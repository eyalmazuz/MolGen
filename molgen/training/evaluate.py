import os
import json
import tomllib
import argparse
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from molgen.models.dt_gpt import sample
from molgen.models.model_factory import get_model
from molgen.models.model_options import ModelType
from molgen.rewards.reward_factory import get_rewards
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.utils.mol_utils import convert_to_molecules, filter_invalid_molecules
from molgen.utils.metrics import calc_qed, calc_sas, calc_diversity, calc_novelty, calc_valid_molecules


def load_model(model_config, args):
    """
    Loads the pre-trained model from the specified checkpoint.

    Returns:
        model: Loaded PyTorch model.
    """
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Initialize model (make sure MolecularGenerator matches your model architecture)
    model_type = ModelType.from_str(args.model_type)

    model = get_model(model_type, model_config).to(args.device)
    model.to(args.device)

    # Load the model weights from the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)

    return model


def generate_molecules(model, tokenizer, reward_func, args, temperature: int = 1, ret: float = 1.0):
    """
    Generate 'k' molecules using the pre-trained model's sample function.

    Args:
        model: Loaded model capable of generating molecular structures.
        k (int): Number of molecules to generate.

    Returns:
        list: List of generated molecules.
    """
    model.eval()  # Set the model to evaluation mode

    gen_smiles = []
    done = True
    for i in range(args.k):
        terminated = False
        init_state = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64)
        init_state = init_state.to(args.device).unsqueeze(0).unsqueeze(0)
        # first state is from env, first rtg is target return, and first timestep is 0
        rtgs = [ret]
        sampled_action = sample(
            model=model,
            x=init_state,
            steps=1,
            temperature=temperature,
            sample=True,
            actions=None,
            rtgs=torch.tensor(rtgs, dtype=torch.float32).to(args.device).unsqueeze(0).unsqueeze(-1),
            # timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device)
        )

        j = 0
        all_states = init_state
        actions = []
        while True:
            if done:
                state, reward_sum, done = ([tokenizer.bos_token_id], 0, False)
            action = sampled_action.cpu().numpy()[0, -1]
            actions += [action]
            state.append(action)
            reward = reward_func(
                tokenizer.decode(state, skip_special_tokens=True)
            )[0]
            done = action == tokenizer.eos_token_id  # mol is complete when [EOS] token is generated
            reward_sum = reward
            j += 1

            # if molecule length exceeds block_size and [EOS] token wasn't generated terminate generation
            if len(state) >= model.block_size // 3 and not done:
                terminated = True

            if done or terminated:
                gen_smiles.append(tokenizer.decode(state, skip_special_tokens=True)[0])
                break

            tensor_state = torch.tensor(state, device=args.device).unsqueeze(0).unsqueeze(0)
            pad_size = tensor_state.shape[-1] - all_states.shape[-1]
            all_states = torch.nn.functional.pad(all_states, (0, pad_size), value=tokenizer.pad_token_id)
            all_states = torch.cat([all_states, tensor_state], dim=1)

            rtgs += [rtgs[-1]]  # - reward]  # TODO: Check this
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep # TODO: check the tensor(actions) to verify its correct
            sampled_action = sample(
                model=model,
                x=all_states,
                steps=1,
                temperature=temperature,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long).to(args.device).unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(args.device).unsqueeze(0),
                attention=torch.tensor(np.tril(np.ones(all_states.shape[1:])), dtype=torch.long).to(args.device).unsqueeze(0)
                # timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
            )

    return gen_smiles

def fail_safe(func: Callable[[Chem.rdchem.Mol], float], mol: Chem.rdchem.Mol) -> float:
    # return func(mol)
    try:
        res = func(mol)
    except Exception as e:
        res = None
        print(f'{mol=}')
    return res


def calc_set_stat(mol_set: List[Chem.rdchem.Mol],
                  func: Callable[[Chem.rdchem.Mol], float],
                  value_range=(0, 1),
                  lst: bool = False,
                  desc=None) -> Tuple[List[float], Dict[str, float]]:
    stats = {}
    if lst:
        batch_rewards = fail_safe(func, mol_set)
    else:
        batch_rewards = np.array([fail_safe(func, mol) for mol in tqdm(mol_set, desc=desc)])

    if isinstance(batch_rewards, dict):
        transformed_batch_rewards = []
        for fn, values in zip(func.reward_fns, batch_rewards.values()):
            if fn.multiplier is not None:
                transformed_batch_rewards.append(list(map(fn.multiplier, values)))
            else:
                transformed_batch_rewards.append(values)

        transformed_batch_rewards = list(zip(*transformed_batch_rewards))
        transformed_batch_rewards = [sum(rewards) for rewards in transformed_batch_rewards]

        batch_rewards['Total Reward'] = transformed_batch_rewards

        for name, value in batch_rewards.items():
            len_batch_rewards = len(value)
            value = [mol for mol in value if mol is not None]
            failed_batch_rewards = len_batch_rewards - len(value)

            value = np.array(value)
            stats[f'{desc} {name} mean'] = value.mean()
            stats[f'{desc} {name} std'] = value.std()
            stats[f'{desc} {name} median'] = np.median(value)
            stats[f'{desc} {name} failed'] = failed_batch_rewards
            start, stop = value_range
            ranges = np.linspace(start, stop, 6)
            for start, stop in [ranges[i:i + 2] for i in range(0, len(ranges) - 1)]:
                stats[f'{start} < {desc} {name} <= {stop}'] = np.count_nonzero((start < value) & (value <= stop))

    else:
        len_batch_rewards = len(batch_rewards)
        batch_rewards = [mol for mol in batch_rewards if mol is not None]
        failed_batch_rewards = len_batch_rewards - len(batch_rewards)

        batch_rewards = np.array(batch_rewards)
        stats[f'{desc} mean'] = batch_rewards.mean()
        stats[f'{desc} std'] = batch_rewards.std()
        stats[f'{desc} median'] = np.median(batch_rewards)
        stats[f'{desc} failed'] = failed_batch_rewards
        start, stop = value_range
        ranges = np.linspace(start, stop, 6)
        for start, stop in [ranges[i:i + 2] for i in range(0, len(ranges) - 1)]:
            stats[f'{start} < {desc} <= {stop}'] = np.count_nonzero((start < batch_rewards) & (batch_rewards <= stop))

    return batch_rewards, stats


def get_top_k_mols(generated_molecules: List[Chem.rdchem.Mol],
                   generated_scores: Union[List[float], Dict[str, List[float]]],
                   top_k: int = 5,
                   score_name: str = 'qed',
                   get_max: bool = True,
                   save_path: str = None) -> Dict[str, float]:
    metrics = {}

    if isinstance(generated_scores, dict):
        sorted_args = np.argsort(generated_scores['Total Reward'])[::-1]

        print(len(sorted_args), len(generated_molecules))
        top_k_molecules = np.array(generated_molecules)[sorted_args][:top_k]

        top_k_generated_scores = {}
        for (name, score) in generated_scores.items():
            # print(name, score[:top_k], sorted_args[:top_k])
            top_k_generated_scores[name] = np.array(score)[sorted_args][:top_k]
            # top_k_scores.append((name, np.array(score)[sorted_args][:top_k]))

        for i in range(top_k):
            molecule = top_k_molecules[i]
            # for i, (molecule, (name, scores)) in enumerate(zip(top_k_molecules, top_k_scores)):
            smiles = Chem.MolToSmiles(molecule)
            try:
                Draw.MolToFile(molecule, f'{save_path}/top_{i + 1}_{smiles.replace("/", "_")}.png')
            except Exception as e:
                print('failed to save ', smiles)
                print(e)

            metrics[f'top_{i + 1}_smiles'] = smiles

            # for j in range(len(top_k_scores)):
            for name, score in top_k_generated_scores.items():
                # name, score = top_k_scores[j][0], top_k_scores[j][1][i]
                if score_name != 'qed':
                    metrics[f'top {i + 1} {name}'] = score[i]

            metrics[f'top {i + 1} qed'] = calc_qed(molecule)
            metrics[f'top {i + 1} sas'] = calc_sas(molecule)
            metrics[f'top {i + 1} len'] = len(smiles)

    else:
        sorted_molecules, sorted_scores = list(
            zip(*list(sorted(zip(generated_molecules, generated_scores), key=lambda x: x[1], reverse=get_max))))
        top_k_molecules, top_k_scores = sorted_molecules[:top_k], sorted_scores[:top_k]
        for i, (molecule, score) in enumerate(zip(top_k_molecules, top_k_scores)):
            smiles = Chem.MolToSmiles(molecule)
            try:
                Draw.MolToFile(molecule, f'{save_path}/top_{i + 1}_{smiles}.png')
            except Exception:
                print('failed to save ', smiles)
            metrics[f'top_{i + 1}_smiles'] = smiles
            if score_name != 'qed':
                metrics[f'top {i + 1} {score_name}'] = score
            metrics[f'top {i + 1} qed'] = calc_qed(molecule)
            metrics[f'top {i + 1} sas'] = calc_sas(molecule)
            metrics[f'top {i + 1} len'] = len(smiles)

    return metrics


def get_stats(generated_smiles: List[str],
              save_path: str = './data',
              folder_name: str = 'results',
              top_k: int = 5,
              train_set: Optional[Dataset] = None,
              run_moses: bool = False,
              reward_fn=None,
              scaffold=None):
    stats = {}
    print('Converting smiles to mols')
    generated_molecules = convert_to_molecules(generated_smiles)

    print('Filtering invlaid mols')
    generated_molecules = filter_invalid_molecules(generated_molecules)

    valid_generated_smiles = [Chem.MolToSmiles(mol) for mol in generated_molecules]

    # Calculating statistics on the generated-set.
    print('Calculating Generated set stats')

    if folder_name:
        generated_path = os.path.join(save_path, folder_name)

    generated_reward_values = {}

    print('Calculating QED')
    generated_qed_values, generated_qed_stats = calc_set_stat(generated_molecules,
                                                              calc_qed,
                                                              lst=False,
                                                              value_range=(0, 1),
                                                              desc='QED')

    if reward_fn is not None and str(reward_fn) != 'QED':
        print(f'Calculating {reward_fn}')
        generated_reward_values, generated_reward_stats = calc_set_stat(valid_generated_smiles,
                                                                        reward_fn,
                                                                        lst=True,
                                                                        value_range=(0, 1),
                                                                        desc=f'{str(reward_fn)}')

        print(f'{len(generated_reward_values)=}')
        if isinstance(generated_reward_values, dict):
            for name, values in generated_reward_values.items():
                print(f'Calculating Sub Reward {name}')
                generated_reward_values_filtered = filter(lambda x: x != 0, values)
                generated_reward_values_filtered = list(generated_reward_values_filtered)

                # generate_and_save_plot(generated_reward_values_filtered,
                #                        sns.kdeplot,
                #                        xlabel=f'{str(name)}',
                #                        ylabel='Density',
                #                        title=f'Generated set {str(name)} density',
                #                        save_path=generated_path,
                #                        name=f"generated_{str(name)}_distribution",
                #                        color='green',
                #                        shade=True)

        else:
            generated_reward_values_filtered = filter(lambda x: x != 0, generated_reward_values)
            generated_reward_values_filtered = list(generated_reward_values_filtered)

    #         generate_and_save_plot(generated_reward_values_filtered,
    #                                sns.kdeplot,
    #                                xlabel=f'{str(reward_fn)}',
    #                                ylabel='Density',
    #                                title=f'Generated set {str(reward_fn)} density',
    #                                save_path=generated_path,
    #                                name=f"generated_{str(reward_fn)}_distribution",
    #                                color='green',
    #                                shade=True)
    #
    # generate_and_save_plot(generated_qed_values,
    #                        sns.kdeplot,
    #                        xlabel='QED',
    #                        ylabel='Density',
    #                        title='Generated set QED density',
    #                        save_path=generated_path,
    #                        name="generated_qed_distribution",
    #                        color='green',
    #                        shade=True)

    print('Calculating SAS')
    generated_sas_values, generated_sas_stats = calc_set_stat(generated_molecules,
                                                              calc_sas,
                                                              lst=False,
                                                              value_range=(1, 10),
                                                              desc='SAS')

    # generate_and_save_plot(generated_sas_values,
    #                        sns.kdeplot,
    #                        xlabel='SAS',
    #                        ylabel='Density',
    #                        title='Generated set SAS density',
    #                        save_path=generated_path,
    #                        name="generated_sas_distribution",
    #                        color='green',
    #                        shade=True)

    if reward_fn is not None and str(reward_fn) != 'QED':
        top_k_metrics = get_top_k_mols(generated_molecules,
                                       generated_reward_values,
                                       top_k=top_k,
                                       score_name=str(reward_fn),
                                       get_max=("Docking" not in str(reward_fn)),
                                       save_path=generated_path)
    else:
        top_k_metrics = get_top_k_mols(generated_molecules,
                                       generated_qed_values,
                                       top_k=top_k,
                                       score_name='qed',
                                       save_path=generated_path)

    stats = {
        **stats,
        **generated_qed_stats,
        **generated_sas_stats,
        **top_k_metrics
    }

    if reward_fn is not None and str(reward_fn) != 'QED':
        stats = {**stats, **generated_reward_stats}

    print('Calculating diversity')
    generated_diversity_score = calc_diversity(generated_smiles)
    stats['diversity'] = generated_diversity_score

    if train_set is not None:
        print('Calculating novelty')
        generated_novelty_score = calc_novelty(train_set.molecules, generated_smiles)
        stats['novelty'] = generated_novelty_score

    print('Calculating percentage of valid mols')
    generated_set_valid_count = calc_valid_molecules(generated_smiles)
    stats['validity'] = generated_set_valid_count

    print('Calculating count of valid smiles')
    stats['count'] = len(valid_generated_smiles)

    print('calculating average SMILES length')
    stats['average_length'] = sum(map(len, generated_smiles)) / len(generated_smiles)

    print(stats)
    if not os.path.exists(generated_path):
        os.makedirs(generated_path)

    with open(f'{generated_path}/stats.json', 'w') as f:
        json.dump(stats, f)

    if not isinstance(generated_reward_values, dict):
        generated_reward_values = {str(reward_fn): generated_reward_values}
    data = {**{'Smiles': valid_generated_smiles},
            **generated_reward_values,
            **{"QED": generated_qed_values},
            **{"SAS": generated_sas_values},
            }

    for k, v in data.items():
        print(f'{k=} {len(v)=}')
    df = pd.DataFrame(data)
    df.to_csv(f'{generated_path}/generated_smiles.csv', index=False)

    if scaffold is not None:
        with open(f'{generated_path}/scaffold.txt', 'w') as f:
            f.write(scaffold)


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using a pre-trained model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pre-trained model checkpoint file.')
    parser.add_argument('--k', type=int, default=100, help='Number of molecules to generate.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to run the model on, 'cpu' or 'cuda'.")

    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--model_type", type=str, required=True, choices=["GPT", "DT"], help="Type of model to use for training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the connfig containing training and model params")

    args = parser.parse_args()

    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    model_config = config["model_config"]

    # Load the model
    model = load_model(model_config=model_config, args=args)

    tokenizer = get_tokenizer(args.tokenizer_path)
    reward_func = get_rewards(config["reward"])

    # Generate 'k' molecules
    molecules = generate_molecules(model, tokenizer, reward_func, args)

    # Evaluate the generated molecules
    get_stats(molecules, folder_name="zinc_bpe_75_results")


if __name__ == "__main__":
    main()
