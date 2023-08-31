import json
import os
import random
from typing import List, Dict, Tuple, Callable, Union

import moses
from torch._C import Value
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import seaborn as sns
import torch
from tqdm import trange, tqdm

from ..utils.metrics import calc_qed, calc_sas, calc_diversity, calc_novelty, calc_valid_molecules
from ..utils.utils import generate_and_save_plot, sample, sample_scaffodls
from ..utils.mol_utils import convert_to_molecules, filter_invalid_molecules

def generate_smiles_scaffolds(model,
                              tokenizer,
                              scaffolds,
                              temprature=1,
                              num_samples=10,
                              size: int=1000,
                              batch_size: int=100,
                              max_len=100,
                              device=torch.device('cuda'), 
                              return_smiles=True,
                              disable=False) -> List[str]:

    print(f'Evaluate {device}')
    model.to(device)
    if return_smiles:
        model.eval()
    gen_smiles = []
    
    if num_samples < len(scaffolds):
        scaffolds_sample = random.sample(scaffolds, num_samples)
    else:
        scaffolds_sample = scaffolds

    print(scaffolds_sample)
    print(f'{size=}, {batch_size=}, {(size // (batch_size * len(scaffolds_sample)))=}')
    for scaffold in tqdm(scaffolds_sample, disable=disable):
        encoding = tokenizer('[BOS]' + scaffold + '[SEP]')
        for batch in range(size // (batch_size * len(scaffolds_sample))):

            tokens = sample(model, encoding['input_ids'],
                        batch_size, max_len, temprature, device,)

            tokens = tokens.tolist()

            for mol in tokens:
                try:
                    end_idx = mol.index(tokenizer.eos_token_id)
                except ValueError:
                    end_idx = len(mol)

                mol = mol[:end_idx+1]
                if return_smiles:
                    len_scaffold = len(encoding['input_ids'])
                    mol = mol[len_scaffold-1:]
                    smiles = tokenizer.decode(mol[1:-1])
                    gen_smiles.append(smiles)

                else:
                    gen_smiles.append(mol)

    return gen_smiles

def generate_smiles_constrained(model,
                              tokenizer,
                              scaffolds,
                              temprature=1,
                              num_samples=10,
                              size: int=1000,
                              batch_size: int=100,
                              max_len=100,
                              device=torch.device('cuda'), 
                              disable=False) -> List[str]:

    print(f'Evaluate {device}')
    model.to(device)
    model.eval()
    gen_smiles = []
    
    if num_samples < len(scaffolds):
        scaffolds_sample = random.sample(scaffolds, num_samples)
    else:
        scaffolds_sample = scaffolds

    for scaffold in scaffolds_sample:
        encoding = tokenizer('[BOS]' + scaffold + '[SEP]')

        tokens = sample(model, encoding, batch_size, max_len, temprature, device)
        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[2 + len(scaffold): end_idx]

            smiles = tokenizer.decode(mol)
            gen_smiles.append(smiles)


    return gen_smiles

def generate_smiles(model,
                    tokenizer,
                    temprature: int=1,
                    size: int=1000,
                    batch_size: int=100,
                    max_len:int=100,
                    device=torch.device('cuda'),
                    return_smiles=True,
                    disable=False) -> List[str]:

    print(f'Evaluate {device}')
    model.to(device)
    if return_smiles:
        model.eval()
    gen_smiles = []

    batches = 1 if size // batch_size == 0 else size // batch_size
    batch_size = batch_size if size > batch_size else size

    
    for batch in trange(batches, disable=disable):
        tokens = sample(model, [tokenizer.bos_token_id], batch_size, max_len, temprature, device)
        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[:end_idx+1]
            if return_smiles:
                smiles = tokenizer.decode(mol[1:-1])
                gen_smiles.append(smiles)
            else:
                gen_smiles.append(mol)

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
                  value_range=(0,1),
                  lst: bool=False,
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
            for start, stop in [ranges[i:i+2] for i in range(0, len(ranges)-1)]:
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
        for start, stop in [ranges[i:i+2] for i in range(0, len(ranges)-1)]:
            stats[f'{start} < {desc} <= {stop}'] = np.count_nonzero((start < batch_rewards) & (batch_rewards <= stop))

    return batch_rewards, stats

def get_top_k_mols(generated_molecules: List[Chem.rdchem.Mol],
                   generated_scores: Union[List[float], Dict[str, List[float]]],
                   top_k: int=5,
                   score_name: str='qed',
                   get_max: bool=True,
                   save_path: str=None) -> Dict[str, float]:
    metrics = {}

    if isinstance(generated_scores, dict):
        sorted_args = np.argsort(generated_scores['Total Reward'])[::-1]

        print(len(sorted_args), len(generated_molecules))
        top_k_molecules = np.array(generated_molecules)[sorted_args][:top_k]
        
        top_k_generated_scores = {}
        for (name, score) in generated_scores.items():
            #print(name, score[:top_k], sorted_args[:top_k])
            top_k_generated_scores[name] = np.array(score)[sorted_args][:top_k]
            # top_k_scores.append((name, np.array(score)[sorted_args][:top_k]))
        
        for i in range(top_k):
            molecule = top_k_molecules[i]
        #for i, (molecule, (name, scores)) in enumerate(zip(top_k_molecules, top_k_scores)):
            smiles = Chem.MolToSmiles(molecule)
            try:
                Draw.MolToFile(molecule, f'{save_path}/top_{i+1}_{smiles.replace("/", "_")}.png')
            except Exception as e:
                print('failed to save ', smiles)
                print(e)

            metrics[f'top_{i+1}_smiles'] = smiles

            #for j in range(len(top_k_scores)):
            for name, score in top_k_generated_scores.items():
                #name, score = top_k_scores[j][0], top_k_scores[j][1][i]
                if score_name != 'qed':
                    metrics[f'top {i+1} {name}'] = score[i]

            metrics[f'top {i+1} qed'] = calc_qed(molecule)
            metrics[f'top {i+1} sas'] = calc_sas(molecule)
            metrics[f'top {i+1} len'] = len(smiles)

    else:
        sorted_molecules, sorted_scores = list(zip(*list(sorted(zip(generated_molecules, generated_scores), key=lambda x: x[1], reverse=get_max))))
        top_k_molecules, top_k_scores = sorted_molecules[:top_k], sorted_scores[:top_k]
        for i, (molecule, score) in enumerate(zip(top_k_molecules, top_k_scores)):
            smiles = Chem.MolToSmiles(molecule)
            try:
                Draw.MolToFile(molecule, f'{save_path}/top_{i+1}_{smiles}.png')
            except Exception:
                print('failed to save ', smiles)
            metrics[f'top_{i+1}_smiles'] = smiles
            if score_name != 'qed':
                metrics[f'top {i+1} {score_name}'] = score
            metrics[f'top {i+1} qed'] = calc_qed(molecule)
            metrics[f'top {i+1} sas'] = calc_sas(molecule)
            metrics[f'top {i+1} len'] = len(smiles)

    return metrics

def get_stats(train_set: Dataset,
              generated_smiles: List[str],
              save_path: str='./data',
              folder_name: str='results',
              top_k: int=5,
              run_moses: bool=False,
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
    
    if str(reward_fn) != 'QED':
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
                generated_reward_values_filtered = filter(lambda x : x != 0, values)
                generated_reward_values_filtered = list(generated_reward_values_filtered)

                generate_and_save_plot(generated_reward_values_filtered,
                                        sns.kdeplot,
                                        xlabel=f'{str(name)}',
                                        ylabel='Density',
                                        title=f'Generated set {str(name)} density',
                                        save_path=generated_path,
                                        name=f"generated_{str(name)}_distribution",
                                        color='green',
                                        shade=True)

        else:
            generated_reward_values_filtered = filter(lambda x : x != 0, generated_reward_values)
            generated_reward_values_filtered = list(generated_reward_values_filtered)

            generate_and_save_plot(generated_reward_values_filtered,
                                    sns.kdeplot,
                                    xlabel=f'{str(reward_fn)}',
                                    ylabel='Density',
                                    title=f'Generated set {str(reward_fn)} density',
                                    save_path=generated_path,
                                    name=f"generated_{str(reward_fn)}_distribution",
                                    color='green',
                                    shade=True)

    generate_and_save_plot(generated_qed_values,
                           sns.kdeplot,
                           xlabel='QED',
                           ylabel='Density',
                           title='Generated set QED density',
                           save_path=generated_path,
                           name="generated_qed_distribution",
                           color='green',
                           shade=True)

    print('Calculating SAS')
    generated_sas_values, generated_sas_stats = calc_set_stat(generated_molecules,
                                                              calc_sas,
                                                              lst=False,
                                                              value_range=(1, 10), 
                                                              desc='SAS')
    
    generate_and_save_plot(generated_sas_values,
                           sns.kdeplot,
                           xlabel='SAS',
                           ylabel='Density',
                           title='Generated set SAS density',
                           save_path=generated_path,
                           name="generated_sas_distribution",
                           color='green',
                           shade=True)

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

    if run_moses:
        print('Running Moses')
        metrics = moses.get_all_metrics(generated_smiles)
        with open(f'{generated_path}/moses_metrics.json', 'w') as f:
            json.dump(metrics, f)
    

def gen_till_train(model, dataset, times: int=10, device=torch.device('cuda')):
    
    results = []
    for i in trange(times):
        count = 0
        test_set = dataset.test_molecules
        not_in_test = True
        while not_in_test:
            smiles_set = generate_smiles(model, dataset.tokenizer, device=device, disable=True)
            for smiles in smiles_set:
                smiles = smiles
                if not smiles or smiles not in test_set:
                    count += 1
                else:
                    not_in_test = False
                    break
        results.append(count)
    
    results = np.array(results)
    return results.mean(), results.std()

def main():
    pass

if __name__ == "__main__":
    main()
