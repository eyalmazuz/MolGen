import json
import os
import random
from typing import List, Dict, Tuple, Callable

import moses
from torch._C import Value
from torch.utils.data import Dataset
import numpy as np
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
        encoding = tokenizer('[BOS]' + scaffold + '[EOS]')

        tokens = sample_scaffodls(model, tokenizer.bos_token_id, encoding['input_ids'], encoding['padding_mask'],
                                  batch_size, max_len, temprature, device,)

        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[1: end_idx]

            smiles = tokenizer.decode(mol)
            gen_smiles.append(smiles)

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
                    disable=False) -> List[str]:

    print(f'Evaluate {device}')
    model.to(device)
    model.eval()
    gen_smiles = []
    
    for batch in trange(size // batch_size, disable=disable):
        tokens = sample(model, [tokenizer.bos_token_id], batch_size, max_len, temprature, device)
        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[1: end_idx]

            smiles = tokenizer.decode(mol)
            gen_smiles.append(smiles)

    return gen_smiles

def fail_safe(func: Callable[[Chem.rdchem.Mol], float], mol: Chem.rdchem.Mol) -> float:
    try:
        res = func(mol)
    except Exception as e:
        res = None
        print(mol)
    return res

def calc_set_stat(mol_set: List[Chem.rdchem.Mol],
                  func: Callable[[Chem.rdchem.Mol], float],
                  value_range=(0,1),
                  desc=None) -> Tuple[List[float], Dict[str, float]]:
    stats = {}
    values = [fail_safe(func, mol) for mol in tqdm(mol_set, desc=desc)]
    len_values = len(values)
    values = [mol for mol in values if mol is not None]
    failed_values = len_values - len(values)


    values = np.array(values)
    stats[f'{desc} mean'] = values.mean()
    stats[f'{desc} std'] = values.std()
    stats[f'{desc} median'] = np.median(values)
    stats[f'{desc} failed'] = failed_values
    start, stop = value_range
    ranges = np.linspace(start, stop, 6)
    for start, stop in [ranges[i:i+2] for i in range(0, len(ranges)-1)]:
        stats[f'{start} < {desc} <= {stop}'] = np.count_nonzero((start < values) & (values <= stop))

    return values, stats

def get_top_k_mols(generated_molecules: List[Chem.rdchem.Mol],
                   generated_score: List[float],
                   top_k: int=5,
                   score_name: str='qed',
                   save_path: str=None) -> Dict[str, float]:
    sorted_molecules, sorted_scores = list(zip(*list(sorted(zip(generated_molecules, generated_score), key=lambda x: x[1], reverse=True))))
    top_k_molecules, top_k_scores = sorted_molecules[:top_k], sorted_scores[:top_k]
    metrics = {}
    for i, (molecule, score) in enumerate(zip(top_k_molecules, top_k_scores)):
        smiles = Chem.MolToSmiles(molecule)
        try:
            Draw.MolToFile(molecule, f'{save_path}/top_{i+1}_{smiles}.png')
        except Exception:
            print('failed to save', smiles)
        metrics[f'top_{i+1}_smiles'] = smiles
        if score_name != 'qed':
            metrics[f'top_{i+1}_{score_name}'] = score
        metrics[f'top_{i+1}_qed'] = calc_qed(molecule)
        metrics[f'top_{i+1}_sas'] = calc_sas(molecule)
        metrics[f'top_{i+1}_len'] = len(smiles)

    return metrics
    


def get_stats(train_set: Dataset,
              generated_smiles: List[str],
              save_path: str='./data',
              folder_name: str='results',
              top_k: int=5,
              run_moses: bool=False,
              reward_fn=None):

    stats = {}
    print('Converting smiles to mols')
    generated_molecules = convert_to_molecules(generated_smiles)

    print('Filtering invlaid mols')
    generated_molecules = filter_invalid_molecules(generated_molecules)

    # Calculating statistics on the generated-set.
    print('Calculating Generated set stats')
    
    if folder_name:
        generated_path = os.path.join(save_path, folder_name)

    print('Calculating QED')
    generated_qed_values, generated_qed_stats = calc_set_stat(generated_molecules,
                                                            calc_qed,
                                                            value_range=(0, 1),
                                                            desc='QED')
    
    if reward_fn is not None:
        print(f'Calculating {reward_fn}')
        generated_reward_values, generated_reward_stats = calc_set_stat(generated_smiles,
                                                                        reward_fn,
                                                                        value_range=(0, 9),
                                                                        desc=f'{str(reward_fn)}')        

        print(f'{len(generated_reward_values)=}')
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
    generated_sas_values, generated_sas_stats = calc_set_stat(generated_molecules, calc_sas, value_range=(1, 10), desc='SAS')
    
    generate_and_save_plot(generated_sas_values,
                           sns.kdeplot,
                           xlabel='SAS',
                           ylabel='Density',
                           title='Generated set SAS density',
                           save_path=generated_path,
                           name="generated_sas_distribution",
                           color='green',
                           shade=True)

    if reward_fn is not None:
        top_k_metrics = get_top_k_mols(generated_molecules,
                                       generated_reward_values,
                                       top_k=top_k,
                                       score_name='ic50',
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

    if reward_fn is not None:
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

    print('calculating average SMILES length')
    stats['average_length'] = sum(map(len, generated_smiles)) / len(generated_smiles)

    print(stats)
    with open(f'{generated_path}/stats.json', 'w') as f:
        json.dump(stats, f)

    with open(f'{generated_path}/generated_smiles.txt', 'w') as f:
        f.write('\n'.join(generated_smiles))

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
