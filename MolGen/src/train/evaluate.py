import json
import os
import random
from typing import List, Dict, Tuple, Callable

import moses
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import seaborn as sns
import torch
from tqdm import trange, tqdm

from ..utils.metrics import calc_qed, calc_sas, calc_diversity, calc_novelty, calc_valid_molecules
from ..utils.utils import generate_and_save_plot
from ..utils.mol_utils import convert_to_molecules, filter_invalid_molecules

def generate_smiles_scaffolds(model, tokenizer, scaffolds, temprature=1, num_samples=10, size=1000,
                                max_len=100, device=torch.device('cuda'), 
                                disable=False) -> List[str]:
    print(f'Evaluate {device}')
    if torch.cuda.device_count() > 1:
        model = model.module
    model.to(device)
    model.eval()
    gen_smiles = []
    
    if num_samples < len(scaffolds):
        scaffolds_sample = random.sample(scaffolds, num_samples)
    else:
        scaffolds_sample = scaffolds

    for scaffold in scaffolds_sample:
        encoding = tokenizer('[BOS]' + scaffold + '[EOS]')

        for i in trange(size // num_samples, disable=disable):
            
            tokens = model.generate(tokenizer.bos_token_id, tokenizer.eos_token_id, encoding['input_ids'], 
                                    encoding['padding_mask'], temprature, max_len, device)

            smiles = tokenizer.decode(tokens[1:-1])
            gen_smiles.append(smiles)

    return gen_smiles

def generate_smiles(model, tokenizer, temprature=1, size=1000, max_len=100,
                    device=torch.device('cuda'), disable=False) -> List[str]:
    print(f'Evaluate {device}')
    if torch.cuda.device_count() > 1:
        model = model.module
    model.to(device)
    model.eval()
    gen_smiles = []
    
    for i in trange(size, disable=disable):
        
        tokens = model.generate(tokenizer.bos_token_id, tokenizer.eos_token_id, temprature, max_len, device)

        smiles = tokenizer.decode(tokens[1:-1])
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
                   save_path: str=None) -> Dict[str, float]:
    sorted_values, _ = list(zip(*list(sorted(zip(generated_molecules, generated_score), key=lambda x: x[1], reverse=True))))
    top_k_molecules = sorted_values[:top_k]
    metrics = {}
    for i, molecule in enumerate(top_k_molecules):
        smiles = Chem.MolToSmiles(molecule)
        Draw.MolToFile(molecule, f'{save_path}/top_{i+1}_{smiles}.png')
        metrics[f'top_{i+1}_qed'] = calc_qed(molecule)
        metrics[f'top_{i+1}_sas'] = calc_sas(molecule)
        metrics[f'top_{i+1}_len'] = len(smiles)

    return metrics
    


def get_stats(train_set: List[str],
              generated_smiles: List[str],
              save_path: str='./data',
              folder_name: str='results',
              top_k: int=5,
              run_moses: bool=False):

    print('Converting smiles to mols')
    generated_molecules = convert_to_molecules(generated_smiles)

    print('Filtering invlaid mols')
    generated_molecules = filter_invalid_molecules(generated_molecules)

    # Calculating statistics on the generated-set.
    print('Calculating Generated set stats')
    
    if folder_name:
        generated_path = os.path.join(save_path, folder_name)

    print('Calculating QED')
    generated_qed_values, generated_qed_stats = calc_set_stat(generated_molecules, calc_qed, value_range=(0, 1), desc='QED')
    
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

    top_k_metrics = get_top_k_mols(generated_molecules, generated_qed_values, top_k=top_k, save_path=generated_path)

    stats = {**generated_qed_stats, **generated_sas_stats, **top_k_metrics}
    
    print('Calculating diversity')
    generated_diversity_score = calc_diversity(generated_smiles)
    stats['diversity'] = generated_diversity_score
    
    if train_set is not None:
        print('Calculating novelty')
        generated_novelty_score = calc_novelty(train_set, generated_smiles)
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
