from datetime import datetime
import sys
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import seaborn as sns
import torch
from tqdm import trange, tqdm

from src.utils.metrics import *
from src.utils.utils import convert_to_mols, filter_invalid_mols

def generate_smiles(model, dataset, temprature=1, size=1000) -> List[Chem.rdchem.Mol]:
    
    model.to('cpu')
    model.eval()
    gen_smiles = []
    for i in trange(size):
        tokens = [dataset.token2id['[BOS]']]
        next_token = ''
        while next_token != dataset.token2id['[EOS]']  and len(tokens) < 36:
            x = torch.tensor([tokens])
            y_pred = model(x)

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits / temprature, dim=0).detach().numpy()
            next_token = np.random.choice(len(last_word_logits), p=p)
            tokens.append(next_token)

        smiles = dataset.decode(tokens[1:-1])
        gen_smiles.append(smiles)

    #gen_smiles = [Chem.MolFromSmiles(smiles) for smiles in gen_smiles]
    return gen_smiles

def get_stats(train_set, generated_smiles, save_path=None):

    print('Converting smiles to mols')
    train_mol_set = convert_to_mols(train_set)
    generated_molecules = convert_to_mols(generated_smiles)

    print('Calculating percentage of valid mols')
    train_set_valid_count = calc_valid_mols(train_mol_set)
    generated_set_valid_count = calc_valid_mols(generated_molecules)

    print(f'Percentage of valid molecules in the train-set: {train_set_valid_count * 100}')
    print(f'Percentage of valid molecules in the generated-set: {generated_set_valid_count * 100}')

    print('Filtering invlaid mols')
    train_set = filter_invalid_mols(train_mol_set)
    generated_molecules = filter_invalid_mols(generated_molecules)

    cur_date = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # Calculating statics on the train-set.
    print('Calculating Train set stats')


    print('Calculating diversity')
    train_diversity_score = calc_diversity(train_set)
    print(f'Train-set diversity score: {train_diversity_score * 100}')

    # print('Calculating QED')
    # train_qeds = [calc_qed(mol) for mol in tqdm(train_mol_set, desc='QED')]
    # train_qed_plot = sns.kdeplot(train_qeds, color='green', shade=True)
    # train_qeds = np.array(train_qeds)
    # print(f'Train QED mean, std and median: {train_qeds.mean()}, {train_qeds.std()}, {np.median(train_qeds)}')
    
    # print('Calculating SAS score')
    # train_sass = [calc_sas(mol) for mol in tqdm(train_mol_set, desc='SAS')]
    # train_sas_plot = sns.kdeplot(train_sass, color='green', shade=True)
    # train_sass = np.array(train_sass)
    # print(f'Train SAS mean, std and median: {train_sass.mean()}, {train_sass.std()}, {np.median(train_sass)}')
    
    # if save_path is not None:
    #     train_path = f'{save_path}/{cur_date}/train'
    #     if not os.path.exists(train_path):
    #         os.makedirs(train_path, exist_ok=True)
    #     train_qed_plot.figure.savefig(f'{train_path}/train_qed_distribution.png')
    #     #train_sas_plot.figure.savefig(f'{train_path}/train_sas_distribution.png')
    #     plt.clf()

    # Calculating statistics on the generated-set.
    print('Calculating Generated set stats')

    print('Calculating diversity')
    generated_diversity_score = calc_diversity(generated_smiles)
    print(f'Generated-set diversity score: {generated_diversity_score * 100}')

    print('Calculating novelty')
    generated_novelty_score = calc_novelty(train_set, generated_smiles)
    print(f'Generated-set novelty score: {generated_novelty_score * 100}')

    print('Calculating QED')
    generated_qeds = [calc_qed(mol) for mol in tqdm(generated_molecules, desc='QED')]
    generated_qed_plot = sns.kdeplot(generated_qeds, color='green', shade=True)
    generated_qeds = np.array(generated_qeds)
    print(f'Generated QED mean, std and median: {generated_qeds.mean()}, {generated_qeds.std()}, {np.median(generated_qeds)}')

    if save_path is not None:
        generated_path = f'{save_path}/{cur_date}/generated'
        if not os.path.exists(generated_path):
            os.makedirs(generated_path, exist_ok=True)
        generated_qed_plot.figure.savefig(f'{generated_path}/generated_qed_distribution.png')
        plt.clf()

    print('Calculating SAS score')
    generated_sass = [sascorer.calculateScore(mol) for mol in tqdm(generated_molecules, desc='SAS')]
    generated_sas_plot = sns.kdeplot(generated_sass, color='green', shade=True)
    generated_sass = np.array(generated_sass)
    print(f'Generated SAS mean, std and median: {generated_sass.mean()}, {generated_sass.std()}, {np.median(generated_sass)}')
    
    if save_path is not None:
        generated_path = f'{save_path}/{cur_date}/generated'
        if not os.path.exists(generated_path):
            os.makedirs(generated_path, exist_ok=True)
        generated_sas_plot.figure.savefig(f'{generated_path}/generated_sas_distribution.png')
        plt.clf()

def main():
    pass

if __name__ == "__main__":
    main()
