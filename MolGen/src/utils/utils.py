import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import seaborn as sns
from tqdm import tqdm

def convert_smiles_to_csv(smiles):
    pass


def convert_to_mols(smiles_list: List[str]) -> List[Chem.rdchem.Mol]:
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]

    return mols


def filter_invalid_mols(mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
    mols = list(filter(lambda x: x != None, mols))

    return mols


def generate_and_save_plot(values,
                           plot_func,
                           xlabel,
                           ylabel,
                           title,
                           save_path,
                           name,
                           **kwargs):
    
    plot = plot_func(values, **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plot.figure.savefig(f'{save_path}/{name}.png')
    plt.clf()


