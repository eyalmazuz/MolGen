import os
from typing import List, Set

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import seaborn as sns
from tqdm import tqdm

def convert_to_molecules(smiles_list: List[str]) -> List[Chem.rdchem.Mol]:
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]

    return mols


def filter_invalid_molecules(mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
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


def get_molecule_scaffold(mol: Chem.rdchem.Mol) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    return scaffold

def convert_to_scaffolds(mols: List[Chem.rdchem.Mol]) -> Set[str]:
    scaffolds = set()
    for mol in tqdm(mols):
        scaffold = get_molecule_scaffold(mol)
        scaffolds.add(scaffold)

    return scaffolds
