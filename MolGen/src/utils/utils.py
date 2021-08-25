import os
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def convert_smiles_to_csv(smiles):
    pass


def convert_to_mols(smiles_list: List[str]) -> List[Chem.rdchem.Mol]:
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]

    return mols


def filter_invalid_mols(mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
    mols = list(filter(lambda x: x != None, mols))

    return mols
