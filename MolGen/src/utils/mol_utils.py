from typing import List, Set

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

def get_molecule_scaffold(mol: Chem.rdchem.Mol) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    return scaffold

def convert_to_scaffolds(mols: List[Chem.rdchem.Mol]) -> Set[str]:
    scaffolds = set()
    for mol in tqdm(mols):
        scaffold = get_molecule_scaffold(mol)
        scaffolds.add(scaffold)

    return scaffolds

def convert_to_molecules(smiles_list: List[str]) -> List[Chem.rdchem.Mol]:
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]

    return mols


def filter_invalid_molecules(mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
    mols = list(filter(lambda x: x != None, mols))

    return mols
