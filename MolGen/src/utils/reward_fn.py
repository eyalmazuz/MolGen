import math
from typing import Callable

from rdkit import Chem
from src.utils.metrics import calc_qed, calc_sas


def qed_reward(smiles: str, fn: Callable[[float], float]=lambda x: x*10) -> float:

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        qed = calc_qed(mol)
        reward = fn(qed)
    else:
        reward = 0

    
    return reward

def sas_reward(smiles: str, fn: Callable[[float], float]=lambda x: -math.exp(x/3)) -> float:

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        sas = calc_sas(mol)
        reward = fn(sas)
    else:
        reward = -1000

    
    return reward

       