from rdkit import Chem
from src.utils.metrics import calc_qed


def qed_reward(mol: Chem.rdchem.Mol, multiplier: float=10.0) -> float:
	qed = calc_qed(mol)
	return multiplier * qed

	