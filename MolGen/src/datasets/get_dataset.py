from enum import Enum


from .smiles_dataset import SmilesDataset
from .scaffold_dataset import ScaffoldDataset

class DatasetOpt(Enum):
	SMILES = 1
	SCAFFOLD = 2

def get_dataset(type=DatasetOpt.SMILES, **kwargs):

	if type == DatasetOpt.SMILES:
		dataset = SmilesDataset(**kwargs)
	
	elif type == DatasetOpt.SCAFFOLD:
		dataset = ScaffoldDataset(**kwargs)

	else:
		raise ValueError("Invalid choice")

	return dataset
