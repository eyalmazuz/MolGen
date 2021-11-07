from .smiles_dataset import SmilesDataset
from .scaffold_dataset import ScaffoldDataset
from ..utils.utils import ModelOpt

def get_dataset(type=ModelOpt.GPT, **kwargs):

	if type == ModelOpt.GPT or type == ModelOpt.RECURRENT:
		dataset = SmilesDataset(**kwargs)
	
	elif type == ModelOpt.TRANSFORMER:
		dataset = ScaffoldDataset(**kwargs)

	else:
		raise ValueError("Invalid choice")

	return dataset
