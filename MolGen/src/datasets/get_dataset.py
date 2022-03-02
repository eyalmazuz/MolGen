from torch.utils import data
from .smiles_dataset import SmilesDataset
from .scaffold_dataset import ScaffoldDataset
from .constrained_dataset import ConstraiedDataset
from ..utils.utils import ModelOpt, TaskOpt

def get_dataset(type=ModelOpt.GPT, task=TaskOpt.CONSTRAINED, **kwargs):

	if type == ModelOpt.GPT or type == ModelOpt.RECURRENT:
		if task == TaskOpt.REGULAR:
			dataset = SmilesDataset(**kwargs)
		else:
			print('getting constrained')
			dataset = ConstraiedDataset(**kwargs)

	elif type == ModelOpt.TRANSFORMER:
		dataset = ScaffoldDataset(**kwargs)

	else:
		raise ValueError("Invalid choice")

	return dataset
