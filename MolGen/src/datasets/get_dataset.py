from torch.utils import data
from .smiles_dataset import SmilesDataset
from .scaffold_dataset import ScaffoldDataset
from .constrained_dataset import ConstraiedDataset
from ..utils.utils import ModelOpt, TaskOpt

def get_dataset(type=ModelOpt.GPT, task=TaskOpt.CONSTRAINED, **kwargs):

	dataset = SmilesDataset(**kwargs)

	return dataset
