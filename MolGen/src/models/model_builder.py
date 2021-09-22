from enum import Enum

from .recurrent import RecurrentConfig, RecurrentModel
from .gpt import GPTConfig, GPT

class ModelOpt(Enum):
	RECURRENT = 1
	GPT = 2

def get_model(type=ModelOpt.RECURRENT, **kwargs):

	if type == ModelOpt.RECURRENT:
		config = RecurrentConfig(**kwargs)
		model = RecurrentModel(config)
	
	elif type == ModelOpt.GPT:
		config = GPTConfig(**kwargs)
		model = GPT(config)

	else:
		raise ValueError("Invalid choice")

	return model
	