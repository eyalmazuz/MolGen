from enum import Enum

import torch.nn as nn
from .recurrent import RecurrentConfig, RecurrentModel
from .gpt import GPTConfig, GPT

class MyDataParallel(nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class ModelOpt(Enum):
	RECURRENT = 1
	GPT = 2
	TRANSFORMER = 3

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
	
