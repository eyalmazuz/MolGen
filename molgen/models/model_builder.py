from enum import Enum

import torch.nn as nn
from .recurrent import RecurrentConfig, RecurrentModel
from .gpt import GPTConfig, GPT
from .transformer import TransformerConfig, Transoformer
from ..utils.utils import ModelOpt

def get_model(type=ModelOpt.RECURRENT, **kwargs):

	print(type)
	if type == ModelOpt.RECURRENT:
		config = RecurrentConfig(**kwargs)
		model = RecurrentModel(config)
	
	elif type == ModelOpt.GPT:
		config = GPTConfig(**kwargs)
		model = GPT(config)

	elif type == ModelOpt.TRANSFORMER:
		config = TransformerConfig(**kwargs)
		model = Transoformer(config)

	else:
		raise ValueError("Invalid choice")

	return model
	
