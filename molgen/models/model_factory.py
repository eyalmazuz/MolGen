from enum import Enum

import torch.nn as nn

from molgen.models.model_options import ModelType
from molgen.models.gpt import GPT
from molgen.models.bert import Bert

def get_model(model_type: str, model_config):
    type_ = ModelType.from_str(model_type)

    match type_:
        case ModelType.GPT:
            return GPT(model_config)

        case ModelType.BERT:
            return Bert(model_config)

        case _:
            raise ValueError(f"Invalid ModelType {type_}")

