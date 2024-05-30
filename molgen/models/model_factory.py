from enum import Enum

import torch.nn as nn

from molgen.models.model_options import ModelType
from molgen.models.gpt import GPT, GPTConfig
from molgen.models.bert import Bert, BertConfig
from molgen.models.transformer import Transformer, TransformerConfig
from molgen.models.recurrent import RNN, RNNConfig

def get_model(model_type: str, model_args):
    type_ = ModelType.from_str(model_type)

    match type_:
        case ModelType.GPT:
            config = GPTConfig(**model_args)
            return GPT(config)

        case ModelType.BERT:
            config = BertConfig(**model_args)
            return Bert(config)

        case ModelType.TRANSFORMER:
            config = TransformerConfig(**model_args)
            return Transformer(config)

        case ModelType.RNN:
            config = RNNConfig(**model_args)
            return RNN(config)

        case _:
            raise ValueError(f"Invalid ModelType {type_}")

