from typing import Any

from dacite import from_dict
from torch.nn import Module

from molgen.models.bert import Bert, BertConfig
from molgen.models.gpt import GPT, GPTConfig
from molgen.models.model_options import ModelType
from molgen.models.transformer import Transformer, TransformerConfig

config_type = type[GPTConfig] | type[BertConfig] | type[TransformerConfig]


def get_model(model_type: ModelType, model_config: dict[str, Any]) -> Module:
    config_cls: config_type
    model_cls: type[Module]

    match model_type:
        case ModelType.GPT:
            config_cls = GPTConfig
            model_cls = GPT
        case ModelType.BERT:
            config_cls = BertConfig
            model_cls = Bert
        case ModelType.TRANSFORMER:
            config_cls = TransformerConfig
            model_cls = Transformer
        case _:
            raise ValueError(f"Invalid model type {model_type}")

    config = from_dict(data_class=config_cls, data=model_config)
    model = model_cls(config)

    return model
