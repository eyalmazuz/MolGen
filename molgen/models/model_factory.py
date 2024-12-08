from typing import Any

from dacite import from_dict
from torch.nn import Module

from molgen.models.gpt import GPT, GPTConfig
from molgen.models.model_options import ModelType

config_type = type[GPTConfig]


def get_model(model_type: str, model_config: dict[str, Any]) -> Module:
    config_cls: config_type
    model_cls: type[Module]

    match model_type.lower():
        case ModelType.GPT:
            config_cls = GPTConfig
            model_cls = GPT
        case _:
            raise ValueError(f"Invalid type {model_type}")

    config = from_dict(data_class=config_cls, data=model_config)
    model = model_cls(config)

    return model
