from enum import StrEnum, auto
from typing import Any

from dacite import from_dict
from torch.nn import Module

from molgen.models.gpt import GPT, GPTConfig

config_type = type[GPTConfig]


class ModelType(StrEnum):
    GPT = auto()


def get_model(model_type: str, model_config: dict[str, Any]) -> Module:
    config_cls: config_type
    model_cls: type[Module]

    match model_type.upper():
        case ModelType.GPT:
            config_cls = GPTConfig
            model_cls = GPT

    config = from_dict(data_class=config_cls, data=model_config)
    model = model_cls(config)

    return model
