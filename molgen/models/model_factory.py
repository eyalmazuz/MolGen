from typing import Any

from dacite import from_dict
from torch import nn

from molgen.models.gpt import GPT, GPTConfig
from molgen.models.llama import Llama, LlamaConfig
from molgen.models.model_options import ModelType

config_type = type[GPTConfig] | type[LlamaConfig]
model_class_type = type[GPT] | type[Llama]


def get_model(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    config_cls: config_type
    model_cls: model_class_type

    match model_type.lower():
        case ModelType.GPT:
            config_cls = GPTConfig
            model_cls = GPT
        case ModelType.LLAMA:
            config_cls = LlamaConfig
            model_cls = Llama
        case _:
            raise ValueError(f"Invalid type {model_type}")

    config = from_dict(data_class=config_cls, data=model_config)
    model = model_cls(config)  # type: ignore

    return model
