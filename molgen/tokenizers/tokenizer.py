from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import torch

TokenizedData: TypeAlias = list[list[int]] | torch.Tensor


class AbstractTokenizer(ABC):

    @abstractmethod
    def encode(self, texts: str | list[str], return_tensors: bool) -> TokenizedData:
        pass


    @abstractmethod
    def decode(self, encodings: TokenizedData, skip_special_tokens: bool) -> list[str]:
        pass


    @classmethod
    @abstractmethod
    def load_pretrained(cls: type["AbstractTokenizer"], path: str, **kwargs: Any) -> "AbstractTokenizer":
        pass


    @abstractmethod
    def __len__(self) -> int:
        pass
