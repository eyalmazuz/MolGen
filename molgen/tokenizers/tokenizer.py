from abc import ABC, abstractmethod
from typing import Any, List, Type, Union

import torch


TokenizedData = Union[List[List[int]], torch.Tensor]


class AbstractTokenizer(ABC):

    @abstractmethod
    def encode(self,
               texts: Union[str, List[str]],
               padding: Union[str, bool],
               truncation: Union[str, bool],
               max_length: int,
               return_tensors: bool) -> TokenizedData:
        pass


    @abstractmethod
    def decode(self, encodings: TokenizedData, skip_special_tokens: bool) -> List[str]:
        pass


    @classmethod
    @abstractmethod
    def load_pretrained(cls: Type["AbstractTokenizer"], path: str, **kwargs: Any) -> "AbstractTokenizer":
        pass


    @abstractmethod
    def __len__(self) -> int:
        pass

    def get_pad_token_id(self) -> int:
        return 0
