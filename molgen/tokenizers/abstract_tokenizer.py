from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

import torch


TokenizedData = Union[List[List[int]], torch.Tensor]


class AbstractTokenizer(ABC):

    @abstractmethod
    def encode(self,
               texts: Union[str, List[str]],
               padding: Union[str, bool],
               truncation: Union[str, bool],
               max_length: int,
               add_bos_token: bool,
               add_eos_token: bool,
               return_tensors: bool) -> TokenizedData:
        pass

    
    @abstractmethod
    def decode(self, encodings: TokenizedData) -> List[str]:
        pass


    @abstractmethod
    @classmethod
    def load_pretrained(cls: Type["AbstractTokenizer"], path: str, **kwargs: Any) -> "AbstractTokenizer":
        pass


    @abstractmethod
    def save_pretrained(self, path: str) -> None:
        pass

    
    @abstractmethod
    def __len__(self) -> int:
        pass
