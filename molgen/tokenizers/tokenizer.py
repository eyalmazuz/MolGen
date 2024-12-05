from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import torch

TokenizedData: TypeAlias = list[list[int]] | torch.Tensor


class AbstractTokenizer(ABC):
    def __init__(
        self,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        sep_token: str | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        if special_tokens:
            self.special_tokens = special_tokens
            self.inverse_special_tokens = {id_: token for token, id_ in self.special_tokens.items()}

        self.bos_token_ = bos_token
        self.eos_token_ = eos_token
        self.pad_token_ = pad_token
        self.sep_token_ = sep_token

        if pad_token is None and eos_token is not None:
            print("pad token is not defined will default to eos token if available")

        if sep_token is None and eos_token is not None:
            print("sep token is not defined will default to eos token if available")

    @abstractmethod
    def encode(self, texts: str | list[str], return_tensors: bool = False) -> TokenizedData:
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

    @property
    def bos_token_id(self) -> int:
        if self.bos_token_ is not None:
            return self.special_tokens[self.bos_token_]
        else:
            raise ValueError("bos token is not defined")

    @property
    def bos_token(self) -> str:
        if self.bos_token_ is not None:
            return self.bos_token_
        else:
            raise ValueError("bos token is not defined")

    @property
    def eos_token_id(self) -> int:
        if self.eos_token_ is not None:
            return self.special_tokens[self.eos_token_]
        else:
            raise ValueError("eos token is not defined")

    @property
    def eos_token(self) -> str:
        if self.eos_token_ is not None:
            return self.eos_token_
        else:
            raise ValueError("eos token is not defined")

    @property
    def pad_token_id(self) -> int:
        if self.pad_token_ is not None:
            return self.special_tokens[self.pad_token_]
        elif self.pad_token_ is None and self.eos_token_ is not None:
            return self.special_tokens[self.eos_token_]
        else:
            raise ValueError("both pad token and eos token are not defined")

    @property
    def pad_token(self) -> str:
        if self.pad_token_ is not None:
            return self.pad_token_
        elif self.pad_token_ is None and self.eos_token_ is not None:
            return self.eos_token_
        else:
            raise ValueError("both pad token and eos token are not defined")

    @property
    def sep_token_id(self) -> int:
        if self.sep_token_ is not None:
            return self.special_tokens[self.sep_token_]
        elif self.sep_token_ is None and self.eos_token_ is not None:
            return self.special_tokens[self.eos_token_]
        else:
            raise ValueError("both sep token and eos token are not defined")

    @property
    def sep_token(self) -> str:
        if self.sep_token_ is not None:
            return self.sep_token_
        elif self.sep_token_ is None and self.eos_token_ is not None:
            return self.eos_token_
        else:
            raise ValueError("both sep token and eos token are not defined")
