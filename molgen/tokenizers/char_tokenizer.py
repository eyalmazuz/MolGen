import json
import os
import re
from typing import Any

import torch

from molgen.tokenizers.tokenizer import AbstractTokenizer, TokenizedData


class CharTokenizer(AbstractTokenizer):
    def __init__(self,
                 token2id: dict[str, int],
                 bos_token: str | None = None,
                 eos_token: str | None = None,
                 pad_token: str | None = None,
                 sep_token: str | None = None,
                 special_tokens: dict[str, int] | None = None) -> None:
        self.tokens_to_ids = token2id
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}

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

    def __len__(self) -> int:
        return len(self.tokens_to_ids)

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

    def encode(self,
               texts: str | list[str],
               return_tensors: bool = False) -> TokenizedData:
        if isinstance(texts, str):
            texts = [texts]

        encodings: list[list[int]] = []
        for text in texts:
            if self.special_tokens is not None and len(self.special_tokens) > 0:
                special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
                chunks: list[str] = re.split(special_pattern, text)
            else:
                chunks = list(text)
            encoding = []
            for chunk in chunks:
                if chunk == "":
                    continue
                else:
                    if self.special_tokens is not None and chunk in self.special_tokens:
                        encoding.append(self.special_tokens[chunk])
                    else:
                        encoding += [self.tokens_to_ids[token] for token in chunk]

            encodings.append(encoding)

        if return_tensors:
            return torch.tensor(encodings)

        return encodings

    def encode_selfies(self,
                       texts: str | list[str],
                       return_tensors: bool = False) -> TokenizedData:

        if isinstance(texts, str):
            texts = [texts]

        encodings: list[list[int]] = []
        for text in texts:
            if self.special_tokens is not None and len(self.special_tokens) > 0:
                special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
                chunks: list[str] = re.split(special_pattern, text)
            else:
                chunks = list(text)
            encoding = []
            for chunk in chunks:
                if chunk == "":
                    continue
                else:
                    if self.special_tokens is not None and chunk in self.special_tokens:
                        encoding.append(self.special_tokens[chunk])
                    else:
                        encoding += [self.tokens_to_ids[token] for token in re.findall(r'\[.*?]|.', chunk)]

            encodings.append(encoding)

        if return_tensors:
            return torch.tensor(encodings)

        return encodings

    def decode(self, encodings: TokenizedData, skip_special_tokens: bool = False) -> list[str]:
        if isinstance(encodings[0], int):
            encodings = [encodings]

        if isinstance(encodings, torch.Tensor):
            encodings = encodings.cpu().numpy().tolist()

        texts = []
        for encoding in encodings:
            token_list = []
            for idx in encoding:
                if skip_special_tokens and self.special_tokens is not None and idx in self.inverse_special_tokens:
                    continue
                elif not skip_special_tokens and self.special_tokens is not None and idx in self.inverse_special_tokens:
                    token_list.append(self.inverse_special_tokens[idx])
                elif idx in self.ids_to_tokens:
                    token_list.append(self.ids_to_tokens[idx])
                else:
                    raise ValueError(f"invalid token id: {idx}")
            text = "".join(token_list)
            texts.append(text)

        return texts

    @classmethod
    def load_pretrained(cls: type["CharTokenizer"], path: str, **kwargs: Any) -> "CharTokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/vocab.json"):
            raise ValueError(f"{path} doesn't contain vocab.json file")

        with open(f"{path}/vocab.json") as fd:
            tokens_to_ids = json.load(fd)

        return cls(tokens_to_ids, **kwargs)
