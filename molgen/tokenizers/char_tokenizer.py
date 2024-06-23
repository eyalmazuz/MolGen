import json
import os
import re
from typing import Any, Dict, List, Optional, Type, Union
import warnings

import torch

from molgen.tokenizers.tokenizer import AbstractTokenizer, TokenizedData


class CharTokenizer(AbstractTokenizer):

    def __init__(self,
                 token2id: Dict[str, int],
                 bos_token: Optional[str]=None,
                 eos_token: Optional[str]=None,
                 pad_token: Optional[str]=None,
                 sep_token: Optional[str]=None,
                 special_tokens: Optional[Dict[str, int]]=None) -> None:
        self.tokens_to_ids = token2id
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}

        self.special_tokens = {}
        self.inverse_special_tokens = {}
        if special_tokens:
            self.special_tokens = special_tokens
            self.inverse_special_tokens = {id_: token for token, id_ in self.special_tokens.items()}

        self.bos_token_  = bos_token
        self.eos_token_  = eos_token
        self.pad_token_  = pad_token
        self.sep_token_  = sep_token

        if pad_token is None:
            print("pad token is not defined will default to eos token if available")

        if sep_token is None:
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
               texts: Union[str, List[str]],
               padding: Union[str, bool]=False,
               truncation: Union[str, bool]=False,
               max_length: Optional[int]=None,
               return_tensors: bool=False) -> TokenizedData:
        if isinstance(texts, str):
            texts = [texts]

        encodings: List[List[int]] = []
        for text in texts:
            if self.special_tokens is not None and len(self.special_tokens) > 0:
                special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
                chunks: List[str] = re.split(special_pattern, text)
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

            if truncation and max_length is not None:
                encoding = encoding[:max_length]

            encodings.append(encoding)

        if (isinstance(padding, bool) and padding) or padding == "longest":
            max_length = max(map(len, encodings))

        if padding == "max_length":
            if max_length is None:
                warnings.warn("when using padding='max_length' length is needed to be specified by the max_length argument defaulting to 512")
                max_length = 512

        if max_length is not None:
            padded_encodings: List[List[int]] = []
            for encoding in encodings:
                encoding = encoding + [self.special_tokens[self.pad_token]] * (max_length - len(encoding))
                padded_encodings.append(encoding)

            encodings = padded_encodings

        if return_tensors:
            return torch.tensor(encodings)

        return encodings


    def decode(self, encodings: TokenizedData, skip_special_tokens: bool=False) -> List[str]:
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
    def load_pretrained(cls: Type["CharTokenizer"], path: str, **kwargs: Any) -> "CharTokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/vocab.json"):
            raise ValueError(f"{path} doesn't contain vocab.json file")

        with open(f"{path}/vocab.json", "r") as f:
            tokens_to_ids = json.load(f)

        return cls(tokens_to_ids, **kwargs)
