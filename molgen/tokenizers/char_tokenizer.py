import json
import os
import re
from typing import Any

import torch

from molgen.tokenizers.tokenizer import AbstractTokenizer, TokenizedData


class CharTokenizer(AbstractTokenizer):
    def __init__(
        self,
        token2id: dict[str, int],
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        sep_token: str | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sep_token=sep_token,
            special_tokens=special_tokens,
        )
        self.tokens_to_ids = token2id
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}

    def __len__(self) -> int:
        return len(self.tokens_to_ids)

    def encode(self, texts: str | list[str], return_tensors: bool = False) -> TokenizedData:
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

    def decode(self, encodings: TokenizedData, skip_special_tokens: bool = False) -> list[str]:
        if isinstance(encodings[0], int) and isinstance(encodings, list):
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
