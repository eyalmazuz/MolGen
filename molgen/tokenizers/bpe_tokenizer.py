import os
import re
from typing import Any

import torch

from molgen.tokenizers.tokenizer import AbstractTokenizer, TokenizedData
from molgen.tokenizers.tokenizers_utils import get_stats, merge


class BPETokenizer(AbstractTokenizer):
    def __init__(
        self,
        merges: dict[tuple[int, int], int],
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
        self.merges = merges

        # build the vocab back from the merges
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def __len__(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def encode(self, texts: str | list[str], return_tensors: bool = False) -> TokenizedData:
        if isinstance(texts, str):
            texts = [texts]

        encodings: list[list[int]] = []
        for text in texts:
            if self.special_tokens is not None and len(self.special_tokens) > 0:
                special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
                chunks: list[str] = re.split(special_pattern, text)
            else:
                chunks = [text]

            encoding = []
            for chunk in chunks:
                if chunk == "":
                    continue
                else:
                    if self.special_tokens is not None and chunk in self.special_tokens:
                        encoding.append(self.special_tokens[chunk])
                    else:
                        encoding += self.__encode_chunk(chunk.encode("utf-8"))

            encodings.append(encoding)

        if return_tensors:
            return torch.tensor(encodings)

        return encodings

    def __encode_chunk(self, text_bytes: bytes) -> list[int]:
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats: dict[tuple[int, int], int] = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, encodings: TokenizedData, skip_special_tokens: bool = False) -> list[str]:
        if isinstance(encodings[0], int):
            encodings = [encodings]

        if isinstance(encodings, torch.Tensor):
            encodings = encodings.cpu().numpy().tolist()

        texts = []
        for encoding in encodings:
            # given ids (list of integers), return Python string
            part_bytes = []
            for idx in encoding:
                if skip_special_tokens and self.special_tokens is not None and idx in self.inverse_special_tokens:
                    continue
                elif not skip_special_tokens and self.special_tokens is not None and idx in self.inverse_special_tokens:
                    part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
                elif idx in self.vocab:
                    part_bytes.append(self.vocab[idx])
                else:
                    raise ValueError(f"invalid token id: {idx}")
            text_bytes = b"".join(part_bytes)
            text = text_bytes.decode("utf-8", errors="replace")
            texts.append(text)

        return texts

    @classmethod
    def load_pretrained(cls: type["BPETokenizer"], path: str, **kwargs: Any) -> "BPETokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/merges.txt"):
            raise ValueError(f"{path} doesn't contain merges.txt file")

        merges = {}
        idx = 256
        with open(f"{path}/merges.txt") as fd:
            for line in fd:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        return cls(merges, **kwargs)
