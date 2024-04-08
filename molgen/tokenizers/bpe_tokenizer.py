import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings

import torch

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer, TokenizedData
from molgen.tokenizers.tokenizers_utils import get_stats, merge


class BPETokenizer(AbstractTokenizer):
    
    def __init__(self, merges: Dict[Tuple[int, int], int], special_tokens: Optional[List[str]]=None) -> None:
        self.merges = merges

        # build the vocab back from the merges
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        # add special tokens if they exists
        if special_tokens is not None:
            self.inverse_special_tokens: Dict[int, str] = {}
            self.special_tokens: Dict[str, int] = {}
            idx = len(self.vocab)
            for offset, special in enumerate(special_tokens):
                self.inverse_special_tokens[idx+offset] = special
                self.special_tokens[special] = idx + offset


    def __len__(self) -> int:
        return len(self.vocab)


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

        if (isinstance(padding, bool) and padding) or padding == "longest":
            max_length = max(map(len, encodings))

        if padding == "max_length":
            if max_length is None:
                warnings.warn("when using padding='max_length' length is needed to be specified by the max_length argument defaulting to 512")
                max_length = 512

        if max_length is not None:
            padded_encodings: List[List[int]] = []
            for encoding in encodings:
                encoding = encoding + [self.special_tokens["<pad>"]] * (max_length - len(encoding))

            padded_encodings.append(encoding)
            
            encodings = padded_encodings

        if return_tensors:
            return torch.tensor(encodings)

        return encodings


    def __encode_chunk(self, text_bytes: bytes) -> List[int]:
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats: Dict[Tuple[int, int], int] = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def decode(self, encodings: TokenizedData, skip_special_tokens: bool=False) -> List[str]:
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
                    part_bytes.append(self.vocab[idx])
                elif idx in self.vocab:
                    part_bytes.append(self.vocab[idx])
                else:
                    raise ValueError(f"invalid token id: {idx}")
            text_bytes = b"".join(part_bytes)
            text = text_bytes.decode("utf-8", errors="replace")
            texts.append(text)

        return texts 


    @classmethod
    def load_pretrained(cls: Type["BPETokenizer"], path: str, **kwargs: Any) -> "BPETokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/merges.txt"):
            raise ValueError(f"{path} doesn't contain merges.txt file")

        merges = {}
        idx = 256
        with open(f"{path}/merges.txt", "r") as f:
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        return cls(merges, **kwargs)

