import json
import os
from typing import Any, Dict, List, Type, Union

import torch

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer, TokenizedData


class BPETokenizer(AbstractTokenizer):

    
    def __init__(self, tokens_to_ids: Dict[str, int], merges: Dict[str, List[str]], model_max_length: int) -> None:
        self.tokens_to_ids = tokens_to_ids
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}
        self.merges = {tuple(pair): merge for merge, pair in merges.items()}
        self.model_max_length = model_max_length

    
    def encode(self,
               texts: Union[str, List[str]],
               padding: Union[str, bool],
               truncation: Union[str, bool],
               max_length: int,
               add_bos_token: bool,
               add_eos_token: bool,
               return_tensors: bool) -> TokenizedData:

        if isinstance(texts, str):
            texts = [texts]

        encodings: List[List[int]] = []

        for text in texts:
            splitted_text = text.split(" ") 
            splits = [list(word) for word in splitted_text]

            for pair, merge in self.merges.items():
                for idx, split in enumerate(splits):
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    splits[idx] = split

            tokens = sum(splits, [])
            
            encoding = [self.tokens_to_ids[token] for token in tokens]

            if add_bos_token:
                encoding = [self.tokens_to_ids["<s>"]] + encoding
            if add_eos_token:
                encoding = encoding + [self.tokens_to_ids["</s>"]] 

            encodings.append(encoding)

        if padding or padding == "longest":
            max_length = max(map(len, encodings))

        elif padding == "max_length":
            if max_length is None:
                max_length = self.model_max_length

        if max_length is not None:
            padded_encodings: List[List[int]] = []
            for encoding in encodings:
                encoding = encoding + [self.tokens_to_ids["<pad>"]] * (max_length - len(encoding))

            padded_encodings.append(encoding)
            
            encodings = padded_encodings

        if return_tensors:
            return torch.tensor(encodings)

        return encodings


    def decode(self, encodings: TokenizedData, skip_special_tokens: bool) -> List[str]:
        if isinstance(encodings[0], int):
            encodings = [encodings]

        if isinstance(encodings, torch.Tensor):
            encodings = encodings.cpu().numpy().tolist()

        texts = []
        for encoding in encodings:
            text = "".join(self.ids_to_tokens[id_] for id_ in encoding)
            texts.append(text)

        return texts 


    @classmethod
    def load_pretrained(cls: Type["BPETokenizer"], path: str, **kwargs: Any) -> "BPETokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/vocab.json"):
            raise ValueError(f"{path} doesn't contain vocab.json file")

        if os.path.isdir(path) and not os.path.exists(f"{path}/merges.json"):
            raise ValueError(f"{path} doesn't contain merges.json file")

        with open(f"{path}/vocab.json", "r") as f:
            tokens_to_ids = json.load(f)

        with open(f"{path}/merges.json", "r") as f:
            merges = json.load(f)

        return cls(tokens_to_ids, merges, **kwargs)


    def save_pretrained(self, path: str) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        config = {
                "type": "BPETokenizer",
                "kwargs": {
                    "model_max_length": self.model_max_length
                    }
                }

        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.tokens_to_ids, f)

        with open(f"{path}/merges.json", "w") as f:
            json.dump(self.merges, f)

