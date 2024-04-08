import json
import os
import re
from typing import Any, Dict, List, Optional, Type, Union

import torch

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer, TokenizedData


class CharTokenizer(AbstractTokenizer):
    def __init__(self,
                 token2id: Dict[str, int],
                 special_tokens: Optional[List[str]]=None,
                 model_max_length: int=256) -> None:
        self.tokens_to_ids = token2id
        self.special_tokens = special_tokens
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}    
        self.model_max_length = model_max_length


    def __len__(self) -> int:
        return len(self.tokens_to_ids)


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
            if self.special_tokens:
                special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
                chunks = re.split(special_pattern, text)
            else:
                chunks = list(text)
            encoding = []
            for chunk in chunks:
                if chunk == "":
                    continue
                else:
                    if chunk in self.special_tokens:
                        encoding.append(self.tokens_to_ids[chunk])
                    else:
                        encoding += [self.tokens_to_ids[token] for token in chunk]
            # encoding = [self.tokens_to_ids[token] for token in special_chunks if token != ""]
            if truncation:
                encoding = encoding[:self.model_max_length]
            encodings.append(encoding)
        
        if (isinstance(padding, bool) and padding) or padding == "longest":
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


    def decode(self, encodings: TokenizedData, skip_special_tokens: bool=False) -> List[str]:
        if isinstance(encodings[0], int):
            encodings = [encodings]

        if isinstance(encodings, torch.Tensor):
            encodings = encodings.cpu().numpy().tolist()

        texts = []
        for encoding in encodings:
            text = "".join(self.ids_to_tokens[id_] if not skip_special_tokens or self.special_tokens is not None and self.ids_to_tokens[id_] not in self.special_tokens else "" for id_ in encoding)
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

 
    def save_pretrained(self, path: str) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        config = {
                "type": "CharTokenizer",
                "kwargs": {
                    "model_max_length": self.model_max_length,
                    "special_tokens": self.special_tokens,
                    }
                }

        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.tokens_to_ids, f)

