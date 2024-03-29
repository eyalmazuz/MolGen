import json
import os
from typing import Dict, List, Optional, Type, Union

import torch

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer


TokenizedData = Union[List[List[int]], torch.Tensor]


class CharTokenizer(AbstractTokenizer):
    
    def __init__(self,
                 token2id: Dict[str, int],
                 model_max_length: int=256) -> None:
        self.tokens_to_ids = token2id
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}    
        self.model_max_length = model_max_length


    def encode(self,
               texts: Union[str, List[str]],
               padding: Union[str, bool]=False,
               truncation: Union[str, bool]=False,
               max_length: Optional[int]=None,
               add_bos_token: bool=False,
               add_eos_token: bool=False,
               return_tensors: bool=False) -> TokenizedData:

        if isinstance(texts, str):
            texts = [texts]

        encodings: TokenizedData = []
        for text in texts:
            encoding = [self.tokens_to_ids[token] for token in list(text)]
            if truncation:
                encoding = encoding[:self.model_max_length]
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
            padded_encodings: TokenizedData = []
            for encoding in encodings:
                encoding = encoding + [self.tokens_to_ids["<pad>"]] * (max_length - len(encoding))

            padded_encodings.append(encoding)
            
            encodings = padded_encodings

        if return_tensors:
            encodings = torch.tensor(encodings)

        return encodings

    def decode(self, encodings: TokenizedData) -> List[str]:
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
    def load_pretrained(cls: Type["CharTokenizer"], path: str) -> "CharTokenizer":
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if os.path.isdir(path) and not os.path.exists(f"{path}/vocab.json"):
            raise ValueError(f"{path} doesn't contain vocab.json file")

        with open(f"{path}/vocab.json", "r") as f:
            tokens_to_ids = json.load(f)

        return cls(tokens_to_ids)

 
    def save_pretrained(self, path: str) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.tokens_to_ids, f)

