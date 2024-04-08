import json
import os
import re
from typing import Any, Dict, List, Optional, Type, Union
import warnings

import torch

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer, TokenizedData


class CharTokenizer(AbstractTokenizer):

    def __init__(self,
                 token2id: Dict[str, int],
                 special_tokens: Optional[List[str]]=None,) -> None:
        self.special_tokens = special_tokens
        self.tokens_to_ids = token2id
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}    

        if self.special_tokens is not None:
            for i, special_token in enumerate(self.special_tokens):
                self.tokens_to_ids[special_token] = len(self.tokens_to_ids) + i
            
        self.ids_to_tokens = {id_: token for token, id_ in self.tokens_to_ids.items()}    

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
                        encoding.append(self.tokens_to_ids[chunk])
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
            token_list = []
            for idx in encoding:
                if skip_special_tokens and self.special_tokens is not None and self.ids_to_tokens[idx] in self.special_tokens:
                    continue
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

