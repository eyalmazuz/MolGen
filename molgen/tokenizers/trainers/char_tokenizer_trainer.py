import json
import os
from typing import Any, Dict, List, Optional, Set


def build_char_tokenizer(data_paths: List[str],
                         save_path: str,
                         bos_token: Optional[str]=None,
                         eos_token: Optional[str]=None,
                         pad_token: Optional[str]=None,
                         extra_special_tokens: Optional[List[str]]=None) -> None:
    unique_tokens: Set[str] = set()

    for path in data_paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} is invalid path")

        with open(path, "r") as f:
            texts = [line.strip() for line in f.readlines()]
            for text in texts:
                unique_tokens |= set(text)

    tokens_to_ids = {token: i for i, token in enumerate(unique_tokens)}

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/config.json", "w") as f:
        config: Dict[str, Any] = {"type": "CharTokenizer", "kwargs": {}}

        config["kwargs"]["bos_token"] = bos_token
        config["kwargs"]["eos_token"] = eos_token
        config["kwargs"]["pad_token"] = pad_token

        special_tokens = [bos_token, eos_token, pad_token]
        if extra_special_tokens:
            special_tokens += extra_special_tokens

        special_to_id = {tok: len(tokens_to_ids) + i for i, tok in enumerate(special_tokens) if tok is not None}
        if special_to_id:
            config["kwargs"]["special_tokens"] = special_to_id

        json.dump(config, f)

    with open(f"{save_path}/vocab.json", "w") as f:
        json.dump(tokens_to_ids, f)
