import json
import os
from typing import Any, Dict, List, Optional, Set


def build_char_tokenizer(data_paths: List[str], save_path: str, add_special_tokens: bool, model_max_length: Optional[int]) -> None:
    unique_tokens: Set[str] = set()

    for path in data_paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} is invalid path")

        with open(path, "r") as f:
            texts = [line.strip() for line in f.readlines()]
            for text in texts:
                unique_tokens |= set(text)


    if add_special_tokens:
        tokens_to_ids = {"<pad>": 0, "<s>": 1, "</s>": 2}
    else:
        tokens_to_ids = {}

    offset = len(tokens_to_ids)
    for i, token in enumerate(unique_tokens):
        tokens_to_ids[token] = i + offset

    
    if not os.path.exists(save_path):
        os.makedirs(path, exist_ok=True)

    with open(f"{path}/config.json", "w") as f:
        config: Dict[str, Any] = {"type": "CharTokenizer", "kwargs": {}}
        if model_max_length is not None:
            config["kwargs"]["model_max_length"] = model_max_length

        json.dump(config, f)

    with open(f"{path}/vocab.json", "w") as f:
        json.dump(tokens_to_ids, f)

