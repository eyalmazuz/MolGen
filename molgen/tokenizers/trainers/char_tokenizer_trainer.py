import json
import os
from typing import Any, Dict, List, Optional, Set


def build_char_tokenizer(data_paths: List[str], save_path: str, special_tokens: Optional[List[str]]) -> None:
    unique_tokens: Set[str] = set()

    for path in data_paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} is invalid path")

        with open(path, "r") as fd:
            texts = [line.strip() for line in fd.readlines()]
            for text in texts:
                unique_tokens |= set(text)

    tokens_to_ids = {token: i for i, token in enumerate(unique_tokens)}

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/config.json", "w") as fd:
        config: Dict[str, Any] = {"type": "CharTokenizer", "kwargs": {}}
        if special_tokens:
            config["kwargs"]["special_tokens"] = special_tokens

        json.dump(config, fd)

    with open(f"{save_path}/vocab.json", "w") as fd:
        json.dump(tokens_to_ids, fd)
