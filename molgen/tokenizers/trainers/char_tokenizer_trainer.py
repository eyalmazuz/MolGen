import json
import os
from typing import Any


def build_char_tokenizer(
    data_paths: list[str],
    save_path: str,
    bos_token: str | None = None,
    eos_token: str | None = None,
    pad_token: str | None = None,
    sep_token: str | None = None,
    extra_special_tokens: list[str] | None = None,
) -> None:
    unique_tokens: set[str] = set()

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
        config: dict[str, Any] = {"type": "CharTokenizer", "kwargs": {}}

        config["kwargs"]["bos_token"] = bos_token
        config["kwargs"]["eos_token"] = eos_token
        config["kwargs"]["pad_token"] = pad_token
        config["kwargs"]["sep_token"] = sep_token

        special_tokens = [bos_token, eos_token, pad_token, sep_token]
        if extra_special_tokens:
            special_tokens += extra_special_tokens

        special_to_id = {tok: len(tokens_to_ids) + i for i, tok in enumerate(special_tokens) if tok is not None}
        if special_to_id:
            config["kwargs"]["special_tokens"] = special_to_id

        json.dump(config, fd)

    with open(f"{save_path}/vocab.json", "w") as fd:
        json.dump(tokens_to_ids, fd)
