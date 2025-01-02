import json
import os

from molgen.tokenizers.bpe_tokenizer import BPETokenizer
from molgen.tokenizers.char_tokenizer import CharTokenizer
from molgen.tokenizers.tokenizer import AbstractTokenizer


def get_tokenizer(path: str) -> AbstractTokenizer:
    if not os.path.isdir(path) or not os.path.exists(path):
        raise ValueError(f"{path} is not valid tokenizer path")

    if not os.path.exists(f"{path}/config.json"):
        raise ValueError(f"config.json not found in {path}")

    with open(f"{path}/config.json", "r") as fd:
        config = json.load(fd)

    tokenizer_type = config["type"]
    kwargs = config.get("kwargs", {})

    tokenizer: AbstractTokenizer
    match tokenizer_type:
        case "CharTokenizer":
            tokenizer = CharTokenizer.load_pretrained(path, **kwargs)
        case "BPETokenizer":
            tokenizer = BPETokenizer.load_pretrained(path, **kwargs)
        case _:
            raise ValueError(f"Invalid tokenizer type {tokenizer_type}")

    return tokenizer
