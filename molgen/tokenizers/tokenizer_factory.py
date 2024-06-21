import json
import os

from molgen.tokenizers.abstract_tokenizer import AbstractTokenizer
from molgen.tokenizers.char_tokenizer import CharTokenizer
from molgen.tokenizers.bpe_tokenizer import BPETokenizer

def get_tokenizer(path: str) -> AbstractTokenizer:
    if not os.path.isdir(path) or not os.path.exists(path):
        raise ValueError(f"{path} is not valid tokenizer path")

    if not os.path.exists(f"{path}/config.json"):
        raise ValueError(f"config.json not found in {path}")

    with open(f"{path}/config.json", "r") as fd:
        config = json.load(fd)

    tokenizer_type = config["type"]
    kwargs = config["kwargs"] if "kwargs" in config else {}

    tokenizer: AbstractTokenizer
    if tokenizer_type == "CharTokenizer":
        tokenizer = CharTokenizer.load_pretrained(path, **kwargs)

    if tokenizer_type == "BPETokenizer":
        tokenizer = BPETokenizer.load_pretrained(path, **kwargs)

    return tokenizer
