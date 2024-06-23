"""
This code is adapted from:
https://github.com/karpathy/minbpe
Under the MIT license
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from molgen.tokenizers.tokenizers_utils import get_stats, merge, render_token

def build_bpe_tokenizer(data_paths: List[str],
                        save_path: str,
                        vocab_size: int,
                        bos_token: Optional[str]=None,
                        eos_token: Optional[str]=None,
                        pad_token: Optional[str]=None,
                        sep_token: Optional[str]=None,
                        extra_special_tokens: Optional[List[str]]=None,
                        verbose: bool=False) -> None:
    texts = []
    for path in data_paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} is invalid path")

        with open(path, "r", encoding="utf-8") as fd:
            texts += [line.strip() for line in fd.readlines()]

    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # input text preprocessing
    ids = [list(ch.encode("utf-8")) for ch in texts]

    # iteratively merge the most common pairs to create new tokens
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
    for i in range(num_merges):
        # count up the number of times every consecutive pair appears
        stats: Dict[Tuple[int, int], int] = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        # find the pair with the highest count
        pair = max(stats, key=lambda pair: stats[pair])
        # mint a new token: assign it the next available id
        idx = 256 + i
        # replace all occurrences of pair in ids with idx
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
        # save the merge
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        # prints
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) had {stats[pair]} occurrences")


    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/config.json", "w") as fd:
        config: Dict[str, Any] = {"type": "BPETokenizer", "kwargs": {}}
        # config["kwargs"]["vocab_size"] = vocab_size # It's probably not needed but might be changed in the future
        config["kwargs"]["bos_token"] = bos_token
        config["kwargs"]["eos_token"] = eos_token
        config["kwargs"]["pad_token"] = pad_token
        config["kwargs"]["sep_token"] = sep_token

        special_tokens = [bos_token, eos_token, pad_token, sep_token]
        if extra_special_tokens:
            special_tokens += extra_special_tokens

        special_to_id = {tok: len(vocab) + i for i, tok in enumerate(special_tokens) if tok is not None}
        if special_to_id:
            config["kwargs"]["special_tokens"] = special_to_id

        json.dump(config, fd)

    with open(f"{save_path}/merges.txt", "w") as fd:
        for idx1, idx2 in merges:
            fd.write(f"{idx1} {idx2}\n")

    inverted_merges = {idx: pair for pair, idx in merges.items()}
    with open(f"{save_path}/vocab.txt", "w", encoding="utf-8") as fd:
        for idx, token in vocab.items():
            # note: many tokens may be partial utf-8 sequences
            # and cannot be decoded into valid strings. Here we're using
            # errors='replace' to replace them with the replacement char �.
            # this also means that we couldn't possibly use .vocab in load()
            # because decoding in this way is a lossy operation!
            s = render_token(token)
            # find the children of this token, if any
            if idx in inverted_merges:
                # if this token has children, render it nicely as a merge
                idx0, idx1 = inverted_merges[idx]
                s0 = render_token(vocab[idx0])
                s1 = render_token(vocab[idx1])
                fd.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
            else:
                # otherwise this is leaf token, just print it
                # (this should just be the first 256 tokens, the bytes)
                fd.write(f"[{s}] {idx}\n")
