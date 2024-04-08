import argparse

from molgen.tokenizers.trainers.char_tokenizer_trainer import build_char_tokenizer
from molgen.tokenizers.trainers.bpe_tokenizer_trainer import build_bpe_tokenizer

def train_tokenizer(args: argparse.Namespace):
    if args.type == "Char":
        build_char_tokenizer(args.data_path, args.save_path, args.special_tokens)

    elif args.type == "BPE":
        build_bpe_tokenizer(args.data_path, args.save_path, args.special_tokens, args.vocab_size, verbose=True)

    else:
        raise ValueError(f"tokenizer type {args.type} is not Supported")

