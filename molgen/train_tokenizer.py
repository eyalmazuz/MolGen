import argparse

from molgen.tokenizers.trainers.char_tokenizer_trainer import build_char_tokenizer

def train_tokenizer(args: argparse.Namespace):
    if args.type == "Char":
        build_char_tokenizer(args.data_path, args.save_path, args.add_special_tokens, args.model_max_length)

    elif args.type == "BPE":
        build_bpe_tokenizer(args.data_path, args.save_path, args.vocab_size, args.add_special_tokens)

    else:
        raise ValueError(f"tokenizer type {args.type} is not Supported")

