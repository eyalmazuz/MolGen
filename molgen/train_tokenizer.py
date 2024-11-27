import argparse

from molgen.tokenizers.trainers.bpe_tokenizer_trainer import build_bpe_tokenizer
from molgen.tokenizers.trainers.char_tokenizer_trainer import build_char_tokenizer


def train_tokenizer(args: argparse.Namespace):
    match args.type.lower():
        case "char":
            build_char_tokenizer(args.data_path,
                                args.save_path,
                                args.bos_token,
                                args.eos_token,
                                args.pad_token,
                                args.sep_token,
                                args.extra_special_tokens)

        case "bpe":
            build_bpe_tokenizer(args.data_path,
                                args.save_path,
                                args.vocab_size,
                                args.bos_token,
                                args.eos_token,
                                args.pad_token,
                                args.sep_token,
                                args.extra_special_tokens,
                                verbose=True)

        case _:
            raise ValueError(f"tokenizer type {args.type} is not Supported")
