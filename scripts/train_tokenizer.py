import argparse

from molgen.tokenizers.trainers.bpe_tokenizer_trainer import build_bpe_tokenizer
from molgen.tokenizers.trainers.char_tokenizer_trainer import build_char_tokenizer


def get_tokenizer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, choices=["Char", "BPE"], help="Type of tokenizer to train")
    parser.add_argument("--vocab-size", type=int, help="Vocab size for tokenizer of type BPE")
    parser.add_argument("--save-path", type=str, required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--data-path", type=str, nargs="+", required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--pad-token", type=str, help="pad token to add to the tokenizer")
    parser.add_argument("--sep-token", type=str, help="sep token to add to the tokenizer")
    parser.add_argument("--bos-token", type=str, help="bos token to add to the tokenizer")
    parser.add_argument("--eos-token", type=str, help="eos token to add to the tokenizer")
    parser.add_argument("--extra-special-tokens", type=str, nargs="+", help="special tokens to add to the tokenizer")

    return parser.parse_args()


def train_tokenizer(args: argparse.Namespace):
    match args.type.lower():
        case "char":
            build_char_tokenizer(
                args.data_path,
                args.save_path,
                args.bos_token,
                args.eos_token,
                args.pad_token,
                args.sep_token,
                args.extra_special_tokens,
            )

        case "bpe":
            build_bpe_tokenizer(
                args.data_path,
                args.save_path,
                args.vocab_size,
                args.bos_token,
                args.eos_token,
                args.pad_token,
                args.sep_token,
                args.extra_special_tokens,
                verbose=True,
            )

        case _:
            raise ValueError(f"tokenizer type {args.type} is not Supported")


if __name__ == "__main__":
    args = get_tokenizer_args()
    if args.type == "Char" and args.vocab_size is not None:
        raise ValueError("Can't specify vocab size with char tokenizer")

    train_tokenizer(args)
