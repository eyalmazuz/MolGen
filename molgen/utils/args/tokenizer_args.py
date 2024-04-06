import argparse


def parse_tokenizer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, choices=["Char", "BPE"], help="Type of tokenizer to train")
    parser.add_argument("--vocab_size", type=int, help="Vocab size for tokenizer of type BPE")
    parser.add_argument("--model_max_length", type=int, help="max length the tokenizer will suport")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--data_path", type=str, nargs="+", required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--add_special_tokens", action="store_true", help="Whether to add special tokens to the tokenizer")


    return parser.parse_args()


def validate_tokenizer_args(args: argparse.Namespace) -> None:
    if args.type == "Char" and args.vocab_size is not None:
        raise ValueError("Can't specify vocab size with char tokenizer")

