import argparse

from molgen.train_tokenizer import train_tokenizer


def get_tokenizer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, choices=["Char", "BPE"], help="Type of tokenizer to train")
    parser.add_argument("--vocab_size", type=int, help="Vocab size for tokenizer of type BPE")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--data_path", type=str, nargs="+", required=True, help="Where to save the trained tokenizer")
    parser.add_argument("--pad_token", type=str, help="pad token to add to the tokenizer")
    parser.add_argument("--sep_token", type=str, help="sep token to add to the tokenizer")
    parser.add_argument("--bos_token", type=str, help="bos token to add to the tokenizer")
    parser.add_argument("--eos_token", type=str, help="eos token to add to the tokenizer")
    parser.add_argument("--extra_special_tokens", type=str, nargs="+", help="special tokens to add to the tokenizer")


    return parser.parse_args()


if __name__ == "__main__":
    args = get_tokenizer_args()
    if args.type == "Char" and args.vocab_size is not None:
        raise ValueError("Can't specify vocab size with char tokenizer")

    train_tokenizer(args)
