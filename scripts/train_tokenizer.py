from molgen.utils.args.tokenizer_args import get_tokenizer_args, validate_tokenizer_args
from molgen.train_tokenizer import train_tokenizer

if __name__ == "__main__":
    args = get_tokenizer_args()
    validate_tokenizer_args(args)
    train_tokenizer(args)
