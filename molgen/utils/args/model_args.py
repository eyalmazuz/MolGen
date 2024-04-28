import argparse


def get_model_parser() -> argparse.ArgumentParser:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size of the model")
    parser.add_argument("--block_size", type=int, default=512, help="Number of maximum tokens the model can handle")
    parser.add_argument("--n_embd", type=int, default=768, help="Model embedding size")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, help="Number of decoder layers")
    parser.add_argument("--embd_pdrop", type=float, default=0.1, help="embedding dropout probability")
    parser.add_argument("--attn_pdrop", type=float, default=0.1, help="attention dropout probability")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="residual dropout probability")

    return parser
