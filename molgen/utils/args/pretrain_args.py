import argparse


def get_model_parser() -> argparse.ArgumentParser:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the connfig containing training and model params")

    return parser
