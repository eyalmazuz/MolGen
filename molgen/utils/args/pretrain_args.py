import argparse


def get_pretrain_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save the model")
    parser.add_argument("--model_type", type=str, required=True, options=["GPT"], help="Type of model to use for training")
    parser.add_argument("--dataset_type", type=str, required=True, options=["SMILES"], help="Type of dataset to use for training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the connfig containing training and model params")

    return parser.parse_args()
