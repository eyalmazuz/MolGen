import argparse


def get_pretrain_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save the model")
    parser.add_argument("--model_type", type=str, required=True, choices=["GPT", "DT"], help="Type of model to use for training")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["SMILES", "DT_SMILES"], help="Type of dataset to use for training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the connfig containing training and model params")

    # Wandb parameters to log results
    parser.add_argument('--wandb_key', type=str, help='wandb api key for user login', default=None)
    parser.add_argument('--wandb_proj', type=str, default='DecisionMol',
                        help='name of wandb project to upload results')
    parser.add_argument('--wandb_entity', type=str, default='bgu-sise',
                        help='wandb entity associated with the project')

    return parser.parse_args()
