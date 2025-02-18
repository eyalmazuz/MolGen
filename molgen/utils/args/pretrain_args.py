import argparse


def get_pretrain_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--model-type", type=str, required=True, choices=["GPT", "DT"], help="Type of model to use for training")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["SMILES", "DT_SMILES", "SELFIES", "DT_SELFIES"], help="Type of dataset to use for training")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the connfig containing training and model params")

    # Wandb parameters to log results
    parser.add_argument('--wandb-key', type=str, help='wandb api key for user login', default=None)
    parser.add_argument('--wandb-proj', type=str, default='DecisionMol',
                        help='name of wandb project to upload results')
    parser.add_argument('--wandb-entity', type=str, default='bgu-sise',
                        help='wandb entity associated with the project')
    parser.add_argument('--wandb-name', type=str, help='wandb run name', default=None)


    return parser.parse_args()
