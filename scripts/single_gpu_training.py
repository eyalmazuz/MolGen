import argparse

import tomllib
import torch
from torch.utils.data import DataLoader

from molgen.datasets.dataset_factory import get_dataset
from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.dataset_utils import LengthBatchSampler, PadCollate
from molgen.models.model_factory import get_model
from molgen.models.model_options import ModelType
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.training.train import pretrain_model
from molgen.utils.train_utils import setup_mixed_precision, setup_torch


def get_pretrain_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--model-type", type=str, required=True, choices=["GPT"], help="Type of model to use for training")  # noqa: E501
    parser.add_argument("--dataset-type", type=str, required=True, choices=["SMILES"], help="Type of dataset to use for training")  # noqa: E501
    parser.add_argument("--config-path", type=str, required=True, help="Path to the connfig containing training and model params")  # noqa: E501

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    train_config = config["train_config"]
    model_config = config["model_config"]

    print("Setting up torch")
    device = setup_torch(train_config["seed"], train_config["device"])
    ctx, scaler = setup_mixed_precision(train_config["device"], train_config["dtype"])

    print(f"Building model {args.model_type} and Dataset {args.dataset_type}")
    model_type = ModelType.from_str(args.model_type)
    dataset_type = DatasetType.from_str(args.dataset_type)

    model = get_model(model_type, model_config).to(device)
    tokenizer = get_tokenizer(args.tokenizer_path)
    train_dataset, val_dataset = get_dataset(dataset_type,
                          model_type,
                          dataset_path=args.data_path,
                          tokenizer=tokenizer)

    train_sampler = LengthBatchSampler(train_dataset, train_config["batch_size"], drop_last=False)
    val_sampler = LengthBatchSampler(val_dataset, train_config["batch_size"], drop_last=False)
    collate_fn = PadCollate(tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset,
                            batch_sampler=train_sampler,
                            collate_fn=collate_fn,
                            num_workers=train_config["num_workers"],
                            pin_memory=True)

    val_dataloader = DataLoader(val_dataset,
                            batch_sampler=val_sampler,
                            collate_fn=collate_fn,
                            num_workers=train_config["num_workers"],
                            pin_memory=True)

    print("Creating Optimizer")
    optimizer = model.configure_optimizers(train_config["weight_decay"],
                                           train_config["learning_rate"],
                                           train_config["betas"],
                                           device)

    if train_config["compile"]:
        print("Compiling model")
        model = torch.compile(model) # type: ignore

    print("Start training")
    model.train()  # Set the model to training mode
    pretrain_model(model, train_dataloader, val_dataloader, optimizer, ctx, scaler, train_config)

if __name__ == "__main__":
    args = get_pretrain_args()
    run_training(args)
