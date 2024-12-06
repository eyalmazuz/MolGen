import argparse
import os

import tomllib
import torch
from torch.distributed import destroy_process_group, init_process_group

from molgen.datasets.dataset_factory import get_dataset
from molgen.datasets.dataset_utils import prepare_data_for_training
from molgen.models.model_factory import get_model
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.training.train import pretrain_model
from molgen.utils.train_utils import setup_mixed_precision, setup_torch
from molgen.utils.utils import get_world_size, is_distributed_run, is_master_process


def get_pretrain_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the model")
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["GPT"], help="Type of model to use for training"
    )
    parser.add_argument(
        "--dataset-type", type=str, required=True, choices=["SMILES"], help="Type of dataset to use for training"
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the connfig containing training and model params"
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    train_config = config["train_config"]
    model_config = config["model_config"]

    if is_distributed_run():
        init_process_group(backend="nccl")
        ddp_world_size = get_world_size()
        train_config["gradient_accumulation_steps"] //= ddp_world_size

    print("Setting up torch")
    device = setup_torch(train_config["seed"], train_config["device"])
    ctx, scaler = setup_mixed_precision(train_config["device"], train_config["dtype"])

    print(f"Building model {args.model_type} and Dataset {args.dataset_type}")

    model = get_model(args.model_type, model_config).to(device)
    tokenizer = get_tokenizer(args.tokenizer_path)
    train_dataset, val_dataset = get_dataset(
        args.dataset_type, args.model_type, dataset_path=args.data_path, tokenizer=tokenizer
    )

    train_dataloader, val_dataloader = prepare_data_for_training(
        train_dataset, val_dataset, tokenizer.pad_token_id, train_config
    )

    print("Creating Optimizer")
    optimizer = model.configure_optimizers(
        train_config["weight_decay"], train_config["learning_rate"], train_config["betas"], device
    )

    if train_config["compile"]:
        print("Compiling model")
        model = torch.compile(model)  # type: ignore

    if train_config["wandb_log"] and is_master_process():
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", None),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=f"{args.model_type}_{args.dataset_type}",
            config=config,
        )

    print("Start training")
    pretrain_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        ctx,
        scaler,
        train_config["checkpoint_dir"],
        train_config["max_steps"],
        train_config["grad_clip"],
        train_config["gardient_accumulation_steps"],
        train_config["eval_every"],
        train_config["log_every"],
        train_config["wandb_log"],
        device,
    )

    if is_distributed_run():
        destroy_process_group()


if __name__ == "__main__":
    args = get_pretrain_args()
    run_training(args)
