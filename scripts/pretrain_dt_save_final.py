import argparse
import os

import tomllib
import torch

from molgen.datasets.dataset_factory import get_dataset
from molgen.datasets.dataset_utils import prepare_data_for_training
from molgen.models.model_factory import get_model
from molgen.models.model_options import ModelType
from molgen.rewards.reward_factory import get_rewards
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.training.train_dt import run_dt_training
from molgen.utils.train_utils import setup_mixed_precision, setup_torch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["DT"])
    parser.add_argument("--dataset-type", type=str, required=True, choices=["DT_SMILES", "DT_SELFIES"])
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--final-ckpt-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    train_config = config["train_config"]
    model_config = config["model_config"]

    print("Setting up torch")
    device = setup_torch(train_config["seed"], train_config["device"])
    setup_mixed_precision(train_config["device"], train_config["dtype"])

    tokenizer = get_tokenizer(args.tokenizer_path)
    reward_func = get_rewards(config["reward"])
    model_config["ignore_index"] = tokenizer.pad_token_id
    model_config["n_goals"] = len(reward_func) if isinstance(reward_func, list) else 1

    print(f"Building model {args.model_type} and Dataset {args.dataset_type}")
    model = get_model(args.model_type, model_config).to(device)
    train_dataset, val_dataset = get_dataset(
        args.dataset_type,
        args.model_type,
        dataset_path=args.data_path,
        tokenizer=tokenizer,
        reward_func=reward_func,
        max_seq_len=model_config["max_seq_len"],
    )
    train_dataloader, val_dataloader = prepare_data_for_training(
        train_dataset, val_dataset, tokenizer.pad_token_id, train_config
    )

    print("Creating Optimizer")
    optimizer = model.configure_optimizers(
        train_config["weight_decay"],
        train_config["learning_rate"],
        train_config["betas"],
        train_config["device"],
    )

    print("Start training")
    run_dt_training(
        model,
        train_dataloader,
        optimizer,
        None,
        None,
        reward_func,
        args.save_path,
        train_config,
        test_dataloader=None,
        device=device,
        wandb_log=train_config["wandb_log"],
    )

    os.makedirs(args.save_path, exist_ok=True)
    epochs = train_config["max_steps"] // len(train_dataset)
    final_epoch = max(0, epochs - 1)
    ckpt_name = args.final_ckpt_name or f"epoch_{final_epoch}.pth"
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": final_epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "token_counter": getattr(train_dataloader.dataset, "tokens", 0),
    }
    ckpt_path = os.path.join(args.save_path, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
