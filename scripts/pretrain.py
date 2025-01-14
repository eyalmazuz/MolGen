import argparse
import os

import tomllib
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from molgen.datasets.dataset_factory import get_dataset
from molgen.datasets.dataset_utils import prepare_data_for_training
from molgen.models.model_factory import get_model
from molgen.models.model_options import ModelType
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.training.train import pretrain_model
from molgen.utils.train_utils import setup_mixed_precision, setup_torch
from molgen.utils.utils import get_world_size, is_distributed_run, is_master_process
from molgen.rewards.reward_factory import get_rewards


def get_pretrain_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer used for training")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the model")
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["GPT", "DT", "LLAMA"], help="Type of model to use for training"
    )
    parser.add_argument(
        "--dataset-type", type=str, required=True, choices=["SMILES", "DT_SMILES", "SELFIES", "DT_SELFIES"], help="Type of dataset to use for training"
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config containing training and model params"
    )

    # Wandb parameters to log results
    parser.add_argument('--wandb-key', type=str, help='wandb api key for user login', default=None)
    parser.add_argument(
        '--wandb-proj', type=str, default='DecisionMol', help='name of wandb project to upload results')
    parser.add_argument('--wandb-entity', type=str, default='bgu-sise', help='wandb entity associated with the project')
    parser.add_argument('--wandb-name', type=str, help='wandb run name', default=None)

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    globals_config_keys = [
        k for k, v in globals().items() if not k.startswith("_") and isinstance(v, int | float | bool | str)
    ]
    globals_config = {k: globals()[k] for k in globals_config_keys}  # will be useful for logging
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
    tokenizer = get_tokenizer(args.tokenizer_path)

    kwargs = {}
    if args.model_type.lower() == ModelType.DT:
        model_config["ignore_index"] = tokenizer.pad_token_id
        reward_func = get_rewards(config["reward"])
        kwargs = {"reward_func": reward_func}
        model_config["n_goals"] = len(reward_func) if isinstance(reward_func, list) else 1

    print(f"Building model {args.model_type} and Dataset {args.dataset_type}")

    model = get_model(args.model_type, model_config).to(device)
    train_dataset, val_dataset = get_dataset(
        args.dataset_type, args.model_type, dataset_path=args.data_path, tokenizer=tokenizer, **kwargs
    )

    train_dataloader, val_dataloader = prepare_data_for_training(
        train_dataset, val_dataset, tokenizer.pad_token_id, train_config
    )

    print("Creating Optimizer")
    optimizer = model.configure_optimizers(
        train_config["weight_decay"], train_config["learning_rate"], train_config["betas"], device
    )
    # Suppose warmup_epochs is the number of epochs to warm up
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / train_config.get("warmup_steps", 0)
        if epoch < train_config.get("warmup_steps", 0)
        else 1.0,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=train_config.get("max_steps"), eta_min=train_config.get("min_lr", 0)
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[train_config.get("warmup_steps", 0)]
    )

    if train_config["compile"]:
        print("Compiling model")
        model = torch.compile(model)  # type: ignore

    if train_config["wandb_log"] and is_master_process():
        import wandb
        wandb.login(key=args.wandb_key)
        wandb.init(  # type: ignore
            project=args.wandb_proj,
            entity=args.wandb_entity,
            name=args.wandb_name if args.wandb_name is not None else f"{args.model_type}_{args.dataset_type}",
            config=config,
        )

    print("Start training")
    pretrain_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        ctx,
        scaler,
        kwargs.get("reward_func", None),
        args.save_path,
        train_config["load_checkpoint"],
        train_config["max_steps"],
        train_config["grad_clip"],
        train_config["gradient_accumulation_steps"],
        train_config["eval_every"],
        train_config["log_every"],
        train_config["wandb_log"],
        device=device,
        globals_config=globals_config,
    )

    if is_distributed_run():
        destroy_process_group()


if __name__ == "__main__":
    args = get_pretrain_args()
    run_training(args)
