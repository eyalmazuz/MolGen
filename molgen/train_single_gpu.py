import os
import tomllib
import wandb
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from molgen.datasets.dataset_factory import get_dataset
from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.dataset_utils import LengthBatchSampler, PadCollate
from molgen.models.model_factory import get_model
from molgen.models.model_options import ModelType
from molgen.tokenizers.tokenizer_factory import get_tokenizer
from molgen.training.train import run_training
from molgen.training.train_dt import run_dt_training
from molgen.utils.train_utils import setup_torch, setup_mixed_precision
from molgen.utils.plot_utils import log_metrics_to_wandb
from molgen.rewards.reward_factory import get_rewards


def single_gpu_training(args) -> None:
    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    train_config = config["train_config"]
    model_config = config["model_config"]

    setup_torch(train_config["seed"], train_config["device"])
    ctx, scaler = setup_mixed_precision(train_config["device"], train_config["dtype"])

    model_type = ModelType.from_str(args.model_type)
    dataset_type = DatasetType.from_str(args.dataset_type)

    model = get_model(model_type, model_config).to(train_config["device"])
    tokenizer = get_tokenizer(args.tokenizer_path)
    train_config["tokenizer"] = os.path.basename(args.tokenizer_path)

    kwargs = {
        "dataset_path": args.data_path,
        "tokenizer": tokenizer,
    }
    if model_type == ModelType.DT:
        kwargs.update({"reward_func": get_rewards(config["reward"])})

    dataset = get_dataset(dataset_type,
                          model_type,
                          **kwargs)

    batch_sampler = LengthBatchSampler(dataset, train_config["batch_size"], drop_last=False)
    collate_fn = PadCollate(
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=tokenizer.pad_token_id,
        max_length=model_config["block_size"] // 3
    )
    dataloader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            # num_workers=train_config["num_workers"]
                            )

    optimizer = model.configure_optimizers(train_config["weight_decay"],
                                           train_config["learning_rate"],
                                           train_config["betas"],
                                           train_config["device"])

    if args.wandb_key is None:
        wandb_run = None
    else:
        wandb_run = log_metrics_to_wandb(
            wandb_key=args.wandb_key,
            project_name=args.wandb_proj,
            project_entity=args.wandb_entity,
            training_config=train_config,
            run_name=f"{os.path.basename(args.data_path).split('.')[0]}_"
                     f"{str(datetime.now().strftime('%m_%d_%H_%M_%S'))}"
        )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    train_config["ckpt_path"] = args.save_path

    if train_config["compile"]:
        model = torch.compile(model)

    if model_type == ModelType.DT:
        run_dt_training(model, dataloader, optimizer, ctx, scaler, kwargs["reward_func"], train_config, wandb_run=wandb_run)
    else:
        run_training(model, dataloader, optimizer, ctx, scaler, train_config)

    wandb.finish()
