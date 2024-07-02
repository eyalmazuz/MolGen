import tomllib

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

    model = get_model(model_type, model_config)
    tokenizer = get_tokenizer(args.tokenizer_path)

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
    collate_fn = PadCollate(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=collate_fn,
                            pin_memory=True)

    optimizer = model.configure_optimizers(train_config["weight_decay"],
                                           train_config["learning_rate"],
                                           train_config["betas"],
                                           train_config["device"])

    if train_config["compile"]:
        model = torch.compile(model)

    if model_type == ModelType.DT:
        run_dt_training(model, dataloader, optimizer, ctx, scaler, kwargs["reward_func"], train_config)
    else:
        run_training(model, dataloader, optimizer, ctx, scaler, train_config)
