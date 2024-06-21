import tomllib

from molgen.models.model_factory import get_model
from molgen.utils.train_utils import setup_torch


def single_gpu_training(args) -> None:
    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    train_config = config["train_config"]
    model_config = config["model_config"]

    ctx, scaler = setup_torch(train_config["seed"], train_config["device"], train_config["dtype"])

    model = get_model(args.model_type, model_config)

    print(model)
