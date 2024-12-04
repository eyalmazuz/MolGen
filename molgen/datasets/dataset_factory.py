from torch.utils.data import Dataset

from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.smiles_datset import PreTrainGPTSmilesDataset
from molgen.models.model_options import ModelType


def get_dataset(dataset_type: DatasetType, model_type: ModelType, **kwargs) -> Dataset:
    match model_type:
        case ModelType.GPT:
            dataset = get_gpt_dataset(dataset_type, **kwargs)
        case _:
            raise ValueError(f"Invalid model type {model_type}")

    return dataset


def get_gpt_dataset(dataset_type: DatasetType, **kwargs) -> Dataset:
    match dataset_type:
        case DatasetType.SMILES:
            dataset = PreTrainGPTSmilesDataset(**kwargs)

    return dataset
