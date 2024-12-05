import os

from rdkit import Chem
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.smiles_dataset import PreTrainGPTSmilesDataset
from molgen.models.model_options import ModelType


def get_dataset(
    dataset_type: DatasetType, model_type: ModelType, dataset_path, tokenizer, **kwargs
) -> tuple[Dataset, Dataset]:
    match model_type:
        case ModelType.GPT:
            dataset = get_gpt_dataset(dataset_type, dataset_path, tokenizer, **kwargs)
        case _:
            raise ValueError(f"Invalid model type {model_type}")

    return dataset


def get_gpt_dataset(dataset_type: DatasetType, dataset_path, tokenizer, **kwargs) -> tuple[Dataset, Dataset]:
    smiles = load_smiles(dataset_path)
    train_smiles, val_smiles = get_train_test_split(smiles, test_size=0.2)
    match dataset_type:
        case DatasetType.SMILES:
            train_dataset = PreTrainGPTSmilesDataset(train_smiles, tokenizer, **kwargs)
            val_dataset = PreTrainGPTSmilesDataset(val_smiles, tokenizer, **kwargs)

    return train_dataset, val_dataset


def get_train_test_split(smiles: list[str], test_size: float) -> tuple[list[str], list[str]]:
    data_size = len(smiles)
    train_smiles = smiles[int(data_size * test_size) :]
    val_smiles = smiles[: int(data_size * test_size)]

    return train_smiles, val_smiles


def load_smiles(dataset_path: str) -> list[str]:
    if not os.path.exists(dataset_path):
        raise ValueError("Invalid path")

    if os.path.isdir(dataset_path):
        print("Given path is a directory, attemping loading all files in the directory")
        smiles = []
        for file_ in tqdm(os.listdir(dataset_path)):
            with open(f"{dataset_path}/{file_}", "r") as f:
                smiles += [s.strip() for s in f.readlines()]

    else:
        print("Loading Data")
        with open(dataset_path, "r") as f:
            smiles = [s.strip() for s in f.readlines()]

    print("Converting SMILES to Canonical SMILES")
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in tqdm(smiles) if Chem.MolFromSmiles(s) is not None]

    return smiles
