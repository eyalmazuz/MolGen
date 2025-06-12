import os

import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.smiles_dataset import PreTrainGPTSmilesDataset, PreTrainDecisionGPTSmilesDataset
from molgen.models.model_options import ModelType


def get_dataset(
    dataset_type: DatasetType, model_type: ModelType, dataset_path, tokenizer, **kwargs
) -> tuple[Dataset, Dataset]:
    match model_type.lower():
        case ModelType.GPT:
            dataset = get_gpt_dataset(dataset_type, dataset_path, tokenizer, **kwargs)
        case ModelType.DT:
            dataset = get_gpt_dataset(dataset_type, dataset_path, tokenizer, **kwargs)
        case _:
            raise ValueError(f"Invalid model type {model_type}")

    return dataset


def get_gpt_dataset(dataset_type: DatasetType, dataset_path, tokenizer, **kwargs) -> tuple[Dataset, Dataset]:
    smiles = load_smiles(dataset_path)
    train_smiles, val_smiles = get_train_test_split(smiles, test_size=0.1)
    match dataset_type.lower():
        case DatasetType.SMILES:
            train_dataset = PreTrainGPTSmilesDataset(train_smiles, tokenizer, **kwargs)
            val_dataset = PreTrainGPTSmilesDataset(val_smiles, tokenizer, **kwargs)
        case DatasetType.SELFIES:
            kwargs["string_type"] = "SELFIES"
            train_dataset = PreTrainGPTSmilesDataset(train_smiles, tokenizer, **kwargs)
            val_dataset = PreTrainGPTSmilesDataset(val_smiles, tokenizer, **kwargs)
        case DatasetType.DT_SMILES:
            train_dataset = PreTrainDecisionGPTSmilesDataset(smiles, tokenizer, **kwargs)
            val_dataset = PreTrainDecisionGPTSmilesDataset(val_smiles, tokenizer, **kwargs)
        case DatasetType.DT_SELFIES:
            kwargs["string_type"] = "SELFIES"
            train_dataset = PreTrainDecisionGPTSmilesDataset(smiles, tokenizer, **kwargs)
            val_dataset = PreTrainDecisionGPTSmilesDataset(val_smiles, tokenizer, **kwargs)

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
        file_extension = os.path.splitext(dataset_path)[-1].lower()
        if file_extension == ".txt" or file_extension == ".text":
            print("Loading Data")
            with open(dataset_path, "r") as f:
                smiles = [s.strip() for s in f.readlines()]
        elif file_extension == ".csv":
            df = pd.read_csv(dataset_path)
            if "smiles" not in df.columns:
                raise ValueError("CSV file must contain a 'smiles' column")
            smiles = df["smiles"].dropna().astype(str).tolist()
        else:
            raise ValueError("Unsupported file format. Only .txt and .csv are supported.")

    print("Converting SMILES to Canonical SMILES")
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in tqdm(smiles) if Chem.MolFromSmiles(s) is not None]

    return smiles
