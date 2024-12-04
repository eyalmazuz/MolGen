import copy
import os

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from molgen.tokenizers.tokenizer import AbstractTokenizer


class PreTrainGPTSmilesDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: AbstractTokenizer) -> None:
        self.dataset = self.load_smiles(dataset_path)
        self.tokenizer = tokenizer


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__ (self, idx: int) -> dict[str, list[str]]:
        smiles: str = self.dataset[idx]
        encoding = self.tokenizer.encode(smiles, return_tensors=False)
        example: list[int] | torch.Tensor = [self.tokenizer.bos_token_id] + encoding[0] + [self.tokenizer.eos_token_id]
        example = torch.tensor(example, dtype=torch.int64)

        labels = copy.deepcopy(example)
        attention_mask = torch.ones_like(example)

        return {
            "input_ids": example.tolist()[:-1],
            "labels": labels.tolist()[1:],
            "attention_mask": attention_mask.tolist()[:-1]
        }


    def load_smiles(self, dataset_path: str) -> list[str]:
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
