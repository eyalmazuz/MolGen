import copy
import os
from typing import Dict, List

from rdkit import Chem
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from molgen.tokenizers.tokenizer import AbstractTokenizer
from molgen.rewards.reward import AbstractReward


class PreTrainGPTSmilesDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: AbstractTokenizer) -> None:
        self.dataset = self.load_smiles(dataset_path)
        self.tokenizer = tokenizer


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__ (self, idx: int) -> Dict[str, List[str]]:
        smiles = self.dataset[idx]
        example = self.tokenizer.encode(smiles)[0]
        example = [self.tokenizer.bos_token_id] + example + [self.tokenizer.eos_token_id]
        example = torch.tensor(example, dtype=torch.int64)

        labels = copy.deepcopy(example)
        attention_mask = torch.ones_like(example)

        return {
            "input_ids": example.tolist()[:-1],
            "labels": labels.tolist()[1:],
            "attention_mask": attention_mask.tolist()[:-1]
        }


    def load_smiles(self, dataset_path: str) -> List[str]:
        if not os.path.exists(dataset_path):
            raise ValueError("Invalid path")

        if os.path.isdir(dataset_path):
            print("Given path is a directory, attempting to load all files in the directory")
            smiles = []
            for file_ in tqdm(os.listdir(dataset_path)):
                with open(f"{dataset_path}/{file_}", "r") as f:
                    smiles += [s.strip() for s in f.readlines()]

        else:
            print("Loading Data")
            with open(dataset_path, "r") as f:
                smiles = [s.strip() for s in f.readlines()]

        print("Converting SMILES to Canonical SMILES")
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in tqdm(smiles) if Chem.MolFromSmiles is not None]

        return smiles


class PreTrainDecisionGPTSmilesDataset(PreTrainGPTSmilesDataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: AbstractTokenizer,
                 reward_func: AbstractReward) -> None:
        super().__init__(dataset_path, tokenizer)
        self.reward_func = reward_func

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        smiles = self.dataset[idx]
        base_item = super().__getitem__(idx)
        reward_to_go = self.reward_func(smiles)
        trajectory_len = len(base_item["input_ids"]) - 1
        trajectory = {
            "reward_to_go": [reward_to_go] * trajectory_len,
            "states": [base_item["input_ids"][:i + 1] for i in range(trajectory_len)],
            "actions": base_item["input_ids"]
        }

        return {
            "reward_to_go": trajectory["reward_to_go"],
            "input_ids": trajectory["states"],
            "labels": trajectory["actions"],
            "attention_mask": base_item["attention_mask"]
        }
