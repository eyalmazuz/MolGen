import copy
import os
from typing import Dict, List, Literal

from rdkit import Chem
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import selfies as sf

from molgen.tokenizers.tokenizer import AbstractTokenizer
from molgen.rewards.reward import AbstractReward


class PreTrainGPTSmilesDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: AbstractTokenizer,
                 string_type: Literal["SMILES", "SELFIES"] = "SMILES") -> None:
        self.string_type = string_type
        self.dataset = self.load_smiles(dataset_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        smiles = self.dataset[idx]
        if self.string_type == "SMILES":
            example = self.tokenizer.encode(smiles)[0]
        elif self.string_type == "SELFIES":
            example = self.tokenizer.encode_selfies(smiles)[0]

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

        if self.string_type == "SMILES":
            print("Converting SMILES to Canonical SMILES")
            smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in tqdm(smiles) if Chem.MolFromSmiles is not None]

        return smiles


class PreTrainDecisionGPTSmilesDataset(PreTrainGPTSmilesDataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: AbstractTokenizer,
                 reward_func: AbstractReward,
                 string_type: Literal["SMILES", "SELFIES"] = "SMILES") -> None:
        super().__init__(dataset_path, tokenizer, string_type)
        self.reward_funcs = reward_func if isinstance(reward_func, list) else [reward_func]
        self.n_goals = len(self.reward_funcs)
        self.n_molecules = len(self.dataset)
        self.reward_memory = [{} for _ in self.reward_funcs]
        self._trajectories = self._precompute_trajectories()

    def _precompute_trajectories(self) -> list[dict[str, list[str]]]:
        """
        Precompute results for all molecules and goals.

        Returns:
            A list of dictionaries containing precomputed trajectories for all molecules and goals.
        """
        results = []
        for mol_idx, smiles in tqdm(
                enumerate(self.dataset), desc="Computing molecule trajectories", total=len(self.dataset)
        ):
            base_item = super().__getitem__(mol_idx)
            trajectory_len = len(base_item["input_ids"])
            states = [base_item["input_ids"][:i + 1] for i in range(trajectory_len)]

            for goal_idx, reward_func in enumerate(self.reward_funcs):  # Iterate over goals
                if self.string_type == "SMILES":
                    reward_to_go = reward_func(smiles)
                    reward_to_go = [reward_to_go] * trajectory_len
                elif self.string_type == "SELFIES":
                    state_selfies = self.tokenizer.decode(states, skip_special_tokens=True)
                    reward_to_go = [
                        self.reward_memory[goal_idx].setdefault(s, reward_func(sf.decoder(s)))
                        if (r := self.reward_memory[goal_idx].get(s)) is None else r
                        for s in state_selfies
                    ]
                    reward_to_go[0] = 0
                    reward_to_go = np.subtract(reward_to_go[-1], reward_to_go).tolist()

                results.append({
                    "rtgs": reward_to_go.copy(),            # trajectory rtg - (block, 1)
                    "input_ids": states.copy(),             # states - (block, state_len)
                    "labels": base_item["labels"].copy(),   # actions - (block, 1)
                    "attention_mask": [1] * trajectory_len,
                    "length": trajectory_len,
                    "goal_idx": goal_idx if self.n_goals > 1 else None,
                    "mol_idx": mol_idx
                })

        return results

    def __len__(self) -> int:
        """
        Return the total number of samples (n_goals * n_molecules).
        """
        return self.n_goals * self.n_molecules

    def __getitem__(self, idx: int) -> dict[str, list[str]]:
        """
        Get a precomputed trajectory based on the global index.

        Args:
            idx: Global index in the range [0, n_goals * n_molecules).

        Returns:
            A dictionary containing the trajectory data for the corresponding goal and molecule.
        """
        return self._trajectories[idx]
