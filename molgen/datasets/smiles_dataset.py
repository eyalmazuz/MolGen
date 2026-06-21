import copy
from typing import Literal, Union

import torch
from torch.utils.data import Dataset
import numpy as np
import selfies as sf
from tqdm import tqdm

from molgen.tokenizers.tokenizer import AbstractTokenizer
from molgen.rewards.reward import AbstractReward

import random

class PreTrainGPTSmilesDataset(Dataset):
    def __init__(
            self,
            smiles: list[str],
            tokenizer: AbstractTokenizer,
            string_type: Literal["SMILES", "SELFIES"] = "SMILES"
    ) -> None:
        self.dataset = smiles
        self.tokenizer = tokenizer
        self.string_type = string_type

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, list[str]]:
        smiles: str = self.dataset[idx]
        if self.string_type == "SMILES":
            encoding = self.tokenizer.encode(smiles, return_tensors=False)
        elif self.string_type == "SELFIES":
            encoding = self.tokenizer.encode_selfies(smiles)[0]

        example: list[int] | torch.Tensor = [self.tokenizer.bos_token_id] + encoding[0] + [self.tokenizer.eos_token_id]
        example = torch.tensor(example, dtype=torch.int64)

        labels = copy.deepcopy(example)

        return {
            "input_ids": example.tolist()[:-1],
            "labels": labels.tolist()[1:],
        }


class PreTrainDecisionGPTSmilesDataset(PreTrainGPTSmilesDataset):
    def __init__(self,
                 smiles: list[str],
                 tokenizer: AbstractTokenizer,
                 reward_func: Union[AbstractReward, list[AbstractReward]],
                 string_type: Literal["SMILES", "SELFIES"] = "SMILES") -> None:
        """
        Initialize the dataset with pre-calculated results for all goals.

        Args:
            smiles: list of smiles strings.
            tokenizer: Tokenizer to convert states to tokens.
            reward_func: A single reward function or a list of reward functions for different goals.
            string_type: Type of string representation ("SMILES" or "SELFIES").
        """
        super().__init__(smiles, tokenizer, string_type)
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

            rtgs = []
            for goal_idx, reward_func in enumerate(self.reward_funcs):  # Iterate over goals
                if self.string_type == "SMILES":
                    reward_to_go = reward_func(smiles)
                    reward_to_go = [reward_to_go] * trajectory_len   # empty list size trajectory_len
                elif self.string_type == "SELFIES":
                    state_selfies = self.tokenizer.decode(states, skip_special_tokens=True)
                    reward_to_go = [
                        self.reward_memory[goal_idx].setdefault(s, reward_func(sf.decoder(s)))
                        if (r := self.reward_memory[goal_idx].get(s)) is None else r
                        for s in state_selfies
                    ]
                    reward_to_go[0] = 0
                    reward_to_go = np.subtract(reward_to_go[-1], reward_to_go).tolist()

                rtgs.append(reward_to_go)

            results.append({
                "rtgs": rtgs.copy(),                    # trajectory rtg - (block, 1)
                "input_ids": states.copy(),             # states - (block, state_len)
                "labels": base_item["labels"].copy(),   # actions - (block, 1)
                "attention_mask": [1] * trajectory_len,
                "length": trajectory_len,
                # "goal_idx": list(range(self.n_goals)) if self.n_goals > 1 else None, 
            })

        return results

    def __len__(self) -> int:
        """
        Return the total number of samples (n_goals * n_molecules).
        """
        return self.n_molecules     # * self.n_goals

    def __getitem__(self, idx: int) -> dict[str, list[str]]:
        """
        Get a precomputed trajectory based on the global index.

        Args:
            idx: Global index in the range [0, n_goals * n_molecules).

        Returns:
            A dictionary containing the trajectory data for the corresponding goal and molecule.
        """
        #making it dynamic for multiple goals
        trajectory = self._trajectories[idx]
        num_goals = random.randint(1, self.n_goals)  # Randomly select a number of goals to include in the trajectory
        all_goals = list(range(self.n_goals))
        selected_goals = random.sample(all_goals, num_goals)
        selected_goals.sort()  # Sort the selected goals to maintain a consistent order

        fillterd_rtgs = [trajectory["rtgs"][goal_idx] for goal_idx in selected_goals] #
        goal_mask = [
            [1] *  len(trajectory["input_ids"])
            for _ in selected_goals
            #goal_mask is a list of lists, where each inner list corresponds to a selected goal and contains 1s for the length of the trajectory
        ]
        return {
            "rtgs": fillterd_rtgs,                    # trajectory rtg - (block, num_selected_goals)
            "input_ids": trajectory["input_ids"],     # states - (block, state_len)
            "labels": trajectory["labels"],           # actions - (block, 1)
            "attention_mask": trajectory["attention_mask"],
            "length": trajectory["length"],
            "goal_idx": selected_goals,               # indices of the selected goals
            "goal_mask": goal_mask,                   # mask for the selected goals
        }
