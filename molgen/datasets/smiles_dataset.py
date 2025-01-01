import copy
from typing import Literal

import torch
from torch.utils.data import Dataset
import numpy as np
import selfies as sf

from molgen.tokenizers.tokenizer import AbstractTokenizer
from molgen.rewards.reward import AbstractReward


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
                 reward_func: AbstractReward,
                 string_type: Literal["SMILES", "SELFIES"] = "SMILES") -> None:
        super().__init__(smiles, tokenizer, string_type)
        self.reward_func = reward_func

    def __getitem__(self, idx: int) -> dict[str, dict[str]]:
        base_item = super().__getitem__(idx)
        trajectory_len = len(base_item["input_ids"])
        states = [base_item["input_ids"][:i + 1] for i in range(trajectory_len)]

        smiles = self.dataset[idx]
        if self.string_type == "SMILES":
            reward_to_go = self.reward_func(smiles)
            reward_to_go = [reward_to_go] * trajectory_len
        if self.string_type == "SELFIES":
            state_selfies = self.tokenizer.decode(states, skip_special_tokens=True)
            reward_to_go = self.reward_func([sf.decoder(s) for s in state_selfies])
            reward_to_go[0] = 0
            reward_to_go = np.subtract(reward_to_go[-1], reward_to_go).tolist()

        return {
            "rtg": reward_to_go,                        # trajectory rtg - (block, 1)
            "input_ids": states,                        # states - (block, state_len)
            "labels": base_item["labels"],              # actions - (block, 1)
            # "attention_mask": base_item["attention_mask"],
        }
