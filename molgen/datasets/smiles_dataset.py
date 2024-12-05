import copy

import torch
from torch.utils.data import Dataset

from molgen.tokenizers.tokenizer import AbstractTokenizer


class PreTrainGPTSmilesDataset(Dataset):
    def __init__(self,
                 smiles: list[str],
                 tokenizer: AbstractTokenizer) -> None:
        self.dataset = smiles
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
