from typing import Dict, List
import random

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import BatchSampler, Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k, v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k, v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class LengthBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        self.lengths = [len(d["input_ids"]) for d in tqdm(dataset, desc="Processing dataset")]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class PadCollate:
    def __init__(self, pad_token_id: int, ignore_index: int = -100, max_length=None) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length

    def __call__(self, batches: List[Dict[str, List[int | List[int]]]]) -> Dict[str, torch.tensor]:
        max_length = max(len(item["input_ids"]) for item in batches) if self.max_length is None else self.max_length

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_rtgs = []

        for batch in batches:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            rtg = batch.get("rtg", None)

            if len(input_ids) < max_length:
                if rtg is None:
                    input_ids += [self.pad_token_id] * (max_length - len(input_ids))
                else:
                    input_ids += [[self.pad_token_id] * max_length] * (max_length - len(input_ids))
                    rtg += [0] * (max_length - len(rtg))

                attention_mask += [0] * (max_length - len(attention_mask))
                labels += [self.ignore_index] * (max_length - len(labels))

            if rtg is not None:
                input_ids = [state + [self.pad_token_id] * (max_length - len(state)) for state in input_ids]
                batch_rtgs.append(rtg)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return_dict = {
            "input_ids": torch.tensor(np.array(batch_input_ids), dtype=torch.int64),
            "attention_mask": torch.tensor(np.array(batch_attention_mask), dtype=torch.int64),
            "labels": torch.tensor(np.array(batch_labels), dtype=torch.int64)
        }
        if len(batch_rtgs) > 0:
            return_dict["rtg"] = torch.tensor(batch_rtgs, dtype=torch.float32)

        return return_dict
