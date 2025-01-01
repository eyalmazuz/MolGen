import random
from itertools import islice
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm

from molgen.utils.utils import get_rank, get_world_size, is_distributed_run


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
                self.samples.append({k: v[: self.chunk_size] for k, v in buffer.items()})
                buffer = {k: v[self.chunk_size :] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class LengthBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        self.lengths = [d["length"] for d in tqdm(dataset, desc="Processing dataset")]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBatchSampler(data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


class PadCollate:
    def __init__(self, pad_token_id: int, ignore_index: int = -100, max_length=None) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length

    def __call__(self, batches: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in batches)
        if self.max_length is not None:
            self.max_length = max_length = max(self.max_length, max_length)

        batch_input_ids = []
        # batch_attention_mask = []
        batch_labels = []
        batch_rtgs = []

        for batch in batches:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            rtg = batch.get("rtg", None)

            if len(input_ids) < max_length:
                if rtg is None:
                    input_ids += [self.pad_token_id] * (max_length - len(input_ids))
                else:
                    input_ids += [[self.pad_token_id] * max_length] * (max_length - len(input_ids))
                    rtg += [0] * (max_length - len(rtg))

                # attention_mask += [0] * (max_length - len(attention_mask))
                labels += [self.ignore_index] * (max_length - len(labels))

            if rtg is not None:
                input_ids = [state + [self.pad_token_id] * (max_length - len(state)) for state in input_ids]
                # attention_mask = [
                #     [0] * max_length if mask == 0 else [1] * (i + 1) + [0] * (max_length - (i + 1))
                #     for i, mask in enumerate(attention_mask)
                # ]
                batch_rtgs.append(rtg)

            batch_input_ids.append(input_ids)
            # batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return_dict = {
            "input_ids": torch.tensor(np.array(batch_input_ids), dtype=torch.int64),
            # "attention_mask": torch.tensor(np.array(batch_attention_mask), dtype=torch.int64),
            "labels": torch.tensor(np.array(batch_labels), dtype=torch.int64)
        }
        if len(batch_rtgs) > 0:
            return_dict["rtg"] = torch.tensor(batch_rtgs, dtype=torch.float32)

        return return_dict


def prepare_data_for_training(
    train_dataset: Dataset, val_dataset: Dataset, pad_token_id: int, train_config: dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    train_sampler: LengthBatchSampler | DistributedLengthBatchSampler
    val_sampler: LengthBatchSampler | DistributedLengthBatchSampler
    if is_distributed_run():
        world_size = get_world_size()
        rank = get_rank()
        seed = 42 + is_distributed_run() * get_rank()
        train_sampler = DistributedLengthBatchSampler(
            train_dataset, train_config["batch_size"], world_size, rank, seed=seed
        )
        val_sampler = DistributedLengthBatchSampler(
            val_dataset, train_config["batch_size"], world_size, rank, seed=seed
        )
    else:
        train_sampler = LengthBatchSampler(train_dataset, train_config["batch_size"], drop_last=False)
        val_sampler = LengthBatchSampler(val_dataset, train_config["batch_size"], drop_last=False)
    collate_fn = PadCollate(pad_token_id)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
