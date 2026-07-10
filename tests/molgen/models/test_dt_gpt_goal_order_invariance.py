import random

import torch

from molgen.datasets.smiles_dataset import PreTrainDecisionGPTSmilesDataset
from molgen.models.dt_gpt import DTGPTConfig, DtGPT


def _build_model(n_goals: int = 3) -> DtGPT:
    config = DTGPTConfig(
        vocab_size=32,
        block_size=12,
        max_seq_len=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        model_type="reward_conditioned",
        n_goals=n_goals,
        ignore_index=-100,
    )
    return DtGPT(config)


def _permute_goals(rtgs: torch.Tensor, goal: torch.Tensor, goal_mask: torch.Tensor, perm: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rtgs[:, perm, :], goal[:, perm], goal_mask[:, perm, :]


class _MockTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, smiles, return_tensors=False):
        return [[3, 4, 5]]


class _MockReward:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, smiles):
        return self.value


def test_dataset_keeps_goal_rtg_alignment_for_random_order(monkeypatch):
    monkeypatch.setattr(random, "randint", lambda a, b: 2)
    monkeypatch.setattr(random, "sample", lambda seq, k: [2, 0])

    dataset = PreTrainDecisionGPTSmilesDataset(
        smiles=["CCO"],
        tokenizer=_MockTokenizer(),
        reward_func=[_MockReward(0.1), _MockReward(0.2), _MockReward(0.3)],
        string_type="SMILES",
    )

    sample = dataset[0]

    assert sample["goal_idx"] == [0, 2]
    assert sample["rtgs"][0] == [0.1, 0.1, 0.1, 0.1]
    assert sample["rtgs"][1] == [0.3, 0.3, 0.3, 0.3]
    assert len(sample["goal_mask"]) == 2
    assert all(mask == [1, 1, 1, 1] for mask in sample["goal_mask"])


def test_reward_conditioned_forward_is_goal_order_invariant():
    torch.manual_seed(0)
    model = _build_model(n_goals=3)
    model.eval()

    batch_size = 2
    seq_len = 4

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len, 1))
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    targets = labels.clone()

    rtgs = torch.tensor(
        [
            [[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]],
            [[4.0, 4.1, 4.2, 4.3], [5.0, 5.1, 5.2, 5.3], [6.0, 6.1, 6.2, 6.3]],
        ],
        dtype=torch.float32,
    )
    goal = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    goal_mask = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
            [[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]],
        ],
        dtype=torch.bool,
    )

    logits_ref, loss_ref = model(
        input_ids=input_ids,
        labels=labels,
        targets=targets,
        rtgs=rtgs,
        goal=goal,
        goal_mask=goal_mask,
    )

    perm = [2, 0, 1]
    rtgs_perm, goal_perm, goal_mask_perm = _permute_goals(rtgs, goal, goal_mask, perm)

    logits_perm, loss_perm = model(
        input_ids=input_ids,
        labels=labels,
        targets=targets,
        rtgs=rtgs_perm,
        goal=goal_perm,
        goal_mask=goal_mask_perm,
    )

    assert torch.allclose(logits_ref, logits_perm, atol=1e-6, rtol=1e-6)
    assert torch.allclose(loss_ref, loss_perm, atol=1e-6, rtol=1e-6)
