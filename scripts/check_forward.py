import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molgen.models.dt_gpt import DTGPTConfig, DtGPT


def build_model(n_goals: int, block_size: int, vocab_size: int, n_embd: int, n_layer: int, n_head: int):
    config = DTGPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_goals=n_goals,
    )
    return DtGPT(config), config


def run_dt(model, config: DTGPTConfig, device: torch.device):
    batch_size = 1
    state_size = 1
    target_vocab = config.vocab_size
    test_goal_counts = list(range(1, config.n_goals + 1))

    for goal_count in test_goal_counts:
        seq_len = min(5, config.max_seq_len)
        input_ids = torch.randint(0, target_vocab, (batch_size, seq_len, state_size), device=device)
        labels = torch.randint(0, target_vocab, (batch_size, seq_len), device=device)
        targets = torch.randint(0, target_vocab, (batch_size, seq_len), device=device)
        rtgs = torch.randn(batch_size, goal_count, seq_len, device=device)
        attention_mask = None
        goals = torch.randint(0, config.n_goals, (batch_size, goal_count), device=device)

        model.zero_grad(set_to_none=True)
        logits, loss = model(
            input_ids=input_ids,
            labels=labels,
            targets=targets,
            rtgs=rtgs,
            attention_mask=attention_mask,
            goal=goals,
        )
        print(f"goals={goal_count} logits shape: {tuple(logits.shape)}")
        print(f"goals={goal_count} loss: {loss.item():.6f}")
        assert torch.isfinite(loss), f"loss is not finite for goal_count={goal_count}"
        loss.backward()
        print(f"goals={goal_count} backward: ok")


def main():
    parser = argparse.ArgumentParser(description="Smoke test a model forward pass.")
    parser.add_argument("--block-size", type=int, default=90)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument(
        "--goal-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Goal counts to test. The model is built with enough goal embeddings for the largest count.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = build_model(
        max(args.goal_counts),
        args.block_size,
        args.vocab_size,
        args.n_embd,
        args.n_layer,
        args.n_head,
    )
    model = model.to(device)
    model.train()

    print("model config:", asdict(config))

    run_dt(model, config, device)

    print("smoke test: ok")


if __name__ == "__main__":
    main()
