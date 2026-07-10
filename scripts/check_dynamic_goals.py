from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molgen.models.dt_gpt import DTGPTConfig, DtGPT


def build_model(max_goals: int, block_size: int, vocab_size: int, n_embd: int, n_layer: int, n_head: int) -> DtGPT:
    config = DTGPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        n_goals=max_goals,
        model_type="reward_conditioned",
    )
    return DtGPT(config)


def run_smoke_test(goal_counts: list[int], device: torch.device) -> None:
    batch_size = 1
    state_seq_len = 10
    state_size = 1
    vocab_size = 100
    max_goals = max(goal_counts)
    model_block_size = state_seq_len * (max_goals + 2)

    model = build_model(
        max_goals=max_goals,
        block_size=model_block_size,
        vocab_size=vocab_size,
        n_embd=32,
        n_layer=2,
        n_head=4,
    ).to(device)

    expected_shape = (batch_size, state_seq_len, vocab_size)

    print("Initializing dynamic goals smoke test...")
    print(f"Model supports up to {max_goals} goals")
    print("-" * 60)

    for active_goals in goal_counts:
        print(f"\n" + "="*50)
        print(f"🚀 EXPERIMENT: {active_goals} Active Goals")
        
        # הרכבה והדפסה של מבנה הבלוק התיאורטי כדי שנראה בעיניים
        block_structure = [f"RTG_{i}" for i in range(active_goals)] + ["State", "Action"]
        print(f"🔍 Block Structure (1 Timestep): {block_structure}")
        print(f"📏 Tokens per timestep (n_layers): {len(block_structure)}")
        print(
            f"🔗 Expected total sequence length (model_block_size={model_block_size}): "
            f"{state_seq_len * len(block_structure)}"
        )
        print("-" * 50)
        input_ids = torch.randint(0, vocab_size, (batch_size, state_seq_len, state_size), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, state_seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, state_seq_len), device=device)
        rtgs = torch.rand((batch_size, active_goals, state_seq_len), device=device)
        goals = torch.randint(0, max_goals, (batch_size, active_goals), device=device)

        model.train()
        logits_train, loss = model(
            input_ids=input_ids,
            labels=labels,
            targets=targets,
            rtgs=rtgs,
            goal=goals,
        )
        assert logits_train.shape == expected_shape, f"Expected {expected_shape}, got {logits_train.shape}"
        assert loss is not None, "Training loss is None"
        assert torch.isfinite(loss), f"Training loss is not finite: {loss}"
        loss.backward()
        print(f"[train] active_goals={active_goals} | logits={tuple(logits_train.shape)} | loss={loss.item():.4f}")

        model.eval()
        logits_eval, eval_loss = model(
            input_ids=input_ids,
            labels=None,
            targets=None,
            rtgs=rtgs,
            goal=goals,
        )
        assert logits_eval.shape == expected_shape, f"Expected {expected_shape}, got {logits_eval.shape}"
        assert eval_loss is None, f"Expected eval loss to be None, got {eval_loss}"
        print(f"[eval ] active_goals={active_goals} | logits={tuple(logits_eval.shape)}")

        model.zero_grad(set_to_none=True)
        print("-" * 60)

    print("Dynamic goals smoke test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test DtGPT with arbitrary numbers of active goals.")
    parser.add_argument(
        "--goal-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Active goal counts to test. The model is built with enough goal embeddings for the largest count.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.goal_counts:
        raise ValueError("--goal-counts must contain at least one positive integer")
    if min(args.goal_counts) < 1:
        raise ValueError("--goal-counts must be positive integers")

    device = torch.device(args.device)
    run_smoke_test(args.goal_counts, device)


if __name__ == "__main__":
    main()
