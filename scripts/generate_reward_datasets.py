from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import selfies as sf
import torch
import tomllib
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from molgen.datasets.dataset_options import DatasetType
from molgen.models.dt_gpt import sample
from molgen.models.model_factory import get_model
from molgen.rewards.reward_factory import get_rewards
from molgen.tokenizers.tokenizer_factory import get_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "rb") as fd:
        return tomllib.load(fd)


def load_checkpoint(model, checkpoint_path: str, device: str):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def canonicalize_smiles(smiles: str) -> tuple[str, bool]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False
    return Chem.MolToSmiles(mol), True


def safe_reward_value(reward_fn, smiles: str) -> float:
    # Report raw chemical metrics even when training used a scaled reward.
    previous_eval = reward_fn.eval
    try:
        reward_fn.eval = True
        value = reward_fn(smiles)
    except Exception:
        return float("nan")
    finally:
        reward_fn.eval = previous_eval

    if value is None:
        return float("nan")
    return float(value)


def model_rtg_from_raw_target(reward_fn, raw_target: float) -> float:
    """Apply the same reward scaling used to construct training RTGs."""
    if reward_fn.scale is None:
        return float(raw_target)
    return float(reward_fn.scale(raw_target))


def permute_goal_inputs(
    goal_ids: list[int],
    rtg_values: list[float],
    goal_mask: list[bool],
    permutation: list[int],
) -> tuple[list[int], list[float], list[bool]]:
    """Permute goals while preserving every (goal id, RTG, mask) association."""
    if not (len(goal_ids) == len(rtg_values) == len(goal_mask) == len(permutation)):
        raise ValueError("Goal IDs, RTGs, masks, and permutation must have equal lengths")
    if sorted(permutation) != list(range(len(goal_ids))):
        raise ValueError("permutation must contain every goal position exactly once")
    return (
        [goal_ids[i] for i in permutation],
        [rtg_values[i] for i in permutation],
        [goal_mask[i] for i in permutation],
    )


def decode_sequence(tokenizer, token_ids: list[int], dataset_type: DatasetType) -> str:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)[0]
    if dataset_type == DatasetType.DT_SELFIES:
        try:
            decoded = sf.decoder(decoded)
        except Exception:
            pass
    return decoded


def make_condition_specs(n_goals: int) -> list[dict[str, Any]]:
    if n_goals < 2:
        raise ValueError(f"Expected at least 2 goals, got {n_goals}")

    return [
        {
            "generation_condition": "reward_A_only",
            "active_goal_ids": [0],
            "output_file": "generated_reward_A.csv",
        },
        {
            "generation_condition": "reward_B_only",
            "active_goal_ids": [1],
            "output_file": "generated_reward_B.csv",
        },
        {
            "generation_condition": "reward_A_and_B",
            "active_goal_ids": [0, 1],
            "output_file": "generated_reward_A_and_B.csv",
        },
    ]


@torch.no_grad()
def generate_conditioned_smiles(
    model,
    tokenizer,
    reward_functions,
    dataset_type: DatasetType,
    generation_condition: str,
    active_goal_ids: list[int],
    reward_targets: list[float],
    num_molecules: int,
    batch_size: int,
    max_generation_length: int,
    temperature: float,
    device: str,
) -> list[dict[str, Any]]:
    total_goals = model.config.n_goals
    if not active_goal_ids:
        raise ValueError("At least one goal must be active")
    if len(set(active_goal_ids)) != len(active_goal_ids):
        raise ValueError("Active goal IDs must be unique")
    if min(active_goal_ids) < 0 or max(active_goal_ids) >= total_goals:
        raise ValueError(f"Active goal IDs must be in [0, {total_goals})")

    raw_reward_targets = list(reward_targets[:2]) + [0.0] * max(0, total_goals - 2)
    all_model_rtg_targets = [
        model_rtg_from_raw_target(reward_functions[i], raw_reward_targets[i])
        for i in range(total_goals)
    ]
    active_model_rtgs = [all_model_rtg_targets[i] for i in active_goal_ids]
    active_goal_mask = [True] * len(active_goal_ids)

    rows: list[dict[str, Any]] = []
    remaining = num_molecules

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        n_active_goals = len(active_goal_ids)
        permutation = torch.randperm(n_active_goals).tolist()
        ordered_goal_ids, ordered_rtgs, ordered_goal_mask = permute_goal_inputs(
            active_goal_ids, active_model_rtgs, active_goal_mask, permutation
        )
        goal_ids = torch.tensor(ordered_goal_ids, device=device, dtype=torch.long).unsqueeze(0)
        goal_template = torch.tensor(ordered_goal_mask, dtype=torch.bool, device=device).view(1, n_active_goals, 1)
        rtg_template = torch.tensor(ordered_rtgs, dtype=torch.float32, device=device).view(1, n_active_goals, 1)
        sequences = [[tokenizer.bos_token_id] for _ in range(current_batch)]
        finished = torch.zeros(current_batch, dtype=torch.bool, device=device)

        for _ in range(max_generation_length):
            seq_len = len(sequences[0])
            input_ids = torch.tensor(sequences, dtype=torch.long, device=device).unsqueeze(-1)
            rtgs = rtg_template.repeat(current_batch, 1, seq_len)
            goal = goal_ids.repeat(current_batch, 1)
            goal_mask = goal_template.repeat(current_batch, 1, seq_len)

            next_tokens = sample(
                model=model,
                x=input_ids,
                steps=1,
                temperature=temperature,
                sample=True,
                actions=None,
                rtgs=rtgs,
                attention=None,
                goal=goal,
                goal_mask=goal_mask,
            ).squeeze(-1)

            if finished.any():
                next_tokens = next_tokens.clone()
                next_tokens[finished] = tokenizer.eos_token_id

            token_list = next_tokens.tolist()
            for idx, token_id in enumerate(token_list):
                sequences[idx].append(token_id)

            finished |= next_tokens.eq(tokenizer.eos_token_id)
            if bool(finished.all()):
                break

        for seq in sequences:
            decoded = decode_sequence(tokenizer, seq, dataset_type)
            canonical_smiles, is_valid = canonicalize_smiles(decoded)

            row: dict[str, Any] = {
                "smiles": canonical_smiles if is_valid else decoded,
                "generation_condition": generation_condition,
                "reward_A_target": float(reward_targets[0]),
                "reward_B_target": float(reward_targets[1]),
                "reward_A_model_rtg": float(all_model_rtg_targets[0]),
                "reward_B_model_rtg": float(all_model_rtg_targets[1]),
                "goal_ids": json.dumps(ordered_goal_ids),
                "goal_mask": json.dumps(ordered_goal_mask),
                "is_valid": bool(is_valid),
                "reward_A_actual": np.nan,
                "reward_B_actual": np.nan,
            }

            if is_valid:
                row["reward_A_actual"] = safe_reward_value(reward_functions[0], row["smiles"])
                row["reward_B_actual"] = safe_reward_value(reward_functions[1], row["smiles"])

            rows.append(row)

        remaining -= current_batch

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate separate molecule datasets for reward A, reward B, and both.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model/reward config TOML.")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the trained tokenizer directory.")
    parser.add_argument("--model-type", type=str, default="DT", choices=["GPT", "DT", "LLAMA"], help="Model family used by the checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where CSVs will be written.")
    parser.add_argument("--num-molecules", type=int, default=1000, help="Number of molecules per dataset.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of molecules to generate in parallel.")
    parser.add_argument("--reward-a-target", type=float, default=1.0, help="Target RTG for reward A.")
    parser.add_argument("--reward-b-target", type=float, default=1.0, help="Target RTG for reward B.")
    parser.add_argument("--max-generation-length", type=int, default=100, help="Maximum number of sampled tokens per molecule.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=DatasetType.DT_SMILES,
        choices=[DatasetType.DT_SMILES, DatasetType.DT_SELFIES],
        help="Controls how decoded token sequences are converted to molecules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config_path)
    model_config = config["model_config"]

    tokenizer = get_tokenizer(args.tokenizer_path)
    reward_functions = get_rewards(config["reward"])
    if not isinstance(reward_functions, list) or len(reward_functions) < 2:
        raise ValueError("This script expects a goal-conditioned model with at least two reward functions.")

    model = get_model(args.model_type, model_config).to(args.device)
    model = load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_specs = make_condition_specs(model.config.n_goals)
    reward_targets = [args.reward_a_target, args.reward_b_target]
    dataset_type = DatasetType(args.dataset_type)

    for spec in condition_specs:
        print(f"Generating {spec['generation_condition']}...")
        rows = generate_conditioned_smiles(
            model=model,
            tokenizer=tokenizer,
            reward_functions=reward_functions,
            dataset_type=dataset_type,
            generation_condition=spec["generation_condition"],
            active_goal_ids=spec["active_goal_ids"],
            reward_targets=reward_targets,
            num_molecules=args.num_molecules,
            batch_size=args.batch_size,
            max_generation_length=args.max_generation_length,
            temperature=args.temperature,
            device=args.device,
        )

        df = pd.DataFrame(rows)
        df = df[
            [
                "smiles",
                "generation_condition",
                "reward_A_target",
                "reward_B_target",
                "reward_A_model_rtg",
                "reward_B_model_rtg",
                "goal_ids",
                "goal_mask",
                "is_valid",
                "reward_A_actual",
                "reward_B_actual",
            ]
        ]
        output_path = output_dir / spec["output_file"]
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
