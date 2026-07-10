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
    try:
        value = reward_fn(smiles)
    except Exception:
        return float("nan")

    if value is None:
        return float("nan")
    return float(value)


def decode_sequence(tokenizer, token_ids: list[int], dataset_type: DatasetType) -> str:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)[0]
    if dataset_type == DatasetType.DT_SELFIES:
        try:
            decoded = sf.decoder(decoded)
        except Exception:
            pass
    return decoded


def build_goal_mask(n_goals: int, active_goal_indices: list[int]) -> list[bool]:
    mask = [False] * n_goals
    for idx in active_goal_indices:
        mask[idx] = True
    return mask


def make_condition_specs(n_goals: int) -> list[dict[str, Any]]:
    if n_goals < 2:
        raise ValueError(f"Expected at least 2 goals, got {n_goals}")

    return [
        {
            "generation_condition": "reward_A_only",
            "goal_mask": build_goal_mask(n_goals, [0]),
            "output_file": "generated_reward_A.csv",
        },
        {
            "generation_condition": "reward_B_only",
            "goal_mask": build_goal_mask(n_goals, [1]),
            "output_file": "generated_reward_B.csv",
        },
        {
            "generation_condition": "reward_A_and_B",
            "goal_mask": build_goal_mask(n_goals, [0, 1]),
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
    goal_mask_1d: list[bool],
    reward_targets: list[float],
    num_molecules: int,
    batch_size: int,
    max_generation_length: int,
    temperature: float,
    device: str,
) -> list[dict[str, Any]]:
    n_goals = len(goal_mask_1d)
    full_reward_targets = list(reward_targets[:2]) + [0.0] * max(0, n_goals - 2)
    goal_ids = torch.arange(n_goals, device=device, dtype=torch.long).unsqueeze(0)
    goal_template = torch.tensor(goal_mask_1d, dtype=torch.bool, device=device).view(1, n_goals, 1)
    rtg_template = torch.tensor(full_reward_targets, dtype=torch.float32, device=device).view(1, n_goals, 1)

    rows: list[dict[str, Any]] = []
    remaining = num_molecules

    while remaining > 0:
        current_batch = min(batch_size, remaining)
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
                "goal_mask": json.dumps(goal_mask_1d),
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
            goal_mask_1d=spec["goal_mask"],
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
