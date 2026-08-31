"""Read-only validation of the frozen Track + BlueROV DreamerV3 teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_teacher(checkpoint: Path, config: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu")
    encoder_shape = tuple(state["world_model.encoder.0.weight"].shape)
    actor_shape = tuple(state["actor.mean.weight"].shape)
    with config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "task": cfg["task"]["name"],
        "robot": cfg["task"]["drone_model"]["name"],
        "trajectory": cfg["task"]["trajectory"],
        "encoder_input_dim": encoder_shape[1],
        "actor_output_dim": actor_shape[0],
        "future_traj_steps": cfg["task"]["future_traj_steps"],
        "sim_dt": cfg["task"]["sim"]["dt"],
    }
    expected = {
        "task": "Track",
        "robot": "bluerov",
        "trajectory": "lemniscate",
        "encoder_input_dim": 38,
        "actor_output_dim": 6,
        "future_traj_steps": 4,
        "sim_dt": 0.016,
    }
    mismatches = {
        key: {"expected": value, "actual": report[key]}
        for key, value in expected.items()
        if report[key] != value
    }
    report["valid"] = not mismatches
    report["mismatches"] = mismatches
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_teacher(args.checkpoint, args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
