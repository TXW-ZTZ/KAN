"""Separate Scheme-D posterior reconstruction, prior transition, and semantic errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from world_dynamics.data import split_episode_indices
from world_dynamics.frozen_world_model import FrozenWorldModel, symexp
from world_dynamics.metrics import regression_report
from world_dynamics.semantics import semanticize


PHYSICAL_INDICES = (0, 1, 2, 3, 4, 5, 9, 10, 11)
PHYSICAL_NAMES = (
    "position_error_body_x",
    "position_error_body_y",
    "position_error_body_z",
    "velocity_error_body_x",
    "velocity_error_body_y",
    "velocity_error_body_z",
    "angular_velocity_body_x",
    "angular_velocity_body_y",
    "angular_velocity_body_z",
)
SELECTED_ABSOLUTE_NAMES = (
    "relative_position_world_x_m","relative_position_world_y_m","relative_position_world_z_m",
    "quaternion_w","quaternion_x","quaternion_y","quaternion_z",
    "linear_velocity_world_x_mps","linear_velocity_world_y_mps","linear_velocity_world_z_mps",
    "angular_velocity_world_x_radps","angular_velocity_world_y_radps","angular_velocity_world_z_radps",
)
RAW_GROUPS = {
    "future_targets_world": tuple(range(0, 12)),
    "quaternion_wxyz": tuple(range(12, 16)),
    "world_linear_angular_velocity": tuple(range(16, 22)),
    "heading_world": tuple(range(22, 25)),
    "up_world": tuple(range(25, 28)),
    "previous_throttle_observation": tuple(range(28, 34)),
    "episode_phase_repeated": tuple(range(34, 38)),
}


def _decode(model: FrozenWorldModel, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    state = torch.cat((h, z.reshape(z.shape[0], -1)), dim=-1)
    return symexp(model.decoder(state))


def _selected_absolute(raw: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """Return the 13 primary physical channels with a valid quaternion.

    Quaternion predictions are normalized and, when a reference is supplied,
    sign-aligned because q and -q encode the same physical attitude.
    """
    quaternion=raw[:,12:16]
    quaternion=quaternion/quaternion.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    if reference is not None:
        reference_q=reference[:,12:16]
        reference_q=reference_q/reference_q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        sign=torch.where((quaternion*reference_q).sum(-1,keepdim=True)<0,-1.0,1.0)
        quaternion=quaternion*sign
    return torch.cat((raw[:,0:3],quaternion,raw[:,16:22]),-1)


def _quaternion_angle_degrees(prediction: torch.Tensor,target: torch.Tensor) -> dict:
    p=prediction[:,3:7]; y=target[:,3:7]
    cosine=(p*y).sum(-1).abs().clamp(0,1)
    angle=torch.rad2deg(2*torch.acos(cosine))
    return {"mean":float(angle.mean()),"median":float(angle.median()),"q95":float(torch.quantile(angle,.95))}


def _compact(report: dict) -> dict:
    return {
        "mean_r2": report["mean_r2"],
        "median_r2": report["median_r2"],
        "per_output": report["per_output"],
    }


def _raw_group_summary(report: dict) -> dict:
    rows = report["per_output"]
    result = {}
    for name, indices in RAW_GROUPS.items():
        r2 = [rows[index]["r2"] for index in indices if rows[index]["r2"] is not None]
        result[name] = {
            "mean_r2": sum(r2) / len(r2) if r2 else None,
            "mean_rmse": sum(rows[index]["rmse"] for index in indices) / len(indices),
        }
    return result


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    source = args.source_dir.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    cfg = OmegaConf.load(args.teacher_config)
    model = FrozenWorldModel.from_checkpoint(args.checkpoint, cfg.algo, device)
    test_episodes = split_episode_indices(len(manifest["episode_files"]), args.split_seed).test

    current_raw_rows, actual_raw_rows, posterior_raw_rows, prior_raw_rows = [], [], [], []
    current_semantic_rows, actual_semantic_rows = [], []
    posterior_semantic_rows, prior_semantic_rows = [], []
    posterior_reward_rows, prior_reward_rows, actual_reward_rows = [], [], []
    kl_rows = []

    for episode_index in test_episodes:
        item = manifest["episode_files"][episode_index]
        payload = torch.load(source / item["path"], map_location="cpu", weights_only=True)
        obs = payload["raw_observation"].float().to(device)
        actions = payload["teacher_action_operational"].float().to(device)
        uniforms = payload["posterior_uniform"].float().to(device)
        semantics = payload["semantics"].float().to(device)
        steps = len(obs)

        h, z = model.initial(1, device, obs.dtype)
        previous_action = torch.zeros(1, 6, device=device, dtype=obs.dtype)
        hs, zs, logits = [], [], []
        for step in range(steps):
            h, z, posterior_logits = model.filter_step(
                obs[step : step + 1], h, z, previous_action, uniforms[step : step + 1]
            )
            hs.append(h.squeeze(0))
            zs.append(z.squeeze(0))
            logits.append(posterior_logits.squeeze(0))
            previous_action = actions[step : step + 1]
        hs = torch.stack(hs)
        zs = torch.stack(zs)
        logits = torch.stack(logits)

        posterior_raw = _decode(model, hs[1:], zs[1:])
        posterior_state = torch.cat((hs[1:], zs[1:].reshape(steps - 1, -1)), dim=-1)
        posterior_reward = model.reward(posterior_state)

        generator = torch.Generator(device="cpu").manual_seed(args.prior_seed + episode_index)
        prior_observation_chunks, prior_reward_chunks = [], []
        for start in range(0, steps - 1, args.chunk_size):
            stop = min(steps - 1, start + args.chunk_size)
            count = stop - start
            prior_uniforms = torch.rand(
                (count, args.mc_samples, model.stoch_dim, model.classes), generator=generator
            ).to(device)
            prior_observation, prior_reward, _, prior_logits = model.expected_imagine(
                hs[start:stop], zs[start:stop], actions[start:stop], prior_uniforms
            )
            prior_observation_chunks.append(prior_observation.mean(1))
            prior_reward_chunks.append(prior_reward.mean(1))
            kl_rows.append(model.categorical_kl(logits[start + 1 : stop + 1], prior_logits).cpu())
        prior_raw = torch.cat(prior_observation_chunks)
        prior_reward = torch.cat(prior_reward_chunks)

        current_raw_rows.append(obs[:-1].cpu())
        actual_raw_rows.append(obs[1:].cpu())
        posterior_raw_rows.append(posterior_raw.cpu())
        prior_raw_rows.append(prior_raw.cpu())
        current_semantic_rows.append(semantics[:-1].cpu())
        actual_semantic_rows.append(semantics[1:].cpu())
        posterior_semantic_rows.append(semanticize(posterior_raw).cpu())
        prior_semantic_rows.append(semanticize(prior_raw).cpu())
        posterior_reward_rows.append(posterior_reward.cpu())
        prior_reward_rows.append(prior_reward.cpu())
        actual_reward_rows.append(payload["reward"][:-1].float())

    current_raw = torch.cat(current_raw_rows)
    actual_raw = torch.cat(actual_raw_rows)
    posterior_raw = torch.cat(posterior_raw_rows)
    prior_raw = torch.cat(prior_raw_rows)
    current_semantic = torch.cat(current_semantic_rows)[:, PHYSICAL_INDICES]
    actual_semantic = torch.cat(actual_semantic_rows)[:, PHYSICAL_INDICES]
    posterior_semantic = torch.cat(posterior_semantic_rows)[:, PHYSICAL_INDICES]
    prior_semantic = torch.cat(prior_semantic_rows)[:, PHYSICAL_INDICES]
    raw_names = tuple(f"observation_{index}" for index in range(38))

    posterior_raw_report = regression_report(posterior_raw, actual_raw, raw_names)
    prior_raw_report = regression_report(prior_raw, actual_raw, raw_names)
    selected_actual=_selected_absolute(actual_raw)
    selected_posterior=_selected_absolute(posterior_raw,actual_raw)
    selected_prior=_selected_absolute(prior_raw,actual_raw)
    selected_current=_selected_absolute(current_raw,actual_raw)
    posterior_selected=regression_report(selected_posterior,selected_actual,SELECTED_ABSOLUTE_NAMES)
    prior_selected=regression_report(selected_prior,selected_actual,SELECTED_ABSOLUTE_NAMES)
    persistence_selected=regression_report(selected_current,selected_actual,SELECTED_ABSOLUTE_NAMES)
    posterior_absolute = regression_report(posterior_semantic, actual_semantic, PHYSICAL_NAMES)
    prior_absolute = regression_report(prior_semantic, actual_semantic, PHYSICAL_NAMES)
    posterior_delta = regression_report(
        posterior_semantic - current_semantic,
        actual_semantic - current_semantic,
        tuple(f"delta_{name}" for name in PHYSICAL_NAMES),
    )
    prior_delta = regression_report(
        prior_semantic - current_semantic,
        actual_semantic - current_semantic,
        tuple(f"delta_{name}" for name in PHYSICAL_NAMES),
    )
    persistence = regression_report(current_semantic, actual_semantic, PHYSICAL_NAMES)
    reward_names = ("reward",)
    actual_reward = torch.cat(actual_reward_rows)
    posterior_reward_report = regression_report(
        torch.cat(posterior_reward_rows), actual_reward, reward_names
    )
    prior_reward_report = regression_report(torch.cat(prior_reward_rows), actual_reward, reward_names)

    result = {
        "protocol": "Scheme-D nominal test-split world-model stage audit",
        "source_dataset": str(source),
        "checkpoint": str(args.checkpoint.resolve()),
        "test_episode_indices": list(test_episodes),
        "test_samples": len(actual_raw),
        "mc_samples": args.mc_samples,
        "mean_prior_posterior_kl": float(torch.cat(kl_rows).mean()),
        "posterior_decoder_reconstruction": {
            "raw_observation": _compact(posterior_raw_report),
            "raw_groups": _raw_group_summary(posterior_raw_report),
            "absolute_next_physical_semantic": _compact(posterior_absolute),
            "one_step_delta_physical_semantic": _compact(posterior_delta),
            "reward": _compact(posterior_reward_report),
            "selected_absolute_physical_state": _compact(posterior_selected),
            "selected_quaternion_angle_degrees": _quaternion_angle_degrees(selected_posterior,selected_actual),
        },
        "prior_one_step_prediction": {
            "raw_observation": _compact(prior_raw_report),
            "raw_groups": _raw_group_summary(prior_raw_report),
            "absolute_next_physical_semantic": _compact(prior_absolute),
            "one_step_delta_physical_semantic": _compact(prior_delta),
            "reward": _compact(prior_reward_report),
            "selected_absolute_physical_state": _compact(prior_selected),
            "selected_quaternion_angle_degrees": _quaternion_angle_degrees(selected_prior,selected_actual),
        },
        "persistence_baseline_absolute_next_physical_semantic": _compact(persistence),
        "persistence_baseline_selected_absolute_physical_state": _compact(persistence_selected),
        "interpretation_rule": (
            "posterior reconstruction isolates representation/decoder floor; the prior-minus-posterior "
            "drop adds transition error; absolute-versus-delta uses identical residuals but different "
            "target variance and therefore audits the 16 ms target-resolution effect"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "test_samples": result["test_samples"],
        "posterior_raw_mean_r2": posterior_raw_report["mean_r2"],
        "prior_raw_mean_r2": prior_raw_report["mean_r2"],
        "posterior_absolute_mean_r2": posterior_absolute["mean_r2"],
        "prior_absolute_mean_r2": prior_absolute["mean_r2"],
        "posterior_delta_mean_r2": posterior_delta["mean_r2"],
        "prior_delta_mean_r2": prior_delta["mean_r2"],
        "persistence_mean_r2": persistence["mean_r2"],
        "prior_selected_absolute_mean_r2": prior_selected["mean_r2"],
        "persistence_selected_absolute_mean_r2": persistence_selected["mean_r2"],
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--prior-seed", type=int, default=2026082703)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
