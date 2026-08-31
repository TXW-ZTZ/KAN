"""Collect Scheme-C Track/BlueROV episodes with explicit posterior randomness.

This script imports MarineGym only to run the existing frozen teacher. It never
edits MarineGym, and every output is written under the caller's new Scheme-C
output root. A dedicated torch.Generator produces posterior uniforms, so the
recorded randomness does not consume or alter the simulator RNG stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv


ROOT = Path(__file__).resolve().parent
for candidate in (Path.cwd(), ROOT.parents[1] if ROOT.parent.name == "experiments" else ROOT):
    if (candidate / "marinegym").is_dir():
        sys.path.insert(0, str(candidate))
        break

from preflight import inspect_teacher  # noqa: E402
from world_dynamics.semantics import SEMANTIC_NAMES, semanticize as semanticize_track_bluerov  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _runtime_config(path: Path, seed: int, *, match_training_scenario: bool = False):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.headless = True
    cfg.enable_livestream = False
    cfg.mode = "evaluate"
    cfg.seed = seed
    cfg.task.env.num_envs = 1
    cfg.env = cfg.task.env
    cfg.sim = cfg.task.sim
    if "eval" in cfg:
        cfg.eval.enabled = False
    if match_training_scenario:
        # Preserve evaluation execution semantics while making every stochastic
        # dynamics/disturbance setting identical to the checkpoint's training
        # scenario.  The rollout seed remains independent from the training seed.
        cfg.task.randomization.evaluate = OmegaConf.create(
            OmegaConf.to_container(cfg.task.randomization.train, resolve=True)
        )
        cfg.task.disturbances.evaluate = OmegaConf.create(
            OmegaConf.to_container(cfg.task.disturbances.train, resolve=True)
        )
    return cfg


def _make_env_and_policy(cfg):
    from marinegym.envs.isaac_env import IsaacEnv
    from marinegym.learning import ALGOS

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device
    )
    return base_env, env, policy


def _sample_posterior(logits: torch.Tensor, *, unimix: float, uniform: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    mixed_logits = torch.log(probs.clamp_min(1e-8))
    gumbel = -torch.log(-torch.log(uniform.clamp(1e-7, 1.0 - 1e-7)))
    index = torch.argmax(mixed_logits + gumbel, dim=-1)
    return torch.nn.functional.one_hot(index, logits.shape[-1]).to(logits.dtype)


@torch.no_grad()
def recorded_teacher_step(policy, tensordict, posterior_generator: torch.Generator):
    obs = tensordict[("agents", "observation")].to(policy.device)
    leading = obs.shape[:-1]
    flat_obs = obs.reshape(-1, policy.obs_dim)
    policy._ensure_carry(flat_obs.shape[0], flat_obs.device, flat_obs.dtype)
    reset = policy._policy_reset_mask(tensordict, leading).to(flat_obs.device)
    if reset.any():
        keep = (~reset).reshape(-1, 1).to(flat_obs.dtype)
        policy._carry_h *= keep
        policy._carry_z *= keep.reshape(-1, 1, 1)
        policy._carry_prev_action *= keep

    wm = policy.world_model
    h = wm.gru(
        torch.cat((policy._carry_z.reshape(flat_obs.shape[0], -1), policy._carry_prev_action), -1),
        policy._carry_h,
    )
    embed = wm.encoder(torch.sign(flat_obs) * torch.log1p(flat_obs.abs()))
    logits = wm.posterior(torch.cat((h, embed), -1))
    uniform = torch.rand(
        logits.shape, device=logits.device, dtype=logits.dtype, generator=posterior_generator
    )
    z = _sample_posterior(logits, unimix=wm.posterior.unimix, uniform=uniform)
    state = wm.state(h, z)
    actor_dist = policy.actor(state)
    mu = actor_dist.base_dist.loc
    action = policy.actor._squash(mu)
    policy._carry_h = h.detach()
    policy._carry_z = z.detach()
    policy._carry_prev_action = action.detach()
    tensordict.set(("agents", "action"), action.reshape(*leading, policy.action_dim))
    return tensordict, {
        "raw_observation": flat_obs.detach(),
        "semantics": semanticize_track_bluerov(flat_obs).detach(),
        "teacher_mu_operational": mu.detach(),
        "teacher_action_operational": action.detach(),
        "posterior_uniform": uniform.detach(),
    }


def _stack(rows):
    return {key: torch.cat([row[key].cpu() for row in rows], 0) for key in rows[0]}


def collect(args: argparse.Namespace) -> Path:
    preflight = inspect_teacher(args.checkpoint, args.teacher_config)
    if not preflight["valid"]:
        raise RuntimeError(f"teacher preflight failed: {preflight['mismatches']}")
    _seed_everything(args.seed)
    cfg = _runtime_config(
        args.teacher_config,
        args.seed,
        match_training_scenario=args.match_training_scenario,
    )
    from marinegym import init_simulation_app

    simulation_app = init_simulation_app(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root.resolve() / f"track_bluerov_paired_seed{args.seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    try:
        base_env, env, policy = _make_env_and_policy(cfg)
        state = torch.load(args.checkpoint, map_location=policy.device, weights_only=True)
        incompatibility = policy.load_state_dict(state, strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(str(incompatibility))
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        base_env.eval()
        env.eval()
        env.set_seed(args.seed)
        posterior_generator = torch.Generator(device=policy.device).manual_seed(args.posterior_seed)
        manifest = {
            "schema_version": 1,
            "protocol": "Scheme-C frozen full-context teacher with explicit posterior uniforms",
            "scope": {"task": "Track", "robot": "BlueROV", "trajectory": "lemniscate"},
            "created_utc": stamp,
            "seed": args.seed,
            "posterior_seed": args.posterior_seed,
            "teacher": preflight,
            "teacher_checkpoint_sha256": _sha256(args.checkpoint),
            "teacher_config_sha256": _sha256(args.teacher_config),
            "environment_executes": "recorded full-context frozen teacher action only",
            "posterior_rng_isolated_from_environment": True,
            "dynamics_condition": (
                "matches_teacher_training_scenario"
                if args.match_training_scenario
                else "teacher_evaluation_scenario"
            ),
            "runtime_scenario": {
                "execution_mode": "evaluate",
                "training_scenario_copied_to_evaluate": bool(args.match_training_scenario),
                "randomization": OmegaConf.to_container(
                    cfg.task.randomization.evaluate, resolve=True
                ),
                "disturbances": OmegaConf.to_container(
                    cfg.task.disturbances.evaluate, resolve=True
                ),
            },
            "semantic_names": list(SEMANTIC_NAMES),
            "episode_files": [],
        }
        for episode_index in range(args.episodes):
            policy._carry_h = policy._carry_z = policy._carry_prev_action = None
            td = env.reset()
            rows = []
            for step in range(int(base_env.max_episode_length)):
                td, labels = recorded_teacher_step(policy, td, posterior_generator)
                transition = env.step(td)
                next_td = transition.get("next")
                labels["reward"] = next_td[("agents", "reward")].reshape(-1, 1).detach()
                labels["done"] = next_td["done"].reshape(-1, 1).detach().bool()
                labels["terminated"] = next_td["terminated"].reshape(-1, 1).detach().bool()
                labels["step"] = torch.full((1, 1), step, device=policy.device, dtype=torch.int32)
                rows.append(labels)
                if bool(labels["done"].any()):
                    break
                td = next_td.clone()
            payload = _stack(rows)
            path = output / f"episode_{episode_index:04d}.pt"
            torch.save(payload, path)
            manifest["episode_files"].append({
                "episode": episode_index,
                "path": path.name,
                "steps": int(payload["step"].shape[0]),
                "return": float(payload["reward"].sum()),
                "sha256": _sha256(path),
            })
            print(
                f"episode={episode_index + 1}/{args.episodes} steps={payload['step'].shape[0]} "
                f"return={float(payload['reward'].sum()):.3f}", flush=True
            )
        manifest["episodes"] = len(manifest["episode_files"])
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output
    finally:
        simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--posterior-seed", type=int, default=2026082701)
    parser.add_argument(
        "--match-training-scenario",
        action="store_true",
        help=(
            "Copy the teacher config's train randomization/disturbance blocks "
            "to evaluate so the rollout differs from training only by seed."
        ),
    )
    args = parser.parse_args()
    print(collect(args))


if __name__ == "__main__":
    main()
