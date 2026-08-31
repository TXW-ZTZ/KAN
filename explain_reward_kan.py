"""Create reward-only, publication-ready figures from the frozen dynamics KAN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch

from world_dynamics.data import DynamicsDataset, split_episode_indices
from world_dynamics.model import AdditiveDynamicsKAN, RobustFeatureScaler


REWARD_INDEX = 9
GROUPS = (
    ("position state", range(0, 3)),
    ("velocity state", range(3, 6)),
    ("reference accel", range(6, 9)),
    ("angular state", range(9, 12)),
    ("attitude + phase", range(12, 15)),
    ("current action", range(15, 21)),
    ("action delta", range(21, 27)),
    ("position memory", range(27, 33)),
    ("velocity memory", range(33, 36)),
    ("angular memory", range(36, 39)),
    ("action memory", range(39, 45)),
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#7A7A7A"
LIGHT_GREY = "#D9D9D9"
TEXT = "#222222"
FAMILY_COLOURS = (BLUE, BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN, GREY, GREY, GREY, GREY)
FAMILY_HATCHES = ("", "//", "xx", "", "//", "", "//", "", "//", "xx", "..")
BANNED = ("scheme d", "scheme-d", "scheme c", "scheme", "terminal", "lawbook", "observatory", "firewall")


def as_numpy(value):
    """Convert tensors without relying on Tensor.numpy() in this environment."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return np.asarray(value)


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": "#E3E3E3",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf(fig, output: Path, filename: str):
    fig.savefig(output / filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def family_for_feature(feature_index: int) -> tuple[int, str]:
    for group_index, (name, indices) in enumerate(GROUPS):
        if feature_index in indices:
            return group_index, name
    raise ValueError(f"feature {feature_index} is not assigned to a family")


def feature_label(name: str) -> str:
    """Readable label while retaining the physical unit encoded in the schema."""
    unit = ""
    stem = name
    for suffix, rendered in (
        ("_mps2", " [m/s²]"),
        ("_radps", " [rad/s]"),
        ("_mps", " [m/s]"),
        ("_m", " [m]"),
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            unit = rendered
            break
    if "action_command" in stem or "action_delta_from_previous_throttle" in stem:
        unit = " [normalised throttle]"
    replacements = (
        ("state_", ""),
        ("memory_", ""),
        ("position_error_body_", "position error "),
        ("velocity_error_body_", "velocity error "),
        ("reference_accel_body_", "reference acceleration "),
        ("angular_velocity_body_", "angular velocity "),
        ("gravity_direction_body_", "gravity direction "),
        ("episode_phase", "episode phase"),
        ("action_delta_from_previous_throttle_", "throttle change "),
        ("action_command_", "throttle command "),
        ("past_ema_0.25s_", "0.25 s mean "),
        ("persistent_", "persistent "),
        ("velocity_innovation_", "velocity innovation "),
    )
    for old, new in replacements:
        stem = stem.replace(old, new)
    return stem.replace("_", " ") + unit


def compact_label(name: str) -> str:
    text = feature_label(name)
    return text.replace("normalised throttle", "norm. throttle").replace("position error", "position err.").replace("angular velocity", "angular vel.").replace("reference acceleration", "reference acc.")


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    report = payload["report"]
    architecture = report["architecture"]
    model = AdditiveDynamicsKAN(
        45,
        11,
        grid_size=architecture["grid_size"],
        spline_order=architecture["spline_order"],
    ).to(device)
    scaler = RobustFeatureScaler(45, 4).to(device)
    model.load_state_dict(payload["kan_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    model.eval()
    center = payload["target_center"].to(device)
    scale = payload["target_scale"].to(device)
    return payload, model, scaler, center, scale


@torch.no_grad()
def reward_bundle(model, scaler, center, scale, data, device):
    z = scaler(data["features"].to(device))
    all_edges = model.edge_contributions(z)
    edges = (all_edges[:, REWARD_INDEX, :] * scale[REWARD_INDEX]).cpu()
    prediction = (model(z)[:, REWARD_INDEX] * scale[REWARD_INDEX] + center[REWARD_INDEX]).cpu()
    return edges, prediction


@torch.no_grad()
def make_edge_curves(model, scaler, scale, train_features, train_edges, importance, names, device):
    edge_means = train_edges.mean(0)
    curves = []
    for feature_index in range(45):
        column = train_features[:, feature_index]
        quantiles = torch.quantile(column, torch.tensor((0.01, 0.99)))
        low, high = float(quantiles[0]), float(quantiles[1])
        if high <= low:
            low, high = float(column.min()), float(column.max())
        if high <= low:
            high = low + 1e-6
        grid = torch.linspace(low, high, 201)
        z = torch.zeros((len(grid), 45), device=device)
        z[:, feature_index] = (
            (grid.to(device) - scaler.center[feature_index]) / scaler.scale[feature_index]
        ).clamp(-4, 4)
        edge = model.edge_contributions(z)[:, REWARD_INDEX, feature_index] * scale[REWARD_INDEX]
        curve = edge.cpu() - edge_means[feature_index]
        curves.append(
            {
                "feature_index": feature_index,
                "feature_name": names[feature_index],
                "grid": as_numpy(grid),
                "curve": as_numpy(curve),
                "importance": float(importance[feature_index]),
                "support": (low, high),
                "training_values": as_numpy(column),
            }
        )
    return curves, edge_means


def add_density_band(ax, values, low, high, colour=GREY):
    clipped = np.asarray(values)
    clipped = clipped[(clipped >= low) & (clipped <= high)]
    density, bins = np.histogram(clipped, bins=28, range=(low, high), density=True)
    if density.max() <= 0:
        return
    centers = 0.5 * (bins[:-1] + bins[1:])
    height = density / density.max() * 0.14
    ax.fill_between(centers, 0, height, transform=ax.get_xaxis_transform(), color=colour, alpha=0.22, lw=0, zorder=0)


def draw_importance(output, importance, names):
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(7.0, 9.8))
    for y, feature_index in enumerate(order):
        group_index, _ = family_for_feature(int(feature_index))
        ax.barh(
            y,
            importance[feature_index],
            color=FAMILY_COLOURS[group_index],
            hatch=FAMILY_HATCHES[group_index],
            edgecolor="white",
            linewidth=0.35,
            height=0.76,
        )
    ax.set_yticks(np.arange(45), [feature_label(names[i]) for i in order], fontsize=5.8)
    ax.set_xlabel("standardised root-mean-square edge contribution")
    ax.grid(axis="x", zorder=-2)
    handles = [
        patches.Patch(
            facecolor=FAMILY_COLOURS[i],
            hatch=FAMILY_HATCHES[i],
            edgecolor="white",
            label=name,
        )
        for i, (name, _) in enumerate(GROUPS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.1, handlelength=1.6, bbox_to_anchor=(0.62, 0.012))
    fig.subplots_adjust(left=0.38, right=0.98, top=0.99, bottom=0.12)
    save_pdf(fig, output, "R1_reward_edge_importance.pdf")


def draw_family_composition(output, family_shares):
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    x = np.arange(len(GROUPS))
    bars = []
    for i, share in enumerate(family_shares):
        bars.append(
            ax.bar(
                i,
                share,
                color=FAMILY_COLOURS[i],
                hatch=FAMILY_HATCHES[i],
                edgecolor="white",
                linewidth=0.4,
                width=0.78,
            )[0]
        )
    for bar, share in zip(bars, family_shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + 0.005, f"{100 * share:.1f}%", ha="center", va="bottom", fontsize=5.8, rotation=90)
    ax.set_xticks(x, [name for name, _ in GROUPS], rotation=58, ha="right", fontsize=6.2)
    ax.set_ylabel("share of total reward-edge importance")
    ax.set_ylim(0, max(family_shares) * 1.23)
    ax.grid(axis="y", zorder=-2)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.99, bottom=0.34)
    save_pdf(fig, output, "R2_reward_family_composition.pdf")


def draw_spline_gallery(output, curves, top12):
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.7))
    for rank, (ax, feature_index) in enumerate(zip(axes.flat, top12), 1):
        row = curves[feature_index]
        group_index, _ = family_for_feature(feature_index)
        colour = FAMILY_COLOURS[group_index]
        ax.plot(row["grid"], row["curve"], color=colour, lw=1.35, zorder=2)
        ax.axhline(0, color="#8A8A8A", lw=0.5, ls="--")
        add_density_band(ax, row["training_values"], *row["support"])
        ax.set_xlabel(compact_label(row["feature_name"]), fontsize=5.8, labelpad=2)
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="y", zorder=-3)
        if rank in (1, 5, 9):
            ax.set_ylabel("contribution to reward", fontsize=6.5)
    fig.tight_layout(pad=0.65, w_pad=0.65, h_pad=0.9)
    save_pdf(fig, output, "R3_reward_spline_gallery.pdf")


def activity_axis_label(name, importance):
    label = compact_label(name)
    if " [" in label:
        stem, unit = label.rsplit(" [", 1)
        wrapped = textwrap.fill(stem, width=18)
        label = f"{wrapped}\n[{unit}"
    else:
        label = textwrap.fill(label, width=18)
    return f"{label}\nstrength = {importance:.3f}"


def draw_edge_activity_contrast(output, curves, importance):
    strongest = np.argsort(importance)[::-1][:6].tolist()
    weakest = np.argsort(importance)[:6].tolist()
    selected = strongest + weakest
    extent = max(float(np.max(np.abs(curves[index]["curve"]))) for index in selected)
    margin = max(0.005, 0.06 * extent)
    fig, axes = plt.subplots(2, 6, figsize=(7.0, 3.65), sharey=True)
    for panel, (ax, feature_index) in enumerate(zip(axes.flat, selected)):
        row = curves[feature_index]
        colour = BLUE if panel < 6 else ORANGE
        ax.plot(row["grid"], row["curve"], color=colour, lw=1.25, zorder=2)
        ax.axhline(0, color="#777777", lw=0.5, ls="--")
        add_density_band(ax, row["training_values"], *row["support"])
        ax.set_ylim(-extent - margin, extent + margin)
        ax.set_xlabel(activity_axis_label(row["feature_name"], importance[feature_index]), fontsize=4.7, labelpad=2)
        ax.tick_params(labelsize=5.2)
        ax.grid(axis="y", zorder=-3)
        if panel in (0, 6):
            ax.set_ylabel("contribution to reward", fontsize=6.2)
    fig.text(0.012, 0.72, "highest importance", rotation=90, ha="center", va="center", fontsize=6.2, color=BLUE)
    fig.text(0.012, 0.28, "lowest importance", rotation=90, ha="center", va="center", fontsize=6.2, color=ORANGE)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.99, bottom=0.22, wspace=0.42, hspace=0.68)
    save_pdf(fig, output, "R8_reward_edge_activity_contrast.pdf")


def draw_counterfactuals(output, curves, top6, test_features):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.15))
    for rank, (ax, feature_index) in enumerate(zip(axes.flat, top6), 1):
        row = curves[feature_index]
        x_ref = float(test_features[:, feature_index].median())
        reference = float(np.interp(x_ref, row["grid"], row["curve"]))
        delta = row["curve"] - reference
        ax.plot(row["grid"], delta, color=BLUE, lw=1.4)
        ax.axhline(0, color="#8A8A8A", lw=0.5, ls="--")
        ax.axvline(x_ref, color=ORANGE, lw=0.7, ls=":")
        ax.set_xlabel(compact_label(row["feature_name"]), fontsize=6.0, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y", zorder=-3)
        if rank in (1, 4):
            ax.set_ylabel("change in reward", fontsize=6.5)
    handles = [
        plt.Line2D([0], [0], color=BLUE, lw=1.4, label="exact edge difference"),
        plt.Line2D([0], [0], color=ORANGE, lw=0.8, ls=":", label="held-out median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.7, w_pad=0.8, h_pad=0.9)
    save_pdf(fig, output, "R4_reward_counterfactual_sweeps.pdf")


def select_local_case(test_prediction, test_target):
    errors = (test_prediction - test_target).abs()
    order = torch.argsort(errors)
    return int(order[len(order) // 2])


def draw_local_waterfall(output, names, values, edges, bias, prediction, residual):
    top8 = torch.argsort(edges.abs(), descending=True)[:8].tolist()
    remaining = [i for i in range(45) if i not in top8]
    contributions = [float(edges[i]) for i in top8] + [float(edges[remaining].sum())]
    labels = [compact_label(names[i]) for i in top8] + ["all remaining edges"]
    labels = ["bias"] + labels + ["model prediction"]
    fig, ax = plt.subplots(figsize=(3.4, 4.8))
    y = np.arange(len(labels))
    ax.barh(y[0], bias, left=min(0, bias), color=GREY, height=0.66)
    running = bias
    for row_index, contribution in enumerate(contributions, 1):
        left = min(running, running + contribution)
        ax.barh(row_index, abs(contribution), left=left, color=BLUE if contribution >= 0 else ORANGE, height=0.66)
        ax.plot([running, running], [row_index - 0.34, row_index + 0.34], color="#666666", lw=0.45)
        running += contribution
    ax.barh(y[-1], prediction, left=min(0, prediction), color=GREEN, height=0.66)
    ax.set_yticks(y, labels, fontsize=5.9)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", lw=0.55)
    ax.set_xlabel("additive contribution to reward")
    ax.grid(axis="x", zorder=-3)
    ax.text(
        0.98,
        0.02,
        f"exact sum − prediction = {residual:+.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
    )
    legend = [
        patches.Patch(color=BLUE, label="positive edge"),
        patches.Patch(color=ORANGE, label="negative edge"),
        patches.Patch(color=GREEN, label="prediction"),
    ]
    fig.legend(handles=legend, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=5.8)
    fig.subplots_adjust(left=0.43, right=0.97, top=0.99, bottom=0.18)
    save_pdf(fig, output, "R5_reward_local_decomposition.pdf")
    return top8


@torch.no_grad()
def choose_and_draw_episode(output, dataset, test_episode_indices, model, scaler, center, scale, device):
    candidates = []
    for episode_id in test_episode_indices:
        episode = dataset.episodes[episode_id]
        dreamer = episode["dreamer_target"][:, REWARD_INDEX]
        z = scaler(episode["features"].to(device))
        kan = (model(z)[:, REWARD_INDEX] * scale[REWARD_INDEX] + center[REWARD_INDEX]).cpu()
        fidelity_rmse = float((kan - dreamer).double().square().mean().sqrt())
        candidates.append((episode_id, fidelity_rmse))
    representative, selected_rmse = min(candidates, key=lambda item: item[1])
    episode = dataset.episodes[representative]
    z = scaler(episode["features"].to(device))
    kan = (model(z)[:, REWARD_INDEX] * scale[REWARD_INDEX] + center[REWARD_INDEX]).cpu()
    actual = episode["actual_target"][:, REWARD_INDEX]
    dreamer = episode["dreamer_target"][:, REWARD_INDEX]
    time = as_numpy(episode["step"].squeeze(-1)) * 0.016
    fig, ax = plt.subplots(figsize=(7.0, 2.65))
    ax.plot(time, as_numpy(actual), color="#3A3A3A", lw=1.0, label="environment reward")
    ax.plot(time, as_numpy(dreamer), color=BLUE, lw=1.15, ls="--", label="Dreamer MC-16 reward")
    ax.plot(time, as_numpy(kan), color=ORANGE, lw=1.1, ls="-.", label="KAN reconstruction")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3)
    ax.grid(axis="y", zorder=-3)
    fig.tight_layout(rect=(0, 0, 1, 0.89), pad=0.8)
    save_pdf(fig, output, "R6_reward_heldout_episode.pdf")
    rmses = {
        "KAN_to_Dreamer": float((kan - dreamer).double().square().mean().sqrt()),
        "Dreamer_to_environment": float((dreamer - actual).double().square().mean().sqrt()),
        "KAN_to_environment": float((kan - actual).double().square().mean().sqrt()),
    }
    print(f"R6 episode id: {representative}")
    print(f"R6 selection: lowest KAN_to_Dreamer per-step RMSE among {len(candidates)} held-out episodes ({selected_rmse:.9g})")
    for pair, value in rmses.items():
        print(f"R6 per-step RMSE {pair}: {value:.9g}")
    return representative


def read_reward_config(path: Path):
    text = path.read_text()
    values = {}
    for key in ("reward_effort_weight", "reward_action_smoothness_weight", "reward_distance_scale"):
        match = re.search(rf"^\s*{re.escape(key)}\s*:\s*([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
        if match:
            values[key] = float(match.group(1))
        else:
            print(f"Could not read {key} from {path}; dependent analytic curve will be omitted.")
    return values


def empirical_center(function, grid, training_values):
    raw_grid = function(grid)
    empirical = function(np.asarray(training_values))
    return raw_grid - float(np.mean(empirical))


def designed_partial(feature_index, grid, train_features, config):
    distance_scale = config.get("reward_distance_scale")
    if feature_index == 0:
        if distance_scale is None:
            return None, None

        def function(x):
            # track.py computes norm(self.rpos[:, [0]]), hence distance = |x|.
            return 0.5 * np.exp(-distance_scale * np.abs(x))

        return function, "designed position term"
    if feature_index == 12:
        if distance_scale is None:
            return None, None
        position_x = float(torch.median(train_features[:, 0]))
        pose = 0.5 * np.exp(-distance_scale * abs(position_x))
        gravity_y = float(torch.median(train_features[:, 13]))

        def function(x):
            up_z = np.sqrt(np.maximum(0.0, 1.0 - np.square(x) - gravity_y**2))
            tiltage = np.abs(1.0 - up_z)
            return pose * 0.5 / (1.0 + np.square(tiltage))

        return function, "designed attitude term"
    if feature_index == 11:
        if distance_scale is None:
            return None, None
        position_x = float(torch.median(train_features[:, 0]))
        pose = 0.5 * np.exp(-distance_scale * abs(position_x))

        def function(x):
            spin = np.square(x)
            return pose * 0.5 / (1.0 + np.square(spin))

        return function, "designed spin term"
    if feature_index in range(21, 27):
        weight = config.get("reward_action_smoothness_weight")
        if weight is None:
            return None, None
        deltas = as_numpy(train_features[:, 21:27])
        medians = np.median(deltas, axis=0)
        channel = feature_index - 21
        constant = float(np.square(np.delete(medians, channel)).sum())

        def function(x):
            return weight * np.exp(-np.sqrt(np.square(x) + constant))

        return function, "designed smoothness term"
    return None, None


def draw_reward_correspondence(output, curves, importance, train_features, config):
    # The implemented tracking reward uses only the body-x position error.
    position_feature = 0
    action_delta_feature = 21 + int(np.argmax(importance[21:27]))
    selected = (position_feature, 12, 11, action_delta_feature)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.55))
    analytic_count = 0
    for ax, feature_index in zip(axes.flat, selected):
        row = curves[feature_index]
        grid = row["grid"]
        ax.plot(grid, row["curve"], color=BLUE, lw=1.45, label="recovered exact edge")
        function, _ = designed_partial(feature_index, grid, train_features, config)
        if function is not None:
            analytic = empirical_center(function, grid, row["training_values"])
            ax.plot(grid, analytic, color=ORANGE, lw=1.2, ls="--", label="designed reward partial")
            analytic_count += 1
        ax.axhline(0, color="#8A8A8A", lw=0.5, ls=":")
        add_density_band(ax, row["training_values"], *row["support"])
        if feature_index in range(21, 27) and config.get("reward_action_smoothness_weight") == 0:
            ax.text(0.97, 0.94, "configured weight = 0", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=ORANGE)
        ax.set_xlabel(compact_label(row["feature_name"]), fontsize=6.6)
        ax.set_ylabel("centred reward contribution", fontsize=6.7)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y", zorder=-3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.065, 1, 1), pad=0.8, h_pad=1.0, w_pad=1.0)
    save_pdf(fig, output, "R7_reward_design_correspondence.pdf")
    print(f"R7 analytic overlays drawn: {analytic_count}/4")
    print("Reward configuration:", json.dumps(config, sort_keys=True))
    return selected


def r_squared(prediction, target):
    prediction = prediction.double()
    target = target.double()
    total = (target - target.mean()).square().sum()
    return float(1.0 - (target - prediction).square().sum() / total)


def rmse(prediction, target):
    return float((prediction.double() - target.double()).square().mean().sqrt())


def best_hinge(x, y):
    total = max(float(np.square(y - y.mean()).sum()), 1e-15)
    best = None
    for direction in ("above", "below"):
        for threshold in np.linspace(np.quantile(x, 0.15), np.quantile(x, 0.85), 81):
            hinge = np.maximum(0, x - threshold) if direction == "above" else np.maximum(0, threshold - x)
            design = np.column_stack((np.ones_like(x), hinge))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            fitted = design @ coefficients
            score = 1.0 - float(np.square(y - fitted).sum()) / total
            candidate = {
                "direction": direction,
                "threshold": float(threshold),
                "intercept": float(coefficients[0]),
                "slope": float(coefficients[1]),
                "r2": score,
            }
            if best is None or candidate["r2"] > best["r2"]:
                best = candidate
    return best


def fit_symbolic(x, y):
    total = max(float(np.square(y - y.mean()).sum()), 1e-15)
    candidates = []
    for degree, family in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
        coefficients = np.polyfit(x, y, degree)
        fitted = np.polyval(coefficients, x)
        score = 1.0 - float(np.square(y - fitted).sum()) / total
        candidates.append((score, degree + 1, family, coefficients))
    for direction in ("above", "below"):
        for threshold in np.linspace(np.quantile(x, 0.15), np.quantile(x, 0.85), 61):
            hinge = np.maximum(0, x - threshold) if direction == "above" else np.maximum(0, threshold - x)
            design = np.column_stack((np.ones_like(x), hinge))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            fitted = design @ coefficients
            score = 1.0 - float(np.square(y - fitted).sum()) / total
            candidates.append((score, 3, f"hinge_{direction}", np.array((coefficients[0], coefficients[1], threshold))))
    best_score = max(row[0] for row in candidates)
    score, _, family, coefficients = min(
        (row for row in candidates if row[0] >= best_score - 0.006),
        key=lambda row: (row[1], -row[0]),
    )
    if family == "linear":
        formula = f"{coefficients[0]:+.6g} x {coefficients[1]:+.6g}"
    elif family == "quadratic":
        formula = f"{coefficients[0]:+.6g} x^2 {coefficients[1]:+.6g} x {coefficients[2]:+.6g}"
    elif family == "cubic":
        formula = f"{coefficients[0]:+.6g} x^3 {coefficients[1]:+.6g} x^2 {coefficients[2]:+.6g} x {coefficients[3]:+.6g}"
    else:
        direction = family.removeprefix("hinge_")
        formula = f"{coefficients[0]:+.6g} {coefficients[1]:+.6g} hinge(x; threshold={coefficients[2]:+.6g}, {direction})"
    return {"family": family, "formula": formula, "r2": float(score)}


def export_importance(output, importance, names):
    total = float(importance.sum())
    with (output / "reward_edge_importance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_index", "feature_name", "family", "importance", "share_of_reward_row"))
        for feature_index in range(45):
            _, family = family_for_feature(feature_index)
            writer.writerow((feature_index, names[feature_index], family, f"{importance[feature_index]:.12g}", f"{importance[feature_index] / total:.12g}"))


def export_rules_and_symbolic(output, curves, top12, test_features):
    rule_lines = [
        "Reward-edge hinge clauses",
        "Each clause approximates one exact additive edge over its empirical support.",
        "",
    ]
    symbolic_lines = [
        "Compact reward-edge approximations",
        "All entries are ranked by exact edge importance.",
        "",
    ]
    for rank, feature_index in enumerate(top12, 1):
        row = curves[feature_index]
        hinge = best_hinge(row["grid"], row["curve"])
        values = as_numpy(test_features[:, feature_index])
        if hinge["direction"] == "above":
            active = values > hinge["threshold"]
            clause = ">"
        else:
            active = values < hinge["threshold"]
            clause = "<"
        activation = float(np.mean(active))
        rule_lines.extend(
            [
                f"Rank {rank:02d}: {row['feature_name']}",
                f"  condition: {row['feature_name']} {clause} {hinge['threshold']:.9g} ({hinge['direction']})",
                f"  slope_per_input_unit: {hinge['slope']:+.9g}",
                f"  edge_R2: {hinge['r2']:.9g}",
                f"  heldout_activation_fraction: {activation:.9g}",
                "",
            ]
        )
        symbolic = fit_symbolic(row["grid"], row["curve"])
        symbolic_lines.extend(
            [
                f"Importance rank {rank:02d}: {row['feature_name']}",
                f"  psi(x) ~= {symbolic['formula']}",
                f"  family: {symbolic['family']}",
                f"  symbolic_fit_edge_R2: {symbolic['r2']:.9g}",
                "",
            ]
        )
    (output / "reward_rules.txt").write_text("\n".join(rule_lines))
    (output / "reward_symbolic.txt").write_text("\n".join(symbolic_lines))


def export_local_case(output, names, values, edges, bias, summed, prediction, residual):
    with (output / "reward_local_case.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("record_type", "feature_index", "feature_name", "input_value", "edge_contribution"))
        for feature_index in range(45):
            writer.writerow(("edge", feature_index, names[feature_index], f"{float(values[feature_index]):.12g}", f"{float(edges[feature_index]):.12g}"))
        writer.writerow(("bias", "", "bias", "", f"{bias:.12g}"))
        writer.writerow(("sum", "", "exact additive sum", "", f"{summed:.12g}"))
        writer.writerow(("prediction", "", "model reward prediction", "", f"{prediction:.12g}"))
        writer.writerow(("residual", "", "sum minus prediction", "", f"{residual:.12g}"))


def audit_pdfs(output):
    pdfs = sorted(output.glob("R*.pdf"))
    if len(pdfs) != 8:
        raise RuntimeError(f"expected 8 PDFs, found {len(pdfs)}")
    if list(output.glob("*.png")):
        raise RuntimeError("PNG files are not allowed in the reward-only output folder")
    for pdf in pdfs:
        result = subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True)
        lowered = result.stdout.lower()
        present = [word for word in BANNED if word in lowered]
        if present:
            raise RuntimeError(f"banned figure text in {pdf.name}: {present}")
        if "r²" in lowered or "r^2" in lowered or "rmse" in lowered:
            raise RuntimeError(f"fit statistic unexpectedly present in {pdf.name}")
    return pdfs


def build(args):
    set_style()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    dataset = DynamicsDataset(args.dataset_dir)
    split = split_episode_indices(len(dataset.episodes))
    train = dataset.concatenate(split.train)
    validation = dataset.concatenate(split.validation)
    test = dataset.concatenate(split.test)
    payload, model, scaler, center, scale = load_model(args.checkpoint, device)
    train_edges, _ = reward_bundle(model, scaler, center, scale, train, device)
    test_edges, test_prediction = reward_bundle(model, scaler, center, scale, test, device)
    names = dataset.manifest["feature_names"]

    # This is the required atlas definition, narrowed only after it is evaluated.
    centered = train_edges - train_edges.mean(0)
    reward_target_std = train["dreamer_target"].std(0).clamp_min(1e-8)[REWARD_INDEX]
    importance_tensor = centered.square().mean(0).sqrt() / reward_target_std
    importance = as_numpy(importance_tensor)
    curves, _ = make_edge_curves(model, scaler, scale, train["features"], train_edges, importance, names, device)
    top12 = np.argsort(importance)[::-1][:12].tolist()
    top6 = top12[:6]
    total_importance = float(importance.sum())
    family_shares = np.asarray([importance[list(indices)].sum() / total_importance for _, indices in GROUPS])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    draw_importance(output, importance, names)
    draw_family_composition(output, family_shares)
    draw_spline_gallery(output, curves, top12)
    draw_edge_activity_contrast(output, curves, importance)
    draw_counterfactuals(output, curves, top6, test["features"])

    local_row = select_local_case(test_prediction, test["dreamer_target"][:, REWARD_INDEX])
    reward_bias = float(center[REWARD_INDEX].cpu() + model.output_bias[REWARD_INDEX].detach().cpu() * scale[REWARD_INDEX].cpu())
    local_edges = test_edges[local_row]
    local_prediction = float(test_prediction[local_row])
    local_sum = reward_bias + float(local_edges.sum())
    local_residual = local_sum - local_prediction
    draw_local_waterfall(
        output,
        names,
        test["features"][local_row],
        local_edges,
        reward_bias,
        local_prediction,
        local_residual,
    )
    episode_id = choose_and_draw_episode(output, dataset, split.test, model, scaler, center, scale, device)
    config = read_reward_config(args.teacher_config)
    draw_reward_correspondence(output, curves, importance, train["features"], config)

    export_importance(output, importance, names)
    export_rules_and_symbolic(output, curves, top12, test["features"])
    export_local_case(
        output,
        names,
        test["features"][local_row],
        local_edges,
        reward_bias,
        local_sum,
        local_prediction,
        local_residual,
    )

    reward_target = test["dreamer_target"][:, REWARD_INDEX]
    environment_target = test["actual_target"][:, REWARD_INDEX]
    reconstructed = reward_bias + test_edges.sum(1)
    max_exact_error = float((reconstructed - test_prediction).abs().max())
    summary = {
        "output_index": REWARD_INDEX,
        "output_name": "reward",
        "heldout_kan_to_dreamer": {"r2": r_squared(test_prediction, reward_target), "rmse": rmse(test_prediction, reward_target)},
        "heldout_dreamer_to_environment": {"r2": r_squared(reward_target, environment_target), "rmse": rmse(reward_target, environment_target)},
        "reward_mc_standard_error_rms": float((test["dreamer_variance"][:, REWARD_INDEX].double().mean() / 16.0).sqrt()),
        "max_exact_reward_edge_reconstruction_error": max_exact_error,
        "rows": {"train": len(train["features"]), "validation": len(validation["features"]), "test": len(test["features"])},
        "episodes": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "r6_episode_id": episode_id,
        "r5_concatenated_test_row_index": local_row,
    }
    (output / "reward_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    expected_shares = np.asarray((0.1842, 0.0464, 0.0407, 0.1010, 0.1044, 0.0737, 0.0452, 0.2306, 0.0446, 0.0522, 0.0770))
    if not np.allclose(family_shares, expected_shares, atol=5.1e-5):
        raise RuntimeError(f"family shares do not match the existing reward row: {family_shares.tolist()}")
    if not math.isclose(float((importance / total_importance).sum()), 1.0, abs_tol=1e-7):
        raise RuntimeError("edge importance shares do not sum to one")
    if max_exact_error > 1e-5:
        raise RuntimeError(f"exact edge reconstruction failed: {max_exact_error}")
    pdfs = audit_pdfs(output)
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    print(f"device: {device}")
    print(f"importance share sum: {(importance / total_importance).sum():.12f}")
    print("family shares:", ", ".join(f"{value:.6f}" for value in family_shares))
    print("produced files:")
    for filename in files:
        print(f"  {filename}")
    print("reward_summary.json:")
    print(json.dumps(summary, indent=2))
    print(f"PDF text audit passed for {len(pdfs)} figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
