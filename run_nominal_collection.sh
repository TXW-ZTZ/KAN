#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MARINEGYM_ROOT="${MARINEGYM_ROOT:-/home/ztz/MarineGym}"
TEACHER_DIR="${TEACHER_DIR:-/home/ztz/trianed-model/Track/BlueRov/dreamerv3}"
CONDA_SETUP="${CONDA_SETUP:-/home/ztz/miniconda3/etc/profile.d/conda.sh}"
ISAAC_SETUP="${ISAAC_SETUP:-/home/ztz/isaac410/setup_conda_env.sh}"

source "$CONDA_SETUP"
source "$ISAAC_SETUP"
conda activate "${CONDA_ENV:-marinegym}"

set -u

cd "$MARINEGYM_ROOT"

python "$SCRIPT_DIR/collect_paired_teacher.py" \
  --checkpoint "$TEACHER_DIR/best_policy.pt" \
  --teacher-config "$TEACHER_DIR/config.yaml" \
  --output-root "${OUTPUT_ROOT:-outputs/kan_world_dynamics_track_bluerov/source_episodes}" \
  --episodes 128 \
  --seed 20260827 \
  --posterior-seed 2026082701 \
  --match-training-scenario
