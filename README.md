# Scheme D — frozen Dreamer world-dynamics KAN audit

This directory is a self-contained, offline Scheme-D experiment for the
`Track / BlueROV / lemniscate` setting. It does not modify or import the
runtime MarineGym control package, does not execute Isaac Sim, and never
places the KAN in the control loop. The source episodes and the trained
Dreamer checkpoint are opened read-only; all generated data and results are
written under a new Scheme-D output root.

## Current formal result: nominal in-distribution rerun

The original result below used Scheme-C `evaluate` episodes with flow and
payload enabled. It is retained as an OOD baseline, not as the formal Scheme-D
result. The current run preserves `mode=evaluate` execution semantics but copies
the teacher config's `train` randomization/disturbance blocks to `evaluate`, so
randomization, flow, and payload are all off and the rollout differs from the
teacher training scenario only by its independent seed (`20260827` versus `0`).

Current held-out results:

```text
KAN -> Dreamer physical mean R2:   0.285  (old flow+payload: 0.190)
KAN -> Dreamer reward R2:          0.881
Dreamer -> Isaac selected absolute physical mean R2: 0.8432
Dreamer -> Isaac 16-ms increment stress-test R2:    -165.72
Dreamer -> Isaac reward R2:        0.957
```

The `-165.72` value is retained as a **16-ms innovation-resolution stress
test**, not as the primary physical-state summary.  A float64 recomputation
reproduces every stored R² to within `2.4e-5`.  For each output,
`R² = 1 - (RMSE / target_std)²`; the physical-increment RMSE is `5.3–17.9`
target standard deviations, so the strongly negative values are mathematically
expected and are not a denominator implementation error.  Removing the 10%
largest shared physical residual rows post hoc still gives mean `R² = -115.89`;
therefore outlier deletion does not repair the result and is not used in the
formal analysis.

The formal primary physical-variable set is now the 13 directly observed
absolute next-state channels: world-frame relative position `(x,y,z)`, attitude
quaternion `(w,x,y,z)`, world linear velocity `(x,y,z)`, and world angular
velocity `(x,y,z)`.  Quaternion predictions are normalized and sign-aligned
before scoring because `q` and `-q` encode the same attitude.  The set excludes
derived finite-difference semantics, repeated phase, throttle, and redundant
heading/up channels.  The uniform 13-variable mean Dreamer-prior R² is `0.8432`
(individual range `0.610–0.978`):

```text
Absolute physical group       Dreamer prior R2   Persistence R2
relative position (world)          0.783              0.9991
quaternion wxyz                    0.906              0.9997
world linear velocity              0.820              0.9967
world angular velocity             0.843              0.9955
```

These values are numerically reasonable, but they remain substantially below
the no-change baseline.  They are therefore a decoder calibration result, not
evidence that the model resolves one-step physical innovations.  Closing the
disturbances did not repair that physical reality gap.
The sign-invariant prior attitude error is `17.81°` on average (`13.08°`
median; `44.26°` at the 95th percentile), which is reported alongside the
quaternion-component R².
The stage audit further localizes the problem: posterior/prior raw-observation
mean R2 is `0.881/0.880`, posterior/prior absolute physical-semantic mean R2 is
`0.459/0.460`, and posterior/prior 16-ms delta mean R2 is `-166.20/-166.80`.
Because posterior and prior are nearly identical, the main limitation is the
representation/decoder floor plus semantic and delta-resolution amplification,
not the prior transition.

Formal outputs:

- source: `outputs/kan_world_dynamics_track_bluerov/source_episodes/track_bluerov_paired_seed20260827_20260830T082113Z`
- dataset: `outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_nominal_seed20260827_128ep`
- KAN: `outputs/kan_world_dynamics_track_bluerov/kan/direct_45x11_nominal_seed20260827`
- stage audit: `outputs/kan_world_dynamics_track_bluerov/diagnostics/nominal_stage_audit_seed20260827.json`
- R²/resolution audit: `outputs/kan_world_dynamics_track_bluerov/diagnostics/nominal_r2_resolution_audit_seed20260827.json`
- atlas: `outputs/kan_world_dynamics_track_bluerov/explanations/figure4paper_dynamics_atlas_nominal`

Reproduce the nominal source collection with
`bash experiments/kan_world_dynamics_track_bluerov/run_nominal_collection.sh`.

## Question and claim boundary

The experiment asks which named state, action, and causal-history variables
explain the frozen Dreamer world model's one-step predictions. For output
`j`, the direct additive KAN is

```text
x_t in R^45
y^D_t = E_Dreamer_MC16[Delta p, Delta v, Delta omega, reward, risk | s_t, a_t]

y_hat^KAN_j(x_t) = b_j + sum_{i=1}^{45} psi_{j,i}(x_{t,i}),  j=1,...,11
```

The trained explainer contains exactly `45 × 11 = 495` learnable cubic-spline
edges. No MLP explainer or MLP baseline is trained. The read-only frozen
Dreamer teacher naturally retains the neural heads from its original
checkpoint; those heads are replayed only to construct the offline targets.
Two errors must never be conflated:

```text
explainer fidelity error = KAN - Dreamer
world-model reality error = Dreamer - recorded Isaac outcome
```

The first error validates the explainer. Only a small second error would
allow a physical-dynamics interpretation. The present result does **not**
support that interpretation for the nine physical state increments.

## Frozen evidence protocol

- Source: 128 previously recorded BlueROV episodes, reused read-only.
- Teacher: frozen `best_policy.pt` Dreamer checkpoint.
- Replay: exact recorded posterior filtering from episode reset with recorded
  actions and stored posterior randomness.
- Target: MC16 expectation under the Dreamer prior after `s_t, a_t`, plus its
  predictive variance and prior-posterior KL.
- Inputs: 15 semantic state variables, 6 actions, 6 action deltas, and 18
  strictly causal memory/innovation variables.
- Outputs: three body-frame position-error increments, three velocity-error
  increments, three angular-velocity increments, reward, and termination risk.
- Split: by whole episode (`90 / 19 / 19` train/validation/test), never by row.
- Data volume: 33,032 transitions (`23,197 / 4,024 / 5,811`).

The dynamics dataset manifest records SHA-256 hashes for both the source
manifest and frozen teacher, making the extraction auditable.

### Nominal feature-selection revision

Disabling flow and payload does not require deleting any explicit disturbance
feature: the current 45-D schema contains none.  It does, however, change the
preferred interpretation and next model revision:

- keep the existing 45-D state/action/history schema only for the
  **KAN-to-Dreamer fidelity** model, because history can proxy information in
  the recurrent latent state;
- use a separate nominal **physical-state** model with a Markov-oriented core:
  relative position, vehicle linear velocity and reference velocity as
  separate variables, reference acceleration, angular velocity, gravity/
  attitude, previous throttle, and current action;
- demote episode phase and the 18 EMA/persistence/innovation features to an
  ablation for the physical model.  They may improve fit but should not be
  presented as hydrodynamic state variables;
- if flow or payload is enabled later, add measured flow-relative velocity and
  payload mass/offset explicitly rather than expecting history features to
  identify hidden disturbances.

The current compressed velocity-error input is insufficient for a physical
law claim: nominal drag depends on vehicle velocity, whereas tracking-error
kinematics also depend on reference velocity.  Their difference alone does not
separate those mechanisms.

## Result summary

Held-out KAN-to-Dreamer R² by output:

```text
Delta p:      0.097, 0.245, 0.394
Delta v:      0.089, 0.098, 0.256
Delta omega:  0.155, 0.226, 0.149
reward:       0.852
risk:        -0.129
```

The mean R² over the nine physical Dreamer outputs is `0.190`. This is a
partial explanation, consistent with Scheme D being harder than direct policy
distillation. Reward is captured well. Termination risk is not captured.

Dreamer-to-Isaac reward agreement is strong (`R² = 0.949`, `RMSE = 0.0383`),
but all nine physical-increment R² values are strongly negative. The actual
termination target is constant zero on the retained non-final transitions, so
its R² is intentionally reported as undefined instead of a misleading number.
Thus the spline curves and symbolic rules are explicitly labeled as laws of
the frozen Dreamer surrogate, not certified hydrodynamic laws.

## Outputs

- Dataset: `outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_seed20260827_128ep`
- KAN: `outputs/kan_world_dynamics_track_bluerov/kan/direct_45x11_seed20260827`
- Official figure4paper atlas:
  `outputs/kan_world_dynamics_track_bluerov/explanations/figure4paper_dynamics_atlas`

The atlas contains 29 matched PDF/PNG figure pairs, an indexed JSON audit,
per-output metrics CSV, symbolic spline approximations, and IF-THEN rules.
Its distinctive Scheme-D panels include a three-way evidence simplex,
state/action/history influence chord, triple-model process traces, error
spectra, dual-error episode map, local contribution sunburst, and a three-layer
error budget. PDF is the paper-ready vector master; PNG is for quick review.

## Reproduce

Run from the MarineGym repository root with the existing environment:

```bash
python experiments/kan_world_dynamics_track_bluerov/extract_dynamics_targets.py \
  --source-dir outputs/kan_memory_residual_track_bluerov/source_episodes/track_bluerov_paired_seed20260827_20260826T230205Z \
  --checkpoint /home/ztz/trianed-model/Track/BlueRov/dreamerv3/best_policy.pt \
  --teacher-config /home/ztz/trianed-model/Track/BlueRov/dreamerv3/config.yaml \
  --output-dir outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_seed20260827_128ep \
  --mc-samples 16 --prior-seed 2026082703 --device auto

python experiments/kan_world_dynamics_track_bluerov/train_kan.py \
  outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_seed20260827_128ep \
  --output-dir outputs/kan_world_dynamics_track_bluerov/kan/direct_45x11_seed20260827 \
  --seed 20260827 --device auto

python experiments/kan_world_dynamics_track_bluerov/explain_dynamics_kan.py \
  --dataset-dir outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_seed20260827_128ep \
  --checkpoint outputs/kan_world_dynamics_track_bluerov/kan/direct_45x11_seed20260827/best_kan.pt \
  --output-dir outputs/kan_world_dynamics_track_bluerov/explanations/figure4paper_dynamics_atlas \
  --device auto

PYTHONPATH=experiments/kan_world_dynamics_track_bluerov \
  python -m unittest discover \
  -s experiments/kan_world_dynamics_track_bluerov/tests -v
```

The tests cover feature causality, episode-reset history, deterministic frozen
world-model replay, tensor shapes, exact additive edge reconstruction, and
constant-target metric handling.  Reproduce the denominator/outlier audit with:

```bash
PYTHONPATH=experiments/kan_world_dynamics_track_bluerov \
python experiments/kan_world_dynamics_track_bluerov/audit_r2_resolution.py \
  --dataset-dir outputs/kan_world_dynamics_track_bluerov/dynamics_data/mc16_nominal_seed20260827_128ep \
  --checkpoint outputs/kan_world_dynamics_track_bluerov/kan/direct_45x11_nominal_seed20260827/best_kan.pt \
  --stage-audit outputs/kan_world_dynamics_track_bluerov/diagnostics/nominal_stage_audit_seed20260827.json \
  --output outputs/kan_world_dynamics_track_bluerov/diagnostics/nominal_r2_resolution_audit_seed20260827.json
```
