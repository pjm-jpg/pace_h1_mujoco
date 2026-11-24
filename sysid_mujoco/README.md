# MuJoCo SysID

This directory contains a MuJoCo-based system identification pipeline.

## Visualize the H1 tests

Preview the exact command timing used by the Holosoma real-H1 collector before
running any hardware experiment.  This imports `H1SysidPlan` directly from the
local Holosoma checkout, applies commands at 50 Hz with zero-order hold, and
steps MuJoCo at 500 Hz:

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind step --joint left_knee_joint --amplitude-rad 0.01
```

The preview supports separate positive and negative steps, combined steps,
single-frequency and segmented multi-sine tests, and chirps. It includes the
two-second pre-hold, two-second post-hold, and the final one-second safety hold
used by the hardware collector. Add `--headless` for a fast numerical check.
Use `--preserve-collisions` when the preview must also reject nonadjacent robot
self-contact. The output plot and metrics are written under
`sysid_mujoco/results/h1/real_command_preview/`.

Preview the recommended progression before using the same plans on hardware:

```bash
python sysid_mujoco/preview_h1_real_experiment.py --kind positive-step --amplitude-rad 0.01
python sysid_mujoco/preview_h1_real_experiment.py --kind negative-step --amplitude-rad 0.01
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind sine --amplitude-rad 0.01 --sine-frequency-hz 0.5
```

Run the complete simulation-only Experiment 8 check for all nine upper-body
joints with the calibrated lower-body model:

```bash
MPLCONFIGDIR=/tmp/matplotlib-h1-exp8 \
  /home/pjm/.pixi/bin/pixi run -e mujoco \
  python sysid_mujoco/validate_h1_upper_body_experiments.py
```

This runs one bidirectional step and one 0.5/1/2 Hz segmented multi-sine plan
per joint (18 runs total) without initializing DDS. Results are saved under
`sysid_mujoco/results/h1/upper_body_experiment_08/`; the experiment design,
acceptance criteria, and current 18/18 result are documented in
`docs/experiments/experiment_08_h1_upper_body_simulation_validation.md`.

Open MuJoCo's interactive viewer and run a visible left-knee step test:

```bash
python sysid_mujoco/view_h1_motion.py
```

Other simulation-only motions can be selected explicitly:

```bash
python sysid_mujoco/view_h1_motion.py --motion sine --amplitude 0.2 --frequency 0.5
python sysid_mujoco/view_h1_motion.py --motion chirp --amplitude 0.15 --frequency 2.0
```

The displayed amplitudes are deliberately large enough to see in simulation;
do not reuse them as real-H1 experiment limits.

Generate deterministic fitting and held-out validation recordings with known
hidden left-knee parameters:

```bash
python datasets/generate_h1_synthetic_dataset.py
```

The generator writes `synthetic_left_knee_fit.pt`,
`synthetic_left_knee_validation.pt`, the hidden-truth XML, and a provenance
JSON under `datasets/h1/`. The fitting entry point reads only the `.pt`
recording; use the truth JSON only after optimization when checking recovery.

Run the reproducible three-parameter recovery experiment:

```bash
python sysid_mujoco/cmaes_fit.py \
  --robot h1 \
  --dataset datasets/h1/synthetic_left_knee_fit.pt \
  --joints left_knee_joint \
  --score-joints left_knee_joint \
  --population-size 16 \
  --max-iters 50 \
  --seed 7 \
  --output-dir sysid_mujoco/results/h1/synthetic_recovery

python sysid_mujoco/eval_fit.py \
  --robot h1 \
  --dataset datasets/h1/synthetic_left_knee_validation.pt \
  --identified-model-xml \
    sysid_mujoco/results/h1/synthetic_recovery/identified_fixed_base.xml \
  --joints left_knee_joint \
  --output-dir sysid_mujoco/results/h1/synthetic_validation \
  --no-show
```

With the deterministic, noise-free default data, seed 7 recovers all three
hidden parameters to better than 0.001% and reduces held-out position RMSE by
more than 99.99%. This is a software-pipeline check, not evidence that the same
parameters describe the real robot.

## H1: DDS conversion and CMA-ES

Experiment 1 passive observations contain no commands and must be checked,
not converted for fitting:

```bash
python datasets/analyze_h1_observe_log.py \
  datasets/h1/raw/<timestamp>_h1_observe/raw.npz
```

This writes `state_report.json` and `state_overview.png` beside the recording.
The GO `MotorState.lost` value is treated as a cumulative communication-loss
counter. Historical nonzero values and small background bursts are allowed;
the default run allowance is 10 counts plus 0.5 count per elapsed second.
The converter deliberately rejects a manifest or NPZ with
`command_available=false` or an all-zero `cmd_seq`.

The H1 integration uses the 19-DOF Holosoma MJCF and its real GO2-protocol
mapping. DDS has 20 physical motor slots; physical motor ID 9 is unused. The
converter maps physical motor order to the model's joint order and never treats
slot 9 as a joint.

Convert a JSONL, CSV, or NPZ recording to the repository dataset format:

```bash
python datasets/convert_h1_dds_log.py raw_h1_dds.jsonl datasets/h1/knee_fit.pt
```

Required signals are `timestamp`, `q_cmd`, `q_real`, and `dq_real`. Optional
signals are `dq_cmd`, `kp`, `kd`, `tau_ff`, and `tau_est`; missing command velocities and
feed-forward torques become zero, while missing gains use `config.py`. Arrays
may contain 20 values in physical DDS motor order or 19 values in H1 joint
order. By default the output is uniformly resampled to 500 Hz. Varying gains
are rejected because the current MuJoCo actuator model uses one Kp/Kd pair per
joint for a complete trajectory. The fallback gains match the current
`h1_19dof_loco_real`/`h1_19dof_loco_wbt_real` runtime override
(`h1_19dof.motor_kp/motor_kd`), not the standalone values stored in
`h1_19dof_real_go`. Recorded DDS gains always take precedence.

State channels are linearly resampled. DDS command channels (`q_cmd`,
`dq_cmd`, `kp`, `kd`, and `tau_ff`) use zero-order hold because a LowCmd stays
active until the next write. A generic `tau` input column is deliberately not
accepted as `tau_ff`; measured torque must be named `tau_est` so it cannot be
mistaken for a simulation control.

The companion Holosoma collector writes `manifest.json` and `raw.npz` with 20
physical motor slots. Convert its output directly:

```bash
python datasets/convert_h1_dds_log.py \
  /path/to/records/h1/<run>/raw.npz \
  datasets/h1/left_knee_step_fit.pt \
  --start-time-s 8 --end-time-s 22
```

`--start-time-s` and `--end-time-s` crop in the raw recording's relative time
before uniform resampling. Use them to exclude the suspended prepare/return
ramps from a joint-excitation fit; the output stores the requested and actual
crop ranges for traceability.

Run the PACE-style CMA-ES entry point on a single knee first:

```bash
python sysid_mujoco/cmaes_fit.py \
  --robot h1 \
  --dataset datasets/h1/knee_fit_0p5hz.pt datasets/h1/knee_fit_2hz.pt \
  --joints left_knee_joint \
  --score-joints left_knee_joint \
  --command-delay-ms 0 \
  --population-size 32 \
  --max-iters 100
```

This command uses the same fixed-base model builder, data loader, MuJoCo
rollout, and position/velocity residual as `my_fit.py`; only the optimizer is
changed to CMA-ES. Results include `best_params.json`, the optimization log,
and nominal/identified fixed-base XML files under
`sysid_mujoco/results/h1/cmaes_<timestamp>/`.

`--score-joints` defaults to the identified joints, so stationary H1 joints do
not dominate a single-knee loss. `--command-delay-ms` applies an explicit
zero-order-held delay to the replayed command; estimate it from held-out step
data instead of asking inertia or damping to absorb a timing error. Every
dataset supplied in one run must have identical recorded Kp/Kd values.

Replay a held-out trajectory against both models and generate the required
three-curve comparison:

```bash
python sysid_mujoco/eval_fit.py \
  --robot h1 \
  --dataset datasets/h1/left_knee_validation.pt \
  --identified-model-xml sysid_mujoco/results/h1/cmaes_<timestamp>/identified_fixed_base.xml \
  --joints left_knee_joint \
  --command-delay-ms 0 \
  --no-show
```

The plot contains desired, real measured, original MuJoCo, and identified
MuJoCo trajectories. `metrics.json` stores baseline and identified RMSE/MAE
for both the selected joints and the complete model.

Use different recordings for fitting and validation. Do not identify link
inertia, armature, PD gains, and delay together in the first knee experiment;
those quantities can compensate for one another.

## H1: combine identified lower-body joints

After selecting one `best_params.json` per joint, merge them into a fixed-base
replay model with `merge_identified_models.py --output-dir ...`. To transfer
the same parameters into the original floating-base model while preserving its
relative mesh paths, use the exact-output form:

```bash
python sysid_mujoco/merge_identified_models.py \
  --base-xml robot_model/h1/h1.xml \
  --best-params <joint-1>/best_params.json <joint-2>/best_params.json \
  --output-xml robot_model/h1/h1_calibrated_lower_body.xml
```

The command writes a parameter provenance manifest beside the output XML. It
does not overwrite `h1.xml`. The current ten-joint candidate and its scene are
`h1_calibrated_lower_body.xml` and
`scene_flat_calibrated_lower_body.xml`.

Freeze/check the 41 official processed datasets, summarize all combined-model
replays, and run the floating-base structural/numerical smoke test with:

```bash
python datasets/freeze_h1_lower_body_dataset_manifest.py
python sysid_mujoco/summarize_h1_lower_body_evaluations.py
python sysid_mujoco/validate_h1_calibrated_model.py
```

The floating-base check confirms topology, exact parameter transfer, unchanged
non-identified joints, model compilation, and finite short simulation. It is
not a standing or policy-safety test. See
`docs/experiments/experiment_07_h1_lower_body_combined_model.md` for the
selected parameters, 41 replay results, and remaining limitations.

## H1: full-body simulation and policy regression

After merging all 19 identified joints, compare the nominal, lower-body and
full-body floating-base models with deterministic full-body tracking and
drop/contact tests:

```bash
python sysid_mujoco/validate_h1_full_body_regression.py
```

Run the saved FastSAC checkpoint through the official Holosoma evaluator on
both the nominal and full-body calibrated models with:

```bash
python sysid_mujoco/run_h1_holosoma_policy_regression.py
```

This is an offline CPU smoke test and never opens DDS. A successful result
proves model/policy pipeline compatibility, not quantitative locomotion parity
or real-robot safety. See
`docs/experiments/experiment_11_h1_full_body_simulation_policy_regression.md`.

By default, it fits per-joint `damping`, `armature`, and `frictionloss` on the
recorded trajectory. From the repository root, run:

```bash
python sysid_mujoco/my_fit.py
```

Inertial identification is optional and has three hierarchical modes:

- `--identify-link-mass`: mass only (`InertiaType.Mass`).
- `--identify-center-of-mass`: mass and CoM (`InertiaType.MassIpos`).
- `--identify-inertia-tensor`: full inertia, including mass and CoM
  (`InertiaType.Pseudo`).

When multiple flags are supplied, the most complete requested mode is used.
For example, full inertial identification is enabled with:

```bash
python sysid_mujoco/my_fit.py \
  --identify-inertia-tensor
```

By default these parameters are added to every massive body downstream of an
articulated joint. Limit them to specific MuJoCo bodies with, for example:

```bash
python sysid_mujoco/my_fit.py \
  --identify-center-of-mass \
  --inertial-bodies link1 link2
```

The full-inertia mode uses MuJoCo SysID's pseudo-inertia Cholesky
parameterization, which guarantees physical consistency without singularities.
The optional bound controls are `--link-mass-scale-bounds`,
`--center-of-mass-offset-bounds`, `--inertia-tensor-scale-bounds`, and
`--inertia-tensor-shear-bounds`.

Notes:

- `my_fit.py` creates a fixed-base robot model and uses MuJoCo's integrated PD
  controller. It saves an HTML report under `sysid_mujoco/results/<robot>/`,
  where tracking and parameter values can be inspected.
