# MuJoCo SysID

This directory contains a MuJoCo-based system identification pipeline.

## Visualize the H1 tests

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
  datasets/h1/left_knee_step_fit.pt
```

Run the PACE-style CMA-ES entry point on a single knee first:

```bash
python sysid_mujoco/cmaes_fit.py \
  --robot h1 \
  --dataset datasets/h1/knee_fit.pt \
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
