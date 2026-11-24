"""Run a diagnostic raw-ONNX H1 rollout on nominal and calibrated models.

This helper is only suitable for policies whose ONNX output is already the
final normalized joint action.  FastSAC checkpoints exported by Holosoma also
need the saved observation normalizer and action-boundary transform, so their
results from this helper must not be treated as policy-regression evidence.
Use ``run_h1_holosoma_policy_regression.py`` for the official FastSAC path.

The rollout is offline and never creates a DDS publisher.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from sysid_mujoco.validate_h1_full_body_regression import MODEL_SPECS


DEFAULT_POLICY = Path("/home/pjm/Desktop/holosoma/logs/walk_h1.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--policy-rate-hz", type=float, default=50.0)
    parser.add_argument("--forward-speed-m-s", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "policy_model_comparison_v1"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every model completes both trials without falling.",
    )
    args = parser.parse_args()
    if args.duration_s <= 0 or args.policy_rate_hz <= 0:
        parser.error("Duration and policy rate must be positive.")
    return args


def _metadata(session: ort.InferenceSession) -> dict[str, object]:
    raw = session.get_modelmeta().custom_metadata_map
    required = ("dof_names", "kp", "kd", "action_scale")
    missing = [key for key in required if key not in raw]
    if missing:
        raise RuntimeError(f"ONNX policy metadata is missing {missing}.")
    parsed = {key: json.loads(raw[key]) for key in required}
    experiment = json.loads(raw["experiment_config"]) if "experiment_config" in raw else None
    parsed["experiment_config"] = experiment
    parsed["iteration"] = raw.get("iteration")
    return parsed


def _model_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos: list[int] = []
    qvel: list[int] = []
    actuators: list[int] = []
    for name in config.H1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"Model is missing joint or actuator {name!r}.")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
        actuators.append(actuator_id)
    return np.asarray(qpos), np.asarray(qvel), np.asarray(actuators)


def _rotation_matrix(data: mujoco.MjData) -> np.ndarray:
    pelvis_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise RuntimeError("H1 pelvis body is missing.")
    return data.xmat[pelvis_id].reshape(3, 3).copy()


def _tilt_deg(data: mujoco.MjData) -> float:
    rotation = _rotation_matrix(data)
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))


def _warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        str(index): int(warning.number)
        for index, warning in enumerate(data.warning)
        if warning.number
    }


def _observation(
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
    default_pose: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    rotation = _rotation_matrix(data)
    # Holosoma's locomotion observation rotates world-frame root angular
    # velocity into the robot frame.  Projected gravity uses the same inverse
    # base rotation.
    base_ang_vel = rotation.T @ data.qvel[3:6]
    projected_gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])
    values = (
        0.25 * base_ang_vel,
        projected_gravity,
        command[:2],
        command[2:3],
        data.qpos[qpos_addresses] - default_pose,
        0.05 * data.qvel[qvel_addresses],
        last_action,
        np.sin(phase),
        np.cos(phase),
    )
    observation = np.concatenate(values).astype(np.float32)[None, :]
    if observation.shape != (1, 70):
        raise RuntimeError(f"Unexpected policy observation shape {observation.shape}.")
    return observation


def _run_trial(
    scene: Path,
    session: ort.InferenceSession,
    metadata: dict[str, object],
    duration_s: float,
    policy_rate_hz: float,
    command: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    # Reproduce the policy's training simulator rather than the standalone
    # scene defaults.  Holosoma trained this checkpoint at 200 Hz with robot
    # self-collision disabled and the terrain in a separate collision class.
    model.opt.timestep = 1.0 / 200.0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "floor":
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 1
        elif model.geom_contype[geom_id] != 0 and model.geom_conaffinity[geom_id] != 0:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 2
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos_addresses, qvel_addresses, actuator_ids = _model_addresses(model)
    default_pose = data.qpos[qpos_addresses].copy()
    kp = np.asarray(metadata["kp"], dtype=np.float64)
    kd = np.asarray(metadata["kd"], dtype=np.float64)
    action_scale = np.asarray(metadata["action_scale"], dtype=np.float64)
    if action_scale.ndim == 0:
        action_scale = np.full(19, float(action_scale))
    for label, value in (("kp", kp), ("kd", kd), ("action_scale", action_scale)):
        if value.shape != (19,):
            raise RuntimeError(f"Policy {label} has shape {value.shape}, expected (19,).")

    policy_steps = max(1, int(round(1.0 / (policy_rate_hz * model.opt.timestep))))
    last_action = np.zeros(19, dtype=np.float64)
    target = default_pose.copy()
    standing = np.linalg.norm(command[:2]) < 0.01 and abs(float(command[2])) < 0.01
    phase = np.asarray(
        [math.pi, math.pi] if standing else [0.0, -math.pi],
        dtype=np.float64,
    )
    phase_increment = 2.0 * math.pi / (1.15 * policy_rate_hz)
    limits = model.actuator_ctrlrange[actuator_ids]
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    traces: dict[str, list[float]] = {
        "time": [],
        "base_x": [],
        "base_z": [],
        "tilt_deg": [],
        "action_rms": [],
        "torque_rms": [],
    }
    fall_time: float | None = None
    max_action = 0.0
    max_torque = 0.0
    steps = int(math.ceil(duration_s / model.opt.timestep))
    policy_update_index = 0
    for step in range(steps):
        if step % policy_steps == 0:
            if standing:
                phase[:] = math.pi
            elif policy_update_index > 0:
                phase = np.fmod(phase + phase_increment + math.pi, 2.0 * math.pi) - math.pi
            observation = _observation(
                data,
                qpos_addresses,
                qvel_addresses,
                default_pose,
                last_action,
                command,
                phase,
            )
            action = session.run([output_name], {input_name: observation})[0]
            last_action = np.asarray(action, dtype=np.float64).reshape(19)
            last_action = np.clip(last_action, -100.0, 100.0)
            target = default_pose + action_scale * last_action
            policy_update_index += 1

        q = data.qpos[qpos_addresses]
        dq = data.qvel[qvel_addresses]
        torque = kp * (target - q) - kd * dq
        torque = np.clip(torque, limits[:, 0], limits[:, 1])
        data.ctrl[actuator_ids] = torque
        mujoco.mj_step(model, data)

        if not all(
            np.all(np.isfinite(array))
            for array in (data.qpos, data.qvel, data.qacc, data.ctrl, last_action)
        ):
            raise RuntimeError(f"Policy rollout became non-finite for {scene.name}.")
        current_tilt = _tilt_deg(data)
        current_z = float(data.qpos[2])
        if fall_time is None and (current_z < 0.55 or current_tilt > 60.0):
            fall_time = float(data.time)
        traces["time"].append(float(data.time))
        traces["base_x"].append(float(data.qpos[0]))
        traces["base_z"].append(current_z)
        traces["tilt_deg"].append(current_tilt)
        traces["action_rms"].append(float(np.sqrt(np.mean(last_action**2))))
        traces["torque_rms"].append(float(np.sqrt(np.mean(torque**2))))
        max_action = max(max_action, float(np.max(np.abs(last_action))))
        max_torque = max(max_torque, float(np.max(np.abs(torque))))

    array_traces = {key: np.asarray(value) for key, value in traces.items()}
    warnings = _warning_counts(data)
    completed_upright = fall_time is None
    metrics = {
        "duration_s": float(data.time),
        "policy_rate_hz": policy_rate_hz,
        "physics_rate_hz": float(1.0 / model.opt.timestep),
        "command_m_s_rad_s": command.tolist(),
        "all_finite": True,
        "warnings": warnings,
        "fall_time_s": fall_time,
        "completed_upright": completed_upright,
        "base_x_displacement_m": float(data.qpos[0] - array_traces["base_x"][0]),
        "base_height_min_m": float(np.min(array_traces["base_z"])),
        "base_height_final_m": float(array_traces["base_z"][-1]),
        "maximum_tilt_deg": float(np.max(array_traces["tilt_deg"])),
        "maximum_abs_action": max_action,
        "maximum_abs_torque_nm": max_torque,
        "passed": completed_upright and not warnings,
    }
    return metrics, array_traces


def _plot(
    all_traces: dict[str, dict[str, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for model_name, trials in all_traces.items():
        stand = trials["stand"]
        forward = trials["forward"]
        axes[0, 0].plot(stand["time"], stand["base_z"], label=model_name)
        axes[0, 1].plot(stand["time"], stand["tilt_deg"], label=model_name)
        axes[1, 0].plot(forward["time"], forward["base_x"], label=model_name)
        axes[1, 1].plot(forward["time"], forward["tilt_deg"], label=model_name)
    axes[0, 0].set(title="Standing policy: base height", ylabel="Base z [m]")
    axes[0, 1].set(title="Standing policy: torso tilt", ylabel="Tilt [deg]")
    axes[1, 0].set(title="Forward policy: displacement", ylabel="Base x [m]")
    axes[1, 1].set(title="Forward policy: torso tilt", ylabel="Tilt [deg]")
    for axis in axes.flat:
        axis.set_xlabel("Time [s]")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("H1 ONNX policy on nominal and calibrated MuJoCo models")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    policy = args.policy.expanduser().resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    metadata = _metadata(session)
    if metadata["dof_names"] != list(config.H1_JOINT_NAMES):
        raise RuntimeError("Policy joint order does not match the calibrated model.")
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if input_shape != [1, 70] or output_shape != [1, 19]:
        raise RuntimeError(f"Unexpected ONNX shapes: input={input_shape}, output={output_shape}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    all_traces: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    trial_commands = {
        "stand": np.asarray([0.0, 0.0, 0.0]),
        "forward": np.asarray([args.forward_speed_m_s, 0.0, 0.0]),
    }
    for model_name, (model_path, scene_path) in MODEL_SPECS.items():
        trial_results: dict[str, object] = {}
        trial_traces: dict[str, dict[str, np.ndarray]] = {}
        for trial_name, command in trial_commands.items():
            metrics, traces = _run_trial(
                scene_path,
                session,
                metadata,
                args.duration_s,
                args.policy_rate_hz,
                command,
            )
            trial_results[trial_name] = metrics
            trial_traces[trial_name] = traces
        results[model_name] = {
            "model": str(model_path.resolve()),
            "scene": str(scene_path.resolve()),
            "trials": trial_results,
            "passed": all(bool(item["passed"]) for item in trial_results.values()),
        }
        all_traces[model_name] = trial_traces

    overall_passed = all(bool(item["passed"]) for item in results.values())
    report = {
        "policy": str(policy),
        "policy_iteration": metadata["iteration"],
        "onnx_input_shape": input_shape,
        "onnx_output_shape": output_shape,
        "joint_order_match": True,
        "results": results,
        "overall_passed": overall_passed,
        "scope_limit": (
            "Diagnostic raw-ONNX rollout only. It does not restore Holosoma "
            "FastSAC observation normalization or action-boundary transforms, "
            "so it is not an official FastSAC policy-regression result. DDS "
            "transport is not involved."
        ),
    }
    report_path = output_dir / "policy_comparison_summary.json"
    plot_path = output_dir / "policy_comparison.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(all_traces, plot_path)
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved plot:   {plot_path}")
    if args.strict and not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
