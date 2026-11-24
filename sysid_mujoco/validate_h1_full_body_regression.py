"""Run deterministic floating-base regression tests for three H1 models.

The comparison covers the nominal model, the lower-body calibrated model and
the full-body calibrated model.  It is a model-integration gate, not a claim
that open-loop joint PD can balance a free-standing humanoid.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


MODEL_SPECS = {
    "nominal": (
        REPO_ROOT / "robot_model" / "h1" / "h1.xml",
        REPO_ROOT / "robot_model" / "h1" / "scene_flat.xml",
    ),
    "lower_body_calibrated": (
        REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml",
        REPO_ROOT
        / "robot_model"
        / "h1"
        / "scene_flat_calibrated_lower_body.xml",
    ),
    "full_body_calibrated": (
        REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_full_body.xml",
        REPO_ROOT
        / "robot_model"
        / "h1"
        / "scene_flat_calibrated_full_body.xml",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airborne-duration-s", type=float, default=4.0)
    parser.add_argument("--drop-duration-s", type=float, default=3.0)
    parser.add_argument("--drop-height-m", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "full_body_regression_v1"
        ),
    )
    args = parser.parse_args()
    if args.airborne_duration_s <= 0 or args.drop_duration_s <= 0:
        parser.error("Durations must be positive.")
    if args.drop_height_m < 0:
        parser.error("--drop-height-m cannot be negative.")
    return args


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def _addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_ids: list[int] = []
    for name in config.H1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"Missing H1 joint or actuator {name!r}.")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
    return (
        np.asarray(qpos_addresses, dtype=np.int32),
        np.asarray(dof_addresses, dtype=np.int32),
        np.asarray(actuator_ids, dtype=np.int32),
    )


def _topology(model: mujoco.MjModel) -> dict[str, object]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsensor": int(model.nsensor),
        "joint_names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "actuator_names": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    }


def _dynamics_fingerprint(model: mujoco.MjModel) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for name in config.H1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_id = int(model.jnt_dofadr[joint_id])
        values[name] = {
            "armature": float(model.dof_armature[dof_id]),
            "damping": float(model.dof_damping[dof_id]),
            "frictionloss": float(model.dof_frictionloss[dof_id]),
        }
    return values


def _warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        str(index): int(warning.number)
        for index, warning in enumerate(data.warning)
        if warning.number
    }


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def _apply_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    desired: np.ndarray,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    actuator_ids: np.ndarray,
) -> np.ndarray:
    q = data.qpos[qpos_addresses]
    dq = data.qvel[dof_addresses]
    torque = np.asarray(config.Kp) * (desired - q) - np.asarray(config.Kd) * dq
    limits = model.actuator_ctrlrange[actuator_ids]
    torque = np.clip(torque, limits[:, 0], limits[:, 1])
    data.ctrl[actuator_ids] = torque
    return torque


def _assert_finite(data: mujoco.MjData, label: str) -> None:
    arrays = (data.qpos, data.qvel, data.qacc, data.ctrl)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError(f"{label} became non-finite.")


def _base_tilt_rad(data: mujoco.MjData) -> float:
    # MuJoCo free-joint quaternion order is w, x, y, z.  The world-z
    # component of the rotated body z-axis is 1 - 2(x^2 + y^2).
    quat = data.qpos[3:7]
    up_z = float(np.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))
    return float(math.acos(up_z))


def _total_normal_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    wrench = np.zeros(6, dtype=np.float64)
    total = 0.0
    for contact_index in range(data.ncon):
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        total += abs(float(wrench[0]))
    return total


def _airborne_trial(scene: Path, duration_s: float) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    _reset_home(model, data)
    model.opt.gravity[:] = 0.0
    data.qpos[2] += 0.5
    mujoco.mj_forward(model, data)

    qpos_addresses, dof_addresses, actuator_ids = _addresses(model)
    center = data.qpos[qpos_addresses].copy()
    amplitudes = np.asarray(
        [0.02] * 10 + [0.015] + [0.025] * 8,
        dtype=np.float64,
    )
    frequencies = np.asarray([0.5, 1.0, 2.0] * 7, dtype=np.float64)[:19]
    phases = np.linspace(0.0, math.pi, 19, endpoint=False)

    times: list[float] = []
    rms_samples: list[float] = []
    base_position: list[np.ndarray] = []
    max_control = 0.0
    steps = int(math.ceil(duration_s / model.opt.timestep))
    for step in range(steps):
        t = step * model.opt.timestep
        envelope = min(1.0, t / 0.5, max(0.0, (duration_s - t) / 0.5))
        desired = center + envelope * amplitudes * np.sin(
            2.0 * math.pi * frequencies * t + phases
        )
        torque = _apply_pd(
            model,
            data,
            desired,
            qpos_addresses,
            dof_addresses,
            actuator_ids,
        )
        mujoco.mj_step(model, data)
        _assert_finite(data, f"Airborne trial for {scene.name}")
        error = desired - data.qpos[qpos_addresses]
        times.append(float(data.time))
        rms_samples.append(float(np.sqrt(np.mean(np.square(error)))))
        base_position.append(data.qpos[:3].copy())
        max_control = max(max_control, float(np.max(np.abs(torque))))

    base_xyz = np.asarray(base_position)
    metrics = {
        "duration_s": float(data.time),
        "steps": steps,
        "all_finite": True,
        "warnings": _warning_counts(data),
        "tracking_rmse_rad": float(np.sqrt(np.mean(np.square(rms_samples)))),
        "tracking_peak_rms_error_rad": float(np.max(rms_samples)),
        "base_translation_from_reaction_m": float(
            np.linalg.norm(base_xyz[-1] - base_xyz[0])
        ),
        "maximum_abs_control_nm": max_control,
        "passed": not _warning_counts(data) and float(np.max(rms_samples)) < 0.10,
    }
    traces = {
        "time": np.asarray(times),
        "tracking_rms": np.asarray(rms_samples),
        "base_z": base_xyz[:, 2],
    }
    return metrics, traces


def _drop_trial(
    scene: Path,
    duration_s: float,
    drop_height_m: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    _reset_home(model, data)
    data.qpos[2] += drop_height_m
    mujoco.mj_forward(model, data)

    qpos_addresses, dof_addresses, actuator_ids = _addresses(model)
    desired = data.qpos[qpos_addresses].copy()
    times: list[float] = []
    base_z: list[float] = []
    tilt: list[float] = []
    contacts: list[int] = []
    normal_force: list[float] = []
    first_contact_s: float | None = None
    fall_time_s: float | None = None
    max_control = 0.0
    steps = int(math.ceil(duration_s / model.opt.timestep))
    for _ in range(steps):
        torque = _apply_pd(
            model,
            data,
            desired,
            qpos_addresses,
            dof_addresses,
            actuator_ids,
        )
        mujoco.mj_step(model, data)
        _assert_finite(data, f"Drop trial for {scene.name}")
        current_tilt = _base_tilt_rad(data)
        current_base_z = float(data.qpos[2])
        if data.ncon and first_contact_s is None:
            first_contact_s = float(data.time)
        if fall_time_s is None and (current_base_z < 0.55 or current_tilt > math.radians(60)):
            fall_time_s = float(data.time)
        times.append(float(data.time))
        base_z.append(current_base_z)
        tilt.append(current_tilt)
        contacts.append(int(data.ncon))
        normal_force.append(_total_normal_contact_force(model, data))
        max_control = max(max_control, float(np.max(np.abs(torque))))

    warnings = _warning_counts(data)
    metrics = {
        "duration_s": float(data.time),
        "steps": steps,
        "drop_height_m": drop_height_m,
        "all_finite": True,
        "warnings": warnings,
        "first_contact_s": first_contact_s,
        "base_height_min_m": float(np.min(base_z)),
        "base_height_final_m": float(base_z[-1]),
        "maximum_tilt_deg": float(np.degrees(np.max(tilt))),
        "final_tilt_deg": float(np.degrees(tilt[-1])),
        "maximum_contact_count": int(np.max(contacts)),
        "peak_total_normal_contact_force_n": float(np.max(normal_force)),
        "maximum_abs_control_nm": max_control,
        "open_loop_pd_fall_time_s": fall_time_s,
        "passed": first_contact_s is not None and not warnings,
        "interpretation": (
            "Fall status is descriptive only; joint PD without a balance policy "
            "is not expected to stabilize a floating-base humanoid."
        ),
    }
    traces = {
        "time": np.asarray(times),
        "base_z": np.asarray(base_z),
        "tilt_deg": np.degrees(np.asarray(tilt)),
        "contacts": np.asarray(contacts),
        "normal_force": np.asarray(normal_force),
    }
    return metrics, traces


def _plot(traces: dict[str, dict[str, dict[str, np.ndarray]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, trials in traces.items():
        airborne = trials["airborne"]
        drop = trials["drop"]
        axes[0, 0].plot(airborne["time"], airborne["tracking_rms"], label=label)
        axes[0, 1].plot(drop["time"], drop["base_z"], label=label)
        axes[1, 0].plot(drop["time"], drop["tilt_deg"], label=label)
        axes[1, 1].plot(drop["time"], drop["normal_force"], label=label)
    axes[0, 0].set(title="Airborne full-body tracking", ylabel="RMS error [rad]")
    axes[0, 1].set(title="Drop/contact base height", ylabel="Base z [m]")
    axes[1, 0].set(title="Drop/contact torso tilt", ylabel="Tilt [deg]", xlabel="Time [s]")
    axes[1, 1].set(
        title="Drop/contact normal force",
        ylabel="Total normal force [N]",
        xlabel="Time [s]",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("H1 nominal vs calibrated floating-base regression")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, mujoco.MjModel] = {}
    for label, (model_path, scene_path) in MODEL_SPECS.items():
        if not model_path.is_file() or not scene_path.is_file():
            raise FileNotFoundError(f"Missing model assets for {label}.")
        models[label] = mujoco.MjModel.from_xml_path(str(model_path))

    topologies = {label: _topology(model) for label, model in models.items()}
    reference_topology = topologies["nominal"]
    topology_equal = all(value == reference_topology for value in topologies.values())
    if not topology_equal:
        raise RuntimeError("The three H1 model topologies are not identical.")
    if reference_topology["actuator_names"] != list(config.H1_JOINT_NAMES):
        raise RuntimeError("H1 actuator order does not match config.H1_JOINT_NAMES.")

    dynamics = {label: _dynamics_fingerprint(model) for label, model in models.items()}
    changed_joint_counts: dict[str, int] = {}
    nominal_dynamics = dynamics["nominal"]
    for label, values in dynamics.items():
        changed_joint_counts[label] = sum(
            any(abs(values[name][key] - nominal_dynamics[name][key]) > 1e-12 for key in values[name])
            for name in config.H1_JOINT_NAMES
        )

    results: dict[str, object] = {}
    traces: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for label, (model_path, scene_path) in MODEL_SPECS.items():
        airborne_metrics, airborne_traces = _airborne_trial(
            scene_path,
            args.airborne_duration_s,
        )
        drop_metrics, drop_traces = _drop_trial(
            scene_path,
            args.drop_duration_s,
            args.drop_height_m,
        )
        results[label] = {
            "model": str(model_path.resolve()),
            "scene": str(scene_path.resolve()),
            "changed_joint_count_from_nominal": changed_joint_counts[label],
            "airborne_full_body_tracking": airborne_metrics,
            "ground_drop_contact": drop_metrics,
            "passed": bool(airborne_metrics["passed"] and drop_metrics["passed"]),
        }
        traces[label] = {"airborne": airborne_traces, "drop": drop_traces}

    overall_passed = topology_equal and all(bool(value["passed"]) for value in results.values())
    report = {
        "purpose": "training-model integration regression before policy rollout",
        "topology_equal": topology_equal,
        "topology": reference_topology,
        "changed_joint_counts_from_nominal": changed_joint_counts,
        "models": results,
        "overall_passed": overall_passed,
        "scope_limit": (
            "This validates model structure, numerical stability, joint-control "
            "interfaces and computable contact response. It does not validate "
            "contact identification or closed-loop locomotion policy performance."
        ),
    }
    report_path = output_dir / "regression_summary.json"
    plot_path = output_dir / "regression_comparison.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(traces, plot_path)
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved plot:   {plot_path}")
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
