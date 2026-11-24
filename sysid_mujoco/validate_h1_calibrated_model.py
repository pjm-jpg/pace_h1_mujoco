"""Validate a floating-base H1 model produced from identified joint parameters.

This is an offline structural and numerical smoke test.  It does not establish
that the calibrated model is ready for deployment on hardware or that its
contact parameters are correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nominal-model",
        type=Path,
        default=REPO_ROOT / "robot_model" / "h1" / "h1.xml",
    )
    parser.add_argument(
        "--calibrated-model",
        type=Path,
        default=REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml",
    )
    parser.add_argument(
        "--calibrated-scene",
        type=Path,
        default=(
            REPO_ROOT
            / "robot_model"
            / "h1"
            / "scene_flat_calibrated_lower_body.xml"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "robot_model"
            / "h1"
            / "h1_calibrated_lower_body_params.json"
        ),
    )
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "real_lower_body_combined_v1"
            / "floating_model_validation.json"
        ),
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    return args


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


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
        "has_free_joint": bool(np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
    }


def _joint_dynamics(model: mujoco.MjModel, joint_name: str) -> dict[str, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint {joint_name!r} is missing from the model.")
    dof_id = int(model.jnt_dofadr[joint_id])
    return {
        "armature": float(model.dof_armature[dof_id]),
        "damping": float(model.dof_damping[dof_id]),
        "frictionloss": float(model.dof_frictionloss[dof_id]),
    }


def _verify_parameters(
    nominal: mujoco.MjModel,
    calibrated: mujoco.MjModel,
    parameters: dict[str, dict[str, float]],
) -> dict[str, object]:
    calibrated_values: dict[str, dict[str, float]] = {}
    maximum_error = 0.0
    for joint_name, expected in parameters.items():
        actual = _joint_dynamics(calibrated, joint_name)
        calibrated_values[joint_name] = actual
        for attribute, expected_value in expected.items():
            maximum_error = max(
                maximum_error,
                abs(actual[attribute] - float(expected_value)),
            )

    untouched = [name for name in config.H1_JOINT_NAMES if name not in parameters]
    maximum_untouched_change = 0.0
    for joint_name in untouched:
        nominal_values = _joint_dynamics(nominal, joint_name)
        calibrated_joint = _joint_dynamics(calibrated, joint_name)
        for attribute in nominal_values:
            maximum_untouched_change = max(
                maximum_untouched_change,
                abs(nominal_values[attribute] - calibrated_joint[attribute]),
            )

    if maximum_error > 1e-9:
        raise RuntimeError(f"Calibrated parameter mismatch: {maximum_error:.3e}")
    if maximum_untouched_change > 1e-12:
        raise RuntimeError(
            "A non-identified joint changed by "
            f"{maximum_untouched_change:.3e}."
        )
    return {
        "identified_joint_count": len(parameters),
        "identified_values": calibrated_values,
        "maximum_parameter_error": maximum_error,
        "untouched_joint_count": len(untouched),
        "maximum_untouched_joint_change": maximum_untouched_change,
    }


def _actuated_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses = []
    dof_addresses = []
    for joint_name in config.H1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} is missing from the scene.")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _short_simulation(scene: Path, duration_s: float) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos_addresses, dof_addresses = _actuated_joint_addresses(model)
    desired = data.qpos[qpos_addresses].copy()
    kp = np.asarray(config.Kp, dtype=np.float64)
    kd = np.asarray(config.Kd, dtype=np.float64)
    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    actuator_ids = np.asarray([actuator_names.index(name) for name in config.H1_JOINT_NAMES])

    base_z = []
    contacts = []
    maximum_joint_error = 0.0
    maximum_control = 0.0
    steps = int(np.ceil(duration_s / model.opt.timestep))
    for _ in range(steps):
        q = data.qpos[qpos_addresses]
        dq = data.qvel[dof_addresses]
        torque = kp * (desired - q) - kd * dq
        ranges = model.actuator_ctrlrange[actuator_ids]
        torque = np.clip(torque, ranges[:, 0], ranges[:, 1])
        data.ctrl[actuator_ids] = torque
        mujoco.mj_step(model, data)
        if not (
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        ):
            raise RuntimeError("The calibrated floating-base simulation became non-finite.")
        base_z.append(float(data.qpos[2]))
        contacts.append(int(data.ncon))
        maximum_joint_error = max(
            maximum_joint_error,
            float(np.max(np.abs(desired - data.qpos[qpos_addresses]))),
        )
        maximum_control = max(maximum_control, float(np.max(np.abs(torque))))

    warnings = {
        str(index): int(warning.number)
        for index, warning in enumerate(data.warning)
        if warning.number
    }
    return {
        "duration_s": float(steps * model.opt.timestep),
        "steps": steps,
        "timestep_s": float(model.opt.timestep),
        "all_finite": True,
        "base_height_min_m": min(base_z),
        "base_height_max_m": max(base_z),
        "contact_count_max": max(contacts),
        "maximum_joint_error_rad": maximum_joint_error,
        "maximum_abs_control_nm": maximum_control,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    nominal_path = args.nominal_model.expanduser().resolve()
    calibrated_path = args.calibrated_model.expanduser().resolve()
    scene_path = args.calibrated_scene.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (nominal_path, calibrated_path, scene_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nominal = mujoco.MjModel.from_xml_path(str(nominal_path))
    calibrated = mujoco.MjModel.from_xml_path(str(calibrated_path))
    nominal_topology = _topology(nominal)
    calibrated_topology = _topology(calibrated)
    if nominal_topology != calibrated_topology:
        raise RuntimeError("Nominal and calibrated model topology differ.")
    if not calibrated_topology["has_free_joint"]:
        raise RuntimeError("The calibrated training candidate has no free joint.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameter_check = _verify_parameters(
        nominal,
        calibrated,
        manifest["parameters"],
    )
    simulation = _short_simulation(scene_path, args.duration_s)

    report = {
        "nominal_model": str(nominal_path),
        "calibrated_model": str(calibrated_path),
        "calibrated_scene": str(scene_path),
        "parameter_manifest": str(manifest_path),
        "topology_equal": True,
        "topology": calibrated_topology,
        "parameter_check": parameter_check,
        "short_simulation": simulation,
        "scope": (
            "Offline structural/numerical smoke test only; contact dynamics and "
            "policy safety are not validated."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved validation report to {output}")


if __name__ == "__main__":
    main()
