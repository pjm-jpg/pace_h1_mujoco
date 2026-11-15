"""Generate deterministic H1 knee datasets with known hidden dynamics.

The fitting recording uses a smooth multi-sine target.  The validation
recording uses an unseen sequence of position steps.  Both are generated from
the same fixed-base MuJoCo model and use the real Holosoma runtime PD gains.
The hidden joint parameters are written to provenance files but are never read
by the fitting entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from sysid_mujoco.common import (  # noqa: E402
    build_actuator_gain_map,
    build_fixed_base_model_xml,
    get_actuated_joint_and_actuator_names,
)


TRUTH_ATTRIBUTES = ("armature", "frictionloss", "damping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "h1",
    )
    parser.add_argument(
        "--joint",
        choices=config.H1_JOINT_NAMES,
        default="left_knee_joint",
    )
    parser.add_argument("--fit-duration", type=float, default=6.0)
    parser.add_argument("--validation-duration", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=0.22)
    parser.add_argument("--truth-armature", type=float, default=0.22)
    parser.add_argument("--truth-frictionloss", type=float, default=0.35)
    parser.add_argument("--truth-damping", type=float, default=2.2)
    parser.add_argument(
        "--measurement-noise-std",
        type=float,
        default=0.0,
        help="Optional zero-mean position noise in radians (default: none).",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.fit_duration <= 1.0 or args.validation_duration <= 1.0:
        parser.error("durations must be greater than one second")
    if not 0.0 < args.amplitude <= 0.25:
        parser.error("--amplitude must be in (0, 0.25] rad for the default zero pose")
    truth = (args.truth_armature, args.truth_frictionloss, args.truth_damping)
    if not np.all(np.isfinite(truth)) or np.any(np.asarray(truth) < 0.0):
        parser.error("hidden dynamics must be finite and nonnegative")
    if args.measurement_noise_std < 0.0 or not np.isfinite(args.measurement_noise_std):
        parser.error("--measurement-noise-std must be finite and nonnegative")
    return args


def _write_truth_xml(
    nominal_xml: Path,
    output_xml: Path,
    joint_name: str,
    truth: dict[str, float],
) -> None:
    tree = ET.parse(nominal_xml)
    root = tree.getroot()
    for joint in root.iter("joint"):
        if joint.get("name") == joint_name:
            for attribute, value in truth.items():
                joint.set(attribute, f"{value:.12g}")
            output_xml.parent.mkdir(parents=True, exist_ok=True)
            tree.write(output_xml, encoding="utf-8", xml_declaration=True)
            return
    raise ValueError(f"Joint {joint_name!r} was not found in {nominal_xml}")


def _smooth_gate(active_time: np.ndarray, ramp_duration: float = 0.4) -> np.ndarray:
    clipped = np.clip(active_time / ramp_duration, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * clipped)


def fit_reference(times: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth multi-frequency position reference and analytic velocity."""
    active = np.maximum(times - 0.5, 0.0)
    gate = _smooth_gate(active)
    frequencies = np.asarray([0.35, 0.85, 1.65])
    weights = np.asarray([0.50, 0.30, 0.20])
    phase = 2.0 * np.pi * active[:, None] * frequencies[None, :]
    raw_position = np.sum(weights[None, :] * np.sin(phase), axis=1)
    raw_velocity = np.sum(
        weights[None, :] * (2.0 * np.pi * frequencies)[None, :] * np.cos(phase),
        axis=1,
    )
    # Differentiate the half-cosine ramp inside its transition interval.
    gate_velocity = np.zeros_like(times)
    ramp_mask = (active > 0.0) & (active < 0.4)
    gate_velocity[ramp_mask] = (
        0.5 * np.pi / 0.4 * np.sin(np.pi * active[ramp_mask] / 0.4)
    )
    position = amplitude * gate * raw_position
    velocity = amplitude * (gate_velocity * raw_position + gate * raw_velocity)
    position[times <= 0.5] = 0.0
    velocity[times <= 0.5] = 0.0
    return position, velocity


def validation_reference(
    times: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unseen held steps; velocity commands remain zero like DDS position steps."""
    position = np.zeros_like(times)
    position[(times >= 1.0) & (times < 2.0)] = 0.75 * amplitude
    position[(times >= 2.0) & (times < 3.2)] = -0.45 * amplitude
    position[(times >= 3.2) & (times < 4.2)] = 0.35 * amplitude
    return position, np.zeros_like(times)


def _simulate_dataset(
    model: mujoco.MjModel,
    joint_names: list[str],
    joint_name: str,
    duration: float,
    reference_fn,
    amplitude: float,
    kp: np.ndarray,
    kd: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
) -> dict[str, object]:
    dt = 1.0 / config.frequency_collection
    num_samples = int(round(duration / dt)) + 1
    times = np.arange(num_samples, dtype=np.float64) * dt
    target_position, target_velocity = reference_fn(times, amplitude)

    desired_qpos = np.zeros((num_samples, model.nu), dtype=np.float64)
    desired_qvel = np.zeros_like(desired_qpos)
    target_index = joint_names.index(joint_name)
    desired_qpos[:, target_index] = target_position
    desired_qvel[:, target_index] = target_velocity

    data = mujoco.MjData(model)
    measured_qpos = np.zeros((num_samples, model.nq), dtype=np.float64)
    measured_qvel = np.zeros((num_samples, model.nv), dtype=np.float64)
    measured_qpos[0] = data.qpos
    measured_qvel[0] = data.qvel
    for sample in range(num_samples - 1):
        data.ctrl[:] = desired_qpos[sample]
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError(f"Non-finite synthetic state at sample {sample}")
        measured_qpos[sample + 1] = data.qpos
        measured_qvel[sample + 1] = data.qvel

    if noise_std > 0.0:
        measured_qpos += rng.normal(0.0, noise_std, measured_qpos.shape)

    return {
        "time": torch.from_numpy(times),
        "dof_pos": torch.from_numpy(measured_qpos),
        "dof_vel": torch.from_numpy(measured_qvel),
        "des_dof_pos": torch.from_numpy(desired_qpos),
        "des_dof_vel": torch.from_numpy(desired_qvel),
        "des_dof_torque": torch.zeros_like(torch.from_numpy(desired_qpos)),
        "kp": torch.from_numpy(kp.copy()),
        "kd": torch.from_numpy(kd.copy()),
        "joint_names": list(joint_names),
        "frequency_hz": float(config.frequency_collection),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gains = build_actuator_gain_map(
        list(config.H1_JOINT_NAMES),
        np.asarray(config.Kp, dtype=np.float64),
        np.asarray(config.Kd, dtype=np.float64),
    )
    nominal_xml = build_fixed_base_model_xml("h1", actuator_gains=gains)
    truth = {
        "armature": float(args.truth_armature),
        "frictionloss": float(args.truth_frictionloss),
        "damping": float(args.truth_damping),
    }
    truth_xml = args.output_dir / "synthetic_left_knee_truth.xml"
    _write_truth_xml(nominal_xml, truth_xml, args.joint, truth)
    model = mujoco.MjModel.from_xml_path(str(truth_xml))
    model.opt.timestep = 1.0 / config.frequency_collection
    joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, model)
    if joint_names != list(config.H1_JOINT_NAMES):
        raise ValueError(
            "H1 actuator order differs from config.H1_JOINT_NAMES: "
            f"{joint_names}"
        )

    rng = np.random.default_rng(args.seed)
    fit = _simulate_dataset(
        model, joint_names, args.joint, args.fit_duration, fit_reference,
        args.amplitude, np.asarray(config.Kp), np.asarray(config.Kd), rng,
        args.measurement_noise_std,
    )
    validation = _simulate_dataset(
        model, joint_names, args.joint, args.validation_duration,
        validation_reference, args.amplitude, np.asarray(config.Kp),
        np.asarray(config.Kd), rng, args.measurement_noise_std,
    )
    fit_path = args.output_dir / "synthetic_left_knee_fit.pt"
    validation_path = args.output_dir / "synthetic_left_knee_validation.pt"
    torch.save(fit, fit_path)
    torch.save(validation, validation_path)

    nominal_joint = mujoco.MjModel.from_xml_path(str(nominal_xml)).joint(args.joint)
    provenance = {
        "robot": "h1",
        "joint": args.joint,
        "frequency_hz": config.frequency_collection,
        "fit_dataset": str(fit_path.resolve()),
        "validation_dataset": str(validation_path.resolve()),
        "truth_model_xml": str(truth_xml.resolve()),
        "nominal_parameters": {
            attribute: float(np.asarray(getattr(nominal_joint, attribute)).reshape(-1)[0])
            for attribute in TRUTH_ATTRIBUTES
        },
        "hidden_truth_parameters": truth,
        "fit_motion": "smooth multi-sine (0.35, 0.85, 1.65 Hz)",
        "validation_motion": "unseen held steps",
        "amplitude_rad": args.amplitude,
        "measurement_noise_std_rad": args.measurement_noise_std,
        "seed": args.seed,
    }
    provenance_path = args.output_dir / "synthetic_left_knee_truth.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Fit dataset:        {fit_path} ({len(fit['time'])} samples)")
    print(f"Validation dataset: {validation_path} ({len(validation['time'])} samples)")
    print(f"Truth model:        {truth_xml}")
    print(f"Provenance:         {provenance_path}")
    print(f"Hidden parameters:  {truth}")


if __name__ == "__main__":
    main()
