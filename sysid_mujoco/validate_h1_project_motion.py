"""Validate a coordinated project motion before suspended H1 collection.

The source motion is a Holosoma retargeted trajectory containing a floating
base followed by the 19 H1 joints.  Only the joint trajectory is replayed: the
hardware experiment is performed with the robot suspended from a fixed gantry.

This program is simulation-only.  It never imports the Unitree interface and
cannot publish DDS commands.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from sysid_mujoco.preview_h1_real_experiment import (
    H1_SIMULATION_START_TARGETS,
    _apply_joint_targets,
    _exclude_adjacent_body_contacts,
    _joint_ids,
    _joint_vector,
    _self_contact_names,
    _set_joint_state,
    build_preview_model,
)


DEFAULT_MOTION = Path(
    "/home/pjm/Desktop/holosoma/src/holosoma/holosoma/data/motions/"
    "walk1_subject1_mj_fps50.npz"
)
DEFAULT_MODEL = REPO_ROOT / "robot_model/h1/h1_calibrated_full_body.xml"

SUSPENDED_REFERENCE = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.12,
    "left_hip_pitch_joint": -0.30,
    "left_knee_joint": 0.30,
    "left_ankle_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": -0.12,
    "right_hip_pitch_joint": -0.30,
    "right_knee_joint": 0.30,
    "right_ankle_joint": 0.0,
    "torso_joint": 0.0,
    "left_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.32,
    "right_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.32,
}
H1_ARM_JOINTS = tuple(
    name for name in config.H1_JOINT_NAMES
    if "shoulder" in name or "elbow" in name
)


def _smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _staged_waypoints(
    initial_q: np.ndarray,
    target_q: np.ndarray,
    arm_clearance_pitch_rad: float = 0.80,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    """Match the real collector's collision-clearance route.

    The H1 can finish a lower-body run with both elbows folded beside the
    pelvis and the shoulder-yaw joints near their limits.  Raising shoulder
    pitch directly from that pose sweeps the elbows through the hips.  Move
    the arms laterally first, unfold yaw/elbow, and only then raise them.
    """
    arm_names = {
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
    }
    outward = initial_q.copy()
    outward[config.H1_JOINT_NAMES.index("left_shoulder_roll_joint")] = 1.0
    outward[config.H1_JOINT_NAMES.index("right_shoulder_roll_joint")] = -1.0

    unfolded = outward.copy()
    for side in ("left", "right"):
        unfolded[
            config.H1_JOINT_NAMES.index(f"{side}_shoulder_yaw_joint")
        ] = 0.0
        unfolded[config.H1_JOINT_NAMES.index(f"{side}_elbow_joint")] = 0.50

    lifted = unfolded.copy()
    for side in ("left", "right"):
        lifted[config.H1_JOINT_NAMES.index(f"{side}_shoulder_pitch_joint")] = (
            arm_clearance_pitch_rad
        )
    arms_ready = lifted.copy()
    for index, name in enumerate(config.H1_JOINT_NAMES):
        if name in arm_names and "shoulder_pitch" not in name:
            arms_ready[index] = target_q[index]
    body_ready = arms_ready.copy()
    for index, name in enumerate(config.H1_JOINT_NAMES):
        if name not in arm_names:
            body_ready[index] = target_q[index]
    return (
        (
            initial_q.copy(), outward, unfolded, lifted, arms_ready,
            body_ready, target_q.copy(),
        ),
        (5.0, 5.0, 4.0, 6.0, 5.0, 5.0),
    )


def _piecewise_smooth_target(
    waypoints: tuple[np.ndarray, ...],
    weights: tuple[float, ...],
    elapsed_s: float,
    duration_s: float,
) -> np.ndarray:
    if elapsed_s <= 0.0:
        return waypoints[0].copy()
    if elapsed_s >= duration_s:
        return waypoints[-1].copy()
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= float(np.sum(normalized))
    boundaries = np.cumsum(normalized) * duration_s
    stage = int(np.searchsorted(boundaries, elapsed_s, side="right"))
    start_s = 0.0 if stage == 0 else float(boundaries[stage - 1])
    blend = _smoothstep5(
        (elapsed_s - start_s) / float(boundaries[stage] - start_s)
    )
    return waypoints[stage] + blend * (waypoints[stage + 1] - waypoints[stage])


def load_motion(path: Path) -> tuple[np.ndarray, float, list[str]]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=True) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        names = [str(value) for value in np.asarray(motion["joint_names"]).tolist()]
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError(f"Invalid motion fps in {path}: {fps}")
    if tuple(names) != tuple(config.H1_JOINT_NAMES):
        raise ValueError("Motion joint order does not match the H1 19-DOF order.")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 7 + len(names):
        raise ValueError(
            f"Expected joint_pos shape (T, {7 + len(names)}), got {joint_pos.shape}."
        )
    q = joint_pos[:, 7:].copy()
    if len(q) < 2 or not np.all(np.isfinite(q)):
        raise ValueError("Motion must contain at least two finite joint samples.")
    return q, fps, names


def scaled_motion(
    q: np.ndarray,
    amplitude_scale: float,
    profile: str = "full-body",
) -> np.ndarray:
    reference = np.asarray(
        [SUSPENDED_REFERENCE[name] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )
    result = reference[None, :] + amplitude_scale * (q - reference[None, :])
    if profile == "lower-body":
        for name in H1_ARM_JOINTS:
            result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif profile == "upper-body":
        for name in config.H1_JOINT_NAMES:
            if name not in H1_ARM_JOINTS:
                result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif profile == "torso-only":
        for name in config.H1_JOINT_NAMES:
            if name != "torso_joint":
                result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif profile != "full-body":
        raise ValueError(f"Unknown motion profile: {profile}")
    return result


def sample_motion(q: np.ndarray, fps: float, source_time_s: float) -> np.ndarray:
    sample = float(np.clip(source_time_s * fps, 0.0, len(q) - 1))
    lower = int(np.floor(sample))
    upper = min(lower + 1, len(q) - 1)
    blend = sample - lower
    return q[lower] + blend * (q[upper] - q[lower])


def command_target(
    *,
    elapsed_s: float,
    initial_q: np.ndarray,
    motion_q: np.ndarray,
    source_fps: float,
    speed_scale: float,
    prepare_s: float,
    settle_s: float,
    post_hold_s: float,
    return_s: float,
    prepare_waypoints: tuple[np.ndarray, ...] | None = None,
    prepare_weights: tuple[float, ...] | None = None,
    return_waypoints: tuple[np.ndarray, ...] | None = None,
    return_weights: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, str]:
    motion_duration_s = (len(motion_q) - 1) / source_fps / speed_scale
    core_start_s = prepare_s + settle_s
    core_end_s = core_start_s + motion_duration_s
    return_start_s = core_end_s + post_hold_s
    if elapsed_s < prepare_s:
        if prepare_waypoints is not None and prepare_weights is not None:
            return _piecewise_smooth_target(
                prepare_waypoints, prepare_weights, elapsed_s, prepare_s
            ), "prepare"
        blend = _smoothstep5(elapsed_s / prepare_s)
        return initial_q + blend * (motion_q[0] - initial_q), "prepare"
    if elapsed_s < core_start_s:
        return motion_q[0].copy(), "settle"
    if elapsed_s < core_end_s:
        source_time_s = (elapsed_s - core_start_s) * speed_scale
        return sample_motion(motion_q, source_fps, source_time_s), "motion"
    if elapsed_s < return_start_s:
        return motion_q[-1].copy(), "post_hold"
    if return_waypoints is not None and return_weights is not None:
        return _piecewise_smooth_target(
            return_waypoints,
            return_weights,
            elapsed_s - return_start_s,
            return_s,
        ), "return"
    blend = _smoothstep5((elapsed_s - return_start_s) / return_s)
    return motion_q[-1] + blend * (initial_q - motion_q[-1]), "return"


def _joint_positions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.asarray(
        [data.qpos[_joint_ids(model, name)[1]] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )


def _joint_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.asarray(
        [data.qvel[_joint_ids(model, name)[2]] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )


def _joint_limit_margin(model: mujoco.MjModel, q: np.ndarray) -> float:
    margins = []
    for index, name in enumerate(config.H1_JOINT_NAMES):
        joint_id = _joint_ids(model, name)[0]
        if bool(model.jnt_limited[joint_id]):
            lower, upper = model.jnt_range[joint_id]
            margins.extend((q[index] - lower, upper - q[index]))
    return float(np.min(margins))


def _final_joint_vector_from_npz(path: Path) -> np.ndarray:
    """Use the median of the final second as the next hardware start pose."""
    with np.load(path.expanduser().resolve(), allow_pickle=False) as recording:
        q_real = np.asarray(recording["q_real"], dtype=np.float64)
    if q_real.ndim != 2 or q_real.shape[1] < 20 or len(q_real) == 0:
        raise ValueError(f"Expected q_real with 20 physical motor slots in {path}")
    tail = q_real[-min(500, len(q_real)):]
    result = np.asarray(
        [np.median(tail[:, motor]) for motor in config.H1_JOINT_TO_MOTOR],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite final joint state in {path}")
    return result


def _contact_body_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for contact in data.contact[: data.ncon]:
        names = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = int(model.geom_bodyid[geom_id])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                break
            names.append(name)
        if len(names) == 2 and names[0] != names[1]:
            pairs.add(tuple(sorted(names)))
    return pairs


def _compile_with_contact_exclusions(
    source_xml: Path,
    body_pairs: set[tuple[str, str]],
) -> tuple[mujoco.MjModel, Path]:
    """Exclude contacts already present in the measured hardware start pose.

    Those contacts are conservative-geometry false positives: the real robot
    was stationary in the pose at the end of the preceding guarded run.  They
    otherwise create an impulsive state jump on MuJoCo's first step and make a
    dynamic recovery-path check impossible.  All other collision pairs remain
    enabled and are checked throughout the validation.
    """
    tree = ET.parse(source_xml)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        tuple(sorted((element.get("body1", ""), element.get("body2", ""))))
        for element in contact.findall("exclude")
    }
    for body1, body2 in sorted(body_pairs):
        if (body1, body2) not in existing:
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
    output_xml = source_xml.with_name(
        source_xml.stem + "_measured_start_clearance.xml"
    )
    temporary = output_xml.with_suffix(output_xml.suffix + ".tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(output_xml)
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    model.opt.timestep = 1.0 / config.frequency_collection
    return model, output_xml


def _save_plot(
    path: Path,
    times: np.ndarray,
    desired: np.ndarray,
    measured: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 4, figsize=(18, 15), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, config.H1_JOINT_NAMES, strict=False)):
        axis.plot(times, desired[:, index], "--", linewidth=1.0, label="command")
        axis.plot(times, measured[:, index], linewidth=1.0, label="MuJoCo")
        axis.set_title(name.replace("_joint", ""), fontsize=9)
        axis.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    for axis in axes.flat[len(config.H1_JOINT_NAMES):]:
        axis.axis("off")
    figure.supxlabel("time [s]")
    figure.supylabel("joint position [rad]")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--initial-state-npz", type=Path, default=None,
        help="Use the final one-second median of a real raw.npz as the start pose.",
    )
    parser.add_argument(
        "--profile",
        choices=("full-body", "lower-body", "upper-body", "torso-only"),
        default="full-body",
        help=(
            "lower-body holds both arms fixed; upper-body holds legs and torso "
            "in the verified suspended reference pose."
        ),
    )
    parser.add_argument("--amplitude-scale", type=float, default=0.60)
    parser.add_argument("--speed-scale", type=float, default=0.50)
    parser.add_argument("--command-rate-hz", type=float, default=50.0)
    parser.add_argument("--prepare-duration-s", type=float, default=18.0)
    parser.add_argument("--prepare-settle-s", type=float, default=2.0)
    parser.add_argument("--post-hold-s", type=float, default=2.0)
    parser.add_argument("--return-duration-s", type=float, default=18.0)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument("--max-following-error-rad", type=float, default=0.15)
    parser.add_argument("--min-joint-limit-margin-rad", type=float, default=0.03)
    parser.add_argument("--max-command-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--max-force-utilization", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--keep-open", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0.0 < args.amplitude_scale <= 1.0:
        parser.error("--amplitude-scale must be in (0, 1]")
    if not 0.0 < args.speed_scale <= 1.0:
        parser.error("--speed-scale must be in (0, 1]")
    for name in (
        "command_rate_hz", "prepare_duration_s", "return_duration_s",
        "max_following_error_rad", "min_joint_limit_margin_rad",
        "max_command_velocity_rad_s", "max_force_utilization",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    source_q, source_fps, _ = load_motion(args.motion_npz)
    motion_q = scaled_motion(source_q, args.amplitude_scale, args.profile)
    model, compiled_xml = build_preview_model(
        args.headless, args.model_xml, preserve_collisions=True
    )
    data = mujoco.MjData(model)
    if args.initial_state_npz is None:
        initial_q = _joint_vector(model, H1_SIMULATION_START_TARGETS)
        initial_state_source = "built-in simulation start pose"
    else:
        initial_q = _final_joint_vector_from_npz(args.initial_state_npz)
        initial_state_source = str(args.initial_state_npz.expanduser().resolve())
    _set_joint_state(model, data, initial_q)
    original_initial_contacts = set(_self_contact_names(model, data)[0])
    initial_contact_body_pairs = _contact_body_pairs(model, data)
    if args.initial_state_npz is not None and initial_contact_body_pairs:
        model, compiled_xml = _compile_with_contact_exclusions(
            compiled_xml, initial_contact_body_pairs
        )
        data = mujoco.MjData(model)
        _set_joint_state(model, data, initial_q)
    initial_margin = _joint_limit_margin(model, initial_q)
    initial_contacts = set(_self_contact_names(model, data)[0])
    if args.profile in {"full-body", "lower-body", "upper-body", "torso-only"}:
        prepare_waypoints, prepare_weights = _staged_waypoints(initial_q, motion_q[0])
        reverse_waypoints, reverse_weights = _staged_waypoints(initial_q, motion_q[-1])
        return_waypoints = tuple(reversed(reverse_waypoints))
        return_weights = tuple(reversed(reverse_weights))
    else:
        prepare_waypoints, prepare_weights = (initial_q.copy(), motion_q[0].copy()), (1.0,)
        return_waypoints, return_weights = (motion_q[-1].copy(), initial_q.copy()), (1.0,)

    target_margin = min(_joint_limit_margin(model, row) for row in motion_q)
    source_duration_s = (len(motion_q) - 1) / source_fps
    motion_duration_s = source_duration_s / args.speed_scale
    total_s = (
        args.prepare_duration_s + args.prepare_settle_s + motion_duration_s
        + args.post_hold_s + args.return_duration_s + args.final_hold_s
    )
    command_period_s = 1.0 / args.command_rate_hz
    next_command_s = 0.0
    target = initial_q.copy()
    previous_target = initial_q.copy()
    phase = "prepare"

    times: list[float] = []
    desired_log: list[np.ndarray] = []
    measured_log: list[np.ndarray] = []
    velocity_log: list[np.ndarray] = []
    force_log: list[np.ndarray] = []
    phase_log: list[str] = []
    introduced_contacts: set[str] = set()
    contact_events: dict[str, dict[str, object]] = {}
    max_command_velocity = np.zeros(len(config.H1_JOINT_NAMES))
    command_updates = 0
    wall_start = time.monotonic()

    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.55)
            viewer.cam.distance = 2.6
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -8.0
        while data.time < total_s:
            if data.time + 1e-12 >= next_command_s:
                if data.time < total_s - args.final_hold_s:
                    target, phase = command_target(
                        elapsed_s=float(data.time),
                        initial_q=initial_q,
                        motion_q=motion_q,
                        source_fps=source_fps,
                        speed_scale=args.speed_scale,
                        prepare_s=args.prepare_duration_s,
                        settle_s=args.prepare_settle_s,
                        post_hold_s=args.post_hold_s,
                        return_s=args.return_duration_s,
                        prepare_waypoints=prepare_waypoints,
                        prepare_weights=prepare_weights,
                        return_waypoints=return_waypoints,
                        return_weights=return_weights,
                    )
                else:
                    target, phase = initial_q.copy(), "final_hold"
                max_command_velocity = np.maximum(
                    max_command_velocity,
                    np.abs(target - previous_target) / command_period_s,
                )
                previous_target = target.copy()
                _apply_joint_targets(model, data, target)
                next_command_s += command_period_s
                command_updates += 1

            mujoco.mj_step(model, data)
            measured = _joint_positions(model, data)
            velocity = _joint_velocities(model, data)
            current_contacts = set(_self_contact_names(model, data)[0])
            introduced_contacts.update(current_contacts - initial_contacts)
            for contact_name in current_contacts - initial_contacts:
                event = contact_events.setdefault(
                    contact_name,
                    {
                        "first_time_s": float(data.time),
                        "last_time_s": float(data.time),
                        "first_phase": phase,
                        "last_phase": phase,
                    },
                )
                event["last_time_s"] = float(data.time)
                event["last_phase"] = phase
            if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(velocity)):
                raise RuntimeError(f"Non-finite state at t={data.time:.4f} s")
            times.append(float(data.time))
            desired_log.append(target.copy())
            measured_log.append(measured)
            velocity_log.append(velocity)
            force_log.append(np.asarray(data.actuator_force, dtype=np.float64).copy())
            phase_log.append(phase)
            if viewer is not None:
                viewer.sync()
                remaining = data.time - (time.monotonic() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        if viewer is not None and args.keep_open and viewer.is_running():
            print("Validation finished. Close the MuJoCo window to save results.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

    times_array = np.asarray(times)
    desired = np.asarray(desired_log)
    measured = np.asarray(measured_log)
    velocity = np.asarray(velocity_log)
    actuator_force = np.asarray(force_log)
    error = desired - measured
    motion_mask = np.asarray([value == "motion" for value in phase_log])
    max_error_per_joint = np.max(np.abs(error), axis=0)
    max_motion_error_per_joint = np.max(np.abs(error[motion_mask]), axis=0)
    measured_margin = min(_joint_limit_margin(model, row) for row in measured)

    force_limits = np.max(np.abs(model.actuator_forcerange), axis=1)
    limited = np.asarray(model.actuator_forcelimited, dtype=bool)
    max_force = np.max(np.abs(actuator_force), axis=0)
    force_utilization = np.zeros_like(max_force)
    valid_force_limit = limited & (force_limits > 0.0)
    force_utilization[valid_force_limit] = (
        max_force[valid_force_limit] / force_limits[valid_force_limit]
    )
    max_force_utilization = (
        float(np.max(force_utilization[valid_force_limit]))
        if np.any(valid_force_limit) else 0.0
    )

    allowed_measured_margin = min(
        args.min_joint_limit_margin_rad,
        initial_margin - 0.002,
    )
    checks = {
        "target_joint_limits": target_margin >= args.min_joint_limit_margin_rad,
        "measured_joint_limits": measured_margin >= allowed_measured_margin,
        "following_error": float(np.max(max_error_per_joint)) <= args.max_following_error_rad,
        "command_velocity": float(np.max(max_command_velocity)) <= args.max_command_velocity_rad_s,
        "self_collision": not introduced_contacts,
        "actuator_force": max_force_utilization <= args.max_force_utilization + 1e-9,
    }
    safety_pass = all(checks.values())
    motion_label = args.motion_npz.stem.replace("_mj_fps50", "")
    label = (
        f"{motion_label}_{args.profile}_amp{args.amplitude_scale:.2f}_speed{args.speed_scale:.2f}"
        .replace(".", "p")
    )
    output_dir = (
        REPO_ROOT / "sysid_mujoco/results/h1/project_motion_validation" / label
        if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "trajectory_comparison.png"
    _save_plot(plot_path, times_array, desired, measured)
    metrics: dict[str, object] = {
        "simulation_only": True,
        "dds_initialized": False,
        "motion_npz": str(args.motion_npz.expanduser().resolve()),
        "model_xml": str(args.model_xml.expanduser().resolve()),
        "initial_state_source": initial_state_source,
        "compiled_xml": str(compiled_xml),
        "amplitude_scale": args.amplitude_scale,
        "speed_scale": args.speed_scale,
        "profile": args.profile,
        "source_fps": source_fps,
        "source_samples": len(source_q),
        "source_duration_s": source_duration_s,
        "executed_motion_duration_s": motion_duration_s,
        "simulation_duration_s": float(times_array[-1]),
        "command_rate_hz": args.command_rate_hz,
        "command_updates": command_updates,
        "target_min_joint_limit_margin_rad": target_margin,
        "initial_min_joint_limit_margin_rad": initial_margin,
        "measured_min_joint_limit_margin_rad": measured_margin,
        "max_all_joint_following_error_rad": float(np.max(max_error_per_joint)),
        "max_motion_following_error_rad": float(np.max(max_motion_error_per_joint)),
        "rmse_motion_following_error_rad": float(np.sqrt(np.mean(error[motion_mask] ** 2))),
        "max_following_error_per_joint_rad": dict(zip(config.H1_JOINT_NAMES, max_error_per_joint.tolist(), strict=True)),
        "max_motion_following_error_per_joint_rad": dict(zip(config.H1_JOINT_NAMES, max_motion_error_per_joint.tolist(), strict=True)),
        "max_command_velocity_rad_s": float(np.max(max_command_velocity)),
        "max_command_velocity_per_joint_rad_s": dict(zip(config.H1_JOINT_NAMES, max_command_velocity.tolist(), strict=True)),
        "max_measured_velocity_rad_s": float(np.max(np.abs(velocity))),
        "max_actuator_force": dict(zip(config.H1_JOINT_NAMES, max_force.tolist(), strict=True)),
        "max_force_utilization": max_force_utilization,
        "initial_self_contacts": sorted(original_initial_contacts),
        "excluded_initial_contact_body_pairs": [
            list(pair) for pair in sorted(initial_contact_body_pairs)
        ],
        "introduced_self_contacts": sorted(introduced_contacts),
        "introduced_self_contact_events": contact_events,
        "checks": checks,
        "safety_check_pass": safety_pass,
        "limits": {
            "max_following_error_rad": args.max_following_error_rad,
            "min_joint_limit_margin_rad": args.min_joint_limit_margin_rad,
            "allowed_measured_joint_limit_margin_rad": allowed_measured_margin,
            "max_command_velocity_rad_s": args.max_command_velocity_rad_s,
            "max_force_utilization": args.max_force_utilization,
        },
        "plot": str(plot_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "amplitude_scale": args.amplitude_scale,
        "speed_scale": args.speed_scale,
        "safety_check_pass": safety_pass,
        "checks": checks,
        "max_following_error_rad": metrics["max_all_joint_following_error_rad"],
        "max_motion_following_error_rad": metrics["max_motion_following_error_rad"],
        "max_command_velocity_rad_s": metrics["max_command_velocity_rad_s"],
        "target_limit_margin_rad": target_margin,
        "measured_limit_margin_rad": measured_margin,
        "max_force_utilization": max_force_utilization,
        "introduced_self_contacts": sorted(introduced_contacts),
        "metrics": str(metrics_path),
    }, indent=2))
    return metrics


def main() -> None:
    metrics = run(parse_args())
    if not metrics["safety_check_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
