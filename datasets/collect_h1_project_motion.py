"""Collect a guarded, coordinated H1 project-motion response on hardware.

The default command is a dry run.  Hardware publishing requires both
``--execute`` and ``--confirm H1_PROJECT_MOTION``.  Lower- and upper-body
profiles hold the unexcited joints in the known suspended reference posture.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


DEFAULT_HOLOSOMA_ROOT = Path("/home/pjm/Desktop/holosoma")
DEFAULT_MOTION = (
    DEFAULT_HOLOSOMA_ROOT
    / "src/holosoma/holosoma/data/motions/walk1_subject1_mj_fps50.npz"
)
DEFAULT_MODEL = REPO_ROOT / "robot_model/h1/h1_calibrated_full_body.xml"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/h1/raw"

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
ARM_JOINTS = tuple(
    name for name in config.H1_JOINT_NAMES
    if "shoulder" in name or "elbow" in name
)


@dataclass(frozen=True)
class ProjectMotionPlan:
    motion_npz: str
    profile: str = "lower-body"
    amplitude_scale: float = 0.60
    speed_scale: float = 0.50
    command_rate_hz: float = 50.0
    prepare_duration_s: float = 18.0
    prepare_settle_s: float = 2.0
    post_hold_s: float = 2.0
    return_duration_s: float = 18.0

    @property
    def motion_source_duration_s(self) -> float:
        q, fps = _load_motion(Path(self.motion_npz))
        return (len(q) - 1) / fps

    @property
    def motion_execution_duration_s(self) -> float:
        return self.motion_source_duration_s / self.speed_scale

    @property
    def total_duration_s(self) -> float:
        return (
            self.prepare_duration_s + self.prepare_settle_s
            + self.motion_execution_duration_s + self.post_hold_s
            + self.return_duration_s
        )


def _smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _load_motion(path: Path) -> tuple[np.ndarray, float]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=True) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        names = tuple(str(value) for value in np.asarray(motion["joint_names"]).tolist())
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    if names != tuple(config.H1_JOINT_NAMES):
        raise ValueError("Motion joint order does not match H1 19-DOF order.")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 26:
        raise ValueError(f"Expected joint_pos shape (T, 26), got {joint_pos.shape}.")
    q = joint_pos[:, 7:].copy()
    if len(q) < 2 or fps <= 0.0 or not np.all(np.isfinite(q)):
        raise ValueError("Motion contains invalid samples or fps.")
    return q, fps


def _scaled_motion(source_q: np.ndarray, plan: ProjectMotionPlan) -> np.ndarray:
    reference = np.asarray(
        [SUSPENDED_REFERENCE[name] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )
    result = reference + plan.amplitude_scale * (source_q - reference)
    if plan.profile == "lower-body":
        for name in ARM_JOINTS:
            result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif plan.profile == "upper-body":
        for name in config.H1_JOINT_NAMES:
            if name not in ARM_JOINTS:
                result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif plan.profile == "torso-only":
        for name in config.H1_JOINT_NAMES:
            if name != "torso_joint":
                result[:, config.H1_JOINT_NAMES.index(name)] = SUSPENDED_REFERENCE[name]
    elif plan.profile != "full-body":
        raise ValueError(f"Unsupported profile: {plan.profile}")
    return result


def _staged_waypoints(
    initial_q: np.ndarray,
    target_q: np.ndarray,
    arm_clearance_pitch_rad: float = 0.80,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    """Build the simulation-reviewed arms-first clearance path."""
    arm_names = set(ARM_JOINTS)
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
        lifted[
            config.H1_JOINT_NAMES.index(f"{side}_shoulder_pitch_joint")
        ] = arm_clearance_pitch_rad

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


def _sample_motion(q: np.ndarray, fps: float, source_time_s: float) -> np.ndarray:
    sample = float(np.clip(source_time_s * fps, 0.0, len(q) - 1))
    lower = int(np.floor(sample))
    upper = min(lower + 1, len(q) - 1)
    blend = sample - lower
    return q[lower] + blend * (q[upper] - q[lower])


def _target(
    elapsed_s: float,
    initial_q: np.ndarray,
    motion_q: np.ndarray,
    fps: float,
    plan: ProjectMotionPlan,
    prepare_waypoints: tuple[np.ndarray, ...] | None = None,
    prepare_weights: tuple[float, ...] | None = None,
    return_waypoints: tuple[np.ndarray, ...] | None = None,
    return_weights: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, str]:
    core_start_s = plan.prepare_duration_s + plan.prepare_settle_s
    core_end_s = core_start_s + plan.motion_execution_duration_s
    return_start_s = core_end_s + plan.post_hold_s
    if elapsed_s < plan.prepare_duration_s:
        if prepare_waypoints is not None and prepare_weights is not None:
            return _piecewise_smooth_target(
                prepare_waypoints,
                prepare_weights,
                elapsed_s,
                plan.prepare_duration_s,
            ), "prepare"
        blend = _smoothstep5(elapsed_s / plan.prepare_duration_s)
        return initial_q + blend * (motion_q[0] - initial_q), "prepare"
    if elapsed_s < core_start_s:
        return motion_q[0].copy(), "settle"
    if elapsed_s < core_end_s:
        source_time_s = (elapsed_s - core_start_s) * plan.speed_scale
        return _sample_motion(motion_q, fps, source_time_s), "motion"
    if elapsed_s < return_start_s:
        return motion_q[-1].copy(), "post_hold"
    if return_waypoints is not None and return_weights is not None:
        return _piecewise_smooth_target(
            return_waypoints,
            return_weights,
            elapsed_s - return_start_s,
            plan.return_duration_s,
        ), "return"
    blend = _smoothstep5((elapsed_s - return_start_s) / plan.return_duration_s)
    return motion_q[-1] + blend * (initial_q - motion_q[-1]), "return"


def _joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path.expanduser().resolve()).getroot()
    result = {}
    for joint in root.iter("joint"):
        name, value = joint.get("name"), joint.get("range")
        if name in config.H1_JOINT_NAMES and value:
            lower, upper = (float(part) for part in value.split())
            result[name] = (lower, upper)
    missing = set(config.H1_JOINT_NAMES) - set(result)
    if missing:
        raise ValueError(f"Model is missing H1 joint limits: {sorted(missing)}")
    return result


def _validate_plan(
    plan: ProjectMotionPlan,
    motion_q: np.ndarray,
    fps: float,
    model_xml: Path,
    min_limit_margin_rad: float,
    max_command_velocity_rad_s: float,
) -> dict[str, object]:
    if not 0.0 < plan.amplitude_scale <= 1.0:
        raise ValueError("amplitude scale must be in (0, 1]")
    if not 0.0 < plan.speed_scale <= 1.0:
        raise ValueError("speed scale must be in (0, 1]")
    limits = _joint_limits(model_xml)
    minimum_margin = float("inf")
    for index, name in enumerate(config.H1_JOINT_NAMES):
        lower, upper = limits[name]
        minimum_margin = min(
            minimum_margin,
            float(np.min(motion_q[:, index]) - lower),
            float(upper - np.max(motion_q[:, index])),
        )
    source_velocity = np.diff(motion_q, axis=0) * fps * plan.speed_scale
    maximum_velocity = float(np.max(np.abs(source_velocity)))
    maximum_step = maximum_velocity / plan.command_rate_hz
    if minimum_margin < min_limit_margin_rad:
        raise ValueError(
            f"Target joint-limit margin {minimum_margin:.4f} rad is below "
            f"{min_limit_margin_rad:.4f} rad."
        )
    if maximum_velocity > max_command_velocity_rad_s:
        raise ValueError(
            f"Target velocity {maximum_velocity:.4f} rad/s exceeds "
            f"{max_command_velocity_rad_s:.4f} rad/s."
        )
    return {
        "minimum_target_joint_limit_margin_rad": minimum_margin,
        "maximum_target_velocity_rad_s": maximum_velocity,
        "maximum_target_step_rad": maximum_step,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", type=Path, default=DEFAULT_MOTION)
    parser.add_argument(
        "--profile",
        choices=("lower-body", "upper-body", "torso-only", "full-body"),
        default="lower-body",
    )
    parser.add_argument("--amplitude-scale", type=float, default=0.60)
    parser.add_argument("--speed-scale", type=float, default=0.50)
    parser.add_argument("--command-rate-hz", type=float, default=50.0)
    parser.add_argument("--prepare-duration-s", type=float, default=18.0)
    parser.add_argument("--prepare-settle-s", type=float, default=2.0)
    parser.add_argument("--post-hold-s", type=float, default=2.0)
    parser.add_argument("--return-duration-s", type=float, default=18.0)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA_ROOT)
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--max-following-error-rad", type=float, default=0.15)
    parser.add_argument("--min-joint-limit-margin-rad", type=float, default=0.03)
    parser.add_argument("--max-command-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--max-state-age-ms", type=float, default=20.0)
    parser.add_argument("--max-temperature-c", type=float, default=70.0)
    parser.add_argument("--max-motor-lost-rate-hz", type=float, default=0.5)
    parser.add_argument("--motor-lost-burst-allowance", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = ProjectMotionPlan(
        motion_npz=str(args.motion_npz.expanduser().resolve()),
        profile=args.profile,
        amplitude_scale=args.amplitude_scale,
        speed_scale=args.speed_scale,
        command_rate_hz=args.command_rate_hz,
        prepare_duration_s=args.prepare_duration_s,
        prepare_settle_s=args.prepare_settle_s,
        post_hold_s=args.post_hold_s,
        return_duration_s=args.return_duration_s,
    )
    source_q, fps = _load_motion(Path(plan.motion_npz))
    motion_q = _scaled_motion(source_q, plan)
    validation = _validate_plan(
        plan, motion_q, fps, args.model_xml,
        args.min_joint_limit_margin_rad,
        args.max_command_velocity_rad_s,
    )
    preview = {
        **asdict(plan),
        **validation,
        "motion_source_duration_s": plan.motion_source_duration_s,
        "motion_execution_duration_s": plan.motion_execution_duration_s,
        "total_duration_s": plan.total_duration_s,
        "publish_lowcmd": bool(args.execute),
        "simulation_gate": "requires matching MuJoCo validation before execution",
        "safety": "robot suspended, feet clear, one lowcmd publisher only",
    }
    print(json.dumps(preview, indent=2))
    if args.profile == "full-body":
        raise SystemExit(
            "Full-body walk1 is blocked: simulation found elbow/hip self-contact. "
            "Use --profile lower-body or --profile upper-body."
        )
    if not args.execute:
        print("Dry run only. Add --execute --confirm H1_PROJECT_MOTION after review.")
        return
    if args.confirm != "H1_PROJECT_MOTION":
        raise SystemExit("Hardware execution requires --confirm H1_PROJECT_MOTION.")

    source = args.holosoma_root.expanduser().resolve() / "src/holosoma_inference"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from holosoma_inference.collect_h1_sysid import _check_state, _read_motor_lost, _send_hold
    from holosoma_inference.config.config_values import inference
    from holosoma_inference.sdk.unitree.unitree_interface_mp import UnitreeInterfaceMP
    from holosoma_inference.utils.rate import RateLimiter

    robot_config = inference.h1_19dof_loco_real.robot
    if tuple(robot_config.dof_names) != tuple(config.H1_JOINT_NAMES):
        raise RuntimeError("Runtime H1 joint order differs from the reviewed plan.")
    if tuple(robot_config.joint2motor) != tuple(config.H1_JOINT_TO_MOTOR):
        raise RuntimeError("Runtime H1 motor mapping differs from the reviewed plan.")

    interface = UnitreeInterfaceMP(
        robot_config,
        domain_id=args.domain_id,
        interface_str=args.interface,
        use_joystick=False,
    )
    try:
        motion_mode = interface.get_motion_mode()
        print("MotionSwitcher: " + json.dumps(motion_mode))
        if motion_mode["status"] != 0 or motion_mode["name"]:
            raise RuntimeError(
                "High-level motion service still owns control or CheckMode failed: "
                f"{motion_mode}"
            )
        state = interface.get_low_state()
        initial_q = np.asarray(
            state[0, 7 : 7 + robot_config.num_joints], dtype=np.float64
        )
        if not np.all(np.isfinite(initial_q)):
            raise RuntimeError("Initial H1 joint state is not finite.")
        initial_delta = np.max(np.abs(motion_q[0] - initial_q))
        if initial_delta > 1.6:
            raise RuntimeError(
                f"Prepare delta {initial_delta:.3f} rad exceeds 1.600 rad."
            )

        if plan.profile in {"lower-body", "upper-body", "torso-only"}:
            prepare_waypoints, prepare_weights = _staged_waypoints(
                initial_q, motion_q[0]
            )
            reverse_waypoints, reverse_weights = _staged_waypoints(
                initial_q, motion_q[-1]
            )
            return_waypoints = tuple(reversed(reverse_waypoints))
            return_weights = tuple(reversed(reverse_weights))
        else:
            prepare_waypoints = prepare_weights = None
            return_waypoints = return_weights = None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        motion_label = Path(plan.motion_npz).stem.replace("_mj_fps50", "")
        output_dir = (
            args.output_root.expanduser().resolve()
            / (
                f"{stamp}_h1_{motion_label}_{plan.profile.replace('-', '_')}"
                f"_amp{plan.amplitude_scale:.2f}_speed{plan.speed_scale:.2f}"
            )
        )
        metadata = {
            **preview,
            "kind": "project-motion",
            "command_available": True,
            "profile": plan.profile,
            "center_q": initial_q.tolist(),
            "motion_first_q": motion_q[0].tolist(),
            "motion_last_q": motion_q[-1].tolist(),
            "joint_names": list(config.H1_JOINT_NAMES),
            "gain_source": "h1_19dof_loco_real.robot",
            "motion_mode_before": motion_mode,
        }
        zeros = np.zeros_like(initial_q)
        rate = RateLimiter(plan.command_rate_hz)
        motor_lost_baseline = _read_motor_lost(interface.get_raw_motor_state())
        interface.send_low_command(initial_q, zeros, zeros, initial_q)
        interface.start_sysid_recording(
            str(output_dir), metadata, plan.total_duration_s + 5.0, 500.0
        )
        started = time.monotonic()
        result, abort_reason = "completed", None
        last_measured = initial_q.copy()
        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= plan.total_duration_s:
                    break
                state = interface.get_low_state()
                measured = np.asarray(
                    state[0, 7 : 7 + robot_config.num_joints], dtype=np.float64
                )
                last_measured = measured.copy()
                target, _ = _target(
                    elapsed,
                    initial_q,
                    motion_q,
                    fps,
                    plan,
                    prepare_waypoints,
                    prepare_weights,
                    return_waypoints,
                    return_weights,
                )
                _check_state(
                    interface, target, measured, args,
                    motor_lost_baseline, elapsed,
                )
                interface.send_low_command(target, zeros, zeros, measured)
                rate.sleep()
        except BaseException as exc:
            result = "aborted"
            abort_reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                final_hold = initial_q if result == "completed" else last_measured
                _send_hold(interface, final_hold, max(1, round(plan.command_rate_hz)), rate)
            finally:
                saved = interface.stop_sysid_recording(result, abort_reason)
        print(json.dumps(saved, indent=2))
    finally:
        interface.close()


if __name__ == "__main__":
    main()
