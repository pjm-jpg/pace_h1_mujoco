"""Replay the exact Holosoma H1 hardware experiment command plan in MuJoCo.

This program imports ``H1SysidPlan`` from the local Holosoma checkout so the
step/sine/chirp timing is not independently reimplemented.  It never
initializes DDS and cannot send a real robot command.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from sysid_mujoco.common import (  # noqa: E402
    _absolutize_file_attributes,
    _disable_all_collisions,
    _remove_all_by_tag,
    _rewrite_actuators_as_general,
)
from sysid_mujoco.view_h1_motion import (  # noqa: E402
    add_viewer_scene,
    actuator_for_joint,
    initialize_position_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=config.H1_JOINT_NAMES, default="left_knee_joint")
    parser.add_argument(
        "--kind",
        choices=(
            "prepare", "hold", "step", "positive-step", "negative-step", "sine",
            "multi-sine", "chirp"
        ),
        default="prepare",
    )
    parser.add_argument("--amplitude-rad", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--step-hold-s", type=float, default=2.0)
    parser.add_argument("--step-base-s", type=float, default=1.0)
    parser.add_argument("--sine-duration-s", type=float, default=10.0)
    parser.add_argument("--sine-frequency-hz", type=float, default=0.5)
    parser.add_argument("--multi-sine-frequencies-hz", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--multi-sine-amplitudes-rad", type=float, nargs="+", default=(0.04, 0.04, 0.02))
    parser.add_argument("--multi-sine-segment-duration-s", type=float, default=10.0)
    parser.add_argument("--multi-sine-hold-s", type=float, default=2.0)
    parser.add_argument("--chirp-duration-s", type=float, default=20.0)
    parser.add_argument("--chirp-f0-hz", type=float, default=0.5)
    parser.add_argument("--chirp-f1-hz", type=float, default=5.0)
    parser.add_argument("--fade-s", type=float, default=0.5)
    parser.add_argument("--pre-hold-s", type=float, default=2.0)
    parser.add_argument("--post-hold-s", type=float, default=2.0)
    parser.add_argument("--command-rate-hz", type=float, default=50.0)
    parser.add_argument("--prepare-duration-s", type=float, default=6.0)
    parser.add_argument("--prepare-settle-s", type=float, default=2.0)
    parser.add_argument("--return-duration-s", type=float, default=6.0)
    parser.add_argument(
        "--staged-prepare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Holosoma arms-first staged prepare and reversed return route.",
    )
    parser.add_argument("--arm-clearance-pitch-rad", type=float, default=0.8)
    parser.add_argument(
        "--return-to-initial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the collector return behavior after the experiment.",
    )
    parser.add_argument("--max-following-error-rad", type=float, default=0.15)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--holosoma-root",
        type=Path,
        default=Path("/home/pjm/Desktop/holosoma"),
        help="Checkout containing the real H1 collector and H1SysidPlan.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml",
        help="Floating-base H1 source model used to build the fixed-base preview.",
    )
    parser.add_argument(
        "--initial-state-npz",
        type=Path,
        default=None,
        help="Optional H1 DDS raw.npz whose median q_real defines the preview start pose.",
    )
    parser.add_argument(
        "--preserve-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve robot collision geoms so self-contact checks are real. "
            "The default remains collision-free for backward compatibility."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: sysid_mujoco/results/h1/real_command_preview/<joint>_<kind>; "
            "sine directories also include frequency."
        ),
    )
    parser.add_argument("--headless", action="store_true", help="Run quickly without the MuJoCo window.")
    parser.add_argument(
        "--keep-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the final viewer frame open until the window is closed.",
    )
    args = parser.parse_args()
    if args.final_hold_s < 0.0:
        parser.error("--final-hold-s must be nonnegative")
    if args.max_following_error_rad <= 0.0:
        parser.error("--max-following-error-rad must be positive")
    return args


def load_hardware_plan(args: argparse.Namespace):
    source = args.holosoma_root.expanduser().resolve() / "src" / "holosoma_inference"
    collector = source / "holosoma_inference" / "collect_h1_sysid.py"
    if not collector.is_file():
        raise FileNotFoundError(f"Holosoma H1 collector not found: {collector}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from holosoma_inference.collect_h1_sysid import (  # noqa: PLC0415
        H1_JOINT_NAMES,
        H1_JOINT_TO_MOTOR,
        H1_SUSPENDED_PREPARE_TARGETS,
        H1SysidPlan,
    )

    if tuple(H1_JOINT_NAMES) != tuple(config.H1_JOINT_NAMES):
        raise RuntimeError("Holosoma and MuJoCo H1 joint orders differ.")
    if tuple(H1_JOINT_TO_MOTOR) != tuple(config.H1_JOINT_TO_MOTOR):
        raise RuntimeError("Holosoma and MuJoCo H1 joint-to-motor mappings differ.")
    plan = H1SysidPlan(
        joint=args.joint,
        kind=args.kind,
        amplitude_rad=args.amplitude_rad,
        pre_hold_s=args.pre_hold_s,
        post_hold_s=args.post_hold_s,
        step_hold_s=args.step_hold_s,
        step_base_s=args.step_base_s,
        repeats=args.repeats,
        sine_duration_s=args.sine_duration_s,
        sine_frequency_hz=args.sine_frequency_hz,
        multi_sine_frequencies_hz=tuple(args.multi_sine_frequencies_hz),
        multi_sine_amplitudes_rad=tuple(args.multi_sine_amplitudes_rad),
        multi_sine_segment_duration_s=args.multi_sine_segment_duration_s,
        multi_sine_hold_s=args.multi_sine_hold_s,
        chirp_duration_s=args.chirp_duration_s,
        chirp_f0_hz=args.chirp_f0_hz,
        chirp_f1_hz=args.chirp_f1_hz,
        fade_s=args.fade_s,
        command_rate_hz=args.command_rate_hz,
        prepare_duration_s=args.prepare_duration_s,
        prepare_settle_s=args.prepare_settle_s,
        return_duration_s=args.return_duration_s,
        staged_prepare=getattr(args, "staged_prepare", False),
        arm_clearance_pitch_rad=getattr(args, "arm_clearance_pitch_rad", 0.8),
        return_to_initial=getattr(args, "return_to_initial", True),
    )
    plan.validate(tuple(H1_JOINT_NAMES))
    return plan, dict(H1_SUSPENDED_PREPARE_TARGETS)


H1_SIMULATION_START_TARGETS = {
    "left_hip_pitch_joint": -0.40,
    "left_knee_joint": 0.80,
    "left_ankle_joint": -0.40,
    "right_hip_pitch_joint": -0.40,
    "right_knee_joint": 0.80,
    "right_ankle_joint": -0.40,
}


def _exclude_adjacent_body_contacts(root: ET.Element) -> None:
    """Disable only direct kinematic-neighbor contacts in a preview model."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        (element.get("body1"), element.get("body2"))
        for element in contact.findall("exclude")
    }

    def visit(parent: ET.Element) -> None:
        parent_name = parent.get("name")
        for child in parent.findall("body"):
            child_name = child.get("name")
            pair = (parent_name, child_name)
            if parent_name and child_name and pair not in existing:
                ET.SubElement(
                    contact,
                    "exclude",
                    {"body1": parent_name, "body2": child_name},
                )
                existing.add(pair)
            visit(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_level_body in worldbody.findall("body"):
            visit(top_level_body)


def build_preview_model(
    headless: bool,
    source_xml: Path,
    preserve_collisions: bool,
) -> tuple[mujoco.MjModel, Path]:
    """Build a fixed-base position-controlled model for command-plan previews."""
    source_xml = source_xml.expanduser().resolve()
    if not source_xml.is_file():
        raise FileNotFoundError(source_xml)
    output_dir = REPO_ROOT / "sysid_mujoco" / "generated" / "h1"
    output_dir.mkdir(parents=True, exist_ok=True)
    collision_label = "collisions" if preserve_collisions else "collision_free"
    output_xml = output_dir / f"{source_xml.stem}_preview_{collision_label}.xml"

    tree = ET.parse(source_xml)
    root = tree.getroot()
    _remove_all_by_tag(root, "freejoint")
    _remove_all_by_tag(root, "keyframe")
    _remove_all_by_tag(root, "accelerometer")
    _remove_all_by_tag(root, "gyro")
    _remove_all_by_tag(root, "framepos")
    _remove_all_by_tag(root, "framequat")
    gains = {
        name: (float(kp), float(kd))
        for name, kp, kd in zip(
            config.H1_JOINT_NAMES,
            config.Kp,
            config.Kd,
            strict=True,
        )
    }
    _rewrite_actuators_as_general(root, actuator_gains=gains)
    _absolutize_file_attributes(root, source_xml.parent)
    if preserve_collisions:
        _exclude_adjacent_body_contacts(root)
    else:
        _disable_all_collisions(root)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_xml.stem}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            tree.write(stream, encoding="utf-8", xml_declaration=True)
            stream.flush()
        temporary_path.replace(output_xml)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    compile_path = output_xml if headless else add_viewer_scene(output_xml)
    model = mujoco.MjModel.from_xml_path(str(compile_path))
    model.opt.timestep = 1.0 / config.frequency_collection
    return model, compile_path


def _joint_vector(model: mujoco.MjModel, overrides: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(config.H1_JOINT_NAMES), dtype=np.float64)
    for index, joint_name in enumerate(config.H1_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        values[index] = overrides.get(joint_name, 0.0)
        if bool(model.jnt_limited[joint_id]):
            lower, upper = model.jnt_range[joint_id]
            if not lower <= values[index] <= upper:
                raise ValueError(f"{joint_name} target {values[index]} outside [{lower}, {upper}]")
    return values


def _set_joint_state(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    for index, joint_name in enumerate(config.H1_JOINT_NAMES):
        joint_id, qpos_address, dof_address = _joint_ids(model, joint_name)
        data.qpos[qpos_address] = q[index]
        data.qvel[dof_address] = 0.0
        actuator_id, _ = actuator_for_joint(model, joint_name)
        data.ctrl[actuator_id] = q[index]
    mujoco.mj_forward(model, data)


def _initial_joint_vector_from_npz(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as recording:
        q_real = np.asarray(recording["q_real"], dtype=np.float64)
    if q_real.ndim != 2 or q_real.shape[0] == 0 or q_real.shape[1] < 20:
        raise ValueError(f"Expected nonempty q_real with 20 motor slots in {path}")
    values = np.asarray(
        [np.median(q_real[:, motor]) for motor in config.H1_JOINT_TO_MOTOR],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Initial joint state contains non-finite values: {path}")
    return values


def _apply_joint_targets(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
    for index, joint_name in enumerate(config.H1_JOINT_NAMES):
        actuator_id, _ = actuator_for_joint(model, joint_name)
        data.ctrl[actuator_id] = target[index]


def _self_contact_names(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[list[str], list[str]]:
    nonadjacent = []
    adjacent = []
    for contact in data.contact[: data.ncon]:
        labels = []
        body_ids = []
        for geom_id in (contact.geom1, contact.geom2):
            geom_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom#{geom_id}"
            )
            body_id = int(model.geom_bodyid[geom_id])
            body_ids.append(body_id)
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body#{body_id}"
            )
            labels.append(f"{body_name}/{geom_name}")
        label = f"{labels[0]} <-> {labels[1]}"
        is_adjacent = (
            int(model.body_parentid[body_ids[0]]) == body_ids[1]
            or int(model.body_parentid[body_ids[1]]) == body_ids[0]
        )
        (adjacent if is_adjacent else nonadjacent).append(label)
    return nonadjacent, adjacent


def _joint_ids(model: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
    return joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _save_plot(
    output_path: Path,
    times: np.ndarray,
    desired: np.ndarray,
    measured: np.ndarray,
    velocity: np.ndarray,
    plan,
    final_hold_s: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = desired - measured
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(times, desired, "--", label="50 Hz desired", color="tab:orange")
    axes[0].plot(times, measured, label="500 Hz MuJoCo response", color="tab:blue")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend()
    axes[1].plot(times, error, color="tab:red")
    axes[1].set_ylabel("following error [rad]")
    axes[2].plot(times, velocity, color="tab:green")
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].set_xlabel("time [s]")
    phase_boundaries = (
        plan.prepare_duration_s,
        plan.core_start_s,
        plan.return_start_s,
        plan.total_duration_s,
        plan.total_duration_s + final_hold_s,
    )
    for axis in axes:
        for boundary in phase_boundaries:
            axis.axvline(boundary, color="black", alpha=0.25, linewidth=0.8)
        axis.grid(True, alpha=0.3)
    amplitude_label = (
        f"amplitudes={list(plan.multi_sine_amplitudes_rad)} rad"
        if plan.kind == "multi-sine"
        else f"amplitude={plan.amplitude_rad:.4f} rad"
    )
    figure.suptitle(
        f"H1 real-command simulation preview: {plan.joint} / {plan.kind} / {amplitude_label}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    plan, prepare_targets = load_hardware_plan(args)
    model, model_xml = build_preview_model(
        args.headless,
        args.model_xml,
        args.preserve_collisions,
    )
    data = mujoco.MjData(model)
    if args.initial_state_npz is None:
        initial_q = _joint_vector(model, H1_SIMULATION_START_TARGETS)
        initial_state_source = "built-in simulation start targets"
    else:
        initial_q = _initial_joint_vector_from_npz(args.initial_state_npz)
        initial_state_source = str(args.initial_state_npz.expanduser().resolve())
    prepare_q = initial_q.copy()
    for joint_name, value in prepare_targets.items():
        prepare_q[config.H1_JOINT_NAMES.index(joint_name)] = value
    for index, joint_name in enumerate(config.H1_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if bool(model.jnt_limited[joint_id]):
            lower, upper = model.jnt_range[joint_id]
            if not lower <= prepare_q[index] <= upper:
                raise ValueError(
                    f"Prepare target for {joint_name} ({prepare_q[index]}) "
                    f"outside [{lower}, {upper}]"
                )
    _set_joint_state(model, data, initial_q)
    initial_contacts = set(_self_contact_names(model, data)[0])
    initialize_position_targets(model, data)
    actuator_id, _ = actuator_for_joint(model, plan.joint)
    joint_id, qpos_address, dof_address = _joint_ids(model, plan.joint)
    joint_index = config.H1_JOINT_NAMES.index(plan.joint)
    center = float(prepare_q[joint_index])
    kp = float(config.Kp[config.H1_JOINT_NAMES.index(plan.joint)])
    kd = float(config.Kd[config.H1_JOINT_NAMES.index(plan.joint)])
    motor_id = int(config.H1_JOINT_TO_MOTOR[config.H1_JOINT_NAMES.index(plan.joint)])
    joint_limited = bool(model.jnt_limited[joint_id])
    joint_range = tuple(float(v) for v in model.jnt_range[joint_id])
    excitation_amplitude = (
        max(plan.multi_sine_amplitudes_rad)
        if plan.kind == "multi-sine"
        else plan.amplitude_rad
    )
    possible_targets = (center - excitation_amplitude, center + excitation_amplitude)
    if joint_limited and (
        possible_targets[0] < joint_range[0] or possible_targets[1] > joint_range[1]
    ):
        raise ValueError(
            f"Requested targets {possible_targets} exceed {plan.joint} range {joint_range}."
        )

    simulation_duration = plan.total_duration_s + args.final_hold_s
    command_period = 1.0 / plan.command_rate_hz
    next_command_time = 0.0
    desired = center
    command_updates = 0
    times: list[float] = []
    desired_history: list[float] = []
    measured_history: list[float] = []
    velocity_history: list[float] = []
    all_joint_error_max = 0.0
    per_joint_error_max = np.zeros(len(config.H1_JOINT_NAMES), dtype=np.float64)
    min_joint_limit_margin = float("inf")
    initial_joint_limit_margin = float("inf")
    for index, joint_name in enumerate(config.H1_JOINT_NAMES):
        this_joint_id, _, _ = _joint_ids(model, joint_name)
        if bool(model.jnt_limited[this_joint_id]):
            lower, upper = model.jnt_range[this_joint_id]
            initial_joint_limit_margin = min(
                initial_joint_limit_margin,
                float(initial_q[index] - lower),
                float(upper - initial_q[index]),
            )
    min_foot_separation = float("inf")
    contacts: set[str] = set()
    introduced_contacts: set[str] = set()
    arm_contacts_after_lift: set[str] = set()
    contacts_after_prepare: set[str] = set()
    adjacent_contacts: set[str] = set()
    contact_events: dict[str, dict[str, float | int]] = {}
    last_report = -0.5
    wall_start = time.monotonic()

    print("H1 real-hardware command simulation preview")
    print(f"  source plan:  {args.holosoma_root}/src/holosoma_inference/.../collect_h1_sysid.py")
    print(f"  joint:        {plan.joint} (physical motor ID {motor_id})")
    print(f"  kind:         {plan.kind}")
    if plan.kind == "multi-sine":
        schedule = list(zip(plan.multi_sine_frequencies_hz, plan.multi_sine_amplitudes_rad))
        print(f"  multi-sine:   {schedule} (Hz, rad)")
    else:
        print(f"  amplitude:    {plan.amplitude_rad:.4f} rad")
    print(f"  Kp / Kd:      {kp:.3f} / {kd:.3f}")
    print(f"  command rate: {plan.command_rate_hz:.1f} Hz (zero-order hold)")
    print(f"  MuJoCo rate:  {config.frequency_collection:.1f} Hz")
    print(f"  plan time:    {plan.total_duration_s:.2f} s + {args.final_hold_s:.2f} s final hold")
    print(f"  prepare:      {plan.prepare_duration_s:.2f} s ramp + {plan.prepare_settle_s:.2f} s settle")
    print(f"  return:       {plan.return_duration_s:.2f} s smooth ramp")
    print("  hardware DDS: DISABLED (simulation only)")

    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.55)
            viewer.cam.distance = 2.4
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -10.0
        while data.time < simulation_duration:
            if data.time + 1e-12 >= next_command_time:
                if data.time < plan.total_duration_s:
                    target = plan.command_target(initial_q, prepare_q, joint_index, data.time)
                else:
                    # Match Holosoma's one-second final safety hold.
                    target = (
                        initial_q.copy()
                        if plan.return_to_initial
                        else prepare_q.copy()
                    )
                desired = float(target[joint_index])
                _apply_joint_targets(model, data, target)
                command_updates += 1
                next_command_time += command_period

            mujoco.mj_step(model, data)
            measured = float(data.qpos[qpos_address])
            velocity = float(data.qvel[dof_address])
            times.append(float(data.time))
            desired_history.append(desired)
            measured_history.append(measured)
            velocity_history.append(velocity)
            measured_all = np.empty(len(config.H1_JOINT_NAMES), dtype=np.float64)
            for index, joint_name in enumerate(config.H1_JOINT_NAMES):
                this_joint_id, this_qpos, _ = _joint_ids(model, joint_name)
                measured_all[index] = data.qpos[this_qpos]
                if bool(model.jnt_limited[this_joint_id]):
                    lower, upper = model.jnt_range[this_joint_id]
                    min_joint_limit_margin = min(
                        min_joint_limit_margin,
                        float(measured_all[index] - lower),
                        float(upper - measured_all[index]),
                    )
            absolute_joint_error = np.abs(target - measured_all)
            per_joint_error_max = np.maximum(per_joint_error_max, absolute_joint_error)
            all_joint_error_max = max(all_joint_error_max, float(np.max(absolute_joint_error)))
            current_contacts, current_adjacent_contacts = _self_contact_names(model, data)
            contacts.update(current_contacts)
            introduced_contacts.update(set(current_contacts) - initial_contacts)
            first_stage_end_s = plan.prepare_duration_s * 6.0 / 24.0
            if plan.staged_prepare and data.time >= first_stage_end_s:
                arm_contacts_after_lift.update(
                    name
                    for name in current_contacts
                    if "shoulder" in name or "elbow" in name
                )
            if data.time >= plan.prepare_duration_s:
                contacts_after_prepare.update(current_contacts)
            adjacent_contacts.update(current_adjacent_contacts)
            for contact_name in current_contacts:
                event = contact_events.setdefault(
                    contact_name,
                    {
                        "first_time_s": float(data.time),
                        "last_time_s": float(data.time),
                        "samples": 0,
                        "target_min_rad": desired,
                        "target_max_rad": desired,
                        "measured_min_rad": measured,
                        "measured_max_rad": measured,
                    },
                )
                event["last_time_s"] = float(data.time)
                event["samples"] = int(event["samples"]) + 1
                event["target_min_rad"] = min(float(event["target_min_rad"]), desired)
                event["target_max_rad"] = max(float(event["target_max_rad"]), desired)
                event["measured_min_rad"] = min(
                    float(event["measured_min_rad"]), measured
                )
                event["measured_max_rad"] = max(
                    float(event["measured_max_rad"]), measured
                )
            left_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_contact_point")
            right_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_contact_point")
            min_foot_separation = min(
                min_foot_separation,
                float(np.linalg.norm(data.xpos[left_foot] - data.xpos[right_foot])),
            )
            if not np.isfinite(measured) or not np.isfinite(velocity):
                raise RuntimeError(f"Non-finite MuJoCo state at t={data.time:.3f} s")

            if data.time - last_report >= 0.5:
                print(
                    f"t={data.time:6.2f} target={desired:+.5f} "
                    f"actual={measured:+.5f} error={desired - measured:+.5f}",
                    flush=True,
                )
                last_report = float(data.time)
            if viewer is not None:
                viewer.sync()
                remaining = data.time - (time.monotonic() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

        if viewer is not None and args.keep_open and viewer.is_running():
            print("Preview finished. Close the MuJoCo window to save results.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

    times_array = np.asarray(times)
    desired_array = np.asarray(desired_history)
    measured_array = np.asarray(measured_history)
    velocity_array = np.asarray(velocity_history)
    following_error = desired_array - measured_array
    max_error = float(np.max(np.abs(following_error)))
    run_label = f"{plan.joint}_{plan.kind}"
    if plan.kind == "sine":
        frequency_label = f"{plan.sine_frequency_hz:g}".replace(".", "p")
        run_label += f"_{frequency_label}hz"
    output_dir = (
        REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "real_command_preview"
        / run_label
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "command_preview.png"
    _save_plot(
        plot_path,
        times_array,
        desired_array,
        measured_array,
        velocity_array,
        plan,
        args.final_hold_s,
    )
    metrics: dict[str, object] = {
        "simulation_only": True,
        "dds_initialized": False,
        "model_xml": str(model_xml),
        "source_model_xml": str(args.model_xml.expanduser().resolve()),
        "collision_check_enabled": bool(args.preserve_collisions),
        "plan_source": str(
            args.holosoma_root.expanduser().resolve()
            / "src/holosoma_inference/holosoma_inference/collect_h1_sysid.py"
        ),
        "plan": asdict(plan),
        "joint": plan.joint,
        "motor_id": motor_id,
        "excitation_center_rad": center,
        "kp": kp,
        "kd": kd,
        "simulation_rate_hz": config.frequency_collection,
        "command_rate_hz": plan.command_rate_hz,
        "command_updates": command_updates,
        "simulation_duration_s": float(times_array[-1]),
        "final_hold_s": args.final_hold_s,
        "target_min_rad": float(np.min(desired_array)),
        "target_max_rad": float(np.max(desired_array)),
        "measured_min_rad": float(np.min(measured_array)),
        "measured_max_rad": float(np.max(measured_array)),
        "final_target_rad": float(desired_array[-1]),
        "final_measured_rad": float(measured_array[-1]),
        "final_return_error_rad": float(desired_array[-1] - measured_array[-1]),
        "rmse_following_error_rad": float(np.sqrt(np.mean(following_error**2))),
        "max_following_error_rad": max_error,
        "max_abs_velocity_rad_s": float(np.max(np.abs(velocity_array))),
        "max_all_joint_following_error_rad": all_joint_error_max,
        "max_following_error_per_joint_rad": {
            name: float(value)
            for name, value in zip(
                config.H1_JOINT_NAMES,
                per_joint_error_max,
                strict=True,
            )
        },
        "following_error_limit_rad": args.max_following_error_rad,
        "minimum_joint_limit_margin_rad": min_joint_limit_margin,
        "initial_joint_limit_margin_rad": initial_joint_limit_margin,
        "joint_limit_not_worsened": (
            min_joint_limit_margin
            >= min(0.0, initial_joint_limit_margin) - 0.002
        ),
        "minimum_foot_point_separation_m": min_foot_separation,
        "self_contacts": sorted(contacts),
        "self_contact_count": len(contacts),
        "self_contact_events": contact_events,
        "ignored_adjacent_contacts": sorted(adjacent_contacts),
        "ignored_adjacent_contact_count": len(adjacent_contacts),
        "prepare_targets": prepare_targets,
        "simulation_start_targets": {
            name: float(value)
            for name, value in zip(config.H1_JOINT_NAMES, initial_q, strict=True)
        },
        "initial_state_source": initial_state_source,
        "initial_self_contacts": sorted(initial_contacts),
        "introduced_self_contacts": sorted(introduced_contacts),
        "arm_contacts_after_lift": sorted(arm_contacts_after_lift),
        "contacts_after_prepare": sorted(contacts_after_prepare),
        "safety_check_pass": (
            all_joint_error_max <= args.max_following_error_rad
            and min_joint_limit_margin
            >= min(0.0, initial_joint_limit_margin) - 0.002
            and (
                not args.preserve_collisions
                or (
                    not introduced_contacts
                    and not arm_contacts_after_lift
                    and not contacts_after_prepare
                )
            )
        ),
        "joint_range_rad": list(joint_range),
    }
    metrics_path = output_dir / "preview_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Safety check: {'PASS' if metrics['safety_check_pass'] else 'FAIL'}")
    print(f"Maximum following error: {max_error:.6f} rad")
    print(f"Maximum all-joint following error: {all_joint_error_max:.6f} rad")
    print(f"Minimum joint-limit margin: {min_joint_limit_margin:.6f} rad")
    print(f"Initial joint-limit margin: {initial_joint_limit_margin:.6f} rad")
    print(f"Minimum foot-point separation: {min_foot_separation:.6f} m")
    print(f"Self contacts: {len(contacts)}")
    print(f"Initial self contacts: {len(initial_contacts)}")
    print(f"Introduced self contacts: {len(introduced_contacts)}")
    print(f"Arm contacts after lift: {len(arm_contacts_after_lift)}")
    print(f"Contacts after prepare: {len(contacts_after_prepare)}")
    print(f"Ignored adjacent-link contacts: {len(adjacent_contacts)}")
    print(f"Plot: {plot_path}")
    print(f"Metrics: {metrics_path}")
    return metrics


def main() -> None:
    metrics = run(parse_args())
    if not metrics["safety_check_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
