"""Visualize simple fixed-base H1 joint tests in the MuJoCo viewer.

This is a simulation-only visualization tool.  Its default motion amplitude is
intentionally large enough to see on screen and must not be copied to a real
robot experiment.
"""

from __future__ import annotations

import argparse
import math
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
from sysid_mujoco.common import build_fixed_base_model_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint",
        default="left_knee_joint",
        choices=config.H1_JOINT_NAMES,
        help="H1 joint to move (default: left knee).",
    )
    parser.add_argument(
        "--motion",
        choices=("step", "sine", "chirp"),
        default="step",
        help="Desired-position test trajectory.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.15,
        help="Motion amplitude in radians (simulation only; default: 0.15).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.5,
        help="Sine frequency, or chirp final frequency, in Hz.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the viewer (useful for a smoke test).",
    )
    parser.add_argument(
        "--keep-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the final frame visible until the viewer is closed (default: true).",
    )
    args = parser.parse_args()
    if args.amplitude <= 0.0:
        parser.error("--amplitude must be positive")
    if args.duration <= 1.0:
        parser.error("--duration must be greater than one second")
    if args.frequency <= 0.0:
        parser.error("--frequency must be positive")
    return args


def target_offset(args: argparse.Namespace, elapsed: float) -> float:
    """Return a smooth test offset after a one-second settling period."""
    if args.motion == "step":
        step_on = min(2.0, args.duration / 3.0)
        step_off = 0.6 * args.duration
        return args.amplitude if step_on <= elapsed < step_off else 0.0

    motion_time = elapsed - 1.0
    if motion_time <= 0.0:
        return 0.0
    if args.motion == "sine":
        return args.amplitude * math.sin(2.0 * math.pi * args.frequency * motion_time)

    # Linear chirp from 0.1 Hz to --frequency over the active test duration.
    active_duration = max(args.duration - 1.0, model_timestep_floor())
    start_frequency = 0.1
    chirp_rate = (args.frequency - start_frequency) / active_duration
    phase = 2.0 * math.pi * (
        start_frequency * motion_time + 0.5 * chirp_rate * motion_time**2
    )
    return args.amplitude * math.sin(phase)


def model_timestep_floor() -> float:
    return 1.0e-9


def add_viewer_scene(fixed_base_xml: Path) -> Path:
    """Add the lighting and floor that live in scene_flat.xml."""
    tree = ET.parse(fixed_base_xml)
    root = tree.getroot()

    statistic = root.find("statistic")
    if statistic is None:
        statistic = ET.SubElement(root, "statistic")
    statistic.set("center", "0 0 0.6")
    statistic.set("extent", "1.5")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.7 0.7 0.7",
            "ambient": "0.35 0.35 0.35",
            "specular": "0.25 0.25 0.25",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.92 0.95 0.98 1"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-20"})

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.35 0.45 0.60",
            "rgb2": "0.85 0.90 0.95",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "sysid_ground_texture",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.20 0.24 0.28",
            "rgb2": "0.30 0.34 0.38",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "sysid_ground_material",
            "texture": "sysid_ground_texture",
            "texuniform": "true",
            "texrepeat": "8 8",
            "reflectance": "0.05",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Generated H1 XML has no worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": "1 -1 4",
            "dir": "-0.2 0.2 -1",
            "directional": "true",
            "diffuse": "0.9 0.9 0.9",
            "castshadow": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "sysid_viewer_floor",
            "type": "plane",
            "size": "0 0 0.05",
            "material": "sysid_ground_material",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    viewer_xml = fixed_base_xml.with_name("h1_fixed_base_viewer.xml")
    tree.write(viewer_xml, encoding="utf-8", xml_declaration=True)
    return viewer_xml


def build_model(headless: bool) -> mujoco.MjModel:
    gains = {
        name: (float(kp), float(kd))
        for name, kp, kd in zip(
            config.H1_JOINT_NAMES,
            config.Kp,
            config.Kd,
            strict=True,
        )
    }
    xml_path = build_fixed_base_model_xml("h1", actuator_gains=gains)
    if not headless:
        xml_path = add_viewer_scene(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 1.0 / config.frequency_collection
    return model


def actuator_for_joint(model: mujoco.MjModel, requested_joint: str) -> tuple[int, int]:
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name == requested_joint:
            return actuator_id, int(model.jnt_qposadr[joint_id])
    raise ValueError(f"No actuator found for joint {requested_joint!r}")


def initialize_position_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = data.qpos[qpos_address]


def clip_target(model: mujoco.MjModel, actuator_id: int, target: float) -> float:
    if not model.actuator_ctrllimited[actuator_id]:
        return target
    lower, upper = model.actuator_ctrlrange[actuator_id]
    return float(np.clip(target, lower, upper))


def step_simulation(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: int,
    qpos_address: int,
    initial_target: float,
    wall_start: float,
    last_report: float,
) -> float:
    desired = clip_target(
        model,
        actuator_id,
        initial_target + target_offset(args, data.time),
    )
    data.ctrl[actuator_id] = desired
    mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise RuntimeError(f"non-finite simulation state at t={data.time:.3f} s")

    if data.time - last_report >= 0.5:
        measured = float(data.qpos[qpos_address])
        print(
            f"t={data.time:6.2f} s  target={desired:+.4f} rad  "
            f"actual={measured:+.4f} rad  error={desired - measured:+.4f} rad",
            flush=True,
        )
        last_report = data.time

    if not args.headless:
        remaining = data.time - (time.monotonic() - wall_start)
        if remaining > 0.0:
            time.sleep(remaining)
    return last_report


def run(args: argparse.Namespace) -> None:
    model = build_model(args.headless)
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    actuator_id, qpos_address = actuator_for_joint(model, args.joint)
    initial_target = float(data.ctrl[actuator_id])

    print("H1 MuJoCo motion viewer")
    print(f"  joint:     {args.joint}")
    print(f"  motion:    {args.motion}")
    print(f"  amplitude: {args.amplitude:.4f} rad")
    print(f"  duration:  {args.duration:.2f} s")
    print("  NOTE: this amplitude is for simulation visualization only.")

    wall_start = time.monotonic()
    last_report = -1.0
    if args.headless:
        while data.time < args.duration:
            last_report = step_simulation(
                args, model, data, actuator_id, qpos_address,
                initial_target, wall_start, last_report,
            )
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0
        while viewer.is_running() and data.time < args.duration:
            last_report = step_simulation(
                args, model, data, actuator_id, qpos_address,
                initial_target, wall_start, last_report,
            )
            viewer.sync()
        if args.keep_open and viewer.is_running():
            print("Motion finished. Close the MuJoCo window to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
