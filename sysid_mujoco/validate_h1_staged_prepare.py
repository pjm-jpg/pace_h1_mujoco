"""Validate the arms-first H1 prepare route from a recorded real start pose.

This is a kinematic collision/limit check.  It is intentionally separate from
the dynamic preview because the measured boot pose has real surface contact
that the MJCF represents as several centimetres of initial penetration.
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

import config  # noqa: E402
from sysid_mujoco.preview_h1_real_experiment import (  # noqa: E402
    _initial_joint_vector_from_npz,
    _joint_ids,
    _self_contact_names,
    _set_joint_state,
    build_preview_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state-npz", type=Path, required=True)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml",
    )
    parser.add_argument(
        "--holosoma-root", type=Path, default=Path("/home/pjm/Desktop/holosoma")
    )
    parser.add_argument("--arm-clearance-pitch-rad", type=float, default=0.8)
    parser.add_argument("--prepare-duration-s", type=float, default=24.0)
    parser.add_argument("--return-duration-s", type=float, default=24.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "upper_body_experiment_09"
            / "staged_prepare_kinematic"
        ),
    )
    return parser.parse_args()


def _load_plan_symbols(holosoma_root: Path):
    source = holosoma_root.resolve() / "src" / "holosoma_inference"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from holosoma_inference.collect_h1_sysid import (  # noqa: PLC0415
        H1_SUSPENDED_PREPARE_TARGETS,
        _piecewise_smooth_target,
        _staged_prepare_route,
    )

    return (
        H1_SUSPENDED_PREPARE_TARGETS,
        _piecewise_smooth_target,
        _staged_prepare_route,
    )


def _contact_distances(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    distances: dict[str, float] = {}
    for contact in data.contact[: data.ncon]:
        labels = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body#{body_id}"
            )
            geom_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom#{geom_id}"
            )
            labels.append(f"{body_name}/{geom_name}")
        label = f"{labels[0]} <-> {labels[1]}"
        distances[label] = min(distances.get(label, float("inf")), float(contact.dist))
    return distances


def main() -> None:
    args = parse_args()
    (
        prepare_targets,
        piecewise_target,
        staged_route,
    ) = _load_plan_symbols(args.holosoma_root)
    model, model_path = build_preview_model(True, args.model_xml, True)
    data = mujoco.MjData(model)
    initial = _initial_joint_vector_from_npz(args.initial_state_npz)
    prepared = initial.copy()
    for name, value in prepare_targets.items():
        prepared[config.H1_JOINT_NAMES.index(name)] = value
    waypoints, stage_weights = staged_route(
        initial, prepared, args.arm_clearance_pitch_rad
    )

    sample_rate_hz = 100.0
    forward_times = np.linspace(
        0.0,
        args.prepare_duration_s,
        round(args.prepare_duration_s * sample_rate_hz) + 1,
    )
    return_times = np.linspace(
        0.0,
        args.return_duration_s,
        round(args.return_duration_s * sample_rate_hz) + 1,
    )[1:]
    forward = [
        piecewise_target(
            waypoints, stage_weights, float(t), args.prepare_duration_s
        )
        for t in forward_times
    ]
    reverse = [
        piecewise_target(
            tuple(reversed(waypoints)),
            tuple(reversed(stage_weights)),
            float(t),
            args.return_duration_s,
        )
        for t in return_times
    ]
    trajectory = np.asarray(forward + reverse)
    times = np.concatenate((forward_times, args.prepare_duration_s + return_times))

    _set_joint_state(model, data, initial)
    initial_contacts = set(_self_contact_names(model, data)[0])
    initial_distances = _contact_distances(model, data)
    introduced: set[str] = set()
    contact_counts = []
    minimum_distances = []
    min_limit_margin = float("inf")
    initial_limit_margin = float("inf")
    first_stage_end_s = args.prepare_duration_s * stage_weights[0] / sum(stage_weights)
    arm_contact_depths_during_lift: dict[str, list[float]] = {
        name: []
        for name in initial_contacts
        if "elbow" in name and "hip" in name
    }
    contacts_at_lift_end: set[str] = set()
    contacts_at_prepare: set[str] = set()

    for joint_index, joint_name in enumerate(config.H1_JOINT_NAMES):
        joint_id, _, _ = _joint_ids(model, joint_name)
        if bool(model.jnt_limited[joint_id]):
            lower, upper = model.jnt_range[joint_id]
            initial_limit_margin = min(
                initial_limit_margin,
                float(initial[joint_index] - lower),
                float(upper - initial[joint_index]),
            )

    for sample_index, (time_s, q) in enumerate(zip(times, trajectory, strict=True)):
        _set_joint_state(model, data, q)
        contacts = set(_self_contact_names(model, data)[0])
        distances = _contact_distances(model, data)
        introduced.update(contacts - initial_contacts)
        contact_counts.append(len(contacts))
        minimum_distances.append(min(distances.values(), default=0.0))
        if time_s <= first_stage_end_s + 1e-9:
            for name in arm_contact_depths_during_lift:
                arm_contact_depths_during_lift[name].append(
                    min(0.0, distances.get(name, 0.0))
                )
        if sample_index == len(forward) - 1:
            contacts_at_prepare = contacts.copy()
        if abs(time_s - first_stage_end_s) <= 0.5 / sample_rate_hz:
            contacts_at_lift_end = contacts.copy()
        for joint_index, joint_name in enumerate(config.H1_JOINT_NAMES):
            joint_id, _, _ = _joint_ids(model, joint_name)
            if bool(model.jnt_limited[joint_id]):
                lower, upper = model.jnt_range[joint_id]
                min_limit_margin = min(
                    min_limit_margin,
                    float(q[joint_index] - lower),
                    float(upper - q[joint_index]),
                )

    arm_contact_monotonic = True
    for depths in arm_contact_depths_during_lift.values():
        if any(next_depth + 1e-4 < depth for depth, next_depth in zip(depths, depths[1:])):
            arm_contact_monotonic = False
    arm_contacts_at_lift_end = {
        name for name in contacts_at_lift_end if "elbow" in name or "shoulder" in name
    }
    limit_not_worsened = min_limit_margin >= min(0.0, initial_limit_margin) - 0.002
    accepted = (
        not introduced
        and arm_contact_monotonic
        and not arm_contacts_at_lift_end
        and not contacts_at_prepare
        and limit_not_worsened
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "accepted": accepted,
        "simulation_only": True,
        "dds_initialized": False,
        "validation_type": "kinematic collision and joint-limit scan",
        "model_xml": str(model_path),
        "initial_state_npz": str(args.initial_state_npz.resolve()),
        "sample_rate_hz": sample_rate_hz,
        "sample_count": len(trajectory),
        "prepare_duration_s": args.prepare_duration_s,
        "return_duration_s": args.return_duration_s,
        "arm_clearance_pitch_rad": args.arm_clearance_pitch_rad,
        "initial_contacts": sorted(initial_contacts),
        "initial_contact_minimum_distances_m": initial_distances,
        "introduced_contacts": sorted(introduced),
        "arm_contact_penetration_monotonically_resolves": arm_contact_monotonic,
        "contacts_at_lift_end": sorted(contacts_at_lift_end),
        "arm_contacts_at_lift_end": sorted(arm_contacts_at_lift_end),
        "contacts_at_prepare": sorted(contacts_at_prepare),
        "initial_joint_limit_margin_rad": initial_limit_margin,
        "minimum_joint_limit_margin_rad": min_limit_margin,
        "joint_limit_not_worsened": limit_not_worsened,
    }
    report_path = args.output_dir / "kinematic_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in (
        "left_shoulder_pitch_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
    ):
        axes[0].plot(times, trajectory[:, config.H1_JOINT_NAMES.index(name)], label=name)
    axes[0].set_ylabel("target position [rad]")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(times, minimum_distances, color="tab:red", label="minimum contact distance")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("contact distance [m]")
    axes[1].set_xlabel("route time [s]")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.axvline(first_stage_end_s, color="tab:green", linestyle="--")
        axis.axvline(args.prepare_duration_s, color="tab:purple", linestyle="--")
    figure.suptitle("H1 staged arms-first prepare: kinematic collision validation")
    figure.tight_layout()
    plot_path = args.output_dir / "kinematic_validation.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
