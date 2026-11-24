"""Run Experiment 8: offline previews for all nine H1 upper-body joints.

The script reuses the exact command-plan implementation imported from the
Holosoma collector.  DDS is never initialized.  Each joint receives a
bidirectional step and a 0.5/1/2 Hz segmented multi-sine preview.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.preview_h1_real_experiment import run


PROFILES = {
    "left_elbow_joint": {"step": 0.05, "multi": (0.05, 0.04, 0.025)},
    "right_elbow_joint": {"step": 0.05, "multi": (0.05, 0.04, 0.025)},
    "left_shoulder_pitch_joint": {"step": 0.05, "multi": (0.05, 0.04, 0.025)},
    "right_shoulder_pitch_joint": {"step": 0.05, "multi": (0.05, 0.04, 0.025)},
    "left_shoulder_roll_joint": {"step": 0.02, "multi": (0.03, 0.025, 0.015)},
    "right_shoulder_roll_joint": {"step": 0.02, "multi": (0.03, 0.025, 0.015)},
    "left_shoulder_yaw_joint": {"step": 0.04, "multi": (0.04, 0.035, 0.02)},
    "right_shoulder_yaw_joint": {"step": 0.04, "multi": (0.04, 0.035, 0.02)},
    "torso_joint": {"step": 0.03, "multi": (0.03, 0.025, 0.015)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml",
    )
    parser.add_argument(
        "--holosoma-root",
        type=Path,
        default=Path("/home/pjm/Desktop/holosoma"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "upper_body_experiment_08"
        ),
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        choices=tuple(PROFILES),
        default=tuple(PROFILES),
        help="Optional subset for focused diagnostics.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _preview_args(
    args: argparse.Namespace,
    joint: str,
    kind: str,
    profile: dict[str, object],
) -> argparse.Namespace:
    output_dir = args.output_root / joint / kind
    return argparse.Namespace(
        joint=joint,
        kind=kind,
        amplitude_rad=float(profile["step"]),
        repeats=1,
        step_hold_s=2.0,
        step_base_s=1.0,
        sine_duration_s=10.0,
        sine_frequency_hz=0.5,
        multi_sine_frequencies_hz=(0.5, 1.0, 2.0),
        multi_sine_amplitudes_rad=tuple(profile["multi"]),
        multi_sine_segment_duration_s=10.0,
        multi_sine_hold_s=2.0,
        chirp_duration_s=20.0,
        chirp_f0_hz=0.5,
        chirp_f1_hz=5.0,
        fade_s=0.5,
        pre_hold_s=2.0,
        post_hold_s=2.0,
        command_rate_hz=50.0,
        prepare_duration_s=6.0,
        prepare_settle_s=2.0,
        return_duration_s=6.0,
        staged_prepare=False,
        arm_clearance_pitch_rad=0.8,
        max_following_error_rad=0.12,
        final_hold_s=1.0,
        holosoma_root=args.holosoma_root,
        model_xml=args.model_xml,
        initial_state_npz=None,
        preserve_collisions=True,
        output_dir=output_dir,
        headless=True,
        keep_open=False,
    )


def _run_acceptance(metrics: dict[str, object]) -> tuple[bool, list[str]]:
    failures = []
    center = float(metrics["excitation_center_rad"])
    if not float(metrics["target_min_rad"]) < center < float(metrics["target_max_rad"]):
        failures.append("trajectory does not excite both command directions")
    if float(metrics["max_all_joint_following_error_rad"]) > 0.12:
        failures.append("all-joint following error exceeds 0.12 rad")
    if float(metrics["minimum_joint_limit_margin_rad"]) <= 0.10:
        failures.append("joint-limit margin is at most 0.10 rad")
    if int(metrics["self_contact_count"]) != 0:
        failures.append("nonadjacent self-contact detected")
    if abs(float(metrics["final_return_error_rad"])) > 0.05:
        failures.append("final return error exceeds 0.05 rad")
    if not bool(metrics["collision_check_enabled"]):
        failures.append("collision checking was disabled")
    if not bool(metrics["safety_check_pass"]):
        failures.append("preview safety check failed")
    return not failures, failures


def _save_summary_plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{str(row['joint']).replace('_joint', '').replace('_', ' ')}\n{row['kind']}"
        for row in rows
    ]
    errors = [float(row["max_all_joint_following_error_rad"]) for row in rows]
    margins = [float(row["minimum_joint_limit_margin_rad"]) for row in rows]
    velocities = [float(row["max_abs_velocity_rad_s"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["tab:blue" if row["kind"] == "step" else "tab:green" for row in rows]

    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].bar(x, errors, color=colors)
    axes[0].axhline(0.12, color="tab:red", linestyle="--", label="0.12 rad limit")
    axes[0].set_ylabel("max all-joint error [rad]")
    axes[0].legend()
    axes[1].bar(x, margins, color=colors)
    axes[1].axhline(0.10, color="tab:red", linestyle="--", label="0.10 rad minimum")
    axes[1].set_ylabel("minimum limit margin [rad]")
    axes[1].legend()
    axes[2].bar(x, velocities, color=colors)
    axes[2].set_ylabel("max velocity [rad/s]")
    axes[2].set_xticks(x, labels, rotation=55, ha="right")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("H1 Experiment 8 - upper-body command-plan simulation checks")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.model_xml = args.model_xml.expanduser().resolve()
    args.holosoma_root = args.holosoma_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_profiles = {joint: PROFILES[joint] for joint in args.joints}
    for joint, profile in selected_profiles.items():
        for kind in ("step", "multi-sine"):
            preview_args = _preview_args(args, joint, kind, profile)
            if args.verbose:
                metrics = run(preview_args)
            else:
                with redirect_stdout(io.StringIO()):
                    metrics = run(preview_args)
            accepted, failures = _run_acceptance(metrics)
            row = {
                "joint": joint,
                "kind": kind,
                "accepted": accepted,
                "failures": failures,
                "motor_id": metrics["motor_id"],
                "kp": metrics["kp"],
                "kd": metrics["kd"],
                "target_min_rad": metrics["target_min_rad"],
                "target_max_rad": metrics["target_max_rad"],
                "measured_min_rad": metrics["measured_min_rad"],
                "measured_max_rad": metrics["measured_max_rad"],
                "rmse_following_error_rad": metrics["rmse_following_error_rad"],
                "max_following_error_rad": metrics["max_following_error_rad"],
                "max_all_joint_following_error_rad": metrics[
                    "max_all_joint_following_error_rad"
                ],
                "max_abs_velocity_rad_s": metrics["max_abs_velocity_rad_s"],
                "minimum_joint_limit_margin_rad": metrics[
                    "minimum_joint_limit_margin_rad"
                ],
                "final_return_error_rad": metrics["final_return_error_rad"],
                "self_contacts": metrics["self_contacts"],
                "metrics": str(preview_args.output_dir / "preview_metrics.json"),
                "plot": str(preview_args.output_dir / "command_preview.png"),
            }
            rows.append(row)
            print(
                f"{'PASS' if accepted else 'FAIL'}  {joint:34s} {kind:10s} "
                f"error={float(row['max_all_joint_following_error_rad']):.4f} "
                f"margin={float(row['minimum_joint_limit_margin_rad']):.4f} "
                f"contacts={len(row['self_contacts'])}"
            )

    report = {
        "experiment": "H1 upper-body command-plan simulation validation",
        "simulation_only": True,
        "dds_initialized": False,
        "source_model_xml": str(args.model_xml),
        "holosoma_plan_source": str(
            args.holosoma_root
            / "src/holosoma_inference/holosoma_inference/collect_h1_sysid.py"
        ),
        "joint_count": len(selected_profiles),
        "run_count": len(rows),
        "accepted_count": sum(bool(row["accepted"]) for row in rows),
        "all_accepted": all(bool(row["accepted"]) for row in rows),
        "acceptance": {
            "max_all_joint_following_error_rad": 0.12,
            "minimum_joint_limit_margin_rad": 0.10,
            "maximum_final_return_error_rad": 0.05,
            "requires_bidirectional_command": True,
            "requires_no_nonadjacent_self_contact": True,
        },
        "profiles": selected_profiles,
        "runs": rows,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = args.output_root / "upper_body_validation_summary.png"
    _save_summary_plot(rows, plot_path)
    print(
        f"Accepted {report['accepted_count']}/{report['run_count']} runs.\n"
        f"Summary: {summary_path}\nPlot: {plot_path}"
    )
    if not report["all_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
