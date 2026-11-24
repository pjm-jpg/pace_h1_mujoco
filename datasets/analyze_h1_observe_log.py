"""Validate a passive H1 LowState recording and write Experiment 1 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


STATE_MOTOR_FIELDS = ("q_real", "dq_real", "tau_est", "temperature")
COMMAND_FIELDS = ("q_cmd", "dq_cmd", "kp", "kd", "tau_ff")


def _load_manifest(raw_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = raw_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected manifest next to recording: {manifest_path}")
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _motor_matrix(data: dict[str, np.ndarray], name: str, samples: int) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Observe log has no {name} field.")
    values = np.asarray(data[name])
    if values.shape != (samples, 20):
        raise ValueError(f"{name} must have shape ({samples}, 20), got {values.shape}.")
    return values


def _finite_range(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"min": None, "max": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def analyze(
    raw_path: Path,
    output_dir: Path | None = None,
    *,
    min_rate_hz: float = 450.0,
    max_rate_hz: float = 550.0,
    max_gap_s: float = 0.020,
    max_temperature_c: float = 70.0,
    max_motor_lost_rate_hz: float = 0.5,
    motor_lost_burst_allowance: int = 10,
    make_plot: bool = True,
) -> dict[str, Any]:
    raw_path = raw_path.expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    manifest_path, manifest = _load_manifest(raw_path)
    output_dir = raw_path.parent if output_dir is None else output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(raw_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    if "t_state_ns" not in data:
        raise KeyError("Observe log has no t_state_ns field.")
    timestamps_ns = np.asarray(data["t_state_ns"], dtype=np.int64).reshape(-1)
    samples = int(timestamps_ns.size)
    if samples < 2:
        raise ValueError("Observe log must contain at least two state samples.")
    motor_arrays = {name: _motor_matrix(data, name, samples) for name in STATE_MOTOR_FIELDS}
    # Schema v1 mislabeled GO MotorState.lost as motor_error. It is a
    # cumulative communication-loss counter, so retain read compatibility.
    motor_lost_key = "motor_lost" if "motor_lost" in data else "motor_error"
    motor_arrays["motor_lost"] = _motor_matrix(data, motor_lost_key, samples)

    intervals_ns = np.diff(timestamps_ns)
    non_increasing = int(np.count_nonzero(intervals_ns <= 0))
    positive_intervals_s = intervals_ns[intervals_ns > 0].astype(np.float64) * 1e-9
    if not positive_intervals_s.size:
        raise ValueError("Observe log has no positive timestamp intervals.")
    duration_s = float((timestamps_ns[-1] - timestamps_ns[0]) * 1e-9)
    median_interval_s = float(np.median(positive_intervals_s))
    median_rate_hz = float(1.0 / median_interval_s)
    largest_gap_s = float(np.max(positive_intervals_s))
    p99_gap_s = float(np.percentile(positive_intervals_s, 99.0))
    if "state_seq" not in data:
        raise KeyError("Observe log has no state_seq field.")
    state_seq = np.asarray(data["state_seq"], dtype=np.uint64).reshape(-1)
    if state_seq.shape != (samples,):
        raise ValueError(f"state_seq must have shape ({samples},), got {state_seq.shape}.")
    state_sequence_gaps = int(np.count_nonzero(np.diff(state_seq) != 1))

    manifest_command_available = manifest.get("command_available")
    if manifest_command_available is None:
        manifest_command_available = manifest.get("experiment", {}).get("command_available")
    raw_command_available = None
    if "command_available" in data and np.asarray(data["command_available"]).size == 1:
        raw_command_available = bool(np.asarray(data["command_available"]).reshape(-1)[0])
    cmd_seq = np.asarray(data.get("cmd_seq", np.ones(samples)), dtype=np.uint64).reshape(-1)
    t_cmd_ns = np.asarray(data.get("t_cmd_ns", np.ones(samples)), dtype=np.int64).reshape(-1)
    commands_zero = bool(cmd_seq.shape == (samples,) and t_cmd_ns.shape == (samples,)
                         and np.all(cmd_seq == 0) and np.all(t_cmd_ns == 0))
    commands_marked_unavailable = all(
        name in data
        and np.asarray(data[name]).shape == (samples, 20)
        and np.isnan(np.asarray(data[name], dtype=np.float64)).all()
        for name in COMMAND_FIELDS
    )

    state_nonfinite_counts = {
        name: int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        for name, values in motor_arrays.items()
        if name != "motor_lost"
    }
    for name, width in (("imu_quat", 4), ("imu_gyro", 3), ("imu_accel", 3)):
        if name not in data or np.asarray(data[name]).shape != (samples, width):
            raise ValueError(f"{name} must have shape ({samples}, {width}).")
        state_nonfinite_counts[name] = int(
            np.size(data[name]) - np.count_nonzero(np.isfinite(data[name]))
        )

    temperature = np.asarray(motor_arrays["temperature"], dtype=np.float64)
    motor_lost = np.asarray(motor_arrays["motor_lost"], dtype=np.uint32)
    valid_motor_ids = np.asarray(config.H1_JOINT_TO_MOTOR, dtype=np.int64)
    valid_lost = motor_lost[:, valid_motor_ids]
    valid_lost_changes = np.count_nonzero(np.diff(valid_lost, axis=0) != 0, axis=0)
    valid_lost_increase = valid_lost[-1].astype(np.int64) - valid_lost[0].astype(np.int64)
    motor_lost_allowance = motor_lost_burst_allowance + max_motor_lost_rate_hz * duration_s
    max_valid_temperature = float(np.max(temperature[:, valid_motor_ids]))
    malformed = int(manifest.get("stats", {}).get("malformed", -1))
    dropped = int(manifest.get("stats", {}).get("dropped", -1))
    experiment_metadata = manifest.get("experiment", {})
    publish_lowcmd = experiment_metadata.get("publish_lowcmd")
    expected_duration_s = experiment_metadata.get("observe_duration_s")

    joint_stats: dict[str, Any] = {}
    for joint_name, motor_id in zip(
        config.H1_JOINT_NAMES, config.H1_JOINT_TO_MOTOR, strict=True
    ):
        joint_stats[joint_name] = {
            "motor_id": int(motor_id),
            "q_real_rad": _finite_range(motor_arrays["q_real"][:, motor_id]),
            "dq_real_rad_s": _finite_range(motor_arrays["dq_real"][:, motor_id]),
            "tau_est": _finite_range(motor_arrays["tau_est"][:, motor_id]),
            "temperature_c": _finite_range(temperature[:, motor_id]),
            "motor_lost_start": int(motor_lost[0, motor_id]),
            "motor_lost_end": int(motor_lost[-1, motor_id]),
            "motor_lost_changes": int(np.count_nonzero(np.diff(motor_lost[:, motor_id]) != 0)),
        }

    motor_stats = {
        str(motor_id): {
            "q_real_rad": _finite_range(motor_arrays["q_real"][:, motor_id]),
            "dq_real_rad_s": _finite_range(motor_arrays["dq_real"][:, motor_id]),
            "tau_est": _finite_range(motor_arrays["tau_est"][:, motor_id]),
            "temperature_c": _finite_range(temperature[:, motor_id]),
            "motor_lost_start": int(motor_lost[0, motor_id]),
            "motor_lost_end": int(motor_lost[-1, motor_id]),
            "motor_lost_changes": int(np.count_nonzero(np.diff(motor_lost[:, motor_id]) != 0)),
        }
        for motor_id in range(20)
    }

    imu_quat = np.asarray(data["imu_quat"], dtype=np.float64)
    quaternion_norm = np.linalg.norm(imu_quat, axis=1)
    checks = [
        _check("manifest_completed", manifest.get("result") == "completed", str(manifest.get("result"))),
        _check(
            "passive_mode_marked",
            manifest_command_available is False and raw_command_available is False,
            f"manifest={manifest_command_available}, raw={raw_command_available}",
        ),
        _check(
            "no_commands_recorded",
            commands_zero and commands_marked_unavailable
            and int(manifest.get("stats", {}).get("commands", -1)) == 0
            and publish_lowcmd is False,
            "publish_lowcmd=false, cmd_seq/t_cmd_ns are zero, command arrays are NaN",
        ),
        _check("twenty_motor_slots", all(v.shape == (samples, 20) for v in motor_arrays.values()), str(samples)),
        _check(
            "mapping_is_h1_19dof",
            manifest.get("num_motors") == 20
            and manifest.get("num_joints") == 19
            and manifest.get("joint_names") == list(config.H1_JOINT_NAMES)
            and manifest.get("joint_to_motor") == list(config.H1_JOINT_TO_MOTOR)
            and manifest.get("unused_motor_id") == config.H1_UNUSED_MOTOR_ID
            and len(set(config.H1_JOINT_TO_MOTOR)) == 19
            and config.H1_UNUSED_MOTOR_ID not in config.H1_JOINT_TO_MOTOR,
            f"unused motor ID={config.H1_UNUSED_MOTOR_ID}",
        ),
        _check("timestamps_strictly_increasing", non_increasing == 0, f"violations={non_increasing}"),
        _check("state_sequence_contiguous", state_sequence_gaps == 0, f"gaps={state_sequence_gaps}"),
        _check(
            "duration_matches_plan",
            expected_duration_s is None
            or abs(duration_s - float(expected_duration_s))
            <= max(0.1, 0.02 * float(expected_duration_s)),
            f"actual={duration_s:.3f} s, planned={expected_duration_s}",
        ),
        _check(
            "median_rate_in_range",
            min_rate_hz <= median_rate_hz <= max_rate_hz,
            f"{median_rate_hz:.3f} Hz (expected {min_rate_hz:g}-{max_rate_hz:g})",
        ),
        _check("largest_gap_within_limit", largest_gap_s <= max_gap_s, f"{largest_gap_s:.6f} s"),
        _check("no_malformed_samples", malformed == 0, str(malformed)),
        _check("no_dropped_samples", dropped == 0, str(dropped)),
        _check("state_values_finite", all(v == 0 for v in state_nonfinite_counts.values()), str(state_nonfinite_counts)),
        _check(
            "valid_motor_lost_growth_within_limit",
            bool(np.all(valid_lost_increase >= 0))
            and bool(np.all(valid_lost_increase <= motor_lost_allowance)),
            f"increase_by_joint={valid_lost_increase.tolist()}, allowance={motor_lost_allowance:.3f}",
        ),
        _check(
            "temperature_within_limit",
            max_valid_temperature <= max_temperature_c,
            f"max={max_valid_temperature:.1f} C, limit={max_temperature_c:.1f} C",
        ),
        _check(
            "imu_quaternion_norm_plausible",
            bool(np.all(np.isfinite(quaternion_norm)))
            and float(np.min(quaternion_norm)) >= 0.9
            and float(np.max(quaternion_norm)) <= 1.1,
            f"min={np.min(quaternion_norm):.6f}, max={np.max(quaternion_norm):.6f}",
        ),
    ]

    report: dict[str, Any] = {
        "experiment": "H1 DDS passive LowState observation",
        "source_raw": str(raw_path),
        "source_manifest": str(manifest_path),
        "overall_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "timing": {
            "samples": samples,
            "duration_s": duration_s,
            "median_interval_s": median_interval_s,
            "median_rate_hz": median_rate_hz,
            "p99_gap_s": p99_gap_s,
            "largest_gap_s": largest_gap_s,
            "non_increasing_timestamps": non_increasing,
            "state_sequence_gaps": state_sequence_gaps,
        },
        "recorder": {
            "malformed": malformed,
            "dropped": dropped,
            "commands": int(manifest.get("stats", {}).get("commands", -1)),
            "manifest_command_available": manifest_command_available,
            "raw_command_available": raw_command_available,
            "command_arrays_all_nan": commands_marked_unavailable,
        },
        "motor_lost_policy": {
            "rate_hz": max_motor_lost_rate_hz,
            "burst_allowance": motor_lost_burst_allowance,
            "run_allowance": motor_lost_allowance,
            "increase_by_joint": valid_lost_increase.tolist(),
            "change_events_by_joint": valid_lost_changes.tolist(),
        },
        "state_nonfinite_counts": state_nonfinite_counts,
        "imu": {
            "quaternion_norm_min": float(np.min(quaternion_norm)),
            "quaternion_norm_median": float(np.median(quaternion_norm)),
            "quaternion_norm_max": float(np.max(quaternion_norm)),
        },
        "unused_motor_id_9": motor_stats[str(config.H1_UNUSED_MOTOR_ID)],
        "joint_stats": joint_stats,
        "motor_stats": motor_stats,
    }
    report_path = output_dir / "state_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) * 1e-9
        joint_q = motor_arrays["q_real"][:, valid_motor_ids]
        joint_dq = motor_arrays["dq_real"][:, valid_motor_ids]
        joint_tau = motor_arrays["tau_est"][:, valid_motor_ids]
        figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
        axes[0, 0].plot(time_s[1:], intervals_ns.astype(np.float64) * 1e-6, linewidth=0.8)
        axes[0, 0].axhline(max_gap_s * 1e3, color="tab:red", linestyle="--")
        axes[0, 0].set(title="LowState intervals", ylabel="dt [ms]")
        axes[0, 1].plot(time_s, joint_q, linewidth=0.7)
        axes[0, 1].set(title="19-joint positions", ylabel="q [rad]")
        axes[1, 0].plot(time_s, joint_dq, linewidth=0.7)
        axes[1, 0].set(title="19-joint velocities", ylabel="dq [rad/s]")
        axes[1, 1].plot(time_s, joint_tau, linewidth=0.7)
        axes[1, 1].set(title="19-joint estimated torques", ylabel="tau_est")
        axes[2, 0].plot(time_s, temperature, linewidth=0.7)
        axes[2, 0].axhline(max_temperature_c, color="tab:red", linestyle="--")
        axes[2, 0].set(title="20 physical motor temperatures", ylabel="temperature [C]", xlabel="time [s]")
        axes[2, 1].plot(time_s, quaternion_norm, color="tab:purple")
        axes[2, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axes[2, 1].set(title="IMU quaternion norm", ylabel="norm", xlabel="time [s]")
        for axis in axes.flat:
            axis.grid(True, alpha=0.3)
        figure.suptitle(f"H1 passive LowState check - {'PASS' if report['overall_pass'] else 'FAIL'}")
        figure.tight_layout()
        figure.savefig(output_dir / "state_overview.png", dpi=160)
        plt.close(figure)

    print(f"Experiment 1 check: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"Samples: {samples}, duration: {duration_s:.3f} s, median rate: {median_rate_hz:.3f} Hz")
    print(f"Largest gap: {largest_gap_s * 1e3:.3f} ms, p99 gap: {p99_gap_s * 1e3:.3f} ms")
    print(f"Report: {report_path}")
    if make_plot:
        print(f"Plot: {output_dir / 'state_overview.png'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="Passive Holosoma raw.npz recording.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-rate-hz", type=float, default=450.0)
    parser.add_argument("--max-rate-hz", type=float, default=550.0)
    parser.add_argument("--max-gap-ms", type=float, default=20.0)
    parser.add_argument("--max-temperature-c", type=float, default=70.0)
    parser.add_argument("--max-motor-lost-rate-hz", type=float, default=0.5)
    parser.add_argument("--motor-lost-burst-allowance", type=int, default=10)
    args = parser.parse_args()
    if args.min_rate_hz <= 0 or args.max_rate_hz <= args.min_rate_hz:
        parser.error("rate limits must satisfy 0 < min < max")
    if args.max_gap_ms <= 0 or args.max_temperature_c <= 0:
        parser.error("gap and temperature limits must be positive")
    if args.max_motor_lost_rate_hz < 0 or args.motor_lost_burst_allowance < 0:
        parser.error("motor lost rate and burst allowance must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    report = analyze(
        args.raw,
        args.output_dir,
        min_rate_hz=args.min_rate_hz,
        max_rate_hz=args.max_rate_hz,
        max_gap_s=args.max_gap_ms * 1e-3,
        max_temperature_c=args.max_temperature_c,
        max_motor_lost_rate_hz=args.max_motor_lost_rate_hz,
        motor_lost_burst_allowance=args.motor_lost_burst_allowance,
    )
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
