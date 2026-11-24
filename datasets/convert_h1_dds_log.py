"""Convert H1 low-level DDS recordings to this repository's SysID dataset.

The input may be JSONL, CSV, or NPZ. Signals can contain either 20 values in
physical DDS motor order (motor ID 9 is removed through ``joint2motor``) or 19
values already in H1 model joint order. Array-valued CSV/JSON fields accept
JSON arrays; CSV files may alternatively use numbered columns such as
``q_cmd_0`` ... ``q_cmd_19``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


FIELD_ALIASES = {
    "timestamp": ("timestamp", "time", "t", "stamp", "timestamp_ns"),
    "q_cmd": ("q_cmd", "cmd_q", "commanded_position", "des_dof_pos"),
    "dq_cmd": ("dq_cmd", "cmd_dq", "commanded_velocity", "des_dof_vel"),
    "q_real": ("q_real", "q", "motor_q", "position", "dof_pos"),
    "dq_real": ("dq_real", "dq", "motor_dq", "velocity", "dof_vel"),
    "kp": ("kp", "motor_kp"),
    "kd": ("kd", "motor_kd"),
    # Never alias a generic `tau` column to tau_ff: motor logs commonly use
    # `tau` for measured/estimated torque, which is not a simulation input.
    "tau_ff": ("tau_ff", "cmd_tau", "feedforward_torque", "des_dof_torque"),
    "tau_est": ("tau_est", "estimated_torque", "motor_tau", "measured_torque"),
}


def _parse_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return np.nan
    if value[0] in "[({":
        return json.loads(value.replace("(", "[").replace(")", "]"))
    return float(value)


def _read_records(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    raise ValueError("Input must be .jsonl, .ndjson, .csv, or .npz")


def _reject_passive_observe_log(
    path: Path,
    records: list[dict[str, Any]] | dict[str, Any],
) -> None:
    """Keep state-only Experiment 1 logs out of the SysID fitting path."""
    command_available: bool | None = None
    if isinstance(records, dict) and "command_available" in records:
        value = np.asarray(records["command_available"])
        if value.size == 1:
            command_available = bool(value.reshape(-1)[0])

    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_value = manifest.get("command_available")
        if manifest_value is None:
            manifest_value = manifest.get("experiment", {}).get("command_available")
        if manifest_value is not None:
            command_available = bool(manifest_value)

    if command_available is False:
        raise ValueError(
            "Passive H1 observe log has command_available=false and cannot be "
            "converted into a CMA-ES/SysID dataset. Run "
            "datasets/analyze_h1_observe_log.py on it instead."
        )

    if isinstance(records, dict) and "cmd_seq" in records:
        cmd_seq = np.asarray(records["cmd_seq"])
        if cmd_seq.size and np.all(cmd_seq == 0):
            raise ValueError(
                "DDS log contains no recorded commands (cmd_seq is always zero); "
                "it cannot be used for parameter identification."
            )


def _find_key(keys: set[str], canonical: str) -> str | None:
    return next((alias for alias in FIELD_ALIASES[canonical] if alias in keys), None)


def _numbered_columns(row: dict[str, Any], canonical: str) -> list[str]:
    for alias in FIELD_ALIASES[canonical]:
        columns = [key for key in row if key.startswith(f"{alias}_") and key[len(alias) + 1 :].isdigit()]
        if columns:
            return sorted(columns, key=lambda key: int(key.rsplit("_", 1)[1]))
    return []


def _extract(records: list[dict[str, Any]] | dict[str, Any], canonical: str) -> np.ndarray | None:
    if isinstance(records, dict):
        key = _find_key(set(records), canonical)
        return None if key is None else np.asarray(records[key], dtype=np.float64)
    if not records:
        raise ValueError("The DDS log is empty.")
    key = _find_key(set(records[0]), canonical)
    if key is not None:
        return np.asarray([_parse_cell(row[key]) for row in records], dtype=np.float64)
    columns = _numbered_columns(records[0], canonical)
    if columns:
        return np.asarray(
            [[_parse_cell(row[column]) for column in columns] for row in records],
            dtype=np.float64,
        )
    return None


def _to_joint_order(values: np.ndarray, signal_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :] if values.size in {19, 20} else values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{signal_name} must be a samples-by-motors array, got {values.shape}.")
    if values.shape[1] == len(config.H1_JOINT_NAMES):
        return values
    if values.shape[1] == 20:
        # Physical DDS slot 9 is never selected by H1_JOINT_TO_MOTOR.
        return values[:, np.asarray(config.H1_JOINT_TO_MOTOR)]
    raise ValueError(
        f"{signal_name} has {values.shape[1]} columns; expected 19 joint-order "
        "values or 20 physical DDS motor slots."
    )


def _normalize_time(raw_time: np.ndarray, unit: str) -> np.ndarray:
    raw_time = np.asarray(raw_time, dtype=np.float64).reshape(-1)
    if raw_time.size < 2 or not np.all(np.isfinite(raw_time)):
        raise ValueError("timestamp must contain at least two finite samples.")
    if np.any(np.diff(raw_time) <= 0):
        raise ValueError("timestamp must be strictly increasing.")
    if unit == "auto":
        median_dt = float(np.median(np.diff(raw_time)))
        if median_dt > 1e5:
            scale = 1e-9
        elif median_dt > 1e2:
            scale = 1e-6
        elif median_dt > 0.1:
            scale = 1e-3
        else:
            scale = 1.0
    else:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    return (raw_time - raw_time[0]) * scale


def _resample(times: np.ndarray, values: np.ndarray, new_times: np.ndarray, zoh: bool = False) -> np.ndarray:
    if zoh:
        indices = np.searchsorted(times, new_times, side="right") - 1
        return values[np.clip(indices, 0, len(times) - 1)]
    return np.column_stack([np.interp(new_times, times, values[:, index]) for index in range(values.shape[1])])


def convert(args: argparse.Namespace) -> Path:
    records = _read_records(args.input)
    _reject_passive_observe_log(args.input, records)
    raw_time = _extract(records, "timestamp")
    if raw_time is None:
        raise KeyError("DDS log has no timestamp/time field.")
    times = _normalize_time(raw_time, args.timestamp_unit)

    required = {}
    for name in ("q_cmd", "q_real", "dq_real"):
        values = _extract(records, name)
        if values is None:
            raise KeyError(f"DDS log has no {name} field.")
        required[name] = _to_joint_order(values, name)

    optional = {}
    defaults = {
        "dq_cmd": np.zeros((len(times), 19)),
        "tau_ff": np.zeros((len(times), 19)),
        "kp": np.broadcast_to(config.Kp, (len(times), 19)),
        "kd": np.broadcast_to(config.Kd, (len(times), 19)),
    }
    for name, default in defaults.items():
        values = _extract(records, name)
        optional[name] = default.copy() if values is None else _to_joint_order(values, name)
        if optional[name].shape[0] == 1:
            optional[name] = np.broadcast_to(optional[name], (len(times), 19)).copy()

    tau_est = _extract(records, "tau_est")
    if tau_est is not None:
        optional["tau_est"] = _to_joint_order(tau_est, "tau_est")

    arrays = {**required, **optional}
    for name, values in arrays.items():
        if values.shape[0] != len(times):
            raise ValueError(f"{name} has {values.shape[0]} samples but timestamp has {len(times)}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    source_start_s = float(times[0])
    source_end_s = float(times[-1])
    requested_start = getattr(args, "start_time_s", None)
    requested_end = getattr(args, "end_time_s", None)
    crop_start_s = 0.0 if requested_start is None else float(requested_start)
    crop_end_s = source_end_s if requested_end is None else float(requested_end)
    selection = (times >= crop_start_s) & (times <= crop_end_s)
    if np.count_nonzero(selection) < 2:
        raise ValueError(
            f"Time crop [{crop_start_s}, {crop_end_s}] contains fewer than two samples "
            f"inside source range [{source_start_s}, {source_end_s}]."
        )
    times = times[selection]
    arrays = {name: values[selection] for name, values in arrays.items()}
    actual_crop_start_s = float(times[0])
    actual_crop_end_s = float(times[-1])
    times = times - times[0]

    if len(times) > 2:
        median_dt = float(np.median(np.diff(times)))
        largest_gap = float(np.max(np.diff(times)))
        if largest_gap > args.max_gap_factor * median_dt:
            raise ValueError(
                f"DDS log contains a {largest_gap:.6g}s gap, more than "
                f"{args.max_gap_factor:g}x its median interval {median_dt:.6g}s."
            )

    if args.frequency:
        new_times = np.arange(0.0, times[-1] + 0.5 / args.frequency, 1.0 / args.frequency)
        # Commands are discrete DDS setpoints and remain active until the next
        # write. State channels are observations and may be linearly resampled.
        zero_order_hold = {"q_cmd", "dq_cmd", "kp", "kd", "tau_ff"}
        arrays = {
            name: _resample(times, values, new_times, zoh=name in zero_order_hold)
            for name, values in arrays.items()
        }
        times = new_times

    gain_variation = max(
        float(np.max(np.ptp(arrays["kp"], axis=0))),
        float(np.max(np.ptp(arrays["kd"], axis=0))),
    )
    if gain_variation > 1e-9 and not args.allow_varying_gains:
        raise ValueError(
            "Kp/Kd vary during the recording, but the current MuJoCo actuator model "
            "uses one gain per trajectory. Split the log into constant-gain runs or "
            "pass --allow-varying-gains to use the median gains."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "time": torch.from_numpy(times),
        "dof_pos": torch.from_numpy(arrays["q_real"]),
        "dof_vel": torch.from_numpy(arrays["dq_real"]),
        "des_dof_pos": torch.from_numpy(arrays["q_cmd"]),
        "des_dof_vel": torch.from_numpy(arrays["dq_cmd"]),
        "des_dof_torque": torch.from_numpy(arrays["tau_ff"]),
        "kp": np.median(arrays["kp"], axis=0),
        "kd": np.median(arrays["kd"], axis=0),
        "kp_history": torch.from_numpy(arrays["kp"]),
        "kd_history": torch.from_numpy(arrays["kd"]),
        "joint_names": list(config.H1_JOINT_NAMES),
        "joint_to_motor": list(config.H1_JOINT_TO_MOTOR),
        "unused_motor_id": config.H1_UNUSED_MOTOR_ID,
        "source_path": str(args.input.resolve()),
        "source_time_range_s": [source_start_s, source_end_s],
        "requested_crop_range_s": [crop_start_s, crop_end_s],
        "actual_crop_range_s": [actual_crop_start_s, actual_crop_end_s],
    }
    if "tau_est" in arrays:
        dataset["tau_est"] = torch.from_numpy(arrays["tau_est"])
    torch.save(dataset, args.output)
    print(f"Saved {len(times)} H1 samples at ~{1 / np.median(np.diff(times)):.3f} Hz to {args.output}")
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw H1 DDS log (.jsonl, .csv, or .npz).")
    parser.add_argument("output", type=Path, help="Output repository dataset (.pt).")
    parser.add_argument("--timestamp-unit", choices=("auto", "s", "ms", "us", "ns"), default="auto")
    parser.add_argument(
        "--frequency", type=float, default=config.frequency_collection,
        help="Uniform output frequency in Hz; use 0 to preserve original timestamps.",
    )
    parser.add_argument(
        "--max-gap-factor", type=float, default=5.0,
        help="Reject a log when its largest timestamp gap exceeds this multiple of the median interval.",
    )
    parser.add_argument("--allow-varying-gains", action="store_true")
    parser.add_argument(
        "--start-time-s",
        type=float,
        default=None,
        help="Optional inclusive crop start in seconds relative to the first raw sample.",
    )
    parser.add_argument(
        "--end-time-s",
        type=float,
        default=None,
        help="Optional inclusive crop end in seconds relative to the first raw sample.",
    )
    args = parser.parse_args()
    if args.frequency is not None and args.frequency < 0:
        parser.error("--frequency must be nonnegative")
    if args.max_gap_factor <= 1.0:
        parser.error("--max-gap-factor must be greater than 1")
    if args.start_time_s is not None and args.start_time_s < 0.0:
        parser.error("--start-time-s must be nonnegative")
    if args.end_time_s is not None and args.end_time_s <= 0.0:
        parser.error("--end-time-s must be positive")
    if (
        args.start_time_s is not None
        and args.end_time_s is not None
        and args.end_time_s <= args.start_time_s
    ):
        parser.error("--end-time-s must be greater than --start-time-s")
    return args


if __name__ == "__main__":
    convert(parse_args())
