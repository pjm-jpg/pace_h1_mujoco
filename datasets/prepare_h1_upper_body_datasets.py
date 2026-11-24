"""Prepare the frozen H1 upper-body and torso SysID datasets.

The raw run selection is intentionally explicit.  This keeps aborted prepares,
passive observations, superseded pilot runs, and any future recordings out of
the official fitting set unless this file is reviewed and updated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.convert_h1_dds_log import convert  # noqa: E402


RAW_ROOT = REPO_ROOT / "datasets" / "h1" / "raw"
REAL_ROOT = REPO_ROOT / "datasets" / "h1" / "real"


@dataclass(frozen=True)
class OfficialRuns:
    joint_key: str
    joint_name: str
    step_run: str
    multisine_run: str
    sine_run: str
    step_amplitude_rad: float


OFFICIAL_RUNS = (
    OfficialRuns("left_elbow", "left_elbow_joint", "20261117_195340_left_elbow_joint_step", "20261117_195602_left_elbow_joint_multi-sine", "20261117_200009_left_elbow_joint_sine", 0.02),
    OfficialRuns("right_elbow", "right_elbow_joint", "20261117_200121_right_elbow_joint_step", "20261117_200302_right_elbow_joint_multi-sine", "20261117_200434_right_elbow_joint_sine", 0.02),
    OfficialRuns("left_shoulder_pitch", "left_shoulder_pitch_joint", "20261117_200552_left_shoulder_pitch_joint_step", "20261117_200731_left_shoulder_pitch_joint_multi-sine", "20261117_200924_left_shoulder_pitch_joint_sine", 0.02),
    OfficialRuns("right_shoulder_pitch", "right_shoulder_pitch_joint", "20261117_201118_right_shoulder_pitch_joint_step", "20261117_201321_right_shoulder_pitch_joint_multi-sine", "20261117_201508_right_shoulder_pitch_joint_sine", 0.02),
    OfficialRuns("left_shoulder_roll", "left_shoulder_roll_joint", "20261117_201650_left_shoulder_roll_joint_step", "20261117_201833_left_shoulder_roll_joint_multi-sine", "20261117_202003_left_shoulder_roll_joint_sine", 0.02),
    OfficialRuns("right_shoulder_roll", "right_shoulder_roll_joint", "20261117_202221_right_shoulder_roll_joint_step", "20261117_202332_right_shoulder_roll_joint_multi-sine", "20261117_202545_right_shoulder_roll_joint_sine", 0.02),
    OfficialRuns("left_shoulder_yaw", "left_shoulder_yaw_joint", "20261117_202707_left_shoulder_yaw_joint_step", "20261117_202826_left_shoulder_yaw_joint_multi-sine", "20261117_203014_left_shoulder_yaw_joint_sine", 0.04),
    OfficialRuns("right_shoulder_yaw", "right_shoulder_yaw_joint", "20261117_203144_right_shoulder_yaw_joint_step", "20261117_203253_right_shoulder_yaw_joint_multi-sine", "20261117_203442_right_shoulder_yaw_joint_sine", 0.04),
    OfficialRuns("torso", "torso_joint", "20261117_203609_torso_joint_step", "20261117_203741_torso_joint_multi-sine", "20261117_203947_torso_joint_sine", 0.03),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REAL_ROOT / "upper_body_official_datasets.json",
    )
    parser.add_argument(
        "--max-gap-factor",
        type=float,
        default=15.0,
        help="Conversion guard; isolated gaps remain recorded in raw manifests.",
    )
    return parser.parse_args()


def _relative(path: Path | str) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _validate_raw_run(run_name: str, joint_name: str, kind: str) -> tuple[Path, dict]:
    run_dir = RAW_ROOT / run_name
    raw_path = run_dir / "raw.npz"
    manifest_path = run_dir / "manifest.json"
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete raw run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    if manifest.get("result") != "completed":
        raise ValueError(f"Raw run was not completed: {run_dir}")
    if manifest.get("command_available") is not True:
        raise ValueError(f"Raw run has no command stream: {run_dir}")
    if experiment.get("joint") != joint_name or experiment.get("kind") != kind:
        raise ValueError(
            f"Raw run metadata mismatch in {run_dir}: expected {joint_name}/{kind}, "
            f"got {experiment.get('joint')}/{experiment.get('kind')}"
        )
    stats = manifest.get("stats", {})
    if stats.get("dropped") != 0 or stats.get("malformed") != 0:
        raise ValueError(f"Raw run has dropped or malformed samples: {run_dir}")
    return raw_path, manifest


def _convert_one(
    raw_path: Path,
    output: Path,
    start_s: float,
    end_s: float,
    max_gap_factor: float,
) -> dict:
    args = argparse.Namespace(
        input=raw_path,
        output=output,
        timestamp_unit="auto",
        frequency=500.0,
        max_gap_factor=max_gap_factor,
        allow_varying_gains=False,
        start_time_s=start_s,
        end_time_s=end_s,
    )
    convert(args)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    times = np.asarray(payload["time"], dtype=float)
    return {
        "dataset": _relative(output),
        "source_raw": _relative(payload["source_path"]),
        "samples": int(times.size),
        "duration_s": float(times[-1] - times[0]),
        "requested_crop_range_s": list(payload["requested_crop_range_s"]),
        "actual_crop_range_s": list(payload["actual_crop_range_s"]),
    }


def main() -> None:
    args = parse_args()
    entries: list[dict] = []
    diagnostics: list[dict] = []
    raw_runs: list[dict] = []
    reference_joint_names = None
    reference_joint_to_motor = None
    reference_kp = None
    reference_kd = None

    for runs in OFFICIAL_RUNS:
        step_raw, step_manifest = _validate_raw_run(runs.step_run, runs.joint_name, "step")
        multi_raw, multi_manifest = _validate_raw_run(runs.multisine_run, runs.joint_name, "multi-sine")
        sine_raw, sine_manifest = _validate_raw_run(runs.sine_run, runs.joint_name, "sine")
        raw_runs.extend(
            {
                "joint_key": runs.joint_key,
                "kind": kind,
                "raw": _relative(raw),
                "stats": manifest["stats"],
            }
            for kind, raw, manifest in (
                ("step", step_raw, step_manifest),
                ("multi-sine", multi_raw, multi_manifest),
                ("sine", sine_raw, sine_manifest),
            )
        )

        output_dir = REAL_ROOT / runs.joint_key
        official_specs = (
            ("fit", "fit_sine_0p5hz.pt", multi_raw, 8.0, 22.0),
            ("fit", "fit_sine_2hz.pt", multi_raw, 32.0, 46.0),
            ("validation", "validation_sine_1hz.pt", sine_raw, 8.0, 22.0),
            (
                "validation",
                f"validation_step_pm{runs.step_amplitude_rad:.2f}".replace(".", "p") + ".pt",
                step_raw,
                8.0,
                24.0,
            ),
        )
        for use, name, raw_path, start_s, end_s in official_specs:
            converted = _convert_one(raw_path, output_dir / name, start_s, end_s, args.max_gap_factor)
            entries.append({"joint_key": runs.joint_key, "joint_name": runs.joint_name, "use": use, **converted})

        diagnostic = _convert_one(
            multi_raw,
            output_dir / "diagnostic_multisine_1hz_internal.pt",
            20.0,
            34.0,
            args.max_gap_factor,
        )
        diagnostics.append(
            {
                "joint_key": runs.joint_key,
                "joint_name": runs.joint_name,
                "use": "diagnostic-only",
                "reason": "Same raw multi-sine recording as the fit segments; not independent validation.",
                **diagnostic,
            }
        )

        payload = torch.load(REPO_ROOT / entries[-1]["dataset"], map_location="cpu", weights_only=False)
        joint_names = list(payload["joint_names"])
        joint_to_motor = list(payload["joint_to_motor"])
        kp = np.asarray(payload["kp"], dtype=float)
        kd = np.asarray(payload["kd"], dtype=float)
        if reference_joint_names is None:
            reference_joint_names = joint_names
            reference_joint_to_motor = joint_to_motor
            reference_kp = kp
            reference_kd = kd
        if joint_names != reference_joint_names or joint_to_motor != reference_joint_to_motor:
            raise ValueError(f"Joint mapping differs for {runs.joint_name}")
        if not np.array_equal(kp, reference_kp) or not np.array_equal(kd, reference_kd):
            raise ValueError(f"Recorded PD gains differ for {runs.joint_name}")

    if len(entries) != 36 or len(diagnostics) != 9:
        raise AssertionError("Expected 36 official datasets and 9 internal diagnostics")
    assert reference_joint_names is not None
    assert reference_joint_to_motor is not None
    assert reference_kp is not None
    assert reference_kd is not None
    manifest = {
        "schema": "h1-upper-body-official-datasets-v1",
        "dataset_count": len(entries),
        "diagnostic_dataset_count": len(diagnostics),
        "joint_keys": [runs.joint_key for runs in OFFICIAL_RUNS],
        "selection_policy": (
            "Only the 36 datasets explicitly listed in datasets are official. "
            "The 0.5 and 2 Hz segments from each multi-sine run are fitting data. "
            "The separately recorded 1 Hz sine and the step run are independent "
            "validation data. Internal 1 Hz multi-sine segments are diagnostic only. "
            "Passive observations, aborted prepares, and superseded pilot runs are excluded."
        ),
        "crop_policy_s": {
            "fit_multisine_0p5hz": [8.0, 22.0],
            "diagnostic_multisine_1hz": [20.0, 34.0],
            "fit_multisine_2hz": [32.0, 46.0],
            "independent_sine_1hz": [8.0, 22.0],
            "step": [8.0, 24.0],
        },
        "joint_names": reference_joint_names,
        "joint_to_motor": reference_joint_to_motor,
        "unused_motor_id": 9,
        "kp": reference_kp.tolist(),
        "kd": reference_kd.tolist(),
        "raw_runs": raw_runs,
        "datasets": entries,
        "diagnostic_datasets": diagnostics,
    }
    output = args.manifest.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(entries)} official datasets and {len(diagnostics)} diagnostics in {output}")


if __name__ == "__main__":
    main()
