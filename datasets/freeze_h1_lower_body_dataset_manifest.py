"""Freeze the processed datasets used by the H1 lower-body calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = REPO_ROOT / "datasets" / "h1" / "real"
JOINT_KEYS = (
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REAL_ROOT / "lower_body_official_datasets.json",
    )
    return parser.parse_args()


def _relative(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _dataset_use(name: str) -> str:
    if name.startswith("fit_"):
        return "fit"
    if name.startswith("validation_"):
        return "validation"
    raise ValueError(f"Unexpected official dataset name: {name}")


def main() -> None:
    args = parse_args()
    entries = []
    reference_joint_names: list[str] | None = None
    reference_joint_to_motor: list[int] | None = None
    reference_kp: np.ndarray | None = None
    reference_kd: np.ndarray | None = None

    for joint_key in JOINT_KEYS:
        for path in sorted((REAL_ROOT / joint_key).glob("*.pt")):
            if path.name.startswith("hold_"):
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            joint_names = list(payload["joint_names"])
            joint_to_motor = np.asarray(payload["joint_to_motor"], dtype=int).tolist()
            kp = np.asarray(payload["kp"], dtype=float)
            kd = np.asarray(payload["kd"], dtype=float)
            if reference_joint_names is None:
                reference_joint_names = joint_names
                reference_joint_to_motor = joint_to_motor
                reference_kp = kp
                reference_kd = kd
            if joint_names != reference_joint_names:
                raise ValueError(f"Joint order differs in {path}")
            if joint_to_motor != reference_joint_to_motor:
                raise ValueError(f"DDS motor mapping differs in {path}")
            if not np.array_equal(kp, reference_kp) or not np.array_equal(kd, reference_kd):
                raise ValueError(f"PD gains differ in {path}")

            time = np.asarray(payload["time"], dtype=float)
            entries.append(
                {
                    "joint_key": joint_key,
                    "use": _dataset_use(path.stem),
                    "dataset": _relative(path),
                    "source_raw": _relative(payload["source_path"]),
                    "samples": int(time.size),
                    "duration_s": float(time[-1] - time[0]),
                    "requested_crop_range_s": list(payload["requested_crop_range_s"]),
                    "actual_crop_range_s": list(payload["actual_crop_range_s"]),
                }
            )

    if len(entries) != 41:
        raise ValueError(f"Expected 41 official datasets, found {len(entries)}")
    assert reference_joint_names is not None
    assert reference_joint_to_motor is not None
    assert reference_kp is not None
    assert reference_kd is not None
    manifest = {
        "schema": "h1-lower-body-official-datasets-v1",
        "dataset_count": len(entries),
        "joint_keys": list(JOINT_KEYS),
        "selection_policy": (
            "Only the datasets explicitly listed below are official. Hold-only "
            "diagnostics, aborted runs, obsolete arm poses, and all other raw "
            "logs are excluded from fitting and validation."
        ),
        "joint_names": reference_joint_names,
        "joint_to_motor": reference_joint_to_motor,
        "unused_motor_id": 9,
        "kp": reference_kp.tolist(),
        "kd": reference_kd.tolist(),
        "datasets": entries,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(entries)} datasets in {output}")


if __name__ == "__main__":
    main()
