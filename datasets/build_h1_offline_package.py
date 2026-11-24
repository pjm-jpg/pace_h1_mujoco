"""Freeze the official H1 data inventory and convert project-motion segments.

This script is intentionally deterministic.  It combines the reviewed small-
signal manifests with the explicitly selected large-motion recordings, assigns
fit/validation/blind roles, and classifies every raw manifest as selected,
diagnostic, superseded, or aborted.  Re-running it never selects an older run
when a reviewed replacement is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.convert_h1_dds_log import convert  # noqa: E402


RAW_ROOT = REPO_ROOT / "datasets" / "h1" / "raw"
REAL_ROOT = REPO_ROOT / "datasets" / "h1" / "real"
OUTPUT_MANIFEST = REAL_ROOT / "full_range_official_manifest.json"
PROJECT_OUTPUT = REAL_ROOT / "project_motion"

LOWER_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint", "right_hip_yaw_joint",
    "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "right_ankle_joint",
]
UPPER_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]


# Explicit selection is preferable to filename heuristics for scientific data.
PROJECT_RUNS = [
    ("20261120_152448_h1_walk1_lower_body_amp0.60_speed0.50", "fit", LOWER_JOINTS),
    ("20261120_152657_h1_walk1_lower_body_amp0.75_speed0.50", "validation", LOWER_JOINTS),
    ("20261120_152849_h1_walk1_lower_body_amp0.90_speed0.50", "validation", LOWER_JOINTS),
    ("20261120_154112_h1_sub3_largebox_003_upper_body_amp0.60_speed0.50", "fit", UPPER_JOINTS),
    ("20261120_154411_h1_sub3_largebox_003_upper_body_amp0.90_speed0.30", "validation", UPPER_JOINTS),
    ("20261120_155247_h1_h1_upper_yaw_elbow_excitation_upper_body_amp0.60_speed0.50", "fit", UPPER_JOINTS),
    ("20261120_155811_h1_h1_upper_yaw_elbow_excitation_upper_body_amp0.90_speed0.50", "validation", UPPER_JOINTS),
    ("20261120_161005_h1_h1_torso_large_torso_only_amp0.60_speed0.50", "fit", ["torso_joint"]),
    ("20261120_161326_h1_h1_torso_large_torso_only_amp0.90_speed0.50", "validation", ["torso_joint"]),
    ("20261120_161757_h1_walk1_subject1_lower_body_amp0.75_speed0.40", "blind", LOWER_JOINTS),
    ("20261120_162020_h1_sub3_largebox_003_upper_body_amp0.60_speed0.30", "blind", UPPER_JOINTS),
    # The 16:24 attempt was interrupted/stale after a collision.  This is its
    # successful same-condition replacement and is the only run selected.
    ("20261120_163031_h1_h1_upper_yaw_elbow_excitation_upper_body_amp0.60_speed0.75", "blind", UPPER_JOINTS),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _project_record(run_name: str, role: str, active_joints: list[str], convert_data: bool) -> dict:
    run_dir = RAW_ROOT / run_name
    manifest = _load(run_dir / "manifest.json")
    if manifest.get("result") != "completed":
        raise RuntimeError(f"Selected project run is not complete: {run_name}")
    experiment = manifest["experiment"]
    start_s = float(experiment["prepare_duration_s"] + experiment["prepare_settle_s"])
    end_s = start_s + float(experiment["motion_execution_duration_s"])
    output = PROJECT_OUTPUT / f"{run_name}.pt"
    if convert_data:
        args = SimpleNamespace(
            input=run_dir / "raw.npz",
            output=output,
            timestamp_unit="auto",
            frequency=500.0,
            # Selected project logs are below the 20 ms protocol limit.  A
            # factor of 12 allows those reviewed isolated gaps at a 2 ms rate.
            max_gap_factor=12.0,
            allow_varying_gains=False,
            start_time_s=start_s,
            end_time_s=end_s,
        )
        convert(args)
    return {
        "run": run_name,
        "role": role,
        "raw": _relative(run_dir / "raw.npz"),
        "dataset": _relative(output),
        "profile": experiment["profile"],
        "motion_npz": experiment["motion_npz"],
        "amplitude_scale": experiment["amplitude_scale"],
        "speed_scale": experiment["speed_scale"],
        "active_joints": active_joints,
        "crop_range_s": [start_s, end_s],
        "stats": manifest["stats"],
    }


def build(convert_data: bool = True) -> dict:
    lower = _load(REAL_ROOT / "lower_body_official_datasets.json")
    upper = _load(REAL_ROOT / "upper_body_official_datasets.json")
    small_datasets = lower["datasets"] + upper["datasets"]
    small_diagnostic = upper.get("diagnostic_datasets", [])
    selected_small_raw = {entry["source_raw"] for entry in small_datasets}
    selected_project_names = {item[0] for item in PROJECT_RUNS}

    project_runs = [
        _project_record(name, role, joints, convert_data)
        for name, role, joints in PROJECT_RUNS
    ]

    inventory = []
    for manifest_path in sorted(RAW_ROOT.glob("*/manifest.json")):
        manifest = _load(manifest_path)
        run_name = manifest_path.parent.name
        raw_relative = _relative(manifest_path.parent / "raw.npz")
        result = manifest.get("result")
        if result != "completed":
            classification = "excluded_aborted"
            reason = manifest.get("abort_reason") or "run did not complete"
        elif run_name in selected_project_names:
            classification = "selected_project_motion"
            reason = "explicitly selected complete project-motion recording"
        elif raw_relative in selected_small_raw:
            classification = "selected_small_signal"
            reason = "referenced by reviewed lower/upper official manifest"
        else:
            kind = manifest.get("experiment", {}).get("kind", "")
            if kind in {"observe", "prepare", "hold"} or "_h1_observe" in run_name or "_h1_prepare" in run_name:
                classification = "excluded_diagnostic"
                reason = "passive/prepare/hold diagnostic; not an identification trajectory"
            else:
                classification = "excluded_superseded_or_unreviewed"
                reason = "not in reviewed official list; newer/corrected run takes precedence where duplicated"
        inventory.append({
            "run": run_name,
            "result": result,
            "classification": classification,
            "reason": reason,
            "raw": raw_relative,
            "stats": manifest.get("stats", {}),
        })

    counts: dict[str, int] = {}
    for item in inventory:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    payload = {
        "schema": "h1-full-range-official-manifest-v1",
        "selection_policy": (
            "Only explicitly listed official small-signal datasets and project_runs are usable. "
            "For repeated or corrected experiments, the latest reviewed successful run is selected."
        ),
        "controller_scope": "h1_19dof_loco_real.robot gains; suspended fixed-base experiments",
        "small_signal": {
            "lower_manifest": "datasets/h1/real/lower_body_official_datasets.json",
            "upper_manifest": "datasets/h1/real/upper_body_official_datasets.json",
            "official_dataset_count": len(small_datasets),
            "diagnostic_dataset_count": len(small_diagnostic),
            "datasets": small_datasets,
        },
        "project_runs": project_runs,
        "raw_inventory_counts": counts,
        "raw_inventory": inventory,
    }
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_MANIFEST}")
    print(json.dumps({"small_datasets": len(small_datasets), "project_runs": len(project_runs), **counts}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-convert", action="store_true", help="Only rebuild the JSON inventory.")
    args = parser.parse_args()
    build(convert_data=not args.no_convert)


if __name__ == "__main__":
    main()
