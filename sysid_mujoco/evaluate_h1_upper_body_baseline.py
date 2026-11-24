"""Replay all official H1 upper-body datasets before upper-body calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.eval_fit import _build_eval_model, simulate_open_loop  # noqa: E402
from sysid_mujoco.common import load_processed_dataset  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "h1" / "real" / "upper_body_official_datasets.json"
DEFAULT_BASE_MODEL = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml"
DEFAULT_OUTPUT = REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "upper_body_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model-xml", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    base_model = args.base_model_xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not base_model.is_file():
        raise FileNotFoundError(base_model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])
    if len(datasets) != 36:
        raise ValueError(f"Expected 36 official datasets, found {len(datasets)}")

    first_dataset = _resolve_repo_path(datasets[0]["dataset"])
    model, generated_xml = _build_eval_model(
        "h1",
        first_dataset,
        source_model_xml=base_model,
    )
    rows: list[dict] = []
    for index, entry in enumerate(datasets, start=1):
        dataset_path = _resolve_repo_path(entry["dataset"])
        trajectory = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
            actuator_mode="general",
        )
        simulated_qpos, _ = simulate_open_loop(model, trajectory)
        if not np.all(np.isfinite(simulated_qpos)):
            raise ValueError(f"Non-finite baseline replay for {dataset_path}")
        joint_name = entry["joint_name"]
        joint_index = trajectory.joint_names.index(joint_name)
        measured = trajectory.measured_qpos[:, joint_index]
        desired = trajectory.desired_qpos[:, joint_index]
        simulated = simulated_qpos[:, joint_index]
        baseline_error = simulated - measured
        tracking_error = measured - desired
        row = {
            "joint_key": entry["joint_key"],
            "joint_name": joint_name,
            "use": entry["use"],
            "dataset": entry["dataset"],
            "samples": int(trajectory.times.size),
            "duration_s": float(trajectory.times[-1] - trajectory.times[0]),
            "baseline_rmse_rad": float(np.sqrt(np.mean(baseline_error**2))),
            "baseline_mae_rad": float(np.mean(np.abs(baseline_error))),
            "measured_tracking_rmse_rad": float(np.sqrt(np.mean(tracking_error**2))),
            "measured_tracking_max_abs_rad": float(np.max(np.abs(tracking_error))),
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(datasets)}] {entry['joint_key']} {Path(entry['dataset']).stem}: "
            f"baseline RMSE={row['baseline_rmse_rad']:.6f} rad"
        )

    joint_summary = {}
    for joint_key in manifest["joint_keys"]:
        joint_rows = [row for row in rows if row["joint_key"] == joint_key]
        fit_rows = [row for row in joint_rows if row["use"] == "fit"]
        validation_rows = [row for row in joint_rows if row["use"] == "validation"]
        joint_summary[joint_key] = {
            "fit_baseline_rmse_mean_rad": float(np.mean([row["baseline_rmse_rad"] for row in fit_rows])),
            "validation_baseline_rmse_mean_rad": float(np.mean([row["baseline_rmse_rad"] for row in validation_rows])),
            "all_baseline_rmse_mean_rad": float(np.mean([row["baseline_rmse_rad"] for row in joint_rows])),
        }

    payload = {
        "schema": "h1-upper-body-baseline-v1",
        "manifest": str(manifest_path),
        "base_model_xml": str(base_model),
        "generated_fixed_base_xml": str(generated_xml),
        "interpretation": (
            "Baseline uses the calibrated lower-body XML while retaining nominal "
            "upper-body and torso joint parameters."
        ),
        "dataset_count": len(rows),
        "joint_summary": joint_summary,
        "datasets": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Baseline summary written to {output_dir}")


if __name__ == "__main__":
    main()
