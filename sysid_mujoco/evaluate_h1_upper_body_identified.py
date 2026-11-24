"""Evaluate the combined H1 full-body calibration on all official datasets."""

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

from sysid_mujoco.common import load_processed_dataset  # noqa: E402
from sysid_mujoco.eval_fit import (  # noqa: E402
    _build_eval_model,
    plot_joint_trajectories,
    simulate_open_loop,
)


DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "h1" / "real" / "upper_body_official_datasets.json"
DEFAULT_BASE_MODEL = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml"
DEFAULT_IDENTIFIED_MODEL = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_full_body.xml"
DEFAULT_OUTPUT = REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "upper_body_identified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model-xml", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--identified-model-xml", type=Path, default=DEFAULT_IDENTIFIED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _errors(measured: np.ndarray, simulated: np.ndarray) -> tuple[float, float]:
    error = simulated - measured
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    base_model_xml = args.base_model_xml.expanduser().resolve()
    identified_model_xml = args.identified_model_xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["datasets"]
    if len(rows) != 36:
        raise ValueError(f"Expected 36 official datasets, found {len(rows)}")
    first_dataset = _repo_path(rows[0]["dataset"])
    baseline_model, baseline_fixed_xml = _build_eval_model(
        "h1", first_dataset, source_model_xml=base_model_xml
    )
    identified_model, identified_fixed_xml = _build_eval_model(
        "h1", first_dataset, source_model_xml=identified_model_xml
    )
    if baseline_model.nq != identified_model.nq or baseline_model.nu != identified_model.nu:
        raise ValueError("Baseline and identified model dimensions differ")

    results = []
    for index, entry in enumerate(rows, start=1):
        dataset_path = _repo_path(entry["dataset"])
        baseline_trajectory = load_processed_dataset(
            dataset_path, baseline_model, mujoco, actuator_mode="general"
        )
        identified_trajectory = load_processed_dataset(
            dataset_path, identified_model, mujoco, actuator_mode="general"
        )
        baseline_qpos, _ = simulate_open_loop(baseline_model, baseline_trajectory)
        identified_qpos, _ = simulate_open_loop(identified_model, identified_trajectory)
        if not np.all(np.isfinite(identified_qpos)):
            raise ValueError(f"Non-finite identified replay: {dataset_path}")
        joint_name = entry["joint_name"]
        joint_index = baseline_trajectory.joint_names.index(joint_name)
        measured = baseline_trajectory.measured_qpos[:, joint_index]
        desired = baseline_trajectory.desired_qpos[:, joint_index]
        baseline = baseline_qpos[:, joint_index]
        identified = identified_qpos[:, joint_index]
        baseline_rmse, baseline_mae = _errors(measured, baseline)
        identified_rmse, identified_mae = _errors(measured, identified)
        improvement = 100.0 * (baseline_rmse - identified_rmse) / baseline_rmse
        result = {
            "joint_key": entry["joint_key"],
            "joint_name": joint_name,
            "use": entry["use"],
            "dataset": entry["dataset"],
            "baseline_rmse_rad": baseline_rmse,
            "identified_rmse_rad": identified_rmse,
            "rmse_improvement_percent": improvement,
            "baseline_mae_rad": baseline_mae,
            "identified_mae_rad": identified_mae,
        }
        results.append(result)
        plot_dir = output_dir / entry["joint_key"] / dataset_path.stem
        plot_joint_trajectories(
            times=baseline_trajectory.times,
            desired_qpos=desired[:, None],
            measured_qpos=measured[:, None],
            simulated_qpos=baseline[:, None],
            identified_qpos=identified[:, None],
            joint_names=[joint_name],
            robot="h1",
            dataset_path=dataset_path,
            output_path=plot_dir / "joint_trajectories.png",
            show=False,
        )
        (plot_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            f"[{index:02d}/{len(rows)}] {entry['joint_key']} {dataset_path.stem}: "
            f"{baseline_rmse:.6f} -> {identified_rmse:.6f} rad "
            f"({improvement:+.1f}%)"
        )

    joint_summary = {}
    for joint_key in manifest["joint_keys"]:
        joint_rows = [row for row in results if row["joint_key"] == joint_key]
        summary = {}
        for use in ("fit", "validation"):
            selected = [row for row in joint_rows if row["use"] == use]
            baseline_mean = float(np.mean([row["baseline_rmse_rad"] for row in selected]))
            identified_mean = float(np.mean([row["identified_rmse_rad"] for row in selected]))
            summary[f"{use}_baseline_rmse_mean_rad"] = baseline_mean
            summary[f"{use}_identified_rmse_mean_rad"] = identified_mean
            summary[f"{use}_rmse_improvement_percent"] = (
                100.0 * (baseline_mean - identified_mean) / baseline_mean
            )
            summary[f"{use}_all_improved"] = all(
                row["identified_rmse_rad"] < row["baseline_rmse_rad"] for row in selected
            )
        joint_summary[joint_key] = summary

    payload = {
        "schema": "h1-upper-body-identified-evaluation-v1",
        "manifest": str(manifest_path),
        "baseline_source_model_xml": str(base_model_xml),
        "identified_source_model_xml": str(identified_model_xml),
        "baseline_fixed_model_xml": str(baseline_fixed_xml),
        "identified_fixed_model_xml": str(identified_fixed_xml),
        "dataset_count": len(results),
        "joint_summary": joint_summary,
        "datasets": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Evaluation summary written to {output_dir}")


if __name__ == "__main__":
    main()
