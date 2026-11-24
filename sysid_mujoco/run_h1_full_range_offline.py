"""Replay all official H1 project motions and identify a shared command delay.

The per-joint armature/friction/damping parameters are taken from the existing
19-joint calibrated model.  This stage estimates the remaining controller-wide
delay only on development recordings, then reports original, calibrated, and
calibrated-plus-delay results on validation and untouched blind recordings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
from sysid_mujoco.common import apply_command_delay, load_processed_dataset  # noqa: E402
from sysid_mujoco.eval_fit import _build_eval_model, simulate_open_loop  # noqa: E402


MANIFEST = REPO_ROOT / "datasets" / "h1" / "real" / "full_range_official_manifest.json"
RESULT_ROOT = REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "full_range_offline"
ORIGINAL_XML = REPO_ROOT / "robot_model" / "h1" / "h1.xml"
CALIBRATED_XML = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_full_body.xml"


def _rmse(error: np.ndarray) -> tuple[float, np.ndarray]:
    per_joint = np.sqrt(np.mean(error**2, axis=0))
    return float(np.mean(per_joint)), per_joint


def _simulate(model: mujoco.MjModel, dataset: Path, delay_ms: float = 0.0):
    trajectory = load_processed_dataset(dataset, model, mujoco, actuator_mode="general")
    if delay_ms:
        trajectory = apply_command_delay(trajectory, delay_ms * 1e-3)
    qpos, _ = simulate_open_loop(model, trajectory)
    return trajectory, qpos


def _active_indices(joint_names: list[str], active: list[str]) -> list[int]:
    missing = sorted(set(active) - set(joint_names))
    if missing:
        raise ValueError(f"Unknown active joints: {missing}")
    return [joint_names.index(name) for name in active]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runs = manifest["project_runs"]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    first_dataset = REPO_ROOT / runs[0]["dataset"]
    original_model, original_fixed_xml = _build_eval_model(
        "h1", first_dataset, source_model_xml=ORIGINAL_XML
    )
    calibrated_model, calibrated_fixed_xml = _build_eval_model(
        "h1", first_dataset, source_model_xml=CALIBRATED_XML
    )

    # Cache the calibrated no-delay trajectories used by delay selection.
    loaded: dict[str, tuple[object, np.ndarray]] = {}
    fit_runs = [run for run in runs if run["role"] == "fit"]
    delay_grid_ms = np.arange(0.0, 20.0 + 0.1, 2.0)
    delay_scores = []
    for delay_ms in delay_grid_ms:
        squared_error_sum = 0.0
        value_count = 0
        for run in fit_runs:
            dataset = REPO_ROOT / run["dataset"]
            trajectory, qpos = _simulate(calibrated_model, dataset, float(delay_ms))
            indices = _active_indices(trajectory.joint_names, run["active_joints"])
            error = qpos[:, indices] - trajectory.measured_qpos[:, indices]
            squared_error_sum += float(np.sum(error**2))
            value_count += int(error.size)
        delay_scores.append(float(np.sqrt(squared_error_sum / value_count)))
        print(f"delay={delay_ms:4.1f} ms fit RMS={delay_scores[-1]:.6f} rad")
    best_delay_ms = float(delay_grid_ms[int(np.argmin(delay_scores))])
    print(f"Selected shared delay: {best_delay_ms:.1f} ms (fit data only)")

    rows = []
    trace_payload: dict[str, dict] = {}
    joint_accumulator: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        dataset = REPO_ROOT / run["dataset"]
        trajectory, original_qpos = _simulate(original_model, dataset, 0.0)
        _, calibrated_qpos = _simulate(calibrated_model, dataset, 0.0)
        _, identified_qpos = _simulate(calibrated_model, dataset, best_delay_ms)
        indices = _active_indices(trajectory.joint_names, run["active_joints"])
        measured = trajectory.measured_qpos[:, indices]
        original_mean, original_per = _rmse(original_qpos[:, indices] - measured)
        calibrated_mean, calibrated_per = _rmse(calibrated_qpos[:, indices] - measured)
        identified_mean, identified_per = _rmse(identified_qpos[:, indices] - measured)
        row = {
            "run": run["run"],
            "role": run["role"],
            "profile": run["profile"],
            "amplitude_scale": run["amplitude_scale"],
            "speed_scale": run["speed_scale"],
            "samples": int(len(trajectory.times)),
            "original_rmse_rad": original_mean,
            "calibrated_rmse_rad": calibrated_mean,
            "identified_rmse_rad": identified_mean,
            "calibrated_improvement_pct": 100.0 * (original_mean - calibrated_mean) / original_mean,
            "identified_improvement_pct": 100.0 * (original_mean - identified_mean) / original_mean,
        }
        rows.append(row)
        print(run["run"], json.dumps(row, ensure_ascii=False))
        for local_index, joint_name in enumerate(run["active_joints"]):
            bucket = joint_accumulator.setdefault(
                joint_name, {"original": [], "calibrated": [], "identified": []}
            )
            bucket["original"].append(float(original_per[local_index]))
            bucket["calibrated"].append(float(calibrated_per[local_index]))
            bucket["identified"].append(float(identified_per[local_index]))

        # Keep full traces only for one representative record per motion family.
        representative = {
            "20261120_152849_h1_walk1_lower_body_amp0.90_speed0.50": "left_knee_joint",
            "20261120_155811_h1_h1_upper_yaw_elbow_excitation_upper_body_amp0.90_speed0.50": "left_elbow_joint",
            "20261120_161326_h1_h1_torso_large_torso_only_amp0.90_speed0.50": "torso_joint",
        }
        if run["run"] in representative:
            joint_name = representative[run["run"]]
            index = trajectory.joint_names.index(joint_name)
            trace_payload[run["run"]] = {
                "joint": joint_name,
                "time": trajectory.times.tolist(),
                "desired": trajectory.desired_qpos[:, index].tolist(),
                "real": trajectory.measured_qpos[:, index].tolist(),
                "original": original_qpos[:, index].tolist(),
                "identified": identified_qpos[:, index].tolist(),
            }

    role_summary = {}
    for role in ("fit", "validation", "blind"):
        selected = [row for row in rows if row["role"] == role]
        role_summary[role] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("original_rmse_rad", "calibrated_rmse_rad", "identified_rmse_rad")
        }
    joint_summary = {
        joint: {model: float(np.mean(values)) for model, values in models.items()}
        for joint, models in joint_accumulator.items()
    }
    result = {
        "schema": "h1-full-range-offline-identification-v1",
        "method": "existing 19-joint armature/frictionloss/damping calibration plus fit-only shared-delay grid search",
        "original_model": str(ORIGINAL_XML),
        "calibrated_model": str(CALIBRATED_XML),
        "original_fixed_model": str(original_fixed_xml),
        "calibrated_fixed_model": str(calibrated_fixed_xml),
        "delay_grid_ms": delay_grid_ms.tolist(),
        "delay_fit_rms_rad": delay_scores,
        "identified_shared_delay_ms": best_delay_ms,
        "role_summary": role_summary,
        "joint_summary": joint_summary,
        "runs": rows,
    }
    (RESULT_ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULT_ROOT / "representative_traces.json").write_text(
        json.dumps(trace_payload), encoding="utf-8"
    )
    with (RESULT_ROOT / "run_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _plot_summary(rows, delay_grid_ms, delay_scores)
    _plot_traces(trace_payload)
    print(f"Wrote offline results to {RESULT_ROOT}")


def _plot_summary(rows: list[dict], delays: np.ndarray, scores: list[float]) -> None:
    labels = [f"{row['profile']}\n{row['amplitude_scale']:.2f}@{row['speed_scale']:.2f}\n{row['role']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.26
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2.3, 1]})
    axes[0].bar(x - width, [r["original_rmse_rad"] for r in rows], width, label="Original MuJoCo")
    axes[0].bar(x, [r["calibrated_rmse_rad"] for r in rows], width, label="19-joint calibrated")
    axes[0].bar(x + width, [r["identified_rmse_rad"] for r in rows], width, label="Calibrated + delay")
    axes[0].set_ylabel("Mean active-joint RMSE [rad]")
    axes[0].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].plot(delays, scores, marker="o")
    axes[1].set_xlabel("Shared command delay [ms]")
    axes[1].set_ylabel("Fit RMS [rad]")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULT_ROOT / "offline_summary.png", dpi=180)
    plt.close(figure)


def _plot_traces(traces: dict[str, dict]) -> None:
    figure, axes = plt.subplots(len(traces), 1, figsize=(13, 9), squeeze=False)
    for axis, (run, trace) in zip(axes[:, 0], traces.items(), strict=True):
        time = np.asarray(trace["time"])
        axis.plot(time, trace["desired"], "--", color="tab:orange", label="Command", linewidth=1.0)
        axis.plot(time, trace["real"], color="black", label="Real H1", linewidth=1.3)
        axis.plot(time, trace["original"], color="tab:red", label="Original MuJoCo", linewidth=1.0)
        axis.plot(time, trace["identified"], color="tab:green", label="Calibrated MuJoCo", linewidth=1.0)
        axis.set_title(f"{trace['joint']} — {run}", fontsize=9)
        axis.set_ylabel("Position [rad]")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    figure.tight_layout()
    figure.savefig(RESULT_ROOT / "representative_three_curve_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
