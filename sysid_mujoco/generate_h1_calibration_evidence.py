"""Generate held-out evidence for the H1 calibrated joint model.

Only datasets marked ``validation`` in the frozen official manifest and
project runs marked ``validation`` or ``blind`` are used in the headline
results.  Fit recordings are intentionally excluded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import load_processed_dataset  # noqa: E402
from sysid_mujoco.eval_fit import _build_eval_model, simulate_open_loop  # noqa: E402


MANIFEST = REPO_ROOT / "datasets/h1/real/full_range_official_manifest.json"
PROJECT_SUMMARY = REPO_ROOT / "sysid_mujoco/results/h1/full_range_offline/summary.json"
ORIGINAL_XML = REPO_ROOT / "robot_model/h1/h1.xml"
CALIBRATED_XML = REPO_ROOT / "robot_model/h1/h1_calibrated_full_body.xml"
RESULT_ROOT = REPO_ROOT / "sysid_mujoco/results/h1/calibration_evidence_v1"


def joint_name(joint_key: str) -> str:
    return f"{joint_key}_joint"


def replay(dataset: Path, active_joint: str) -> tuple[dict, dict[str, np.ndarray]]:
    original_model, _ = _build_eval_model("h1", dataset, source_model_xml=ORIGINAL_XML)
    calibrated_model, _ = _build_eval_model("h1", dataset, source_model_xml=CALIBRATED_XML)

    trajectory = load_processed_dataset(dataset, original_model, mujoco, actuator_mode="general")
    calibrated_trajectory = load_processed_dataset(
        dataset, calibrated_model, mujoco, actuator_mode="general"
    )
    if trajectory.joint_names != calibrated_trajectory.joint_names:
        raise ValueError(f"Joint order mismatch for {dataset}")
    index = trajectory.joint_names.index(active_joint)
    original_q, original_dq = simulate_open_loop(original_model, trajectory)
    calibrated_q, calibrated_dq = simulate_open_loop(calibrated_model, calibrated_trajectory)

    measured_q = trajectory.measured_qpos[:, index]
    measured_dq = trajectory.measured_qvel[:, index]
    original_error = original_q[:, index] - measured_q
    calibrated_error = calibrated_q[:, index] - measured_q
    original_dq_error = original_dq[:, index] - measured_dq
    calibrated_dq_error = calibrated_dq[:, index] - measured_dq
    original_rmse = float(np.sqrt(np.mean(original_error**2)))
    calibrated_rmse = float(np.sqrt(np.mean(calibrated_error**2)))
    metrics = {
        "original_rmse_rad": original_rmse,
        "calibrated_rmse_rad": calibrated_rmse,
        "improvement_pct": 100.0 * (original_rmse - calibrated_rmse) / original_rmse,
        "original_velocity_rmse_rad_s": float(np.sqrt(np.mean(original_dq_error**2))),
        "calibrated_velocity_rmse_rad_s": float(np.sqrt(np.mean(calibrated_dq_error**2))),
        "samples": int(len(trajectory.times)),
        "duration_s": float(trajectory.times[-1] - trajectory.times[0]),
    }
    traces = {
        "time": trajectory.times,
        "command": trajectory.desired_qpos[:, index],
        "real": measured_q,
        "original": original_q[:, index],
        "calibrated": calibrated_q[:, index],
    }
    return metrics, traces


def cluster_bootstrap(rows: list[dict], samples: int = 20_000, seed: int = 20261124) -> list[float]:
    """Bootstrap the aggregate reduction by joint, preserving within-joint tests."""
    rng = np.random.default_rng(seed)
    joints = sorted({row["joint_key"] for row in rows})
    grouped = {joint: [row for row in rows if row["joint_key"] == joint] for joint in joints}
    reductions = []
    for _ in range(samples):
        selected = rng.choice(joints, size=len(joints), replace=True)
        sampled_rows = [row for joint in selected for row in grouped[joint]]
        original = np.mean([row["original_rmse_rad"] for row in sampled_rows])
        calibrated = np.mean([row["calibrated_rmse_rad"] for row in sampled_rows])
        reductions.append(float(100.0 * (original - calibrated) / original))
    return reductions


def short_project_label(row: dict) -> str:
    family = {
        "lower-body": "Lower-body",
        "upper-body": "Upper-body",
        "torso-only": "Torso",
    }[row["profile"]]
    return f"{family}\n{row['amplitude_scale']:.2f}x @ {row['speed_scale']:.2f}x\n{row['role']}"


def plot_dashboard(validation_rows: list[dict], project_rows: list[dict], summary: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    axis = axes[0, 0]
    colors = {
        "leg": "tab:blue",
        "arm": "tab:orange",
        "torso": "tab:purple",
    }
    for group in ("leg", "arm", "torso"):
        selected = [row for row in validation_rows if row["body_group"] == group]
        axis.scatter(
            [row["original_rmse_rad"] for row in selected],
            [row["calibrated_rmse_rad"] for row in selected],
            label=f"{group} (n={len(selected)})",
            color=colors[group],
            s=42,
            alpha=0.85,
        )
    limit = 1.08 * max(
        max(row["original_rmse_rad"] for row in validation_rows),
        max(row["calibrated_rmse_rad"] for row in validation_rows),
    )
    axis.plot([0, limit], [0, limit], "k--", linewidth=1, label="no improvement")
    axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Original RMSE [rad]", ylabel="Calibrated RMSE [rad]")
    axis.set_title("39 held-out small-signal tests")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    by_joint = []
    for key in sorted({row["joint_key"] for row in validation_rows}):
        selected = [row for row in validation_rows if row["joint_key"] == key]
        original = np.mean([row["original_rmse_rad"] for row in selected])
        calibrated = np.mean([row["calibrated_rmse_rad"] for row in selected])
        by_joint.append((key, 100.0 * (original - calibrated) / original))
    by_joint.sort(key=lambda pair: pair[1])
    y = np.arange(len(by_joint))
    values = [value for _, value in by_joint]
    axis.barh(y, values, color=["tab:green" if value >= 0 else "tab:red" for value in values])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, [name.replace("_", " ") for name, _ in by_joint], fontsize=7)
    axis.set_xlabel("Mean RMSE reduction [%]")
    axis.set_title("Held-out improvement by joint")
    axis.grid(axis="x", alpha=0.25)

    axis = axes[1, 0]
    x = np.arange(len(project_rows))
    width = 0.36
    axis.bar(
        x - width / 2,
        [row["original_rmse_rad"] for row in project_rows],
        width,
        label="Original",
        color="tab:red",
    )
    axis.bar(
        x + width / 2,
        [row["calibrated_rmse_rad"] for row in project_rows],
        width,
        label="Calibrated",
        color="tab:green",
    )
    axis.set_xticks(x, [short_project_label(row) for row in project_rows], rotation=28, ha="right", fontsize=7)
    axis.set_ylabel("Mean active-joint RMSE [rad]")
    axis.set_title("8 held-out project-motion runs")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    labels = ["Small signal\n(validation)", "Project motion\n(validation)", "Project motion\n(blind)"]
    original = [
        summary["small_signal_validation"]["original_mean_rmse_rad"],
        summary["project_motion"]["validation"]["original_mean_rmse_rad"],
        summary["project_motion"]["blind"]["original_mean_rmse_rad"],
    ]
    calibrated = [
        summary["small_signal_validation"]["calibrated_mean_rmse_rad"],
        summary["project_motion"]["validation"]["calibrated_mean_rmse_rad"],
        summary["project_motion"]["blind"]["calibrated_mean_rmse_rad"],
    ]
    x = np.arange(3)
    axis.bar(x - width / 2, original, width, label="Original", color="tab:red")
    axis.bar(x + width / 2, calibrated, width, label="Calibrated", color="tab:green")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean RMSE [rad]")
    axis.set_title("Aggregate held-out results")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, (before, after) in enumerate(zip(original, calibrated, strict=True)):
        reduction = 100.0 * (before - after) / before
        axis.text(index, max(before, after) * 1.03, f"{reduction:+.1f}%", ha="center", fontsize=9)

    figure.suptitle("H1 calibrated model — held-out joint-response evidence", fontsize=15)
    figure.tight_layout()
    figure.savefig(RESULT_ROOT / "heldout_evidence_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_traces(representatives: list[dict]) -> None:
    figure, axes = plt.subplots(len(representatives), 2, figsize=(15, 10), squeeze=False)
    for row_index, record in enumerate(representatives):
        trace = record["trace"]
        time = trace["time"]
        axis = axes[row_index, 0]
        axis.plot(time, trace["command"], "--", color="tab:orange", linewidth=1.0, label="Command")
        axis.plot(time, trace["real"], color="black", linewidth=1.35, label="Real H1")
        axis.plot(time, trace["original"], color="tab:red", linewidth=1.05, label="Original MuJoCo")
        axis.plot(time, trace["calibrated"], color="tab:green", linewidth=1.05, label="Calibrated MuJoCo")
        axis.set_title(f"{record['joint_key']} — {record['dataset_name']}", fontsize=10)
        axis.set_ylabel("Joint position [rad]")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=7)

        axis = axes[row_index, 1]
        axis.plot(time, trace["original"] - trace["real"], color="tab:red", linewidth=0.9, label="Original error")
        axis.plot(time, trace["calibrated"] - trace["real"], color="tab:green", linewidth=0.9, label="Calibrated error")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title(
            f"RMSE {record['original_rmse_rad']:.4f} → {record['calibrated_rmse_rad']:.4f} rad "
            f"({record['improvement_pct']:+.1f}%)",
            fontsize=10,
        )
        axis.set_ylabel("Simulation − real [rad]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time [s]")
    figure.suptitle("Representative held-out trajectories", fontsize=15)
    figure.tight_layout()
    figure.savefig(RESULT_ROOT / "heldout_representative_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: dict) -> None:
    small = summary["small_signal_validation"]
    validation = summary["project_motion"]["validation"]
    blind = summary["project_motion"]["blind"]
    report = f"""# H1 校准模型：关节响应留出验证

## 结论

在未参与本轮评估拟合的留出数据上，校准模型总体上比原始 MuJoCo 更接近真机关节响应，但并非每一个动作都改善。

- 小信号留出测试：{small['count']} 条，{small['improved_count']} 条改善；平均 RMSE `{small['original_mean_rmse_rad']:.6f}` → `{small['calibrated_mean_rmse_rad']:.6f}` rad，降低 `{small['reduction_pct']:.2f}%`。
- 同一批数据的速度 RMSE：`{small['original_mean_velocity_rmse_rad_s']:.6f}` → `{small['calibrated_mean_velocity_rmse_rad_s']:.6f}` rad/s，降低 `{small['velocity_reduction_pct']:.2f}%`。
- 关节聚类 bootstrap 95% CI：`[{small['cluster_bootstrap_reduction_ci95_pct'][0]:.2f}%, {small['cluster_bootstrap_reduction_ci95_pct'][1]:.2f}%]`。
- 项目动作 validation：{validation['count']} 条，{validation['improved_count']} 条改善；平均 RMSE 降低 `{validation['reduction_pct']:.2f}%`。
- 项目动作 blind：{blind['count']} 条，{blind['improved_count']} 条改善；平均 RMSE 降低 `{blind['reduction_pct']:.2f}%`。

## 证据边界

这些结果支持“固定基座/吊起条件下，校准模型的关节位置响应总体更准确”。它们不证明落地接触、整机平衡或策略级行为更准确。负改善的数据点已保留，没有从统计中删除。

## 文件

- `heldout_evidence_dashboard.png`：全部留出测试概览。
- `heldout_representative_trajectories.png`：真机、原始 MuJoCo、校准 MuJoCo 时序曲线。
- `small_signal_validation_metrics.csv`：逐数据集结果。
- `summary.json`：机器可读汇总及数据来源。
"""
    (RESULT_ROOT / "README.md").write_text(report, encoding="utf-8")


def role_summary(rows: list[dict]) -> dict:
    original = float(np.mean([row["original_rmse_rad"] for row in rows]))
    calibrated = float(np.mean([row["calibrated_rmse_rad"] for row in rows]))
    return {
        "count": len(rows),
        "improved_count": sum(row["calibrated_rmse_rad"] < row["original_rmse_rad"] for row in rows),
        "original_mean_rmse_rad": original,
        "calibrated_mean_rmse_rad": calibrated,
        "reduction_pct": 100.0 * (original - calibrated) / original,
    }


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validation_entries = [
        entry for entry in manifest["small_signal"]["datasets"] if entry["use"] == "validation"
    ]

    representative_paths = {
        "datasets/h1/real/left_knee/validation_sine_1hz.pt",
        "datasets/h1/real/left_shoulder_pitch/validation_sine_1hz.pt",
        "datasets/h1/real/torso/validation_sine_1hz.pt",
    }
    rows: list[dict] = []
    representatives: list[dict] = []
    for number, entry in enumerate(validation_entries, start=1):
        dataset = REPO_ROOT / entry["dataset"]
        active_joint = joint_name(entry["joint_key"])
        metrics, traces = replay(dataset, active_joint)
        body_group = (
            "torso"
            if entry["joint_key"] == "torso"
            else "arm"
            if any(token in entry["joint_key"] for token in ("shoulder", "elbow"))
            else "leg"
        )
        row = {
            "joint_key": entry["joint_key"],
            "joint_name": active_joint,
            "body_group": body_group,
            "dataset": entry["dataset"],
            "source_raw": entry["source_raw"],
            **metrics,
        }
        rows.append(row)
        print(
            f"[{number:02d}/{len(validation_entries)}] {entry['joint_key']}: "
            f"{metrics['original_rmse_rad']:.6f} -> {metrics['calibrated_rmse_rad']:.6f} "
            f"({metrics['improvement_pct']:+.1f}%)"
        )
        if entry["dataset"] in representative_paths:
            representatives.append(
                {
                    **row,
                    "dataset_name": dataset.name,
                    "trace": traces,
                }
            )

    with (RESULT_ROOT / "small_signal_validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    project = json.loads(PROJECT_SUMMARY.read_text(encoding="utf-8"))
    project_rows = [row for row in project["runs"] if row["role"] in {"validation", "blind"}]
    bootstrap = cluster_bootstrap(rows)
    summary = {
        "schema": "h1-calibration-heldout-evidence-v1",
        "claim": "joint-response accuracy under fixed-base/suspended replay",
        "generated_from": {
            "official_manifest": str(MANIFEST),
            "project_summary": str(PROJECT_SUMMARY),
            "original_model": str(ORIGINAL_XML),
            "calibrated_model": str(CALIBRATED_XML),
        },
        "fit_data_excluded_from_headline": True,
        "small_signal_validation": {
            **role_summary(rows),
            "joint_count": len({row["joint_key"] for row in rows}),
            "original_mean_velocity_rmse_rad_s": float(
                np.mean([row["original_velocity_rmse_rad_s"] for row in rows])
            ),
            "calibrated_mean_velocity_rmse_rad_s": float(
                np.mean([row["calibrated_velocity_rmse_rad_s"] for row in rows])
            ),
            "cluster_bootstrap_reduction_ci95_pct": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
        },
        "project_motion": {
            role: role_summary([row for row in project_rows if row["role"] == role])
            for role in ("validation", "blind")
        },
        "limitations": [
            "Fixed-base/suspended replay only.",
            "Does not validate ground contact, floating-base balance, or policy-level behavior.",
            "Some held-out motions regress; all negative results are retained.",
        ],
        "small_signal_rows": rows,
        "project_motion_rows": project_rows,
    }
    small = summary["small_signal_validation"]
    small["velocity_reduction_pct"] = 100.0 * (
        small["original_mean_velocity_rmse_rad_s"]
        - small["calibrated_mean_velocity_rmse_rad_s"]
    ) / small["original_mean_velocity_rmse_rad_s"]
    (RESULT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_dashboard(rows, project_rows, summary)
    plot_traces(representatives)
    write_report(summary)
    print(f"Wrote held-out evidence to {RESULT_ROOT}")


if __name__ == "__main__":
    main()
