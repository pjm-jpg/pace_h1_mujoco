"""Create one compact publication-style figure for H1 calibration evidence."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.generate_h1_calibration_evidence import replay  # noqa: E402


EVIDENCE_ROOT = REPO_ROOT / "sysid_mujoco/results/h1/calibration_evidence_v1"
ABLATION_ROOT = REPO_ROOT / "sysid_mujoco/results/h1/ablation_m0_m4_v1"
OUTPUT_ROOT = REPO_ROOT / "sysid_mujoco/results/h1/publication_evidence_v1"
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")

ORANGE = "#B65A4A"
BLUE = "#356FA3"
SKY = "#78A9CF"
BLACK = "#202020"
DARK_GREY = "#4D4D4D"
MID_GREY = "#888888"
LIGHT_GREY = "#C7C7C7"
GRID = "#E6E6E6"


def _style() -> None:
    if not CJK_FONT.is_file():
        raise FileNotFoundError(f"未找到中文字体：{CJK_FONT}")
    font_manager.fontManager.addfont(str(CJK_FONT))
    cjk_family = font_manager.FontProperties(fname=str(CJK_FONT)).get_name()
    plt.rcParams.update(
        {
            # 直接注册字体文件，避免不同机器上的 Matplotlib 字体缓存找不到中文字体。
            "font.family": cjk_family,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(axis, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _load() -> tuple[list[dict], dict, dict]:
    with (EVIDENCE_ROOT / "small_signal_validation_metrics.csv").open(
        encoding="utf-8"
    ) as stream:
        small_rows = list(csv.DictReader(stream))
    evidence = json.loads((EVIDENCE_ROOT / "summary.json").read_text(encoding="utf-8"))
    ablation = json.loads((ABLATION_ROOT / "summary.json").read_text(encoding="utf-8"))
    return small_rows, evidence, ablation


def _validation_scatter(axis, rows: list[dict], evidence: dict) -> None:
    original = np.asarray([float(row["original_rmse_rad"]) for row in rows])
    calibrated = np.asarray([float(row["calibrated_rmse_rad"]) for row in rows])
    limit = 1.06 * float(max(np.max(original), np.max(calibrated)))
    axis.scatter(
        original,
        calibrated,
        s=24,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.85,
        zorder=3,
    )
    axis.plot([0, limit], [0, limit], color=MID_GREY, linestyle="--", linewidth=1.0)
    axis.set(xlim=(0, limit), ylim=(0, limit))
    axis.set_xlabel("原始模型 RMSE [rad]")
    axis.set_ylabel("校准模型 RMSE [rad]")
    axis.set_title("留出集关节响应")
    small = evidence["small_signal_validation"]
    ci = small["cluster_bootstrap_reduction_ci95_pct"]
    axis.text(
        0.04,
        0.94,
        f"39/39 条均改善\n平均降低  {small['reduction_pct']:.1f}%\n95% 置信区间  {ci[0]:.1f}–{ci[1]:.1f}%",
        transform=axis.transAxes,
        va="top",
        ha="left",
        color=BLACK,
        linespacing=1.35,
    )
    _clean_axis(axis)
    _panel_label(axis, "a")


def _trajectory(axis) -> None:
    # 选用未参与拟合的腰部 1 Hz 数据：波形清晰，适合作为代表性可视化。
    dataset = REPO_ROOT / "datasets/h1/real/torso/validation_sine_1hz.pt"
    with redirect_stdout(io.StringIO()):
        metrics, trace = replay(dataset, "torso_joint")
    time = np.asarray(trace["time"], dtype=np.float64)
    time = time - time[0]
    mask = (time >= 2.2) & (time <= 5.2)
    axis.plot(time[mask], np.asarray(trace["real"])[mask], color=BLACK, linewidth=1.7, label="H1 真机")
    axis.plot(
        time[mask],
        np.asarray(trace["original"])[mask],
        color=ORANGE,
        linewidth=1.15,
        label="原始模型",
    )
    axis.plot(
        time[mask],
        np.asarray(trace["calibrated"])[mask],
        color=BLUE,
        linewidth=1.55,
        label="校准模型",
    )
    axis.set_xlabel("时间 [s]")
    axis.set_ylabel("关节位置 [rad]")
    axis.set_title("代表性留出轨迹：腰部，1 Hz")
    axis.legend(frameon=False, ncol=3, loc="lower left")
    axis.text(
        0.98,
        0.94,
        f"RMSE  {metrics['original_rmse_rad']:.4f} → {metrics['calibrated_rmse_rad']:.4f} rad",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    _clean_axis(axis, "y")
    _panel_label(axis, "b")


def _ablation(axis, ablation: dict) -> None:
    data = ablation["small_signal_validation"]
    metrics = [
        ("position_rmse_rad", "位置"),
        ("dynamic_rmse_rad", "动态"),
        ("velocity_rmse_rad_s", "速度"),
        ("amplitude_relative_error", "1 Hz 幅值"),
    ]
    x = np.arange(len(metrics))
    styles = {
        "M0": (ORANGE, "o", "-", 1.5),
        "M1": (LIGHT_GREY, "s", "--", 1.0),
        "M2": (MID_GREY, "^", "--", 1.0),
        "M3": (DARK_GREY, "D", "-.", 1.2),
        "M4": (BLUE, "o", "-", 2.0),
    }
    labels = {
        "M0": "M0 原始模型",
        "M1": "M1 + 偏置",
        "M2": "M2 + 延迟",
        "M3": "M3 + 偏置 + 延迟",
        "M4": "M4 完整动力学",
    }
    for model in ("M0", "M1", "M2", "M3", "M4"):
        values = [100.0 * data[model][key] / data["M0"][key] for key, _ in metrics]
        color, marker, linestyle, width = styles[model]
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=width,
            markersize=4.5,
            label=labels[model],
            zorder=4 if model == "M4" else 2,
        )
    axis.axhline(100.0, color=ORANGE, linewidth=0.8, alpha=0.45)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("相对 M0 的误差 [%]")
    axis.set_ylim(0, 112)
    axis.set_title("留出集消融实验（越低越好）")
    axis.legend(frameon=False, ncol=2, loc="lower left")
    _clean_axis(axis, "y")
    _panel_label(axis, "c")


def _project_generalization(axis, evidence: dict, ablation: dict) -> None:
    runs = evidence["project_motion_rows"]
    with (ABLATION_ROOT / "project_motion_ablation_metrics.csv").open(
        encoding="utf-8"
    ) as stream:
        ablation_rows = list(csv.DictReader(stream))
    by_run_model = {
        (row["run"], row["model"]): float(row["position_rmse_rad"])
        for row in ablation_rows
    }
    x_positions = {"M3": 0.0, "M4": 1.0}
    role_style = {
        "validation": (DARK_GREY, "o", "验证集"),
        "blind": (BLUE, "s", "盲测集"),
    }
    role_seen: set[str] = set()
    for index, run in enumerate(runs):
        base = by_run_model[(run["run"], "M0")]
        values = [100.0 * by_run_model[(run["run"], model)] / base for model in ("M3", "M4")]
        color, marker, label = role_style[run["role"]]
        axis.plot([0, 1], values, color=color, linewidth=0.7, alpha=0.45, zorder=1)
        for x, value in zip((0, 1), values, strict=True):
            axis.scatter(
                x,
                value,
                s=24,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.4,
                label=label if run["role"] not in role_seen and x == 0 else None,
                zorder=3,
            )
        role_seen.add(run["role"])
    means = []
    for model in ("M3", "M4"):
        ratios = [
            100.0 * by_run_model[(run["run"], model)] / by_run_model[(run["run"], "M0")]
            for run in runs
        ]
        means.append(float(np.mean(ratios)))
    axis.scatter([0, 1], means, marker="_", s=520, linewidth=2.5, color=BLACK, zorder=4, label="均值")
    axis.axhline(100.0, color=ORANGE, linestyle="--", linewidth=1.0, label="M0 基线")
    axis.set_xticks([0, 1], ["M3\n简单基线", "M4\n完整动力学"])
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylabel("相对 M0 的 RMSE [%]")
    axis.set_title("8 条项目动作的泛化结果")
    axis.legend(frameon=False, ncol=2, loc="lower left")
    _clean_axis(axis, "y")
    _panel_label(axis, "d")


def _ablation_heatmap(axis, ablation: dict) -> None:
    """Compact M0--M4 summary; numbers are normalized errors, not improvements."""
    data = ablation["small_signal_validation"]
    models = ("M0", "M1", "M2", "M3", "M4")
    metrics = (
        ("position_rmse_rad", "位置"),
        ("dynamic_rmse_rad", "动态"),
        ("velocity_rmse_rad_s", "速度"),
        ("amplitude_relative_error", "1 Hz 幅值"),
    )
    values = np.asarray(
        [
            [100.0 * data[model][key] / data["M0"][key] for key, _ in metrics]
            for model in models
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "calibration_evidence", [BLUE, "#DCEEF7", "#F2F2F2"]
    )
    axis.imshow(values, vmin=20, vmax=103, cmap=cmap, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value < 55 else BLACK,
                fontweight="bold" if row == 4 else "normal",
            )
    axis.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    axis.set_yticks(
        np.arange(len(models)),
        ["M0 原始", "M1 + 偏置", "M2 + 延迟", "M3 + 两者", "M4 完整动力学"],
    )
    axis.tick_params(length=0)
    axis.get_yticklabels()[-1].set_color(BLUE)
    axis.get_yticklabels()[-1].set_fontweight("bold")
    axis.set_title("M0–M4 留出集消融")
    axis.text(
        0.99,
        1.01,
        "M0 = 100，越低越好",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=DARK_GREY,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    _panel_label(axis, "a")


def _project_reduction(axis, evidence: dict) -> None:
    """Show group means as bars and retain every individual project-motion result."""
    runs = evidence["project_motion_rows"]
    groups = (("validation", "验证集", DARK_GREY, 0.0), ("blind", "盲测集", BLUE, 1.0))
    offsets = np.asarray([-0.20, -0.10, 0.0, 0.10, 0.20])
    for role, label, color, center in groups:
        selected = [run for run in runs if run["role"] == role]
        reductions = np.asarray([float(run["calibrated_improvement_pct"]) for run in selected])
        local_offsets = offsets[: len(reductions)]
        local_offsets = local_offsets - np.mean(local_offsets)
        summary = evidence["project_motion"][role]
        mean_reduction = float(summary["reduction_pct"])
        axis.bar(
            center,
            mean_reduction,
            width=0.58,
            color=color,
            alpha=0.18,
            edgecolor=color,
            linewidth=1.2,
            zorder=1,
        )
        axis.scatter(
            center + local_offsets,
            reductions,
            s=48,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axis.text(
            center,
            80,
            f"平均改善 {mean_reduction:.1f}%\n{summary['improved_count']}/{summary['count']} 条动作改善",
            ha="center",
            va="top",
            color=color,
            fontweight="bold",
        )
    axis.axhline(0.0, color=ORANGE, linestyle="--", linewidth=1.1)
    axis.set_xticks([0, 1], ["验证集\n5 条动作", "盲测集\n3 条新动作"])
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(-15, 85)
    axis.set_ylabel("相对原始模型的 RMSE 改善 [%]")
    axis.set_title("项目动作验证（越高越好）")
    axis.text(
        0.02,
        0.02,
        "圆点：单条动作　浅色柱：组内平均值",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=DARK_GREY,
    )
    axis.text(
        1.53,
        1.0,
        "0：与原始模型相同",
        ha="right",
        va="bottom",
        fontsize=8,
        color=ORANGE,
    )
    _clean_axis(axis, "y")
    _panel_label(axis, "b")


def main() -> None:
    _style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    small_rows, evidence, ablation = _load()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.15))
    _validation_scatter(axes[0], small_rows, evidence)
    _trajectory(axes[1])
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.90, wspace=0.28)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUTPUT_ROOT / f"h1_calibration_core_validation.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.15))
    _ablation_heatmap(axes[0], ablation)
    _project_reduction(axes[1], evidence)
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.88, wspace=0.34)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUTPUT_ROOT / f"h1_calibration_ablation_generalization.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    caption = """# H1 校准模型科研展示图

## 图 1：核心验证

**a**，39 条未参与拟合的关节响应验证；虚线表示两种模型误差相同，全部数据点均位于虚线下方。  
**b**，腰部 1 Hz 留出轨迹示例；完整动力学校准显著修正了原始模型的响应幅值。

## 图 2：消融与泛化

**a**，M0–M4 留出集消融；每个指标归一化为 M0=100，越低越好。  
**b**，8 条未参与选参的项目动作；空心菱形表示组内平均改善。两条退化动作被完整保留。

颜色：真机为黑色，原始模型为橙色，验证集为深灰色，盲测集及完整动力学校准为蓝色。
"""
    (OUTPUT_ROOT / "README.md").write_text(caption, encoding="utf-8")
    print(f"Wrote two publication figures to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
