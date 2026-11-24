"""Generate the auditable H1 experiment report in Markdown and PDF."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "datasets" / "h1" / "real" / "full_range_official_manifest.json"
RESULT_PATH = REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "full_range_offline" / "summary.json"
PARAM_PATH = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_full_body_params.json"
OUTPUT_MD = REPO_ROOT / "docs" / "experiments" / "h1_full_range_system_identification_report.md"
OUTPUT_PDF = REPO_ROOT / "docs" / "experiments" / "h1_full_range_system_identification_report.pdf"
FIGURE_ROOT = REPO_ROOT / "sysid_mujoco" / "results" / "h1" / "full_range_offline"


def _font() -> fm.FontProperties:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    path = next((item for item in candidates if item.is_file()), None)
    return fm.FontProperties(fname=str(path)) if path else fm.FontProperties()


FONT = _font()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(manifest: dict, result: dict, params: dict) -> str:
    counts = manifest["raw_inventory_counts"]
    role_rows = []
    for role, values in result["role_summary"].items():
        role_rows.append([
            role,
            f"{values['original_rmse_rad']:.6f}",
            f"{values['calibrated_rmse_rad']:.6f}",
            f"{_pct(values['original_rmse_rad'], values['calibrated_rmse_rad']):.1f}%",
        ])
    project_rows = []
    for run in manifest["project_runs"]:
        project_rows.append([
            run["run"].split("_")[1], run["profile"], run["role"],
            f"{run['amplitude_scale']:.2f}", f"{run['speed_scale']:.2f}",
            f"{run['stats']['median_state_rate_hz']:.1f}",
            f"{1000 * run['stats']['largest_state_gap_s']:.2f}",
        ])
    parameter_rows = [
        [joint, f"{values['armature']:.6g}", f"{values['frictionloss']:.6g}", f"{values['damping']:.6g}"]
        for joint, values in params["parameters"].items()
    ]
    run_rows = [[
        row["role"], row["profile"], f"{row['amplitude_scale']:.2f}",
        f"{row['speed_scale']:.2f}", f"{row['original_rmse_rad']:.6f}",
        f"{row['calibrated_rmse_rad']:.6f}", f"{row['calibrated_improvement_pct']:.1f}%",
    ] for row in result["runs"]]
    aborted = [item for item in manifest["raw_inventory"] if item["classification"] == "excluded_aborted"]
    aborted_lines = "\n".join(f"- `{item['run']}`：{item['reason']}" for item in aborted)
    return f"""# H1 全范围真机实验与离线参数辨识报告

生成日期：2026-11-20  
实验平台：Unitree H1（19 DoF 控制模型，20 个物理电机槽位）  
控制增益范围：`h1_19dof_loco_real.robot`  
实验边界：固定吊装、足端离地、单一 LowCmd 发布者

## 1. 结论摘要

- 已冻结可追溯的正式数据清单：**77 个小幅正式 `.pt` 数据集**（41 下身、36 上身/腰）、9 个内部诊断段，以及 **12 个大幅项目动作数据集**。
- 原始目录共 108 次记录：102 次完成、6 次中止。重复实验只保留最后一次经审核的成功记录；碰撞、中止、旧手臂姿态和被更新记录替代的数据均排除。
- 19 个关节已分别通过小幅正弦数据执行 CMA-ES，辨识参数为 `armature / frictionloss / damping`，模型文件为 `robot_model/h1/h1_calibrated_full_body.xml`。
- 大幅动作只用 4 条开发记录拟合共享控制延迟；0–20 ms 网格的最优值为 **{result['identified_shared_delay_ms']:.1f} ms**，因此最终不人为增加延迟。
- 现有校准模型在开发集平均 RMSE 从 **{result['role_summary']['fit']['original_rmse_rad']:.6f}** 降到 **{result['role_summary']['fit']['calibrated_rmse_rad']:.6f} rad**；验证集从 **{result['role_summary']['validation']['original_rmse_rad']:.6f}** 降到 **{result['role_summary']['validation']['calibrated_rmse_rad']:.6f} rad**；盲测从 **{result['role_summary']['blind']['original_rmse_rad']:.6f}** 降到 **{result['role_summary']['blind']['calibrated_rmse_rad']:.6f} rad**。
- 总体泛化成立，但不是每个关节都改善：右膝、右髋俯仰以及高速肘部存在局部退化。因此当前模型适合作为训练模型/域随机化中心值，不应宣称已经精确恢复所有 CAD 惯量。

## 2. 正式数据选择规则

选择规则：只有 `full_range_official_manifest.json` 明确列出的数据可用于拟合、验证或盲测。同条件重复实验取最后一次完整、经人工确认的成功记录。

- 小幅正式数据：{manifest['small_signal']['official_dataset_count']} 个；对应 {counts.get('selected_small_signal', 0)} 条唯一原始记录。
- 大幅正式数据：{len(manifest['project_runs'])} 条。
- 中止并排除：{counts.get('excluded_aborted', 0)} 条。
- prepare/observe/hold 诊断排除：{counts.get('excluded_diagnostic', 0)} 条。
- 旧版本、重复或未进入正式清单：{counts.get('excluded_superseded_or_unreviewed', 0)} 条。

明确的中止记录：

{aborted_lines}

关键重跑选择包括：左踝 1 Hz 使用 22:07:47 的清障重跑；右踝 0.5/1/2 Hz 使用 21:53:41、21:53:01、21:51:21；左髋侧摆使用新手臂姿态下的 22:48:04；左肘阶跃使用 19:53:40；上肢 yaw/elbow 0.60@0.75 使用 16:30:31，排除 16:24:14 的碰撞/陈旧 LowState 记录。

## 3. 实验矩阵

小幅实验覆盖 19 个关节。每个关节含拟合频段（0.5、2 Hz）及独立验证（1 Hz、阶跃）；上身 multi-sine 内部 1 Hz 仅用于诊断，不与独立 1 Hz 混用。下身还包含膝、踝、髋三个方向；上身包含双侧肩 pitch/roll/yaw、双肘和 torso。

大幅项目动作：

{_md_table(['时间', 'profile', '用途', '幅值比例', '速度比例', '状态率 Hz', '最大间隔 ms'], project_rows)}

所有正式大幅记录均满足：dropped=0、malformed=0、states_before_command=0，状态率约 500 Hz，最大间隔小于 20 ms。

## 4. 数据处理与辨识流程

1. 原始 `raw.npz` 按动作 manifest 裁切：去除 prepare/settle 与 post-hold/return，只保留真实激励段。
2. 状态以线性插值重采样到 500 Hz；离散命令、Kp/Kd 与前馈力矩使用零阶保持。
3. 小幅拟合：每个关节以 0.5/2 Hz 为拟合数据，CMA-ES 搜索 `armature / frictionloss / damping`；1 Hz 与阶跃独立验证。
4. 大幅泛化：60% 幅值记录用于开发；75/90% 作为验证；不同速度 0.40、0.30、0.75 作为盲测。
5. 共享延迟只在开发集上从 0–20 ms 搜索，验证/盲测不参与选参。
6. MuJoCo 回放采用固定基座、无接触，使用真机记录的 Kp/Kd 与命令轨迹；每条轨迹只用首帧真机状态初始化，之后自由演化。

## 5. 19 关节辨识参数

{_md_table(['关节', 'armature', 'frictionloss', 'damping'], parameter_rows)}

## 6. 大幅动作离线结果

分组平均：

{_md_table(['数据角色', '原始 MuJoCo RMSE', '校准 MuJoCo RMSE', '改善'], role_rows)}

逐条结果：

{_md_table(['用途', 'profile', '幅值', '速度', '原始 RMSE', '校准 RMSE', '改善'], run_rows)}

校准模型对 lower-body、largebox upper-body 和 torso 有稳定总体收益；yaw/elbow 的 90% 验证与 0.75x 盲测略有退化，说明摩擦/阻尼在高速或大幅区间具有非线性，单组常数不能在所有工作点同时最优。

## 7. 数据质量例外

- `20261116_223218_right_hip_pitch_joint_sine` 全记录出现 31.1 ms 单点间隔，但该点位于正式 2 Hz 裁切区间之外，不影响正式数据。
- `20261117_201508_right_shoulder_pitch_joint_sine` 独立 1 Hz 验证段有 23.8 ms 单点间隔；它不参与 CMA-ES 拟合，且同关节有 multi-sine 内部 1 Hz 诊断段提供冗余。报告保留该事实，不把它当作无缺陷记录。
- 大幅正式 12 条记录没有超过 20 ms 的状态间隔。

## 8. 适用范围与剩余不确定性

- 本次辨识对应固定增益控制器；更换 Kp/Kd 后需重新验证。
- 固定吊装数据能可靠辨识执行器闭环动态，但不能辨识足地摩擦、接触恢复系数或承重结构形变。项目训练中应继续对地面摩擦与质量/CoM 做域随机化。
- 没有载物质量实验；上肢拿取物体时应将 payload 作为训练随机变量。
- 当前大幅动作验证没有支持高维全身 CAD 惯量自由拟合。由于存在参数耦合和不可辨识性，本轮没有用验证/盲测数据强行改写 link inertia。
- 推荐训练时以 `h1_calibrated_full_body.xml` 为中心，同时对局部退化关节（右膝、右髋俯仰、双肘）保留更宽的 damping/friction/armature 随机化范围。

## 9. 可复现产物

- 正式数据清单：`datasets/h1/real/full_range_official_manifest.json`
- 大幅动作 `.pt`：`datasets/h1/real/project_motion/`
- 19 关节参数：`robot_model/h1/h1_calibrated_full_body_params.json`
- 校准模型：`robot_model/h1/h1_calibrated_full_body.xml`
- 离线结果：`sysid_mujoco/results/h1/full_range_offline/summary.json`
- 逐条指标：`sysid_mujoco/results/h1/full_range_offline/run_metrics.csv`
- 重建清单：`pixi run -e mujoco python datasets/build_h1_offline_package.py`
- 重跑离线辨识：`MPLCONFIGDIR=/tmp/h1_mpl pixi run -e mujoco python sysid_mujoco/run_h1_full_range_offline.py`

本报告的正式结论只以以上机器可读产物为依据。
"""


def _text_page(pdf: PdfPages, title: str, paragraphs: list[str], page: str = "") -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0.08, 0.06, 0.84, 0.88])
    axis.axis("off")
    axis.text(0, 1.0, title, fontproperties=FONT, fontsize=18, weight="bold", va="top")
    y = 0.94
    for paragraph in paragraphs:
        if paragraph.startswith("•"):
            lines = textwrap.wrap(paragraph, width=52, subsequent_indent="  ")
        else:
            # CJK glyphs are visually wider than Latin characters; the lower
            # width keeps long Chinese sentences inside the A4 text box.
            lines = textwrap.wrap(paragraph, width=56)
        axis.text(0, y, "\n".join(lines), fontproperties=FONT, fontsize=10.5, va="top", linespacing=1.45)
        y -= 0.034 * max(1, len(lines)) + 0.022
    if page:
        axis.text(1, 0, page, fontproperties=FONT, fontsize=8, ha="right", va="bottom", color="0.4")
    pdf.savefig(figure)
    plt.close(figure)


def _table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[list[str]], widths=None, fontsize=7.5) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title(title, fontproperties=FONT, fontsize=16, pad=18)
    table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center", loc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.25)
    for cell in table.get_celld().values():
        cell.get_text().set_fontproperties(FONT)
        cell.set_edgecolor("0.75")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.imshow(plt.imread(image_path))
    axis.axis("off")
    axis.set_title(title, fontproperties=FONT, fontsize=15, pad=12)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def build_pdf(manifest: dict, result: dict, params: dict) -> None:
    counts = manifest["raw_inventory_counts"]
    with PdfPages(OUTPUT_PDF) as pdf:
        _text_page(pdf, "H1 全范围真机实验与离线参数辨识报告", [
            "正式版 · 2026-11-20",
            "范围：19 DoF 小幅关节激励 + 大幅项目动作；固定吊装、足端离地。",
            f"正式数据：{manifest['small_signal']['official_dataset_count']} 个小幅数据集 + {len(manifest['project_runs'])} 条大幅动作。",
            f"主要结果：验证集 RMSE {result['role_summary']['validation']['original_rmse_rad']:.6f} → {result['role_summary']['validation']['calibrated_rmse_rad']:.6f} rad；盲测 {result['role_summary']['blind']['original_rmse_rad']:.6f} → {result['role_summary']['blind']['calibrated_rmse_rad']:.6f} rad。",
            "结论：现有 19 关节 CMA-ES 参数可作为训练模型中心值；共享延迟辨识为 0 ms。局部高速关节仍需通过域随机化覆盖。",
        ])
        _text_page(pdf, "1. 数据冻结与排除规则", [
            "只有 full_range_official_manifest.json 明确列出的记录可用于拟合、验证或盲测。重复实验取最后一次经审核成功记录。",
            f"原始记录 108 次：完成 102 次，中止 {counts.get('excluded_aborted', 0)} 次。正式小幅唯一原始记录 {counts.get('selected_small_signal', 0)} 条，正式大幅记录 {counts.get('selected_project_motion', 0)} 条。",
            "• 排除 16:24 的碰撞/LowState stale 上肢记录，保留 16:30 同条件重跑。",
            "• 排除左髋侧摆旧手臂姿态与跟随误差中止记录，保留 22:48 新姿态成功记录。",
            "• 左踝、右踝、左肘等重复实验均使用最后一次审核通过的数据。",
            "• prepare、observe、hold 仅作诊断，不进入动力学拟合。",
        ])
        project_rows = [[
            run["run"][9:15], run["profile"], run["role"], f"{run['amplitude_scale']:.2f}",
            f"{run['speed_scale']:.2f}", f"{run['stats']['median_state_rate_hz']:.1f}",
            f"{1000*run['stats']['largest_state_gap_s']:.2f}",
        ] for run in manifest["project_runs"]]
        _table_page(pdf, "2. 大幅动作正式数据矩阵", ["时刻", "profile", "角色", "幅值", "速度", "Hz", "最大间隔 ms"], project_rows, fontsize=8)
        _text_page(pdf, "3. 辨识方法与数据隔离", [
            "小幅阶段：19 个关节分别使用 0.5/2 Hz 拟合 armature、frictionloss、damping；独立 1 Hz 与阶跃用于验证。",
            "大幅阶段：4 条 60% 记录为开发集；5 条 75/90% 记录为验证；不同速度的 lower 0.40、upper 0.30、yaw/elbow 0.75 三条记录为盲测。",
            "动作裁切去除 prepare/settle/post-hold/return，重采样至 500 Hz。状态线性插值，命令与增益零阶保持。",
            f"共享延迟在开发集 0–20 ms 网格搜索，最优为 {result['identified_shared_delay_ms']:.1f} ms。验证与盲测不参与选参。",
            "回放模型固定基座、无接触，只在首帧使用真机状态初始化，之后不重置仿真状态。",
        ])
        parameter_rows = [[j, f"{v['armature']:.5g}", f"{v['frictionloss']:.5g}", f"{v['damping']:.5g}"] for j, v in params["parameters"].items()]
        _table_page(pdf, "4. CMA-ES 参数（下身）", ["关节", "armature", "frictionloss", "damping"], parameter_rows[:10], widths=[0.38, .2, .2, .2], fontsize=8.5)
        _table_page(pdf, "5. CMA-ES 参数（腰与上身）", ["关节", "armature", "frictionloss", "damping"], parameter_rows[10:], widths=[0.38, .2, .2, .2], fontsize=8.5)
        role_rows = [[role, f"{v['original_rmse_rad']:.6f}", f"{v['calibrated_rmse_rad']:.6f}", f"{_pct(v['original_rmse_rad'],v['calibrated_rmse_rad']):.1f}%"] for role, v in result["role_summary"].items()]
        _table_page(pdf, "6. 分组泛化结果", ["角色", "原始 RMSE", "校准 RMSE", "改善"], role_rows, widths=[.25]*4, fontsize=11)
        run_rows = [[r["role"], r["profile"], f"{r['amplitude_scale']:.2f}", f"{r['speed_scale']:.2f}", f"{r['original_rmse_rad']:.5f}", f"{r['calibrated_rmse_rad']:.5f}", f"{r['calibrated_improvement_pct']:.1f}%"] for r in result["runs"]]
        _table_page(pdf, "7. 逐条大幅动作回放结果", ["角色", "profile", "幅值", "速度", "原始", "校准", "改善"], run_rows, fontsize=8)
        joint_rows = [[joint.replace("_joint", ""), f"{v['original']:.5f}", f"{v['calibrated']:.5f}", f"{_pct(v['original'],v['calibrated']):.1f}%"] for joint, v in result["joint_summary"].items()]
        _table_page(pdf, "8. 按关节汇总（所有大幅记录）", ["关节", "原始 RMSE", "校准 RMSE", "改善"], joint_rows, widths=[.4,.2,.2,.2], fontsize=7.5)
        _image_page(pdf, "9. 数据集误差与共享延迟扫描", FIGURE_ROOT / "offline_summary.png")
        _image_page(pdf, "10. 原始 MuJoCo / 真机 / 校准 MuJoCo 三曲线", FIGURE_ROOT / "representative_three_curve_comparison.png")
        _text_page(pdf, "11. 结论、限制与训练建议", [
            "总体结论：校准模型在开发、验证、盲测三个集合均降低平均误差，证明小幅单关节辨识可以泛化到本项目的大幅动作范围。",
            "局部例外：右膝、右髋俯仰和双肘在部分大幅/高速记录上没有改善。单一常数摩擦/阻尼无法完全表达速度、温度和负载相关非线性。",
            "训练建议：使用 h1_calibrated_full_body.xml 作为中心模型；对上述局部退化关节设置更宽的 actuator dynamics 随机化。",
            "接触边界：吊装数据不能辨识地面摩擦/恢复系数。保持地面摩擦域随机化，而不是把某一次实验室地面参数写死。",
            "载荷边界：未进行持物 payload 辨识；上肢任务需随机化手端附加载荷与 CoM。",
            "本轮没有使用验证/盲测数据强行拟合高维 link inertia，以避免不可辨识参数产生表面上更低、实际更差的误差。",
            "所有选择、裁切、参数和指标均已保存为机器可读 JSON/CSV，可完整复现。",
        ])


def main() -> None:
    manifest, result, params = _load(MANIFEST_PATH), _load(RESULT_PATH), _load(PARAM_PATH)
    OUTPUT_MD.write_text(build_markdown(manifest, result, params), encoding="utf-8")
    build_pdf(manifest, result, params)
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
