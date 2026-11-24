# H1 全范围真机实验与离线参数辨识报告

生成日期：2026-11-20  
实验平台：Unitree H1（19 DoF 控制模型，20 个物理电机槽位）  
控制增益范围：`h1_19dof_loco_real.robot`  
实验边界：固定吊装、足端离地、单一 LowCmd 发布者

## 1. 结论摘要

- 已冻结可追溯的正式数据清单：**77 个小幅正式 `.pt` 数据集**（41 下身、36 上身/腰）、9 个内部诊断段，以及 **12 个大幅项目动作数据集**。
- 原始目录共 108 次记录：102 次完成、6 次中止。重复实验只保留最后一次经审核的成功记录；碰撞、中止、旧手臂姿态和被更新记录替代的数据均排除。
- 19 个关节已分别通过小幅正弦数据执行 CMA-ES，辨识参数为 `armature / frictionloss / damping`，模型文件为 `robot_model/h1/h1_calibrated_full_body.xml`。
- 大幅动作只用 4 条开发记录拟合共享控制延迟；0–20 ms 网格的最优值为 **0.0 ms**，因此最终不人为增加延迟。
- 现有校准模型在开发集平均 RMSE 从 **0.007592** 降到 **0.004911 rad**；验证集从 **0.009028** 降到 **0.006604 rad**；盲测从 **0.008388** 降到 **0.006226 rad**。
- 总体泛化成立，但不是每个关节都改善：右膝、右髋俯仰以及高速肘部存在局部退化。因此当前模型适合作为训练模型/域随机化中心值，不应宣称已经精确恢复所有 CAD 惯量。

## 2. 正式数据选择规则

选择规则：只有 `full_range_official_manifest.json` 明确列出的数据可用于拟合、验证或盲测。同条件重复实验取最后一次完整、经人工确认的成功记录。

- 小幅正式数据：77 个；对应 60 条唯一原始记录。
- 大幅正式数据：12 条。
- 中止并排除：6 条。
- prepare/observe/hold 诊断排除：12 条。
- 旧版本、重复或未进入正式清单：18 条。

明确的中止记录：

- `20261116_195710_left_knee_joint_hold`：RuntimeError: Motor communication-loss counters changed: {1: [970, 975]}.
- `20261116_223454_left_hip_roll_joint_step`：RuntimeError: Joint following error exceeded the configured safety limit.
- `20261117_192258_h1_prepare`：RuntimeError: Joint following error exceeded the configured safety limit.
- `20261117_192359_h1_prepare`：RuntimeError: Joint following error exceeded the configured safety limit.
- `20261117_195206_h1_prepare`：RuntimeError: Motor communication-loss counters exceeded allowance 21.60: {12: {'start': 182, 'current': 204, 'increase': 22}}.
- `20261120_162414_h1_h1_upper_yaw_elbow_excitation_upper_body_amp0.60_speed0.75`：RuntimeError: LowState is stale: 25.9 ms.

关键重跑选择包括：左踝 1 Hz 使用 22:07:47 的清障重跑；右踝 0.5/1/2 Hz 使用 21:53:41、21:53:01、21:51:21；左髋侧摆使用新手臂姿态下的 22:48:04；左肘阶跃使用 19:53:40；上肢 yaw/elbow 0.60@0.75 使用 16:30:31，排除 16:24:14 的碰撞/陈旧 LowState 记录。

## 3. 实验矩阵

小幅实验覆盖 19 个关节。每个关节含拟合频段（0.5、2 Hz）及独立验证（1 Hz、阶跃）；上身 multi-sine 内部 1 Hz 仅用于诊断，不与独立 1 Hz 混用。下身还包含膝、踝、髋三个方向；上身包含双侧肩 pitch/roll/yaw、双肘和 torso。

大幅项目动作：

| 时间 | profile | 用途 | 幅值比例 | 速度比例 | 状态率 Hz | 最大间隔 ms |
|---|---|---|---|---|---|---|
| 152448 | lower-body | fit | 0.60 | 0.50 | 501.0 | 6.71 |
| 152657 | lower-body | validation | 0.75 | 0.50 | 500.8 | 6.40 |
| 152849 | lower-body | validation | 0.90 | 0.50 | 500.7 | 9.68 |
| 154112 | upper-body | fit | 0.60 | 0.50 | 499.8 | 8.08 |
| 154411 | upper-body | validation | 0.90 | 0.30 | 500.1 | 8.30 |
| 155247 | upper-body | fit | 0.60 | 0.50 | 501.0 | 10.15 |
| 155811 | upper-body | validation | 0.90 | 0.50 | 501.6 | 8.51 |
| 161005 | torso-only | fit | 0.60 | 0.50 | 503.7 | 8.49 |
| 161326 | torso-only | validation | 0.90 | 0.50 | 503.8 | 15.12 |
| 161757 | lower-body | blind | 0.75 | 0.40 | 502.3 | 16.83 |
| 162020 | upper-body | blind | 0.60 | 0.30 | 501.5 | 15.26 |
| 163031 | upper-body | blind | 0.60 | 0.75 | 500.5 | 6.85 |

所有正式大幅记录均满足：dropped=0、malformed=0、states_before_command=0，状态率约 500 Hz，最大间隔小于 20 ms。

## 4. 数据处理与辨识流程

1. 原始 `raw.npz` 按动作 manifest 裁切：去除 prepare/settle 与 post-hold/return，只保留真实激励段。
2. 状态以线性插值重采样到 500 Hz；离散命令、Kp/Kd 与前馈力矩使用零阶保持。
3. 小幅拟合：每个关节以 0.5/2 Hz 为拟合数据，CMA-ES 搜索 `armature / frictionloss / damping`；1 Hz 与阶跃独立验证。
4. 大幅泛化：60% 幅值记录用于开发；75/90% 作为验证；不同速度 0.40、0.30、0.75 作为盲测。
5. 共享延迟只在开发集上从 0–20 ms 搜索，验证/盲测不参与选参。
6. MuJoCo 回放采用固定基座、无接触，使用真机记录的 Kp/Kd 与命令轨迹；每条轨迹只用首帧真机状态初始化，之后自由演化。

## 5. 19 关节辨识参数

| 关节 | armature | frictionloss | damping |
|---|---|---|---|
| left_knee_joint | 0.424055 | 3.41166 | 5.02497 |
| right_knee_joint | 0.43638 | 3.05874 | 6.79966 |
| left_ankle_joint | 0.151869 | 1.12759 | 0.0237066 |
| right_ankle_joint | 0.106533 | 0.577623 | 0.00039372 |
| left_hip_pitch_joint | 2.20702e-05 | 2.36745 | 1.15657 |
| right_hip_pitch_joint | 7.73094e-06 | 1.54802 | 6.9199 |
| left_hip_roll_joint | 0.862611 | 5.14777 | 1.9692e-05 |
| right_hip_roll_joint | 0.128591 | 2.40119 | 0.571023 |
| left_hip_yaw_joint | 0.148286 | 2.8971 | 1.99122e-05 |
| right_hip_yaw_joint | 0.0953905 | 3.1749 | 2.34214e-05 |
| left_elbow_joint | 0.0309128 | 0.269525 | 0.000111726 |
| right_elbow_joint | 0.0441613 | 0.303969 | 2.63622e-05 |
| left_shoulder_pitch_joint | 0.0232834 | 1.25286 | 1.26143 |
| right_shoulder_pitch_joint | 0.0264026 | 1.43209 | 1.92645e-05 |
| left_shoulder_roll_joint | 0.0797404 | 1.087 | 0.000439399 |
| right_shoulder_roll_joint | 0.138805 | 1.02228 | 2.96934e-05 |
| left_shoulder_yaw_joint | 0.000491171 | 0.340624 | 0.308015 |
| right_shoulder_yaw_joint | 2.19687e-06 | 0.175392 | 0.544709 |
| torso_joint | 1.08055e-06 | 1.97691 | 0.0878298 |

## 6. 大幅动作离线结果

分组平均：

| 数据角色 | 原始 MuJoCo RMSE | 校准 MuJoCo RMSE | 改善 |
|---|---|---|---|
| fit | 0.007592 | 0.004911 | 35.3% |
| validation | 0.009028 | 0.006604 | 26.9% |
| blind | 0.008388 | 0.006226 | 25.8% |

逐条结果：

| 用途 | profile | 幅值 | 速度 | 原始 RMSE | 校准 RMSE | 改善 |
|---|---|---|---|---|---|---|
| fit | lower-body | 0.60 | 0.50 | 0.012025 | 0.009301 | 22.7% |
| validation | lower-body | 0.75 | 0.50 | 0.012982 | 0.010453 | 19.5% |
| validation | lower-body | 0.90 | 0.50 | 0.013268 | 0.011451 | 13.7% |
| fit | upper-body | 0.60 | 0.50 | 0.007812 | 0.004528 | 42.0% |
| validation | upper-body | 0.90 | 0.30 | 0.007656 | 0.004253 | 44.4% |
| fit | upper-body | 0.60 | 0.50 | 0.004624 | 0.004571 | 1.2% |
| validation | upper-body | 0.90 | 0.50 | 0.004773 | 0.005183 | -8.6% |
| fit | torso-only | 0.60 | 0.50 | 0.005907 | 0.001244 | 78.9% |
| validation | torso-only | 0.90 | 0.50 | 0.006460 | 0.001679 | 74.0% |
| blind | lower-body | 0.75 | 0.40 | 0.012942 | 0.009563 | 26.1% |
| blind | upper-body | 0.60 | 0.30 | 0.007794 | 0.004579 | 41.2% |
| blind | upper-body | 0.60 | 0.75 | 0.004428 | 0.004537 | -2.4% |

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
