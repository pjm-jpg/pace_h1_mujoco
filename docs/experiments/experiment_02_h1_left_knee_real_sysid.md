# 实验二：H1左膝悬空实机辨识

## 目的

在H1固定吊装、双脚离地条件下，验证从GO2 DDS真机命令与状态记录，到MuJoCo固定基座回放、CMA-ES参数拟合和独立轨迹验证的完整闭环。首轮只辨识 `left_knee_joint` 的等效 `armature`、`frictionloss` 和 `damping`。

这些数值是当前H1模型、悬空构型、位置PD增益和MuJoCo参数化下的**等效关节参数**，不是电机转子或减速器的独立台架物理常数。

## 实验条件

- 机器人：Unitree H1，19个有效关节，GO2 20槽位协议；物理电机ID 9未使用。
- 被测关节：`left_knee_joint`，物理电机ID 5。
- 实际增益：`Kp=300`，`Kd=6`。
- LowState：约500 Hz；LowCmd：50 Hz零阶保持。
- 机器人固定吊装，双脚完全离地。
- MotionSwitcher的 `name` 为空，高层 `ai` 服务未占用控制权。
- 每条主动实验均执行：6秒平滑进入准备姿态、2秒稳定、核心实验、6秒平滑返回。
- 准备姿态的腿部目标：左右髋横滚 `+0.12/-0.12 rad`，髋俯仰 `-0.30 rad`，膝 `+0.30 rad`，踝 `0 rad`。

## 准备姿态验证

原始记录：

```text
datasets/h1/raw/20261116_204017_h1_prepare/
```

结果：

- 采样率 `500.30 Hz`，最大状态间隔 `8.18 ms`；
- dropped/malformed均为0；
- 温度 `29–35 °C`；
- 左膝准备目标/实测为 `0.300/0.310 rad`；
- 右膝准备目标/实测为 `0.300/0.314 rad`；
- 全身最大跟踪误差 `0.0721 rad`，低于 `0.15 rad`保护阈值；
- 最大力矩约 `14.30 N·m`，出现在髋横滚打开双腿阶段。

## 方向阶跃与回差

诊断记录：

```text
datasets/h1/raw/20261116_204242_left_knee_joint_positive-step/
datasets/h1/raw/20261116_204506_left_knee_joint_negative-step/
datasets/h1/raw/20261116_204724_left_knee_joint_positive-step/
```

`+0.01 rad`阶跃只产生约 `+0.0003–0.0004 rad`运动。第一次 `-0.01 rad`阶跃产生约 `-0.00965 rad`运动，但回到基准命令后位置未恢复，第二次同方向阶跃只产生约 `-0.00066 rad`。单次 `+0.02 rad`阶跃产生约 `+0.00509 rad`运动，约52 ms达到实测稳态变化的90%。

这些结果证明左膝存在明显的重力偏置、静摩擦和方向回差。同方向重复阶跃不能视为两次独立、相同初态的激励，因此阶跃仅作为诊断数据，不作为首轮主要拟合轨迹。

## 正弦数据

| 用途 | 频率 | 原始记录 | 基频输出振幅 | 增益 | 相位滞后 |
|---|---:|---|---:|---:|---:|
| 拟合 | 0.5 Hz | `20261116_204944_left_knee_joint_sine` | 0.008902 rad | 0.445 | 28.71° |
| 验证 | 1.0 Hz | `20261116_205144_left_knee_joint_sine` | 0.009165 rad | 0.458 | 33.13° |
| 拟合 | 2.0 Hz | `20261116_205441_left_knee_joint_sine` | 0.010352 rad | 0.519 | 47.61° |

三条命令振幅均为 `0.02 rad`，持续10秒。稳定周期重复性良好；2 Hz记录包含20个周期，正弦段最大速度约 `0.211 rad/s`、最大左膝力矩约 `5.63 N·m`。

## 数据裁剪与转换

原始29秒记录包含准备和返回动作。为避免大幅准备动作主导左膝正弦损失，只保留每条记录的 `8–22 s`：2秒预保持、10秒正弦、2秒后保持。转换器先裁剪，再以500 Hz重采样；命令、增益和前馈力矩使用零阶保持，状态使用线性插值。

生成的数据：

```text
datasets/h1/real/left_knee/fit_sine_0p5hz.pt
datasets/h1/real/left_knee/fit_sine_2hz.pt
datasets/h1/real/left_knee/validation_sine_1hz.pt
datasets/h1/real/left_knee/validation_positive_step_0p02.pt
```

0.5 Hz和2 Hz只用于拟合；1 Hz和单次 `+0.02 rad` 阶跃完全不参与优化，只用于验证泛化。

## 未校准MuJoCo基线

| 数据 | 原始MuJoCo RMSE |
|---|---:|
| 0.5 Hz拟合 | 0.008326 rad |
| 1 Hz验证 | 0.008281 rad |
| 2 Hz拟合 | 0.008300 rad |

原始MuJoCo几乎完全跟随位置命令，而真机振幅明显更小并存在回差，形成了清楚的sim-to-real差距。

## CMA-ES配置

- 拟合轨迹：0.5 Hz和2 Hz；
- 评分关节：仅 `left_knee_joint`；
- 参数：`armature`、`frictionloss`、`damping`；
- 范围：armature `[0.001, 0.5]`，frictionloss `[0, 7]`，damping `[0, 7]`；
- population 16，最多40代，位置损失，命令延迟固定为0；
- 主随机种子：20261116；复核随机种子：42。

## 最终结果

主优化结果：

```text
armature    = 0.4240549348
frictionloss = 3.4116553297
damping     = 5.0249655603
```

第二随机种子结果：

```text
armature    = 0.4239256938
frictionloss = 3.4096590510
damping     = 5.0255091493
```

两个种子收敛到同一区域，结果未卡在设置边界。

## 独立验证

| 数据 | 原始RMSE | 辨识后RMSE | 降幅 |
|---|---:|---:|---:|
| 0.5 Hz拟合 | 0.008326 rad | 0.003739 rad | 55.1% |
| 1 Hz独立验证 | 0.008281 rad | 0.003710 rad | 55.2% |
| 2 Hz拟合 | 0.008300 rad | 0.003938 rad | 52.6% |
| +0.02 rad单次阶跃验证 | 0.007493 rad | 0.003789 rad | 49.4% |

独立1 Hz轨迹得到约55%的误差下降，独立单次阶跃也得到约49%的下降，因此结果不是只记住拟合频率或正弦波形。

主要输出：

```text
sysid_mujoco/results/h1/real_left_knee_cmaes_v1/best_params.json
sysid_mujoco/results/h1/real_left_knee_cmaes_v1/identified_fixed_base.xml
sysid_mujoco/results/h1/real_left_knee_cmaes_v1/eval_1hz/joint_trajectories.png
```

## 当前结论与限制

本实验完成了H1左膝的第一轮真实数据辨识闭环，证明平台能够让未参与优化的轨迹获得明显改善。当前参数可以作为悬空固定基座左膝模型的第一版校准参数。

仍需注意：

- 当前只辨识左膝，不能直接复制到髋、踝、右膝或上肢；
- 当前损失只使用关节位置，速度未进入目标函数；
- 当前没有单独辨识命令延迟；相位滞后同时包含动力学、摩擦和可能的延迟；
- MuJoCo的 `frictionloss` 不能完整表达真实传动回差，阶跃回中仍可能存在结构性误差；
- 参数是在当前吊装构型与PD增益下得到的等效结果，接入全身训练前仍需逐关节扩展并进行全身策略验证。
