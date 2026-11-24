# 实验三：H1右膝悬空实机辨识

## 目的

在与左膝实验相同的固定吊装、双脚离地、位置PD控制条件下，辨识
`right_knee_joint` 的等效 `armature`、`frictionloss` 和 `damping`，并用
未参与优化的1 Hz正弦和单次正阶跃检查泛化能力。

## 实验条件

- 被测关节：`right_knee_joint`，GO2物理电机ID 2；
- 实际增益：`Kp=300`、`Kd=6`；
- LowState约500 Hz，LowCmd为50 Hz零阶保持；
- 机器人固定吊装，双脚完全离地；
- 每条记录均包含6秒准备、2秒稳定、核心激励和6秒返回；
- 准备姿态和安全限制与左膝实验完全相同。

## 原始记录

```text
datasets/h1/raw/20261116_210913_right_knee_joint_positive-step/  # +0.01 rad诊断
datasets/h1/raw/20261116_211153_right_knee_joint_positive-step/  # +0.02 rad验证
datasets/h1/raw/20261116_211300_right_knee_joint_sine/           # 0.5 Hz拟合
datasets/h1/raw/20261116_211419_right_knee_joint_sine/           # 1.0 Hz验证
datasets/h1/raw/20261116_211540_right_knee_joint_sine/           # 2.0 Hz拟合
```

所有正式记录均为约500 Hz，`dropped=0`、`malformed=0`，右膝电机通信
丢失计数增量为0，温度保持30 °C。正弦实验中的最大右膝速度约
0.271 rad/s，估计力矩范围约为 -6.0 至 3.84 N·m。

## 阶跃诊断

`+0.01 rad`命令只产生约0.000585 rad平均位移，说明小信号受静摩擦或
死区显著影响。`+0.02 rad`命令产生约0.00572 rad平均位移，核心区间
峰峰值约0.01325 rad；返回基准命令后位置没有完全恢复，存在明显回差。

阶跃只作为独立验证，不进入首轮CMA-ES拟合。

## 正弦响应

| 用途 | 频率 | 基频输出振幅 | 增益 | 相位滞后 |
|---|---:|---:|---:|---:|
| 拟合 | 0.5 Hz | 0.012983 rad | 0.649 | 30.96° |
| 验证 | 1.0 Hz | 0.012516 rad | 0.626 | 35.09° |
| 拟合 | 2.0 Hz | 0.011607 rad | 0.582 | 41.97° |

三条命令的振幅均为0.02 rad。增益随频率升高而下降，相位滞后随频率
升高而增加，趋势一致且可辨认。

## 数据转换

正弦记录裁剪原始时间8--22秒，保留预保持、10秒激励和后保持；阶跃
记录裁剪8--15秒。转换后统一为500 Hz：

```text
datasets/h1/real/right_knee/fit_sine_0p5hz.pt
datasets/h1/real/right_knee/fit_sine_2hz.pt
datasets/h1/real/right_knee/validation_sine_1hz.pt
datasets/h1/real/right_knee/validation_positive_step_0p02.pt
```

## CMA-ES配置与边界复核

- 拟合数据：0.5 Hz和2 Hz；
- 独立验证：1 Hz和单次 `+0.02 rad` 阶跃；
- 评分关节：仅 `right_knee_joint`；
- population 16，40代，位置损失，命令延迟固定为0；
- armature范围 `[0.001, 0.5]`；
- frictionloss范围 `[0, 7]`；
- 首次damping范围 `[0, 7]`，随机种子20261116；
- 复核damping范围扩大为 `[0, 12]`，随机种子42。

首次结果为 `(0.436398, 3.058716, 6.798216)`。扩大阻尼上界并更换随机种子
后得到：

```text
armature     = 0.4363796724
frictionloss = 3.0587443653
damping      = 6.7996619323
```

两次损失均约为 `3.409523e-08`，参数收敛到同一区域。阻尼没有随扩大后的
上界继续增大，因此最终采用宽边界复核版。

## 独立验证

| 数据 | 原始MuJoCo RMSE | 辨识后RMSE | 降幅 |
|---|---:|---:|---:|
| 0.5 Hz拟合 | 0.008230 rad | 0.004215 rad | 48.8% |
| 1 Hz独立验证 | 0.008285 rad | 0.004507 rad | 45.6% |
| 2 Hz拟合 | 0.008155 rad | 0.005106 rad | 37.4% |
| +0.02 rad阶跃验证 | 0.007619 rad | 0.004840 rad | 36.5% |

未参与拟合的1 Hz和阶跃均明显改善，说明结果不是只记住两条拟合轨迹。
右膝改善幅度低于左膝，且真实响应存在方向回差，因此当前三个MuJoCo
参数仍只能视为当前吊装构型和PD增益下的等效近似。

## 输出

```text
sysid_mujoco/results/h1/real_right_knee_cmaes_v2_wide/best_params.json
sysid_mujoco/results/h1/real_right_knee_cmaes_v2_wide/identified_fixed_base.xml
sysid_mujoco/results/h1/real_right_knee_cmaes_v2_wide/eval_1hz/joint_trajectories.png
sysid_mujoco/results/h1/real_right_knee_cmaes_v2_wide/eval_positive_step/joint_trajectories.png
```

## 结论

右膝真实数据的采集、转换、多轨迹CMA-ES拟合和独立验证闭环已经完成。
下一步不应直接把单膝XML当作全身训练模型，而应继续辨识左右踝和髋，
最后把各关节结果合并到同一份H1 MJCF，再进行全身仿真回放和策略A/B验证。
