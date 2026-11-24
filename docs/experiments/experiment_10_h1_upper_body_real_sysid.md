# 实验十：H1上肢与躯干真机参数辨识

## 实验目的

在固定吊装条件下完成H1剩余9个关节的真机辨识，并与已经完成的10个下肢关节合并，形成19自由度完整校准模型。本实验辨识：

- 左右肩俯仰、侧摆、旋转；
- 左右肘；
- 躯干旋转。

每个关节只优化MuJoCo中的等效 `armature`、`frictionloss` 和 `damping`。控制增益使用真机日志中实际记录的固定Kp/Kd，命令延迟固定为0；已辨识的下肢参数在上肢拟合期间保持冻结。

## 正式数据选择

正式清单为：

`datasets/h1/real/upper_body_official_datasets.json`

清单显式固定27个完成状态的真机日志：每关节一条双向阶跃、一条0.5/1/2 Hz分段正弦和一条独立1 Hz正弦。全部日志满足：

- `result=completed`、`command_available=true`；
- `dropped=0`、`malformed=0`；
- 中位接收频率范围约500.20--505.75 Hz；
- 最大单次时间间隔为23.79 ms，出现在右肩俯仰独立1 Hz记录；该间隔前后命令连续，保留并在清单中记录。

下列记录不进入正式集合：被动observe、失败或中止的prepare、旧手臂姿态数据，以及被新记录替代的左肘早期阶跃。

## 数据拆分

统一转换脚本为：

```bash
/home/pjm/.pixi/bin/pixi run -e mujoco \
  python datasets/prepare_h1_upper_body_datasets.py
```

每个关节生成4条正式数据和1条诊断数据：

| 数据 | 原始记录时间段 | 用途 |
|---|---:|---|
| `fit_sine_0p5hz.pt` | 8--22 s | CMA-ES拟合 |
| `fit_sine_2hz.pt` | 32--46 s | CMA-ES拟合 |
| `validation_sine_1hz.pt` | 独立正弦8--22 s | 独立频段验证 |
| `validation_step_pm*.pt` | 阶跃8--24 s | 独立瞬态验证 |
| `diagnostic_multisine_1hz_internal.pt` | 分段正弦20--34 s | 同源诊断，不计作独立验证 |

因此正式集合共36条：18条拟合、18条独立验证。内部1 Hz片段与拟合频段来自同一条原始记录，只作为诊断，避免数据泄漏。

## 基线模型

上肢辨识不是从完全原始H1模型重新开始，而是以：

`robot_model/h1/h1_calibrated_lower_body.xml`

为底座。这样10个下肢参数保持固定，上肢关节不会吸收已知的下肢模型误差。上肢仍使用辨识前标称参数的基线结果保存在：

`sysid_mujoco/results/h1/upper_body_baseline/`

独立验证基线平均RMSE为 `0.0064868 rad`。误差最大的关节组是肩俯仰和肩侧摆，验证平均RMSE约为 `0.0090--0.0131 rad`。

## CMA-ES配置

- 每个关节独立拟合0.5和2 Hz两条轨迹；
- population 16，40代，固定随机种子20261117；
- 仅目标关节位置进入损失；
- 速度权重0，命令延迟0 ms；
- 初始范围：`armature=[0.001, 0.5]`、`frictionloss=[0, 3]`、`damping=[0, 5]`；
- 左右肩旋转和躯干的armature触及0.001下界，因此用 `armature=[0, 0.5]` 重新做边界复核并采用复核结果。

正式运行入口：

```bash
/home/pjm/.pixi/bin/pixi run -e mujoco \
  python sysid_mujoco/run_h1_upper_body_cmaes.py --workers 3
```

## 识别参数

| 关节 | armature | frictionloss | damping |
|---|---:|---:|---:|
| left_elbow_joint | 0.030913 | 0.269525 | 0.000112 |
| right_elbow_joint | 0.044161 | 0.303969 | 0.000026 |
| left_shoulder_pitch_joint | 0.023283 | 1.252861 | 1.261430 |
| right_shoulder_pitch_joint | 0.026403 | 1.432086 | 0.000019 |
| left_shoulder_roll_joint | 0.079740 | 1.087002 | 0.000439 |
| right_shoulder_roll_joint | 0.138805 | 1.022279 | 0.000030 |
| left_shoulder_yaw_joint | 0.000491 | 0.340624 | 0.308015 |
| right_shoulder_yaw_joint | 0.000002 | 0.175392 | 0.544709 |
| torso_joint | 0.000001 | 1.976914 | 0.087830 |

接近0的参数应解释为当前控制、姿态和频率范围内的等效最优值，不等同于真实电机转子惯量或机械摩擦的直接测量。

## 独立验证结果

三曲线/四曲线回放（期望、真机、辨识前MuJoCo、校准MuJoCo）和逐数据指标位于：

`sysid_mujoco/results/h1/upper_body_identified/`

| 关节 | 验证基线RMSE | 校准RMSE | 改善 |
|---|---:|---:|---:|
| 左肘 | 0.003862 | 0.003620 | 6.3% |
| 右肘 | 0.003402 | 0.003125 | 8.1% |
| 左肩俯仰 | 0.009335 | 0.003849 | 58.8% |
| 右肩俯仰 | 0.013088 | 0.009827 | 24.9% |
| 左肩侧摆 | 0.009001 | 0.006590 | 26.8% |
| 右肩侧摆 | 0.010484 | 0.008813 | 15.9% |
| 左肩旋转 | 0.003092 | 0.001960 | 36.6% |
| 右肩旋转 | 0.002569 | 0.001756 | 31.7% |
| 躯干 | 0.003549 | 0.001291 | 63.6% |

汇总结果：

- 18条拟合数据：平均RMSE `0.007702 -> 0.004699 rad`，改善39.0%；
- 18条独立验证：平均RMSE `0.006487 -> 0.004537 rad`，改善30.1%；
- 18/18条独立验证均改善，没有退化记录。

## 完整19关节模型

最终浮动基座模型、场景和参数来源清单为：

- `robot_model/h1/h1_calibrated_full_body.xml`
- `robot_model/h1/scene_flat_calibrated_full_body.xml`
- `robot_model/h1/h1_calibrated_full_body_params.json`

参数清单包含下肢10关节和本实验9关节，共19关节。离线结构与数值检查结果位于：

`sysid_mujoco/results/h1/full_body_combined_v1/floating_model_validation.json`

检查确认：拓扑与标称H1一致、保留浮动基座、19个关节参数写回误差小于 `5e-12`、3秒仿真全程有限且无MuJoCo warning。

## 结论与局限

实验完成了从真机DDS日志、正式数据冻结、分段转换、基线回放、CMA-ES辨识、独立验证到19关节模型合并的闭环。结果证明这些等效关节参数在吊装、固定基座和当前小幅激励范围内，整体上比标称模型更接近真机。

这仍不证明模型已经准确描述落地接触、浮动基座耦合、行走大动作或策略部署安全性。下一阶段应先在仿真中进行完整模型稳定性、接触和策略回归测试，再考虑受保护的真机策略验证。
