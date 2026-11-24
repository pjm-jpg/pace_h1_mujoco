# 实验五：H1右踝悬空实机辨识

## 目的与条件

辨识 `right_ankle_joint` 在H1固定吊装、双脚离地和真实位置PD增益下的
等效 `armature`、`frictionloss` 和 `damping`。右踝对应GO2物理电机ID 11，
实际增益为 `Kp=40`、`Kd=2`。LowState约500 Hz，LowCmd为50 Hz零阶保持。

## 阶跃与幅值选择

`±0.02 rad`阶跃中，正向实际移动约0.02007 rad，随后负向移动约
-0.00838 rad，最终残留约0.01184 rad。提高到 `±0.03 rad` 后，正向和
负向分别移动约+0.03228和-0.03220 rad，最终相对初始仅残留0.00149 rad。
因此正式正弦幅值采用0.03 rad，不需要升至0.04或0.05 rad。

## 正式原始记录

用户在首轮0.5/1/2 Hz采集后重新采集了三条时序质量更好的记录。正式转换、
拟合和验证只使用以下新记录：

```text
datasets/h1/raw/20261116_215341_right_ankle_joint_sine/  # 0.5 Hz拟合
datasets/h1/raw/20261116_215301_right_ankle_joint_sine/  # 1.0 Hz验证
datasets/h1/raw/20261116_215121_right_ankle_joint_sine/  # 2.0 Hz拟合
datasets/h1/raw/20261116_214530_right_ankle_joint_step/  # ±0.03 rad验证
```

较早的 `214647/214824/214952` 三条正弦只保留原始记录，不参与最终结果。
一次使用旧数据启动的优化被中断，最终结果写入独立的 `v3_allnew` 目录，未
混用任何旧转换数据。

新三条正弦均无丢弃、无畸形、无状态序号缺口；最大状态间隔小于3.64 ms，
ID 11通信丢失计数增量为0，温度为29--30 °C。

## 正弦响应

| 用途 | 频率 | 基频输出振幅 | 增益 | 相位滞后 |
|---|---:|---:|---:|---:|
| 拟合 | 0.5 Hz | 0.019942 rad | 0.665 | 42.47° |
| 验证 | 1.0 Hz | 0.019821 rad | 0.661 | 48.37° |
| 拟合 | 2.0 Hz | 0.018277 rad | 0.611 | 69.44° |

增益随频率上升而下降，相位滞后随频率上升而增加，三条记录趋势一致。2 Hz
正弦最大速度约0.394 rad/s，估计力矩范围约-1.29至0.71 N·m。

## 转换数据

正弦记录裁剪原始时间8--22秒，双向阶跃裁剪8--18秒，统一重采样为500 Hz：

```text
datasets/h1/real/right_ankle/fit_sine_0p5hz.pt
datasets/h1/real/right_ankle/fit_sine_2hz.pt
datasets/h1/real/right_ankle/validation_sine_1hz.pt
datasets/h1/real/right_ankle/validation_step_pm0p03.pt
```

## CMA-ES结果

- 拟合：0.5 Hz和2 Hz；
- 独立验证：1 Hz和 `±0.03 rad` 阶跃；
- population 16，40代，位置损失，命令延迟固定为0；
- armature `[0.001,0.5]`、frictionloss `[0,7]`、damping `[0,12]`。

随机种子20261116：

```text
armature     = 0.1065326821
frictionloss = 0.5776230209
damping      = 0.0003937205
```

随机种子42：

```text
armature     = 0.1065500575
frictionloss = 0.5775742149
damping      = 0.0001393557
```

两次最终损失均约 `7.7771e-06`，armature和frictionloss差异小于0.02%。
damping在两次运行中都接近零，说明它在当前数据与模型参数化下是弱可辨识项，
不应将该小数视为精确的物理阻尼常数。

## 独立验证

| 数据 | 原始MuJoCo RMSE | 辨识后RMSE | 降幅 |
|---|---:|---:|---:|
| 0.5 Hz拟合 | 0.011659 rad | 0.002947 rad | 74.7% |
| 1 Hz独立验证 | 0.009620 rad | 0.003051 rad | 68.3% |
| 2 Hz拟合 | 0.008579 rad | 0.003666 rad | 57.3% |
| 双向阶跃验证 | 0.012498 rad | 0.004922 rad | 60.6% |

未参与优化的1 Hz和双向阶跃均明显改善，证明结果并非只记住拟合频率。
辨识后模型仍不能完全复现中心偏置和传动回差，这属于当前单一
`frictionloss` 参数化的结构性限制。

## 输出与结论

```text
sysid_mujoco/results/h1/real_right_ankle_cmaes_v3_allnew/best_params.json
sysid_mujoco/results/h1/real_right_ankle_cmaes_v3_allnew/identified_fixed_base.xml
sysid_mujoco/results/h1/real_right_ankle_cmaes_v3_allnew/eval_1hz/joint_trajectories.png
sysid_mujoco/results/h1/real_right_ankle_cmaes_v3_allnew/eval_step_pm0p03/joint_trajectories.png
```

右踝采集、转换、拟合和独立验证闭环已经完成。至此左右膝与左右踝均已有
第一版实机校准结果。下一阶段应设计髋关节实验；在全部下肢关节完成前，
不要把任一单关节XML直接作为最终全身训练模型。
