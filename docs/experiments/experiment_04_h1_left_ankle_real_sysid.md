# 实验四：H1左踝悬空实机辨识

## 目的与条件

在H1固定吊装、双脚离地条件下辨识 `left_ankle_joint` 的等效
`armature`、`frictionloss` 和 `damping`。左踝对应GO2物理电机ID 10，
实际位置PD增益为 `Kp=40`、`Kd=2`，LowState约500 Hz、LowCmd为50 Hz
零阶保持。每条记录均包含6秒准备、2秒稳定、核心激励和6秒返回。

这些参数只描述当前悬空构型和控制增益下的等效关节响应，不包含脚底接触、
鞋底柔性和地面摩擦。

## 幅值选择与阶跃回差

预先在MuJoCo中验证了 `0.02--0.05 rad` 阶跃以及 `0.03/0.04 rad` 的
0.5、1、2 Hz正弦，均无自碰撞且保有足够关节限位余量。

真机先执行 `±0.02 rad` 双向阶跃：正向移动约0.01778 rad，但回零后仍
残留约0.01714 rad；随后的负向只移动约-0.00331 rad。随后执行
`±0.03 rad`：正向移动约0.03287 rad，负向从中间回零位置移动约
-0.01035 rad，最终仍相对初始残留约0.02291 rad。这证明左踝具有明显的
静摩擦和方向回差。

0.03 rad、0.5 Hz正弦的基频响应仅约0.0050 rad，且前几周期持续改变中心
位置，因此只保留作幅值敏感性诊断。提高到0.04 rad后，第3--5周期峰峰值
稳定在约0.02343 rad，故正式正弦幅值确定为0.04 rad，没有升至0.05 rad。

## 原始记录

```text
datasets/h1/raw/20261116_212743_left_ankle_joint_step/  # ±0.02 rad诊断
datasets/h1/raw/20261116_212921_left_ankle_joint_step/  # ±0.03 rad验证
datasets/h1/raw/20261116_213038_left_ankle_joint_sine/  # 0.03 rad幅值诊断
datasets/h1/raw/20261116_213240_left_ankle_joint_sine/  # 0.04 rad, 0.5 Hz拟合
datasets/h1/raw/20261116_213356_left_ankle_joint_sine/  # 0.04 rad, 1.0 Hz验证
datasets/h1/raw/20261116_213524_left_ankle_joint_sine/  # 0.04 rad, 2.0 Hz拟合
```

正式正弦记录均无丢弃、无畸形、无状态序号缺口，ID 10通信丢失计数增量
为0，温度保持29 °C。2 Hz正弦最大速度约0.396 rad/s，估计力矩范围约
-1.93至1.21 N·m。

## 正弦响应

| 用途 | 频率 | 基频输出振幅 | 增益 | 相位滞后 |
|---|---:|---:|---:|---:|
| 拟合 | 0.5 Hz | 0.013224 rad | 0.331 | 47.43° |
| 验证 | 1.0 Hz | 0.013723 rad | 0.343 | 57.48° |
| 拟合 | 2.0 Hz | 0.013278 rad | 0.333 | 73.60° |

三条正式正弦的命令幅值均约0.04 rad。基频响应幅值一致，相位滞后随频率
增加，具备多频辨识所需的信息。

## 数据转换

正弦记录裁剪原始时间8--22秒；双向阶跃裁剪8--18秒。生成：

```text
datasets/h1/real/left_ankle/fit_sine_0p5hz.pt
datasets/h1/real/left_ankle/fit_sine_2hz.pt
datasets/h1/real/left_ankle/validation_sine_1hz.pt
datasets/h1/real/left_ankle/validation_step_pm0p03.pt
```

`±0.03 rad`阶跃在15.744秒的零位保持段出现一次11.43 ms调度间隔；状态
序号连续且无丢包。转换该文件时明确将最大间隔门限由中位间隔5倍放宽到
6倍，其他文件使用默认门限。

## CMA-ES配置与结果

- 拟合轨迹：0.5 Hz和2 Hz；
- 独立验证：1 Hz和 `±0.03 rad` 双向阶跃；
- 评分关节：仅 `left_ankle_joint`；
- population 16，40代，位置损失，命令延迟固定为0；
- armature `[0.001,0.5]`、frictionloss `[0,7]`、damping `[0,12]`。

随机种子20261116：

```text
armature     = 0.1518691436
frictionloss = 1.1275868518
damping      = 0.0237065708
```

随机种子42：

```text
armature     = 0.1520190446
frictionloss = 1.1271061501
damping      = 0.0306636863
```

两次损失均约 `2.7071e-05`。armature和frictionloss高度一致；damping绝对值
都很小，说明其在当前数据与参数化下影响较弱，不应把小数点后的差异解释成
精确物理阻尼。

## 独立验证

| 数据 | 原始MuJoCo RMSE | 辨识后RMSE | 降幅 |
|---|---:|---:|---:|
| 0.5 Hz拟合 | 0.020841 rad | 0.005100 rad | 75.5% |
| 1 Hz独立验证 | 0.019181 rad | 0.005390 rad | 71.9% |
| 2 Hz拟合 | 0.016449 rad | 0.005115 rad | 68.9% |
| 双向阶跃验证 | 0.019603 rad | 0.006482 rad | 66.9% |

未参与优化的1 Hz和双向阶跃均明显改善，说明结果具备轨迹泛化能力。辨识后
模型仍不能重现所有中心漂移和方向回差；这是单一MuJoCo `frictionloss` 参数
无法完整描述真实传动滞回的结构性限制。

## 输出与结论

### 1 Hz离地复测

为排除首次1 Hz记录可能触地的疑问，在明确确认脚底离地后，以完全相同的
`0.04 rad、1 Hz、10 s`设置重新采集：

```text
datasets/h1/raw/20261116_220747_left_ankle_joint_sine/
```

复测与原记录的去中心波形相关系数为0.9914，相位只差0.03°；响应幅值由
0.01372增至0.01487 rad（+8.3%），稳定周期峰峰值增加约11.2%。两次的
力矩范围完全相同，力矩RMS、IMU扰动、前后静态位移和其他关节耦合均接近，
没有发现原记录存在明显触地所应产生的削顶、相位突变或全身联动证据。

复测数据单独保存为：

```text
datasets/h1/real/left_ankle/validation_sine_1hz_rerun_clearance.pt
```

使用原辨识模型评估复测数据，RMSE由0.018722 rad降至0.004945 rad，下降
73.6%；与首次1 Hz验证的71.9%改善一致。因此左踝辨识结论对这次复测稳定。

主要输出：

```text
sysid_mujoco/results/h1/real_left_ankle_cmaes_v1/best_params.json
sysid_mujoco/results/h1/real_left_ankle_cmaes_v1/identified_fixed_base.xml
sysid_mujoco/results/h1/real_left_ankle_cmaes_v1/eval_1hz/joint_trajectories.png
sysid_mujoco/results/h1/real_left_ankle_cmaes_v1/eval_step_pm0p03/joint_trajectories.png
```

左踝采集、转换、拟合和独立验证闭环已经完成。该结果可作为悬空左踝模型
的第一版校准参数；在进入全身训练模型前，还需完成右踝及髋关节辨识，并将
各关节结果合并到同一份MJCF后做联合回放验证。
