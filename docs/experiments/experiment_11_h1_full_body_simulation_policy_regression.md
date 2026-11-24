# 实验十一：H1 全身校准模型仿真集成与策略回归

## 一、实验目的

实验十已经把 19 个关节的真实响应参数写入完整 H1 MuJoCo 模型。本实验不再采集真机数据，而是回答三个问题：

1. 全身校准模型是否保持原模型的拓扑、关节顺序和控制接口；
2. 全身关节同时运动、自由落体和足地接触时，模型是否数值稳定；
3. Holosoma 原生 FastSAC 策略评估链路能否直接加载全身校准模型并完成闭环仿真。

## 二、实验对象

- 标称模型：`robot_model/h1/h1.xml`
- 下肢校准模型：`robot_model/h1/h1_calibrated_lower_body.xml`
- 全身校准模型：`robot_model/h1/h1_calibrated_full_body.xml`
- 策略检查点：`model_0050000.pt`，FastSAC 第 50000 次迭代
- 正式策略运行器：Holosoma `holosoma.eval_agent`
- 仿真后端：MuJoCo Classic CPU
- 物理频率：200 Hz
- 策略频率：50 Hz

## 三、实验步骤

### 1. 模型结构一致性检查

加载标称、下肢校准和全身校准三个模型，比较自由度、刚体、碰撞几何、传感器、关节名称及执行器顺序。

结果完全一致：`nq=26`、`nv=25`、`nu=19`、20 个关节（含自由关节）、23 个刚体、41 个几何体和 21 个传感器。相对标称模型，下肢模型修改 10 个关节参数，全身模型修改 19 个关节参数。

### 2. 悬空全身多频跟踪

关闭重力并让机器人离开地面，19 个关节同时执行 0.5、1 和 2 Hz 的小幅多频运动。记录跟踪误差、基座反作用位移、控制力矩、数值异常和 MuJoCo warning。

| 模型 | 全身跟踪 RMSE | 峰值 RMS 误差 | MuJoCo warning |
| --- | ---: | ---: | ---: |
| 标称 | 0.006777 rad | 0.009595 rad | 0 |
| 下肢校准 | 0.009600 rad | 0.012510 rad | 0 |
| 全身校准 | 0.010503 rad | 0.012517 rad | 0 |

三个模型全部保持有限数值，峰值均远低于 0.10 rad 的集成门限。这里的误差差别不是辨识精度排名；输入是统一的人造全身激励，校准模型具有更真实的阻尼、摩擦和转子惯量，响应不应与标称模型相同。

### 3. 自由落体与接触检查

从相同姿态抬高 0.05 m 后释放，并用开环关节 PD 保持关节姿态。三个模型均在约 0.102 s 首次接触地面，全程没有 NaN、Inf 或 MuJoCo warning。

机器人随后倾倒是预期结果：自由基座人形机器人不能依靠固定关节目标的开环 PD 保持平衡。标称、下肢和全身模型的倾倒时刻分别约为 1.428、1.454 和 1.456 s，因此该指标只描述接触演化，不作为模型失败判据。

### 4. Holosoma 原生策略链路回归

使用同一 FastSAC 检查点，分别加载 Holosoma 标称 H1 XML 和本项目全身校准 XML。由于本机没有 CUDA，原检查点默认的 Warp GPU 后端不能启动，所以改用 Holosoma 已支持的 MuJoCo Classic CPU 后端；其余策略配置保持来自检查点。

两个模型均完成 500 个 headless evaluation step，进程返回码均为 0：

| 模型 | 官方评估步数 | 返回码 | 结果 |
| --- | ---: | ---: | --- |
| Holosoma 标称 H1 | 500 | 0 | 通过兼容性冒烟测试 |
| 全身校准 H1 | 500 | 0 | 通过兼容性冒烟测试 |

这证明完整校准 XML 可以被现有 Holosoma 策略栈直接加载，关节、执行器、观测构造和仿真循环没有结构性或运行时错误。

### 5. 为什么不采用自写 ONNX 回放结果

早期诊断脚本直接把 ONNX 输出当作最终关节动作，两个模型都会跌倒。进一步检查确认，FastSAC 正式链路还会恢复检查点中的观测归一化和动作边界变换；原始 ONNX 元数据不足以单独重建这两部分。因此该结果只能说明“简化回放器不完整”，不能用来判断校准模型或策略优劣。正式结论只采用 Holosoma 原生评估器。

## 四、实验结论

实验十一通过，结论限定如下：

- 全身校准模型保持了原 H1 的完整拓扑和 19 关节控制接口；
- 全身联合运动和足地接触均可计算，未出现数值异常；
- 现有 Holosoma FastSAC 策略评估链路可以加载全身校准模型并完成 500 步闭环冒烟测试；
- 这一阶段不需要再次连接真机。

本实验尚未证明校准模型上的策略奖励、稳定时间或行走速度优于标称模型，因为当前 headless evaluator 不输出这些量，也不能仅凭返回码确认过程中是否发生环境 reset。下一阶段应在校准模型中训练或微调策略，并增加明确的奖励、跌倒次数、存活时间、速度跟踪误差和能耗统计，再与标称模型做定量 A/B 验证。

## 五、复现实验

```bash
/home/pjm/.pixi/bin/pixi run -e mujoco \
  python sysid_mujoco/validate_h1_full_body_regression.py

/home/pjm/.pixi/bin/pixi run -e mujoco \
  python sysid_mujoco/run_h1_holosoma_policy_regression.py
```

输出位置：

- `sysid_mujoco/results/h1/full_body_regression_v1/regression_summary.json`
- `sysid_mujoco/results/h1/full_body_regression_v1/regression_comparison.png`
- `sysid_mujoco/results/h1/holosoma_policy_smoke_v1/summary.json`
- `sysid_mujoco/results/h1/holosoma_policy_smoke_v1/nominal.log`
- `sysid_mujoco/results/h1/holosoma_policy_smoke_v1/full_body_calibrated.log`
