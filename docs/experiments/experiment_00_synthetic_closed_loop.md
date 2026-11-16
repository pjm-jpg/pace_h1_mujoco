# 实验零：H1 左膝仿真软件闭环验证

## 1. 实验状态

- 结论：**通过（PASS）**
- 执行时间：2026-11-15 16:59:40 至 17:00:39（Asia/Shanghai）
- 项目分支：`pjm_pace`
- 环境：MuJoCo 3.13.0、PyTorch 2.15.0+cu132、SciPy 1.18.1
- 对象：固定基座、无接触的 H1，辨识关节为 `left_knee_joint`
- 数据性质：完全由 MuJoCo 生成的确定性、无噪声合成数据，不是真机数据

本实验验证的是从“生成记录”到“参数优化、写回模型、独立轨迹评估”的整条软件链路。这里的“闭环”包含位置 PD 控制形成的关节响应，以及辨识结果重新回放验证形成的软件工作闭环。

## 2. 实验目的

在连接真实 H1 之前，先回答以下问题：

1. H1 的 19 关节固定基座模型能否在 MuJoCo 中稳定运行；
2. 项目的统一 `.pt` 数据结构能否被拟合和评估程序正确读取；
3. CMA-ES 能否只根据位置指令和关节响应反推出左膝的三个隐藏动力学参数；
4. 找到的参数能否写回一个新的 MuJoCo XML；
5. 校准模型能否在未参与拟合的阶跃轨迹上优于原始模型。

## 3. 实验设置

### 3.1 模型和控制器

- H1 有效关节数：19
- 左膝在数据关节顺序中的索引：3
- 仿真和采样频率：500 Hz（时间步长 0.002 s）
- 左膝位置控制增益：`Kp=300`、`Kd=6`
- 前馈力矩：0
- 命令延迟：0 ms
- 基座：固定
- 碰撞：关闭

### 3.2 原始参数和隐藏真值

| 参数 | 原始 MuJoCo | 合成“虚拟真机”隐藏真值 | CMA-ES 搜索范围 |
|---|---:|---:|---:|
| `armature` | 0.1 | 0.22 | 0.001–0.5 |
| `frictionloss` | 0.0 | 0.35 | 0.0–3.0 |
| `damping` | 1.0 | 2.2 | 0.0–5.0 |

隐藏真值只用于产生合成测量数据和实验结束后的核对，`cmaes_fit.py` 不读取真值 JSON 或真值 XML。

### 3.3 拟合数据

- 文件：`datasets/h1/synthetic_left_knee_fit.pt`
- 时长：6.0 s
- 样本数：3001
- 激励：0.35、0.85、1.65 Hz 的平滑多频正弦叠加
- 设定幅值参数：0.22 rad
- 实际指令范围：-0.189471 至 0.206717 rad
- 合成测量响应范围：-0.189080 至 0.210327 rad
- 测量噪声标准差：0 rad

### 3.4 独立验证数据

- 文件：`datasets/h1/synthetic_left_knee_validation.pt`
- 时长：5.0 s
- 样本数：2501
- 激励：从未用于拟合的保持阶跃序列
- 指令区间：
  - 0–1.0 s：0 rad
  - 1.0–2.0 s：0.165 rad
  - 2.0–3.2 s：-0.099 rad
  - 3.2–4.2 s：0.077 rad
  - 4.2–5.0 s：0 rad
- 合成测量响应范围：-0.133475 至 0.185837 rad

拟合使用多频正弦，而验证使用阶跃，二者相互独立。这样可以检查参数是否描述了动力学，而不仅是记住一条拟合轨迹。

## 4. 执行过程

### 4.1 生成合成记录

```bash
python datasets/generate_h1_synthetic_dataset.py
```

程序先生成带隐藏参数的固定基座 H1 模型，再使用该模型分别运行拟合轨迹和验证轨迹，保存指令、位置、速度、PD 增益和时间轴。

### 4.2 CMA-ES 参数辨识

```bash
python sysid_mujoco/cmaes_fit.py \
  --robot h1 \
  --dataset datasets/h1/synthetic_left_knee_fit.pt \
  --joints left_knee_joint \
  --score-joints left_knee_joint \
  --population-size 16 \
  --max-iters 50 \
  --seed 7 \
  --output-dir sysid_mujoco/results/h1/synthetic_recovery
```

CMA-ES 每次提出一组 `armature`、`frictionloss` 和 `damping`，MuJoCo 用同一条多频正弦指令回放，损失函数计算左膝位置残差的均方值。速度权重为 0，因此本次优化目标只使用关节位置。

优化实际执行 46 代（迭代编号 0–45），每代 16 个候选参数。全局最优损失从第 0 代的 `3.3153e-08` 降至第 41 代的 `1.1151e-18`，之后没有出现更优解。

### 4.3 未见阶跃验证

```bash
python sysid_mujoco/eval_fit.py \
  --robot h1 \
  --dataset datasets/h1/synthetic_left_knee_validation.pt \
  --identified-model-xml \
    sysid_mujoco/results/h1/synthetic_recovery/identified_fixed_base.xml \
  --joints left_knee_joint \
  --output-dir sysid_mujoco/results/h1/synthetic_validation \
  --no-show
```

评估程序从验证数据的实测初始状态开始，分别用原始固定基座模型和校准模型回放同一条阶跃指令，然后与合成测量响应比较。

## 5. 辨识结果

| 参数 | 隐藏真值 | 辨识结果 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| `armature` | 0.22 | 0.219999779141 | -2.209e-7 | 0.0001004% |
| `frictionloss` | 0.35 | 0.349998363196 | -1.637e-6 | 0.0004677% |
| `damping` | 2.2 | 2.200004843524 | +4.844e-6 | 0.0002202% |

三个参数都恢复到了真值的 0.001% 以内。最终 XML 与名义 XML 的动力学差异只发生在 `left_knee_joint`：

```xml
<joint name="left_knee_joint"
       armature="0.219999779141"
       frictionloss="0.349998363196"
       damping="2.20000484352" />
```

## 6. 独立验证结果

| 模型 | 左膝位置 RMSE | 左膝位置 MAE |
|---|---:|---:|
| 原始 MuJoCo | 0.00711107 rad（0.407434°） | 0.00294788 rad（0.168901°） |
| 辨识后 MuJoCo | 1.41402e-8 rad（8.102e-7°） | 6.71997e-9 rad（3.850e-7°） |

- RMSE 降低：99.999801%
- RMSE 改善倍数：约 502,896 倍
- MAE 降低：99.999772%

### 6.1 曲线解读

![实验零左膝阶跃验证曲线](../../sysid_mujoco/results/h1/synthetic_validation/joint_trajectories.png)

- 橙色虚线 `desired`：阶跃位置指令；它不是期望仿真响应瞬间跳到的位置。
- 蓝色 `measured`：隐藏真值模型生成的合成测量，可把它理解为本实验里的“虚拟真机”。
- 红色 `original MuJoCo`：使用原始参数的模型响应。
- 绿色 `identified MuJoCo`：写入 CMA-ES 结果后的模型响应。

阶跃发生后，蓝色曲线存在超调和衰减振荡，这是受惯量、阻尼、摩擦和 PD 控制共同影响的正常动态响应。原始模型在图的整体尺度下已经看起来比较接近，但在每次阶跃后的峰值、振荡和收敛过程中仍有系统性偏差。绿色曲线几乎完全覆盖蓝色曲线；这与 `1.414e-8 rad` 的独立验证 RMSE 一致。

绿色曲线没有覆盖橙色阶跃线并不是失败。辨识目标是复现实际关节如何响应指令，而不是强迫物理状态无延迟、无超调地等于指令。

## 7. 本实验证明了什么

本实验在理想合成条件下证明：

1. H1 固定基座模型、19 关节顺序和左膝控制可以稳定运行；
2. 合成数据生成器输出的数据能被统一加载和回放；
3. MuJoCo SysID 残差、CMA-ES 搜索和参数边界连接正确；
4. 三个待辨识参数确实会影响左膝瞬态响应，并且在本次激励下能够被优化器恢复；
5. 最优参数可以正确写入新的固定基座 XML；
6. 从多频正弦得到的参数可以预测未参与拟合的阶跃响应；
7. 评估程序可以同时输出原始模型、校准模型的定量指标和对比曲线。

因此，进入真实 DDS 数据实验前，软件主链路已经具备可用基础。

## 8. 本实验没有证明什么

近乎零误差是预期的理想结果，不应直接理解成真实 H1 也能达到同样精度。本实验没有验证：

1. 真实 H1 的 `armature`、`frictionloss`、`damping` 就是本实验的隐藏数值；
2. DDS 的 20 槽位到 19 关节映射、无效电机 ID 9 排除和真实时间戳已经正确；
3. 实机通信延迟、命令保持、丢包、编码器噪声和滤波已经建模；
4. 电机饱和、死区、温度、减速器柔性、回差和库仑摩擦都能被这三个参数充分表达；
5. 多关节同时运动时的耦合也能达到相同精度；
6. 地面接触、足底摩擦、浮动基座和完整行走策略已经校准；
7. 校准模型已经可以直接用于策略训练。本实验输出的是固定基座、关闭碰撞的辨识模型。

此外，合成数据与拟合回放使用同一类 MuJoCo 模型，且没有噪声，属于“模型结构完全匹配”的软件自检。真实实验通常不会得到如此小的误差。

## 9. 生成文件及用途

| 文件 | 内容和用途 |
|---|---|
| `datasets/h1/synthetic_left_knee_truth.xml` | 带隐藏真值的合成数据生成模型，只用于实验零 |
| `datasets/h1/synthetic_left_knee_truth.json` | 记录真值、名义值、轨迹和随机种子，优化器不读取 |
| `datasets/h1/synthetic_left_knee_fit.pt` | 6 秒多频正弦拟合记录 |
| `datasets/h1/synthetic_left_knee_validation.pt` | 5 秒独立阶跃验证记录 |
| `sysid_mujoco/results/h1/synthetic_recovery/optimization_log.csv` | 每代最佳损失、全局最佳损失和参数 |
| `sysid_mujoco/results/h1/synthetic_recovery/best_params.json` | 最终参数、损失、关节、数据源和随机种子 |
| `sysid_mujoco/results/h1/synthetic_recovery/nominal_fixed_base.xml` | 本次回放使用的原始固定基座模型 |
| `sysid_mujoco/results/h1/synthetic_recovery/identified_fixed_base.xml` | 写入辨识参数后的固定基座模型 |
| `sysid_mujoco/results/h1/synthetic_validation/metrics.json` | 原始模型和辨识模型的 RMSE、MAE |
| `sysid_mujoco/results/h1/synthetic_validation/joint_trajectories.png` | 指令、合成测量、原始模型和辨识模型的对比图 |

这些合成数据和运行结果由 `.gitignore` 排除，不会默认提交到 Git。本文档记录了关键配置和结果；如需长期归档完整原始产物，应另行保存本次结果目录。

## 10. 最终结论和下一步

实验零判定为 **PASS**。当前结果足以说明软件辨识平台在无噪声、模型匹配的合成条件下工作正确，不需要继续通过调整算法来重复追求更小的合成误差。

下一步进入实验一：连接 H1 后只订阅和记录 LowState/DDS 状态，不发送运动命令。首先验证通信频率、时间戳、19 关节顺序、物理电机 ID 9 排除、数据完整性和日志转换，暂不进行任何参数拟合。
