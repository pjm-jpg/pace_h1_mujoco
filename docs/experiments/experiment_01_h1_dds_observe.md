# 实验一：H1 DDS 被动状态采集

## 状态

- 软件功能：已完成并通过自动测试
- 实机采集：尚未执行
- 目标：只订阅 `rt/lowstate`，不调用 `send_low_command()`，不发布 `rt/lowcmd`
- 默认时长：10 s
- 预计接收频率：约 500 Hz

## 实验目的

本实验只验证真实 H1 的 DDS 状态链路、20槽位数据和19关节映射。它没有命令轨迹和动态激励，不能用于参数辨识。

需要验证：

1. 能稳定收到 `rt/lowstate`；
2. 时间戳严格递增，接收频率接近500 Hz；
3. `q_real`、`dq_real`、`tau_est`、温度、通信丢包计数和IMU字段完整；
4. 原始数据包含20个物理电机槽位；
5. 物理电机ID 9不在19关节映射中；
6. 记录期间没有malformed、dropped或明显长间隔；
7. 本程序没有发送LowCmd。

## 已实现的软件保护

Holosoma的 `observe` 路径与 `hold/step/chirp` 路径分离。observe路径不会调用 `send_low_command()`，并使用独立确认词 `H1_OBSERVE`。

被动日志使用以下不可混淆标记：

```text
command_available = false
cmd_seq = 0
t_cmd_ns = 0
q_cmd, dq_cmd, kp, kd, tau_ff = NaN
publish_lowcmd = false
```

PACE的DDS转换器会拒绝 `command_available=false` 或 `cmd_seq` 全零的记录，防止被动日志进入CMA-ES。

## 实验前准备

1. H1必须放在可靠的固定支架或悬挂架上；
2. 急停必须在操作人员手边；
3. 停止所有策略、Unitree示例和其他 `rt/lowcmd` 发布程序；
4. 确定连接H1的有线网卡名称：

```bash
ip -br addr
```

5. 激活Holosoma实机环境：

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_inference_setup.sh

which python
python -c "import holosoma_inference, unitree_sdk2py; print('H1 environment: OK')"
```

Python应来自：

```text
/home/pjm/.holosoma_deps/miniconda3/envs/hsinference/bin/python
```

## 第一步：只预览计划

将 `<H1_NETWORK_INTERFACE>` 替换成实际网卡名称：

```bash
PYTHONPATH=src/holosoma_inference \
python -m holosoma_inference.collect_h1_sysid \
  --kind observe \
  --observe-duration-s 10 \
  --interface <H1_NETWORK_INTERFACE> \
  --output-root \
  /home/pjm/Desktop/humanoid-projects/pace/sim2real-robot-identification/datasets/h1/raw
```

预览必须显示：

```json
"kind": "observe",
"total_duration_s": 10.0,
"publish_lowcmd": false,
"command_available": false
```

预览命令没有 `--execute`，不会初始化实机DDS接口。

## 第二步：执行10秒被动采集

确认机器人固定、急停可用、其他LowCmd发布者全部关闭，并且预览内容正确后执行：

```bash
PYTHONPATH=src/holosoma_inference \
python -m holosoma_inference.collect_h1_sysid \
  --kind observe \
  --observe-duration-s 10 \
  --interface <H1_NETWORK_INTERFACE> \
  --output-root \
  /home/pjm/Desktop/humanoid-projects/pace/sim2real-robot-identification/datasets/h1/raw \
  --execute \
  --confirm H1_OBSERVE
```

输出目录形式：

```text
datasets/h1/raw/<timestamp>_h1_observe/
├── manifest.json
└── raw.npz
```

如果LowState超过20 ms没有更新或温度超过70°C，采集会中止。observe中止路径同样不会发送保持命令。GO消息中的 `MotorState.lost` 是累计通信丢包计数：历史值允许非零，采集结束后检查它在10秒内是否增长。

## 第三步：自动检查日志

退出Holosoma环境后，进入PACE的MuJoCo环境：

```bash
cd /home/pjm/Desktop/humanoid-projects/pace/sim2real-robot-identification

/home/pjm/.pixi/bin/pixi run -e mujoco \
python datasets/analyze_h1_observe_log.py \
  datasets/h1/raw/<timestamp>_h1_observe/raw.npz
```

程序在同一目录生成：

```text
state_report.json
state_overview.png
```

报告自动统计：

- 样本数、实际时长、中位频率；
- 中位时间间隔、99%时间间隔、最大间隔；
- 非递增时间戳数量；
- malformed和dropped数量；
- 20个物理电机的位置、速度、估计力矩、温度和累计通信丢包计数；
- 19个关节到物理电机的映射；
- ID 9的单独状态；
- IMU四元数模长；
- 所有状态字段的NaN/Inf数量；
- LowCmd不可用标记和命令计数；
- 每项PASS/FAIL和总判定。

## 默认验收标准

| 检查项 | 默认通过条件 |
|---|---|
| manifest结果 | `completed` |
| 命令发布标记 | `publish_lowcmd=false` |
| 命令数量 | `commands=0` |
| 命令字段 | 全部NaN，且 `cmd_seq=t_cmd_ns=0` |
| 电机数据维度 | `N × 20` |
| H1映射 | 19个唯一电机ID，不包含ID 9 |
| 时间戳 | 严格递增 |
| 中位频率 | 450–550 Hz |
| 最大时间间隔 | 不超过20 ms |
| malformed/dropped | 均为0 |
| 状态NaN/Inf | 0 |
| 有效电机通信丢包计数 | 历史值可非零；默认允许10个背景突发计数，再按每秒0.5个累计 |
| 最高温度 | 不超过70°C |
| IMU四元数模长 | 0.9–1.1 |
| 实际时长 | 与计划时长偏差不超过0.1 s或2% |

如果总结果为FAIL，不进入保持或运动实验。先根据失败项目检查网卡、DDS频率、时间戳、温度、通信丢包计数或映射。

## 实验通过后能说明什么

通过只说明真实H1的状态数据能够可靠进入软件，并且物理槽位到模型关节的映射正确。它不说明MuJoCo参数准确，也不说明CMA-ES能够拟合真机。

实验一通过后的下一步是实验二：当前位置保持。实验二才首次发送命令，用来验证命令记录、实际Kp/Kd、命令—状态时间对齐和安全接管。
