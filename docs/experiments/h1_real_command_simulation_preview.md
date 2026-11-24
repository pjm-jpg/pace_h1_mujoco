# H1真机实验命令的MuJoCo预演

## 目的

在向真实H1发送任何命令前，先在固定基座MuJoCo模型中执行同一套命令时序，检查目标关节、物理电机映射、PD增益、幅值、频率、持续时间、跟踪误差和速度。

预演器直接从Holosoma导入 `H1SysidPlan`，不单独复制阶跃、单频正弦和扫频公式。因此真机采集器改变时序后，预演器会读取新的定义。预演器不初始化DDS，不能向真机发送命令。

## 与真机命令一致的部分

- H1 19关节顺序及20槽位物理电机映射；
- 实际Holosoma运行增益；
- 50 Hz LowCmd更新频率；
- 命令之间使用零阶保持；
- 6秒五次平滑插值进入悬空准备姿态；
- 2秒准备姿态稳定保持；
- 左右髋横滚分别为 `+0.12/-0.12 rad`，使双腿分开；
- 髋俯仰、膝、踝为 `-0.30/+0.30/0 rad`，使小腿接近竖直；
- 2秒预保持；
- 阶跃的正向、回零、负向、回零和重复次数；
- 单独正向阶跃和单独负向阶跃；
- 单频正弦的频率、持续时间和淡入淡出；
- chirp的起止频率、相位和淡入淡出；
- 2秒后保持；
- 6秒五次平滑插值返回启动姿态；
- 采集器异常/完成时使用的1秒最终安全保持；
- 0.15 rad跟踪误差保护阈值。

MuJoCo以500 Hz运行。当前预演使用名义固定基座模型，因此它用于检查命令设计和软件时序，不代表真实H1一定具有相同响应。

## 先验证悬空准备姿态

正式激励前先单独预演 `prepare`。它只执行“进入准备姿态、稳定、返回”，不会叠加单关节激励：

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind prepare --joint left_knee_joint
```

时序为6秒进入、2秒稳定、6秒返回，随后预演器再保持最终姿态1秒。准备姿态只改动双腿十个关节；躯干和手臂保持启动值。安全检查同时要求：无MuJoCo自碰撞、所有关节保留正的限位余量、全身最大跟踪误差不超过阈值。

## 可视化命令

进入PACE环境：

```bash
cd /home/pjm/Desktop/humanoid-projects/pace/sim2real-robot-identification
/home/pjm/.pixi/bin/pixi shell -e mujoco
```

### 当前位置保持

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind hold \
  --joint left_knee_joint
```

机器人应该保持初始位置。由于重力和名义模型误差，实际关节位置可能与零目标存在很小的稳态偏差。

### 默认小幅左右膝阶跃

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind step \
  --joint left_knee_joint \
  --amplitude-rad 0.01
```

核心阶跃时序为：

```text
0–6 s      平滑进入悬空准备姿态
6–8 s      准备姿态稳定
8–10 s     核心预保持
10–22 s    正负阶跃及回中
22–24 s    核心后保持
24–30 s    平滑返回启动姿态
30–31 s    最终安全保持（仅预演器）
```

`0.01 rad`约为 `0.573°`，在窗口中运动很小，这是计划中的真实首轮安全幅值。曲线比肉眼观察更容易判断响应。

### 分方向阶跃预演

真机实验应先把两个方向分开，以便发现重力、摩擦、线缆或机械结构造成的不对称。仿真命令为：

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind positive-step --joint left_knee_joint --amplitude-rad 0.01

python sysid_mujoco/preview_h1_real_experiment.py \
  --kind negative-step --joint left_knee_joint --amplitude-rad 0.01
```

每条记录的核心实验均为2秒预保持、两次“2秒阶跃+1秒回零”和2秒后保持。外层再包含6秒准备、2秒稳定、6秒返回和预演器1秒最终保持，共25秒。正向记录只出现 `+0.01 rad`，负向记录只出现 `-0.01 rad`。

### 单频正弦预演

两个方向的阶跃均通过后，先运行0.5 Hz、0.01 rad的小幅单频正弦：

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind sine --joint left_knee_joint --amplitude-rad 0.01 \
  --sine-frequency-hz 0.5 --sine-duration-s 10
```

核心命令包含2秒预保持、10秒正弦和2秒后保持；加上准备、稳定、返回和预演器最终保持后共29秒。正弦开始和结束各有0.5秒淡入淡出。随后可分别预演1、2、3 Hz，每个频率写入独立目录，不会覆盖已有结果：

```bash
python sysid_mujoco/preview_h1_real_experiment.py --kind sine --sine-frequency-hz 1 --headless
python sysid_mujoco/preview_h1_real_experiment.py --kind sine --sine-frequency-hz 2 --headless
python sysid_mujoco/preview_h1_real_experiment.py --kind sine --sine-frequency-hz 3 --headless
```

50 Hz命令频率下，软件拒绝超过5 Hz的单频正弦和chirp终止频率。

### 扫频预演

只有小阶跃实机实验通过后，才考虑实机扫频。仿真预演命令为：

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind chirp \
  --joint left_knee_joint \
  --amplitude-rad 0.02 \
  --chirp-f0-hz 0.5 \
  --chirp-f1-hz 5.0
```

扫频核心时序为2秒预保持、20秒扫频和2秒后保持；加上准备、稳定、返回和预演器最终保持后共39秒。

## 快速无界面检查

给任意命令加上 `--headless`，即可不打开窗口并快速生成结果：

```bash
python sysid_mujoco/preview_h1_real_experiment.py \
  --kind step --joint left_knee_joint --amplitude-rad 0.01 --headless
```

每次预演生成：

```text
sysid_mujoco/results/h1/real_command_preview/<joint>_<kind>/
├── command_preview.png
└── preview_metrics.json
```

曲线包含50 Hz目标位置、500 Hz MuJoCo响应、跟踪误差和关节速度。指标文件还包含全身最大跟踪误差、最小关节限位余量、双脚参考点最小间距和自碰撞列表。

## 加入悬空准备姿态后的左膝验证

| 命令 | 幅值 | 被测膝最大误差 | 全身最大误差 | 自碰撞 | 检查 |
|---|---:|---:|---:|---:|---|
| prepare | 0 | 0.007822 rad | 0.043336 rad | 0 | PASS |
| positive-step | +0.01 rad | 0.010249 rad | 0.043336 rad | 0 | PASS |
| negative-step | -0.01 rad | 0.010406 rad | 0.043336 rad | 0 | PASS |
| sine 0.5 Hz | ±0.01 rad | 0.007822 rad | 0.043336 rad | 0 | PASS |
| chirp 0.5–5 Hz | ±0.01 rad | 0.010962 rad | 0.043336 rad | 0 | PASS |

左膝映射为物理电机ID 5，预演使用 `Kp=300`、`Kd=6`。五项检查的最小关节限位余量均为 `0.339363 rad`，双脚参考点最小距离均为 `0.335165 m`，没有检测到自碰撞。

这些PASS只表示命令在当前名义MuJoCo模型中没有触发跟踪误差保护，不是实机安全认证。实机仍必须从被动observe和当前位置保持开始，并由固定支架、急停和现场人员提供安全保障。
