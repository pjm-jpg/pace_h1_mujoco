# H1 19DoF Isaac Sim / PhysX 迁移包

`h1_calibrated_physx.usd` 是带浮动基座、几何和关节动力学的机器人资产。可在 Isaac Sim 5.1 打开，或者在 Isaac Lab 中通过 `UsdFileCfg` 载入。配套启动脚本让 Holosoma 的 H1 实验读取这个 USD，并按现有的关节顺序填入对应配置。

## Holosoma

解压迁移包后，从已安装 Isaac Sim 5.1 和 Isaac Lab 2.3.0 的 Holosoma 项目根目录执行。将 `H1_PACKAGE` 设成解压目录的绝对路径。

```bash
source scripts/source_isaacsim_setup.sh
H1_PACKAGE=/绝对路径/h1_isaacsim_v1
python "$H1_PACKAGE/holosoma_isaacsim.py" smoke \
  exp:h1-19dof simulator:isaacsim logger:disabled --training.num-envs 2
```

出现 `HOLOSOMA PASS` 表示原生训练环境完成初始化、重置、短步进和参数读回。正式训练使用 `train` 子命令，后面沿用你的 Holosoma 训练选项，例如：

```bash
python "$H1_PACKAGE/holosoma_isaacsim.py" train \
  exp:h1-19dof simulator:isaacsim logger:disabled \
  --training.name h1_physx_calibrated
```

启动脚本读取现有实验 preset，可把算法、任务、奖励、环境数等按项目需要配置。上述示例使用 `h1-19dof` 步态 preset。

## 独立 Isaac Lab 加载

```bash
python "$H1_PACKAGE/build_usd.py" smoke
```

将 USD 接到自己的 `ArticulationCfg.spawn` 时，使用 `UsdFileCfg(usd_path=...)`。如果任务定义了 actuator 配置，让 `armature`、`friction`、`dynamic_friction`、`viscous_friction` 继承 USD（设为 `None`），或按关节名显式传入同样的值。Holosoma 启动脚本处理了其原生 backend 对 armature 和静摩擦的显式赋值。

## 参数及来源

- `armature` 以 kg·m² 写入 PhysX 关节附加惯量。
- `frictionloss` 以 N·m 同时作为静摩擦阈值和动摩擦力矩。本次辨识使用单一库仑摩擦参数，因此用同一数值初始化两项。
- `damping` 以 N·m·s/rad 读回为 PhysX 粘性摩擦。USD 的旋转关节属性按度计，因此写入时乘以 π/180。策略 PD 增益沿用 Holosoma 实验设置。
- 机械结构、连杆名义质量与惯量来自 Holosoma 的 H1 URDF。脚地接触由使用该资产的环境配置决定。

参考 [NVIDIA PhysX 关节轴属性](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/PhysxJointAxisAPI.html) 和 [Isaac Lab 执行器说明](https://isaac-sim.github.io/IsaacLab/main/source/concepts/actuators.html)。

`h1_calibrated_full_body_params.json` 保存原始辨识值及来源。`source/` 提供重建用 URDF 和网格。可在相同 Isaac Sim 环境运行 `python build_usd.py export --usd /绝对路径/新文件.usd` 重建资产。

## 实测记录

在 RTX 4090、Isaac Sim 5.1.0、Isaac Lab v2.3.0 和 Holosoma `c3d07956` 上验证。独立 Isaac Lab 场景运行了 20 个物理步，Holosoma H1 步态任务在两个并行环境完成重置和五个控制步。两条路径均从 PhysX 读回 19 个关节的参数，惯量和动静摩擦与 float32 参考值一致，粘性阻尼的最大绝对差为 `4.77e-7 N·m·s/rad`。训练入口也完成了 8 个并行环境的一轮 PPO 更新，成功写出 `model_00000.pt`。在解压目录运行 `sha256sum -c SHA256SUMS` 可核对主要文件。

这是从 MuJoCo 辨识结果迁移到 PhysX 的模型。迁移测试衡量资产加载、关节参数和数值状态。跨引擎轨迹误差及训练收益仍需要独立评估。
