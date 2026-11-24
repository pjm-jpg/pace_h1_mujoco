# H1 19DoF 全身校准模型 v1

这是一个可迁移的 MuJoCo 模型包。模型保持原 H1 的拓扑、关节顺序、执行器名称、网格与名义质量/惯量不变，将真机辨识得到的 19 个关节被动动力学参数写入模型：

- `armature`：等效转子惯量；
- `damping`：关节粘性阻尼；
- `frictionloss`：关节库仑摩擦损失。

## 文件

- `h1_calibrated_full_body.xml`：正式校准机器人模型；
- `scene_flat_calibrated_full_body.xml`：包含平地的直接运行场景；
- `h1_calibrated_full_body_params.json`：19 关节参数及来源；
- `meshes/`：模型所需的全部 STL；
- `verify_model.py`：独立完整性与参数检查；
- `SHA256SUMS`：核心文件校验值。

## 独立验证

在安装了 `mujoco` 的 Python 环境中运行：

```bash
python verify_model.py
```

预期输出：模型和场景均成功编译，19 个关节参数与清单一致。

## 迁移到 Holosoma（MuJoCo/MJWarp）

不需要覆盖原始 `h1.xml`。将整个目录复制到 Holosoma：

```bash
cp -a h1_calibrated_full_body_v1 \
  /home/pjm/Desktop/holosoma/src/holosoma/holosoma/data/robots/
```

随后通过两个命令行参数选择校准模型：

```bash
--robot.asset.asset-root @holosoma/data/robots \
--robot.asset.xml-file h1_calibrated_full_body_v1/h1_calibrated_full_body.xml
```

例如原生 MuJoCo sim2sim：

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_mujoco_setup.sh
python -m holosoma.run_sim \
  simulator:mujoco \
  robot:h1-19dof \
  terrain:terrain-locomotion-plane \
  --robot.asset.asset-root @holosoma/data/robots \
  --robot.asset.xml-file h1_calibrated_full_body_v1/h1_calibrated_full_body.xml
```

训练命令使用 MJWarp 时，也添加相同的两个 `--robot.asset...` 参数即可。这样原模型仍保留，可做严格 A/B 对照。

## IsaacGym / IsaacSim 注意事项

Holosoma 的默认 H1 训练配置使用 IsaacGym 和 URDF，而不是这个 MuJoCo XML。因此仅复制 XML **不会改变 IsaacGym 训练动力学**。跨到 PhysX 后端时，应以 `h1_calibrated_full_body_params.json` 为来源，单独映射 `armature`、关节摩擦和阻尼，并重新做后端一致性验证。MuJoCo 的 `frictionloss` 与 PhysX 摩擦参数不是完全等价的，不能无验证地逐值照搬。

## 已验证范围

- 名义模型与校准模型拓扑一致；
- 19 个关节参数写入误差小于 `5e-12`；
- MuJoCo 模型及平地场景可编译；
- 短时仿真保持有限值；
- 39 条未参与拟合的关节响应测试全部改善，位置 RMSE 均值降低约 39%。

当前参数主要校准关节执行动力学；质量、惯量、地面接触与摩擦仍使用名义模型值，不应将此包表述为完整接触动力学数字孪生。
