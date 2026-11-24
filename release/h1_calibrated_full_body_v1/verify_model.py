"""Verify the portable H1 calibrated MuJoCo package."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
MODEL_XML = ROOT / "h1_calibrated_full_body.xml"
SCENE_XML = ROOT / "scene_flat_calibrated_full_body.xml"
PARAMETERS_JSON = ROOT / "h1_calibrated_full_body_params.json"


def main() -> None:
    payload = json.loads(PARAMETERS_JSON.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    scene = mujoco.MjModel.from_xml_path(str(SCENE_XML))

    maximum_error = 0.0
    for joint_name, expected in parameters.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"模型缺少关节：{joint_name}")
        dof_id = int(model.jnt_dofadr[joint_id])
        actual = {
            "armature": float(model.dof_armature[dof_id]),
            "damping": float(model.dof_damping[dof_id]),
            "frictionloss": float(model.dof_frictionloss[dof_id]),
        }
        maximum_error = max(
            maximum_error,
            *(abs(actual[name] - float(expected[name])) for name in actual),
        )

    if maximum_error > 1e-9:
        raise RuntimeError(f"参数检查失败，最大误差：{maximum_error:.3e}")
    if model.nu != 19 or len(parameters) != 19:
        raise RuntimeError(f"预期 19 个执行器和参数，实际为 nu={model.nu}, 参数={len(parameters)}")

    print("验证通过")
    print(f"机器人模型：nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"平地场景：nq={scene.nq}, nv={scene.nv}, nu={scene.nu}")
    print(f"已校准关节：{len(parameters)}")
    print(f"最大参数误差：{maximum_error:.3e}")


if __name__ == "__main__":
    main()
