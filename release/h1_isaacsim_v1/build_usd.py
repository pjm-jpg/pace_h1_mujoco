"""Export the H1 URDF and identified joint dynamics as a self-contained PhysX USD."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
USD = ROOT / "h1_calibrated_physx.usd"


def load_parameters():
    parameters = json.loads((ROOT / "h1_calibrated_full_body_params.json").read_text())["parameters"]
    if len(parameters) != 19 or any(
        set(row) != {"armature", "damping", "frictionloss"}
        or any(not math.isfinite(value) or value < 0 for value in row.values())
        for row in parameters.values()
    ):
        raise ValueError("Expected 19 H1 joints with finite, nonnegative identified parameters.")
    return parameters


def export(output: Path):
    from isaaclab.sim import UrdfConverter, UrdfConverterCfg
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    parameters = load_parameters()
    with TemporaryDirectory(prefix="h1_urdf_import_") as directory:
        tree = ET.parse(ROOT / "source" / "h1.urdf")
        for mesh in tree.iter("mesh"):
            filename = ROOT / "source" / "meshes" / Path(mesh.attrib["filename"]).name
            if not filename.is_file():
                raise FileNotFoundError(filename)
            mesh.set("filename", str(filename))
        urdf = Path(directory) / "h1.urdf"
        tree.write(urdf, encoding="utf-8", xml_declaration=True)
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(urdf), usd_dir=directory, usd_file_name="h1_nominal.usd",
            force_usd_conversion=True, fix_base=False, merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none", gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ))
        imported = Usd.Stage.Open(converter.usd_path)
        imported.SetEditTarget(imported.GetSessionLayer())
        instance_paths = {prim.GetPath() for prim in imported.Traverse(Usd.TraverseInstanceProxies())
                          if prim.IsInstanceable()}
        for path in sorted(instance_paths, key=lambda path: path.pathElementCount):
            imported.GetPrimAtPath(path).SetInstanceable(False)
        stage = Usd.Stage.Open(imported.Flatten(addSourceFileComment=False))
        if not stage.GetDefaultPrim() or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise RuntimeError("The imported asset requires a default prim and meter units.")
        # The URDF uses solid colors. Express them directly in USD for portability.
        materials = [UsdShade.Material(prim) for prim in stage.Traverse() if prim.IsA(UsdShade.Material)]
        for material in materials:
            material_prim = material.GetPrim()
            for child in material_prim.GetChildren():
                if not child.IsA(UsdShade.Shader):
                    continue
                color_attr = child.GetAttribute("inputs:diffuse_color")
                color = color_attr.Get() if color_attr else None
                for attribute in child.GetAuthoredAttributes():
                    child.RemoveProperty(attribute.GetName())
                shader = UsdShade.Shader(child)
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(color) if color is not None else Gf.Vec3f(0.5)
                )
                shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                for output_attr in material.GetOutputs():
                    material_prim.RemoveProperty(output_attr.GetFullName())
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        joints = {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
        if set(joints) != set(parameters):
            raise RuntimeError(f"USD joint names differ from the manifest: {sorted(set(joints) ^ set(parameters))}")
        for name, prim in joints.items():
            row = parameters[name]
            if not prim.ApplyAPI("PhysxJointAxisAPI", "angular"):
                raise RuntimeError(f"Unable to apply PhysxJointAxisAPI to {prim.GetPath()}")
            attributes = {
                "armature": row["armature"],
                "staticFrictionEffort": row["frictionloss"],
                "dynamicFrictionEffort": row["frictionloss"],
                "viscousFrictionCoefficient": row["damping"] * math.pi / 180.0,
            }
            for field, value in attributes.items():
                prim.CreateAttribute(f"physxJointAxis:angular:{field}", Sdf.ValueTypeNames.Float).Set(value)
        if not stage.GetRootLayer().Export(str(output)):
            raise RuntimeError(f"Could not write {output}")
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output)))
        if len(layers) != 1 or assets or unresolved:
            raise RuntimeError(f"USD has external dependencies: {layers}, {assets}, {unresolved}")
        print(f"EXPORTED {output} ({len(joints)} joints, self-contained)", flush=True)


def verify_readback(robot):
    import torch

    parameters = load_parameters()
    names = robot.joint_names
    if len(names) != 19 or set(names) != set(parameters):
        raise RuntimeError(f"Unexpected PhysX joints: {names}")
    view = robot.root_physx_view
    armature = view.get_dof_armatures().cpu()
    friction = view.get_dof_friction_properties().cpu()
    expected_armature = armature.new_tensor([parameters[name]["armature"] for name in names])
    expected_friction = friction.new_tensor([
        [parameters[name]["frictionloss"], parameters[name]["frictionloss"], parameters[name]["damping"]]
        for name in names
    ])
    torch.testing.assert_close(armature, expected_armature.expand_as(armature), rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(friction, expected_friction.expand_as(friction), rtol=2e-5, atol=1e-7)
    print("PHYSX PASS " + json.dumps({
        "joints": len(names), "instances": armature.shape[0],
        "max_armature_error": (armature - expected_armature).abs().max().item(),
        "max_friction_error_by_field": (friction - expected_friction).abs().amax(dim=(0, 1)).tolist(),
    }), flush=True)


def smoke(path: Path, device: str):
    import torch
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext, UsdFileCfg

    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/H1", spawn=UsdFileCfg(usd_path=str(path)),
        actuators={"all": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    ))
    sim.reset()
    verify_readback(robot)
    for _ in range(20):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
    if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(robot.data.root_pos_w).all():
        raise RuntimeError("Non-finite simulation state")
    print("STANDALONE PASS 20 PhysX steps", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "smoke"))
    parser.add_argument("--usd", type=Path, default=USD)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    try:
        if args.mode == "export":
            export(args.usd.resolve())
        else:
            smoke(args.usd.resolve(), args.device)
    except Exception:
        traceback.print_exc()
        app.close(skip_cleanup=True, exit_code=1)
        raise
    else:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
