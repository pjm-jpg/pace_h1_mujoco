"""Use the calibrated H1 USD with an existing Holosoma experiment preset."""

from dataclasses import replace
import sys
import traceback

from build_usd import USD, load_parameters, verify_readback


def calibrated_config(config):
    if config.simulator.config.name != "isaacsim":
        raise ValueError("The calibrated PhysX USD requires simulator:isaacsim.")
    if not USD.is_file():
        raise FileNotFoundError(USD)
    robot = config.robot
    names = robot.dof_names
    parameters = load_parameters()
    if len(names) != 19 or set(names) != set(parameters):
        raise ValueError("The selected experiment does not match the 19 calibrated H1 joints.")
    # Holosoma sets armature and static friction from RobotConfig when spawning.
    # Dynamic and viscous friction are read from the USD by Isaac Lab.
    robot = replace(
        robot,
        asset=replace(robot.asset, usd_file=str(USD)),
        dof_armature_list=[parameters[name]["armature"] for name in names],
        dof_joint_friction_list=[parameters[name]["frictionloss"] for name in names],
    )
    return replace(config, robot=robot)


def smoke(config):
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.eval_utils import init_sim_imports
    from holosoma.utils.helpers import get_class

    app = init_sim_imports(config)
    try:
        import torch

        env = get_class(config.env_class)(get_tyro_env_config(config), device=getattr(config, "device", None) or "cuda:0")
        robot = env.simulator._robot
        verify_readback(robot)
        env.reset_all()
        actions = torch.zeros((env.num_envs, env.dim_actions), device=env.device)
        for _ in range(5):
            _, reward, _, _ = env.step({"actions": actions})
            if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite H1 training-environment state")
        verify_readback(robot)
        print(f"HOLOSOMA PASS {env.num_envs} environments, reset + 5 control steps", flush=True)
    except Exception:
        traceback.print_exc()
        app.close(skip_cleanup=True, exit_code=1)
        raise
    else:
        app.close(skip_cleanup=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"smoke", "train"}:
        raise SystemExit("Usage: python holosoma_isaacsim.py {smoke,train} exp:h1-19dof simulator:isaacsim [Holosoma options]")
    import tyro
    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    config = calibrated_config(tyro.cli(AnnotatedExperimentConfig, args=sys.argv[2:], config=TYRO_CONIFG))
    if sys.argv[1] == "smoke":
        smoke(config)
    else:
        from holosoma.train_agent import train

        train(config)


if __name__ == "__main__":
    main()
