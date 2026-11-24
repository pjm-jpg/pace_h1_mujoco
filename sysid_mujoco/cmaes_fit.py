"""PACE-style CMA-ES identification using the native MuJoCo replay pipeline.

This entry point deliberately reuses the model construction, dataset loading,
trajectory conversion, and residual function from ``my_fit.py``. CMA-ES only
replaces the local optimizer. The first H1 workflow defaults to the left knee.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from functools import partial
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from sysid_mujoco.common import (  # noqa: E402
    build_actuator_gain_map,
    build_fixed_base_model_xml,
    build_parameter_dict,
    clip_parameter_values_inside_bounds,
    get_actuated_joint_and_actuator_names,
    load_dataset_actuator_gains,
)
ATTRIBUTES = ("armature", "frictionloss", "damping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default=config.robot)
    parser.add_argument(
        "--base-model-xml",
        type=Path,
        default=None,
        help=(
            "Optional source model to calibrate. For upper-body H1 fits, use "
            "the combined lower-body calibrated XML so previously identified "
            "parameters remain fixed."
        ),
    )
    parser.add_argument("--dataset", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--joints", nargs="+", default=["left_knee_joint"],
        help="Exact MuJoCo joint names to identify (default: left knee).",
    )
    parser.add_argument(
        "--score-joints", nargs="+", default=None,
        help="Joints included in the trajectory loss (defaults to --joints).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.35, help="Initial sigma in normalized [-1, 1] space.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    parser.add_argument("--damping-bounds", nargs=2, type=float, default=(0.0, 5.0))
    # H1's nominal MJCF armature is 0.1, so the default bounds must contain it.
    parser.add_argument("--armature-bounds", nargs=2, type=float, default=(0.001, 0.5))
    parser.add_argument("--frictionloss-bounds", nargs=2, type=float, default=(0.0, 3.0))
    args = parser.parse_args()
    if args.population_size < 2 or args.max_iters < 1:
        parser.error("population size must be >= 2 and max iterations >= 1")
    if not 0.0 < args.sigma <= 1.0:
        parser.error("--sigma must be in (0, 1]")
    if args.velocity_weight < 0:
        parser.error("--velocity-weight must be nonnegative")
    if args.command_delay_ms < 0:
        parser.error("--command-delay-ms must be nonnegative")
    if args.score_joints is None:
        args.score_joints = list(args.joints)
    return args


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / args.robot / f"cmaes_{stamp}"


def _physical(normalized: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + 0.5 * (normalized + 1.0) * (upper - lower)


def _normalized(physical: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 2.0 * (physical - lower) / (upper - lower) - 1.0


def _loss(residual_fn, parameter_dict, physical: np.ndarray) -> float:
    result = residual_fn(physical, parameter_dict, return_pred_all=False)
    # MuJoCo sysid versions differ here: some always return the diagnostic
    # triple while others return only the residual unless predictions are asked.
    residual = result[0] if isinstance(result, tuple) and len(result) == 3 else result
    if isinstance(residual, (list, tuple)):
        # Multiple ModelSequences are returned as one residual array per
        # trajectory. Their lengths need not match, so np.asarray would create
        # an object/ragged array instead of one CMA-ES objective vector.
        residual = np.concatenate(
            [np.asarray(sequence, dtype=np.float64).reshape(-1) for sequence in residual]
        )
    else:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(residual)):
        return float("inf")
    return float(np.mean(np.square(residual)))


def _save_identified_xml(source_xml: Path, output_xml: Path, values: dict[str, dict[str, float]]) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    for joint in root.iter("joint"):
        joint_values = values.get(joint.get("name", ""))
        if joint_values:
            for attribute, value in joint_values.items():
                joint.set(attribute, f"{value:.12g}")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    from cmaes import CMA
    import mujoco
    from mujoco import sysid
    from sysid_mujoco.my_fit import (
        add_joint_velocity_sensors,
        build_model_sequences_from_source,
        joint_position_velocity_residual,
    )

    for dataset in args.dataset:
        if not dataset.exists():
            raise FileNotFoundError(dataset)

    dataset_kp, dataset_kd = load_dataset_actuator_gains(args.dataset[0])
    for dataset_path in args.dataset[1:]:
        kp, kd = load_dataset_actuator_gains(dataset_path)
        if not np.allclose(kp, dataset_kp) or not np.allclose(kd, dataset_kd):
            raise ValueError(
                "All CMA-ES trajectories must use identical fixed Kp/Kd gains; "
                f"{dataset_path} differs from {args.dataset[0]}."
            )
    fixed_xml = build_fixed_base_model_xml(
        args.robot,
        source_model_xml=args.base_model_xml,
    )
    discovery_spec = mujoco.MjSpec.from_file(str(fixed_xml))
    discovery_spec.option.timestep = 1.0 / config.frequency_collection
    discovery_model = discovery_spec.compile()
    actuated_joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, discovery_model)
    unknown = sorted(set(args.joints) - set(actuated_joint_names))
    if unknown:
        raise ValueError(f"Unknown joints {unknown}; available joints: {actuated_joint_names}")
    unknown_score_joints = sorted(set(args.score_joints) - set(actuated_joint_names))
    if unknown_score_joints:
        raise ValueError(
            f"Unknown score joints {unknown_score_joints}; available joints: {actuated_joint_names}"
        )
    if len(dataset_kp) != len(actuated_joint_names) or len(dataset_kd) != len(actuated_joint_names):
        raise ValueError("Dataset Kp/Kd length does not match the H1 actuator count.")

    fixed_xml = build_fixed_base_model_xml(
        args.robot,
        actuator_gains=build_actuator_gain_map(actuated_joint_names, dataset_kp, dataset_kd),
        source_model_xml=args.base_model_xml,
    )
    model_spec = mujoco.MjSpec.from_file(str(fixed_xml))
    model_spec.option.timestep = 1.0 / config.frequency_collection
    add_joint_velocity_sensors(model_spec)
    model = model_spec.compile()

    measurements, controls, initial_states = build_model_sequences_from_source(
        sysid=sysid,
        mujoco=mujoco,
        model=model,
        robot=args.robot,
        dataset_paths=args.dataset,
        chunk_size=args.chunk_size,
        command_delay_s=args.command_delay_ms * 1e-3,
    )
    sequence_names = [f"sequence_{index:03d}" for index in range(len(measurements))]
    model_sequences = [
        sysid.ModelSequences(
            args.robot, model_spec, sequence_names[index], initial_states[index],
            controls[index], measurements[index],
        )
        for index in range(len(measurements))
    ]
    bounds = {
        "damping": tuple(args.damping_bounds),
        "armature": tuple(args.armature_bounds),
        "frictionloss": tuple(args.frictionloss_bounds),
    }
    params = build_parameter_dict(sysid, model, args.joints, bounds, model_spec=model_spec)
    clip_parameter_values_inside_bounds(params)
    lower, upper = params.get_bounds()
    initial = params.as_vector()
    mean = np.clip(_normalized(initial, lower, upper), -0.999, 0.999)

    residual_fn = sysid.build_residual_fn(
        models_sequences=model_sequences,
        modify_residual=partial(
            joint_position_velocity_residual,
            velocity_weight=args.velocity_weight,
            score_joints=args.score_joints,
        ),
        resample_true=False,
    )
    optimizer = CMA(
        mean=mean,
        sigma=args.sigma,
        bounds=np.tile(np.array([-1.0, 1.0]), (mean.size, 1)),
        population_size=args.population_size,
        seed=args.seed,
    )

    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "optimization_log.csv"
    best_loss = float("inf")
    best_vector = initial.copy()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "generation_best", "global_best", *[
            f"{joint}_{attribute}" for joint in args.joints for attribute in ATTRIBUTES
        ]])
        for iteration in range(args.max_iters):
            solutions = []
            generation_best = None
            for _ in range(optimizer.population_size):
                candidate = optimizer.ask()
                physical = _physical(candidate, lower, upper)
                score = _loss(residual_fn, params, physical)
                solutions.append((candidate, score))
                if generation_best is None or score < generation_best[0]:
                    generation_best = (score, physical.copy())
                if score < best_loss:
                    best_loss = score
                    best_vector = physical.copy()
            optimizer.tell(solutions)
            assert generation_best is not None
            writer.writerow([iteration, generation_best[0], best_loss, *best_vector])
            stream.flush()
            print(
                f"iteration={iteration:04d} generation_best={generation_best[0]:.8g} "
                f"global_best={best_loss:.8g}"
            )
            if optimizer.should_stop():
                break

    values: dict[str, dict[str, float]] = {}
    cursor = 0
    for joint in args.joints:
        values[joint] = {}
        for attribute in ATTRIBUTES:
            values[joint][attribute] = float(best_vector[cursor])
            cursor += 1
    result = {
        "robot": args.robot,
        "base_model_xml": str(
            (args.base_model_xml or fixed_xml).expanduser().resolve()
        ),
        "datasets": [str(path.resolve()) for path in args.dataset],
        "joints": args.joints,
        "loss": best_loss,
        "parameters": values,
        "frequency_hz": config.frequency_collection,
        "velocity_weight": args.velocity_weight,
        "score_joints": args.score_joints,
        "command_delay_ms": args.command_delay_ms,
        "seed": args.seed,
    }
    (output_dir / "best_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(fixed_xml, output_dir / "nominal_fixed_base.xml")
    _save_identified_xml(fixed_xml, output_dir / "identified_fixed_base.xml", values)
    print(f"Best loss: {best_loss:.8g}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
