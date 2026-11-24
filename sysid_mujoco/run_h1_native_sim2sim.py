"""Launch Holosoma's native H1 MuJoCo sim2sim with one model choice.

This launcher intentionally does not reproduce policy evaluation.  It executes
Holosoma's existing ``holosoma.run_sim`` bridge unchanged.  The calibrated
variant differs only by two robot-asset CLI overrides; the inference process,
ONNX policy, DDS bridge, timing, controls, terrain, gantry and viewer remain the
same as the nominal Holosoma sim2sim setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLOSOMA_ROOT = Path("/home/pjm/Desktop/holosoma")
DEFAULT_PYTHON = Path(
    "/home/pjm/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
)
CALIBRATED_ASSET_ROOT = REPO_ROOT / "robot_model"
CALIBRATED_XML = "h1/h1_calibrated_full_body.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("nominal", "calibrated"), required=True)
    parser.add_argument("--domain-id", type=int, default=7)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "holosoma.run_sim",
        "simulator:mujoco",
        "robot:h1-19dof",
        "terrain:terrain-locomotion-plane",
        "--simulator.config.bridge.enabled",
        "True",
        "--simulator.config.bridge.interface",
        args.interface,
        "--simulator.config.bridge.domain-id",
        str(args.domain_id),
    ]
    if args.model == "calibrated":
        command.extend(
            [
                "--robot.asset.asset-root",
                str(CALIBRATED_ASSET_ROOT),
                "--robot.asset.xml-file",
                CALIBRATED_XML,
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    args.holosoma_root = args.holosoma_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()

    required = [
        args.python,
        args.holosoma_root / "src" / "holosoma" / "holosoma" / "run_sim.py",
    ]
    if args.model == "calibrated":
        required.append(CALIBRATED_ASSET_ROOT / CALIBRATED_XML)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    command = build_command(args)
    print("Holosoma native sim2sim; no Holosoma source files are modified.", flush=True)
    print(f"MuJoCo model: {args.model}", flush=True)
    print("Command:", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    source_root = args.holosoma_root / "src" / "holosoma"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root)] + ([existing_pythonpath] if existing_pythonpath else [])
    )
    library_dir = args.python.parent.parent / "lib"
    existing_library_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(library_dir)] + ([existing_library_path] if existing_library_path else [])
    )
    completed = subprocess.run(command, cwd=args.holosoma_root, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
