"""Run the official Holosoma FastSAC evaluator on two H1 MuJoCo models.

The script compares the original Holosoma H1 model with this repository's
full-body calibrated model.  It uses MuJoCo Classic on CPU, writes only under
the requested output directory and /tmp, and never opens DDS or commands a
real robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLOSOMA_ROOT = Path("/home/pjm/Desktop/holosoma")
DEFAULT_CHECKPOINT = DEFAULT_HOLOSOMA_ROOT / (
    "logs/hv-h1-manager/"
    "20260610_071135-h1_19dof_fast_sac_manager-locomotion/model_0050000.pt"
)
DEFAULT_PYTHON = Path(
    "/home/pjm/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--max-eval-steps", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "sysid_mujoco"
            / "results"
            / "h1"
            / "holosoma_policy_smoke_v1"
        ),
    )
    args = parser.parse_args()
    if args.max_eval_steps <= 0:
        parser.error("--max-eval-steps must be positive.")
    return args


def _run(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    return return_code, time.monotonic() - started


def main() -> None:
    args = parse_args()
    holosoma_root = args.holosoma_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    python = args.python.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    calibrated_asset_root = (REPO_ROOT / "robot_model" / "h1").resolve()
    calibrated_xml = calibrated_asset_root / "h1_calibrated_full_body.xml"
    for required in (
        holosoma_root / "src" / "holosoma" / "holosoma" / "eval_agent.py",
        checkpoint,
        python,
        calibrated_xml,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path("/tmp/h1_holosoma_policy_regression")
    temporary_root.mkdir(parents=True, exist_ok=True)
    local_checkpoint = temporary_root / checkpoint.name
    shutil.copy2(checkpoint, local_checkpoint)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(holosoma_root / "src" / "holosoma"),
            "WANDB_MODE": "offline",
            "WARP_CACHE_PATH": str(temporary_root / "warp_cache"),
        }
    )
    base_command = [
        str(python),
        "-m",
        "holosoma.eval_agent",
        "--checkpoint",
        str(local_checkpoint),
        "--training.headless",
        "True",
        "--training.num-envs",
        "1",
        "--training.max-eval-steps",
        str(args.max_eval_steps),
        "--training.export-onnx",
        "False",
        "--simulator.config.mujoco-backend",
        "CLASSIC",
    ]
    trials = {
        "nominal": [],
        "full_body_calibrated": [
            "--robot.asset.asset-root",
            str(calibrated_asset_root),
            "--robot.asset.xml-file",
            calibrated_xml.name,
        ],
    }
    results: dict[str, object] = {}
    for name, extra_args in trials.items():
        command = base_command + extra_args
        print(f"\n=== {name} ===")
        return_code, elapsed_s = _run(
            command,
            temporary_root,
            env,
            output_dir / f"{name}.log",
        )
        results[name] = {
            "return_code": return_code,
            "completed_requested_steps": return_code == 0,
            "elapsed_wall_s": elapsed_s,
            "command": command,
            "log": str(output_dir / f"{name}.log"),
        }

    passed = all(item["return_code"] == 0 for item in results.values())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "official Holosoma policy-pipeline compatibility smoke test",
        "checkpoint_source": str(checkpoint),
        "checkpoint_iteration": 50000,
        "backend": "MuJoCo Classic CPU",
        "physics_rate_hz": 200,
        "policy_rate_hz": 50,
        "max_eval_steps": args.max_eval_steps,
        "results": results,
        "overall_passed": passed,
        "interpretation": (
            "A zero return code means the official Holosoma evaluator loaded "
            "the model and checkpoint and completed the requested evaluation "
            "loop without a runtime error."
        ),
        "scope_limit": (
            "Holosoma does not emit fall, reset, reward, or tracking scalars in "
            "this headless smoke mode. This result proves pipeline compatibility, "
            "not quantitative locomotion parity or real-robot safety."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
