"""Run resumable per-joint CMA-ES fits for the official H1 upper-body data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "h1" / "real" / "upper_body_official_datasets.json"
DEFAULT_BASE_MODEL = REPO_ROOT / "robot_model" / "h1" / "h1_calibrated_lower_body.xml"
DEFAULT_RESULTS = REPO_ROOT / "sysid_mujoco" / "results" / "h1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model-xml", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--joints", nargs="+", default=None, help="Optional joint keys to run.")
    parser.add_argument("--tag", default="upper_v1", help="Result-directory suffix.")
    parser.add_argument("--armature-bounds", nargs=2, type=float, default=(0.001, 0.5))
    return parser.parse_args()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _run_fit(
    joint_key: str,
    joint_name: str,
    fit_paths: list[Path],
    base_model: Path,
    results_root: Path,
    population_size: int,
    max_iters: int,
    seed: int,
    tag: str,
    armature_bounds: tuple[float, float],
) -> tuple[str, str]:
    output_dir = results_root / f"real_{joint_key}_cmaes_{tag}"
    result_path = output_dir / "best_params.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if Path(previous.get("base_model_xml", "")).resolve() == base_model:
            return joint_key, "skipped (matching result exists)"
        raise ValueError(f"Existing result uses a different base model: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "sysid_mujoco" / "cmaes_fit.py"),
        "--robot", "h1",
        "--base-model-xml", str(base_model),
        "--dataset", *[str(path) for path in fit_paths],
        "--joints", joint_name,
        "--score-joints", joint_name,
        "--population-size", str(population_size),
        "--max-iters", str(max_iters),
        "--seed", str(seed),
        "--armature-bounds", str(armature_bounds[0]), str(armature_bounds[1]),
        "--output-dir", str(output_dir),
    ]
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-h1-upper-cmaes")
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{joint_key} failed; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return joint_key, f"completed, loss={result['loss']:.8g}"


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    base_model = args.base_model_xml.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = []
    for joint_key in manifest["joint_keys"]:
        if args.joints is not None and joint_key not in args.joints:
            continue
        rows = [
            row for row in manifest["datasets"]
            if row["joint_key"] == joint_key and row["use"] == "fit"
        ]
        if len(rows) != 2:
            raise ValueError(f"Expected two fitting datasets for {joint_key}")
        jobs.append(
            (
                joint_key,
                rows[0]["joint_name"],
                [_resolve_repo_path(row["dataset"]) for row in rows],
            )
        )

    print(
        f"Starting {len(jobs)} fits with {args.workers} workers, "
        f"population={args.population_size}, iterations={args.max_iters}"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_fit,
                joint_key,
                joint_name,
                fit_paths,
                base_model,
                results_root,
                args.population_size,
                args.max_iters,
                args.seed,
                args.tag,
                tuple(args.armature_bounds),
            ): joint_key
            for joint_key, joint_name, fit_paths in jobs
        }
        for future in as_completed(futures):
            joint_key = futures[future]
            try:
                completed_key, status = future.result()
            except Exception as error:
                print(f"{joint_key}: FAILED: {error}", flush=True)
                raise
            print(f"{completed_key}: {status}", flush=True)


if __name__ == "__main__":
    main()
