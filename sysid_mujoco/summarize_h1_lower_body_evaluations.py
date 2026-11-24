"""Summarize all formal replays of the combined H1 lower-body model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "sysid_mujoco"
    / "results"
    / "h1"
    / "real_lower_body_combined_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-count", type=int, default=41)
    return parser.parse_args()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    baseline = sum(float(row["baseline_rmse_rad"]) for row in rows)
    identified = sum(float(row["identified_rmse_rad"]) for row in rows)
    return {
        "count": len(rows),
        "baseline_mean_rmse_rad": baseline / len(rows),
        "identified_mean_rmse_rad": identified / len(rows),
        "pooled_reduction_percent": 100.0 * (baseline - identified) / baseline,
        "improved_count": sum(
            float(row["identified_rmse_rad"]) < float(row["baseline_rmse_rad"])
            for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    evaluation_root = result_root / "evaluations"
    rows = []
    for metrics_path in sorted(evaluation_root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline = float(payload["metrics"]["baseline"]["rmse_mean"])
        identified = float(payload["metrics"]["identified"]["rmse_mean"])
        rows.append(
            {
                "joint_key": metrics_path.parents[1].name,
                "evaluation": metrics_path.parent.name,
                "dataset": payload["dataset"],
                "baseline_rmse_rad": baseline,
                "identified_rmse_rad": identified,
                "reduction_percent": 100.0 * (baseline - identified) / baseline,
                "metrics": str(metrics_path),
                "plot": str(metrics_path.with_name("joint_trajectories.png")),
            }
        )
    if len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} evaluation files, found {len(rows)}"
        )

    by_evaluation: dict[str, dict[str, float | int]] = {}
    for evaluation in sorted({str(row["evaluation"]) for row in rows}):
        selected = [row for row in rows if row["evaluation"] == evaluation]
        by_evaluation[evaluation] = _aggregate(selected)
    step_rows = [row for row in rows if "step" in str(row["evaluation"])]
    summary = {
        "result_root": str(result_root),
        "all": _aggregate(rows),
        "step_validations": _aggregate(step_rows),
        "by_evaluation": by_evaluation,
        "evaluations": rows,
    }
    output = result_root / "evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["all"], indent=2))
    print(f"Saved evaluation summary to {output}")


if __name__ == "__main__":
    main()
