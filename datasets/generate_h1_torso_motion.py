"""Generate a large-range, multi-frequency H1 torso identification motion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from datasets.collect_h1_project_motion import SUSPENDED_REFERENCE


DEFAULT_OUTPUT = REPO_ROOT / "datasets/h1/motions/h1_torso_large_mj_fps50.npz"


def generate_motion(fps: float = 50.0) -> tuple[np.ndarray, dict[str, object]]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    torso_index = config.H1_JOINT_NAMES.index("torso_joint")
    center = np.asarray(
        [SUSPENDED_REFERENCE[name] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )
    center[torso_index] = 0.04
    samples: list[np.ndarray] = []
    segments: list[dict[str, object]] = []

    def add(name: str, duration_s: float, frequency_hz: float, amplitude_rad: float) -> None:
        start_s = len(samples) / fps
        count = int(round(duration_s * fps))
        for sample in range(count):
            q = center.copy()
            t = sample / fps
            q[torso_index] += amplitude_rad * np.sin(2.0 * np.pi * frequency_hz * t)
            samples.append(q)
        segments.append(
            {
                "name": name,
                "start_s": start_s,
                "duration_s": duration_s,
                "frequency_hz": frequency_hz,
                "amplitude_rad": amplitude_rad,
            }
        )

    add("initial_hold", 1.0, 0.0, 0.0)
    add("torso_0p25hz", 8.0, 0.25, 0.24)
    add("slow_settle", 1.0, 0.0, 0.0)
    add("torso_0p5hz", 8.0, 0.50, 0.20)
    add("medium_settle", 1.0, 0.0, 0.0)
    add("torso_1hz", 8.0, 1.0, 0.12)
    add("final_hold", 1.0, 0.0, 0.0)
    samples.append(center.copy())

    q = np.asarray(samples, dtype=np.float64)
    root = np.zeros((len(q), 7), dtype=np.float64)
    root[:, 2] = 1.0
    root[:, 3] = 1.0
    return np.concatenate((root, q), axis=1), {
        "fps": fps,
        "duration_s": (len(q) - 1) / fps,
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    joint_pos, metadata = generate_motion(args.fps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    segments = metadata["segments"]
    np.savez_compressed(
        output,
        fps=np.asarray([args.fps], dtype=np.float64),
        joint_names=np.asarray(config.H1_JOINT_NAMES),
        joint_pos=joint_pos,
        segment_names=np.asarray([item["name"] for item in segments]),
        segment_start_s=np.asarray([item["start_s"] for item in segments]),
        segment_duration_s=np.asarray([item["duration_s"] for item in segments]),
        segment_frequency_hz=np.asarray([item["frequency_hz"] for item in segments]),
        segment_amplitude_rad=np.asarray([item["amplitude_rad"] for item in segments]),
    )
    print(f"saved: {output}")
    print(f"samples: {len(joint_pos)}")
    print(f"duration_s: {metadata['duration_s']:.3f}")


if __name__ == "__main__":
    main()
