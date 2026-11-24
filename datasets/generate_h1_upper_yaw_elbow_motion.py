"""Generate the guarded H1 shoulder-yaw and elbow excitation trajectory."""

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


DEFAULT_OUTPUT = (
    REPO_ROOT / "datasets/h1/motions/h1_upper_yaw_elbow_excitation_mj_fps50.npz"
)


def _append_segment(
    samples: list[np.ndarray],
    center: np.ndarray,
    fps: float,
    duration_s: float,
    updater,
) -> None:
    count = int(round(duration_s * fps))
    # Exclude each segment's last point so adjacent segments share one sample.
    for index in range(count):
        t = index / fps
        value = center.copy()
        updater(value, t)
        samples.append(value)


def generate_motion(fps: float = 50.0) -> tuple[np.ndarray, dict[str, object]]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    index = {name: i for i, name in enumerate(config.H1_JOINT_NAMES)}
    center = np.asarray(
        [SUSPENDED_REFERENCE[name] for name in config.H1_JOINT_NAMES],
        dtype=np.float64,
    )

    # Hold the upper arms forward and away from the torso.  Shoulder yaw then
    # rotates the already-clear forearms, while elbow flexion stays outside the
    # hip and pelvis envelope.
    center[index["left_shoulder_pitch_joint"]] = -0.80
    center[index["right_shoulder_pitch_joint"]] = -0.80
    center[index["left_shoulder_roll_joint"]] = 0.65
    center[index["right_shoulder_roll_joint"]] = -0.65
    center[index["left_shoulder_yaw_joint"]] = 0.0
    center[index["right_shoulder_yaw_joint"]] = 0.0
    center[index["left_elbow_joint"]] = 0.85
    center[index["right_elbow_joint"]] = 0.85

    samples: list[np.ndarray] = []
    segment_table: list[dict[str, object]] = []

    def add(name: str, duration_s: float, updater) -> None:
        start = len(samples) / fps
        _append_segment(samples, center, fps, duration_s, updater)
        segment_table.append(
            {"name": name, "start_s": start, "duration_s": duration_s}
        )

    zero = lambda _q, _t: None
    add("initial_hold", 1.0, zero)

    def yaw_only(q: np.ndarray, t: float) -> None:
        wave = np.sin(2.0 * np.pi * 0.5 * t)
        q[index["left_shoulder_yaw_joint"]] = 0.65 * wave
        q[index["right_shoulder_yaw_joint"]] = -0.65 * wave

    add("shoulder_yaw", 8.0, yaw_only)
    add("yaw_settle", 1.0, zero)

    def elbow_only(q: np.ndarray, t: float) -> None:
        wave = np.sin(2.0 * np.pi * 0.5 * t)
        q[index["left_elbow_joint"]] = 0.85 + 0.60 * wave
        q[index["right_elbow_joint"]] = 0.85 + 0.60 * wave

    add("elbow", 8.0, elbow_only)
    add("elbow_settle", 1.0, zero)

    def coupled(q: np.ndarray, t: float) -> None:
        yaw_wave = np.sin(2.0 * np.pi * 0.5 * t)
        elbow_wave = np.sin(2.0 * np.pi * 0.75 * t)
        q[index["left_shoulder_yaw_joint"]] = 0.55 * yaw_wave
        q[index["right_shoulder_yaw_joint"]] = -0.55 * yaw_wave
        q[index["left_elbow_joint"]] = 0.85 + 0.55 * elbow_wave
        q[index["right_elbow_joint"]] = 0.85 + 0.55 * elbow_wave

    add("coupled_yaw_elbow", 8.0, coupled)
    add("final_hold", 1.0, zero)
    samples.append(center.copy())

    q = np.asarray(samples, dtype=np.float64)
    root = np.zeros((len(q), 7), dtype=np.float64)
    root[:, 2] = 1.0
    root[:, 3] = 1.0
    joint_pos = np.concatenate((root, q), axis=1)
    metadata = {
        "fps": fps,
        "duration_s": (len(q) - 1) / fps,
        "segments": segment_table,
        "purpose": "large-amplitude shoulder-yaw and elbow identification",
    }
    return joint_pos, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    joint_pos, metadata = generate_motion(args.fps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=np.asarray([args.fps], dtype=np.float64),
        joint_names=np.asarray(config.H1_JOINT_NAMES),
        joint_pos=joint_pos,
        segment_names=np.asarray([item["name"] for item in metadata["segments"]]),
        segment_start_s=np.asarray(
            [item["start_s"] for item in metadata["segments"]], dtype=np.float64
        ),
        segment_duration_s=np.asarray(
            [item["duration_s"] for item in metadata["segments"]], dtype=np.float64
        ),
    )
    print(f"saved: {output}")
    print(f"samples: {len(joint_pos)}")
    print(f"duration_s: {metadata['duration_s']:.3f}")


if __name__ == "__main__":
    main()
