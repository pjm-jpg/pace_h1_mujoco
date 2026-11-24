from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from sysid_mujoco.common import ProcessedTrajectory, apply_command_delay


_CONVERTER_PATH = Path(__file__).resolve().parents[1] / "datasets" / "convert_h1_dds_log.py"
_CONVERTER_SPEC = importlib.util.spec_from_file_location("convert_h1_dds_log", _CONVERTER_PATH)
assert _CONVERTER_SPEC is not None and _CONVERTER_SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_CONVERTER_SPEC)
_CONVERTER_SPEC.loader.exec_module(_CONVERTER)
convert = _CONVERTER.convert

_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "datasets" / "analyze_h1_observe_log.py"
_ANALYZER_SPEC = importlib.util.spec_from_file_location("analyze_h1_observe_log", _ANALYZER_PATH)
assert _ANALYZER_SPEC is not None and _ANALYZER_SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)
analyze_observe = _ANALYZER.analyze


def test_h1_npz_conversion_uses_motor_mapping_and_command_zoh(tmp_path: Path):
    timestamps = np.asarray([0, 20_000_000, 40_000_000, 60_000_000], dtype=np.int64)
    q_cmd = np.zeros((4, 20), dtype=np.float64)
    q_cmd[1:, 5] = 0.2
    q_real = np.zeros((4, 20), dtype=np.float64)
    q_real[:, 5] = np.asarray([0.0, 0.02, 0.08, 0.15])
    dq_real = np.zeros_like(q_real)
    tau_est = np.zeros_like(q_real)
    tau_est[:, 5] = 1.25
    kp = np.broadcast_to(np.arange(20, dtype=np.float64), (4, 20)).copy()
    kd = np.ones((4, 20), dtype=np.float64)
    tau_ff = np.zeros((4, 20), dtype=np.float64)
    source = tmp_path / "raw.npz"
    np.savez(
        source,
        timestamp=timestamps,
        q_cmd=q_cmd,
        q_real=q_real,
        dq_real=dq_real,
        tau_est=tau_est,
        kp=kp,
        kd=kd,
        tau_ff=tau_ff,
    )
    output = tmp_path / "converted.pt"
    convert(
        Namespace(
            input=source,
            output=output,
            timestamp_unit="auto",
            frequency=500.0,
            allow_varying_gains=False,
            max_gap_factor=5.0,
        )
    )

    data = torch.load(output, map_location="cpu", weights_only=False)
    left_knee = data["joint_names"].index("left_knee_joint")
    # The 50 Hz command changes at 20 ms and remains held on the 500 Hz grid.
    assert data["des_dof_pos"][9, left_knee].item() == 0.0
    assert data["des_dof_pos"][10, left_knee].item() == 0.2
    assert data["des_dof_pos"][15, left_knee].item() == 0.2
    assert data["tau_est"][:, left_knee].unique().tolist() == [1.25]
    assert data["kp"][left_knee] == 5.0
    assert 9 not in data["joint_to_motor"]


def test_h1_conversion_crops_before_resampling_and_records_source_range(tmp_path: Path):
    samples = 10
    timestamps = np.arange(samples, dtype=np.int64) * 1_000_000_000
    values = np.zeros((samples, 20), dtype=np.float64)
    values[:, 5] = np.arange(samples)
    source = tmp_path / "raw.npz"
    np.savez(
        source,
        timestamp=timestamps,
        q_cmd=values,
        q_real=values,
        dq_real=np.zeros_like(values),
    )
    output = tmp_path / "cropped.pt"
    convert(
        Namespace(
            input=source,
            output=output,
            timestamp_unit="auto",
            frequency=0,
            allow_varying_gains=False,
            max_gap_factor=5.0,
            start_time_s=2.0,
            end_time_s=5.0,
        )
    )

    data = torch.load(output, map_location="cpu", weights_only=False)
    left_knee = data["joint_names"].index("left_knee_joint")
    np.testing.assert_allclose(data["time"].numpy(), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(data["dof_pos"][:, left_knee].numpy(), [2.0, 3.0, 4.0, 5.0])
    assert data["source_time_range_s"] == [0.0, 9.0]
    assert data["requested_crop_range_s"] == [2.0, 5.0]
    assert data["actual_crop_range_s"] == [2.0, 5.0]


def test_apply_command_delay_uses_zero_order_hold(tmp_path: Path):
    times = np.arange(5, dtype=np.float64) * 0.01
    controls = np.arange(5, dtype=np.float64)[:, None]
    trajectory = ProcessedTrajectory(
        source_path=tmp_path / "data.pt",
        sequence_name="delay",
        times=times,
        measured_qpos=np.zeros((5, 1)),
        measured_qvel=np.zeros((5, 1)),
        desired_qpos=controls.copy(),
        ctrl=controls,
        joint_names=["joint"],
        actuator_names=["actuator"],
        kp=np.ones(1),
        kd=np.ones(1),
    )

    delayed = apply_command_delay(trajectory, 0.015)
    np.testing.assert_array_equal(delayed.ctrl[:, 0], [0.0, 0.0, 0.0, 1.0, 2.0])


def test_passive_observe_report_passes_and_converter_rejects(tmp_path: Path):
    samples = 101
    timestamps = np.arange(samples, dtype=np.int64) * 2_000_000
    zeros = np.zeros((samples, 20), dtype=np.float32)
    command_nan = np.full((samples, 20), np.nan, dtype=np.float32)
    raw = tmp_path / "raw.npz"
    np.savez(
        raw,
        t_state_ns=timestamps,
        timestamp=timestamps,
        state_seq=np.arange(1, samples + 1, dtype=np.uint64),
        q_real=zeros,
        dq_real=zeros,
        tau_est=zeros,
        temperature=np.full((samples, 20), 35, dtype=np.int16),
        motor_lost=np.tile(np.arange(20, dtype=np.uint32), (samples, 1)),
        imu_quat=np.broadcast_to(np.array([1, 0, 0, 0], dtype=np.float32), (samples, 4)),
        imu_gyro=np.zeros((samples, 3), dtype=np.float32),
        imu_accel=np.broadcast_to(np.array([0, 0, -9.81], dtype=np.float32), (samples, 3)),
        cmd_seq=np.zeros(samples, dtype=np.uint64),
        t_cmd_ns=np.zeros(samples, dtype=np.int64),
        q_cmd=command_nan,
        dq_cmd=command_nan,
        kp=command_nan,
        kd=command_nan,
        tau_ff=command_nan,
        command_available=np.asarray(False),
    )
    manifest = {
        "result": "completed",
        "command_available": False,
        "num_motors": 20,
        "num_joints": 19,
        "joint_names": list(_ANALYZER.config.H1_JOINT_NAMES),
        "joint_to_motor": list(_ANALYZER.config.H1_JOINT_TO_MOTOR),
        "unused_motor_id": _ANALYZER.config.H1_UNUSED_MOTOR_ID,
        "stats": {"malformed": 0, "dropped": 0, "commands": 0},
        "experiment": {
            "kind": "observe",
            "publish_lowcmd": False,
            "observe_duration_s": 0.2,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze_observe(raw, make_plot=True)
    assert report["overall_pass"]
    assert report["timing"]["median_rate_hz"] == 500.0
    assert report["recorder"]["commands"] == 0
    assert (tmp_path / "state_report.json").is_file()
    assert (tmp_path / "state_overview.png").is_file()

    with np.testing.assert_raises_regex(ValueError, "command_available=false"):
        convert(
            Namespace(
                input=raw,
                output=tmp_path / "must_not_exist.pt",
                timestamp_unit="auto",
                frequency=500.0,
                allow_varying_gains=False,
                max_gap_factor=5.0,
            )
        )
