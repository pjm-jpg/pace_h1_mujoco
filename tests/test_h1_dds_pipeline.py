from argparse import Namespace
import importlib.util
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
