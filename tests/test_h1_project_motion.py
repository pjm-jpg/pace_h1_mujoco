from pathlib import Path

import numpy as np

from datasets.collect_h1_project_motion import (
    ARM_JOINTS,
    DEFAULT_MODEL,
    DEFAULT_MOTION,
    ProjectMotionPlan,
    SUSPENDED_REFERENCE,
    _load_motion,
    _scaled_motion,
    _staged_waypoints,
    _target,
    _validate_plan,
)
import config
from datasets.generate_h1_upper_yaw_elbow_motion import generate_motion
from datasets.generate_h1_torso_motion import generate_motion as generate_torso_motion


def test_lower_body_walk1_plan_matches_reviewed_gate() -> None:
    plan = ProjectMotionPlan(motion_npz=str(DEFAULT_MOTION))
    source_q, fps = _load_motion(Path(plan.motion_npz))
    motion_q = _scaled_motion(source_q, plan)
    validation = _validate_plan(plan, motion_q, fps, DEFAULT_MODEL, 0.03, 3.0)

    assert np.isclose(plan.motion_source_duration_s, 11.58)
    assert np.isclose(plan.motion_execution_duration_s, 23.16)
    assert np.isclose(plan.total_duration_s, 63.16)
    assert validation["minimum_target_joint_limit_margin_rad"] > 0.27
    assert validation["maximum_target_velocity_rad_s"] < 1.64

    for name in ARM_JOINTS:
        index = config.H1_JOINT_NAMES.index(name)
        assert np.allclose(motion_q[:, index], SUSPENDED_REFERENCE[name])


def test_project_motion_returns_to_initial_pose() -> None:
    plan = ProjectMotionPlan(motion_npz=str(DEFAULT_MOTION))
    source_q, fps = _load_motion(Path(plan.motion_npz))
    motion_q = _scaled_motion(source_q, plan)
    initial_q = np.linspace(-0.1, 0.1, len(config.H1_JOINT_NAMES))

    start, phase = _target(0.0, initial_q, motion_q, fps, plan)
    end, end_phase = _target(
        plan.total_duration_s, initial_q, motion_q, fps, plan
    )
    assert phase == "prepare"
    assert end_phase == "return"
    assert np.allclose(start, initial_q)
    assert np.allclose(end, initial_q)


def test_upper_body_profile_holds_non_arm_joints() -> None:
    plan = ProjectMotionPlan(
        motion_npz=str(DEFAULT_MOTION),
        profile="upper-body",
    )
    source_q, _ = _load_motion(Path(plan.motion_npz))
    motion_q = _scaled_motion(source_q, plan)

    for name in config.H1_JOINT_NAMES:
        if name not in ARM_JOINTS:
            index = config.H1_JOINT_NAMES.index(name)
            assert np.allclose(motion_q[:, index], SUSPENDED_REFERENCE[name])


def test_upper_body_clearance_route_and_return() -> None:
    plan = ProjectMotionPlan(
        motion_npz=str(DEFAULT_MOTION),
        profile="upper-body",
        prepare_duration_s=24.0,
        return_duration_s=24.0,
    )
    source_q, fps = _load_motion(Path(plan.motion_npz))
    motion_q = _scaled_motion(source_q, plan)
    initial_q = np.linspace(-0.2, 0.2, len(config.H1_JOINT_NAMES))
    prepare_waypoints, prepare_weights = _staged_waypoints(initial_q, motion_q[0])
    reverse_waypoints, reverse_weights = _staged_waypoints(initial_q, motion_q[-1])
    return_waypoints = tuple(reversed(reverse_waypoints))
    return_weights = tuple(reversed(reverse_weights))

    start, _ = _target(
        0.0, initial_q, motion_q, fps, plan,
        prepare_waypoints, prepare_weights, return_waypoints, return_weights,
    )
    end, phase = _target(
        plan.total_duration_s, initial_q, motion_q, fps, plan,
        prepare_waypoints, prepare_weights, return_waypoints, return_weights,
    )
    assert np.allclose(start, initial_q)
    assert phase == "return"
    assert np.allclose(end, initial_q)


def test_generated_yaw_elbow_motion_has_large_smooth_coverage() -> None:
    joint_pos, metadata = generate_motion(50.0)
    q = joint_pos[:, 7:]
    index = {name: i for i, name in enumerate(config.H1_JOINT_NAMES)}

    assert q.shape == (1401, len(config.H1_JOINT_NAMES))
    assert np.isclose(metadata["duration_s"], 28.0)
    assert np.allclose(q[0], q[-1])
    assert np.max(np.abs(np.diff(q, axis=0))) < 0.055
    assert np.ptp(q[:, index["left_shoulder_yaw_joint"]]) >= 1.29
    assert np.ptp(q[:, index["right_shoulder_yaw_joint"]]) >= 1.29
    assert np.ptp(q[:, index["left_elbow_joint"]]) >= 1.19
    assert np.ptp(q[:, index["right_elbow_joint"]]) >= 1.19

    for name in config.H1_JOINT_NAMES:
        if name not in ARM_JOINTS:
            assert np.allclose(q[:, index[name]], SUSPENDED_REFERENCE[name])


def test_torso_only_profile_and_generated_motion_cover_dance_range() -> None:
    joint_pos, metadata = generate_torso_motion(50.0)
    source_q = joint_pos[:, 7:]
    plan = ProjectMotionPlan(
        motion_npz="unused.npz",
        profile="torso-only",
        amplitude_scale=0.90,
    )
    q = _scaled_motion(source_q, plan)
    torso_index = config.H1_JOINT_NAMES.index("torso_joint")

    assert q.shape == (1401, len(config.H1_JOINT_NAMES))
    assert np.isclose(metadata["duration_s"], 28.0)
    assert np.min(q[:, torso_index]) <= -0.17
    assert np.max(q[:, torso_index]) >= 0.25
    assert np.ptp(q[:, torso_index]) >= 0.43
    for name in config.H1_JOINT_NAMES:
        if name != "torso_joint":
            index = config.H1_JOINT_NAMES.index(name)
            assert np.allclose(q[:, index], SUSPENDED_REFERENCE[name])
