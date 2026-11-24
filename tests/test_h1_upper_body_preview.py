import xml.etree.ElementTree as ET

from sysid_mujoco.preview_h1_real_experiment import (
    _exclude_adjacent_body_contacts,
    _initial_joint_vector_from_npz,
)
from sysid_mujoco.validate_h1_upper_body_experiments import (
    PROFILES,
    _run_acceptance,
)


def _passing_metrics() -> dict[str, object]:
    return {
        "excitation_center_rad": 0.2,
        "target_min_rad": 0.15,
        "target_max_rad": 0.25,
        "max_all_joint_following_error_rad": 0.06,
        "minimum_joint_limit_margin_rad": 0.30,
        "self_contact_count": 0,
        "final_return_error_rad": 0.01,
        "collision_check_enabled": True,
        "safety_check_pass": True,
    }


def test_adjacent_contact_exclusions_only_cover_direct_neighbors() -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis">
              <body name="torso">
                <body name="upper_arm" />
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    _exclude_adjacent_body_contacts(root)
    _exclude_adjacent_body_contacts(root)

    pairs = [
        (exclude.get("body1"), exclude.get("body2"))
        for exclude in root.findall("./contact/exclude")
    ]
    assert pairs.count(("pelvis", "torso")) == 1
    assert pairs.count(("torso", "upper_arm")) == 1
    assert ("pelvis", "upper_arm") not in pairs


def test_upper_body_profiles_stay_within_planned_amplitude_limit() -> None:
    assert len(PROFILES) == 9
    for profile in PROFILES.values():
        assert 0.0 < float(profile["step"]) <= 0.05
        assert all(0.0 < float(value) <= 0.05 for value in profile["multi"])

    for side in ("left", "right"):
        roll = PROFILES[f"{side}_shoulder_roll_joint"]
        assert roll["step"] == 0.02
        assert roll["multi"] == (0.03, 0.025, 0.015)


def test_acceptance_accepts_a_safe_bidirectional_preview() -> None:
    accepted, failures = _run_acceptance(_passing_metrics())
    assert accepted
    assert failures == []


def test_acceptance_rejects_nonadjacent_self_contact() -> None:
    metrics = _passing_metrics()
    metrics["self_contact_count"] = 1

    accepted, failures = _run_acceptance(metrics)
    assert not accepted
    assert "nonadjacent self-contact detected" in failures


def test_initial_joint_vector_maps_20_motor_slots_to_19_joints(tmp_path) -> None:
    import numpy as np

    from config import H1_JOINT_TO_MOTOR

    q_real = np.vstack((np.arange(20), np.arange(20) + 2.0))
    path = tmp_path / "raw.npz"
    np.savez(path, q_real=q_real)

    actual = _initial_joint_vector_from_npz(path)
    expected = np.asarray(H1_JOINT_TO_MOTOR, dtype=np.float64) + 1.0
    np.testing.assert_allclose(actual, expected)
