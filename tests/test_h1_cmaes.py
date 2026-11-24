import argparse
import json

import numpy as np
import pytest

from sysid_mujoco.cmaes_fit import _loss
from sysid_mujoco.merge_identified_models import merge_parameter_files
from sysid_mujoco.merge_identified_models import resolve_output_paths
from sysid_mujoco.merge_identified_models import write_merged_xml


def test_cmaes_loss_flattens_multiple_trajectory_residuals():
    def residual_fn(*_args, **_kwargs):
        residuals = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        return residuals, ["predicted"], ["measured"]

    loss = _loss(residual_fn, object(), np.zeros(3))

    assert loss == np.mean(np.square([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_cmaes_loss_accepts_single_flat_residual():
    def residual_fn(*_args, **_kwargs):
        return np.array([1.0, -1.0])

    assert _loss(residual_fn, object(), np.zeros(3)) == 1.0


def test_merge_identified_parameters_and_write_xml(tmp_path):
    source_a = tmp_path / "a.json"
    source_b = tmp_path / "b.json"
    source_a.write_text(
        json.dumps(
            {
                "robot": "h1",
                "parameters": {
                    "left_knee_joint": {
                        "armature": 0.4,
                        "damping": 5.0,
                        "frictionloss": 3.0,
                    }
                },
            }
        )
    )
    source_b.write_text(
        json.dumps(
            {
                "robot": "h1",
                "parameters": {
                    "right_knee_joint": {
                        "armature": 0.5,
                        "damping": 6.0,
                        "frictionloss": 2.0,
                    }
                },
            }
        )
    )
    base_xml = tmp_path / "base.xml"
    base_xml.write_text(
        '<mujoco><worldbody><body><joint name="left_knee_joint"/>'
        '<joint name="right_knee_joint"/></body></worldbody></mujoco>'
    )

    robot, parameters = merge_parameter_files([source_a, source_b])
    output_xml = tmp_path / "calibrated.xml"
    write_merged_xml(base_xml, output_xml, parameters)

    assert robot == "h1"
    text = output_xml.read_text()
    assert 'name="left_knee_joint" armature="0.4" damping="5" frictionloss="3"' in text
    assert 'name="right_knee_joint" armature="0.5" damping="6" frictionloss="2"' in text


def test_merge_rejects_duplicate_joint(tmp_path):
    payload = {
        "robot": "h1",
        "parameters": {"left_knee_joint": {"armature": 0.4}},
    }
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="more than one"):
        merge_parameter_files(paths)


def test_exact_output_path_keeps_manifest_beside_model(tmp_path):
    output_xml = tmp_path / "models" / "h1_calibrated.xml"
    args = argparse.Namespace(output_xml=output_xml, output_dir=None)

    resolved_xml, manifest = resolve_output_paths(args)

    assert resolved_xml == output_xml.resolve()
    assert manifest == output_xml.resolve().with_name("h1_calibrated_params.json")
