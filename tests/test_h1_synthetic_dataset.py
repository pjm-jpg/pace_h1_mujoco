import numpy as np

from datasets.generate_h1_synthetic_dataset import fit_reference
from datasets.generate_h1_synthetic_dataset import validation_reference


def test_fit_reference_settles_then_excites_multiple_frequencies():
    times = np.arange(0.0, 3.0, 0.002)
    position, velocity = fit_reference(times, 0.22)
    assert np.all(position[times <= 0.5] == 0.0)
    assert np.all(velocity[times <= 0.5] == 0.0)
    assert 0.05 < np.max(np.abs(position)) <= 0.22
    assert np.all(np.isfinite(velocity))


def test_validation_reference_is_distinct_held_step_sequence():
    times = np.arange(0.0, 5.0, 0.002)
    position, velocity = validation_reference(times, 0.2)
    np.testing.assert_allclose(position[times < 1.0], 0.0)
    np.testing.assert_allclose(position[(times >= 1.0) & (times < 2.0)], 0.15)
    np.testing.assert_allclose(position[(times >= 2.0) & (times < 3.2)], -0.09)
    np.testing.assert_allclose(velocity, 0.0)
