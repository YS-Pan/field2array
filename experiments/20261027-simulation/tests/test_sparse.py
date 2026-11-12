"""Independent sparse-fault oracles, using only synthetic observations.

Truth is used only to construct and assess test cases. It is never supplied to
the solver or its model-selection rule. No measurement files are opened.
"""

from dataclasses import is_dataclass
import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from array_reconstruct.sparse import prepare_sparse_faults, solve_sparse_faults


def _complex_matrix(rng, rows, columns):
    return (rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))) / np.sqrt(rows)


@pytest.fixture
def mixed_fault_problem():
    rng = np.random.default_rng(20261027)
    a = _complex_matrix(rng, 100, 20)
    v = _complex_matrix(rng, 67, 20)
    nominal = (0.8 + 0.4 * rng.random(20)) * np.exp(1j * rng.uniform(-np.pi, np.pi, 20))
    gain = 0.73 * np.exp(0.41j)
    coefficients = gain * nominal
    support = np.array([2, 8, 17])
    # An amplitude error, a phase error, and one dead element, respectively.
    coefficients[support] *= np.array([0.43, np.exp(0.85j), 0.0])
    return a, v, nominal, gain, coefficients, support


def _fixed_support_oracle(a, y, nominal, support, weights=None):
    """Independent physical-coordinate augmented LS using NumPy's solver."""
    if weights is None:
        weights = np.ones(a.shape[0])
    design = np.column_stack((a @ nominal, a[:, np.asarray(support, dtype=int)]))
    fitted = np.linalg.lstsq(weights[:, None] * design, weights * y, rcond=None)[0]
    correction = np.zeros(a.shape[1], dtype=complex)
    correction[np.asarray(support, dtype=int)] = fitted[1:]
    return fitted[0], correction, fitted[0] * nominal + correction


def _assert_selected_path_consistency(result, a, y, nominal, v, vy, weights=None, validation_weights=None):
    weights = np.ones(len(y)) if weights is None else weights
    validation_weights = np.ones(len(vy)) if validation_weights is None else validation_weights
    sizes = np.asarray(result.support_size_curve)
    training = np.asarray(result.training_score_curve)
    validation = np.asarray(result.validation_score_curve)
    support = np.asarray(result.support, dtype=int)
    assert sizes.ndim == training.ndim == validation.ndim == 1
    assert len(sizes) == len(training) == len(validation) >= 1
    assert_array_equal(sizes, np.arange(len(sizes)))
    assert result.support_size == len(support) == len(np.unique(support))
    assert np.all(np.isfinite(training)) and np.all(training >= 0)
    assert np.all(np.isfinite(validation)) and np.all(validation >= 0)
    assert np.all(np.diff(training) <= 2e-13 * max(1.0, np.max(training)))
    assert len(result.support_path) == len(result.gain_path) == len(sizes)
    for index, path_support in enumerate(result.support_path):
        path_support = np.asarray(path_support, dtype=int)
        assert len(path_support) == sizes[index]
        assert len(np.unique(path_support)) == len(path_support)
        assert np.all((path_support >= 0) & (path_support < a.shape[1]))
        path_gain, _, path_coefficients = _fixed_support_oracle(a, y, nominal, path_support, weights)
        assert_allclose(result.gain_path[index], path_gain, atol=3e-13, rtol=2e-12)
        path_training_error = weights * (y - a @ path_coefficients)
        path_validation_error = validation_weights * (vy - v @ path_coefficients)
        assert_allclose(training[index], np.mean(np.abs(path_training_error) ** 2),
                        atol=2e-13, rtol=1e-11)
        assert_allclose(validation[index], np.mean(np.abs(path_validation_error) ** 2),
                        atol=2e-13, rtol=1e-11)
    chosen = np.flatnonzero(sizes == result.support_size)
    assert len(chosen) == 1
    chosen = chosen[0]
    train_error = weights * (y - a @ result.coefficients)
    validation_error = validation_weights * (vy - v @ result.coefficients)
    assert_allclose(training[chosen], np.mean(np.abs(train_error) ** 2), atol=2e-13, rtol=1e-11)
    assert_allclose(validation[chosen], np.mean(np.abs(validation_error) ** 2), atol=2e-13, rtol=1e-11)
    assert validation[chosen] <= np.min(validation) + 2e-13 * max(1.0, np.max(validation))
    assert_allclose(result.predicted, a @ result.coefficients, atol=2e-13, rtol=1e-11)
    assert_allclose(result.residual, y - result.predicted, atol=2e-13, rtol=1e-11)
    assert_allclose(result.coefficients, result.baseline_gain * nominal + result.correction,
                    atol=2e-13, rtol=1e-11)
    outside = np.setdiff1d(np.arange(a.shape[1]), support)
    assert_array_equal(np.asarray(result.correction)[outside], np.zeros(len(outside), dtype=complex))
    assert "validation" in result.selection_method.lower()


def test_common_complex_gain_has_no_element_faults(mixed_fault_problem):
    a, v, nominal, gain, _, _ = mixed_fault_problem
    y, vy = a @ (gain * nominal), v @ (gain * nominal)
    result = solve_sparse_faults(a, y, nominal, v, vy, max_faults=5)
    assert is_dataclass(result)
    assert result.support_size == 0
    assert np.asarray(result.support).size == 0
    assert_allclose(result.baseline_gain, gain, atol=2e-13)
    assert_allclose(result.coefficients, gain * nominal, atol=3e-13)
    assert_allclose(result.correction, 0, atol=0)
    assert_allclose(result["coefficients"], result.coefficients, atol=0)
    _assert_selected_path_consistency(result, a, y, nominal, v, vy)


def test_three_mixed_faults_are_recovered_with_independent_validation_matrix(mixed_fault_problem):
    a, v, nominal, gain, coefficients, support = mixed_fault_problem
    y, vy = a @ coefficients, v @ coefficients
    result = solve_sparse_faults(a, y, nominal, v, vy, max_faults=5)
    assert result.support_size == 3
    assert_array_equal(np.sort(result.support), support)
    assert_allclose(result.coefficients, coefficients, atol=3e-12, rtol=3e-12)
    assert_allclose(result.baseline_gain, gain, atol=3e-12)
    assert result.validation_relative_error < 2e-12
    oracle_gain, oracle_delta, oracle_coefficients = _fixed_support_oracle(a, y, nominal, result.support)
    assert_allclose(result.baseline_gain, oracle_gain, atol=2e-13, rtol=2e-12)
    assert_allclose(result.correction, oracle_delta, atol=3e-13, rtol=2e-12)
    assert_allclose(result.coefficients, oracle_coefficients, atol=3e-13, rtol=2e-12)
    _assert_selected_path_consistency(result, a, y, nominal, v, vy)


def test_hand_oracle_whitens_by_inverse_std_and_rejects_training_only_fault():
    # WA has three mutually orthogonal nonzero columns: h=(1,1,0),
    # f=(0,0,1), v=(1,-1,0). The first column is entirely common gain.
    # y_train = h+f+v/4, but independent validation contains only h+f.
    a = np.zeros((3, 8), dtype=complex)
    a[:, :3] = [[1, 0, 1], [2, 0, -2], [0, 1, 0]]
    nominal = np.zeros(8, dtype=complex)
    nominal[0] = 1
    y = np.array([1.25, 1.5, 1.0], dtype=complex)
    weights = np.array([1.0, 0.5, 1.0])  # 1/sigma; not 1/sigma**2.
    v = np.eye(8, dtype=complex)
    vy = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=complex)
    result = solve_sparse_faults(a, y, nominal, v, vy, max_faults=2, weights=weights)
    assert_array_equal(result.support, [1])
    assert result.support_size == 1
    assert_allclose(result.baseline_gain, 1.0, atol=2e-14)
    assert_allclose(result.coefficients, vy, atol=2e-14)
    assert_array_equal(result.support_size_curve, [0, 1, 2])
    assert_allclose(result.training_score_curve, [3 / 8, 1 / 24, 0], atol=2e-15)
    assert_allclose(result.validation_score_curve, [1 / 8, 0, 1 / 128], atol=2e-15)
    assert 0 not in result.support  # Its projected column is exactly zero.
    _assert_selected_path_consistency(result, a, y, nominal, v, vy, weights)


def test_noisy_fit_matches_augmented_weighted_ls_in_physical_coordinates():
    rng = np.random.default_rng(718)
    a, v = _complex_matrix(rng, 35, 8), _complex_matrix(rng, 41, 8)
    nominal = np.exp(1j * np.linspace(-0.4, 0.7, 8))
    weights = np.geomspace(0.2, 8, 35)
    y = (0.9 + 0.2j) * (a @ nominal) + (1.1 - 0.3j) * a[:, 3]
    y += 0.02 * (rng.normal(size=35) + 1j * rng.normal(size=35))
    gain, delta, coefficients = _fixed_support_oracle(a, y, nominal, [3], weights)
    # The validation target is this independently computed constrained optimum.
    vy = v @ coefficients
    validation_weights = np.geomspace(0.3, 2.0, len(vy))
    result = solve_sparse_faults(a, y, nominal, v, vy, max_faults=1,
                                 weights=weights, validation_weights=validation_weights)
    assert_array_equal(result.support, [3])
    assert_allclose(result.baseline_gain, gain, atol=3e-13, rtol=2e-12)
    assert_allclose(result.correction, delta, atol=3e-13, rtol=2e-12)
    assert_allclose(result.coefficients, coefficients, atol=3e-13, rtol=2e-12)
    wrong_gain, _, wrong_coefficients = _fixed_support_oracle(a, y, nominal, [3], weights**2)
    assert abs(wrong_gain - gain) > 1e-4
    assert np.linalg.norm(wrong_coefficients - coefficients) > 1e-4
    _assert_selected_path_consistency(result, a, y, nominal, v, vy, weights, validation_weights)


def test_external_validation_observations_alone_change_selected_order():
    a, v, nominal = np.eye(8, dtype=complex), -np.eye(8, dtype=complex), np.ones(8, dtype=complex)
    faulted = nominal.copy()
    faulted[0] += 1
    y = a @ faulted
    no_fault_validation = solve_sparse_faults(a, y, nominal, v, v @ nominal, max_faults=1)
    fault_validation = solve_sparse_faults(a, y, nominal, v, v @ faulted, max_faults=1)
    assert no_fault_validation.support_size == 0
    assert fault_validation.support_size == 1
    assert_array_equal(fault_validation.support, [0])
    assert_allclose(no_fault_validation.training_score_curve, fault_validation.training_score_curve, atol=0)
    assert "truth" not in inspect.signature(solve_sparse_faults).parameters


def test_validation_weights_change_selection_by_weighted_mse():
    a, v, nominal = np.eye(8, dtype=complex), -np.eye(8, dtype=complex), np.ones(8, dtype=complex)
    y = nominal.copy()
    y[0] = 2
    validation_coefficients = np.full(8, 1.125, dtype=complex)
    validation_coefficients[0] = 2
    vy = v @ validation_coefficients
    unweighted = solve_sparse_faults(a, y, nominal, v, vy, max_faults=1)
    weights = np.ones(8)
    weights[0] = 0.01
    weighted = solve_sparse_faults(a, y, nominal, v, vy, max_faults=1, validation_weights=weights)
    assert unweighted.support_size == 1
    assert weighted.support_size == 0
    _assert_selected_path_consistency(weighted, a, y, nominal, v, vy, validation_weights=weights)


def test_column_rescaling_and_matching_nominal_coordinate_change_are_invariant(mixed_fault_problem):
    a, v, nominal, _, coefficients, _ = mixed_fault_problem
    y, vy = a @ coefficients, v @ coefficients
    original = solve_sparse_faults(a, y, nominal, v, vy, max_faults=5)
    scale = np.geomspace(0.1, 10, a.shape[1]) * np.exp(1j * np.linspace(-0.3, 0.4, a.shape[1]))
    changed = solve_sparse_faults(a * scale, y, nominal / scale, v * scale, vy, max_faults=5)
    assert_array_equal(changed.support, original.support)
    assert_allclose(changed.coefficients * scale, original.coefficients, atol=4e-12, rtol=4e-12)
    assert_allclose(changed.baseline_gain, original.baseline_gain, atol=4e-12)
    assert_allclose(changed.predicted, original.predicted, atol=4e-12)


def test_prepared_cache_matches_uncached_and_accepts_omitted_nominal(mixed_fault_problem):
    a, v, nominal, _, coefficients, _ = mixed_fault_problem
    y, vy = a @ coefficients, v @ coefficients
    weights = np.geomspace(0.5, 2, len(y))
    prepared = prepare_sparse_faults(a, nominal, weights=weights)
    direct = solve_sparse_faults(a, y, nominal, v, vy, max_faults=5, weights=weights)
    cached = solve_sparse_faults(prepared, y, validation_matrix=v, validation_y=vy, max_faults=5)
    assert_array_equal(cached.support, direct.support)
    for name in ("coefficients", "correction", "baseline_gain", "predicted", "residual",
                 "training_score_curve", "validation_score_curve"):
        assert_allclose(cached[name], direct[name], atol=3e-13, rtol=2e-12)


def test_zero_observations_choose_zero_gain_without_nans(mixed_fault_problem):
    a, v, nominal, _, _, _ = mixed_fault_problem
    y, vy = np.zeros(a.shape[0], dtype=complex), np.zeros(v.shape[0], dtype=complex)
    result = solve_sparse_faults(a, y, nominal, v, vy, max_faults=5)
    assert result.support_size == 0
    assert_allclose(result.baseline_gain, 0, atol=0)
    assert_allclose(result.coefficients, 0, atol=0)
    assert np.isfinite(result.validation_relative_error)
    assert result.validation_relative_error == 0
    _assert_selected_path_consistency(result, a, y, nominal, v, vy)


def test_zero_fault_cap_is_baseline_only_and_default_cap_obeys_minority_assumption():
    rng = np.random.default_rng(435)
    a, v = _complex_matrix(rng, 25, 12), _complex_matrix(rng, 21, 12)
    nominal = np.ones(12, dtype=complex)
    y = a @ nominal + 0.1 * (rng.normal(size=25) + 1j * rng.normal(size=25))
    vy = v @ nominal
    baseline = solve_sparse_faults(a, y, nominal, v, vy, max_faults=0)
    assert_array_equal(baseline.support_size_curve, [0])
    assert baseline.support_size == 0
    default = solve_sparse_faults(a, y, nominal, v, vy)
    assert np.max(default.support_size_curve) <= min(32, a.shape[1] // 4)
    _assert_selected_path_consistency(default, a, y, nominal, v, vy)


def test_default_cap_is_zero_below_four_elements_and_hard_cap_is_32():
    small = np.eye(3, dtype=complex)
    small_nominal = np.ones(3, dtype=complex)
    result = solve_sparse_faults(small, small_nominal, small_nominal, -small, -small_nominal)
    assert result.max_faults == 0
    assert_array_equal(result.support_size_curve, [0])
    large = np.eye(132, dtype=complex)
    nominal = np.ones(132, dtype=complex)
    with pytest.raises(ValueError):
        solve_sparse_faults(large, nominal, nominal, -large, -nominal, max_faults=33)


@pytest.mark.parametrize("change", [
    {"nominal": np.zeros(8)},
    {"nominal": np.ones(7)},
    {"y": np.ones(7)},
    {"y": np.full(8, np.nan)},
    {"A": np.full((8, 8), np.inf)},
    {"validation_matrix": np.ones((10, 7))},
    {"validation_y": np.ones(9)},
    {"validation_matrix": None, "validation_y": None},
    {"weights": np.zeros(8)},
    {"weights": np.full(8, -1.0)},
    {"weights": np.full(8, np.nan)},
    {"weights": np.ones(8, dtype=complex)},
    {"weights": np.ones(7)},
    {"validation_weights": np.zeros(10)},
    {"validation_weights": np.ones(9)},
    {"max_faults": -1},
    {"max_faults": 3},  # N=8 admits at most floor(N/4)=2 sparse faults.
    {"max_faults": 1.5},
    {"max_faults": True},
])
def test_invalid_inputs_are_rejected(change):
    inputs = {
        "A": np.eye(8, dtype=complex),
        "y": np.ones(8, dtype=complex),
        "nominal": np.ones(8, dtype=complex),
        "validation_matrix": np.ones((10, 8), dtype=complex),
        "validation_y": np.ones(10, dtype=complex),
        "max_faults": 1,
    }
    inputs.update(change)
    with pytest.raises(ValueError):
        solve_sparse_faults(**inputs)
