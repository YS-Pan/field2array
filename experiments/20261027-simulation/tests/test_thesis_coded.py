"""Independent small synthetic oracles for joint coded measurements.

No measured inputs or full experiment runners are used. Coefficient truth and
state errors construct observations only; the inverse receives nominal codes.
"""

import inspect

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from array_reconstruct import thesis_coded as coded
from array_reconstruct.inverse import prepare_tikhonov, solve_tikhonov


def _joint(matrix, codes):
    return np.vstack([matrix * pattern[None, :] for pattern in codes])


def _fields(matrix, coefficients, codes):
    return np.asarray([matrix @ (pattern * coefficients) for pattern in codes])


@pytest.mark.parametrize("side", [2, 3])
@pytest.mark.parametrize("count", [1, 2, 4])
def test_patterns_follow_row_column_parities_in_row_major_order(side, count):
    row, column = np.indices((side, side))
    ones = np.ones(side * side)
    row_code = (-1.0) ** row.ravel()
    column_code = (-1.0) ** column.ravel()
    checker = (-1.0) ** (row + column).ravel()
    expected = {1: [ones], 2: [ones, checker], 4: [ones, row_code, column_code, checker]}[count]
    actual = coded.code_patterns(side, count)
    assert actual.shape == (count, side * side)
    assert np.iscomplexobj(actual)
    assert_array_equal(actual, expected)
    assert_array_equal(np.abs(actual), np.ones_like(actual.real))


def test_k2_cancels_opposite_parity_gram_terms_but_preserves_same_parity_ambiguity():
    matrix = np.ones((1, 4), dtype=complex)
    codes = coded.code_patterns(2, 2)
    prepared = coded.prepare_coded_ridge(matrix, codes)
    checker = np.array([1, -1, -1, 1])
    expected_gram = 1 + checker[:, None] * checker[None, :]
    assert_allclose(prepared.joint_matrix.conj().T @ prepared.joint_matrix, expected_gram, atol=0)
    assert np.linalg.matrix_rank(matrix) == 1
    assert np.linalg.matrix_rank(prepared.joint_matrix) == 2
    # Elements 0 and 3 are still indistinguishable under both K2 states.
    assert_allclose(prepared.joint_matrix @ [1, 0, 0, -1], 0, atol=0)
    assert expected_gram[0, 3] == 2
    assert expected_gram[0, 1] == 0


def test_k4_two_by_two_hadamard_operator_has_exact_column_normalization():
    matrix = np.ones((1, 4), dtype=complex)
    codes = coded.code_patterns(2, 4)
    prepared = coded.prepare_coded_ridge(matrix, codes)
    hadamard = np.array([
        [1, 1, 1, 1], [1, 1, -1, -1],
        [1, -1, 1, -1], [1, -1, -1, 1],
    ], dtype=complex)
    assert_array_equal(prepared.joint_matrix, hadamard)
    assert_allclose(prepared.joint_matrix.conj().T @ prepared.joint_matrix, 4 * np.eye(4), atol=0)
    assert_allclose(prepared.column_scale, np.full(4, 2.0), atol=0)
    assert_allclose(prepared.singular_values, np.ones(4), atol=5e-15)


def test_k4_larger_array_still_preserves_within_parity_class_ambiguity():
    matrix = np.ones((1, 16), dtype=complex)
    prepared = coded.prepare_coded_ridge(matrix, coded.code_patterns(4, 4))
    assert np.linalg.matrix_rank(prepared.joint_matrix) == 4
    ambiguity = np.zeros(16)
    ambiguity[0], ambiguity[2] = 1, -1  # row 0, columns 0 and 2 share all four codes.
    assert_allclose(prepared.joint_matrix @ ambiguity, 0, atol=0)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_prepared_operator_and_ridge_candidates_match_explicit_stack_svd(count):
    rng = np.random.default_rng(783)
    matrix = rng.normal(size=(11, 9)) + 1j * rng.normal(size=(11, 9))
    validation_matrix = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
    codes = coded.code_patterns(3, count)
    coefficients = (0.8 + 0.3 * rng.random(9)) * np.exp(0.3j * rng.normal(size=9))
    fields = _fields(matrix, coefficients, codes)
    validation_fields = _fields(validation_matrix, coefficients, codes)
    fields += 0.03 * (rng.normal(size=fields.shape) + 1j * rng.normal(size=fields.shape))
    validation_fields += 0.02 * (
        rng.normal(size=validation_fields.shape) + 1j * rng.normal(size=validation_fields.shape)
    )
    joint = _joint(matrix, codes)
    validation_joint = _joint(validation_matrix, codes)
    flattened = fields.ravel()
    validation_flattened = validation_fields.ravel()
    nominal_field = joint @ np.ones(matrix.shape[1])
    gain = np.vdot(nominal_field, flattened) / np.vdot(nominal_field, nominal_field).real
    prior = gain * np.ones(matrix.shape[1], dtype=complex)
    alphas = np.array([0.0, 0.03, 0.3, 1.0])
    prepared = coded.prepare_coded_ridge(matrix, codes)
    standard_prepared = prepare_tikhonov(joint)
    assert_allclose(prepared.joint_matrix, joint, atol=0)
    assert_allclose(prepared.column_scale, np.sqrt(count) * np.linalg.norm(matrix, axis=0), rtol=3e-15)
    assert_allclose(prepared.singular_values, standard_prepared.singular_values, atol=3e-14, rtol=3e-14)

    candidate_scores = []
    candidates = []
    for alpha in alphas:
        actual = coded.solve_coded_ridge(
            prepared, fields, validation_matrix, validation_fields, alphas=[alpha],
        )
        expected = solve_tikhonov(
            standard_prepared, flattened, alphas=[alpha], prior=prior,
            validation_matrix=validation_joint, validation_y=validation_flattened,
        )
        assert_allclose(actual["gain"], gain, atol=3e-14, rtol=3e-14)
        assert_allclose(actual["coefficients"], expected.coefficients, atol=5e-13, rtol=5e-13)
        assert_allclose(actual["score_curve"], expected.score_curve, atol=3e-14, rtol=3e-13)
        assert_allclose(actual["selection_field_nrmse"], expected.validation_relative_error, atol=3e-13)
        assert_allclose(actual["training_field_nrmse"], expected.relative_residual, atol=3e-13)
        assert_allclose(actual["regularized_effective_rank"], expected.effective_rank, atol=3e-13)
        candidate_scores.append(expected.score_curve[0])
        candidates.append(expected.coefficients)
    selected = coded.solve_coded_ridge(
        prepared, fields, validation_matrix, validation_fields, alphas=alphas,
    )
    chosen = int(np.argmin(candidate_scores))
    assert selected["selected_alpha"] == alphas[chosen]
    assert_array_equal(selected["alpha_curve"], alphas)
    assert_allclose(selected["score_curve"], candidate_scores, atol=3e-14, rtol=3e-13)
    assert_allclose(selected["coefficients"], candidates[chosen], atol=5e-13, rtol=5e-13)


def test_k4_ridge_matches_hand_solution_and_common_gain_uses_training_only():
    matrix = np.ones((1, 4), dtype=complex)
    codes = coded.code_patterns(2, 4)
    coefficients = np.array([1 + 2j, 2 - 1j, -1 + 0.5j, 0.2 + 0.7j])
    fields = _fields(matrix, coefficients, codes)
    alpha = 0.35
    prepared = coded.prepare_coded_ridge(matrix, codes)
    result = coded.solve_coded_ridge(prepared, fields, matrix, fields, alphas=[alpha])
    expected_gain = 0.55 + 0.55j
    # B=H/2 is unitary, so lambda=alpha and every correction shrinks equally.
    expected_coefficients = expected_gain + (coefficients - expected_gain) / (1 + alpha**2)
    assert_allclose(result["gain"], expected_gain, atol=3e-15)
    assert_allclose(result["coefficients"], expected_coefficients, atol=5e-15)
    altered_validation = coded.solve_coded_ridge(
        prepared, fields, matrix, -fields, alphas=[0.0, alpha, 1.0],
    )
    assert_allclose(altered_validation["gain"], expected_gain, atol=3e-15)


@pytest.mark.parametrize("count", [2, 4])
def test_uncoded_repetition_does_not_improve_normalized_singular_spectrum(count):
    rng = np.random.default_rng(174)
    matrix = rng.normal(size=(9, 4)) + 1j * rng.normal(size=(9, 4))
    codes = np.ones((count, 4), dtype=complex)
    single = coded.prepare_coded_ridge(matrix, np.ones((1, 4), dtype=complex))
    repeated = coded.prepare_coded_ridge(matrix, codes)
    assert_allclose(repeated.column_scale, np.sqrt(count) * single.column_scale, rtol=3e-15)
    assert_allclose(repeated.singular_values, single.singular_values, atol=2e-14, rtol=2e-14)
    assert_allclose(
        np.linalg.svd(repeated.joint_matrix, compute_uv=False),
        np.sqrt(count) * np.linalg.svd(single.joint_matrix, compute_uv=False),
        atol=2e-14, rtol=2e-14,
    )


def test_noise_sigma_is_fixed_even_for_zero_clean_coded_states():
    matrix = np.ones((1, 4), dtype=complex)
    coefficients = np.ones(4, dtype=complex)
    codes = coded.code_patterns(2, 4)
    unit_noise = np.array([[1], [1j], [-1], [-1j]], dtype=complex)
    sigma = 0.04  # uncoded RMS 4 / 100; no state-specific rescaling.
    result = coded.coded_observations(matrix, coefficients, codes, unit_noise, sigma)
    assert_array_equal(result["clean"], np.array([[4], [0], [0], [0]], dtype=complex))
    assert_array_equal(result["noise"], sigma * unit_noise)
    assert_array_equal(result["field"], result["clean"] + sigma * unit_noise)
    assert np.all(np.abs(result["noise"][1:]) == sigma)


def test_observation_state_errors_only_modify_actual_codes_without_mutating_inputs():
    matrix = np.array([[1, 2j, -0.3, 0.4j], [0.2j, 0.5, -1j, 2]], dtype=complex)
    coefficients = np.array([0.8 + 0.3j, 1.2 - 0.2j, 0.9j, -0.1 + 0.7j])
    codes = coded.code_patterns(2, 2)
    state_errors = np.ones_like(codes)
    state_errors[1] = 10 ** (np.array([0.4, -0.7, 0.2, -0.3]) / 20) * np.exp(
        1j * np.deg2rad([3, -2, 4, 1])
    )
    unit_noise = np.array([[1 + 1j, -0.5j], [0.3 - 0.1j, -1]], dtype=complex)
    original_codes, original_errors = codes.copy(), state_errors.copy()
    result = coded.coded_observations(
        matrix, coefficients, codes, unit_noise, 0.015, state_errors=state_errors,
    )
    expected = _fields(matrix, coefficients, codes * state_errors)
    assert_allclose(result["clean"], expected, atol=2e-15)
    assert_array_equal(result["noise"], 0.015 * unit_noise)
    assert_allclose(result["field"], expected + 0.015 * unit_noise, atol=2e-15)
    assert_array_equal(codes, original_codes)
    assert_array_equal(state_errors, original_errors)


@pytest.mark.parametrize("amplitude_std_db,phase_std_deg", [(0.2, 3.0), (1.0, 5.0)])
def test_state_error_factors_use_exact_gaussian_streams_and_keep_reference_unchanged(
    amplitude_std_db, phase_std_deg,
):
    seed, count = 20261112, 17
    factors, amplitude_db, phase_deg = coded.state_error_factors(
        seed, count, amplitude_std_db=amplitude_std_db, phase_std_deg=phase_std_deg,
    )
    assert factors.shape == amplitude_db.shape == phase_deg.shape == (4, count)
    assert_array_equal(factors[0], np.ones(count, dtype=complex))
    assert_array_equal(amplitude_db[0], np.zeros(count))
    assert_array_equal(phase_deg[0], np.zeros(count))
    for state_id in (1, 2, 3):
        amplitude_seed, phase_seed = np.random.SeedSequence(
            seed, spawn_key=(64002, state_id),
        ).spawn(2)
        expected_amplitude = np.random.default_rng(amplitude_seed).normal(
            0.0, amplitude_std_db, count,
        )
        expected_phase = np.random.default_rng(phase_seed).normal(0.0, phase_std_deg, count)
        expected_factors = 10 ** (expected_amplitude / 20) * np.exp(
            1j * np.deg2rad(expected_phase)
        )
        assert_array_equal(amplitude_db[state_id], expected_amplitude)
        assert_array_equal(phase_deg[state_id], expected_phase)
        assert_allclose(factors[state_id], expected_factors, atol=0, rtol=2e-15)
    repeated = coded.state_error_factors(
        seed, count, amplitude_std_db=amplitude_std_db, phase_std_deg=phase_std_deg,
    )
    for actual, expected in zip(repeated, (factors, amplitude_db, phase_deg)):
        assert_array_equal(actual, expected)


def test_unmodeled_state_error_can_bias_coefficients_despite_zero_residual():
    matrix = np.ones((1, 4), dtype=complex)
    coefficients = np.ones(4, dtype=complex)
    codes = coded.code_patterns(2, 4)
    state_errors = np.ones_like(codes)
    epsilon = 0.2
    state_errors[1, 0] += epsilon
    observed = coded.coded_observations(
        matrix, coefficients, codes, np.zeros((4, 1), dtype=complex), 0.0,
        state_errors=state_errors,
    )
    assert_allclose(observed["clean"].ravel(), [4, epsilon, 0, 0], atol=3e-15)
    prepared = coded.prepare_coded_ridge(matrix, codes)
    fit = coded.solve_coded_ridge(
        prepared, observed["field"], matrix, observed["field"], alphas=[0.0],
    )
    expected = coefficients + epsilon / 4 * np.array([1, 1, -1, -1])
    assert_allclose(fit["coefficients"], expected, atol=5e-15)
    assert np.linalg.norm(fit["coefficients"] - coefficients) > 0.09
    assert_allclose(prepared.joint_matrix @ fit["coefficients"], observed["field"].ravel(), atol=5e-15)
    assert fit["training_field_nrmse"] < 1e-14
    parameters = inspect.signature(coded.solve_coded_ridge).parameters
    assert not any("truth" in name or "state_error" in name for name in parameters)
    # An oracle that knows the actual codes solves a different observation model.
    actual_prepared = coded.prepare_coded_ridge(matrix, codes * state_errors)
    oracle_fit = coded.solve_coded_ridge(
        actual_prepared, observed["field"], matrix, observed["field"], alphas=[0.0],
    )
    assert_allclose(oracle_fit["coefficients"], coefficients, atol=5e-15)


def test_canonical_noise_states_are_shared_across_k1_k2_k4_and_separated_by_plane():
    sample_count = 31
    streams = []
    for seed in (20261031, 20261101, 20261102):
        for plane_id in range(3):
            full = coded.noise_unit_samples(seed, plane_id, [0, 1, 2, 3], sample_count)
            pair = coded.noise_unit_samples(seed, plane_id, [0, 3], sample_count)
            single = coded.noise_unit_samples(seed, plane_id, [0], sample_count)
            assert full.shape == (4, sample_count)
            assert_array_equal(pair, full[[0, 3]])
            assert_array_equal(single, full[[0]])
            for state_id in range(4):
                rng = np.random.default_rng(np.random.SeedSequence(
                    seed, spawn_key=(64001, plane_id, state_id),
                ))
                expected = (
                    rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)
                ) / np.sqrt(2)
                assert_array_equal(full[state_id], expected)
            streams.extend(full)
    assert len(streams) == 36
    for index, stream in enumerate(streams):
        for other in streams[index + 1:]:
            assert not np.array_equal(stream, other)


@pytest.mark.parametrize("count", [2, 4])
def test_equal_budget_uncoded_averaging_reduces_noise_power_by_state_count(count):
    sample_count, sigma = 20000, 0.07
    unit_noise = coded.noise_unit_samples(20261102, 2, list(range(count)), sample_count)
    matrix = np.ones((sample_count, 4), dtype=complex)
    coefficients = np.ones(4, dtype=complex)
    codes = np.ones((count, 4), dtype=complex)
    observed = coded.coded_observations(matrix, coefficients, codes, unit_noise, sigma)
    assert_array_equal(observed["noise"], sigma * unit_noise)
    assert_allclose(np.mean(np.abs(unit_noise) ** 2, axis=1), 1, atol=0.03)
    assert_allclose(np.var(unit_noise.real), 0.5, atol=0.02)
    assert_allclose(np.var(unit_noise.imag), 0.5, atol=0.02)
    assert abs(np.mean(unit_noise**2)) < 0.025
    averaged_noise = np.mean(observed["field"], axis=0) - (matrix @ coefficients)
    assert_allclose(averaged_noise, sigma * np.mean(unit_noise, axis=0), atol=2e-15)
    assert_allclose(np.mean(np.abs(averaged_noise) ** 2), sigma**2 / count, rtol=0.03)
