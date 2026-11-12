"""Independent analytic and optimization checks; all observations are synthetic.

These tests use hand-evaluated Green functions, sampled plane-wave eigenmodes,
and a separately assembled convex least-squares objective. No measured files,
measurement statistics, or inverse-generated coefficient truth are used.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.physics import (
    C0_M_S,
    angular_spectrum_backprop,
    green_matrix,
    planar_scan,
    rectangular_array,
)
from array_reconstruct.inverse import prepare_tikhonov, solve_tikhonov
from array_reconstruct.diagnostics import (
    coefficient_metrics,
    detect_anomalies,
    classification_metrics,
    matrix_identifiability,
    sampling_report,
)


def _synthetic_matrix():
    return np.array([
        [1 + 1j, 2 - 1j],
        [2 + 0.3j, -1 + 2j],
        [-0.4 + 1.2j, 3 + 0.2j],
        [1.4 - 0.1j, 0.2 + 0.7j],
        [0.7 + 0.6j, -0.3 - 1.2j],
        [-1.1 + 0.2j, 1.7 - 0.9j],
    ])


def _objective_oracle(a, y, weights, prior, alpha):
    """Solve the documented physical-coordinate objective independently.

    The production penalty is lambda^2 ||D(c-prior)||^2, where D is the
    weighted training-column norm. It is not unscaled isotropic ridge.
    The oracle solves a small, well-conditioned Hermitian normal equation;
    production instead uses an SVD of the column-normalized matrix.
    """
    weighted = weights[:, None] * a
    scale = np.sqrt(np.sum(abs(weighted) ** 2, axis=0))
    lam = alpha * np.linalg.norm(weighted / scale, ord=2)
    penalty = lam**2 * np.diag(scale**2)
    normal = weighted.conj().T @ weighted + penalty
    rhs = weighted.conj().T @ (weights * y) + penalty @ prior
    return np.linalg.solve(normal, rhs), lam, scale


def test_scalar_outgoing_green_function_hand_evaluated_phase_and_decay():
    # At f=c, wavelength=1 m. R=1/4 gives 4i; R=1/2 gives -2.
    result = green_matrix([[0, 0, 0]], [[0, 0, 0.25], [0, 0, 0.5]], C0_M_S)
    assert_allclose(result[:, 0], [4j, -2], atol=1e-14)
    assert_allclose(result[1, 0] / result[0, 0], 0.5j, atol=1e-14)


@pytest.mark.parametrize("orientation,expected", [
    ((1, 0, 0), -8 / np.pi + 4j * (1 - 4 / np.pi**2)),
    ((0, 0, 1), 16 / np.pi + 32j / np.pi**2),
])
def test_dipole_transverse_and_longitudinal_exact_axial_values(orientation, expected):
    result = green_matrix([[0, 0, 0]], [[0, 0, 0.25]], C0_M_S,
                          model="dipole", source_orientation=orientation,
                          receive_orientation=orientation)
    assert_allclose(result[0, 0], expected, rtol=2e-14)


def test_dipole_complex_receive_mode_uses_hermitian_projection():
    same = green_matrix([[0, 0, 0]], [[0, 0, 0.25]], C0_M_S,
                        model="dipole", source_orientation=(1, 1j, 0),
                        receive_orientation=(1, 1j, 0))
    orthogonal = green_matrix([[0, 0, 0]], [[0, 0, 0.25]], C0_M_S,
                              model="dipole", source_orientation=(1, 1j, 0),
                              receive_orientation=(1, -1j, 0))
    assert_allclose(same[0, 0], -8 / np.pi + 4j * (1 - 4 / np.pi**2), rtol=2e-14)
    assert_allclose(orthogonal, 0, atol=1e-14)


@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("frequency_hz", [10.7e9, 12.7e9])
def test_arbitrary_element_geometry_translation_and_chunk_invariance(staggered, frequency_hz):
    elements = rectangular_array(4, 3, 0.011)
    if staggered:
        elements[np.repeat(np.arange(3) % 2 == 1, 4), 0] += 0.0055
    scans = planar_scan(7, 6, 0.00835, z_m=0.060)
    expected = green_matrix(elements, scans, frequency_hz, chunk_size=1000)
    shift = np.array([0.123, -0.421, 0.037])
    actual = green_matrix(elements + shift, scans + shift, frequency_hz, chunk_size=3)
    assert_allclose(actual, expected, rtol=4e-14, atol=5e-13)


def test_rectangular_grid_element_order_has_x_fast_and_centered_origin():
    coordinates = rectangular_array(3, 2, 0.011, 0.017, z_m=0.004)
    expected = np.array([
        [-0.011, -0.0085, 0.004], [0, -0.0085, 0.004], [0.011, -0.0085, 0.004],
        [-0.011, 0.0085, 0.004], [0, 0.0085, 0.004], [0.011, 0.0085, 0.004],
    ])
    assert_allclose(coordinates, expected, atol=0, rtol=0)


def test_green_kernel_rejects_singularity_and_complex_coordinates():
    with pytest.raises(ValueError):
        green_matrix([[0, 0, 0]], [[0, 0, 0]], 11.7e9)
    with pytest.raises(ValueError):
        green_matrix([[1j, 0, 0]], [[0, 0, 0.06]], 11.7e9)


def _plane_wave(nx=32, ny=24, mode_x=3, mode_y=2, dx=0.00835, dy=0.00835):
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    kx = 2 * np.pi * mode_x / (nx * dx)
    ky = 2 * np.pi * mode_y / (ny * dy)
    wave = np.exp(1j * (kx * x[None, :] + ky * y[:, None]))
    return wave, kx, ky


@pytest.mark.parametrize("mode_x,mode_y", [(3, 2), (-4, 1), (0, 0)])
def test_angular_spectrum_reverses_known_upgoing_plane_wave(mode_x, mode_y):
    wave, kx, ky = _plane_wave(mode_x=mode_x, mode_y=mode_y)
    k = 2 * np.pi * 11.7e9 / C0_M_S
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    measured = wave * np.exp(1j * kz * 0.06)
    result = angular_spectrum_backprop(measured, 0.00835, 0.00835, 11.7e9,
                                       0.06, padding_factor=1, max_angle_deg=90)
    assert_allclose(result["image"], wave, rtol=4e-14, atol=4e-14)


def test_angular_spectrum_removes_evanescent_mode_without_exponential_gain():
    propagating, kx, ky = _plane_wave()
    evanescent, high_kx, _ = _plane_wave(mode_x=15, mode_y=0)
    k = 2 * np.pi * 11.7e9 / C0_M_S
    assert high_kx > k
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    # This field already contains the measured evanescent amplitude. Even a
    # huge backward distance must discard it, never multiply by exp(+|kz|z).
    measured = propagating * np.exp(1j * kz * 3.0) + 2 * evanescent
    result = angular_spectrum_backprop(measured, 0.00835, 0.00835, 11.7e9,
                                       3.0, padding_factor=1, max_angle_deg=90)
    assert np.all(np.isfinite(result["image"]))
    assert_allclose(result["image"], propagating, rtol=1e-12, atol=1e-12)
    assert_allclose(result["spectrum"][~result["propagating_mask"]], 0, atol=0)


def test_angular_spectrum_probe_regularization_has_known_complex_shrinkage():
    wave, kx, ky = _plane_wave()
    k = 2 * np.pi * 11.7e9 / C0_M_S
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    transfer, alpha = 0.4j, 0.04
    measured = transfer * wave * np.exp(1j * kz * 0.06)
    result = angular_spectrum_backprop(measured, 0.00835, 0.00835, 11.7e9,
                                       0.06, padding_factor=1, max_angle_deg=90,
                                       probe_transfer=transfer, regularization=alpha)
    assert_allclose(result["image"], 0.8 * wave, rtol=4e-14, atol=4e-14)


def test_angular_spectrum_zero_probe_transfer_and_odd_padding_crop():
    rng = np.random.default_rng(13)
    field = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    # All discrete spatial frequencies are propagating for these coarse steps.
    result = angular_spectrum_backprop(field, 0.03, 0.03, 11.7e9, 0,
                                       padding_factor=2, max_angle_deg=90)
    assert_allclose(result["image"], field, rtol=1e-14, atol=1e-14)
    blocked = angular_spectrum_backprop(field, 0.03, 0.03, 11.7e9, 0,
                                        padding_factor=2, max_angle_deg=90,
                                        probe_transfer=0j, regularization=0)
    assert_allclose(blocked["image"], 0, atol=0)
    assert np.all(np.isfinite(blocked["probe_compensation"]))


@pytest.mark.parametrize("alpha", [0.0, 0.07, 0.8])
def test_complex_weighted_regularized_prior_matches_independent_objective(alpha):
    a = _synthetic_matrix()
    y = np.array([1 + 2j, -3 + 0.7j, 2 - 4j, 0.2 + 0.3j, -0.4j, 3.2])
    weights = np.array([0.4, 1.3, 2.0, 0.7, 1.1, 2.2])
    prior = np.array([0.3 + 0.9j, -0.6 + 0.2j])
    expected, lam, scale = _objective_oracle(a, y, weights, prior, alpha)
    result = solve_tikhonov(a, y, [alpha], weights=weights, prior=prior)
    assert_allclose(result.coefficients, expected, rtol=3e-13, atol=3e-13)
    assert_allclose(result.lambda_absolute, lam, rtol=1e-14)
    assert_allclose(result.column_scale, scale, rtol=1e-14)


def test_weighted_single_column_hand_solution_and_nonzero_prior():
    a = np.array([[1 + 1j], [2 - 1j]])
    y = np.array([2 - 1j, -1 + 3j])
    # sum(w² |a|²)=13; sum(w² conj(a)y)=-1-7i.
    # alpha=1 => lambda=1, and D²=13: c=(-1-7i+13*(1+2i))/26.
    result = solve_tikhonov(a, y, [1.0], weights=[2, 1], prior=[1 + 2j])
    assert_allclose(result.coefficients, [(12 + 19j) / 26], rtol=2e-14)


def test_complex_column_rescaling_preserves_physical_solution_and_prior():
    a = _synthetic_matrix()
    y = np.arange(6) - 0.4j * np.arange(6)[::-1]
    prior = np.array([0.2 + 0.5j, -0.7 + 0.1j])
    weights = np.array([0.5, 0.8, 1.1, 1.6, 0.9, 1.3])
    factors = np.array([3.0j, -0.2 + 0.1j])
    original = solve_tikhonov(a, y, [0.17], weights=weights, prior=prior)
    transformed = solve_tikhonov(a * factors, y, [0.17], weights=weights,
                                 prior=prior / factors)
    assert_allclose(transformed.coefficients * factors, original.coefficients,
                    rtol=4e-13, atol=4e-13)
    assert_allclose(transformed.predicted, original.predicted, rtol=4e-13, atol=4e-13)


def test_holdout_alpha_scores_come_only_from_training_solutions():
    a = _synthetic_matrix()
    y = np.array([1 + 2j, -3 + 0.7j, 2 - 4j, 0.2 + 0.3j, -0.4j, 3.2])
    weights = np.array([0.4, 1.3, 2.0, 0.7, 1.1, 2.2])
    prior = np.array([0.3 + 0.9j, -0.6 + 0.2j])
    train, held = np.array([0, 2, 3, 5]), np.array([1, 4])
    alphas = np.array([0.0, 0.07, 0.8])
    candidates = [_objective_oracle(a[train], y[train], weights[train], prior, alpha)[0]
                  for alpha in alphas]
    scores = [np.mean(abs(weights[held] * (a[held] @ c - y[held]))**2) for c in candidates]
    expected_index = np.argmin(scores)
    result = solve_tikhonov(a, y, alphas, weights=weights, prior=prior,
                            validation_indices=held, refit=False)
    assert result.selection_method == "heldout_rows"
    assert result.alpha == alphas[expected_index]
    assert_allclose(result.score_curve, scores, rtol=4e-13)
    assert_allclose(result.coefficients, candidates[expected_index], rtol=4e-13, atol=4e-13)
    assert result.training_count == 4 and result.validation_count == 2
    # A much larger holdout row must not alter the training scale/SVD.
    altered = a.copy()
    altered[held] *= 1e6
    prepared = prepare_tikhonov(altered, weights=weights, validation_indices=held)
    assert_allclose(prepared.column_scale, np.linalg.norm(weights[train, None] * a[train], axis=0))


def test_holdout_refit_preserves_selection_error_and_exposes_refit():
    a = _synthetic_matrix()
    y = np.array([1, 2j, 3, -2, 5j, -6])
    args = dict(alphas=[0.01, 0.1, 1], validation_indices=[1, 4])
    held_fit = solve_tikhonov(a, y, refit=False, **args)
    refitted = solve_tikhonov(a, y, refit=True, **args)
    assert refitted.refit and not held_fit.refit
    assert refitted.alpha == held_fit.alpha
    assert refitted.validation_relative_error == held_fit.validation_relative_error
    expected, _, _ = _objective_oracle(a, y, np.ones(6), np.zeros(2), refitted.alpha)
    assert_allclose(refitted.coefficients, expected, rtol=3e-13, atol=3e-13)
    assert not np.allclose(refitted.coefficients, held_fit.coefficients)


def test_external_validation_selects_alpha_without_entering_the_training_objective():
    a = _synthetic_matrix()
    y = np.array([1, 2j, 3, -2, 5j, -6])
    av = np.array([[2 + 3j, -1], [-3j, 0.2 + 1j]])
    yv = np.array([2 - 1j, -0.3j])
    wv = np.array([0.2, 3.0])
    alphas = [0.02, 0.4, 1.7]
    candidates = [_objective_oracle(a, y, np.ones(6), np.zeros(2), alpha)[0]
                  for alpha in alphas]
    scores = [np.mean(abs(wv * (av @ c - yv))**2) for c in candidates]
    result = solve_tikhonov(a, y, alphas, validation_matrix=av, validation_y=yv,
                            validation_weights=wv)
    assert result.selection_method == "independent_validation"
    assert not result.refit
    assert_allclose(result.score_curve, scores, rtol=3e-13)
    assert_allclose(result.coefficients, candidates[np.argmin(scores)], rtol=3e-13, atol=3e-13)


def test_underdetermined_zero_regularization_retains_prior_nullspace():
    # A=[1,i] has only one nonzero singular value, but two unknowns.
    # The minimum correction to prior [1,2] is A^H*(2+2i)/2.
    a = np.array([[1, 1j]])
    result = solve_tikhonov(a, [3 + 4j], [0], prior=[1, 2],
                            validation_matrix=a, validation_y=[3 + 4j])
    assert_allclose(result.coefficients, [2 + 1j, 3 - 1j], rtol=2e-14)
    assert result.numerical_rank == 1
    assert result.numerical_rank < a.shape[1]
    with pytest.raises(ValueError, match="no finite score"):
        solve_tikhonov(a, [3 + 4j], [0])


def test_training_unobservable_column_and_overlapping_validation_inputs_rejected():
    with pytest.raises(ValueError, match="zero column"):
        prepare_tikhonov([[1, 0], [1, 2]], validation_indices=[1])
    with pytest.raises(ValueError, match="not both"):
        solve_tikhonov(_synthetic_matrix(), np.ones(6), [0.1], validation_indices=[0],
                       validation_matrix=[[1, 1]], validation_y=[1])


def test_gcv_score_matches_independent_hat_matrix_trace_and_weighted_residual():
    a = _synthetic_matrix()
    y = np.array([1 + 2j, -3 + 0.7j, 2 - 4j, 0.2 + 0.3j, -0.4j, 3.2])
    weights = np.array([0.4, 1.3, 2.0, 0.7, 1.1, 2.2])
    prior = np.array([0.3 + 0.9j, -0.6 + 0.2j])
    alphas = [0, 0.07, 0.8]
    scores = []
    for alpha in alphas:
        coefficients, lam, scale = _objective_oracle(a, y, weights, prior, alpha)
        weighted = weights[:, None] * a
        normal = weighted.conj().T @ weighted + lam**2 * np.diag(scale**2)
        hat = weighted @ np.linalg.solve(normal, weighted.conj().T)
        degrees = np.trace(hat).real
        residual_squared = np.linalg.norm(weights * (a @ coefficients - y))**2
        scores.append(len(y) * residual_squared / (len(y) - degrees)**2)
    result = solve_tikhonov(a, y, alphas, weights=weights, prior=prior)
    assert result.selection_method == "gcv"
    assert result.validation_relative_error is None
    assert_allclose(result.score_curve, scores, rtol=4e-13)


def test_common_gain_alignment_is_explicit_and_does_not_claim_absolute_calibration():
    truth = np.array([1, 0.7 + 0.5j, -1.2j, 0.3 - 0.4j])
    gain = 2 * np.exp(1j * np.pi / 3)
    raw = coefficient_metrics(gain * truth, truth, alignment="none")
    aligned = coefficient_metrics(gain * truth, truth, alignment="complex")
    assert_allclose(raw["amplitude_rmse_db"], 20 * np.log10(2), rtol=1e-14)
    assert_allclose(raw["phase_rmse_deg"], 60, rtol=1e-14)
    assert_allclose(raw["complex_nrmse"], np.sqrt(3), rtol=1e-14)
    assert raw["absolute_calibration_tested"]
    assert not aligned["absolute_calibration_tested"]
    assert_allclose(aligned["complex_nrmse"], 0, atol=1e-14)
    assert_allclose(aligned["alignment_factor"], [np.real(1 / gain), np.imag(1 / gain)], atol=1e-14)


def test_phase_wrap_low_amplitude_mask_and_failed_active_coefficients_remain_visible():
    truth = np.array([np.exp(1j * np.deg2rad(179)), 1, 0, 0.01j])
    estimate = np.array([np.exp(1j * np.deg2rad(-179)), 0, 0.3j, -0.01])
    metrics = coefficient_metrics(estimate, truth, amplitude_floor_relative=0.1)
    assert_allclose(metrics["phase_rmse_deg"], 2, atol=1e-12)
    assert metrics["phase_scored_count"] == 1
    assert metrics["active_truth_count"] == 2
    assert metrics["active_phase_unscored_count"] == 1
    assert metrics["inactive_truth_count"] == 2
    assert np.all(np.isnan(metrics["phase_error_deg"][[1, 2, 3]]))
    assert_allclose(metrics["inactive_estimate_max_relative"], 0.3)
    assert metrics["complex_nrmse"] > 0.7
    assert metrics["amplitude_rmse_db"] > 200
    missing = coefficient_metrics(np.zeros(2), [1, 1])
    assert missing["phase_rmse_deg"] is None
    assert missing["active_phase_unscored_count"] == 2
    assert missing["complex_nrmse"] == 1


def test_anomalies_compare_commanded_ramp_and_taper_and_flag_unexpected_activation():
    nominal = np.array([1, 0.5j, np.exp(1j * 0.8), 0])
    healthy = detect_anomalies(nominal, nominal)
    assert not np.any(healthy["fault_mask"])
    altered = nominal.copy()
    altered[0] = 0
    altered[1] *= np.exp(1j * np.deg2rad(60))
    altered[3] = 0.3
    result = detect_anomalies(altered, nominal)
    assert np.array_equal(result["fault_mask"], [True, True, False, True])
    assert np.array_equal(result["phase_fault_mask"], [False, True, False, False])
    assert np.array_equal(result["unexpected_activation_mask"], [False, False, False, True])
    rates = classification_metrics(result["fault_mask"], [True, True, False, True])
    assert rates["precision"] == rates["recall"] == 1
    assert rates["false_positive_rate"] == 0


def test_rectangular_spectrum_does_not_hide_underdetermined_nullspace():
    a = np.array([[1, 1j]])
    # np.linalg.cond(A)==1, although the data leave a coefficient direction unknown.
    assert_allclose(np.linalg.cond(a), 1)
    report = matrix_identifiability(a)
    assert report["numerical_rank"] == 1
    assert report["nullity"] == 1
    assert not report["full_column_rank"]
    assert np.isinf(report["condition_number"])
    assert not report["physical_identifiability_certified"]
    cached = matrix_identifiability(singular_values=[np.sqrt(2)], matrix_shape=(1, 2))
    assert cached["nullity"] == report["nullity"]
    assert cached["condition_number"] == report["condition_number"]


def test_noise_resolution_rank_is_not_numeric_rank_or_physical_certificate():
    report = matrix_identifiability(np.diag([1, 1e-3, 1e-12]), relative_noise_level=1e-2)
    assert report["numerical_rank"] == 3
    assert report["full_column_rank"]
    assert report["noise_resolved_mode_count"] == 1
    assert_allclose(report["condition_number"], 1e12)
    assert not report["physical_identifiability_certified"]


@pytest.mark.parametrize("frequency_hz", [10.7e9, 11.7e9, 12.7e9])
def test_reference_scan_sampling_only_claims_propagating_band_coverage(frequency_hz):
    report = sampling_report(frequency_hz, 0.00835, z_m=0.060)
    assert report["full_propagating_spectrum_nyquist"]
    assert not report["evanescent_spectrum_covered"]
    assert not report["element_identifiability_inferred"]


def test_sampling_relaxed_bandlimit_and_arbitrary_spectrum_roi_are_distinct():
    # lambda=30 mm, step=20 mm. The relaxed angle is asin(.75); an ROI safe
    # from aliases of arbitrary propagating spectra is only asin(.5)=30 deg.
    report = sampling_report(C0_M_S / 0.03, 0.02, z_m=0.06,
                             scan_extent_m=0.22, array_extent_m=0.10)
    assert not report["full_propagating_spectrum_nyquist"]
    assert_allclose(report["bandlimited_spectrum_angle_deg"], np.rad2deg(np.arcsin(0.75)))
    assert_allclose(report["arbitrary_propagating_spectrum_central_roi_deg"], 30)
    assert_allclose(report["truncation_geometry_angle_deg"], 45)
