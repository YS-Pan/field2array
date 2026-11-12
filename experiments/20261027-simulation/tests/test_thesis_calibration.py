"""Small scientific checks for the isolated virtual calibration experiment.

These tests use tiny in-memory arrays.  They do not read measured inputs, open
the thesis PDF, or execute the registered 32x32 evaluation matrix.
"""
from __future__ import annotations

import inspect

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from array_reconstruct import thesis_calibration as calibration
from array_reconstruct.inverse import prepare_tikhonov


def _phasor(degrees):
    return np.exp(1j * np.deg2rad(degrees))


def test_attenuation_quantization_respects_code_grid_and_physical_limits():
    config = calibration.CalibrationConfig()
    values = np.array([-1.0, 0.0, 0.09, 0.11, 25.39, 25.4, 99.0])
    original = values.copy()
    codes = calibration.quantize_attenuation(values, config)

    assert_allclose(codes, [0.0, 0.0, 0.0, 0.2, 25.4, 25.4, 25.4])
    assert_allclose(codes / config.attenuation_step_db,
                    np.rint(codes / config.attenuation_step_db))
    assert_array_equal(calibration.quantize_attenuation(codes, config), codes)
    assert_array_equal(values, original)


def test_phase_codes_are_periodic_quantized_and_idempotent():
    config = calibration.CalibrationConfig()
    values = np.array([-360.0, -0.01, 0.0, 2.7, 2.9, 359.0, 360.0,
                       720.0 + config.phase_step_deg])
    original = values.copy()
    codes = calibration.quantize_phase(values, config)

    # Compare physical phase, so a signed versus unsigned representation is
    # immaterial.  In particular 359 degrees rounds across the 360/0 seam.
    assert_allclose(_phasor(codes), _phasor([0, 0, 0, 0, 5.625, 0, 0, 5.625]),
                    atol=2e-14)
    assert_allclose(_phasor(calibration.quantize_phase(codes, config)),
                    _phasor(codes), atol=2e-14)
    assert_array_equal(values, original)


def test_hardware_uses_cumulative_absolute_codes_and_cannot_revive_a_dead_element():
    initial = np.array([0j, 2 * _phasor(30), 0.5 * _phasor(-20)])
    attenuation = np.array([0.0, 6.0, 2.0])
    phase = np.array([15.0, 45.0, 350.0])
    initial_copy, attenuation_copy, phase_copy = (
        initial.copy(), attenuation.copy(), phase.copy())

    actual = calibration.apply_hardware(initial, attenuation, phase, 1.0)
    expected = initial * 10 ** (-attenuation / 20) * _phasor(phase + attenuation)

    assert_allclose(actual, expected, atol=2e-15)
    assert actual[0] == 0j
    assert_allclose(np.abs(actual), np.abs(initial) * 10 ** (-attenuation / 20))
    assert np.sum(np.abs(actual) ** 2) < np.sum(np.abs(initial) ** 2)
    assert_array_equal(initial, initial_copy)
    assert_array_equal(attenuation, attenuation_copy)
    assert_array_equal(phase, phase_copy)


def test_amplitude_control_only_attenuates_above_the_linear_mean_and_accumulates():
    config = calibration.CalibrationConfig()
    estimate = np.array([0.5, 1.0, 2.0], dtype=complex)
    attenuation = np.array([1.0, 2.0, 3.0])
    phase = np.array([5.625, 11.25, 16.875])
    originals = [value.copy() for value in (estimate, attenuation, phase)]

    first = calibration.control_update(estimate, attenuation, phase, "a", config)
    expected_delta = np.maximum(20 * np.log10(np.abs(estimate) / np.mean(np.abs(estimate))), 0)
    expected_codes = calibration.quantize_attenuation(attenuation + expected_delta, config)

    assert_allclose(first["requested_attenuation_delta_db"], expected_delta, atol=2e-14)
    assert_allclose(first["attenuation_db"], expected_codes)
    assert_allclose(first["attenuation_db"][:2], attenuation[:2])
    assert first["attenuation_db"][2] > attenuation[2]
    assert_array_equal(first["phase_command_deg"], phase)
    assert not np.any(first["excluded_mask"])

    second = calibration.control_update(
        estimate, first["attenuation_db"], first["phase_command_deg"], "a", config)
    assert second["attenuation_db"][2] > first["attenuation_db"][2]
    assert_allclose(second["attenuation_db"], calibration.quantize_attenuation(
        first["attenuation_db"] + expected_delta, config))
    for value, original in zip((estimate, attenuation, phase), originals):
        assert_array_equal(value, original)


def test_exclusion_is_derived_from_the_estimate_and_does_not_change_excluded_codes():
    config = calibration.CalibrationConfig(phase_step_deg=1.0)
    estimate = np.array([0j, 1e-6 * _phasor(90), _phasor(20), 2 * _phasor(-20)])
    attenuation = np.array([1.0, 2.0, 3.0, 4.0])
    phase = np.array([30.0, 40.0, 50.0, 60.0])
    update = calibration.control_update(estimate, attenuation, phase, "ap", config)

    assert_array_equal(update["excluded_mask"], [True, True, False, False])
    assert_array_equal(update["attenuation_db"][:2], attenuation[:2])
    assert_array_equal(update["phase_command_deg"][:2], phase[:2])
    assert_allclose(update["requested_attenuation_delta_db"][:2], 0)
    # Only the eligible magnitudes 1 and 2 enter the mean.  Including either
    # excluded low channel would demand unnecessary extra attenuation.
    assert_allclose(update["requested_attenuation_delta_db"][2:],
                    [0.0, 20 * np.log10(2 / 1.5)], atol=2e-14)

    # A changed estimate changes the mask without any separate truth/label
    # input.  This is a control eligibility decision, not a fault diagnosis.
    revised = estimate.copy()
    revised[:2] = 1.0
    changed = calibration.control_update(revised, attenuation, phase, "ap", config)
    assert not np.any(changed["excluded_mask"])
    names = inspect.signature(calibration.control_update).parameters
    assert not any("truth" in name or "fault" in name for name in names)


def test_phase_control_uses_circular_mean_across_the_branch_cut():
    config = calibration.CalibrationConfig(phase_step_deg=1.0)
    estimate = _phasor(np.array([179.0, -179.0]))
    attenuation = np.array([1.0, 2.0])
    phase = np.array([10.0, 20.0])
    update = calibration.control_update(estimate, attenuation, phase, "p", config)

    # The circular reference is 180 degrees: the required increments are
    # +1/-1 degrees.  An arithmetic mean of np.angle gives zero and fails.
    assert_allclose(_phasor(update["phase_command_deg"]), _phasor([11.0, 19.0]),
                    atol=2e-14)
    assert_array_equal(update["attenuation_db"], attenuation)
    assert_allclose(update["requested_attenuation_delta_db"], 0)


def test_undefined_circular_mean_preserves_phase_but_still_allows_amplitude_control():
    config = calibration.CalibrationConfig(phase_step_deg=1.0)
    # Unequal magnitudes still give opposite unit phasors.  Their circular
    # resultant is zero, so there is no preferred phase reference.
    estimate = np.array([1.0 + 0j, -2.0 + 0j])
    attenuation = np.array([1.0, 2.0])
    phase = np.array([10.0, 20.0])
    update = calibration.control_update(estimate, attenuation, phase, "ap", config)

    assert not update["phase_mean_defined"]
    assert update["circular_phase_target_deg"] is None
    assert_array_equal(update["phase_command_deg"], phase)
    assert_array_equal(update["requested_phase_delta_deg"], np.zeros(2))
    assert update["attenuation_db"][1] > attenuation[1]
    assert_allclose(update["requested_attenuation_delta_db"],
                    [0, 20 * np.log10(2 / 1.5)], atol=2e-14)


def test_amplitude_update_saturates_without_exceeding_hardware_range():
    config = calibration.CalibrationConfig()
    estimate = np.array([1.0, 10.0], dtype=complex)
    attenuation = np.array([0.0, config.attenuation_max_db])
    phase = np.zeros(2)
    update = calibration.control_update(estimate, attenuation, phase, "a", config)

    assert update["requested_attenuation_delta_db"][1] > 0
    assert_allclose(update["attenuation_db"][1], config.attenuation_max_db, atol=1e-13)
    assert update["saturated_mask"][1]
    assert np.all(update["attenuation_db"] <= config.attenuation_max_db + 1e-13)


def test_all_zero_estimate_is_excluded_and_produces_a_finite_noop():
    config = calibration.CalibrationConfig()
    attenuation = np.array([1.0, 2.0, 3.0])
    phase = np.array([0.0, 5.625, 11.25])
    update = calibration.control_update(
        np.zeros(3, dtype=complex), attenuation, phase, "ap", config)

    assert np.all(update["excluded_mask"])
    assert_array_equal(update["attenuation_db"], attenuation)
    assert_array_equal(update["phase_command_deg"], phase)
    assert_array_equal(update["requested_attenuation_delta_db"], np.zeros(3))
    assert np.all(np.isfinite(update["attenuation_db"]))
    assert np.all(np.isfinite(update["phase_command_deg"]))


def test_fixed_noise_does_not_shrink_when_control_attenuates_the_field():
    matrix = np.array([[1, 0], [0, 1j], [1, 1], [1j, -1]], dtype=complex)
    coefficients = np.array([1 + 0.3j, -0.5 + 0.2j])
    sigma = 0.02
    bright = calibration.fixed_noise_measurement(matrix, coefficients, sigma, 17)
    dim = calibration.fixed_noise_measurement(matrix, 0.1 * coefficients, sigma, 17)

    assert_array_equal(bright["noise"], dim["noise"])
    assert bright["noise_sigma"] == dim["noise_sigma"] == sigma
    assert_allclose(dim["field_clean"], 0.1 * bright["field_clean"], atol=2e-16)
    assert_allclose(bright["field"], bright["field_clean"] + bright["noise"])
    assert_allclose(dim["field"], dim["field_clean"] + dim["noise"])
    assert_allclose(bright["realized_snr_db"] - dim["realized_snr_db"], 20.0, atol=2e-13)
    different = calibration.fixed_noise_measurement(matrix, coefficients, sigma, 18)
    assert not np.array_equal(bright["noise"], different["noise"])


def test_zero_field_keeps_receiver_noise_without_undefined_array_values():
    result = calibration.fixed_noise_measurement(np.eye(3), np.zeros(3), 0.1, 19)
    assert_array_equal(result["field_clean"], np.zeros(3))
    assert_array_equal(result["field"], result["noise"])
    assert np.all(np.isfinite(result["field"]))
    assert np.any(np.abs(result["noise"]) > 0)


def test_fitting_interface_has_no_truth_test_or_fault_label_channel():
    signature = inspect.signature(calibration.fit_current_state)
    assert set(signature.parameters) == {
        "prepared", "y", "validation_matrix", "validation_y", "alphas"}
    assert not any(parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
                   for parameter in signature.parameters.values())


def test_fit_prior_common_gain_uses_training_data_only(monkeypatch):
    matrix = np.array([[1, 0], [0, 2j], [1 + 1j, 0.5]], dtype=complex)
    field = np.array([1.2 + 0.8j, -0.2 + 2.1j, 1.3 + 1.8j])
    prepared = prepare_tikhonov(matrix)
    validation_matrix = np.eye(2, dtype=complex)
    nominal_field = matrix @ np.ones(2, dtype=complex)
    expected_gain = np.vdot(nominal_field, field) / np.vdot(nominal_field, nominal_field)
    original_solver = calibration.solve_tikhonov
    captured_priors = []

    def capture_prior(*args, **kwargs):
        captured_priors.append(np.asarray(kwargs["prior"]).copy())
        return original_solver(*args, **kwargs)

    def forbid_scoring(*args, **kwargs):
        raise AssertionError("Fitting must finish before truth-dependent scoring")

    monkeypatch.setattr(calibration, "solve_tikhonov", capture_prior)
    monkeypatch.setattr(calibration, "score_estimate", forbid_scoring, raising=False)
    results = []
    for validation_y in (np.ones(2), np.array([15j, -12.0])):
        count_before = len(captured_priors)
        results.append(calibration.fit_current_state(
            prepared, field, validation_matrix, validation_y, np.array([0.0, 0.1, 1.0])))
        assert len(captured_priors) > count_before

    for prior in captured_priors:
        assert_allclose(prior, expected_gain * np.ones(2), atol=2e-14)
    for estimate, metadata in results:
        assert_allclose(metadata["common_gain"], [expected_gain.real, expected_gain.imag], atol=2e-14)
        assert estimate.shape == (2,)
        assert np.all(np.isfinite(estimate))
    for prior in captured_priors[1:]:
        assert_array_equal(prior, captured_priors[0])
