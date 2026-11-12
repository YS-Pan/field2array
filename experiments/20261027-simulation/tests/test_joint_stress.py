"""Numerical generator checks, independent of geometry recovery assertions."""
from dataclasses import replace

import numpy as np
import pytest

from array_reconstruct.joint_stress import (
    C_M_S, apply_scenario, independent_scalar_field, independent_scalar_matrix,
    local_calibration_transfer, scenario_manifest, state_error_factors,
    stress_scenarios, wrong_calibration_template,
)
from array_reconstruct.joint_virtual import VirtualFitData, VirtualSession, VirtualTruth


@pytest.mark.parametrize("q", [0., 1., 2., 2.4])
@pytest.mark.parametrize("distance", [60., 72.])
def test_zero_apertures_equal_hand_point_formula_with_offsets(q, distance):
    elements = np.array([[.002, -.004], [-.006, .009]])
    samples = np.array([[.017, -.011], [-.013, .021], [.002, .003]])
    geometry = np.array([distance, 1.2345, -3.4567])
    actual = independent_scalar_matrix(elements, samples, 12.7e9, geometry, q=q)
    expected = np.empty_like(actual)
    for row, sample in enumerate(samples):
        for col, element in enumerate(elements):
            xyz = np.r_[sample + geometry[1:] * .001 - element, distance * .001]
            radius = np.linalg.norm(xyz)
            expected[row, col] = (distance * .001) ** q / radius ** (q + 1) * np.exp(
                -2j * np.pi * 12.7e9 / C_M_S * radius)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-13)


def test_independent_generator_does_not_call_fitting_kernel(monkeypatch):
    import array_reconstruct.joint_geometry as fit

    def forbidden(*args, **kwargs):
        raise AssertionError("independent generator called fitting kernel")

    monkeypatch.setattr(fit, "scalar_kernel", forbidden)
    matrix = independent_scalar_matrix([[0., 0.]], [[.01, -.02]], 12.7e9,
        [62.3456, 1.2345, -3.4567], source_aperture_mm=4., probe_aperture_mm=4.)
    assert matrix.shape == (1, 1) and np.isfinite(matrix).all()


@pytest.mark.parametrize("q", [0., 1., 2.4])
def test_far_field_normalized_square_apertures_have_sinc_product(q):
    # Known direction and 1000 m range make curvature negligible without
    # assuming any actual element or probe directivity.
    direction = np.array([.6, .48, .64])
    radius = 1000.
    samples = np.array([radius * direction[:2]])
    geometry = [radius * direction[2] * 1000., 0., 0.]
    wavelength = .03
    point = independent_scalar_matrix([[0., 0.]], samples, C_M_S / wavelength,
                                       geometry, q=q)[0, 0]
    finite = independent_scalar_matrix([[0., 0.]], samples, C_M_S / wavelength,
        geometry, q=q, source_aperture_mm=6., probe_aperture_mm=4.)[0, 0]
    expected = np.prod([np.sinc(side * direction[axis] / wavelength)
                        for side in (.006, .004) for axis in (0, 1)])
    assert abs(finite / point - expected) < 5e-6
    assert expected < .98  # The test checks a measurable effect, not only unity.


def test_unequal_aperture_exchange_and_spatial_reversal_reciprocity():
    xy = np.array([[.018, -.026], [-.043, .021], [.001, .002]])
    first = independent_scalar_matrix([[0., 0.]], xy, 12.7e9, [51., 0., 0.],
        source_aperture_mm=(6., 5.), probe_aperture_mm=(4., 3.))
    second = independent_scalar_matrix([[0., 0.]], -xy, 12.7e9, [51., 0., 0.],
        source_aperture_mm=(4., 3.), probe_aperture_mm=(6., 5.))
    np.testing.assert_allclose(first, second, rtol=2e-14, atol=1e-13)


def test_equal_square_matches_independent_triangular_displacement_integral():
    # Convolution of two uniform equal squares has separable triangular
    # displacement density. Split at its cusp and integrate each half.
    side, frequency, d = .006, 12.7e9, .05
    x, y = .034, -.029
    nodes, weights = np.polynomial.legendre.leggauss(12)
    axis = []
    for lower, upper in ((-side, 0.), (0., side)):
        places = lower + (nodes + 1) * (upper - lower) / 2
        mass = weights * (upper - lower) / 2 * (side - np.abs(places)) / side**2
        axis.extend(zip(places, mass))
    expected = 0j
    for dx, wx in axis:
        for dy, wy in axis:
            rr = np.sqrt(d*d + (x+dx)**2 + (y+dy)**2)
            expected += wx * wy * (d / rr)**2 / rr * np.exp(-2j*np.pi*frequency/C_M_S*rr)
    actual = independent_scalar_matrix([[0., 0.]], [[x, y]], frequency,
        [d * 1000, 0., 0.], source_aperture_mm=6., probe_aperture_mm=6.,
        quadrature_order=7)[0, 0]
    assert abs(actual - expected) / abs(expected) < 2e-12


@pytest.mark.parametrize("q", [2., 2.4])
def test_gauss_convergence_over_near_field_stress_extent(q):
    samples = np.array([[0., 0.], [.04, -.03], [.2, .2], [-.39, .36]])
    elements = np.array([[0., 0.], [.006, -.004]])
    values = {order: independent_scalar_matrix(elements, samples, 12.7e9,
        [50., 1.2345, -3.4567], q=q, source_aperture_mm=6.,
        probe_aperture_mm=4., quadrature_order=order) for order in (3, 5, 7, 9)}
    point = independent_scalar_matrix(elements, samples, 12.7e9,
                                       [50., 1.2345, -3.4567], q=q)
    normalize = np.abs(point)
    assert np.max(np.abs(values[7] - values[9]) / normalize) < 1e-10
    assert np.max(np.abs(values[3] - values[9]) / normalize) < 1e-5
    assert np.max(np.abs(values[5] - values[9]) / normalize) < 1e-9


def test_matrix_field_chunk_sign_and_joint_translation_agree():
    rng = np.random.default_rng(43101)
    elements = rng.normal(0, .01, (4, 2))
    samples = rng.normal(0, .02, (11, 2))
    coefficient = rng.normal(size=4) + 1j * rng.normal(size=4)
    keywords = dict(source_aperture_mm=4., probe_aperture_mm=4., q=2.4)
    first = independent_scalar_matrix(elements, samples, 12.7e9,
                                       [62.3456, 1.2345, -3.4567], chunk_size=2, **keywords)
    field = independent_scalar_field(elements, samples, 12.7e9,
        [62.3456, 1.2345, -3.4567], coefficient, chunk_size=3, **keywords)
    np.testing.assert_allclose(field, first @ coefficient, rtol=1e-14, atol=1e-13)
    reverse = independent_scalar_matrix(elements, samples, 12.7e9,
        [62.3456, 1.2345, -3.4567], sign=1, **keywords)
    np.testing.assert_array_equal(reverse, first.conj())
    translated = independent_scalar_matrix(elements + [.017, -.013], samples + [.017, -.013],
        12.7e9, [62.3456, 1.2345, -3.4567], **keywords)
    np.testing.assert_allclose(translated, first, rtol=3e-14, atol=1e-13)


def test_state_common_factors_removed_and_strengths_exactly_distinguished():
    factors = state_error_factors(64, np.random.default_rng(43101), phase_std_deg=10., gain_std_db=.3)
    phase, db = np.rad2deg(np.angle(factors)), 20*np.log10(abs(factors))
    assert abs(phase.mean()) < 1e-13 and abs(db.mean()) < 1e-13
    assert abs(phase.std() - 10) < 1e-13 and abs(db.std() - .3) < 1e-13
    np.testing.assert_array_equal(factors, state_error_factors(64, np.random.default_rng(43101),
                                                             phase_std_deg=10., gain_std_db=.3))


def test_height_example_normalizes_nominal_transfer_and_keeps_height_phase():
    unity = local_calibration_transfer(32, 12.7e9, np.random.default_rng(43101),
                                       height_mm=10., height_jitter_mm=0.)
    np.testing.assert_array_equal(unity, np.ones(32))
    rng = np.random.default_rng(43102)
    heights = 10 + rng.uniform(-.5, .5, 32)
    expected = 10/heights * np.exp(-2j*np.pi*12.7e9/C_M_S*(heights-10)*.001)
    actual = local_calibration_transfer(32, 12.7e9, np.random.default_rng(43102))
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_wrong_templates_do_not_create_phase_for_amplitude_information():
    original = np.linspace(.7, 1.2, 16)
    amplitude = wrong_calibration_template(original, np.random.default_rng(43101), mode="historical_version")
    assert not np.iscomplexobj(amplitude) and np.all(amplitude > 0)
    permutation = wrong_calibration_template(original, np.random.default_rng(43101))
    np.testing.assert_array_equal(np.sort(permutation), original)
    assert not np.array_equal(permutation, original)
    complex_version = wrong_calibration_template(original.astype(complex), np.random.default_rng(43101),
                                                 mode="historical_version")
    assert np.std(np.angle(complex_version)) > .1


def _small_protocol_case():
    elements = np.array([[0., 0.], [.011, 0.], [0., .011], [.011, .011]])
    xy = np.array([[-.013, -.009], [.002, .004], [.019, -.012], [.024, .018]])
    weights = np.array([.8*np.exp(-.03j), 1.1*np.exp(.05j), .9*np.exp(.1j), 1.2*np.exp(-.08j)])
    geometry = np.array([62.3456, 1.2345, -3.4567])
    sessions, clean, gains, states = [], {}, {}, {}
    for side, angle in ((134, 0.), (101, 30.)):
        name = f"session{side}"
        steering = np.exp(1j * elements[:, 0] * np.sin(np.deg2rad(angle)) * 2*np.pi*12.7e9/C_M_S)
        gain = 1. if angle == 0 else np.exp(.02 + .2j)
        generated = gain * independent_scalar_field(elements, xy, 12.7e9, geometry, steering * weights)
        noise = np.array([.001+.002j, -.002+.001j, .003-.004j, .001-.001j])
        sessions.append(VirtualSession(name, 12.7e9, angle, xy.copy(), steering,
            generated + noise, np.array([0, 2]), np.array([0, 1]), np.array([2, 3]), side, "train"))
        clean[name], gains[name], states[name] = generated, gain, np.ones(4, complex)
    calibration = {"f12.7": abs(weights) / np.exp(np.log(abs(weights)).mean())}
    data = VirtualFitData(elements, tuple(sessions), calibration, {"case_index": 0, "seed": 43101})
    truth = VirtualTruth(geometry, {"f12.7": weights}, clean, gains, states)
    return data, truth


def test_protocol_point_control_and_confirmation_seal():
    data, truth = _small_protocol_case()
    generated, hidden = apply_scenario(data, truth, "point_q2_control", 43101)
    for old, new in zip(data.sessions, generated.sessions):
        np.testing.assert_array_equal(old.observed_field, new.observed_field)
        np.testing.assert_array_equal(hidden.clean_fields[old.name], truth.clean_fields[old.name])
    with pytest.raises(ValueError, match="sealed"):
        apply_scenario(data, truth, "point_q2_control", 88201)


def test_protocol_session_phase_positive_and_noise_pairing():
    data, truth = _small_protocol_case()
    result, hidden = apply_scenario(data, truth, "session_phase45_control", 43101)
    np.testing.assert_array_equal(result.sessions[0].observed_field, data.sessions[0].observed_field)
    name = data.sessions[1].name
    np.testing.assert_allclose(hidden.clean_fields[name], truth.clean_fields[name] * np.exp(1j*np.pi/4))
    np.testing.assert_allclose(hidden.session_scalars[name], truth.session_scalars[name] * np.exp(1j*np.pi/4))
    old_noise = data.sessions[1].observed_field - truth.clean_fields[name]
    new_noise = result.sessions[1].observed_field - hidden.clean_fields[name]
    np.testing.assert_allclose(new_noise, old_noise, rtol=1e-10, atol=1e-14)


def test_window_injection_only_changes_101_generation_and_never_fit_coordinates():
    data, truth = _small_protocol_case()
    result, hidden = apply_scenario(data, truth, "window101_x1p5", 43101)
    for old, new in zip(data.sessions, result.sessions):
        np.testing.assert_array_equal(old.samples_xy_m, new.samples_xy_m)
    np.testing.assert_array_equal(hidden.clean_fields["session134"], truth.clean_fields["session134"])
    shifted = data.sessions[1]
    expected = truth.session_scalars[shifted.name] * independent_scalar_field(data.elements_m,
        shifted.samples_xy_m + [.0015, 0.], shifted.frequency_hz, truth.geometry_mm,
        shifted.steering * truth.weights["f12.7"])
    np.testing.assert_array_equal(hidden.clean_fields[shifted.name], expected)
    assert np.linalg.norm(expected-truth.clean_fields[shifted.name]) > .1


def test_calibration_only_stress_does_not_change_any_field_and_is_relative():
    data, truth = _small_protocol_case()
    original_amp = data.calibration_relative_amplitude["f12.7"].copy()
    result, hidden = apply_scenario(data, truth, "calibration_gain1", 43101)
    assert abs(np.log(result.calibration_relative_amplitude["f12.7"]).mean()) < 1e-14
    assert not np.array_equal(result.calibration_relative_amplitude["f12.7"], original_amp)
    np.testing.assert_array_equal(data.calibration_relative_amplitude["f12.7"], original_amp)
    for old, new in zip(data.sessions, result.sessions):
        np.testing.assert_array_equal(old.observed_field, new.observed_field)
        np.testing.assert_array_equal(truth.clean_fields[old.name], hidden.clean_fields[old.name])


def test_state_error_injection_is_distinct_from_static_response():
    data, truth = _small_protocol_case()
    result, hidden = apply_scenario(data, truth, "state_phase3_gain0p3", 43101)
    np.testing.assert_array_equal(hidden.weights["f12.7"], truth.weights["f12.7"])
    np.testing.assert_array_equal(hidden.state_factors["session134"], np.ones(4))
    assert not np.array_equal(hidden.state_factors["session101"], np.ones(4))
    assert not np.array_equal(result.sessions[1].observed_field, data.sessions[1].observed_field)


def test_outside_domain_generation_does_not_clip_or_change_original_truth():
    data, truth = _small_protocol_case()
    result, hidden = apply_scenario(data, truth, "distance72_outside", 43101)
    assert hidden.geometry_mm[0] == 72. and truth.geometry_mm[0] == 62.3456
    assert result.metadata["stress"]["acceptance_must_not_use_scenario_or_truth"] is True


def test_scenario_inventory_labels_hypotheses_and_contains_independent_control():
    manifest = scenario_manifest()
    scenarios = stress_scenarios()
    assert manifest["hardware_facts"] is False
    assert len({scenario.id for scenario in scenarios}) == len(scenarios)
    assert any(s.positive_control for s in scenarios)
    assert any(s.source_aperture_mm > 0 and s.probe_aperture_mm > 0 for s in scenarios)


@pytest.mark.parametrize("kwargs", [
    {"geometry_mm": [0., 0., 0.]}, {"q": -1}, {"sign": 0},
    {"source_aperture_mm": -1}, {"probe_aperture_mm": [1, -1]},
    {"quadrature_order": 0}, {"chunk_size": 1.5},
])
def test_invalid_generator_domain_rejected(kwargs):
    arguments = dict(elements_m=[[0., 0.]], samples_xy_m=[[0., 0.]],
                     frequency_hz=12.7e9, geometry_mm=[60., 0., 0.])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        independent_scalar_matrix(**arguments)
