"""Numerical definitions of the virtual paper-inspired aperture-cell pipeline."""
import numpy as np
import pytest

from array_reconstruct.physics import C0_M_S, angular_spectrum_backprop, green_matrix
from array_reconstruct.thesis_aperture import (FourierCellSampler, amplitude_circular_mean,
    cell_bounds, cell_owners, fit_aperture, prepare_aperture, reference_normalize, training_gain)
from array_reconstruct.validation import array_geometry, scan_geometry


def test_plane_wave_forward_backward_phase_closure():
    # Periodic exact Fourier mode avoids finite-window leakage in this sign test.
    n, step, f, z = 21, .00835, 11.7e9, .06
    x = np.arange(n) * step
    xx, yy = np.meshgrid(x, x)
    kx, ky = 2 * np.pi / (n * step), -4 * np.pi / (n * step)
    k = 2 * np.pi * f / C0_M_S
    kz = np.sqrt(k * k - kx * kx - ky * ky)
    source = np.exp(1j * (kx * xx + ky * yy))
    measured = source * np.exp(1j * kz * z)
    result = angular_spectrum_backprop(measured, step, step, f, z,
                                      padding_factor=1, max_angle_deg=90)
    np.testing.assert_allclose(result["image"], source, atol=3e-14)


def test_constant_complex_cell_integral_and_no_pixel_shortcut():
    n, step = 19, .00835
    constant = 2.3 * np.exp(2.2j)
    result = angular_spectrum_backprop(np.full((n, n), constant), step, step, 11.7e9, 0,
                                      padding_factor=1, max_angle_deg=90)
    centres = np.array([[.0013, -.0031], [.0151, .0073]])
    sampler = FourierCellSampler.create(centres, .011, result["kx"], result["ky"], 0, 0)
    np.testing.assert_allclose(sampler.evaluate(result["spectrum"]), constant, atol=1e-13)
    np.testing.assert_allclose(sampler.complex_mean(result["spectrum"]), constant, atol=1e-13)
    assert sampler.points_xy_m.shape == (2, 9, 2)
    assert sampler.weights.sum() == pytest.approx(1)


def test_fourier_resampling_physical_origin_and_gauss_plane_wave_integral():
    n, step = 189, .00835
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n, step))
    ky = kx.copy()
    spectrum = np.zeros((n, n), dtype=complex)
    iy, ix = n // 2 - 2, n // 2 + 3
    spectrum[iy, ix] = n * n
    x0, y0 = -94 * step, -94 * step
    centres = np.array([[.0134, -.0271], [.0244, .0169]])
    sampler = FourierCellSampler.create(centres, .011, kx, ky, x0, y0, quadrature_order=5)
    expected = np.exp(1j * (kx[ix] * (sampler.points_xy_m[..., 0] - x0) + ky[iy] * (sampler.points_xy_m[..., 1] - y0)))
    np.testing.assert_allclose(sampler.evaluate(spectrum), expected, atol=2e-15)
    exact_mean = np.exp(1j * (kx[ix] * (centres[:, 0] - x0) + ky[iy] * (centres[:, 1] - y0)))
    exact_mean *= np.sinc(kx[ix] * .011 / (2 * np.pi)) * np.sinc(ky[iy] * .011 / (2 * np.pi))
    np.testing.assert_allclose(sampler.complex_mean(spectrum), exact_mean, atol=1e-10)


@pytest.mark.parametrize("shape", [(9, 11), (10, 12), (9, 12), (10, 11)])
def test_fourier_resampling_recovers_non_square_complex_grid(shape):
    ny, nx = shape
    dx, dy, x0, y0 = .00835, .0061, -.047, .023
    rng = np.random.default_rng(481)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    spectrum = np.fft.fftshift(np.fft.fft2(field))
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, dy))
    row, col = np.indices(shape)
    centres = np.column_stack((x0 + col.ravel() * dx, y0 + row.ravel() * dy))
    sampler = FourierCellSampler.create(centres, .001, kx, ky, x0, y0, quadrature_order=5)
    np.testing.assert_allclose(sampler.evaluate(spectrum)[:, 12], field.ravel(), atol=2e-14)


def test_staggered_cells_have_half_open_shared_boundaries():
    centres = np.array([[0, 0], [.011, 0], [.0055, .011]])
    bounds = cell_bounds(centres, .011)
    points = np.array([[.0055, 0], [0, .0055], [.0055, .0055], [-.0055, -.0055], [.011, .0165]])
    np.testing.assert_array_equal(cell_owners(points, bounds), [1, 2, 2, 0, -1])
    for a in range(len(bounds)):
        for b in range(a):
            overlap_width = np.minimum(bounds[a, 1], bounds[b, 1]) - np.maximum(bounds[a, 0], bounds[b, 0])
            assert np.any(overlap_width <= 0)


def test_circular_phase_wrap_and_undefined_resultant():
    values = np.exp(1j * np.deg2rad([[179., -179.], [0., 180.]]))
    proxy, valid, coherence = amplitude_circular_mean(values, np.array([.5, .5]))
    assert abs(abs(np.angle(proxy[0])) - np.pi) < 1e-12
    assert valid.tolist() == [True, False]
    assert proxy[1] == 0
    assert coherence[0] > .99


def test_reference_small_values_masked_and_nominal_exact():
    ref = np.array([2 + 3j, 1e-12, 0])
    nominal = np.array([np.exp(.3j), 1, 1])
    coefficient, valid, threshold = reference_normalize(ref, ref, nominal)
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(coefficient, [nominal[0], 0, 0])
    assert threshold == pytest.approx(abs(ref[0]) * 1e-6)


def test_nominal_point_field_reference_normalization_is_exact():
    elements = array_geometry(3)
    samples = scan_geometry(19)
    point = green_matrix(elements, samples, 11.7e9)
    nominal = np.ones(len(elements), dtype=complex)
    field = point @ nominal
    prepared = prepare_aperture(elements, (19, 19), .00835, 11.7e9, .06, field)
    fitted, arrays = fit_aperture(prepared, field, point, nominal)
    for name in ("asm_complex_reference", "asm_amp_circular_reference"):
        np.testing.assert_allclose(fitted[name]["coefficients"], nominal, atol=2e-15)
    np.testing.assert_array_equal(arrays["reference_valid_complex"], True)
    np.testing.assert_array_equal(arrays["reference_valid_amp_circular"], True)


def test_training_gain_uses_observed_field_only():
    a = np.array([[1, 2j], [3j, 4], [5, 6]])
    proxy = np.array([1 + 2j, 3 - 4j])
    gain = -.4 + .7j
    assert training_gain(a, a @ proxy * gain, proxy) == pytest.approx(gain)
    with pytest.raises(ValueError, match="observable"):
        training_gain(a, np.ones(3), np.zeros(2))
