"""Independent checks of synthetic truth; no measured data are read."""

import unittest

import numpy as np

from array_reconstruct.simulation import (
    C0, DEFAULT_SPACING_M, build_rectangular_array, generate_virtual_case,
    inject_faults, make_feed_weights, make_scan_grid, nearest_neighbor_coupling,
    simulate_measurement, standard_faults,
)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.frequency = 11.7e9
        self.point = {"source_aperture_m": 0.0, "directionality_power": 0.0}

    def test_point_source_analytic_outgoing_kernel(self):
        source = np.array([[0.0, 0.0, 0.0]])
        scans = np.array([[0.0, 0.0, 0.06], [0.01, -0.02, 0.07]])
        weight = np.array([0.7 + 0.4j])
        result = simulate_measurement(source, weight, self.frequency, scans, self.point)
        radii = np.linalg.norm(scans, axis=1)
        analytic = weight[0] * np.exp(2j * np.pi * self.frequency / C0 * radii) / radii
        np.testing.assert_allclose(result["field"], analytic, rtol=2e-14)
        np.testing.assert_array_equal(result["noise"], np.zeros(2))

    def test_staggered_coordinates_and_neighbors(self):
        positions = build_rectangular_array((4, 5), 0.011, row_offset_m=0.0055)
        coords = positions.reshape(4, 5, 3)
        np.testing.assert_allclose(np.mean(positions, axis=0), 0, atol=1e-17)
        np.testing.assert_allclose(np.diff(coords[:, :, 0], axis=1), 0.011)
        np.testing.assert_allclose(np.diff(coords[:, 0, 1]), 0.011)
        np.testing.assert_allclose(coords[1, :, 0] - coords[0, :, 0], 0.0055)
        coupling = nearest_neighbor_coupling(positions, 0.03j)
        np.testing.assert_allclose(coupling, coupling.T)
        self.assertEqual(np.count_nonzero(coupling[7]) - 1, 6)
        self.assertEqual(coupling[4, 5], 0)  # adjacent flattened IDs are not neighbors
        rectangular = nearest_neighbor_coupling(build_rectangular_array((4, 5)), 0.03)
        self.assertEqual(np.count_nonzero(rectangular[7]) - 1, 4)

    def test_steered_beam_sign_at_nonzero_angle(self):
        positions = build_rectangular_array((6, 6))
        theta, phi = np.deg2rad([25.0, 30.0])
        direction = np.array([np.sin(theta) * np.cos(phi),
                              np.sin(theta) * np.sin(phi), np.cos(theta)])
        mirrored = direction * [-1, -1, 1]
        weights = make_feed_weights(positions, self.frequency, beam_deg=(25, 30))
        fields = simulate_measurement(positions, weights, self.frequency,
                                      np.array([direction, mirrored]) * 100, self.point)["field"]
        self.assertGreater(abs(fields[0]), 10 * abs(fields[1]))
        self.assertAlmostEqual(abs(fields[0]), len(positions) / 100, places=6)

    def test_faults_are_feed_amplitude_not_power(self):
        healthy = np.ones(64, dtype=complex)
        faults = standard_faults((8, 8))
        faulted = inject_faults(healthy, faults, array_shape=(8, 8))
        self.assertAlmostEqual(abs(faulted[9]), 0.01)
        self.assertAlmostEqual(abs(faulted[14]), 10 ** (-6 / 20))
        self.assertAlmostEqual(np.angle(faulted[49]), np.pi / 3)
        for index in (45, 46, 53, 54):
            self.assertAlmostEqual(abs(faulted[index]), 10 ** (-6 / 20))
            self.assertAlmostEqual(np.angle(faulted[index]), np.pi / 3)
        self.assertEqual(np.count_nonzero(abs(faulted - 1) > 1e-12), 7)
        dead = inject_faults(healthy, [{"kind": "dead", "index": 0, "zero": True}])
        self.assertEqual(dead[0], 0)
        np.testing.assert_array_equal(healthy, np.ones(64))

    def test_coupling_maps_feed_to_radiation_before_gain(self):
        positions = build_rectangular_array((1, 2))
        feeds = np.array([0.0, 1.0 + 0.5j])
        coupling = np.array([[1.0, 0.2j], [0.1, 1.0]], complex)
        gain = 1.3 * np.exp(0.4j)
        scans = make_scan_grid((3, 4))
        result = simulate_measurement(positions, feeds, self.frequency, scans,
                                      {**self.point, "coupling_matrix": coupling,
                                       "complex_gain": gain, "coupling_known": False})
        np.testing.assert_allclose(result["equivalent_coefficients"], coupling @ feeds)
        self.assertGreater(abs(result["equivalent_coefficients"][0]), 0)
        self.assertIsNone(result["known_coupling_matrix"])
        expected = simulate_measurement(positions, coupling @ feeds, self.frequency,
                                        scans, self.point)["field"] * gain
        np.testing.assert_allclose(result["field"], expected, rtol=1e-14)

    def test_finite_aperture_far_field_sinc_and_quadrature_convergence(self):
        wavelength = C0 / self.frequency
        source = np.zeros((1, 3))
        direction = np.array([[0.4, 0.3, np.sqrt(0.75)]])
        scans = 1000 * direction
        widths = wavelength * np.array([0.9, 0.7])
        point = simulate_measurement(source, np.ones(1), self.frequency, scans, self.point)["field"]
        finite = simulate_measurement(source, np.ones(1), self.frequency, scans,
                                      {**self.point, "source_aperture_m": widths,
                                       "source_quadrature_order": 9})["field"]
        expected_ratio = np.prod(np.sinc(widths * direction[0, :2] / wavelength))
        np.testing.assert_allclose(finite / point, expected_ratio, rtol=2e-5, atol=2e-5)
        near_scans = make_scan_grid((6, 7), step_m=0.012, z_m=0.025)
        settings = {"source_aperture_m": widths, "probe_aperture_m": 0.3 * wavelength,
                    "source_quadrature_order": 7, "probe_quadrature_order": 7}
        a = simulate_measurement(source, np.ones(1), self.frequency, near_scans, settings)
        b = simulate_measurement(source, np.ones(1), self.frequency, near_scans,
                                 {**settings, "source_quadrature_order": 11,
                                  "probe_quadrature_order": 11})
        self.assertLess(np.linalg.norm(a["field"] - b["field"]) / np.linalg.norm(b["field"]), 2e-8)

    def test_translation_directionality_and_blocking(self):
        positions = build_rectangular_array((3, 4))
        scans = make_scan_grid((5, 6))
        weights = make_feed_weights(positions, normal_amp_std_db=0.2,
                                    normal_phase_std_deg=3, seed=4)
        settings = {"source_aperture_m": 0.003, "probe_aperture_m": 0.002,
                    "directionality_power": 1.2, "probe_directionality_power": 0.4}
        first = simulate_measurement(positions, weights, self.frequency, scans, settings)
        shift = np.array([0.12, -0.03, 0.4])
        second = simulate_measurement(positions + shift, weights, self.frequency, scans + shift,
                                      {**settings, "chunk_size": 7, "element_chunk_size": 5})
        np.testing.assert_allclose(first["field"], second["field"], rtol=1e-13, atol=2e-12)
        source = np.zeros((1, 3))
        two = np.array([[0, 0, 0.06], [0.036, 0, 0.048]], float)
        directional = simulate_measurement(source, np.ones(1), self.frequency, two,
                                           {**self.point, "directionality_power": 2})["field"]
        self.assertAlmostEqual(abs(directional[1] / directional[0]), 0.8 ** 2)

    def test_noise_statistics_and_reproducible_separate_geometry(self):
        source = np.zeros((1, 3))
        scans = np.tile([0.0, 0.0, 0.06], (20000, 1))
        config = {**self.point, "snr_db": 30, "seed": 18, "scan_jitter_std_m": (1e-4, 2e-4, 0),
                  "distance_bias_m": 0.001, "complex_gain": 0.5j}
        result = simulate_measurement(source, np.ones(1), self.frequency, scans, config)
        again = simulate_measurement(source, np.ones(1), self.frequency, scans, config)
        np.testing.assert_array_equal(result["field"], again["field"])
        np.testing.assert_array_equal(result["nominal_scan_positions"], scans)
        np.testing.assert_allclose(result["actual_scan_positions"][:, 2], 0.061)
        noise = result["noise"]
        expected = result["metadata"]["expected_noise_mean_power"]
        self.assertAlmostEqual(np.mean(abs(noise) ** 2) / expected, 1, delta=0.025)
        self.assertAlmostEqual(np.var(noise.real) / expected, 0.5, delta=0.02)
        self.assertAlmostEqual(np.var(noise.imag) / expected, 0.5, delta=0.02)
        self.assertLess(abs(np.mean(noise)) / np.sqrt(expected), 0.02)
        self.assertLess(abs(np.corrcoef(noise.real, noise.imag)[0, 1]), 0.02)

    def test_case_fixed_spacing_across_frequency_and_truth_labels(self):
        first = generate_virtual_case((3, 3), 10.7e9, scan_shape=(4, 4), config=self.point)
        second = generate_virtual_case((3, 3), 12.7e9, scan_shape=(4, 4), config=self.point)
        np.testing.assert_array_equal(first["nominal_element_positions"], second["nominal_element_positions"])
        self.assertAlmostEqual(np.diff(first["nominal_element_positions"][:2, 0])[0], DEFAULT_SPACING_M)
        self.assertEqual(first["metadata"]["data_origin"], "virtual_only")
        self.assertEqual(np.count_nonzero(first["fault_mask"]), 0)

    def test_invalid_config_and_singular_geometry_fail(self):
        source = np.zeros((1, 3))
        scans = np.array([[0, 0, 0.06]], float)
        with self.assertRaises(ValueError):
            simulate_measurement(source, np.ones(1), self.frequency, scans, {"typo": 1})
        with self.assertRaises(ValueError):
            simulate_measurement(source, np.ones(1), self.frequency, source, self.point)
        with self.assertRaises(ValueError):
            simulate_measurement(source, np.zeros(1), self.frequency, scans,
                                 {**self.point, "snr_db": 30})


if __name__ == "__main__":
    unittest.main()
