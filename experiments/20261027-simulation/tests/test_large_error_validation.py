"""Small independent checks of the explicitly synthetic large-error runner.

These tests never open measured inputs and never run the 54-case experiment.
Truth is constructed here for scoring; fitting receives observations only.
"""

from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from array_reconstruct import large_error_validation as large_error
from array_reconstruct.inverse import prepare_tikhonov
from array_reconstruct.simulation import (
    C0,
    build_rectangular_array,
    make_scan_grid,
    simulate_measurement,
)
from array_reconstruct.sparse import prepare_sparse_faults


@pytest.fixture
def policy():
    path = Path(__file__).resolve().parents[1] / "configs" / "large_error_validation_policy.json"
    return json.loads(path.read_text(encoding="utf8"))


def _scenario(distribution="uniform", *, half_range=2.5, amp_std=None, phase_std=5.0):
    return {
        "name": "unit_test_only",
        "amplitude_distribution": distribution,
        "amplitude_std_db": amp_std,
        "amplitude_half_range_db": half_range,
        "phase_std_deg": phase_std,
    }


def test_registered_matrix_is_30_main_and_24_endpoint_paired_cases(policy):
    cases = large_error.case_matrix(policy)
    assert len(cases) == 54
    assert len({case["name"] for case in cases}) == 54
    assert Counter(case["group"] for case in cases) == {"main": 30, "endpoint": 24}

    pairs = defaultdict(list)
    for case in cases:
        pairs[case["pair_id"]].append(case)
        assert case["seed"] in policy["seeds"]
        assert case["scenario_name"] == case["scenario"]["name"]
    assert len(pairs) == 27
    for members in pairs.values():
        assert len(members) == 2
        assert {member["with_faults"] for member in members} == {False, True}
        for key in ("seed", "frequency_hz", "scenario_name", "group"):
            assert members[0][key] == members[1][key]

    main = [case for case in cases if case["group"] == "main"]
    endpoints = [case for case in cases if case["group"] == "endpoint"]
    assert {case["frequency_hz"] for case in main} == {policy["main_frequency_hz"]}
    assert {case["scenario_name"] for case in main} == {
        scenario["name"] for scenario in policy["scenarios"]
    }
    assert {case["frequency_hz"] for case in endpoints} == set(policy["endpoint_frequencies_hz"])
    assert {case["scenario_name"] for case in endpoints} == set(policy["endpoint_scenarios"])


@pytest.mark.parametrize(
    "scenario",
    [
        _scenario(half_range=2.5, phase_std=5.0),
        _scenario(half_range=5.0, phase_std=10.0),
        _scenario("gaussian", half_range=None, amp_std=5.0, phase_std=3.0),
    ],
)
def test_distribution_scales_are_exact_draw_parameters_not_interchanged(scenario):
    # Full span 5 dB means U[-2.5,+2.5]; Gaussian std 5 dB means N(0,5^2).
    # Comparing exact seeded draws catches scale confusion without statistical
    # tolerance or an unreliable small-sample standard deviation assertion.
    count, seed = 128, 20261102
    result = large_error.sample_excitation(count, scenario, seed, with_faults=False)
    amplitude_seed, phase_seed, _ = np.random.SeedSequence(seed).spawn(3)
    amplitude_rng = np.random.default_rng(amplitude_seed)
    if scenario["amplitude_distribution"] == "uniform":
        half_range = scenario["amplitude_half_range_db"]
        expected_amplitude = amplitude_rng.uniform(-half_range, half_range, count)
        assert np.all(np.abs(result["amp_error_db"]) <= half_range)
    else:
        expected_amplitude = amplitude_rng.normal(0.0, scenario["amplitude_std_db"], count)
        assert np.any(np.abs(expected_amplitude) > scenario["amplitude_std_db"])
    expected_phase = np.random.default_rng(phase_seed).normal(0.0, scenario["phase_std_deg"], count)
    assert_array_equal(result["amp_error_db"], expected_amplitude)
    assert_array_equal(result["phase_error_deg"], expected_phase)
    expected_healthy = (
        result["nominal"] * 10 ** (expected_amplitude / 20)
        * np.exp(1j * np.deg2rad(expected_phase))
    )
    assert_allclose(result["healthy"], expected_healthy, atol=0, rtol=2e-15)
    assert_array_equal(result["truth"], result["healthy"])
    assert len(result["fault_indices"]) == 0


def test_registered_span5_and_phase5_have_distinct_units(policy):
    scenario = next(s for s in policy["scenarios"] if s["name"] == "span5_phase5")
    assert scenario["amplitude_distribution"] == "uniform"
    assert scenario["amplitude_half_range_db"] == 2.5
    assert scenario["amplitude_std_db"] is None
    assert scenario["phase_std_deg"] == 5.0


def test_healthy_and_faulted_share_background_and_seeded_physical_faults(policy):
    count, seed = 64, policy["seeds"][0]
    scenario = next(s for s in policy["scenarios"] if s["name"] == "span5_phase5")
    healthy = large_error.sample_excitation(count, scenario, seed, with_faults=False)
    faulted = large_error.sample_excitation(count, scenario, seed, with_faults=True)
    for key in ("nominal", "healthy", "amp_error_db", "phase_error_deg"):
        assert_array_equal(healthy[key], faulted[key])
    indices = np.asarray(faulted["fault_indices"], dtype=int)
    assert len(indices) == len(np.unique(indices)) == policy["faults"]["count"] == 5
    assert np.all((indices >= 0) & (indices < count))
    outside = np.setdiff1d(np.arange(count), indices)
    assert_array_equal(faulted["truth"][outside], healthy["truth"][outside])
    assert np.count_nonzero(faulted["truth"] == 0) == 1
    assert np.count_nonzero(faulted["truth"] != healthy["truth"]) == 5
    expected_multipliers = np.array([
        0,
        10 ** (-6 / 20),
        np.exp(1j * np.deg2rad(60)),
        10 ** (-6 / 20) * np.exp(1j * np.deg2rad(40)),
        10 ** (-6 / 20) * np.exp(1j * np.deg2rad(40)),
    ])
    assert_allclose(
        faulted["truth"][indices] / healthy["truth"][indices],
        expected_multipliers, atol=2e-15, rtol=2e-15,
    )

    # Scenario changes may rescale normal errors, but must not move the faults.
    for other_scenario in policy["scenarios"]:
        other = large_error.sample_excitation(count, other_scenario, seed, with_faults=True)
        assert_array_equal(other["fault_indices"], indices)


@pytest.mark.parametrize("source_fraction,order", [(0.0, 3), (0.1, 3), (0.2, 5)])
def test_cached_truth_and_noise_match_independent_simulate_measurement(source_fraction, order):
    frequency, seed = 11.7e9, 819
    elements = build_rectangular_array((4, 4), 0.011, row_offset_m=0.0055)
    samples = make_scan_grid((9, 9), step_m=0.00835, z_m=0.06, center_xy_m=(0.001, -0.002))
    rng = np.random.default_rng(740)
    coefficients = (0.3 + rng.random(16)) * np.exp(1j * rng.uniform(-np.pi, np.pi, 16))
    matrix = large_error.cached_truth_matrix(
        elements, samples, frequency,
        source_fraction=source_fraction, quadrature_order=order, chunk_size=7,
    )
    assert matrix.shape == (81, 16)
    actual = large_error.cached_measurement(matrix, coefficients, seed, snr_db=40.0)
    expected = simulate_measurement(
        elements, coefficients, frequency, samples,
        {"source_aperture_m": source_fraction * C0 / frequency,
         "source_quadrature_order": order, "directionality_power": 0.0,
         "snr_db": 40.0, "seed": seed, "chunk_size": 11, "element_chunk_size": 5},
    )
    for key in ("field_clean", "noise", "field"):
        assert_allclose(actual[key], expected[key], rtol=5e-13, atol=3e-12)
    assert_allclose(actual["signal_power"], expected["metadata"]["signal_mean_power"], rtol=5e-13)
    assert_allclose(actual["expected_noise_power"], expected["metadata"]["expected_noise_mean_power"], rtol=5e-13)
    assert_allclose(actual["realized_snr_db"], expected["metadata"]["realized_snr_db"], atol=2e-11)


def test_paired_measurements_share_unit_gaussian_but_scale_to_each_clean_power(policy):
    scenario = next(s for s in policy["scenarios"] if s["name"] == "span5_phase5")
    healthy = large_error.sample_excitation(16, scenario, 530, with_faults=False)
    faulted = large_error.sample_excitation(16, scenario, 530, with_faults=True)
    elements = build_rectangular_array((4, 4), 0.011, row_offset_m=0.0055)
    samples = make_scan_grid((9, 9), step_m=0.00835, z_m=0.06)
    matrix = large_error.cached_truth_matrix(elements, samples, 11.7e9)
    seed = 100530
    measurements = [large_error.cached_measurement(matrix, state["truth"], seed, snr_db=40)
                    for state in (healthy, faulted)]
    noise_seed = np.random.SeedSequence(seed).spawn(3)[2]
    rng = np.random.default_rng(noise_seed)
    expected_standard = rng.normal(size=len(samples)) + 1j * rng.normal(size=len(samples))
    normalized = []
    for measurement in measurements:
        expected_power = float(np.mean(np.abs(measurement["field_clean"]) ** 2)) / 10000
        assert_allclose(measurement["expected_noise_power"], expected_power, rtol=2e-15)
        normalized.append(measurement["noise"] / np.sqrt(expected_power / 2))
        assert_allclose(normalized[-1], expected_standard, atol=5e-16, rtol=5e-16)
    assert_allclose(normalized[0], normalized[1], atol=5e-16, rtol=5e-16)
    assert measurements[0]["signal_power"] != measurements[1]["signal_power"]


def test_registered_nine_noise_streams_are_distinct_and_pair_across_scenarios(policy):
    # Isolate RNG allocation with a small fixed operator; no physical runner or
    # full experimental case is needed to detect replicate/plane seed reuse.
    rng = np.random.default_rng(741)
    matrix = rng.normal(size=(31, 16)) + 1j * rng.normal(size=(31, 16))
    allocated_seeds = []
    unit_noise_by_stream = {}
    for replicate_seed in policy["seeds"]:
        plane_seeds = large_error.measurement_seeds(replicate_seed)
        assert plane_seeds == {
            "training": replicate_seed * 10 + 100000,
            "validation": replicate_seed * 10 + 100001,
            "test": replicate_seed * 10 + 100002,
        }
        allocated_seeds.extend(plane_seeds.values())
        for plane, measurement_seed in plane_seeds.items():
            noise_seed = np.random.SeedSequence(measurement_seed).spawn(3)[2]
            noise_rng = np.random.default_rng(noise_seed)
            expected_unit_noise = (
                noise_rng.normal(size=len(matrix)) + 1j * noise_rng.normal(size=len(matrix))
            )
            unit_noise_by_stream[(replicate_seed, plane)] = expected_unit_noise
            for scenario in policy["scenarios"]:
                for with_faults in (False, True):
                    excitation = large_error.sample_excitation(
                        matrix.shape[1], scenario, replicate_seed, with_faults,
                    )
                    measurement = large_error.cached_measurement(
                        matrix, excitation["truth"], measurement_seed,
                        snr_db=policy["measurement_snr_db"],
                    )
                    expected_clean = matrix @ excitation["truth"]
                    assert_allclose(measurement["field_clean"], expected_clean, atol=0, rtol=2e-15)
                    assert measurement["expected_noise_power"] > 0
                    unit_noise = measurement["noise"] / np.sqrt(
                        measurement["expected_noise_power"] / 2
                    )
                    assert_allclose(unit_noise, expected_unit_noise, atol=1e-15, rtol=1e-15)
                    observed_unit_noise = (measurement["field"] - expected_clean) / np.sqrt(
                        measurement["expected_noise_power"] / 2
                    )
                    assert_allclose(observed_unit_noise, expected_unit_noise, atol=5e-13, rtol=5e-13)
    assert len(allocated_seeds) == len(set(allocated_seeds)) == 9
    assert len(unit_noise_by_stream) == 9
    streams = list(unit_noise_by_stream.values())
    for index, stream in enumerate(streams):
        for other in streams[index + 1:]:
            # This catches accidental reuse; it does not purport to establish
            # statistical independence from a finite random sample.
            assert not np.allclose(stream, other, atol=1e-15, rtol=1e-15)


def test_common_gain_is_complex_least_squares_with_conjugation():
    matrix = np.eye(2, dtype=complex)
    nominal = np.array([1 + 1j, 2 - 1j])
    observations = np.array([2 + 3j, -1 + 4j])
    # conj(h) @ y = (5+i)+(-6+7i), while conj(h) @ h = 2+5.
    expected = (-1 + 8j) / 7
    gain = large_error.estimate_common_gain(matrix, observations, nominal)
    assert_allclose(gain, expected, atol=2e-15, rtol=2e-15)
    residual = observations - gain * (matrix @ nominal)
    assert_allclose(np.vdot(matrix @ nominal, residual), 0, atol=5e-15)


def test_healthy_precision_is_null_while_legacy_scoring_is_preserved(policy):
    nominal = np.ones(16, dtype=complex)
    record, aligned, flagged, oracle_flags = large_error.score_estimate(
        (1.7 - 0.4j) * nominal, nominal, nominal, np.array([], dtype=int), policy,
    )
    assert_allclose(aligned, nominal, atol=3e-15)
    assert not np.any(flagged)
    assert not np.any(oracle_flags)
    assert record["metrics"]["fault_precision"] is None
    assert record["metrics"]["fault_recall"] is None
    assert record["injected_fault_detection"]["precision"] is None
    assert record["injected_fault_detection"]["recall"] is None
    assert record["legacy_metrics"]["fault_precision"] == 1.0
    assert record["legacy_metrics"]["fault_recall"] == 1.0
    assert record["legacy_joint_acceptance_passed"] is True
    assert record["coefficient_acceptance_passed"] is True


def test_nominal_threshold_oracle_is_distinct_from_physical_injection_labels(policy):
    nominal = np.ones(16, dtype=complex)
    truth = nominal.copy()
    truth[0] = 0  # The sole physical injected fault.
    truth[1] *= 10 ** (6 / 20)  # A large normal error can cross the detector threshold.
    injected = np.array([0])
    record, aligned, flagged, oracle_flags = large_error.score_estimate(
        truth.copy(), truth, nominal, injected, policy,
    )
    assert_allclose(aligned, truth, atol=3e-15)
    assert_array_equal(np.flatnonzero(flagged), [0, 1])
    assert_array_equal(flagged, oracle_flags)
    injected_metrics = record["injected_fault_detection"]
    assert injected_metrics["true_positives"] == 1
    assert injected_metrics["false_positives"] == 1
    assert injected_metrics["precision"] == 0.5
    assert injected_metrics["recall"] == 1.0
    assert record["oracle_nominal_threshold_detection"]["precision"] == 1.0
    assert record["oracle_nominal_threshold_detection"]["recall"] == 1.0
    assert record["truth_through_detector"]["flagged_indices"] == [0, 1]
    assert record["coefficient_acceptance_passed"] is True
    assert record["legacy_joint_acceptance_passed"] is False


def test_healthy_null_precision_is_retained_when_normal_errors_cross_thresholds(policy):
    nominal = np.ones(16, dtype=complex)
    truth = nominal.copy()
    truth[0] *= 10 ** (6 / 20)
    truth[1] *= np.exp(1j * np.deg2rad(40))
    record, _, flagged, oracle_flags = large_error.score_estimate(
        truth, truth, nominal, np.array([], dtype=int), policy,
    )
    assert_array_equal(np.flatnonzero(flagged), [0, 1])
    assert_array_equal(flagged, oracle_flags)
    assert record["injected_fault_detection"]["false_positives"] == 2
    assert record["injected_fault_detection"]["precision"] is None
    assert record["injected_fault_detection"]["recall"] is None
    assert record["metrics"]["fault_precision"] is None
    assert record["metrics"]["fault_recall"] is None
    assert record["legacy_metrics"]["fault_precision"] == 0.0
    assert record["truth_through_detector"]["normal_noninjected_threshold_crossings"] == 2
    assert record["oracle_nominal_threshold_detection"]["precision"] == 1.0
    assert record["coefficient_acceptance_passed"] is True
    assert record["legacy_joint_acceptance_passed"] is False


def test_detection_gauge_does_not_use_truth_alignment(policy):
    nominal = np.ones(16, dtype=complex)
    first_truth = nominal.copy()
    first_truth[0] = 0
    second_truth = first_truth.copy()
    second_truth[1] *= 10 ** (6 / 20)
    estimate = (0.8 + 0.6j) * second_truth
    first = large_error.score_estimate(estimate, first_truth, nominal, np.array([0]), policy)
    second = large_error.score_estimate(estimate, second_truth, nominal, np.array([0]), policy)
    assert_array_equal(first[2], second[2])
    assert_array_equal(np.flatnonzero(first[3]), [0])
    assert_array_equal(np.flatnonzero(second[3]), [0, 1])


def test_fitting_has_no_truth_or_fault_label_input_and_never_calls_scorer(monkeypatch):
    parameters = inspect.signature(large_error.fit_methods).parameters
    assert not any("truth" in name or "fault_indices" in name for name in parameters)
    matrix = np.eye(8, dtype=complex)
    nominal = np.ones(8, dtype=complex)
    field = nominal.copy()
    field[0] = 2
    prepared = prepare_tikhonov(matrix)
    sparse_prepared = prepare_sparse_faults(matrix, nominal)

    def scoring_is_forbidden(*args, **kwargs):
        raise AssertionError("Truth-dependent scoring must occur after all fits are fixed")

    monkeypatch.setattr(large_error, "score_estimate", scoring_is_forbidden)
    fits = large_error.fit_methods(
        prepared, sparse_prepared, matrix, field, field,
        alphas=np.array([0.0, 0.1, 1.0]), max_faults=1,
    )
    assert set(fits) == {
        "nominal_common_gain", "ridge_zero", "sparse_nominal_gain", "ridge_nominal_centered",
    }
    for fit in fits.values():
        assert np.asarray(fit["coefficients"]).shape == nominal.shape
        assert np.all(np.isfinite(fit["coefficients"]))


def test_independent_validation_observations_can_change_selected_sparse_fit():
    matrix = np.eye(8, dtype=complex)
    nominal = np.ones(8, dtype=complex)
    field = nominal.copy()
    field[0] = 2
    prepared = prepare_tikhonov(matrix)
    sparse_prepared = prepare_sparse_faults(matrix, nominal)
    settings = {"alphas": np.array([0.0, 0.1, 1.0]), "max_faults": 1}
    healthy_validation = large_error.fit_methods(
        prepared, sparse_prepared, matrix, field, nominal, **settings,
    )
    faulty_validation = large_error.fit_methods(
        prepared, sparse_prepared, matrix, field, field, **settings,
    )
    # With no correction, the training LS common gain is 9/8. A single-element
    # correction instead fits [2,1,...,1] exactly. Only external observations
    # change here; the training system, nominal prior and candidate path agree.
    assert_allclose(
        healthy_validation["sparse_nominal_gain"]["coefficients"],
        np.full(8, 9 / 8), atol=2e-13,
    )
    assert_allclose(faulty_validation["sparse_nominal_gain"]["coefficients"], field, atol=2e-13)
    assert_allclose(
        healthy_validation["nominal_common_gain"]["coefficients"],
        faulty_validation["nominal_common_gain"]["coefficients"], atol=0,
    )
