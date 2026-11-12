"""Small pipeline boundary checks, using exclusively fabricated observations."""

import csv
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct import cli
from array_reconstruct.validation import (
    C0,
    array_geometry,
    evaluate_coefficients,
    excitation,
    scan_geometry,
)


def _observation_payload(*, zero_field=False, underdetermined=False, coefficients=None):
    elements = array_geometry(2)
    samples = scan_geometry(7)
    if underdetermined:
        samples = samples[:2]
    frequency = 11.7e9
    if coefficients is None:
        coefficients = np.array([1 + 0.3j, 0.2 - 0.4j, -0.3 + 0.8j, 0.7])

    def analytic_field(points):
        distance = np.sqrt(np.sum((points[:, None] - elements[None]) ** 2, axis=2))
        return (np.exp(2j * np.pi * frequency / C0 * distance) / distance) @ coefficients

    validation = scan_geometry(3, z=0.067)
    validation[:, 0] += 0.003
    return dict(
        provenance="synthetic", elements_m=elements, samples_m=samples,
        frequency_hz=frequency,
        field=np.zeros(len(samples), complex) if zero_field else analytic_field(samples),
        validation_samples_m=validation,
        validation_field=np.zeros(len(validation), complex) if zero_field else analytic_field(validation),
    )


@pytest.mark.parametrize("provenance", [None, "actual_measurement", "Synthetic"])
def test_reconstruct_refuses_non_synthetic_archive_before_observation_access(tmp_path, provenance):
    archive = tmp_path / "unapproved.npz"
    payload = {} if provenance is None else {"provenance": provenance}
    # Intentionally omit all observation fields: provenance must be checked first.
    np.savez(archive, **payload)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="synthetic measurements only"):
        cli.main(["reconstruct", "--input", str(archive), "--output", str(output)])
    assert not output.exists()


def test_reconstruct_does_not_access_stored_truth_or_use_it_for_selection(tmp_path):
    payload = _observation_payload()
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    np.savez(first, **payload, true_coefficients=np.array([100j, -30, 7, 999]))
    # Object payload is deliberately unreadable with allow_pickle=False. A
    # numeric-only output comparison would not prove that truth was never read.
    np.savez(second, **payload, true_coefficients=np.array([{"forbidden": "truth"}], dtype=object))
    outputs = [tmp_path / "first_result", tmp_path / "second_result"]
    for archive, output in zip((first, second), outputs):
        assert cli.main(["reconstruct", "--input", str(archive), "--output", str(output)]) == 0
    assert (outputs[0] / "coefficients.csv").read_bytes() == (outputs[1] / "coefficients.csv").read_bytes()
    reports = [json.loads((output / "reconstruction.json").read_text()) for output in outputs]
    assert reports[0] == reports[1]
    assert reports[0]["truth_used"] is False
    assert reports[0]["selection_method"] == "independent_validation"


def test_positive_steering_phase_points_far_field_to_positive_angle():
    positions = array_geometry(8)
    nominal, _, _ = excitation(positions, 11.7e9, 1, faults=False, beam_deg=20)
    # Outgoing exp(+ikR) has far-field factor exp(-ik s.r). This independent
    # direct sum therefore reaches N at +20 degrees, not at -20 degrees.
    k = 2 * np.pi * 11.7e9 / C0
    forward = np.exp(-1j * k * positions[:, 0] * np.sin(np.deg2rad(20)))
    backward = np.exp(+1j * k * positions[:, 0] * np.sin(np.deg2rad(20)))
    assert_allclose(abs(np.dot(nominal, forward)), len(nominal), rtol=1e-14)
    assert abs(np.dot(nominal, backward)) < 0.2 * len(nominal)


def test_pipeline_excludes_low_amplitude_phase_without_hiding_complex_error():
    truth = np.array([1, 1, 1, 0, 0.01j])
    estimated = np.array([1, 1, 1, 0.01j, -0.01])
    metrics, _, _ = evaluate_coefficients(estimated, truth, np.ones(5), np.array([3, 4]))
    assert metrics["active_phase_count"] == 3
    assert metrics["phase_unscored_count"] == 2
    assert metrics["complex_nrmse"] > 0
    # Tiny alignment phase from the low-amplitude entries may affect all active
    # elements, but the arbitrary 90 degree low-amplitude phase is not averaged.
    assert metrics["active_phase_rmse_deg"] < 0.01


def test_cli_zero_coefficients_have_no_exported_phase(tmp_path):
    archive = tmp_path / "zero.npz"
    np.savez(archive, **_observation_payload(zero_field=True))
    output = tmp_path / "zero_result"
    cli.main(["reconstruct", "--input", str(archive), "--output", str(output)])
    with (output / "coefficients.csv").open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(row["amplitude"]) == 0 for row in rows)
    assert all(row["phase_deg"] == "" for row in rows)


def test_cli_sparse_nonzero_weak_coefficients_have_no_exported_phase(tmp_path):
    archive = tmp_path / "weak.npz"
    np.savez(archive, **_observation_payload(coefficients=np.array([1, 0.01j, 0, 0])))
    output = tmp_path / "weak_result"
    cli.main(["reconstruct", "--input", str(archive), "--output", str(output)])
    with (output / "coefficients.csv").open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    assert_allclose([float(row["amplitude"]) for row in rows], [1, 0.01, 0, 0], atol=1e-10)
    assert rows[0]["phase_deg"] != ""
    assert all(row["phase_deg"] == "" for row in rows[1:])


def test_sparse_truth_does_not_turn_dead_coefficients_into_phase_targets():
    truth = np.array([1, 0, 0, 0, 0])
    metrics, _, _ = evaluate_coefficients(truth, truth, np.ones(5), np.array([1, 2, 3, 4]))
    assert metrics["active_phase_count"] == 1
    assert metrics["phase_unscored_count"] == 4
    assert np.isfinite(metrics["active_amplitude_rmse_db"])
    assert metrics["active_amplitude_rmse_db"] == 0


def test_sparse_truth_still_excludes_nonzero_weak_element_phase():
    truth = np.array([1, 0.01j, 0, 0, 0])
    estimated = np.array([1, -0.01, 0, 0, 0])
    metrics, _, _ = evaluate_coefficients(estimated, truth, np.ones(5), np.array([1, 2, 3, 4]))
    assert metrics["active_phase_count"] == 1
    assert metrics["phase_unscored_count"] == 4
    assert metrics["active_phase_rmse_deg"] < 0.01


def test_underdetermined_cli_report_exposes_nullspace(tmp_path):
    archive = tmp_path / "underdetermined.npz"
    np.savez(archive, **_observation_payload(underdetermined=True))
    output = tmp_path / "underdetermined_result"
    cli.main(["reconstruct", "--input", str(archive), "--output", str(output)])
    report = json.loads((output / "reconstruction.json").read_text())
    assert report["numerical_rank"] == 2
    assert report["elements"] == 4
    assert report["data_nullity"] == 2
    assert report["rank_deficient"]
    assert report["column_scaled_condition_number"] is None


@pytest.mark.parametrize("side", [2, 3, 4])
def test_cli_supported_small_array_faults_stay_in_bounds(side):
    positions = array_geometry(side)
    nominal, truth, fault_indices = excitation(positions, 11.7e9, 1, faults=True)
    assert nominal.shape == truth.shape == (side * side,)
    assert np.all((fault_indices >= 0) & (fault_indices < side * side))
    assert len(np.unique(fault_indices)) == len(fault_indices)
