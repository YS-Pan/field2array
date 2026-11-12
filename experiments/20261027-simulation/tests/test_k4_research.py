"""Small execution-contract checks; these never run the scientific matrix."""
from copy import deepcopy
from hashlib import sha256
import inspect
from itertools import count
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from array_reconstruct import k4_research as research


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SEEDS = {20261029, 20261030, 20261031, 20261101, 20261102,
                    20261111, 20261112, 20261113}


def _tiny_policy():
    policy = json.loads((ROOT / "configs/k4_development_20261112.json").read_text(encoding="utf8"))
    policy["geometry"].update(array_side=3, scan_side=5)
    policy.update(seeds=[81312001, 81312002], max_cases=8,
                  max_total_seconds=60., max_case_seconds=30.)
    policy["state_error_regimes"] = [
        {"name": "mild", "amplitude_std_db": .2, "phase_std_deg": 3., "domain": "declared"},
        {"name": "double", "amplitude_std_db": .4, "phase_std_deg": 6., "domain": "stress"},
    ]
    return policy


def _one_case(policy):
    policy = deepcopy(policy)
    policy["seeds"] = policy["seeds"][:1]
    policy["fault_states"] = [False]
    policy["state_error_regimes"] = policy["state_error_regimes"][:1]
    policy["max_cases"] = 1
    return policy


def _write_policy(directory, policy):
    path = directory / "policy.json"
    path.write_text(json.dumps(policy, indent=2), encoding="utf8")
    return path


def _quiet(*args, **kwargs):
    pass


def _run(directory, policy, operators=(), **kwargs):
    directory.mkdir(parents=True, exist_ok=True)
    return research.run_k4_research(directory / "output", _write_policy(directory, policy),
                                    operators, progress=_quiet, **kwargs)


def _load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


@pytest.fixture(scope="module")
def generated_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("k4-small-contract")
    policy = _tiny_policy()
    report = _run(directory, policy, generate_operators=True)
    output = directory / "output"
    operators = _load(Path(report["operators"][0]["path"]))
    archives = {case["name"]: _load(output / case["archive"]) for case in report["cases"]}
    return policy, output, report, operators, archives


@pytest.mark.parametrize("filename,expected_count", [
    ("k4_development_20261112.json", 16),
    ("k4_confirmation_20261112.json", 20),
])
def test_shipped_policies_freeze_explicitly_disjoint_roots(filename, expected_count):
    policy = json.loads((ROOT / "configs" / filename).read_text(encoding="utf8"))
    assert len(research.build_case_matrix(policy)) == expected_count
    assert policy["frozen"] is True
    assert HISTORICAL_SEEDS <= set(policy["excluded_seeds"])
    assert not set(policy["seeds"]) & set(policy["excluded_seeds"])
    other_name = ("k4_confirmation_20261112.json" if policy["stage"] == "development"
                  else "k4_development_20261112.json")
    other = json.loads((ROOT / "configs" / other_name).read_text(encoding="utf8"))
    assert set(other["seeds"]) <= set(policy["excluded_seeds"])


def test_tiny_generated_run_persists_complete_denominator_and_receipts(generated_run):
    policy, output, report, operators, archives = generated_run
    assert report["registered_case_count"] == report["completed_case_count"] == 8
    assert report["failed_case_count"] == report["timed_out_case_count"] == 0
    assert report["in_progress"] is False
    assert report["stage"] == "development"
    assert report["synthetic_only"] is True
    assert report["actual_measurements_read"] is False
    assert len(report["summary"]) == 4
    assert {group["domain"] for group in report["summary"]} == {"declared", "stress"}
    assert all(group["denominator"] == 2 and group["coverage"] == 1. for group in report["summary"])
    assert sum(group["denominator"] for group in report["summary"]) == 8
    assert len(archives) == 8
    assert_array_equal(operators["elements_m"].shape, (9, 3))
    assert_array_equal(operators["point_training"].shape, (25, 9))
    snapshot = (output / "policy_snapshot.json").read_bytes()
    assert json.loads(snapshot) == policy
    assert report["config_sha256"] == sha256(snapshot).hexdigest()
    receipt = json.loads((output / "run_receipt.json").read_text(encoding="utf8"))
    assert receipt["registered_case_count"] == 8
    assert receipt["completed_case_count"] == 8
    assert receipt["config_sha256"] == report["config_sha256"]
    assert receipt["source_hashes"] == report["source_hashes"]
    source = report["operators"][0]
    assert source["generated"] is True
    assert source["sha256"] == sha256(Path(source["path"]).read_bytes()).hexdigest()
    assert all(value["passed"] for value in source["verification"].values())
    registered = json.loads((output / "case_matrix.json").read_text(encoding="utf8"))
    assert len(registered["cases"]) == 8
    assert all(case["status"] == "pending" for case in registered["cases"])
    for case in report["cases"]:
        assert case["archive_sha256"] == sha256((output / case["archive"]).read_bytes()).hexdigest()
        saved = json.loads((output / (case["name"] + ".json")).read_text(encoding="utf8"))
        assert saved["status"] == "completed"
        assert case["observation_counts"] == {"training": 100, "validation": 36, "test": 32}


def test_noise_and_state_errors_use_paired_exact_namespaced_streams(generated_run):
    policy, _, report, _, archives = generated_run
    for seed in policy["seeds"]:
        cases = [case for case in report["cases"] if case["seed"] == seed]
        reference = archives[cases[0]["name"]]
        for case in cases:
            arrays = archives[case["name"]]
            for key in ("healthy_coefficients", "normal_amplitude_error_db", "normal_phase_error_deg",
                        "paired_fault_indices"):
                assert_array_equal(arrays[key], reference[key])
            assert_array_equal(arrays["state_error_factors"][0], np.ones(9))
            multiplier = 1 if case["regime"]["name"] == "mild" else 2
            assert_array_equal(arrays["state_error_amplitude_db"], multiplier * reference["state_error_amplitude_db"])
            assert_array_equal(arrays["state_error_phase_deg"], multiplier * reference["state_error_phase_deg"])
            for plane_id, plane in enumerate(research.PLANES):
                samples = len(arrays[research.POINT_KEYS[plane]])
                expected = []
                for state_id in range(4):
                    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(64001, plane_id, state_id)))
                    expected.append((rng.normal(size=samples) + 1j * rng.normal(size=samples)) / np.sqrt(2))
                assert_array_equal(arrays[plane + "_unit_noise"], expected)
                assert_array_equal(arrays[plane + "_unit_noise"], reference[plane + "_unit_noise"])
                assert_array_equal(arrays[plane + "_actual_codes"], arrays["canonical_codes"] * arrays["state_error_factors"])
                sigma = np.sqrt(np.mean(np.abs(arrays[plane + "_uncoded_field_clean"]) ** 2)) / 100
                assert_allclose(arrays[plane + "_absolute_noise_sigma"], sigma, rtol=2e-15)
                assert_allclose(case["noise_sigmas"][plane], sigma, rtol=2e-15)
                stored_sigma = arrays[plane + "_absolute_noise_sigma"]
                assert_array_equal(arrays[plane + "_noise"], stored_sigma * arrays[plane + "_unit_noise"])
                assert_allclose(arrays[plane + "_field"], arrays[plane + "_clean"] + arrays[plane + "_noise"], atol=0, rtol=0)
                for state_id, snr in enumerate(case["state_snr"][plane]):
                    power = np.mean(np.abs(arrays[plane + "_clean"][state_id]) ** 2)
                    assert snr["absolute_noise_sigma"] == case["noise_sigmas"][plane]
                    assert_allclose(snr["expected_snr_db"], 10 * np.log10(power / sigma**2), rtol=2e-15)
        for faults in policy["fault_states"]:
            paired = [case for case in cases if case["with_faults"] == faults]
            assert paired[0]["noise_sigmas"] == paired[1]["noise_sigmas"]
    first, second = [archives[next(case["name"] for case in report["cases"] if case["seed"] == seed)]
                     for seed in policy["seeds"]]
    assert not np.array_equal(first["training_unit_noise"], second["training_unit_noise"])


def test_saved_scores_masks_and_test_predictions_use_the_documented_gauges(generated_run):
    _, _, report, operators, archives = generated_run
    for case in report["cases"]:
        arrays = archives[case["name"]]
        truth, estimate = arrays["true_coefficients"], arrays["coefficients"]
        scalar = np.vdot(estimate, truth) / np.vdot(estimate, estimate).real
        aligned = scalar * estimate
        threshold = .1 * np.max(np.abs(truth))
        active = (np.abs(truth) > 0) & (np.abs(truth) >= threshold)
        aligned_phase = active & (np.abs(aligned) > 0) & (np.abs(aligned) >= threshold)
        raw_phase = active & (np.abs(estimate) > 0) & (np.abs(estimate) >= threshold)
        assert_allclose(arrays["aligned_coefficients"], aligned, atol=2e-15, rtol=2e-15)
        assert_allclose(case["metrics"]["alignment_complex"], [scalar.real, scalar.imag], atol=2e-15, rtol=2e-14)
        assert_array_equal(arrays["amplitude_score_mask"], active)
        assert_array_equal(arrays["aligned_phase_score_mask"], aligned_phase)
        assert_array_equal(arrays["unaligned_phase_score_mask"], raw_phase)
        assert case["aligned_mask_counts"]["truth_active_count"] == int(active.sum())
        assert case["aligned_mask_counts"]["phase_scored_count"] == int(aligned_phase.sum())
        assert case["aligned_mask_counts"]["truth_active_but_estimate_low_count"] == int((active & ~aligned_phase).sum())
        assert case["aligned_mask_counts"]["truth_inactive_count"] == int(case["with_faults"])
        raw_nrmse = np.linalg.norm(estimate - truth) / np.linalg.norm(truth)
        assert_allclose(case["unaligned_metrics"]["complex_nrmse"], raw_nrmse, atol=2e-15, rtol=2e-13)
        assert_allclose(case["metrics"]["unaligned_complex_nrmse"], raw_nrmse, atol=2e-15, rtol=2e-13)
        assert_allclose(case["metrics"]["complex_nrmse"], np.linalg.norm(aligned - truth) / np.linalg.norm(truth), atol=2e-15, rtol=2e-13)
        prediction = operators["point_test"] @ estimate
        assert_allclose(arrays["uncoded_test_prediction"], prediction, atol=2e-13, rtol=2e-15)
        clean = arrays["test_uncoded_field_clean"]
        assert_allclose(case["independent_test_clean_field_nrmse"], np.linalg.norm(prediction - clean) / np.linalg.norm(clean), atol=2e-15, rtol=2e-13)
        assert case["truth_used_for_selection"] is case["test_used_for_selection"] is False
        assert case["selection_method"] == "independent_validation"


def test_low_recovered_amplitude_mask_and_dead_channel_complex_loss_are_explicit():
    truth = np.array([1, .05, 0, 1], dtype=complex)
    estimate = np.array([2, 1, .5, .01j], dtype=complex)
    metrics, counts, active, phase = research._raw_metrics(estimate, truth)
    assert_array_equal(active, [True, False, False, True])
    assert_array_equal(phase, [True, False, False, False])
    assert counts["truth_active_count"] == 2
    assert counts["phase_scored_count"] == counts["truth_active_but_estimate_low_count"] == 1
    assert counts["phase_unscored_count"] == 3
    assert_allclose(metrics["complex_nrmse"], np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
    estimate[2] = 0
    without_dead_error = research._raw_metrics(estimate, truth)[0]["complex_nrmse"]
    assert without_dead_error < metrics["complex_nrmse"]


def test_solver_boundary_accepts_only_training_and_validation_observations(tmp_path, monkeypatch):
    original = research.solve_coded_ridge
    assert not any("truth" in key or "test" in key or "fault" in key
                   for key in inspect.signature(original).parameters)
    calls = []

    def restricted(prepared, fields, validation_matrix, validation_fields, alphas=None):
        assert fields.shape == (4, 25)
        assert validation_matrix.shape == (9, 9)
        assert validation_fields.shape == (4, 9)
        assert_array_equal(prepared.codes, research.code_patterns(3, 4))
        calls.append((fields.copy(), validation_fields.copy()))
        return original(prepared, fields, validation_matrix, validation_fields, alphas)

    monkeypatch.setattr(research, "solve_coded_ridge", restricted)
    report = _run(tmp_path, _one_case(_tiny_policy()), generate_operators=True)
    assert report["completed_case_count"] == len(calls) == 1
    saved = _load(tmp_path / "output" / report["cases"][0]["archive"])
    assert_array_equal(calls[0][0], saved["training_field"])
    assert_array_equal(calls[0][1], saved["validation_field"])


@pytest.mark.parametrize("issue", ["provenance", "frequency", "geometry", "shape", "truth_model"])
def test_input_mismatches_are_rejected_or_retained_as_preparation_failures(tmp_path, generated_run, issue):
    _, _, _, arrays, _ = generated_run
    payload = {key: value.copy() for key, value in arrays.items()}
    if issue == "provenance":
        payload["provenance"] = "measured"
    elif issue == "frequency":
        payload["frequency_hz"] = 10.7e9
    elif issue == "geometry":
        payload["elements_m"][0, 0] += .001
    elif issue == "shape":
        payload["point_validation"] = payload["point_validation"][:-1]
    else:
        payload["truth_training"] *= 1.1
    source = tmp_path / "operators.npz"
    np.savez(source, **payload)
    policy = _one_case(_tiny_policy())
    if issue in ("provenance", "frequency"):
        with pytest.raises(ValueError, match="synthetic|frequencies"):
            _run(tmp_path, policy, [source])
        assert not (tmp_path / "output").exists()
    else:
        report = _run(tmp_path, policy, [source])
        assert report["registered_case_count"] == report["failed_case_count"] == 1
        assert report["completed_case_count"] == 0
        assert report["summary"][0]["denominator"] == 1
        assert report["summary"][0]["coverage"] == 0
        assert report["summary"][0]["target_pass_counts"]["baseline"] == 0
        assert "Operator preparation" in report["cases"][0]["error"]


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_solver_failure_keeps_case_in_denominator_and_continues(tmp_path, monkeypatch, failure):
    policy = _tiny_policy()
    policy["state_error_regimes"] = policy["state_error_regimes"][:1]
    policy["max_cases"] = 4
    original = research.solve_coded_ridge
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 and failure == "exception":
            raise RuntimeError("injected solver failure")
        result = original(*args, **kwargs)
        if calls == 1:
            result["coefficients"][0] = np.nan
        return result

    monkeypatch.setattr(research, "solve_coded_ridge", fail_first)
    report = _run(tmp_path, policy, generate_operators=True)
    assert calls == report["registered_case_count"] == 4
    assert report["failed_case_count"] == 1
    assert report["completed_case_count"] == 3
    assert all(case["status"] != "pending" for case in report["cases"])
    healthy = next(group for group in report["summary"] if not group["with_faults"])
    assert healthy["denominator"] == 2
    assert healthy["coverage"] == .5
    assert healthy["counts"]["failed"] == 1
    assert healthy["target_pass_counts"]["baseline"] <= 1
    assert healthy["all_cases_baseline_passed"] is False
    assert (tmp_path / "output" / (report["cases"][0]["name"] + ".json")).exists()


def test_total_budget_skips_preparation_but_keeps_registered_denominator(tmp_path, monkeypatch):
    policy = _tiny_policy()
    policy["max_total_seconds"] = .5
    ticks = count()
    monkeypatch.setattr(research, "perf_counter", lambda: float(next(ticks)))

    def forbidden(*args, **kwargs):
        pytest.fail("Exhausted budget must not prepare scientific operators")

    monkeypatch.setattr(research, "_operators", forbidden)
    report = _run(tmp_path, policy, generate_operators=True)
    assert report["registered_case_count"] == report["timed_out_case_count"] == 8
    assert report["completed_case_count"] == report["failed_case_count"] == 0
    assert all(case["status"] == "timed_out" for case in report["cases"])
    assert sum(group["denominator"] for group in report["summary"]) == 8
    assert all(group["coverage"] == 0 and group["target_pass_counts"]["baseline"] == 0
               for group in report["summary"])
    assert (tmp_path / "output/run_receipt.json").exists()


def test_case_budget_preserves_computed_evidence_but_fails_acceptance(tmp_path, monkeypatch):
    policy = _one_case(_tiny_policy())
    policy.update(max_case_seconds=.001, max_total_seconds=1000.)
    ticks = count()
    monkeypatch.setattr(research, "perf_counter", lambda: next(ticks) * .01)
    report = _run(tmp_path, policy, generate_operators=True)
    case = report["cases"][0]
    assert report["registered_case_count"] == report["timed_out_case_count"] == 1
    assert report["completed_case_count"] == 0
    assert case["status"] == "timed_out"
    assert case["target_passed"]["baseline"] is False
    assert "metrics" in case
    assert (tmp_path / "output" / case["archive"]).exists()
    assert report["summary"][0]["coverage"] == 0


@pytest.mark.parametrize("issue", ["not_frozen", "no_exclusions", "excluded_overlap", "duplicate_seed",
                                   "fractional_seed", "matrix_budget", "phase_mask", "baseline_relaxed"])
def test_invalid_confirmation_contract_fails_before_output(tmp_path, issue):
    policy = _tiny_policy()
    policy.update(stage="confirmation", frozen=True)
    if issue == "not_frozen":
        policy["frozen"] = False
    elif issue == "no_exclusions":
        policy.pop("excluded_seeds")
    elif issue == "excluded_overlap":
        policy["seeds"][0] = 20261111  # Explicitly excluded historical confirmation root.
    elif issue == "duplicate_seed":
        policy["seeds"][1] = policy["seeds"][0]
    elif issue == "fractional_seed":
        policy["seeds"][0] = 1.5
    elif issue == "matrix_budget":
        policy["max_cases"] = 7
    elif issue == "phase_mask":
        policy["phase_mask_ratio"] = .2
    else:
        policy["targets"]["baseline"]["complex_nrmse"] = .2
    with pytest.raises(ValueError):
        _run(tmp_path, policy, generate_operators=True)
    assert not (tmp_path / "output").exists()


def test_operator_generation_requires_explicit_opt_in_and_outputs_are_not_overwritten(tmp_path):
    policy = _one_case(_tiny_policy())
    with pytest.raises(ValueError, match="explicitly enable generation"):
        _run(tmp_path, policy)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("retained evidence", encoding="utf8")
    with pytest.raises(FileExistsError):
        _run(tmp_path, policy, generate_operators=True)
    assert sentinel.read_text(encoding="utf8") == "retained evidence"
