"""Synthetic tests of identifiability and truth-isolated rejection contracts."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_diagnostics import (
    DEVELOPMENT_THRESHOLDS,
    INTERNAL_SHAPE_PROTOCOL_V2,
    SENSITIVITY_POLICY_V2,
    assess_fit_diagnostics,
    assess_sensitivity_admissibility,
    audit_completed_numerical_abstention,
    audit_virtual_gates,
    data_only_geometry_singular_values,
    group_near_modes,
    profile_geometry_singular_values,
    fit_record_binding,
    validate_numerical_abstention_policy,
)


def _fit(geometry=(62.2, 1.3, -2.4), cost=1.0, success=True):
    return {"geometry_mm": list(geometry), "cost": cost, "success": success,
            "status": 1 if success else 0, "message": "synthetic", "nfev": 8}


def _decision(fit=None, frozen=False):
    fit = _fit() if fit is None else fit
    return assess_fit_diagnostics(
        fit, predictive_nrmse=0.01, modes=group_near_modes([{**fit, "start_id": "a"}, {**fit, "start_id": "b"}]),
        sensitivity_geometries_mm={"lambda0.1": fit, "lambda10": fit, "subset": fit},
        thresholds={"frozen": frozen})


def test_geometry_projection_removes_unidentifiable_direction_and_handles_units():
    # First two geometry directions are independently observable.  The third
    # is exactly a gain direction, despite a large unprojected derivative.
    j = np.zeros((5, 4))
    j[0, 0], j[1, 1], j[2, 2], j[2, 3] = 30, 20, 700, 7
    result = profile_geometry_singular_values(j, geometry_scales_mm=(10, 10, 10))
    assert_allclose(result["singular_values"], [3, 2, 0], atol=1e-13)
    assert result["rank"] == 2
    assert result["condition_number"] is None
    assert result["nuisance_rank"] == 1
    assert result["prior_included"] is True


def test_projection_is_invariant_to_nuisance_basis_and_redundant_columns():
    rng = np.random.default_rng(918)
    geometry = rng.normal(size=(19, 3))
    nuisance = rng.normal(size=(19, 2))
    a = profile_geometry_singular_values(geometry, nuisance_jacobian=nuisance,
                                         residual_kind="data_only_full")
    changed = np.column_stack((3 * nuisance[:, 0] + 4 * nuisance[:, 1],
                               -2 * nuisance[:, 0] + nuisance[:, 1], nuisance[:, 0]))
    b = profile_geometry_singular_values(geometry, nuisance_jacobian=changed,
                                         residual_kind="data_only_full")
    assert_allclose(a["singular_values"], b["singular_values"], rtol=1e-13)
    assert a["nuisance_rank"] == b["nuisance_rank"] == 2
    assert a["prior_included"] is False


def test_nuisance_parameter_units_do_not_hide_exact_geometric_confounding():
    nuisance = np.zeros((5, 2))
    nuisance[0, 0], nuisance[1, 1] = 1, 1
    geom = np.zeros((5, 3))
    geom[2, 0], geom[3, 1], geom[1, 2] = 1, 1, 1
    result = profile_geometry_singular_values(geom, nuisance_jacobian=nuisance * [1e12, 1e-8])
    assert result["rank"] == 2
    complex_result = data_only_geometry_singular_values(
        [{"geometry_jacobian": geom.astype(complex), "nuisance_jacobian": nuisance.astype(complex) * [1e12, 1e-8]}])
    assert complex_result["rank"] == 2


def test_cancellation_roundoff_does_not_create_geometry_information():
    rng = np.random.default_rng(32)
    nuisance = rng.normal(size=(30, 4))
    geometry = nuisance @ rng.normal(size=(4, 3))
    result = profile_geometry_singular_values(geometry, nuisance_jacobian=nuisance)
    assert result["rank"] == 0


def test_profile_data_label_does_not_claim_unpenalized_data_identifiability():
    result = profile_geometry_singular_values(np.eye(3), residual_kind="data_profile")
    assert "not_pure_data_information" in result["interpretation"]
    with pytest.raises(ValueError, match="real and imaginary"):
        profile_geometry_singular_values(np.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="disjoint"):
        profile_geometry_singular_values(np.eye(3), nuisance_columns=[0])


def test_complex_data_block_projection_matches_full_real_nuisance_projection():
    rng = np.random.default_rng(120)
    geom = rng.normal(size=(23, 3)) + 1j * rng.normal(size=(23, 3))
    nuisance = rng.normal(size=(23, 5)) + 1j * rng.normal(size=(23, 5))
    # A complex nuisance delta has independent real and imaginary parts.
    real_nuisance = np.block([[nuisance.real, -nuisance.imag],
                              [nuisance.imag, nuisance.real]])
    reference = profile_geometry_singular_values(
        np.vstack((geom.real, geom.imag)), nuisance_jacobian=real_nuisance,
        residual_kind="data_only_full", geometry_scales_mm=(10, 10, 10))
    actual = data_only_geometry_singular_values(
        [{"name": "f", "geometry_jacobian": geom, "nuisance_jacobian": nuisance}],
        geometry_scales_mm=(10, 10, 10))
    assert_allclose(reference["singular_values"], actual["singular_values"], rtol=1e-13)
    assert actual["nuisance_rank"] == 10


def test_complex_projection_does_not_treat_numerical_cancellation_as_rank():
    rng = np.random.default_rng(121)
    nuisance = rng.normal(size=(23, 5)) + 1j * rng.normal(size=(23, 5))
    geom = nuisance @ (rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)))
    result = data_only_geometry_singular_values(
        [{"geometry_jacobian": geom, "nuisance_jacobian": nuisance}])
    assert result["rank"] == 0


def test_complete_link_mode_groups_do_not_chain_and_keep_failed_better_starts():
    records = [_fit((60, x, 0), 1 + x) for x in (0, 0.009, 0.018)]
    records.append(_fit((65, 2, 2), 0.9, False))
    records.append({"geometry_mm": [np.nan, 0, 0], "cost": np.inf,
                    "success": False, "message": "failed linear solve"})
    result = group_near_modes(records, coordinate_tolerance_mm=0.01)
    assert [m["record_indices"] for m in result["modes"]] == [[0, 1], [2]]
    assert result["unresolved_near_failed_starts"] == [3]
    assert result["failed_record_indices"] == [3, 4]
    assert result["records"][4]["message"] == "failed linear solve"
    assert result["envelope_is_confidence_region"] is False
    assert_allclose(result["maximum_near_mode_separation_mm"], 0.018)


def test_zero_cost_mode_envelope_uses_explicit_absolute_floor():
    result = group_near_modes([_fit(cost=0), _fit((63, 1, 0), cost=1e-13)],
                             absolute_cost_envelope=1e-12)
    assert len(result["modes"]) == 2
    failed = group_near_modes([_fit(success=False)])
    assert failed["status"] == "no_converged_valid_start"
    assert len(failed["records"]) == 1


def test_observable_decision_ignores_all_truth_fields():
    fit = _fit()
    before = _decision(fit, frozen=True)
    fit["final_score"] = {"geometry_error_mm": [500, 1000, -80]}
    fit["truth_geometry_mm"] = [0, 0, 0]
    assert _decision(fit, frozen=True) == before
    assert before["would_accept_under_rule"] is True
    assert before["accepted_for_precision_claim"] is True
    assert before["truth_used_for_decision"] is False
    assert before["external_virtual_gates_still_required"] is True
    assert _decision(frozen=False)["accepted_for_precision_claim"] is False


def test_missing_diagnostics_are_unknown_and_never_pass():
    result = assess_fit_diagnostics(_fit())
    assert result["status"] == "uncertain"
    assert result["would_accept_under_rule"] is False
    assert set(result["unknowns"]) == {
        "prediction_diagnostic_missing", "multistart_modes_missing",
        "lambda_subset_sensitivity_missing"}


def test_strings_cannot_be_mistaken_for_positive_boolean_evidence():
    fit = _fit()
    fit["success"] = "false"
    assert _decision(fit)["would_accept_under_rule"] is False
    with pytest.raises(ValueError, match="explicit boolean"):
        assess_fit_diagnostics(_fit(), thresholds={"frozen": "false"})
    records = [_scene("confirmation"), _scene("ideal", (0, 0, 0)), _scene("stress")]
    prereq = _prerequisites()
    prereq["v0_passed"] = "unknown"
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=prereq)
    assert result["gates"]["v0_passed"] is False
    assert result["real_fit_gate_open"] is False


@pytest.mark.parametrize("which,expected", [
    ("prediction", "prediction_threshold_exceeded"),
    ("nonconverged", "selected_fit_not_converged"),
    ("boundary", "hard_boundary_hit"),
    ("mode", "separated_near_optimal_modes"),
    ("sensitivity", "lambda_subset_geometry_drift"),
])
def test_each_observable_failure_prevents_precision_claim(which, expected):
    fit = _fit((70, 1, 0) if which == "boundary" else (62, 1, 0), success=which != "nonconverged")
    modes = group_near_modes([fit, _fit((64, 1, 0))] if which == "mode" else [_fit(fit["geometry_mm"])])
    result = assess_fit_diagnostics(
        fit, predictive_nrmse=0.051 if which == "prediction" else 0.01,
        modes=modes,
        sensitivity_geometries_mm=[[64, 1, 0]] if which == "sensitivity" else [fit["geometry_mm"]],
        thresholds={"frozen": True})
    assert expected in result["rejection_reasons"] + result["uncertainty_reasons"]
    assert result["accepted_for_precision_claim"] is False


def _scene(kind, error=(0.1, 0.2, -0.1), identifier="example"):
    decision = _decision(frozen=True)
    return {"id": identifier, "kind": kind, "full_array": True,
            "seed": sum(ord(c) for c in identifier) + 1000, "snr_db": 40,
            "identifiable_positive_control": kind == "ideal", "same_mode_multistart_difference_mm": 0,
            "fit_binding": deepcopy(decision["fit_binding"]),
            "diagnostic_decision": decision,
            "final_score": {"geometry_error_mm": list(error),
                            "fit_binding": deepcopy(decision["fit_binding"]),
                            "unseen_angles": {"angle": {"clean_nrmse": 0.02}},
                            "weights": {"f": {"amplitude_rmse_db": 0.1,
                                                "phase_rmse_deg": 1, "complex_nrmse": 0.03}}}}


def _prerequisites():
    return dict.fromkeys(("v0_passed", "v1_completed", "confirmation_frozen_before_scores",
                          "available_information_confirmed", "stress_classes_complete",
                          "stability_checks_complete"), True)


def _manifest(records):
    return {record["id"]: {"seed": record["seed"], "snr_db": record["snr_db"],
                           "unseen_angle_ids": list(record["final_score"]["unseen_angles"])}
            for record in records if record["kind"] == "confirmation"}


def test_truth_audit_detects_confident_wrong_stress_without_rewriting_decision():
    records = [_scene("confirmation", identifier="a"), _scene("confirmation", identifier="b"),
               _scene("ideal", (0.001, 0.002, 0.001)),
               _scene("stress", (1.2, 0, 0), "wrong calibration")]
    before = deepcopy(records[-1]["diagnostic_decision"])
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=2, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["real_fit_gate_open"] is False
    assert result["false_precision_acceptances"] == ["wrong calibration"]
    assert result["gates"]["stress_no_false_precision_acceptance"] is False
    assert records[-1]["diagnostic_decision"] == before


def test_rejection_is_successful_stress_behavior_and_all_explicit_gates_required():
    records = [_scene("confirmation"), _scene("ideal", (0.001, 0, 0)),
               _scene("stress", (2, 0, 0))]
    records[-1]["diagnostic_decision"] = _decision(_fit(success=False), frozen=True)
    records[-1]["fit_binding"] = records[-1]["diagnostic_decision"]["fit_binding"]
    records[-1]["final_score"]["fit_binding"] = records[-1]["diagnostic_decision"]["fit_binding"]
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=_prerequisites(), require_weights=True,
                                  expected_confirmation_manifest=_manifest(records))
    assert result["real_fit_gate_open"] is True
    assert result["population_reliability_or_confidence_coverage_claim"] is False
    missing = audit_virtual_gates(records, thresholds={"frozen": True}, confirmation_count=1)
    assert missing["real_fit_gate_open"] is False
    assert "v0_passed" in missing["missing_evidence"]
    unfrozen = audit_virtual_gates(records, confirmation_count=1, prerequisites=_prerequisites())
    assert unfrozen["real_fit_gate_open"] is False
    assert unfrozen["gates"]["thresholds_frozen"] is False


def test_confirmation_p90_and_worst_axis_are_both_enforced():
    records = [_scene("confirmation", (0.1, 0, 0), str(i)) for i in range(12)]
    records[-1]["final_score"]["geometry_error_mm"] = [1.01, 0, 0]
    result = audit_virtual_gates(records, thresholds={"frozen": True})
    assert result["axis_absolute_error_p90_mm"][0] < 0.5
    assert result["axis_absolute_error_max_mm"][0] == 1.01
    assert result["gates"]["confirmation_geometry"] is False
    assert DEVELOPMENT_THRESHOLDS["frozen"] is False


def test_full_array_claim_and_clean_holdout_scores_must_be_explicit():
    record = _scene("confirmation")
    del record["full_array"]
    record["final_score"]["unseen_angles"] = {}
    result = audit_virtual_gates([record], confirmation_count=1)
    assert result["gates"]["complete_1024_columns"] is None
    assert result["gates"]["unseen_angle_clean_prediction"] is None


def test_single_start_or_unverified_sensitivity_refit_is_not_complete_evidence():
    fit = _fit()
    result = assess_fit_diagnostics(fit, predictive_nrmse=0.01,
                                    modes=group_near_modes([fit]),
                                    sensitivity_geometries_mm={"lambda0.1": fit["geometry_mm"]})
    assert result["would_accept_under_rule"] is False
    assert "insufficient_converged_starts" in result["unknowns"]
    assert "sensitivity_label_missing:subset" in result["unknowns"]
    assert "sensitivity_refit_convergence_unverified" in result["unknowns"]


def test_audit_cannot_retroactively_freeze_or_change_observable_rules():
    records = [_scene("confirmation"), _scene("ideal", (0, 0, 0)), _scene("stress")]
    result = audit_virtual_gates(records, thresholds={"frozen": True, "prediction_nrmse_max": 0.005},
                                  confirmation_count=1, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["gates"]["observable_rule_binding"] is False
    assert result["real_fit_gate_open"] is False
    records[0]["diagnostic_decision"] = _decision(frozen=False)
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["gates"]["observable_rule_binding"] is False


def test_inconsistent_full_array_flags_and_non_boolean_stress_decisions_fail_closed():
    records = [_scene("confirmation"), _scene("ideal", (0, 0, 0)), _scene("stress", (2, 0, 0))]
    records[0]["full_array"] = False
    records[0]["element_count"] = 1024
    records[2]["diagnostic_decision"]["would_accept_under_rule"] = "True"
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["gates"]["complete_1024_columns"] is False
    assert result["gates"]["stress_no_false_precision_acceptance"] is None
    assert result["real_fit_gate_open"] is False


def test_duplicate_confirmation_cases_and_score_fit_swaps_do_not_pass():
    records = [_scene("confirmation"), _scene("confirmation"), _scene("ideal", (0, 0, 0)), _scene("stress")]
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=2, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["gates"]["confirmation_manifest_binding"] is False
    records = [records[0], records[2], records[3]]
    records[0]["final_score"]["fit_binding"] = {"sha256": "unrelated fit"}
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=_manifest(records))
    assert result["gates"]["fit_decision_score_binding"] is False


def test_null_manifest_fields_cannot_match_missing_case_identity():
    records = [_scene("confirmation"), _scene("ideal", (0, 0, 0)), _scene("stress")]
    manifest = _manifest(records)
    del records[0]["seed"]
    del records[0]["snr_db"]
    manifest[records[0]["id"]]["seed"] = None
    manifest[records[0]["id"]]["snr_db"] = None
    result = audit_virtual_gates(records, thresholds={"frozen": True},
                                  confirmation_count=1, prerequisites=_prerequisites(),
                                  expected_confirmation_manifest=manifest)
    assert result["gates"]["confirmation_manifest_binding"] is False
    assert result["real_fit_gate_open"] is False


def test_duplicate_or_missing_start_identity_is_not_multistart_evidence():
    fit = _fit()
    sensitivities = {name: fit for name in ("lambda0.1", "lambda10", "subset")}
    for attempts in ([{**fit, "start_id": "same"}] * 2, [fit, fit]):
        result = assess_fit_diagnostics(fit, predictive_nrmse=0.01,
                                        modes=group_near_modes(attempts),
                                        sensitivity_geometries_mm=sensitivities)
        assert result["would_accept_under_rule"] is False
        assert "insufficient_converged_starts" in result["unknowns"]


def _sensitivity_candidate(fit, score=0.02, *, inner_passed=True, source="internal_observed"):
    return {"fit": fit, "inner_audit_passed": inner_passed,
            "internal_validation": {"protocol_id": INTERNAL_SHAPE_PROTOCOL_V2,
                "source": source, "noisy_nrmse": score,
                "fit_binding": _decision(fit)["fit_binding"]}}


def test_v2_separates_all_tested_drift_from_prediction_admissible_drift():
    fit = _fit()
    close = _fit((62.3, 1.3, -2.4))
    far = _fit((68.2, 1.3, -2.4))
    candidates = {"lambda0.1": _sensitivity_candidate(close),
                  "lambda10": _sensitivity_candidate(far, 0.06),
                  "subset": _sensitivity_candidate(close)}
    report = assess_sensitivity_admissibility(fit["geometry_mm"], candidates)
    assert_allclose(report["all_tested_drift_mm"], 6.0)
    assert_allclose(report["admissible_drift_mm"], 0.1)
    assert report["excluded_by_prediction_labels"] == ["lambda10"]
    assert report["evidence_complete"] is True
    assert report["unstable_admissible"] is False
    assert report["truth_used"] is False
    assert report["sealed_evaluation_used"] is False
    assert report["cross_lambda_objective_costs_compared"] is False
    result = assess_fit_diagnostics(fit, predictive_nrmse=0.02,
        modes=group_near_modes([{**fit,"start_id":"a"},{**fit,"start_id":"b"}]),
        sensitivity_geometries_mm=candidates,
        thresholds={"sensitivity_policy_version": SENSITIVITY_POLICY_V2})
    assert result["would_accept_under_rule"] is True
    assert result["accepted_for_precision_claim"] is False  # not frozen
    assert result["sensitivity_policy_version"] == SENSITIVITY_POLICY_V2


def test_v2_nonconvergence_or_unreviewed_inner_is_unknown_never_prediction_excluded():
    fit = _fit()
    candidates = {"lambda0.1": _sensitivity_candidate(_fit(success=False), 0.9),
                  "lambda10": _sensitivity_candidate(fit, 0.9, inner_passed=False),
                  "subset": _sensitivity_candidate(fit)}
    report = assess_sensitivity_admissibility(fit["geometry_mm"], candidates)
    assert report["evidence_complete"] is False
    assert report["excluded_by_prediction_labels"] == []
    assert set(report["unknown_labels"]) == {"lambda0.1", "lambda10"}


def test_v2_cannot_ignore_an_admissible_remote_branch_or_use_geometry_to_exclude_it():
    fit = _fit()
    far = _fit((63.5, 1.3, -2.4), cost=1000000)
    candidates = {name: _sensitivity_candidate(far) for name in ("lambda0.1","lambda10","subset")}
    # Huge training cost is intentionally irrelevant across different lambda.
    report = assess_sensitivity_admissibility(fit["geometry_mm"], candidates)
    assert report["unstable_admissible"] is True
    assert len(report["admissible_labels"]) == 3
    assert report["excluded_by_prediction_labels"] == []
    result = assess_fit_diagnostics(fit, predictive_nrmse=0.01,
        modes=group_near_modes([{**fit,"start_id":"a"},{**fit,"start_id":"b"}]),
        sensitivity_geometries_mm=candidates,
        thresholds={"sensitivity_policy_version": SENSITIVITY_POLICY_V2, "frozen": True})
    assert result["would_accept_under_rule"] is False
    assert "admissible_lambda_subset_geometry_drift" in result["uncertainty_reasons"]


def test_v2_all_excluded_or_missing_labels_remain_insufficient_evidence():
    fit = _fit()
    report = assess_sensitivity_admissibility(fit["geometry_mm"],
        {"lambda0.1": _sensitivity_candidate(fit, 0.2)})
    assert report["evidence_complete"] is False
    assert "no_sufficient_admissible_sensitivity_alternative" in report["unknowns"]
    assert "sensitivity_label_missing:subset" in report["unknowns"]


def test_v2_requires_the_frozen_internal_observed_protocol_and_exact_fit_binding():
    fit = _fit()
    candidate = _sensitivity_candidate(fit, source="sealed_clean_truth")
    report = assess_sensitivity_admissibility(fit["geometry_mm"],
        {name: candidate for name in ("lambda0.1","lambda10","subset")})
    assert len(report["unknown_labels"]) == 3
    candidate = _sensitivity_candidate(fit)
    candidate["internal_validation"]["fit_binding"] = {"sha256":"another fit"}
    report = assess_sensitivity_admissibility(fit["geometry_mm"],
        {name: candidate for name in ("lambda0.1","lambda10","subset")})
    assert len(report["unknown_labels"]) == 3


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _seal_abstention(record):
    lock=record["numerical_abstention_lock"]
    lock["observable_lock_sha256"]=_content_hash(record["diagnostic_decision"])
    for attempt in lock["attempts"]:
        if "inner_diagnostics" in attempt:
            attempt["inner_diagnostics_sha256"]=_content_hash(attempt["inner_diagnostics"])
    record["numerical_abstention_lock_sha256"]=_content_hash(lock)
    record["score_barrier_abstention_lock_sha256"]=_content_hash(lock)
    return record


def _abstention_policy():
    return {"enabled":True,"frozen":True,"required_stress_families":["family"]}


def _abstention_record(inner=False):
    fit={"id":"controlled_refusal","success":False,"status":-3}
    decision=assess_fit_diagnostics(fit,thresholds={"frozen":True})
    record={"id":"controlled_refusal","kind":"stress","stress_family":"family","full_array":True,
        "fit":fit,"fit_binding":fit_record_binding(fit),"diagnostic_decision":decision,
        "final_score":{"fit_binding":fit_record_binding(fit)},"selected_endpoint_available":False,
        "freeze_sha256":"a"*64,"split_sha256":"b"*64,"attempt_seal_sha256":"c"*64,
        "score_barrier_sha256":"d"*64,"source_sha256":{"solver.py":"e"*64},
        "dataset_sha256":{name:"f"*64 for name in ("fit_data.npz","manifest.json","truth_for_scoring_only.npz")},
        "truth_used_only_after_global_locks":True}
    budget=None
    attempt={"trajectory_id":"t0","status":"completed_no_estimate","started":True,
        "contract_sha256":"1"*64,"terminal_sha256":"2"*64,"budget_ledger_sha256":"3"*64,
        "budget_ledger_verified":True,"reserved_profile_evaluations":3,
        "reason_code":"profile_evaluation_budget_exhausted","exception_type":"EvaluationBudgetExhausted",
        "budget_exhausted_after_completed_reservations":True}
    if inner:
        budget={"n_starts":3,"maxiter":5000,"newton_steps":250,"random_seed":20261028,
                "stationarity_tol":1e-12,"min_hessian_rcond":1e-12,"epsilon":1e-8}
        diag={"success":False,"n_starts":3,"random_seed":20261028,"stationarity_tol":1e-12,
              "epsilon":1e-8,"chosen_start":0,"stationarity":0.1,"hessian_rcond":0.01,"hessian_positive_definite":True,
              "starts":[{"index":i,"objective":1.0,"stationarity":0.1,"converged":False,
                         "optimizer_success":True,"optimizer_message":"optimizer terminated; exact gradient still above criterion",
                         "nfev":15,"njev":15,"newton_steps":250,"hessian_rcond":0.01,
                         "hessian_positive_definite":True} for i in range(3)]}
        attempt.update(exception_type="AmplitudeProfileError",reason_code="amplitude_inner_stationarity_rejected",
                       reserved_profile_evaluations=1,inner_budget=budget,inner_diagnostics=diag)
    record["numerical_abstention_lock"]={
        "schema_version":1,"status":"completed_numerical_abstention","all_planned_trajectories_terminal":True,
        "observable_decision_frozen_before_truth":True,"decision_fit_binding":record["fit_binding"],
        **{key:deepcopy(record[key]) for key in ("freeze_sha256","split_sha256","attempt_seal_sha256","source_sha256","dataset_sha256")},
        "attempt_plan":{"profile_evaluation_caps":{"t0":3},"trajectory_contract_sha256":{"t0":"1"*64},
                        "inner_budgets":{"t0":budget}},"attempts":[attempt]}
    return _seal_abstention(record)


def test_numerical_abstention_is_default_off_and_needs_genuine_complete_bound_cap():
    record=_abstention_record()
    assert audit_completed_numerical_abstention(record)["classification"]=="unknown"
    audit=audit_completed_numerical_abstention(record,policy=_abstention_policy())
    assert audit["classification"]=="completed_numerical_abstention"
    assert audit["physical_wrong_model_identification_claim"] is False
    record["numerical_abstention_lock"]["attempts"][0]["reserved_profile_evaluations"]=2
    _seal_abstention(record)
    audit=audit_completed_numerical_abstention(record,policy=_abstention_policy())
    assert audit["classification"]=="unknown"
    assert "budget_exhaustion_not_proven:t0" in audit["reasons"]


@pytest.mark.parametrize("exception",["MemoryError","ImportError","RuntimeError","DeviceOOM","ValueError"])
def test_unexpected_software_or_device_exceptions_never_count_as_stress_rejection(exception):
    record=_abstention_record()
    record["numerical_abstention_lock"]["attempts"][0]["exception_type"]=exception
    _seal_abstention(record)
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"


@pytest.mark.parametrize("tamper",["unstarted","partial","missing_terminal","duplicate_terminal","source","split","decision","barrier"])
def test_incomplete_or_rebound_abstention_is_unknown_even_if_payload_is_rehashed(tamper):
    record=_abstention_record()
    lock=record["numerical_abstention_lock"]
    if tamper=="unstarted":lock["attempts"][0]["started"]=False
    elif tamper=="partial":lock["attempts"][0]["status"]="partial"
    elif tamper=="missing_terminal":lock["attempts"]=[]
    elif tamper=="duplicate_terminal":lock["attempts"]*=2
    elif tamper=="source":record["source_sha256"]["solver.py"]="0"*64
    elif tamper=="split":record["split_sha256"]="0"*64
    elif tamper=="decision":record["diagnostic_decision"]["would_accept_under_rule"]=True
    _seal_abstention(record)
    if tamper=="barrier":record["score_barrier_abstention_lock_sha256"]="0"*64
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"


def test_same_named_amplitude_error_needs_three_full_starts_and_frozen_inner_budget():
    record=_abstention_record(inner=True)
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="completed_numerical_abstention"
    record["numerical_abstention_lock"]["attempts"][0]["inner_diagnostics"]["starts"].pop()
    _seal_abstention(record)
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"
    record=_abstention_record(inner=True)
    record["numerical_abstention_lock"]["attempt_plan"]["inner_budgets"]["t0"]["maxiter"]=10
    # Separate the actual budget from the altered plan, as a real binding check.
    record["numerical_abstention_lock"]["attempts"][0]["inner_budget"]={"n_starts":3,"maxiter":5000,
        "newton_steps":250,"random_seed":20261028,"stationarity_tol":1e-12,"min_hessian_rcond":1e-12,"epsilon":1e-8}
    _seal_abstention(record)
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"


def test_strict_stress_family_endpoint_requirement_and_core_gate_are_not_weakened():
    main=_scene("confirmation",identifier="main")
    ideal=_scene("ideal",(0,0,0),"ideal")
    numerical=_abstention_record()
    kwargs={"thresholds":{"frozen":True},"confirmation_count":1,"prerequisites":_prerequisites(),
            "expected_confirmation_manifest":_manifest([main]),"stress_abstention_policy":_abstention_policy()}
    audit=audit_virtual_gates([main,ideal,numerical],**kwargs)
    assert audit["completed_numerical_abstentions"]==["controlled_refusal"]
    assert audit["gates"]["stress_valid_endpoint_per_required_family"] is False
    assert audit["real_fit_gate_open"] is False
    endpoint=_scene("stress",(2,0,0),"valid_endpoint")
    endpoint["stress_family"]="family"
    endpoint["fit"]={**_fit(),"inner_audit_passed":True,"fit_id":"a"*64}
    endpoint["selected_endpoint_available"]=True
    endpoint["endpoint_binding_verified_before_truth"]=True
    endpoint["selected_terminal_sha256"]="b"*64
    endpoint["truth_used_only_after_global_locks"]=True
    endpoint["diagnostic_decision"]=assess_fit_diagnostics(endpoint["fit"],predictive_nrmse=0.1,thresholds={"frozen":True})
    endpoint["fit_binding"]=endpoint["diagnostic_decision"]["fit_binding"]
    endpoint["final_score"]["fit_binding"]=endpoint["fit_binding"]
    audit=audit_virtual_gates([main,ideal,numerical,endpoint],**kwargs)
    assert audit["gates"]["stress_valid_endpoint_per_required_family"] is True
    assert audit["real_fit_gate_open"] is True
    main["final_score"]["geometry_error_mm"]=[2,0,0]
    assert audit_virtual_gates([main,ideal,numerical,endpoint],**kwargs)["real_fit_gate_open"] is False


def test_usable_inner_start_cannot_be_relabelled_failed_to_manufacture_abstention():
    record=_abstention_record(inner=True)
    diagnostic=record["numerical_abstention_lock"]["attempts"][0]["inner_diagnostics"]
    diagnostic["starts"][1]["stationarity"]=1e-14
    # The false converged flag is intentionally retained despite usable numbers.
    _seal_abstention(record)
    assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"


def test_inner_fallback_selection_and_top_diagnostics_must_match_all_start_records():
    for tamper in ("selection","summary"):
        record=_abstention_record(inner=True)
        diagnostic=record["numerical_abstention_lock"]["attempts"][0]["inner_diagnostics"]
        if tamper=="selection":diagnostic["chosen_start"]=1
        else:diagnostic["stationarity"]=1e-14
        _seal_abstention(record)
        assert audit_completed_numerical_abstention(record,policy=_abstention_policy())["classification"]=="unknown"


def test_placeholder_stress_error_does_not_satisfy_required_endpoint_family():
    main=_scene("confirmation",identifier="main")
    records=[main,_scene("ideal",(0,0,0),"ideal"),_abstention_record(),_scene("stress",(0,0,0),"placeholder")]
    records[-1]["stress_family"]="family"
    audit=audit_virtual_gates(records,thresholds={"frozen":True},confirmation_count=1,
        prerequisites=_prerequisites(),expected_confirmation_manifest=_manifest([main]),
        stress_abstention_policy=_abstention_policy())
    assert audit["gates"]["stress_valid_endpoint_per_required_family"] is False
    assert audit["real_fit_gate_open"] is False


def test_valid_family_endpoint_cannot_hide_another_software_failure_with_placeholder_score():
    main=_scene("confirmation",identifier="main")
    ideal=_scene("ideal",(0,0,0),"ideal")
    endpoint=_scene("stress",(2,0,0),"real_endpoint")
    fit={**_fit(),"inner_audit_passed":True,"fit_id":"a"*64}
    endpoint.update(fit=fit,stress_family="family",selected_endpoint_available=True,
        endpoint_binding_verified_before_truth=True,selected_terminal_sha256="b"*64,
        truth_used_only_after_global_locks=True)
    endpoint["diagnostic_decision"]=assess_fit_diagnostics(fit,predictive_nrmse=.1,thresholds={"frozen":True})
    endpoint["fit_binding"]=endpoint["diagnostic_decision"]["fit_binding"]
    endpoint["final_score"]["fit_binding"]=endpoint["fit_binding"]
    software=_abstention_record()
    software["id"]="software_failure"
    software["final_score"]["geometry_error_mm"]=[0,0,0]
    software["exception"]="MemoryError"
    software["numerical_abstention_lock"]["attempts"][0]["exception_type"]="MemoryError"
    _seal_abstention(software)
    audit=audit_virtual_gates([main,ideal,endpoint,software],thresholds={"frozen":True},
        confirmation_count=1,prerequisites=_prerequisites(),expected_confirmation_manifest=_manifest([main]),
        stress_abstention_policy=_abstention_policy())
    assert audit["gates"]["stress_valid_endpoint_per_required_family"] is True
    assert "software_failure" in audit["stress_unknowns"]
    assert audit["real_fit_gate_open"] is False


def test_abstention_policy_cannot_omit_endpoint_families_or_use_nonboolean_flags():
    with pytest.raises(ValueError,match="required stress families"):
        validate_numerical_abstention_policy({"enabled":True})
    with pytest.raises(ValueError,match="explicit boolean"):
        validate_numerical_abstention_policy({"enabled":"True"})
    with pytest.raises(ValueError,match="may not remove"):
        validate_numerical_abstention_policy({"enabled":True,"frozen":True,"require_valid_endpoint_per_family":False})
