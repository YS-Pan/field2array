"""Observable joint-fit diagnostics and separate synthetic truth audits.

This module has no file access.  Fit rejection never inspects coefficient or
geometry truth.  Default thresholds are development hypotheses: they must be
locked before confirmation, and an objective envelope is not a confidence set.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import numpy as np


DEVELOPMENT_THRESHOLDS = {
    "schema_version": 1,
    "status": "development_not_frozen",
    "frozen": False,
    "prediction_nrmse_max": 0.05,
    "separated_modes_mm_max": 1.0,
    "sensitivity_drift_mm_max": 1.0,
    "boundary_tolerance_mm": 1e-5,
    "minimum_completed_starts": 2,
    "minimum_converged_starts": 2,
    "required_sensitivity_labels": ["lambda0.1", "lambda10", "subset"],
    "require_sensitivity_convergence": True,
    "bounds_mm": [[50.0, 70.0], [-10.0, 10.0], [-10.0, 10.0]],
    "confirmation_axis_p90_mm_max": 0.5,
    "confirmation_axis_error_mm_max": 1.0,
    "unseen_angle_clean_nrmse_max": 0.05,
    "ideal_axis_error_mm_max": 0.02,
    "same_mode_multistart_difference_mm_max": 0.01,
    "amplitude_rmse_db_max": 0.5,
    "phase_rmse_deg_max": 5.0,
    "weight_complex_nrmse_max": 0.10,
    "note": "Observable rules only; truth enters audit_virtual_gates afterward.",
}

SENSITIVITY_POLICY_V2 = "admissible_internal_shape_v2"
INTERNAL_SHAPE_PROTOCOL_V2 = "internal_plus15_single_complex_scalar_disjoint_observed_shape_v1"
NUMERICAL_ABSTENTION_POLICY_V1 = "completed_numerical_abstention_v1"


def _thresholds(value):
    result = deepcopy(DEVELOPMENT_THRESHOLDS)
    if value is not None:
        result.update(dict(value))
    sensitivity_version = result.get("sensitivity_policy_version", "all_tested_drift_v1")
    if sensitivity_version not in ("all_tested_drift_v1", SENSITIVITY_POLICY_V2):
        raise ValueError("unknown sensitivity_policy_version")
    if sensitivity_version == SENSITIVITY_POLICY_V2:
        result.setdefault("internal_validation_protocol_id", INTERNAL_SHAPE_PROTOCOL_V2)
        result.setdefault("minimum_admissible_sensitivity_candidates", 1)
        minimum = result["minimum_admissible_sensitivity_candidates"]
        if isinstance(minimum, (bool, np.bool_)) or int(minimum) != minimum or minimum < 1:
            raise ValueError("minimum_admissible_sensitivity_candidates must be a positive integer")
        if not isinstance(result["internal_validation_protocol_id"], str) or not result["internal_validation_protocol_id"]:
            raise ValueError("internal_validation_protocol_id must be nonempty")
    if not isinstance(result["frozen"], (bool, np.bool_)):
        raise ValueError("frozen must be an explicit boolean")
    result["frozen"] = bool(result["frozen"])
    for name in ("minimum_completed_starts", "minimum_converged_starts"):
        if isinstance(result[name], (bool, np.bool_)) or int(result[name]) != result[name] or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
        result[name] = int(result[name])
    if not isinstance(result["require_sensitivity_convergence"], (bool, np.bool_)):
        raise ValueError("require_sensitivity_convergence must be an explicit boolean")
    if not isinstance(result["required_sensitivity_labels"], (list, tuple)) or not all(isinstance(s, str) and s for s in result["required_sensitivity_labels"]):
        raise ValueError("required_sensitivity_labels must contain nonempty strings")
    for name in (
        "prediction_nrmse_max", "separated_modes_mm_max",
        "sensitivity_drift_mm_max", "boundary_tolerance_mm",
        "confirmation_axis_p90_mm_max", "confirmation_axis_error_mm_max",
        "unseen_angle_clean_nrmse_max", "ideal_axis_error_mm_max", "same_mode_multistart_difference_mm_max",
        "amplitude_rmse_db_max", "phase_rmse_deg_max", "weight_complex_nrmse_max",
    ):
        if not np.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    bounds = np.asarray(result["bounds_mm"], dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds_mm must contain three finite increasing intervals")
    result["bounds_mm"] = bounds.tolist()
    return result


def _real_matrix(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must already stack real and imaginary rows")
    result = np.asarray(raw, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real matrix with nonzero rows")
    return result


def _normalize_nuisance_columns(matrix):
    """Remove arbitrary nuisance parameter units before rank truncation."""
    peak = np.max(abs(matrix), axis=0)
    scaled = np.divide(matrix, peak, out=np.zeros_like(matrix), where=peak > 0)
    norm = np.linalg.norm(scaled, axis=0)
    return np.divide(scaled, norm, out=np.zeros_like(matrix), where=norm > 0)


def profile_geometry_singular_values(
    jacobian, geometry_columns=(0, 1, 2), nuisance_columns=None, *,
    nuisance_jacobian=None, geometry_scales_mm=(1.0, 1.0, 1.0),
    residual_kind="augmented_profile", rcond=None,
):
    """Project real geometry columns off nuisance tangents, in per-mm units.

    ``geometry_scales_mm[j]`` is d(geometry_mm[j])/d(parameter[j]); normalized
    t coordinates therefore use (10, 10, 10).  With a profile Jacobian, nuisance
    columns normally contain the remaining outer gain parameters.  For a pure
    data endpoint use fixed-b geometry columns plus *all* realified b/g tangent
    columns through ``nuisance_jacobian`` and label ``data_only_full``.  Merely
    dropping penalty rows from a ridge-profile Jacobian is ``data_profile``.
    """
    kinds = {
        "augmented_profile": "prior_assisted_profile_information",
        "data_profile": "data_rows_of_penalized_profile_not_pure_data_information",
        "data_only_full": "unpenalized_data_information_after_all_supplied_nuisances",
    }
    if residual_kind not in kinds:
        raise ValueError("unknown residual_kind")
    matrix = _real_matrix(jacobian, "jacobian")
    cols = tuple(int(i) for i in geometry_columns)
    if len(cols) != 3 or len(set(cols)) != 3 or min(cols) < 0 or max(cols) >= matrix.shape[1]:
        raise ValueError("three distinct valid geometry columns required")
    if nuisance_columns is None:
        nuisance_columns = [i for i in range(matrix.shape[1]) if i not in cols]
    other = tuple(int(i) for i in nuisance_columns)
    if len(set(other)) != len(other) or any(i < 0 or i >= matrix.shape[1] or i in cols for i in other):
        raise ValueError("nuisance columns must be distinct and disjoint from geometry")
    scales = np.asarray(geometry_scales_mm, dtype=float)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("three positive geometry scales in mm required")
    geom = matrix[:, cols] / scales
    nuisance = matrix[:, other]
    if nuisance_jacobian is not None:
        extra = _real_matrix(nuisance_jacobian, "nuisance_jacobian")
        if extra.shape[0] != len(matrix):
            raise ValueError("nuisance and geometry Jacobian rows must match")
        nuisance = np.column_stack((nuisance, extra))
    cutoff = np.finfo(float).eps * max(matrix.shape[0], nuisance.shape[1], 3) if rcond is None else float(rcond)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("rcond must be finite and positive")
    nuisance_rank = 0
    nuisance_singular = np.empty(0)
    projected = geom.copy()
    if nuisance.shape[1]:
        u, nuisance_singular, _ = np.linalg.svd(_normalize_nuisance_columns(nuisance), full_matrices=False)
        nuisance_rank = int(np.sum(nuisance_singular > cutoff * nuisance_singular[0])) if len(nuisance_singular) else 0
        q = u[:, :nuisance_rank]
        projected -= q @ (q.T @ geom)
    singular = np.linalg.svd(projected, compute_uv=False)
    singular = np.pad(singular, (0, max(0, 3 - len(singular))))
    # Reference the unprojected scale: cancellation noise is not information.
    tolerance = cutoff * max(float(np.linalg.norm(geom, ord=2)), float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    return {
        "residual_kind": residual_kind, "interpretation": kinds[residual_kind],
        "geometry_order": ["d", "dx", "dy"], "units": "residual_per_mm",
        "geometry_scales_mm": scales.tolist(), "singular_values": singular.tolist(),
        "rank": rank, "condition_number": float(singular[0] / singular[-1]) if rank == 3 else None,
        "rank_tolerance": tolerance, "rcond": cutoff,
        "nuisance_rank": nuisance_rank, "nuisance_columns": nuisance.shape[1],
        "nuisance_singular_values": nuisance_singular.tolist(),
        "nuisance_column_normalization": "unit_l2_before_rank_cutoff_zero_columns_kept",
        "row_count": matrix.shape[0],
        "prior_included": residual_kind == "augmented_profile",
        "confidence_interval": False,
    }


def data_only_geometry_singular_values(blocks, *, geometry_scales_mm=(1.0, 1.0, 1.0), rcond=None):
    """Pure-data endpoint projection, without a doubled real nuisance matrix.

    Every frequency/chain block supplies complex ``geometry_jacobian`` (m,3)
    at fixed b/g and complex ``nuisance_jacobian`` (m,q) containing all b/g
    tangents.  Each nuisance coefficient must be freely complex: a column
    stands for both its amplitude and quadrature direction.  Blocks must have
    independent nuisance variables and already include their frozen weights.
    This does not differentiate a penalized b estimate or append penalty rows.
    """
    projected, original, metadata = [], [], []
    for index, block in enumerate(blocks):
        geom = np.asarray(block["geometry_jacobian"], dtype=complex)
        nuisance = np.asarray(block["nuisance_jacobian"], dtype=complex)
        if geom.ndim != 2 or geom.shape[1] != 3 or geom.shape[0] == 0 or not np.all(np.isfinite(geom)):
            raise ValueError("each complex geometry Jacobian must have finite shape (m,3)")
        if nuisance.ndim != 2 or nuisance.shape[0] != geom.shape[0] or not np.all(np.isfinite(nuisance)):
            raise ValueError("each complex nuisance Jacobian must have matching finite rows")
        cutoff = np.finfo(float).eps * max(2 * nuisance.shape[0], 2 * nuisance.shape[1], 3) if rcond is None else float(rcond)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("rcond must be finite and positive")
        result, rank = geom.copy(), 0
        if nuisance.shape[1]:
            u, s, _ = np.linalg.svd(_normalize_nuisance_columns(nuisance), full_matrices=False)
            rank = int(np.sum(s > cutoff * s[0])) if len(s) else 0
            q = u[:, :rank]
            result -= q @ (q.conj().T @ geom)
        projected.append(result)
        original.append(geom)
        metadata.append({"name": str(block.get("name", index)), "row_count_complex": len(geom),
                         "nuisance_columns_complex": nuisance.shape[1], "nuisance_rank_complex": rank,
                         "nuisance_rank_real": 2 * rank, "rcond": cutoff})
    if not projected:
        raise ValueError("at least one independent data block required")
    complex_projected = np.vstack(projected)
    complex_original = np.vstack(original)
    real_projected = np.vstack((complex_projected.real, complex_projected.imag))
    report = profile_geometry_singular_values(real_projected, geometry_scales_mm=geometry_scales_mm,
                                              residual_kind="data_only_full", rcond=rcond)
    # Preserve the original scale when all projected directions cancel.
    original_real = np.vstack((complex_original.real, complex_original.imag)) / np.asarray(geometry_scales_mm)
    tolerance = max(report["rank_tolerance"], report["rcond"] * float(np.linalg.norm(original_real, ord=2)))
    singular = np.asarray(report["singular_values"])
    report["rank_tolerance"] = tolerance
    report["rank"] = int(np.sum(singular > tolerance))
    report["condition_number"] = float(singular[0] / singular[-1]) if report["rank"] == 3 else None
    report["nuisance_blocks"] = metadata
    report["nuisance_rank"] = sum(item["nuisance_rank_real"] for item in metadata)
    report["nuisance_columns"] = 2 * sum(item["nuisance_columns_complex"] for item in metadata)
    return report


def _finite_geometry(value):
    try:
        result = np.asarray(value, dtype=float)
        return result if result.shape == (3,) and np.all(np.isfinite(result)) else None
    except (TypeError, ValueError):
        return None


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _plain(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (complex, np.complexfloating)):
        return [_plain(value.real), _plain(value.imag)]
    return value


def _fit_binding(fit):
    fields = ("id", "fit_id", "geometry_mm", "x", "cost", "success", "status", "model", "branch", "lambda")
    summary = {name: _plain(fit[name]) for name in fields if name in fit}
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"schema_version": 1, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "summary": summary}


def fit_record_binding(fit):
    """Public identity helper for internal-validation and final-score records.

    The caller separately binds dataset and source manifests.  Volatile timing
    and counters are excluded; outer parameters, geometry, objective, numerical
    status and model/lambda identity (when present) are retained.
    """
    return _fit_binding(fit)


def _start_identity(record):
    if "start" in record:
        try:
            start = np.asarray(record["start"], dtype=float)
            if start.ndim == 1 and len(start) >= 3 and np.all(np.isfinite(start)):
                return "coordinates:" + json.dumps(start.tolist(), separators=(",", ":"))
        except (TypeError, ValueError):
            pass
    identity = record.get("start_id")
    if isinstance(identity, str) and identity:
        return "label:" + identity
    if isinstance(identity, (int, np.integer)) and not isinstance(identity, (bool, np.bool_)) and identity >= 0:
        return "label:" + str(identity)
    return None


def group_near_modes(records, coordinate_tolerance_mm=0.01,
                     relative_cost_envelope=0.05, absolute_cost_envelope=1e-12):
    """Keep all starts/failures and group converged near-optimal geometries.

    Greedy complete-link groups avoid chaining far-apart endpoints through many
    intermediate starts.  Distances use max absolute axis displacement.  The
    relative envelope applies to the supplied objective cost, not NRMSE.
    """
    if any(not np.isfinite(v) or v < 0 for v in (coordinate_tolerance_mm, relative_cost_envelope, absolute_cost_envelope)):
        raise ValueError("mode tolerances must be finite and nonnegative")
    retained, candidates = [], []
    fields = ("id", "start_id", "start", "geometry_mm", "x", "cost", "success",
              "status", "message", "nfev", "njev", "optimality", "active_mask",
              "counts", "elapsed_seconds", "failure", "exception")
    for index, record in enumerate(records):
        item = {key: _plain(record[key]) for key in fields if key in record}
        item["record_index"] = index
        geometry = _finite_geometry(record.get("geometry_mm"))
        cost = record.get("cost")
        finite_cost = not isinstance(cost, (bool, np.bool_)) and isinstance(cost, (int, float, np.number)) and np.isfinite(cost) and cost >= 0
        item["valid_geometry"] = geometry is not None
        item["valid_cost"] = bool(finite_cost)
        item["converged"] = _true(record.get("success"))
        item["start_identity"] = _start_identity(record)
        retained.append(item)
        if geometry is not None and finite_cost:
            candidates.append((float(cost), index, geometry, item["converged"]))
    converged = sorted((c for c in candidates if c[3]), key=lambda c: (c[0], c[1]))
    if not converged:
        return {"status": "no_converged_valid_start", "records": retained,
                "modes": [], "near_record_indices": [], "best_cost": None,
                "maximum_near_mode_separation_mm": None, "unresolved_near_failed_starts": [],
                "envelope_is_confidence_region": False}
    best = converged[0][0]
    limit = best + max(absolute_cost_envelope, abs(best) * relative_cost_envelope)
    near = [c for c in converged if c[0] <= limit]
    clusters = []
    for candidate in near:
        for cluster in clusters:
            if all(np.max(abs(candidate[2] - old[2])) <= coordinate_tolerance_mm for old in cluster):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    modes = [{"mode_index": i, "geometry_mm": group[0][2].tolist(),
              "cost": group[0][0], "record_indices": [c[1] for c in group],
              "maximum_within_mode_axis_span_mm": float(np.ptp(np.array([c[2] for c in group]), axis=0).max())}
             for i, group in enumerate(clusters)]
    positions = np.array([c[2] for c in near])
    identities = [item["start_identity"] for item in retained]
    unique_starts = {identity for identity in identities if identity is not None}
    unique_converged = {retained[c[1]]["start_identity"] for c in converged if retained[c[1]]["start_identity"] is not None}
    return {
        "status": "grouped", "records": retained, "modes": modes,
        "near_record_indices": [c[1] for c in near], "best_cost": best,
        "objective_cost_limit": limit,
        "relative_cost_envelope": relative_cost_envelope,
        "absolute_cost_envelope": absolute_cost_envelope,
        "coordinate_tolerance_mm": coordinate_tolerance_mm,
        "maximum_near_mode_separation_mm": float(np.ptp(positions, axis=0).max()),
        "unresolved_near_failed_starts": [c[1] for c in candidates if not c[3] and c[0] <= limit],
        "failed_record_indices": [item["record_index"] for item in retained if not item["converged"] or not item["valid_geometry"] or not item["valid_cost"]],
        "completed_start_count": len(retained), "converged_start_count": len(converged),
        "unique_start_count": len(unique_starts), "unique_converged_start_count": len(unique_converged),
        "start_identity_complete": all(identity is not None for identity in identities),
        "duplicate_start_identity": len(unique_starts) < sum(identity is not None for identity in identities),
        "envelope_is_confidence_region": False,
    }


def assess_sensitivity_admissibility(reference_geometry_mm, candidates, *, thresholds=None):
    """Separate all tested drift from competing validated explanations.

    This explicit V2 policy uses *only* converged fits, an explicit passed inner
    audit, and the frozen internal +15-degree observed shape protocol.  Geometry
    drift, boundaries, truth, sealed fields and cross-lambda training cost never
    decide admissibility.  Nonconvergence is unknown, not a successful exclusion.

    Each label maps to {fit: <record>, inner_audit_passed: bool,
    internal_validation: {protocol_id, source: 'internal_observed', noisy_nrmse,
    fit_binding}}.  The protocol id certifies one complex scalar on a predefined
    stripe and disjoint evaluation; its actual source/mask must be frozen by the
    caller.  Fit binding is the same object used in diagnostic decisions.
    """
    proposed = dict(thresholds or {})
    proposed.setdefault("sensitivity_policy_version", SENSITIVITY_POLICY_V2)
    rule = _thresholds(proposed)
    if rule.get("sensitivity_policy_version") != SENSITIVITY_POLICY_V2:
        raise ValueError("admissibility requires the explicit V2 sensitivity policy")
    reference = _finite_geometry(reference_geometry_mm)
    if reference is None:
        raise ValueError("finite reference geometry required")
    items = candidates if isinstance(candidates, Mapping) else {}
    unknowns = []
    for label in rule["required_sensitivity_labels"]:
        if label not in items:
            unknowns.append("sensitivity_label_missing:" + label)
    if not isinstance(candidates, Mapping):
        unknowns.append("sensitivity_labels_unknown")
    records, all_drifts, accepted_drifts = {}, [], []
    for label, item in items.items():
        reasons = []
        value = item if isinstance(item, Mapping) else {}
        fit = value.get("fit", value)
        fit = fit if isinstance(fit, Mapping) else {}
        geometry = _finite_geometry(fit.get("geometry_mm"))
        drift = float(np.max(abs(geometry-reference))) if geometry is not None else None
        if drift is not None:
            all_drifts.append(drift)
        if geometry is None:
            reasons.append("invalid_sensitivity_geometry")
        if not _true(fit.get("success")):
            reasons.append("sensitivity_fit_not_converged_or_unknown")
        if not _true(value.get("inner_audit_passed")):
            reasons.append("sensitivity_inner_audit_unverified")
        validation = value.get("internal_validation", {})
        validation = validation if isinstance(validation, Mapping) else {}
        if validation.get("protocol_id") != rule["internal_validation_protocol_id"] or validation.get("source") != "internal_observed":
            reasons.append("sensitivity_internal_protocol_unverified")
        if validation.get("fit_binding") != _fit_binding(fit):
            reasons.append("sensitivity_validation_fit_binding_mismatch")
        score = validation.get("noisy_nrmse")
        valid_score = (isinstance(score, (int, float, np.number)) and not isinstance(score, (bool, np.bool_))
                       and np.isfinite(score) and score >= 0)
        if not valid_score:
            reasons.append("sensitivity_internal_score_missing_or_invalid")
        if reasons:
            classification = "unknown"
            unknowns.extend(str(label) + ":" + reason for reason in reasons)
        elif score > rule["prediction_nrmse_max"]:
            classification = "excluded_by_prediction"
        else:
            classification = "admissible"
            accepted_drifts.append(drift)
        records[str(label)] = {
            "classification": classification, "geometry_mm": _plain(geometry),
            "drift_mm": drift, "internal_observed_shape_nrmse": _plain(score),
            "reasons": reasons, "fit_binding": _fit_binding(fit),
            "boundary_used_for_admissibility": False, "cost_used_for_admissibility": False,
        }
    admissible = [name for name, item in records.items() if item["classification"] == "admissible"]
    if len(admissible) < rule["minimum_admissible_sensitivity_candidates"]:
        unknowns.append("no_sufficient_admissible_sensitivity_alternative")
    admissible_drift = max(accepted_drifts) if accepted_drifts else None
    return {
        "policy_version": SENSITIVITY_POLICY_V2,
        "all_tested_drift_mm": max(all_drifts) if all_drifts else None,
        "admissible_drift_mm": admissible_drift,
        "admissible_labels": admissible,
        "excluded_by_prediction_labels": [name for name, item in records.items() if item["classification"] == "excluded_by_prediction"],
        "unknown_labels": [name for name, item in records.items() if item["classification"] == "unknown"],
        "candidates": records, "unknowns": unknowns,
        "evidence_complete": not unknowns,
        "unstable_admissible": bool(admissible_drift is not None and admissible_drift > rule["sensitivity_drift_mm_max"]),
        "thresholds": rule, "truth_used": False, "sealed_evaluation_used": False,
        "cross_lambda_objective_costs_compared": False,
        "interpretation": "conditional stability among independently admissible explanations; excluded candidates remain reported, not robust performance over every tested penalty",
    }


def assess_fit_diagnostics(fit, *, predictive_nrmse=None, modes=None,
                           sensitivity_geometries_mm=None, thresholds=None):
    """Make an observable decision, without reading any truth field in fit.

    Missing prediction/multistart/sensitivity evidence gives uncertainty.
    ``candidate`` means this finite diagnostic rule found no objection.  A
    precision claim additionally requires frozen thresholds and external gates.
    """
    rule = _thresholds(thresholds)
    reject, uncertain, unknown = [], [], []
    geometry = _finite_geometry(fit.get("geometry_mm"))
    bounds = np.asarray(rule["bounds_mm"])
    boundary = None
    if geometry is None:
        reject.append("invalid_geometry")
    else:
        if np.any((geometry < bounds[:, 0]) | (geometry > bounds[:, 1])):
            reject.append("geometry_outside_hard_bounds")
        boundary = bool(np.any(np.minimum(abs(geometry - bounds[:, 0]), abs(geometry - bounds[:, 1])) <= rule["boundary_tolerance_mm"]))
        if boundary:
            uncertain.append("hard_boundary_hit")
    if "success" not in fit:
        unknown.append("convergence_unknown")
    elif not _true(fit["success"]):
        reject.append("selected_fit_not_converged")
    if predictive_nrmse is None:
        unknown.append("prediction_diagnostic_missing")
    elif not np.isscalar(predictive_nrmse) or not np.isfinite(predictive_nrmse) or predictive_nrmse < 0:
        reject.append("invalid_prediction_diagnostic")
    elif predictive_nrmse > rule["prediction_nrmse_max"]:
        reject.append("prediction_threshold_exceeded")
    mode_separation = None
    if modes is None:
        unknown.append("multistart_modes_missing")
    elif modes.get("status") != "grouped" or not modes.get("modes"):
        uncertain.append("no_converged_valid_mode")
    else:
        mode_separation = modes.get("maximum_near_mode_separation_mm")
        if mode_separation is None or not np.isfinite(mode_separation):
            unknown.append("mode_separation_unknown")
        elif mode_separation > rule["separated_modes_mm_max"]:
            uncertain.append("separated_near_optimal_modes")
        if modes.get("unresolved_near_failed_starts"):
            uncertain.append("unresolved_near_optimal_failed_starts")
        if modes.get("unique_start_count", 0) < rule["minimum_completed_starts"]:
            unknown.append("insufficient_completed_starts")
        if modes.get("unique_converged_start_count", 0) < rule["minimum_converged_starts"]:
            unknown.append("insufficient_converged_starts")
        if not modes.get("start_identity_complete", False):
            unknown.append("start_identity_missing")
        if modes.get("duplicate_start_identity", False):
            unknown.append("duplicate_start_identity")
    drift = None
    sensitivity_report = None
    if rule.get("sensitivity_policy_version") == SENSITIVITY_POLICY_V2 and geometry is not None:
        sensitivity_report = assess_sensitivity_admissibility(geometry, sensitivity_geometries_mm, thresholds=rule)
        drift = sensitivity_report["admissible_drift_mm"]
        unknown.extend(sensitivity_report["unknowns"])
        if sensitivity_report["unstable_admissible"]:
            uncertain.append("admissible_lambda_subset_geometry_drift")
    elif sensitivity_geometries_mm is None or len(sensitivity_geometries_mm) == 0:
        unknown.append("lambda_subset_sensitivity_missing")
    else:
        if isinstance(sensitivity_geometries_mm, Mapping):
            labels = set(sensitivity_geometries_mm)
            for label in rule["required_sensitivity_labels"]:
                if label not in labels:
                    unknown.append("sensitivity_label_missing:" + label)
            values = list(sensitivity_geometries_mm.values())
        else:
            values = list(sensitivity_geometries_mm)
            if rule["required_sensitivity_labels"]:
                unknown.append("sensitivity_labels_unknown")
        alternatives = [_finite_geometry(value.get("geometry_mm") if isinstance(value, Mapping) else value) for value in values]
        if rule["require_sensitivity_convergence"] and any(not isinstance(value, Mapping) or not _true(value.get("success")) for value in values):
            unknown.append("sensitivity_refit_convergence_unverified")
        if geometry is None or any(value is None for value in alternatives):
            unknown.append("invalid_sensitivity_geometry")
        else:
            drift = float(np.max(np.abs(np.array(alternatives) - geometry)))
            if drift > rule["sensitivity_drift_mm_max"]:
                uncertain.append("lambda_subset_geometry_drift")
    eligible = not (reject or uncertain or unknown)
    return {
        "status": "rejected" if reject else "uncertain" if uncertain or unknown else "candidate",
        "would_accept_under_rule": eligible,
        "accepted_for_precision_claim": bool(eligible and rule["frozen"]),
        "external_virtual_gates_still_required": True,
        "rejection_reasons": reject, "uncertainty_reasons": uncertain, "unknowns": unknown,
        "boundary_hit": boundary, "predictive_nrmse": _plain(predictive_nrmse),
        "maximum_near_mode_separation_mm": mode_separation,
        "maximum_sensitivity_drift_mm": drift,
        "sensitivity_policy_version": rule.get("sensitivity_policy_version", "all_tested_drift_v1"),
        "sensitivity_admissibility": sensitivity_report,
        "thresholds": rule, "truth_used_for_decision": False,
        "fit_binding": _fit_binding(fit),
    }


def _scores(record):
    return record.get("final_score", record)


def _content_sha256(value):
    body = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _positive_integer(value, allow_zero=False):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= (0 if allow_zero else 1)


def validate_numerical_abstention_policy(value=None):
    """Resolve a default-off policy; this changes no geometry/field threshold."""
    policy = {"version": NUMERICAL_ABSTENTION_POLICY_V1, "enabled": False,
              "frozen": False, "require_valid_endpoint_per_family": True,
              "required_stress_families": []}
    if value is not None:
        policy.update(dict(value))
    if policy["version"] != NUMERICAL_ABSTENTION_POLICY_V1:
        raise ValueError("unknown numerical abstention policy version")
    for key in ("enabled", "frozen", "require_valid_endpoint_per_family"):
        if not isinstance(policy[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be an explicit boolean")
        policy[key] = bool(policy[key])
    families = policy["required_stress_families"]
    if not isinstance(families, list) or not all(isinstance(f, str) and f for f in families) or len(set(families)) != len(families):
        raise ValueError("unique explicit required stress families needed")
    if policy["enabled"] and policy["require_valid_endpoint_per_family"] and not families:
        raise ValueError("enabled strict abstention policy needs required stress families")
    if policy["enabled"] and policy["frozen"] and not policy["require_valid_endpoint_per_family"]:
        raise ValueError("frozen abstention policy may not remove required-family endpoint evidence")
    return policy


def _audited_inner_rejection(attempt, planned_budget):
    """Require all three complete inner diagnostics, not an exception name."""
    budget = attempt.get("inner_budget")
    diag = attempt.get("inner_diagnostics")
    if not isinstance(budget, Mapping) or budget != planned_budget or not isinstance(diag, Mapping):
        return False
    if (budget.get("n_starts") != 3 or not _positive_integer(budget.get("maxiter"))
            or not _positive_integer(budget.get("newton_steps"), allow_zero=True)
            or not _positive_integer(budget.get("random_seed"), allow_zero=True)):
        return False
    for key in ("stationarity_tol", "min_hessian_rcond", "epsilon"):
        number = budget.get(key)
        if not isinstance(number, (int,float,np.number)) or isinstance(number,(bool,np.bool_)) or not np.isfinite(number) or number <= 0:
            return False
    if budget["stationarity_tol"] > 1e-8 or budget["min_hessian_rcond"] >= 1:
        return False
    if (diag.get("success") is not False or diag.get("n_starts") != 3
            or diag.get("random_seed") != budget["random_seed"]
            or diag.get("stationarity_tol") != budget["stationarity_tol"]
            or diag.get("epsilon") != budget["epsilon"]
            or attempt.get("inner_diagnostics_sha256") != _content_sha256(diag)):
        return False
    starts = diag.get("starts")
    if not isinstance(starts,list) or len(starts)!=3 or not all(isinstance(s,Mapping) for s in starts) or [s.get("index") for s in starts] != [0,1,2]:
        return False
    for start in starts:
        if (start.get("converged") is not False or not isinstance(start.get("optimizer_success"),(bool,np.bool_))
                or not isinstance(start.get("hessian_positive_definite"),(bool,np.bool_))
                or not isinstance(start.get("optimizer_message"),str) or not start["optimizer_message"]
                or not _positive_integer(start.get("nfev")) or not _positive_integer(start.get("njev"),allow_zero=True)
                or not _positive_integer(start.get("newton_steps"),allow_zero=True)
                or start["newton_steps"]>budget["newton_steps"]):
            return False
        for key in ("objective","stationarity","hessian_rcond"):
            number=start.get(key)
            if not isinstance(number,(int,float,np.number)) or isinstance(number,(bool,np.bool_)) or not np.isfinite(number) or number<0:
                return False
        numerically_usable=(start["stationarity"]<=budget["stationarity_tol"]
                            and _true(start["hessian_positive_definite"])
                            and start["hessian_rcond"]>=budget["min_hessian_rcond"])
        # Production success is rebuilt from the numerical criteria, rather
        # than trusting a false 'converged' label on an otherwise usable start.
        if numerically_usable:
            return False
    chosen = diag.get("chosen_start")
    if not _positive_integer(chosen,allow_zero=True) or chosen not in (0,1,2):
        return False
    selected=next(s for s in starts if s["index"]==chosen)
    if chosen!=min(range(3),key=lambda i:(starts[i]["objective"],i)):
        return False
    if (diag.get("stationarity")!=selected["stationarity"]
            or diag.get("hessian_rcond")!=selected["hessian_rcond"]
            or diag.get("hessian_positive_definite")!=selected["hessian_positive_definite"]):
        return False
    reason=attempt.get("reason_code")
    if reason=="amplitude_inner_stationarity_rejected":
        return selected["stationarity"]>budget["stationarity_tol"]
    if reason=="amplitude_inner_hessian_rejected":
        return selected["hessian_positive_definite"] is False or selected["hessian_rcond"]<budget["min_hessian_rcond"]
    return False


def audit_completed_numerical_abstention(record, *, policy=None):
    """Audit a pre-truth terminal lock already verified against actual files.

    This module is I/O free.  The runner's global barrier must independently
    verify every contract, terminal and reservation ledger before exporting the
    lock.  Matching hashes here bind that verified evidence; a same-named raised
    exception or an asserted counter by itself is never sufficient.
    """
    rule=validate_numerical_abstention_policy(policy)
    failures=[]
    if not rule["enabled"] or not rule["frozen"]:
        failures.append("numerical_abstention_policy_disabled_or_unfrozen")
    decision=record.get("diagnostic_decision",{})
    if not isinstance(decision,Mapping):decision={}
    lock=record.get("numerical_abstention_lock")
    fit=record.get("fit",{})
    if not isinstance(fit,Mapping):fit={}
    if (record.get("selected_endpoint_available") is not False
            or fit.get("success") is not False
            or _finite_geometry(fit.get("geometry_mm")) is not None
            or fit_record_binding(fit)!=record.get("fit_binding")):
        failures.append("no_estimate_status_unverified")
    if (decision.get("would_accept_under_rule") is not False or decision.get("truth_used_for_decision") is not False
            or record.get("truth_used_only_after_global_locks") is not True):
        failures.append("observable_rejection_not_frozen_before_truth")
    if not isinstance(lock,Mapping):
        return {"classification":"unknown", "reasons":failures+["numerical_abstention_lock_missing"],"policy":rule}
    if (lock.get("schema_version")!=1 or lock.get("status")!="completed_numerical_abstention"
            or lock.get("all_planned_trajectories_terminal") is not True
            or lock.get("observable_decision_frozen_before_truth") is not True
            or lock.get("decision_fit_binding")!=decision.get("fit_binding")
            or record.get("fit_binding")!=decision.get("fit_binding")):
        failures.append("abstention_lock_status_or_decision_binding_invalid")
    for name in ("freeze_sha256","split_sha256","attempt_seal_sha256"):
        if not _digest_string(record.get(name)) or lock.get(name)!=record.get(name):
            failures.append("abstention_binding_invalid:"+name)
    for name in ("source_sha256","dataset_sha256"):
        expected=record.get(name)
        if not isinstance(expected,Mapping) or not expected or not all(_digest_string(v) for v in expected.values()) or lock.get(name)!=expected:
            failures.append("abstention_binding_invalid:"+name)
    dataset=record.get("dataset_sha256",{})
    if not isinstance(dataset,Mapping) or not {"fit_data.npz","manifest.json","truth_for_scoring_only.npz"} <= set(dataset):
        failures.append("complete_dataset_hash_binding_missing")
    lock_hash=_content_sha256(lock)
    if (record.get("numerical_abstention_lock_sha256")!=lock_hash
            or record.get("score_barrier_abstention_lock_sha256")!=lock_hash
            or not _digest_string(record.get("score_barrier_sha256"))
            or lock.get("observable_lock_sha256")!=_content_sha256(decision)):
        failures.append("pretruth_lock_or_score_barrier_binding_invalid")
    plan=lock.get("attempt_plan",{})
    if not isinstance(plan,Mapping):plan={}
    caps=plan.get("profile_evaluation_caps",{})
    contracts=plan.get("trajectory_contract_sha256",{})
    inner_budgets=plan.get("inner_budgets",{})
    attempts=lock.get("attempts",[])
    if (not isinstance(caps,Mapping) or not caps or not all(isinstance(k,str) and k and _positive_integer(v) for k,v in caps.items())
            or not isinstance(contracts,Mapping) or set(contracts)!=set(caps) or not all(_digest_string(v) for v in contracts.values())
            or not isinstance(inner_budgets,Mapping) or set(inner_budgets)!=set(caps)
            or not isinstance(attempts,list) or len(attempts)!=len(caps)
            or not all(isinstance(a,Mapping) for a in attempts)
            or {a.get("trajectory_id") for a in attempts}!=set(caps)):
        failures.append("planned_terminal_matrix_incomplete_or_duplicate")
        attempts=[]
    for attempt in attempts:
        identity=attempt["trajectory_id"]
        if (attempt.get("status")!="completed_no_estimate" or attempt.get("started") is not True
                or attempt.get("contract_sha256")!=contracts[identity]
                or not _digest_string(attempt.get("terminal_sha256"))
                or not _digest_string(attempt.get("budget_ledger_sha256"))
                or attempt.get("budget_ledger_verified") is not True
                or not _positive_integer(attempt.get("reserved_profile_evaluations"))
                or attempt["reserved_profile_evaluations"]>caps[identity]):
            failures.append("attempt_terminal_or_budget_evidence_invalid:"+identity)
            continue
        reason, exception=attempt.get("reason_code"),attempt.get("exception_type")
        if reason=="profile_evaluation_budget_exhausted" and exception=="EvaluationBudgetExhausted":
            if (attempt["reserved_profile_evaluations"]!=caps[identity]
                    or attempt.get("budget_exhausted_after_completed_reservations") is not True):
                failures.append("budget_exhaustion_not_proven:"+identity)
        elif exception=="AmplitudeProfileError" and reason in ("amplitude_inner_stationarity_rejected","amplitude_inner_hessian_rejected"):
            if not _audited_inner_rejection(attempt,inner_budgets[identity]):
                failures.append("complete_three_start_inner_rejection_unverified:"+identity)
        else:
            failures.append("unsupported_or_unexpected_exception:"+identity)
    return {"classification":"unknown" if failures else "completed_numerical_abstention",
            "reasons":failures,"policy":rule,"attempt_count":len(attempts),
            "physical_wrong_model_identification_claim":False,"truth_used":False,
            "file_evidence_requires_independent_runner_barrier_verification":True}


def _full_array(record):
    flags = [record[name] for name in ("full_array", "full_1024") if name in record]
    counts = [record[name] for name in ("element_count", "n_elements") if name in record]
    if any(not _true(flag) for flag in flags) or any(count != 1024 for count in counts):
        return False
    return True if flags or counts else None


def _valid_stress_endpoint_record(record):
    """Require a real selected terminal, not a placeholder finite error."""
    fit=record.get("fit")
    decision=record.get("diagnostic_decision",{})
    if not isinstance(fit,Mapping) or not isinstance(decision,Mapping):return False
    binding=fit_record_binding(fit)
    return bool(record.get("selected_endpoint_available") is True
        and record.get("endpoint_binding_verified_before_truth") is True
        and _digest_string(record.get("selected_terminal_sha256"))
        and _digest_string(fit.get("fit_id"))
        and _finite_geometry(fit.get("geometry_mm")) is not None
        and fit.get("success") is True and fit.get("inner_audit_passed") is True
        and record.get("truth_used_only_after_global_locks") is True
        and record.get("fit_binding")==binding and decision.get("fit_binding")==binding
        and _scores(record).get("fit_binding")==binding)


def audit_virtual_gates(records, *, thresholds=None, confirmation_count=12,
                        prerequisites=None, require_weights=False,
                        expected_confirmation_manifest=None, stress_abstention_policy=None):
    """Audit frozen synthetic truth scores; this never changes fit decisions.

    Each record has kind='confirmation'|'stress'|'ideal', full_array=True,
    final_score with geometry_error_mm/unseen_angles/weights, and
    diagnostic_decision from assess_fit_diagnostics.  The record and score each
    retain the decision's fit_binding.  A frozen manifest maps unique case ids
    to seed, snr_db and unseen_angle_ids; it is not inferred from scores.
    Paired noise levels share
    seeds: the empirical quantile is not a population coverage estimate.
    """
    if isinstance(confirmation_count, (bool, np.bool_)) or int(confirmation_count) != confirmation_count or confirmation_count <= 0:
        raise ValueError("confirmation_count must be a positive integer")
    rule = _thresholds(thresholds)
    items = list(records)
    main = [r for r in items if r.get("kind") == "confirmation"]
    stress = [r for r in items if r.get("kind") == "stress"]
    ideal = [r for r in items if r.get("kind") == "ideal"]
    gates, missing = {}, []
    manifest = expected_confirmation_manifest
    case_ids = [r.get("id") for r in main]
    if manifest is None:
        gates["confirmation_manifest_binding"] = None
        missing.append("frozen_confirmation_manifest")
    else:
        identities_valid = (len(case_ids) == len(set(str(i) for i in case_ids))
                            and all(isinstance(i, str) and i for i in case_ids)
                            and set(case_ids) == set(manifest)
                            and len(manifest) == confirmation_count)
        for record in main:
            expected = manifest.get(record.get("id"), {})
            angles = expected.get("unseen_angle_ids")
            seed, snr = expected.get("seed"), expected.get("snr_db")
            valid_seed = isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0
            valid_snr = isinstance(snr, (int, float, np.number)) and not isinstance(snr, (bool, np.bool_)) and np.isfinite(snr)
            identities_valid = identities_valid and bool(
                "seed" in record and "snr_db" in record and valid_seed and valid_snr
                and record.get("seed") == expected["seed"]
                and record.get("snr_db") == expected["snr_db"]
                and isinstance(angles, (list, tuple)) and len(angles) > 0
                and all(isinstance(angle, str) and angle for angle in angles)
                and len(set(angles)) == len(angles)
                and set(_scores(record).get("unseen_angles", {})) == set(angles))
        gates["confirmation_manifest_binding"] = bool(identities_valid)
    errors = [_finite_geometry(_scores(r).get("geometry_error_mm")) for r in main]
    valid_errors = len(main) == confirmation_count and all(e is not None for e in errors)
    p90 = maximum = None
    if valid_errors:
        absolute = abs(np.array(errors))
        p90 = np.quantile(absolute, 0.9, axis=0)
        maximum = absolute.max(axis=0)
        gates["confirmation_geometry"] = bool(np.all(p90 <= rule["confirmation_axis_p90_mm_max"]) and np.all(maximum <= rule["confirmation_axis_error_mm_max"]))
    else:
        gates["confirmation_geometry"] = None
        missing.append("complete_confirmation_geometry_scores")
    main_decisions = [r.get("diagnostic_decision", {}) for r in main]
    if main and all(d.get("truth_used_for_decision") is False and isinstance(d.get("would_accept_under_rule"), (bool, np.bool_)) for d in main_decisions):
        gates["confirmation_observable_acceptance"] = all(_true(d["would_accept_under_rule"]) for d in main_decisions)
    else:
        gates["confirmation_observable_acceptance"] = None
        missing.append("confirmation_observable_decisions")
    angle_errors = []
    angles_complete = len(main) == confirmation_count
    for record in main:
        angles = _scores(record).get("unseen_angles", {})
        if not angles:
            angles_complete = False
        for value in angles.values():
            clean = value.get("clean_nrmse")
            if clean is None or not np.isfinite(clean) or clean < 0:
                angles_complete = False
            else:
                angle_errors.append(float(clean))
    gates["unseen_angle_clean_prediction"] = bool(max(angle_errors) <= rule["unseen_angle_clean_nrmse_max"]) if angles_complete and angle_errors else None
    if gates["unseen_angle_clean_prediction"] is None:
        missing.append("complete_unseen_angle_clean_scores")
    array_flags = [_full_array(r) for r in items]
    gates["complete_1024_columns"] = bool(all(array_flags)) if items and all(v is not None for v in array_flags) else None
    if gates["complete_1024_columns"] is None:
        missing.append("explicit_full_array_provenance")
    abstention_policy=validate_numerical_abstention_policy(stress_abstention_policy)
    if abstention_policy["enabled"]:
        gates["numerical_abstention_policy_frozen"]=abstention_policy["frozen"]
    false_acceptances, stress_unknown, numerical_abstentions = [], [], []
    abstention_audits, endpoint_families = {}, set()
    for index, record in enumerate(stress):
        error = _finite_geometry(_scores(record).get("geometry_error_mm"))
        decision = record.get("diagnostic_decision", {})
        accepted = decision.get("would_accept_under_rule")
        if error is not None and abstention_policy["enabled"] and not _valid_stress_endpoint_record(record):
            # Another valid endpoint in the same family cannot hide a software
            # failure carrying a fabricated finite placeholder score.
            stress_unknown.append(record.get("id",index))
        elif error is None and accepted is False and abstention_policy["enabled"]:
            abstention=audit_completed_numerical_abstention(record,policy=abstention_policy)
            identifier=record.get("id",index)
            abstention_audits[str(identifier)]=abstention
            if abstention["classification"]=="completed_numerical_abstention":
                numerical_abstentions.append(identifier)
            else:
                stress_unknown.append(identifier)
        elif error is None or not isinstance(accepted, (bool, np.bool_)) or decision.get("truth_used_for_decision") is not False:
            stress_unknown.append(record.get("id", index))
        elif np.max(abs(error)) > rule["confirmation_axis_error_mm_max"] and _true(accepted):
            false_acceptances.append(record.get("id", index))
            if isinstance(record.get("stress_family"),str) and _valid_stress_endpoint_record(record):endpoint_families.add(record["stress_family"])
        elif isinstance(record.get("stress_family"),str) and _valid_stress_endpoint_record(record):
            endpoint_families.add(record["stress_family"])
    gates["stress_no_false_precision_acceptance"] = False if false_acceptances else True if stress and not stress_unknown else None
    if gates["stress_no_false_precision_acceptance"] is None:
        missing.append("stress_truth_and_observable_decisions")
    if abstention_policy["enabled"] and abstention_policy["require_valid_endpoint_per_family"]:
        family_labels=[r.get("stress_family") for r in stress]
        gates["stress_family_labels_complete"]=all(isinstance(f,str) and bool(f) for f in family_labels)
        # A newly included family cannot evade coverage by omission from the
        # supplied required list. Formal runner separately binds its manifest.
        required_families=set(abstention_policy["required_stress_families"])|{f for f in family_labels if isinstance(f,str) and f}
        absent=required_families-endpoint_families
        gates["stress_valid_endpoint_per_required_family"]=not absent
        if absent:missing.extend("stress_family_needs_valid_endpoint:"+family for family in sorted(absent))
    decisions = main_decisions + [r.get("diagnostic_decision", {}) for r in stress]
    gates["observable_rule_binding"] = bool(decisions and all(_plain(d.get("thresholds")) == _plain(rule) for d in decisions))
    bindings_valid = bool(main or stress or ideal)
    for record in main + stress + ideal:
        decision = record.get("diagnostic_decision", {})
        binding = decision.get("fit_binding")
        bindings_valid = bindings_valid and bool(binding and record.get("fit_binding") == binding
                                                 and _scores(record).get("fit_binding") == binding)
        if "fit" in record:
            bindings_valid = bindings_valid and _fit_binding(record["fit"]) == binding
    gates["fit_decision_score_binding"] = bool(bindings_valid)
    ideal_errors = [_finite_geometry(_scores(r).get("geometry_error_mm")) for r in ideal]
    if ideal and all(e is not None for e in ideal_errors):
        gates["ideal_geometry"] = bool(np.max(abs(np.array(ideal_errors))) <= rule["ideal_axis_error_mm_max"])
        ideal_consistency = []
        for record in ideal:
            difference = record.get("same_mode_multistart_difference_mm")
            consistent = (difference is not None and np.isfinite(difference) and 0 <= difference <= rule["same_mode_multistart_difference_mm_max"])
            fit_success = record.get("diagnostic_decision", {}).get("fit_binding", {}).get("summary", {}).get("success")
            ideal_consistency.append(bool(consistent and _true(fit_success) and _true(record.get("identifiable_positive_control"))))
        gates["ideal_convergence_and_multistart"] = all(ideal_consistency)
    else:
        gates["ideal_geometry"] = None
        gates["ideal_convergence_and_multistart"] = None
        missing.append("ideal_identifiable_geometry_scores")
    weight_failures, weights_missing = [], []
    if require_weights:
        for index, record in enumerate(main):
            weights = _scores(record).get("weights", {})
            if not weights:
                weights_missing.append(record.get("id", index))
            for group, score in weights.items():
                for key, limit in (("amplitude_rmse_db", "amplitude_rmse_db_max"), ("phase_rmse_deg", "phase_rmse_deg_max"), ("complex_nrmse", "weight_complex_nrmse_max")):
                    value = score.get(key)
                    if value is None or not np.isfinite(value) or value < 0:
                        weights_missing.append([record.get("id", index), group, key])
                    elif value > rule[limit]:
                        weight_failures.append([record.get("id", index), group, key, float(value)])
        gates["claimed_weight_recovery"] = not weight_failures if not weights_missing and main else None
    required = ("v0_passed", "v1_completed", "confirmation_frozen_before_scores",
                "available_information_confirmed", "stress_classes_complete",
                "stability_checks_complete")
    prereq = dict(prerequisites or {})
    for name in required:
        gates[name] = _true(prereq[name]) if name in prereq else None
        if name not in prereq:
            missing.append(name)
    gates["thresholds_frozen"] = bool(rule["frozen"])
    return {
        "status": "passed" if all(v is True for v in gates.values()) else "failed" if any(v is False for v in gates.values()) else "incomplete",
        "real_fit_gate_open": bool(all(v is True for v in gates.values())),
        "gates": gates, "missing_evidence": missing,
        "confirmation_count": len(main), "expected_confirmation_count": confirmation_count,
        "axis_absolute_error_p90_mm": _plain(p90), "axis_absolute_error_max_mm": _plain(maximum),
        "maximum_unseen_angle_clean_nrmse": max(angle_errors) if angle_errors else None,
        "stress_count": len(stress), "false_precision_acceptances": false_acceptances,
        "stress_unknowns": stress_unknown, "weight_failures": weight_failures,
        "completed_numerical_abstentions":numerical_abstentions,
        "numerical_abstention_audits":abstention_audits,
        "numerical_abstention_policy":abstention_policy,
        "weight_scores_missing": weights_missing,
        "thresholds": rule, "truth_used_for_audit_only": True,
        "population_reliability_or_confidence_coverage_claim": False,
    }
