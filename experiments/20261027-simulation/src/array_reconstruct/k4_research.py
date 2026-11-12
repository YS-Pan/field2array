"""Bounded, policy-driven K4 virtual research using the preserved ridge solver.

This entry point keeps development and confirmation labels explicit.  It never
changes the historical runners, and coefficient truth enters scoring only.
Budgets are cooperative: a running NumPy/SciPy operation is not interrupted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import sys
from time import perf_counter
import warnings

import numpy as np
import scipy

from .large_error_validation import cached_truth_matrix, sample_excitation, score_estimate
from .physics import green_matrix
from .simulation import C0, simulate_measurement
from .thesis_coded import (NOISE_NAMESPACE, STATE_ERROR_NAMESPACE, code_patterns,
                           coded_observations, noise_unit_samples,
                           prepare_coded_ridge, solve_coded_ridge, state_error_factors)
from .validation import array_geometry, scan_geometry


PLANES = ("training", "validation", "test")
POINT_KEYS = dict(zip(PLANES, ("samples_m", "validation_samples_m", "test_samples_m")))
BASELINE_LIMITS = {"active_amplitude_rmse_db": .5, "active_phase_rmse_deg": 5., "complex_nrmse": .1}
LEGACY_POLICY = {"independent_model_limits": {**BASELINE_LIMITS, "fault_precision": .95,
                 "fault_recall": .95, "healthy_false_positive_rate": .01}}
PHASE_MASK_RATIO = .1


def _finite_number(value, name, *, minimum=0., strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} is out of range")
    return float(value)


def _unique_integers(values, name):
    if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{name} must be a list of nonnegative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def validate_policy(policy):
    """Validate the execution contract before creating output or loading data.

    The policy is self contained: geometry/truth/scenario use the historical
    field names; new fields are stage, frequencies_hz, seeds, fault_states,
    state_error_regimes, alphas, max_cases, and max_total_seconds.  Confirmation
    additionally requires frozen=true and an explicit excluded_seeds list.
    Optional named targets use the same three coefficient metric keys.
    """
    if policy.get("schema_version") != 1 or policy.get("synthetic_only") is not True:
        raise ValueError("schema_version=1 and synthetic_only=true are required")
    if policy.get("stage") not in ("development", "confirmation"):
        raise ValueError("stage must be development or confirmation")
    seeds = _unique_integers(policy["seeds"], "seeds")
    excluded = _unique_integers(policy.get("excluded_seeds", []), "excluded_seeds")
    if not seeds or set(seeds) & set(excluded):
        raise ValueError("Run seeds must be nonempty and disjoint from excluded_seeds")
    if policy["stage"] == "confirmation" and (policy.get("frozen") is not True or "excluded_seeds" not in policy):
        raise ValueError("Confirmation requires frozen=true and explicit excluded_seeds")
    frequencies = policy["frequencies_hz"]
    if not isinstance(frequencies, list) or not frequencies or len(frequencies) != len(set(frequencies)):
        raise ValueError("frequencies_hz must be nonempty and unique")
    for value in frequencies:
        _finite_number(value, "frequency", strict=True)
    faults = policy["fault_states"]
    if not isinstance(faults, list) or not faults or any(type(x) is not bool for x in faults) or len(faults) != len(set(faults)):
        raise ValueError("fault_states must be a nonempty unique list of booleans")
    g = policy["geometry"]
    for key in ("array_side", "scan_side"):
        if type(g[key]) is not int or g[key] < (3 if key == "array_side" else 2):
            raise ValueError(f"geometry.{key} is too small or is not an integer")
    for key in ("pitch_m", "scan_step_m", "scan_distance_m"):
        _finite_number(g[key], "geometry." + key, strict=True)
    if not np.isclose(g["row_offset_m"], g["pitch_m"] / 2, rtol=0, atol=1e-15):
        raise ValueError("Only the preserved half-pitch staggered array is supported")
    for key in ("validation_x_offset_m", "test_y_offset_m", "test_z_offset_m"):
        _finite_number(g[key], "geometry." + key, minimum=-1e100)
    if g["scan_distance_m"] + g["test_z_offset_m"] <= 0:
        raise ValueError("Test plane must have positive distance")
    truth = policy["truth"]
    _finite_number(truth["source_aperture_wavelength_fraction"], "source aperture", strict=True)
    if type(truth["quadrature_order"]) is not int or truth["quadrature_order"] < 1:
        raise ValueError("Positive integer quadrature_order required")
    if truth.get("coupling", "identity") != "identity" or truth.get("common_gain", [1., 0.]) != [1., 0.]:
        raise ValueError("Current runner requires identity coupling and common gain 1")
    if any(truth.get(key, 0.) != 0 for key in ("directionality_power", "probe_aperture_m")):
        raise ValueError("Current truth operator supports isotropic sources and a point probe")
    scenario = policy["scenario"]
    if scenario["amplitude_distribution"] not in ("uniform", "gaussian"):
        raise ValueError("Unsupported amplitude_distribution")
    key = "amplitude_half_range_db" if scenario["amplitude_distribution"] == "uniform" else "amplitude_std_db"
    _finite_number(scenario[key], key)
    _finite_number(scenario["phase_std_deg"], "scenario phase_std_deg")
    _finite_number(policy["initial_snr_db"], "initial_snr_db", minimum=-300.)
    alphas = policy["alphas"]
    if not isinstance(alphas, list) or not alphas or len(alphas) != len(set(alphas)):
        raise ValueError("alphas must be nonempty and unique")
    for value in alphas:
        _finite_number(value, "alpha")
    regimes = policy["state_error_regimes"]
    if not isinstance(regimes, list) or not regimes:
        raise ValueError("state_error_regimes must be nonempty")
    names = []
    for regime in regimes:
        name = regime["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Regime names must contain only letters, digits, _ or -")
        names.append(name)
        for key in ("amplitude_std_db", "phase_std_deg"):
            _finite_number(regime[key], "regime " + key)
        if regime.get("domain", "declared") not in ("declared", "stress"):
            raise ValueError("Regime domain must be declared or stress")
    if len(names) != len(set(names)):
        raise ValueError("Regime names must be unique")
    for name, limits in policy.get("targets", {}).items():
        if set(limits) != set(BASELINE_LIMITS):
            raise ValueError("Every target must specify exactly the three coefficient limits")
        for value in limits.values():
            _finite_number(value, "target limit", strict=True)
        if name == "baseline" and limits != BASELINE_LIMITS:
            raise ValueError("The original baseline limits cannot be changed")
    for key in ("phase_mask_ratio", "phase_min_relative_amplitude"):
        if policy.get(key, PHASE_MASK_RATIO) != PHASE_MASK_RATIO:
            raise ValueError("The preserved scoring phase mask is fixed at 0.1")
    if type(policy["max_cases"]) is not int or policy["max_cases"] < 1:
        raise ValueError("Positive integer max_cases required")
    _finite_number(policy["max_total_seconds"], "max_total_seconds", strict=True)
    if "max_case_seconds" in policy:
        _finite_number(policy["max_case_seconds"], "max_case_seconds", strict=True)
    return policy


def build_case_matrix(policy):
    """Enumerate the full denominator in advance, including stress cases."""
    validate_policy(policy)
    cases = []
    for frequency in policy["frequencies_hz"]:
        for seed in policy["seeds"]:
            for faults in policy["fault_states"]:
                pair = f"f{frequency:.12g}_seed{seed}"
                for regime in policy["state_error_regimes"]:
                    name = pair + ("_faulted_" if faults else "_healthy_") + regime["name"]
                    cases.append({"name": name, "pair_id": pair, "frequency_hz": frequency,
                                  "seed": seed, "with_faults": faults, "regime": dict(regime),
                                  "domain": regime.get("domain", "declared"), "status": "pending"})
    if len(cases) > policy["max_cases"]:
        raise ValueError(f"Matrix has {len(cases)} cases, exceeding max_cases={policy['max_cases']}")
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Frequency formatting produced duplicate case names")
    return cases


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def _geometry(policy):
    g = policy["geometry"]
    elements = array_geometry(g["array_side"], g["pitch_m"])
    train = scan_geometry(g["scan_side"], g["scan_step_m"], g["scan_distance_m"])
    validation = train[::3].copy()
    validation[:, 0] += g["validation_x_offset_m"]
    test = train[1::3].copy()
    test[:, 1] += g["test_y_offset_m"]
    test[:, 2] += g["test_z_offset_m"]
    return elements, dict(zip(PLANES, (train, validation, test)))


def _verify_operators(elements, points, matrices, truth_matrices, frequency, policy):
    """Check every operator row against independent physical calculations.

    A full point matrix comparison and a full-aperture independent field probe
    validate legacy caches against the current policy rather than trusting NPZ
    metadata.  The fixed verification excitation is never a research case.
    """
    rng = np.random.default_rng(918273)
    probe = (.7 + .6 * rng.random(len(elements))) * np.exp(.8j * rng.normal(size=len(elements)))
    checks = {}
    truth = policy["truth"]
    for plane in PLANES:
        shape = (len(points[plane]), len(elements))
        for operator in (matrices[plane], truth_matrices[plane]):
            if operator.shape != shape or not np.all(np.isfinite(operator)):
                raise ValueError(f"Invalid {plane} operator shape or nonfinite values")
        point_error = _relative(matrices[plane], green_matrix(elements, points[plane], frequency))
        original = simulate_measurement(elements, probe, frequency, points[plane], config={
            "source_aperture_m": truth["source_aperture_wavelength_fraction"] * C0 / frequency,
            "source_quadrature_order": truth["quadrature_order"], "directionality_power": 0.,
            "snr_db": None, "seed": 13579})
        truth_error = _relative(truth_matrices[plane] @ probe, original["field_clean"])
        if max(point_error, truth_error) > 5e-13:
            raise ValueError(f"{plane} operator disagrees with the policy: {point_error=}, {truth_error=}")
        checks[plane] = {"point_relative_error": point_error, "truth_probe_relative_error": truth_error,
                         "tolerance": 5e-13, "passed": True}
    return checks


def _operator_inputs(paths, frequencies):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    result = {}
    for value in paths:
        path = Path(value).resolve()
        with np.load(path, allow_pickle=False) as source:
            if str(source["provenance"]) != "synthetic":
                raise ValueError("Only synthetic operator inputs are accepted")
            frequency = float(source["frequency_hz"])
        if frequency not in frequencies or frequency in result:
            raise ValueError("Operator inputs must uniquely match requested frequencies")
        result[frequency] = path
    return result


def _operators(path, frequency, policy, output, config_hash):
    elements, points = _geometry(policy)
    generated = path is None
    if generated:
        matrices = {plane: green_matrix(elements, samples, frequency) for plane, samples in points.items()}
        truth = policy["truth"]
        truth_matrices = {plane: cached_truth_matrix(elements, samples, frequency,
                          truth["source_aperture_wavelength_fraction"], truth["quadrature_order"])
                          for plane, samples in points.items()}
    else:
        with np.load(path, allow_pickle=False) as source:
            if float(source["frequency_hz"]) != frequency or str(source["provenance"]) != "synthetic":
                raise ValueError("Operator frequency/provenance mismatch")
            for key, expected in [("elements_m", elements), *[(POINT_KEYS[p], points[p]) for p in PLANES]]:
                actual = source[key]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-14):
                    raise ValueError(f"Operator geometry differs from policy: {key}")
            matrices = {plane: source["point_" + plane] for plane in PLANES}
            truth_matrices = {plane: source["truth_" + plane] for plane in PLANES}
    verification = _verify_operators(elements, points, matrices, truth_matrices, frequency, policy)
    if generated:
        path = output / f"operators_{frequency:.12g}Hz.npz"
        np.savez(path, provenance="synthetic", config_sha256=config_hash, frequency_hz=frequency,
                 elements_m=elements, **{POINT_KEYS[p]: points[p] for p in PLANES},
                 **{"point_" + p: matrices[p] for p in PLANES},
                 **{"truth_" + p: truth_matrices[p] for p in PLANES})
    return elements, points, matrices, truth_matrices, {"path": str(path), "sha256": _hash(path),
           "generated": generated, "verification": verification, "frequency_hz": frequency}


def _relative(estimate, reference):
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-300))


def _raw_metrics(estimate, truth):
    threshold = PHASE_MASK_RATIO * np.max(np.abs(truth))
    active = (np.abs(truth) > 0) & (np.abs(truth) >= threshold)
    phase_active = active & (np.abs(estimate) > 0) & (np.abs(estimate) >= threshold)
    amp = 20 * np.log10(np.maximum(np.abs(estimate[active]), 1e-15) / np.abs(truth[active]))
    phase = np.rad2deg(np.angle(estimate[phase_active] * np.conj(truth[phase_active])))
    metrics = {"active_amplitude_rmse_db": float(np.sqrt(np.mean(amp ** 2))) if np.any(active) else None,
               "active_phase_rmse_deg": float(np.sqrt(np.mean(phase ** 2))) if np.any(phase_active) else None,
               "complex_nrmse": _relative(estimate, truth)}
    masks = {"truth_active_count": int(np.sum(active)), "truth_inactive_count": int(np.sum(~active)),
             "phase_scored_count": int(np.sum(phase_active)),
             "truth_active_but_estimate_low_count": int(np.sum(active & ~phase_active)),
             "phase_unscored_count": int(np.sum(~phase_active)),
             "absolute_amplitude_threshold": float(threshold)}
    return metrics, masks, active, phase_active


def _passes(metrics, limits):
    return all(metrics.get(key) is not None and np.isfinite(metrics[key]) and metrics[key] <= limit
               for key, limit in limits.items())


def _case(case, policy, prepared, points, matrices, truth_matrices, elements, output, config_hash):
    started = perf_counter()
    seed, regime = case["seed"], case["regime"]
    codes = prepared.codes
    excitation = sample_excitation(len(elements), policy["scenario"], seed, case["with_faults"])
    errors, amp, phase = state_error_factors(seed, len(elements), regime["amplitude_std_db"], regime["phase_std_deg"])
    clean = {plane: matrix @ excitation["truth"] for plane, matrix in truth_matrices.items()}
    sigmas = {plane: float(np.sqrt(np.mean(np.abs(field) ** 2)) * 10 ** (-policy["initial_snr_db"] / 20))
              for plane, field in clean.items()}
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas.values()):
        raise ValueError("Fixed-noise reference must have positive finite RMS")
    unit_noise = {plane: noise_unit_samples(seed, i, (0, 1, 2, 3), len(points[plane]))
                  for i, plane in enumerate(PLANES)}
    observations = {plane: coded_observations(truth_matrices[plane], excitation["truth"], codes,
                    unit_noise[plane], sigmas[plane], errors) for plane in PLANES}
    t_fit = perf_counter()
    fit = solve_coded_ridge(prepared, observations["training"]["field"], matrices["validation"],
                            observations["validation"]["field"], policy["alphas"])
    fit_seconds = perf_counter() - t_fit
    if not np.all(np.isfinite(fit["coefficients"])):
        raise ValueError("Solver returned nonfinite coefficients")
    # Only now expose truth and fault labels to coefficient/detection scoring.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, aligned, flagged, oracle_flags = score_estimate(fit["coefficients"], excitation["truth"],
                                                               excitation["nominal"], excitation["fault_indices"], LEGACY_POLICY)
    raw, raw_masks, _, raw_phase_mask = _raw_metrics(fit["coefficients"], excitation["truth"])
    _, masks, active, phase_mask = _raw_metrics(aligned, excitation["truth"])
    prediction = matrices["test"] @ fit["coefficients"]
    targets = {"baseline": BASELINE_LIMITS, **policy.get("targets", {})}
    passed = {name: _passes(scores["metrics"], limits) for name, limits in targets.items()}
    result = {**case, **scores, "status": "completed", "target_passed": passed,
              "unaligned_metrics": raw, "aligned_mask_counts": masks, "unaligned_mask_counts": raw_masks,
              "selected_alpha": fit["selected_alpha"], "selection_method": fit["selection_method"],
              "selection_field_nrmse": fit["selection_field_nrmse"], "training_field_nrmse": fit["training_field_nrmse"],
              "regularized_effective_rank": fit["regularized_effective_rank"],
              "normalized_lambda_absolute": fit["normalized_lambda_absolute"],
              "training_common_gain": [fit["gain"].real, fit["gain"].imag],
              "noise_sigmas": sigmas, "physical_fault_indices": excitation["fault_indices"].tolist(),
              "paired_fault_indices": excitation["paired_fault_indices"].tolist(),
              "measurement_states": 4, "observation_counts": {p: 4 * len(points[p]) for p in PLANES},
              "truth_used_for_selection": False, "test_used_for_selection": False,
              "independent_test_clean_field_nrmse": _relative(prediction, clean["test"]),
              "independent_test_noisy_field_nrmse": _relative(prediction, observations["test"]["field"][0]),
              "fit_seconds": fit_seconds}
    arrays = {"provenance": "synthetic", "config_sha256": config_hash, "frequency_hz": case["frequency_hz"],
              "elements_m": elements, "canonical_codes": codes, "state_error_factors": errors,
              "state_error_amplitude_db": amp, "state_error_phase_deg": phase,
              "true_coefficients": excitation["truth"], "healthy_coefficients": excitation["healthy"],
              "normal_amplitude_error_db": excitation["amp_error_db"], "normal_phase_error_deg": excitation["phase_error_deg"],
              "injected_fault_indices": excitation["fault_indices"], "paired_fault_indices": excitation["paired_fault_indices"],
              "fault_multipliers": excitation["fault_multipliers"], "aligned_coefficients": aligned,
              "flagged_mask": flagged, "oracle_flag_mask": oracle_flags, "amplitude_score_mask": active,
              "aligned_phase_score_mask": phase_mask, "unaligned_phase_score_mask": raw_phase_mask,
              "uncoded_test_prediction": prediction,
              **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)}}
    result["state_snr"] = {}
    for plane in PLANES:
        arrays[POINT_KEYS[plane]] = points[plane]
        arrays[plane + "_unit_noise"] = unit_noise[plane]
        arrays[plane + "_absolute_noise_sigma"] = sigmas[plane]
        arrays[plane + "_uncoded_field_clean"] = clean[plane]
        for key in ("clean", "field", "noise", "actual_codes"):
            arrays[plane + "_" + key] = observations[plane][key]
        powers = np.mean(np.abs(observations[plane]["clean"]) ** 2, axis=1)
        noise_powers = np.mean(np.abs(observations[plane]["noise"]) ** 2, axis=1)
        result["state_snr"][plane] = [{"canonical_state_id": i, "clean_rms": float(np.sqrt(power)),
                "absolute_noise_sigma": sigmas[plane],
                "expected_snr_db": float(10 * np.log10(power / sigmas[plane] ** 2)) if power > 0 else None,
                "realized_snr_db": float(10 * np.log10(power / noise_powers[i])) if power > 0 and noise_powers[i] > 0 else None}
                for i, power in enumerate(powers)]
    archive = output / (case["name"] + ".npz")
    np.savez_compressed(archive, **arrays)
    result.update({"archive": archive.name, "archive_sha256": _hash(archive), "seconds": perf_counter() - started})
    if any(not np.isfinite(scores["metrics"][key]) for key in BASELINE_LIMITS):
        result.update(status="failed", error="Nonfinite coefficient score (including an empty phase mask)")
    elif result["seconds"] > policy.get("max_case_seconds", float("inf")):
        result.update(status="timed_out", error="Cooperative per-case elapsed-time budget exceeded after computation")
    if result["status"] != "completed":
        result["target_passed"] = dict.fromkeys(targets, False)
    return result


def summarize_cases(cases, targets):
    groups = []
    keys = sorted({(c["frequency_hz"], c["regime"]["name"], c["domain"], c["with_faults"]) for c in cases})
    for frequency, regime, domain, faults in keys:
        rows = [c for c in cases if (c["frequency_hz"], c["regime"]["name"], c["domain"], c["with_faults"]) ==
                (frequency, regime, domain, faults)]
        completed = [row for row in rows if row["status"] == "completed"]
        counts = {status: sum(c["status"] == status for c in rows) for status in
                  ("completed", "failed", "timed_out", "rejected", "pending")}
        metrics = {}
        for key in BASELINE_LIMITS:
            values = [r["metrics"][key] for r in completed if np.isfinite(r["metrics"][key])]
            metrics[key] = {"mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)} if values else None
        groups.append({"frequency_hz": frequency, "regime": regime, "domain": domain, "with_faults": faults,
                       "denominator": len(rows), "counts": counts, "coverage": len(completed) / len(rows),
                       "target_pass_counts": {t: sum(r.get("target_passed", {}).get(t, False) and r["status"] == "completed" for r in rows) for t in targets},
                       "all_cases_baseline_passed": all(r["status"] == "completed" and r.get("target_passed", {}).get("baseline", False) for r in rows),
                       "completed_case_metrics": metrics,
                       "injected_detection_counts": {key: sum(r["injected_fault_detection"][key] for r in completed)
                        for key in ("true_positives", "false_positives", "false_negatives", "true_negatives")}})
    return groups


def run_k4_research(output, policy_path, operator_paths=(), *, generate_operators=False, progress=print):
    """Execute one bounded policy into a new explicit output directory.

    Operator paths are synthetic NPZ inputs in the preserved historical format;
    their embedded frequency selects each cache.  Missing caches are generated
    only with generate_operators=True.  Existing outputs are never overwritten.
    """
    output, policy_path = Path(output).resolve(), Path(policy_path).resolve()
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    cases = build_case_matrix(policy)
    inputs = _operator_inputs(operator_paths, policy["frequencies_hz"])
    if not generate_operators and set(inputs) != set(policy["frequencies_hz"]):
        raise ValueError("Supply one operator input per frequency, or explicitly enable generation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy_snapshot.json").write_bytes(policy_bytes)
    config_hash = sha256(policy_bytes).hexdigest()
    started = perf_counter()
    targets = {"baseline": BASELINE_LIMITS, **policy.get("targets", {})}
    sources = [Path(__file__), *[Path(__file__).with_name(name + ".py") for name in
               ("thesis_coded", "large_error_validation", "inverse", "validation", "diagnostics", "physics", "simulation")]]
    launcher = Path(__file__).resolve().parents[2] / "scripts/run_k4_research.py"
    unavailable_sources = {}
    if launcher.is_file():
        sources.append(launcher)
    else:
        unavailable_sources[str(launcher)] = "Repository launcher is not distributed with the installed numerical package"
    report = {"schema_version": 1, "stage": policy["stage"], "synthetic_only": True,
              "actual_measurements_read": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": config_hash, "policy_source": str(policy_path), "policy": policy,
              "source_hashes": {str(p): _hash(p) for p in sources}, "unavailable_sources": unavailable_sources, "targets": targets,
              "budget_semantics": "Cooperative wall-clock budgets; no interruption inside NumPy/SciPy; prep counts toward total",
              "scoring": {"gauge": "evaluation-only single least-squares complex scalar", "phase_mask_ratio": PHASE_MASK_RATIO,
                "phase_mask": "truth and aligned estimate amplitudes >=0.1*max(abs(truth)); both strictly positive",
                "off_elements_in_complex_nrmse": True, "rejection_rule": "none", "baseline_required_coverage": 1.},
              "seed_policy": {"excitation": "SeedSequence(seed).spawn(3): amplitude, phase, fault positions",
                "noise_spawn_key": [NOISE_NAMESPACE, "plane_id", "canonical_state_id"],
                "state_errors_spawn_key": [STATE_ERROR_NAMESPACE, "canonical_state_id"],
                "state_errors_substreams": "spawn(2): amplitude, phase", "plane_ids": dict(zip(PLANES, range(3))),
                "pairing": "Same normal coefficients/fault locations/unit noise/state standard normals across healthy/faulted, regimes and frequencies; independent replicate seeds",
                "reference_state_unchanged": True, "state_errors_fixed_across_planes": True,
                "noise_sigma": "Each case/plane initial uncoded clean RMS * 10^(-initial_snr_db/20), constant across all four states"},
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
                "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "operator_inputs": [{"path": str(path), "sha256": _hash(path), "frequency_hz": frequency}
                                  for frequency, path in inputs.items()],
              "operators": [], "conditioning": [], "in_progress": True, "cases": cases}
    _write_json(output / "case_matrix.json", {"config_sha256": config_hash, "cases": cases})

    def checkpoint():
        report["total_seconds"] = perf_counter() - started
        report["summary"] = summarize_cases(cases, targets)
        _write_json(output / "results.json", report)
        _write_json(output / "group_summary.json", {"config_sha256": config_hash, "groups": report["summary"]})

    checkpoint()
    for frequency in policy["frequencies_hz"]:
        selected = [case for case in cases if case["frequency_hz"] == frequency]
        if perf_counter() - started >= policy["max_total_seconds"]:
            for case in selected:
                case.update(status="timed_out", error="Cooperative total run budget exhausted before preparation")
            checkpoint()
            continue
        t_prepare = perf_counter()
        try:
            progress(f"Prepare K4 at {frequency / 1e9:g} GHz", flush=True)
            elements, points, matrices, truth_matrices, source = _operators(inputs.get(frequency), frequency, policy, output, config_hash)
            report["operators"].append(source)
            prepared = prepare_coded_ridge(matrices["training"], code_patterns(policy["geometry"]["array_side"], 4))
            nominal_baseline = prepared.joint_matrix @ np.ones(len(elements), dtype=complex)
            if not np.isfinite(np.vdot(nominal_baseline, nominal_baseline).real) or np.vdot(nominal_baseline, nominal_baseline).real <= 0:
                raise ValueError("Nominal coded baseline has no finite nonzero observable field")
            s = prepared.singular_values
            report["conditioning"].append({"frequency_hz": frequency, "matrix_shape": list(prepared.joint_matrix.shape),
                "normalized_condition_number": float(s[0] / s[-1]), "normalized_sigma_max": float(s[0]),
                "normalized_sigma_min": float(s[-1]), "numerical_rank": int(np.sum(s > prepared.prepared.rank_tolerance)),
                "direct_svd": True, "seconds": perf_counter() - t_prepare})
            np.savez_compressed(output / f"conditioning_{frequency:.12g}Hz.npz", singular_values=s, column_scale=prepared.column_scale)
        except Exception as exc:
            for case in selected:
                case.update(status="failed", error=f"Operator preparation: {type(exc).__name__}: {exc}")
            checkpoint()
            continue
        checkpoint()
        for case in selected:
            if perf_counter() - started >= policy["max_total_seconds"]:
                case.update(status="timed_out", error="Cooperative total run budget exhausted before case")
            else:
                t_case = perf_counter()
                try:
                    case.update(_case(case, policy, prepared, points, matrices, truth_matrices, elements, output, config_hash))
                except Exception as exc:
                    case.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=perf_counter() - t_case)
            _write_json(output / (case["name"] + ".json"), case)
            checkpoint()
            progress(f"[{sum(c['status'] != 'pending' for c in cases)}/{len(cases)}] {case['name']}: {case['status']} "
                     f"baseline={case.get('target_passed', {}).get('baseline', False)}", flush=True)
        del prepared, matrices, truth_matrices
    # Preparation failures and budget-skipped cases also get individual records.
    for case in cases:
        _write_json(output / (case["name"] + ".json"), case)
    report.update(in_progress=False, completed_utc=datetime.now(timezone.utc).isoformat(),
                  registered_case_count=len(cases), completed_case_count=sum(c["status"] == "completed" for c in cases),
                  failed_case_count=sum(c["status"] == "failed" for c in cases),
                  timed_out_case_count=sum(c["status"] == "timed_out" for c in cases), rejected_case_count=0)
    checkpoint()
    _write_json(output / "run_receipt.json", {key: report[key] for key in
                ("stage", "started_utc", "completed_utc", "config_sha256", "source_hashes", "unavailable_sources", "operator_inputs", "operators", "registered_case_count",
                 "completed_case_count", "failed_case_count", "timed_out_case_count", "rejected_case_count", "total_seconds")})
    return report
