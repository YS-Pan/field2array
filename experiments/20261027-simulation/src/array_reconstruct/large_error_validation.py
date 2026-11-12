"""Preregistered virtual sensitivity to large *feed* amplitude/phase errors.

This separate experiment preserves the original algorithms and acceptance
limits.  It never opens real measurement data.  Independent finite-aperture
truth operators, point-source dictionaries, SVDs and sparse preparations are
reused across paired excitations.  Truth enters generation and scoring only.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy

from .diagnostics import classification_metrics
from .inverse import prepare_tikhonov, solve_tikhonov
from .physics import green_matrix
from .simulation import C0, simulate_measurement
from .sparse import prepare_sparse_faults, solve_sparse_faults
from .validation import accepts, array_geometry, evaluate_coefficients, scan_geometry


METHODS = ("nominal_common_gain", "ridge_zero", "sparse_nominal_gain", "ridge_nominal_centered")


def case_matrix(policy):
    """Return the frozen 30 main + 24 endpoint paired cases, before execution."""
    cases = []
    frequencies = [policy["main_frequency_hz"], *policy["endpoint_frequencies_hz"]]
    for frequency in frequencies:
        main = frequency == policy["main_frequency_hz"]
        for scenario in policy["scenarios"]:
            if not main and scenario["name"] not in policy["endpoint_scenarios"]:
                continue
            for seed in policy["seeds"]:
                token = f"{frequency / 1e9:.1f}".replace(".", "p")
                pair_id = f"f{token}_{scenario['name']}_seed{seed}"
                for with_faults in (False, True):
                    cases.append({"name": pair_id + ("_faulted" if with_faults else "_healthy"),
                                  "pair_id": pair_id, "frequency_hz": frequency,
                                  "scenario": dict(scenario), "scenario_name": scenario["name"],
                                  "seed": int(seed), "with_faults": with_faults,
                                  "group": "main" if main else "endpoint"})
    return cases


def sample_excitation(count, scenario, seed, with_faults):
    """Pair healthy/faulted feeds; split normal and fault-location RNG streams.

    Uniform span 5 dB means [-2.5,+2.5] dB, whose population std is 5/sqrt(12).
    Phase inputs are Gaussian standard deviations, not full ranges.  The
    underlying phase standard normals and uniform draws are shared across
    scenarios for the same replicate; no true errors are given to a solver.
    """
    if count < 5:
        raise ValueError("At least five elements are required")
    amp_seed, phase_seed, fault_seed = np.random.SeedSequence(seed).spawn(3)
    amp_rng = np.random.default_rng(amp_seed)
    if scenario["amplitude_distribution"] == "gaussian":
        amp = amp_rng.normal(0, scenario["amplitude_std_db"], count)
    elif scenario["amplitude_distribution"] == "uniform":
        half = float(scenario["amplitude_half_range_db"])
        amp = amp_rng.uniform(-half, half, count)
    else:
        raise ValueError("Unknown amplitude error distribution")
    phase = np.random.default_rng(phase_seed).normal(0, scenario["phase_std_deg"], count)
    nominal = np.ones(count, dtype=complex)
    healthy = nominal * 10 ** (amp / 20) * np.exp(1j * np.deg2rad(phase))
    truth = healthy.copy()
    selected = np.sort(np.random.default_rng(fault_seed).choice(count, 5, replace=False))
    multipliers = np.ones(count, dtype=complex)
    if with_faults:
        multipliers[selected] = [0., 10 ** (-6 / 20), np.exp(1j * np.deg2rad(60)),
                                10 ** (-6 / 20) * np.exp(1j * np.deg2rad(40)),
                                10 ** (-6 / 20) * np.exp(1j * np.deg2rad(40))]
        truth *= multipliers
    return {"nominal": nominal, "healthy": healthy, "truth": truth,
            "amp_error_db": amp, "phase_error_deg": phase,
            "fault_indices": selected if with_faults else np.empty(0, dtype=int),
            "paired_fault_indices": selected, "fault_multipliers": multipliers}


def cached_truth_matrix(elements, samples, frequency_hz, source_fraction=.1,
                        quadrature_order=3, chunk_size=256):
    """Independent scalar area-average aperture operator, not point green_matrix.

    This implements the same physical quadrature as simulate_measurement but
    materializes the operator once. Agreement is checked against that original
    field-accumulation implementation before any experiment at each frequency.
    """
    elements = np.asarray(elements, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if (elements.ndim != 2 or samples.ndim != 2 or elements.shape[1] != 3
            or samples.shape[1] != 3 or not len(elements) or not len(samples)
            or not np.all(np.isfinite(elements)) or not np.all(np.isfinite(samples))):
        raise ValueError("Finite (count,3) geometry is required")
    if frequency_hz <= 0 or source_fraction < 0 or quadrature_order < 1 or chunk_size < 1:
        raise ValueError("Invalid aperture or frequency configuration")
    width = source_fraction * C0 / frequency_hz
    nodes, weights = leggauss(quadrature_order)
    nodes = nodes * width / 2
    weights = weights / 2
    k = 2 * np.pi * frequency_hz / C0
    matrix = np.zeros((len(samples), len(elements)), dtype=complex)
    for lo in range(0, len(samples), chunk_size):
        receivers = samples[lo:lo + chunk_size]
        block = matrix[lo:lo + chunk_size]
        for j, sy in enumerate(nodes):
            for i, sx in enumerate(nodes):
                dx = receivers[:, None, 0] - (elements[None, :, 0] + sx)
                dy = receivers[:, None, 1] - (elements[None, :, 1] + sy)
                dz = receivers[:, None, 2] - elements[None, :, 2]
                distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                if np.any(distance <= np.finfo(float).tiny):
                    raise ValueError("Source and receiver quadrature points coincide")
                block += weights[i] * weights[j] * np.exp(1j * k * distance) / distance
    return matrix


def cached_measurement(truth_matrix, coefficients, seed, snr_db=40):
    """Fast measurement with the original simulator's third-stream complex AWGN."""
    clean = np.asarray(truth_matrix) @ np.asarray(coefficients)
    signal_power = float(np.mean(np.abs(clean) ** 2))
    noise_power = 0. if snr_db is None else signal_power * 10 ** (-float(snr_db) / 10)
    if signal_power == 0 and snr_db is not None:
        raise ValueError("Finite SNR is undefined for a zero field")
    noise_seed = np.random.SeedSequence(seed).spawn(3)[2]
    rng = np.random.default_rng(noise_seed)
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=len(clean)) + 1j * rng.normal(size=len(clean)))
    actual_noise_power = float(np.mean(np.abs(noise) ** 2))
    return {"field": clean + noise, "field_clean": clean, "noise": noise,
            "signal_power": signal_power, "expected_noise_power": noise_power,
            "realized_snr_db": float(10 * np.log10(signal_power / actual_noise_power))
            if actual_noise_power > 0 and signal_power > 0 else None}


def measurement_seeds(replicate_seed):
    """Unique plane seeds for the registered three replicates; paired elsewhere."""
    return {plane: int(replicate_seed) * 10 + 100000 + i
            for i, plane in enumerate(("training", "validation", "test"))}


def estimate_common_gain(matrix, y, nominal):
    """Training-only physical-coordinate complex least-squares common gain."""
    baseline = np.asarray(matrix) @ np.asarray(nominal)
    norm = float(np.vdot(baseline, baseline).real)
    if norm <= 0:
        raise ValueError("Nominal excitation has no observable training field")
    return complex(np.vdot(baseline, y) / norm)


def fit_methods(prepared, sparse_prepared, validation_matrix, field, validation_field,
                alphas=None, max_faults=None):
    """Fit all comparison methods without access to truth, labels or test fields."""
    matrix, nominal = prepared.matrix, sparse_prepared.nominal
    gain = estimate_common_gain(matrix, field, nominal)
    baseline = gain * nominal
    records = {"nominal_common_gain": {"coefficients": baseline,
               "common_gain": [gain.real, gain.imag], "selection_method": "training_only_common_gain_least_squares",
               "selected_alpha": None, "selected_support_count": 0,
               "selection_field_nrmse": _relative(validation_matrix @ baseline, validation_field),
               "training_field_nrmse": _relative(matrix @ baseline, field)}}
    for name, prior in (("ridge_zero", None), ("ridge_nominal_centered", baseline)):
        fit = solve_tikhonov(prepared, field, alphas=alphas, prior=prior,
                            validation_matrix=validation_matrix, validation_y=validation_field)
        records[name] = {"coefficients": fit.coefficients, "common_gain": [gain.real, gain.imag] if prior is not None else None,
                         "selection_method": fit.selection_method, "selected_alpha": fit.alpha,
                         "selected_support_count": None, "selection_field_nrmse": fit.validation_relative_error,
                         "training_field_nrmse": fit.relative_residual, "regularized_effective_rank": fit.effective_rank,
                         "alpha_curve": fit.alpha_curve, "score_curve": fit.score_curve}
    fit = solve_sparse_faults(sparse_prepared, field, validation_matrix=validation_matrix,
                             validation_y=validation_field, max_faults=max_faults)
    records["sparse_nominal_gain"] = {"coefficients": fit.coefficients,
               "common_gain": [float(fit.gain.real), float(fit.gain.imag)],
               "selection_method": fit.selection_method, "selected_alpha": None,
               "selected_support_count": fit.support_size, "support": fit.support,
               "selection_field_nrmse": fit.validation_relative_error,
               "training_field_nrmse": fit.relative_residual,
               "support_size_curve": fit.support_size_curve, "score_curve": fit.validation_score_curve,
               "training_score_curve": fit.training_score_curve, "gain_path": fit.gain_path,
               "support_path_flat": np.concatenate(fit.support_path),
               "support_path_offsets": np.cumsum([0, *[len(p) for p in fit.support_path]])}
    return {name: records[name] for name in METHODS}


def _relative(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1e-300))


def _classification(predicted, actual, healthy_null=False):
    record = classification_metrics(predicted, actual)
    if healthy_null and not np.any(actual):
        record["precision"] = None
        record["recall"] = None
    return record


def score_estimate(estimate, truth, nominal, injected_indices, policy):
    """Preserve old scores, explicitly distinguishing dense deviations and faults."""
    injected_indices = np.asarray(injected_indices, dtype=int)
    legacy, aligned, flagged = evaluate_coefficients(estimate, truth, nominal, injected_indices)
    oracle_legacy, _, oracle_flags = evaluate_coefficients(truth, truth, nominal, injected_indices)
    injected = np.zeros(len(truth), dtype=bool)
    injected[injected_indices] = True
    metrics = dict(legacy)
    if not len(injected_indices):
        metrics["fault_precision"] = None
        metrics["fault_recall"] = None
    limits = policy["independent_model_limits"]
    coefficient_pass = all(legacy[k] <= limits[k] for k in
                           ("active_amplitude_rmse_db", "active_phase_rmse_deg", "complex_nrmse"))
    truth_control = _classification(oracle_flags, injected, healthy_null=True)
    truth_control.update({"flagged_indices": np.flatnonzero(oracle_flags).tolist(),
                          "normal_noninjected_threshold_crossings": int(np.sum(oracle_flags & ~injected)),
                          "legacy_joint_acceptance_passed": accepts(oracle_legacy, policy)})
    record = {"metrics": metrics, "legacy_metrics": legacy,
              "coefficient_acceptance_passed": coefficient_pass,
              "legacy_joint_acceptance_passed": accepts(legacy, policy),
              "injected_fault_detection": _classification(flagged, injected, healthy_null=True),
              "oracle_nominal_threshold_detection": _classification(flagged, oracle_flags),
              "truth_through_detector": truth_control,
              "truth_used_for_model_selection": False}
    # Same global oracle gauge as the old full-vector score, then evaluate the
    # non-injected group so large normal variations cannot hide behind faults.
    normal = ~injected
    true_amp = np.abs(truth)
    active = normal & (true_amp >= .1 * np.max(true_amp))
    phase_active = active & (np.abs(aligned) >= .1 * np.max(true_amp))
    amplitude_error = 20 * np.log10(np.maximum(np.abs(aligned[active]), 1e-15) / true_amp[active])
    phase_error = np.rad2deg(np.angle(aligned[phase_active] * np.conj(truth[phase_active])))
    recovered_ratio = aligned[normal] / nominal[normal]
    true_ratio = truth[normal] / nominal[normal]
    record["noninjected_metrics"] = {
        "count": int(np.sum(normal)), "complex_nrmse": _relative(aligned[normal], truth[normal]),
        "amplitude_rmse_db": float(np.sqrt(np.mean(amplitude_error ** 2))),
        "phase_rmse_deg": float(np.sqrt(np.mean(phase_error ** 2))),
        "true_amplitude_std_db": float(np.std(20 * np.log10(np.maximum(np.abs(true_ratio), 1e-15)))),
        "estimated_amplitude_std_db": float(np.std(20 * np.log10(np.maximum(np.abs(recovered_ratio), 1e-15)))),
        "true_phase_std_deg": float(np.std(np.rad2deg(np.angle(true_ratio)))),
        "estimated_phase_std_deg": float(np.std(np.rad2deg(np.angle(recovered_ratio))))}
    return record, aligned, flagged, oracle_flags


def _sample_statistics(amp, phase):
    def describe(values):
        return {"mean": float(np.mean(values)), "sample_std_ddof1": float(np.std(values, ddof=1)),
                "population_std_ddof0": float(np.std(values)), "minimum": float(np.min(values)),
                "maximum": float(np.max(values)), "realized_range": float(np.ptp(values))}
    return {"amplitude_db": describe(amp), "phase_deg": describe(phase)}


def _json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_truth_cache(elements, planes, frequency, truth_matrices, policy):
    rng = np.random.default_rng(918273)
    coefficients = (.7 + .6 * rng.random(len(elements))) * np.exp(.8j * rng.normal(size=len(elements)))
    checks = {}
    for i, (name, points) in enumerate(planes.items()):
        seed = 13579 + i
        cached = cached_measurement(truth_matrices[name], coefficients, seed, policy["measurement_snr_db"])
        original = simulate_measurement(elements, coefficients, frequency, points, config={
            "source_aperture_m": policy["truth"]["source_aperture_wavelength_fraction"] * C0 / frequency,
            "source_quadrature_order": policy["truth"]["quadrature_order"], "directionality_power": 0.,
            "snr_db": policy["measurement_snr_db"], "seed": seed})
        errors = {key: _relative(cached[key], original[key]) for key in ("field_clean", "noise", "field")}
        if max(errors.values()) > 5e-13:
            raise AssertionError(f"Truth cache differs from original simulator: {name}: {errors}")
        checks[name] = {"relative_errors": errors, "tolerance": 5e-13, "passed": True,
                        "verification_seed": seed, "coefficient_seed": 918273}
    return checks


def _write_elements(path, side, excitation, fitted_arrays, oracle_flags):
    fields = ["index", "row", "column", "normal_amp_error_db", "normal_phase_error_deg",
              "injected_fault", "oracle_nominal_threshold_flag", "true_real", "true_imag"]
    for method in METHODS:
        fields += [method + "_real", method + "_imag", method + "_aligned_real", method + "_aligned_imag", method + "_flagged"]
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        injected = set(excitation["fault_indices"].tolist())
        for index, true in enumerate(excitation["truth"]):
            row = [index, index // side, index % side, excitation["amp_error_db"][index],
                   excitation["phase_error_deg"][index], index in injected, bool(oracle_flags[index]), true.real, true.imag]
            for method in METHODS:
                coefficient, aligned, flagged = fitted_arrays[method]
                row += [coefficient[index].real, coefficient[index].imag, aligned[index].real, aligned[index].imag, bool(flagged[index])]
            writer.writerow(row)


def _write_metrics(path, cases):
    fields = ["case", "frequency_hz", "scenario", "seed", "with_faults", "method",
              "active_amplitude_rmse_db", "active_phase_rmse_deg", "complex_nrmse",
              "coefficient_acceptance_passed", "legacy_joint_acceptance_passed",
              "injected_precision", "injected_recall", "injected_false_positive_rate",
              "oracle_precision", "oracle_recall", "oracle_false_positive_rate",
              "independent_test_clean_field_nrmse", "independent_test_noisy_field_nrmse",
              "coefficient_error_reduction_vs_nominal_fraction", "selected_alpha", "selected_support_count"]
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in cases:
            for name, method in case["methods"].items():
                row = {"case": case["name"], "frequency_hz": case["frequency_hz"], "scenario": case["scenario_name"],
                       "seed": case["seed"], "with_faults": case["with_faults"], "method": name}
                row.update({key: method["metrics"][key] for key in fields[6:9]})
                for key in fields[9:11] + fields[17:]:
                    row[key] = method.get(key)
                for prefix, source in (("injected", "injected_fault_detection"), ("oracle", "oracle_nominal_threshold_detection")):
                    for key in ("precision", "recall", "false_positive_rate"):
                        row[prefix + "_" + key] = method[source][key]
                writer.writerow(row)


def run_large_error_validation(output, policy_path, progress=print):
    """Execute the frozen 54 cases, refusing to overwrite a historical output."""
    output, policy_path = Path(output), Path(policy_path)
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    config_hash = sha256(policy_bytes).hexdigest()
    cases = case_matrix(policy)
    if len(cases) != 54:
        raise ValueError("Preregistered matrix must contain exactly 54 cases")
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy_snapshot.json").write_bytes(policy_bytes)
    _json_write(output / "case_matrix.json", {"config_sha256": config_hash, "cases": cases})
    g = policy["geometry"]
    elements = array_geometry(g["array_side"], g["pitch_m"])
    samples = scan_geometry(g["scan_side"], g["scan_step_m"], g["scan_distance_m"])
    validation = samples[::3].copy(); validation[:, 0] += g["validation_x_offset_m"]
    test = samples[1::3].copy(); test[:, 1] += g["test_y_offset_m"]; test[:, 2] += g["test_z_offset_m"]
    planes = {"training": samples, "validation": validation, "test": test}
    started = perf_counter()
    records, caches = [], []
    report = {"schema_version": 1, "synthetic_only": True, "actual_measurements_read": False,
              "universal_identifiability_claim": False, "in_progress": True,
              "config_sha256": config_hash, "policy": policy,
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "thread_environment": {k: os.environ.get(k) for k in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "source_hashes": {str(path.name): _file_hash(path) for path in
                                (Path(__file__), Path(__file__).with_name("simulation.py"),
                                 Path(__file__).with_name("inverse.py"), Path(__file__).with_name("sparse.py"),
                                 Path(__file__).with_name("validation.py"), Path(__file__).with_name("physics.py"),
                                 Path(__file__).with_name("diagnostics.py"), policy_path)},
              "cache": caches, "cases": records}
    for frequency in dict.fromkeys(case["frequency_hz"] for case in cases):
        t0 = perf_counter()
        progress(f"Prepare {frequency / 1e9:.1f} GHz: point and independent aperture operators", flush=True)
        matrices = {name: green_matrix(elements, points, frequency) for name, points in planes.items()}
        truth_matrices = {name: cached_truth_matrix(elements, points, frequency,
            policy["truth"]["source_aperture_wavelength_fraction"], policy["truth"]["quadrature_order"])
            for name, points in planes.items()}
        verification = _verify_truth_cache(elements, planes, frequency, truth_matrices, policy)
        progress(f"Prepare {frequency / 1e9:.1f} GHz: cache verified; SVD and sparse projection", flush=True)
        prepared = prepare_tikhonov(matrices["training"])
        sparse_prepared = prepare_sparse_faults(matrices["training"], np.ones(len(elements), dtype=complex))
        cache_name = f"operators_{frequency / 1e9:.1f}GHz.npz".replace(".0GHz", "GHz")
        np.savez(output / cache_name, provenance="synthetic", config_sha256=config_hash,
                 frequency_hz=frequency, elements_m=elements, samples_m=samples,
                 validation_samples_m=validation, test_samples_m=test,
                 point_training=matrices["training"], point_validation=matrices["validation"], point_test=matrices["test"],
                 truth_training=truth_matrices["training"], truth_validation=truth_matrices["validation"], truth_test=truth_matrices["test"],
                 singular_values=prepared.singular_values, column_scale=prepared.column_scale)
        cache_record = {"frequency_hz": frequency, "file": cache_name, "sha256": _file_hash(output / cache_name),
                        "verification_against_simulate_measurement": verification,
                        "prepare_seconds": perf_counter() - t0, "svd_computations": 1, "sparse_preparations": 1,
                        "column_scaled_condition_number": float(prepared.singular_values[0] / prepared.singular_values[-1]),
                        "numerical_rank": int(np.sum(prepared.singular_values > prepared.rank_tolerance))}
        caches.append(cache_record)
        progress(f"Prepared {frequency / 1e9:.1f} GHz in {cache_record['prepare_seconds']:.1f}s", flush=True)
        for config in (case for case in cases if case["frequency_hz"] == frequency):
            t0 = perf_counter(); name = config["name"]
            excitation = sample_excitation(len(elements), config["scenario"], config["seed"], config["with_faults"])
            plane_seeds = measurement_seeds(config["seed"])
            measurements = {plane: cached_measurement(truth_matrices[plane], excitation["truth"],
                plane_seeds[plane], policy["measurement_snr_db"]) for plane in planes}
            fitted = fit_methods(prepared, sparse_prepared, matrices["validation"], measurements["training"]["field"],
                                 measurements["validation"]["field"], alphas=policy["alphas"], max_faults=policy["sparse_support_cap"])
            # Generation and fitting have completed before coefficient truth,
            # physical injection labels or independent test fields enter scoring.
            saved = {"provenance": "synthetic", "config_sha256": config_hash, "frequency_hz": frequency,
                     "elements_m": elements, "samples_m": samples, "validation_samples_m": validation,
                     "test_samples_m": test, "nominal": excitation["nominal"], "healthy_coefficients": excitation["healthy"],
                     "true_coefficients": excitation["truth"], "normal_amplitude_error_db": excitation["amp_error_db"],
                     "normal_phase_error_deg": excitation["phase_error_deg"], "injected_fault_indices": excitation["fault_indices"],
                     "paired_fault_indices": excitation["paired_fault_indices"], "fault_multipliers": excitation["fault_multipliers"]}
            for plane, prefix in (("training", ""), ("validation", "validation_"), ("test", "test_")):
                for key in ("field", "field_clean", "noise"):
                    saved[prefix + key] = measurements[plane][key]
            result = {**config, "normal_error_statistics": _sample_statistics(excitation["amp_error_db"], excitation["phase_error_deg"]),
                      "pairing": {"replicate_seed": config["seed"], "measurement_seeds": plane_seeds,
                                  "paired_fault_indices": excitation["paired_fault_indices"].tolist(),
                                  "same_normal_errors_and_unit_noise_in_healthy_fault_pair": True},
                      "measurement_statistics": {plane: {key: measurement[key] for key in
                                  ("signal_power", "expected_noise_power", "realized_snr_db")}
                                  for plane, measurement in measurements.items()}, "methods": {}}
            fitted_arrays = {}
            for method, fit in fitted.items():
                estimate = fit["coefficients"]
                scores, aligned, flagged, oracle_flags = score_estimate(estimate, excitation["truth"], excitation["nominal"],
                                                                       excitation["fault_indices"], policy)
                record = {key: value for key, value in fit.items() if not isinstance(value, np.ndarray)}
                if "support" in fit:
                    record["support"] = fit["support"].tolist()
                test_prediction = matrices["test"] @ estimate
                record.update(scores)
                record.update({"independent_test_clean_field_nrmse": _relative(test_prediction, measurements["test"]["field_clean"]),
                               "independent_test_noisy_field_nrmse": _relative(test_prediction, measurements["test"]["field"])})
                result["methods"][method] = record
                saved["coefficients__" + method] = estimate
                saved["aligned_coefficients__" + method] = aligned
                saved["flagged_mask__" + method] = flagged
                saved["test_predicted__" + method] = test_prediction
                for key, value in fit.items():
                    if isinstance(value, np.ndarray) and key != "coefficients":
                        saved[key + "__" + method] = value
                fitted_arrays[method] = (estimate, aligned, flagged)
            result["truth_through_detector"] = scores["truth_through_detector"]
            saved["oracle_flag_mask"] = oracle_flags
            baseline = result["methods"]["nominal_common_gain"]
            for method in result["methods"].values():
                for key, baseline_error in (("coefficient", baseline["metrics"]["complex_nrmse"]),
                    ("test_clean_field", baseline["independent_test_clean_field_nrmse"])):
                    error = method["metrics"]["complex_nrmse"] if key == "coefficient" else method["independent_test_clean_field_nrmse"]
                    method[key + "_error_reduction_vs_nominal_fraction"] = 1 - error / max(baseline_error, 1e-300)
            np.savez_compressed(output / (name + ".npz"), **saved)
            _write_elements(output / (name + "_elements.csv"), g["array_side"], excitation, fitted_arrays, oracle_flags)
            result.update({"archive": name + ".npz", "archive_sha256": _file_hash(output / (name + ".npz")),
                           "seconds": perf_counter() - t0})
            records.append(result)
            report["total_seconds"] = perf_counter() - started
            _json_write(output / "results.json", report)
            _write_metrics(output / "metrics.csv", records)
            display = ", ".join(f"{method}={result['methods'][method]['metrics']['complex_nrmse']:.4f}" for method in METHODS)
            progress(f"[{len(records)}/54] {name}: {display} ({result['seconds']:.1f}s)", flush=True)
        # Discard this frequency's expensive arrays before preparing the next.
        del prepared, sparse_prepared, matrices, truth_matrices
    report["in_progress"] = False
    report["total_seconds"] = perf_counter() - started
    report["completed_case_count"] = len(records)
    report["completed_method_fits"] = len(records) * len(METHODS)
    report["summary"] = {method: {
        "coefficient_pass_count": sum(case["methods"][method]["coefficient_acceptance_passed"] for case in records),
        "legacy_joint_pass_count": sum(case["methods"][method]["legacy_joint_acceptance_passed"] for case in records),
        "maximum_complex_nrmse": max(case["methods"][method]["metrics"]["complex_nrmse"] for case in records)} for method in METHODS}
    _json_write(output / "results.json", report)
    progress(f"Completed {len(records)} cases / {report['completed_method_fits']} method fits in {report['total_seconds']:.1f}s", flush=True)
    return report
