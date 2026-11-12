"""Virtual known-excitation coding, inspired by the thesis literature review.

This is a project extension, not a reproduction of thesis chapter 5.  Hardware
base errors are fixed across coded states.  An independent receiver noise RMS
is fixed to the initial uncoded field on each measurement plane; weak coded
fields are never rescaled back to 40 dB.  Truth and final test observations are
excluded from the fitting interface.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from .inverse import prepare_tikhonov, solve_tikhonov
from .large_error_validation import sample_excitation, score_estimate


STATE_IDS = {1: (0,), 2: (0, 3), 4: (0, 1, 2, 3)}
METHODS = ("uncoded_K1", "uncoded_average_K2", "uncoded_average_K4", "coded_ideal_K2",
           "coded_ideal_K4", "coded_imperfect_K2", "coded_imperfect_K4",
           "coded_strong_imperfect_K2", "coded_strong_imperfect_K4")
NOISE_NAMESPACE = 64001
STATE_ERROR_NAMESPACE = 64002


def code_patterns(side, count):
    """Fixed logical-index patterns: 1, row flip, column flip, checkerboard."""
    if count not in STATE_IDS or int(side) != side or side < 1:
        raise ValueError("Positive integer side and code count 1, 2 or 4 required")
    row, col = np.indices((int(side), int(side)))
    canonical = np.stack((np.ones_like(row), (-1.) ** row, (-1.) ** col, (-1.) ** (row + col)))
    return canonical.reshape(4, -1)[list(STATE_IDS[count])].astype(complex)


def noise_unit_samples(seed, plane_id, state_ids, sample_count):
    """Documented proper-complex streams; one independent key per state/plane."""
    if plane_id not in (0, 1, 2) or sample_count < 1 or len(set(state_ids)) != len(state_ids):
        raise ValueError("Unique state IDs, valid plane ID and positive count required")
    rows = []
    for state_id in state_ids:
        sequence = np.random.SeedSequence(int(seed), spawn_key=(NOISE_NAMESPACE, int(plane_id), int(state_id)))
        rng = np.random.default_rng(sequence)
        rows.append((rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)) / np.sqrt(2))
    return np.asarray(rows)


def state_error_factors(seed, count, amplitude_std_db=.2, phase_std_deg=3.):
    """Unknown, fixed-per-code additional errors; reference code is unchanged."""
    errors = np.ones((4, count), dtype=complex)
    amp, phase = np.zeros((4, count)), np.zeros((4, count))
    for state_id in (1, 2, 3):
        amp_seed, phase_seed = np.random.SeedSequence(int(seed), spawn_key=(STATE_ERROR_NAMESPACE, state_id)).spawn(2)
        amp[state_id] = np.random.default_rng(amp_seed).normal(0, amplitude_std_db, count)
        phase[state_id] = np.random.default_rng(phase_seed).normal(0, phase_std_deg, count)
        errors[state_id] = 10 ** (amp[state_id] / 20) * np.exp(1j * np.deg2rad(phase[state_id]))
    return errors, amp, phase


def coded_observations(truth_matrix, truth, codes, unit_noise, noise_sigma, state_errors=None):
    """Generate states with a common *absolute* receiver noise RMS sigma."""
    h = np.asarray(truth_matrix, dtype=complex)
    c = np.asarray(truth, dtype=complex)
    codes = np.asarray(codes, dtype=complex)
    unit_noise = np.asarray(unit_noise, dtype=complex)
    if h.ndim != 2 or c.shape != (h.shape[1],) or codes.ndim != 2 or codes.shape[1] != len(c):
        raise ValueError("Inconsistent truth operator, coefficients or code shapes")
    if unit_noise.shape != (len(codes), h.shape[0]) or not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("Noise shape or absolute sigma is invalid")
    errors = np.ones_like(codes) if state_errors is None else np.asarray(state_errors, dtype=complex)
    if errors.shape != codes.shape:
        raise ValueError("State errors must match code shape")
    clean = (h @ (codes * errors * c[None, :]).T).T
    noise = noise_sigma * unit_noise
    return {"clean": clean, "field": clean + noise, "noise": noise,
            "noise_sigma": float(noise_sigma), "actual_codes": codes * errors}


@dataclass
class CodedRidgePrepared:
    matrix: np.ndarray
    codes: np.ndarray
    joint_matrix: np.ndarray
    prepared: object
    column_scale: np.ndarray
    singular_values: np.ndarray


def prepare_coded_ridge(matrix, codes):
    """Direct SVD of the same column-normalized stacked point-source operator."""
    matrix, codes = np.asarray(matrix, dtype=complex), np.asarray(codes, dtype=complex)
    if matrix.ndim != 2 or codes.ndim != 2 or codes.shape[1] != matrix.shape[1] or len(codes) < 1:
        raise ValueError("Codes must contain one value for each operator column")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(codes)):
        raise ValueError("Operator and codes must be finite")
    joint = np.vstack([matrix * code[None, :] for code in codes])
    prepared = prepare_tikhonov(joint)
    return CodedRidgePrepared(matrix, codes.copy(), joint, prepared, prepared.column_scale, prepared.singular_values)


def solve_coded_ridge(prepared, fields, validation_matrix, validation_fields, alphas=None):
    """Training common gain + nominal-centered ridge; independent validation only."""
    fields, validation_fields = np.asarray(fields, complex), np.asarray(validation_fields, complex)
    validation_matrix = np.asarray(validation_matrix, complex)
    k, m = len(prepared.codes), prepared.matrix.shape[0]
    if fields.shape != (k, m) or validation_fields.shape != (k, len(validation_matrix)):
        raise ValueError("One training/validation field per known coding state required")
    joint_validation = np.vstack([validation_matrix * code[None, :] for code in prepared.codes])
    observations, validation_observations = fields.ravel(), validation_fields.ravel()
    nominal = np.ones(prepared.matrix.shape[1], dtype=complex)
    baseline = prepared.joint_matrix @ nominal
    gain = complex(np.vdot(baseline, observations) / np.vdot(baseline, baseline).real)
    prior = gain * nominal
    fit = solve_tikhonov(prepared.prepared, observations, prior=prior, alphas=alphas,
                        validation_matrix=joint_validation, validation_y=validation_observations)
    return {"coefficients": fit.coefficients, "gain": gain, "selected_alpha": fit.alpha,
            "score_curve": fit.score_curve, "alpha_curve": fit.alpha_curve,
            "selection_field_nrmse": fit.validation_relative_error,
            "training_field_nrmse": fit.relative_residual,
            "regularized_effective_rank": fit.effective_rank,
            "selection_method": fit.selection_method,
            "normalized_lambda_absolute": fit.lambda_absolute}


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _relative(estimate, reference):
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-300))


def _snr_statistics(observations):
    result = []
    for clean, noise in zip(observations["clean"], observations["noise"]):
        signal = float(np.mean(np.abs(clean) ** 2)); actual = float(np.mean(np.abs(noise) ** 2))
        sigma2 = observations["noise_sigma"] ** 2
        result.append({"clean_rms": float(np.sqrt(signal)), "absolute_noise_sigma": observations["noise_sigma"],
                       "expected_state_snr_db": float(10 * np.log10(signal / sigma2)) if signal and sigma2 else None,
                       "realized_state_snr_db": float(10 * np.log10(signal / actual)) if signal and actual else None})
    return result


def _summary(cases):
    groups = []
    for with_faults in (False, True):
        for method in METHODS:
            rows = [case["methods"][method] for case in cases if case["with_faults"] == with_faults]
            if not rows:
                continue
            metrics = {}
            for key in ("active_amplitude_rmse_db", "active_phase_rmse_deg", "complex_nrmse"):
                values = [row["metrics"][key] for row in rows]
                metrics[key] = {"mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)}
            for key in ("independent_test_clean_field_nrmse", "independent_test_noisy_field_nrmse"):
                values = [row[key] for row in rows]
                metrics[key] = {"mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)}
            groups.append({"with_faults": with_faults, "method": method, "replicate_count": len(rows),
                           "metrics": metrics, "coefficient_pass_count": sum(row["coefficient_acceptance_passed"] for row in rows),
                           "legacy_joint_pass_count": sum(row["legacy_joint_acceptance_passed"] for row in rows),
                           "injected_detection_counts": {key: sum(row["injected_fault_detection"][key] for row in rows) for key in
                            ("true_positives", "false_positives", "false_negatives", "true_negatives")}})
    return groups


def _write_metrics(path, cases):
    rows = []
    for case in cases:
        for method, fit in case["methods"].items():
            row = {"case": case["name"], "seed": case["seed"], "with_faults": case["with_faults"], "method": method,
                   "measurement_states": fit["measurement_states"], "training_observation_count": fit["training_observation_count"],
                   "validation_observation_count": fit["validation_observation_count"],
                   "selected_alpha": fit["selected_alpha"], "coefficient_passed": fit["coefficient_acceptance_passed"],
                   "legacy_joint_passed": fit["legacy_joint_acceptance_passed"]}
            row.update({key: fit["metrics"][key] for key in ("active_amplitude_rmse_db", "active_phase_rmse_deg", "complex_nrmse")})
            row.update({key: fit[key] for key in ("independent_test_clean_field_nrmse", "independent_test_noisy_field_nrmse")})
            row.update({"injected_" + key: value for key, value in fit["injected_fault_detection"].items()})
            rows.append(row)
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_thesis_coded(output, policy_path, operator_path, progress=print):
    """Run six sealed synthetic cases and nine explicitly paired methods."""
    output, policy_path, operator_path = Path(output), Path(policy_path), Path(operator_path)
    policy_bytes = policy_path.read_bytes(); policy = json.loads(policy_bytes)
    config_hash = sha256(policy_bytes).hexdigest()
    reference_path = Path(policy["reference_policy"])
    if not reference_path.is_absolute():
        reference_path = policy_path.resolve().parents[1] / reference_path
    reference_bytes = reference_path.read_bytes(); reference = json.loads(reference_bytes)
    alphas = reference["alphas"]
    if len(alphas) != 25 or policy["confirmation_seeds"] != [20261111, 20261112, 20261113]:
        raise ValueError("Frozen confirmation seeds and original alpha grid required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy_snapshot.json").write_bytes(policy_bytes)
    (output / "reference_policy_snapshot.json").write_bytes(reference_bytes)
    with np.load(operator_path, allow_pickle=False) as source:
        if str(source["provenance"]) != "synthetic" or float(source["frequency_hz"]) != 11.7e9:
            raise ValueError("Only the audited 11.7GHz synthetic operator cache is accepted")
        points = {name: source[key] for name, key in (("training", "samples_m"), ("validation", "validation_samples_m"), ("test", "test_samples_m"))}
        matrices = {plane: source["point_" + plane] for plane in points}
        truth_matrices = {plane: source["truth_" + plane] for plane in points}
        elements = source["elements_m"]
    if len(elements) != 1024 or len(points["training"]) != 3969:
        raise ValueError("The preregistered 32x32 array and 63x63 scan are required")
    side = policy["geometry"]["array_side"]
    codes = {k: code_patterns(side, k) for k in (1, 2, 4)}
    started = perf_counter()
    prepared, conditioning = {}, {}
    spectra = {}
    for k in (1, 2, 4):
        t0 = perf_counter(); progress(f"Prepare direct joint SVD K={k}", flush=True)
        prepared[k] = prepare_coded_ridge(matrices["training"], codes[k])
        p = prepared[k]; s = p.singular_values
        conditioning[str(k)] = {"states": k, "canonical_state_ids": list(STATE_IDS[k]),
             "matrix_shape": list(p.joint_matrix.shape), "column_normalization": "Euclidean norms of all stacked training rows",
             "normalized_sigma_max": float(s[0]), "normalized_sigma_min": float(s[-1]),
             "normalized_condition_number": float(s[0] / s[-1]),
             "numerical_rank": int(np.sum(s > p.prepared.rank_tolerance)),
             "noise_resolved_modes_at_relative_0p01": int(np.sum(s / s[0] > .01)),
             "stable_rank": float(np.sum((s / s[0]) ** 2)),
             "minimum_column_norm": float(np.min(p.column_scale)), "maximum_column_norm": float(np.max(p.column_scale)),
             "preparation_seconds": perf_counter() - t0, "direct_svd": True}
        spectra["singular_values_K" + str(k)] = s
        spectra["column_scale_K" + str(k)] = p.column_scale
        progress(f"K={k}: condition {s[0] / s[-1]:.6g}, s_min {s[-1]:.6g} ({perf_counter()-t0:.1f}s)", flush=True)
    np.savez_compressed(output / "conditioning.npz", provenance="synthetic", **spectra)
    source_paths = [Path(__file__), Path(__file__).with_name("inverse.py"), Path(__file__).with_name("large_error_validation.py"),
                    Path(__file__).with_name("validation.py"), Path(__file__).with_name("diagnostics.py"), policy_path, reference_path,
                    Path(__file__).resolve().parents[2] / "scripts/run_thesis_coded.py",
                    Path(__file__).resolve().parents[2] / "tests/test_thesis_coded.py"]
    cases = []
    report = {"schema_version": 1, "synthetic_only": True, "actual_measurements_read": False,
              "thesis_reproduction": False, "thesis_relation": policy["coded"]["status"], "config_sha256": config_hash,
              "reference_policy_sha256": sha256(reference_bytes).hexdigest(), "policy": policy,
              "operator_source": str(operator_path), "operator_source_sha256": _hash(operator_path),
              "source_hashes": {str(p): _hash(p) for p in source_paths}, "conditioning": conditioning,
              "seed_policy": {"entropy": "confirmation replicate seed", "noise_spawn_key": [NOISE_NAMESPACE, "plane_id", "canonical_state_id"],
                 "plane_ids": {"training": 0, "validation": 1, "test": 2}, "canonical_state_ids": {"reference": 0, "row_flip": 1, "column_flip": 2, "checkerboard": 3},
                 "state_error_spawn_key": [STATE_ERROR_NAMESPACE, "canonical_state_id"], "state_error_substreams": "spawn(2): amplitude, phase",
                 "pairing": "Same unit noise across healthy/faulted and coded ideal/mild-imperfect/strong-imperfect/uncoded alternatives; each case/plane sigma fixed from its own initial uncoded clean RMS. K2 uses canonical state IDs [0,3], including the identical checkerboard errors/noise used by K4. Mild and strong state errors share standard Gaussian draws scaled to their registered standard deviations.",
                 "state_errors_fixed_across_all_planes": True},
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "in_progress": True, "cases": cases}
    for seed in policy["confirmation_seeds"]:
        errors, error_amp, error_phase = state_error_factors(seed, len(elements),
                policy["coded"]["nonideal_control"]["extra_amplitude_std_db"],
                policy["coded"]["nonideal_control"]["extra_phase_std_deg"])
        strong_errors, strong_error_amp, strong_error_phase = state_error_factors(seed, len(elements),
                policy["coded"]["strong_nonideal_control"]["extra_amplitude_std_db"],
                policy["coded"]["strong_nonideal_control"]["extra_phase_std_deg"])
        error_sets = {"ideal": None, "imperfect": errors, "strong_imperfect": strong_errors}
        latent_noise = {plane: noise_unit_samples(seed, plane_id, STATE_IDS[4], len(samples))
                        for plane_id, (plane, samples) in enumerate(points.items())}
        for with_faults in policy["fault_states"]:
            t0 = perf_counter()
            name = f"seed{seed}_" + ("faulted" if with_faults else "healthy")
            excitation = sample_excitation(len(elements), policy["scenario"], seed, with_faults)
            uncoded_clean = {plane: h @ excitation["truth"] for plane, h in truth_matrices.items()}
            sigmas = {plane: float(np.sqrt(np.mean(np.abs(field) ** 2)) * 10 ** (-policy["initial_snr_db"] / 20))
                      for plane, field in uncoded_clean.items()}
            observations = {}
            for accuracy, state_errors in error_sets.items():
                observations[accuracy] = {plane: coded_observations(h, excitation["truth"], codes[4], latent_noise[plane],
                        sigmas[plane], state_errors=state_errors)
                        for plane, h in truth_matrices.items()}
            arrays = {"provenance": "synthetic", "config_sha256": config_hash, "frequency_hz": 11.7e9, "elements_m": elements,
                "samples_m": points["training"], "validation_samples_m": points["validation"], "test_samples_m": points["test"],
                "nominal": excitation["nominal"], "true_coefficients": excitation["truth"], "healthy_coefficients": excitation["healthy"],
                "normal_amplitude_error_db": excitation["amp_error_db"], "normal_phase_error_deg": excitation["phase_error_deg"],
                "injected_fault_indices": excitation["fault_indices"], "paired_fault_indices": excitation["paired_fault_indices"],
                "fault_multipliers": excitation["fault_multipliers"], "canonical_codes": codes[4], "state_error_factors": errors,
                "state_error_amplitude_db": error_amp, "state_error_phase_deg": error_phase,
                "strong_state_error_factors": strong_errors, "strong_state_error_amplitude_db": strong_error_amp,
                "strong_state_error_phase_deg": strong_error_phase}
            for plane in points:
                arrays[plane + "_unit_noise"] = latent_noise[plane]
                arrays[plane + "_absolute_noise_sigma"] = sigmas[plane]
                arrays[plane + "_uncoded_field_clean"] = uncoded_clean[plane]
                for accuracy in error_sets:
                    for key in ("clean", "field", "noise"):
                        arrays[plane + "_" + accuracy + "_" + key] = observations[accuracy][plane][key]
            result = {"name": name, "seed": seed, "with_faults": with_faults,
                "physical_fault_indices": excitation["fault_indices"].tolist(), "normal_sample_statistics": {
                     "amplitude_std_db_ddof1": float(np.std(excitation["amp_error_db"], ddof=1)),
                     "amplitude_min_db": float(np.min(excitation["amp_error_db"])), "amplitude_max_db": float(np.max(excitation["amp_error_db"])),
                     "phase_std_deg_ddof1": float(np.std(excitation["phase_error_deg"], ddof=1))},
                "noise_sigmas": sigmas, "state_snr": {accuracy: {plane: _snr_statistics(observation) for plane, observation in states.items()}
                                                         for accuracy, states in observations.items()}, "methods": {}}
            for method in METHODS:
                k = 1 if method == "uncoded_K1" else int(method[-1])
                state_ids = list(STATE_IDS[k])
                uncoded = method.startswith("uncoded")
                accuracy = "strong_imperfect" if "strong_imperfect" in method else "imperfect" if "imperfect" in method else "ideal"
                if uncoded:
                    fields = {plane: (uncoded_clean[plane] + sigmas[plane] * np.mean(latent_noise[plane][state_ids], axis=0))[None, :]
                              for plane in ("training", "validation")}
                    solver = prepared[1]
                else:
                    fields = {plane: observations[accuracy][plane]["field"][state_ids] for plane in ("training", "validation")}
                    solver = prepared[k]
                fit = solve_coded_ridge(solver, fields["training"], matrices["validation"], fields["validation"], alphas)
                scores, aligned, flagged, oracle_flags = score_estimate(fit["coefficients"], excitation["truth"], excitation["nominal"],
                                                                        excitation["fault_indices"], reference)
                prediction = matrices["test"] @ fit["coefficients"]
                test_noisy = observations["ideal"]["test"]["field"][0]
                record = {key: value for key, value in fit.items() if not isinstance(value, np.ndarray) and key != "gain"}
                record.update(scores)
                record.update({"common_gain": [fit["gain"].real, fit["gain"].imag], "measurement_states": k,
                    "canonical_state_ids": state_ids, "repeated_uncoded_average": uncoded,
                    "additional_unknown_state_errors": accuracy != "ideal", "state_error_regime": accuracy,
                    "controls_coupling_matrix": "identity",
                    "training_observation_count": k * len(points["training"]), "validation_observation_count": k * len(points["validation"]),
                    "post_average_training_count": len(points["training"]) if uncoded else k * len(points["training"]),
                    "independent_test_clean_field_nrmse": _relative(prediction, uncoded_clean["test"]),
                    "independent_test_noisy_field_nrmse": _relative(prediction, test_noisy),
                    "independent_test_uses_uncoded_reference": True, "test_used_for_selection": False})
                result["methods"][method] = record
                for key, value in fit.items():
                    if isinstance(value, np.ndarray):
                        arrays[key + "__" + method] = value
                arrays["aligned_coefficients__" + method] = aligned
                arrays["flagged_mask__" + method] = flagged
                arrays["uncoded_test_predicted__" + method] = prediction
                arrays["fitted_training_observations__" + method] = fields["training"]
                arrays["fitted_validation_observations__" + method] = fields["validation"]
            result["truth_through_detector"] = scores["truth_through_detector"]
            arrays["oracle_flag_mask"] = oracle_flags
            np.savez_compressed(output / (name + ".npz"), **arrays)
            result.update({"archive": name + ".npz", "archive_sha256": _hash(output / (name + ".npz")), "seconds": perf_counter() - t0})
            cases.append(result)
            report["total_seconds"] = perf_counter() - started
            _write_json(output / "results.json", report); _write_metrics(output / "metrics.csv", cases)
            progress(f"[{len(cases)}/6] {name}: " + ", ".join(f"{m}={result['methods'][m]['metrics']['complex_nrmse']:.4f}" for m in METHODS), flush=True)
    report.update({"in_progress": False, "completed_case_count": len(cases), "completed_method_fits": len(cases) * len(METHODS),
                   "total_seconds": perf_counter() - started, "summary": _summary(cases)})
    _write_json(output / "results.json", report)
    _write_json(output / "group_summary.json", {"config_sha256": config_hash, "groups": report["summary"]})
    _write_json(output / "run_receipt.json", {"synthetic_only": True, "completed_case_count": len(cases),
        "completed_method_fits": len(cases) * len(METHODS), "seconds": report["total_seconds"], "config_sha256": config_hash,
        "operator_source_sha256": report["operator_source_sha256"], "source_hashes": report["source_hashes"],
        "seed_policy": report["seed_policy"], "requested_agent_configuration": policy["agent_configuration"]["requested"],
        "actual_agent_configuration_independently_verified": None,
        "descendant_configuration_details": [{"role": "coded runner agent", "requested_model": "gpt-6.1-sol", "requested_reasoning_effort": "ultra", "requested_fork_turns": "none", "actual_independently_verified": None},
                                             {"role": "coded mathematical tests subagent", "requested_model": "gpt-6.1-sol", "requested_reasoning_effort": "ultra", "requested_fork_turns": "none", "actual_independently_verified": None}],
        "new_software_installed": False})
    progress(f"Completed coded extension: 6 cases, {len(cases) * len(METHODS)} fits, {report['total_seconds']:.1f}s", flush=True)
    return report
