"""Synthetic fixed-length feedback calibration inspired by thesis table 5-2.

The controller only receives an estimate and its accumulated commands.  A
separate virtual hardware response produces observations; hidden coefficients
and injection labels enter evaluation after each estimate and command are fixed.
This is a scalar method experiment, not a reproduction of the thesis apparatus.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .inverse import prepare_tikhonov, solve_tikhonov
from .large_error_validation import sample_excitation
from .validation import evaluate_coefficients


STRATEGIES = {
    "once_ap": ("ap",),
    "sequential_a_p_ap": ("a", "p", "ap"),
    "repeated_ap": ("ap", "ap", "ap"),
}


@dataclass(frozen=True)
class CalibrationConfig:
    attenuation_step_db: float = 0.2
    attenuation_max_db: float = 25.4
    phase_step_deg: float = 5.625
    exclusion_relative_median: float = 0.1
    initial_snr_db: float = 40.0
    seeds: tuple[int, ...] = (20261111, 20261112, 20261113)
    fault_states: tuple[bool, ...] = (False, True)
    coupling_deg_per_db: tuple[float, ...] = (0.0, 1.0)
    alphas: tuple[float, ...] = field(default_factory=lambda: tuple(np.logspace(-8, 0, 25)))


def _wrap_deg(values):
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def quantize_attenuation(values, config):
    """Round cumulative attenuation to its code grid, then enforce bounds."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Attenuation commands must be finite")
    return np.clip(np.rint(values / config.attenuation_step_db) * config.attenuation_step_db,
                   0.0, config.attenuation_max_db)


def quantize_phase(values_deg, config):
    """Periodic phase codes; 360 degrees and zero are the same physical code."""
    values = np.asarray(values_deg, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Phase commands must be finite")
    count = int(round(360.0 / config.phase_step_deg))
    if not np.isclose(count * config.phase_step_deg, 360.0):
        raise ValueError("Phase step must divide 360 degrees")
    return (np.rint(_wrap_deg(values) / config.phase_step_deg).astype(np.int64) % count) * config.phase_step_deg


def apply_hardware(initial_coefficients, attenuation_db, phase_command_deg,
                   attenuation_phase_coupling_deg_per_db):
    """Apply absolute accumulated codes to fixed initial hardware coefficients.

    The 1 degree/dB option is an explicit sensitivity assumption.  No
    phase-to-amplitude interaction or mutual coupling is added in this test.
    """
    initial = np.asarray(initial_coefficients, dtype=complex)
    attenuation = np.asarray(attenuation_db, dtype=float)
    phase = np.asarray(phase_command_deg, dtype=float)
    if initial.shape != attenuation.shape or initial.shape != phase.shape:
        raise ValueError("One amplitude and phase command per initial coefficient is required")
    return initial * 10 ** (-attenuation / 20.0) * np.exp(1j * np.deg2rad(
        _wrap_deg(phase) + attenuation_phase_coupling_deg_per_db * attenuation))


def _control_mask(estimate, config):
    amplitudes = np.abs(np.asarray(estimate, dtype=complex))
    median = float(np.median(amplitudes))
    # An identically zero estimate supplies no meaningful phase or mean target.
    eligible = (amplitudes > 0) & (amplitudes >= config.exclusion_relative_median * median)
    return amplitudes, eligible


def control_update(estimate, attenuation_db, phase_command_deg, mode, config):
    """Use current estimated field coefficients, never true faults or coefficients."""
    if mode not in ("a", "p", "ap"):
        raise ValueError("mode must be a, p or ap")
    estimate = np.asarray(estimate, dtype=complex)
    attenuation = np.asarray(attenuation_db, dtype=float).copy()
    phase = np.asarray(phase_command_deg, dtype=float).copy()
    if estimate.shape != attenuation.shape or estimate.shape != phase.shape or estimate.ndim != 1:
        raise ValueError("Equal-sized coefficient and command vectors are required")
    amplitudes, eligible = _control_mask(estimate, config)
    delta = np.zeros(len(estimate))
    phase_delta = np.zeros(len(estimate))
    amplitude_target = phase_target = None
    phase_mean_defined = False
    if np.any(eligible):
        amplitude_target = float(np.mean(amplitudes[eligible]))
        phase_resultant = np.mean(estimate[eligible] / amplitudes[eligible])
        phase_mean_defined = bool(abs(phase_resultant) > 1e-12)
        if phase_mean_defined:
            phase_target = float(np.rad2deg(np.angle(phase_resultant)))
        if "a" in mode:
            delta[eligible] = np.maximum(20 * np.log10(amplitudes[eligible] / amplitude_target), 0.0)
            attenuation[eligible] = quantize_attenuation(attenuation[eligible] + delta[eligible], config)
        if "p" in mode and phase_mean_defined:
            phase_delta[eligible] = _wrap_deg(phase_target - np.rad2deg(np.angle(estimate[eligible])))
            phase[eligible] = quantize_phase(phase[eligible] + phase_delta[eligible], config)
    return {"attenuation_db": attenuation, "phase_command_deg": phase,
            "excluded_mask": ~eligible, "requested_attenuation_delta_db": delta,
            "requested_phase_delta_deg": phase_delta,
            "saturated_mask": attenuation >= config.attenuation_max_db - 1e-12,
            "linear_amplitude_target": amplitude_target, "circular_phase_target_deg": phase_target,
            "phase_mean_defined": phase_mean_defined}


def fixed_noise_measurement(truth_matrix, coefficients, sigma, seed):
    """Receiver noise has fixed absolute complex RMS sigma after attenuation."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    clean = np.asarray(truth_matrix) @ np.asarray(coefficients)
    rng = np.random.default_rng(seed)
    noise = sigma / np.sqrt(2.0) * (rng.normal(size=len(clean)) + 1j * rng.normal(size=len(clean)))
    signal_power = float(np.mean(np.abs(clean) ** 2))
    actual_noise_power = float(np.mean(np.abs(noise) ** 2))
    return {"field": clean + noise, "field_clean": clean, "noise": noise,
            "noise_sigma": sigma, "signal_power": signal_power,
            "expected_noise_power": sigma * sigma, "actual_noise_power": actual_noise_power,
            "realized_snr_db": float(10 * np.log10(signal_power / actual_noise_power))
            if signal_power > 0 and actual_noise_power > 0 else None}


def fit_current_state(prepared, y, validation_matrix, validation_y, alphas):
    """Known nominal shape; training-only common gain; noisy validation alpha."""
    nominal = np.ones(prepared.matrix.shape[1], dtype=complex)
    h = prepared.matrix @ nominal
    gain = np.vdot(h, y) / np.vdot(h, h).real
    fit = solve_tikhonov(prepared, y, alphas=alphas, prior=gain * nominal,
                         validation_matrix=validation_matrix, validation_y=validation_y)
    return fit.coefficients, {"common_gain": [float(gain.real), float(gain.imag)],
        "selected_alpha": fit.alpha, "selection_method": fit.selection_method,
        "selection_field_nrmse": fit.validation_relative_error,
        "training_field_nrmse": fit.relative_residual,
        "regularized_effective_rank": fit.effective_rank,
        "alpha_curve": fit.alpha_curve, "score_curve": fit.score_curve}


def _noise_seed(seed, stage, plane):
    # Explicit namespaces prevent collisions between replicate, stage and plane.
    return int(np.random.SeedSequence([2026102803, int(seed), int(stage), int(plane)])
               .generate_state(1, dtype=np.uint64)[0])


def _relative(estimate, reference):
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-300))


def _group_metrics(coefficients, mask):
    values = np.asarray(coefficients)[mask]
    amplitudes = np.abs(values)
    active = amplitudes > 1e-15
    amp_db = 20 * np.log10(amplitudes[active])
    angles = np.angle(values[active])
    if len(angles):
        resultant = np.mean(np.exp(1j * angles))
        center = float(np.angle(resultant))
        residual = np.angle(np.exp(1j * (angles - center)))
        phase_std = float(np.rad2deg(np.sqrt(max(0., -2 * np.log(max(abs(resultant), 1e-300))))))
    else:
        center, residual, phase_std = None, np.empty(0), None
    return {"count": int(len(values)), "active_count": int(np.sum(active)),
        "zero_or_unscored_phase_count": int(np.sum(~active)),
        "amplitude_mean_db": float(np.mean(amp_db)) if len(amp_db) else None,
        "amplitude_std_db": float(np.std(amp_db)) if len(amp_db) else None,
        "phase_circular_std_deg": phase_std,
        "phase_rms_about_circular_mean_deg": float(np.rad2deg(np.sqrt(np.mean(residual ** 2)))) if len(residual) else None,
        "amplitude_rmse_vs_unit_nominal_db": float(np.sqrt(np.mean(amp_db ** 2))) if len(amp_db) else None,
        "phase_rmse_vs_zero_nominal_deg": float(np.rad2deg(np.sqrt(np.mean(angles ** 2)))) if len(angles) else None,
        "phase_circular_mean_deg": float(np.rad2deg(center)) if center is not None else None,
        "complex_nrmse_vs_unit_nominal": _relative(values, np.ones(len(values))) if len(values) else None}


def _evaluate_stage(estimate, coefficients, initial, injected_indices, excluded, attenuation,
                    phase_commands, matrices, policy, config):
    """Evaluation only: grouping uses original injection labels after control."""
    injected = np.zeros(len(initial), dtype=bool)
    injected[injected_indices] = True
    normal = ~injected
    dead = np.abs(initial) == 0
    nominal = np.ones(len(initial), dtype=complex)
    recovery, aligned, flagged = evaluate_coefficients(estimate, coefficients, nominal, injected_indices)
    limits = policy["coefficient_limits"]
    coefficient_pass = all(recovery[key] <= value for key, value in limits.items())
    power = float(np.sum(np.abs(coefficients) ** 2))
    initial_power = float(np.sum(np.abs(initial) ** 2))
    boresight = float(abs(np.sum(coefficients)) ** 2)
    initial_boresight = float(abs(np.sum(initial)) ** 2)
    true_test = matrices["truth_test"] @ coefficients
    predicted_test = matrices["point_test"] @ estimate
    record = {"reconstruction_metrics": recovery, "coefficient_acceptance_passed": coefficient_pass,
        "all_channel_true_metrics": _group_metrics(coefficients, np.ones(len(initial), dtype=bool)),
        "noninjected_true_metrics": _group_metrics(coefficients, normal),
        "injected_true_metrics": _group_metrics(coefficients, injected),
        "noninjected_estimated_metrics": _group_metrics(estimate, normal),
        "coefficient_power": power, "coefficient_power_ratio_vs_initial": power / initial_power,
        "coefficient_power_change_db": float(10 * np.log10(max(power / initial_power, 1e-300))),
        "boresight_absolute_power": boresight,
        "boresight_absolute_power_ratio_vs_initial": boresight / initial_boresight,
        "boresight_power_change_db": float(10 * np.log10(max(boresight / initial_boresight, 1e-300))),
        "coherent_efficiency": boresight / max(len(coefficients) * power, 1e-300),
        "independent_test_clean_field_nrmse": _relative(predicted_test, true_test),
        "control_exclusion": {"excluded_count": int(np.sum(excluded)),
            "normal_false_exclusion_count": int(np.sum(excluded & normal)),
            "injected_exclusion_count": int(np.sum(excluded & injected)),
            "true_dead_exclusion_count": int(np.sum(excluded & dead)),
            "true_dead_not_excluded_count": int(np.sum(~excluded & dead))},
        "commands": {"attenuation_sum_db": float(np.sum(attenuation)),
            "attenuation_mean_db": float(np.mean(attenuation)), "attenuation_max_db": float(np.max(attenuation)),
            "attenuation_saturated_count": int(np.sum(attenuation >= config.attenuation_max_db - 1e-12)),
            "nonzero_attenuation_count": int(np.sum(attenuation > 0)),
            "nonzero_phase_command_count": int(np.sum(phase_commands != 0))},
        "truth_used_for_control_or_alpha_selection": False}
    return record, aligned, flagged, predicted_test, true_test


def _hash(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _configuration(policy, reference):
    section = policy["calibration"]
    if section["attenuation_range_db"][0] != 0:
        raise ValueError("This attenuation-only implementation starts at zero dB")
    return CalibrationConfig(attenuation_step_db=section["attenuation_step_db"],
        attenuation_max_db=section["attenuation_range_db"][1], phase_step_deg=section["phase_step_deg"],
        initial_snr_db=policy["initial_snr_db"], seeds=tuple(policy["confirmation_seeds"]),
        fault_states=tuple(policy["fault_states"]),
        coupling_deg_per_db=tuple(section["amplitude_to_phase_deg_per_db"]), alphas=tuple(reference["alphas"]))


def run_calibration(output, policy_path, operator_path, progress=print):
    """Run the registered 36 trajectories; refuse to overwrite prior results."""
    started = perf_counter()
    output, policy_path, operator_path = Path(output), Path(policy_path), Path(operator_path)
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    root = Path(__file__).resolve().parents[2]
    reference_path = root / policy["reference_policy"]
    reference = json.loads(reference_path.read_text(encoding="utf8"))
    config = _configuration(policy, reference)
    if policy.get("synthetic_only") is not True or policy["geometry"]["array_side"] != 32:
        raise ValueError("Only the explicitly synthetic 32x32 registered experiment is supported")
    policy_hash = sha256(policy_bytes).hexdigest()
    # Verify cached observation physics and geometry before using its arrays.
    baseline_results_path = operator_path.parent / "results.json"
    baseline_results = json.loads(baseline_results_path.read_text(encoding="utf8"))
    cache_record = next(item for item in baseline_results["cache"] if item["file"] == operator_path.name)
    if _hash(operator_path) != cache_record["sha256"]:
        raise ValueError("Cached operator hash differs from its source record")
    with np.load(operator_path, allow_pickle=False) as data:
        if str(data["provenance"].item()) != "synthetic" or float(data["frequency_hz"]) != policy["geometry"]["frequency_hz"]:
            raise ValueError("Wrong provenance or frequency in operator cache")
        matrices = {key: data[key] for key in ("point_training", "point_validation", "point_test",
                                               "truth_training", "truth_validation", "truth_test")}
        elements = data["elements_m"]
        samples = {"training": data["samples_m"], "validation": data["validation_samples_m"], "test": data["test_samples_m"]}
    g = policy["geometry"]
    if matrices["point_training"].shape != (g["scan_side"] ** 2, g["array_side"] ** 2):
        raise ValueError("Cached matrix shape differs from registered geometry")
    if not np.allclose(samples["training"][:, 2], g["scan_distance_m"], rtol=0, atol=1e-14):
        raise ValueError("Cached scan distance differs from policy")
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy_snapshot.json").write_bytes(policy_bytes)
    (output / "reference_policy_snapshot.json").write_bytes(reference_path.read_bytes())
    progress("Calibration: preparing one reusable point-model SVD", flush=True)
    prepared = prepare_tikhonov(matrices["point_training"])
    records = []
    report = {"schema_version": 1, "synthetic_only": True, "actual_measurements_read": False,
        "in_progress": True, "policy_sha256": policy_hash, "policy": policy,
        "configuration": asdict(config), "operator_cache": {"path": str(operator_path),
            "sha256": cache_record["sha256"], "original_simulator_verification": cache_record["verification_against_simulate_measurement"]},
        "source_hashes": {name: _hash(root / "src/array_reconstruct" / name) for name in
            ("thesis_calibration.py", "inverse.py", "large_error_validation.py", "validation.py", "simulation.py")},
        "method_notes": {"controller_mean_domain": "linear amplitude; circular phase",
            "quantizer": "numpy nearest-even integer rounding; clip cumulative attenuation; phase modulo 64",
            "noise_namespace": "SeedSequence([2026102803, replicate_seed, stage_index, plane_index]) -> uint64; paired across strategies, coupling environments and physical fault states",
            "scan_cost": "Each measurement round contains one training and one validation scan; once_ap has two rounds, other strategies have four. Final clean test field is evaluation only.",
            "power_definition": "sum(abs(c)^2) is coefficient power proxy; boresight abs(sum(c))^2 is absolute scalar far-field intensity up to a common constant, never independently peak-normalized; neither is measured EIRP in watts.",
            "fixed_length": "No truth-selected stopping; table-5-2-inspired trajectory has exactly A, P, AP writes.",
            "controls_do_not_receive": ["true coefficients", "injected labels", "test fields"]},
        "agent_configuration": {"requested_model": "gpt-6.1-sol", "requested_reasoning_effort": "ultra",
            "requested_fork_turns": "none", "actual_independently_verified": None},
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "trajectories": records}
    for seed in config.seeds:
        for with_faults in config.fault_states:
            excitation = sample_excitation(len(elements), policy["scenario"], seed, with_faults)
            initial = excitation["truth"]
            sigmas = {plane: float(np.sqrt(np.mean(np.abs(matrices["truth_" + plane] @ initial) ** 2)))
                * 10 ** (-config.initial_snr_db / 20) for plane in ("training", "validation")}
            for coupling in config.coupling_deg_per_db:
                for strategy, schedule in STRATEGIES.items():
                    name = f"seed{seed}_{'faulted' if with_faults else 'healthy'}_coupling{coupling:g}_{strategy}"
                    attenuation = np.zeros(len(initial))
                    phase = np.zeros(len(initial))
                    saved = {"provenance": "synthetic", "policy_sha256": policy_hash,
                        "initial_coefficients": initial, "healthy_initial_coefficients": excitation["healthy"],
                        "normal_amplitude_error_db": excitation["amp_error_db"], "normal_phase_error_deg": excitation["phase_error_deg"],
                        "injected_fault_indices": excitation["fault_indices"], "elements_m": elements,
                        "training_samples_m": samples["training"], "validation_samples_m": samples["validation"], "test_samples_m": samples["test"]}
                    stages = []
                    for stage in range(len(schedule) + 1):
                        coefficients = apply_hardware(initial, attenuation, phase, coupling)
                        measurements = {plane: fixed_noise_measurement(matrices["truth_" + plane], coefficients,
                            sigmas[plane], _noise_seed(seed, stage, i))
                            for i, plane in enumerate(("training", "validation"))}
                        estimate, fit = fit_current_state(prepared, measurements["training"]["field"],
                            matrices["point_validation"], measurements["validation"]["field"], config.alphas)
                        # Fix commands using estimates before any true coefficient scoring.
                        mode = schedule[stage] if stage < len(schedule) else None
                        update = control_update(estimate, attenuation, phase, mode, config) if mode else None
                        _, eligible = _control_mask(estimate, config)
                        evaluation, aligned, flagged, prediction, true_test = _evaluate_stage(estimate,
                            coefficients, initial, excitation["fault_indices"], ~eligible, attenuation, phase,
                            matrices, policy, config)
                        prefix = f"stage{stage}__"
                        for key, value in {"attenuation_db": attenuation, "phase_command_deg": phase,
                            "true_coefficients": coefficients, "estimated_coefficients": estimate,
                            "aligned_estimated_coefficients": aligned, "excluded_mask": ~eligible,
                            "legacy_nominal_flag_mask": flagged, "test_predicted_clean": prediction,
                            "test_true_clean": true_test, "alpha_curve": fit["alpha_curve"], "score_curve": fit["score_curve"]}.items():
                            saved[prefix + key] = value
                        for plane, measurement in measurements.items():
                            for key in ("field", "field_clean", "noise"):
                                saved[prefix + plane + "_" + key] = measurement[key]
                        stage_record = {"stage": stage, "write_after_measurement": mode,
                            "fit": {key: value for key, value in fit.items() if not isinstance(value, np.ndarray)},
                            "measurements": {plane: {key: value for key, value in measurement.items()
                                if not isinstance(value, np.ndarray)} for plane, measurement in measurements.items()},
                            "noise_seeds": {plane: _noise_seed(seed, stage, i) for i, plane in enumerate(("training", "validation"))},
                            **evaluation}
                        if update is not None:
                            saved[prefix + "requested_attenuation_delta_db"] = update["requested_attenuation_delta_db"]
                            saved[prefix + "requested_phase_delta_deg"] = update["requested_phase_delta_deg"]
                            stage_record["control_update"] = {"linear_amplitude_target": update["linear_amplitude_target"],
                                "circular_phase_target_deg": update["circular_phase_target_deg"],
                                "phase_mean_defined": update["phase_mean_defined"],
                                "saturated_count": int(np.sum(update["saturated_mask"])),
                                "excluded_count": int(np.sum(update["excluded_mask"]))}
                            attenuation, phase = update["attenuation_db"], update["phase_command_deg"]
                        stages.append(stage_record)
                    archive_path = output / (name + ".npz")
                    np.savez_compressed(archive_path, **saved)
                    record = {"name": name, "seed": seed, "with_faults": with_faults,
                        "amplitude_to_phase_deg_per_db": coupling, "strategy": strategy,
                        "write_schedule": list(schedule), "measurement_rounds": len(stages),
                        "full_training_and_validation_scans": 2 * len(stages),
                        "training_observation_count": len(stages) * len(samples["training"]),
                        "validation_observation_count": len(stages) * len(samples["validation"]),
                        "initial_noise_sigmas": sigmas, "stages": stages, "final": stages[-1],
                        "archive": archive_path.name, "archive_sha256": _hash(archive_path)}
                    records.append(record)
                    report["elapsed_seconds"] = perf_counter() - started
                    _write_json(output / "results.json", report)
                    progress(f"Calibration [{len(records)}/36] {name}: normal amplitude std={record['final']['noninjected_true_metrics']['amplitude_std_db']:.3f} dB, phase std={record['final']['noninjected_true_metrics']['phase_circular_std_deg']:.3f} deg, boresight change={record['final']['boresight_power_change_db']:.3f} dB", flush=True)
    groups = defaultdict(list)
    for record in records:
        groups[(record["with_faults"], record["amplitude_to_phase_deg_per_db"], record["strategy"])].append(record)
    group_summary = []
    for (faulted, coupling, strategy), values in groups.items():
        summary = {"with_faults": faulted, "amplitude_to_phase_deg_per_db": coupling,
            "strategy": strategy, "replicate_count": len(values), "seeds": [item["seed"] for item in values],
            "measurement_rounds_per_trajectory": values[0]["measurement_rounds"], "metrics": {}}
        paths = {"true_noninjected_amplitude_std_db": ("noninjected_true_metrics", "amplitude_std_db"),
            "true_noninjected_phase_circular_std_deg": ("noninjected_true_metrics", "phase_circular_std_deg"),
            "true_noninjected_amplitude_rmse_vs_unit_nominal_db": ("noninjected_true_metrics", "amplitude_rmse_vs_unit_nominal_db"),
            "reconstruction_complex_nrmse": ("reconstruction_metrics", "complex_nrmse"),
            "reconstruction_amplitude_rmse_db": ("reconstruction_metrics", "active_amplitude_rmse_db"),
            "reconstruction_phase_rmse_deg": ("reconstruction_metrics", "active_phase_rmse_deg")}
        for key in ("coefficient_power_ratio_vs_initial", "coefficient_power_change_db", "boresight_absolute_power_ratio_vs_initial",
                    "boresight_power_change_db", "coherent_efficiency", "independent_test_clean_field_nrmse"):
            paths[key] = (key,)
        for key, path in paths.items():
            numbers = []
            for value in values:
                leaf = value["final"]
                for component in path:
                    leaf = leaf[component]
                numbers.append(float(leaf))
            summary["metrics"][key] = {"mean": float(np.mean(numbers)), "minimum": min(numbers), "maximum": max(numbers)}
        summary["coefficient_pass_count"] = sum(value["final"]["coefficient_acceptance_passed"] for value in values)
        summary["normal_false_exclusions_sum"] = sum(value["final"]["control_exclusion"]["normal_false_exclusion_count"] for value in values)
        group_summary.append(summary)
    report.update({"in_progress": False, "completed_trajectory_count": len(records),
        "completed_state_estimates": sum(len(record["stages"]) for record in records),
        "elapsed_seconds": perf_counter() - started, "group_summary": group_summary})
    _write_json(output / "results.json", report)
    _write_json(output / "group_summary.json", {"policy_sha256": policy_hash, "groups": group_summary})
    _write_json(output / "execution_receipt.json", {"synthetic_only": True, "actual_measurements_read": False,
        "policy_sha256": policy_hash, "reference_policy_sha256": _hash(reference_path),
        "results_sha256": _hash(output / "results.json"), "source_hashes": report["source_hashes"],
        "completed_trajectories": len(records), "completed_state_estimates": report["completed_state_estimates"],
        "runtime_seconds": report["elapsed_seconds"], "new_software_installed": False,
        "implementation_details": report["method_notes"], "agent_configuration": report["agent_configuration"]})
    return report
