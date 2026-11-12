"""Paper-inspired scalar ASM and physical-cell proxies, for virtual tests only.

The cell statistics are bandwidth-limited aperture-field proxies. Interpreting
them as coefficients in the point-source dictionary is an explicit diagnostic
comparison, not an identity or a feed-port calibration. Fourier resampling only
integrates the measured band-limited field; padding supplies no new information.
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
from numpy.polynomial.legendre import leggauss
import scipy

from .inverse import prepare_tikhonov, solve_tikhonov
from .large_error_validation import (cached_measurement, cached_truth_matrix,
    estimate_common_gain, measurement_seeds, sample_excitation, score_estimate)
from .physics import angular_spectrum_backprop, green_matrix
from .validation import array_geometry, scan_geometry


METHODS = ("asm_complex_global", "asm_complex_reference",
           "asm_amp_circular_reference", "ridge_nominal_centered", "nominal_common_gain")


def _relative(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1e-300))


def cell_bounds(centres_xy_m, width_m):
    """Known physical square cells, half-open at upper x/y boundaries.

    Adjacent rows may shift in x. They never overlap in area when their y pitch
    is the cell width. The half-open convention assigns every shared edge once.
    """
    centres = np.asarray(centres_xy_m, dtype=float)
    width = float(width_m)
    if centres.ndim != 2 or centres.shape[1] != 2 or not np.all(np.isfinite(centres)):
        raise ValueError("centres_xy_m must be a finite (cells,2) array")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width_m must be positive")
    return np.stack((centres - width / 2, centres + width / 2), axis=1)


def cell_owners(points_xy_m, bounds_m):
    """Return cell index, or -1 outside; reject overlapping half-open cells."""
    points = np.asarray(points_xy_m, dtype=float)
    bounds = np.asarray(bounds_m, dtype=float)
    membership = np.all((points[:, None, :] >= bounds[None, :, 0, :]) &
                        (points[:, None, :] < bounds[None, :, 1, :]), axis=2)
    if np.any(membership.sum(axis=1) > 1):
        raise ValueError("Cells overlap at one or more supplied points")
    return np.where(membership.any(axis=1), membership.argmax(axis=1), -1)


@dataclass
class FourierCellSampler:
    """Separable exact evaluation of the finite FFT series at Gauss nodes.

    The input spectrum is fftshift-centred and unnormalised, matching physics.py.
    x0/y0 are the physical coordinates of padded array index zero. No image-pixel
    lookup or phase interpolation is used. The fixed Gauss rule approximates the
    area statistics, with normalised weights summing to one for each cell.
    """
    basis_x: np.ndarray
    basis_y: np.ndarray
    indices_x: np.ndarray
    indices_y: np.ndarray
    weights: np.ndarray
    points_xy_m: np.ndarray
    bounds_m: np.ndarray
    spectrum_shape: tuple

    @classmethod
    def create(cls, centres_xy_m, width_m, kx, ky, x0_m, y0_m, quadrature_order=3):
        if isinstance(quadrature_order, bool) or quadrature_order not in (3, 5):
            raise ValueError("Cell quadrature order must be fixed at 3 or 5")
        bounds = cell_bounds(centres_xy_m, width_m)
        centres = np.asarray(centres_xy_m, dtype=float)
        nodes, weights = leggauss(quadrature_order)
        xx, yy = np.meshgrid(nodes * width_m / 2, nodes * width_m / 2, indexing="xy")
        offsets = np.column_stack((xx.ravel(), yy.ravel()))
        points = centres[:, None, :] + offsets[None, :, :]
        x, ix = np.unique(points[..., 0], return_inverse=True)
        y, iy = np.unique(points[..., 1], return_inverse=True)
        kx, ky = np.asarray(kx), np.asarray(ky)
        return cls(np.exp(1j * (x[:, None] - x0_m) * kx),
                   np.exp(1j * (y[:, None] - y0_m) * ky),
                   ix.reshape(points.shape[:2]), iy.reshape(points.shape[:2]),
                   np.outer(weights / 2, weights / 2).ravel(), points, bounds,
                   (len(ky), len(kx)))

    def evaluate(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        if spectrum.shape != self.spectrum_shape or not np.all(np.isfinite(spectrum)):
            raise ValueError("Spectrum has inconsistent shape or nonfinite values")
        grid = self.basis_y @ spectrum @ self.basis_x.T / np.prod(self.spectrum_shape)
        return grid[self.indices_y, self.indices_x]

    def complex_mean(self, spectrum):
        return self.evaluate(spectrum) @ self.weights


def amplitude_circular_mean(values, weights, phase_relative_floor=1e-12):
    """Mean linear amplitude and a weighted *circular* mean of local phases.

    Very small local samples do not define phase. A vanishing circular resultant
    marks an undefined phase rather than assigning a fictitious angle.
    """
    values = np.asarray(values, dtype=complex)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("Expected (cells,nodes) values and one weight per node")
    if not np.all(np.isfinite(values)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Values must be finite and normalised weights nonnegative")
    amplitude = np.abs(values)
    active = amplitude > phase_relative_floor * amplitude.max(axis=1, keepdims=True)
    units = np.divide(values, amplitude, out=np.zeros_like(values), where=active)
    phase_weight = active @ weights
    resultant = units @ weights
    valid = (phase_weight > 0) & (np.abs(resultant) > 1e-12 * phase_weight)
    proxy = (amplitude @ weights) * np.exp(1j * np.angle(resultant))
    proxy[~valid] = 0
    coherence = np.divide(np.abs(resultant), phase_weight, out=np.zeros_like(phase_weight), where=phase_weight > 0)
    return proxy, valid, coherence


def reference_normalize(proxy, reference, nominal, relative_floor=1e-6, proxy_valid=None):
    """Use only the known nominal reference; mask small references explicitly.

    Masked coefficient proxies are set to zero and remain in full-vector scoring
    and field prediction, so masking cannot silently improve a reported score.
    """
    proxy, reference, nominal = (np.asarray(x, dtype=complex) for x in (proxy, reference, nominal))
    if proxy.shape != reference.shape or proxy.shape != nominal.shape:
        raise ValueError("Proxy, nominal and reference must have equal shape")
    if relative_floor <= 0 or not np.isfinite(relative_floor):
        raise ValueError("Reference floor must be positive")
    threshold = float(relative_floor * np.max(np.abs(reference)))
    valid = np.abs(reference) > max(threshold, np.finfo(float).tiny)
    if proxy_valid is not None:
        valid &= np.asarray(proxy_valid, dtype=bool)
    result = np.zeros_like(proxy)
    np.divide(proxy * nominal, reference, out=result, where=valid)
    return result, valid, threshold


def training_gain(matrix, field, coefficient_proxy):
    """Fit one complex scale on training observations, never coefficient truth."""
    predicted = np.asarray(matrix) @ np.asarray(coefficient_proxy)
    denominator = float(np.vdot(predicted, predicted).real)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("Proxy has no observable training field")
    return complex(np.vdot(predicted, field) / denominator)


def prepare_aperture(elements, scan_shape, step_m, frequency_hz, distance_m,
                     nominal_field, cell_width_m=.011, quadrature_order=5,
                     padding_factor=3, max_angle_deg=90.0):
    """Freeze known geometry and nominal point-model reference before cases."""
    nominal_asm = angular_spectrum_backprop(np.asarray(nominal_field).reshape(scan_shape),
        step_m, step_m, frequency_hz, distance_m, padding_factor=padding_factor,
        max_angle_deg=max_angle_deg, window="none")
    before_y = nominal_asm["crop_slices"][0].start
    before_x = nominal_asm["crop_slices"][1].start
    x0 = -((scan_shape[1] - 1) / 2 + before_x) * step_m
    y0 = -((scan_shape[0] - 1) / 2 + before_y) * step_m
    sampler = FourierCellSampler.create(np.asarray(elements)[:, :2], cell_width_m,
        nominal_asm["kx"], nominal_asm["ky"], x0, y0, quadrature_order)
    values = sampler.evaluate(nominal_asm["spectrum"])
    circular, circular_valid, coherence = amplitude_circular_mean(values, sampler.weights)
    return {"sampler": sampler, "reference_complex": values @ sampler.weights,
            "reference_amp_circular": circular, "reference_circular_valid": circular_valid,
            "reference_circular_coherence": coherence, "reference_values": values,
            "reference_asm": nominal_asm, "scan_shape": tuple(scan_shape),
            "step_m": step_m, "frequency_hz": frequency_hz, "distance_m": distance_m,
            "padding_factor": padding_factor, "max_angle_deg": max_angle_deg}


def fit_aperture(prepared, field, point_matrix, nominal, relative_floor=1e-6):
    """Fit A/B/C using only training field, known geometry and nominal model."""
    asm = angular_spectrum_backprop(np.asarray(field).reshape(prepared["scan_shape"]),
        prepared["step_m"], prepared["step_m"], prepared["frequency_hz"], prepared["distance_m"],
        padding_factor=prepared["padding_factor"], max_angle_deg=prepared["max_angle_deg"], window="none")
    sampler = prepared["sampler"]
    values = sampler.evaluate(asm["spectrum"])
    raw_complex = values @ sampler.weights
    raw_circular, circular_valid, coherence = amplitude_circular_mean(values, sampler.weights)
    normalized_complex, mask_complex, floor_complex = reference_normalize(
        raw_complex, prepared["reference_complex"], nominal, relative_floor)
    normalized_circular, mask_circular, floor_circular = reference_normalize(
        raw_circular, prepared["reference_amp_circular"], nominal, relative_floor,
        circular_valid & prepared["reference_circular_valid"])
    proxies = {"asm_complex_global": raw_complex,
               "asm_complex_reference": normalized_complex,
               "asm_amp_circular_reference": normalized_circular}
    fitted = {}
    for name, proxy in proxies.items():
        gain = training_gain(point_matrix, field, proxy)
        fitted[name] = {"coefficients": gain * proxy, "common_gain": [gain.real, gain.imag],
                       "selection_method": "training_only_global_complex_gain", "selected_alpha": None,
                       "raw_proxy_is_port_coefficient": False,
                       "training_field_nrmse": _relative(point_matrix @ (gain * proxy), field)}
    return fitted, {"asm_image_padded": asm["image_padded"], "asm_spectrum": asm["spectrum"],
                   "cell_quadrature_values": values, "raw_proxy_complex": raw_complex,
                   "raw_proxy_amp_circular": raw_circular, "proxy_circular_valid": circular_valid,
                   "proxy_circular_coherence": coherence, "reference_normalized_complex": normalized_complex,
                   "reference_normalized_amp_circular": normalized_circular,
                   "reference_valid_complex": mask_complex, "reference_valid_amp_circular": mask_circular,
                   "reference_floor_complex": np.asarray(floor_complex),
                   "reference_floor_amp_circular": np.asarray(floor_circular)}


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _resolve_policy(policy, split):
    """Accept the unified policy's common sections plus aperture/split sections."""
    settings = policy.get("aperture", {})
    scenario = policy.get("scenario")
    if scenario is None:
        scenario = next(x for x in policy["scenarios"] if x["name"] == "span5_phase5")
    seed_key = "confirmation" if split == "confirmatory" else split
    seeds = policy.get(seed_key + "_seeds", policy.get("splits", {}).get(split))
    if isinstance(seeds, dict):
        seeds = seeds["seeds"]
    if seeds is None:
        raise ValueError(f"Policy does not contain explicit {split} seeds")
    return settings, scenario, [int(seed) for seed in seeds]


def run_thesis_aperture(output, policy_path, split, progress=print):
    """Run a locked synthetic benchmark, refusing to overwrite prior outputs."""
    if split not in ("development", "confirmatory"):
        raise ValueError("split must be development or confirmatory")
    output, policy_path = Path(output), Path(policy_path)
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    project_root = Path(__file__).resolve().parents[2]
    reference_path = project_root / policy["reference_policy"]
    reference_policy = json.loads(reference_path.read_bytes())
    effective = {**reference_policy, **policy}
    effective["measurement_snr_db"] = policy["initial_snr_db"]
    # Development is precisely the preceding registered three replicates, not
    # a newly sampled development set or any confirmation seed.
    effective.setdefault("development_seeds", reference_policy["seeds"])
    settings, scenario, seeds = _resolve_policy(effective, split)
    config_hash = sha256(policy_bytes).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy_snapshot.json").write_bytes(policy_bytes)
    g = policy["geometry"]
    f = g["frequency_hz"]
    q = settings.get("cell_quadrature_order", 5)
    padding = settings.get("padding_factor", 3)
    angle = settings.get("max_angle_deg", 90.)
    floor = settings.get("reference_relative_floor", 1e-6)
    if padding != 3 or q != 5 or angle != 90:
        raise ValueError("Locked experiment requires factor-3 padding, 5x5 quadrature, all propagating modes")
    if "5x5" not in settings["cell"] or g["array_side"] != 32 or g["scan_side"] != 63 or f != 11.7e9:
        raise ValueError("Unified policy does not declare the frozen aperture experiment")
    if effective["measurement_snr_db"] != 40 or floor != 1e-6:
        raise ValueError("Frozen initial SNR and reference floor are 40 dB and 1e-6")
    if split == "confirmatory" and set(seeds) & set(reference_policy["seeds"]):
        raise ValueError("Confirmation seeds must be disjoint from development")
    elements = array_geometry(g["array_side"], g["pitch_m"])
    samples = scan_geometry(g["scan_side"], g["scan_step_m"], g["scan_distance_m"])
    validation = samples[::3].copy(); validation[:, 0] += g["validation_x_offset_m"]
    test = samples[1::3].copy(); test[:, 1] += g["test_y_offset_m"]; test[:, 2] += g["test_z_offset_m"]
    planes = {"training": samples, "validation": validation, "test": test}
    nominal = np.ones(len(elements), dtype=complex)
    started = perf_counter()
    progress("Preparing identical point dictionaries and independent finite-aperture truth", flush=True)
    matrices = {name: green_matrix(elements, points, f) for name, points in planes.items()}
    truth_matrices = {name: cached_truth_matrix(elements, points, f,
        policy["truth"]["source_aperture_wavelength_fraction"], policy["truth"]["quadrature_order"])
        for name, points in planes.items()}
    ridge = prepare_tikhonov(matrices["training"])
    aperture = prepare_aperture(elements, (g["scan_side"],) * 2, g["scan_step_m"], f,
        g["scan_distance_m"], matrices["training"] @ nominal, g["pitch_m"], q, padding, angle)
    reference = aperture["reference_asm"]
    np.savez_compressed(output / "reference.npz", provenance="synthetic", config_sha256=config_hash,
        elements_m=elements, samples_m=samples, validation_samples_m=validation, test_samples_m=test,
        cell_bounds_m=aperture["sampler"].bounds_m, cell_quadrature_xy_m=aperture["sampler"].points_xy_m,
        cell_quadrature_weights=aperture["sampler"].weights,
        nominal_point_field=matrices["training"] @ nominal,
        reference_complex=aperture["reference_complex"], reference_amp_circular=aperture["reference_amp_circular"],
        reference_quadrature_values=aperture["reference_values"],
        reference_circular_valid=aperture["reference_circular_valid"],
        reference_circular_coherence=aperture["reference_circular_coherence"],
        reference_asm_image_padded=reference["image_padded"], reference_asm_spectrum=reference["spectrum"],
        kx=reference["kx"], ky=reference["ky"], retained_mask=reference["retained_mask"],
        singular_values=ridge.singular_values, column_scale=ridge.column_scale)
    cases = [{"name": f"seed{seed}_{'faulted' if faults else 'healthy'}", "seed": seed,
              "with_faults": faults} for seed in seeds for faults in (False, True)]
    _write_json(output / "case_matrix.json", {"split": split, "config_sha256": config_hash, "cases": cases})
    report = {"schema_version": 1, "synthetic_only": True, "actual_measurements_read": False,
        "split": split, "in_progress": True, "config_sha256": config_hash, "policy": policy,
        "reference_policy_sha256": _hash(reference_path),
        "resolved_seeds": seeds, "resolved_alphas": effective["alphas"],
        "resolved_scoring_limits": effective["independent_model_limits"],
        "derived_runtime_configuration": {"cell_quadrature_order": q, "cell_quadrature_count": q*q,
            "cell_width_m": g["pitch_m"], "padding_factor": padding, "max_angle_deg": angle,
            "reference_relative_floor": floor, "measurement_snr_db": effective["measurement_snr_db"],
            "derivation": "root aperture cell specifies 5x5; propagation specifies factor 3 and propagating-only; root initial_snr_db; alpha and legacy scoring from reference_policy"},
        "reference_archive_sha256": _hash(output / "reference.npz"),
        "definition": {"raw_proxy": "area statistic of scalar propagating-only ASM aperture field",
            "complex_mean": "normalised fixed Gauss area mean of complex Fourier-resampled field",
            "amp_circular": "linear amplitude mean times exp(i*weighted circular phase mean)",
            "cell_boundaries": "known-centre square, half-open upper boundaries; staggered rows do not overlap",
            "normalization": "B/C divide their own known nominal point-field pipeline reference, then fit training common gain",
            "gain": "all ASM coefficient proxies globally scaled by least squares to training field only",
            "scoring": "proxy-as-point-coefficient accuracy is diagnostic; test field is independent physical consistency",
            "registration": "known geometry; no Sobel edge search, no fitted alignment, no unknown probe compensation",
            "truth_field_consistency": "forward prediction at independent test plane; z=0 finite-source self-field is not used as an aperture truth",
            "masked_cells": "zero filled and retained in all coefficient and field scores",
            "resampling": "finite FFT Fourier-series evaluation at fixed Gauss nodes; padding is not sampling refinement"},
        "source_hashes": {name: _hash(Path(__file__).with_name(name)) for name in
            ("thesis_aperture.py", "physics.py", "large_error_validation.py", "inverse.py", "validation.py")},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "thread_environment": {k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "agent_configuration": {"requested": {"model": "gpt-6.1-sol", "reasoning_effort": "ultra", "fork_turns": "none"},
                                 "actual_independently_verified": None},
        "preparation": {"seconds": perf_counter() - started,
            "column_scaled_condition_number": float(ridge.singular_values[0] / ridge.singular_values[-1]),
            "numerical_rank": int(np.sum(ridge.singular_values > ridge.rank_tolerance)),
            "retained_spectral_modes": int(reference["retained_mask"].sum()),
            "padding_factor": padding, "cell_quadrature_order": q, "cell_quadrature_count": q * q},
        "cases": []}
    for config in cases:
        t0 = perf_counter()
        excitation = sample_excitation(len(elements), scenario, config["seed"], config["with_faults"])
        seeds_plane = measurement_seeds(config["seed"])
        measurements = {name: cached_measurement(truth_matrices[name], excitation["truth"], seeds_plane[name],
            effective["measurement_snr_db"]) for name in planes}
        field = measurements["training"]["field"]
        fitted, arrays = fit_aperture(aperture, field, matrices["training"], nominal, floor)
        gain = estimate_common_gain(matrices["training"], field, nominal)
        base = gain * nominal
        fitted["nominal_common_gain"] = {"coefficients": base, "common_gain": [gain.real, gain.imag],
            "selection_method": "training_only_global_complex_gain", "selected_alpha": None,
            "training_field_nrmse": _relative(matrices["training"] @ base, field)}
        fit = solve_tikhonov(ridge, field, prior=base, alphas=effective["alphas"],
            validation_matrix=matrices["validation"], validation_y=measurements["validation"]["field"])
        fitted["ridge_nominal_centered"] = {"coefficients": fit.coefficients,
            "common_gain": [gain.real, gain.imag], "selection_method": fit.selection_method,
            "selected_alpha": fit.alpha, "training_field_nrmse": fit.relative_residual,
            "selection_field_nrmse": fit.validation_relative_error}
        saved = {"provenance": "synthetic", "config_sha256": config_hash, "frequency_hz": f,
            "true_coefficients": excitation["truth"], "healthy_coefficients": excitation["healthy"], "nominal": nominal,
            "normal_amplitude_error_db": excitation["amp_error_db"], "normal_phase_error_deg": excitation["phase_error_deg"],
            "injected_fault_indices": excitation["fault_indices"], "paired_fault_indices": excitation["paired_fault_indices"],
            "fault_multipliers": excitation["fault_multipliers"], "ridge_alpha_curve": fit.alpha_curve,
            "ridge_score_curve": fit.score_curve,
            "reference_complex": aperture["reference_complex"],
            "reference_amp_circular": aperture["reference_amp_circular"], **arrays}
        for plane, prefix in (("training", ""), ("validation", "validation_"), ("test", "test_")):
            for key in ("field", "field_clean", "noise"):
                saved[prefix + key] = measurements[plane][key]
        record = {**config, "frequency_hz": f, "scenario": scenario, "measurement_seeds": seeds_plane,
            "truth_used_for_model_selection": False, "methods": {},
            "reference_masking": {"complex_invalid_count": int(np.sum(~arrays["reference_valid_complex"])),
                "amp_circular_invalid_count": int(np.sum(~arrays["reference_valid_amp_circular"])),
                "complex_floor": float(arrays["reference_floor_complex"]),
                "amp_circular_floor": float(arrays["reference_floor_amp_circular"]),
                "relative_floor": floor}}
        for name in METHODS:
            fit_record = fitted[name]
            estimate = fit_record["coefficients"]
            scores, aligned, flagged, oracle = score_estimate(estimate, excitation["truth"], nominal,
                excitation["fault_indices"], effective)
            prediction = matrices["test"] @ estimate
            method = {k: v for k, v in fit_record.items() if k != "coefficients"}
            method.update(scores)
            method.update({"independent_test_clean_field_nrmse": _relative(prediction, measurements["test"]["field_clean"]),
                "independent_test_noisy_field_nrmse": _relative(prediction, measurements["test"]["field"]),
                "coefficient_score_interpretation": "proxy-as-point-coefficient diagnostic" if name.startswith("asm_") else "point-model coefficient diagnostic"})
            record["methods"][name] = method
            saved["coefficients__" + name] = estimate
            saved["aligned_coefficients__" + name] = aligned
            saved["flagged_mask__" + name] = flagged
            saved["test_predicted__" + name] = prediction
        # Clean-vs-noisy proxy stability is a field-estimator diagnostic only;
        # it is not an aperture z=0 truth or a port-coefficient accuracy score.
        _, clean_arrays = fit_aperture(aperture, measurements["training"]["field_clean"], matrices["training"], nominal, floor)
        record["raw_proxy_noise_stability"] = {"complex_nrmse": _relative(arrays["raw_proxy_complex"], clean_arrays["raw_proxy_complex"]),
            "amp_circular_nrmse": _relative(arrays["raw_proxy_amp_circular"], clean_arrays["raw_proxy_amp_circular"])}
        saved["clean_raw_proxy_complex"] = clean_arrays["raw_proxy_complex"]
        saved["clean_raw_proxy_amp_circular"] = clean_arrays["raw_proxy_amp_circular"]
        saved["oracle_flag_mask"] = oracle
        saved["scores_json"] = json.dumps(record, ensure_ascii=False, allow_nan=False)
        archive = output / (config["name"] + ".npz")
        np.savez_compressed(archive, **saved)
        record.update({"archive": archive.name, "archive_sha256": _hash(archive), "seconds": perf_counter() - t0})
        report["cases"].append(record)
        report["total_seconds"] = perf_counter() - started
        _write_json(output / "results.json", report)
        progress(f"[{len(report['cases'])}/{len(cases)}] {config['name']}: " + ", ".join(
            f"{name} coeff={record['methods'][name]['metrics']['complex_nrmse']:.4f} test={record['methods'][name]['independent_test_clean_field_nrmse']:.4f}"
            for name in METHODS), flush=True)
    report["in_progress"] = False
    report["completed_case_count"] = len(cases)
    report["completed_method_fits"] = len(cases) * len(METHODS)
    report["total_seconds"] = perf_counter() - started
    report["summary"] = {name: {"coefficient_pass_count": sum(c["methods"][name]["coefficient_acceptance_passed"] for c in report["cases"]),
        "legacy_joint_pass_count": sum(c["methods"][name]["legacy_joint_acceptance_passed"] for c in report["cases"]),
        "median_complex_nrmse": float(np.median([c["methods"][name]["metrics"]["complex_nrmse"] for c in report["cases"]])),
        "median_independent_test_clean_field_nrmse": float(np.median([c["methods"][name]["independent_test_clean_field_nrmse"] for c in report["cases"]]))}
        for name in METHODS}
    _write_json(output / "results.json", report)
    fields = ["case", "seed", "with_faults", "method", "active_amplitude_rmse_db", "active_phase_rmse_deg", "complex_nrmse",
        "coefficient_acceptance_passed", "legacy_joint_acceptance_passed", "independent_test_clean_field_nrmse", "independent_test_noisy_field_nrmse", "selected_alpha"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in report["cases"]:
            for name, method in case["methods"].items():
                row = {"case": case["name"], "seed": case["seed"], "with_faults": case["with_faults"], "method": name}
                row.update({key: method["metrics"][key] for key in fields[4:7]})
                row.update({key: method[key] for key in fields[7:]})
                writer.writerow(row)
    return report
