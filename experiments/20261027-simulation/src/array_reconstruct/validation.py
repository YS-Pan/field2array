"""Reproducible synthetic experiments; this module never opens measured data."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .physics import green_matrix, angular_spectrum_backprop
from .inverse import prepare_tikhonov, solve_tikhonov
from .simulation import simulate_measurement

C0 = 299792458.0


def array_geometry(side: int, pitch: float = 0.011) -> np.ndarray:
    """Reference staggered layout, row-major; translation to its centroid."""
    row, col = np.indices((side, side))
    points = np.column_stack(((pitch * col + pitch / 2 * (row % 2)).ravel(),
                              (pitch * row).ravel(), np.zeros(side * side)))
    points[:, :2] -= points[:, :2].mean(axis=0)
    return points


def scan_geometry(side: int, step: float = 0.00835, z: float = 0.06) -> np.ndarray:
    axis = (np.arange(side) - (side - 1) / 2) * step
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def excitation(positions, frequency_hz, seed, faults=True, beam_deg=0, random_faults=False):
    rng = np.random.default_rng(seed)
    n = len(positions)
    nominal = np.exp(1j * 2 * np.pi * frequency_hz / C0 * positions[:, 0] * np.sin(np.deg2rad(beam_deg)))
    truth = nominal * 10 ** (rng.normal(0, 0.10, n) / 20) * np.exp(1j * np.deg2rad(rng.normal(0, 1, n)))
    indices = []
    if faults:
        indices = sorted({k for k in (0, n // 3, n // 2 + 2, n - 2, n - 3) if 0 <= k < n})
        for k in range(n):
            if len(indices) >= min(5, n):
                break
            if k not in indices:
                indices.append(k)
        indices.sort()
        if random_faults:
            indices = sorted(rng.choice(n, min(5, n), replace=False).tolist())
        truth[indices[0]] = 0
        if len(indices) > 1:
            truth[indices[1]] *= 10 ** (-6 / 20)
        if len(indices) > 2:
            truth[indices[2]] *= np.exp(1j * np.deg2rad(60))
        truth[indices[3:]] *= 10 ** (-6 / 20) * np.exp(1j * np.deg2rad(40))
    return nominal, truth, np.asarray(indices, dtype=int)


def evaluate_coefficients(estimate, truth, nominal, fault_indices):
    """Oracle alignment is evaluation-only; detection uses nominal robust gauge."""
    scalar = np.vdot(estimate, truth) / max(np.vdot(estimate, estimate).real, 1e-300)
    aligned = scalar * estimate
    active = (np.abs(truth) > 0) & (np.abs(truth) >= 0.1 * np.max(np.abs(truth)))
    amp_error = 20 * np.log10(np.maximum(np.abs(aligned[active]), 1e-15) / np.abs(truth[active]))
    phase_active = active & (np.abs(aligned) > 0) & (np.abs(aligned) >= 0.1 * np.max(np.abs(truth)))
    phase_error = np.rad2deg(np.angle(aligned[phase_active] * np.conj(truth[phase_active])))
    ratios = estimate / nominal
    scale = np.median(np.abs(ratios))
    phase = np.angle(np.sum(ratios / np.maximum(np.abs(ratios), 1e-15)))
    observed = ratios / max(scale, 1e-15) * np.exp(-1j * phase)
    adb = 20 * np.log10(np.maximum(np.abs(observed), 1e-15))
    pdeg = np.rad2deg(np.angle(observed))
    flagged = (np.abs(adb) >= 3) | ((np.abs(pdeg) >= 20) & (np.abs(observed) >= 0.1))
    actual = np.zeros(len(truth), dtype=bool)
    actual[fault_indices] = True
    tp = int(np.sum(flagged & actual)); fp = int(np.sum(flagged & ~actual))
    fn = int(np.sum(~flagged & actual))
    metrics = {
        "active_amplitude_rmse_db": float(np.sqrt(np.mean(amp_error ** 2))) if np.any(active) else float("inf"),
        "active_phase_rmse_deg": float(np.sqrt(np.mean(phase_error ** 2))) if np.any(phase_active) else float("inf"),
        "complex_nrmse": float(np.linalg.norm(aligned - truth) / max(np.linalg.norm(truth), 1e-300)),
        "unaligned_complex_nrmse": float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-300)),
        "fault_precision": tp / (tp + fp) if tp + fp else (1.0 if not len(fault_indices) else 0.0),
        "fault_recall": tp / (tp + fn) if tp + fn else 1.0,
        "healthy_false_positive_rate": fp / max(int(np.sum(~actual)), 1),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "flagged_indices": np.flatnonzero(flagged).tolist(),
        "actual_fault_indices": fault_indices.tolist(),
        "alignment_complex": [float(scalar.real), float(scalar.imag)],
        "active_phase_count": int(np.sum(phase_active)),
        "phase_unscored_count": int(np.sum(~phase_active)),
    }
    return metrics, aligned, flagged


def accepts(metrics, policy):
    limits = policy["independent_model_limits"]
    return all(metrics[key] >= value if key in ("fault_precision", "fault_recall") else metrics[key] <= value
               for key, value in limits.items())


def _plot_case(output, name, side, truth, aligned, measured, scan_side, field_image):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.subplots_adjust(left=.065, right=.97, bottom=.09, top=.89, wspace=.40, hspace=.32)
    images = [20 * np.log10(np.maximum(np.abs(truth), 1e-3)).reshape(side, side),
              20 * np.log10(np.maximum(np.abs(aligned), 1e-3)).reshape(side, side),
              np.abs(aligned - truth).reshape(side, side),
              np.rad2deg(np.angle(truth)).reshape(side, side),
              np.rad2deg(np.angle(aligned)).reshape(side, side),
              np.abs(field_image)]
    titles = ["True amplitude (dB)", "Recovered amplitude (dB)", "Complex coefficient error",
              "True phase (deg)", "Recovered phase (deg)", "Back-propagated field\n(not port weights)"]
    for i, (ax, data, title) in enumerate(zip(axes.flat, images, titles)):
        settings = {"vmin": -40, "vmax": 1, "cmap": "viridis"} if i < 2 else ({"vmin": -180, "vmax": 180, "cmap": "twilight"} if i in (3, 4) else {"cmap": "magma"})
        if i in (3, 4):
            data = data.copy()
            mask = np.abs(truth).reshape(side, side) < .1
            if i == 4:
                mask |= np.abs(aligned).reshape(side, side) < .1
            data[mask] = np.nan
        item = ax.imshow(data, origin="lower", **settings)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("column / scan x"); ax.set_ylabel("row / scan y")
        fig.colorbar(item, ax=ax, shrink=.8)
    fig.suptitle(name + " | synthetic data only", fontsize=13)
    fig.savefig(output / (name + ".png"), dpi=150)
    plt.close(fig)


def run_validation(output, policy_path, full=False, seed_base=20261027, progress=print, random_faults=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(policy_path).read_text(encoding="utf8"))
    configs = [
        {"name": "point_source_sanity", "side": 8, "frequency_hz": 11.7e9, "source_fraction": 0., "snr_db": None},
        {"name": "independent_10p7GHz", "side": 8, "frequency_hz": 10.7e9},
        {"name": "independent_11p7GHz", "side": 8, "frequency_hz": 11.7e9},
        {"name": "independent_12p7GHz", "side": 8, "frequency_hz": 12.7e9},
        {"name": "steered_and_common_gain", "side": 8, "frequency_hz": 11.7e9, "beam_deg": 20., "gain": 1.7 * np.exp(.8j)},
        {"name": "mild_geometry_probe_mismatch", "side": 8, "frequency_hz": 11.7e9,
         "jitter_m": .00005, "distance_bias_m": .00005, "probe_fraction": .1, "source_fraction": .12},
        {"name": "stress_wrong_distance_3mm", "side": 8, "frequency_hz": 11.7e9,
         "distance_bias_m": .003, "stress": True},
        {"name": "stress_10dB_noise", "side": 8, "frequency_hz": 11.7e9, "snr_db": 10, "stress": True},
    ]
    if full:
        configs += [{"name": "full_32x32_11p7GHz", "side": 32, "frequency_hz": 11.7e9},
                    {"name": "full_32x32_healthy", "side": 32, "frequency_hz": 11.7e9, "faults": False},
                    {"name": "full_32x32_10p7GHz", "side": 32, "frequency_hz": 10.7e9},
                    {"name": "full_32x32_12p7GHz", "side": 32, "frequency_hz": 12.7e9}]
    configs += [{"name": "healthy_8x8", "side": 8, "frequency_hz": 11.7e9, "faults": False},
                {"name": "sensitivity_30dB_noise", "side": 8, "frequency_hz": 10.7e9, "snr_db": 30, "stress": True}]
    cache = {}; cases = []
    t0 = perf_counter()
    for i, config in enumerate(configs):
        start = perf_counter(); name = config["name"]; side = config["side"]; frequency = config["frequency_hz"]
        progress(f"Starting {name}", flush=True)
        elements = array_geometry(side)
        scan_side = 33 if side == 8 else 63
        samples = scan_geometry(scan_side)
        # Independent coordinates: selection plane displaced laterally by a
        # half step; final test plane is at a distinct, known z and offset.
        validation_samples = samples[::3].copy(); validation_samples[:, 0] += .004175
        test_samples = samples[1::3].copy(); test_samples[:, 1] -= .0027; test_samples[:, 2] += .007
        key = (side, frequency)
        if key not in cache:
            matrix = green_matrix(elements, samples, frequency)
            av = green_matrix(elements, validation_samples, frequency)
            at = green_matrix(elements, test_samples, frequency)
            prepared = prepare_tikhonov(matrix)
            cache[key] = (matrix, av, at, prepared)
        matrix, av, at, prepared = cache[key]
        nominal, feed, faults = excitation(elements, frequency, seed_base + i,
                                          faults=config.get("faults", True), beam_deg=config.get("beam_deg", 0),
                                          random_faults=random_faults)
        sim_config = {
            "source_aperture_m": config.get("source_fraction", .1) * C0 / frequency,
            "source_quadrature_order": 3, "directionality_power": 0.,
            "probe_aperture_m": config.get("probe_fraction", 0.) * C0 / frequency,
            "snr_db": config.get("snr_db", 40.), "complex_gain": config.get("gain", 1.+0j),
            "scan_jitter_std_m": config.get("jitter_m", 0.),
            "distance_bias_m": config.get("distance_bias_m", 0.), "seed": seed_base + 100 * i,
        }
        measurement = simulate_measurement(elements, feed, frequency, samples, config=sim_config)
        validation = simulate_measurement(elements, feed, frequency, validation_samples, config={**sim_config, "seed": sim_config["seed"] + 1})
        test = simulate_measurement(elements, feed, frequency, test_samples, config={**sim_config, "seed": sim_config["seed"] + 2})
        fit = solve_tikhonov(prepared, measurement["field"], validation_matrix=av, validation_y=validation["field"])
        truth = measurement["equivalent_coefficients"]
        metrics, aligned, flagged = evaluate_coefficients(fit.coefficients, truth, nominal, faults)
        field_error = float(np.linalg.norm(at @ fit.coefficients - test["field_clean"]) / np.linalg.norm(test["field_clean"]))
        s = prepared.singular_values
        # A worst-observed source perturbation exposes information loss even
        # when a small handpicked set of faults happens to reconstruct well.
        hard_delta = prepared.vh[-1].conj() / prepared.column_scale
        hard_delta *= .1 * np.linalg.norm(nominal) / np.linalg.norm(hard_delta)
        hard_visibility = float(np.linalg.norm(matrix @ hard_delta) / np.linalg.norm(matrix @ nominal))
        result = {
            "name": name, "side": side, "elements": side * side, "frequency_hz": frequency,
            "scan_shape": [scan_side, scan_side], "scan_step_m": .00835, "scan_distance_m": .06,
            "source_aperture_m": sim_config["source_aperture_m"], "source_quadrature_order": 3,
            "source_directionality_power": 0., "probe_aperture_m": sim_config["probe_aperture_m"],
            "snr_db": sim_config["snr_db"], "seed": sim_config["seed"], "excitation_seed": seed_base + i,
            "stress_case": bool(config.get("stress", False)),
            "metrics": metrics, "coefficient_acceptance_passed": accepts(metrics, policy),
            "independent_test_field_nrmse": field_error,
            "selected_alpha": fit.alpha, "selection_method": fit.selection_method,
            "selection_field_nrmse": fit.validation_relative_error,
            "training_field_nrmse": fit.relative_residual,
            "column_scaled_condition_number": float(s[0] / s[-1]),
            "numerical_rank": fit.numerical_rank, "regularized_effective_rank": fit.effective_rank,
            "ten_percent_hard_mode_field_ratio": hard_visibility,
            "hard_mode_below_40dB_total_noise_norm": hard_visibility < .01,
            "point_model_pair_mahalanobis_complex_40dB": 2 * hard_visibility * np.sqrt(len(samples)) * 100,
            "seconds": perf_counter() - start,
            "truth_used_for_regularization_selection": False,
        }
        cases.append(result)
        np.savez_compressed(output / (name + ".npz"), provenance="synthetic",
            elements_m=elements, samples_m=samples, field=measurement["field"],
            frequency_hz=frequency, nominal=nominal, true_coefficients=truth,
            recovered_coefficients=fit.coefficients, aligned_coefficients=aligned,
            singular_values=s, test_samples_m=test_samples, test_field=test["field"],
            validation_samples_m=validation_samples, validation_field=validation["field"])
        with (output / (name + "_elements.csv")).open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "row", "col", "true_amplitude", "estimated_amplitude_aligned",
                             "true_phase_deg", "estimated_phase_deg_aligned", "injected_fault", "flagged"])
            for n in range(len(truth)):
                phase_true = np.rad2deg(np.angle(truth[n])) if abs(truth[n]) >= .1 else ""
                phase_est = np.rad2deg(np.angle(aligned[n])) if abs(aligned[n]) >= .1 else ""
                writer.writerow([n, n // side, n % side, abs(truth[n]), abs(aligned[n]), phase_true,
                                 phase_est, n in faults, bool(flagged[n])])
        bp = angular_spectrum_backprop(measurement["field"].reshape(scan_side, scan_side), .00835, .00835, frequency, .06)
        bp_field = bp["image"] if isinstance(bp, dict) else bp
        _plot_case(output, name, side, truth, aligned, measurement["field"], scan_side, bp_field)
        (output / "results.json").write_text(json.dumps({"cases": cases, "in_progress": True}, indent=2), encoding="utf8")
        progress(f"Finished {name}: pass={result['coefficient_acceptance_passed']}, NRMSE={metrics['complex_nrmse']:.4f}, condition={result['column_scaled_condition_number']:.3g}", flush=True)
    ambiguity = None
    if full:
        ambiguity = ambiguity_demonstration(output, cache[(32, 11.7e9)][3])
    nominal_cases = [case for case in cases if not case["stress_case"]]
    report = {"schema_version": 1, "synthetic_only": True, "actual_measurements_read": False,
              "full_scale_included": full, "policy": policy,
              "nominal_scenarios_passed": all(c["coefficient_acceptance_passed"] for c in nominal_cases),
              "universal_identifiability_claim": False, "real_data_gate_open": False,
              "seed_base": seed_base, "random_fault_locations": random_faults, "total_seconds": perf_counter() - t0,
              "independent_truth_ambiguity": ambiguity, "cases": cases}
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def ambiguity_demonstration(output, prepared):
    """Quantify two distinct source states under proper complex white noise.

    For E|noise_m|^2=sigma^2, d=||mu_plus-mu_minus||/sigma; KL=d^2.
    Unlike a comparison with the total noise norm, this includes coherent
    accumulation over all observations. Truth uses finite-aperture quadrature.
    """
    elements = array_geometry(32); samples = scan_geometry(63)
    nominal = np.ones(1024, dtype=complex)
    delta = prepared.vh[-1].conj() / prepared.column_scale
    delta *= .1 * np.linalg.norm(nominal) / np.linalg.norm(delta)
    config = {"source_aperture_m": .1 * C0 / 11.7e9, "directionality_power": 0., "snr_db": None}
    base = simulate_measurement(elements, nominal, 11.7e9, samples, config=config)["field_clean"]
    change = simulate_measurement(elements, delta, 11.7e9, samples, config=config)["field_clean"]
    plus, minus = nominal + delta, nominal - delta
    sigma = np.linalg.norm(base) / np.sqrt(len(base)) / 100
    d = 2 * np.linalg.norm(change) / sigma
    gauge = np.vdot(minus, plus) / np.vdot(minus, minus)
    result = {"elements": 1024, "frequency_hz": 11.7e9, "snr_db": 40,
        "coefficient_pair_relative_gap": float(np.linalg.norm(plus - minus) / np.linalg.norm(nominal)),
        "coefficient_pair_gap_after_complex_alignment": float(np.linalg.norm(plus - gauge * minus) / np.linalg.norm(plus)),
        "independent_truth_field_pair_relative_gap": float(2 * np.linalg.norm(change) / np.linalg.norm(base)),
        "complex_noise_sigma_per_sample": float(sigma),
        "mahalanobis_complex_distance": float(d), "proper_complex_gaussian_kl": float(d * d),
        "conclusion": ("Two materially different radiation states are statistically nearly indistinguishable at the assumed noise; generic unique recovery is unsupported."
                       if d < .1 else "This pair does not establish statistical indistinguishability at the assumed noise."),
        "derivation": "sigma^2=E|noise_m|^2; KL=||mu_plus-mu_minus||^2/sigma^2"}
    np.savez_compressed(Path(output) / "ambiguity_pair.npz", provenance="synthetic",
        coefficients_plus=plus, coefficients_minus=minus, field_plus=base + change,
        field_minus=base - change, source_delta=delta, singular_values=prepared.singular_values)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    s = prepared.singular_values
    ax.semilogy(np.arange(1, len(s) + 1), s / s[0])
    ax.set(xlabel="Singular mode index", ylabel="Normalized singular value",
           title="32 x 32 array: some radiation modes are weakly observed")
    ax.grid(alpha=.25)
    fig.savefig(Path(output) / "singular_spectrum.png", dpi=170); plt.close(fig)
    return result
