"""Synthetic-truth metrics and explicitly conditional identifiability checks."""

import numpy as np
from scipy.linalg import svdvals


C0 = 299792458.0


def _pair(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if (estimate.ndim != 1 or estimate.shape != reference.shape or estimate.size == 0
            or not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference))):
        raise ValueError("estimate and reference must be equal-sized finite complex vectors")
    if np.linalg.norm(reference) == 0:
        raise ValueError("The reference vector must have nonzero norm")
    return estimate, reference


def align_coefficients(estimate, reference, alignment="none"):
    """Return estimate aligned by one common scalar and that scalar.

    Complex alignment deliberately removes common amplitude and phase error;
    it must not be reported as absolute calibration accuracy. Phase alignment
    preserves amplitude error. This operation uses synthetic truth/reference,
    never unknown real coefficients, and is solely an evaluation convention.
    """
    estimate, reference = _pair(estimate, reference)
    if alignment == "none":
        factor = 1.0 + 0j
    elif alignment in ("complex", "phase"):
        product = np.vdot(estimate, reference)
        if alignment == "phase":
            factor = product / abs(product) if abs(product) > 0 else 1.0 + 0j
        else:
            denominator = np.vdot(estimate, estimate).real
            factor = product / denominator if denominator > 0 else 1.0 + 0j
    else:
        raise ValueError("alignment must be 'none', 'phase', or 'complex'")
    return factor * estimate, complex(factor)


def coefficient_metrics(estimate, truth, *, alignment="none", amplitude_floor_relative=0.1):
    """Compare equivalent radiating weights, with explicit phase-score masks.

    The amplitude floor is relative to the strongest truth coefficient. Phase
    is not scored where either truth or estimate is below the floor; amplitude
    and full-vector complex error remain scored so a failed estimate cannot
    disappear from all metrics. Zero/dead truth elements have no true phase.
    """
    estimate, truth = _pair(estimate, truth)
    if not 0 < amplitude_floor_relative < 1:
        raise ValueError("amplitude_floor_relative must lie strictly between 0 and 1")
    aligned, factor = align_coefficients(estimate, truth, alignment)
    true_amp, recovered_amp = np.abs(truth), np.abs(aligned)
    floor = amplitude_floor_relative * np.max(true_amp)
    active = true_amp >= floor
    phase_mask = active & (recovered_amp >= floor)
    amplitude_error = np.full(truth.shape, np.nan)
    amplitude_error[active] = 20 * np.log10(np.maximum(recovered_amp[active],
                                                      np.max(true_amp) * 1e-15) / true_amp[active])
    phase_error = np.full(truth.shape, np.nan)
    phase_error[phase_mask] = np.rad2deg(np.angle(aligned[phase_mask] * np.conj(truth[phase_mask])))

    def rms(values):
        return float(np.sqrt(np.mean(values * values))) if values.size else None

    def maximum(values):
        return float(np.max(np.abs(values))) if values.size else None

    return {
        "alignment": alignment,
        "alignment_factor": [float(factor.real), float(factor.imag)],
        "absolute_calibration_tested": alignment == "none",
        "complex_nrmse": float(np.linalg.norm(aligned - truth) / np.linalg.norm(truth)),
        "amplitude_rmse_db": rms(amplitude_error[active]),
        "phase_rmse_deg": rms(phase_error[phase_mask]),
        "amplitude_max_abs_db": maximum(amplitude_error[active]),
        "phase_max_abs_deg": maximum(phase_error[phase_mask]),
        "amplitude_error_db": amplitude_error,
        "phase_error_deg": phase_error,
        "amplitude_scored_mask": active,
        "phase_scored_mask": phase_mask,
        "active_truth_count": int(np.sum(active)),
        "phase_scored_count": int(np.sum(phase_mask)),
        "active_phase_unscored_count": int(np.sum(active & ~phase_mask)),
        "inactive_truth_count": int(np.sum(~active)),
        "inactive_estimate_max_relative": float(np.max(recovered_amp[~active]) / np.max(true_amp))
        if np.any(~active) else 0.0,
        "amplitude_floor_relative": float(amplitude_floor_relative),
    }


def detect_anomalies(
    estimate,
    nominal,
    *,
    amplitude_threshold_db=3.0,
    phase_threshold_deg=20.0,
    amplitude_floor_relative=0.1,
    alignment="none",
):
    """Classify deviations from a known commanded/calibrated nominal vector.

    A common calibration offset is only removed when explicitly requested.
    Without a trusted nominal excitation this function cannot label hardware
    faults; a commanded phase ramp or taper is not itself a fault.
    """
    estimate, nominal = _pair(estimate, nominal)
    if (not np.isfinite(amplitude_threshold_db) or not np.isfinite(phase_threshold_deg)
            or amplitude_threshold_db <= 0 or not 0 < phase_threshold_deg <= 180):
        raise ValueError("Anomaly thresholds must be finite and positive; phase cannot exceed 180 deg")
    if not 0 < amplitude_floor_relative < 1:
        raise ValueError("amplitude_floor_relative must lie strictly between 0 and 1")
    aligned, factor = align_coefficients(estimate, nominal, alignment)
    nominal_amp, estimate_amp = np.abs(nominal), np.abs(aligned)
    floor = amplitude_floor_relative * np.max(nominal_amp)
    nominal_active = nominal_amp >= floor
    phase_mask = nominal_active & (estimate_amp >= floor)
    amplitude_db = np.full(nominal.shape, np.nan)
    amplitude_db[nominal_active] = 20 * np.log10(np.maximum(estimate_amp[nominal_active],
                                                         np.max(nominal_amp) * 1e-15) / nominal_amp[nominal_active])
    phase_deg = np.full(nominal.shape, np.nan)
    phase_deg[phase_mask] = np.rad2deg(np.angle(aligned[phase_mask] * np.conj(nominal[phase_mask])))
    amplitude_fault = nominal_active & (np.abs(amplitude_db) >= amplitude_threshold_db - 1e-12)
    # Unexpected activation of a nominally off element is explicitly flagged.
    unexpected_activation = ~nominal_active & (estimate_amp >= floor)
    phase_fault = phase_mask & (np.abs(phase_deg) >= phase_threshold_deg - 1e-12)
    return {
        "fault_mask": amplitude_fault | phase_fault | unexpected_activation,
        "amplitude_fault_mask": amplitude_fault,
        "phase_fault_mask": phase_fault,
        "unexpected_activation_mask": unexpected_activation,
        "phase_scored_mask": phase_mask,
        "amplitude_deviation_db": amplitude_db,
        "phase_deviation_deg": phase_deg,
        "alignment": alignment,
        "alignment_factor": [float(factor.real), float(factor.imag)],
        "amplitude_threshold_db": float(amplitude_threshold_db),
        "phase_threshold_deg": float(phase_threshold_deg),
    }


def classification_metrics(predicted_fault_mask, true_fault_mask):
    predicted = np.asarray(predicted_fault_mask, dtype=bool)
    truth = np.asarray(true_fault_mask, dtype=bool)
    if predicted.ndim != 1 or predicted.shape != truth.shape or predicted.size == 0:
        raise ValueError("Fault masks must be nonempty equal-sized vectors")
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else (1.0 if not np.any(truth) else 0.0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "exact_mask_match": bool(np.array_equal(predicted, truth)),
    }


fault_metrics = classification_metrics


def matrix_identifiability(A=None, *, singular_values=None, matrix_shape=None, relative_noise_level=None):
    """Return algebraic diagnostics, not a certificate of physical recovery.

    Pass an already prepared SVD spectrum to avoid repeating large factorizations.
    ``noise_resolved_mode_count`` is merely the count s/smax > relative_noise;
    it ignores coefficient direction, calibration and systematic model errors.
    Full rank alone therefore never approves element-level recovery.
    """
    if singular_values is None:
        a = np.asarray(A, dtype=np.complex128)
        if a.ndim != 2 or min(a.shape) < 1 or not np.all(np.isfinite(a)):
            raise ValueError("A must be a nonempty finite matrix")
        s, shape = svdvals(a, check_finite=False), a.shape
    else:
        s = np.asarray(singular_values, dtype=float)
        if matrix_shape is None:
            raise ValueError("matrix_shape is required with a supplied spectrum")
        shape = tuple(matrix_shape)
    if (len(shape) != 2 or min(shape) < 1 or s.ndim != 1 or s.size != min(shape)
            or not np.all(np.isfinite(s)) or np.any(s < 0)):
        raise ValueError("Invalid singular spectrum or matrix_shape")
    s = np.sort(s)[::-1]
    smax = float(s[0])
    tolerance = max(shape) * np.finfo(float).eps * smax
    rank = int(np.sum(s > tolerance))
    full_column_rank = rank == shape[1]
    relative = s / smax if smax > 0 else np.zeros_like(s)
    result = {
        "sample_count": int(shape[0]), "coefficient_count": int(shape[1]),
        "numerical_rank": rank, "nullity": int(shape[1] - rank),
        "full_column_rank": full_column_rank,
        "condition_number": float(s[0] / s[-1]) if full_column_rank else float("inf"),
        "smallest_to_largest": float(relative[-1]) if full_column_rank else 0.0,
        "stable_rank": float(np.sum(relative ** 2)),
        "singular_values": s, "relative_singular_values": relative,
        "numerical_rank_tolerance": float(tolerance),
        "physical_identifiability_certified": False,
    }
    if relative_noise_level is not None:
        if not np.isfinite(relative_noise_level) or relative_noise_level <= 0:
            raise ValueError("relative_noise_level must be finite and positive")
        result["noise_resolved_mode_count"] = int(np.sum(relative > relative_noise_level))
        result["relative_noise_level"] = float(relative_noise_level)
    return result


def sampling_report(frequency_hz, step_m, *, z_m=None, scan_extent_m=None, array_extent_m=None):
    """Nyquist/truncation warnings for a Cartesian plane; no inference from channel IDs.

    The relaxed lambda/(2 sin(theta)) angle assumes the entire effective spectrum
    is band-limited. The more conservative lambda/(1+sin(theta)) angle protects
    a central ROI from arbitrary propagating spectra outside that ROI.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    step = np.broadcast_to(np.asarray(step_m, dtype=float), (2,)).copy()
    if not np.all(np.isfinite(step)) or np.any(step <= 0):
        raise ValueError("step_m must be one or two positive finite spacings")
    wavelength = C0 / frequency_hz
    central = wavelength / step - 1
    roi_angles = [float(np.rad2deg(np.arcsin(min(1.0, value)))) if value >= 0 else None for value in central]
    result = {
        "frequency_hz": float(frequency_hz), "wavelength_m": wavelength,
        "half_wavelength_m": wavelength / 2,
        "step_m": step.tolist(), "step_over_wavelength": (step / wavelength).tolist(),
        "full_propagating_spectrum_nyquist": bool(np.all(step <= wavelength / 2)),
        "nyquist_frequency_limit_hz": (C0 / (2 * step)).tolist(),
        "bandlimited_spectrum_angle_deg": np.rad2deg(np.arcsin(np.minimum(1, wavelength / (2 * step)))).tolist(),
        "arbitrary_propagating_spectrum_central_roi_deg": roi_angles,
        "evanescent_spectrum_covered": False,
        "element_identifiability_inferred": False,
    }
    if z_m is not None:
        if not np.isfinite(z_m) or z_m <= 0:
            raise ValueError("z_m must be finite and positive")
        result["z_over_wavelength"] = float(z_m / wavelength)
    if scan_extent_m is not None or array_extent_m is not None:
        if scan_extent_m is None or array_extent_m is None or z_m is None:
            raise ValueError("Both extents and z_m are needed for the geometric coverage estimate")
        scan = np.broadcast_to(np.asarray(scan_extent_m, dtype=float), (2,))
        array = np.broadcast_to(np.asarray(array_extent_m, dtype=float), (2,))
        if not np.all(np.isfinite(scan)) or not np.all(np.isfinite(array)) or np.any(scan <= 0) or np.any(array < 0):
            raise ValueError("Scan extents must be positive and array extents nonnegative")
        result["truncation_geometry_angle_deg"] = np.rad2deg(np.arctan(np.maximum(0, scan - array) / (2 * z_m))).tolist()
    return result
