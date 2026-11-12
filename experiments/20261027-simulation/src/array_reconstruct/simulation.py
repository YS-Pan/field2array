"""Independent scalar virtual array and near-field measurement generator.

All data produced here are synthetic.  Nothing in this module reads measured
data, and it deliberately does not import the inverse/physics implementation.
The time convention is exp(-i omega t), with outgoing exp(+i k R)/R waves.
Finite rectangular source and probe apertures are *area averages*, so their
point limits retain the same coefficient units (there is no 1/(4 pi) factor).
This is a controllable scalar model, not a full-wave electromagnetic solver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss


C0 = 299_792_458.0
REFERENCE_FREQUENCY_HZ = 11.7e9
DEFAULT_SPACING_M = 0.5 * C0 / REFERENCE_FREQUENCY_HZ
DEFAULT_FREQUENCIES_HZ = (10.7e9, 11.7e9, 12.7e9)


def _positive_frequency(frequency_hz: float) -> float:
    value = float(frequency_hz)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    return value


def _shape_pair(shape: Sequence[int], name: str = "shape") -> tuple[int, int]:
    values = tuple(shape)
    if len(values) != 2 or any(int(v) != v or v < 1 for v in values):
        raise ValueError(f"{name} must contain two positive integers")
    return int(values[0]), int(values[1])


def _xy_pair(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    pair = np.asarray(value, dtype=float)
    if pair.ndim == 0:
        pair = np.repeat(pair, 2)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must be a finite scalar or (x, y) pair")
    if np.any(pair <= 0 if positive else pair < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return pair


def _positions(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError(f"{name} must have shape (nonzero count, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _jitter_std(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.repeat(result, 3)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be a nonnegative scalar or (x, y, z) vector")
    return result


def build_rectangular_array(
    shape: Sequence[int] = (8, 8),
    spacing_m: float | Sequence[float] | None = None,
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    row_offset_m: float = 0.0,
) -> np.ndarray:
    """Return centered (N,3) coordinates, row/y then column/x in C order.

    ``shape=(ny,nx)``, ``spacing_m=(dx,dy)``.  An omitted spacing is half a
    wavelength at ``frequency_hz``; use a fixed explicit spacing for sweeps.
    ``generate_virtual_case`` always uses the fixed 11.7 GHz design spacing.
    Odd-numbered rows shift in +x by row_offset_m, followed by recentering at
    the coordinate centroid.  For the reference staggered configuration use
    spacing_m=0.011 and row_offset_m=0.0055 explicitly; hardware is unverified.
    """
    ny, nx = _shape_pair(shape)
    frequency_hz = _positive_frequency(frequency_hz)
    spacing = _xy_pair(
        0.5 * C0 / frequency_hz if spacing_m is None else spacing_m,
        "spacing_m", positive=True,
    )
    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must be a finite (x, y, z) vector")
    if not np.isfinite(row_offset_m):
        raise ValueError("row_offset_m must be finite")
    x = (np.arange(nx) - (nx - 1) / 2) * spacing[0]
    y = (np.arange(ny) - (ny - 1) / 2) * spacing[1]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xx += (np.arange(ny) % 2)[:, None] * row_offset_m
    xx -= np.mean(xx)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny))) + center


def make_scan_grid(
    shape: Sequence[int] = (25, 25),
    step_m: float | Sequence[float] = 8.35e-3,
    z_m: float = 60e-3,
    center_xy_m: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Return centered xy scan points; shape=(ny,nx), step=(dx,dy)."""
    center = np.asarray(center_xy_m, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_m must be a finite (x,y) pair")
    if not np.isfinite(z_m):
        raise ValueError("z_m must be finite")
    return build_rectangular_array(shape, step_m, center_m=(*center, z_m))


def make_feed_weights(
    element_positions: np.ndarray,
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    beam_deg: Sequence[float] = (0.0, 0.0),
    normal_amp_std_db: float = 0.0,
    normal_phase_std_deg: float = 0.0,
    seed: int | None = 0,
    taper: np.ndarray | None = None,
) -> np.ndarray:
    """Known feed weights with steering and independent small normal errors.

    beam_deg=(theta,phi): theta is measured from +z and phi from +x toward +y.
    Positive steering phase exp(+ik r_n dot s) focuses the outgoing far field
    toward s under the exp(-i omega t) convention.  Coupling is applied later.
    """
    positions = _positions(element_positions, "element_positions")
    frequency_hz = _positive_frequency(frequency_hz)
    beam = np.asarray(beam_deg, dtype=float)
    if beam.shape != (2,) or not np.all(np.isfinite(beam)):
        raise ValueError("beam_deg must be a finite (theta, phi) pair")
    if abs(beam[0]) > 90:
        raise ValueError("beam theta must lie between -90 and 90 degrees")
    for value in (normal_amp_std_db, normal_phase_std_deg):
        if not np.isfinite(value) or value < 0:
            raise ValueError("normal error standard deviations must be nonnegative")
    theta, phi = np.deg2rad(beam)
    direction = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi), np.cos(theta)])
    amplitudes = np.ones(len(positions)) if taper is None else np.asarray(taper, float)
    if amplitudes.shape != (len(positions),) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("taper must contain one finite amplitude per element")
    if np.any(amplitudes < 0):
        raise ValueError("taper amplitudes must be nonnegative")
    rng = np.random.default_rng(seed)
    amp_errors_db = rng.normal(0, normal_amp_std_db, len(positions))
    phase_errors = np.deg2rad(rng.normal(0, normal_phase_std_deg, len(positions)))
    phases = 2 * np.pi * frequency_hz / C0 * (positions @ direction)
    return amplitudes * 10 ** (amp_errors_db / 20) * np.exp(1j * (phases + phase_errors))


def _fault_indices(fault: Mapping[str, Any], count: int,
                   array_shape: Sequence[int] | None) -> np.ndarray:
    if "indices" in fault:
        raw = np.atleast_1d(fault["indices"])
        if not np.all(np.isfinite(raw)) or np.any(raw != raw.astype(int)):
            raise ValueError("fault indices must be integers")
        indices = raw.astype(int)
    elif "index" in fault:
        return _fault_indices({"indices": [fault["index"]]}, count, array_shape)
    elif "start" in fault and "shape" in fault and array_shape is not None:
        ny, nx = _shape_pair(array_shape, "array_shape")
        if ny * nx != count:
            raise ValueError("array_shape must match the weight count")
        start = np.asarray(fault["start"])
        sy, sx = _shape_pair(fault["shape"], "fault shape")
        if start.shape != (2,) or np.any(start != start.astype(int)):
            raise ValueError("fault start must be integer (row, column)")
        row, col = start.astype(int)
        if row < 0 or col < 0 or row + sy > ny or col + sx > nx:
            raise ValueError("fault cluster extends outside the array")
        indices = (np.arange(row, row + sy)[:, None] * nx
                   + np.arange(col, col + sx)[None, :]).ravel()
    else:
        raise ValueError("fault needs indices/index or start+shape+array_shape")
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("fault indices are outside the weight vector")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fault indices must be unique within each fault")
    return indices


def inject_faults(
    feed_weights: np.ndarray,
    faults: Sequence[Mapping[str, Any]],
    array_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Return a copy with multiplicative *feed* faults, before coupling.

    Fault dictionaries contain kind and indices (or index).  A cluster may use
    start=(row,col), shape=(rows,cols), with array_shape supplied.  Kinds:
    dead: amplitude_db=-40, or zero=True; amplitude: amplitude_db=-6;
    phase: phase_deg=60; cluster: amplitude_db=-6 and phase_deg=60.
    Explicitly overlapping faults compound in the listed order.  A dead feed
    may still have nonzero equivalent radiation after mutual coupling.
    """
    weights = np.asarray(feed_weights, dtype=complex).copy()
    if weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights)):
        raise ValueError("feed_weights must be a nonempty finite complex vector")
    for fault in faults:
        indices = _fault_indices(fault, len(weights), array_shape)
        kind = fault.get("kind")
        if kind == "dead":
            amp_db = float(fault.get("amplitude_db", -40.0))
            if not np.isfinite(amp_db):
                raise ValueError("fault amplitude_db must be finite; use zero=True")
            factor = 0.0 if fault.get("zero", False) else 10 ** (amp_db / 20)
        elif kind in ("amplitude", "phase", "cluster"):
            amp_db = float(fault.get("amplitude_db", -6 if kind != "phase" else 0))
            phase_deg = float(fault.get("phase_deg", 60 if kind != "amplitude" else 0))
            if not np.isfinite(amp_db) or not np.isfinite(phase_deg):
                raise ValueError("fault amplitude_db and phase_deg must be finite")
            factor = 10 ** (amp_db / 20) * np.exp(1j * np.deg2rad(phase_deg))
        else:
            raise ValueError(f"unknown fault kind: {kind!r}")
        weights[indices] *= factor
    return weights


def standard_faults(shape: Sequence[int] = (8, 8)) -> list[dict[str, Any]]:
    """A reproducible nonoverlapping diagnostic set, for arrays at least 6x6."""
    ny, nx = _shape_pair(shape)
    if min(ny, nx) < 6:
        raise ValueError("standard_faults requires both dimensions at least 6")
    return [
        {"kind": "dead", "index": nx + 1, "amplitude_db": -40.0},
        {"kind": "amplitude", "index": nx + nx - 2, "amplitude_db": -6.0},
        {"kind": "phase", "index": (ny - 2) * nx + 1, "phase_deg": 60.0},
        {"kind": "cluster", "start": (ny - 3, nx - 3), "shape": (2, 2),
         "amplitude_db": -6.0, "phase_deg": 60.0},
    ]


def nearest_neighbor_coupling(element_positions: np.ndarray,
                              strength: complex = 0.03) -> np.ndarray:
    """Synthetic C=I+strength*A for immediate rectangular/staggered neighbors.

    A has no row wrapping.  Within a row it links consecutive columns; between
    adjacent rows it links x separations up to half a column pitch.  Thus an
    aligned square grid has four neighbors and a half-staggered interior has
    six.  Strength is a field
    amplitude coefficient per neighbor, not a calibrated S-parameter model.
    Explicit coupling_matrix accepts any finite complete feed-to-radiation C.
    """
    positions = _positions(element_positions, "element_positions")
    strength = complex(strength)
    if not np.isfinite(strength):
        raise ValueError("coupling strength must be finite")
    matrix = np.eye(len(positions), dtype=complex)
    if strength == 0:
        return matrix
    if not np.allclose(positions[:, 2], positions[0, 2], rtol=0, atol=1e-12):
        raise ValueError("automatic coupling requires a planar xy array")
    adjacency = np.zeros(matrix.shape, dtype=bool)
    ys = np.unique(positions[:, 1])
    if len(ys) > 2 and not np.allclose(np.diff(ys), np.diff(ys)[0], rtol=1e-6, atol=1e-12):
        raise ValueError("automatic coupling requires equally spaced rows")
    rows = []
    pitches = []
    for y in ys:
        indices = np.flatnonzero(np.isclose(positions[:, 1], y, rtol=0, atol=1e-12))
        indices = indices[np.argsort(positions[indices, 0])]
        rows.append(indices)
        steps = np.diff(positions[indices, 0])
        if len(steps):
            if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
                raise ValueError("automatic coupling requires equally spaced columns")
            pitches.append(steps[0])
            adjacency[indices[:-1], indices[1:]] = True
            adjacency[indices[1:], indices[:-1]] = True
    if pitches and not np.allclose(pitches, pitches[0], rtol=1e-6, atol=1e-12):
        raise ValueError("automatic coupling requires a common column pitch")
    tolerance = (0.5 * pitches[0] if pitches else 0.0) + 1e-12
    for first, second in zip(rows[:-1], rows[1:]):
        linked = np.abs(positions[first, None, 0] - positions[second, 0]) <= tolerance
        adjacency[np.ix_(first, second)] = linked
        adjacency[np.ix_(second, first)] = linked.T
    matrix[adjacency] += strength
    return matrix


def _aperture_nodes(widths_m: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    nodes, weights = leggauss(order)
    axes = [(np.array([0.0]), np.array([1.0])) if width == 0
            else (nodes * width / 2, weights / 2) for width in widths_m]
    x, y = np.meshgrid(axes[0][0], axes[1][0], indexing="xy")
    wx, wy = np.meshgrid(axes[0][1], axes[1][1], indexing="xy")
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), (wx * wy).ravel()


def _independent_aperture_field(
    elements: np.ndarray, coefficients: np.ndarray, scans: np.ndarray, k: float,
    source_nodes: np.ndarray, source_weights: np.ndarray,
    probe_nodes: np.ndarray, probe_weights: np.ndarray,
    directionality_power: float, probe_directionality_power: float,
    chunk_size: int, element_chunk_size: int,
) -> np.ndarray:
    """Accumulate bounded-size blocks without making a full transfer matrix."""
    field = np.zeros(len(scans), dtype=complex)
    exponent = directionality_power + probe_directionality_power
    for lo in range(0, len(scans), chunk_size):
        hi = min(lo + chunk_size, len(scans))
        block = np.zeros(hi - lo, dtype=complex)
        for elo in range(0, len(elements), element_chunk_size):
            ehi = min(elo + element_chunk_size, len(elements))
            source_centers = elements[elo:ehi]
            for probe_node, probe_weight in zip(probe_nodes, probe_weights):
                receivers = scans[lo:hi] + probe_node
                for source_node, source_weight in zip(source_nodes, source_weights):
                    sources = source_centers + source_node
                    dx = receivers[:, None, 0] - sources[None, :, 0]
                    dy = receivers[:, None, 1] - sources[None, :, 1]
                    dz = receivers[:, None, 2] - sources[None, :, 2]
                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if np.any(distance <= np.finfo(float).tiny):
                        raise ValueError("source and probe quadrature points coincide")
                    kernel = np.exp(1j * k * distance) / distance
                    if exponent:
                        kernel *= np.maximum(dz / distance, 0.0) ** exponent
                    block += (probe_weight * source_weight
                              * (kernel @ coefficients[elo:ehi]))
        field[lo:hi] = block
    return field


def simulate_measurement(
    element_positions: np.ndarray,
    feed_weights: np.ndarray,
    frequency_hz: float,
    scan_positions: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate independent finite-aperture virtual truth and complex AWGN.

    Config keys (all optional):
      source_aperture_m=0.1*wavelength; source_quadrature_order=3;
      directionality_power=1; probe_aperture_m=0; probe_quadrature_order=3;
      probe_directionality_power=0; scan_jitter_std_m=0;
      element_jitter_std_m=0; distance_bias_m=0; snr_db=None;
      complex_gain=1+0j; coupling_matrix=None; coupling_strength=0j;
      coupling_known=False; seed=0; chunk_size=256; element_chunk_size=256;
      return_samples=False.

    Apertures are xy widths (scalar or pair); jitter is iid per point and axis
    (scalar or xyz standard deviations).  Positive distance bias shifts the
    entire scan in +z.  Directionality is max(cos(theta),0)**p for p>0;
    p=0 explicitly means isotropic.  Probe normal faces the source plane.
    C is a *complete* map: equivalent_coefficients=C@feed_weights.  The actual
    measured field is complex_gain*H@equivalent_coefficients + complex AWGN.
    Noise power is referenced to the post-gain noise-free scan mean power.
    Geometry/coupling truth is returned for scoring; hidden truth must not be
    passed to the inverse solver in unknown-calibration validation scenarios.
    """
    elements = _positions(element_positions, "element_positions")
    scans = _positions(scan_positions, "scan_positions")
    frequency_hz = _positive_frequency(frequency_hz)
    feeds = np.asarray(feed_weights, dtype=complex)
    if feeds.shape != (len(elements),) or not np.all(np.isfinite(feeds)):
        raise ValueError("feed_weights must have one finite complex value per element")
    settings = dict(
        source_aperture_m=0.1 * C0 / frequency_hz, source_quadrature_order=3,
        directionality_power=1.0, probe_aperture_m=0.0, probe_quadrature_order=3,
        probe_directionality_power=0.0, scan_jitter_std_m=0.0,
        element_jitter_std_m=0.0, distance_bias_m=0.0, snr_db=None,
        complex_gain=1.0 + 0j, coupling_matrix=None, coupling_strength=0j,
        coupling_known=False, seed=0, chunk_size=256, element_chunk_size=256,
        return_samples=False,
    )
    supplied = {} if config is None else dict(config)
    unknown = supplied.keys() - settings.keys()
    if unknown:
        raise ValueError(f"unknown simulation config keys: {sorted(unknown)}")
    settings.update(supplied)
    source_aperture = _xy_pair(settings["source_aperture_m"], "source_aperture_m")
    probe_aperture = _xy_pair(settings["probe_aperture_m"], "probe_aperture_m")
    source_nodes, source_weights = _aperture_nodes(
        source_aperture, settings["source_quadrature_order"])
    probe_nodes, probe_weights = _aperture_nodes(
        probe_aperture, settings["probe_quadrature_order"])
    powers = [float(settings[name]) for name in
              ("directionality_power", "probe_directionality_power")]
    if any(not np.isfinite(power) or power < 0 for power in powers):
        raise ValueError("directionality powers must be finite and nonnegative")
    sizes = [settings["chunk_size"], settings["element_chunk_size"]]
    if any(not isinstance(size, (int, np.integer)) or size < 1 for size in sizes):
        raise ValueError("chunk sizes must be positive integers")
    distance_bias = float(settings["distance_bias_m"])
    gain = complex(settings["complex_gain"])
    if not np.isfinite(distance_bias) or not np.isfinite(gain):
        raise ValueError("distance_bias_m and complex_gain must be finite")
    scan_std = _jitter_std(settings["scan_jitter_std_m"], "scan_jitter_std_m")
    element_std = _jitter_std(settings["element_jitter_std_m"], "element_jitter_std_m")
    # Independent streams keep noise comparisons meaningful when jitter changes.
    element_seed, scan_seed, noise_seed = np.random.SeedSequence(settings["seed"]).spawn(3)
    actual_elements = elements + np.random.default_rng(element_seed).normal(
        size=elements.shape) * element_std
    actual_scans = scans + np.random.default_rng(scan_seed).normal(size=scans.shape) * scan_std
    actual_scans[:, 2] += distance_bias
    coupling = settings["coupling_matrix"]
    if coupling is not None and complex(settings["coupling_strength"]) != 0:
        raise ValueError("provide coupling_matrix or coupling_strength, not both")
    if coupling is None:
        coupling = nearest_neighbor_coupling(elements, settings["coupling_strength"])
    else:
        coupling = np.asarray(coupling, dtype=complex)
    if coupling.shape != (len(elements), len(elements)) or not np.all(np.isfinite(coupling)):
        raise ValueError("coupling_matrix must be a finite (N,N) complete linear map")
    equivalent = coupling @ feeds
    ungained = _independent_aperture_field(
        actual_elements, equivalent, actual_scans, 2 * np.pi * frequency_hz / C0,
        source_nodes, source_weights, probe_nodes, probe_weights,
        *powers, *sizes,
    )
    clean = gain * ungained
    noise = np.zeros_like(clean)
    signal_power = float(np.mean(np.abs(clean) ** 2))
    snr_db = settings["snr_db"]
    noise_power = 0.0
    if snr_db is not None:
        snr_db = float(snr_db)
        if np.isnan(snr_db) or snr_db == -np.inf:
            raise ValueError("snr_db must be finite or positive infinity")
        if signal_power == 0 and np.isfinite(snr_db):
            raise ValueError("finite SNR is undefined for an identically zero signal")
        noise_power = signal_power * 10 ** (-snr_db / 10)
        if not np.isfinite(noise_power):
            raise ValueError("snr_db produces nonfinite noise power")
        rng = np.random.default_rng(noise_seed)
        noise = np.sqrt(noise_power / 2) * (rng.normal(size=len(clean))
                                          + 1j * rng.normal(size=len(clean)))
    actual_noise_power = float(np.mean(np.abs(noise) ** 2))
    actual_snr = (float(10 * np.log10(signal_power / actual_noise_power))
                  if signal_power > 0 and actual_noise_power > 0 else None)
    result: dict[str, Any] = {
        "field": clean + noise,
        "field_clean": clean,
        "field_without_gain": ungained,
        "noise": noise,
        "feed_weights": feeds.copy(),
        "equivalent_coefficients": equivalent,
        "nominal_element_positions": elements.copy(),
        "actual_element_positions": actual_elements,
        "nominal_scan_positions": scans.copy(),
        "actual_scan_positions": actual_scans,
        "true_coupling_matrix": coupling.copy(),
        "known_coupling_matrix": coupling.copy() if settings["coupling_known"] else None,
        "metadata": {
            "data_origin": "virtual_only",
            "truth_model": "independent_scalar_rectangular_aperture_quadrature",
            "time_convention": "exp(-i omega t)",
            "outgoing_kernel": "exp(+i k R)/R; no 1/(4*pi)",
            "aperture_normalization": "area_average",
            "coefficient_definition": "equivalent_coefficients = C @ feed_weights",
            "fault_domain": "feed_before_coupling",
            "frequency_hz": frequency_hz,
            "source_aperture_m": source_aperture.tolist(),
            "source_quadrature_order": settings["source_quadrature_order"],
            "probe_aperture_m": probe_aperture.tolist(),
            "probe_quadrature_order": settings["probe_quadrature_order"],
            "directionality_power": powers[0],
            "probe_directionality_power": powers[1],
            "scan_jitter_std_m": scan_std.tolist(),
            "element_jitter_std_m": element_std.tolist(),
            "distance_bias_m": distance_bias,
            "complex_gain_real_imag": [gain.real, gain.imag],
            "coupling_known": bool(settings["coupling_known"]),
            "target_snr_db": snr_db,
            "realized_snr_db": actual_snr,
            "signal_mean_power": signal_power,
            "expected_noise_mean_power": noise_power,
            "seed": settings["seed"],
        },
    }
    if settings["return_samples"]:
        result["source_quadrature_offsets_m"] = source_nodes
        result["source_quadrature_weights"] = source_weights
        result["probe_quadrature_offsets_m"] = probe_nodes
        result["probe_quadrature_weights"] = probe_weights
    return result


def generate_virtual_case(
    array_shape: Sequence[int] = (8, 8),
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    spacing_m: float | Sequence[float] = DEFAULT_SPACING_M,
    scan_shape: Sequence[int] | None = None,
    scan_step_m: float | Sequence[float] = 8.35e-3,
    scan_z_m: float = 60e-3,
    beam_deg: Sequence[float] = (0.0, 0.0),
    normal_amp_std_db: float = 0.2,
    normal_phase_std_deg: float = 2.0,
    faults: Sequence[Mapping[str, Any]] = (),
    config: Mapping[str, Any] | None = None,
    seed: int | None = 0,
    row_offset_m: float = 0.0,
) -> dict[str, Any]:
    """Convenience virtual case, retaining healthy and faulted feed truths.

    The default scan spans the element-center footprint plus 60 mm per side;
    this is a geometry choice, not a proof of truncation adequacy.  Physical
    array spacing remains 0.5 lambda at 11.7 GHz across frequency sweeps.
    No faults are injected unless requested, e.g. standard_faults((8,8)).
    """
    shape = _shape_pair(array_shape, "array_shape")
    positions = build_rectangular_array(shape, spacing_m, row_offset_m=row_offset_m)
    if scan_shape is None:
        step = _xy_pair(scan_step_m, "scan_step_m", positive=True)
        extent = np.ptp(positions[:, :2], axis=0) + 0.12
        nxy = np.ceil(extent / step).astype(int) + 1
        scan_shape = (int(nxy[1]), int(nxy[0]))
    scans = make_scan_grid(scan_shape, scan_step_m, scan_z_m)
    healthy = make_feed_weights(positions, frequency_hz, beam_deg,
                                normal_amp_std_db, normal_phase_std_deg, seed)
    faulted = inject_faults(healthy, faults, array_shape=shape)
    settings = {"seed": seed, **({} if config is None else dict(config))}
    result = simulate_measurement(positions, faulted, frequency_hz, scans, settings)
    result.update({"healthy_feed_weights": healthy,
                   "fault_mask": ~np.isclose(faulted, healthy, rtol=1e-12, atol=1e-14),
                   "faults": [dict(fault) for fault in faults],
                   "array_shape": shape, "scan_shape": _shape_pair(scan_shape)})
    result["metadata"]["spacing_m"] = _xy_pair(spacing_m, "spacing_m").tolist()
    result["metadata"]["element_order"] = "C order; row/y first, column/x fastest"
    result["metadata"]["spacing_reference_frequency_hz"] = (
        REFERENCE_FREQUENCY_HZ if np.allclose(spacing_m, DEFAULT_SPACING_M,
                                             rtol=1e-12, atol=0) else None)
    result["metadata"]["row_offset_m"] = float(row_offset_m)
    return result
