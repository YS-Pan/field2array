"""Independent, hypothetical generator-side stressors for virtual geometry work.

The finite-aperture kernel is a normalized scalar spatial average over uniform
square source and probe apertures. It models neither absolute collecting area
nor an identified hardware element/probe. It deliberately imports no fit kernel.
Geometry here may lie outside fit bounds, so a 72 mm truth can test rejection.
Truth enters only generation; no function in this module fits or accepts a fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .joint_virtual import VirtualFitData, VirtualTruth

C_M_S = 299792458.0


def _real(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _nonnegative(value, name):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _aperture_sides(value, name):
    sides = _real(value, name)
    if sides.ndim == 0:
        sides = np.repeat(sides, 2)
    if sides.shape != (2,) or np.any(sides < 0):
        raise ValueError(f"{name} must be a nonnegative side or (x,y) sides in mm")
    return sides * .001


def _relative_axis(source_side, probe_side, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    source = [(0., 1.)] if source_side == 0 else list(zip(nodes * source_side / 2, weights / 2))
    probe = [(0., 1.)] if probe_side == 0 else list(zip(nodes * probe_side / 2, weights / 2))
    # Sum identical quadrature displacements without rounding coordinates.
    # Equal source/probe sides reduce GL3's 81 2-D pairs to 25 displacements.
    terms = {}
    for source_node, source_weight in source:
        for probe_node, probe_weight in probe:
            shift = float(probe_node - source_node)
            terms[shift] = terms.get(shift, 0.) + source_weight * probe_weight
    return tuple(terms.items())


def _generator_inputs(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                      q, sign, source_aperture_mm, probe_aperture_mm,
                      quadrature_order, chunk_size):
    elements = _real(elements_m, "elements_m")
    samples = _real(samples_xy_m, "samples_xy_m")
    if (elements.ndim != 2 or not len(elements) or elements.shape[1] not in (2, 3)
            or (elements.shape[1] == 3 and np.any(elements[:, 2] != 0))):
        raise ValueError("elements_m must be a nonempty planar (n,2) or (n,3) array")
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
        raise ValueError("samples_xy_m must be nonempty (m,2)")
    geometry = _real(geometry_mm, "geometry_mm", (3,)) * .001
    if geometry[0] <= 0:
        raise ValueError("generator distance must be positive")
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency_hz must be positive and finite")
    exponent = _nonnegative(q, "q")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    order = _positive_integer(quadrature_order, "quadrature_order")
    chunk = _positive_integer(chunk_size, "chunk_size")
    source = _aperture_sides(source_aperture_mm, "source_aperture_mm")
    probe = _aperture_sides(probe_aperture_mm, "probe_aperture_mm")
    terms = tuple((x, y, wx * wy)
                  for x, wx in _relative_axis(source[0], probe[0], order)
                  for y, wy in _relative_axis(source[1], probe[1], order))
    return elements, samples, geometry, 2 * np.pi * frequency / C_M_S, exponent, terms, chunk


def _independent_block(elements, samples, geometry, k, q, sign, terms):
    distance, offset_x, offset_y = geometry
    center_x = samples[:, 0, None] + offset_x - elements[None, :, 0]
    center_y = samples[:, 1, None] + offset_y - elements[None, :, 1]
    answer = np.zeros((len(samples), len(elements)), dtype=complex)
    # Independent direct propagation arithmetic, never optimizer helpers.
    for shift_x, shift_y, weight in terms:
        xx, yy = center_x + shift_x, center_y + shift_y
        radius = np.sqrt(distance * distance + xx * xx + yy * yy)
        answer += weight * ((distance / radius) ** q / radius
                            * np.exp(sign * 1j * k * radius))
    return answer


def independent_scalar_matrix(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                              *, q=2., sign=-1, source_aperture_mm=0.,
                              probe_aperture_mm=0., quadrature_order=3,
                              chunk_size=512):
    """Return the independently averaged scalar propagation matrix.

    Aperture sizes are full square sides in mm (or rectangular x/y sides).
    Each aperture has unit integrated weight, including at zero side length.
    Positive geometry offsets move the scan plane; all coordinates are metres.
    No fit-domain bound, location rounding or clipping is performed.
    """
    elements, samples, geometry, k, exponent, terms, chunk = _generator_inputs(
        elements_m, samples_xy_m, frequency_hz, geometry_mm, q, sign,
        source_aperture_mm, probe_aperture_mm, quadrature_order, chunk_size)
    answer = np.empty((len(samples), len(elements)), dtype=complex)
    for first in range(0, len(samples), chunk):
        answer[first:first + chunk] = _independent_block(
            elements, samples[first:first + chunk], geometry, k, exponent, sign, terms)
    return answer


def independent_scalar_field(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                             coefficients, *, q=2., sign=-1,
                             source_aperture_mm=0., probe_aperture_mm=0.,
                             quadrature_order=3, chunk_size=512):
    """Propagate a coefficient vector without storing the complete matrix."""
    elements, samples, geometry, k, exponent, terms, chunk = _generator_inputs(
        elements_m, samples_xy_m, frequency_hz, geometry_mm, q, sign,
        source_aperture_mm, probe_aperture_mm, quadrature_order, chunk_size)
    coefficient = np.asarray(coefficients, dtype=complex)
    if coefficient.shape != (len(elements),) or not np.all(np.isfinite(coefficient)):
        raise ValueError("coefficients must be finite and match the elements")
    answer = np.empty(len(samples), dtype=complex)
    for first in range(0, len(samples), chunk):
        block = _independent_block(elements, samples[first:first + chunk],
                                   geometry, k, exponent, sign, terms)
        answer[first:first + chunk] = block @ coefficient
    return answer


def _centered_draw(rng, count, standard_deviation):
    strength = _nonnegative(standard_deviation, "standard deviation")
    if count < 1:
        raise ValueError("at least one element is required")
    if count == 1 or strength == 0:
        return np.zeros(count)
    draw = rng.normal(size=count)
    draw -= draw.mean()
    spread = draw.std()
    if spread == 0:
        raise ValueError("degenerate random draw")
    return draw * (strength / spread)


def state_error_factors(count, rng, *, phase_std_deg=0., gain_std_db=0.):
    """Centered element errors; common log gain/phase is reserved for g.

    Requested strengths are exact empirical population standard deviations,
    not measured hardware statistics or normal-distribution guarantees.
    """
    count = _positive_integer(count, "count")
    db = _centered_draw(rng, count, gain_std_db)
    phase = _centered_draw(rng, count, phase_std_deg)
    return 10 ** (db / 20) * np.exp(1j * np.deg2rad(phase))


def local_calibration_transfer(count, frequency_hz, rng, *, height_mm=10.,
                               height_jitter_mm=.5, gain_std_db=0., sign=-1):
    """Illustrative aligned-point local transfer, normalized at nominal h.

    h=10 mm is only a scenario. Per-element uniform height jitter, 1/h gain
    and free-space phase are hypothetical, without embedded/probe de-embedding.
    """
    count = _positive_integer(count, "count")
    height = float(height_mm)
    jitter = _nonnegative(height_jitter_mm, "height_jitter_mm")
    frequency = float(frequency_hz)
    if not np.isfinite(height) or height <= jitter or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("height must exceed nonnegative jitter; frequency must be positive")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    actual = height + rng.uniform(-jitter, jitter, count)
    gain = 10 ** (_centered_draw(rng, count, gain_std_db) / 20)
    transfer = gain * height / actual * np.exp(sign * 1j * 2 * np.pi * frequency / C_M_S
                                               * (actual - height) * .001)
    return transfer


def wrong_calibration_template(values, rng, *, mode="unit_permutation",
                               gain_std_db=1., phase_std_deg=10.):
    """Return a deliberately wrong existing template, never infer its phase.

    Real amplitude input remains real and positive. Complex input permits an
    explicitly conditional historical complex-template test; this function
    never creates a zero phase from amplitude-only information.
    """
    original = np.asarray(values)
    if original.ndim != 1 or not len(original) or not np.all(np.isfinite(original)):
        raise ValueError("template must be a finite nonempty vector")
    if not np.iscomplexobj(original) and np.any(original <= 0):
        raise ValueError("amplitude-only template must be positive")
    if mode == "unit_permutation":
        permutation = rng.permutation(len(original))
        if len(original) > 1 and np.array_equal(permutation, np.arange(len(original))):
            permutation = np.roll(permutation, 1)
        return original[permutation].copy()
    if mode != "historical_version":
        raise ValueError("mode must be unit_permutation or historical_version")
    factors = state_error_factors(len(original), rng, gain_std_db=gain_std_db,
                                  phase_std_deg=phase_std_deg if np.iscomplexobj(original) else 0.)
    return original * (factors if np.iscomplexobj(original) else np.abs(factors))


@dataclass(frozen=True)
class StressScenario:
    id: str
    family: str
    expected_behavior: str
    detectability_limit: str
    positive_control: bool = False
    q: float = 2.
    source_aperture_mm: float = 0.
    probe_aperture_mm: float = 0.
    window101_offset_mm: tuple[float, float] = (0., 0.)
    session_phase_deg: float = 0.
    state_phase_std_deg: float = 0.
    state_gain_std_db: float = 0.
    calibration_gain_std_db: float = 0.
    calibration_height_mm: float | None = None
    calibration_height_jitter_mm: float = 0.
    calibration_wrong_mode: str | None = None
    historical_loaded_phase_std_deg: float = 0.
    static_amplitude_peak_to_peak_db: float | None = None
    static_phase_std_deg: float | None = None
    truth_distance_mm: float | None = None


def stress_scenarios():
    """Fixed finite scenario inventory; all strengths are labelled assumptions."""
    ambiguous = "Low residual alone can be misleading; assess held-angle shape, modes, subsets and regularization stability without truth."
    static_limit = "Shared static error is representable by b; C0 need not reject it. A wrong prior may bias geometry without an obvious residual."
    return (
        StressScenario("point_q2_control", "matched_positive", "Independent point generator agrees with the declared q2 fit model.", "A positive implementation control does not establish identifiability.", positive_control=True),
        StressScenario("session_phase45_control", "session_reference", "Extra common 45 degree phase on nonreference sessions is absorbed by g.", "Session phase is identifiable only under the fixed reference gauge.", positive_control=True, session_phase_deg=45.),
        StressScenario("aperture_source4_probe4", "independent_aperture", "Independent 4 mm source and 4 mm probe square averages challenge a point kernel.", ambiguous, source_aperture_mm=4., probe_aperture_mm=4.),
        StressScenario("aperture_source6_probe4", "independent_aperture", "Increase only source side to 6 mm to examine finite-aperture sensitivity.", ambiguous, source_aperture_mm=6., probe_aperture_mm=4.),
        StressScenario("point_q2p4", "wrong_directivity", "Generate q=2.4 and retain q=2 fitting.", ambiguous, q=2.4),
        StressScenario("window101_x1p5", "relative_window_origin", "Only 101 grids start 1.5 mm farther in x; 134 grid origin remains fixed.", "A shared-offset compromise can bias position with a small residual; independent session translations must not absorb the test.", window101_offset_mm=(1.5, 0.)),
        StressScenario("state_phase3_gain0p3", "element_state_error", "Nonreference states acquire element phase std3 degree and log gain std0.3 dB.", ambiguous, state_phase_std_deg=3., state_gain_std_db=.3),
        StressScenario("state_phase10_gain0p3", "element_state_error", "Nonreference states acquire element phase std10 degree and log gain std0.3 dB.", ambiguous, state_phase_std_deg=10., state_gain_std_db=.3),
        StressScenario("calibration_gain0p3", "local_calibration_gain", "Add relative local calibration gain std0.3 dB; scan field is unchanged.", "C0 should be unchanged. An amplitude prior may absorb or bias this discrepancy.", calibration_gain_std_db=.3),
        StressScenario("calibration_gain1", "local_calibration_gain", "Add relative local calibration gain std1 dB; scan field is unchanged.", "C0 should be unchanged. A strong wrong calibration may still produce low field residual.", calibration_gain_std_db=1.),
        StressScenario("calibration_h10_jitter0p5", "local_calibration_height", "Illustrative h=10 mm with per-element uniform +/-0.5 mm; local transfer amplitude corrupts calibration and its phase enters phase-only compensation.", "The nominal height is not actual chamber evidence. Resulting static phase can be represented by b.", calibration_height_mm=10., calibration_height_jitter_mm=.5),
        StressScenario("calibration_wrong_unit", "wrong_calibration_units", "Permute amplitude calibration unit labels consistently across frequencies.", "C0 should be unchanged. Weak calibration contrast or a weak prior may make the wrong labels undetectable.", calibration_wrong_mode="unit_permutation"),
        StressScenario("calibration_historical_amplitude", "wrong_calibration_version", "Use a hypothetical historical amplitude template with std1 dB relative change.", "Amplitude-only data supplies no historical phase; no zero phase is invented.", calibration_wrong_mode="historical_version"),
        StressScenario("loaded_historical_phase10", "wrong_loaded_version", "Hypothetical historical phase-only loading leaves a static element phase change of std10 degree.", static_limit, historical_loaded_phase_std_deg=10.),
        StressScenario("static_amplitude10db", "large_static_error", "Increase static amplitude peak-to-peak to 10 dB with original phases.", static_limit, static_amplitude_peak_to_peak_db=10.),
        StressScenario("static_phase10deg", "large_static_error", "Increase static phase empirical std to 10 degrees with original amplitudes.", static_limit, static_phase_std_deg=10.),
        StressScenario("distance72_outside", "outside_fit_domain", "Generate d=72 mm; fitting retains its 50-70 mm bound.", "A boundary hit is informative but not sufficient for a correct rejection; b can conceal an out-of-domain truth.", truth_distance_mm=72.),
        StressScenario("combined_aperture4_state3_cal0p3", "combined_after_single_factors", "Combine 4/4 mm apertures, state phase3 degree/gain0.3 dB and local calibration gain0.3 dB only after separate factors.", ambiguous, source_aperture_mm=4., probe_aperture_mm=4., state_phase_std_deg=3., state_gain_std_db=.3, calibration_gain_std_db=.3),
    )


def scenario_manifest():
    return {
        "schema_version": 1,
        "status": "development_design_no_V3_execution",
        "hardware_facts": False,
        "source_plan": "outputs/joint_geometry_plan_20261028/plan.json",
        "optimizer_review": "outputs/channel_calibration_review_20261028/optimizer_review.json",
        "execution_policy": "configs/joint_geometry_execution_policy.json",
        "independent_kernel": "Normalized uniform square source/probe scalar spatial average, tensor Gauss-Legendre order3; no optimizer import; no inferred hardware dimensions.",
        "truth_policy": "Truth is generator/scorer-only. Scenario IDs and truth errors must never decide acceptance or tune a fit. Diagnostics must use fit data only.",
        "stage_order": "V0 then V1; V2/V3 require root freeze. No 882xx case generated here.",
        "noise_pairing": "Retain each baseline noise direction and rescale by the scenario/baseline clean-field norm ratio, preserving nominal per-session SNR.",
        "session_state_definition": "Element error is independent per nonreference electronic state, centered in log gain and phase; common components belong to g. Static b error is a separate factor.",
        "model_provenance": {"requested_model": "gpt-6.1-sol", "requested_reasoning": "ultra", "actual_independently_verified": None},
        "scenarios": [asdict(scenario) for scenario in stress_scenarios()],
    }


def _normalize_amplitude(value):
    value = np.asarray(value, dtype=float)
    return value / np.exp(np.log(value).mean())


def apply_scenario(data: "VirtualFitData", truth: "VirtualTruth", scenario,
                   seed, *, allow_confirmation=False, chunk_size=512):
    """Generate a paired new case using the joint_virtual dataclass protocol.

    Public sample coordinates intentionally retain assumed origins even for a
    generator-only 101-grid displacement. Baseline objects are never modified.
    The caller must freeze before setting allow_confirmation for 882xx roots.
    This adapter is a generator and must never be called by a fitting function.
    """
    from .joint_virtual import VirtualFitData, VirtualTruth, frequency_key

    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if 88200 <= seed < 88300 and not allow_confirmation:
        raise ValueError("882xx confirmation remains sealed; a separate root freeze is required")
    inventory = stress_scenarios()
    lookup = {item.id: item for item in inventory}
    selected = lookup.get(scenario) if isinstance(scenario, str) else scenario
    if not isinstance(selected, StressScenario) or selected.id not in lookup or selected != lookup[selected.id]:
        raise ValueError("scenario must be an unchanged registered StressScenario or its id")
    scenario_index = list(lookup).index(selected.id)
    sequence = np.random.SeedSequence([int(seed), int(data.metadata.get("case_index", 0)), 7391, scenario_index])
    streams = sequence.spawn(3)
    state_rng, calibration_rng, version_rng = [np.random.default_rng(ss) for ss in streams]
    n = len(data.elements_m)
    weights = {key: np.array(value, complex, copy=True) for key, value in truth.weights.items()}
    calibration = {key: np.array(value, float, copy=True) for key, value in data.calibration_relative_amplitude.items()}
    frequencies = {frequency_key(ss.frequency_hz): ss.frequency_hz for ss in data.sessions}
    permutation = version_rng.permutation(n)
    if n > 1 and np.array_equal(permutation, np.arange(n)):
        permutation = np.roll(permutation, 1)
    for key, value in weights.items():
        if selected.static_amplitude_peak_to_peak_db is not None:
            db = 20 * np.log10(np.abs(value))
            spread = np.ptp(db)
            if spread == 0:
                raise ValueError("cannot enlarge a constant static amplitude without a new stated draw")
            db = db.mean() + (db - db.mean()) * selected.static_amplitude_peak_to_peak_db / spread
            value = 10 ** (db / 20) * np.exp(1j * np.angle(value))
        if selected.static_phase_std_deg is not None:
            phase = np.unwrap(np.angle(value))
            if phase.std() == 0:
                raise ValueError("cannot enlarge constant static phase without a new stated draw")
            phase = phase.mean() + (phase - phase.mean()) * np.deg2rad(selected.static_phase_std_deg) / phase.std()
            value = np.abs(value) * np.exp(1j * phase)
        if selected.historical_loaded_phase_std_deg:
            value *= state_error_factors(n, version_rng, phase_std_deg=selected.historical_loaded_phase_std_deg)
        if selected.calibration_height_mm is not None:
            transfer = local_calibration_transfer(n, frequencies[key], calibration_rng,
                height_mm=selected.calibration_height_mm,
                height_jitter_mm=selected.calibration_height_jitter_mm)
            calibration[key] *= np.abs(transfer)
            # Explicit hypothetical negative compensation of the extra local
            # phase; its height/phase is never supplied to the fitting model.
            value *= np.exp(-1j * np.angle(transfer))
        if selected.calibration_gain_std_db:
            calibration[key] *= np.abs(state_error_factors(n, calibration_rng,
                gain_std_db=selected.calibration_gain_std_db))
        if selected.calibration_wrong_mode == "unit_permutation":
            calibration[key] = calibration[key][permutation]
        elif selected.calibration_wrong_mode == "historical_version":
            calibration[key] = wrong_calibration_template(calibration[key], version_rng,
                                                          mode="historical_version")
        calibration[key] = _normalize_amplitude(calibration[key])
        weights[key] = value
    geometry = np.array(truth.geometry_mm, float, copy=True)
    if selected.truth_distance_mm is not None:
        geometry[0] = selected.truth_distance_mm
    regenerate = (selected.id == "point_q2_control" or selected.q != 2.
        or selected.source_aperture_mm > 0 or selected.probe_aperture_mm > 0
        or any(selected.window101_offset_mm) or selected.state_phase_std_deg > 0
        or selected.state_gain_std_db > 0 or selected.calibration_height_mm is not None
        or selected.historical_loaded_phase_std_deg > 0
        or selected.static_amplitude_peak_to_peak_db is not None
        or selected.static_phase_std_deg is not None or selected.truth_distance_mm is not None)
    sessions, clean, scalars, states = [], {}, {}, {}
    for session in data.sessions:
        key, name = frequency_key(session.frequency_hz), session.name
        gain = complex(truth.session_scalars[name])
        state = np.array(truth.state_factors[name], complex, copy=True)
        if session.angle_deg != 0:
            gain *= np.exp(1j * np.deg2rad(selected.session_phase_deg))
            if selected.state_phase_std_deg or selected.state_gain_std_db:
                state *= state_error_factors(n, state_rng,
                    phase_std_deg=selected.state_phase_std_deg,
                    gain_std_db=selected.state_gain_std_db)
        if regenerate:
            xy = np.array(session.samples_xy_m, copy=True)
            if session.grid_side == 101:
                xy += np.asarray(selected.window101_offset_mm) * .001
            generated = gain * independent_scalar_field(data.elements_m, xy,
                session.frequency_hz, geometry, session.steering * weights[key] * state,
                q=selected.q, source_aperture_mm=selected.source_aperture_mm,
                probe_aperture_mm=selected.probe_aperture_mm, chunk_size=chunk_size)
        else:
            generated = np.array(truth.clean_fields[name], copy=True)
            generated *= gain / truth.session_scalars[name]
        old_clean = truth.clean_fields[name]
        old_norm = np.linalg.norm(old_clean)
        if old_norm == 0:
            raise ValueError("baseline clean field must have nonzero norm for noise pairing")
        noise = session.observed_field - old_clean
        observed = generated + noise * (np.linalg.norm(generated) / old_norm)
        sessions.append(replace(session, observed_field=observed))
        clean[name], scalars[name], states[name] = generated, gain, state
    metadata = dict(data.metadata)
    metadata["stress"] = {
        "scenario": asdict(selected), "hypothesis_not_hardware_fact": True,
        "quadrature_order": 3, "generator_independent_of_fit_kernel": True,
        "root_seed": int(seed), "stream_entropy": sequence.entropy,
        "streams": {key: {"spawn_key": list(ss.spawn_key)}
                    for key, ss in zip(("state_errors", "calibration", "version"), streams)},
        "noise_pairing": "baseline direction, scaled by clean norm ratio",
        "truth_used_only_by_generator": True,
        "acceptance_must_not_use_scenario_or_truth": True,
    }
    return (VirtualFitData(np.array(data.elements_m, copy=True), tuple(sessions), calibration, metadata),
            VirtualTruth(geometry, weights, clean, scalars, states))
