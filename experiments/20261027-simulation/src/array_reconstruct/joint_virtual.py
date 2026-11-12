"""Independent, truth-separated synthetic development for shared geometry.

All chamber choices below are labelled hypotheses.  No measured field is read.
The public fit object deliberately contains neither geometry nor clean fields.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
import csv
import hashlib
import io
import json
from typing import Any

import numpy as np

C_M_S = 299792458.0
STREAM_NAMES = ("geometry", "weights", "noise", "calibration", "session_scalars", "state_errors")
TRAIN_ANGLES = (0.0, 30.0, -30.0, -15.0)
INTERNAL_ANGLES = (15.0,)
SEALED_ANGLES = (-45.0, 45.0)
EXPLICIT_GEOMETRIES_MM = (
    (62.3456, 1.2345, -3.4567),
    (51.7312, -8.2345, 8.7654),
    (68.6421, 7.8912, -6.5432),
)


@dataclass(frozen=True)
class VirtualSession:
    name: str
    frequency_hz: float
    angle_deg: float
    samples_xy_m: np.ndarray
    steering: np.ndarray
    observed_field: np.ndarray
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    grid_side: int
    split: str


@dataclass(frozen=True)
class VirtualFitData:
    elements_m: np.ndarray
    sessions: tuple[VirtualSession, ...]
    calibration_relative_amplitude: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VirtualTruth:
    geometry_mm: np.ndarray
    weights: dict[str, np.ndarray]
    clean_fields: dict[str, np.ndarray]
    session_scalars: dict[str, complex]
    state_factors: dict[str, np.ndarray]


def frequency_key(frequency_hz: float) -> str:
    return f"f{frequency_hz / 1e9:.1f}"


def reference_elements(array_size: int, path: Path | str | None = None) -> tuple[np.ndarray, dict]:
    """Read the exact registered 1024-unit order, or a labelled small formula."""
    if array_size not in (2, 3, 4, 8, 16, 32):
        raise ValueError("supported array sides: 2,3,4,8,16,32")
    if array_size == 32:
        path = (Path(path).resolve() if path is not None else
                files("array_reconstruct").joinpath("data/reference_element_geometry.csv"))
        raw = path.read_bytes()
        with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
            records = list(csv.DictReader(handle))
        records.sort(key=lambda row: int(row["unit"]))
        if [int(row["unit"]) for row in records] != list(range(1024)):
            raise ValueError("registered geometry must contain each unit 0..1023 exactly once")
        xy = np.array([[float(row["x_mm"]), float(row["y_mm"])] for row in records])
        provenance = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "order": "ascending unit", "formula_only": False}
    else:
        rr, cc = np.meshgrid(np.arange(array_size), np.arange(array_size), indexing="ij")
        xy = np.column_stack((11 * cc.ravel() + 5.5 * (rr.ravel() % 2), 11 * rr.ravel()))
        provenance = {"formula": "x_mm=11*col+5.5*(row%2);y_mm=11*row", "order": "row-major small-array formula", "formula_only": True}
    center = (xy.max(axis=0) + xy.min(axis=0)) / 2
    provenance["center_subtracted_mm"] = center.tolist()
    return (xy - center) * 1e-3, provenance


def scan_grid(grid_side: int) -> np.ndarray:
    axis = (np.arange(grid_side) - (grid_side - 1) / 2) * .003
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def scalar_regions(grid_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Spatially interleaved, disjoint scalar-calibration and score regions.

    One in five diagonal-index classes calibrates a single complex gain.
    All remaining points score shape; no fitted per-element held-angle weights.
    """
    row, col = np.indices((grid_side, grid_side))
    calibration = (col - 2 * row) % 5 == 0
    return np.flatnonzero(calibration), np.flatnonzero(~calibration)


def independent_field(elements_m: np.ndarray, samples_xy_m: np.ndarray,
                      frequency_hz: float, geometry_mm: np.ndarray,
                      coefficients: np.ndarray, *, q: float = 2., sign: int = -1,
                      block_size: int = 512) -> np.ndarray:
    """Generator-side scalar propagation, independent of optimizer helpers."""
    d, dx, dy = np.asarray(geometry_mm, dtype=float) * .001
    k = 2 * np.pi * frequency_hz / C_M_S
    answer = np.empty(len(samples_xy_m), dtype=complex)
    for first in range(0, len(samples_xy_m), block_size):
        pp = samples_xy_m[first:first + block_size]
        xx = pp[:, 0, None] + dx - elements_m[None, :, 0]
        yy = pp[:, 1, None] + dy - elements_m[None, :, 1]
        rr = np.sqrt(xx * xx + yy * yy + d * d)
        matrix = (d / rr) ** q * np.exp(sign * 1j * k * rr) / rr
        answer[first:first + len(pp)] = matrix @ coefficients
    return answer


def _static_weights(rng: np.random.Generator, n: int, phase_std_deg: float) -> np.ndarray:
    db = rng.uniform(-1., 1., n)
    db = (db - (db.min() + db.max()) / 2) * (5. / np.ptp(db))
    phase = rng.normal(size=n)
    phase -= phase.mean()
    phase *= phase_std_deg / phase.std()
    return 10 ** (db / 20) * np.exp(1j * np.deg2rad(phase))


def generate_virtual_case(seed: int, array_size: int = 8, *, case_index: int = 0,
                          snr_db: float | None = 40., phase_std_deg: float = 5.,
                          train_stride: int | None = None,
                          calibration_noise_db: float = .15,
                          state_error_std_deg: float = 0.,
                          geometry_mm: np.ndarray | None = None,
                          allow_confirmation: bool = False) -> tuple[VirtualFitData, VirtualTruth]:
    """Generate all observations once; each method receives the same fit data.

    ``case_index=3`` chooses continuous random geometry from its own stream.
    Confirmation roots are deliberately disabled until an explicit freeze.
    """
    if 88200 <= seed < 88300 and not allow_confirmation:
        raise ValueError("882xx confirmation remains sealed; a separate freeze is required")
    if case_index < 0:
        raise ValueError("case_index must be nonnegative")
    if not np.isfinite(phase_std_deg) or phase_std_deg < 0:
        raise ValueError("phase_std_deg must be finite and nonnegative")
    if train_stride is not None and train_stride <= 0:
        raise ValueError("train_stride must be positive")
    if snr_db is not None and not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite or None")
    root = np.random.SeedSequence([int(seed), int(case_index)])
    sequences = root.spawn(len(STREAM_NAMES))
    rng = {key: np.random.default_rng(ss) for key, ss in zip(STREAM_NAMES, sequences)}
    elements, element_source = reference_elements(array_size)
    if geometry_mm is None:
        geometry = (np.array(EXPLICIT_GEOMETRIES_MM[case_index]) if case_index < 3 else
                    rng["geometry"].uniform([50.4, -9.5, -9.5], [69.6, 9.5, 9.5]))
    else:
        geometry = np.asarray(geometry_mm, dtype=float).copy()
    if geometry.shape != (3,) or not np.all(np.isfinite(geometry)):
        raise ValueError("geometry must have three entries")
    train_stride = train_stride or {2: 12, 3: 12, 4: 10, 8: 8, 16: 6, 32: 5}[array_size]
    weights = {frequency_key(f): _static_weights(rng["weights"], len(elements), phase_std_deg)
               for f in (12.7e9, 12.1e9, 12.5e9)}
    calibration = {}
    for key, bb in weights.items():
        amp = np.abs(bb) * 10 ** (rng["calibration"].normal(0, calibration_noise_db, len(bb)) / 20)
        calibration[key] = amp / np.exp(np.log(amp).mean())
    sessions, clean, scalars, states = [], {}, {}, {}
    # Actual relative origins remain unknown; these are explicitly coaxial,
    # centered synthetic windows, with the registered physical 3 mm pitch.
    specs = [(12.7e9, angle, 134 if angle in (0., 30.) else 101)
             for angle in (0., 30., -30., -45., -15., 15., 45.)]
    specs += [(12.1e9, 0., 134), (12.5e9, 0., 134)]
    for frequency, angle, side in specs:
        key = frequency_key(frequency)
        name = f"{key}_angle{angle:+g}"
        xy = scan_grid(side)
        # Explicit hypothetical electronic steering axis x.  For exp(-ikR)
        # propagation, -sin(angle) phase excites the chosen +x far direction.
        steering = np.exp(-1j * 2 * np.pi * frequency / C_M_S * elements[:, 0] * np.sin(np.deg2rad(angle)))
        g = (1. + 0j) if angle == 0 else np.exp(rng["session_scalars"].normal(0, .05) +
                                               1j * np.deg2rad(rng["session_scalars"].normal(0, 20.)))
        state_phase = rng["state_errors"].normal(0, state_error_std_deg, len(elements))
        state_phase -= state_phase.mean()
        state = np.exp(1j * np.deg2rad(state_phase))
        yy = g * independent_field(elements, xy, frequency, geometry, steering * weights[key] * state)
        # Noise stream is consumed even for a noiseless paired case, keeping
        # every other stream and all session noise draws aligned across SNRs.
        zz = (rng["noise"].normal(size=len(yy)) + 1j * rng["noise"].normal(size=len(yy))) / np.sqrt(2)
        sigma = 0. if snr_db is None else np.linalg.norm(yy) / np.sqrt(len(yy)) * 10 ** (-snr_db / 20)
        observed = yy + sigma * zz
        rr, cc = np.indices((side, side))
        train = np.flatnonzero((rr % train_stride == 0) & (cc % train_stride == 0))
        cal_idx, score_idx = scalar_regions(side)
        split = ("train" if frequency != 12.7e9 or angle in TRAIN_ANGLES else
                 "internal" if angle in INTERNAL_ANGLES else "sealed")
        sessions.append(VirtualSession(name, frequency, angle, xy, steering, observed,
                                       train, cal_idx, score_idx, side, split))
        clean[name], scalars[name], states[name] = yy, g, state
    metadata = {
        "synthetic": True, "seed": seed, "case_index": case_index, "array_size": array_size,
        "snr_db": snr_db, "phase_std_deg": phase_std_deg,
        "weight_amplitude_peak_to_peak_db": 5., "train_stride": train_stride,
        "calibration_noise_std_db": calibration_noise_db,
        "state_error_std_deg": state_error_std_deg,
        "random_streams": {key: {"entropy": ss.entropy, "spawn_key": list(ss.spawn_key)}
                           for key, ss in zip(STREAM_NAMES, sequences)},
        "element_source": element_source, "scan_pitch_m": .003,
        "assumptions": {
            "window_origins": "centered and coaxial synthetic assumption; real relative origins unknown",
            "electronic_steering": "ideal phase ramp on element x; u=exp(-i*k*x*sin(angle)); scalar propagation exp(-i*k*R); positive angle toward +x",
            "geometry_offset": "positive dx,dy move scan samples: X=x_nominal+dx-x_element",
            "generator_kernel": "independently implemented scalar cos^2/R matched positive control",
            "session_gain": "g_ref=1 for broadside at each frequency; other rho~N(0,.05),phase~N(0,20deg)",
            "calibration": f"relative amplitude only, lognormal {calibration_noise_db:g}dB assumed error; no loaded phase evidence",
            "hardware_statistics": False},
        "folds": {"train_angles_deg": list(TRAIN_ANGLES), "internal_angles_deg": list(INTERNAL_ANGLES),
                  "sealed_angles_deg": list(SEALED_ANGLES),
                  "scalar_regions": "(col-2*row)%5==0 calibration; complement evaluation, full grid"},
        "model_provenance": {"requested_model": "gpt-6.1-sol", "requested_reasoning": "ultra", "actual_independently_verified": None},
    }
    return VirtualFitData(elements, tuple(sessions), calibration, metadata), VirtualTruth(geometry, weights, clean, scalars, states)


def observation_fingerprint(data: VirtualFitData) -> str:
    """Hash all fixed public observations/conventions except training indices."""
    digest = hashlib.sha256()
    def add(name, value):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    add("elements_m", data.elements_m)
    for ss in data.sessions:
        digest.update(json.dumps([ss.name, ss.frequency_hz, ss.angle_deg, ss.grid_side, ss.split]).encode("utf-8"))
        for key in ("samples_xy_m", "steering", "observed_field", "calibration_indices", "evaluation_indices"):
            add(ss.name+"/"+key, getattr(ss, key))
    for key in sorted(data.calibration_relative_amplitude):
        add("calibration/"+key, data.calibration_relative_amplitude[key])
    digest.update(json.dumps(data.metadata.get("assumptions", {}), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def with_training_stride(data: VirtualFitData, stride: int) -> VirtualFitData:
    """Change only public training indices on the existing physical 3 mm grid.

    The observations, noise realization, steering, calibration and all three
    evaluation folds remain byte-identical. No truth input is accepted.
    """
    if isinstance(stride, bool) or int(stride) != stride or stride <= 0:
        raise ValueError("training stride must be a positive integer")
    stride = int(stride)
    sessions, layouts = [], {}
    for ss in data.sessions:
        row, col = np.indices((ss.grid_side, ss.grid_side))
        indices = np.flatnonzero((row % stride == 0) & (col % stride == 0))
        changed = replace(ss, train_indices=indices) if ss.split == "train" else ss
        sessions.append(changed)
        if ss.split == "train":
            selected = ss.samples_xy_m[indices]
            layouts[ss.name] = {"n_training": len(indices), "nominal_pitch_mm": 3*stride,
                "training_min_xy_mm": (selected.min(axis=0)*1000).tolist(),
                "training_max_xy_mm": (selected.max(axis=0)*1000).tolist(),
                "training_mean_xy_mm": (selected.mean(axis=0)*1000).tolist(),
                "full_window_min_xy_mm": (ss.samples_xy_m.min(axis=0)*1000).tolist(),
                "full_window_max_xy_mm": (ss.samples_xy_m.max(axis=0)*1000).tolist()}
    metadata = dict(data.metadata, train_stride=int(stride), sampling_control={
        "source_train_stride": data.metadata["train_stride"], "new_train_stride": int(stride),
        "physical_grid_pitch_mm": 3, "changed": "training indices only",
        "unchanged_public_observation_sha256": observation_fingerprint(data),
        "observations_steering_calibration_and_folds_unchanged": True,
        "window_assumption": "retain centered 134/101 original grids and their 1.5mm half-step difference; no recenter/crop",
        "index_rule": "row%stride==0 and col%stride==0; retained endpoint asymmetry is recorded",
        "session_training_layouts": layouts})
    return replace(data, sessions=tuple(sessions), metadata=metadata)


def save_virtual_fit_data(directory: Path | str, data: VirtualFitData) -> None:
    """Persist a public-only fit artifact, including matched sampling controls."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    public = {"elements_m": data.elements_m}
    descriptions = []
    for index, ss in enumerate(data.sessions):
        prefix = f"session{index}_"
        for key in ("samples_xy_m", "steering", "observed_field", "train_indices", "calibration_indices", "evaluation_indices"):
            public[prefix + key] = getattr(ss, key)
        descriptions.append({"name": ss.name, "frequency_hz": ss.frequency_hz, "angle_deg": ss.angle_deg,
                             "grid_side": ss.grid_side, "split": ss.split, "array_prefix": prefix})
    for key, value in data.calibration_relative_amplitude.items():
        public["calibration_" + key] = value
    np.savez_compressed(directory / "fit_data.npz", **public)
    manifest = dict(data.metadata, sessions=descriptions)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def save_virtual_case(directory: Path | str, data: VirtualFitData, truth: VirtualTruth) -> None:
    save_virtual_fit_data(directory, data)
    hidden = {"geometry_mm": truth.geometry_mm}
    for index, ss in enumerate(data.sessions):
        prefix = f"session{index}_"
        hidden[prefix+"clean_field"] = truth.clean_fields[ss.name]
        hidden[prefix+"scalar"] = np.array(truth.session_scalars[ss.name])
        hidden[prefix+"state_factor"] = truth.state_factors[ss.name]
    for key, value in truth.weights.items():
        hidden["weights_"+key] = value
    np.savez_compressed(Path(directory) / "truth_for_scoring_only.npz", **hidden)


def load_virtual_fit_data(directory: Path | str) -> VirtualFitData:
    """Load public observations only. This never opens the truth artifact."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True:
        raise ValueError("only labelled synthetic observations are accepted")
    with np.load(directory / "fit_data.npz", allow_pickle=False) as values:
        sessions = []
        for item in manifest["sessions"]:
            pp = item["array_prefix"]
            sessions.append(VirtualSession(item["name"], item["frequency_hz"], item["angle_deg"],
                *[values[pp+key].copy() for key in ("samples_xy_m", "steering", "observed_field", "train_indices",
                                                   "calibration_indices", "evaluation_indices")],
                item["grid_side"], item["split"]))
        calibration = {key.removeprefix("calibration_"): values[key].copy()
                       for key in values.files if key.startswith("calibration_")}
        elements = values["elements_m"].copy()
    metadata = {key: value for key, value in manifest.items() if key != "sessions"}
    return VirtualFitData(elements, tuple(sessions), calibration, metadata)


def build_truth_reference_binding(owner_directory: Path | str, referenced_directory: Path | str) -> dict:
    """Seal reference bytes without parsing latent truth arrays.

    An existing unbound historical link is reported, not rewritten. Existing
    bound links must still verify; a new binding cannot repair stale sources.
    """
    owner, target = Path(owner_directory).resolve(), Path(referenced_directory).resolve()
    def hashes(folder, names):
        return {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    chain, seen, legacy = [], {owner}, []
    current = target
    for _ in range(8):
        if current in seen:
            raise ValueError("cyclic synthetic truth reference")
        seen.add(current)
        row = {"dataset_directory": str(current),
               "files_sha256": hashes(current, ("manifest.json", "fit_data.npz"))}
        truth_file = current / "truth_for_scoring_only.npz"
        if truth_file.exists():
            row["files_sha256"][truth_file.name] = hashlib.sha256(truth_file.read_bytes()).hexdigest()
            row["terminal_truth_artifact"] = True
            chain.append(row)
            break
        ref_path = current / "truth_reference.json"
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        row["files_sha256"][ref_path.name] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
        row["terminal_truth_artifact"] = False
        next_directory = Path(ref["dataset_directory"]).resolve()
        if "hash_binding" in ref:
            verify_truth_reference_binding(ref["hash_binding"], expected_owner=current, expected_target=next_directory)
        else:
            legacy.append(str(ref_path))
        chain.append(row)
        current = next_directory
    else:
        raise ValueError("synthetic truth reference exceeds eight links")
    return {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "owner_dataset_directory": str(owner), "referenced_dataset_directory": str(target),
        "owner_files_sha256": hashes(owner, ("manifest.json", "fit_data.npz")),
        "owner_reference_sha256": (hashlib.sha256((owner/"truth_reference.json").read_bytes()).hexdigest()
                                   if (owner/"truth_reference.json").exists() else None),
        "reference_chain": chain, "historical_unbound_reference_paths": legacy,
        "role": "opaque SHA256 provenance sealing only; truth bytes hashed without array parsing",
        "truth_bytes_read_for_hash": True, "truth_array_values_parsed": False}


def verify_truth_reference_binding(binding: dict, *, expected_owner=None, expected_target=None) -> dict:
    """Check exact owner/public/manifest/truth bytes before final truth loading."""
    if binding.get("schema_version") != 2:
        raise ValueError("unsupported synthetic reference binding")
    owner = Path(binding["owner_dataset_directory"]).resolve()
    target = Path(binding["referenced_dataset_directory"]).resolve()
    if expected_owner is not None and owner != Path(expected_owner).resolve():
        raise ValueError("synthetic reference binding belongs to another owner")
    if expected_target is not None and target != Path(expected_target).resolve():
        raise ValueError("synthetic reference binding points to another target")
    def verify_files(folder, values):
        allowed = {"manifest.json", "fit_data.npz", "truth_for_scoring_only.npz", "truth_reference.json"}
        for name, expected in values.items():
            if name not in allowed or hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"stale synthetic reference bytes: {folder/name}")
    if not {"manifest.json", "fit_data.npz"}.issubset(binding["owner_files_sha256"]):
        raise ValueError("owner public or manifest hash missing")
    verify_files(owner, binding["owner_files_sha256"])
    if binding.get("owner_reference_sha256") is not None:
        if hashlib.sha256((owner/"truth_reference.json").read_bytes()).hexdigest() != binding["owner_reference_sha256"]:
            raise ValueError("stale owner truth-reference metadata")
    chain = binding["reference_chain"]
    if not chain or len(chain) > 8:
        raise ValueError("invalid synthetic reference binding chain")
    expected_directory, seen = target, {owner}
    terminal = None
    for index, row in enumerate(chain):
        current = Path(row["dataset_directory"]).resolve()
        if current != expected_directory or current in seen:
            raise ValueError("broken or cyclic synthetic reference binding chain")
        seen.add(current)
        values = row["files_sha256"]
        if not {"manifest.json", "fit_data.npz"}.issubset(values):
            raise ValueError("synthetic reference omitted public or manifest hash")
        verify_files(current, values)
        if row["terminal_truth_artifact"]:
            if "truth_for_scoring_only.npz" not in values or index != len(chain)-1:
                raise ValueError("invalid terminal truth binding")
            terminal = current / "truth_for_scoring_only.npz"
        else:
            if "truth_reference.json" not in values:
                raise ValueError("reference metadata hash missing")
            link = json.loads((current / "truth_reference.json").read_text(encoding="utf-8"))
            expected_directory = Path(link["dataset_directory"]).resolve()
    if terminal is None:
        raise ValueError("reference binding has no terminal truth artifact")
    return {"verified": True, "terminal_truth_artifact": str(terminal),
            "historical_unbound_reference_paths": binding.get("historical_unbound_reference_paths", []),
            "truth_array_values_parsed": False}


def load_virtual_truth_for_scoring(directory: Path | str, *, supplementary_binding=None) -> VirtualTruth:
    """Explicitly separate final scorer entry; fit/selection must not call it."""
    directory = Path(directory)
    initial_directory = directory.resolve()
    supplement = None
    if supplementary_binding is not None:
        supplement = (json.loads(Path(supplementary_binding).read_text(encoding="utf-8"))
                      if isinstance(supplementary_binding, (str, Path)) else supplementary_binding)
    visited = set()
    while not (directory / "truth_for_scoring_only.npz").exists():
        resolved = directory.resolve()
        if resolved in visited or len(visited) >= 8:
            raise ValueError("invalid synthetic truth-reference chain")
        visited.add(resolved)
        reference = json.loads((directory / "truth_reference.json").read_text(encoding="utf-8"))
        target = Path(reference["dataset_directory"])
        if "hash_binding" in reference:
            verified = verify_truth_reference_binding(reference["hash_binding"], expected_owner=directory, expected_target=target)
        elif supplement is not None and resolved == initial_directory:
            verified = verify_truth_reference_binding(supplement, expected_owner=directory, expected_target=target)
        else:
            raise ValueError("historical truth reference is unbound; explicit supplementary binding required")
        directory = Path(verified["terminal_truth_artifact"]).parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / "truth_for_scoring_only.npz", allow_pickle=False) as values:
        weights = {key.removeprefix("weights_"): values[key].copy() for key in values.files if key.startswith("weights_")}
        clean, scalars, states = {}, {}, {}
        for item in manifest["sessions"]:
            pp, name = item["array_prefix"], item["name"]
            clean[name] = values[pp+"clean_field"].copy()
            scalars[name] = complex(values[pp+"scalar"].item())
            states[name] = values[pp+"state_factor"].copy()
        geometry = values["geometry_mm"].copy()
    return VirtualTruth(geometry, weights, clean, scalars, states)


def make_problem(data: VirtualFitData, lambda_reg: float, *, model: str = "M2",
                 backend: str = "normal", normal_qr_check: str = "first",
                 branch: str = "C0", template_strength: float = 0.,
                 amplitude_strength: float = 0., amplitude_n_starts: int = 3,
                 compute_device: str = "cpu", gpu_fallback: str = "raise"):
    """Fit-only adapter; no truth is accepted by this function."""
    from .joint_geometry import Session, FrequencyGroup, JointProblem
    sessions = [ss for ss in data.sessions if ss.split == "train"]
    frequency_counts = {}
    for ss in sessions:
        key = frequency_key(ss.frequency_hz)
        frequency_counts[key] = frequency_counts.get(key, 0) + 1
    grouped: dict[str, list] = {}
    for ss in sessions:
        key = ss.name if model == "M1" else frequency_key(ss.frequency_hz)
        grouped.setdefault(key, []).append(ss)
    if branch not in ("C0", "Cphaseonly_hypothesis", "Cwrong", "Csoft_amplitude"):
        raise ValueError("branch requires its own validated nonlinear adapter")
    groups = []
    for key, items in grouped.items():
        native = []
        for ss in items:
            ii = ss.train_indices
            # Norm and all weighting constants are frozen from training only.
            weight = 1 / (np.sqrt(len(frequency_counts) * frequency_counts[frequency_key(ss.frequency_hz)]) *
                          np.linalg.norm(ss.observed_field[ii]))
            native.append(Session(ss.name, ss.frequency_hz, ss.samples_xy_m[ii], ss.steering,
                                  ss.observed_field[ii], weight=weight))
        template = None
        if branch in ("Cphaseonly_hypothesis", "Cwrong"):
            template = data.calibration_relative_amplitude[frequency_key(items[0].frequency_hz)].astype(complex)
            if branch == "Cwrong":
                # Fixed cyclic unit permutation: wrong information, not truth.
                template = np.roll(template, max(1, len(template) // 3))
        groups.append(FrequencyGroup(key, tuple(native), lambda_reg,
                                     template_prior=template,
                                     template_strength=template_strength if template is not None else 0.,
                                     amplitude_prior=(data.calibration_relative_amplitude[frequency_key(items[0].frequency_hz)]
                                                      if branch == "Csoft_amplitude" else None),
                                     amplitude_strength=amplitude_strength if branch == "Csoft_amplitude" else 0.,
                                     amplitude_n_starts=amplitude_n_starts))
    problem = JointProblem(data.elements_m, tuple(groups), q=2, sign=-1, backend=backend,
                           normal_condition_max=1e8, normal_qr_check=normal_qr_check)
    if compute_device == "cpu":
        return problem
    from .joint_geometry_gpu import GPUJointProblem
    return GPUJointProblem.from_problem(problem, device=compute_device, fallback=gpu_fallback)


def fit_record(fit) -> dict:
    def serial(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, (np.integer, np.floating)): return value.item()
        if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [serial(v) for v in value]
        if isinstance(value, complex): return [value.real, value.imag]
        return value
    return serial({key: getattr(fit, key) for key in
                   ("geometry_mm", "x", "cost", "success", "status", "message", "nfev", "njev",
                    "optimality", "active_mask", "counts", "elapsed_seconds")} | {
                       "linear_diagnostics": fit.evaluation.diagnostics,
                       "gains": fit.evaluation.gains})


def scalar_prediction_metrics(problem, fit, session: VirtualSession, *, clean_field=None,
                              block_size: int = 512) -> dict:
    """Full physical grid/columns; only one scalar uses calibration stripe."""
    from .joint_geometry import Session
    key = frequency_key(session.frequency_hz)
    prediction = np.empty(len(session.samples_xy_m), dtype=complex)
    for first in range(0, len(prediction), block_size):
        ii = slice(first, first + block_size)
        subset = Session(session.name, session.frequency_hz, session.samples_xy_m[ii], session.steering,
                         session.observed_field[ii])
        prediction[ii] = problem.predict(subset, fit.geometry_mm, fit.b[key])
    cc, ee = session.calibration_indices, session.evaluation_indices
    scalar = np.vdot(prediction[cc], session.observed_field[cc]) / np.vdot(prediction[cc], prediction[cc])
    answer = {"scalar_complex": [float(scalar.real), float(scalar.imag)],
              "noisy_nrmse": float(np.linalg.norm(scalar * prediction[ee] - session.observed_field[ee]) /
                                    np.linalg.norm(session.observed_field[ee])),
              "n_calibration": len(cc), "n_evaluation": len(ee),
              "full_grid": True, "columns": len(problem.elements_m)}
    if clean_field is not None:
        answer["clean_nrmse"] = float(np.linalg.norm(scalar * prediction[ee] - clean_field[ee]) / np.linalg.norm(clean_field[ee]))
    return answer


def internal_selection_score(problem, fit, data: VirtualFitData) -> dict:
    sessions = [ss for ss in data.sessions if ss.split == "internal"]
    metrics = {ss.name: scalar_prediction_metrics(problem, fit, ss) for ss in sessions}
    return {"sessions": metrics, "mean_noisy_nrmse": float(np.mean([vv["noisy_nrmse"] for vv in metrics.values()]))}


def m1_spatial_selection_score(problem, fit, data: VirtualFitData) -> dict:
    """Same-state spatial interpolation for independent-response M1.

    This does not predict any unseen angle and fits no new response at scoring.
    Frequency and angle averaging remain exactly equal to M2 training weights.
    """
    from .joint_geometry import Session
    metrics, frequencies = {}, {}
    for ss in data.sessions:
        if ss.split != "train":
            continue
        ee = np.setdiff1d(np.arange(len(ss.observed_field)), ss.train_indices, assume_unique=True)
        prediction = np.empty(len(ee), complex)
        for first in range(0, len(ee), 512):
            ii = ee[first:first+512]
            subset = Session(ss.name, ss.frequency_hz, ss.samples_xy_m[ii], ss.steering, ss.observed_field[ii])
            prediction[first:first+len(ii)] = problem.predict(subset, fit.geometry_mm, fit.b[ss.name])
        error = float(np.linalg.norm(prediction-ss.observed_field[ee])/np.linalg.norm(ss.observed_field[ee]))
        metrics[ss.name] = {"noisy_nrmse": error, "n_evaluation": len(ee),
                            "n_training": len(ss.train_indices), "columns": len(data.elements_m)}
        frequencies.setdefault(frequency_key(ss.frequency_hz), []).append(error)
    return {"sessions": metrics,
            "mean_noisy_nrmse": float(np.mean([np.mean(values) for values in frequencies.values()])),
            "interpretation": "M1 same-state spatial interpolation; not unseen-angle prediction"}


def final_m1_truth_score(fit, truth: VirtualTruth) -> dict:
    error = fit.geometry_mm-truth.geometry_mm
    return {"geometry_error_mm": error.tolist(), "geometry_max_abs_error_mm": float(np.max(np.abs(error))),
            "interpretation": "M1 geometry truth only; no unseen-angle shared-response prediction exists"}


def final_truth_score(problem, fit, data: VirtualFitData, truth: VirtualTruth) -> dict:
    """Only call once model/hyperparameters/start selection is complete."""
    metrics = {ss.name: scalar_prediction_metrics(problem, fit, ss, clean_field=truth.clean_fields[ss.name])
               for ss in data.sessions if ss.split == "sealed"}
    coefficients = {}
    for key, expected in truth.weights.items():
        recovered = fit.b[key]
        gauge = np.vdot(recovered, expected) / np.vdot(recovered, recovered)
        aligned = gauge * recovered
        phase = np.angle(aligned / expected)
        coefficients[key] = {
            "amplitude_rmse_db": float(np.sqrt(np.mean((20 * np.log10(np.abs(aligned / expected))) ** 2))),
            "phase_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(phase ** 2)))),
            "complex_nrmse": float(np.linalg.norm(aligned - expected) / np.linalg.norm(expected)),
            "public_complex_gauge": [float(gauge.real), float(gauge.imag)]}
    error = fit.geometry_mm - truth.geometry_mm
    return {"geometry_error_mm": error.tolist(), "geometry_max_abs_error_mm": float(np.max(np.abs(error))),
            "unseen_angles": metrics, "weights": coefficients,
            "full_1024_columns": len(data.elements_m) == 1024,
            "interpretation": "synthetic truth scoring only; optimizer success is not physical acceptance"}
