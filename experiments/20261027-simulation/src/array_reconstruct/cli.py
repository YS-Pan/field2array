"""Synthetic spatial inversion and a separate, narrowly scoped real adapter."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .physics import green_matrix
from .inverse import solve_tikhonov
from .simulation import simulate_measurement
from .validation import array_geometry, scan_geometry, excitation, run_validation, C0


def simulate(args):
    frequency = args.frequency_ghz * 1e9
    elements = array_geometry(args.side)
    samples = scan_geometry(33 if args.side <= 8 else 63)
    selection = samples[::3].copy(); selection[:, 0] += .004175
    nominal, truth, faults = excitation(elements, frequency, args.seed, faults=not args.no_faults)
    config = {"source_aperture_m": .1 * C0 / frequency, "directionality_power": 0,
              "snr_db": args.snr_db, "seed": args.seed}
    measured = simulate_measurement(elements, truth, frequency, samples, config=config)
    heldout = simulate_measurement(elements, truth, frequency, selection, config={**config, "seed": args.seed + 1})
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, provenance="synthetic", elements_m=elements, samples_m=samples,
        frequency_hz=frequency, field=measured["field"], nominal=nominal,
        true_coefficients=truth, fault_indices=faults,
        validation_samples_m=selection, validation_field=heldout["field"])
    print(f"Synthetic measurement saved: {target.resolve()}")


def reconstruct(args):
    # np.load never enables pickle. Only required observations/coordinates are
    # passed to the solver; stored truth is deliberately not accessed.
    with np.load(args.input, allow_pickle=False) as dataset:
        if "provenance" not in dataset or str(dataset["provenance"].item()) != "synthetic":
            raise ValueError("This spatial-scan entry permits synthetic measurements only; use analyze-real for authorized channel observations.")
        frequency = float(dataset["frequency_hz"])
        elements, samples, field = dataset["elements_m"], dataset["samples_m"], dataset["field"]
        nominal = dataset["nominal"] if "nominal" in dataset else None
        matrix = green_matrix(elements, samples, frequency)
        options = {}
        if "validation_samples_m" in dataset and "validation_field" in dataset:
            options = {"validation_matrix": green_matrix(elements, dataset["validation_samples_m"], frequency),
                       "validation_y": dataset["validation_field"]}
        method = getattr(args, "method", "ridge")
        if method == "sparse":
            from .sparse import solve_sparse_faults
            if not options or "nominal" not in dataset:
                raise ValueError("Sparse diagnosis requires a known nominal excitation and independent validation samples")
            fit = solve_sparse_faults(matrix, field, dataset["nominal"], **options,
                                     max_faults=getattr(args, "max_faults", None))
        else:
            fit = solve_tikhonov(matrix, field, **options)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with (output / "coefficients.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["element_index", "x_m", "y_m", "z_m", "real", "imag", "amplitude", "phase_deg"])
        for i, (position, value) in enumerate(zip(elements, fit.coefficients)):
            phase = float(np.rad2deg(np.angle(value))) if abs(value) > 0 and abs(value) >= .1 * np.max(np.abs(fit.coefficients)) else ""
            writer.writerow([i, *position, value.real, value.imag, abs(value), phase])
    if method == "sparse":
        from scipy.linalg import svdvals
        scale = np.linalg.norm(matrix, axis=0)
        s = svdvals(matrix / scale)
        rank = int(np.sum(s > max(matrix.shape) * np.finfo(float).eps * s[0]))
    else:
        s = fit.singular_values
        rank = fit.numerical_rank
    result = {"provenance": "synthetic", "quantity": "equivalent_radiation_coefficients",
              "truth_used": False, "method": method,
              "alpha": fit.alpha if method == "ridge" else None,
              "selection_method": fit.selection_method if method == "ridge" else "independent_validation_support_size",
              "relative_field_residual": fit.relative_residual,
              "validation_relative_error": fit.validation_relative_error,
              "column_scaled_condition_number": float(s[0] / s[-1]) if rank == len(elements) else None,
              "numerical_rank": rank, "data_nullity": len(elements) - rank,
              "rank_deficient": rank < len(elements),
              "effective_rank": fit.effective_rank if method == "ridge" else None, "elements": len(elements),
              "absolute_port_amplitude_phase_calibrated": False,
              "warning": "A small field residual does not prove unique or accurate element weights; consult validation and ambiguity report."}
    if method == "sparse":
        result.update(selected_correction_count=fit.selected_fault_count, support=np.asarray(fit.support).tolist(),
                      assumptions=["Known nominal excitation", "Most elements close to nominal", "Sparse large deviations"])
    if nominal is not None and np.any(np.abs(nominal) > 0):
        from .diagnostics import detect_anomalies
        if np.all(np.abs(nominal) > 0):
            ratio = fit.coefficients / nominal
            scale = np.median(np.abs(ratio))
            phase = np.angle(np.sum(ratio / np.maximum(np.abs(ratio), 1e-15)))
            calibrated = fit.coefficients / max(scale, 1e-15) * np.exp(-1j * phase)
            diagnostic = detect_anomalies(calibrated, nominal)
        else:
            diagnostic = detect_anomalies(fit.coefficients, nominal, alignment="complex")
        result["flagged_indices"] = np.flatnonzero(diagnostic["fault_mask"]).tolist()
        result["diagnostic_reference"] = "Known nominal; one common complex gain removed; majority-normal assumption"
        with (output / "anomalies.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["element_index", "amplitude_deviation_db", "phase_deviation_deg", "flagged"])
            for i in range(len(elements)):
                amp, phase = diagnostic["amplitude_deviation_db"][i], diagnostic["phase_deviation_deg"][i]
                writer.writerow([i, amp if np.isfinite(amp) else "", phase if np.isfinite(phase) else "", bool(diagnostic["fault_mask"][i])])
    (output / "reconstruction.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    print(f"Results: {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulation-first near-field array reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate", help="Run synthetic comparisons and stress tests")
    validation.add_argument("--output", default="outputs/validation")
    validation.add_argument("--policy", default="configs/validation_policy.json")
    validation.add_argument("--full", action="store_true", help="Include the 1024-element layout")
    validation.add_argument("--seed", type=int, default=20261027)
    validation.add_argument("--random-faults", action="store_true", help="Use seeded distinct fault locations for each case")
    sparse = commands.add_parser("validate-sparse", help="Evaluate the explicit sparse-fault prior on a synthetic validation archive")
    sparse.add_argument("--dataset-directory", required=True)
    sparse.add_argument("--output", default="outputs/sparse_validation")
    sim = commands.add_parser("simulate", help="Export coherent synthetic measurements to NPZ")
    sim.add_argument("--side", type=int, choices=range(2, 33), default=8)
    sim.add_argument("--frequency-ghz", type=float, default=11.7)
    sim.add_argument("--snr-db", type=float, default=40)
    sim.add_argument("--seed", type=int, default=20261027)
    sim.add_argument("--no-faults", action="store_true")
    sim.add_argument("--output", default="outputs/virtual_measurement.npz")
    rec = commands.add_parser("reconstruct", help="Reconstruct an exported synthetic NPZ without reading its truth")
    rec.add_argument("--input", required=True)
    rec.add_argument("--output", default="outputs/reconstruction")
    rec.add_argument("--method", choices=("ridge", "sparse"), default="ridge")
    rec.add_argument("--max-faults", type=int, default=None, help="Maximum correction support (default min(32,N/4)); not a forced fault count")
    real = commands.add_parser("analyze-real", help="Analyze the authorized H/V 0/180 channel-response CSV runs")
    real.add_argument("--archive", required=True)
    real.add_argument("--mapping", default="metadata/chamber_element_mapping.csv")
    real.add_argument("--policy", default="configs/real_measurement_policy.json")
    real.add_argument("--output", default="outputs/real_measurement")
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = run_validation(args.output, args.policy, full=args.full, seed_base=args.seed, random_faults=args.random_faults)
        print(f"Nominal scenarios passed: {report['nominal_scenarios_passed']}; real-data gate remains closed")
        return 0 if report["nominal_scenarios_passed"] else 2
    if args.command == "validate-sparse":
        from .sparse_validation import run_sparse_validation
        report = run_sparse_validation(args.dataset_directory, args.output)
        print(f"Sparse-prior nominal scenarios passed: {report['nominal_scenarios_passed']}; real-data gate remains closed")
        return 0 if report["nominal_scenarios_passed"] else 2
    if args.command == "simulate":
        simulate(args)
    elif args.command == "analyze-real":
        from .real_measurement import analyze_real_measurement
        result = analyze_real_measurement(args.archive, args.mapping, args.output, args.policy)
        print(json.dumps({"provenance": result["provenance"], "main_members": result["main_members"],
                          "repeat_members": result["repeat_members"], "observations": result["observations"]}, indent=2))
    else:
        reconstruct(args)
    return 0
