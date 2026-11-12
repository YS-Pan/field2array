"""Evaluate the separately declared sparse-fault prior on synthetic archives."""
import json
from pathlib import Path

import numpy as np

from .physics import green_matrix, angular_spectrum_backprop
from .sparse import solve_sparse_faults
from .validation import evaluate_coefficients, accepts, _plot_case


def run_sparse_validation(dataset_directory, output_directory, progress=print):
    source = Path(dataset_directory); output = Path(output_directory)
    baseline = json.loads((source / "results.json").read_text(encoding="utf8"))
    if baseline.get("synthetic_only") is not True:
        raise ValueError("Only an explicitly synthetic validation archive is accepted")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for record in baseline["cases"]:
        name = record["name"]
        progress(f"Sparse diagnostic: {name}", flush=True)
        with np.load(source / (name + ".npz"), allow_pickle=False) as data:
            if str(data["provenance"].item()) != "synthetic":
                raise ValueError("Non-synthetic input blocked")
            elements, points = data["elements_m"], data["samples_m"]
            frequency = float(data["frequency_hz"])
            nominal, field = data["nominal"], data["field"]
            matrix = green_matrix(elements, points, frequency)
            validation_matrix = green_matrix(elements, data["validation_samples_m"], frequency)
            # Preparation depends on nominal steering, so never cache by f alone.
            fit = solve_sparse_faults(matrix, field, nominal,
                validation_matrix=validation_matrix, validation_y=data["validation_field"],
                max_faults=min(32, len(elements) // 4))
            # Coefficient truth and injected fault labels first enter here.
            truth = data["true_coefficients"]
            faults = np.asarray(record["metrics"]["actual_fault_indices"], dtype=int)
            metrics, aligned, flagged = evaluate_coefficients(fit.coefficients, truth, nominal, faults)
            test_matrix = green_matrix(elements, data["test_samples_m"], frequency)
            test_error = float(np.linalg.norm(test_matrix @ fit.coefficients - data["test_field"]) / np.linalg.norm(data["test_field"]))
            result = {"name": name, "method": "sparse_nominal_plus_common_gain", "seed_base": baseline["seed_base"],
                "side": record["side"], "frequency_hz": frequency, "stress_case": record["stress_case"],
                "metrics": metrics, "coefficient_acceptance_passed": accepts(metrics, baseline["policy"]),
                "baseline_tikhonov_passed": record["coefficient_acceptance_passed"],
                "baseline_tikhonov_nrmse": record["metrics"]["complex_nrmse"],
                "selected_support_count": fit.selected_fault_count, "support": np.asarray(fit.support).tolist(),
                "common_gain": [float(fit.gain.real), float(fit.gain.imag)],
                "selection_field_nrmse": fit.validation_relative_error,
                "training_field_nrmse": fit.relative_residual,
                "independent_test_noisy_field_nrmse": test_error,
                "truth_used_for_model_selection": False}
            results.append(result)
            np.savez_compressed(output / (name + ".npz"), provenance="synthetic",
                coefficients=fit.coefficients, aligned_coefficients=aligned, truth=truth,
                selected_support=np.asarray(fit.support), flagged_indices=np.flatnonzero(flagged))
            scan_shape = tuple(record["scan_shape"])
            bp = angular_spectrum_backprop(field.reshape(scan_shape), .00835, .00835, frequency, .06)
            _plot_case(output, name, record["side"], truth, aligned, field, scan_shape[0], bp["image"])
            progress(f"  pass={result['coefficient_acceptance_passed']}, NRMSE={metrics['complex_nrmse']:.4f}, precision={metrics['fault_precision']:.3f}, recall={metrics['fault_recall']:.3f}", flush=True)
        (output / "results.json").write_text(json.dumps({"cases": results, "in_progress": True}, indent=2), encoding="utf8")
    report = {"schema_version": 1, "synthetic_only": True, "baseline_archive": str(source),
        "seed_base": baseline["seed_base"], "cases": results,
        "additional_assumptions": ["Known nominal excitation", "Most elements close to nominal",
                                   "Large deviations sparse", "Common complex gain is free"],
        "nominal_scenarios_passed": all(r["coefficient_acceptance_passed"] for r in results if not r["stress_case"]),
        "universal_identifiability_claim": False, "real_data_gate_open": False}
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
