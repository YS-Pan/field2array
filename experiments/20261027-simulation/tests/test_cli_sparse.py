"""Exercise the public sparse CLI using fabricated observations only."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.validation import C0, array_geometry, scan_geometry


@pytest.mark.parametrize("side", [8, 2])
def test_sparse_cli_default_support_cap_and_truth_independent_exports(tmp_path, side):
    """Default and small arrays must work without an explicit --max-faults."""
    elements = array_geometry(side)
    count = len(elements)
    frequency = 11.7e9
    nominal = np.linspace(0.8, 1.2, count) * np.exp(1j * np.linspace(-0.4, 0.6, count))
    expected = (0.73 * np.exp(0.41j)) * nominal
    expected[-1] *= 0.35
    samples = scan_geometry(17 if side == 8 else 7)
    validation = scan_geometry(13 if side == 8 else 5, z=0.067)
    validation[:, 0] += 0.003

    def fabricated_field(points):
        distance = np.sqrt(np.sum((points[:, None] - elements[None]) ** 2, axis=2))
        return (np.exp(2j * np.pi * frequency / C0 * distance) / distance) @ expected

    payload = dict(
        provenance="synthetic", elements_m=elements, samples_m=samples,
        frequency_hz=frequency, nominal=nominal, field=fabricated_field(samples),
        validation_samples_m=validation, validation_field=fabricated_field(validation),
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    exports = []
    for index, stored_truth in enumerate((
        np.full(count, 999j),
        np.array([{"forbidden": "stored truth"}], dtype=object),
    )):
        archive, output = tmp_path / f"input_{index}.npz", tmp_path / f"result_{index}"
        # The second truth/fault payload cannot be read with allow_pickle=False.
        # It must be irrelevant to both support selection and anomaly export.
        np.savez(archive, **payload, true_coefficients=stored_truth,
                 fault_indices=np.array([{"forbidden": "fault labels"}], dtype=object))
        completed = subprocess.run(
            [sys.executable, "-m", "array_reconstruct", "reconstruct", "--input", str(archive),
             "--method", "sparse", "--output", str(output)],
            cwd=root, env=env, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        exports.append({name: (output / name).read_bytes() for name in (
            "coefficients.csv", "anomalies.csv", "reconstruction.json",
        )})

    assert exports[0] == exports[1]
    with (tmp_path / "result_1" / "coefficients.csv").open(newline="", encoding="utf8") as stream:
        coefficients = list(csv.DictReader(stream))
    recovered = np.array([complex(float(row["real"]), float(row["imag"])) for row in coefficients])
    assert_allclose(recovered, expected, rtol=1e-9, atol=1e-9)
    with (tmp_path / "result_1" / "anomalies.csv").open(newline="", encoding="utf8") as stream:
        anomalies = list(csv.DictReader(stream))
    assert [int(row["element_index"]) for row in anomalies if row["flagged"] == "True"] == [count - 1]
    assert_allclose(float(anomalies[-1]["amplitude_deviation_db"]), 20 * np.log10(0.35), atol=1e-9)
    report = json.loads(exports[1]["reconstruction.json"])
    assert report["elements"] == count
    assert report["truth_used"] is False
    assert report["selection_method"] == "independent_validation_support_size"
    assert report["flagged_indices"] == [count - 1]
    assert report["selected_correction_count"] == 1
    assert "Known nominal" in report["diagnostic_reference"]
