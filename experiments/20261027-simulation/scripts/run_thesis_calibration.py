"""Run the explicitly synthetic table-5-2-inspired calibration experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from array_reconstruct.thesis_calibration import run_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/thesis_methods_20261028/calibration")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/thesis_methods_policy.json")
    parser.add_argument("--operators", type=Path, default=ROOT / "outputs/large_error_validation_20261028_independent_noise/operators_11.7GHz.npz")
    args = parser.parse_args()
    run_calibration(args.output, args.policy, args.operators)


if __name__ == "__main__":
    main()
