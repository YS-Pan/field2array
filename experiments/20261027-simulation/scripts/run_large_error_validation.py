"""Run the preregistered, synthetic-only large feed-error experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Set before importing NumPy/SciPy; respects an explicitly configured setting.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from array_reconstruct.large_error_validation import run_large_error_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/large_error_validation_20261028_independent_noise")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/large_error_validation_policy.json")
    args = parser.parse_args()
    run_large_error_validation(args.output, args.policy)


if __name__ == "__main__":
    main()
