"""Run the locked synthetic paper-inspired physical-cell ASM experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from array_reconstruct.thesis_aperture import run_thesis_aperture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmatory"), required=True)
    args = parser.parse_args()
    run_thesis_aperture(args.output, args.policy, args.split)


if __name__ == "__main__":
    main()
