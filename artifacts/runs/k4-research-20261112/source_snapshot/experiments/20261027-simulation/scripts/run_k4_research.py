"""Run one bounded, policy-driven K4 virtual development or confirmation matrix."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from array_reconstruct.k4_research import run_k4_research


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--policy", type=Path, required=True, help="Self-contained frozen JSON policy")
    parser.add_argument("--operators", type=Path, action="append", default=[], help="Synthetic operator NPZ, repeated for multiple frequencies")
    parser.add_argument("--generate-operators", action="store_true", help="Generate missing operators under the explicit output directory")
    args = parser.parse_args()
    report = run_k4_research(args.output, args.policy, args.operators, generate_operators=args.generate_operators)
    raise SystemExit(1 if report["failed_case_count"] or report["timed_out_case_count"] else 0)
