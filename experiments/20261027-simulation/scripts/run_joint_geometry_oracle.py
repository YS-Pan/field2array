"""Known-response numerical control only; cannot release a real-data gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
ROOT = Path(__file__).resolve().parents[1]
RUN_SOURCE_BYTES = Path(__file__).read_bytes()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import least_squares
from array_reconstruct.physics import green_matrix, planar_scan, C0_M_S
from array_reconstruct.joint_geometry import scalar_kernel, deterministic_starts_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(8, 32), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "result.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()
    geometry_file = ROOT / "metadata/reference_element_geometry.csv"
    if args.size == 32:
        import csv
        with geometry_file.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        elements = np.array([[float(r['x_mm'])-173.25, float(r['y_mm'])-170.5, 0]
                             for r in records]) * .001
    else:
        row, col = np.indices((args.size, args.size))
        x = 11*col + 5.5*(row % 2)
        y = 11*row
        elements = np.c_[(x-(x.max()+x.min())/2).ravel(),
                         (y-(y.max()+y.min())/2).ravel(), np.zeros(x.size)]*.001
    truth = np.array([62.3456, 1.2345, -3.4567])
    rng = np.random.default_rng(np.random.SeedSequence([43103, args.size, 991]))
    amp_db = rng.uniform(-2.5, 2.5, len(elements))
    amp_db = (amp_db-amp_db.min())*5/np.ptp(amp_db)-2.5
    b = 10**(amp_db/20) * np.exp(1j*np.deg2rad(rng.normal(0, 5, len(elements))))
    frequency = 12.7e9
    sessions = []
    for index, angle in enumerate((0., 30., -30.)):
        side = (134 if index < 2 else 101) if args.size == 32 else 51
        dense = planar_scan(side, side, .003, z_m=0)
        indices = np.arange(side*side).reshape(side, side)[::3, ::3].ravel()
        xy = dense[indices, :2]
        points = np.c_[xy[:, 0]+truth[1]*.001,
                       xy[:, 1]+truth[2]*.001, np.full(len(xy), truth[0]*.001)]
        steering = np.exp(-2j*np.pi*frequency/C0_M_S*elements[:, 0]*np.sin(np.deg2rad(angle)))
        gain = 1 if index == 0 else np.exp(.1*index+1j*.3*index)
        # Independent existing generator; the fitted kernel does not generate y.
        observed = gain * (green_matrix(elements, points, frequency, pattern_exponent=2).conj() @ (steering*b))
        sessions.append((xy, steering, observed, 1/(np.sqrt(3)*np.linalg.norm(observed))))
    cached_x = None
    cached_value = None
    calls = 0
    def evaluate(x):
        nonlocal cached_x, cached_value, calls
        if cached_x is not None and np.array_equal(x, cached_x):
            return cached_value
        calls += 1
        geom = np.array([60., 0., 0.])+10*x[:3]
        residuals, derivatives = [], []
        for index, (xy, u, y, w) in enumerate(sessions):
            gi = 3+2*(index-1)
            gain = 1 if index == 0 else np.exp(x[gi]+1j*x[gi+1])
            pred = gain * (scalar_kernel(elements, xy, frequency, geom) @ (u*b))
            jac = np.zeros((len(xy), 7), complex)
            for axis in range(3):
                jac[:, axis] = -w*gain*(scalar_kernel(elements, xy, frequency, geom, derivative=axis) @ (u*b))
            if index:
                jac[:, gi], jac[:, gi+1] = -w*pred, -1j*w*pred
            residuals.append(w*(y-pred))
            derivatives.append(jac)
        r, j = np.concatenate(residuals), np.vstack(derivatives)
        cached_x, cached_value = x.copy(), (np.r_[r.real, r.imag], np.vstack([j.real, j.imag]))
        return cached_value
    check_x = np.array([.2, .1, -.3, .1, .2, -.1, -.2])
    r, jac = evaluate(check_x)
    direction = np.array([.4, -.3, .2, .1, -.2, .3, .4])
    direction /= np.linalg.norm(direction)
    checks = []
    for h in (1e-3, 1e-4, 1e-5):
        delta = (evaluate(check_x+h*direction)[0]-evaluate(check_x-h*direction)[0])/(2*h)
        checks.append({"step_normalized": h,
                       "relative_directional_error": float(np.linalg.norm(delta-jac@direction)/np.linalg.norm(jac@direction))})
    records = []
    for start in deterministic_starts_mm():
        x0 = np.r_[(start-[60, 0, 0])/10, np.zeros(4)]
        result = least_squares(lambda x:evaluate(x)[0], x0, jac=lambda x:evaluate(x)[1],
                               bounds=([-1,-1,-1,-3,-np.inf,-3,-np.inf],
                                       [1,1,1,3,np.inf,3,np.inf]),
                               method='trf', max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        recovered = np.array([60.,0.,0.])+10*result.x[:3]
        records.append({"start_mm":start.tolist(), "geometry_mm":recovered.tolist(),
                        "geometry_mm_4dp":[f"{v:.4f}" for v in recovered],
                        "error_mm":(recovered-truth).tolist(), "cost":float(result.cost),
                        "success":bool(result.success), "status":int(result.status),
                        "message":result.message, "nfev":result.nfev})
    coords = np.array([r['geometry_mm'] for r in records])
    same_mode = [r for r in records if r['cost'] <= 1e-12 and r['success']]
    passed = (len(same_mode) >= 2 and np.max(np.abs(np.array([r['error_mm'] for r in same_mode]))) <= .02
              and np.max(np.ptp(np.array([r['geometry_mm'] for r in same_mode]),axis=0)) <= .01
              and min(r['relative_directional_error'] for r in checks) <= 1e-5)
    report = {"kind":"synthetic_known_response_numerical_positive_control",
              "real_release_eligible":False, "oracle":"Exact b supplied to fitting, never used as applicable-information evidence",
              "size":args.size,"columns":len(elements),"seed":[43103,args.size,991],
              "truth_geometry_mm":truth.tolist(),"amplitude_pp_db":5,"phase_std_deg":5,
              "known_response":True,"session_gains_estimated":True,"reference_gain":1,
              "train_spacing_on_original_3mm_grid_mm":9,"frequency_hz":frequency,
              "derivative_checks":checks,"all_starts":records,"same_mode_success_count":len(same_mode),
              "same_mode_gate_passed":bool(passed),"calls":calls,"elapsed_seconds":time.perf_counter()-started,
              "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(RUN_SOURCE_BYTES if p==Path(__file__).resolve() else p.read_bytes()).hexdigest()
                               for p in [Path(__file__).resolve(), geometry_file,
                                         ROOT/'src/array_reconstruct/joint_geometry.py',ROOT/'src/array_reconstruct/physics.py']}}
    with report_path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['size','same_mode_success_count','same_mode_gate_passed','elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
