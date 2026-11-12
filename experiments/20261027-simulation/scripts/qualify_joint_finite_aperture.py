"""CPU-only math/contracts by default; optional later single-device qualification.

No input datasets, truth/held files or virtual generators are loaded. The CPU
harness executes the GPU class's arithmetic and inherited QR/profile code on
torch CPU tensors solely to check dispatch. It is never a production backend
and never establishes CUDA numerical, memory, timing or 1024-column fitness.
"""
from __future__ import annotations

import os
for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_key] = "1"
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch

from array_reconstruct.joint_geometry import FrequencyGroup, JointProblem, Session, scalar_kernel
from array_reconstruct.joint_geometry_finite_gpu import ApertureQuadrature, GPUFiniteApertureJointProblem
from array_reconstruct.joint_finite_aperture_reference import CPUFiniteApertureReference, literal_finite_kernel
from array_reconstruct.joint_newton import profile_gradient_hessian, scalar_kernel_second


def relative(value, reference):
    return float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference), 1e-30))


class CPUArithmeticHarness(GPUFiniteApertureJointProblem):
    """Qualification-only CPU execution; bypasses production CUDA constructor."""
    def __init__(self, elements, groups, q=2., sign=-1, *, chunk_size=5,
                 source_aperture_mm=4., probe_aperture_mm=4.):
        # No torch.cuda call, synchronization, device query or allocation.
        JointProblem.__init__(self, elements, groups, q, sign, backend="qr")
        self.torch, self.device = torch, torch.device("cpu")
        self.cd, self.rd = torch.complex128, torch.float64
        self.backend = "CPU_emulation_of_finite_CUDA_arithmetic_NOT_CUDA_qualification"
        self.fallback = "raise"
        self._quadrature = ApertureQuadrature.gl3(source_aperture_mm, probe_aperture_mm)
        self.kernel_chunk_size = chunk_size
        self.counts.update(gpu_failures=0, cpu_fallbacks=0, gpu_qr_factorizations=0,
            gpu_hessian_calls=0, kernel_second_derivative_calls=0,
            gpu_prediction_calls=0, gpu_validation_calls=0)
        self._initialize_finite_counters()
        self._cpu_reference = self._device_state = None
        self._resident_sessions = {}
        self.gpu_elapsed_seconds = self.gpu_hessian_seconds = self.upload_setup_seconds = 0.
        self._elements = self._tensor(self.elements_m[:, :2], self.rd)
        self._eye = torch.eye(len(elements), dtype=self.cd)
        self._resident_groups = [(g, [self._session(s) for s in g.sessions],
                                  g.lambda_reg*self._eye, torch.zeros(len(elements), dtype=self.cd)) for g in groups]

    def _sync(self):
        return None

    def runtime_descriptor(self):
        return {"backend": self.backend, "device": "cpu", "cuda_used": False,
                "counts": dict(self.counts), "fallback_policy": self.fallback,
                "failure_ledger": list(self.failure_ledger)}


def fixture(columns=6, samples=11):
    # Hand-constructed local observations: no scientific truth, generation
    # registry, frozen development case or confirmation seed is consulted.
    row, col = np.divmod(np.arange(columns), max(2, int(np.ceil(np.sqrt(columns)))))
    elements = np.c_[.011*col+.0055*(row % 2), .011*row]
    elements -= elements.mean(axis=0)
    z = np.linspace(-.037, .041, samples)
    xy = np.c_[z, .031*np.sin(np.arange(samples)*1.13)]
    groups = []
    for group_index, frequency in enumerate((12.7e9, 12.1e9)):
        sessions = []
        for index in range(3 if group_index == 0 else 1):
            coord = xy if index != 2 else xy+np.array((.00031, -.00017))
            u = np.exp(1j*(index+.3)*(elements[:, 0]*130+elements[:, 1]*47))
            observation = (np.sin(np.arange(samples)*.83+.4*index)+.3
                           +1j*np.cos(np.arange(samples)*.61+group_index+.2*index))
            weight = np.linspace(.13, .21, samples)
            weight[::5] = 0.
            sessions.append(Session(f"f{group_index}s{index}", frequency, coord, u, observation, weight))
        groups.append(FrequencyGroup(f"f{group_index}", sessions, .043))
    return elements, tuple(groups)


def compare_evaluations(reference, actual):
    out = {"residual": relative(actual.residual, reference.residual),
           "jacobian": relative(actual.jacobian, reference.jacobian)}
    for field in ("b", "db", "data_residuals", "weighted_data_residuals", "penalty_residuals",
                  "data_jacobians", "weighted_data_jacobians", "penalty_jacobians"):
        out[field] = max(relative(getattr(actual, field)[name], value)
                         for name, value in getattr(reference, field).items())
    return out


def qualify(*, cuda=False, columns=6, samples=11):
    started = time.perf_counter()
    torch.set_num_threads(1)
    elements, groups = fixture(columns, samples)
    reference = CPUFiniteApertureReference(elements, groups)
    tested = (GPUFiniteApertureJointProblem(elements, groups, fallback="raise", kernel_chunk_size=5)
              if cuda else CPUArithmeticHarness(elements, groups))
    x = reference.initial_parameters((61.37, -.83, 2.19), {"f0s1": 1.17+.23j, "f0s2": .86-.19j})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "cuda_used": cuda,
        "scope": "local synthetic CPU mathematical/contracts check" if not cuda else "single-device synthetic numerical qualification; no fit or scientific gate",
        "columns": columns, "samples_per_session": samples,
        "truth_held_real_files_read": False, "confirmation_seeds_generated": False,
        "parameter_layout": reference.parameter_layout, "aperture": tested.quadrature.descriptor(),
        "cuda_1024_qualification": "not_run" if not cuda else "this receipt only",
        "amplitude_extension": "pending", "kernel_finite_differences": [], "profile_finite_differences": []}

    native = tested._session(groups[0].sessions[0])
    tx = tested._tensor(x, tested.rd)
    kernel = tested._kernel(native, tx)
    actual_K = tested._numpy(kernel["K"])
    actual_first = [tested._numpy(v) for v in kernel["first"]]
    actual_second = {(j, k): tested._numpy(tested._second(kernel, j, k)) for j in range(3) for k in range(j, 3)}
    args = (elements, native["session"].samples_xy_m, native["session"].frequency_hz, x[:3]*10+[60., 0., 0.])
    report["literal_kernel_parity"] = {"K": relative(actual_K, literal_finite_kernel(*args)),
        "first": max(relative(actual_first[j], literal_finite_kernel(*args, derivative=j)) for j in range(3)),
        "second": max(relative(actual_second[j, k], literal_finite_kernel(*args, derivative=(j, k))) for j, k in actual_second)}
    # Several steps are necessary: one lucky finite-difference step is not a
    # stability interval. Both first and second differences use the literal
    # quadrature, independent of the Torch propagation implementation.
    steps = (3e-4, 1e-4, 3e-5, 1e-5)
    for step in steps:
        first_errors, second_errors = [], []
        for j in range(3):
            plus, minus = np.array(args[-1]), np.array(args[-1])
            plus[j] += 10*step; minus[j] -= 10*step
            finite = (literal_finite_kernel(*args[:-1], plus)-literal_finite_kernel(*args[:-1], minus))/(2*step)
            first_errors.append(relative(finite, actual_first[j]))
            for k in range(3):
                finite_second = (literal_finite_kernel(*args[:-1], plus, derivative=k)-literal_finite_kernel(*args[:-1], minus, derivative=k))/(2*step)
                second_errors.append(relative(finite_second, actual_second[min(j, k), max(j, k)]))
        report["kernel_finite_differences"].append({"step_t": step, "first_max_relative": max(first_errors),
            "second_max_relative": max(second_errors), "passed": max(first_errors+second_errors) <= 1e-5})

    initial_kernel_calls = tested.counts["kernel_calls"]
    initial_derivative_calls = tested.counts["kernel_derivative_calls"]
    a, b = reference.evaluate(x), tested.evaluate(x)
    training_unique_kernel_calls = tested.counts["kernel_calls"]-initial_kernel_calls
    training_first_matrix_calls = tested.counts["kernel_derivative_calls"]-initial_derivative_calls
    report["profile_parity"] = compare_evaluations(a, b)
    ca, cb = profile_gradient_hessian(reference, x), profile_gradient_hessian(tested, x)
    report["profile_parity"].update(gradient=relative(cb["gradient"], ca["gradient"]),
        exact_hessian=relative(cb["hessian"], ca["hessian"]))
    report["exact_hessian_vs_gauss_newton_relative"] = relative(cb["hessian"], b.jacobian.T@b.jacobian)
    report["profile_hessian_diagnostics"] = cb["diagnostics"]
    p = len(x)
    for step in steps:
        residual_columns, gradient_columns, cost_gradient = [], [], []
        for j in range(p):
            delta = np.eye(p)[j]*step
            ep, em = reference.evaluate(x+delta), reference.evaluate(x-delta)
            residual_columns.append((ep.residual-em.residual)/(2*step))
            gradient_columns.append((ep.jacobian.T@ep.residual-em.jacobian.T@em.residual)/(2*step))
            cost_gradient.append((ep.cost-em.cost)/(2*step))
        row = {"step_t_or_gain": step,
            "residual_jacobian_relative": relative(np.stack(residual_columns, axis=1), a.jacobian),
            "cost_gradient_relative": relative(np.asarray(cost_gradient), ca["gradient"]),
            "exact_hessian_relative": relative(np.stack(gradient_columns, axis=1), ca["hessian"])}
        row["passed"] = max(row[k] for k in row if k not in ("step_t_or_gain", "passed")) <= 1e-5
        report["profile_finite_differences"].append(row)

    # Shifted, rectangular-coordinate validation exercises the inherited
    # prediction/validation hooks, full db and gain/layout cross terms.
    val = replace(groups[0].sessions[1], samples_xy_m=groups[0].sessions[1].samples_xy_m+[.00071, -.00029])
    validation_errors = []
    for gain_override in (None, {val.name: .91+.12j}):
        va = reference.validation_evaluate(x, {groups[0].name: [val]}, gain_override)
        vb = tested.validation_evaluate(x, {groups[0].name: [val]}, gain_override)
        validation_errors += [relative(vb[key], va[key]) for key in ("residual", "jacobian")]
        validation_errors.append(relative(vb["predictions"][val.name], va["predictions"][val.name]))
    report["validation_max_relative"] = max(validation_errors)
    report["prediction_relative"] = relative(tested.predict(val, a.geometry_mm, a.b[groups[0].name], .93+.17j),
        reference.predict(val, a.geometry_mm, a.b[groups[0].name], .93+.17j))
    stationary = max(g["stationarity"] for g in b.diagnostics["groups"].values())
    report["linear_stationarity_max"] = stationary

    # The point limit retains 81 normalized duplicate terms deliberately.
    zero = (GPUFiniteApertureJointProblem(elements, groups, fallback="raise", source_aperture_mm=0., probe_aperture_mm=0.)
            if cuda else CPUArithmeticHarness(elements, groups, source_aperture_mm=0., probe_aperture_mm=0.))
    zero_kernel = zero._kernel(zero._session(groups[0].sessions[0]), zero._tensor(x, zero.rd))
    point_K = scalar_kernel(*args)
    report["point_limit_relative"] = {"K": relative(zero._numpy(zero_kernel["K"]), point_K),
        "first": max(relative(zero._numpy(zero_kernel["first"][j]), scalar_kernel(*args, derivative=j)) for j in range(3)),
        "second": max(relative(zero._numpy(zero._second(zero_kernel, j, k)), scalar_kernel_second(*args, j=j, k=k)) for j in range(3) for k in range(j, 3))}
    # Reciprocity / axis exchange and translation tests detect swapped axes,
    # wrong signs and a mm/m conversion error independently of derivatives.
    xy = args[1]
    rectangle = literal_finite_kernel(elements, xy, args[2], args[3], source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.))
    swapped = literal_finite_kernel(elements[:, ::-1], xy[:, ::-1], args[2], np.array(args[3])[[0, 2, 1]], source_aperture_mm=(6., 2.), probe_aperture_mm=(4., 3.))
    translated = literal_finite_kernel(elements+[.0031, -.0017], xy+[.0031, -.0017], args[2], args[3])
    conjugate = literal_finite_kernel(elements, xy, args[2], args[3], sign=1)
    reciprocity = literal_finite_kernel(elements, xy, args[2], args[3], source_aperture_mm=6., probe_aperture_mm=4.)
    reversed_apertures = literal_finite_kernel(elements, xy, args[2], args[3], source_aperture_mm=4., probe_aperture_mm=6.)
    report["coordinate_contracts"] = {"axis_exchange": relative(swapped, rectangle),
        "common_translation": relative(translated, literal_finite_kernel(*args)),
        "phase_sign_conjugation": relative(conjugate, literal_finite_kernel(*args).conj()),
        "source_probe_exchange": relative(reciprocity, reversed_apertures)}
    rectangle_backend = (GPUFiniteApertureJointProblem(elements, groups, source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.))
                         if cuda else CPUArithmeticHarness(elements, groups, source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.)))
    rect_native = rectangle_backend._session(groups[0].sessions[0])
    rect_torch = rectangle_backend._kernel(rect_native, rectangle_backend._tensor(x, rectangle_backend.rd))
    report["coordinate_contracts"]["torch_rectangular_kernel"] = relative(rectangle_backend._numpy(rect_torch["K"]), rectangle)

    # C0 only is checked before CUDA initialization, even during CPU tests.
    rejects = {}
    for name, group in (("amplitude", replace(groups[0], amplitude_prior=np.ones(columns), amplitude_strength=.1)),
                        ("template", replace(groups[0], template_prior=np.ones(columns), template_strength=.1)),
                        ("nonzero_b0", replace(groups[0], b0=np.ones(columns)))):
        try:
            GPUFiniteApertureJointProblem(elements, [group])
        except NotImplementedError:
            rejects[name] = True
        else:
            rejects[name] = False
    try:
        tested.fallback = "cpu"
    except ValueError:
        rejects["mutable_cpu_fallback"] = True
    else:
        rejects["mutable_cpu_fallback"] = False
    for name, instance, attribute, replacement in (
            ("quadrature_reassignment", tested, "quadrature", ApertureQuadrature.gl3((12., 2.), (5., 9.))),
            ("CPU_source_aperture_reassignment", reference, "source_aperture_mm", (12., 2.)),
            ("CPU_probe_aperture_reassignment", reference, "probe_aperture_mm", (5., 9.))):
        try:
            setattr(instance, attribute, replacement)
        except AttributeError:
            rejects[name] = True
        else:
            rejects[name] = False
    # Inject failures separately after all numerical comparisons. The base
    # failure ledger must record and re-raise without constructing point CPU.
    def injected_failure(*args, **kwargs):
        raise RuntimeError("qualification injected finite-kernel failure")
    original = tested._kernel
    tested._kernel = injected_failure
    failed_x = x.copy(); failed_x[0] += .017
    try:
        tested.evaluate(failed_x)
    except RuntimeError as error:
        rejects["profile_failure_raises"] = "injected" in str(error)
    else:
        rejects["profile_failure_raises"] = False
    tested._kernel = original
    report["rejection_contracts"] = rejects
    report["no_cpu_fallback"] = tested.counts["cpu_fallbacks"] == 0 and tested._cpu_reference is None
    report["injected_failure_ledger_count"] = len(tested.failure_ledger)
    report["tested_runtime"] = tested.runtime_descriptor()
    report["counter_contract"] = {"term_count": len(tested.quadrature.terms) == 81,
        "normalization": abs(sum(w for _, _, w in tested.quadrature.terms)-1.) <= 1e-14,
        "positive_weights": all(w > 0 for _, _, w in tested.quadrature.terms),
        "chunk_bound": tested.counts["finite_peak_sample_chunk_rows"] <= tested.kernel_chunk_size,
        "streamed_terms": tested.counts["finite_max_active_quadrature_terms"] == 1,
        "unique_training_kernel_count3": training_unique_kernel_calls == 3,
        "unique_training_first_matrix_count9": training_first_matrix_calls == 9,
        "term_blocks_multiple_of81": tested.counts["finite_quadrature_term_blocks"] % 81 == 0,
        "complete_second_outputs": tested.counts["finite_second_output_matrices"] == tested.counts["kernel_second_derivative_calls"]}
    threshold = 1e-5
    report["passed"] = bool(max(report["literal_kernel_parity"].values()) < 1e-10
        and max(report["profile_parity"].values()) < 1e-8
        and sum(row["passed"] for row in report["kernel_finite_differences"]) >= 2
        and sum(row["passed"] for row in report["profile_finite_differences"]) >= 2
        and report["validation_max_relative"] < 1e-8 and report["prediction_relative"] < 1e-10
        and max(report["point_limit_relative"].values()) < 1e-10
        and max(report["coordinate_contracts"].values()) < 1e-10
        and report["exact_hessian_vs_gauss_newton_relative"] > 1e-5
        and stationary < 1e-8 and all(rejects.values()) and report["no_cpu_fallback"]
        and all(report["counter_contract"].values()))
    report["threshold"] = threshold
    report["elapsed_seconds"] = time.perf_counter()-started
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.columns <= 0 or args.samples < 3:
        parser.error("columns positive and samples>=3 required")
    if args.mode == "cpu" and args.columns > 16:
        parser.error("CPU-only qualification is intentionally small; use <=16 columns")
    paths = [Path(__file__), ROOT/"src/array_reconstruct/joint_geometry_finite_gpu.py",
        ROOT/"src/array_reconstruct/joint_finite_aperture_reference.py",
        ROOT/"src/array_reconstruct/joint_geometry_gpu.py", ROOT/"src/array_reconstruct/joint_geometry.py",
        ROOT/"src/array_reconstruct/joint_newton.py", ROOT/"src/array_reconstruct/joint_amplitude.py", ROOT/"src/array_reconstruct/physics.py",
        ROOT/"src/array_reconstruct/__init__.py", ROOT/"tests/test_joint_finite_aperture_backend.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = qualify(cuda=args.mode == "cuda", columns=args.columns, samples=args.samples)
    report["source_sha256"] = hashes
    report["source_unchanged_during_run"] = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path.relative_to(ROOT))] for path in paths)
    report["requested_model"] = {"model": "gpt-6.1-sol", "reasoning": "ultra", "fork_turns": "none", "actual_independently_verified": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cuda_used": report["cuda_used"],
        "elapsed_seconds": report["elapsed_seconds"], "output": str(args.output)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["passed"] and report["source_unchanged_during_run"] else 2)


if __name__ == "__main__":
    main()
