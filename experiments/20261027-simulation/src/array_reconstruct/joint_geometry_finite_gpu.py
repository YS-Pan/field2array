"""Optional C0 finite source/probe CUDA backend, not yet CUDA-qualified.

Each aperture is a normalized uniform, axis-aligned square (rectangles may
be supplied for coordinate contracts). Full sides are mm, physical inputs
are metres, and t=((d_mm-60)/10, dx_mm/10, dy_mm/10). The 4/4 mm GL3
hypothesis has 81 explicit source/probe 2-D pairs. No hardware dimensions
are asserted. Only one quadrature term and one sample chunk are temporary;
81 full kernel matrices are never simultaneously resident.

The frozen GPU base's complete QR, implicit derivative and exact profile
Hessian machinery is reused. Both kernel hooks are overridden. Prediction
and validation inherit the same dynamic hooks. Point-kernel CPU fallback
is forbidden. Nonlinear amplitude-prior extension remains pending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint_finite_aperture_reference import _sides, require_c0
from .joint_geometry_gpu import GPUJointProblem
from .physics import C0_M_S


@dataclass(frozen=True)
class ApertureQuadrature:
    source_aperture_mm: tuple[float, float]
    probe_aperture_mm: tuple[float, float]
    terms: tuple[tuple[float, float, float], ...]

    @classmethod
    def gl3(cls, source_aperture_mm=4., probe_aperture_mm=4.):
        source = _sides(source_aperture_mm, "source_aperture_mm")
        probe = _sides(probe_aperture_mm, "probe_aperture_mm")
        nodes, weights = np.polynomial.legendre.leggauss(3)
        source_points = [(source[0]*sx*.0005, source[1]*sy*.0005, wx*wy/4)
                         for sx, wx in zip(nodes, weights) for sy, wy in zip(nodes, weights)]
        probe_points = [(probe[0]*px*.0005, probe[1]*py*.0005, wx*wy/4)
                        for px, wx in zip(nodes, weights) for py, wy in zip(nodes, weights)]
        terms = tuple((px-sx, py-sy, float(sw*pw))
                      for sx, sy, sw in source_points for px, py, pw in probe_points)
        return cls(source, probe, terms)

    def descriptor(self):
        return {"source_full_sides_mm": list(self.source_aperture_mm),
                "probe_full_sides_mm": list(self.probe_aperture_mm),
                "quadrature_order_per_aperture_axis": 3,
                "explicit_quadrature_terms": len(self.terms),
                "weight_sum": float(sum(w for _, _, w in self.terms)),
                "normalization": "unit spatial average for each aperture; no area factor",
                "relative_displacement_m": "probe_node-source_node; added to scan+mechanical-element",
                "axes": "columns +x, rows +y; parallel planar apertures; no rotation/tilt",
                "hypothesis_not_hardware_fact": True}


class GPUFiniteApertureJointProblem(GPUJointProblem):
    """Strict CUDA-only C0 backend with complete first/second kernel hooks.

    Aperture configuration is read-only because the inherited profile and
    Hessian caches key on x. Construct a new problem for another aperture.
    """
    @property
    def quadrature(self):
        return self._quadrature

    @property
    def fallback(self):
        return "raise"

    @fallback.setter
    def fallback(self, value):
        if value != "raise":
            raise ValueError("finite aperture requires fallback='raise'; point-kernel CPU fallback is forbidden")

    def __init__(self, elements_m, groups, q=2., sign=-1, *, device="cuda:0",
                 fallback="raise", source_aperture_mm=4., probe_aperture_mm=4.,
                 kernel_chunk_size=256, **options):
        self.fallback = fallback
        groups = tuple(groups)
        require_c0(groups)
        if isinstance(kernel_chunk_size, bool) or int(kernel_chunk_size) != kernel_chunk_size or kernel_chunk_size <= 0:
            raise ValueError("kernel_chunk_size must be a positive integer")
        self._quadrature = ApertureQuadrature.gl3(source_aperture_mm, probe_aperture_mm)
        self.kernel_chunk_size = int(kernel_chunk_size)
        super().__init__(elements_m, groups, q, sign, device=device, fallback=fallback, **options)
        self.backend = "cuda_qr_complex128_finite_aperture_gl3_C0"
        self._initialize_finite_counters()

    def _initialize_finite_counters(self):
        self.counts.update(finite_kernel_constructions=0, finite_second_constructions=0,
            finite_quadrature_term_blocks=0, finite_first_output_matrices=0,
            finite_second_output_matrices=0, finite_max_active_quadrature_terms=0,
            finite_peak_sample_chunk_rows=0)

    @classmethod
    def from_problem(cls, problem, *, device="cuda:0", fallback="raise", **aperture_options):
        if hasattr(problem, "quadrature"):
            aperture_options.setdefault("source_aperture_mm", problem.quadrature.source_aperture_mm)
            aperture_options.setdefault("probe_aperture_mm", problem.quadrature.probe_aperture_mm)
        elif hasattr(problem, "source_aperture_mm"):
            aperture_options.setdefault("source_aperture_mm", problem.source_aperture_mm)
            aperture_options.setdefault("probe_aperture_mm", problem.probe_aperture_mm)
        return cls(problem.elements_m, problem.groups, problem.q, problem.sign,
            device=device, fallback=fallback, normal_condition_max=problem.normal_condition_max,
            normal_qr_check=problem.normal_qr_check, qr_coefficient_rtol=problem.qr_coefficient_rtol,
            qr_residual_rtol=problem.qr_residual_rtol, **aperture_options)

    def runtime_descriptor(self):
        out = super().runtime_descriptor()
        out.update(aperture=self.quadrature.descriptor(),
            finite_hypothesis_mutability="aperture configuration read-only; construct a new problem to change it",
            kernel_chunk_size=self.kernel_chunk_size, finite_backend_scope="C0 only; amplitude extension pending",
            finite_cuda_qualification="must be established by a separate queued CUDA qualification receipt",
            finite_memory_contract="4 complete K/first outputs per unique grid; up to 6 cached second outputs; quadrature intermediates one term x <=chunk rows x n columns",
            finite_counter_scope="term_blocks counts actual quadrature iterations per sample chunk, including predict/validation and separately computed second outputs; first/second output counters count matrices; finite_max_active_quadrature_terms is a structural loop bound, not live tensor count or measured memory")
        return out

    def _accumulate(self, native, tx, second_axes=None):
        """Stream every GL3 term; never form a term x sample x element array."""
        torch = self.torch
        d = (60.+10.*tx[0])*.001
        dx, dy = tx[1]*.01, tx[2]*.01
        m, n = len(native["session"].samples_xy_m), len(self.elements_m)
        out = torch.zeros((m, n), device=self.device, dtype=self.cd)
        first = None if second_axes is not None else tuple(torch.zeros_like(out) for _ in range(3))
        wave = self.sign*2j*np.pi*native["session"].frequency_hz/C0_M_S
        for start in range(0, m, self.kernel_chunk_size):
            stop = min(start+self.kernel_chunk_size, m)
            center_x = native["xy"][start:stop, None, 0]+dx-self._elements[None, :, 0]
            center_y = native["xy"][start:stop, None, 1]+dy-self._elements[None, :, 1]
            self.counts["finite_peak_sample_chunk_rows"] = max(self.counts["finite_peak_sample_chunk_rows"], stop-start)
            for shift_x, shift_y, weight in self.quadrature.terms:
                xx, yy = center_x+shift_x, center_y+shift_y
                radius = torch.sqrt(xx*xx+yy*yy+d*d)
                point = weight*d**self.q*torch.exp(wave*radius)/radius**(self.q+1)
                beta = wave/radius-(self.q+1)/radius**2
                v = (d, xx, yy)
                logs = (beta*d+self.q/d, beta*xx, beta*yy)
                if second_axes is None:
                    out[start:stop] += point
                    for j in range(3):
                        first[j][start:stop] += point*.01*logs[j]
                else:
                    j, k = second_axes
                    gamma = -wave/radius**3+2*(self.q+1)/radius**4
                    log_second = gamma*v[j]*v[k]
                    if j == k:
                        log_second = log_second+beta
                    if j == k == 0:
                        log_second = log_second-self.q/d**2
                    out[start:stop] += point*1e-4*(logs[j]*logs[k]+log_second)
                self.counts["finite_quadrature_term_blocks"] += 1
                self.counts["finite_max_active_quadrature_terms"] = 1
        return out, first

    def _kernel(self, native, tx):
        K, first = self._accumulate(native, tx)
        self.counts["finite_kernel_constructions"] += 1
        self.counts["finite_first_output_matrices"] += 3
        # Point-kernel radius/log arrays are deliberately absent. A base-class
        # _second call would fail, rather than silently differentiate mean K.
        return {"K": K, "first": first, "second": {}, "finite_native": native,
                "finite_tx": tx[:3].clone(), "finite_quadrature": self.quadrature}

    def _second(self, kernel, j, k):
        if (isinstance(j, bool) or isinstance(k, bool) or not isinstance(j, (int, np.integer))
                or not isinstance(k, (int, np.integer)) or j not in (0, 1, 2) or k not in (0, 1, 2)):
            raise ValueError("second derivative axes must be 0, 1 or 2")
        if kernel.get("finite_quadrature") is not self.quadrature:
            raise ValueError("second derivative requires this backend's finite kernel")
        key = min(j, k), max(j, k)
        if key not in kernel["second"]:
            kernel["second"][key] = self._accumulate(kernel["finite_native"], kernel["finite_tx"], key)[0]
            self.counts["kernel_second_derivative_calls"] += 1
            self.counts["finite_second_constructions"] += 1
            self.counts["finite_second_output_matrices"] += 1
        return kernel["second"][key]


def make_finite_gpu_problem(data, lambda_reg, *, model="M2", branch="C0",
                            device="cuda:0", fallback="raise",
                            source_aperture_mm=4., probe_aperture_mm=4.,
                            kernel_chunk_size=256):
    """Public fit-data adapter; no truth or held-field inspection occurs here."""
    if branch != "C0":
        raise NotImplementedError("finite aperture adapter currently supports C0 only")
    from .joint_virtual import make_problem
    cpu_layout = make_problem(data, lambda_reg, model=model, branch="C0", backend="qr")
    return GPUFiniteApertureJointProblem.from_problem(cpu_layout, device=device, fallback=fallback,
        source_aperture_mm=source_aperture_mm, probe_aperture_mm=probe_aperture_mm,
        kernel_chunk_size=kernel_chunk_size)
