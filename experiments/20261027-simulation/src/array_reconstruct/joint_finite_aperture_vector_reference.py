"""Chunked NumPy GL3 reference for the fixed C0 aperture hypothesis.

No CUDA, stress generator, data loader or file access occurs in this module.
The literal reference's profile/QR layer is reused transparently; its every
kernel access dispatches through the overridden hook here, including raw
zero-weight Jacobians, exact profile Hessians, prediction and validation.
The physical model is the same normalized scalar GL3 hypothesis, not new
independent physical evidence. The pointwise chain rule uses R_j and R_jk,
independently of the GPU beta/gamma implementation.
"""
from __future__ import annotations

import numpy as np

from .joint_geometry import _geometry, _real_array
from .joint_finite_aperture_reference import CPUFiniteApertureReference, _sides


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _axes(derivative):
    if derivative is None:
        return ()
    if isinstance(derivative, (int, np.integer)) and not isinstance(derivative, (bool, np.bool_)):
        value = (derivative,)
    elif isinstance(derivative, tuple) and len(derivative) == 2:
        value = derivative
    else:
        raise ValueError("derivative must be None, an integer axis, or a second-axis tuple")
    if any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or j not in (0, 1, 2) for j in value):
        raise ValueError("derivative axes must be integer 0, 1 or 2")
    return tuple(int(j) for j in value)


def vector_finite_kernel(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                         q=2., sign=-1, *, source_aperture_mm=4.,
                         probe_aperture_mm=4., derivative=None,
                         sample_chunk_size=128, element_chunk_size=256,
                         counters=None):
    """Return K, K_j or K_jk with bounded sample/element temporary blocks.

    Only the quadrature axes have an 81-term loop; sample/element pairs use
    NumPy broadcasting within each block. Coordinates are metres, aperture
    full sides are mm, and one normalized t unit is 0.01 m. Probe minus
    source shifts are added to scan+mechanical-element. Unit spatial
    averaging supplies no extra collecting area or hardware calibration.
    """
    elements = _real_array(elements_m, "elements_m")
    samples = _real_array(samples_xy_m, "samples_xy_m")
    if (elements.ndim != 2 or not len(elements) or elements.shape[1] not in (2, 3)
            or (elements.shape[1] == 3 and np.any(elements[:, 2] != 0))):
        raise ValueError("elements_m must be nonempty planar (n,2) or (n,3)")
    if samples.ndim != 2 or not len(samples) or samples.shape[1] != 2:
        raise ValueError("samples_xy_m must be nonempty (m,2)")
    frequency, exponent = float(frequency_hz), float(q)
    if (not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(exponent)
            or exponent < 0 or sign not in (-1, 1)):
        raise ValueError("frequency positive, q nonnegative and sign +/-1 required")
    axes = _axes(derivative)
    rows = _positive_integer(sample_chunk_size, "sample_chunk_size")
    cols = _positive_integer(element_chunk_size, "element_chunk_size")
    source = np.array(_sides(source_aperture_mm, "source_aperture_mm"))*.0005
    probe = np.array(_sides(probe_aperture_mm, "probe_aperture_mm"))*.0005
    nodes = np.array((-np.sqrt(3./5.), 0., np.sqrt(3./5.)))
    weights = np.array((5./18., 4./9., 5./18.))
    source_points = tuple((source[0]*nodes[i], source[1]*nodes[j], weights[i]*weights[j])
                          for i in range(3) for j in range(3))
    probe_points = tuple((probe[0]*nodes[i], probe[1]*nodes[j], weights[i]*weights[j])
                         for i in range(3) for j in range(3))
    terms = tuple((px-sx, py-sy, sw*pw) for sx, sy, sw in source_points for px, py, pw in probe_points)
    d, dx, dy = _geometry(geometry_mm)*.001
    wave = sign*2j*np.pi*frequency/299792458.
    answer = np.empty((len(samples), len(elements)), np.complex128)
    for start in range(0, len(samples), rows):
        stop = min(start+rows, len(samples))
        for first in range(0, len(elements), cols):
            last = min(first+cols, len(elements))
            center_x = samples[start:stop, 0, None]+dx-elements[None, first:last, 0]
            center_y = samples[start:stop, 1, None]+dy-elements[None, first:last, 1]
            block = np.zeros((stop-start, last-first), np.complex128)
            for shift_x, shift_y, weight in terms:
                xx, yy = center_x+shift_x, center_y+shift_y
                radius = np.sqrt(d*d+xx*xx+yy*yy)
                value = (d/radius)**exponent/radius*np.exp(wave*radius)
                if axes:
                    j = axes[0]
                    vj = d if j == 0 else xx if j == 1 else yy
                    radius_j = vj/radius
                    radial_log = wave-(exponent+1)/radius
                    log_j = radial_log*radius_j+(exponent/d if j == 0 else 0.)
                    if len(axes) == 1:
                        value *= .01*log_j
                    else:
                        k = axes[1]
                        vk = d if k == 0 else xx if k == 1 else yy
                        radius_k = vk/radius
                        log_k = radial_log*radius_k+(exponent/d if k == 0 else 0.)
                        radius_jk = (float(j == k)-radius_j*radius_k)/radius
                        log_jk = ((exponent+1)*radius_j*radius_k/radius**2
                                  + radial_log*radius_jk
                                  - (exponent/d**2 if j == k == 0 else 0.))
                        value *= .0001*(log_j*log_k+log_jk)
                block += weight*value
            answer[start:stop, first:last] = block
            if counters is not None:
                counters["vector_blocks"] = counters.get("vector_blocks", 0)+1
                counters["vector_quadrature_term_blocks"] = counters.get("vector_quadrature_term_blocks", 0)+81
                counters["vector_point_term_evaluations"] = counters.get("vector_point_term_evaluations", 0)+81*(stop-start)*(last-first)
                counters["vector_peak_block_rows"] = max(counters.get("vector_peak_block_rows", 0), stop-start)
                counters["vector_peak_block_columns"] = max(counters.get("vector_peak_block_columns", 0), last-first)
    return answer


class VectorFiniteApertureReference(CPUFiniteApertureReference):
    """Only-kernel replacement plus diagnostic labels; old profile unchanged.

    The inherited dense QR/profile layer keeps p*(m+n)*n derivative matrices
    in its state. Vectorization bounds propagation temporaries, not those
    profile allocations. This is a CPU numerical reference, not a new
    production optimizer or a separate physical model.
    """
    @property
    def q(self):
        return self._finite_q

    @q.setter
    def q(self, value):
        # JointProblem assigns this exactly once during construction. Later
        # mutation would invalidate its same-x profile cache just as changing
        # the aperture would. A different kernel requires a new instance.
        if hasattr(self, "_finite_q"):
            raise AttributeError("q is fixed; construct a new vector reference")
        self._finite_q = float(value)

    @property
    def sign(self):
        return self._finite_sign

    @sign.setter
    def sign(self, value):
        if hasattr(self, "_finite_sign"):
            raise AttributeError("sign is fixed; construct a new vector reference")
        self._finite_sign = int(value)

    def __init__(self, elements_m, groups, q=2., sign=-1, *,
                 source_aperture_mm=4., probe_aperture_mm=4.,
                 sample_chunk_size=128, element_chunk_size=256, **options):
        self._sample_chunk_size = _positive_integer(sample_chunk_size, "sample_chunk_size")
        self._element_chunk_size = _positive_integer(element_chunk_size, "element_chunk_size")
        super().__init__(elements_m, groups, q, sign, source_aperture_mm=source_aperture_mm,
                         probe_aperture_mm=probe_aperture_mm, **options)
        self.counts.update(vector_kernel_constructions=0, vector_blocks=0,
            vector_quadrature_term_blocks=0, vector_point_term_evaluations=0,
            vector_peak_block_rows=0, vector_peak_block_columns=0)

    @property
    def sample_chunk_size(self):
        return self._sample_chunk_size

    @property
    def element_chunk_size(self):
        return self._element_chunk_size

    def _literal(self, session, geometry, derivative=None):
        # Existing profile hook name retained deliberately. Its implementation
        # is entirely the new vectorized kernel; the literal function is never
        # called by this subclass, including on failure.
        key = ("kernel_calls" if derivative is None else "kernel_second_derivative_calls"
               if isinstance(derivative, tuple) else "kernel_derivative_calls")
        self.counts[key] += 1
        self.counts["vector_kernel_constructions"] += 1
        return vector_finite_kernel(self.elements_m, session.samples_xy_m,
            session.frequency_hz, geometry, self.q, self.sign,
            source_aperture_mm=self.source_aperture_mm, probe_aperture_mm=self.probe_aperture_mm,
            derivative=derivative, sample_chunk_size=self.sample_chunk_size,
            element_chunk_size=self.element_chunk_size, counters=self.counts)

    def _evaluate(self, x, geometry, gains):
        result = super()._evaluate(x, geometry, gains)
        for diagnostic in result.diagnostics["groups"].values():
            diagnostic.update(backend_used="cpu_vector_gl3_qr_reference",
                profile_layer="unchanged CPUFiniteApertureReference dense QR/implicit derivatives")
        result.diagnostics["vector_reference"] = self.runtime_descriptor()
        return result

    def profile_gradient_hessian(self, x):
        result = super().profile_gradient_hessian(x)
        result["diagnostics"].update(backend_used="cpu_vector_gl3_qr_reference",
            kernel="chunked NumPy R_j/R_jk chain rule; no literal/GPU/stress call",
            profile_layer="unchanged CPUFiniteApertureReference exact envelope formula")
        return result

    def runtime_descriptor(self):
        return {"backend": "cpu_vector_gl3_qr_reference", "device": "cpu", "CUDA_used": False,
            "dtype": "complex128/float64", "source_full_sides_mm": list(self.source_aperture_mm),
            "probe_full_sides_mm": list(self.probe_aperture_mm),
            "sample_chunk_size": self.sample_chunk_size, "element_chunk_size": self.element_chunk_size,
            "readonly_kernel_parameters": ["q", "sign", "source_aperture_mm", "probe_aperture_mm"],
            "aperture_configuration_readonly": True, "C0_only": True,
            "profile_memory_scope": "inherited dense p*(m+n)*n derivative matrices; kernel chunking does not remove them",
            "counts": dict(self.counts)}
