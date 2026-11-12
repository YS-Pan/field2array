"""Continuous shared geometry with exact, regularized complex variable projection.

This module has no file access. All coordinates passed to the scalar kernel are
metres; the public geometry is (d, dx, dy) in millimetres. The first three outer
variables are ((d-60)/10, dx/10, dy/10). No geometry rounding or clipping occurs.
Each frequency/measurement-chain group owns one complex response vector and one
fixed physical ridge penalty. Its first session fixes the complex gauge g=1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Mapping, Sequence

import numpy as np
from scipy import linalg, optimize
from scipy.linalg.lapack import get_lapack_funcs

from .physics import C0_M_S
from .joint_amplitude import implicit_derivative, solve_amplitude


BOUNDS_MM = np.array(((50., 70.), (-10., 10.), (-10., 10.)))


def _real_array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    out = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def _complex_array(value, name, shape=None):
    out = np.array(value, dtype=complex, copy=True)
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def _readonly(value):
    value.flags.writeable = False
    return value


def _geometry(value):
    out = _real_array(value, "geometry_mm", (3,))
    if np.any(out < BOUNDS_MM[:, 0]) or np.any(out > BOUNDS_MM[:, 1]):
        raise ValueError("geometry outside fixed 50-70 mm / +/-10 mm bounds")
    return out


def geometry_to_parameters(geometry_mm):
    return (_geometry(geometry_mm) - np.array((60., 0., 0.))) / 10.


def parameters_to_geometry(t):
    return _geometry(_real_array(t, "normalized geometry", (3,)) * 10.
                     + np.array((60., 0., 0.)))


def scalar_kernel(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                  q=2., sign=-1, derivative=None, chunk_size=512):
    """Scalar d**q exp(sign*i*k*R)/R**(q+1), or one t derivative.

    Positive offsets move the scanning plane in +x/+y. ``derivative`` 0/1/2
    means d/dt_d, d/dt_dx, d/dt_dy, respectively (one t unit is 0.01 m).
    The elements must be planar at z=0, with shape (n,2) or (n,3).
    """
    elements = _real_array(elements_m, "elements_m")
    samples = _real_array(samples_xy_m, "samples_xy_m")
    if (elements.ndim != 2 or elements.shape[0] == 0
            or elements.shape[1] not in (2, 3)):
        raise ValueError("elements_m must have shape (n,2) or (n,3)")
    if elements.shape[1] == 3 and np.any(elements[:, 2] != 0):
        raise ValueError("the registered scalar model requires elements at z=0")
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
        raise ValueError("samples_xy_m must have nonempty shape (m,2)")
    frequency = float(frequency_hz)
    exponent = float(q)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency_hz must be positive and finite")
    if not np.isfinite(exponent) or exponent < 0 or sign not in (-1, 1):
        raise ValueError("q must be nonnegative and sign must be +/-1")
    if derivative not in (None, 0, 1, 2):
        raise ValueError("derivative must be None, 0, 1, or 2")
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    d, dx, dy = _geometry(geometry_mm) * 1e-3
    k = 2 * np.pi * frequency / C0_M_S
    out = np.empty((len(samples), len(elements)), dtype=complex)
    for start in range(0, len(samples), int(chunk_size)):
        stop = min(start + int(chunk_size), len(samples))
        x = samples[start:stop, None, 0] + dx - elements[None, :, 0]
        y = samples[start:stop, None, 1] + dy - elements[None, :, 1]
        radius = np.sqrt(x*x + y*y + d*d)
        block = d**exponent * np.exp(sign * 1j * k * radius) / radius**(exponent+1)
        if derivative is not None:
            factor = sign * 1j * k - (exponent+1) / radius
            if derivative == 0:
                factor = exponent/d + factor*d/radius
            else:
                factor = factor * (x if derivative == 1 else y) / radius
            block *= 0.01 * factor
        out[start:stop] = block
    return out


@dataclass(frozen=True)
class Session:
    name: str
    frequency_hz: float
    samples_xy_m: np.ndarray
    steering: np.ndarray
    field: np.ndarray
    weight: float | np.ndarray = 1.

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("session name must be a nonempty string")
        frequency = float(self.frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency_hz must be positive and finite")
        xy = _real_array(self.samples_xy_m, "samples_xy_m")
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
            raise ValueError("samples_xy_m must have nonempty shape (m,2)")
        u = _complex_array(self.steering, "steering")
        if u.ndim != 1 or not len(u):
            raise ValueError("steering must be a nonempty complex vector")
        y = _complex_array(self.field, "field", (len(xy),))
        w = _real_array(self.weight, "weight")
        if w.ndim == 0:
            w = np.full(len(xy), float(w))
        if w.shape != (len(xy),) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weight must be fixed nonnegative scalar/vector with positive mass")
        for name, value in (("samples_xy_m", xy), ("steering", u),
                            ("field", y), ("weight", w)):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "frequency_hz", frequency)


@dataclass(frozen=True)
class FrequencyGroup:
    name: str
    sessions: Sequence[Session]
    lambda_reg: float
    b0: np.ndarray | None = None
    template_prior: np.ndarray | None = None
    template_strength: float = 0.
    amplitude_prior: np.ndarray | None = None
    amplitude_strength: float = 0.
    amplitude_epsilon: float = 1e-8
    amplitude_n_starts: int = 3
    amplitude_random_seed: int = 20261028
    amplitude_maxiter: int = 5000
    amplitude_newton_steps: int = 250
    amplitude_stationarity_tol: float = 1e-12
    amplitude_min_hessian_rcond: float = 1e-12

    def __post_init__(self):
        sessions = tuple(self.sessions)
        if not self.name or not sessions:
            raise ValueError("each named frequency/chain group needs sessions")
        if len({s.name for s in sessions}) != len(sessions):
            raise ValueError("session names must be unique within each group")
        if len({s.frequency_hz for s in sessions}) != 1:
            raise ValueError("a frequency group must contain exactly one frequency")
        n = len(sessions[0].steering)
        if any(len(s.steering) != n for s in sessions):
            raise ValueError("steering lengths must match within a group")
        lam = float(self.lambda_reg)
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lambda_reg must be finite and nonnegative")
        prior = np.zeros(n, complex) if self.b0 is None else _complex_array(self.b0, "b0", (n,))
        strength = float(self.template_strength)
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("template_strength must be finite and nonnegative")
        template = None
        if self.template_prior is not None:
            template = _complex_array(self.template_prior, "template_prior", (n,))
            if linalg.norm(template) == 0:
                raise ValueError("template_prior must be nonzero")
            template = _readonly(template)
        if strength > 0 and template is None:
            raise ValueError("positive template_strength requires a complex template_prior")
        amp_strength = float(self.amplitude_strength)
        amp_epsilon = float(self.amplitude_epsilon)
        amp_tolerance = float(self.amplitude_stationarity_tol)
        amp_rcond = float(self.amplitude_min_hessian_rcond)
        if not np.isfinite(amp_strength) or amp_strength < 0:
            raise ValueError("amplitude_strength must be finite and nonnegative")
        if not np.isfinite(amp_epsilon) or amp_epsilon <= 0:
            raise ValueError("amplitude_epsilon must be finite and positive")
        if not 0 < amp_tolerance <= 1e-8 or not 0 < amp_rcond < 1:
            raise ValueError("amplitude stationarity tolerance must be <=1e-8 and Hessian rcond floor positive")
        for name, minimum in (("amplitude_n_starts", 1), ("amplitude_maxiter", 0),
                              ("amplitude_newton_steps", 0), ("amplitude_random_seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        amplitude = None
        if self.amplitude_prior is not None:
            amplitude = _real_array(self.amplitude_prior, "amplitude_prior", (n,))
            if np.any(amplitude <= 0):
                raise ValueError("amplitude_prior must contain positive relative amplitudes")
            amplitude = _readonly(amplitude)
        if amp_strength > 0 and amplitude is None:
            raise ValueError("positive amplitude_strength requires amplitude_prior")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "lambda_reg", lam)
        object.__setattr__(self, "b0", _readonly(prior))
        object.__setattr__(self, "template_prior", template)
        object.__setattr__(self, "template_strength", strength)
        object.__setattr__(self, "amplitude_prior", amplitude)
        object.__setattr__(self, "amplitude_strength", amp_strength)
        object.__setattr__(self, "amplitude_epsilon", amp_epsilon)
        object.__setattr__(self, "amplitude_stationarity_tol", amp_tolerance)
        object.__setattr__(self, "amplitude_min_hessian_rcond", amp_rcond)


@dataclass
class ProfileEvaluation:
    x: np.ndarray
    geometry_mm: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    b: dict[str, np.ndarray]
    db: dict[str, np.ndarray]
    gains: dict[str, complex]
    data_residuals: dict[str, np.ndarray]
    weighted_data_residuals: dict[str, np.ndarray]
    penalty_residuals: dict[str, np.ndarray]
    data_jacobians: dict[str, np.ndarray]
    weighted_data_jacobians: dict[str, np.ndarray]
    penalty_jacobians: dict[str, np.ndarray]
    amplitude_residuals: dict[str, np.ndarray]
    amplitude_jacobians: dict[str, np.ndarray]
    diagnostics: dict

    @property
    def cost(self):
        return float(0.5 * np.dot(self.residual, self.residual))


@dataclass
class FitResult:
    geometry_mm: np.ndarray
    evaluation: ProfileEvaluation
    success: bool
    status: int
    message: str
    nfev: int
    njev: int
    optimality: float
    active_mask: np.ndarray
    counts: dict
    elapsed_seconds: float
    x: np.ndarray

    @property
    def b(self):
        return self.evaluation.b

    @property
    def gains(self):
        return self.evaluation.gains

    @property
    def cost(self):
        return self.evaluation.cost


class _Factor:
    """An augmented QR/SVD, or guarded Cholesky, never an explicit inverse."""
    def __init__(self, B, c, backend, threshold, counts):
        self.B = B
        self.backend = backend
        self.condition_h_1 = None
        self.fallback_reason = None
        counts["linear_factorizations"] += 1
        if backend == "normal":
            H = B.conj().T @ B
            try:
                factor, lower = linalg.cho_factor(H, lower=True, check_finite=False)
                rcond, info = get_lapack_funcs("pocon", (factor,))(
                    factor, float(linalg.norm(H, 1)), uplo="L")
                self.condition_h_1 = float(1/rcond) if rcond > 0 else np.inf
                if info != 0 or not np.isfinite(self.condition_h_1) or self.condition_h_1 > threshold:
                    self.fallback_reason = "normal matrix condition threshold exceeded"
                else:
                    self.factor = factor, lower
                    self.b = linalg.cho_solve(self.factor, B.conj().T @ c, check_finite=False)
                    return
            except linalg.LinAlgError:
                self.fallback_reason = "normal Cholesky factorization failed"
            self.backend = "qr"
            counts["normal_fallbacks"] += 1
            counts["linear_factorizations"] += 1
        if self.backend == "qr":
            self.Q, self.R = linalg.qr(B, mode="economic", check_finite=False)
            diagonal = np.abs(np.diag(self.R))
            if B.shape[0] < B.shape[1] or np.min(diagonal) <= np.finfo(float).eps * max(B.shape) * np.max(diagonal):
                raise linalg.LinAlgError("augmented system is rank deficient; use positive fixed regularization")
            self.b = linalg.solve_triangular(self.R, self.Q.conj().T @ c, check_finite=False)
        elif self.backend == "svd":
            self.U, self.s, self.Vh = linalg.svd(B, full_matrices=False, check_finite=False)
            if len(self.s) != B.shape[1] or self.s[-1] <= np.finfo(float).eps * max(B.shape) * self.s[0]:
                raise linalg.LinAlgError("augmented system is rank deficient")
            self.b = self.Vh.conj().T @ ((self.U.conj().T @ c)/self.s)

    def derivative(self, derivative_b_product, derivative_adjoint_residual):
        # r=c-Bb; q=B_j b; H b_j=B_j^H r-B^H q.
        if self.backend == "qr":
            z = linalg.solve_triangular(self.R.conj().T, derivative_adjoint_residual,
                                       lower=True, check_finite=False)
            return linalg.solve_triangular(self.R, z-self.Q.conj().T @ derivative_b_product,
                                           check_finite=False)
        rhs = derivative_adjoint_residual - self.B.conj().T @ derivative_b_product
        if self.backend == "normal":
            return linalg.cho_solve(self.factor, rhs, check_finite=False)
        return self.Vh.conj().T @ ((self.Vh @ derivative_adjoint_residual)/(self.s*self.s)
                                  - (self.U.conj().T @ derivative_b_product)/self.s)


class AmplitudeProfileError(RuntimeError):
    """An inner branch failed: no outer profile derivative may be used."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"amplitude inner branch failed stationary/curvature gate in {diagnostics['group']}")


class JointProblem:
    """Frozen training problem; groups share geometry and have independent b.

    ``normal_condition_max`` limits estimated cond_1(B^H B), not cond(B).
    Every accelerated group is checked against augmented QR at its first valid
    evaluation (or always); failed checks fall back to QR and remain recorded.
    W, lambda, b0 and the identity penalty are immutable for this instance.
    """
    def __init__(self, elements_m, groups, q=2., sign=-1, backend="qr",
                 normal_condition_max=1e8, normal_qr_check="first",
                 qr_coefficient_rtol=1e-6, qr_residual_rtol=1e-8):
        elements = _real_array(elements_m, "elements_m")
        if (elements.ndim != 2 or not len(elements) or elements.shape[1] not in (2, 3)
                or (elements.shape[1] == 3 and np.any(elements[:, 2] != 0))):
            raise ValueError("elements_m must be nonempty planar (n,2) or (n,3)")
        self.elements_m = _readonly(elements)
        self.groups = tuple(groups)
        if not self.groups or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("nonempty, uniquely named frequency groups required")
        sessions = [s for g in self.groups for s in g.sessions]
        if len({s.name for s in sessions}) != len(sessions):
            raise ValueError("session names must be globally unique")
        if any(len(s.steering) != len(elements) for s in sessions):
            raise ValueError("each steering vector must match the elements")
        if backend not in ("qr", "svd", "normal"):
            raise ValueError("backend must be qr, svd, or normal")
        if normal_qr_check not in ("first", "always"):
            raise ValueError("normal_qr_check must be first or always; unverified acceleration is disabled")
        if not np.isfinite(normal_condition_max) or normal_condition_max <= 1:
            raise ValueError("normal_condition_max must be finite and greater than one")
        if not np.isfinite(q) or q < 0 or sign not in (-1, 1):
            raise ValueError("q must be nonnegative and sign must be +/-1")
        self.q, self.sign, self.backend = float(q), int(sign), backend
        self.normal_condition_max = float(normal_condition_max)
        self.normal_qr_check = normal_qr_check
        self.qr_coefficient_rtol = float(qr_coefficient_rtol)
        self.qr_residual_rtol = float(qr_residual_rtol)
        self.parameter_layout = ["t_d", "t_dx", "t_dy"]
        self._gain_indices = {}
        for group in self.groups:
            for session in group.sessions[1:]:
                self._gain_indices[session.name] = len(self.parameter_layout)
                self.parameter_layout.extend((f"rho:{group.name}:{session.name}",
                                              f"phi:{group.name}:{session.name}"))
        self._qr_verified = set()
        self.qr_checks = []
        self.counts = {key: 0 for key in ("evaluations", "cache_hits", "kernel_calls",
                         "kernel_derivative_calls", "linear_factorizations", "linear_derivative_solves",
                         "normal_fallbacks", "qr_verifications", "failed_evaluations",
                         "amplitude_solves", "amplitude_starts", "amplitude_nfev", "amplitude_njev",
                         "amplitude_newton_steps", "amplitude_derivative_solves")}
        self.failure_ledger = []
        self._last = None

    @property
    def parameter_count(self):
        return len(self.parameter_layout)

    def initial_parameters(self, geometry_mm=(60., 0., 0.), gains=None):
        x = np.zeros(self.parameter_count)
        x[:3] = geometry_to_parameters(geometry_mm)
        for name, index in self._gain_indices.items():
            gain = complex((gains or {}).get(name, 1.))
            if not np.isfinite(gain) or abs(gain) == 0:
                raise ValueError("initial gains must be finite and nonzero")
            x[index:index+2] = np.log(abs(gain)), np.angle(gain)
        return x

    def _gains(self, x):
        gains = {s.name: 1.+0j for g in self.groups for s in g.sessions}
        for name, index in self._gain_indices.items():
            gains[name] = np.exp(x[index] + 1j*x[index+1])
        if not all(np.isfinite(v) for v in gains.values()):
            raise ValueError("nonfinite outer session gains")
        return gains

    def gains_from_parameters(self, x):
        """Public parameter-layout decoder, including every fixed reference g=1."""
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        parameters_to_geometry(x[:3])
        return self._gains(x)

    def predict(self, session, geometry_mm, b, g=1.):
        b = _complex_array(b, "b", (len(self.elements_m),))
        K = scalar_kernel(self.elements_m, session.samples_xy_m, session.frequency_hz,
                          geometry_mm, self.q, self.sign)
        return complex(g) * (K @ (session.steering * b))

    def evaluate(self, x):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        if self._last is not None and np.array_equal(x, self._last.x):
            self.counts["cache_hits"] += 1
            return self._last
        geometry = parameters_to_geometry(x[:3])
        gains = self._gains(x)
        self.counts["evaluations"] += 1
        try:
            evaluation = self._evaluate(x, geometry, gains)
        except (ValueError, linalg.LinAlgError, FloatingPointError, AmplitudeProfileError):
            self.counts["failed_evaluations"] += 1
            raise
        self._last = evaluation
        return evaluation

    def _evaluate(self, x, geometry, gains):
        residuals, jacobians, bs, dbs, diagnostics = [], [], {}, {}, {}
        data, weighted, penalties = {}, {}, {}
        data_jacobians, weighted_jacobians, penalty_jacobians = {}, {}, {}
        amplitude_residuals, amplitude_jacobians = {}, {}
        p, n = self.parameter_count, len(self.elements_m)
        # A per-evaluation cache shares kernels for truly identical frequency/grid.
        kernel_cache = {}
        for group in self.groups:
            m = sum(len(s.field) for s in group.sessions)
            penalty_n = n*(1+int(group.template_strength > 0))
            B = np.zeros((m+penalty_n, n), complex)
            c = np.zeros(m+penalty_n, complex)
            blocks, cursor = [], 0
            for session in group.sessions:
                stop = cursor+len(session.field)
                key = session.frequency_hz, session.samples_xy_m.shape, session.samples_xy_m.tobytes()
                if key not in kernel_cache:
                    kernel_cache[key] = scalar_kernel(self.elements_m, session.samples_xy_m,
                        session.frequency_hz, geometry, self.q, self.sign)
                    self.counts["kernel_calls"] += 1
                B[cursor:stop] = (kernel_cache[key] * session.steering[None, :]
                                  * (session.weight*gains[session.name])[:, None])
                c[cursor:stop] = session.weight * session.field
                blocks.append((session, slice(cursor, stop), key))
                cursor = stop
            B[m:m+n] = group.lambda_reg*np.eye(n)
            c[m:m+n] = group.lambda_reg*group.b0
            if group.template_strength > 0:
                a = group.template_prior
                B[m+n:] = group.template_strength*(np.eye(n)-np.outer(a, a.conj())/np.vdot(a, a).real)
            factor = _Factor(B, c, self.backend, self.normal_condition_max, self.counts)
            verification, baseline = None, None
            if factor.backend == "normal" and (self.normal_qr_check == "always" or group.name not in self._qr_verified):
                baseline = _Factor(B, c, "qr", self.normal_condition_max, self.counts)
                rnormal, rqr = c-B@factor.b, c-B@baseline.b
                coefficient_error = float(linalg.norm(factor.b-baseline.b)/max(linalg.norm(baseline.b), 1e-30))
                residual_error = float(linalg.norm(rnormal-rqr)/max(linalg.norm(c), 1e-30))
                passed = coefficient_error <= self.qr_coefficient_rtol and residual_error <= self.qr_residual_rtol
                verification = {"group": group.name, "geometry_mm": geometry.tolist(),
                    "coefficient_relative_difference": coefficient_error,
                    "residual_relative_difference": residual_error, "passed": bool(passed)}
                self.qr_checks.append(verification)
                self.counts["qr_verifications"] += 1
                if not passed:
                    baseline.fallback_reason = "normal solution failed augmented QR verification"
                    baseline.condition_h_1 = factor.condition_h_1
                    factor = baseline
                    self.counts["normal_fallbacks"] += 1
            b = factor.b
            r = c-B@b
            J = np.zeros((m+penalty_n, p), complex)
            db = np.zeros((n, p), complex)
            amplitude_active = group.amplitude_strength > 0
            checking_derivatives = verification is not None and factor.backend == "normal" and not amplitude_active
            if checking_derivatives:
                baseline_r = c-B@baseline.b
                baseline_db, baseline_J = np.zeros_like(db), np.zeros_like(J)
            for j in (range(p) if not amplitude_active else ()):
                if j >= 3 and not any(self._gain_indices.get(s.name, -10) in (j, j-1) for s in group.sessions):
                    continue
                qproduct = np.zeros(m+penalty_n, complex)
                adjoint = np.zeros(n, complex)
                if checking_derivatives:
                    baseline_q, baseline_adjoint = np.zeros_like(qproduct), np.zeros_like(adjoint)
                derivative_cache = {}
                for session, sl, key in blocks:
                    if j < 3:
                        if key not in derivative_cache:
                            derivative_cache[key] = scalar_kernel(self.elements_m, session.samples_xy_m,
                                session.frequency_hz, geometry, self.q, self.sign, derivative=j)
                            self.counts["kernel_derivative_calls"] += 1
                        Bj = derivative_cache[key] * session.steering[None, :] * (session.weight*gains[session.name])[:, None]
                    elif j == self._gain_indices.get(session.name, -10):
                        Bj = B[sl]
                    elif j == self._gain_indices.get(session.name, -10)+1:
                        Bj = 1j*B[sl]
                    else:
                        continue
                    qproduct[sl] = Bj@b
                    adjoint += Bj.conj().T@r[sl]
                    if checking_derivatives:
                        baseline_q[sl] = Bj@baseline.b
                        baseline_adjoint += Bj.conj().T@baseline_r[sl]
                db[:, j] = factor.derivative(qproduct, adjoint)
                J[:, j] = -qproduct-B@db[:, j]
                self.counts["linear_derivative_solves"] += 1
                if checking_derivatives:
                    baseline_db[:, j] = baseline.derivative(baseline_q, baseline_adjoint)
                    baseline_J[:, j] = -baseline_q-B@baseline_db[:, j]
                    self.counts["linear_derivative_solves"] += 1
            if checking_derivatives:
                derivative_error = float(linalg.norm(db-baseline_db)/max(linalg.norm(baseline_db), 1e-30))
                jacobian_error = float(linalg.norm(J-baseline_J)/max(linalg.norm(baseline_J), 1e-30))
                verification.update(db_relative_difference=derivative_error,
                                    jacobian_relative_difference=jacobian_error)
                verification["passed"] = bool(verification["passed"] and
                    derivative_error <= self.qr_coefficient_rtol and jacobian_error <= self.qr_coefficient_rtol)
                if verification["passed"]:
                    self._qr_verified.add(group.name)
                else:
                    baseline.fallback_reason = "normal derivatives failed augmented QR verification"
                    baseline.condition_h_1 = factor.condition_h_1
                    factor, b, r, db, J = baseline, baseline.b, baseline_r, baseline_db, baseline_J
                    self.counts["normal_fallbacks"] += 1
            amplitude_diagnostic = None
            stationarity_normalization = "||B^H r||_2 / max(||B||_F ||r||_2, eps64 ||B||_F ||c||_2, 1e-30)"
            if amplitude_active:
                if verification is not None and verification["passed"]:
                    verification["scope"] = "linear initializer only; final amplitude derivative uses exact real Hessian"
                    self._qr_verified.add(group.name)
                solution = solve_amplitude(B, c, 0., group.amplitude_prior, group.amplitude_strength,
                    initial=b, epsilon=group.amplitude_epsilon, n_starts=group.amplitude_n_starts,
                    random_seed=group.amplitude_random_seed, maxiter=group.amplitude_maxiter,
                    stationarity_tol=group.amplitude_stationarity_tol,
                    min_hessian_rcond=group.amplitude_min_hessian_rcond,
                    newton_steps=group.amplitude_newton_steps)
                self.counts["amplitude_solves"] += 1
                self.counts["amplitude_starts"] += len(solution.starts)
                self.counts["amplitude_nfev"] += sum(s.nfev for s in solution.starts)
                self.counts["amplitude_njev"] += sum(s.njev for s in solution.starts)
                self.counts["amplitude_newton_steps"] += sum(s.newton_steps for s in solution.starts)
                objective_scale = max(abs(solution.objective), 1e-30)
                coefficient_scale = max(linalg.norm(solution.b), group.amplitude_epsilon*np.sqrt(n))
                start_records = []
                for index, start in enumerate(solution.starts):
                    record = asdict(start)
                    gap = float(start.objective-solution.objective)
                    record.update(selected=index == solution.chosen_start,
                        objective_gap_from_selected=gap,
                        objective_relative_gap_from_selected=gap/objective_scale,
                        relative_b_distance_from_selected=float(linalg.norm(
                            solution.start_coefficients[:, index]-solution.b)/coefficient_scale))
                    start_records.append(record)
                converged_objectives = [s.objective for s in solution.starts if s.converged]
                amplitude_diagnostic = {"group": group.name, "geometry_mm": geometry.tolist(),
                    "x": x.tolist(), "success": solution.success, "stationarity": solution.stationarity,
                    "stationarity_tol": group.amplitude_stationarity_tol,
                    "hessian_rcond": solution.hessian_rcond,
                    "hessian_positive_definite": solution.hessian_positive_definite,
                    "chosen_start": solution.chosen_start, "selected_objective": solution.objective,
                    "starts": start_records,
                    "stationary_start_count": len(converged_objectives),
                    "converged_objective_range": ([float(min(converged_objectives)), float(max(converged_objectives))]
                                                  if converged_objectives else None),
                    "relative_objective_gap_normalization": "(objective_i-selected_objective)/max(abs(selected_objective),1e-30); signed gaps retain failed lower objectives",
                    "max_converged_relative_b_spread": solution.max_converged_relative_b_spread,
                    "has_failed_lower_objective_start": solution.has_failed_lower_objective_start,
                    "initialization": "linear augmented solve at the current outer point; fixed random-phase seed; no cross-evaluation warm start",
                    "random_seed": int(group.amplitude_random_seed), "n_starts": int(group.amplitude_n_starts),
                    "branch_global_optimality_established": False,
                    "global_profile_smoothness_established": False,
                    "profile_derivative_scope": "exact full-Hessian derivative of the selected local stationary branch; not a derivative across best-of-starts branch switches",
                    "branch_selection": "lowest objective among converged stationary starts; all failed and alternate starts retained",
                    "branch_interpretation": "coefficient spread alone does not prove geometry error; inspect objective closeness, profile continuity, geometry and prediction stability. Start indices label initializations, so index changes alone do not prove different branches.",
                    "strength": group.amplitude_strength, "epsilon": group.amplitude_epsilon,
                    "penalty": "strength * center(log(sqrt(abs(b)^2+epsilon^2))-log(relative_amplitudes)); common log scale removed; phases unconstrained"}
                if not solution.success:
                    self.failure_ledger.append(amplitude_diagnostic)
                    raise AmplitudeProfileError(amplitude_diagnostic)
                active_indices = list(range(3))
                for session in group.sessions[1:]:
                    index = self._gain_indices[session.name]
                    active_indices.extend((index, index+1))
                def derivatives():
                    for j in active_indices:
                        matrix = np.zeros_like(B)
                        derivative_cache = {}
                        for session, sl, key in blocks:
                            if j < 3:
                                if key not in derivative_cache:
                                    derivative_cache[key] = scalar_kernel(self.elements_m, session.samples_xy_m,
                                        session.frequency_hz, geometry, self.q, self.sign, derivative=j)
                                    self.counts["kernel_derivative_calls"] += 1
                                matrix[sl] = derivative_cache[key]*session.steering[None, :]*(session.weight*gains[session.name])[:, None]
                            elif j == self._gain_indices.get(session.name, -10):
                                matrix[sl] = B[sl]
                            elif j == self._gain_indices.get(session.name, -10)+1:
                                matrix[sl] = 1j*B[sl]
                        yield matrix
                active_db, active_J = implicit_derivative(solution, derivatives())
                db[:, active_indices] = active_db
                k = len(B)
                J[:, active_indices] = -(active_J[:k]+1j*active_J[k:2*k])
                amplitude_J = np.zeros((n, p))
                amplitude_J[:, active_indices] = -active_J[-n:]
                amplitude_jacobians[group.name] = amplitude_J
                amplitude_residuals[group.name] = -solution.amplitude_residual
                self.counts["amplitude_derivative_solves"] += len(active_indices)
                b, r = solution.b, -solution.data_residual
                stationarity = solution.stationarity
                stationarity_normalization = "||gradient_full_real_objective||_inf / max(1, ||stack(B^H c)||_inf)"
            else:
                gradient = B.conj().T@r
                stationarity = float(linalg.norm(gradient)/max(linalg.norm(B, "fro")*linalg.norm(r),
                                                            np.finfo(float).eps*linalg.norm(B, "fro")*linalg.norm(c), 1e-30))
            for session, sl, key in blocks:
                weighted[session.name] = r[sl].copy()
                A = kernel_cache[key]*session.steering[None, :]*gains[session.name]
                data[session.name] = session.field-A@b
                weighted_jacobians[session.name] = J[sl].copy()
                if np.all(session.weight > 0):
                    unweighted_J = J[sl]/session.weight[:, None]
                else:
                    unweighted_J = -A@db
                    for j in range(3):
                        Kj = scalar_kernel(self.elements_m, session.samples_xy_m, session.frequency_hz,
                            geometry, self.q, self.sign, derivative=j)
                        self.counts["kernel_derivative_calls"] += 1
                        unweighted_J[:, j] -= gains[session.name]*Kj@(session.steering*b)
                    if session.name in self._gain_indices:
                        index = self._gain_indices[session.name]
                        unweighted_J[:, index] -= A@b
                        unweighted_J[:, index+1] -= 1j*(A@b)
                data_jacobians[session.name] = unweighted_J
            penalties[group.name] = r[m:].copy()
            penalty_jacobians[group.name] = J[m:].copy()
            bs[group.name], dbs[group.name] = b.copy(), db
            diagnostics[group.name] = {"backend_requested": self.backend,
                "backend_used": "nonlinear_amplitude_exact_hessian" if amplitude_active else factor.backend,
                "linear_initialization_backend": factor.backend if amplitude_active else None,
                "normal_condition_h_1": factor.condition_h_1,
                "normal_condition_max": self.normal_condition_max,
                "fallback_reason": factor.fallback_reason, "qr_verification": verification,
                "stationarity": stationarity, "lambda_reg": group.lambda_reg,
                "stationarity_normalization": stationarity_normalization,
                "inner_solver": "nonlinear relative amplitude; exact real stationary Hessian" if amplitude_active else "complex linear augmented least squares",
                "amplitude": amplitude_diagnostic,
                "penalty": "fixed physical lambda * (b-b0), L=I", "reference_session": group.sessions[0].name,
                "template_strength": group.template_strength,
                "template_penalty": "mu * (I-aa^H/(a^Ha)) b; unknown common complex scale eliminated" if group.template_strength > 0 else None,
                "template_scale": ([float(np.real(np.vdot(group.template_prior, b)/np.vdot(group.template_prior, group.template_prior))),
                                    float(np.imag(np.vdot(group.template_prior, b)/np.vdot(group.template_prior, group.template_prior)))]
                                   if group.template_strength > 0 else None),
                "reference_gain": [1., 0.], "data_residual_norm": float(linalg.norm(r[:m])),
                "penalty_residual_norm": float(linalg.norm(r[m:]))}
            residuals.append(r)
            jacobians.append(J)
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        amp_r = np.concatenate(list(amplitude_residuals.values())) if amplitude_residuals else np.empty(0)
        amp_J = np.vstack(list(amplitude_jacobians.values())) if amplitude_jacobians else np.empty((0, p))
        return ProfileEvaluation(x.copy(), geometry, np.r_[r.real, r.imag, amp_r], np.vstack((J.real, J.imag, amp_J)),
            bs, dbs, gains, data, weighted, penalties, data_jacobians, weighted_jacobians,
            penalty_jacobians, amplitude_residuals, amplitude_jacobians,
            {"groups": diagnostics, "qr_checks": list(self.qr_checks),
                "parameter_layout": list(self.parameter_layout), "counts": dict(self.counts),
                "gauge": "first session g=1 separately in every frequency/chain group"})

    def validation_evaluate(self, x, validation_groups: Mapping[str, Sequence[Session]], gains=None):
        """Held-out residual and full derivative, including training db/dx.

        Unseen sessions use a caller-frozen scalar (default one). No validation
        samples influence b or g. A scalar calibrated elsewhere can be supplied;
        this method does not fit that scalar or differentiate its calibration.
        """
        evaluation = self.evaluate(x)
        residuals, jacobians, predictions = [], [], {}
        group_lookup = {g.name: g for g in self.groups}
        for group_name, sessions in validation_groups.items():
            if group_name not in group_lookup:
                raise ValueError("validation group must have a trained response")
            for session in sessions:
                if session.frequency_hz != group_lookup[group_name].sessions[0].frequency_hz:
                    raise ValueError("validation frequency must match the trained group")
                gain = complex((gains or {}).get(session.name, evaluation.gains.get(session.name, 1.)))
                K = scalar_kernel(self.elements_m, session.samples_xy_m, session.frequency_hz,
                                  evaluation.geometry_mm, self.q, self.sign)
                A = gain*K*session.steering[None, :]
                prediction = A@evaluation.b[group_name]
                J = -A@evaluation.db[group_name]
                for j in range(3):
                    Kj = scalar_kernel(self.elements_m, session.samples_xy_m, session.frequency_hz,
                        evaluation.geometry_mm, self.q, self.sign, derivative=j)
                    J[:, j] -= gain*Kj@(session.steering*evaluation.b[group_name])
                if session.name in self._gain_indices and not (gains and session.name in gains):
                    index = self._gain_indices[session.name]
                    J[:, index] -= prediction
                    J[:, index+1] -= 1j*prediction
                predictions[session.name] = prediction
                residuals.append(session.weight*(session.field-prediction))
                jacobians.append(session.weight[:, None]*J)
        if not residuals:
            raise ValueError("nonempty validation sessions required")
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        return {"residual": np.r_[r.real, r.imag], "jacobian": np.vstack((J.real, J.imag)),
                "predictions": predictions, "evaluation": evaluation}

    def fit(self, start_geometry_mm=(60., 0., 0.), max_nfev=1500, fixed_geometry=False,
            start_gains=None, ftol=1e-8, gtol=1e-7, xtol=1e-7, gain_log_bounds=(-3., 3.),
            start_parameters=None, checkpoint_callback=None):
        """Bounded TRF; fixed_geometry can be True or a three-axis bool mask.

        The default exp(-3)..exp(3) gain bounds are numerical assumptions, never
        measurement drift facts. Phases remain unwrapped and unbounded.
        An optional callback receives small JSON-compatible initial, accepted
        iteration and final records. This module performs no checkpoint I/O.
        Raising StopIteration on an accepted iteration stops TRF with status -2.
        """
        started = time.perf_counter()
        initial_counts = dict(self.counts)
        if start_parameters is None:
            x0 = self.initial_parameters(start_geometry_mm, start_gains)
        else:
            if start_gains is not None:
                raise ValueError("use either start_parameters or start_gains, not both")
            x0 = _real_array(start_parameters, "start_parameters", (self.parameter_count,))
            parameters_to_geometry(x0[:3])
            self._gains(x0)
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError("checkpoint_callback must be callable")
        mask = np.full(3, fixed_geometry, dtype=bool) if np.ndim(fixed_geometry) == 0 else np.asarray(fixed_geometry, dtype=bool)
        if mask.shape != (3,):
            raise ValueError("fixed_geometry must be bool or a three-axis bool mask")
        free = np.r_[~mask, np.ones(self.parameter_count-3, bool)]
        lower, upper = np.full(self.parameter_count, -np.inf), np.full(self.parameter_count, np.inf)
        lower[:3], upper[:3] = -1., 1.
        log_bounds = _real_array(gain_log_bounds, "gain_log_bounds", (2,))
        if log_bounds[0] >= log_bounds[1]:
            raise ValueError("gain_log_bounds must be strictly increasing")
        for index in self._gain_indices.values():
            lower[index], upper[index] = log_bounds
        def expand(z):
            x = x0.copy()
            x[free] = z
            return x
        last_accepted, last_cost, last_nfev, iteration = x0.copy(), None, 0, 0
        def checkpoint(event, x, cost, nfev, **extra):
            if checkpoint_callback is not None:
                checkpoint_callback({"event": event, "x": x.tolist(),
                    "geometry_mm": parameters_to_geometry(x[:3]).tolist(),
                    "gains": {k: [float(v.real), float(v.imag)] for k, v in self._gains(x).items()},
                    "cost": None if cost is None else float(cost), "nfev": int(nfev),
                    "iteration": int(iteration), "max_nfev": int(max_nfev),
                    "counts": {k: self.counts[k]-initial_counts[k] for k in self.counts},
                    "elapsed_seconds": time.perf_counter()-started, **extra})
        def accepted_iteration(intermediate_result):
            nonlocal last_accepted, last_cost, last_nfev, iteration
            iteration += 1
            last_accepted = expand(intermediate_result.x)
            last_cost, last_nfev = float(intermediate_result.cost), int(intermediate_result.nfev)
            checkpoint("iteration", last_accepted, last_cost, last_nfev)
        try:
            if checkpoint_callback is not None:
                last_cost = self.evaluate(x0).cost
                last_nfev = 1
                checkpoint("initial", x0, last_cost, last_nfev)
            if np.any(free):
                result = optimize.least_squares(lambda z: self.evaluate(expand(z)).residual, x0[free],
                    jac=lambda z: self.evaluate(expand(z)).jacobian[:, free], bounds=(lower[free], upper[free]),
                    method="trf", max_nfev=max_nfev, ftol=ftol, gtol=gtol, xtol=xtol,
                    callback=accepted_iteration if checkpoint_callback is not None else None)
                x = expand(result.x)
                success, status, message = bool(result.success), int(result.status), str(result.message)
                nfev, njev, optimality = int(result.nfev), int(result.njev), float(result.optimality)
                active = np.zeros(self.parameter_count, int)
                active[free] = result.active_mask
            else:
                x, success, status, message = x0, True, 1, "all outer variables fixed; inner solve only"
                nfev, njev, optimality, active = 1, 1, 0., np.zeros(self.parameter_count, int)
            evaluation = self.evaluate(x)
        except (ValueError, linalg.LinAlgError, FloatingPointError, AmplitudeProfileError) as error:
            checkpoint("failed", last_accepted, last_cost, last_nfev,
                       success=False, status=-3, message=str(error),
                       failed_attempt=getattr(error, "diagnostics", None))
            raise
        checkpoint("final", x, evaluation.cost, nfev, success=success, status=status, message=message,
                   njev=njev, optimality=optimality)
        counts = {k: self.counts[k]-initial_counts[k] for k in self.counts}
        return FitResult(evaluation.geometry_mm.copy(), evaluation, success, status, message,
                         nfev, njev, optimality, active, counts, time.perf_counter()-started, x.copy())


def deterministic_starts_mm(count=12):
    """Old twelve starts, optionally followed by twelve fixed LHS starts.

    The original prefix includes lattice/half-lattice neighbourhoods and is
    unchanged for historical reproduction. Extra starts are a truth-independent
    12-stratum Latin hypercube from NumPy seed 43103, within the same hard bounds.
    """
    if count not in (12, 24):
        raise ValueError("count must be 12 or 24")
    original = np.array(((60., 0., 0.), (51., -9., -9.), (69., 9., 9.),
        (51., 9., -9.), (69., -9., 9.), (55., -5.5, 0.), (65., 5.5, 0.),
        (55., 0., -5.5), (65., 0., 5.5), (60., -9., 5.5),
        (60., 9., -5.5), (60., 5.5, 5.5)))
    if count == 12:
        return original
    rng = np.random.default_rng(43103)
    strata = np.column_stack([rng.permutation(12) for _ in range(3)])
    unit = (strata+rng.uniform(size=(12, 3)))/12
    extra = BOUNDS_MM[:, 0]+unit*np.ptp(BOUNDS_MM, axis=1)
    return np.vstack((original, extra))


def summarize_multistart(results, relative_cost_tolerance=.01, alias_distance_mm=.5):
    """Preserve every termination and flag separated near minima, not CIs."""
    if not results:
        raise ValueError("nonempty fit results required")
    order = sorted(range(len(results)), key=lambda i: results[i].cost)
    best = results[order[0]]
    converged_order = [i for i in order if results[i].success]
    best_converged = converged_order[0] if converged_order else None
    threshold = best.cost*(1+relative_cost_tolerance) + 1e-24
    near = [i for i in order if results[i].cost <= threshold]
    separated = [(i, j) for pos, i in enumerate(near) for j in near[pos+1:]
                 if np.max(np.abs(results[i].geometry_mm-results[j].geometry_mm)) >= alias_distance_mm]
    return {"best_index": order[0], "best_observed_index": order[0],
        "best_converged_index": best_converged, "best_observed_converged": best.success,
        "numerical_status": "uncertain: best observed trajectory did not converge" if not best.success
            else "converged numerical minimum; physical identifiability requires separate diagnostics",
        "cost_order": order, "near_indices": near,
        "near_alias_pairs": separated, "near_alias_detected": bool(separated),
        "near_cost_relative_tolerance": relative_cost_tolerance,
        "alias_separation_mm": alias_distance_mm,
        "starts": [{"geometry_mm": r.geometry_mm.tolist(), "cost": r.cost,
            "success": r.success, "status": r.status, "message": r.message,
            "nfev": r.nfev, "njev": r.njev, "counts": r.counts,
            "elapsed_seconds": r.elapsed_seconds, "boundary_active": r.active_mask[:3].tolist()}
            for r in results], "interpretation": "objective near-minimum set; not a confidence interval"}
