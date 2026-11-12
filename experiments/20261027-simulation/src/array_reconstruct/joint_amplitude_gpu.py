"""Double-precision tensor port of the exact relative-amplitude solver.

All physical residuals, gradients, full Hessians, Newton systems and implicit
derivatives are evaluated on the input tensor device.  NumPy double inputs
may be explicitly uploaded; existing tensors must already have the required
dtype and device.  Explicit ``device='cpu'`` is a supported smoke-test path,
and is identified as CPU in the returned diagnostics.

The CPU work is restricted to deterministic phase-start construction,
SciPy L-BFGS control over small coordinate/gradient vectors, and LAPACK
``dpocon`` condition estimates from copied Cholesky factors.  There is no
CPU physical-solver fallback and no explicit matrix inverse.  Every such
transfer and condition-estimation call is recorded.  Importing this module
does not query or initialize CUDA.

The objective and original-coordinate derivative are mathematical ports of
``joint_amplitude.py``; the Newton-first stages and affine whitening follow
``joint_amplitude_fast.py``.  In particular, this is the full residual-weighted
Hessian of the smoothed log-amplitude prior, not a Gauss-Newton approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable

import numpy as np
from scipy import optimize
from scipy.linalg.lapack import dpocon
import torch

from .joint_amplitude import AmplitudeStart
from .joint_amplitude_fast import _integer_budget

__all__ = ["GPUAmplitudeSolution", "solve_amplitude_gpu", "implicit_derivative_gpu"]


_COUNT_KEYS = (
    "objective_evaluations", "gradient_evaluations", "hessian_evaluations",
    "residual_evaluations", "quadratic_hessian_constructions",
    "quadratic_gram_matrix_products", "cholesky_factorization_attempts",
    "cholesky_factorization_successes", "quadratic_factorization_attempts",
    "newton_factorization_attempts", "gate_factorization_attempts",
    "rcond_estimation_calls", "cpu_dpocon_calls", "cholesky_solve_calls",
    "triangular_solve_calls", "initial_native_lstsq_calls",
    "initial_native_rank_svd_calls", "newton_iterations",
    "newton_direction_attempts", "newton_accepted_steps",
    "newton_rejected_directions", "line_search_trials",
    "line_search_nonfinite_trials", "armijo_acceptances",
    "rounding_refinement_acceptances", "lbfgs_objective_calls",
    "whitening_forward_products", "host_scalar_reads",
    "device_to_host_scalar_transfers", "device_to_host_scalar_bytes",
    "device_to_host_tensor_transfers", "device_to_host_tensor_bytes",
    "host_to_device_tensor_transfers", "host_to_device_tensor_bytes",
    "cpu_numpy_materializations", "input_array_materializations",
    "coordinate_to_host_calls", "gradient_to_host_calls",
    "cholesky_to_host_calls", "calibration_to_host_calls",
    "coordinate_from_host_calls", "phase_start_from_host_calls",
    "input_from_host_calls", "physical_cpu_fallback_calls",
)


def _counts():
    return dict.fromkeys(_COUNT_KEYS, 0)


def _item(value, counts):
    counts["host_scalar_reads"] += 1
    if value.device.type == "cuda":
        counts["device_to_host_scalar_transfers"] += 1
        counts["device_to_host_scalar_bytes"] += value.element_size()
    return value.detach().item()


def _all_finite(value, counts):
    return bool(_item(torch.isfinite(value).all(), counts))


def _to_numpy(value, counts, category):
    counts[category + "_to_host_calls"] += 1
    counts["cpu_numpy_materializations"] += 1
    if value.device.type == "cuda":
        counts["device_to_host_tensor_transfers"] += 1
        counts["device_to_host_tensor_bytes"] += value.numel()*value.element_size()
    # Copy also on CPU: SciPy's mutable coordinates must never alias a state.
    return value.detach().contiguous().cpu().numpy().copy()


def _from_numpy(value, device, dtype, counts, category):
    array = np.ascontiguousarray(value)
    counts[category + "_from_host_calls"] += 1
    counts["input_array_materializations"] += 1
    tensor = torch.from_numpy(array)
    if tensor.dtype != dtype:
        raise TypeError("host input dtype must already have the requested double precision")
    if device.type == "cuda":
        counts["host_to_device_tensor_transfers"] += 1
        counts["host_to_device_tensor_bytes"] += array.nbytes
        return tensor.to(device=device, dtype=dtype)
    return tensor.detach()


def _input_tensor(value, device, dtype, counts, name):
    if isinstance(value, torch.Tensor):
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype must be {dtype}; lower precision is rejected")
        if value.device != device:
            raise ValueError(f"{name} device {value.device} does not match {device}")
        return value.detach()
    array = np.asarray(value)
    expected = np.dtype(np.complex128 if dtype == torch.complex128 else np.float64)
    # Real double source data can be promoted to complex double exactly;
    # float32/complex64 sources are rejected rather than silently upcast.
    if dtype == torch.complex128 and array.dtype == np.dtype(np.float64):
        array = array.astype(np.complex128)
    if array.dtype != expected:
        raise TypeError(f"{name} dtype must be {expected}; double precision input required")
    return _from_numpy(array, device, dtype, counts, "input")


def _stack(value):
    return torch.cat((value.real, value.imag))


def _complex(value):
    n = value.numel()//2
    return torch.complex(value[:n], value[n:])


def _inf_norm(value, counts):
    return float(_item(torch.linalg.vector_norm(value, ord=float("inf")), counts))


def _parameter(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite Python/NumPy real scalar")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


@dataclass
class GPUAmplitudeSolution:
    b: torch.Tensor
    residual: torch.Tensor
    data_residual: torch.Tensor
    ridge_residual: torch.Tensor
    amplitude_residual: torch.Tensor
    objective: float
    stationarity: float
    success: bool
    chosen_start: int
    starts: tuple[AmplitudeStart, ...]
    start_coefficients: torch.Tensor
    max_converged_relative_b_spread: float
    has_failed_lower_objective_start: bool
    epsilon: float
    ridge: float
    strength: float
    relative_amplitudes: torch.Tensor
    hessian_rcond: float
    hessian_positive_definite: bool
    hessian: torch.Tensor = field(repr=False)
    _B: torch.Tensor = field(repr=False)
    _y: torch.Tensor = field(repr=False)
    _factor: tuple[torch.Tensor, bool] | None = field(repr=False)
    gpu_diagnostics: dict = field(default_factory=dict, repr=False)
    fast_diagnostics: dict = field(default_factory=dict, repr=False)


@dataclass
class _State:
    z: torch.Tensor
    value: float
    gradient: torch.Tensor
    gradient_norm: float
    finite: bool


@dataclass
class _Gate:
    z: torch.Tensor
    hessian: torch.Tensor
    factor: tuple[torch.Tensor, bool] | None
    rcond: float
    stationarity: float
    success: bool


class _GPUAmplitudeObjective:
    """The original physical objective, without host matrix arithmetic."""

    def __init__(self, B, y, ridge, a, strength, epsilon, counts):
        self.B, self.y, self.n = B, y, B.shape[1]
        self.ridge, self.strength, self.epsilon = ridge, strength, epsilon
        self.loga = torch.log(a)
        counts["quadratic_gram_matrix_products"] += 1
        gram = B.mH @ B
        counts["quadratic_hessian_constructions"] += 1
        H0 = torch.cat((torch.cat((gram.real, -gram.imag), dim=1),
                        torch.cat((gram.imag, gram.real), dim=1)), dim=0)
        H0.diagonal().add_(ridge**2)
        self.quadratic_hessian = (H0+H0.T)*0.5
        self.stationarity_scale = max(1.0, _inf_norm(_stack(B.mH @ y), counts))

    def prior(self, z):
        x, v = z[:self.n], z[self.n:]
        s = x*x + v*v + self.epsilon**2
        ell = 0.5*torch.log(s)-self.loga
        h = ell-ell.mean()
        return h, s, x/s, v/s

    def residual(self, z):
        r = self.B @ _complex(z)-self.y
        return torch.cat((_stack(r), self.ridge*z,
                          self.strength*self.prior(z)[0]))

    def value_gradient(self, z):
        r = self.B @ _complex(z)-self.y
        h, _, dx, dv = self.prior(z)
        gradient = _stack(self.B.mH @ r)+self.ridge**2*z
        gradient = gradient+self.strength**2*torch.cat((dx*h, dv*h))
        value = 0.5*(torch.vdot(r, r).real+self.ridge**2*torch.dot(z, z)
                     +self.strength**2*torch.dot(h, h))
        return value, gradient

    def hessian(self, z):
        H = self.quadratic_hessian.clone()
        if not self.strength:
            return H
        n = self.n
        x, v = z[:n], z[n:]
        h, s, dx, dv = self.prior(z)
        d = torch.cat((dx, dv))
        prior_H = -torch.outer(d, d)/n
        j = torch.arange(n, device=z.device)
        prior_H[j, j] += dx*dx+h*(1/s-2*x*x/(s*s))
        prior_H[n+j, n+j] += dv*dv+h*(1/s-2*v*v/(s*s))
        cross = dx*dv-2*h*x*v/(s*s)
        prior_H[j, n+j] += cross
        prior_H[n+j, j] += cross
        H = H+self.strength**2*prior_H
        return (H+H.T)*0.5


class _TrackedObjective:
    def __init__(self, problem, counts):
        self.problem, self.counts = problem, counts

    def evaluate(self, z):
        self.counts["objective_evaluations"] += 1
        self.counts["gradient_evaluations"] += 1
        z = z.detach().clone()
        value, gradient = self.problem.value_gradient(z)
        host_value = float(_item(value, self.counts))
        gradient_norm = _inf_norm(gradient, self.counts)
        finite = bool(np.isfinite(host_value) and np.isfinite(gradient_norm)
                      and _all_finite(z, self.counts))
        return _State(z, host_value, gradient, gradient_norm, finite)

    def hessian(self, z):
        self.counts["hessian_evaluations"] += 1
        return self.problem.hessian(z)

    def residual(self, z):
        self.counts["residual_evaluations"] += 1
        return self.problem.residual(z)


def _stationarity(state, scale):
    return state.gradient_norm/scale if np.isfinite(state.gradient_norm) else float("inf")


def _cholesky(H, counts, kind):
    counts["cholesky_factorization_attempts"] += 1
    counts[kind + "_factorization_attempts"] += 1
    lower, info = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
    if int(_item(info, counts)) != 0:
        return None
    counts["cholesky_factorization_successes"] += 1
    return lower


def _factor_and_rcond(H, counts, kind):
    if not _all_finite(H, counts):
        return None, 0.0, "nonfinite_hessian"
    lower = _cholesky(H, counts, kind)
    if lower is None:
        return None, 0.0, "cholesky_failed"
    anorm = float(_item(torch.linalg.matrix_norm(H, ord=1), counts))
    lower_cpu = np.array(_to_numpy(lower, counts, "cholesky"),
                         dtype=np.float64, order="F", copy=True)
    counts["rcond_estimation_calls"] += 1
    counts["cpu_dpocon_calls"] += 1
    rcond, info = dpocon(lower_cpu, anorm, uplo="L")
    if info != 0 or not np.isfinite(rcond):
        return None, 0.0, "cpu_dpocon_failed"
    return (lower, True), float(rcond), "available"


def _original_gate(tracked, state, tolerance, min_rcond):
    H = tracked.hessian(state.z)
    factor, rcond, _ = _factor_and_rcond(H, tracked.counts, "gate")
    stationarity = _stationarity(state, tracked.problem.stationarity_scale)
    success = bool(state.finite and stationarity <= tolerance
                   and factor is not None and rcond >= min_rcond)
    return _Gate(state.z, H, factor, rcond, stationarity, success)


def _newton_stage(tracked, state, tolerance, budget, damping_metric,
                  metric_scale, max_damping_attempts, max_line_search):
    counts = tracked.counts
    before_iterations = counts["newton_iterations"]
    before_steps = counts["newton_accepted_steps"]
    before_trials = counts["line_search_trials"]
    reason, damping_values = "iteration_limit", []
    for _ in range(budget):
        if not state.finite:
            reason = "nonfinite_value_or_gradient"
            break
        if _stationarity(state, tracked.problem.stationarity_scale) <= tolerance:
            reason = "stationarity_satisfied"
            break
        counts["newton_iterations"] += 1
        H = tracked.hessian(state.z)
        if not _all_finite(H, counts):
            reason = "nonfinite_hessian"
            break
        hscale = max(1.0, float(_item(H.diagonal().abs().max(), counts)))
        first_damping = (1e-8*hscale if damping_metric is None
                         else 1e-8*max(1.0, hscale/metric_scale))
        damping, direction = 0.0, None
        for _ in range(max_damping_attempts):
            counts["newton_direction_attempts"] += 1
            if damping == 0:
                candidate = H
            elif damping_metric is None:
                candidate = H.clone()
                candidate.diagonal().add_(damping)
            else:
                candidate = H+damping*damping_metric
            lower = _cholesky(candidate, counts, "newton")
            if lower is not None:
                counts["cholesky_solve_calls"] += 1
                proposed = torch.cholesky_solve(
                    -state.gradient[:, None], lower, upper=False).squeeze(1)
                slope = float(_item(torch.dot(proposed, state.gradient), counts))
                if np.isfinite(slope) and slope < 0 and _all_finite(proposed, counts):
                    direction = proposed
                    break
            counts["newton_rejected_directions"] += 1
            damping = first_damping if damping == 0 else damping*10.0
        if direction is None:
            reason = "no_descent_direction"
            break
        step = 1.0
        for _ in range(max_line_search):
            counts["line_search_trials"] += 1
            trial_z = state.z+step*direction
            if not _all_finite(trial_z, counts):
                counts["line_search_nonfinite_trials"] += 1
                step *= 0.5
                continue
            trial = tracked.evaluate(trial_z)
            if not trial.finite:
                counts["line_search_nonfinite_trials"] += 1
                step *= 0.5
                continue
            armijo = trial.value <= state.value+1e-4*step*slope
            rounding_refinement = (
                abs(trial.value-state.value)
                <= 32*np.finfo(float).eps*max(1.0, abs(state.value))
                and trial.gradient_norm < 0.5*state.gradient_norm
            )
            if armijo or rounding_refinement:
                state = trial
                damping_values.append(float(damping))
                counts["newton_accepted_steps"] += 1
                counts["armijo_acceptances" if armijo
                       else "rounding_refinement_acceptances"] += 1
                break
            step *= 0.5
        else:
            reason = "line_search_failed"
            break
    if state.finite and _stationarity(state, tracked.problem.stationarity_scale) <= tolerance:
        reason = "stationarity_satisfied"
    elif budget == 0:
        reason = "zero_budget"
    return state, {
        "requested_iterations": budget,
        "actual_iterations": counts["newton_iterations"]-before_iterations,
        "accepted_steps": counts["newton_accepted_steps"]-before_steps,
        "line_search_trials": counts["line_search_trials"]-before_trials,
        "accepted_direction_damping": damping_values, "termination": reason,
    }


def _from_coordinate(coordinate, reference, lower, counts):
    coordinate_tensor = _from_numpy(
        np.asarray(coordinate, dtype=np.float64), reference.device, torch.float64,
        counts, "coordinate")
    if lower is None:
        return coordinate_tensor
    counts["triangular_solve_calls"] += 1
    return reference+torch.linalg.solve_triangular(
        lower.T, coordinate_tensor[:, None], upper=True).squeeze(1)


def _lbfgs_stage(tracked, state, reference, lower, lower_inf_norm,
                 tolerance, maxiter, max_line_search):
    counts = tracked.counts
    summary = {
        "used": maxiter > 0, "whitened": lower is not None,
        "optimizer_success": False, "optimizer_message": "zero L-BFGS budget",
        "optimizer_iterations": 0, "optimizer_nfev": 0, "optimizer_njev": 0,
        "exception_type": None, "original_stationarity_after": None,
        "maxiter": maxiter, "maxfun": (maxiter+1)*max_line_search+1,
        "maxls": max_line_search, "control_backend": "scipy_cpu_lbfgsb",
        "physical_backend": "torch_"+reference.device.type,
    }
    if maxiter == 0:
        summary["original_stationarity_after"] = _stationarity(
            state, tracked.problem.stationarity_scale)
        return state, summary
    if lower is None:
        native_coordinate = state.z
        coordinate_gtol = tolerance*tracked.problem.stationarity_scale
    else:
        counts["whitening_forward_products"] += 1
        native_coordinate = lower.T @ (state.z-reference)
        coordinate_gtol = tolerance*tracked.problem.stationarity_scale/max(1.0, lower_inf_norm)
    coordinate = _to_numpy(native_coordinate, counts, "coordinate")
    last_state, last_coordinate = None, None

    def evaluate(coordinate_value):
        nonlocal last_state, last_coordinate
        counts["lbfgs_objective_calls"] += 1
        z = _from_coordinate(coordinate_value, reference, lower, counts)
        last_state = tracked.evaluate(z)
        last_coordinate = np.array(coordinate_value, dtype=np.float64, copy=True)
        if lower is None:
            gradient = last_state.gradient
        else:
            counts["triangular_solve_calls"] += 1
            gradient = torch.linalg.solve_triangular(
                lower, last_state.gradient[:, None], upper=False).squeeze(1)
        return last_state.value, _to_numpy(gradient, counts, "gradient")

    try:
        result = optimize.minimize(
            evaluate, coordinate, method="L-BFGS-B", jac=True,
            options={"maxiter": maxiter, "maxfun": summary["maxfun"],
                     "gtol": coordinate_gtol, "ftol": 1e-15,
                     "maxls": max_line_search, "maxcor": 20},
        )
        summary.update(
            optimizer_success=bool(result.success), optimizer_message=str(result.message),
            optimizer_iterations=int(result.nit), optimizer_nfev=int(result.nfev),
            optimizer_njev=int(result.njev),
        )
        if last_coordinate is not None and np.array_equal(result.x, last_coordinate):
            state = last_state
        elif np.shape(result.x) == tuple(state.z.shape) and np.all(np.isfinite(result.x)):
            state = tracked.evaluate(_from_coordinate(result.x, reference, lower, counts))
        else:
            summary["optimizer_success"] = False
            summary["optimizer_message"] += "; nonfinite or invalid final coefficients"
    except (ValueError, FloatingPointError, OverflowError) as exc:
        summary["optimizer_success"] = False
        summary["exception_type"] = type(exc).__name__
        summary["optimizer_message"] = str(exc)
    summary["original_stationarity_after"] = _stationarity(
        state, tracked.problem.stationarity_scale)
    return state, summary


@torch.no_grad()
def solve_amplitude_gpu(
    B, y, ridge, a, strength, initial=None, *, epsilon=1e-8,
    n_starts=3, random_seed=20261028, maxiter=5000,
    stationarity_tol=1e-12, min_hessian_rcond=1e-12, newton_steps=250,
    newton_first_steps=None, whitening_min_rcond=None,
    max_damping_attempts=16, max_line_search=35, lbfgs_max_line_search=40,
    device=None,
):
    """Solve the exact amplitude profile on CUDA, or explicitly on CPU.

    Tensor inputs must be complex128 (B/y/initial) or float64 (a), on the
    selected device; lower precision tensors are rejected.  NumPy source
    arrays must also be double precision, and their explicit input transfers
    are recorded.  Device defaults to ``B.device`` for tensor B and to
    ``cuda:0`` otherwise.  No autograd graph is built through optimization.

    The success gate uses original-coordinate stationarity (default 1e-12,
    never looser than 1e-8), an unshifted full-Hessian Cholesky factor and its
    1-norm reciprocal condition estimate.  All three deterministic starts and
    failed branches are retained.  Newton uses at most 250 iterations per
    start in total; unused initial-stage iterations transfer to polishing.
    L-BFGS uses at most 5000 iterations and can stop earlier normally.
    """
    if device is None:
        selected_device = B.device if isinstance(B, torch.Tensor) else torch.device("cuda:0")
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and selected_device.index is None:
        selected_device = (B.device if isinstance(B, torch.Tensor)
                           and B.device.type == "cuda" else torch.device("cuda:0"))
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("only explicit CPU and CUDA tensor devices are supported")
    shared_counts = _counts()
    B = _input_tensor(B, selected_device, torch.complex128, shared_counts, "B")
    y = _input_tensor(y, selected_device, torch.complex128, shared_counts, "y")
    a = _input_tensor(a, selected_device, torch.float64, shared_counts, "a")
    if B.ndim != 2 or y.shape != (B.shape[0],) or a.shape != (B.shape[1],):
        raise ValueError("B must be (m,n), y (m,), and a (n,)")
    if not B.shape[0] or not B.shape[1]:
        raise ValueError("empty inner problem")
    if not (_all_finite(B, shared_counts) and _all_finite(y, shared_counts)
            and _all_finite(a, shared_counts) and bool(_item((a > 0).all(), shared_counts))):
        raise ValueError("finite B/y and strictly positive finite a required")
    ridge = _parameter(ridge, "ridge")
    strength = _parameter(strength, "strength")
    epsilon = _parameter(epsilon, "epsilon", positive=True)
    stationarity_tol = _parameter(stationarity_tol, "stationarity_tol", positive=True)
    if stationarity_tol > 1e-8:
        raise ValueError("stationarity_tol must be at most 1e-8")
    min_hessian_rcond = _parameter(min_hessian_rcond, "min_hessian_rcond", positive=True)
    if min_hessian_rcond > 1:
        raise ValueError("min_hessian_rcond must be in (0,1]")
    if whitening_min_rcond is None:
        whitening_min_rcond = max(min_hessian_rcond, np.finfo(float).eps)
    whitening_min_rcond = _parameter(whitening_min_rcond, "whitening_min_rcond", positive=True)
    if whitening_min_rcond > 1:
        raise ValueError("whitening_min_rcond must be in (0,1]")
    n_starts = _integer_budget(n_starts, "n_starts")
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    maxiter = _integer_budget(maxiter, "maxiter", 5000)
    newton_steps = _integer_budget(newton_steps, "newton_steps", 250)
    if newton_first_steps is None:
        newton_first_steps = (4*newton_steps)//5
    newton_first_steps = _integer_budget(newton_first_steps, "newton_first_steps", newton_steps)
    max_damping_attempts = _integer_budget(max_damping_attempts, "max_damping_attempts")
    max_line_search = _integer_budget(max_line_search, "max_line_search")
    lbfgs_max_line_search = _integer_budget(lbfgs_max_line_search, "lbfgs_max_line_search")
    if min(max_damping_attempts, max_line_search, lbfgs_max_line_search) < 1:
        raise ValueError("damping and line-search attempt limits must be positive")
    tf32_disabled = None
    if selected_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tf32_disabled = bool(not torch.backends.cuda.matmul.allow_tf32
                             and not torch.backends.cudnn.allow_tf32)
    n = B.shape[1]
    initial_driver = None
    if initial is None:
        augmented = torch.cat((B, ridge*torch.eye(n, dtype=torch.complex128,
                                                 device=selected_device)), dim=0)
        rhs = torch.cat((y, torch.zeros(n, dtype=torch.complex128, device=selected_device)))
        initial_driver = "gels" if selected_device.type == "cuda" else "gelsd"
        if selected_device.type == "cuda":
            # CUDA's gels driver assumes full column rank.  Check it natively
            # and fail explicitly; there is no hidden host least-squares solve.
            shared_counts["initial_native_rank_svd_calls"] += 1
            singular = torch.linalg.svdvals(augmented)
            threshold = torch.finfo(torch.float64).eps*max(augmented.shape)*singular[0]
            rank = int(_item(torch.count_nonzero(singular > threshold), shared_counts))
            if rank < n:
                raise RuntimeError("native CUDA gels initialization requires numerical full column rank; "
                                   "supply a public physical initial vector or regularize the model")
        shared_counts["initial_native_lstsq_calls"] += 1
        initial = torch.linalg.lstsq(augmented, rhs, driver=initial_driver).solution
        del augmented, rhs
        initial_source = "native_double_augmented_complex_lstsq_"+initial_driver
    else:
        initial = _input_tensor(initial, selected_device, torch.complex128,
                                shared_counts, "initial")
        initial_source = "caller_provided_physical_coefficients"
    if initial.shape != (n,) or not _all_finite(initial, shared_counts):
        raise ValueError("initial must be a finite complex128 (n,) vector on the selected device")
    initial = initial.detach()
    problem = _GPUAmplitudeObjective(B, y, ridge, a, strength, epsilon, shared_counts)
    H0 = problem.quadratic_hessian
    h0_factor, h0_rcond, whitening_reason = _factor_and_rcond(H0, shared_counts, "quadratic")
    if h0_factor is not None and h0_rcond < whitening_min_rcond:
        whitening_reason = "quadratic_rcond_below_whitening_threshold"
    whitening_available = bool(h0_factor is not None and h0_rcond >= whitening_min_rcond)
    lower = h0_factor[0] if whitening_available else None
    damping_metric = H0 if whitening_available else None
    metric_scale = (max(np.finfo(float).tiny,
                        float(_item(H0.diagonal().abs().max(), shared_counts)))
                    if whitening_available else 1.0)
    lower_inf_norm = (float(_item(torch.linalg.matrix_norm(lower, ord=float("inf")), shared_counts))
                      if whitening_available else 1.0)
    reference = _stack(initial)
    initial_starts = [initial]
    if n_starts > 1:
        a_cpu = _to_numpy(a, shared_counts, "calibration")
        initial_norm = float(_item(torch.linalg.vector_norm(initial), shared_counts))
        amplitude_norm = float(_item(torch.linalg.vector_norm(a), shared_counts))
        start_scale = max(initial_norm, epsilon*np.sqrt(n))/amplitude_norm
        rng = np.random.default_rng(random_seed)
        for _ in range(n_starts-1):
            phase_start = start_scale*a_cpu*np.exp(1j*rng.uniform(-np.pi, np.pi, n))
            initial_starts.append(_from_numpy(phase_start, selected_device, torch.complex128,
                                              shared_counts, "phase_start"))

    records, solutions, stage_diagnostics, branch_counts = [], [], [], []
    best_success = best_failed = None
    for index, start in enumerate(initial_starts):
        counts = _counts()
        branch_counts.append(counts)
        tracked = _TrackedObjective(problem, counts)
        state = tracked.evaluate(_stack(start))
        initial_value = state.value
        initial_stationarity = _stationarity(state, problem.stationarity_scale)
        state, first_stage = _newton_stage(
            tracked, state, stationarity_tol, newton_first_steps,
            damping_metric, metric_scale, max_damping_attempts, max_line_search)
        gate = None
        if _stationarity(state, problem.stationarity_scale) <= stationarity_tol:
            gate = _original_gate(tracked, state, stationarity_tol, min_hessian_rcond)
        fallback_needed = gate is None or not gate.success
        final_budget = newton_steps-counts["newton_iterations"]
        if fallback_needed:
            state, fallback = _lbfgs_stage(
                tracked, state, reference, lower, lower_inf_norm,
                stationarity_tol, maxiter, lbfgs_max_line_search)
            state, final_stage = _newton_stage(
                tracked, state, stationarity_tol, final_budget, damping_metric,
                metric_scale, max_damping_attempts, max_line_search)
        else:
            fallback = {
                "used": False, "whitened": lower is not None, "optimizer_success": None,
                "optimizer_message": "skipped: initial Newton stage passed original gate",
                "optimizer_iterations": 0, "optimizer_nfev": 0, "optimizer_njev": 0,
                "exception_type": None, "original_stationarity_after": gate.stationarity,
                "maxiter": maxiter, "maxfun": (maxiter+1)*lbfgs_max_line_search+1,
                "maxls": lbfgs_max_line_search, "control_backend": "scipy_cpu_lbfgsb",
                "physical_backend": "torch_"+selected_device.type,
            }
            final_stage = {
                "requested_iterations": final_budget, "actual_iterations": 0,
                "accepted_steps": 0, "line_search_trials": 0,
                "accepted_direction_damping": [],
                "termination": "skipped: original gate already passed",
            }
        # Coordinates are immutable clones.  Identity reuse needs no hidden
        # GPU equality reduction or host synchronization.
        if gate is None or gate.z is not state.z:
            gate = _original_gate(tracked, state, stationarity_tol, min_hessian_rcond)
        optimizer_success = (bool(fallback["optimizer_success"]) if fallback_needed
                             else first_stage["termination"] == "stationarity_satisfied")
        optimizer_message = (str(fallback["optimizer_message"]) if fallback_needed
                             else "Newton-first original-coordinate stationarity satisfied")
        record = AmplitudeStart(
            index, state.value, gate.stationarity, gate.success, optimizer_success,
            optimizer_message, counts["objective_evaluations"], counts["gradient_evaluations"],
            counts["newton_accepted_steps"], gate.rcond, gate.factor is not None)
        records.append(record)
        solutions.append(state.z)
        stage_diagnostics.append({
            "index": index, "initialization": initial_source if index == 0
            else "numpy_seeded_public_amplitude_shape_phases_uploaded",
            "initial_objective": initial_value, "initial_stationarity": initial_stationarity,
            "initial_newton": first_stage, "fallback_needed": fallback_needed,
            "lbfgs": fallback, "final_newton": final_stage,
            "original_gate_success": gate.success,
            "original_hessian_positive_definite": gate.factor is not None,
            "original_hessian_rcond": gate.rcond, "global_optimality_certified": False,
        })
        candidate = (index, state, gate, tracked)
        if gate.success:
            if best_success is None or state.value < best_success[1].value:
                best_success = candidate
        elif best_failed is None or (
            np.isfinite(state.value) and (not np.isfinite(best_failed[1].value)
                                         or state.value < best_failed[1].value)
        ):
            best_failed = candidate

    chosen, state, gate, tracked = best_success if best_success is not None else best_failed
    record = records[chosen]
    residual = tracked.residual(state.z)
    m = B.shape[0]
    b = _complex(state.z)
    start_coefficients = torch.stack([_complex(z) for z in solutions], dim=1)
    eligible = [j for j, item in enumerate(records) if item.converged]
    denominator = max(float(_item(torch.linalg.vector_norm(b), tracked.counts)), epsilon*np.sqrt(n))
    spread = max((float(_item(torch.linalg.vector_norm(start_coefficients[:, j]-b),
                              tracked.counts))/denominator for j in eligible), default=0.0)
    failed_lower = any(not item.converged and item.objective < record.objective
                       -1e-10*max(1.0, abs(record.objective)) for item in records)
    solution = GPUAmplitudeSolution(
        b=b, residual=residual, data_residual=_complex(residual[:2*m]),
        ridge_residual=residual[2*m:2*m+2*n], amplitude_residual=residual[2*m+2*n:],
        objective=record.objective, stationarity=record.stationarity, success=record.converged,
        chosen_start=chosen, starts=tuple(records), start_coefficients=start_coefficients,
        max_converged_relative_b_spread=spread, has_failed_lower_objective_start=failed_lower,
        epsilon=epsilon, ridge=ridge, strength=strength, relative_amplitudes=a.clone(),
        hessian_rcond=gate.rcond, hessian_positive_definite=gate.factor is not None,
        hessian=gate.hessian, _B=B, _y=y, _factor=gate.factor)
    for diagnostic, counts in zip(stage_diagnostics, branch_counts):
        diagnostic["counts"] = counts.copy()
    total_counts = {key: shared_counts[key]+sum(counts[key] for counts in branch_counts)
                    for key in _COUNT_KEYS}
    diagnostics = {
        "solver": "torch_double_newton_first_exact_amplitude",
        "device": str(selected_device), "device_type": selected_device.type,
        "physical_backend": "torch_"+selected_device.type,
        "actual_cuda_physical_arithmetic": selected_device.type == "cuda",
        "complex_dtype": str(B.dtype), "real_dtype": str(residual.dtype),
        "tf32_disabled": tf32_disabled, "autograd_through_iterations": False,
        "physical_cpu_fallback": False, "physical_objective_approximated": False,
        "original_coordinate_gate": True, "global_optimality_certified": False,
        "objective_identity": "tensor port of joint_amplitude._AmplitudeObjective",
        "source_dependencies": ["array_reconstruct/joint_amplitude_gpu.py",
                                "array_reconstruct/joint_amplitude_fast.py",
                                "array_reconstruct/joint_amplitude.py"],
        "stationarity_tolerance": stationarity_tol, "min_hessian_rcond": min_hessian_rcond,
        "n_starts": n_starts, "random_seed": random_seed,
        "initial_native_lstsq_driver": initial_driver,
        "newton_iteration_budget_per_start": newton_steps,
        "initial_newton_iteration_budget_per_start": newton_first_steps,
        "reserved_final_newton_iterations_per_start": newton_steps-newton_first_steps,
        "final_newton_budget_policy": "total budget minus actual initial Newton iterations",
        "lbfgs_iteration_budget_per_start": maxiter,
        "whitening": {
            "available": whitening_available, "quadratic_hessian_rcond": h0_rcond,
            "min_rcond": whitening_min_rcond, "guard_status": whitening_reason,
            "reference": initial_source,
            "mapping": "z=reference+solve(L.T,w); gradient_w=solve(L,gradient_z)",
            "fallback_coordinates": "whitened" if whitening_available else "physical",
            "newton_damping_metric": "H0" if whitening_available else "scaled_identity",
            "preconditioner_differentiated_in_implicit_derivative": False,
        },
        "condition_estimator": "CPU scipy LAPACK dpocon on copied native Cholesky; no inverse",
        "lbfgs_control_backend": "CPU SciPy; coordinates and gradients only",
        "phase_start_backend": "NumPy public amplitude shape and deterministic phases, explicit upload",
        "shared_setup_counts": shared_counts.copy(), "starts": stage_diagnostics,
        "counts": total_counts,
        "count_semantics": {
            "objective_and_gradient": "each joint native physical value_gradient call counts once in each",
            "hessian": "full original tensor Hessian constructions including curvature gates",
            "factorization": "actual torch.linalg.cholesky_ex calls including failed calls",
            "line_search_trials": "Newton trials; L-BFGS trial work is in lbfgs_objective_calls",
            "transfers": "explicit tensor copies and scalar item reads; CPU smoke copies are not CUDA transfers",
            "implicit_derivative": "recorded separately after solve; solve counts remain fixed",
        },
    }
    solution.gpu_diagnostics = diagnostics
    solution.fast_diagnostics = diagnostics
    return solution


@torch.no_grad()
def implicit_derivative_gpu(
    solution: GPUAmplitudeSolution, dB: Iterable[torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native db/dtheta and the full original real residual Jacobian.

    Each derivative is a complex128 tensor on the solution device.  The fixed
    right-hand side, ridge, relative amplitude shape, strength and epsilon do
    not depend on theta.  The exact unshifted stationary Hessian is used;
    neither damping nor the optimizer's whitening transform is differentiated.
    """
    if not solution.success or solution._factor is None:
        raise RuntimeError("amplitude branch failed stationary/curvature gate")
    if isinstance(dB, torch.Tensor) and dB.ndim == 2:
        dB = [dB]
    B, b, r = solution._B, solution.b, solution.data_residual
    lower, is_lower = solution._factor
    if not is_lower:
        raise RuntimeError("original stationary factor must be lower Cholesky")
    counts = _counts()
    n = b.numel()
    s = b.abs()**2+solution.epsilon**2
    dx, dv = b.real/s, b.imag/s
    db_columns, jac_columns = [], []
    for matrix in dB:
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("each dB must be a native complex128 Tensor")
        if matrix.dtype != torch.complex128 or matrix.device != B.device:
            raise TypeError("each dB must have complex128 dtype and the solution device")
        matrix = matrix.detach()
        if matrix.shape != B.shape or not _all_finite(matrix, counts):
            raise ValueError("each dB must be finite and have the shape of B")
        partial_data = matrix @ b
        partial_gradient = _stack(matrix.mH @ r+B.mH @ partial_data)
        counts["cholesky_solve_calls"] += 1
        dz = torch.cholesky_solve(-partial_gradient[:, None], lower, upper=False).squeeze(1)
        db = _complex(dz)
        data_jac = partial_data+B @ db
        amp_jac = dx*dz[:n]+dv*dz[n:]
        amp_jac = amp_jac-amp_jac.mean()
        jac_columns.append(torch.cat((_stack(data_jac), solution.ridge*dz,
                                      solution.strength*amp_jac)))
        db_columns.append(db)
    history = solution.gpu_diagnostics.setdefault("implicit_derivative_calls", [])
    history.append({"parameters": len(db_columns), "device": str(B.device),
                    "counts": counts, "physical_cpu_fallback": False,
                    "original_unshifted_hessian": True})
    if not db_columns:
        return (torch.empty((n, 0), dtype=torch.complex128, device=B.device),
                torch.empty((solution.residual.numel(), 0), dtype=torch.float64, device=B.device))
    return torch.stack(db_columns, dim=1), torch.stack(jac_columns, dim=1)
