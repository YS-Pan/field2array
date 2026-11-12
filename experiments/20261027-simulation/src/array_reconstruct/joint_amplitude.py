"""Nonlinear relative-amplitude inner problem for joint geometry profiles.

The calibration supplies positive *relative* amplitudes, never phases.  For
``b = x + 1j*v`` this module minimizes one half the squared norm of

    [Re(B b-y), Im(B b-y), ridge*x, ridge*v,
     strength * P(0.5*log(x*x+v*v+epsilon**2)-log(a))].

``P`` subtracts the mean, analytically eliminating an unknown common log
scale.  The epsilon is fixed in physical coefficient units and is not changed
with geometry.  At amplitudes comparable to epsilon the prior is explicitly
the log of a smoothed magnitude, rather than log(abs(b)); derivatives there
remain finite.  No common phase is constrained by the prior.

The profile derivative differentiates the real stationary equations using the
full Hessian, including the residual-weighted second derivatives of log
magnitude.  A stationary local branch is not a certificate of global optimality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import linalg, optimize
from scipy.linalg.lapack import dpocon


def _stack(b: np.ndarray) -> np.ndarray:
    return np.concatenate((b.real, b.imag))


def _complex(z: np.ndarray) -> np.ndarray:
    n = len(z) // 2
    return z[:n] + 1j * z[n:]


def _adjoint_product(B: np.ndarray, r: np.ndarray) -> np.ndarray:
    # Avoid materializing a second tall complex matrix just to conjugate B.
    return (B.T @ r.conj()).conj()


@dataclass(frozen=True)
class AmplitudeStart:
    index: int
    objective: float
    stationarity: float
    converged: bool
    optimizer_success: bool
    optimizer_message: str
    nfev: int
    njev: int
    newton_steps: int
    hessian_rcond: float
    hessian_positive_definite: bool


@dataclass
class AmplitudeSolution:
    b: np.ndarray
    residual: np.ndarray
    data_residual: np.ndarray
    ridge_residual: np.ndarray
    amplitude_residual: np.ndarray
    objective: float
    stationarity: float
    success: bool
    chosen_start: int
    starts: tuple[AmplitudeStart, ...]
    start_coefficients: np.ndarray
    max_converged_relative_b_spread: float
    has_failed_lower_objective_start: bool
    epsilon: float
    ridge: float
    strength: float
    relative_amplitudes: np.ndarray
    hessian_rcond: float
    hessian_positive_definite: bool
    hessian: np.ndarray = field(repr=False)
    _B: np.ndarray = field(repr=False)
    _y: np.ndarray = field(repr=False)
    _factor: tuple[np.ndarray, bool] | None = field(repr=False)


class _AmplitudeObjective:
    def __init__(self, B, y, ridge, a, strength, epsilon):
        self.B, self.y = B, y
        self.n = B.shape[1]
        self.ridge, self.strength, self.epsilon = ridge, strength, epsilon
        self.loga = np.log(a)
        gram = _adjoint_product(B, B)
        self.quadratic_hessian = np.block(
            [[gram.real, -gram.imag], [gram.imag, gram.real]]
        )
        self.quadratic_hessian.flat[:: 2 * self.n + 1] += ridge**2
        self.quadratic_hessian = (
            self.quadratic_hessian + self.quadratic_hessian.T
        ) * 0.5
        self.stationarity_scale = max(
            1.0, float(np.linalg.norm(_stack(_adjoint_product(B, y)), np.inf))
        )

    def prior(self, z):
        n = self.n
        x, v = z[:n], z[n:]
        s = x*x + v*v + self.epsilon**2
        ell = 0.5*np.log(s) - self.loga
        h = ell - ell.mean()
        return h, s, x/s, v/s

    def residual(self, z):
        r = self.B @ _complex(z) - self.y
        h = self.prior(z)[0]
        return np.concatenate((_stack(r), self.ridge*z, self.strength*h))

    def value_gradient(self, z):
        r = self.B @ _complex(z) - self.y
        h, _, dx, dv = self.prior(z)
        gradient = _stack(_adjoint_product(self.B, r)) + self.ridge**2*z
        gradient += self.strength**2 * np.concatenate((dx*h, dv*h))
        value = 0.5*(np.vdot(r, r).real + self.ridge**2*np.dot(z,z)
                     + self.strength**2*np.dot(h,h))
        return float(value), gradient

    def hessian(self, z):
        n = self.n
        x, v = z[:n], z[n:]
        h, s, dx, dv = self.prior(z)
        H = self.quadratic_hessian.copy()
        if not self.strength:
            return H
        # D.T P D = D.T D - outer(D.T 1,D.T 1)/n.
        d = np.concatenate((dx, dv))
        prior_H = -np.outer(d, d)/n
        j = np.arange(n)
        prior_H[j,j] += dx*dx + h*(1/s - 2*x*x/(s*s))
        prior_H[n+j,n+j] += dv*dv + h*(1/s - 2*v*v/(s*s))
        cross = dx*dv - 2*h*x*v/(s*s)
        prior_H[j,n+j] += cross
        prior_H[n+j,j] += cross
        H += self.strength**2*prior_H
        return (H+H.T)*0.5


def _factor_hessian(H):
    try:
        factor = linalg.cho_factor(H, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return None, 0.0
    rcond, info = dpocon(factor[0], float(linalg.norm(H, 1)), uplo="L")
    if info != 0 or not np.isfinite(rcond):
        return None, 0.0
    return factor, float(rcond)


def _newton_polish(problem, z, tolerance, max_steps):
    """Exact-Hessian Armijo refinement; failed iterations remain unsuccessful."""
    z = z.copy()
    accepted = 0
    for _ in range(max_steps):
        value, gradient = problem.value_gradient(z)
        if np.linalg.norm(gradient, np.inf)/problem.stationarity_scale <= tolerance:
            break
        H = problem.hessian(z)
        diagonal_scale = max(1.0, float(np.max(np.abs(np.diag(H)))))
        damping = 0.0
        direction = None
        for _ in range(16):
            candidate = H.copy()
            candidate.flat[::len(z)+1] += damping
            try:
                factor = linalg.cho_factor(candidate, lower=True, check_finite=False)
                direction = linalg.cho_solve(factor, -gradient, check_finite=False)
                if np.dot(direction, gradient) < 0:
                    break
            except linalg.LinAlgError:
                pass
            direction = None
            damping = (1e-8*diagonal_scale if damping == 0 else damping*10)
        if direction is None:
            break
        slope = float(np.dot(gradient, direction))
        step = 1.0
        for _ in range(35):
            trial = z + step*direction
            trial_value, trial_gradient = problem.value_gradient(trial)
            # At the arithmetic floor the objective can round upward while
            # an exact Newton step materially improves the directly computed
            # stationary equations.  Admit only an O(eps) objective change
            # together with a strict gradient decrease, then recheck the
            # original unshifted stationarity criterion on the next iteration.
            rounding_refinement = (
                abs(trial_value-value) <= 32*np.finfo(float).eps*max(1.0,abs(value))
                and linalg.norm(trial_gradient,np.inf) < .5*linalg.norm(gradient,np.inf)
            )
            if trial_value <= value + 1e-4*step*slope or rounding_refinement:
                z = trial
                accepted += 1
                break
            step *= 0.5
        else:
            break
    return z, accepted


def solve_amplitude(
    B, y, ridge, a, strength, initial=None, *, epsilon=1e-8,
    n_starts=3, random_seed=20261028, maxiter=500,
    stationarity_tol=1e-10, min_hessian_rcond=1e-12, newton_steps=25,
):
    """Solve a local amplitude profile with deterministic phase multistarts.

    ``B`` already includes fixed observation weights; ``ridge`` and
    ``strength`` are residual multipliers and stay fixed along an outer
    trajectory.  ``a`` must be strictly positive and may be multiplied by any
    common positive constant without changing the problem.  ``initial`` is
    optional complex b, never a truth-only initialization.  The default first
    start is a stable augmented complex SVD ridge fit; additional starts use
    the calibration amplitude shape and independent random phases.  All
    branches and failures are retained.  Success requires a strict stationary
    point and a positive definite, sufficiently conditioned *exact* Hessian.
    """
    B = np.asarray(B, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    a = np.asarray(a, dtype=float)
    if B.ndim != 2 or y.shape != (B.shape[0],) or a.shape != (B.shape[1],):
        raise ValueError("B must be (m,n), y (m,), and a (n,)")
    if not B.shape[0] or not B.shape[1]:
        raise ValueError("empty inner problem")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(y))
            and np.all(np.isfinite(a)) and np.all(a > 0)):
        raise ValueError("finite B/y and strictly positive finite a required")
    if (not np.isfinite(ridge) or ridge < 0 or not np.isfinite(strength)
            or strength < 0 or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("nonnegative fixed penalties and positive epsilon required")
    if n_starts < 1 or maxiter < 0 or newton_steps < 0:
        raise ValueError("invalid solve budget")
    n = B.shape[1]
    if initial is None:
        augmented = np.vstack((B, ridge*np.eye(n)))
        rhs = np.concatenate((y, np.zeros(n, dtype=complex)))
        initial = linalg.lstsq(augmented, rhs, lapack_driver="gelsd")[0]
        del augmented, rhs
    else:
        initial = np.asarray(initial, dtype=complex)
        if initial.shape != (n,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial must be a finite complex (n,) vector")
    problem = _AmplitudeObjective(B, y, ridge, a, strength, epsilon)
    rng = np.random.default_rng(random_seed)
    # The arithmetic normalization is only for starts; the actual prior has
    # its unknown common log scale projected out.
    start_scale = max(float(linalg.norm(initial)), epsilon*np.sqrt(n))/linalg.norm(a)
    starts = [initial]
    starts.extend(start_scale*a*np.exp(1j*rng.uniform(-np.pi,np.pi,n))
                  for _ in range(n_starts-1))
    records, solutions = [], []
    for index, start in enumerate(starts):
        result = optimize.minimize(
            problem.value_gradient, _stack(start), method="L-BFGS-B", jac=True,
            options={"maxiter": maxiter, "gtol": stationarity_tol*problem.stationarity_scale,
                     "ftol": 1e-15, "maxls": 40, "maxcor": 20},
        )
        z, steps = _newton_polish(problem, result.x, stationarity_tol, newton_steps)
        value, gradient = problem.value_gradient(z)
        stationarity = float(linalg.norm(gradient, np.inf)/problem.stationarity_scale)
        H = problem.hessian(z)
        factor, rcond = _factor_hessian(H)
        success = bool(np.isfinite(value) and stationarity <= stationarity_tol
                       and factor is not None and rcond >= min_hessian_rcond)
        records.append(AmplitudeStart(
            index, value, stationarity, success, bool(result.success), str(result.message),
            int(result.nfev), int(result.njev), steps, rcond, factor is not None,
        ))
        solutions.append(z)
    # Prefer a converged local minimum; if none passed, return the best failed
    # branch with success=False so callers cannot quietly profile through it.
    eligible = [j for j,r in enumerate(records) if r.converged]
    chosen = min(eligible or range(len(records)), key=lambda j: records[j].objective)
    record = records[chosen]
    z = solutions[chosen]
    H = problem.hessian(z)
    factor, _ = _factor_hessian(H)
    residual = problem.residual(z)
    m = B.shape[0]
    start_coefficients = np.column_stack([_complex(s) for s in solutions])
    b = _complex(z)
    spread = max((float(linalg.norm(start_coefficients[:,j]-b)
                        /max(linalg.norm(b),epsilon*np.sqrt(n)))
                  for j in eligible),default=0.0)
    failed_lower = any(not r.converged and r.objective < record.objective
                       - 1e-10*max(1.0,abs(record.objective)) for r in records)
    return AmplitudeSolution(
        b=b, residual=residual,
        data_residual=B@b-y,
        ridge_residual=residual[2*m:2*m+2*n],
        amplitude_residual=residual[2*m+2*n:], objective=record.objective,
        stationarity=record.stationarity, success=record.converged,
        chosen_start=chosen, starts=tuple(records),
        start_coefficients=start_coefficients,
        max_converged_relative_b_spread=spread,
        has_failed_lower_objective_start=failed_lower,epsilon=float(epsilon),
        ridge=float(ridge), strength=float(strength), relative_amplitudes=a.copy(),
        hessian_rcond=record.hessian_rcond,
        hessian_positive_definite=record.hessian_positive_definite,
        hessian=H, _B=B, _y=y, _factor=factor,
    )


def implicit_derivative(
    solution: AmplitudeSolution, dB: Iterable[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return db/dtheta (n,p) and full real profile Jacobian (rows,p).

    Each ``dB[j]`` is with respect to one real outer parameter.  An iterable
    allows callers to generate one tall derivative matrix at a time.  No
    derivative is returned for an unconverged, singular or indefinite branch.
    The fixed RHS, ridge, a, strength and epsilon do not depend on theta.
    """
    if not solution.success or solution._factor is None:
        raise RuntimeError("amplitude branch failed stationary/curvature gate")
    if isinstance(dB, np.ndarray) and dB.ndim == 2:
        dB = [dB]
    B, b = solution._B, solution.b
    r = solution.data_residual
    z = _stack(b)
    n = len(b)
    s = abs(b)**2 + solution.epsilon**2
    dx, dv = b.real/s, b.imag/s
    db_columns, jac_columns = [], []
    for matrix in dB:
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.shape != B.shape or not np.all(np.isfinite(matrix)):
            raise ValueError("each dB must be finite and have the shape of B")
        partial_data = matrix@b
        partial_gradient = _stack(
            _adjoint_product(matrix,r) + _adjoint_product(B,partial_data)
        )
        dz = linalg.cho_solve(solution._factor,-partial_gradient,check_finite=False)
        db = _complex(dz)
        data_jac = partial_data+B@db
        amp_jac = dx*dz[:n]+dv*dz[n:]
        amp_jac -= amp_jac.mean()
        jac_columns.append(np.concatenate((
            _stack(data_jac),solution.ridge*dz,solution.strength*amp_jac
        )))
        db_columns.append(db)
    if not db_columns:
        return np.empty((n,0),complex),np.empty((len(solution.residual),0))
    return np.column_stack(db_columns),np.column_stack(jac_columns)
