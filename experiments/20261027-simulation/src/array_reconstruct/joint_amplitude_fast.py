"""Newton-first solver for the unchanged relative-amplitude inner problem.

The physical objective, coefficient stacking, residual order, solution data
class and original-coordinate implicit derivative come from
``array_reconstruct.joint_amplitude``.  This module is therefore an additional
solver for that objective, rather than a second definition of its prior.

For a stable Cholesky factor ``H0 = L @ L.T`` of the data-plus-ridge Hessian,
the L-BFGS fallback uses ``z = z_reference + solve(L.T, w)``.  Every objective
and gradient is still evaluated in physical ``z`` coordinates, including all
data-dependent constants and the full smoothed log-amplitude penalty.  Its
gradient is transformed by solving ``L @ gradient_w = gradient_z``.  Whitening
does not alter an undamped exact Newton step, or certify global optimality.

Only the original, undamped full Hessian enters the final success gate and
implicit derivative.  In particular, the implicit derivative does not
differentiate an optimizer coordinate transform or its Cholesky factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import linalg, optimize
from scipy.linalg.lapack import dpocon

from .joint_amplitude import (
    AmplitudeSolution,
    AmplitudeStart,
    _AmplitudeObjective,
    _complex,
    _stack,
    implicit_derivative,
)

__all__ = ["AmplitudeSolution", "AmplitudeStart", "solve_amplitude_fast",
           "implicit_derivative"]


_COUNT_KEYS = (
    "objective_evaluations", "gradient_evaluations", "hessian_evaluations",
    "residual_evaluations", "cholesky_factorization_attempts",
    "cholesky_factorization_successes", "rcond_estimation_calls",
    "cholesky_solve_calls", "triangular_solve_calls",
    "initial_svd_least_squares_calls", "quadratic_hessian_constructions",
    "quadratic_gram_matrix_products", "quadratic_factorization_attempts",
    "newton_factorization_attempts", "gate_factorization_attempts",
    "newton_iterations", "newton_direction_attempts", "newton_accepted_steps",
    "newton_rejected_directions", "line_search_trials",
    "line_search_nonfinite_trials", "armijo_acceptances",
    "rounding_refinement_acceptances", "lbfgs_objective_calls",
    "whitening_forward_products",
)


def _counts():
    return dict.fromkeys(_COUNT_KEYS, 0)


@dataclass
class _State:
    z: np.ndarray
    value: float
    gradient: np.ndarray


@dataclass
class _Gate:
    z: np.ndarray
    hessian: np.ndarray
    factor: tuple[np.ndarray, bool] | None
    rcond: float
    stationarity: float
    success: bool


class _TrackedObjective:
    """Count physical evaluations, including Newton and rejected trials."""

    def __init__(self, problem, counts):
        self.problem, self.counts = problem, counts

    def evaluate(self, z):
        self.counts["objective_evaluations"] += 1
        self.counts["gradient_evaluations"] += 1
        value, gradient = self.problem.value_gradient(z)
        return _State(np.asarray(z, dtype=float).copy(), float(value), gradient)

    def hessian(self, z):
        self.counts["hessian_evaluations"] += 1
        return self.problem.hessian(z)

    def residual(self, z):
        self.counts["residual_evaluations"] += 1
        return self.problem.residual(z)


def _stationarity(state, scale):
    if not np.all(np.isfinite(state.gradient)):
        return float("inf")
    return float(linalg.norm(state.gradient, np.inf)/scale)


def _finite_state(state):
    return bool(np.isfinite(state.value) and np.all(np.isfinite(state.z))
                and np.all(np.isfinite(state.gradient)))


def _factor_and_rcond(H, counts, kind, *, clean_factor=False):
    """Use the original Cholesky/dpocon gate, with actual call accounting."""
    if not np.all(np.isfinite(H)):
        return None, 0.0, "nonfinite_hessian"
    counts["cholesky_factorization_attempts"] += 1
    counts[kind + "_factorization_attempts"] += 1
    try:
        if clean_factor:
            factor = (linalg.cholesky(H, lower=True, check_finite=False), True)
        else:
            factor = linalg.cho_factor(H, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return None, 0.0, "cholesky_failed"
    counts["cholesky_factorization_successes"] += 1
    counts["rcond_estimation_calls"] += 1
    rcond, info = dpocon(factor[0], float(linalg.norm(H, 1)), uplo="L")
    if info != 0 or not np.isfinite(rcond):
        return None, 0.0, "rcond_estimation_failed"
    return factor, float(rcond), "available"


def _original_gate(tracked, state, tolerance, min_rcond):
    H = tracked.hessian(state.z)
    factor, rcond, _ = _factor_and_rcond(H, tracked.counts, "gate")
    stationarity = _stationarity(state, tracked.problem.stationarity_scale)
    success = bool(_finite_state(state) and stationarity <= tolerance
                   and factor is not None and rcond >= min_rcond)
    return _Gate(state.z, H, factor, rcond, stationarity, success)


def _newton_stage(tracked, state, tolerance, budget, damping_metric,
                  max_damping_attempts, max_line_search):
    """Exact full-Hessian directions; damping affects directions only."""
    counts = tracked.counts
    before_iterations = counts["newton_iterations"]
    before_steps = counts["newton_accepted_steps"]
    before_trials = counts["line_search_trials"]
    reason = "iteration_limit"
    damping_values = []
    for _ in range(budget):
        if not _finite_state(state):
            reason = "nonfinite_value_or_gradient"
            break
        if _stationarity(state, tracked.problem.stationarity_scale) <= tolerance:
            reason = "stationarity_satisfied"
            break
        counts["newton_iterations"] += 1
        H = tracked.hessian(state.z)
        if not np.all(np.isfinite(H)):
            reason = "nonfinite_hessian"
            break
        hscale = max(1.0, float(np.max(np.abs(np.diag(H)))))
        if damping_metric is None:
            # A singular/unstable H0 cannot serve as a reliable damping
            # metric.  This explicitly recorded fallback uses scaled I.
            first_damping = 1e-8*hscale
        else:
            mscale = max(np.finfo(float).tiny,
                         float(np.max(np.abs(np.diag(damping_metric)))))
            first_damping = 1e-8*max(1.0, hscale/mscale)
        damping, direction = 0.0, None
        for _ in range(max_damping_attempts):
            counts["newton_direction_attempts"] += 1
            if damping == 0:
                candidate = H
            elif damping_metric is None:
                candidate = H.copy()
                candidate.flat[::len(state.z)+1] += damping
            else:
                candidate = H + damping*damping_metric
            counts["cholesky_factorization_attempts"] += 1
            counts["newton_factorization_attempts"] += 1
            try:
                factor = linalg.cho_factor(candidate, lower=True,
                                           check_finite=False)
                counts["cholesky_factorization_successes"] += 1
                counts["cholesky_solve_calls"] += 1
                proposed = linalg.cho_solve(factor, -state.gradient,
                                           check_finite=False)
                slope = float(np.dot(proposed, state.gradient))
                if np.all(np.isfinite(proposed)) and np.isfinite(slope) and slope < 0:
                    direction = proposed
                    break
            except linalg.LinAlgError:
                pass
            counts["newton_rejected_directions"] += 1
            damping = first_damping if damping == 0 else damping*10.0
        if direction is None:
            reason = "no_descent_direction"
            break
        step = 1.0
        gradient_norm = linalg.norm(state.gradient, np.inf)
        for _ in range(max_line_search):
            counts["line_search_trials"] += 1
            trial_z = state.z + step*direction
            if not np.all(np.isfinite(trial_z)):
                counts["line_search_nonfinite_trials"] += 1
                step *= 0.5
                continue
            trial = tracked.evaluate(trial_z)
            if not _finite_state(trial):
                counts["line_search_nonfinite_trials"] += 1
                step *= 0.5
                continue
            armijo = trial.value <= state.value + 1e-4*step*slope
            rounding_refinement = (
                abs(trial.value-state.value)
                <= 32*np.finfo(float).eps*max(1.0, abs(state.value))
                and linalg.norm(trial.gradient, np.inf) < 0.5*gradient_norm
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
    if _finite_state(state) and (
        _stationarity(state, tracked.problem.stationarity_scale) <= tolerance
    ):
        reason = "stationarity_satisfied"
    elif budget == 0:
        reason = "zero_budget"
    return state, {
        "requested_iterations": budget,
        "actual_iterations": counts["newton_iterations"]-before_iterations,
        "accepted_steps": counts["newton_accepted_steps"]-before_steps,
        "line_search_trials": counts["line_search_trials"]-before_trials,
        "accepted_direction_damping": damping_values,
        "termination": reason,
    }


def _lbfgs_stage(tracked, state, reference, lower, tolerance, maxiter,
                 max_line_search):
    """Optimize exactly the physical objective in optional affine coordinates."""
    counts = tracked.counts
    summary = {
        "used": maxiter > 0, "whitened": lower is not None,
        "optimizer_success": False, "optimizer_message": "zero L-BFGS budget",
        "optimizer_iterations": 0, "optimizer_nfev": 0, "optimizer_njev": 0,
        "exception_type": None, "original_stationarity_after": None,
        "maxiter": maxiter, "maxfun": (maxiter+1)*max_line_search+1,
        "maxls": max_line_search,
    }
    if maxiter == 0:
        summary["original_stationarity_after"] = _stationarity(
            state, tracked.problem.stationarity_scale)
        return state, summary

    if lower is None:
        coordinate = state.z.copy()
        coordinate_gtol = tolerance*tracked.problem.stationarity_scale
    else:
        counts["whitening_forward_products"] += 1
        coordinate = lower.T @ (state.z-reference)
        # ||gradient_z||_inf <= ||L||_inf ||gradient_w||_inf.
        coordinate_gtol = tolerance*tracked.problem.stationarity_scale/max(
            1.0, float(linalg.norm(lower, np.inf)))
    last_state = None

    def evaluate(coordinate_value):
        nonlocal last_state
        counts["lbfgs_objective_calls"] += 1
        if lower is None:
            z = coordinate_value
        else:
            counts["triangular_solve_calls"] += 1
            z = reference + linalg.solve_triangular(
                lower, coordinate_value, lower=True, trans="T", check_finite=False)
        last_state = tracked.evaluate(z)
        if lower is None:
            gradient = last_state.gradient
        else:
            counts["triangular_solve_calls"] += 1
            gradient = linalg.solve_triangular(
                lower, last_state.gradient, lower=True, check_finite=False)
        return last_state.value, gradient

    try:
        result = optimize.minimize(
            evaluate, coordinate, method="L-BFGS-B", jac=True,
            options={"maxiter": maxiter, "gtol": coordinate_gtol,
                     "maxfun": summary["maxfun"], "ftol": 1e-15,
                     "maxls": max_line_search, "maxcor": 20},
        )
        summary.update(
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            optimizer_iterations=int(result.nit),
            optimizer_nfev=int(result.nfev), optimizer_njev=int(result.njev),
        )
        if lower is None:
            final_z = np.asarray(result.x, dtype=float)
        else:
            counts["triangular_solve_calls"] += 1
            final_z = reference + linalg.solve_triangular(
                lower, result.x, lower=True, trans="T", check_finite=False)
        if final_z.shape == state.z.shape and np.all(np.isfinite(final_z)):
            if last_state is not None and np.array_equal(final_z, last_state.z):
                state = last_state
            else:
                state = tracked.evaluate(final_z)
        else:
            summary["optimizer_success"] = False
            summary["optimizer_message"] += "; nonfinite or invalid final coefficients"
    except (ValueError, FloatingPointError, OverflowError, linalg.LinAlgError) as exc:
        # Keep the pre-fallback branch on a numerical optimizer failure;
        # neither a failed line-search trial nor an exception becomes success.
        summary["exception_type"] = type(exc).__name__
        summary["optimizer_message"] = str(exc)
    summary["original_stationarity_after"] = _stationarity(
        state, tracked.problem.stationarity_scale)
    return state, summary


def _integer_budget(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return int(value)


def solve_amplitude_fast(
    B, y, ridge, a, strength, initial=None, *, epsilon=1e-8,
    n_starts=3, random_seed=20261028, maxiter=5000,
    stationarity_tol=1e-12, min_hessian_rcond=1e-12, newton_steps=250,
    newton_first_steps=None, whitening_min_rcond=None,
    max_damping_attempts=16, max_line_search=35, lbfgs_max_line_search=40,
):
    """Solve the original amplitude objective with deterministic multistarts.

    Defaults retain the augmented-SVD first start and two independent seeded
    phase starts of the original solver.  An initial Newton stage receives
    four fifths of at most 250 total Newton iterations; L-BFGS, when needed,
    receives at most 5000 iterations; the remaining Newton budget polishes
    its result, including unused initial-stage iterations.  Successful
    convergence may use less than any budget.

    The return value is the original ``AmplitudeSolution``.  Its
    ``fast_diagnostics`` dictionary additionally records shared setup work,
    every start/stage, actual evaluation/factorization/line-search counts,
    whitening guards, and optimizer exit status separately from the strict
    original-coordinate stationarity/SPD/rcond success gate.  ``nfev`` and
    ``njev`` in each ``AmplitudeStart`` include *all* physical evaluations,
    including Newton line searches and final checks, not only L-BFGS calls.
    ``start_coefficients`` retains each branch's final physical coefficients,
    including failed branches, as in the original solver.
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
    if (not np.isfinite(stationarity_tol) or not 0 < stationarity_tol <= 1e-8):
        raise ValueError("stationarity_tol must be positive and at most 1e-8")
    if not np.isfinite(min_hessian_rcond) or not 0 < min_hessian_rcond <= 1:
        raise ValueError("min_hessian_rcond must be in (0,1]")
    if whitening_min_rcond is None:
        whitening_min_rcond = max(min_hessian_rcond, np.finfo(float).eps)
    if (not np.isfinite(whitening_min_rcond)
            or not 0 < whitening_min_rcond <= 1):
        raise ValueError("whitening_min_rcond must be in (0,1]")
    n_starts = _integer_budget(n_starts, "n_starts")
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    maxiter = _integer_budget(maxiter, "maxiter", 5000)
    newton_steps = _integer_budget(newton_steps, "newton_steps", 250)
    if newton_first_steps is None:
        newton_first_steps = (4*newton_steps)//5
    newton_first_steps = _integer_budget(
        newton_first_steps, "newton_first_steps", newton_steps)
    max_damping_attempts = _integer_budget(max_damping_attempts, "max_damping_attempts")
    max_line_search = _integer_budget(max_line_search, "max_line_search")
    lbfgs_max_line_search = _integer_budget(lbfgs_max_line_search, "lbfgs_max_line_search")
    if min(max_damping_attempts, max_line_search, lbfgs_max_line_search) < 1:
        raise ValueError("damping and line-search attempt limits must be positive")

    shared_counts = _counts()
    n = B.shape[1]
    if initial is None:
        augmented = np.vstack((B, ridge*np.eye(n)))
        rhs = np.concatenate((y, np.zeros(n, dtype=complex)))
        shared_counts["initial_svd_least_squares_calls"] += 1
        initial = linalg.lstsq(augmented, rhs, lapack_driver="gelsd")[0]
        del augmented, rhs
        initial_source = "augmented_complex_svd_ridge_fit"
    else:
        initial = np.asarray(initial, dtype=complex)
        if initial.shape != (n,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial must be a finite complex (n,) vector")
        initial_source = "caller_provided_physical_coefficients"
    problem = _AmplitudeObjective(B, y, ridge, a, strength, epsilon)
    shared_counts["quadratic_hessian_constructions"] += 1
    shared_counts["quadratic_gram_matrix_products"] += 1
    H0 = problem.quadratic_hessian
    h0_factor, h0_rcond, whitening_reason = _factor_and_rcond(
        H0, shared_counts, "quadratic", clean_factor=True)
    if h0_factor is not None and h0_rcond < whitening_min_rcond:
        whitening_reason = "quadratic_rcond_below_whitening_threshold"
    whitening_available = bool(h0_factor is not None
                              and h0_rcond >= whitening_min_rcond)
    lower = h0_factor[0] if whitening_available else None
    damping_metric = H0 if whitening_available else None
    # H0 and this affine reference are optimizer aids only; every physical
    # evaluation and the final derivative remain in the original coordinates.
    reference = _stack(initial)
    rng = np.random.default_rng(random_seed)
    start_scale = max(float(linalg.norm(initial)), epsilon*np.sqrt(n))/linalg.norm(a)
    initial_starts = [initial]
    initial_starts.extend(start_scale*a*np.exp(1j*rng.uniform(-np.pi, np.pi, n))
                          for _ in range(n_starts-1))

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
            damping_metric, max_damping_attempts, max_line_search)
        gate = None
        if _stationarity(state, problem.stationarity_scale) <= stationarity_tol:
            gate = _original_gate(tracked, state, stationarity_tol, min_hessian_rcond)
        fallback_needed = gate is None or not gate.success
        final_budget = newton_steps-counts["newton_iterations"]
        if fallback_needed:
            state, fallback = _lbfgs_stage(
                tracked, state, reference, lower, stationarity_tol, maxiter,
                lbfgs_max_line_search)
            state, final_stage = _newton_stage(
                tracked, state, stationarity_tol, final_budget, damping_metric,
                max_damping_attempts, max_line_search)
        else:
            fallback = {
                "used": False, "whitened": lower is not None,
                "optimizer_success": None,
                "optimizer_message": "skipped: initial Newton stage passed original gate",
                "optimizer_iterations": 0, "optimizer_nfev": 0, "optimizer_njev": 0,
                "exception_type": None,
                "original_stationarity_after": gate.stationarity,
                "maxiter": maxiter, "maxfun": (maxiter+1)*lbfgs_max_line_search+1,
                "maxls": lbfgs_max_line_search,
            }
            final_stage = {
                "requested_iterations": final_budget,
                "actual_iterations": 0, "accepted_steps": 0,
                "line_search_trials": 0, "accepted_direction_damping": [],
                "termination": "skipped: original gate already passed",
            }
        if gate is None or not np.array_equal(gate.z, state.z):
            gate = _original_gate(tracked, state, stationarity_tol, min_hessian_rcond)
        optimizer_success = (bool(fallback["optimizer_success"])
                             if fallback_needed else first_stage["termination"]
                             == "stationarity_satisfied")
        optimizer_message = (str(fallback["optimizer_message"]) if fallback_needed
                             else "Newton-first original-coordinate stationarity satisfied")
        record = AmplitudeStart(
            index, state.value, gate.stationarity, gate.success, optimizer_success,
            optimizer_message, counts["objective_evaluations"],
            counts["gradient_evaluations"], counts["newton_accepted_steps"],
            gate.rcond, gate.factor is not None,
        )
        records.append(record)
        solutions.append(state.z)
        stage_diagnostics.append({
            "index": index, "initialization": initial_source if index == 0
            else "seeded_independent_calibration_shape_phases",
            "initial_objective": initial_value,
            "initial_stationarity": initial_stationarity,
            "initial_newton": first_stage, "fallback_needed": fallback_needed,
            "lbfgs": fallback, "final_newton": final_stage,
            "original_gate_success": gate.success,
            "original_hessian_positive_definite": gate.factor is not None,
            "original_hessian_rcond": gate.rcond,
            "global_optimality_certified": False,
        })
        candidate = (index, state, gate, tracked)
        if gate.success:
            if best_success is None or state.value < best_success[1].value:
                best_success = candidate
        elif best_failed is None or (
            np.isfinite(state.value) and (
                not np.isfinite(best_failed[1].value) or state.value < best_failed[1].value)
        ):
            best_failed = candidate

    chosen, state, gate, tracked = best_success if best_success is not None else best_failed
    record = records[chosen]
    residual = tracked.residual(state.z)
    m = B.shape[0]
    b = _complex(state.z)
    start_coefficients = np.column_stack([_complex(z) for z in solutions])
    eligible = [j for j, item in enumerate(records) if item.converged]
    spread = max((float(linalg.norm(start_coefficients[:, j]-b)
                        /max(linalg.norm(b), epsilon*np.sqrt(n)))
                  for j in eligible), default=0.0)
    failed_lower = any(
        not item.converged and item.objective < record.objective
        - 1e-10*max(1.0, abs(record.objective)) for item in records)
    solution = AmplitudeSolution(
        b=b, residual=residual, data_residual=_complex(residual[:2*m]),
        ridge_residual=residual[2*m:2*m+2*n],
        amplitude_residual=residual[2*m+2*n:], objective=record.objective,
        stationarity=record.stationarity, success=record.converged,
        chosen_start=chosen, starts=tuple(records), start_coefficients=start_coefficients,
        max_converged_relative_b_spread=spread,
        has_failed_lower_objective_start=failed_lower, epsilon=float(epsilon),
        ridge=float(ridge), strength=float(strength), relative_amplitudes=a.copy(),
        hessian_rcond=gate.rcond, hessian_positive_definite=gate.factor is not None,
        hessian=gate.hessian, _B=B, _y=y, _factor=gate.factor,
    )
    for diagnostic, counts in zip(stage_diagnostics, branch_counts):
        diagnostic["counts"] = counts.copy()
    total_counts = {
        key: shared_counts[key]+sum(counts[key] for counts in branch_counts)
        for key in _COUNT_KEYS
    }
    solution.fast_diagnostics = {
        "solver": "newton_first_exact_amplitude_with_optional_whitened_lbfgs",
        "objective_identity": "array_reconstruct.joint_amplitude._AmplitudeObjective",
        "source_dependencies": ["array_reconstruct/joint_amplitude_fast.py",
                                "array_reconstruct/joint_amplitude.py"],
        "implicit_derivative_identity": "array_reconstruct.joint_amplitude.implicit_derivative",
        "physical_objective_approximated": False,
        "original_coordinate_gate": True, "global_optimality_certified": False,
        "stationarity_tolerance": float(stationarity_tol),
        "min_hessian_rcond": float(min_hessian_rcond),
        "n_starts": n_starts, "random_seed": random_seed,
        "newton_iteration_budget_per_start": newton_steps,
        "initial_newton_iteration_budget_per_start": newton_first_steps,
        "reserved_final_newton_iterations_per_start": newton_steps-newton_first_steps,
        "final_newton_budget_policy": "total budget minus actual initial Newton iterations",
        "lbfgs_iteration_budget_per_start": maxiter,
        "whitening": {
            "available": whitening_available, "quadratic_hessian_rcond": h0_rcond,
            "min_rcond": float(whitening_min_rcond), "guard_status": whitening_reason,
            "reference": initial_source,
            "mapping": "z=reference+solve(L.T,w); gradient_w=solve(L,gradient_z)",
            "fallback_coordinates": "whitened" if whitening_available else "physical",
            "newton_damping_metric": "H0" if whitening_available else "scaled_identity",
            "preconditioner_differentiated_in_implicit_derivative": False,
            "purpose": "numerical coordinates and damping metric only; physical objective unchanged",
        },
        "shared_setup_counts": shared_counts.copy(), "starts": stage_diagnostics,
        "counts": total_counts,
        "count_semantics": {
            "objective_and_gradient": "each joint physical value_gradient call counts once in each",
            "hessian": "each full physical Hessian construction, including gate checks",
            "factorization": "actual SciPy Cholesky calls, including unsuccessful calls",
            "line_search_trials": "Newton trials; L-BFGS trial work is in lbfgs_objective_calls",
            "initial_svd": "one augmented complex gelsd solve when no initial is supplied",
            "implicit_derivative": "subsequent derivative calls are outside solve counters",
        },
    }
    return solution
