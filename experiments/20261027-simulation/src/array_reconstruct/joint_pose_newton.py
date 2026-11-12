"""Exact linear-profile pose curvature and isolated bounded Newton polish."""
from __future__ import annotations

import time
import numpy as np
from scipy import linalg, optimize

from .joint_geometry import _real_array
from .joint_newton import NewtonResult, stationarity_metrics
from .joint_pose_kernel import scalar_pose_kernel, scalar_pose_kernel_second


class PoseNewtonResult(NewtonResult):
    @property
    def angles_deg(self):
        return self.evaluation.angles_deg.copy()

    @property
    def pose_u6(self):
        return self.x[:6].copy()


def profile_gradient_hessian(problem, x):
    """Dispatch to the explicit CPU/CUDA pose backend; never old 3-axis math."""
    if getattr(problem, "physical_parameter_count", None) != 6:
        raise TypeError("six-physical-coordinate pose backend required")
    return problem.profile_gradient_hessian(x)


def cpu_profile_gradient_hessian(problem, x):
    """Original exact envelope formula with all six physical/gain cross terms."""
    value = problem.evaluate(x)
    p = problem.parameter_count
    gradient, hessian = np.zeros(p), np.zeros((p, p))
    second_calls = first_calls = 0
    cache = {}
    options = {"q_array": problem.q_array, "q_probe": problem.q_probe, "sign": problem.sign}
    for group in problem.groups:
        b, db = value.b[group.name], value.db[group.name]
        m = sum(len(s.field) for s in group.sessions)
        r = np.concatenate([value.weighted_data_residuals[s.name] for s in group.sessions])
        rj = np.vstack([value.weighted_data_jacobians[s.name] for s in group.sessions])
        Q, adjoint, curvature = np.zeros((m, p), complex), np.zeros((len(b), p), complex), np.zeros((p, p), complex)
        blocks, cursor = [], 0
        for session in group.sessions:
            sl = slice(cursor, cursor+len(session.field)); cursor = sl.stop
            key = session.frequency_hz, session.samples_xy_m.shape, session.samples_xy_m.tobytes()
            if key not in cache:
                cache[key] = scalar_pose_kernel(problem.elements_m, session.samples_xy_m,
                    session.frequency_hz, value.geometry_mm, value.angles_deg, derivatives=True, **options)
                first_calls += 1
            K, dK = cache[key]
            weight = session.steering[None, :]*(session.weight*value.gains[session.name])[:, None]
            for j in range(6):
                Bj = dK[j]*weight
                Q[sl, j] = Bj@b
                adjoint[:, j] += Bj.conj().T@r[sl]
            gi = problem._gain_indices.get(session.name)
            if gi is not None:
                A = K*weight
                prediction = A@b
                Q[sl, gi], Q[sl, gi+1] = prediction, 1j*prediction
                adjoint[:, gi] += A.conj().T@r[sl]
                adjoint[:, gi+1] += -1j*(A.conj().T@r[sl])
                for axis in range(6):
                    z = np.vdot(r[sl], Q[sl, axis])
                    curvature[axis, gi] += z; curvature[gi, axis] += z
                    curvature[axis, gi+1] += 1j*z; curvature[gi+1, axis] += 1j*z
                z = np.vdot(r[sl], prediction)
                curvature[gi, gi] += z
                curvature[gi, gi+1] += 1j*z; curvature[gi+1, gi] += 1j*z
                curvature[gi+1, gi+1] -= z
            blocks.append((session, sl, key, weight))
        for j in range(6):
            for k in range(j, 6):
                second_cache, z = {}, 0j
                for session, sl, key, weight in blocks:
                    if key not in second_cache:
                        second_cache[key] = scalar_pose_kernel_second(problem.elements_m,
                            session.samples_xy_m, session.frequency_hz, value.geometry_mm, value.angles_deg,
                            j=j, k=k, **options)
                        second_calls += 1
                    z += np.vdot(r[sl], (second_cache[key]*weight)@b)
                curvature[j, k] = z; curvature[k, j] = z
        gradient -= (Q.conj().T@r).real
        hessian -= (Q.T@rj.conj()+curvature+adjoint.conj().T@db).real
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise FloatingPointError("nonfinite CPU pose envelope derivatives")
    asymmetry = float(linalg.norm(hessian-hessian.T)/max(linalg.norm(hessian), 1e-30))
    hessian = .5*(hessian+hessian.T)
    residual_gradient = value.jacobian.T@value.residual
    problem.counts["kernel_second_derivative_calls"] += second_calls
    return {"evaluation": value, "gradient": gradient, "hessian": hessian,
        "diagnostics": {"hessian_relative_asymmetry_before_symmetrizing": asymmetry,
            "envelope_gradient_relative_difference": float(linalg.norm(gradient-residual_gradient)/max(linalg.norm(gradient), 1e-30)),
            "second_kernel_calls": second_calls, "additional_first_kernel_calls": first_calls,
            "additional_first_kernel_calls_unit": "scalar_pose_kernel bundles, each K plus six normalized first derivatives",
            "hessian_eigenvalues": linalg.eigvalsh(hessian).tolist(),
            "derivative_scope": "six-physical-coordinate selected linear stationary branch; no global optimum certificate"}}


def outer_bounds(problem, gain_log_bounds=(-3., 3.)):
    if getattr(problem, "physical_parameter_count", None) != 6:
        raise TypeError("six-physical-coordinate pose backend required")
    lower, upper = np.full(problem.parameter_count, -np.inf), np.full(problem.parameter_count, np.inf)
    lower[:6], upper[:6] = -1., 1.
    bounds = _real_array(gain_log_bounds, "gain_log_bounds", (2,))
    if bounds[0] >= bounds[1]:
        raise ValueError("gain_log_bounds must be strictly increasing")
    for index in problem._gain_indices.values():
        lower[index], upper[index] = bounds
    return lower, upper


def physical_free_mask(problem, fixed_geometry):
    mask = np.full(6, fixed_geometry, dtype=bool) if np.ndim(fixed_geometry) == 0 else np.asarray(fixed_geometry, dtype=bool)
    if mask.shape != (6,):
        raise ValueError("fixed_geometry must be bool or a six-physical-axis mask")
    return np.r_[~mask, np.ones(problem.parameter_count-6, bool)]


def polish_profile(problem, start_parameters, maxiter=1500, *, fixed_geometry=False,
                   gtol=1e-7, xtol=1e-7, gain_log_bounds=(-3., 3.),
                   initial_tr_radius=1., initial_barrier_parameter=1e-8,
                   barrier_tol=1e-10, checkpoint_callback=None, bound_start_margin=1e-6):
    """Port of original trust-constr math; six physical bounds, explicit pose."""
    started, initial_counts = time.perf_counter(), dict(problem.counts)
    x0 = _real_array(start_parameters, "start_parameters", (problem.parameter_count,)).copy()
    problem._pose(x0); problem._gains(x0)
    lower, upper = outer_bounds(problem, gain_log_bounds)
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("starting pose/gains outside hard bounds")
    free = physical_free_mask(problem, fixed_geometry)
    if not np.isfinite(bound_start_margin) or bound_start_margin <= 0:
        raise ValueError("bound_start_margin must be finite and positive")
    requested_start = x0.copy()
    for index in np.flatnonzero(free):
        if np.isfinite(lower[index]) and x0[index]-lower[index] < bound_start_margin:
            x0[index] = lower[index]+bound_start_margin
        if np.isfinite(upper[index]) and upper[index]-x0[index] < bound_start_margin:
            x0[index] = upper[index]-bound_start_margin
    startup_nudge = x0-requested_start
    counts = {"profile_hessian_calls": 0, "second_kernel_calls": 0, "additional_first_kernel_calls": 0}
    cached_x = cached_curvature = None
    def expand(z):
        x = x0.copy(); x[free] = z
        return x
    def curvature(x):
        nonlocal cached_x, cached_curvature
        if cached_x is None or not np.array_equal(x, cached_x):
            cached_curvature = profile_gradient_hessian(problem, x)
            cached_x = x.copy(); counts["profile_hessian_calls"] += 1
            for key in ("second_kernel_calls", "additional_first_kernel_calls"):
                counts[key] += cached_curvature["diagnostics"][key]
        return cached_curvature
    def value_gradient(z):
        value = problem.evaluate(expand(z))
        return value.cost, (value.jacobian.T@value.residual)[free]
    def hessian(z):
        return curvature(expand(z))["hessian"][np.ix_(free, free)]
    iteration = 0
    def emit(event, x, cost, nfev, **extra):
        if checkpoint_callback is not None:
            G, angles = problem._pose(x)
            checkpoint_callback({"event": event, "method": "trust-constr-exact-pose-profile-Hessian",
                "x": x.tolist(), "geometry_mm": G.tolist(), "angles_deg": angles.tolist(), "pose_u6": x[:6].tolist(),
                "gains": {k: [float(v.real), float(v.imag)] for k, v in problem.gains_from_parameters(x).items()},
                "cost": float(cost), "nfev": int(nfev), "iteration": iteration,
                "counts": {**{k: problem.counts[k]-initial_counts[k] for k in problem.counts}, **counts},
                "elapsed_seconds": time.perf_counter()-started, **extra})
    initial = problem.evaluate(requested_start)
    emit("initial", requested_start, initial.cost, 1, requested_start_parameters=requested_start.tolist(),
         optimizer_start_parameters=x0.tolist(), startup_nudge=startup_nudge.tolist())
    if not np.any(free):
        final = curvature(x0)
        emit("final", x0, initial.cost, 1, success=True, status=1, message="all outer variables fixed")
        return PoseNewtonResult(x0, initial.geometry_mm.copy(), initial, True, 1, "all outer variables fixed", 0, 1, 1,
            counts["profile_hessian_calls"], 0., {"projected_gradient_inf": 0., "trf_scaled_gradient_inf": 0.},
            counts, time.perf_counter()-started, final["diagnostics"])
    def callback(z, state):
        nonlocal iteration
        iteration += 1
        try:
            emit("iteration", expand(z), state.fun, state.nfev, optimizer_optimality=float(state.optimality),
                 barrier_parameter=float(getattr(state, "barrier_parameter", 0.)))
        except StopIteration:
            return True
        return False
    result = optimize.minimize(value_gradient, x0[free], method="trust-constr", jac=True, hess=hessian,
        bounds=optimize.Bounds(lower[free], upper[free], keep_feasible=True), callback=callback,
        options={"maxiter": int(maxiter), "gtol": gtol, "xtol": xtol, "barrier_tol": barrier_tol,
            "initial_tr_radius": initial_tr_radius, "initial_barrier_parameter": initial_barrier_parameter,
            "initial_barrier_tolerance": initial_barrier_parameter, "verbose": 0})
    x = expand(result.x); final = curvature(x)
    metrics = stationarity_metrics(x[free], final["gradient"][free], lower[free], upper[free])
    totals = {**{k: problem.counts[k]-initial_counts[k] for k in problem.counts}, **counts}
    diagnostics = {**final["diagnostics"], "optimizer_barrier_parameter": float(getattr(result, "barrier_parameter", 0.)),
        "requested_start_parameters": requested_start.tolist(), "optimizer_start_parameters": x0.tolist(),
        "startup_nudge": startup_nudge.tolist(), "bound_start_margin": bound_start_margin,
        "startup_nudge_scope": "initialization only; model coordinates never clipped",
        "optimizer_constraint_violation": float(result.constr_violation), "optimizer_method": "trust-constr",
        "common_projected_stationarity_passed": metrics["projected_gradient_inf"] <= gtol,
        "gtol": gtol, "xtol": xtol, "barrier_tol": barrier_tol,
        "optimizer_convergence_is_not_physical_identifiability": True}
    emit("final", x, final["evaluation"].cost, result.nfev, success=bool(result.success),
         status=int(result.status), message=str(result.message))
    return PoseNewtonResult(x, final["evaluation"].geometry_mm.copy(), final["evaluation"], bool(result.success),
        int(result.status), str(result.message), int(result.nit), int(result.nfev), int(result.njev), int(result.nhev),
        float(result.optimality), metrics, totals, time.perf_counter()-started, diagnostics)
