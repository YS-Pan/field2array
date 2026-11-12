"""Optional exact-curvature profile polishing; the primary solver remains TRF.

The Hessian is local to the selected stationary response branch. It supplies
neither a global minimum certificate nor smoothness across amplitude branch
switches. All physical penalties, observation weights and bounds stay fixed.
This module has no file access or truth-dependent initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import linalg, optimize

from .joint_geometry import (_geometry, _real_array, parameters_to_geometry,
                             scalar_kernel)
from .physics import C0_M_S


def scalar_kernel_second(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                         q=2., sign=-1, j=0, k=0, chunk_size=512):
    """Second scalar-kernel derivative with respect to normalized t_j,t_k."""
    if j not in (0, 1, 2) or k not in (0, 1, 2):
        raise ValueError("second derivative indices must be 0, 1, or 2")
    out = scalar_kernel(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                        q, sign, chunk_size=chunk_size)
    elements = np.asarray(elements_m, dtype=float)
    samples = np.asarray(samples_xy_m, dtype=float)
    d, dx, dy = _geometry(geometry_mm)*.001
    wave = sign*2j*np.pi*float(frequency_hz)/C0_M_S
    for first in range(0, len(samples), int(chunk_size)):
        last = min(first+int(chunk_size), len(samples))
        x = samples[first:last, None, 0]+dx-elements[None, :, 0]
        y = samples[first:last, None, 1]+dy-elements[None, :, 1]
        radius = np.sqrt(d*d+x*x+y*y)
        vj = d if j == 0 else x if j == 1 else y
        vk = d if k == 0 else x if k == 1 else y
        beta = wave/radius-(q+1)/radius**2
        gamma = -wave/radius**3+2*(q+1)/radius**4
        lj = beta*vj+(q/d if j == 0 else 0)
        lk = beta*vk+(q/d if k == 0 else 0)
        ljk = gamma*vj*vk+(beta if j == k else 0)-(q/d**2 if j == k == 0 else 0)
        out[first:last] *= 1e-4*(lj*lk+ljk)
    return out


def profile_gradient_hessian(problem, x):
    """Exact local envelope gradient/Hessian, without second derivatives of b.

    For r=c-Bb and real outer parameters,
    F_j=-Re(r^H B_j b),
    F_jk=-Re(r_k^H B_j b+r^H B_jk b+r^H B_j b_k).
    Here B/r are weighted observation rows. Fixed quadratic and nonlinear
    amplitude priors affect b_k and r_k through the appropriate inner solve.
    No nonlinear amplitude residual is represented as a complex linear row.
    """
    backend_curvature=getattr(problem,"profile_gradient_hessian",None)
    if callable(backend_curvature):
        # A backend implements the same exact-curvature contract without a
        # device-to-CPU kernel fallback. Ordinary CPU problems take the path below.
        return backend_curvature(x)
    evaluation = problem.evaluate(x)
    p = problem.parameter_count
    gradient = np.zeros(p)
    hessian = np.zeros((p, p))
    second_calls = first_calls = 0
    for group in problem.groups:
        b, db = evaluation.b[group.name], evaluation.db[group.name]
        m = sum(len(session.field) for session in group.sessions)
        r = np.concatenate([evaluation.weighted_data_residuals[s.name] for s in group.sessions])
        rj = np.vstack([evaluation.weighted_data_jacobians[s.name] for s in group.sessions])
        Q = np.zeros((m, p), complex)
        adjoint = np.zeros((len(b), p), complex)
        curvature = np.zeros((p, p), complex)
        blocks, cursor = [], 0
        for session in group.sessions:
            stop = cursor+len(session.field)
            sl = slice(cursor, stop)
            key = session.frequency_hz, session.samples_xy_m.shape, session.samples_xy_m.tobytes()
            blocks.append((session, sl, key))
            cursor = stop
        for j in range(3):
            cache = {}
            for session, sl, key in blocks:
                if key not in cache:
                    cache[key] = scalar_kernel(problem.elements_m, session.samples_xy_m,
                        session.frequency_hz, evaluation.geometry_mm, problem.q, problem.sign, derivative=j)
                    first_calls += 1
                Bj = cache[key]*session.steering[None, :]*(session.weight*evaluation.gains[session.name])[:, None]
                Q[sl, j] = Bj@b
                adjoint[:, j] += (Bj.T@r[sl].conj()).conj()
        for session, sl, _ in blocks:
            index = problem._gain_indices.get(session.name)
            if index is None:
                continue
            prediction = session.weight*session.field-r[sl]
            Q[sl, index], Q[sl, index+1] = prediction, 1j*prediction
            # A_s^H r_s is obtained with a temporary un-differentiated kernel.
            K = scalar_kernel(problem.elements_m, session.samples_xy_m, session.frequency_hz,
                              evaluation.geometry_mm, problem.q, problem.sign)
            first_calls += 1
            A = K*session.steering[None, :]*(session.weight*evaluation.gains[session.name])[:, None]
            adjoint[:, index] += (A.T@r[sl].conj()).conj()
            adjoint[:, index+1] += -1j*adjoint[:, index]
            for axis in range(3):
                value = np.vdot(r[sl], Q[sl, axis])
                curvature[axis, index] += value
                curvature[index, axis] += value
                curvature[axis, index+1] += 1j*value
                curvature[index+1, axis] += 1j*value
            value = np.vdot(r[sl], prediction)
            curvature[index, index] += value
            curvature[index, index+1] += 1j*value
            curvature[index+1, index] += 1j*value
            curvature[index+1, index+1] -= value
        for j in range(3):
            for k in range(j, 3):
                cache = {}
                value = 0j
                for session, sl, key in blocks:
                    if key not in cache:
                        cache[key] = scalar_kernel_second(problem.elements_m, session.samples_xy_m,
                            session.frequency_hz, evaluation.geometry_mm, problem.q, problem.sign, j, k)
                        second_calls += 1
                    product = session.weight*evaluation.gains[session.name]*(cache[key]@(session.steering*b))
                    value += np.vdot(r[sl], product)
                curvature[j, k] = value
                curvature[k, j] = value
        gradient -= (Q.conj().T@r).real
        hessian -= (Q.T@rj.conj()+curvature+adjoint.conj().T@db).real
    asymmetry = float(linalg.norm(hessian-hessian.T)/max(linalg.norm(hessian), 1e-30))
    hessian = (hessian+hessian.T)*.5
    residual_gradient = evaluation.jacobian.T@evaluation.residual
    return {"evaluation": evaluation, "gradient": gradient, "hessian": hessian,
        "diagnostics": {"hessian_relative_asymmetry_before_symmetrizing": asymmetry,
            "envelope_gradient_relative_difference": float(linalg.norm(gradient-residual_gradient)/max(linalg.norm(gradient), 1e-30)),
            "second_kernel_calls": second_calls, "additional_first_kernel_calls": first_calls,
            "hessian_eigenvalues": linalg.eigvalsh(hessian).tolist(),
            "derivative_scope": "selected stationary branch only; global profile smoothness and optimality not established"}}


def outer_bounds(problem, gain_log_bounds=(-3., 3.)):
    lower, upper = np.full(problem.parameter_count, -np.inf), np.full(problem.parameter_count, np.inf)
    lower[:3], upper[:3] = -1., 1.
    log_bounds = _real_array(gain_log_bounds, "gain_log_bounds", (2,))
    if log_bounds[0] >= log_bounds[1]:
        raise ValueError("gain_log_bounds must be strictly increasing")
    for index in problem._gain_indices.values():
        lower[index], upper[index] = log_bounds
    return lower, upper


def stationarity_metrics(x, gradient, lower, upper):
    """Common box-projected gradient plus SciPy TRF's box-scaled gradient."""
    projected = x-np.clip(x-gradient, lower, upper)
    scale = np.ones_like(gradient)
    positive = (gradient > 0)&np.isfinite(lower)
    negative = (gradient < 0)&np.isfinite(upper)
    scale[positive] = x[positive]-lower[positive]
    scale[negative] = upper[negative]-x[negative]
    return {"projected_gradient_inf": float(linalg.norm(projected, np.inf)),
            "trf_scaled_gradient_inf": float(linalg.norm(gradient*scale, np.inf))}


@dataclass
class NewtonResult:
    x: np.ndarray
    geometry_mm: np.ndarray
    evaluation: object
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int
    nhev: int
    optimality: float
    stationarity: dict
    counts: dict
    elapsed_seconds: float
    diagnostics: dict

    @property
    def cost(self):
        return self.evaluation.cost

    @property
    def b(self):
        return self.evaluation.b

    @property
    def gains(self):
        return self.evaluation.gains


def polish_profile(problem, start_parameters, maxiter=1500, *, fixed_geometry=False,
                   gtol=1e-7, xtol=1e-7, gain_log_bounds=(-3., 3.),
                   initial_tr_radius=1., initial_barrier_parameter=1e-8,
                   barrier_tol=1e-10, checkpoint_callback=None, bound_start_margin=1e-6):
    """Bounded trust-constr polishing with the exact selected-branch Hessian.

    SciPy optimizer convergence is retained separately from the common box KKT
    diagnostics. A small explicit barrier starts well below normalized field
    objectives; its actual final value is reported. Current TRF runs are not
    modified. All starting parameters must come from public training fits.
    Exact-bound free starting coordinates are explicitly nudged inward by at
    most bound_start_margin to avoid trust-constr's keep_feasible lock. Both
    starts and the nudge are reported. Model evaluations are never clipped.
    """
    started = time.perf_counter()
    initial_counts = dict(problem.counts)
    x0 = _real_array(start_parameters, "start_parameters", (problem.parameter_count,))
    parameters_to_geometry(x0[:3])
    lower, upper = outer_bounds(problem, gain_log_bounds)
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("starting parameters outside existing hard bounds")
    mask = np.full(3, fixed_geometry, dtype=bool) if np.ndim(fixed_geometry) == 0 else np.asarray(fixed_geometry, dtype=bool)
    if mask.shape != (3,):
        raise ValueError("fixed_geometry must be bool or a three-axis mask")
    free = np.r_[~mask, np.ones(problem.parameter_count-3, bool)]
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
        x = x0.copy()
        x[free] = z
        return x
    def curvature(x):
        nonlocal cached_x, cached_curvature
        if cached_x is None or not np.array_equal(x, cached_x):
            cached_curvature = profile_gradient_hessian(problem, x)
            cached_x = x.copy()
            counts["profile_hessian_calls"] += 1
            for key in ("second_kernel_calls", "additional_first_kernel_calls"):
                counts[key] += cached_curvature["diagnostics"][key]
        return cached_curvature
    def value_gradient(z):
        evaluation = problem.evaluate(expand(z))
        return evaluation.cost, (evaluation.jacobian.T@evaluation.residual)[free]
    def hessian(z):
        return curvature(expand(z))["hessian"][np.ix_(free, free)]
    iteration = 0
    def emit(event, x, cost, nfev, **extra):
        if checkpoint_callback is not None:
            checkpoint_callback({"event": event, "method": "trust-constr-exact-profile-Hessian",
                "x": x.tolist(), "geometry_mm": parameters_to_geometry(x[:3]).tolist(),
                "gains": {k: [float(v.real),float(v.imag)] for k,v in problem.gains_from_parameters(x).items()},
                "cost": float(cost), "nfev": int(nfev), "iteration": iteration,
                "counts": {**{k: problem.counts[k]-initial_counts[k] for k in problem.counts}, **counts},
                "elapsed_seconds": time.perf_counter()-started, **extra})
    initial = problem.evaluate(requested_start)
    emit("initial", requested_start, initial.cost, 1,
         optimizer_start_parameters=x0.tolist(),startup_nudge=startup_nudge.tolist())
    if not np.any(free):
        final = curvature(x0)
        emit("final", x0, initial.cost, 1,success=True,status=1,message="all outer variables fixed")
        return NewtonResult(x0, initial.geometry_mm, initial, True, 1, "all outer variables fixed", 0, 1, 1,
            counts["profile_hessian_calls"], 0., {"projected_gradient_inf":0.,"trf_scaled_gradient_inf":0.},
            counts, time.perf_counter()-started, final["diagnostics"])
    def callback(z, state):
        nonlocal iteration
        iteration += 1
        try:
            emit("iteration", expand(z), state.fun, state.nfev,
                 optimizer_optimality=float(state.optimality), barrier_parameter=float(getattr(state,"barrier_parameter",0.)))
        except StopIteration:
            return True
        return False
    result = optimize.minimize(value_gradient, x0[free], method="trust-constr", jac=True,
        hess=hessian, bounds=optimize.Bounds(lower[free], upper[free], keep_feasible=True),
        callback=callback, options={"maxiter":int(maxiter),"gtol":gtol,"xtol":xtol,"barrier_tol":barrier_tol,
            "initial_tr_radius":initial_tr_radius,"initial_barrier_parameter":initial_barrier_parameter,
            "initial_barrier_tolerance":initial_barrier_parameter,"verbose":0})
    x = expand(result.x)
    final = curvature(x)
    metrics = stationarity_metrics(x[free], final["gradient"][free], lower[free], upper[free])
    totals = {**{k:problem.counts[k]-initial_counts[k] for k in problem.counts}, **counts}
    diagnostics = {**final["diagnostics"], "optimizer_barrier_parameter":float(getattr(result,"barrier_parameter",0.)),
        "requested_start_parameters":requested_start.tolist(),"optimizer_start_parameters":x0.tolist(),
        "startup_nudge":startup_nudge.tolist(),"bound_start_margin":bound_start_margin,
        "startup_nudge_scope":"optimization initialization only, to avoid exact-bound keep_feasible lock; model calls are never clipped",
        "optimizer_constraint_violation":float(result.constr_violation),
        "optimizer_method":"trust-constr", "common_projected_stationarity_passed":metrics["projected_gradient_inf"]<=gtol,
        "gtol":gtol,"xtol":xtol,"barrier_tol":barrier_tol,
        "optimizer_convergence_is_not_physical_identifiability":True}
    emit("final",x,final["evaluation"].cost,result.nfev,success=bool(result.success),status=int(result.status),message=str(result.message))
    return NewtonResult(x,final["evaluation"].geometry_mm.copy(),final["evaluation"],bool(result.success),int(result.status),
        str(result.message),int(result.nit),int(result.nfev),int(result.njev),int(result.nhev),float(result.optimality),
        metrics,totals,time.perf_counter()-started,diagnostics)
