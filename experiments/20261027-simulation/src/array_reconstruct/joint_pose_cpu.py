"""Isolated six-physical-parameter, linear-profile CPU pose model.

Angles are a conditional scalar mechanical model, not measured polarization.
No files, truth, production orchestration, or GPU are accessed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import linalg, optimize

from .joint_geometry import (JointProblem, ProfileEvaluation, FitResult, _Factor,
                             _real_array, _complex_array)
from .joint_pose_kernel import (pose_to_parameters, parameters_to_pose,
                                scalar_pose_kernel)

PHYSICAL_PARAMETER_COUNT = 6
POSE_LAYOUT = ("t_d", "t_dx", "t_dy", "t_alpha", "t_beta", "t_gamma")


@dataclass
class PoseEvaluation(ProfileEvaluation):
    angles_deg: np.ndarray

    @property
    def pose_u6(self):
        return self.x[:6].copy()


@dataclass
class PoseFitResult(FitResult):
    angles_deg: np.ndarray

    @property
    def pose_u6(self):
        return self.x[:6].copy()


class PoseJointProblem(JointProblem):
    """Six shared physical coordinates followed by original complex gains.

    Only augmented-QR linear C0/complex-template inner solves are admitted.
    The registered centroid is immutable; no alternative pivot is accepted.
    """
    physical_parameter_count = PHYSICAL_PARAMETER_COUNT

    def __init__(self, elements_m, groups, *, q_array=2., q_probe=0., sign=-1,
                 backend="qr", normal_condition_max=1e8, normal_qr_check="first",
                 qr_coefficient_rtol=1e-6, qr_residual_rtol=1e-8):
        groups = tuple(groups)
        if any(g.amplitude_strength > 0 for g in groups):
            raise NotImplementedError("pose nonlinear amplitude inner solve is unqualified")
        if backend != "qr":
            raise ValueError("isolated CPU pose path requires augmented QR")
        for value in (q_array, q_probe):
            if isinstance(value, (bool, complex)) or not np.isfinite(value) or value < 0:
                raise ValueError("q_array/q_probe must be finite nonnegative real scalars")
        super().__init__(elements_m, groups, q=float(q_array)+float(q_probe), sign=sign,
                         backend=backend, normal_condition_max=normal_condition_max,
                         normal_qr_check=normal_qr_check,
                         qr_coefficient_rtol=qr_coefficient_rtol,
                         qr_residual_rtol=qr_residual_rtol)
        self.q_array, self.q_probe = float(q_array), float(q_probe)
        self.parameter_layout = list(POSE_LAYOUT)
        self._gain_indices = {}
        for group in self.groups:
            for session in group.sessions[1:]:
                self._gain_indices[session.name] = len(self.parameter_layout)
                self.parameter_layout.extend((f"rho:{group.name}:{session.name}",
                                              f"phi:{group.name}:{session.name}"))
        xyz = np.c_[self.elements_m, np.zeros(len(self.elements_m))] if self.elements_m.shape[1] == 2 else self.elements_m
        self.pivot_m = np.mean(xyz, axis=0)
        self.pivot_m.setflags(write=False)
        self.counts["kernel_second_derivative_calls"] = 0

    def _pose(self, x):
        return parameters_to_pose(x[:6])

    def initial_parameters(self, geometry_mm=(60., 0., 0.), gains=None, *, angles_deg=(0., 0., 0.)):
        x = np.zeros(self.parameter_count)
        x[:6] = pose_to_parameters(geometry_mm, angles_deg)
        for name, index in self._gain_indices.items():
            gain = complex((gains or {}).get(name, 1.))
            if not np.isfinite(gain) or abs(gain) == 0:
                raise ValueError("initial gains must be finite and nonzero")
            x[index:index+2] = np.log(abs(gain)), np.angle(gain)
        return x

    def gains_from_parameters(self, x):
        x = _real_array(x, "outer pose parameters", (self.parameter_count,))
        self._pose(x)
        return self._gains(x)

    def _kernel(self, session, geometry_mm, angles_deg, *, derivatives=False):
        return scalar_pose_kernel(self.elements_m, session.samples_xy_m,
            session.frequency_hz, geometry_mm, angles_deg, q_array=self.q_array,
            q_probe=self.q_probe, sign=self.sign, derivatives=derivatives)

    def predict(self, session, geometry_mm, b, g=1., *, angles_deg):
        """Explicit angles are mandatory, including for unseen observations."""
        b = _complex_array(b, "b", (len(self.elements_m),))
        gain = complex(g)
        if not np.isfinite(gain):
            raise ValueError("prediction gain must be finite")
        return gain*(self._kernel(session, geometry_mm, angles_deg)@(session.steering*b))

    def evaluate(self, x):
        x = _real_array(x, "outer pose parameters", (self.parameter_count,))
        if self._last is not None and np.array_equal(x, self._last.x):
            self.counts["cache_hits"] += 1
            return self._last
        geometry, angles = self._pose(x)
        gains = self._gains(x)
        self.counts["evaluations"] += 1
        try:
            value = self._evaluate_pose(x, geometry, angles, gains)
        except (ValueError, linalg.LinAlgError, FloatingPointError) as error:
            self._last = None
            self.counts["failed_evaluations"] += 1
            self.failure_ledger.append({"x": x.tolist(), "geometry_mm": geometry.tolist(),
                "angles_deg": angles.tolist(), "exception_type": type(error).__name__, "reason": str(error)})
            raise
        self._last = value
        return value

    def _evaluate_pose(self, x, geometry, angles, gains):
        p, n = self.parameter_count, len(self.elements_m)
        residuals, jacobians, bs, dbs, diagnostics = [], [], {}, {}, {}
        data, weighted, penalties = {}, {}, {}
        data_jacobians, weighted_jacobians, penalty_jacobians = {}, {}, {}
        kernels = {}
        for group in self.groups:
            m = sum(len(s.field) for s in group.sessions)
            penalty_n = n*(1+int(group.template_strength > 0))
            B, c = np.zeros((m+penalty_n, n), complex), np.zeros(m+penalty_n, complex)
            blocks, cursor = [], 0
            for session in group.sessions:
                key = session.frequency_hz, session.samples_xy_m.shape, session.samples_xy_m.tobytes()
                if key not in kernels:
                    kernels[key] = self._kernel(session, geometry, angles, derivatives=True)
                    self.counts["kernel_calls"] += 1
                    self.counts["kernel_derivative_calls"] += 6
                K, dK = kernels[key]
                sl = slice(cursor, cursor+len(session.field)); cursor = sl.stop
                B[sl] = K*session.steering[None, :]*(session.weight*gains[session.name])[:, None]
                c[sl] = session.weight*session.field
                blocks.append((session, sl, K, dK))
            B[m:m+n] = group.lambda_reg*np.eye(n)
            c[m:m+n] = group.lambda_reg*group.b0
            if group.template_strength > 0:
                a = group.template_prior
                B[m+n:] = group.template_strength*(np.eye(n)-np.outer(a, a.conj())/np.vdot(a, a).real)
            factor = _Factor(B, c, "qr", self.normal_condition_max, self.counts)
            b = factor.b
            r, J, db = c-B@b, np.zeros((len(c), p), complex), np.zeros((n, p), complex)
            for j in range(p):
                if j >= 6 and not any(self._gain_indices.get(s.name, -10) in (j, j-1) for s in group.sessions):
                    continue
                product, adjoint = np.zeros(len(c), complex), np.zeros(n, complex)
                for session, sl, K, dK in blocks:
                    gi = self._gain_indices.get(session.name, -10)
                    if j < 6:
                        Bj = dK[j]*session.steering[None, :]*(session.weight*gains[session.name])[:, None]
                    elif j == gi:
                        Bj = B[sl]
                    elif j == gi+1:
                        Bj = 1j*B[sl]
                    else:
                        continue
                    product[sl] = Bj@b
                    adjoint += Bj.conj().T@r[sl]
                db[:, j] = factor.derivative(product, adjoint)
                J[:, j] = -product-B@db[:, j]
                self.counts["linear_derivative_solves"] += 1
            stationarity = float(linalg.norm(B.conj().T@r)/max(linalg.norm(B, "fro")*linalg.norm(r),
                np.finfo(float).eps*linalg.norm(B, "fro")*linalg.norm(c), 1e-30))
            for session, sl, K, dK in blocks:
                A = K*session.steering[None, :]*gains[session.name]
                prediction = A@b
                unweighted_J = -A@db
                for j in range(6):
                    unweighted_J[:, j] -= gains[session.name]*(dK[j]@(session.steering*b))
                gi = self._gain_indices.get(session.name)
                if gi is not None:
                    unweighted_J[:, gi] -= prediction
                    unweighted_J[:, gi+1] -= 1j*prediction
                data[session.name], weighted[session.name] = session.field-prediction, r[sl].copy()
                data_jacobians[session.name], weighted_jacobians[session.name] = unweighted_J, J[sl].copy()
            penalties[group.name], penalty_jacobians[group.name] = r[m:].copy(), J[m:].copy()
            bs[group.name], dbs[group.name] = b.copy(), db
            diagnostics[group.name] = {"backend_requested": self.backend, "backend_used": "qr",
                "normal_condition_h_1": None, "normal_condition_max": self.normal_condition_max,
                "fallback_reason": None, "qr_verification": None, "stationarity": stationarity,
                "stationarity_normalization": "||B^H r|| / max(||B||_F ||r||, eps ||B||_F ||c||, 1e-30)",
                "lambda_reg": group.lambda_reg, "inner_solver": "complex linear augmented least squares",
                "amplitude": None, "penalty": "fixed physical lambda * (b-b0), L=I",
                "reference_session": group.sessions[0].name, "reference_gain": [1., 0.],
                "template_strength": group.template_strength,
                "template_penalty": "mu * (I-aa^H/(a^Ha)) b" if group.template_strength > 0 else None,
                "data_residual_norm": float(linalg.norm(r[:m])), "penalty_residual_norm": float(linalg.norm(r[m:]))}
            residuals.append(r); jacobians.append(J)
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        if not np.isfinite(r).all() or not np.isfinite(J).all():
            raise FloatingPointError("nonfinite CPU pose residual/Jacobian")
        return PoseEvaluation(x=x.copy(), geometry_mm=geometry.copy(), angles_deg=angles.copy(),
            residual=np.r_[r.real, r.imag], jacobian=np.vstack((J.real, J.imag)),
            b=bs, db=dbs, gains=gains, data_residuals=data, weighted_data_residuals=weighted,
            penalty_residuals=penalties, data_jacobians=data_jacobians,
            weighted_data_jacobians=weighted_jacobians, penalty_jacobians=penalty_jacobians,
            amplitude_residuals={}, amplitude_jacobians={}, diagnostics={"groups": diagnostics,
                "qr_checks": [], "parameter_layout": list(self.parameter_layout), "counts": dict(self.counts),
                "angles_deg": angles.tolist(), "pose_u6": x[:6].tolist(), "pivot_m": self.pivot_m.tolist(),
                "rotation_order": "Ry(beta) Rx(alpha) Rz(gamma), active right handed column vectors",
                "q_array": self.q_array, "q_probe": self.q_probe,
                "gauge": "first session g=1 separately in every frequency/chain group"})

    def profile_gradient_hessian(self, x):
        from .joint_pose_newton import cpu_profile_gradient_hessian
        return cpu_profile_gradient_hessian(self, x)

    def validation_evaluate(self, x, validation_groups, gains=None):
        value = self.evaluate(x)
        residuals, jacobians, predictions = [], [], {}
        lookup = {g.name: g for g in self.groups}
        for name, sessions in validation_groups.items():
            if name not in lookup:
                raise ValueError("validation group must have a trained response")
            for session in sessions:
                if session.frequency_hz != lookup[name].sessions[0].frequency_hz:
                    raise ValueError("validation frequency must match trained group")
                gain = complex((gains or {}).get(session.name, value.gains.get(session.name, 1.)))
                if not np.isfinite(gain):
                    raise ValueError("validation gain must be finite")
                K, dK = self._kernel(session, value.geometry_mm, value.angles_deg, derivatives=True)
                A = gain*K*session.steering[None, :]
                prediction, J = A@value.b[name], -A@value.db[name]
                for j in range(6):
                    J[:, j] -= gain*(dK[j]@(session.steering*value.b[name]))
                if session.name in self._gain_indices and not (gains and session.name in gains):
                    gi = self._gain_indices[session.name]
                    J[:, gi] -= prediction; J[:, gi+1] -= 1j*prediction
                predictions[session.name] = prediction
                residuals.append(session.weight*(session.field-prediction))
                jacobians.append(session.weight[:, None]*J)
        if not residuals:
            raise ValueError("nonempty validation sessions required")
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        return {"residual": np.r_[r.real, r.imag], "jacobian": np.vstack((J.real, J.imag)),
                "predictions": predictions, "evaluation": value}

    def fit(self, start_geometry_mm=(60., 0., 0.), max_nfev=1500, fixed_geometry=False,
            start_gains=None, ftol=1e-8, gtol=1e-7, xtol=1e-7, gain_log_bounds=(-3., 3.),
            start_parameters=None, checkpoint_callback=None, *, start_angles_deg=(0., 0., 0.)):
        """Original bounded TRF semantics, now with six physical coordinates."""
        from .joint_pose_newton import outer_bounds, physical_free_mask
        started, initial_counts = time.perf_counter(), dict(self.counts)
        if start_parameters is None:
            x0 = self.initial_parameters(start_geometry_mm, start_gains, angles_deg=start_angles_deg)
        else:
            if start_gains is not None or np.any(np.asarray(start_angles_deg) != 0):
                raise ValueError("use start_parameters or explicit start gains/angles")
            x0 = _real_array(start_parameters, "start_parameters", (self.parameter_count,)).copy()
            self._pose(x0); self._gains(x0)
        free = physical_free_mask(self, fixed_geometry)
        lower, upper = outer_bounds(self, gain_log_bounds)
        if np.any(x0 < lower) or np.any(x0 > upper):
            raise ValueError("starting pose/gains outside bounds")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError("checkpoint callback must be callable")
        def expand(z):
            x = x0.copy(); x[free] = z
            return x
        iteration, last_x, last_cost, last_nfev = 0, x0.copy(), None, 0
        def checkpoint(event, x, cost, nfev, **extra):
            if checkpoint_callback is not None:
                G, angles = self._pose(x)
                checkpoint_callback({"event": event, "x": x.tolist(), "geometry_mm": G.tolist(),
                    "angles_deg": angles.tolist(), "pose_u6": x[:6].tolist(),
                    "gains": {k: [float(v.real), float(v.imag)] for k, v in self._gains(x).items()},
                    "cost": None if cost is None else float(cost), "nfev": int(nfev), "iteration": iteration,
                    "max_nfev": int(max_nfev), "counts": {k: self.counts[k]-initial_counts[k] for k in self.counts},
                    "elapsed_seconds": time.perf_counter()-started, **extra})
        def callback(intermediate_result):
            nonlocal iteration, last_x, last_cost, last_nfev
            iteration += 1; last_x = expand(intermediate_result.x)
            last_cost = float(intermediate_result.cost); last_nfev = int(intermediate_result.nfev)
            checkpoint("accepted_iteration", last_x, last_cost, last_nfev)
        try:
            initial = self.evaluate(x0); checkpoint("initial", x0, initial.cost, 0)
            if np.any(free):
                result = optimize.least_squares(lambda z: self.evaluate(expand(z)).residual, x0[free],
                    jac=lambda z: self.evaluate(expand(z)).jacobian[:, free], bounds=(lower[free], upper[free]),
                    method="trf", max_nfev=max_nfev, ftol=ftol, gtol=gtol, xtol=xtol, callback=callback)
                x = expand(result.x)
                success, status, message = bool(result.success), int(result.status), str(result.message)
                nfev, njev, optimality = int(result.nfev), int(result.njev), float(result.optimality)
                active = np.zeros(self.parameter_count, int); active[free] = result.active_mask
            else:
                x, success, status, message = x0, True, 1, "all outer variables fixed; inner solve only"
                nfev, njev, optimality, active = 1, 1, 0., np.zeros(self.parameter_count, int)
            value = self.evaluate(x)
        except (ValueError, linalg.LinAlgError, FloatingPointError) as error:
            checkpoint("failed", last_x, last_cost, last_nfev, success=False, status=-3, message=str(error))
            raise
        checkpoint("final", x, value.cost, nfev, success=success, status=status, message=message,
                   njev=njev, optimality=optimality)
        return PoseFitResult(geometry_mm=value.geometry_mm.copy(), angles_deg=value.angles_deg.copy(),
            evaluation=value, success=success, status=status, message=message, nfev=nfev, njev=njev,
            optimality=optimality, active_mask=active,
            counts={k: self.counts[k]-initial_counts[k] for k in self.counts},
            elapsed_seconds=time.perf_counter()-started, x=x.copy())
