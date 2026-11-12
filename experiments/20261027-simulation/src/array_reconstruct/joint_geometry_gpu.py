"""Double precision CUDA variable projection with the CPU mathematical contract.

No files, truth, measurements, initialization choices or scientific gates are
read here. Immutable observations stay on one CUDA device. Only the small outer
optimizer and returned diagnostics/arrays use CPU. QR is the sole linear solve;
there is no unguarded normal-equation or reduced precision acceleration.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy import linalg

from .joint_geometry import (JointProblem, ProfileEvaluation, _complex_array,
                             _real_array, parameters_to_geometry)
from .physics import C0_M_S


class GPUJointProblem(JointProblem):
    """JointProblem-compatible complex128 QR backend on one CUDA device.

    ``fallback='cpu'`` preserves a complete failure ledger and explicitly marks
    CPU evaluations. ``fallback='raise'`` is useful for paired validation: no
    computation can silently leave CUDA. Amplitude priors require their separate
    validated nonlinear CUDA solver and are deliberately rejected for now.
    """
    def __init__(self, elements_m, groups, q=2., sign=-1, *, device="cuda:0",
                 fallback="cpu", **cpu_options):
        import torch
        if fallback not in ("cpu", "raise"):
            raise ValueError("fallback must be cpu or raise")
        cpu_options.pop("backend", None)
        super().__init__(elements_m, groups, q, sign, backend="qr", **cpu_options)
        if any(g.amplitude_strength > 0 for g in self.groups):
            raise NotImplementedError("nonlinear amplitude CUDA path is not validated")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPUJointProblem requires an available CUDA device")
        self.fallback = fallback
        self.backend = "cuda_qr_complex128"
        self.cd, self.rd = torch.complex128, torch.float64
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.counts.update(gpu_failures=0, cpu_fallbacks=0, gpu_qr_factorizations=0,
                           gpu_hessian_calls=0, kernel_second_derivative_calls=0,
                           gpu_prediction_calls=0, gpu_validation_calls=0)
        self._cpu_reference = None
        self._device_state = None
        self._resident_sessions = {}
        self.gpu_elapsed_seconds = 0.
        self.gpu_hessian_seconds = 0.
        self._sync()
        started = time.perf_counter()
        self._elements = self._tensor(self.elements_m[:, :2], self.rd)
        self._eye = torch.eye(len(self.elements_m), device=self.device, dtype=self.cd)
        self._resident_groups = []
        for group in self.groups:
            native = [self._session(session) for session in group.sessions]
            penalty = group.lambda_reg*self._eye
            rhs = group.lambda_reg*self._tensor(group.b0, self.cd)
            if group.template_strength > 0:
                a = self._tensor(group.template_prior, self.cd)
                P = self._eye-a[:, None]*a.conj()[None, :]/torch.vdot(a, a).real
                penalty = torch.cat((penalty, group.template_strength*P))
                rhs = torch.cat((rhs, torch.zeros_like(rhs)))
            self._resident_groups.append((group, native, penalty, rhs))
        self._sync()
        self.upload_setup_seconds = time.perf_counter()-started

    @classmethod
    def from_problem(cls, problem, *, device="cuda:0", fallback="cpu"):
        return cls(problem.elements_m, problem.groups, problem.q, problem.sign,
                   device=device, fallback=fallback,
                   normal_condition_max=problem.normal_condition_max,
                   normal_qr_check=problem.normal_qr_check,
                   qr_coefficient_rtol=problem.qr_coefficient_rtol,
                   qr_residual_rtol=problem.qr_residual_rtol)

    def _tensor(self, array, dtype):
        return self.torch.as_tensor(np.array(array, copy=True), device=self.device, dtype=dtype)

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    def _numpy(self, tensor):
        return tensor.detach().cpu().numpy().copy()

    def _session(self, session):
        key = (session.frequency_hz, session.samples_xy_m.shape,
               hashlib.sha256(session.samples_xy_m.tobytes()).hexdigest())
        return {"session": session, "key": key,
                "xy": self._tensor(session.samples_xy_m, self.rd),
                "u": self._tensor(session.steering, self.cd),
                "w": self._tensor(session.weight, self.rd),
                "y": self._tensor(session.field, self.cd)}

    def runtime_descriptor(self):
        torch = self.torch
        props = torch.cuda.get_device_properties(self.device)
        return {"backend": self.backend, "device": str(self.device), "device_name": props.name,
                "device_total_memory_bytes": props.total_memory,
                "complex_dtype": "complex128", "real_dtype": "float64",
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
                "torch_cpu_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "synchronized_evaluation_seconds": self.gpu_elapsed_seconds,
                "evaluation_timing_scope": "profile attempt through required downloads; includes explicit CPU fallback wall time when used",
                "cpu_fallback_counts": dict(self._cpu_reference.counts) if self._cpu_reference is not None else {},
                "synchronized_hessian_seconds": self.gpu_hessian_seconds,
                "upload_setup_seconds": self.upload_setup_seconds,
                "fallback_policy": self.fallback, "counts": dict(self.counts),
                "counter_scope": "Training-only CUDA kernel calls count actual unique frequency/grid constructions across groups; derivative count is three fused first-derivative matrices per construction. CPU counters may count repeated per-group/zero-weight diagnostics. Prediction and validation have separate call counters.",
                "fallback_scope": "profile evaluation and exact profile Hessian; prediction/validation errors propagate explicitly",
                "failure_ledger": list(self.failure_ledger)}

    def _kernel(self, native, tx):
        torch = self.torch
        d = (60.+10.*tx[0])*.001
        dx, dy = tx[1]*.01, tx[2]*.01
        xx = native["xy"][:, None, 0]+dx-self._elements[None, :, 0]
        yy = native["xy"][:, None, 1]+dy-self._elements[None, :, 1]
        radius = torch.sqrt(xx*xx+yy*yy+d*d)
        wave = self.sign*2j*np.pi*native["session"].frequency_hz/C0_M_S
        K = d**self.q*torch.exp(wave*radius)/radius**(self.q+1)
        beta = wave/radius-(self.q+1)/radius**2
        log_first = (beta*d+self.q/d, beta*xx, beta*yy)
        first = tuple(K*.01*value for value in log_first)
        return {"K": K, "first": first, "radius": radius, "v": (d, xx, yy),
                "beta": beta, "wave": wave, "log_first": log_first, "second": {}}

    def _second(self, kernel, j, k):
        key = min(j, k), max(j, k)
        if key not in kernel["second"]:
            R, v = kernel["radius"], kernel["v"]
            gamma = -kernel["wave"]/R**3+2*(self.q+1)/R**4
            value = gamma*v[j]*v[k]
            if j == k:
                value = value+kernel["beta"]
            if j == k == 0:
                value = value-self.q/v[0]**2
            kernel["second"][key] = kernel["K"]*1e-4*(
                kernel["log_first"][j]*kernel["log_first"][k]+value)
            self.counts["kernel_second_derivative_calls"] += 1
        return kernel["second"][key]

    def evaluate(self, x):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        if self._last is not None and np.array_equal(x, self._last.x):
            self.counts["cache_hits"] += 1
            return self._last
        geometry = parameters_to_geometry(x[:3])
        gains = self._gains(x)
        self.counts["evaluations"] += 1
        self._device_state = None
        started = time.perf_counter()
        try:
            self._sync()
            with self.torch.no_grad():
                value = self._evaluate_gpu(x, geometry, gains)
            self._sync()
        except (RuntimeError, ValueError, linalg.LinAlgError, FloatingPointError) as error:
            self._device_state = None
            self._last = None
            self.counts["failed_evaluations"] += 1
            self.counts["gpu_failures"] += 1
            record = {"x": x.tolist(), "geometry_mm": geometry.tolist(),
                      "exception_type": type(error).__name__, "reason": str(error),
                      "fallback": self.fallback, "device": str(self.device)}
            self.failure_ledger.append(record)
            if self.fallback == "raise":
                raise
            if self._cpu_reference is None:
                self._cpu_reference = JointProblem(self.elements_m, self.groups, self.q, self.sign, backend="qr")
            value = self._cpu_reference.evaluate(x)
            self.counts["cpu_fallbacks"] += 1
            value.diagnostics["gpu"] = {"requested_backend": self.backend,
                                          "backend_used": "cpu_qr_fallback", "failure": record}
        finally:
            self.gpu_elapsed_seconds += time.perf_counter()-started
        value.diagnostics.setdefault("gpu", {}).update(self.runtime_descriptor())
        self._last = value
        return value

    def _evaluate_gpu(self, x, geometry, gains):
        torch = self.torch
        p, n = self.parameter_count, len(self.elements_m)
        tx = self._tensor(x, self.rd)
        cache, state, bs, dbs, diagnostics = {}, {}, {}, {}, {}
        residuals, jacobians = [], []
        data, weighted, penalties = {}, {}, {}
        data_jacobians, weighted_jacobians, penalty_jacobians = {}, {}, {}
        for group, sessions, penalty, penalty_rhs in self._resident_groups:
            blocks, rhs, details, cursor = [], [], [], 0
            for ss in sessions:
                if ss["key"] not in cache:
                    cache[ss["key"]] = self._kernel(ss, tx)
                    self.counts["kernel_calls"] += 1
                    self.counts["kernel_derivative_calls"] += 3
                gi = self._gain_indices.get(ss["session"].name)
                gain = torch.ones((), device=self.device, dtype=self.cd) if gi is None else torch.exp(tx[gi]+1j*tx[gi+1])
                weight = ss["w"][:, None]*gain*ss["u"][None, :]
                A = cache[ss["key"]]["K"]*weight
                sl = slice(cursor, cursor+len(ss["session"].field)); cursor = sl.stop
                blocks.append(A); rhs.append(ss["w"]*ss["y"])
                details.append((ss, sl, weight, gi, gain))
            m = cursor
            B, c = torch.cat(blocks+[penalty]), torch.cat(rhs+[penalty_rhs])
            self.counts["linear_factorizations"] += 1
            self.counts["gpu_qr_factorizations"] += 1
            if B.shape[0] < B.shape[1]:
                raise linalg.LinAlgError("augmented system is rank deficient; use positive fixed regularization")
            Q, R = torch.linalg.qr(B, mode="reduced")
            diagonal = torch.abs(torch.diagonal(R))
            rank_threshold = np.finfo(float).eps*max(B.shape)*torch.max(diagonal)
            if bool(torch.min(diagonal) <= rank_threshold):
                raise linalg.LinAlgError("augmented system is rank deficient; use positive fixed regularization")
            b = torch.linalg.solve_triangular(R, (Q.mH@c)[:, None], upper=True)[:, 0]
            r = c-B@b
            J = torch.zeros((len(c), p), device=self.device, dtype=self.cd)
            db = torch.zeros((n, p), device=self.device, dtype=self.cd)
            products = torch.zeros((len(c), p), device=self.device, dtype=self.cd)
            adjoints = torch.zeros((n, p), device=self.device, dtype=self.cd)
            active = list(range(3))
            for ss, sl, weight, gi, gain in details:
                if gi is not None:
                    active.extend((gi, gi+1))
                for j in range(3):
                    Bj = cache[ss["key"]]["first"][j]*weight
                    products[sl, j] = Bj@b
                    adjoints[:, j] += Bj.mH@r[sl]
                if gi is not None:
                    products[sl, gi] = B[sl]@b
                    products[sl, gi+1] = 1j*products[sl, gi]
                    adjoints[:, gi] += B[sl].mH@r[sl]
                    adjoints[:, gi+1] += -1j*(B[sl].mH@r[sl])
            # All active implicit derivatives share the same QR factors.
            z = torch.linalg.solve_triangular(R.mH, adjoints[:, active], upper=False)
            db[:, active] = torch.linalg.solve_triangular(R, z-Q.mH@products[:, active], upper=True)
            J[:, active] = -products[:, active]-B@db[:, active]
            self.counts["linear_derivative_solves"] += len(active)
            Bnorm, rnorm, cnorm = torch.linalg.norm(B), torch.linalg.norm(r), torch.linalg.norm(c)
            scale = torch.maximum(Bnorm*rnorm, np.finfo(float).eps*Bnorm*cnorm).clamp(min=1e-30)
            stationarity = (torch.linalg.norm(B.mH@r)/scale).item()
            if not bool(torch.isfinite(b).all() and torch.isfinite(r).all() and torch.isfinite(J).all()):
                raise FloatingPointError("nonfinite CUDA QR coefficients/residual/Jacobian")
            for ss, sl, weight, gi, gain in details:
                name = ss["session"].name
                A = cache[ss["key"]]["K"]*gain*ss["u"][None, :]
                raw = ss["y"]-A@b
                if bool(torch.all(ss["w"] > 0)):
                    rawJ = J[sl]/ss["w"][:, None]
                else:
                    rawJ = -A@db
                    for j in range(3):
                        rawJ[:, j] -= gain*(cache[ss["key"]]["first"][j]@(ss["u"]*b))
                    if gi is not None:
                        rawJ[:, gi] -= A@b
                        rawJ[:, gi+1] -= 1j*(A@b)
                data[name], weighted[name] = self._numpy(raw), self._numpy(r[sl])
                data_jacobians[name], weighted_jacobians[name] = self._numpy(rawJ), self._numpy(J[sl])
            bs[group.name], dbs[group.name] = self._numpy(b), self._numpy(db)
            penalties[group.name], penalty_jacobians[group.name] = self._numpy(r[m:]), self._numpy(J[m:])
            template_scale = None
            if group.template_strength > 0:
                a = self._tensor(group.template_prior, self.cd)
                scale_a = (torch.vdot(a, b)/torch.vdot(a, a)).item()
                template_scale = [scale_a.real, scale_a.imag]
            diagnostics[group.name] = {"backend_requested": self.backend, "backend_used": self.backend,
                "linear_initialization_backend": None, "normal_condition_h_1": None,
                "normal_condition_max": self.normal_condition_max, "fallback_reason": None,
                "qr_verification": None, "stationarity": stationarity,
                "stationarity_normalization": "||B^H r||_2 / max(||B||_F ||r||_2, eps64 ||B||_F ||c||_2, 1e-30)",
                "lambda_reg": group.lambda_reg, "inner_solver": "complex linear augmented least squares",
                "amplitude": None, "penalty": "fixed physical lambda * (b-b0), L=I",
                "reference_session": group.sessions[0].name, "reference_gain": [1., 0.],
                "template_strength": group.template_strength, "template_scale": template_scale,
                "template_penalty": "mu * (I-aa^H/(a^Ha)) b; unknown common complex scale eliminated" if group.template_strength > 0 else None,
                "data_residual_norm": torch.linalg.norm(r[:m]).item(),
                "penalty_residual_norm": torch.linalg.norm(r[m:]).item(),
                "qr_rank_diagonal_min": torch.min(diagonal).item(),
                "qr_rank_threshold": rank_threshold.item()}
            residuals.append(r); jacobians.append(J)
            state[group.name] = {"b": b, "db": db, "r": r, "J": J,
                "products": products, "adjoints": adjoints, "details": details, "m": m}
        r, J = torch.cat(residuals), torch.cat(jacobians)
        self._device_state = {"x": x.copy(), "groups": state, "kernels": cache, "curvature": None}
        return ProfileEvaluation(x.copy(), geometry, self._numpy(torch.cat((r.real, r.imag))),
            self._numpy(torch.cat((J.real, J.imag))), bs, dbs, gains, data, weighted, penalties,
            data_jacobians, weighted_jacobians, penalty_jacobians, {}, {},
            {"groups": diagnostics, "qr_checks": [], "parameter_layout": list(self.parameter_layout),
             "counts": dict(self.counts), "gauge": "first session g=1 separately in every frequency/chain group"})

    def profile_gradient_hessian(self, x):
        """Dispatch exact CUDA curvature, with an explicit full CPU fallback."""
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        self.evaluate(x)  # Failed evaluation is already recorded; never double-count it.
        try:
            return self._profile_gradient_hessian_cuda(x)
        except (RuntimeError, linalg.LinAlgError, FloatingPointError) as error:
            self._device_state = None
            self._last = None
            self.counts["gpu_failures"] += 1
            self.counts["failed_evaluations"] += 1
            record = {"operation": "exact_profile_hessian", "x": x.tolist(),
                      "exception_type": type(error).__name__, "reason": str(error),
                      "fallback": self.fallback, "device": str(self.device)}
            self.failure_ledger.append(record)
            if self.fallback == "raise":
                raise
            if self._cpu_reference is None:
                self._cpu_reference = JointProblem(self.elements_m, self.groups, self.q, self.sign, backend="qr")
            from .joint_newton import profile_gradient_hessian
            result = profile_gradient_hessian(self._cpu_reference, x)
            self.counts["cpu_fallbacks"] += 1
            result["diagnostics"].update(gpu_hessian_fallback=True, gpu_failure=record)
            result["evaluation"].diagnostics["gpu"] = {"backend_used": "cpu_qr_fallback", "failure": record}
            self._last = result["evaluation"]
            return result

    def _profile_gradient_hessian_cuda(self, x):
        """Exact envelope curvature, including gain/geometry cross terms, on CUDA."""
        evaluation = self.evaluate(x)
        if self._device_state is None:
            from .joint_newton import profile_gradient_hessian
            result = profile_gradient_hessian(self._cpu_reference, x)
            result["diagnostics"]["gpu_hessian_fallback"] = True
            return result
        state = self._device_state
        if state["curvature"] is not None:
            cached = dict(state["curvature"])
            cached["diagnostics"] = {**cached["diagnostics"], "second_kernel_calls": 0,
                                      "additional_first_kernel_calls": 0, "gpu_curvature_cache_hit": True}
            return cached
        torch = self.torch
        self._sync(); started = time.perf_counter()
        initial_second = self.counts["kernel_second_derivative_calls"]
        with torch.no_grad():
            gradient = torch.zeros(self.parameter_count, device=self.device, dtype=self.rd)
            hessian = torch.zeros((self.parameter_count, self.parameter_count), device=self.device, dtype=self.rd)
            for group in self.groups:
                s = state["groups"][group.name]
                r, rj, products = s["r"][:s["m"]], s["J"][:s["m"]], s["products"][:s["m"]]
                curvature = torch.zeros_like(hessian, dtype=self.cd)
                for ss, sl, weight, gi, gain in s["details"]:
                    kernel = state["kernels"][ss["key"]]
                    for j in range(3):
                        for k in range(j, 3):
                            product = (self._second(kernel, j, k)*weight)@s["b"]
                            value = torch.vdot(r[sl], product)
                            curvature[j, k] += value
                            if j != k:
                                curvature[k, j] += value
                    if gi is not None:
                        for j in range(3):
                            value = torch.vdot(r[sl], products[sl, j])
                            curvature[j, gi] += value; curvature[gi, j] += value
                            curvature[j, gi+1] += 1j*value; curvature[gi+1, j] += 1j*value
                        value = torch.vdot(r[sl], products[sl, gi])
                        curvature[gi, gi] += value
                        curvature[gi, gi+1] += 1j*value; curvature[gi+1, gi] += 1j*value
                        curvature[gi+1, gi+1] -= value
                gradient -= (products.mH@r).real
                hessian -= (products.T@rj.conj()+curvature+s["adjoints"].mH@s["db"]).real
            asymmetry = (torch.linalg.norm(hessian-hessian.T)/torch.linalg.norm(hessian).clamp(min=1e-30)).item()
            hessian = .5*(hessian+hessian.T)
            residual_gradient = self._tensor(evaluation.jacobian.T@evaluation.residual, self.rd)
            difference = (torch.linalg.norm(gradient-residual_gradient)/torch.linalg.norm(gradient).clamp(min=1e-30)).item()
            result = {"evaluation": evaluation, "gradient": self._numpy(gradient), "hessian": self._numpy(hessian),
                "diagnostics": {"hessian_relative_asymmetry_before_symmetrizing": asymmetry,
                    "envelope_gradient_relative_difference": difference,
                    "second_kernel_calls": self.counts["kernel_second_derivative_calls"]-initial_second,
                    "additional_first_kernel_calls": 0, "hessian_eigenvalues": self._numpy(torch.linalg.eigvalsh(hessian)).tolist(),
                    "backend_used": self.backend, "derivative_scope": "selected stationary branch only; global profile smoothness and optimality not established"}}
        self._sync()
        self.gpu_hessian_seconds += time.perf_counter()-started
        self.counts["gpu_hessian_calls"] += 1
        state["curvature"] = result
        return result

    def predict(self, session, geometry_mm, b, g=1.):
        b = _complex_array(b, "b", (len(self.elements_m),))
        geometry = _real_array(geometry_mm, "geometry_mm", (3,))
        x = np.zeros(self.parameter_count); x[:3] = (geometry-np.array((60., 0., 0.)))/10.
        parameters_to_geometry(x[:3])
        if not np.isfinite(complex(g)):
            raise ValueError("prediction gain must be finite")
        with self.torch.no_grad():
            native = self._session(session)
            K = self._kernel(native, self._tensor(x, self.rd))["K"]
            result = complex(g)*(K@(native["u"]*self._tensor(b, self.cd)))
            self.counts["gpu_prediction_calls"] += 1
            return self._numpy(result)

    def validation_evaluate(self, x, validation_groups, gains=None):
        evaluation = self.evaluate(x)
        if self._device_state is None:
            return self._cpu_reference.validation_evaluate(x, validation_groups, gains)
        torch = self.torch
        residuals, jacobians, predictions = [], [], {}
        lookup = {g.name: g for g in self.groups}
        with torch.no_grad():
            tx = self._tensor(x, self.rd)
            cache = {}
            for name, sessions in validation_groups.items():
                if name not in lookup:
                    raise ValueError("validation group must have a trained response")
                state = self._device_state["groups"][name]
                for session in sessions:
                    if session.frequency_hz != lookup[name].sessions[0].frequency_hz:
                        raise ValueError("validation frequency must match the trained group")
                    native = self._session(session)
                    if native["key"] not in cache:
                        cache[native["key"]] = self._kernel(native, tx)
                    kernel = cache[native["key"]]
                    gain = complex((gains or {}).get(session.name, evaluation.gains.get(session.name, 1.)))
                    if not np.isfinite(gain):
                        raise ValueError("validation gain must be finite")
                    A = gain*kernel["K"]*native["u"][None, :]
                    prediction = A@state["b"]
                    J = -A@state["db"]
                    for j in range(3):
                        J[:, j] -= gain*(kernel["first"][j]@(native["u"]*state["b"]))
                    if session.name in self._gain_indices and not (gains and session.name in gains):
                        gi = self._gain_indices[session.name]
                        J[:, gi] -= prediction; J[:, gi+1] -= 1j*prediction
                    predictions[session.name] = self._numpy(prediction)
                    residuals.append(native["w"]*(native["y"]-prediction))
                    jacobians.append(native["w"][:, None]*J)
            if not residuals:
                raise ValueError("nonempty validation sessions required")
            r, J = torch.cat(residuals), torch.cat(jacobians)
            self.counts["gpu_validation_calls"] += 1
            return {"residual": self._numpy(torch.cat((r.real, r.imag))),
                    "jacobian": self._numpy(torch.cat((J.real, J.imag))),
                    "predictions": predictions, "evaluation": evaluation}


def make_gpu_problem(data, lambda_reg, *, model="M2", branch="C0", template_strength=0.,
                     amplitude_strength=0., amplitude_n_starts=3, device="cuda:0",
                     fallback="cpu", **kwargs):
    """Public-fit-data factory; no truth arrays are accepted or inspected."""
    from .joint_virtual import make_problem
    kwargs.pop("backend", None)
    problem = make_problem(data, lambda_reg, model=model, branch=branch, backend="qr",
        template_strength=template_strength, amplitude_strength=amplitude_strength,
        amplitude_n_starts=amplitude_n_starts, **kwargs)
    return GPUJointProblem.from_problem(problem, device=device, fallback=fallback)
