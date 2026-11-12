"""Isolated float64/complex128 CUDA profile solver for a six-coordinate pose.

The rotation and its tiny analytic derivatives are constructed on CPU; kernels,
QR, implicit response derivatives, residuals and profile curvature stay on one
CUDA device. This module performs no file access and has no CPU fallback.
Scientific qualification of the CUDA path is separate from implementation.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy import linalg

from .joint_geometry import _complex_array, _real_array
from .joint_pose_cpu import PoseEvaluation, PoseJointProblem
from .joint_pose_kernel import (parameters_to_pose, pose_to_parameters,
                                rotation_pose_derivatives)
from .physics import C0_M_S


class GPUPoseJointProblem(PoseJointProblem):
    """Six shared pose coordinates followed by independent complex gains.

    Only augmented complex QR with fixed ridge/template rows is supported.
    ``fallback`` is intentionally restricted to ``'raise'``: a failed CUDA
    operation is recorded and propagated, never replaced by CPU computation.
    Nonlinear amplitude priors require separate numerical qualification.
    """

    def __init__(self, elements_m, groups, *, q_array=2., q_probe=0., sign=-1,
                 device="cuda:0", fallback="raise", **cpu_options):
        if fallback != "raise":
            raise ValueError("the isolated pose CUDA backend requires fallback='raise'")
        if cpu_options.pop("backend", "qr") != "qr":
            raise ValueError("the isolated pose CUDA backend requires augmented QR")
        super().__init__(elements_m, groups, q_array=q_array, q_probe=q_probe,
                         sign=sign, backend="qr", **cpu_options)
        if any(group.amplitude_strength > 0 for group in self.groups):
            raise NotImplementedError("nonlinear amplitude pose CUDA path is not qualified")
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPUPoseJointProblem requires an available CUDA device")
        self.fallback = fallback
        self.backend = "cuda_pose_qr_complex128"
        self.cd, self.rd = torch.complex128, torch.float64
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._initialize_device_counters()
        self._sync()
        started = time.perf_counter()
        self._upload_resident_state()
        self._sync()
        self.upload_setup_seconds = time.perf_counter()-started

    @classmethod
    def from_problem(cls, problem, *, device="cuda:0", fallback="raise"):
        if not isinstance(problem, PoseJointProblem):
            raise TypeError("from_problem requires an explicit PoseJointProblem")
        return cls(problem.elements_m, problem.groups, q_array=problem.q_array,
                   q_probe=problem.q_probe, sign=problem.sign, device=device,
                   fallback=fallback, normal_condition_max=problem.normal_condition_max,
                   normal_qr_check=problem.normal_qr_check,
                   qr_coefficient_rtol=problem.qr_coefficient_rtol,
                   qr_residual_rtol=problem.qr_residual_rtol)

    def _initialize_device_counters(self):
        self.counts.update(gpu_failures=0, cpu_fallbacks=0, gpu_qr_factorizations=0,
                           gpu_hessian_calls=0, kernel_second_derivative_calls=0,
                           gpu_prediction_calls=0, gpu_validation_calls=0,
                           gpu_evaluation_failures=0, gpu_hessian_failures=0,
                           gpu_prediction_failures=0, gpu_validation_failures=0,
                           cpu_rotation_derivative_calls=0, gpu_kernel_calls=0,
                           gpu_kernel_derivative_calls=0,
                           gpu_prediction_kernel_calls=0, gpu_validation_kernel_calls=0)
        self._device_state = None
        self.gpu_elapsed_seconds = 0.
        self.gpu_hessian_seconds = 0.
        self.cpu_rotation_derivative_seconds = 0.
        self.upload_setup_seconds = 0.

    def _tensor(self, array, dtype):
        return self.torch.as_tensor(np.array(array, copy=True), device=self.device, dtype=dtype)

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    def _numpy(self, tensor):
        return tensor.detach().cpu().numpy().copy()

    def _upload_resident_state(self):
        """Upload immutable public training arrays; called after CUDA validation."""
        torch = self.torch
        elements = np.zeros((len(self.elements_m), 3))
        elements[:, :2] = self.elements_m[:, :2]
        self._elements = self._tensor(elements, self.rd)
        self._centered_elements = self._tensor(elements-self.pivot_m, self.rd)
        self._pivot = self._tensor(self.pivot_m, self.rd)
        self._eye = torch.eye(len(elements), device=self.device, dtype=self.cd)
        self._resident_groups = []
        for group in self.groups:
            penalty = group.lambda_reg*self._eye
            rhs = group.lambda_reg*self._tensor(group.b0, self.cd)
            if group.template_strength > 0:
                a = self._tensor(group.template_prior, self.cd)
                projection = self._eye-a[:, None]*a.conj()[None, :]/torch.vdot(a, a).real
                penalty = torch.cat((penalty, group.template_strength*projection))
                rhs = torch.cat((rhs, torch.zeros_like(rhs)))
            self._resident_groups.append((group, [self._session(s) for s in group.sessions], penalty, rhs))

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
                "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
                "torch_cpu_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "synchronized_evaluation_seconds": self.gpu_elapsed_seconds,
                "evaluation_timing_scope": "synchronized profile-attempt wall time, including tiny CPU rotation construction and diagnostic downloads; not isolated CUDA-event time",
                "synchronized_hessian_seconds": self.gpu_hessian_seconds,
                "hessian_timing_scope": "synchronized curvature-request wall time after required profile evaluation; cached requests included",
                "cpu_rotation_derivative_seconds": self.cpu_rotation_derivative_seconds,
                "upload_setup_seconds": self.upload_setup_seconds,
                "fallback_policy": self.fallback, "counts": dict(self.counts),
                "cpu_work_scope": "tiny 3x3 rotation derivatives and outer optimizer/diagnostic arrays only",
                "cuda_work_scope": "all large kernels, QR, response derivatives, residuals, Jacobians and Hessian contractions",
                "counter_scope": "training kernel counts unique frequency/grid builds per profile; six fused first derivatives per build; prediction/validation separate; no CPU profile fallback",
                "second_derivative_storage": "one physical coordinate pair at a time; no persistent second-kernel cache",
                "pose_convention": "active column-vector R=Ry(beta) Rx(alpha) Rz(gamma), fixed registered centroid pivot",
                "pattern_split_assumption": {"q_array": self.q_array, "q_probe": self.q_probe,
                    "status": "explicit scalar radiation-pattern hypothesis; not a hardware calibration fact"},
                "failure_ledger": list(self.failure_ledger)}

    def _pose_state(self, x):
        started = time.perf_counter()
        geometry, angles = parameters_to_pose(x[:6])
        rotation, first, second = rotation_pose_derivatives(angles)
        self.cpu_rotation_derivative_seconds += time.perf_counter()-started
        self.counts["cpu_rotation_derivative_calls"] += 1
        R, dR, ddR = (self._tensor(a, self.rd) for a in (rotation, first, second))
        # v_j and v_jk depend on element coordinates, but not sample coordinates.
        va = -(self._centered_elements[None, :, :]@dR.transpose(-1, -2))
        va[0, :, 2] = .01
        va[1, :, 0] = .01
        va[2, :, 1] = .01
        positions = self._elements if not np.any(angles) else self._pivot+self._centered_elements@R.T
        return {"geometry": geometry, "angles": angles, "R": R, "dR": dR, "ddR": ddR,
                "normal": R[:, 2], "normal_first": dR[:, :, 2],
                "normal_second": ddR[:, :, :, 2], "va": va,
                "positions": positions}

    @staticmethod
    def _dot_sample_element(v, a):
        return sum(v[axis]*a[None, :, axis] for axis in range(3))

    @staticmethod
    def _dot_normal(v, normal):
        return sum(v[axis]*normal[axis] for axis in range(3))

    def _first_log(self, kernel, j):
        va = kernel["pose"]["va"][j]
        rj = self._dot_sample_element(kernel["v"], va)/kernel["radius"]
        result = kernel["radial_log"]*rj
        if self.q_array:
            hj = self._dot_normal(kernel["v"], kernel["pose"]["normal_first"][j])
            hj = hj+(va@kernel["pose"]["normal"])[None, :]
            result = result+self.q_array*hj/kernel["h_array"]
        if self.q_probe:
            result = result+self.q_probe*va[None, :, 2]/kernel["h_probe"]
        return result

    def _kernel(self, native, pose, *, derivatives=True):
        torch = self.torch
        d, dx, dy = pose["geometry"]*.001
        positions = pose["positions"]
        v = (native["xy"][:, None, 0]+dx-positions[None, :, 0],
             native["xy"][:, None, 1]+dy-positions[None, :, 1],
             (d-positions[:, 2])[None, :].expand(len(native["xy"]), -1))
        radius = torch.sqrt(sum(value*value for value in v))
        h_array = self._dot_normal(v, pose["normal"])
        h_probe = v[2]
        if (not bool(torch.isfinite(radius).all()) or bool(torch.any(radius <= 0))
                or (self.q_array and bool(torch.any(h_array <= 0)))
                or (self.q_probe and bool(torch.any(h_probe <= 0)))):
            raise ValueError("pose kernel requires positive radius and positive active pattern projections")
        wave = self.sign*2j*np.pi*native["session"].frequency_hz/C0_M_S
        radial_log = wave-(self.q_array+self.q_probe+1)/radius
        if not np.any(pose["angles"]):
            amplitude = d**(self.q_array+self.q_probe)
        else:
            amplitude = h_array**self.q_array*h_probe**self.q_probe
        K = amplitude*torch.exp(wave*radius)/radius**(self.q_array+self.q_probe+1)
        state = {"K": K, "radius": radius, "v": v, "h_array": h_array,
                 "h_probe": h_probe, "pose": pose, "radial_log": radial_log}
        state["first"] = tuple(K*self._first_log(state, j) for j in range(6)) if derivatives else ()
        if not bool(torch.isfinite(K).all()) or any(not bool(torch.isfinite(a).all()) for a in state["first"]):
            raise FloatingPointError("nonfinite CUDA pose kernel/first derivative")
        return state

    def _second(self, kernel, j, k):
        """Construct one K_jk, retaining no second-derivative matrices."""
        if any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer))
               or not 0 <= a < 6 for a in (j, k)):
            raise ValueError("pose second derivative indices must be 0 through 5")
        pose, v, radius = kernel["pose"], kernel["v"], kernel["radius"]
        va, vb = pose["va"][j], pose["va"][k]
        vab = -self._centered_elements@pose["ddR"][j, k].T
        rj, rk = (self._dot_sample_element(v, a)/radius for a in (va, vb))
        rjk = ((va*vb).sum(dim=1)[None, :]+self._dot_sample_element(v, vab)-rj*rk)/radius
        log_second = kernel["radial_log"]*rjk+(self.q_array+self.q_probe+1)*rj*rk/radius**2
        if self.q_array:
            normal, nj, nk = pose["normal"], pose["normal_first"][j], pose["normal_first"][k]
            hj = self._dot_normal(v, nj)+(va@normal)[None, :]
            hk = self._dot_normal(v, nk)+(vb@normal)[None, :]
            hjk = self._dot_normal(v, pose["normal_second"][j, k])
            hjk = hjk+(vb@nj+va@nk+vab@normal)[None, :]
            log_second = log_second+self.q_array*(hjk/kernel["h_array"]-hj*hk/kernel["h_array"]**2)
        if self.q_probe:
            log_second = log_second+self.q_probe*(vab[None, :, 2]/kernel["h_probe"]
                -va[None, :, 2]*vb[None, :, 2]/kernel["h_probe"]**2)
        result = kernel["K"]*(self._first_log(kernel, j)*self._first_log(kernel, k)+log_second)
        self.counts["kernel_second_derivative_calls"] += 1
        if not bool(self.torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite CUDA pose second derivative")
        return result

    def _record_failure(self, operation, x, error):
        self._device_state = None
        self._last = None
        self.counts["gpu_failures"] += 1
        if operation in ("profile_evaluation", "exact_profile_hessian"):
            self.counts["failed_evaluations"] += 1
        failure_key = {"profile_evaluation": "gpu_evaluation_failures",
                       "exact_profile_hessian": "gpu_hessian_failures",
                       "prediction": "gpu_prediction_failures", "validation": "gpu_validation_failures"}[operation]
        self.counts[failure_key] += 1
        geometry, angles = parameters_to_pose(x[:6])
        record = {"operation": operation, "x": x.tolist(), "geometry_mm": geometry.tolist(),
                  "angles_deg": angles.tolist(), "exception_type": type(error).__name__,
                  "reason": str(error), "fallback": "raise", "device": str(self.device)}
        self.failure_ledger.append(record)
        return record

    def evaluate(self, x):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        geometry, angles = parameters_to_pose(x[:6])
        if self._last is not None and np.array_equal(x, self._last.x):
            self.counts["cache_hits"] += 1
            return self._last
        gains = self._gains(x)
        self.counts["evaluations"] += 1
        self._device_state = None
        started = time.perf_counter()
        try:
            self._sync()
            with self.torch.no_grad():
                value = self._evaluate_gpu(x, geometry, angles, gains)
            self._sync()
        except (RuntimeError, ValueError, linalg.LinAlgError, FloatingPointError) as error:
            self._record_failure("profile_evaluation", x, error)
            raise
        finally:
            self.gpu_elapsed_seconds += time.perf_counter()-started
        value.diagnostics["gpu"] = self.runtime_descriptor()
        self._last = value
        return value

    def _evaluate_gpu(self, x, geometry, angles, gains):
        torch = self.torch
        p, n = self.parameter_count, len(self.elements_m)
        tx, pose = self._tensor(x, self.rd), self._pose_state(x)
        cache, state, bs, dbs, diagnostics = {}, {}, {}, {}, {}
        residuals, jacobians = [], []
        data, weighted, penalties = {}, {}, {}
        data_jacobians, weighted_jacobians, penalty_jacobians = {}, {}, {}
        for group, sessions, penalty, penalty_rhs in self._resident_groups:
            blocks, rhs, details, cursor = [], [], [], 0
            for ss in sessions:
                if ss["key"] not in cache:
                    cache[ss["key"]] = self._kernel(ss, pose)
                    self.counts["kernel_calls"] += 1
                    self.counts["gpu_kernel_calls"] += 1
                    self.counts["kernel_derivative_calls"] += 6
                    self.counts["gpu_kernel_derivative_calls"] += 6
                gi = self._gain_indices.get(ss["session"].name)
                gain = torch.ones((), device=self.device, dtype=self.cd) if gi is None else torch.exp(tx[gi]+1j*tx[gi+1])
                weight = ss["w"][:, None]*gain*ss["u"][None, :]
                A = cache[ss["key"]]["K"]*weight
                sl = slice(cursor, cursor+len(ss["session"].field))
                cursor = sl.stop
                blocks.append(A)
                rhs.append(ss["w"]*ss["y"])
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
            active = list(range(6))
            for ss, sl, weight, gi, gain in details:
                if gi is not None:
                    active.extend((gi, gi+1))
                for j in range(6):
                    Bj = cache[ss["key"]]["first"][j]*weight
                    products[sl, j] = Bj@b
                    adjoints[:, j] += Bj.mH@r[sl]
                if gi is not None:
                    products[sl, gi] = B[sl]@b
                    products[sl, gi+1] = 1j*products[sl, gi]
                    adjoint = B[sl].mH@r[sl]
                    adjoints[:, gi] += adjoint
                    adjoints[:, gi+1] -= 1j*adjoint
            z = torch.linalg.solve_triangular(R.mH, adjoints[:, active], upper=False)
            db[:, active] = torch.linalg.solve_triangular(R, z-Q.mH@products[:, active], upper=True)
            J[:, active] = -products[:, active]-B@db[:, active]
            self.counts["linear_derivative_solves"] += len(active)
            Bnorm, rnorm, cnorm = torch.linalg.norm(B), torch.linalg.norm(r), torch.linalg.norm(c)
            scale = torch.maximum(Bnorm*rnorm, np.finfo(float).eps*Bnorm*cnorm).clamp(min=1e-30)
            stationarity = (torch.linalg.norm(B.mH@r)/scale).item()
            if not bool(torch.isfinite(b).all() and torch.isfinite(r).all() and torch.isfinite(J).all()):
                raise FloatingPointError("nonfinite CUDA pose QR coefficients/residual/Jacobian")
            for ss, sl, weight, gi, gain in details:
                name = ss["session"].name
                A = cache[ss["key"]]["K"]*gain*ss["u"][None, :]
                raw = ss["y"]-A@b
                if bool(torch.all(ss["w"] > 0)):
                    rawJ = J[sl]/ss["w"][:, None]
                else:
                    rawJ = -A@db
                    for j in range(6):
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
            residuals.append(r)
            jacobians.append(J)
            state[group.name] = {"b": b, "db": db, "r": r, "J": J,
                "products": products, "adjoints": adjoints, "details": details, "m": m}
        r, J = torch.cat(residuals), torch.cat(jacobians)
        self._device_state = {"x": x.copy(), "groups": state, "kernels": cache, "curvature": None}
        return PoseEvaluation(x=x.copy(), geometry_mm=geometry.copy(), angles_deg=angles.copy(),
            residual=self._numpy(torch.cat((r.real, r.imag))),
            jacobian=self._numpy(torch.cat((J.real, J.imag))), b=bs, db=dbs, gains=gains,
            data_residuals=data, weighted_data_residuals=weighted, penalty_residuals=penalties,
            data_jacobians=data_jacobians, weighted_data_jacobians=weighted_jacobians,
            penalty_jacobians=penalty_jacobians, amplitude_residuals={}, amplitude_jacobians={},
            diagnostics={"groups": diagnostics, "qr_checks": [], "parameter_layout": list(self.parameter_layout),
                "counts": dict(self.counts), "pivot_m": self.pivot_m.tolist(),
                "gauge": "first session g=1 separately in every frequency/chain group"})

    def profile_gradient_hessian(self, x):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        self.evaluate(x)
        started = time.perf_counter()
        try:
            self._sync()
            with self.torch.no_grad():
                result = self._profile_gradient_hessian_cuda(x)
            self._sync()
            return result
        except (RuntimeError, ValueError, linalg.LinAlgError, FloatingPointError) as error:
            self._record_failure("exact_profile_hessian", x, error)
            raise
        finally:
            self.gpu_hessian_seconds += time.perf_counter()-started

    def _profile_gradient_hessian_cuda(self, x):
        evaluation = self.evaluate(x)
        state = self._device_state
        if state is None or not np.array_equal(state["x"], x):
            raise RuntimeError("CUDA pose profile state unavailable; CPU fallback is disabled")
        if state["curvature"] is not None:
            cached = dict(state["curvature"])
            cached["diagnostics"] = {**cached["diagnostics"], "second_kernel_calls": 0,
                                      "additional_first_kernel_calls": 0, "gpu_curvature_cache_hit": True}
            return cached
        torch = self.torch
        initial_second = self.counts["kernel_second_derivative_calls"]
        gradient = torch.zeros(self.parameter_count, device=self.device, dtype=self.rd)
        hessian = torch.zeros((self.parameter_count, self.parameter_count), device=self.device, dtype=self.rd)
        for group in self.groups:
            s = state["groups"][group.name]
            r, rj, products = s["r"][:s["m"]], s["J"][:s["m"]], s["products"][:s["m"]]
            curvature = torch.zeros_like(hessian, dtype=self.cd)
            for j in range(6):
                for k in range(j, 6):
                    # This dictionary lives for one pair only, never all 21.
                    second_cache = {}
                    for ss, sl, weight, gi, gain in s["details"]:
                        if ss["key"] not in second_cache:
                            second_cache[ss["key"]] = self._second(state["kernels"][ss["key"]], j, k)
                        product = (second_cache[ss["key"]]*weight)@s["b"]
                        value = torch.vdot(r[sl], product)
                        curvature[j, k] += value
                        if j != k:
                            curvature[k, j] += value
                    del second_cache
            for ss, sl, weight, gi, gain in s["details"]:
                if gi is not None:
                    for j in range(6):
                        value = torch.vdot(r[sl], products[sl, j])
                        curvature[j, gi] += value
                        curvature[gi, j] += value
                        curvature[j, gi+1] += 1j*value
                        curvature[gi+1, j] += 1j*value
                    value = torch.vdot(r[sl], products[sl, gi])
                    curvature[gi, gi] += value
                    curvature[gi, gi+1] += 1j*value
                    curvature[gi+1, gi] += 1j*value
                    curvature[gi+1, gi+1] -= value
            gradient -= (products.mH@r).real
            hessian -= (products.T@rj.conj()+curvature+s["adjoints"].mH@s["db"]).real
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(hessian).all()):
            raise FloatingPointError("nonfinite CUDA pose profile gradient/Hessian")
        asymmetry = (torch.linalg.norm(hessian-hessian.T)/torch.linalg.norm(hessian).clamp(min=1e-30)).item()
        hessian = .5*(hessian+hessian.T)
        residual_gradient = torch.zeros_like(gradient)
        for native_state in state["groups"].values():
            residual_gradient += (native_state["J"].mH@native_state["r"]).real
        difference = (torch.linalg.norm(gradient-residual_gradient)/torch.linalg.norm(gradient).clamp(min=1e-30)).item()
        result = {"evaluation": evaluation, "gradient": self._numpy(gradient), "hessian": self._numpy(hessian),
            "diagnostics": {"hessian_relative_asymmetry_before_symmetrizing": asymmetry,
                "envelope_gradient_relative_difference": difference,
                "second_kernel_calls": self.counts["kernel_second_derivative_calls"]-initial_second,
                "additional_first_kernel_calls": 0, "hessian_eigenvalues": self._numpy(torch.linalg.eigvalsh(hessian)).tolist(),
                "backend_used": self.backend, "physical_parameter_count": 6,
                "second_derivative_storage": "streamed one coordinate pair; no resident K_jk cache",
                "derivative_scope": "selected stationary branch only; global profile smoothness and optimality not established"}}
        self.counts["gpu_hessian_calls"] += 1
        state["curvature"] = result
        return result

    def predict(self, session, geometry_mm, b, g=1., *, angles_deg):
        b = _complex_array(b, "b", (len(self.elements_m),))
        x = np.zeros(self.parameter_count)
        x[:6] = pose_to_parameters(geometry_mm, angles_deg)
        if not np.isfinite(complex(g)):
            raise ValueError("prediction gain must be finite")
        try:
            with self.torch.no_grad():
                native = self._session(session)
                K = self._kernel(native, self._pose_state(x), derivatives=False)["K"]
                result = complex(g)*(K@(native["u"]*self._tensor(b, self.cd)))
                if not bool(self.torch.isfinite(result).all()):
                    raise FloatingPointError("nonfinite CUDA pose prediction")
                self.counts["gpu_prediction_calls"] += 1
                self.counts["gpu_prediction_kernel_calls"] += 1
                return self._numpy(result)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            self._record_failure("prediction", x, error)
            raise

    def validation_evaluate(self, x, validation_groups, gains=None):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        evaluation = self.evaluate(x)
        if self._device_state is None:
            raise RuntimeError("CUDA pose validation state unavailable; CPU fallback is disabled")
        torch = self.torch
        residuals, jacobians, predictions = [], [], {}
        lookup = {group.name: group for group in self.groups}
        try:
            with torch.no_grad():
                pose = self._pose_state(x)
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
                            cache[native["key"]] = self._kernel(native, pose)
                            self.counts["gpu_validation_kernel_calls"] += 1
                        kernel = cache[native["key"]]
                        gain = complex((gains or {}).get(session.name, evaluation.gains.get(session.name, 1.)))
                        if not np.isfinite(gain):
                            raise ValueError("validation gain must be finite")
                        A = gain*kernel["K"]*native["u"][None, :]
                        prediction = A@state["b"]
                        J = -A@state["db"]
                        for j in range(6):
                            J[:, j] -= gain*(kernel["first"][j]@(native["u"]*state["b"]))
                        if session.name in self._gain_indices and not (gains and session.name in gains):
                            gi = self._gain_indices[session.name]
                            J[:, gi] -= prediction
                            J[:, gi+1] -= 1j*prediction
                        predictions[session.name] = self._numpy(prediction)
                        residuals.append(native["w"]*(native["y"]-prediction))
                        jacobians.append(native["w"][:, None]*J)
                if not residuals:
                    raise ValueError("nonempty validation sessions required")
                r, J = torch.cat(residuals), torch.cat(jacobians)
                if not bool(torch.isfinite(r).all() and torch.isfinite(J).all()):
                    raise FloatingPointError("nonfinite CUDA pose validation residual/Jacobian")
                self.counts["gpu_validation_calls"] += 1
                return {"residual": self._numpy(torch.cat((r.real, r.imag))),
                        "jacobian": self._numpy(torch.cat((J.real, J.imag))),
                        "predictions": predictions, "evaluation": evaluation,
                        "geometry_mm": evaluation.geometry_mm.copy(), "angles_deg": evaluation.angles_deg.copy()}
        except (RuntimeError, ValueError, FloatingPointError) as error:
            self._record_failure("validation", x, error)
            raise
