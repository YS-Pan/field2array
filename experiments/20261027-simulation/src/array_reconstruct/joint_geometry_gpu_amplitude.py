"""Experimental exact nonlinear-amplitude tensor profile and outer curvature.

This separate adapter leaves the frozen linear CUDA backend untouched. A local
stationary inner branch and its exact derivatives do not establish smoothness
of the best-of-starts profile. In particular, the known coarse/weak-ridge basin
switch failure remains excluded from production; this module has no adoption
or scientific-gate authority.

``device='cpu'`` is an explicitly labelled tensor smoke-test path. It never
queries or initializes CUDA. CUDA execution requires the caller's GPU slot.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
from scipy import linalg
import torch

from .joint_amplitude_gpu import (solve_amplitude_gpu, implicit_derivative_gpu,
                                  _COUNT_KEYS)
from .joint_geometry import (AmplitudeProfileError, JointProblem, ProfileEvaluation,
                             _real_array, parameters_to_geometry)
from .joint_geometry_gpu import GPUJointProblem


class GPUAmplitudeJointProblem(GPUJointProblem):
    """Complete relative-amplitude profile with the original groups preserved.

    There is no linear substitute for an amplitude prior: the full original
    real stationary Hessian supplies db and every residual Jacobian column.
    Infrastructure failures may explicitly recompute the same original problem
    with CPU QR/amplitude, if requested. A failed inner stationarity/curvature
    gate always aborts the outer evaluation and retains its complete ledger.
    """
    def __init__(self, elements_m, groups, q=2., sign=-1, *, device="cuda:0",
                 fallback="raise", **cpu_options):
        if fallback not in ("cpu", "raise"):
            raise ValueError("fallback must be cpu or raise")
        cpu_options.pop("backend", None)
        # Validate the *original* groups directly. No copied group has its prior
        # removed, even transiently, and CPU fallback sees exactly these groups.
        JointProblem.__init__(self, elements_m, groups, q, sign, backend="qr", **cpu_options)
        if not any(group.amplitude_strength > 0 for group in self.groups):
            raise ValueError("at least one nonlinear amplitude prior is required")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only explicit cpu and cuda tensor devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("an available CUDA device is required")
            if self.device.index is None:
                self.device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.fallback = fallback
        self.backend = "torch_"+self.device.type+"_nonlinear_amplitude_complex128"
        self.cd, self.rd = torch.complex128, torch.float64
        self.counts.update(gpu_failures=0, cpu_fallbacks=0, gpu_qr_factorizations=0,
            gpu_hessian_calls=0, kernel_second_derivative_calls=0,
            gpu_prediction_calls=0, gpu_validation_calls=0, amplitude_failed_profiles=0,
            amplitude_local_derivative_columns=0, outer_hessian_scalar_reads=0)
        self.counts.update({"amplitude_tensor_"+key: 0 for key in _COUNT_KEYS})
        self.transfer_counts = {key: 0 for key in (
            "host_to_device_tensor_transfers", "host_to_device_tensor_bytes",
            "device_to_host_tensor_transfers", "device_to_host_tensor_bytes",
            "device_to_host_scalar_transfers", "device_to_host_scalar_bytes",
            "host_scalar_reads", "cpu_tensor_materializations")}
        self._cpu_reference = None
        self._device_state = None
        self._last_backend_used = None
        self._resident_sessions = {}
        self.gpu_elapsed_seconds = self.gpu_hessian_seconds = 0.
        self._sync(); started = time.perf_counter()
        self._elements = self._tensor(self.elements_m[:, :2], self.rd)
        self._eye = torch.eye(len(self.elements_m), device=self.device, dtype=self.cd)
        self._resident_groups, self._resident_amplitudes, self._resident_templates = [], {}, {}
        for group in self.groups:
            native = [self._session(session) for session in group.sessions]
            penalty = group.lambda_reg*self._eye
            rhs = group.lambda_reg*self._tensor(group.b0, self.cd)
            if group.template_strength > 0:
                a = self._tensor(group.template_prior, self.cd)
                self._resident_templates[group.name] = a
                P = self._eye-a[:, None]*a.conj()[None, :]/torch.vdot(a, a).real
                penalty = torch.cat((penalty, group.template_strength*P))
                rhs = torch.cat((rhs, torch.zeros_like(rhs)))
            if group.amplitude_strength > 0:
                self._resident_amplitudes[group.name] = self._tensor(group.amplitude_prior, self.rd)
            self._resident_groups.append((group, native, penalty, rhs))
        self._sync(); self.upload_setup_seconds = time.perf_counter()-started
        self.amplitude_evaluation_ledger = []

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tensor(self, array, dtype):
        value = np.array(array, copy=True)
        if self.device.type == "cuda":
            self.transfer_counts["host_to_device_tensor_transfers"] += 1
            self.transfer_counts["host_to_device_tensor_bytes"] += value.nbytes
        else:
            self.transfer_counts["cpu_tensor_materializations"] += 1
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def _numpy(self, tensor):
        if self.device.type == "cuda":
            self.transfer_counts["device_to_host_tensor_transfers"] += 1
            self.transfer_counts["device_to_host_tensor_bytes"] += tensor.numel()*tensor.element_size()
        else:
            self.transfer_counts["cpu_tensor_materializations"] += 1
        return tensor.detach().cpu().numpy().copy()

    def _item(self, tensor):
        self.transfer_counts["host_scalar_reads"] += 1
        if self.device.type == "cuda":
            self.transfer_counts["device_to_host_scalar_transfers"] += 1
            self.transfer_counts["device_to_host_scalar_bytes"] += tensor.element_size()
        return tensor.detach().item()

    def runtime_descriptor(self):
        if self.device.type == "cuda":
            descriptor = super().runtime_descriptor()
        else:
            descriptor = {"backend": self.backend, "device": str(self.device),
                "device_name": "explicit CPU tensor smoke path", "complex_dtype": "complex128",
                "real_dtype": "float64", "torch": torch.__version__, "cuda_runtime": None,
                "actual_cuda_physical_arithmetic": False, "torch_cpu_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "upload_setup_seconds": self.upload_setup_seconds,
                "synchronized_evaluation_seconds": self.gpu_elapsed_seconds,
                "synchronized_hessian_seconds": self.gpu_hessian_seconds,
                "fallback_policy": self.fallback, "counts": dict(self.counts),
                "cpu_fallback_counts": dict(self._cpu_reference.counts) if self._cpu_reference is not None else {},
                "failure_ledger": list(self.failure_ledger)}
        last_cuda = (self.device.type == "cuda" and self._last_backend_used == self.backend
                     if self._last_backend_used is not None else None)
        descriptor.update(transfer_counts=dict(self.transfer_counts),
            configured_physical_backend="torch_"+self.device.type,
            physical_backend=("cpu_full_nonlinear_amplitude" if self._last_backend_used == "cpu_full_nonlinear_amplitude_fallback"
                              else "torch_"+self.device.type),
            actual_cuda_physical_arithmetic=last_cuda,
            last_evaluation_backend_used=self._last_backend_used,
            amplitude_derivative_scope="exact selected local stationary branch; best-of-starts smoothness unestablished",
            amplitude_production_gate_passed=False,
            amplitude_ledger_entries=len(self.amplitude_evaluation_ledger),
            fallback_scope="infrastructure only, using original full priors; failed inner stationarity/curvature is never bypassed",
            transfer_scope="profile/prediction/validation tensor copies and profile scalar reads; inner solver transfers are separately included in amplitude_tensor_* counters; inherited exact-H scalar reads are counted explicitly")
        return descriptor

    def evaluate(self, x):
        x = _real_array(x, "outer parameters", (self.parameter_count,))
        if self._last is not None and np.array_equal(x, self._last.x):
            self.counts["cache_hits"] += 1
            return self._last
        geometry, gains = parameters_to_geometry(x[:3]), self._gains(x)
        self.counts["evaluations"] += 1
        self._device_state = None
        started = time.perf_counter()
        try:
            self._sync()
            with torch.no_grad():
                value = self._evaluate_gpu(x, geometry, gains)
            self._sync()
            self._last_backend_used = self.backend
        except AmplitudeProfileError:
            self._last = self._device_state = None
            self.counts["failed_evaluations"] += 1
            self.counts["amplitude_failed_profiles"] += 1
            self._last_backend_used = "failed_stationary_profile"
            raise
        except (RuntimeError, ValueError, linalg.LinAlgError, FloatingPointError) as error:
            self._last = self._device_state = None
            self.counts["failed_evaluations"] += 1
            self.counts["gpu_failures"] += 1
            record = {"x": x.tolist(), "geometry_mm": geometry.tolist(), "operation": "profile",
                "exception_type": type(error).__name__, "reason": str(error), "fallback": self.fallback,
                "device": str(self.device), "original_nonlinear_priors_preserved": True}
            self.failure_ledger.append(record)
            self._last_backend_used = "failed_tensor_infrastructure"
            if self.fallback == "raise":
                raise
            if self._cpu_reference is None:
                self._cpu_reference = JointProblem(self.elements_m, self.groups, self.q, self.sign, backend="qr")
            value = self._cpu_reference.evaluate(x)
            self.counts["cpu_fallbacks"] += 1
            self._last_backend_used = "cpu_full_nonlinear_amplitude_fallback"
            value.diagnostics["gpu"] = {"backend_used": "cpu_full_nonlinear_amplitude_fallback", "failure": record}
        finally:
            self.gpu_elapsed_seconds += time.perf_counter()-started
        value.diagnostics.setdefault("gpu", {}).update(self.runtime_descriptor())
        self._last = value
        return value

    def profile_gradient_hessian(self, x):
        result = super().profile_gradient_hessian(x)
        if result["diagnostics"].get("gpu_hessian_fallback"):
            self._last_backend_used = "cpu_full_nonlinear_amplitude_fallback"
            gpu = result["evaluation"].diagnostics.setdefault("gpu", {})
            gpu.update(self.runtime_descriptor())
            gpu["backend_used"] = "cpu_full_nonlinear_amplitude_fallback"
            result["diagnostics"]["backend_used"] = "cpu_full_nonlinear_amplitude_fallback"
            result["diagnostics"]["original_nonlinear_priors_preserved"] = True
        return result

    def _amplitude_diagnostic(self, group, solution, x, geometry):
        records = []
        coefficient_scale = max(float(self._item(torch.linalg.norm(solution.b))),
                                group.amplitude_epsilon*np.sqrt(len(solution.b)))
        for index, start in enumerate(solution.starts):
            gap = float(start.objective-solution.objective)
            record = asdict(start)
            record.update(selected=index == solution.chosen_start,
                objective_gap_from_selected=gap,
                objective_relative_gap_from_selected=gap/max(abs(solution.objective), 1e-30),
                relative_b_distance_from_selected=float(self._item(torch.linalg.norm(
                    solution.start_coefficients[:, index]-solution.b)))/coefficient_scale)
            records.append(record)
        objectives = [start.objective for start in solution.starts if start.converged]
        return {"group": group.name, "geometry_mm": geometry.tolist(), "x": x.tolist(),
            "success": solution.success, "stationarity": solution.stationarity,
            "stationarity_tol": group.amplitude_stationarity_tol,
            "hessian_rcond": solution.hessian_rcond,
            "hessian_positive_definite": solution.hessian_positive_definite,
            "chosen_start": solution.chosen_start, "selected_objective": solution.objective,
            "starts": records, "stationary_start_count": len(objectives),
            "converged_objective_range": [float(min(objectives)), float(max(objectives))] if objectives else None,
            "max_converged_relative_b_spread": solution.max_converged_relative_b_spread,
            "has_failed_lower_objective_start": solution.has_failed_lower_objective_start,
            "initialization": "augmented QR at this exact outer point, then fixed fresh phase starts; no cross-evaluation warm starts",
            "random_seed": group.amplitude_random_seed, "n_starts": group.amplitude_n_starts,
            "strength": group.amplitude_strength, "epsilon": group.amplitude_epsilon,
            "branch_selection": "lowest objective among converged stationary starts; failed and alternate starts retained",
            "branch_global_optimality_established": False, "global_profile_smoothness_established": False,
            "profile_derivative_scope": "full real Hessian of selected local stationary branch only; no derivative across best-of-starts basin switches",
            "penalty": "strength * center(log(sqrt(abs(b)^2+epsilon^2))-log(relative_amplitudes)); phase unconstrained",
            "tensor_solver": solution.gpu_diagnostics}

    def _evaluate_gpu(self, x, geometry, gains):
        p, n = self.parameter_count, len(self.elements_m)
        tx = self._tensor(x, self.rd)
        cache, state, bs, dbs, diagnostics = {}, {}, {}, {}, {}
        residuals, jacobians, amp_r, amp_J = [], [], {}, {}
        data, weighted, penalties = {}, {}, {}
        data_J, weighted_J, penalty_J = {}, {}, {}
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
                sl = slice(cursor, cursor+len(ss["session"].field)); cursor = sl.stop
                blocks.append(cache[ss["key"]]["K"]*weight)
                rhs.append(ss["w"]*ss["y"])
                details.append((ss, sl, weight, gi, gain))
            m = cursor
            B, c = torch.cat(blocks+[penalty]), torch.cat(rhs+[penalty_rhs])
            k = len(c)
            self.counts["linear_factorizations"] += 1
            self.counts["gpu_qr_factorizations"] += 1
            if B.shape[0] < n:
                raise linalg.LinAlgError("augmented system is rank deficient")
            Q, R = torch.linalg.qr(B, mode="reduced")
            diagonal = torch.abs(torch.diagonal(R))
            rank_threshold = np.finfo(float).eps*max(B.shape)*torch.max(diagonal)
            if self._item(torch.min(diagonal) <= rank_threshold):
                raise linalg.LinAlgError("augmented system is rank deficient; positive fixed regularization required")
            linear_b = torch.linalg.solve_triangular(R, (Q.mH@c)[:, None], upper=True)[:, 0]
            amplitude_active = group.amplitude_strength > 0
            diagnostic, solution = None, None
            if amplitude_active:
                solution = solve_amplitude_gpu(B, c, 0., self._resident_amplitudes[group.name],
                    group.amplitude_strength, initial=linear_b, epsilon=group.amplitude_epsilon,
                    n_starts=group.amplitude_n_starts, random_seed=group.amplitude_random_seed,
                    maxiter=group.amplitude_maxiter, newton_steps=group.amplitude_newton_steps,
                    stationarity_tol=group.amplitude_stationarity_tol,
                    min_hessian_rcond=group.amplitude_min_hessian_rcond, device=self.device)
                self.counts["amplitude_solves"] += 1
                self.counts["amplitude_starts"] += len(solution.starts)
                self.counts["amplitude_nfev"] += sum(start.nfev for start in solution.starts)
                self.counts["amplitude_njev"] += sum(start.njev for start in solution.starts)
                self.counts["amplitude_newton_steps"] += sum(start.newton_steps for start in solution.starts)
                for key, value in solution.gpu_diagnostics["counts"].items():
                    self.counts["amplitude_tensor_"+key] += value
                diagnostic = self._amplitude_diagnostic(group, solution, x, geometry)
                self.amplitude_evaluation_ledger.append(diagnostic)
                if not solution.success:
                    self.failure_ledger.append(diagnostic)
                    raise AmplitudeProfileError(diagnostic)
                b, r = solution.b, -solution.data_residual
            else:
                b, r = linear_b, c-B@linear_b
            products = torch.zeros((k, p), device=self.device, dtype=self.cd)
            adjoints = torch.zeros((n, p), device=self.device, dtype=self.cd)
            active = list(range(3))
            for ss, sl, weight, gi, gain in details:
                if gi is not None:
                    active.extend((gi, gi+1))
            def derivatives():
                for j in active:
                    matrix = torch.zeros_like(B)
                    for ss, sl, weight, gi, gain in details:
                        if j < 3:
                            matrix[sl] = cache[ss["key"]]["first"][j]*weight
                        elif j == gi:
                            matrix[sl] = B[sl]
                        elif gi is not None and j == gi+1:
                            matrix[sl] = 1j*B[sl]
                    products[:, j] = matrix@b
                    adjoints[:, j] = matrix.mH@r
                    yield matrix
            db = torch.zeros((n, p), device=self.device, dtype=self.cd)
            J = torch.zeros((k, p), device=self.device, dtype=self.cd)
            if amplitude_active:
                active_db, active_J = implicit_derivative_gpu(solution, derivatives())
                db[:, active] = active_db
                J[:, active] = -(active_J[:k]+1j*active_J[k:2*k])
                amplitude_J = torch.zeros((n, p), device=self.device, dtype=self.rd)
                amplitude_J[:, active] = -active_J[-n:]
                amp_r[group.name], amp_J[group.name] = -solution.amplitude_residual, amplitude_J
                self.counts["amplitude_derivative_solves"] += len(active)
                self.counts["amplitude_local_derivative_columns"] += len(active)
                for call in solution.gpu_diagnostics.get("implicit_derivative_calls", []):
                    for key, value in call["counts"].items():
                        self.counts["amplitude_tensor_"+key] += value
                stationarity = solution.stationarity
                normalization = "||gradient_full_real_objective||_inf / max(1, ||stack(B^H c)||_inf)"
            else:
                for _ in derivatives():
                    pass
                z = torch.linalg.solve_triangular(R.mH, adjoints[:, active], upper=False)
                db[:, active] = torch.linalg.solve_triangular(R, z-Q.mH@products[:, active], upper=True)
                J[:, active] = -products[:, active]-B@db[:, active]
                self.counts["linear_derivative_solves"] += len(active)
                Bnorm = torch.linalg.norm(B)
                scale = torch.maximum(Bnorm*torch.linalg.norm(r), np.finfo(float).eps*Bnorm*torch.linalg.norm(c)).clamp(min=1e-30)
                stationarity = float(self._item(torch.linalg.norm(B.mH@r)/scale))
                normalization = "||B^H r||_2 / max(||B||_F ||r||_2, eps64 ||B||_F ||c||_2, 1e-30)"
            if not self._item(torch.isfinite(b).all() & torch.isfinite(r).all() & torch.isfinite(db).all() & torch.isfinite(J).all()):
                raise FloatingPointError("nonfinite exact amplitude profile or implicit derivatives")
            for ss, sl, weight, gi, gain in details:
                name = ss["session"].name
                A = cache[ss["key"]]["K"]*gain*ss["u"][None, :]
                raw = ss["y"]-A@b
                if self._item(torch.all(ss["w"] > 0)):
                    rawJ = J[sl]/ss["w"][:, None]
                else:
                    rawJ = -A@db
                    for j in range(3):
                        rawJ[:, j] -= gain*(cache[ss["key"]]["first"][j]@(ss["u"]*b))
                    if gi is not None:
                        rawJ[:, gi] -= A@b; rawJ[:, gi+1] -= 1j*(A@b)
                data[name], weighted[name] = self._numpy(raw), self._numpy(r[sl])
                data_J[name], weighted_J[name] = self._numpy(rawJ), self._numpy(J[sl])
            bs[group.name], dbs[group.name] = self._numpy(b), self._numpy(db)
            penalties[group.name], penalty_J[group.name] = self._numpy(r[m:]), self._numpy(J[m:])
            template_scale = None
            if group.template_strength > 0:
                a = self._resident_templates[group.name]
                value = self._item(torch.vdot(a, b)/torch.vdot(a, a))
                template_scale = [value.real, value.imag]
            diagnostics[group.name] = {"backend_requested": self.backend, "backend_used": self.backend,
                "linear_initialization_backend": "tensor_complex128_augmented_QR",
                "normal_condition_h_1": None, "normal_condition_max": self.normal_condition_max,
                "fallback_reason": None, "qr_verification": None, "stationarity": stationarity,
                "stationarity_normalization": normalization, "lambda_reg": group.lambda_reg,
                "inner_solver": "nonlinear relative amplitude, original full real stationary Hessian" if amplitude_active else "complex linear augmented QR",
                "amplitude": diagnostic, "penalty": "fixed physical lambda*(b-b0), L=I",
                "reference_session": group.sessions[0].name, "reference_gain": [1., 0.],
                "template_strength": group.template_strength, "template_scale": template_scale,
                "template_penalty": "mu*(I-aa^H/(a^Ha))*b" if group.template_strength > 0 else None,
                "data_residual_norm": float(self._item(torch.linalg.norm(r[:m]))),
                "penalty_residual_norm": float(self._item(torch.linalg.norm(r[m:]))),
                "qr_rank_diagonal_min": float(self._item(torch.min(diagonal))),
                "qr_rank_threshold": float(self._item(rank_threshold))}
            residuals.append(r); jacobians.append(J)
            state[group.name] = {"b": b, "db": db, "r": r, "J": J, "products": products,
                "adjoints": adjoints, "details": details, "m": m, "amplitude_solution": solution}
        r, J = torch.cat(residuals), torch.cat(jacobians)
        amplitude_r = torch.cat(list(amp_r.values()))
        amplitude_J = torch.cat(list(amp_J.values()))
        self._device_state = {"x": x.copy(), "groups": state, "kernels": cache, "curvature": None}
        return ProfileEvaluation(x.copy(), geometry,
            self._numpy(torch.cat((r.real, r.imag, amplitude_r))),
            self._numpy(torch.cat((J.real, J.imag, amplitude_J))), bs, dbs, gains, data, weighted,
            penalties, data_J, weighted_J, penalty_J,
            {key: self._numpy(value) for key, value in amp_r.items()},
            {key: self._numpy(value) for key, value in amp_J.items()},
            {"groups": diagnostics, "qr_checks": [], "parameter_layout": list(self.parameter_layout),
             "counts": dict(self.counts), "gauge": "first session g=1 separately in each group",
             "derivative_scope": "exact selected local stationary amplitude branch; full best-of-starts profile smoothness unestablished"})

    def _profile_gradient_hessian_cuda(self, x):
        # The frozen parent's envelope formula uses actual b/db/r/rj/products/
        # adjoints from the nonlinear stationary solution. It does not assume
        # B^H r=0 and uses no normal-equation derivative or second db.
        cached = self._device_state is not None and self._device_state.get("curvature") is not None
        result = super()._profile_gradient_hessian_cuda(x)
        if not cached and self._device_state is not None:
            self.counts["outer_hessian_scalar_reads"] += 2
            self.transfer_counts["host_scalar_reads"] += 2
            if self.device.type == "cuda":
                self.transfer_counts["device_to_host_scalar_transfers"] += 2
                self.transfer_counts["device_to_host_scalar_bytes"] += 16
        result["diagnostics"].update(amplitude_derivative_scope="exact selected local branch only",
                                     global_profile_smoothness_established=False)
        return result


def make_gpu_amplitude_problem(data, lambda_reg, *, model="M2", amplitude_strength,
                               amplitude_n_starts=3, device="cuda:0", fallback="raise", **kwargs):
    """Experimental public-fit-data factory; no truth or gate bypass is accepted."""
    from .joint_virtual import make_problem
    kwargs.pop("backend", None)
    branch = kwargs.pop("branch", "Csoft_amplitude")
    if branch != "Csoft_amplitude":
        raise ValueError("the amplitude tensor factory requires Csoft_amplitude")
    problem = make_problem(data, lambda_reg, model=model, branch="Csoft_amplitude", backend="qr",
        amplitude_strength=amplitude_strength, amplitude_n_starts=amplitude_n_starts, **kwargs)
    return GPUAmplitudeJointProblem.from_problem(problem, device=device, fallback=fallback)
