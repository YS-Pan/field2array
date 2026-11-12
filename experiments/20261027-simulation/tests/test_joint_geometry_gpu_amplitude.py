"""Exact amplitude/outer tensor adapter contracts, explicit CPU by default.

CUDA is used only if the root GPU scheduler authorizes a process with
ARRAY_RECONSTRUCT_GEOMETRY_AMPLITUDE_DEVICE=cuda:0.
"""
from dataclasses import replace
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose
import torch

from array_reconstruct.joint_geometry import (AmplitudeProfileError, FrequencyGroup,
                                             JointProblem, Session, scalar_kernel)
from array_reconstruct.joint_geometry_gpu_amplitude import GPUAmplitudeJointProblem
from array_reconstruct.joint_newton import polish_profile, profile_gradient_hessian

DEVICE = os.environ.get("ARRAY_RECONSTRUCT_GEOMETRY_AMPLITUDE_DEVICE", "cpu")


def example(*, mixed=False, template=True, zero_weights=True):
    rng = np.random.default_rng(84109)
    elements = np.array([[-.011, -.011], [0., -.011], [.011, -.011],
                         [-.0055, 0.], [.0055, 0.], [.0165, 0.]])
    groups = []
    for group_index, frequency in enumerate((12.7e9, 12.1e9)):
        b = rng.uniform(.7, 1.4, 6)*np.exp(.3j*rng.normal(size=6))
        sessions = []
        for index in range(2 if group_index == 0 else 1):
            xx, yy = np.meshgrid(np.linspace(-.05, .05, 7-index), np.linspace(-.04, .04, 7-index))
            xy = np.column_stack((xx.ravel(), yy.ravel()))
            u = np.exp(-1j*elements[:, 0]*index*65.)
            field = np.exp(.12*index+.2j*index)*(scalar_kernel(elements, xy, frequency,
                (62.3, 1.4, -3.2))@(u*b))
            field += .002*(rng.normal(size=len(xy))+1j*rng.normal(size=len(xy)))
            w = np.linspace(.006, .01, len(xy))
            if zero_weights:
                w[::9] = 0
            sessions.append(Session(f"g{group_index}s{index}", frequency, xy, u, field, w))
        groups.append(FrequencyGroup(f"g{group_index}", sessions, .03,
            b0=.2+.05j*rng.normal(size=6),
            template_prior=b+.1j if template else None, template_strength=.011 if template else 0.,
            amplitude_prior=np.abs(b)*np.exp(.06*rng.normal(size=6)),
            amplitude_strength=0. if mixed and group_index == 1 else .04,
            amplitude_stationarity_tol=1e-13))
    cpu = JointProblem(elements, groups, backend="qr")
    tensor = GPUAmplitudeJointProblem.from_problem(cpu, device=DEVICE, fallback="raise")
    x = cpu.initial_parameters((61.4, -.8, -2.9), {"g0s1": 1.13+.17j})
    return cpu, tensor, x


def compare(a, b, tolerance=2e-7):
    assert_allclose(a.residual, b.residual, rtol=tolerance, atol=2e-10)
    assert_allclose(a.jacobian, b.jacobian, rtol=tolerance, atol=2e-10)
    for field in ("b", "db", "data_residuals", "weighted_data_residuals", "penalty_residuals",
                  "data_jacobians", "weighted_data_jacobians", "penalty_jacobians",
                  "amplitude_residuals", "amplitude_jacobians"):
        assert getattr(a, field).keys() == getattr(b, field).keys()
        for key, value in getattr(a, field).items():
            assert_allclose(value, getattr(b, field)[key], rtol=tolerance, atol=2e-10)


@pytest.mark.parametrize("mixed", [False, True])
def test_original_groups_all_residuals_and_exact_stationary_db(mixed):
    cpu, tensor, x = example(mixed=mixed)
    a, b = cpu.evaluate(x), tensor.evaluate(x)
    compare(a, b)
    assert tensor.groups == cpu.groups
    assert [native[0] for native in tensor._resident_groups] == list(cpu.groups)
    assert all(g["amplitude"]["stationarity"] <= 1e-13 for g in b.diagnostics["groups"].values() if g["amplitude"])
    assert not b.diagnostics["gpu"]["amplitude_production_gate_passed"]
    assert tensor.counts["amplitude_local_derivative_columns"] == (5 if mixed else 8)
    for name, group_state in tensor._device_state["groups"].items():
        if group_state["amplitude_solution"] is not None:
            assert group_state["amplitude_solution"].hessian.shape == (12, 12)
            assert group_state["amplitude_solution"].hessian.dtype == torch.float64
    assert tensor.evaluate(x) is b


def test_full_profile_jacobian_fresh_phase_resolves_and_exact_outer_hessian():
    cpu, tensor, x = example(mixed=True)
    a, b = cpu.evaluate(x), tensor.evaluate(x)
    ch, gh = profile_gradient_hessian(cpu, x), profile_gradient_hessian(tensor, x)
    assert_allclose(gh["gradient"], ch["gradient"], rtol=2e-7, atol=2e-10)
    assert_allclose(gh["hessian"], ch["hessian"], rtol=2e-7, atol=2e-9)
    numeric_J, numeric_H = [], []
    h = 1e-5
    for j in range(tensor.parameter_count):
        step = np.eye(tensor.parameter_count)[j]*h
        plus, minus = tensor.evaluate(x+step), tensor.evaluate(x-step)
        numeric_J.append((plus.residual-minus.residual)/(2*h))
        numeric_H.append((plus.jacobian.T@plus.residual-minus.jacobian.T@minus.residual)/(2*h))
    numeric_J, numeric_H = np.column_stack(numeric_J), np.column_stack(numeric_H)
    assert np.linalg.norm(numeric_J-b.jacobian)/np.linalg.norm(b.jacobian) < 1e-5
    assert np.linalg.norm(numeric_H-gh["hessian"])/np.linalg.norm(gh["hessian"]) < 1e-5
    assert np.linalg.norm(gh["hessian"]-b.jacobian.T@b.jacobian) > 1e-5
    assert not gh["diagnostics"]["global_profile_smoothness_established"]


def test_gpu_prediction_and_heldout_derivative_use_full_nonlinear_db():
    cpu, tensor, x = example()
    validation = {"g0": [replace(cpu.groups[0].sessions[1], samples_xy_m=cpu.groups[0].sessions[1].samples_xy_m+.00027)]}
    ca, ta = cpu.validation_evaluate(x, validation), tensor.validation_evaluate(x, validation)
    assert_allclose(ta["residual"], ca["residual"], rtol=2e-7, atol=2e-10)
    assert_allclose(ta["jacobian"], ca["jacobian"], rtol=2e-7, atol=2e-10)
    assert_allclose(ta["predictions"]["g0s1"], ca["predictions"]["g0s1"], rtol=2e-8, atol=2e-8)
    session = validation["g0"][0]
    c, t = cpu.evaluate(x), tensor.evaluate(x)
    assert_allclose(tensor.predict(session, t.geometry_mm, t.b["g0"], .9+.1j),
                    cpu.predict(session, c.geometry_mm, c.b["g0"], .9+.1j), rtol=2e-8, atol=2e-8)


def test_infrastructure_fallback_preserves_full_amplitude_and_template(monkeypatch):
    cpu, tensor, x = example()
    tensor.fallback = "cpu"
    def fail(*args):
        raise RuntimeError("injected tensor infrastructure failure")
    monkeypatch.setattr(tensor, "_evaluate_gpu", fail)
    actual = tensor.evaluate(x)
    compare(cpu.evaluate(x), actual, tolerance=1e-12)
    assert tensor._device_state is None
    assert tensor._cpu_reference.groups == cpu.groups
    assert all(g.amplitude_strength == .04 for g in tensor._cpu_reference.groups)
    assert actual.diagnostics["gpu"]["backend_used"] == "cpu_full_nonlinear_amplitude_fallback"
    assert actual.diagnostics["gpu"]["physical_backend"] == "cpu_full_nonlinear_amplitude"
    assert actual.diagnostics["gpu"]["actual_cuda_physical_arithmetic"] is False
    assert tensor.failure_ledger[-1]["original_nonlinear_priors_preserved"]
    assert_allclose(profile_gradient_hessian(tensor, x)["hessian"],
                    profile_gradient_hessian(cpu, x)["hessian"], rtol=1e-12, atol=1e-12)


def test_failed_stationary_gate_aborts_without_cpu_substitute():
    cpu, _, x = example()
    groups = [replace(g, amplitude_maxiter=0, amplitude_newton_steps=0) for g in cpu.groups]
    tensor = GPUAmplitudeJointProblem(cpu.elements_m, groups, device=DEVICE, fallback="cpu")
    with pytest.raises(AmplitudeProfileError):
        tensor.evaluate(x)
    assert tensor._last is None and tensor._device_state is None
    assert tensor.counts["amplitude_failed_profiles"] == tensor.counts["failed_evaluations"] == 1
    assert tensor.counts["cpu_fallbacks"] == 0
    assert tensor._cpu_reference is None
    assert tensor.failure_ledger[-1]["starts"]
    with pytest.raises(AmplitudeProfileError):
        profile_gradient_hessian(tensor, x)
    assert tensor.counts["cpu_fallbacks"] == 0


def test_paired_outer_optimizer_and_exact_curvature_dispatch():
    cpu, tensor, x = example(mixed=True)
    expected = cpu.fit(start_parameters=x, max_nfev=50)
    actual = tensor.fit(start_parameters=x, max_nfev=50)
    assert expected.success and actual.success
    assert_allclose(actual.geometry_mm, expected.geometry_mm, rtol=0, atol=3e-5)
    assert_allclose(actual.cost, expected.cost, rtol=1e-8, atol=1e-10)
    polished = polish_profile(tensor, actual.x, maxiter=50, gtol=1e-9, xtol=1e-9,
        initial_barrier_parameter=1e-12, barrier_tol=1e-12)
    assert polished.success
    assert polished.stationarity["projected_gradient_inf"] < 1e-7
    assert polished.counts["gpu_hessian_calls"] > 0
    assert tensor.counts["cpu_fallbacks"] == 0


@pytest.mark.skipif(DEVICE != "cpu", reason="CPU smoke guard applies to explicit CPU only")
def test_explicit_cpu_path_never_queries_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit CPU adapter queried CUDA")
    for key in ("is_available", "synchronize", "get_device_properties", "memory_allocated", "memory_reserved",
                "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    _, tensor, x = example(mixed=True)
    tensor.evaluate(x)
    profile_gradient_hessian(tensor, x)
    descriptor = tensor.runtime_descriptor()
    assert descriptor["device"] == "cpu"
    assert not descriptor["actual_cuda_physical_arithmetic"]
    assert descriptor["transfer_counts"]["device_to_host_tensor_transfers"] == 0
