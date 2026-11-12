"""CUDA/CPU mathematical parity, public synthetic arrays only."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")

from array_reconstruct.joint_geometry import FrequencyGroup, JointProblem, Session, scalar_kernel
from array_reconstruct.joint_geometry_gpu import GPUJointProblem
from array_reconstruct.joint_newton import profile_gradient_hessian


def problem(*, q=2., sign=-1, template=0., zero_weights=False):
    rng = np.random.default_rng(43107)
    elements = np.array([[-.011, -.011], [0., -.011], [.011, -.011],
                         [-.0055, 0.], [.0055, 0.], [.0165, 0.]])
    groups = []
    for group_index, frequency in enumerate((12.7e9, 12.1e9)):
        sessions = []
        b = (1+.2*rng.normal(size=6))*np.exp(.3j*rng.normal(size=6))
        for index in range(3 if group_index == 0 else 1):
            side = 7 if index != 2 else 6
            xy = np.stack(np.meshgrid(np.linspace(-.05, .05, side), np.linspace(-.04, .04, side)), axis=-1).reshape(-1, 2)
            u = np.exp(-1j*elements[:, 0]*(index-.5)*100.)
            gain = np.exp(.13*index+.27j*index)
            field = gain*scalar_kernel(elements, xy, frequency, (62.3, 1.4, -3.2), q, sign)@(u*b)
            field += .003*(rng.normal(size=len(xy))+1j*rng.normal(size=len(xy)))
            w = np.linspace(.005, .011, len(xy))
            if zero_weights:
                w[::7] = 0
            sessions.append(Session(f"g{group_index}s{index}", frequency, xy, u, field, w))
        groups.append(FrequencyGroup(f"g{group_index}", sessions, .003,
            b0=.3+.1j*rng.normal(size=6), template_prior=b+.1j, template_strength=template))
    cpu = JointProblem(elements, groups, q=q, sign=sign)
    gpu = GPUJointProblem.from_problem(cpu, fallback="raise")
    x = cpu.initial_parameters((61.4, -.8, -2.9), {"g0s1": 1.15+.18j, "g0s2": .87-.22j})
    return cpu, gpu, x


def compare(a, b, rtol=2e-10, atol=2e-11):
    assert_allclose(a.residual, b.residual, rtol=rtol, atol=atol)
    assert_allclose(a.jacobian, b.jacobian, rtol=rtol, atol=atol)
    for field in ("b", "db", "data_residuals", "weighted_data_residuals", "penalty_residuals",
                  "data_jacobians", "weighted_data_jacobians", "penalty_jacobians"):
        for key, value in getattr(a, field).items():
            assert_allclose(value, getattr(b, field)[key], rtol=rtol, atol=atol)


@pytest.mark.parametrize("q,sign", [(0., -1), (2., -1), (4., 1)])
@pytest.mark.parametrize("template,zero", [(0., False), (.013, True)])
def test_full_evaluation_boundary_template_gain_and_zero_weights(q, sign, template, zero):
    cpu, gpu, x = problem(q=q, sign=sign, template=template, zero_weights=zero)
    for geometry in ((61.4, -.8, -2.9), (50., -10., 10.), (70., 10., -10.)):
        x[:3] = (np.array(geometry)-[60., 0., 0.])/10.
        a, b = cpu.evaluate(x), gpu.evaluate(x)
        compare(a, b)
        assert max(v["stationarity"] for v in b.diagnostics["groups"].values()) < 1e-10
        assert gpu.counts["cpu_fallbacks"] == 0
    assert gpu.evaluate(x) is b
    assert gpu.counts["cache_hits"] == 1
    with pytest.raises(ValueError, match="outside"):
        gpu.evaluate(np.r_[1.0001, x[1:]])


def test_validation_prediction_full_jacobian():
    cpu, gpu, x = problem(template=.01, zero_weights=True)
    validation = {"g0": [replace(cpu.groups[0].sessions[1], samples_xy_m=cpu.groups[0].sessions[1].samples_xy_m+.00031)]}
    for gains in (None, {"g0s1": .93+.13j}):
        a, b = cpu.validation_evaluate(x, validation, gains), gpu.validation_evaluate(x, validation, gains)
        for key in ("residual", "jacobian"):
            assert_allclose(a[key], b[key], rtol=3e-10, atol=2e-11)
        assert_allclose(a["predictions"]["g0s1"], b["predictions"]["g0s1"], rtol=2e-12, atol=2e-11)
    session = validation["g0"][0]
    a, b = cpu.evaluate(x), gpu.evaluate(x)
    assert_allclose(cpu.predict(session, a.geometry_mm, a.b["g0"], .9+.1j),
                    gpu.predict(session, b.geometry_mm, b.b["g0"], .9+.1j), rtol=2e-12, atol=2e-11)


def test_exact_hessian_matches_cpu_and_full_gradient_finite_difference(monkeypatch):
    cpu, gpu, x = problem(template=.01, zero_weights=True)
    a = profile_gradient_hessian(cpu, x)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU kernel called from CUDA Hessian")
    import array_reconstruct.joint_newton as module
    monkeypatch.setattr(module, "scalar_kernel", forbidden)
    monkeypatch.setattr(module, "scalar_kernel_second", forbidden)
    b = profile_gradient_hessian(gpu, x)
    assert_allclose(a["gradient"], b["gradient"], rtol=2e-10, atol=2e-11)
    assert_allclose(a["hessian"], b["hessian"], rtol=2e-9, atol=2e-10)
    h = 1e-5
    finite = np.column_stack([(gpu.evaluate(x+np.eye(len(x))[j]*h).jacobian.T@gpu.evaluate(x+np.eye(len(x))[j]*h).residual-
                               gpu.evaluate(x-np.eye(len(x))[j]*h).jacobian.T@gpu.evaluate(x-np.eye(len(x))[j]*h).residual)/(2*h)
                              for j in range(len(x))])
    assert_allclose(finite, b["hessian"], rtol=2e-6, atol=2e-8)
    assert linalg.norm(b["hessian"]-b["evaluation"].jacobian.T@b["evaluation"].jacobian) > 1e-6


def test_explicit_fallback_keeps_new_x_and_records_failure(monkeypatch):
    cpu, gpu, x = problem()
    gpu.fallback = "cpu"
    gpu.evaluate(x)
    def fail(*args, **kwargs):
        raise RuntimeError("injected CUDA failure")
    monkeypatch.setattr(gpu, "_evaluate_gpu", fail)
    other = x.copy(); other[0] += .03
    a, b = cpu.evaluate(other), gpu.evaluate(other)
    compare(a, b)
    assert gpu._device_state is None
    assert gpu.counts["cpu_fallbacks"] == gpu.counts["gpu_failures"] == 1
    assert b.diagnostics["gpu"]["backend_used"] == "cpu_qr_fallback"
    assert gpu.failure_ledger[0]["x"] == other.tolist()
    assert_allclose(profile_gradient_hessian(gpu, other)["hessian"], profile_gradient_hessian(cpu, other)["hessian"])


def test_strict_failure_rank_guard_and_amplitude_rejection(monkeypatch):
    cpu, gpu, x = problem()
    def fail(*args, **kwargs):
        raise RuntimeError("injected CUDA failure")
    monkeypatch.setattr(gpu, "_evaluate_gpu", fail)
    with pytest.raises(RuntimeError, match="injected"):
        gpu.evaluate(x)
    assert gpu.counts["cpu_fallbacks"] == 0
    group = replace(cpu.groups[0], amplitude_prior=np.ones(6), amplitude_strength=.01)
    with pytest.raises(NotImplementedError, match="amplitude"):
        GPUJointProblem(cpu.elements_m, [group])
    s = Session("deficient", 12.7e9, np.array([[0., 0.]]), np.ones(6), np.ones(1))
    deficient = GPUJointProblem(cpu.elements_m, [FrequencyGroup("d", [s], 0.)], fallback="raise")
    with pytest.raises(linalg.LinAlgError, match="rank deficient"):
        deficient.evaluate(deficient.initial_parameters())


def test_paired_outer_fit_and_kernel_sharing():
    cpu, gpu, x = problem(template=.005)
    a, b = cpu.fit(start_parameters=x, max_nfev=100), gpu.fit(start_parameters=x, max_nfev=100)
    assert a.success and b.success
    assert_allclose(a.geometry_mm, b.geometry_mm, rtol=0, atol=2e-6)
    assert_allclose(a.cost, b.cost, rtol=1e-10, atol=1e-13)
    compare(a.evaluation, b.evaluation, rtol=2e-7, atol=1e-8)
    # g0s0/g0s1 truly share a frequency/grid; g0s2 is a second window.
    fresh = GPUJointProblem.from_problem(cpu, fallback="raise")
    fresh.evaluate(x)
    assert fresh.counts["kernel_calls"] == 3
    assert fresh.counts["kernel_derivative_calls"] == 9
