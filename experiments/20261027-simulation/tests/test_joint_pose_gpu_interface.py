"""Small CPU execution harness for CUDA equations; never a CUDA qualification.

The production constructor only accepts a CUDA device. Tests deliberately bypass
that constructor and run its tensor equations on CPU, so this file neither
probes nor allocates GPU resources and reads no production arrays.
"""
import inspect

import numpy as np
import pytest

from array_reconstruct.joint_geometry import FrequencyGroup, Session, scalar_kernel
from array_reconstruct.joint_pose_cpu import PoseJointProblem
from array_reconstruct.joint_pose_gpu import GPUPoseJointProblem
from array_reconstruct.joint_pose_kernel import (parameters_to_pose,
    scalar_pose_kernel, scalar_pose_kernel_second)


class CPUExecutionHarness(GPUPoseJointProblem):
    def __init__(self, elements, groups, **options):
        torch = pytest.importorskip("torch")
        PoseJointProblem.__init__(self, elements, groups, **options)
        self.torch, self.device = torch, torch.device("cpu")
        self.cd, self.rd = torch.complex128, torch.float64
        self.fallback = "raise"
        self.backend = "cpu_execution_harness_for_cuda_equations_not_gpu_qualified"
        self._initialize_device_counters()
        self._upload_resident_state()

    def _sync(self):
        pass

    def runtime_descriptor(self):
        return {"backend": self.backend, "device": "cpu", "cuda_qualification": False,
                "counts": dict(self.counts), "failure_ledger": list(self.failure_ledger)}


def make_toy(*, template=True, zero_weight=False):
    # A translated centroid catches accidental rotations around the world origin.
    elements = np.array([[-.021, -.018], [.012, -.018], [-.021, .015], [.012, .015]])
    xy = np.array([(x, y) for x in np.linspace(-.038, .04, 5)
                   for y in np.linspace(-.031, .033, 4)])
    groups = []
    for k, frequency in enumerate((8.8e9, 9.1e9, 9.4e9)):
        sessions = []
        response = (1+np.arange(4)*.025)*np.exp(1j*np.arange(4)*.03)
        for j in range(2):
            steering = np.exp(1j*np.arange(4)*(.21*j+.01*k))
            kernel = scalar_pose_kernel(elements, xy, frequency, [61., .4, -.6], [.24, -.31, .7])
            field = (1+.04*j)*np.exp(.025j*j)*(kernel@(steering*response))
            field += .01*np.exp(1j*np.arange(len(xy))*.19)
            weights = np.linspace(.8, 1.1, len(xy))
            if zero_weight:
                weights[0] = 0.
            sessions.append(Session(f"s{k}_{j}", frequency, xy, steering, field, weights))
        groups.append(FrequencyGroup(f"f{k}", sessions, .4, b0=np.ones(4),
            template_prior=np.ones(4), template_strength=.25 if template else 0.))
    return elements, groups


@pytest.mark.parametrize("q_array,q_probe", [(2., 0.), (1.25, .75), (0., 2.), (0., 0.)])
def test_small_tensor_kernel_and_all_21_pairs_match_cpu(q_array, q_probe):
    elements, groups = make_toy()
    problem = CPUExecutionHarness(elements, groups, q_array=q_array, q_probe=q_probe)
    x = problem.initial_parameters([62., -.7, 1.2], angles_deg=[.3, -.2, 1.1])
    geometry, angles = parameters_to_pose(x[:6])
    session = groups[0].sessions[0]
    state = problem._kernel(problem._session(session), problem._pose_state(x))
    K, first = scalar_pose_kernel(elements, session.samples_xy_m, session.frequency_hz,
        geometry, angles, q_array=q_array, q_probe=q_probe, derivatives=True)
    np.testing.assert_allclose(problem._numpy(state["K"]), K, rtol=3e-14, atol=3e-13)
    np.testing.assert_allclose(np.stack([problem._numpy(a) for a in state["first"]]), first,
                               rtol=3e-13, atol=3e-12)
    for j in range(6):
        for k in range(j, 6):
            expected = scalar_pose_kernel_second(elements, session.samples_xy_m,
                session.frequency_hz, geometry, angles, q_array=q_array, q_probe=q_probe, j=j, k=k)
            np.testing.assert_allclose(problem._numpy(problem._second(state, j, k)), expected,
                                       rtol=1e-12, atol=2e-11)
    assert "second" not in state
    assert problem.counts["kernel_second_derivative_calls"] == 21


def test_zero_pose_legacy_kernel_and_translation_derivative_regression():
    elements, groups = make_toy()
    problem = CPUExecutionHarness(elements, groups)
    session = groups[0].sessions[0]
    x = problem.initial_parameters([62., -.7, 1.2], angles_deg=[0., 0., 0.])
    state = problem._kernel(problem._session(session), problem._pose_state(x))
    args = elements, session.samples_xy_m, session.frequency_hz, [62., -.7, 1.2]
    np.testing.assert_allclose(problem._numpy(state["K"]), scalar_kernel(*args), rtol=3e-14, atol=3e-13)
    for j in range(3):
        np.testing.assert_allclose(problem._numpy(state["first"][j]),
            scalar_kernel(*args, derivative=j), rtol=3e-13, atol=3e-12)
    with pytest.raises(ValueError, match="indices"):
        problem._second(state, True, 0)
    with pytest.raises(ValueError, match="indices"):
        problem._second(state, 1., 0)


@pytest.mark.parametrize("template,zero_weight", [(False, False), (True, True)])
def test_small_p12_profile_and_hessian_match_cpu(template, zero_weight):
    elements, groups = make_toy(template=template, zero_weight=zero_weight)
    cpu, tensor = PoseJointProblem(elements, groups), CPUExecutionHarness(elements, groups)
    x = cpu.initial_parameters([62., -.7, 1.2], angles_deg=[.3, -.2, 1.1],
                              gains={"s0_1": 1.02+.03j, "s1_1": .97-.02j, "s2_1": 1.01+.01j})
    a, b = cpu.evaluate(x), tensor.evaluate(x)
    assert tensor.parameter_count == 12
    assert tuple(tensor._gain_indices.values()) == (6, 8, 10)
    np.testing.assert_array_equal(b.angles_deg, [.3, -.2, 1.1])
    np.testing.assert_allclose(b.residual, a.residual, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(b.jacobian, a.jacobian, rtol=3e-10, atol=3e-10)
    for group in groups:
        np.testing.assert_allclose(b.b[group.name], a.b[group.name], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(b.db[group.name], a.db[group.name], rtol=1e-10, atol=1e-11)
        for session in group.sessions:
            np.testing.assert_allclose(b.data_jacobians[session.name], a.data_jacobians[session.name],
                                       rtol=3e-10, atol=3e-10)
    h_cpu, h_tensor = cpu.profile_gradient_hessian(x), tensor.profile_gradient_hessian(x)
    np.testing.assert_allclose(h_tensor["gradient"], h_cpu["gradient"], rtol=1e-9, atol=3e-9)
    np.testing.assert_allclose(h_tensor["hessian"], h_cpu["hessian"], rtol=2e-9, atol=5e-8)
    assert h_tensor["diagnostics"]["second_kernel_calls"] == 63  # shared grid per frequency
    assert tensor.profile_gradient_hessian(x)["diagnostics"]["second_kernel_calls"] == 0
    assert tensor.counts["cpu_fallbacks"] == 0
    assert all("second" not in s for s in tensor._device_state["kernels"].values())
    # Mixed angle/gain entries must not be accidentally omitted.
    assert np.linalg.norm(h_tensor["hessian"][3:6, 6:]) > 0


def test_predict_and_validation_require_and_preserve_angles():
    elements, groups = make_toy()
    cpu, tensor = PoseJointProblem(elements, groups), CPUExecutionHarness(elements, groups)
    x = cpu.initial_parameters([62., -.7, 1.2], angles_deg=[.3, -.2, 1.1])
    a, b = cpu.evaluate(x), tensor.evaluate(x)
    session = groups[0].sessions[1]
    with pytest.raises(TypeError):
        tensor.predict(session, b.geometry_mm, b.b["f0"])
    expected = cpu.predict(session, a.geometry_mm, a.b["f0"], angles_deg=a.angles_deg)
    actual = tensor.predict(session, b.geometry_mm, b.b["f0"], angles_deg=b.angles_deg)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-11)
    for overrides in (None, {"s0_1": .98+.02j}):
        ac = cpu.validation_evaluate(x, {"f0": [session]}, overrides)
        at = tensor.validation_evaluate(x, {"f0": [session]}, overrides)
        np.testing.assert_allclose(at["residual"], ac["residual"], rtol=2e-10, atol=3e-11)
        np.testing.assert_allclose(at["jacobian"], ac["jacobian"], rtol=2e-10, atol=3e-10)
        np.testing.assert_array_equal(at["angles_deg"], b.angles_deg)


def test_public_constructor_rejects_fallback_and_legacy_problem():
    elements, groups = make_toy()
    with pytest.raises(ValueError, match="fallback='raise'"):
        GPUPoseJointProblem(elements, groups, fallback="cpu")
    with pytest.raises(ValueError, match="augmented QR"):
        GPUPoseJointProblem(elements, groups, backend="normal")
    with pytest.raises(TypeError, match="PoseJointProblem"):
        GPUPoseJointProblem.from_problem(object())
    signature = inspect.signature(GPUPoseJointProblem.predict)
    assert signature.parameters["angles_deg"].default is inspect.Parameter.empty


def test_failed_tensor_operation_records_pose_and_never_falls_back():
    elements, groups = make_toy()
    problem = CPUExecutionHarness(elements, groups)
    x = problem.initial_parameters(angles_deg=[.2, -.1, .5])
    def fail(*args):
        raise RuntimeError("deliberate tensor failure")
    problem._evaluate_gpu = fail
    with pytest.raises(RuntimeError, match="deliberate tensor failure"):
        problem.evaluate(x)
    assert problem._last is None and problem._device_state is None
    assert problem.counts["gpu_failures"] == 1
    assert problem.counts["cpu_fallbacks"] == 0
    record = problem.failure_ledger[-1]
    assert record["operation"] == "profile_evaluation"
    assert record["fallback"] == "raise"
    np.testing.assert_array_equal(record["angles_deg"], [.2, -.1, .5])
    assert problem.counts["gpu_evaluation_failures"] == 1
