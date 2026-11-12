"""No CUDA access: finite C0 hook/reference/strict-failure contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_finite_aperture_reference import literal_finite_kernel
from array_reconstruct.joint_geometry_finite_gpu import GPUFiniteApertureJointProblem
from array_reconstruct.joint_geometry_gpu import GPUJointProblem

_script = Path(__file__).resolve().parents[1]/"scripts/qualify_joint_finite_aperture.py"
_spec = importlib.util.spec_from_file_location("qualify_finite_aperture_contract", _script)
qualification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qualification)


@pytest.mark.parametrize("q,sign", [(0., -1), (2., -1), (2.4, 1)])
@pytest.mark.parametrize("geometry", [(50., -10., 10.), (61.37, -.83, 2.19), (70., 10., -10.)])
def test_first_second_parity_boundaries_and_phase_sign(q, sign, geometry, monkeypatch):
    # Any access to CUDA is a test failure, including an availability query.
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA must not be accessed by CPU contract tests")
    monkeypatch.setattr(qualification.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(GPUJointProblem, "_kernel", forbidden)
    monkeypatch.setattr(GPUJointProblem, "_second", forbidden)
    elements, groups = qualification.fixture(3, 4)
    tested = qualification.CPUArithmeticHarness(elements, groups, q=q, sign=sign,
        chunk_size=3, source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.))
    x = tested.initial_parameters(geometry)
    native = tested._session(groups[0].sessions[0])
    kernel = tested._kernel(native, tested._tensor(x, tested.rd))
    args = (elements, native["session"].samples_xy_m, native["session"].frequency_hz, geometry, q, sign)
    options = {"source_aperture_mm": (2., 6.), "probe_aperture_mm": (3., 4.)}
    assert_allclose(tested._numpy(kernel["K"]), literal_finite_kernel(*args, **options), rtol=2e-13)
    for j in range(3):
        assert_allclose(tested._numpy(kernel["first"][j]), literal_finite_kernel(*args, derivative=j, **options), rtol=2e-13)
        for k in range(j, 3):
            assert_allclose(tested._numpy(tested._second(kernel, j, k)), literal_finite_kernel(*args, derivative=(j, k), **options), rtol=3e-13)
    calls = tested.counts["kernel_second_derivative_calls"]
    assert tested._second(kernel, 1, 0) is kernel["second"][0, 1]
    assert calls == tested.counts["kernel_second_derivative_calls"] == 6
    assert tested.counts["finite_quadrature_term_blocks"] == 81*2*7
    assert tested.counts["finite_max_active_quadrature_terms"] == 1


@pytest.mark.parametrize("bad", [True, .0, -1, 3, "0"])
def test_axis_type_rejected(bad):
    elements, groups = qualification.fixture(3, 4)
    tested = qualification.CPUArithmeticHarness(elements, groups)
    native = tested._session(groups[0].sessions[0])
    kernel = tested._kernel(native, tested._tensor(tested.initial_parameters(), tested.rd))
    with pytest.raises(ValueError, match="axes"):
        tested._second(kernel, bad, 1)
    with pytest.raises(ValueError, match="axes"):
        literal_finite_kernel(elements, native["session"].samples_xy_m, 12.7e9, (60, 0, 0), derivative=(bad, 1))


def test_hessian_and_prediction_failures_cannot_call_cpu(monkeypatch):
    elements, groups = qualification.fixture(3, 4)
    tested = qualification.CPUArithmeticHarness(elements, groups)
    x = tested.initial_parameters()
    tested.evaluate(x)
    def fail(*args, **kwargs):
        raise RuntimeError("injected finite second failure")
    monkeypatch.setattr(tested, "_second", fail)
    with pytest.raises(RuntimeError, match="injected finite second"):
        tested.profile_gradient_hessian(x)
    assert tested._cpu_reference is None and tested.counts["cpu_fallbacks"] == 0
    assert tested.failure_ledger[-1]["operation"] == "exact_profile_hessian"
    monkeypatch.setattr(tested, "_kernel", fail)
    with pytest.raises(RuntimeError, match="injected"):
        tested.predict(groups[0].sessions[0], (60, 0, 0), np.ones(3))
    with pytest.raises(RuntimeError, match="injected"):
        tested.validation_evaluate(x, {groups[0].name: [groups[0].sessions[0]]})
    assert tested._cpu_reference is None and tested.counts["cpu_fallbacks"] == 0


def test_constructor_fallback_rejected_before_cuda():
    elements, groups = qualification.fixture(3, 4)
    with pytest.raises(ValueError, match="fallback"):
        GPUFiniteApertureJointProblem(elements, groups, fallback="cpu")


def test_from_problem_preserves_apertures(monkeypatch):
    elements, groups = qualification.fixture(3, 4)
    source = qualification.CPUArithmeticHarness(elements, groups,
        source_aperture_mm=(6., 2.), probe_aperture_mm=(3., 4.))
    observed = {}
    def capture(self, *args, **kwargs):
        observed.update(kwargs)
    monkeypatch.setattr(GPUFiniteApertureJointProblem, "__init__", capture)
    GPUFiniteApertureJointProblem.from_problem(source)
    assert observed["source_aperture_mm"] == (6., 2.)
    assert observed["probe_aperture_mm"] == (3., 4.)


@pytest.mark.parametrize("backend", ["torch_cpu_harness", "literal_CPU_reference"])
def test_aperture_configuration_readonly_keeps_same_x_profile_and_H_consistent(backend):
    """Regression for independently reproduced 4.15% stale-model residual."""
    from dataclasses import FrozenInstanceError
    elements, groups = qualification.fixture(3, 4)
    if backend == "torch_cpu_harness":
        tested = qualification.CPUArithmeticHarness(elements, groups)
        changes = [("quadrature", qualification.ApertureQuadrature.gl3((12., 2.), (5., 9.)))]
    else:
        tested = qualification.CPUFiniteApertureReference(elements, groups)
        changes = [("source_aperture_mm", (12., 2.)), ("probe_aperture_mm", (5., 9.))]
    x = tested.initial_parameters((61.37, -.83, 2.19), {"f0s1": 1.17+.23j, "f0s2": .86-.19j})
    original = tested.evaluate(x)
    H = tested.profile_gradient_hessian(x)["hessian"].copy()
    session = groups[0].sessions[0]
    prediction = tested.predict(session, original.geometry_mm, original.b[groups[0].name])
    for attribute, replacement in changes:
        with pytest.raises(AttributeError):
            setattr(tested, attribute, replacement)
    if backend == "torch_cpu_harness":
        with pytest.raises(FrozenInstanceError):
            tested.quadrature.source_aperture_mm = (12., 2.)
        with pytest.raises(TypeError):
            tested.quadrature.terms[0] = (0., 0., 1.)
        assert tested.quadrature.source_aperture_mm == (4., 4.)
        assert tested.quadrature.probe_aperture_mm == (4., 4.)
    else:
        assert tested.source_aperture_mm == tested.probe_aperture_mm == (4., 4.)
    assert tested.evaluate(x) is original
    assert_allclose(tested.profile_gradient_hessian(x)["hessian"], H, rtol=0, atol=0)
    assert_allclose(tested.predict(session, original.geometry_mm, original.b[groups[0].name]), prediction, rtol=0, atol=0)
