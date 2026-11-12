"""Vector-reference lineage, full arrays and complete C0 profile contracts."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_geometry import FrequencyGroup, Session
from array_reconstruct.joint_finite_aperture_reference import CPUFiniteApertureReference, literal_finite_kernel
from array_reconstruct.joint_finite_aperture_vector_reference import VectorFiniteApertureReference, vector_finite_kernel


def fixture():
    elements = np.array([[-.011, -.011], [0., -.011], [.011, -.011], [-.0055, 0.]])
    xy = np.c_[np.linspace(-.029, .037, 7), .023*np.sin(np.arange(7)*1.17)]
    groups = []
    for fi, frequency in enumerate((12.7e9, 12.1e9)):
        sessions = []
        for index in range(3 if fi == 0 else 1):
            samples = xy if index != 2 else xy+[.00041, -.00023]
            u = np.exp(1j*(index+.3)*(elements[:, 0]*130+elements[:, 1]*47))
            field = np.sin(np.arange(7)*.83+.4*index)+.3+1j*np.cos(np.arange(7)*.61+fi+.2*index)
            weight = np.linspace(.13, .21, 7); weight[::5] = 0
            sessions.append(Session(f"f{fi}s{index}", frequency, samples, u, field, weight))
        groups.append(FrequencyGroup(f"f{fi}", sessions, .043))
    return elements, tuple(groups)


@pytest.mark.parametrize("q,sign,source,probe", [(2., -1, 4., 4.), (2.4, 1, (2., 6.), (3., 4.)), (0., -1, 0., 0.)])
def test_all_entries_K_first_second_against_old_literal(q, sign, source, probe):
    elements, groups = fixture()
    session = groups[0].sessions[0]
    args = (elements, session.samples_xy_m, session.frequency_hz, (61.37, -.83, 2.19), q, sign)
    options = dict(source_aperture_mm=source, probe_aperture_mm=probe)
    for derivative in (None, 0, 1, 2, (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        reference = literal_finite_kernel(*args, derivative=derivative, **options)
        for chunks in ((2, 3), (128, 256)):
            actual = vector_finite_kernel(*args, derivative=derivative, sample_chunk_size=chunks[0], element_chunk_size=chunks[1], **options)
            assert actual.shape == reference.shape and actual.dtype == np.complex128
            assert_allclose(actual, reference, rtol=3e-13, atol=3e-13)


def test_vector_kernel_multistep_finite_differences():
    elements, groups = fixture()
    samples = groups[0].sessions[0].samples_xy_m
    geometry = np.array((61.37, -.83, 2.19))
    options = dict(source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.), sample_chunk_size=3, element_chunk_size=2)
    for step in (3e-4, 1e-4, 3e-5):
        for j in range(3):
            delta = np.eye(3)[j]*(10*step)
            finite = (vector_finite_kernel(elements, samples, 12.7e9, geometry+delta, **options)-vector_finite_kernel(elements, samples, 12.7e9, geometry-delta, **options))/(2*step)
            analytic = vector_finite_kernel(elements, samples, 12.7e9, geometry, derivative=j, **options)
            assert np.linalg.norm(finite-analytic)/np.linalg.norm(analytic) < 1e-5
            for k in range(3):
                finite = (vector_finite_kernel(elements, samples, 12.7e9, geometry+delta, derivative=k, **options)-vector_finite_kernel(elements, samples, 12.7e9, geometry-delta, derivative=k, **options))/(2*step)
                analytic = vector_finite_kernel(elements, samples, 12.7e9, geometry, derivative=(j, k), **options)
                assert np.linalg.norm(finite-analytic)/np.linalg.norm(analytic) < 1e-5


def test_complete_profile_and_auxiliary_paths_no_literal_bypass(monkeypatch):
    import array_reconstruct.joint_finite_aperture_reference as literal_module
    elements, groups = fixture()
    options = dict(q=2.4, sign=1, source_aperture_mm=(2., 6.), probe_aperture_mm=(3., 4.))
    reference = CPUFiniteApertureReference(elements, groups, **options)
    tested = VectorFiniteApertureReference(elements, groups, sample_chunk_size=3, element_chunk_size=2, **options)
    x = reference.initial_parameters((61.37, -.83, 2.19), {"f0s1": 1.17+.23j, "f0s2": .86-.19j})
    a = reference.evaluate(x)
    ah = reference.profile_gradient_hessian(x)
    validation = {"f0": [replace(groups[0].sessions[1], samples_xy_m=groups[0].sessions[1].samples_xy_m+[.00071, -.00029])]}
    va = reference.validation_evaluate(x, validation)
    fixed_va = reference.validation_evaluate(x, validation, {"f0s1": .91+.12j})
    prediction = reference.predict(validation["f0"][0], a.geometry_mm, a.b["f0"], .93+.17j)
    def forbidden(*args, **kwargs):
        raise AssertionError("new vector reference called the old literal kernel")
    monkeypatch.setattr(literal_module, "literal_finite_kernel", forbidden)
    b = tested.evaluate(x)
    for field in ("residual", "jacobian"):
        assert_allclose(getattr(b, field), getattr(a, field), rtol=3e-12, atol=3e-13)
    for field in ("b", "db", "data_residuals", "weighted_data_residuals", "penalty_residuals", "data_jacobians", "weighted_data_jacobians", "penalty_jacobians"):
        for key, expected in getattr(a, field).items():
            assert_allclose(getattr(b, field)[key], expected, rtol=3e-12, atol=3e-13)
    bh = tested.profile_gradient_hessian(x)
    assert_allclose(bh["gradient"], ah["gradient"], rtol=3e-12, atol=3e-13)
    assert_allclose(bh["hessian"], ah["hessian"], rtol=3e-12, atol=3e-13)
    assert np.linalg.norm(bh["hessian"]-b.jacobian.T@b.jacobian) > 1e-4
    for expected, gain in ((va, None), (fixed_va, {"f0s1": .91+.12j})):
        actual = tested.validation_evaluate(x, validation, gain)
        for key in ("residual", "jacobian"):
            assert_allclose(actual[key], expected[key], rtol=3e-12, atol=3e-13)
    assert_allclose(tested.predict(validation["f0"][0], b.geometry_mm, b.b["f0"], .93+.17j), prediction, rtol=3e-12, atol=3e-13)
    for step in (1e-4, 3e-5):
        J, gradient, H = [], [], []
        for j in range(len(x)):
            delta = np.eye(len(x))[j]*step
            plus, minus = tested.evaluate(x+delta), tested.evaluate(x-delta)
            J.append((plus.residual-minus.residual)/(2*step))
            gradient.append((plus.cost-minus.cost)/(2*step))
            H.append((plus.jacobian.T@plus.residual-minus.jacobian.T@minus.residual)/(2*step))
        assert np.linalg.norm(np.stack(J, axis=1)-b.jacobian)/np.linalg.norm(b.jacobian) < 1e-5
        assert np.linalg.norm(np.array(gradient)-bh["gradient"])/np.linalg.norm(bh["gradient"]) < 1e-5
        assert np.linalg.norm(np.stack(H, axis=1)-bh["hessian"])/np.linalg.norm(bh["hessian"]) < 1e-5
    counts = tested.counts
    assert counts["vector_peak_block_rows"] <= 3
    assert counts["vector_peak_block_columns"] <= 2
    assert counts["vector_quadrature_term_blocks"] == 81*counts["vector_blocks"]


def test_C0_and_readonly_hypothesis_and_chunks():
    elements, groups = fixture()
    tested = VectorFiniteApertureReference(elements, groups)
    original = tested.evaluate(tested.initial_parameters())
    for attribute, value in (("q", 4.), ("sign", 1), ("source_aperture_mm", (12., 2.)), ("probe_aperture_mm", (5., 9.)), ("sample_chunk_size", 1), ("element_chunk_size", 1)):
        with pytest.raises(AttributeError):
            setattr(tested, attribute, value)
    assert tested.evaluate(tested.initial_parameters()) is original
    for group in (replace(groups[0], b0=np.ones(4)), replace(groups[0], amplitude_prior=np.ones(4)), replace(groups[0], template_prior=np.ones(4))):
        with pytest.raises(NotImplementedError, match="C0"):
            VectorFiniteApertureReference(elements, [group])
    for chunk in (True, 0, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            VectorFiniteApertureReference(elements, groups, sample_chunk_size=chunk)
