"""Literal two/three-element CPU derivative checks; no data files are read.

These tests qualify small mathematical implementations only. They provide no
production-array, identifiability, blind-confirmation, or measurement result.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from array_reconstruct.joint_geometry import scalar_kernel
from array_reconstruct.joint_newton import scalar_kernel_second
from array_reconstruct.joint_pose_kernel import (
    parameters_to_pose, pose_to_parameters, rotation_pose_derivatives,
    scalar_pose_kernel, scalar_pose_kernel_second,
)
from array_reconstruct.physics import C0_M_S


_ELEMENTS = np.array(((.013, -.009), (.029, .012), (-.008, .017)))
_SAMPLES = np.array(((-.021, -.013), (.005, .023), (.037, -.007)))
_FREQUENCY = 12.3e9


def _rotate_literal(vector, angles_deg):
    """Apply spin, x tilt, y tilt directly; no derivative/matrix helper reused."""
    a, b, g = np.asarray(angles_deg)*np.pi/180.
    x, y, z = vector
    x, y = np.cos(g)*x-np.sin(g)*y, np.sin(g)*x+np.cos(g)*y
    y, z = np.cos(a)*y-np.sin(a)*z, np.sin(a)*y+np.cos(a)*z
    x, z = np.cos(b)*x+np.sin(b)*z, -np.sin(b)*x+np.cos(b)*z
    return np.array((x, y, z))


def _value_oracle(u, elements=_ELEMENTS, samples=_SAMPLES, qa=2., qp=0., sign=-1):
    """Direct physical value definition independent of implementation helpers."""
    u = np.asarray(u)
    d, dx, dy = ((60.+10.*u[0])*1e-3, u[1]*.01, u[2]*.01)
    angles = u[3:]*np.array((1., 1., 3.))
    elements = np.asarray(elements)
    if elements.shape[1] == 2:
        elements = np.column_stack((elements, np.zeros(len(elements))))
    center = np.mean(elements, axis=0)
    normal = _rotate_literal(np.array((0., 0., 1.)), angles)
    result = np.empty((len(samples), len(elements)), complex)
    for column, element in enumerate(elements):
        position = center+_rotate_literal(element-center, angles)-np.array((dx, dy, 0.))
        for row, (x, y) in enumerate(samples):
            vector = np.array((x, y, d))-position
            radius = np.linalg.norm(vector)
            result[row, column] = (np.dot(normal, vector)**qa*vector[2]**qp
                *np.exp(sign*1j*(2.*np.pi*_FREQUENCY/C0_M_S)*radius)
                /radius**(qa+qp+1.))
    return result


def _relative(actual, expected):
    return np.linalg.norm(actual-expected)/max(np.linalg.norm(actual), np.linalg.norm(expected), 1e-14)


def _first_fd(oracle, u, j, h):
    step = np.eye(6)[j]*h
    return (-oracle(u+2*step)+8*oracle(u+step)-8*oracle(u-step)
            +oracle(u-2*step))/(12*h)


def _second_fd(oracle, u, j, k, h):
    sj, sk = np.eye(6)[j]*h, np.eye(6)[k]*h
    if j == k:
        return (-oracle(u+2*sj)+16*oracle(u+sj)-30*oracle(u)
                +16*oracle(u-sj)-oracle(u-2*sj))/(12*h*h)
    # Tensor product of independent fourth-order value-only difference rules.
    result = np.zeros_like(oracle(u))
    for a, ca in ((-2, 1.), (-1, -8.), (1, 8.), (2, -1.)):
        for b, cb in ((-2, 1.), (-1, -8.), (1, 8.), (2, -1.)):
            result += ca*cb*oracle(u+a*sj+b*sk)
    return result/(144*h*h)


def test_pose_roundtrip_bounds_and_no_clipping():
    geometry = np.array((62.123456789, -3.45678901, 4.56789012))
    angles = np.array((.12345678, -.87654321, 2.3456789))
    u = pose_to_parameters(geometry, angles)
    assert_allclose(u, ((geometry[0]-60.)/10., geometry[1]/10., geometry[2]/10.,
                        angles[0], angles[1], angles[2]/3.), rtol=0., atol=0.)
    actual_geometry, actual_angles = parameters_to_pose(u)
    assert_allclose(actual_geometry, geometry, rtol=0., atol=2e-15)
    assert_allclose(actual_angles, angles, rtol=0., atol=1e-15)
    assert_array_equal(pose_to_parameters((50., -10., 10.), (-1., 1., -3.)),
                       (-1., -1., 1., -1., 1., -1.))
    for invalid in (np.ones(5), np.array((0., 0., 0., 0., 0., 1.00000001)),
                    np.array((0., 0., 0., np.nan, 0., 0.)), np.ones(6, complex)):
        with pytest.raises(ValueError):
            parameters_to_pose(invalid)
    with pytest.raises(ValueError, match="bounds"):
        pose_to_parameters((70.00000001, 0., 0.), (0., 0., 0.))


@pytest.mark.parametrize("angles", [(0., 0., 0.), (.37, -.64, 1.81)])
def test_rotation_full_u_derivatives_match_value_only_differences(angles):
    rotation, first, second = rotation_pose_derivatives(angles)
    u = np.concatenate((np.zeros(3), np.asarray(angles)/np.array((1., 1., 3.))))
    def oracle(v):
        return np.column_stack([_rotate_literal(axis, v[3:]*np.array((1., 1., 3.)))
                                for axis in np.eye(3)])
    assert rotation.dtype == first.dtype == second.dtype == np.float64
    assert first.shape == (6, 3, 3) and second.shape == (6, 6, 3, 3)
    assert_allclose(rotation, oracle(u), rtol=2e-15, atol=1e-16)
    assert_allclose(rotation.T@rotation, np.eye(3), rtol=2e-15, atol=3e-16)
    assert_array_equal(first[:3], 0.)
    assert_array_equal(second[:3], 0.)
    assert_array_equal(second[:, :3], 0.)
    assert_array_equal(first[5, :, 2], 0.)
    assert_array_equal(second[5, :, :, 2], 0.)
    for j in range(3, 6):
        assert _relative(first[j], _first_fd(oracle, u, j, .01)) < 3e-12
        for k in range(j, 6):
            assert_allclose(second[j, k], second[k, j], rtol=0., atol=0.)
            assert _relative(second[j, k], _second_fd(oracle, u, j, k, .03)) < 3e-8


@pytest.mark.parametrize("qa,qp", [(2., 0.), (1., 1.), (0., 2.), (.7, 1.3), (0., 0.), (1.25, .5)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_angles_preserve_legacy_value_translation_j_and_h(qa, qp, sign):
    geometry, angles = (63.234567, 2.34567, -1.23456), (0., 0., 0.)
    value, first = scalar_pose_kernel(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry, angles,
                                      q_array=qa, q_probe=qp, sign=sign, derivatives=True)
    old = scalar_kernel(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry, q=qa+qp, sign=sign)
    assert_array_equal(value, old)
    assert first.shape == (6, 3, 3) and first.dtype == np.complex128
    for j in range(3):
        expected = scalar_kernel(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry,
                                  q=qa+qp, sign=sign, derivative=j)
        assert_allclose(first[j], expected, rtol=3e-15, atol=3e-14)
        for k in range(j, 3):
            expected = scalar_kernel_second(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry,
                                            q=qa+qp, sign=sign, j=j, k=k)
            actual = scalar_pose_kernel_second(_ELEMENTS, _SAMPLES, _FREQUENCY,
                geometry, angles, q_array=qa, q_probe=qp, sign=sign, j=j, k=k)
            assert_allclose(actual, expected, rtol=2e-14, atol=6e-13)


@pytest.mark.parametrize("count,angles", [(2, (0., 0., 0.)), (3, (.43, -.61, 1.7))])
@pytest.mark.parametrize("qa,qp", [(2., 0.), (1., 1.), (0., 2.), (.8, .4), (0., 0.)])
def test_all_six_columns_and_21_pairs_against_independent_value_fd(count, angles, qa, qp):
    elements = _ELEMENTS[:count]
    geometry = np.array((61.234, 1.234, -2.345))
    u = pose_to_parameters(geometry, angles)
    oracle = lambda v: _value_oracle(v, elements=elements, qa=qa, qp=qp)
    value, first = scalar_pose_kernel(elements, _SAMPLES, _FREQUENCY, geometry, angles,
                                      q_array=qa, q_probe=qp, derivatives=True)
    assert_allclose(value, oracle(u), rtol=6e-15, atol=2e-13)
    assert value.dtype == first.dtype == np.complex128
    for j in range(6):
        errors = [_relative(first[j], _first_fd(oracle, u, j, h))
                  for h in (2e-4, 5e-4)]
        assert max(errors) < 1e-8, (j, errors)
        for k in range(j, 6):
            actual = scalar_pose_kernel_second(elements, _SAMPLES, _FREQUENCY,
                geometry, angles, q_array=qa, q_probe=qp, j=j, k=k)
            transpose = scalar_pose_kernel_second(elements, _SAMPLES, _FREQUENCY,
                geometry, angles, q_array=qa, q_probe=qp, j=k, k=j)
            assert actual.shape == (3, count) and actual.dtype == np.complex128
            assert_allclose(actual, transpose, rtol=3e-15, atol=2e-14)
            errors = [_relative(actual, _second_fd(oracle, u, j, k, h))
                      for h in (1e-3, 3e-3)]
            assert min(errors) < 2e-5, (j, k, errors)
            assert max(errors) < 2e-4, (j, k, errors)


def test_centroid_pivot_common_frame_shift_and_sign_conjugacy():
    geometry, angles = (59.123, -1.72, 2.16), (.41, -.32, 1.14)
    translation = np.array((.114, -.092))
    kwargs = dict(q_array=.7, q_probe=1.2)
    negative = scalar_pose_kernel(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry, angles,
                                 derivatives=True, **kwargs)
    positive = scalar_pose_kernel(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry, angles,
                                 sign=1, derivatives=True, **kwargs)
    shifted = scalar_pose_kernel(_ELEMENTS+translation, _SAMPLES+translation,
                                _FREQUENCY, geometry, angles, derivatives=True, **kwargs)
    for a, b, c in zip(negative, positive, shifted):
        assert_allclose(a.conj(), b, rtol=0., atol=0.)
        assert_allclose(a, c, rtol=2e-14, atol=3e-13)
    for j in range(6):
        for k in range(j, 6):
            h = scalar_pose_kernel_second(_ELEMENTS, _SAMPLES, _FREQUENCY, geometry,
                                         angles, j=j, k=k, **kwargs)
            shifted_h = scalar_pose_kernel_second(_ELEMENTS+translation,
                _SAMPLES+translation, _FREQUENCY, geometry, angles, j=j, k=k, **kwargs)
            conjugate = scalar_pose_kernel_second(_ELEMENTS, _SAMPLES, _FREQUENCY,
                geometry, angles, j=j, k=k, sign=1, **kwargs)
            assert_allclose(h.conj(), conjugate, rtol=0., atol=0.)
            assert_allclose(h, shifted_h, rtol=5e-14, atol=3e-13)


@pytest.mark.parametrize("name,value", [
    ("elements_m", ((0., 0., .001),)), ("elements_m", np.empty((0, 2))),
    ("samples_xy_m", ((0., np.nan),)), ("samples_xy_m", ((0., 0., 0.),)),
    ("f_hz", 0.), ("f_hz", True), ("f_hz", complex(1.)),
    ("geometry_mm", (60., 10.0001, 0.)), ("angles_deg", (0., 0., 3.001)),
    ("q_array", -.1), ("q_probe", np.inf), ("q_probe", True),
    ("sign", 0), ("sign", True), ("derivatives", 1),
])
def test_invalid_inputs_fail_explicitly(name, value):
    params = dict(elements_m=_ELEMENTS, samples_xy_m=_SAMPLES, f_hz=_FREQUENCY,
                  geometry_mm=(60., 0., 0.), angles_deg=(0., 0., 0.))
    params[name] = value
    with pytest.raises(ValueError):
        scalar_pose_kernel(**params)


def test_directional_height_rules_and_second_indices():
    with pytest.raises(ValueError, match="h_array"):
        scalar_pose_kernel(_ELEMENTS, ((-10., 0.),), _FREQUENCY,
                           (60., 0., 0.), (0., 1., 0.))
    wide = np.array(((-10., 0.), (10., 0.)))
    with pytest.raises(ValueError, match="h_probe"):
        scalar_pose_kernel(wide, ((0., 0.),), _FREQUENCY,
                           (60., 0., 0.), (0., 1., 0.), q_array=0., q_probe=2.)
    # An unused directional factor imposes no artificial positive-height rule.
    value, jacobian = scalar_pose_kernel(wide, ((-10., 0.),), _FREQUENCY,
        (60., 0., 0.), (0., 1., 0.), q_array=0., q_probe=0., derivatives=True)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(jacobian))
    for j in (-1, 6, True, 1., np.array([1])):
        with pytest.raises(ValueError, match="indices"):
            scalar_pose_kernel_second(_ELEMENTS, _SAMPLES, _FREQUENCY,
                                      (60., 0., 0.), (0., 0., 0.), j=j, k=0)
