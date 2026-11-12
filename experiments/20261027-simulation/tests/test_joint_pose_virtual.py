"""Literal two/three-element preparation checks; no production data or GPU."""
import ast
from pathlib import Path

import numpy as np
import pytest

from array_reconstruct.joint_pose_kernel import scalar_pose_kernel
from array_reconstruct.joint_pose_virtual import independent_pose_field, pose_error_metrics


ELEMENTS = np.array([[-.012, -.009], [.018, -.009], [.003, .015]])
SAMPLES = np.array([[-.03, -.02], [.015, -.01], [0., .024], [.03, .027], [-.01, .01]])
WEIGHTS = np.array([1.+.04j, .93-.08j, 1.07+.02j])


@pytest.mark.parametrize("angles", [[0., 0., 0.], [.4, -.6, 1.3]])
@pytest.mark.parametrize("q_array,q_probe", [(2., 0.), (1.2, .8), (0., 2.)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_independent_forward_matches_fit_kernel_only_in_test(angles, q_array, q_probe, sign):
    geometry = [62., -.8, 1.1]
    expected = scalar_pose_kernel(ELEMENTS, SAMPLES, 9.1e9, geometry, angles,
        q_array=q_array, q_probe=q_probe, sign=sign)@WEIGHTS
    for chunk_rows in (1, 2, 8):
        result = independent_pose_field(ELEMENTS, SAMPLES, 9.1e9, geometry, angles,
            WEIGHTS, q_array=q_array, q_probe=q_probe, sign=sign, chunk_rows=chunk_rows)
        assert result.dtype == np.complex128
        assert result.shape == (len(SAMPLES),)
        np.testing.assert_allclose(result, expected, rtol=8e-14, atol=5e-12)


def test_virtual_truth_outside_fit_bounds_is_not_clipped():
    # At scan x/y=0, two equal-weight elements have the same radius even after
    # a 10-degree in-plane spin. Analytic value uses the unbounded d=82 mm.
    elements = np.array([[-.01, 0., 0.], [.01, 0., 0.]])
    distance = np.hypot(.082, .01)
    expected = 2*.082**2*np.exp(-2j*np.pi*9.1e9/299792458.*distance)/distance**3
    result = independent_pose_field(elements, [[0., 0.]], 9.1e9, [82., 0., 0.],
                                     [0., 0., 10.], [1., 1.])
    np.testing.assert_allclose(result, [expected], rtol=1e-13, atol=2e-12)


def test_independent_literal_noncommuting_tilts_and_rotating_normal():
    # Ry(90) Rx(90) maps centered local x=-/+10 mm to world z=+/-10 mm;
    # R ez=(0,-1,0). This is an analytic ray check independent of fit kernels.
    elements = [[.003, -.01], [.023, -.01]]
    heights_probe = np.array([.05, .07])
    radius = np.hypot(.02, heights_probe)
    expected = np.sum(np.array([1., 2j])*.02*heights_probe
        *np.exp(-2j*np.pi*9.1e9/299792458.*radius)/radius**3)
    result = independent_pose_field(elements, [[.013, -.03]], 9.1e9,
        [60., 0., 0.], [90., 90., 0.], [1., 2j], q_array=1., q_probe=1.)
    np.testing.assert_allclose(result, [expected], rtol=2e-13, atol=3e-12)


@pytest.mark.parametrize("argument,value", [
    ("elements_m", [[0., 0., .001], [.01, 0., 0.]]),
    ("elements_m", [[1j, 0.], [.01, 0.]]),
    ("samples_xy_m", [[0., np.nan]]),
    ("effective_weights", [1., 1.]),
    ("effective_weights", [1., np.inf, 1.]),
    ("frequency_hz", 0.),
    ("q_array", -1.),
    ("q_probe", True),
    ("sign", True),
    ("chunk_rows", 1.5),
    ("chunk_rows", 0),
])
def test_forward_rejects_invalid_shapes_types_and_values(argument, value):
    options = dict(elements_m=ELEMENTS, samples_xy_m=SAMPLES, frequency_hz=9.1e9,
        geometry_mm=[60., 0., 0.], angles_deg=[0., 0., 0.], effective_weights=WEIGHTS)
    options[argument] = value
    with pytest.raises(ValueError):
        independent_pose_field(**options)


def test_invalid_directional_heights_and_zero_radius_are_rejected():
    with pytest.raises(ValueError, match="array directional height"):
        independent_pose_field(ELEMENTS, SAMPLES, 9.1e9, [-60., 0., 0.], [0., 0., 0.], WEIGHTS)
    with pytest.raises(ValueError, match="probe directional height"):
        independent_pose_field(ELEMENTS, SAMPLES, 9.1e9, [-60., 0., 0.], [0., 0., 0.], WEIGHTS,
                               q_array=0., q_probe=2.)
    with pytest.raises(ValueError, match="radius"):
        independent_pose_field([[0., 0.], [.01, 0.]], [[0., 0.]], 9.1e9,
                               [0., 0., 0.], [0., 0., 0.], [1., 1.])
    # An inactive directional factor does not impose a hemisphere restriction.
    isotropic = independent_pose_field(ELEMENTS, SAMPLES, 9.1e9, [-60., 0., 0.],
        [0., 0., 0.], WEIGHTS, q_array=0., q_probe=0.)
    assert isotropic.dtype == np.complex128 and np.all(np.isfinite(isotropic))


def test_metrics_identity_and_common_world_translation():
    identity = pose_error_metrics(ELEMENTS, [60., 1., -2.], [.3, -.2, .7],
                                   [60., 1., -2.], [.3, -.2, .7])
    np.testing.assert_allclose(identity["rotation_geodesic_angle_deg"], 0., atol=1e-14)
    assert identity["total_displacement_mm"] == {"max": 0., "p90": 0.}
    a = pose_error_metrics(ELEMENTS, [60., 0., 0.], [0., 0., 0.],
                            [72., 3., 4.], [0., 0., 0.])
    assert a["geometry_error_mm"] == [12., 3., 4.]
    assert a["angle_error_deg"] == {"alpha": 0., "beta": 0., "gamma": 0.}
    assert a["rotation_only_displacement_mm"] == {"max": 0., "p90": 0.}
    np.testing.assert_allclose(list(a["total_displacement_mm"].values()), [13., 13.], atol=2e-14)
    # Translating every registered element changes the pivot position, not its
    # centered relative pose errors, including a nonzero rotation mismatch.
    b = pose_error_metrics(ELEMENTS, [60., 0., 0.], [.2, -.3, .4],
                            [62., .5, -.7], [-.1, .5, -.8])
    c = pose_error_metrics(ELEMENTS+[.1, -.2], [60., 0., 0.], [.2, -.3, .4],
                            [62., .5, -.7], [-.1, .5, -.8])
    for key in ("rotation_only_displacement_mm", "total_displacement_mm"):
        np.testing.assert_allclose(list(b[key].values()), list(c[key].values()), atol=5e-14)


def test_metrics_rotation_only_literal_two_elements_and_full_turn():
    elements = [[.003, -.01], [.023, -.01]]
    result = pose_error_metrics(elements, [60., 0., 0.], [0., 0., 0.],
                                [60., 0., 0.], [0., 0., 90.])
    expected = np.sqrt(2)*10.
    assert result["angle_error_deg"] == {"alpha": 0., "beta": 0., "gamma": 90.}
    np.testing.assert_allclose(result["rotation_geodesic_angle_deg"], 90., atol=1e-13)
    for key in ("rotation_only_displacement_mm", "total_displacement_mm"):
        np.testing.assert_allclose(list(result[key].values()), [expected, expected], atol=1e-13)
    full_turn = pose_error_metrics(elements, [60., 0., 0.], [0., 0., 0.],
                                   [60., 0., 0.], [0., 0., 360.])
    assert full_turn["angle_error_deg"]["gamma"] == 360.
    assert full_turn["rotation_geodesic_angle_deg"] < 2e-14


def test_metrics_three_element_p90_is_not_replaced_by_maximum():
    result = pose_error_metrics([[0., 0.], [.03, 0.], [.01, .01]],
        [60., 0., 0.], [0., 0., 0.], [60., 0., 0.], [0., 0., 90.])
    maximum = np.sqrt(5200.)/3.
    p90 = .2*np.sqrt(3400.)/3.+.8*maximum
    for key in ("rotation_only_displacement_mm", "total_displacement_mm"):
        np.testing.assert_allclose(result[key]["max"], maximum, atol=1e-13)
        np.testing.assert_allclose(result[key]["p90"], p90, atol=1e-13)
        assert result[key]["p90"] < result[key]["max"]


def test_virtual_module_imports_no_fitted_model_or_optimizer():
    source = Path(__file__).resolve().parents[1]/"src/array_reconstruct/joint_pose_virtual.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert set(modules) <= {"__future__", "numpy", "physics"}
