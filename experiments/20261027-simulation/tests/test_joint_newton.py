"""Exact profile curvature contracts on synthetic fields only.

These checks are numerical implementation tests, not evidence that mechanical
geometry is identifiable from the registered measurements.  No archives or
previous experiment outputs are read.
"""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_geometry import (
    AmplitudeProfileError, FrequencyGroup, JointProblem, Session, scalar_kernel,
)
from array_reconstruct.joint_newton import (
    polish_profile, profile_gradient_hessian, scalar_kernel_second,
)
from array_reconstruct.physics import C0_M_S, planar_scan, rectangular_array


def _independent_kernel(elements, xy, frequency, geometry, q=2., sign=-1):
    """Direct Euclidean definition, independent of either derivative routine."""
    geometry = np.asarray(geometry)
    samples = np.column_stack((xy+geometry[None, 1:]*1e-3,
                               np.full(len(xy), geometry[0]*1e-3)))
    radius = np.linalg.norm(samples[:, None, :]-elements[None, :, :], axis=-1)
    return ((geometry[0]*1e-3)**q
            * np.exp(sign*2j*np.pi*frequency/C0_M_S*radius)
            / radius**(q+1))


def _example(side=4, *, amplitude=False, positive_control=False,
             truth_geometry=(62.3456, 1.2345, -3.4567)):
    # These are separate code-test streams, not frozen confirmation seeds.
    rng = np.random.default_rng(71903+side)
    elements = rectangular_array(side, side, .011)
    xy = planar_scan(side+5, side+4, .008)[:, :2]
    frequency = 12.7e9
    truth_geometry = np.asarray(truth_geometry, dtype=float)
    truth_b = np.exp(rng.uniform(-.25, .25, side*side)
                     + .17j*rng.normal(size=side*side))
    sessions = []
    for index, (ax, ay) in enumerate(((0., 0.), (.23, -.11), (-.17, .19))):
        steering = np.exp(2j*np.pi*frequency/C0_M_S
                          * (elements[:, 0]*ax+elements[:, 1]*ay))
        gain = np.exp(.035*index+.065j*index)
        field = gain*_independent_kernel(elements, xy, frequency, truth_geometry)@(steering*truth_b)
        if not positive_control:
            field += .02*(rng.normal(size=len(xy))+1j*rng.normal(size=len(xy)))
        sessions.append(Session(f"s{index}", frequency, xy, steering, field,
                                .01*np.linspace(.7, 1.1, len(xy))))
    if positive_control:
        # An explicitly oracle *code-correctness* control, never a real-data prior.
        group = FrequencyGroup("f", sessions, .8, b0=truth_b)
    else:
        group = FrequencyGroup("f", sessions, .08,
                               b0=.4+.03j*rng.normal(size=side*side))
        if amplitude:
            group = replace(group,
                template_prior=np.linspace(.9, 1.1, side*side)
                    * np.exp(.04j*np.arange(side*side)), template_strength=.025,
                amplitude_prior=np.linspace(.8, 1.2, side*side), amplitude_strength=.04,
                amplitude_n_starts=1, amplitude_stationarity_tol=1e-13,
                amplitude_epsilon=1e-6)
    problem = JointProblem(elements, [group], backend="qr")
    x = problem.initial_parameters((61.4321, -.87, -2.9),
                                   {"s1": 1.07+.09j, "s2": .97-.08j})
    return problem, x


def _gradient(problem, x):
    # This uses the established full residual/Jacobian API, not the new
    # envelope-gradient implementation being checked.
    evaluation = problem.evaluate(x)
    return evaluation.jacobian.T@evaluation.residual


def _relative_error(actual, expected):
    return np.linalg.norm(actual-expected)/max(np.linalg.norm(actual),
                                              np.linalg.norm(expected), 1e-14)


@pytest.mark.parametrize("q", [0., 2., 4.])
@pytest.mark.parametrize("sign", [-1, 1])
def test_second_kernel_matches_independent_field_differences(q, sign):
    elements = rectangular_array(3, 2, .011)
    xy = np.array(((-.019, -.011), (.0012, .0173), (.024, -.013)))
    frequency = 12.5e9
    geometry = np.array((62.3456, 1.2345, -3.4567))
    base = _independent_kernel(elements, xy, frequency, geometry, q, sign)
    for j in range(3):
        for k in range(j, 3):
            analytic = scalar_kernel_second(elements, xy, frequency, geometry,
                                            q=q, sign=sign, j=j, k=k)
            transpose = scalar_kernel_second(elements, xy, frequency, geometry,
                                             q=q, sign=sign, j=k, k=j)
            assert_allclose(analytic, transpose, rtol=2e-14, atol=1e-13)
            errors = []
            for h_mm in (.03, .01, .003):
                sj, sk = np.eye(3)[j]*h_mm, np.eye(3)[k]*h_mm
                if j == k:
                    numeric = (_independent_kernel(elements, xy, frequency, geometry+sj, q, sign)
                               - 2*base
                               + _independent_kernel(elements, xy, frequency, geometry-sj, q, sign))
                    numeric /= (h_mm/10)**2
                else:
                    numeric = (_independent_kernel(elements, xy, frequency, geometry+sj+sk, q, sign)
                               - _independent_kernel(elements, xy, frequency, geometry+sj-sk, q, sign)
                               - _independent_kernel(elements, xy, frequency, geometry-sj+sk, q, sign)
                               + _independent_kernel(elements, xy, frequency, geometry-sj-sk, q, sign))
                    numeric /= 4*(h_mm/10)**2
                errors.append(_relative_error(numeric, analytic))
            assert sum(e < 1e-5 for e in errors) >= 2, (q, sign, j, k, errors)
            assert min(errors) < 2e-6, (q, sign, j, k, errors)


def test_second_kernel_at_hard_bounds_uses_feasible_one_sided_steps():
    elements = rectangular_array(2, 3, .011)
    xy = np.array(((-.019, -.011), (.0012, .0173)))
    geometry = np.array((50., -10., 10.))
    for j in range(3):
        for k, direction in enumerate((1., 1., -1.)):
            h_mm = .001
            step = np.eye(3)[k]*h_mm*direction
            first = [scalar_kernel(elements, xy, 12.7e9, geometry+s*step,
                                   derivative=j) for s in range(3)]
            numeric = direction*(-3*first[0]+4*first[1]-first[2])/(2*h_mm/10)
            analytic = scalar_kernel_second(elements, xy, 12.7e9, geometry, j=j, k=k)
            assert _relative_error(numeric, analytic) < 1e-5


@pytest.mark.parametrize("amplitude", [False, True])
def test_profile_hessian_matches_full_resolve_gradient_differences(amplitude):
    problem, x = _example(3 if amplitude else 4, amplitude=amplitude)
    result = profile_gradient_hessian(problem, x)
    evaluation = result["evaluation"]
    gradient, hessian = result["gradient"], result["hessian"]
    assert_allclose(gradient, evaluation.jacobian.T@evaluation.residual,
                    rtol=2e-9, atol=2e-11)
    assert_allclose(hessian, hessian.T, rtol=1e-10, atol=1e-11)
    errors = []
    for h in (1e-4, 3e-5):
        numeric = np.column_stack([
            (_gradient(problem, x+h*axis)-_gradient(problem, x-h*axis))/(2*h)
            for axis in np.eye(problem.parameter_count)])
        errors.append(_relative_error(numeric, hessian))
    assert max(errors) < 1e-5, errors
    # A nonzero curvature correction makes this catch a GN substitution.
    gn = evaluation.jacobian.T@evaluation.jacobian
    assert _relative_error(hessian, gn) > 1e-4
    if amplitude:
        inner = evaluation.diagnostics["groups"]["f"]["amplitude"]
        assert inner["hessian_positive_definite"]
        assert inner["stationarity"] <= inner["stationarity_tol"]
        assert not inner["global_profile_smoothness_established"]
        assert "selected local stationary branch" in inner["profile_derivative_scope"]


@pytest.mark.parametrize("side", [8, 16])
def test_n8_n16_profile_hessian_directional_finite_differences(side):
    problem, x = _example(side)
    result = profile_gradient_hessian(problem, x)
    assert_allclose(result["gradient"], _gradient(problem, x), rtol=2e-9, atol=2e-10)
    # Mix all geometry and both complex nuisance gains so their cross terms
    # are exercised.  A separate direction isolates the mechanical block.
    directions = [np.array((.7, -.3, .4, .2, -.1, -.35, .25)),
                  np.array((.3, .5, -.8, 0., 0., 0., 0.))]
    for direction in directions:
        direction /= np.linalg.norm(direction)
        analytic = result["hessian"]@direction
        errors = []
        for h in (1e-4, 3e-5):
            numeric = (_gradient(problem, x+h*direction)
                       -_gradient(problem, x-h*direction))/(2*h)
            errors.append(_relative_error(numeric, analytic))
        assert max(errors) < 1e-5, (side, errors)


def test_amplitude_common_scale_does_not_change_profile_curvature():
    problem, x = _example(3, amplitude=True)
    scaled = JointProblem(problem.elements_m, [replace(problem.groups[0],
        amplitude_prior=237*problem.groups[0].amplitude_prior)], backend="qr")
    original = profile_gradient_hessian(problem, x)
    comparison = profile_gradient_hessian(scaled, x)
    assert_allclose(original["evaluation"].cost, comparison["evaluation"].cost,
                    rtol=2e-10, atol=2e-12)
    assert_allclose(original["gradient"], comparison["gradient"], rtol=2e-8, atol=2e-10)
    assert_allclose(original["hessian"], comparison["hessian"], rtol=2e-7, atol=2e-9)


def test_failed_amplitude_branch_cannot_supply_profile_hessian():
    problem, x = _example(3, amplitude=True)
    failed = JointProblem(problem.elements_m, [replace(problem.groups[0],
        amplitude_maxiter=0, amplitude_newton_steps=0)], backend="qr")
    with pytest.raises(AmplitudeProfileError, match="failed"):
        profile_gradient_hessian(failed, x)


def test_frequency_blocks_have_shared_geometry_and_independent_gain_curvature():
    base, xbase = _example(3)
    other_sessions = [replace(session, name=f"other-{session.name}",
        frequency_hz=12.1e9, field=(1.2+.3j)*session.field)
        for session in base.groups[0].sessions]
    other = replace(base.groups[0], name="other", sessions=other_sessions)
    problem = JointProblem(base.elements_m, [base.groups[0], other], backend="qr")
    x = problem.initial_parameters((61.4321, -.87, -2.9),
        {"s1": 1.07+.09j, "s2": .97-.08j,
         "other-s1": .93-.07j, "other-s2": 1.1+.12j})
    result = profile_gradient_hessian(problem, x)
    assert_allclose(result["hessian"][3:7, 7:], 0, atol=1e-13)
    direction = np.linspace(-.7, .8, problem.parameter_count)
    direction /= np.linalg.norm(direction)
    h = 3e-5
    numeric = (_gradient(problem, x+h*direction)-_gradient(problem, x-h*direction))/(2*h)
    assert _relative_error(numeric, result["hessian"]@direction) < 1e-5
    assert_allclose(xbase[:3], x[:3], atol=0, rtol=0)


def test_polish_and_trf_reach_same_interior_code_control():
    problem, x = _example(3, positive_control=True)
    trf = problem.fit(start_parameters=x, max_nfev=150,
                      ftol=1e-12, gtol=1e-9, xtol=1e-11)
    polished = polish_profile(problem, x, maxiter=250, gtol=1e-9)
    assert trf.success and polished.success
    assert_allclose(polished.geometry_mm, trf.geometry_mm, atol=2e-4, rtol=0)
    assert_allclose(polished.cost, trf.cost, atol=1e-12, rtol=1e-7)
    assert_allclose(polished.evaluation.cost, problem.evaluate(polished.x).cost,
                    atol=0, rtol=0)


def test_polish_preserves_hard_bounds_for_every_evaluation_and_failed_budget():
    problem, x = _example(3, positive_control=True)
    x[:3] = (-1., -1., 1.)
    original = problem.evaluate
    attempts = []

    def watched(parameters):
        parameters = np.asarray(parameters)
        attempts.append(parameters.copy())
        assert np.all(parameters[:3] >= -1.)
        assert np.all(parameters[:3] <= 1.)
        return original(parameters)

    problem.evaluate = watched
    result = polish_profile(problem, x, maxiter=1, gtol=1e-12)
    assert attempts
    assert np.all(result.x[:3] >= -1.) and np.all(result.x[:3] <= 1.)
    assert not result.success
    assert np.isfinite(result.cost)


def test_polish_can_leave_exact_bound_for_a_nearby_interior_minimum():
    truth = np.array((51.2345, -8.1234, 8.4567))
    problem, x = _example(3, positive_control=True, truth_geometry=truth)
    x[:3] = (-1., -1., 1.)
    initial_cost = problem.evaluate(x).cost
    result = polish_profile(problem, x, maxiter=250, gtol=1e-9)
    assert result.success
    assert result.cost < initial_cost*1e-6
    assert_allclose(result.geometry_mm, truth, atol=3e-4, rtol=0)


def test_checkpoint_stop_preserves_fixed_axes_and_unwrapped_gain_phase():
    import json

    problem, x = _example(3, positive_control=True)
    x[4] += 4*np.pi
    checkpoints = []

    def checkpoint(record):
        json.dumps(record)
        checkpoints.append(record)
        if record["event"] == "iteration":
            raise StopIteration

    result = polish_profile(problem, x, fixed_geometry=(True, False, True),
        maxiter=100, checkpoint_callback=checkpoint)
    assert not result.success
    assert [record["event"] for record in checkpoints] == ["initial", "iteration", "final"]
    assert_allclose(result.x[[0, 2]], x[[0, 2]], atol=0, rtol=0)
    assert result.x[4] > 3*np.pi
    assert_allclose(checkpoints[-1]["x"], result.x, atol=0, rtol=0)
    assert checkpoints[-1]["cost"] == result.cost
    assert "b" not in checkpoints[-1] and "hessian" not in checkpoints[-1]


def test_exact_curvature_backend_hook_avoids_cpu_kernel_fallback(monkeypatch):
    import array_reconstruct.joint_newton as module
    calls=[]
    payload={"gradient":np.ones(3),"hessian":np.eye(3),"diagnostics":{"backend":"stub"}}
    class Backend:
        def profile_gradient_hessian(self,x):
            calls.append(np.array(x,copy=True))
            return payload
    def forbidden(*args,**kwargs):
        raise AssertionError("CPU kernel executed instead of backend curvature")
    monkeypatch.setattr(module,"scalar_kernel",forbidden)
    monkeypatch.setattr(module,"scalar_kernel_second",forbidden)
    result=profile_gradient_hessian(Backend(),np.array((.1,.2,.3)))
    assert result is payload
    assert len(calls)==1
    assert_allclose(calls[0],(.1,.2,.3),atol=0,rtol=0)

