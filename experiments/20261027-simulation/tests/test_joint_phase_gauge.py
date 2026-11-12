"""Synthetic contracts for an optional residual representation.

These checks do not open measurements, previous runs, or confirmation datasets.
The known-response fit is an explicit numerical positive control. Objective
equivalence is asserted only where the response-sum guard is valid.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_geometry import (
    AmplitudeProfileError, FrequencyGroup, JointProblem, ProfileEvaluation, Session,
    parameters_to_geometry,
)
from array_reconstruct.joint_phase_gauge import (
    PhaseGaugeError, fit_phase_gauge, phase_gauge_evaluate,
)
from array_reconstruct.physics import C0_M_S, planar_scan, rectangular_array


def _independent_kernel(elements, xy, frequency, geometry):
    geometry = np.asarray(geometry)
    points = np.column_stack((xy + geometry[None, 1:] * 1e-3,
                              np.full(len(xy), geometry[0] * 1e-3)))
    radius = np.linalg.norm(points[:, None, :] - elements[None, :, :], axis=-1)
    return ((geometry[0] * 1e-3) ** 2
            * np.exp(-2j * np.pi * frequency / C0_M_S * radius) / radius ** 3)


def _problem(*, amplitude=False, two_groups=False, positive_control=False):
    # Separate code-test stream, unrelated to development or confirmation seeds.
    rng = np.random.default_rng(79031)
    elements = rectangular_array(3, 3, .011)
    xy = planar_scan(8, 7, .008)[:, :2]
    truth_geometry = np.array((62.3456, 1.2345, -3.4567))
    groups = []
    gains = {}
    for gi, frequency in enumerate((12.7e9, 12.1e9)[:2 if two_groups else 1]):
        truth_b = np.exp(rng.uniform(-.25, .25, len(elements))
                         + 1j * (.4 * gi + .12 * rng.normal(size=len(elements))))
        sessions = []
        for si, (ax, ay) in enumerate(((0., 0.), (.23, -.11), (-.17, .19))):
            name = f"g{gi}s{si}"
            u = np.exp(2j * np.pi * frequency / C0_M_S
                       * (elements[:, 0] * ax + elements[:, 1] * ay))
            gain = np.exp(.035 * si + .065j * si)
            field = gain * (_independent_kernel(elements, xy, frequency,
                                                truth_geometry) @ (u * truth_b))
            if not positive_control:
                field += .02 * (rng.normal(size=len(xy)) + 1j * rng.normal(size=len(xy)))
            sessions.append(Session(name, frequency, xy, u, field,
                                    .02 * np.linspace(.7, 1.1, len(xy))))
            if si:
                gains[name] = (1.07 + .09j) if si == 1 else (.97 - .08j)
        if positive_control:
            # True b is supplied only to this labelled code-correctness control.
            group = FrequencyGroup(f"g{gi}", sessions, .6, b0=truth_b,
                                   template_prior=truth_b, template_strength=.08)
        else:
            b0 = .35 * np.exp(.09j * np.arange(len(elements)))
            template = np.linspace(.9, 1.1, len(elements)) * np.exp(.04j * np.arange(len(elements)))
            group = FrequencyGroup(f"g{gi}", sessions, .08, b0=b0,
                                   template_prior=template, template_strength=.035)
            if amplitude:
                group = replace(group, amplitude_prior=np.linspace(.8, 1.2, len(elements)),
                                amplitude_strength=.05, amplitude_n_starts=1,
                                amplitude_stationarity_tol=1e-13, amplitude_epsilon=1e-6)
        groups.append(group)
    problem = JointProblem(elements, groups, backend="qr")
    x = problem.initial_parameters((61.4321, -.87, -2.9), gains)
    return problem, x, truth_geometry


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(actual),
                                                  np.linalg.norm(expected), 1e-14)


def _row_layout(problem, evaluation):
    """Independent block indices in [all real, all imaginary, real amplitude]."""
    total = sum(sum(len(s.field) for s in g.sessions)
                + len(evaluation.penalty_residuals[g.name]) for g in problem.groups)
    cursor = 0
    for group in problem.groups:
        m = sum(len(s.field) for s in group.sessions)
        stop = cursor + m + len(evaluation.penalty_residuals[group.name])
        yield group, np.r_[np.arange(cursor, cursor + m),
                          np.arange(cursor + total, cursor + m + total)], total
        cursor = stop


@pytest.mark.parametrize("amplitude", [False, True])
def test_objective_gradient_data_and_real_amplitude_blocks_are_preserved(amplitude):
    problem, x, _ = _problem(amplitude=amplitude, two_groups=True)
    raw = problem.evaluate(x)
    old_residual, old_jacobian = raw.residual.copy(), raw.jacobian.copy()
    old_b = {key: value.copy() for key, value in raw.b.items()}
    old_db = {key: value.copy() for key, value in raw.db.items()}
    gauged = phase_gauge_evaluate(problem, x)
    assert gauged["evaluation"] is raw
    assert_allclose(gauged["residual"] @ gauged["residual"],
                    old_residual @ old_residual, rtol=3e-13, atol=3e-14)
    assert_allclose(gauged["jacobian"].T @ gauged["residual"],
                    old_jacobian.T @ old_residual, rtol=3e-12, atol=3e-13)
    assert np.linalg.norm(old_jacobian.T @ old_residual) > 1e-6
    assert np.linalg.norm(gauged["residual"] - old_residual) > 1e-4
    # A nontrivial representation changes GN curvature even with the same F/g.
    assert _relative_error(gauged["jacobian"].T @ gauged["jacobian"],
                           old_jacobian.T @ old_jacobian) > 1e-5
    for group, data_rows, total_complex in _row_layout(problem, raw):
        assert_allclose(gauged["residual"][data_rows], old_residual[data_rows], atol=0, rtol=0)
        assert_allclose(gauged["jacobian"][data_rows], old_jacobian[data_rows], atol=0, rtol=0)
        assert_allclose(raw.b[group.name], old_b[group.name], atol=0, rtol=0)
        assert_allclose(raw.db[group.name], old_db[group.name], atol=0, rtol=0)
        b = raw.b[group.name]
        rotation = np.exp(-1j * np.angle(np.sum(b)))
        # Rotate the complete nonzero-b0 residual, rather than changing b alone.
        n = len(b)
        original_penalty = raw.penalty_residuals[group.name]
        assert_allclose(original_penalty[:n], group.lambda_reg * (group.b0 - b),
                        rtol=1e-13, atol=1e-14)
        assert np.linalg.norm(original_penalty[n:]) > 1e-4
        assert_allclose(gauged["penalty_residuals"][group.name],
                        rotation * original_penalty, rtol=2e-13, atol=2e-14)
    assert_allclose(gauged["residual"][2 * total_complex:],
                    old_residual[2 * total_complex:], atol=0, rtol=0)
    assert_allclose(gauged["jacobian"][2 * total_complex:],
                    old_jacobian[2 * total_complex:], atol=0, rtol=0)
    if amplitude:
        assert len(gauged["residual"]) > 2 * total_complex
        assert np.linalg.norm(gauged["residual"][2 * total_complex:]) > 1e-4
    # The cached physical profile must not be mutated by the optional wrapper.
    assert_allclose(raw.residual, old_residual, atol=0, rtol=0)
    assert_allclose(raw.jacobian, old_jacobian, atol=0, rtol=0)
    assert not gauged["diagnostics"]["physical_objective_changed"]
    assert not gauged["diagnostics"]["global_profile_smoothness_established"]


@pytest.mark.parametrize("amplitude", [False, True])
def test_full_profile_chain_rule_by_complete_resolves(amplitude):
    problem, x, _ = _problem(amplitude=amplitude)
    center = phase_gauge_evaluate(problem, x)
    assert np.all(problem.groups[0].b0 != 0)
    assert problem.groups[0].template_strength > 0
    assert np.linalg.norm(center["evaluation"].penalty_residuals["g0"]) > .01
    assert np.linalg.norm(center["diagnostics"]["groups"]["g0"]["removed_phase_derivative"]) > .1
    # Full residual differences are essential: gradient invariance alone cannot
    # detect a missing, or wrong-sign, residual-rotation chain-rule term.
    for h in (1e-4, 3e-5):
        numeric = np.column_stack([
            (phase_gauge_evaluate(problem, x + h * axis)["residual"]
             - phase_gauge_evaluate(problem, x - h * axis)["residual"]) / (2 * h)
            for axis in np.eye(problem.parameter_count)])
        assert _relative_error(numeric, center["jacobian"]) < 1e-5


def test_group_rotations_use_their_own_b_and_have_no_cross_gain_derivatives():
    problem, x, _ = _problem(two_groups=True)
    gauged = phase_gauge_evaluate(problem, x)
    raw = gauged["evaluation"]
    rotations = []
    for gi, group in enumerate(problem.groups):
        b = raw.b[group.name]
        rotation = np.sum(b).conjugate() / abs(np.sum(b))
        rotations.append(rotation)
        inactive = slice(7, 11) if gi == 0 else slice(3, 7)
        assert_allclose(gauged["penalty_jacobians"][group.name][:, inactive],
                        0, atol=2e-14, rtol=0)
        assert_allclose(gauged["penalty_residuals"][group.name],
                        rotation * raw.penalty_residuals[group.name], atol=2e-14, rtol=2e-13)
    assert abs(rotations[0] - rotations[1]) > .1
    direction = np.linspace(-.7, .8, problem.parameter_count)
    direction /= np.linalg.norm(direction)
    h = 3e-5
    numeric = (phase_gauge_evaluate(problem, x + h * direction)["residual"]
               - phase_gauge_evaluate(problem, x - h * direction)["residual"]) / (2 * h)
    assert _relative_error(numeric, gauged["jacobian"] @ direction) < 1e-5


def _toy_profile(response, *, penalty_strength=.2):
    """An analytic synthetic profile isolates the guard and arg branch cut."""
    group = SimpleNamespace(name="g", lambda_reg=penalty_strength, template_strength=0.,
                            sessions=(SimpleNamespace(name="s", field=np.zeros(2)),))
    problem = SimpleNamespace(groups=(group,), parameter_count=3)

    def evaluate(x):
        x = np.asarray(x, float)
        b, db = response(x)
        data = np.array((x[1] + .2j, -.1 + 1j * x[2]))
        data_j = np.array(((0., 1., 0.), (0., 0., 1j)), complex)
        b0 = np.array((.3 + .2j, .1 - .07j))
        penalty = penalty_strength * (b0 - b)
        penalty_j = -penalty_strength * db
        r = np.r_[data, penalty]
        j = np.vstack((data_j, penalty_j))
        return ProfileEvaluation(x.copy(), parameters_to_geometry(x[:3]),
            np.r_[r.real, r.imag], np.vstack((j.real, j.imag)),
            {"g": b}, {"g": db}, {"s": 1. + 0j}, {"s": data}, {"s": data},
            {"g": penalty}, {"s": data_j}, {"s": data_j}, {"g": penalty_j},
            {}, {}, {})

    problem.evaluate = evaluate
    return problem


@pytest.mark.parametrize("b", [np.zeros(2), np.array((1., -1.)),
                              np.array((1., -1. + 1e-6))])
def test_zero_or_low_coherence_is_a_hard_error_with_evidence(b):
    problem = _toy_profile(lambda x: (b.astype(complex), np.zeros((2, 3), complex)))
    with pytest.raises(PhaseGaugeError) as caught:
        phase_gauge_evaluate(problem, np.zeros(3))
    diagnostics = caught.value.diagnostics
    assert diagnostics["group"] == "g"
    assert diagnostics["coherence"] < diagnostics["min_coherence"]
    assert diagnostics["representation_used"] is False
    assert diagnostics["x"] == [0., 0., 0.]


@pytest.mark.parametrize("b, db", [
    (np.array((1., np.inf), complex), np.zeros((2, 3), complex)),
    (np.array((1., np.nan), complex), np.zeros((2, 3), complex)),
    (np.ones(2, complex), np.full((2, 3), np.nan + 0j)),
])
def test_nonfinite_response_or_profile_derivative_cannot_bypass_guard(b, db):
    problem = _toy_profile(lambda x: (b, db))
    with pytest.raises(PhaseGaugeError):
        phase_gauge_evaluate(problem, np.zeros(3))


def test_guard_is_dimensionless_and_complex_scale_independent():
    for delta, accepted in ((.001, False), (.004, True)):
        b = np.array((1., -1. + delta), complex)
        coherence = abs(sum(b)) / (np.sqrt(len(b)) * np.linalg.norm(b))
        for scale in (1., 1e-8 * (2. - 3j), 1e8 * (-1. + .4j)):
            problem = _toy_profile(lambda x, bb=scale*b: (bb, np.zeros((2, 3), complex)))
            if accepted:
                result = phase_gauge_evaluate(problem, np.zeros(3))
                assert result["diagnostics"]["groups"]["g"]["coherence"] == pytest.approx(coherence, rel=2e-12)
            else:
                with pytest.raises(PhaseGaugeError):
                    phase_gauge_evaluate(problem, np.zeros(3))


@pytest.mark.parametrize("minimum", [0., -.1, 1.1, np.inf, np.nan])
def test_invalid_guard_threshold_is_rejected(minimum):
    problem = _toy_profile(lambda x: (np.ones(2, complex), np.zeros((2, 3), complex)))
    with pytest.raises(ValueError, match="min_coherence"):
        phase_gauge_evaluate(problem, np.zeros(3), min_coherence=minimum)


def test_arg_branch_cut_is_smooth_away_from_zero_sum():
    def response(x):
        b = np.array((1., .9)) * np.exp(1j * (np.pi + x[0]))
        db = np.zeros((2, 3), complex)
        db[:, 0] = 1j * b
        return b, db
    problem = _toy_profile(response)
    x = np.zeros(3)
    center = phase_gauge_evaluate(problem, x)
    for h in (1e-3, 1e-4):
        step = np.array((h, 0., 0.))
        numeric = (phase_gauge_evaluate(problem, x + step)["residual"]
                   - phase_gauge_evaluate(problem, x - step)["residual"]) / (2 * h)
        assert _relative_error(numeric, center["jacobian"][:, 0]) < 1e-6


def test_absent_complex_penalties_need_no_response_sum_coordinate():
    problem = _toy_profile(lambda x: (np.array((1., -1.), complex),
                                     np.zeros((2, 3), complex)), penalty_strength=0.)
    raw = problem.evaluate(np.zeros(3))
    gauged = phase_gauge_evaluate(problem, np.zeros(3))
    assert not gauged["diagnostics"]["groups"]["g"]["enabled"]
    assert_allclose(gauged["residual"], raw.residual, atol=0, rtol=0)
    assert_allclose(gauged["jacobian"], raw.jacobian, atol=0, rtol=0)


def test_feasible_one_sided_derivatives_at_geometry_bounds_and_no_clipping():
    problem, x, _ = _problem()
    x[:3] = (-1., -1., 1.)
    center = phase_gauge_evaluate(problem, x)
    assert_allclose(center["evaluation"].geometry_mm, (50., -10., 10.), atol=0, rtol=0)
    for axis, direction in enumerate((1., 1., -1.)):
        h = 1e-5
        step = np.eye(problem.parameter_count)[axis] * h * direction
        r1 = phase_gauge_evaluate(problem, x + step)["residual"]
        r2 = phase_gauge_evaluate(problem, x + 2 * step)["residual"]
        numeric = direction * (-3 * center["residual"] + 4 * r1 - r2) / (2 * h)
        assert _relative_error(numeric, center["jacobian"][:, axis]) < 1e-5
    with pytest.raises(ValueError, match="bounds"):
        phase_gauge_evaluate(problem, x - np.eye(problem.parameter_count)[0] * 1e-6)


def test_known_response_positive_control_same_optimum_as_original_trf():
    problem, x, truth_geometry = _problem(positive_control=True)
    raw = problem.fit(start_parameters=x, max_nfev=200, ftol=1e-12, gtol=1e-10, xtol=1e-12)
    gauged = fit_phase_gauge(problem, x, max_nfev=200,
                            ftol=1e-12, gtol=1e-10, xtol=1e-12)
    assert raw.success and gauged.success
    assert_allclose(gauged.geometry_mm, raw.geometry_mm, atol=2e-4, rtol=0)
    assert_allclose(gauged.geometry_mm, truth_geometry, atol=2e-4, rtol=0)
    assert gauged.cost < 1e-14 and raw.cost < 1e-14
    assert_allclose(gauged.cost, problem.evaluate(gauged.x).cost, atol=1e-15, rtol=1e-10)
    final = phase_gauge_evaluate(problem, gauged.x)
    assert_allclose(gauged.evaluation.residual, final["residual"], atol=0, rtol=0)
    assert_allclose(gauged.evaluation.jacobian, final["jacobian"], atol=0, rtol=0)


def test_one_evaluation_budget_preserves_failure_and_json_checkpoint():
    problem, x, _ = _problem()
    records = []
    def checkpoint(record):
        json.dumps(record, allow_nan=False)
        records.append(record)
    fit = fit_phase_gauge(problem, x, max_nfev=1, checkpoint_callback=checkpoint)
    assert not fit.success and fit.status == 0 and fit.nfev == 1
    assert_allclose(fit.x, x, atol=0, rtol=0)
    assert [record["event"] for record in records] == ["initial", "final"]
    assert records[-1]["x"] == fit.x.tolist()
    assert records[-1]["cost"] == fit.cost
    assert not records[-1]["success"]
    assert "jacobian" not in records[-1] and "b" not in records[-1]
    assert fit.counts["phase_gauge_calls"] >= 1
    assert fit.counts["phase_gauge_cache_hits"] >= 1
    assert_allclose(fit.cost, problem.evaluate(x).cost, atol=1e-14, rtol=1e-13)


def test_checkpoint_stop_and_restart_from_saved_parameters_reach_same_code_control():
    problem, x, truth_geometry = _problem(positive_control=True)
    records = []
    def checkpoint(record):
        json.dumps(record, allow_nan=False)
        records.append(record)
        if record["event"] == "iteration":
            raise StopIteration
    stopped = fit_phase_gauge(problem, x, max_nfev=200, checkpoint_callback=checkpoint)
    assert not stopped.success and stopped.status == -2
    assert [record["event"] for record in records] == ["initial", "iteration", "final"]
    assert_allclose(records[-1]["x"], stopped.x, atol=0, rtol=0)
    assert records[-1]["cost"] == stopped.cost
    assert records[1]["nfev"] >= records[0]["nfev"]
    resumed = fit_phase_gauge(problem, records[-1]["x"], max_nfev=200,
                             ftol=1e-12, gtol=1e-10, xtol=1e-12)
    assert resumed.success
    assert_allclose(resumed.geometry_mm, truth_geometry, atol=2e-4, rtol=0)
    assert resumed.cost < 1e-14


@pytest.mark.parametrize("failure_kind", ["phase", "amplitude"])
def test_initial_guard_failure_is_json_serializable_and_never_returns_a_fit(monkeypatch, failure_kind):
    problem, x, _ = _problem()
    physical_evaluate = problem.evaluate

    def failed_evaluate(point):
        if failure_kind == "amplitude":
            raise AmplitudeProfileError({"group": "g0", "x": point.tolist(),
                                         "reason": "synthetic injected inner failure"})
        raw = physical_evaluate(point)
        return replace(raw, b={"g0": np.zeros_like(raw.b["g0"])})

    monkeypatch.setattr(problem, "evaluate", failed_evaluate)
    records = []
    error_class = PhaseGaugeError if failure_kind == "phase" else AmplitudeProfileError
    with pytest.raises(error_class) as caught:
        fit_phase_gauge(problem, x, max_nfev=10, checkpoint_callback=records.append)
    assert [record["event"] for record in records] == ["failed"]
    json.dumps(records[0], allow_nan=False)
    assert records[0]["cost"] is None and records[0]["nfev"] == 0
    assert records[0]["x"] == x.tolist()
    assert records[0]["failed_attempt"] == caught.value.diagnostics
    assert records[0]["status"] == -3 and not records[0]["success"]


@pytest.mark.parametrize("after_accepted_iteration", [False, True])
@pytest.mark.parametrize("failure_kind", ["phase", "amplitude"])
def test_trial_failure_preserves_last_accepted_checkpoint(monkeypatch, after_accepted_iteration,
                                                        failure_kind):
    problem, x, _ = _problem()
    physical_evaluate = problem.evaluate
    records = []
    state = {"armed": not after_accepted_iteration, "accepted_x": x.copy()}

    def checkpoint(record):
        json.dumps(record, allow_nan=False)
        records.append(record)
        if record["event"] == "iteration":
            state["armed"] = True
            state["accepted_x"] = np.asarray(record["x"])

    def failed_trial(point):
        if state["armed"] and not np.array_equal(point, state["accepted_x"]):
            if failure_kind == "amplitude":
                raise AmplitudeProfileError({"group": "g0", "x": point.tolist(),
                                             "reason": "synthetic injected trial failure"})
            raw = physical_evaluate(point)
            return replace(raw, b={"g0": np.zeros_like(raw.b["g0"])})
        return physical_evaluate(point)

    monkeypatch.setattr(problem, "evaluate", failed_trial)
    error_class = PhaseGaugeError if failure_kind == "phase" else AmplitudeProfileError
    with pytest.raises(error_class) as caught:
        fit_phase_gauge(problem, x, max_nfev=100, checkpoint_callback=checkpoint)
    expected = ["initial", "iteration", "failed"] if after_accepted_iteration else ["initial", "failed"]
    assert [record["event"] for record in records] == expected
    accepted, failed = records[-2:]
    assert failed["x"] == accepted["x"]
    assert failed["cost"] == accepted["cost"]
    assert failed["nfev"] == accepted["nfev"]
    assert failed["failed_attempt"] == caught.value.diagnostics
    assert failed["failed_attempt"]["x"] != accepted["x"]
    assert failed["status"] == -3 and not failed["success"]


def test_partial_fixed_geometry_gain_bounds_and_unwrapped_phase():
    problem, x, truth_geometry = _problem(positive_control=True)
    x[:3] = problem.initial_parameters(truth_geometry)[:3]
    x[1] -= .1
    fixed = fit_phase_gauge(problem, x, fixed_geometry=(True, False, True),
                           max_nfev=200, gtol=1e-10, xtol=1e-12, ftol=1e-12)
    assert fixed.success
    assert_allclose(fixed.x[[0, 2]], x[[0, 2]], atol=0, rtol=0)
    assert_allclose(fixed.geometry_mm, truth_geometry, atol=2e-4, rtol=0)
    shifted = x.copy()
    shifted[4] += 10 * np.pi
    fit = fit_phase_gauge(problem, shifted, fixed_geometry=True, max_nfev=1)
    assert_allclose(fit.x, shifted, atol=0, rtol=0)
    assert fit.x[4] > np.pi
    invalid = x.copy()
    invalid[3] = 3.01
    with pytest.raises(ValueError, match="bounds"):
        fit_phase_gauge(problem, invalid, max_nfev=1)
    with pytest.raises(ValueError, match="fixed_geometry"):
        fit_phase_gauge(problem, x, fixed_geometry=(True, False), max_nfev=1)


def test_all_fixed_variables_return_single_inner_solve():
    base, _, _ = _problem()
    group = replace(base.groups[0], sessions=base.groups[0].sessions[:1])
    problem = JointProblem(base.elements_m, [group], backend="qr")
    x = problem.initial_parameters((61.23456789, -.456789, 2.3456789))
    records = []
    fit = fit_phase_gauge(problem, x, fixed_geometry=True, checkpoint_callback=records.append)
    assert fit.success and fit.status == 1 and fit.nfev == 1
    assert_allclose(fit.x, x, atol=0, rtol=0)
    assert_allclose(fit.geometry_mm, parameters_to_geometry(x), atol=0, rtol=0)
    assert [record["event"] for record in records] == ["initial", "final"]
    assert fit.evaluation.diagnostics["phase_gauge"]["common_stationarity"] == {
        "projected_gradient_inf": 0., "trf_scaled_gradient_inf": 0.}
