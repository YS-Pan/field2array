"""Synthetic mathematical contracts; no measurements or archives are accessed."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_reconstruct.joint_geometry import (
    AmplitudeProfileError, FrequencyGroup, JointProblem, Session, deterministic_starts_mm,
    geometry_to_parameters, parameters_to_geometry, scalar_kernel,
    summarize_multistart,
)
from array_reconstruct.physics import C0_M_S, planar_scan, rectangular_array


def example(backend="qr", normal_condition_max=1e8):
    rng = np.random.default_rng(43101)
    elements = rectangular_array(4, 4, .011)
    xy = planar_scan(9, 8, .008)[:, :2]
    frequency = 12.7e9
    truth = np.array((62.3456, 1.2345, -3.4567))
    b = (1+.25*rng.normal(size=16))*np.exp(.2j*rng.normal(size=16))
    sessions = []
    for index, angle in enumerate((0., .35, -.25)):
        u = np.exp(1j*2*np.pi*frequency/C0_M_S*elements[:, 0]*angle)
        gain = np.exp(.12*index+.17j*index)
        field = gain*scalar_kernel(elements, xy, frequency, truth)@(u*b)
        field += .01*(rng.normal(size=len(xy))+1j*rng.normal(size=len(xy)))
        sessions.append(Session(f"s{index}", frequency, xy, u, field,
                                np.linspace(.7, 1.1, len(xy))*.01))
    group = FrequencyGroup("f", sessions, .08, .4+.03j*rng.normal(size=16))
    problem = JointProblem(elements, [group], backend=backend,
                           normal_condition_max=normal_condition_max)
    x = problem.initial_parameters((61.4321, -.87, -2.9), {"s1": 1.13+.11j, "s2": .9-.1j})
    return problem, x


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("q", [0., 2., 4.])
def test_kernel_and_normalized_derivatives_match_independent_oracle(q, sign):
    elements = rectangular_array(2, 2, .011)
    xy = np.array(((-.019, -.011), (.0012, .0173)))
    frequency, geometry = 12.5e9, np.array((62.3456, 1.2345, -3.4567))
    samples = np.column_stack((xy+geometry[None, 1:]*1e-3,
                               np.full(len(xy), geometry[0]*1e-3)))
    radius = np.linalg.norm(samples[:, None, :]-elements[None, :, :], axis=-1)
    expected = (geometry[0]*1e-3)**q*np.exp(sign*2j*np.pi*frequency/C0_M_S*radius)/radius**(q+1)
    assert_allclose(scalar_kernel(elements, xy, frequency, geometry, q, sign), expected, rtol=2e-14)
    for j in range(3):
        analytic = scalar_kernel(elements, xy, frequency, geometry, q, sign, derivative=j)
        errors = []
        for h_mm in (.01, .001, .0001):
            step = np.eye(3)[j]*h_mm
            finite = (scalar_kernel(elements, xy, frequency, geometry+step, q, sign)
                      -scalar_kernel(elements, xy, frequency, geometry-step, q, sign))/(2*h_mm/10)
            errors.append(np.linalg.norm(finite-analytic)/np.linalg.norm(analytic))
        assert max(errors) <= 1e-5


def test_profile_full_data_penalty_b_and_validation_jacobians():
    problem, x = example()
    result = problem.evaluate(x)
    validation = {"f": [replace(problem.groups[0].sessions[1],
                                samples_xy_m=problem.groups[0].sessions[1].samples_xy_m+.00031)]}
    val = problem.validation_evaluate(x, validation)
    for h_mm in (.01, .001, .0001):
        for j in range(problem.parameter_count):
            h = h_mm/10 if j < 3 else h_mm/10
            step = np.eye(problem.parameter_count)[j]*h
            plus, minus = problem.evaluate(x+step), problem.evaluate(x-step)
            for finite, analytic in (
                ((plus.residual-minus.residual)/(2*h), result.jacobian[:, j]),
                ((plus.b["f"]-minus.b["f"])/(2*h), result.db["f"][:, j]),
                ((plus.data_residuals["s1"]-minus.data_residuals["s1"])/(2*h), result.data_jacobians["s1"][:, j]),
                ((plus.penalty_residuals["f"]-minus.penalty_residuals["f"])/(2*h), result.penalty_jacobians["f"][:, j]),
            ):
                assert np.linalg.norm(finite-analytic)/np.linalg.norm(analytic) < 1e-5
            finite_val = (problem.validation_evaluate(x+step, validation)["residual"]
                          -problem.validation_evaluate(x-step, validation)["residual"])/(2*h)
            assert np.linalg.norm(finite_val-val["jacobian"][:, j])/np.linalg.norm(val["jacobian"][:, j]) < 1e-5
    assert result.gains["s0"] == 1
    assert result.diagnostics["groups"]["f"]["stationarity"] <= 1e-8
    assert np.linalg.norm(result.penalty_jacobians["f"]) > 0


@pytest.mark.parametrize("backend", ["svd", "normal"])
def test_stable_backends_and_normal_verification(backend):
    qr, x = example()
    comparison, _ = example(backend)
    a, b = qr.evaluate(x), comparison.evaluate(x)
    assert_allclose(a.b["f"], b.b["f"], rtol=1e-10, atol=1e-11)
    assert_allclose(a.residual, b.residual, rtol=1e-9, atol=1e-11)
    assert_allclose(a.jacobian, b.jacobian, rtol=1e-8, atol=1e-10)
    if backend == "normal":
        assert comparison.qr_checks[0]["passed"]
        assert "jacobian_relative_difference" in comparison.qr_checks[0]
        assert b.diagnostics["groups"]["f"]["backend_used"] == "normal"


def test_normal_condition_guard_falls_back_to_augmented_qr():
    problem, x = example("normal", normal_condition_max=1.01)
    result = problem.evaluate(x)
    assert result.diagnostics["groups"]["f"]["backend_used"] == "qr"
    assert "condition threshold" in result.diagnostics["groups"]["f"]["fallback_reason"]
    assert problem.counts["normal_fallbacks"] == 1


def test_hard_boundary_one_sided_jacobian_and_no_clip_or_round():
    problem, x = example()
    x[:3] = (-1., -1., 1.)
    result = problem.evaluate(x)
    assert_allclose(result.geometry_mm, (50., -10., 10.))
    for j, direction in enumerate((1, 1, -1)):
        h = .0001/10
        step = np.eye(problem.parameter_count)[j]*h*direction
        finite = direction*(-3*result.residual+4*problem.evaluate(x+step).residual
                            -problem.evaluate(x+2*step).residual)/(2*h)
        assert np.linalg.norm(finite-result.jacobian[:, j])/np.linalg.norm(result.jacobian[:, j]) <= 1e-5
    with pytest.raises(ValueError, match="bounds"):
        problem.evaluate(x-np.eye(problem.parameter_count)[0]*1e-6)
    geometry = (62.345678901, 1.23456789, -3.45678901)
    assert_allclose(parameters_to_geometry(geometry_to_parameters(geometry)), geometry, atol=1e-14)


def test_frequency_blocks_independent_and_one_session_m1_has_no_gain_parameters():
    base, _ = example()
    s = base.groups[0].sessions[0]
    other = replace(s, name="other", frequency_hz=12.1e9, field=2j*s.field)
    p = JointProblem(base.elements_m, [FrequencyGroup("f1", [s], .08), FrequencyGroup("f2", [other], .08)])
    result = p.evaluate(p.initial_parameters())
    assert p.parameter_count == 3
    assert result.gains == {s.name: 1, other.name: 1}
    assert not np.allclose(result.b["f1"], result.b["f2"])
    fit = p.fit(fixed_geometry=True)
    assert fit.success and fit.nfev == 1 and fit.status == 1
    assert_allclose(fit.geometry_mm, (60, 0, 0))


def test_fixed_geometry_still_fits_session_gains_and_preserves_nonconvergence():
    p, _ = example()
    fit = p.fit((60.1234, 1.9876, -2.3456), fixed_geometry=True, max_nfev=2)
    assert_allclose(fit.geometry_mm, (60.1234, 1.9876, -2.3456), atol=1e-14)
    assert fit.nfev == 2 and not fit.success and fit.status == 0
    assert fit.counts["linear_factorizations"] > 0


def test_problem_freezes_inputs_repeatability_and_twelve_starts():
    p, x = example()
    a = p.evaluate(x)
    b = p.evaluate(x.copy())
    assert a is b
    assert p.counts["cache_hits"] == 1
    for s in p.groups[0].sessions:
        assert not s.weight.flags.writeable
        assert not s.steering.flags.writeable
    starts = deterministic_starts_mm()
    assert starts.shape == (12, 3)
    assert len(np.unique(starts, axis=0)) == 12
    assert_allclose(starts[0], (60, 0, 0))


def test_rank_deficient_unregularized_system_and_invalid_inputs_are_rejected():
    p, _ = example()
    s = replace(p.groups[0].sessions[0], samples_xy_m=np.zeros((1, 2)), field=np.ones(1), weight=1)
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        JointProblem(p.elements_m, [FrequencyGroup("under", [s], 0)]).evaluate(np.zeros(3))
    with pytest.raises(ValueError):
        replace(s, weight=-1)
    with pytest.raises(ValueError, match="unverified"):
        JointProblem(p.elements_m, p.groups, normal_qr_check="none")
    with pytest.raises(ValueError, match="exactly one frequency"):
        FrequencyGroup("bad", [s, replace(s, name="other", frequency_hz=1e9)], 1.)


def test_template_prior_eliminates_unknown_common_complex_scale_and_has_exact_jacobian():
    base, x = example()
    template = np.linspace(.7, 1.3, len(base.elements_m))*np.exp(.05j*np.arange(len(base.elements_m)))
    group = replace(base.groups[0], template_prior=template, template_strength=.13)
    p = JointProblem(base.elements_m, [group])
    scaled = JointProblem(base.elements_m, [replace(group, template_prior=template*(2.3-1.7j))])
    result, comparison = p.evaluate(x), scaled.evaluate(x)
    assert_allclose(result.b["f"], comparison.b["f"], rtol=2e-13, atol=1e-13)
    assert_allclose(result.jacobian, comparison.jacobian, rtol=1e-10, atol=1e-12)
    assert result.penalty_residuals["f"].shape == (32,)
    for j in range(p.parameter_count):
        h = 1e-5
        step = np.eye(p.parameter_count)[j]*h
        finite = (p.evaluate(x+step).residual-p.evaluate(x-step).residual)/(2*h)
        assert np.linalg.norm(finite-result.jacobian[:, j])/np.linalg.norm(result.jacobian[:, j]) <= 1e-5
    assert result.diagnostics["groups"]["f"]["stationarity"] <= 1e-8


def test_multistart_keeps_failed_lower_cost_and_distinguishes_best_converged():
    p, _ = example()
    failed = p.fit(max_nfev=1)
    converged = p.fit(fixed_geometry=True)
    # Preserve a concrete failed result even if its observed objective is lower.
    if failed.cost > converged.cost:
        failed.evaluation.residual *= .01
    summary = summarize_multistart([failed, converged])
    assert summary["best_observed_index"] == 0
    assert summary["best_converged_index"] == 1
    assert not summary["best_observed_converged"]
    assert "uncertain" in summary["numerical_status"]
    assert not summary["starts"][0]["success"]


def amplitude_example(backend="qr", n_starts=3):
    base, x = example(backend)
    n = len(base.elements_m)
    group = replace(base.groups[0], template_prior=np.linspace(.9, 1.1, n)*np.exp(.04j*np.arange(n)),
        template_strength=.025, amplitude_prior=np.linspace(.8, 1.2, n),
        amplitude_strength=.04, amplitude_n_starts=n_starts,
        amplitude_stationarity_tol=1e-13, amplitude_epsilon=1e-6)
    return JointProblem(base.elements_m, [group], backend=backend), x


def test_integrated_amplitude_full_profile_db_and_validation_jacobians():
    p, x = amplitude_example()
    result = p.evaluate(x)
    validation = {"f": [replace(p.groups[0].sessions[1],
        samples_xy_m=p.groups[0].sessions[1].samples_xy_m+.00031)]}
    val = p.validation_evaluate(x, validation)
    for h_mm in (.01, .001, .0001):
        for j in range(p.parameter_count):
            h = h_mm/10
            step = np.eye(p.parameter_count)[j]*h
            plus, minus = p.evaluate(x+step), p.evaluate(x-step)
            for numeric, analytic in (
                ((plus.residual-minus.residual)/(2*h), result.jacobian[:, j]),
                ((plus.b["f"]-minus.b["f"])/(2*h), result.db["f"][:, j]),
                ((plus.amplitude_residuals["f"]-minus.amplitude_residuals["f"])/(2*h), result.amplitude_jacobians["f"][:, j]),
                ((plus.data_residuals["s1"]-minus.data_residuals["s1"])/(2*h), result.data_jacobians["s1"][:, j]),
                ((plus.penalty_residuals["f"]-minus.penalty_residuals["f"])/(2*h), result.penalty_jacobians["f"][:, j]),
            ):
                assert np.linalg.norm(numeric-analytic)/np.linalg.norm(analytic) < 1e-5
            numeric_val = (p.validation_evaluate(x+step, validation)["residual"]
                           -p.validation_evaluate(x-step, validation)["residual"])/(2*h)
            assert np.linalg.norm(numeric_val-val["jacobian"][:, j])/np.linalg.norm(val["jacobian"][:, j]) < 1e-5
    diag = result.diagnostics["groups"]["f"]
    assert diag["backend_used"] == "nonlinear_amplitude_exact_hessian"
    assert diag["stationarity"] < 1e-8
    assert diag["amplitude"]["n_starts"] == 3
    assert diag["amplitude"]["hessian_positive_definite"]
    assert diag["amplitude"]["max_converged_relative_b_spread"] < 1e-8
    assert not diag["amplitude"]["branch_global_optimality_established"]
    amp = diag["amplitude"]
    assert not amp["global_profile_smoothness_established"]
    assert "selected local stationary branch" in amp["profile_derivative_scope"]
    chosen = amp["starts"][amp["chosen_start"]]
    assert chosen["selected"]
    assert chosen["objective"] == amp["selected_objective"]
    assert chosen["objective_gap_from_selected"] == 0
    assert chosen["relative_b_distance_from_selected"] == 0
    for record in amp["starts"]:
        assert record["objective_gap_from_selected"] == record["objective"]-amp["selected_objective"]
        assert record["objective_relative_gap_from_selected"] == record["objective_gap_from_selected"]/max(abs(amp["selected_objective"]), 1e-30)
    assert amp["stationary_start_count"] == sum(record["converged"] for record in amp["starts"])


def test_integrated_amplitude_unknown_scale_repeatability_and_normal_initializer():
    qr, x = amplitude_example()
    normal, _ = amplitude_example("normal")
    group = replace(qr.groups[0], amplitude_prior=237*qr.groups[0].amplitude_prior)
    scaled = JointProblem(qr.elements_m, [group])
    a, b, c = qr.evaluate(x), normal.evaluate(x), scaled.evaluate(x)
    assert_allclose(a.b["f"], b.b["f"], atol=1e-10, rtol=1e-9)
    assert_allclose(a.jacobian, b.jacobian, atol=1e-10, rtol=1e-8)
    assert_allclose(a.b["f"], c.b["f"], atol=1e-10, rtol=1e-9)
    # Returning from another outer point must not change the local objective.
    qr.evaluate(x+np.eye(qr.parameter_count)[0]*1e-4)
    repeated = qr.evaluate(x)
    assert_allclose(a.b["f"], repeated.b["f"], rtol=0, atol=0)
    assert_allclose(a.jacobian, repeated.jacobian, rtol=0, atol=0)
    assert normal.qr_checks[0]["scope"].startswith("linear initializer only")


def test_amplitude_inner_failure_is_recorded_and_profile_derivative_refused():
    p, x = amplitude_example(n_starts=1)
    group = replace(p.groups[0], amplitude_maxiter=0, amplitude_newton_steps=0)
    p = JointProblem(p.elements_m, [group])
    with pytest.raises(AmplitudeProfileError, match="failed"):
        p.evaluate(x)
    assert len(p.failure_ledger) == 1
    assert not p.failure_ledger[0]["success"]
    assert p.counts["failed_evaluations"] == 1
    assert p._last is None


def test_24_start_extension_preserves_original_prefix_and_exact_latin_strata():
    original = deterministic_starts_mm()
    extended = deterministic_starts_mm(count=24)
    assert_allclose(extended[:12], original, atol=0, rtol=0)
    assert_allclose(extended, deterministic_starts_mm(24), atol=0, rtol=0)
    unit = (extended[12:]-np.array((50, -10, -10)))/20
    for j in range(3):
        assert sorted(np.floor(12*unit[:, j]).astype(int)) == list(range(12))
    with pytest.raises(ValueError, match="12 or 24"):
        deterministic_starts_mm(13)


def test_fit_exact_parameter_resume_preserves_unwrapped_gain_phase():
    p, x = example()
    x[4] += 4*np.pi
    gains = p.gains_from_parameters(x)
    assert gains["s0"] == 1
    assert_allclose(gains["s1"], np.exp(x[3]+1j*x[4]))
    fit = p.fit(start_parameters=x, fixed_geometry=True, max_nfev=1)
    assert_allclose(fit.x, x, atol=0, rtol=0)
    assert not fit.success
    with pytest.raises(ValueError, match="either"):
        p.fit(start_parameters=x, start_gains={"s1": 1})


def test_accepted_checkpoint_stop_and_exact_resume_matches_uninterrupted_fit():
    import json
    p, _ = example()
    checkpoints = []
    def checkpoint(record):
        # The monitor can persist the record directly, without b/J arrays.
        json.dumps(record)
        checkpoints.append(record)
        if record["event"] == "iteration":
            raise StopIteration
    stopped = p.fit(max_nfev=80, checkpoint_callback=checkpoint)
    assert not stopped.success and stopped.status == -2
    assert [r["event"] for r in checkpoints] == ["initial", "iteration", "final"]
    assert_allclose(checkpoints[-1]["x"], stopped.x, atol=0, rtol=0)
    assert checkpoints[-1]["cost"] == stopped.cost
    assert "b" not in checkpoints[-1] and "jacobian" not in checkpoints[-1]
    assert checkpoints[1]["nfev"] >= checkpoints[0]["nfev"]
    resumed = p.fit(start_parameters=checkpoints[-1]["x"], max_nfev=80)
    baseline, _ = example()
    uninterrupted = baseline.fit(max_nfev=80)
    assert resumed.success and uninterrupted.success
    assert_allclose(resumed.geometry_mm, uninterrupted.geometry_mm, atol=2e-5, rtol=0)
    assert_allclose(resumed.cost, uninterrupted.cost, atol=1e-12, rtol=1e-9)
