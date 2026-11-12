"""Meaningful correctness checks for the nonlinear relative amplitude block."""

import numpy as np
import pytest
from scipy import linalg

from array_reconstruct.joint_amplitude import (
    _AmplitudeObjective,
    _stack,
    implicit_derivative,
    solve_amplitude,
)


def example(seed=391):
    rng = np.random.default_rng(seed)
    B = (rng.normal(size=(35,9))+1j*rng.normal(size=(35,9)))/np.sqrt(35)
    truth = rng.uniform(0.5,1.5,9)*np.exp(1j*rng.uniform(-np.pi,np.pi,9))
    y = B@truth + .02*(rng.normal(size=35)+1j*rng.normal(size=35))
    a = abs(truth)*np.exp(rng.normal(0,.12,9))
    dB = [(rng.normal(size=B.shape)+1j*rng.normal(size=B.shape))*.02
          for _ in range(3)]
    return B,y,a,dB,truth


def test_full_hessian_includes_nonzero_residual_curvature():
    B,y,a,_,truth = example()
    problem = _AmplitudeObjective(B,y,.1,a,.7,1e-6)
    z = _stack(truth*np.exp(.3j))
    rng = np.random.default_rng(12)
    direction = rng.normal(size=len(z))
    h = 1e-5
    numeric_grad = np.array([
        (problem.value_gradient(z+h*np.eye(len(z))[j])[0]
         -problem.value_gradient(z-h*np.eye(len(z))[j])[0])/(2*h)
        for j in range(len(z))
    ])
    gradient = problem.value_gradient(z)[1]
    np.testing.assert_allclose(gradient,numeric_grad,rtol=1e-7,atol=1e-9)
    numeric_Hd = (problem.value_gradient(z+h*direction)[1]
                  -problem.value_gradient(z-h*direction)[1])/(2*h)
    np.testing.assert_allclose(problem.hessian(z)@direction,numeric_Hd,rtol=1e-7,atol=1e-9)
    # The residual Hessian is material here; a GN substitution would fail.
    hprior,s,dx,dv = problem.prior(z)
    D = np.zeros((len(a),len(z)))
    D[np.arange(len(a)),np.arange(len(a))] = dx
    D[np.arange(len(a)),len(a)+np.arange(len(a))] = dv
    D -= D.mean(axis=0)
    GN = problem.quadratic_hessian + .7**2*D.T@D
    assert linalg.norm(problem.hessian(z)-GN) > 1e-2
    assert linalg.norm(hprior) > .1


def test_calibration_scale_and_phase_never_become_complex_prior():
    B,y,a,_,truth = example()
    p = _AmplitudeObjective(B,y,.1,a,.7,1e-7)
    q = _AmplitudeObjective(B,y,.1,237*a,.7,1e-7)
    np.testing.assert_allclose(p.prior(_stack(truth))[0],q.prior(_stack(truth))[0],atol=2e-15)
    np.testing.assert_allclose(p.prior(_stack(truth))[0],p.prior(_stack(truth*np.exp(1.23j)))[0],atol=2e-15)
    # The common scale is fit from the data: a's arbitrary norm has no effect.
    sol = solve_amplitude(B,y,.03,a,.35)
    scaled = solve_amplitude(B,y,.03,237*a,.35)
    assert sol.success and scaled.success
    np.testing.assert_allclose(sol.b,scaled.b,rtol=2e-9,atol=2e-10)
    assert np.std(np.angle(sol.b)) > .5


def test_profile_jacobian_by_full_resolve_at_three_steps():
    B,y,a,dB,_ = example()
    sol = solve_amplitude(B,y,.04,a,.6,stationarity_tol=1e-13)
    assert sol.success and len(sol.starts) == 3
    assert all(s.converged for s in sol.starts)
    assert sol.max_converged_relative_b_spread < 1e-10
    assert not sol.has_failed_lower_objective_start
    assert sol.start_coefficients.shape == (B.shape[1],3)
    db,J = implicit_derivative(sol,iter(dB))
    assert db.shape == (B.shape[1],3)
    for h in [1e-2,1e-3,1e-4]:
        for j,d in enumerate(dB):
            # Every perturbation re-solves the complete nonlinear subproblem;
            # the unperturbed solution is merely its non-oracle warm start.
            plus = solve_amplitude(B+h*d,y,.04,a,.6,initial=sol.b,n_starts=1,
                                   stationarity_tol=1e-13)
            minus = solve_amplitude(B-h*d,y,.04,a,.6,initial=sol.b,n_starts=1,
                                    stationarity_tol=1e-13)
            assert plus.success and minus.success
            numeric = (plus.residual-minus.residual)/(2*h)
            relative_error = linalg.norm(numeric-J[:,j])/linalg.norm(J[:,j])
            assert relative_error < 1e-5
            np.testing.assert_allclose((plus.b-minus.b)/(2*h),db[:,j],rtol=1e-5,atol=1e-8)
    # Envelope identity includes all penalty rows and has the declared sign.
    for j,d in enumerate(dB):
        expected = np.vdot(sol.data_residual,d@sol.b).real
        assert np.dot(sol.residual,J[:,j]) == pytest.approx(expected,abs=2e-11)


def test_zero_strength_matches_augmented_svd_and_is_repeatable():
    B,y,a,dB,_ = example()
    ridge = .15
    exact = linalg.lstsq(np.vstack((B,ridge*np.eye(B.shape[1]))),
                         np.r_[y,np.zeros(B.shape[1])])[0]
    sol = solve_amplitude(B,y,ridge,a,0)
    repeated = solve_amplitude(B,y,ridge,a,0)
    assert sol.success and repeated.success
    np.testing.assert_allclose(sol.b,exact,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(sol.b,repeated.b,rtol=0,atol=1e-14)
    assert sol.stationarity <= 1e-10
    assert sol.hessian_rcond > 1e-3
    assert np.all(sol.amplitude_residual == 0)
    assert implicit_derivative(sol,dB[0])[1].shape[1] == 1


def test_smoothing_at_zero_is_finite_and_can_have_negative_curvature():
    B = np.eye(3,dtype=complex)
    a = np.array([.2,1,5])
    problem = _AmplitudeObjective(B,np.ones(3),.1,a,1,1e-3)
    z = np.zeros(6)
    assert np.all(np.isfinite(problem.residual(z)))
    assert np.all(np.isfinite(problem.value_gradient(z)[1]))
    assert np.all(np.isfinite(problem.hessian(z)))
    assert np.linalg.eigvalsh(problem.hessian(z)).min() < 0


def test_budget_failure_is_preserved_and_derivative_refused():
    B,y,a,dB,_ = example()
    sol = solve_amplitude(B,y,.1,a,2,maxiter=0,newton_steps=0)
    assert not sol.success
    assert len(sol.starts) == 3
    assert any(not s.converged for s in sol.starts)
    with pytest.raises(RuntimeError,match="failed"):
        implicit_derivative(sol,dB)


@pytest.mark.parametrize("bad",[0,-1,np.nan,np.inf])
def test_nonpositive_or_invalid_calibration_rejected(bad):
    B,y,a,_,_ = example()
    a[1] = bad
    with pytest.raises(ValueError,match="positive"):
        solve_amplitude(B,y,.1,a,.3)


def test_invalid_derivative_shape_rejected():
    B,y,a,_,_ = example()
    sol = solve_amplitude(B,y,.1,a,.3,n_starts=1)
    with pytest.raises(ValueError,match="shape"):
        implicit_derivative(sol,np.zeros((2,2),complex))
