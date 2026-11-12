"""Independent objective/branch/Jacobian checks for the accelerated inner solve."""
import numpy as np
import pytest
from scipy import linalg

from array_reconstruct.joint_amplitude import solve_amplitude,_AmplitudeObjective,_stack
from array_reconstruct.joint_amplitude_fast import solve_amplitude_fast,implicit_derivative


def fixture(seed=73021,ill_conditioned=False):
    rng=np.random.default_rng(seed)
    B=(rng.normal(size=(32,10))+1j*rng.normal(size=(32,10)))/np.sqrt(32)
    if ill_conditioned:
        u,_,v=np.linalg.svd(B,full_matrices=False)
        B=(u*np.geomspace(1,1e-3,10))@v
    truth=rng.uniform(.6,1.4,10)*np.exp(1j*rng.uniform(-np.pi,np.pi,10))
    y=B@truth+.005*(rng.normal(size=32)+1j*rng.normal(size=32))
    a=abs(truth)*np.exp(rng.normal(0,.05,10))
    dB=(rng.normal(size=B.shape)+1j*rng.normal(size=B.shape))*.01
    return B,y,a,dB


def test_fast_same_objective_and_stationary_branch_as_original():
    B,y,a,_=fixture()
    old=solve_amplitude(B,y,.03,a,.15,maxiter=5000,newton_steps=250,stationarity_tol=1e-12)
    fast=solve_amplitude_fast(B,y,.03,a,.15)
    assert old.success and fast.success
    assert len(fast.starts)==3
    assert all(start.converged for start in fast.starts)
    np.testing.assert_allclose(fast.b,old.b,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(fast.residual,old.residual,rtol=1e-8,atol=1e-9)
    assert fast.objective==pytest.approx(.5*np.dot(fast.residual,fast.residual),abs=1e-13)
    assert isinstance(fast.fast_diagnostics,dict)


def test_ill_conditioned_original_stationary_equations_not_relaxed():
    B,y,a,_=fixture(ill_conditioned=True)
    old=solve_amplitude(B,y,1e-4,a,.03,maxiter=5000,newton_steps=250,
                        stationarity_tol=1e-12)
    assert old.success
    # A converged original local branch must remain that same branch, even
    # though whitening changes the numerical coordinates and intermediate path.
    fast=solve_amplitude_fast(B,y,1e-4,a,.03,initial=old.b,n_starts=1)
    assert fast.success
    np.testing.assert_allclose(fast.b,old.b,rtol=1e-7,atol=1e-8)
    problem=_AmplitudeObjective(B,y,1e-4,a,.03,fast.epsilon)
    gradient=problem.value_gradient(_stack(fast.b))[1]
    assert linalg.norm(gradient,np.inf)/problem.stationarity_scale<=1e-12
    np.testing.assert_allclose(fast.hessian,problem.hessian(_stack(fast.b)),rtol=1e-12,atol=1e-13)


def test_fast_implicit_profile_derivative_by_complete_resolves():
    B,y,a,dB=fixture()
    fast=solve_amplitude_fast(B,y,.04,a,.3,stationarity_tol=1e-13)
    assert fast.success
    db,J=implicit_derivative(fast,dB)
    for h in (1e-2,1e-3,1e-4):
        plus=solve_amplitude_fast(B+h*dB,y,.04,a,.3,initial=fast.b,n_starts=1,
                                  stationarity_tol=1e-13)
        minus=solve_amplitude_fast(B-h*dB,y,.04,a,.3,initial=fast.b,n_starts=1,
                                   stationarity_tol=1e-13)
        assert plus.success and minus.success
        numeric=(plus.residual-minus.residual)/(2*h)
        assert linalg.norm(numeric-J[:,0])/linalg.norm(J[:,0])<1e-5
        assert linalg.norm((plus.b-minus.b)/(2*h)-db[:,0])/linalg.norm(db[:,0])<1e-5
    assert np.dot(fast.residual,J[:,0])==pytest.approx(
        np.vdot(fast.data_residual,dB@fast.b).real,abs=1e-11)


def test_unknown_calibration_scale_still_eliminated_exactly():
    B,y,a,_=fixture()
    first=solve_amplitude_fast(B,y,.04,a,.3)
    rescaled=solve_amplitude_fast(B,y,.04,91*a,.3)
    assert first.success and rescaled.success
    np.testing.assert_allclose(first.b,rescaled.b,rtol=1e-8,atol=1e-9)
    assert np.std(np.angle(first.b))>.5


def test_nonzero_augmented_rhs_and_template_penalty_keep_full_objective():
    B,y,a,_=fixture()
    n=len(a)
    template=a*np.exp(1j*np.linspace(-.1,.2,n))
    P=np.eye(n)-np.outer(template,template.conj())/np.vdot(template,template)
    b0=.2*np.exp(1j*np.linspace(-1,1,n))
    augmented=np.vstack((B,.07*np.eye(n),.04*P))
    rhs=np.concatenate((y,.07*b0,np.zeros(n,complex)))
    old=solve_amplitude(augmented,rhs,0.,a,.15,stationarity_tol=1e-12,
                        maxiter=5000,newton_steps=250)
    fast=solve_amplitude_fast(augmented,rhs,0.,a,.15)
    assert old.success and fast.success
    np.testing.assert_allclose(fast.b,old.b,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(fast.residual[:2*len(rhs)],np.r_[
        (augmented@fast.b-rhs).real,(augmented@fast.b-rhs).imag],atol=1e-13)
    assert fast.objective==pytest.approx(old.objective,rel=1e-10,abs=1e-13)


def test_singular_data_hessian_does_not_trigger_hidden_ridge():
    rng=np.random.default_rng(919)
    B=(rng.normal(size=(3,4))+1j*rng.normal(size=(3,4)))
    truth=rng.uniform(.8,1.2,4)*np.exp(1j*rng.uniform(-np.pi,np.pi,4))
    y=B@truth
    # A numerical positive control at an exact stationary point.  The data
    # Hessian is singular but data+amplitude tangent may be locally full rank.
    result=solve_amplitude_fast(B,y,0.,abs(truth),.2,initial=truth,n_starts=1)
    assert result.ridge==0.
    np.testing.assert_allclose(result.residual,0.,atol=1e-12)
    assert result.success
    np.testing.assert_allclose(result.b,truth,rtol=0,atol=1e-12)


def test_acceleration_never_converts_exhausted_budget_to_success():
    B,y,a,dB=fixture()
    result=solve_amplitude_fast(B,y,.03,a,2.,maxiter=0,newton_steps=0)
    assert not result.success
    assert len(result.starts)==3
    with pytest.raises(RuntimeError,match="failed"):
        implicit_derivative(result,dB)
