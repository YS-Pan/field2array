"""Tensor amplitude solver checks; CPU by default, CUDA only by explicit env.

Set ARRAY_RECONSTRUCT_AMPLITUDE_TEST_DEVICE=cuda:0 only after the root GPU
scheduler authorizes this process.  Importing these tests does not query CUDA.
"""
import os
import numpy as np
import pytest
import torch

from array_reconstruct.joint_amplitude_fast import solve_amplitude_fast
from array_reconstruct.joint_amplitude_gpu import solve_amplitude_gpu,implicit_derivative_gpu

DEVICE=os.environ.get("ARRAY_RECONSTRUCT_AMPLITUDE_TEST_DEVICE","cpu")


def fixture():
    rng=np.random.default_rng(84104)
    B=(rng.normal(size=(28,7))+1j*rng.normal(size=(28,7)))/np.sqrt(28)
    b=rng.uniform(.6,1.4,7)*np.exp(1j*rng.uniform(-np.pi,np.pi,7))
    y=B@b+.007*(rng.normal(size=28)+1j*rng.normal(size=28))
    a=abs(b)*np.exp(rng.normal(0,.06,7))
    dB=.015*(rng.normal(size=B.shape)+1j*rng.normal(size=B.shape))
    return B,y,a,dB


def array(tensor):
    return tensor.detach().cpu().numpy()


def test_tensor_solver_same_original_objective_and_strict_gate():
    B,y,a,_=fixture()
    expected=solve_amplitude_fast(B,y,.05,a,.2)
    actual=solve_amplitude_gpu(B,y,.05,a,.2,device=DEVICE)
    assert expected.success and actual.success
    assert len(actual.starts)==3
    assert all(s.converged for s in actual.starts)
    assert actual.b.dtype==torch.complex128
    assert actual.residual.dtype==torch.float64
    assert actual.hessian.dtype==torch.float64
    assert actual.b.device.type==torch.device(DEVICE).type
    np.testing.assert_allclose(array(actual.b),expected.b,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(array(actual.residual),expected.residual,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(array(actual.hessian),expected.hessian,rtol=1e-8,atol=1e-10)
    assert actual.objective==pytest.approx(expected.objective,rel=1e-10,abs=1e-13)
    assert actual.stationarity<=1e-12 and actual.hessian_rcond>=1e-12


def test_tensor_complete_profile_jacobian_by_full_resolves():
    B,y,a,dB=fixture()
    solution=solve_amplitude_gpu(B,y,.05,a,.3,device=DEVICE,stationarity_tol=1e-13)
    assert solution.success
    db,J=implicit_derivative_gpu(solution,[torch.as_tensor(dB,dtype=torch.complex128,device=DEVICE)])
    assert db.dtype==torch.complex128 and J.dtype==torch.float64
    expected=array(J)[:,0]
    for h in (1e-2,1e-3,1e-4):
        plus=solve_amplitude_gpu(B+h*dB,y,.05,a,.3,device=DEVICE,
            initial=solution.b,n_starts=1,stationarity_tol=1e-13)
        minus=solve_amplitude_gpu(B-h*dB,y,.05,a,.3,device=DEVICE,
            initial=solution.b,n_starts=1,stationarity_tol=1e-13)
        assert plus.success and minus.success
        numeric=array(plus.residual-minus.residual)/(2*h)
        assert np.linalg.norm(numeric-expected)/np.linalg.norm(expected)<1e-5
    assert float(torch.dot(solution.residual,J[:,0]).item())==pytest.approx(
        np.vdot(array(solution.data_residual),dB@array(solution.b)).real,abs=1e-11)


def test_tensor_augmented_complex_rhs_and_unknown_amplitude_scale():
    B,y,a,_=fixture()
    n=len(a)
    template=a*np.exp(1j*np.linspace(-.1,.15,n))
    P=np.eye(n)-np.outer(template,template.conj())/np.vdot(template,template)
    b0=.2*np.exp(1j*np.linspace(-1.,1.,n))
    augmented=np.vstack((B,.08*np.eye(n),.03*P))
    rhs=np.r_[y,.08*b0,np.zeros(n,complex)]
    expected=solve_amplitude_fast(augmented,rhs,0.,a,.2)
    first=solve_amplitude_gpu(augmented,rhs,0.,a,.2,device=DEVICE)
    rescaled=solve_amplitude_gpu(augmented,rhs,0.,17*a,.2,device=DEVICE)
    assert expected.success and first.success and rescaled.success
    np.testing.assert_allclose(array(first.b),expected.b,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(array(first.b),array(rescaled.b),rtol=1e-8,atol=1e-9)
    assert np.std(np.angle(array(first.b)))>.5


def test_tensor_non_double_input_is_rejected():
    B,y,a,_=fixture()
    bad=torch.as_tensor(B,dtype=torch.complex64,device=DEVICE)
    with pytest.raises((TypeError,ValueError),match="128|dtype|precision"):
        solve_amplitude_gpu(bad,y,.05,a,.2,device=DEVICE)


def test_tensor_zero_smoothing_and_indefinite_stationary_point_remain_failure():
    B=np.eye(3,dtype=complex)
    a=np.array([.2,1.,5.])
    solution=solve_amplitude_gpu(B,np.zeros(3,complex),.1,a,1.,initial=np.zeros(3,complex),
        n_starts=1,epsilon=1e-3,device=DEVICE)
    assert bool(torch.isfinite(solution.residual).all().item())
    assert not solution.success
    assert not solution.hessian_positive_definite
    with pytest.raises(RuntimeError,match="failed"):
        implicit_derivative_gpu(solution,[B])


def test_tensor_no_false_success_on_exhausted_budget():
    B,y,a,dB=fixture()
    solution=solve_amplitude_gpu(B,y,.03,a,2.,device=DEVICE,maxiter=0,newton_steps=0)
    assert not solution.success
    assert len(solution.starts)==3
    with pytest.raises(RuntimeError,match="failed"):
        implicit_derivative_gpu(solution,[dB])
