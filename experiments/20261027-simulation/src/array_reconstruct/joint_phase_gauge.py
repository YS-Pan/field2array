"""Optional, objective-invariant phase coordinates for complex penalty rows.

This changes the representation supplied to Gauss-Newton, never the physical
penalty or fitted response. Data rows and real log-amplitude rows are unchanged.
The common response sum must remain well conditioned; a failed guard rejects
this representation rather than silently switching residual definitions.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from scipy import optimize

from .joint_geometry import AmplitudeProfileError, FitResult, _real_array, parameters_to_geometry
from .joint_newton import outer_bounds, stationarity_metrics


class PhaseGaugeError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics=diagnostics
        super().__init__(f"penalty phase gauge response-sum guard failed in {diagnostics['group']}")


def phase_gauge_evaluate(problem,x,min_coherence=1e-3):
    """Rotate linear penalty rows with full chain rule, keeping norm/gradient.

    S=sum(b), u=conj(S)/abs(S), phi_j=Im(sum(b_j)/S).
    r_pen_new=u*r_pen; J_pen_new=u*(J_pen-i*r_pen*phi_j).
    Coherence abs(S)/(sqrt(n)*norm(b)) is a numerical conditioning check, not
    evidence about hardware phases or geometry. No global smoothness is claimed
    across a zero response sum, an inner branch switch, or a failed guard.
    """
    if not np.isfinite(min_coherence) or not 0<min_coherence<=1:
        raise ValueError("min_coherence must lie in (0,1]")
    evaluation=problem.evaluate(x)
    residuals,jacobians,penalties,penalty_jacobians=[],[],{},{}
    group_diagnostics={}
    for group in problem.groups:
        for session in group.sessions:
            residuals.append(evaluation.weighted_data_residuals[session.name])
            jacobians.append(evaluation.weighted_data_jacobians[session.name])
        r=evaluation.penalty_residuals[group.name]
        J=evaluation.penalty_jacobians[group.name]
        b=evaluation.b[group.name]
        response_sum=np.sum(b)
        norm=float(np.linalg.norm(b))
        coherence=float(abs(response_sum)/(np.sqrt(len(b))*norm)) if norm>0 else 0.
        enabled=group.lambda_reg>0 or group.template_strength>0
        finite=bool(np.all(np.isfinite(b)) and np.all(np.isfinite(evaluation.db[group.name]))
                    and np.isfinite(response_sum) and np.isfinite(norm) and np.isfinite(coherence))
        if enabled and (not finite or coherence<min_coherence):
            raise PhaseGaugeError({"group":group.name,"x":evaluation.x.tolist(),
                "geometry_mm":evaluation.geometry_mm.tolist(),"coherence":coherence if np.isfinite(coherence) else None,
                "min_coherence":min_coherence,"response_sum":[float(v) if np.isfinite(v) else None for v in (response_sum.real,response_sum.imag)],
                "representation_used":False,"reason":"nonfinite response sum/coefficient derivative" if not finite
                    else "sum(b) is too ill-conditioned for a smooth penalty phase coordinate"})
        if enabled:
            rotation=response_sum.conjugate()/abs(response_sum)
            phase_derivative=np.imag(np.sum(evaluation.db[group.name],axis=0)/response_sum)
            if not np.isfinite(rotation) or not np.all(np.isfinite(phase_derivative)):
                raise PhaseGaugeError({"group":group.name,"x":evaluation.x.tolist(),
                    "geometry_mm":evaluation.geometry_mm.tolist(),"coherence":coherence,
                    "min_coherence":min_coherence,"representation_used":False,
                    "reason":"nonfinite penalty phase rotation or derivative"})
            rr=rotation*r
            JJ=rotation*(J-1j*r[:,None]*phase_derivative[None,:])
        else:
            rotation=1.+0j
            phase_derivative=np.zeros(problem.parameter_count)
            rr,JJ=r.copy(),J.copy()
        residuals.append(rr);jacobians.append(JJ)
        penalties[group.name],penalty_jacobians[group.name]=rr,JJ
        group_diagnostics[group.name]={"enabled":enabled,"coherence":coherence,
            "min_coherence":min_coherence,"rotation":[float(rotation.real),float(rotation.imag)],
            "removed_phase_derivative":phase_derivative.tolist(),
            "phase_radians":float(np.angle(response_sum)),
            "penalty_norm_squared_difference":float(np.vdot(rr,rr).real-np.vdot(r,r).real)}
    r,J=np.concatenate(residuals),np.vstack(jacobians)
    amp_r=np.concatenate(list(evaluation.amplitude_residuals.values())) if evaluation.amplitude_residuals else np.empty(0)
    amp_J=np.vstack(list(evaluation.amplitude_jacobians.values())) if evaluation.amplitude_jacobians else np.empty((0,problem.parameter_count))
    real_r=np.r_[r.real,r.imag,amp_r]
    real_J=np.vstack((J.real,J.imag,amp_J))
    original_gradient=evaluation.jacobian.T@evaluation.residual
    new_gradient=real_J.T@real_r
    diagnostics={"groups":group_diagnostics,"min_coherence":min_coherence,
        "objective_difference":float(.5*np.dot(real_r,real_r)-evaluation.cost),
        "gradient_absolute_difference":float(np.linalg.norm(new_gradient-original_gradient)),
        "gradient_relative_difference":float(np.linalg.norm(new_gradient-original_gradient)/max(np.linalg.norm(original_gradient),1e-30)),
        "physical_objective_changed":False,"data_rows_changed":False,"real_logamplitude_rows_changed":False,
        "chain_rule_included":True,"global_profile_smoothness_established":False}
    return {"evaluation":evaluation,"residual":real_r,"jacobian":real_J,
        "penalty_residuals":penalties,"penalty_jacobians":penalty_jacobians,"diagnostics":diagnostics}


def fit_phase_gauge(problem,start_parameters,max_nfev=1500,*,fixed_geometry=False,
                    ftol=1e-8,gtol=1e-7,xtol=1e-7,gain_log_bounds=(-3.,3.),
                    min_coherence=1e-3,checkpoint_callback=None):
    """The same bounded TRF objective, expressed in gauged penalty coordinates."""
    started=time.perf_counter();initial_counts=dict(problem.counts)
    x0=_real_array(start_parameters,"start_parameters",(problem.parameter_count,))
    parameters_to_geometry(x0[:3]);problem.gains_from_parameters(x0)
    lower,upper=outer_bounds(problem,gain_log_bounds)
    if np.any(x0<lower) or np.any(x0>upper):
        raise ValueError("starting parameters outside existing hard bounds")
    mask=np.full(3,fixed_geometry,dtype=bool) if np.ndim(fixed_geometry)==0 else np.asarray(fixed_geometry,dtype=bool)
    if mask.shape!=(3,):
        raise ValueError("fixed_geometry must be bool or a three-axis mask")
    free=np.r_[~mask,np.ones(problem.parameter_count-3,bool)]
    cached_x=cached=None
    gauge_calls=cache_hits=0
    def expand(z):
        x=x0.copy();x[free]=z;return x
    def evaluate(x):
        nonlocal cached_x,cached,gauge_calls,cache_hits
        if cached_x is not None and np.array_equal(x,cached_x):
            cache_hits+=1;return cached
        cached=phase_gauge_evaluate(problem,x,min_coherence)
        cached_x=x.copy();gauge_calls+=1;return cached
    iteration=0
    def emit(event,x,cost,nfev,**extra):
        if checkpoint_callback is not None:
            checkpoint_callback({"event":event,"method":"TRF-penalty-phase-gauge",
                "x":x.tolist(),"geometry_mm":parameters_to_geometry(x[:3]).tolist(),
                "gains":{k:[float(v.real),float(v.imag)] for k,v in problem.gains_from_parameters(x).items()},
                "cost":None if cost is None else float(cost),"nfev":int(nfev),"iteration":iteration,"max_nfev":int(max_nfev),
                "counts":{**{k:problem.counts[k]-initial_counts[k] for k in problem.counts},
                    "phase_gauge_calls":gauge_calls,"phase_gauge_cache_hits":cache_hits},
                "elapsed_seconds":time.perf_counter()-started,**extra})
    try:
        initial=evaluate(x0)
    except (PhaseGaugeError,AmplitudeProfileError) as error:
        emit("failed",x0,None,0,success=False,status=-3,message=str(error),failed_attempt=error.diagnostics)
        raise
    emit("initial",x0,.5*np.dot(initial["residual"],initial["residual"]),1)
    last_accepted=x0.copy();last_cost=.5*np.dot(initial["residual"],initial["residual"]);last_nfev=1
    def callback(intermediate_result):
        nonlocal iteration,last_accepted,last_cost,last_nfev
        iteration+=1
        last_accepted=expand(intermediate_result.x)
        last_cost=float(intermediate_result.cost);last_nfev=int(intermediate_result.nfev)
        emit("iteration",last_accepted,last_cost,last_nfev)
    if np.any(free):
        try:
            result=optimize.least_squares(lambda z:evaluate(expand(z))["residual"],x0[free],
                jac=lambda z:evaluate(expand(z))["jacobian"][:,free],bounds=(lower[free],upper[free]),
                method="trf",max_nfev=max_nfev,ftol=ftol,gtol=gtol,xtol=xtol,
                callback=callback)
        except (PhaseGaugeError,AmplitudeProfileError) as error:
            emit("failed",last_accepted,last_cost,last_nfev,success=False,status=-3,message=str(error),failed_attempt=error.diagnostics)
            raise
        x=expand(result.x);success=bool(result.success);status=int(result.status);message=str(result.message)
        nfev,njev,optimality=int(result.nfev),int(result.njev),float(result.optimality)
        active=np.zeros(problem.parameter_count,int);active[free]=result.active_mask
    else:
        x=x0;success=True;status=1;message="all outer variables fixed; inner solve only"
        nfev,njev,optimality,active=1,1,0.,np.zeros(problem.parameter_count,int)
    final=evaluate(x)
    raw=final["evaluation"]
    common=stationarity_metrics(x[free],(final["jacobian"].T@final["residual"])[free],lower[free],upper[free]) if np.any(free) else {"projected_gradient_inf":0.,"trf_scaled_gradient_inf":0.}
    diagnostics={**raw.diagnostics,"phase_gauge":{**final["diagnostics"],"common_stationarity":common,
        "residual_representation":"common response phase removed only from complex linear penalty rows"}}
    evaluation=replace(raw,residual=final["residual"],jacobian=final["jacobian"],
        penalty_residuals=final["penalty_residuals"],penalty_jacobians=final["penalty_jacobians"],diagnostics=diagnostics)
    counts={**{k:problem.counts[k]-initial_counts[k] for k in problem.counts},
            "phase_gauge_calls":gauge_calls,"phase_gauge_cache_hits":cache_hits}
    emit("final",x,evaluation.cost,nfev,success=success,status=status,message=message,njev=njev,optimality=optimality)
    return FitResult(evaluation.geometry_mm.copy(),evaluation,success,status,message,nfev,njev,
        optimality,active,counts,time.perf_counter()-started,x.copy())
