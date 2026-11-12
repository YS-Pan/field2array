"""CPU endpoint pose information at fixed b/g, without a profile evaluation.

The caller owns authorization, the frozen train-only problem and public endpoint
bindings. No loader, optimizer, GPU, file access or confidence/precision gate is
provided. Penalty-assisted sensitivity is reported separately from pure data.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import linalg

from .joint_pose_kernel import pose_to_parameters, scalar_pose_kernel

PHYSICAL_SCALES = np.array([.1, .1, .1, 1., 1., 1./3.])
ORDER = ("d", "dx", "dy", "alpha", "beta", "gamma")


def _need(value, message):
    if not value:
        raise ValueError(message)


def _work():
    return dict(device="cpu", profile_evaluations=0, optimizer_calls=0,
        second_kernel_calls=0, kernel_first_bundle_calls=0,
        kernel_first_derivative_matrices=0, nuisance_qr_calls=0,
        nuisance_R_svd_calls=0, pose_svd_calls=0, reference_eigvalsh_calls=0,
        kernel_seconds=0., projection_seconds=0., summary_seconds=0.)


def _project(N, Z, rtol, work):
    """Complex nuisance span: one column also includes its i-quadrature."""
    tick = time.perf_counter()
    N, Z = np.asarray(N, dtype=np.complex128), np.asarray(Z, dtype=np.complex128)
    _need(N.ndim == Z.ndim == 2 and Z.shape[1] == 6 and len(N) == len(Z)
          and len(Z) > 0 and np.isfinite(N).all() and np.isfinite(Z).all(), "finite matching N(m,q),Z(m,6) required")
    norms = linalg.norm(N, axis=0)
    _need(np.isfinite(norms).all(), "nuisance column norms overflowed")
    nonzero = norms > 0
    rank, singular, tolerance = 0, np.empty(0), 0.
    if np.any(nonzero):
        # Normalization changes parameter units, never the exact column span.
        scaled = np.array(N[:, nonzero] / norms[nonzero], order="F")
        Q, R = linalg.qr(scaled, mode="economic", overwrite_a=True, check_finite=False)
        work["nuisance_qr_calls"] += 1
        U, singular, _ = linalg.svd(R, full_matrices=False, check_finite=False)
        work["nuisance_R_svd_calls"] += 1
        tolerance = rtol * float(singular[0])
        rank = int(np.count_nonzero(singular > tolerance))
        coordinates = U[:, :rank].conj().T @ (Q.conj().T @ Z)
        projected = Z - Q @ (U[:, :rank] @ coordinates)
    else:
        projected = Z.copy()
    _need(np.isfinite(projected).all(), "nonfinite projected pose tangent")
    work["projection_seconds"] += time.perf_counter() - tick
    return projected, dict(rows_complex=len(Z), nuisance_columns_complex=N.shape[1],
        nuisance_columns_real=2*N.shape[1], nuisance_rank_complex=rank,
        nuisance_rank_real=2*rank, nuisance_singular_values=singular.tolist(),
        nuisance_rank_tolerance=tolerance, nuisance_rtol=rtol,
        zero_nuisance_columns=int(np.count_nonzero(~nonzero)),
        nuisance_column_normalization="unit_l2_before_QR_R_SVD;zero_columns_omitted")


def _spectrum(matrix, unprojected_gram, rtol, work):
    # Never request an m-by-m U for a production tall six-column matrix.
    _, singular, vh = linalg.svd(matrix, full_matrices=matrix.shape[0] < 6, check_finite=False)
    work["pose_svd_calls"] += 1
    singular = np.pad(singular, (0, max(0, 6-len(singular))))
    reference = linalg.eigvalsh(.5*(unprojected_gram+unprojected_gram.T), check_finite=False)
    work["reference_eigvalsh_calls"] += 1
    raw_norm = float(np.sqrt(max(0., float(reference[-1]))))
    tolerance = rtol * max(raw_norm, float(singular[0]))
    rank = int(np.count_nonzero(singular > tolerance))
    _need(np.isfinite(singular).all() and np.isfinite(raw_norm) and np.isfinite(tolerance), "nonfinite pose information spectrum")
    raw_columns = np.sqrt(np.maximum(0., np.diag(unprojected_gram)))
    columns = linalg.norm(matrix, axis=0)
    return dict(column_norms=columns.tolist(), unprojected_column_norms=raw_columns.tolist(),
        column_norm_retention=[float(p/r) if r > 0 else None for p,r in zip(columns,raw_columns)], singular_values=singular.tolist(),
        rank=rank, rank_tolerance=tolerance, rank_rtol=rtol,
        unprojected_tangent_spectral_norm=raw_norm,
        singular_values_relative_to_unprojected=(singular/raw_norm).tolist() if raw_norm > 0 else None,
        right_singular_vectors_rows=vh.tolist(),
        singular_vector_basis_uniqueness_claim=False,
        condition_number=float(singular[0]/singular[-1]) if rank == 6 else None,
        below_tolerance_directions=[vh[j].tolist() for j in range(6) if singular[j] <= tolerance],
        status="rank_deficient_unknown" if rank < 6 else "finite_local_rank6_not_precision_or_uniqueness_pass")


def _summary(projected_blocks, raw_gram, rtol, work, kind):
    tick = time.perf_counter()
    combined = np.vstack(projected_blocks)
    real = np.vstack((combined.real, combined.imag))
    normalized = _spectrum(real, raw_gram, rtol, work)
    scaled = real * PHYSICAL_SCALES[None, :]
    physical = _spectrum(scaled, raw_gram*PHYSICAL_SCALES[:, None]*PHYSICAL_SCALES[None, :], rtol, work)
    work["summary_seconds"] += time.perf_counter()-tick
    return dict(residual_kind=kind, order=list(ORDER), row_count_real=len(real),
        normalized_u6=normalized, per_mm_per_degree=physical,
        normalized_u6_definition=["(d_mm-60)/10", "dx_mm/10", "dy_mm/10", "alpha_deg", "beta_deg", "gamma_deg/3"],
        physical_column_multiplier_from_u6=PHYSICAL_SCALES.tolist(),
        physical_units=["per_mm"]*3+["per_degree"]*3,
        rank_reference="unprojected fixed-b/g tangent scale;not projection-cancellation noise",
        precision_or_identifiability_gate=None, confidence_interval=False,
        population_probability_or_global_uniqueness_claim=False)


def _penalty(group, n):
    lam, mu = float(group.lambda_reg), float(group.template_strength)
    _need(np.isfinite(lam) and lam >= 0 and np.isfinite(mu) and mu >= 0, "fixed finite nonnegative lambda/mu required")
    _need(float(getattr(group, "amplitude_strength", 0.)) == 0., "nonlinear amplitude prior is outside this diagnostic")
    rows = [lam*np.eye(n, dtype=np.complex128)]
    if mu > 0:
        a = np.asarray(group.template_prior, dtype=np.complex128)
        _need(a.shape == (n,) and np.isfinite(a).all() and linalg.norm(a) > 0, "valid nonzero fixed complex template required")
        rows.append(mu*(np.eye(n)-np.outer(a, a.conj())/np.vdot(a, a).real))
    # b0 is constant; it supplies neither geometry derivatives nor nuisance rows.
    return np.vstack(rows)


def projected_pose_information(problem, geometry_mm, angles_deg, b_by_group, gains,
                               *, nuisance_rtol=1e-10, include_augmented=True):
    """Analyze six real pose tangents at an explicitly supplied public endpoint.

    Each frequency/chain's complete free complex b and nonreference complex-gain
    tangents are projected out. This never reads profile Jacobians/Hessians or
    calls problem.evaluate/predict; analysis CPU work does not consume profiles.
    """
    started = time.perf_counter()
    _need(type(include_augmented) is bool and not isinstance(nuisance_rtol, (bool, np.bool_))
          and np.isfinite(nuisance_rtol) and 0 < nuisance_rtol < 1, "explicit boolean augmentation and0<rtol<1 required")
    rtol = float(nuisance_rtol)
    pose_u6 = pose_to_parameters(geometry_mm, angles_deg)
    groups = tuple(problem.groups)
    _need(groups and len({g.name for g in groups}) == len(groups)
          and set(b_by_group) == {g.name for g in groups}, "exact nonempty group/b inventory required")
    names = [s.name for g in groups for s in g.sessions]
    _need(len(names) == len(set(names)) and set(gains) == set(names), "exact unique session/gain inventory required")
    elements = np.asarray(problem.elements_m, dtype=np.float64)
    n = len(elements)
    _need(n > 0 and elements.ndim == 2 and elements.shape[1] in (2, 3) and np.isfinite(elements).all(), "finite registered element coordinates required")
    work, reports, pure, augmented = _work(), [], [], []
    raw_gram = np.zeros((6, 6))
    for group in groups:
        sessions = tuple(group.sessions)
        _need(sessions and complex(gains[sessions[0].name]) == 1.+0j, "first session must retain fixed reference g=1")
        b = np.asarray(b_by_group[group.name], dtype=np.complex128)
        _need(b.shape == (n,) and np.isfinite(b).all(), "finite full-column b required")
        m = sum(len(s.samples_xy_m) for s in sessions)
        N = np.zeros((m, n+len(sessions)-1), dtype=np.complex128)
        Z = np.empty((m, 6), dtype=np.complex128)
        cursor = 0
        for index, session in enumerate(sessions):
            gain = complex(gains[session.name])
            u, w = np.asarray(session.steering, dtype=np.complex128), np.asarray(session.weight, dtype=np.float64)
            count = len(session.samples_xy_m)
            if w.ndim == 0: w = np.full(count, float(w))
            _need(np.isfinite(gain) and gain != 0 and u.shape == (n,) and np.isfinite(u).all()
                  and w.shape == (count,) and np.isfinite(w).all() and np.all(w >= 0) and np.any(w > 0), "finite fixed gain/steering/training weights required")
            tick = time.perf_counter()
            K, dK = scalar_pose_kernel(elements, session.samples_xy_m, session.frequency_hz,
                geometry_mm, angles_deg, q_array=problem.q_array, q_probe=problem.q_probe,
                sign=problem.sign, derivatives=True)
            work["kernel_first_bundle_calls"] += 1
            work["kernel_first_derivative_matrices"] += 6
            span = slice(cursor, cursor+count)
            N[span, :n] = K*u[None, :]*(w*gain)[:, None]
            if index: N[span, n+index-1] = N[span, :n]@b
            Z[span] = np.einsum("jmn,n->mj", dK, u*b)*(w*gain)[:, None]
            work["kernel_seconds"] += time.perf_counter()-tick
            cursor += count
            del K, dK
        raw_gram += (Z.conj().T@Z).real
        _need(np.isfinite(raw_gram).all(), "unprojected tangent Gram overflowed")
        projected, data_report = _project(N, Z, rtol, work)
        pure.append(projected)
        row = dict(group=group.name, reference_session=sessions[0].name, data=data_report)
        if include_augmented:
            penalty = _penalty(group, n)
            N_aug = np.zeros((m+len(penalty), N.shape[1]), dtype=np.complex128)
            N_aug[:m], N_aug[m:, :n] = N, penalty
            Z_aug = np.zeros((m+len(penalty), 6), dtype=np.complex128); Z_aug[:m] = Z
            del N
            projected_aug, aug_report = _project(N_aug, Z_aug, rtol, work)
            augmented.append(projected_aug)
            row["augmented"] = dict(aug_report, penalty_rows=len(penalty),
                fixed_lambda=float(group.lambda_reg), fixed_mu=float(group.template_strength),
                interpretation="prior_assisted_endpoint_sensitivity_not_pure_data_or_exact_profile_Hessian")
            del N_aug, Z_aug, penalty
        else:
            del N
        del Z
        reports.append(row)
    result = dict(order=list(ORDER), geometry_mm=np.asarray(geometry_mm).tolist(),
        angles_deg=np.asarray(angles_deg).tolist(), pose_u6=pose_u6.tolist(), groups=reports,
        data_only=_summary(pure, raw_gram, rtol, work, "unpenalized_data_fixed_b_g_full_nuisance_projection"),
        W_is_noise_precision=False, input_scope="caller-bound immutable native training groups and public fitted b/g",
        nuisance_gain_semantics="one freely complex nonreference log-gain column includes rho and i*phi directions",
        Z_semantics="positive fixed-b/g prediction tangent W*g*K_j*(u*b);not a penalized profile J",
        sensitivity_scope="fixed-endpoint linearized tangent projection;not a reprofiled residual derivative or exact profile Hessian",
        old_profile_J_or_H_used=False, caller_endpoint_and_stage_binding_required=True,
        confidence_interval=False, precision_or_identifiability_gate=None)
    if include_augmented:
        result["augmented_prior_assisted"] = _summary(augmented, raw_gram, rtol, work, "prior_assisted_augmented_nuisance_projection")
    work["total_seconds"] = time.perf_counter()-started
    result["analysis_work"] = work
    return result
