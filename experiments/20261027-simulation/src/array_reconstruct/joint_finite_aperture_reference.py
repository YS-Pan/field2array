"""Independent literal GL3 reference for the hypothetical C0 aperture model.

This is a small CPU qualification reference, not a fast full-array solver.
It never imports the stress generator or the CUDA implementation. Aperture
dimensions are full x/y sides in millimetres, with unit integrated weight.
The probe and source are each planar squares, mutually parallel to the array.
No area factor or measured element/probe response is inferred.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .joint_geometry import (JointProblem, ProfileEvaluation, _complex_array,
                             _geometry, _real_array, parameters_to_geometry)


def _sides(value, name):
    out = _real_array(value, name)
    if out.ndim == 0:
        out = np.repeat(out, 2)
    if out.shape != (2,) or np.any(out < 0):
        raise ValueError(f"{name} must be a nonnegative side or (x,y) sides in mm")
    return tuple(float(v) for v in out)


def require_c0(groups):
    """Reject every information-bearing prior, including a nonzero ridge b0."""
    for group in groups:
        if (np.any(group.b0 != 0) or group.template_prior is not None
                or group.template_strength != 0 or group.amplitude_prior is not None
                or group.amplitude_strength != 0):
            raise NotImplementedError("finite aperture backend currently supports C0 only; nonlinear amplitude pending")


def literal_finite_kernel(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                          q=2., sign=-1, *, source_aperture_mm=4.,
                          probe_aperture_mm=4., derivative=None):
    """Literal m x n x source_x/y x probe_x/y GL3 quadrature.

    ``derivative`` is None, a t-axis 0/1/2, or a second-axis pair. One t unit
    is 0.01 m. The displacement is scan+mechanical+probe-source-element;
    columns are +x and rows +y. These loops intentionally retain all 81
    terms, without the stress generator's repeated-shift merging.
    """
    elements = _real_array(elements_m, "elements_m")
    samples = _real_array(samples_xy_m, "samples_xy_m")
    if (elements.ndim != 2 or not len(elements) or elements.shape[1] not in (2, 3)
            or (elements.shape[1] == 3 and np.any(elements[:, 2] != 0))):
        raise ValueError("elements_m must be nonempty planar (n,2) or (n,3)")
    if samples.ndim != 2 or not len(samples) or samples.shape[1] != 2:
        raise ValueError("samples_xy_m must be nonempty (m,2)")
    frequency, exponent = float(frequency_hz), float(q)
    if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(exponent) or exponent < 0 or sign not in (-1, 1):
        raise ValueError("frequency positive, q nonnegative and sign +/-1 required")
    if derivative is None:
        axes = ()
    elif isinstance(derivative, (int, np.integer)) and not isinstance(derivative, bool):
        axes = (int(derivative),)
    elif isinstance(derivative, tuple) and len(derivative) == 2:
        axes = derivative
    else:
        raise ValueError("derivative must be None, an axis or a second-axis tuple")
    if any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a not in (0, 1, 2) for a in axes):
        raise ValueError("derivative axes must be 0, 1 or 2")
    source = np.asarray(_sides(source_aperture_mm, "source_aperture_mm"))*.001
    probe = np.asarray(_sides(probe_aperture_mm, "probe_aperture_mm"))*.001
    # Closed-form GL3 is independent of leggauss used by the CUDA backend.
    nodes = (-np.sqrt(3./5.), 0., np.sqrt(3./5.))
    weights = (5./18., 4./9., 5./18.)
    d, dx, dy = _geometry(geometry_mm)*.001
    wave = sign*2j*np.pi*frequency/299792458.
    answer = np.zeros((len(samples), len(elements)), complex)
    for mi, sample in enumerate(samples):
        for ni, element in enumerate(elements):
            total = 0j
            for sx, wsx in zip(nodes, weights):
                for sy, wsy in zip(nodes, weights):
                    for px, wpx in zip(nodes, weights):
                        for py, wpy in zip(nodes, weights):
                            x = sample[0]+dx+px*probe[0]/2-element[0]-sx*source[0]/2
                            y = sample[1]+dy+py*probe[1]/2-element[1]-sy*source[1]/2
                            v = np.array((d, x, y))
                            radius = np.sqrt(np.dot(v, v))
                            value = (d/radius)**exponent/radius*np.exp(wave*radius)
                            if axes:
                                # Chain rule via R_j and R_jk; deliberately no
                                # CUDA beta/gamma helper or stress arithmetic.
                                radial_log = wave-(exponent+1)/radius
                                j = axes[0]
                                radius_j = v[j]/radius
                                log_j = radial_log*radius_j+(exponent/d if j == 0 else 0.)
                                if len(axes) == 1:
                                    value *= .01*log_j
                                else:
                                    k = axes[1]
                                    radius_k = v[k]/radius
                                    log_k = radial_log*radius_k+(exponent/d if k == 0 else 0.)
                                    radius_jk = (float(j == k)-radius_j*radius_k)/radius
                                    log_jk = ((exponent+1)*radius_j*radius_k/radius**2
                                              + radial_log*radius_jk
                                              - (exponent/d**2 if j == k == 0 else 0.))
                                    value *= .0001*(log_j*log_k+log_jk)
                            total += wsx*wsy*wpx*wpy*value
            answer[mi, ni] = total
    return answer


class CPUFiniteApertureReference(JointProblem):
    """Small C0 QR reference with complete-profile derivatives, fixed aperture.

    Construct a new problem to change the physical hypothesis: same-x
    profile caching is valid only while aperture configuration is fixed.
    """
    @property
    def source_aperture_mm(self):
        return self._source_aperture_mm

    @property
    def probe_aperture_mm(self):
        return self._probe_aperture_mm

    def __init__(self, elements_m, groups, q=2., sign=-1, *,
                 source_aperture_mm=4., probe_aperture_mm=4., **options):
        groups = tuple(groups)
        require_c0(groups)
        options.pop("backend", None)
        super().__init__(elements_m, groups, q, sign, backend="qr", **options)
        self._source_aperture_mm = _sides(source_aperture_mm, "source_aperture_mm")
        self._probe_aperture_mm = _sides(probe_aperture_mm, "probe_aperture_mm")
        self._reference_state = None
        self.counts.update(kernel_second_derivative_calls=0)

    def _literal(self, session, geometry, derivative=None):
        if derivative is None:
            self.counts["kernel_calls"] += 1
        elif isinstance(derivative, tuple):
            self.counts["kernel_second_derivative_calls"] += 1
        else:
            self.counts["kernel_derivative_calls"] += 1
        return literal_finite_kernel(self.elements_m, session.samples_xy_m,
            session.frequency_hz, geometry, self.q, self.sign,
            source_aperture_mm=self.source_aperture_mm,
            probe_aperture_mm=self.probe_aperture_mm, derivative=derivative)

    def _evaluate(self, x, geometry, gains):
        p, n = self.parameter_count, len(self.elements_m)
        cache, state = {}, {}
        residuals, jacobians = [], []
        bs, dbs, data, weighted, penalties = {}, {}, {}, {}, {}
        dataJ, weightedJ, penaltyJ, diagnostics = {}, {}, {}, {}
        for group in self.groups:
            blocks, details, rhs, cursor = [], [], [], 0
            for session in group.sessions:
                key = (session.frequency_hz, session.samples_xy_m.shape, session.samples_xy_m.tobytes())
                if key not in cache:
                    cache[key] = (self._literal(session, geometry),
                        tuple(self._literal(session, geometry, j) for j in range(3)))
                K, first = cache[key]
                raw = gains[session.name]*K*session.steering[None, :]
                sl = slice(cursor, cursor+len(session.field)); cursor = sl.stop
                blocks.append(session.weight[:, None]*raw)
                rhs.append(session.weight*session.field)
                details.append((session, sl, key, raw))
            B = np.vstack(blocks+[group.lambda_reg*np.eye(n)])
            c = np.concatenate(rhs+[np.zeros(n, complex)])
            self.counts["linear_factorizations"] += 1
            Qr, R = linalg.qr(B, mode="economic")
            diagonal = np.abs(np.diag(R))
            if B.shape[0] < n or np.min(diagonal) <= np.finfo(float).eps*max(B.shape)*np.max(diagonal):
                raise linalg.LinAlgError("augmented system is rank deficient")
            b = linalg.solve_triangular(R, Qr.conj().T@c)
            r = c-B@b
            matrices = np.zeros((p, len(c), n), complex)
            for session, sl, key, raw in details:
                multiplier = (session.weight*gains[session.name])[:, None]*session.steering[None, :]
                for j in range(3):
                    matrices[j, sl] = cache[key][1][j]*multiplier
                gi = self._gain_indices.get(session.name)
                if gi is not None:
                    matrices[gi, sl], matrices[gi+1, sl] = B[sl], 1j*B[sl]
            products = np.stack([matrix@b for matrix in matrices], axis=1)
            adjoints = np.stack([matrix.conj().T@r for matrix in matrices], axis=1)
            z = linalg.solve_triangular(R.conj().T, adjoints, lower=True)
            db = linalg.solve_triangular(R, z-Qr.conj().T@products)
            J = -products-B@db
            self.counts["linear_derivative_solves"] += p
            for session, sl, key, raw in details:
                name = session.name
                data[name], weighted[name] = session.field-raw@b, r[sl].copy()
                weightedJ[name] = J[sl].copy()
                rawJ = -raw@db
                for j in range(3):
                    rawJ[:, j] -= gains[name]*(cache[key][1][j]@(session.steering*b))
                gi = self._gain_indices.get(name)
                if gi is not None:
                    rawJ[:, gi] -= raw@b
                    rawJ[:, gi+1] -= 1j*(raw@b)
                dataJ[name] = rawJ
            bs[group.name], dbs[group.name] = b, db
            penalties[group.name], penaltyJ[group.name] = r[cursor:], J[cursor:]
            stationarity = linalg.norm(B.conj().T@r)/max(linalg.norm(B)*linalg.norm(r),
                np.finfo(float).eps*linalg.norm(B)*linalg.norm(c), 1e-30)
            diagnostics[group.name] = {"backend_used": "cpu_literal_gl3_qr_reference",
                                      "stationarity": float(stationarity), "amplitude": None}
            residuals.append(r); jacobians.append(J)
            state[group.name] = (details, b, db, r, J, matrices, products, adjoints)
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        self._reference_state = (x.copy(), state)
        return ProfileEvaluation(x.copy(), geometry, np.r_[r.real, r.imag], np.vstack((J.real, J.imag)),
            bs, dbs, gains, data, weighted, penalties, dataJ, weightedJ, penaltyJ, {}, {},
            {"groups": diagnostics, "parameter_layout": list(self.parameter_layout), "counts": dict(self.counts)})

    def profile_gradient_hessian(self, x):
        evaluation = self.evaluate(x)
        p = self.parameter_count
        gradient, hessian = np.zeros(p), np.zeros((p, p))
        initial_second = self.counts["kernel_second_derivative_calls"]
        second_cache = {}
        for group in self.groups:
            details, b, db, r, J, matrices, products, adjoints = self._reference_state[1][group.name]
            curvature = np.zeros((p, p), complex)
            for session, sl, key, raw in details:
                for j in range(3):
                    for k in range(j, 3):
                        skey = key+(j, k)
                        if skey not in second_cache:
                            second_cache[skey] = self._literal(session, evaluation.geometry_mm, (j, k))
                        product = session.weight*evaluation.gains[session.name]*(second_cache[skey]@(session.steering*b))
                        value = np.vdot(r[sl], product)
                        curvature[j, k] += value
                        if j != k:
                            curvature[k, j] += value
                gi = self._gain_indices.get(session.name)
                if gi is not None:
                    for j in range(3):
                        value = np.vdot(r[sl], products[sl, j])
                        curvature[j, gi] += value; curvature[gi, j] += value
                        curvature[j, gi+1] += 1j*value; curvature[gi+1, j] += 1j*value
                    value = np.vdot(r[sl], products[sl, gi])
                    curvature[gi, gi] += value
                    curvature[gi, gi+1] += 1j*value; curvature[gi+1, gi] += 1j*value
                    curvature[gi+1, gi+1] -= value
            gradient -= (products.conj().T@r).real
            hessian -= (products.T@J.conj()+curvature+adjoints.conj().T@db).real
        asymmetry = linalg.norm(hessian-hessian.T)/max(linalg.norm(hessian), 1e-30)
        hessian = .5*(hessian+hessian.T)
        return {"evaluation": evaluation, "gradient": gradient, "hessian": hessian,
            "diagnostics": {"hessian_relative_asymmetry_before_symmetrizing": float(asymmetry),
                "envelope_gradient_relative_difference": float(linalg.norm(gradient-evaluation.jacobian.T@evaluation.residual)/max(linalg.norm(gradient), 1e-30)),
                "second_kernel_calls": self.counts["kernel_second_derivative_calls"]-initial_second,
                "additional_first_kernel_calls": 0, "backend_used": "cpu_literal_gl3_qr_reference",
                "derivative_scope": "C0 exact local profile; not a scientific identification gate"}}

    def predict(self, session, geometry_mm, b, g=1.):
        b = _complex_array(b, "b", (len(self.elements_m),))
        if not np.isfinite(complex(g)):
            raise ValueError("prediction gain must be finite")
        return complex(g)*(self._literal(session, geometry_mm)@(session.steering*b))

    def validation_evaluate(self, x, validation_groups, gains=None):
        evaluation = self.evaluate(x)
        lookup = {g.name: g for g in self.groups}
        residuals, jacobians, predictions = [], [], {}
        for name, sessions in validation_groups.items():
            if name not in lookup:
                raise ValueError("validation group must have a trained response")
            for session in sessions:
                if session.frequency_hz != lookup[name].sessions[0].frequency_hz:
                    raise ValueError("validation frequency must match the trained group")
                gain = complex((gains or {}).get(session.name, evaluation.gains.get(session.name, 1.)))
                if not np.isfinite(gain):
                    raise ValueError("validation gain must be finite")
                K = self._literal(session, evaluation.geometry_mm)
                A = gain*K*session.steering[None, :]
                prediction = A@evaluation.b[name]
                J = -A@evaluation.db[name]
                for j in range(3):
                    J[:, j] -= gain*(self._literal(session, evaluation.geometry_mm, j)@(session.steering*evaluation.b[name]))
                if session.name in self._gain_indices and not (gains and session.name in gains):
                    gi = self._gain_indices[session.name]
                    J[:, gi] -= prediction; J[:, gi+1] -= 1j*prediction
                predictions[session.name] = prediction
                residuals.append(session.weight*(session.field-prediction))
                jacobians.append(session.weight[:, None]*J)
        if not residuals:
            raise ValueError("nonempty validation sessions required")
        r, J = np.concatenate(residuals), np.vstack(jacobians)
        return {"residual": np.r_[r.real, r.imag], "jacobian": np.vstack((J.real, J.imag)),
                "predictions": predictions, "evaluation": evaluation}
