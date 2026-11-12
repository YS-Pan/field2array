"""Complex Tikhonov inversion for equivalent radiating element coefficients.

The forward convention is exp(-i omega t), with outgoing exp(+i k R).
No measured-data loader is used in this module. A calibrated forward matrix is
needed before its coefficients can be interpreted as physical feed quantities.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import svd


@dataclass
class PreparedTikhonov:
    """Reusable decomposition of the training rows of a forward matrix.

    ``weights`` are inverse noise standard deviations, not inverse variances.
    With D = diag(column_scale), B = W A_train D^-1. The penalty is
    (alpha * s_max(B))^2 ||D (c - prior)||_2^2. Column scaling uses training
    rows only, so validation observations never affect the decomposition.
    """

    matrix: np.ndarray
    weights: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray
    column_scale: np.ndarray
    u: np.ndarray
    singular_values: np.ndarray
    vh: np.ndarray
    rank_tolerance: float
    refit_prepared: Optional["PreparedTikhonov"] = None


@dataclass
class InverseResult:
    coefficients: np.ndarray
    alpha: float
    lambda_absolute: float
    predicted: np.ndarray
    residual: np.ndarray
    selection_method: str
    alpha_curve: np.ndarray
    score_curve: np.ndarray
    singular_values: np.ndarray
    effective_rank: float
    numerical_rank: int
    relative_residual: float
    column_scale: np.ndarray
    validation_relative_error: Optional[float]
    training_count: int
    validation_count: int
    refit: bool

    def __getitem__(self, key):
        """Allow simple mapping-style access in reporting code."""
        return getattr(self, key)


def _matrix(a):
    a = np.asarray(a, dtype=np.complex128)
    if a.ndim != 2 or min(a.shape) < 1 or not np.all(np.isfinite(a)):
        raise ValueError("A must be a finite nonempty complex matrix (samples, elements)")
    return a


def _indices(indices, size):
    if indices is None:
        return np.empty(0, dtype=int)
    indices = np.asarray(indices)
    if indices.dtype == bool:
        if indices.shape != (size,):
            raise ValueError("Boolean validation mask must have one entry per sample")
        indices = np.flatnonzero(indices)
    if indices.ndim != 1 or not np.all(np.equal(indices, indices.astype(int))):
        raise ValueError("validation_indices must be one-dimensional integer indices")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("validation_indices are out of bounds")
    if np.unique(indices).size != indices.size:
        raise ValueError("validation_indices must not contain duplicates")
    return np.sort(indices)


def prepare_tikhonov(A, *, weights=None, validation_indices=None):
    """Factor A once for many synthetic excitation cases.

    We deliberately use an SVD rather than forming A^H A, which squares the
    condition number. Reuse the returned object at 1024-element scale.
    """
    a = _matrix(A)
    m, _ = a.shape
    if weights is None:
        weights = np.ones(m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (m,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite positive inverse noise standard deviations")
    valid = _indices(validation_indices, m)
    train = np.setdiff1d(np.arange(m), valid, assume_unique=True)
    if train.size == 0:
        raise ValueError("At least one training sample is required")
    weighted = weights[train, None] * a[train]
    scale = np.linalg.norm(weighted, axis=0)
    if np.any(scale == 0):
        raise ValueError("A contains an unobservable zero column on the training samples")
    b = weighted / scale[None, :]
    u, s, vh = svd(b, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    tol = max(b.shape) * np.finfo(float).eps * s[0]
    return PreparedTikhonov(a, weights, train, valid, scale, u, s, vh, tol)


def _ridge_solution(prepared, y, prior, alpha):
    s = prepared.singular_values
    lam = float(alpha) * s[0]
    centered = y[prepared.training_indices] - prepared.matrix[prepared.training_indices] @ prior
    rhs = prepared.weights[prepared.training_indices] * centered
    projection = prepared.u.conj().T @ rhs
    if lam == 0:
        good = s > prepared.rank_tolerance
        factors = np.divide(1.0, s, out=np.zeros_like(s), where=good)
        influence = good.astype(float)
    else:
        factors = s / (s * s + lam * lam)
        influence = s * s / (s * s + lam * lam)
    correction = prepared.vh.conj().T @ (factors * projection)
    return prior + correction / prepared.column_scale, float(np.sum(influence)), lam


def solve_tikhonov(
    A,
    y,
    alphas=None,
    *,
    validation_indices=None,
    validation_matrix=None,
    validation_y=None,
    validation_weights=None,
    weights=None,
    prior=None,
    refit=True,
):
    """Fit complex weights and choose a dimensionless regularization strength.

    A may be a matrix or a PreparedTikhonov object. Supply either internal
    validation row indices or an independent validation matrix and field.
    Independent off-grid/second-plane synthetic observations are preferable.
    Without validation this uses GCV, which is algorithm selection rather than
    independent physical validation. ``prior`` defaults to zero; supplying the
    nominal array weights introduces an explicit assumption about the answer.

    If internal rows are held out, alpha is selected on training rows and the
    coefficients are optionally refit to all rows. The selected held-out error
    is always from the training fit, never the refit. The full-row decomposition
    is cached in the PreparedTikhonov object for later calls.
    """
    if isinstance(A, PreparedTikhonov):
        if weights is not None or validation_indices is not None:
            raise ValueError("Specify weights and validation rows when preparing A")
        prepared = A
    else:
        prepared = prepare_tikhonov(A, weights=weights, validation_indices=validation_indices)
    a = prepared.matrix
    y = np.asarray(y, dtype=np.complex128)
    if y.shape != (a.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("y must contain one finite complex observation per row of A")
    if prior is None:
        prior = np.zeros(a.shape[1], dtype=np.complex128)
    prior = np.asarray(prior, dtype=np.complex128)
    if prior.shape != (a.shape[1],) or not np.all(np.isfinite(prior)):
        raise ValueError("prior must contain one finite complex coefficient per column")
    if alphas is None:
        alphas = np.logspace(-8, 0, 25)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0 or not np.all(np.isfinite(alphas)) or np.any(alphas < 0):
        raise ValueError("alphas must be a nonempty vector of finite nonnegative values")
    alphas = np.unique(alphas)
    external = validation_matrix is not None or validation_y is not None
    if external and prepared.validation_indices.size:
        raise ValueError("Use internal validation indices or external validation data, not both")
    if external:
        if validation_matrix is None or validation_y is None:
            raise ValueError("Both validation_matrix and validation_y are required")
        av = _matrix(validation_matrix)
        yv = np.asarray(validation_y, dtype=np.complex128)
        if av.shape[1] != a.shape[1] or yv.shape != (av.shape[0],) or not np.all(np.isfinite(yv)):
            raise ValueError("External validation data have inconsistent dimensions or nonfinite values")
        wv = np.ones(av.shape[0]) if validation_weights is None else np.asarray(validation_weights, dtype=float)
        method = "independent_validation"
    elif prepared.validation_indices.size:
        vi = prepared.validation_indices
        av, yv, wv = a[vi], y[vi], prepared.weights[vi]
        method = "heldout_rows"
    else:
        if validation_weights is not None:
            raise ValueError("validation_weights require external validation data")
        av = yv = wv = None
        method = "gcv"
    if wv is not None and (wv.shape != yv.shape or not np.all(np.isfinite(wv)) or np.any(wv <= 0)):
        raise ValueError("validation_weights must be finite positive inverse standard deviations")

    # All candidate coefficients are evaluated together to avoid repeating the
    # expensive U^H y and V matrix products for every regularization value.
    s = prepared.singular_values
    ti = prepared.training_indices
    rhs = prepared.weights[ti] * (y[ti] - a[ti] @ prior)
    projection = prepared.u.conj().T @ rhs
    lambdas = alphas * s[0]
    denominator = s[:, None] ** 2 + lambdas[None, :] ** 2
    factors = np.divide(s[:, None], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    influence = s[:, None] * factors
    zero_alpha = alphas == 0
    if np.any(zero_alpha):
        good = s > prepared.rank_tolerance
        factors[:, zero_alpha] = np.divide(1.0, s, out=np.zeros_like(s), where=good)[:, None]
        influence[:, zero_alpha] = good[:, None]
    candidates = prior[:, None] + (prepared.vh.conj().T @ (factors * projection[:, None])) / prepared.column_scale[:, None]
    effective_ranks = np.sum(influence, axis=0)
    if method == "gcv":
        # ||(I-UU^H)y||^2 plus the residual in U's subspace; complex-observation
        # counting is consistent because real and imaginary channels double
        # both the number of observations and the effective degrees of freedom.
        perpendicular = max(0.0, float(np.vdot(rhs, rhs).real - np.vdot(projection, projection).real))
        residual_norms = perpendicular + np.sum(np.abs((1 - influence) * projection[:, None]) ** 2, axis=0)
        remaining = ti.size - effective_ranks
        scores = np.divide(ti.size * residual_norms, remaining ** 2,
                           out=np.full(alphas.size, np.inf), where=remaining > np.sqrt(np.finfo(float).eps))
    else:
        differences = wv[:, None] * (av @ candidates - yv[:, None])
        scores = np.mean(np.abs(differences) ** 2, axis=0)
    if not np.any(np.isfinite(scores)):
        raise ValueError("Regularization selection has no finite score; supply independent validation data")
    chosen = int(np.nanargmin(scores))
    alpha = float(alphas[chosen])
    coefficients = candidates[:, chosen]
    validation_error = None
    if method != "gcv":
        validation_error = float(np.linalg.norm(wv * (av @ coefficients - yv)) /
                                 max(np.linalg.norm(wv * yv), np.finfo(float).tiny))
    did_refit = bool(refit and prepared.validation_indices.size)
    fitted = prepared
    if did_refit:
        if prepared.refit_prepared is None:
            prepared.refit_prepared = prepare_tikhonov(a, weights=prepared.weights)
        fitted = prepared.refit_prepared
        coefficients, effective_rank, lam = _ridge_solution(fitted, y, prior, alpha)
    else:
        effective_rank, lam = float(effective_ranks[chosen]), float(lambdas[chosen])
    prediction = a @ coefficients
    residual = y - prediction
    relative_residual = float(np.linalg.norm(prepared.weights * residual) /
                              max(np.linalg.norm(prepared.weights * y), np.finfo(float).tiny))
    return InverseResult(
        coefficients=coefficients, alpha=alpha, lambda_absolute=lam,
        predicted=prediction, residual=residual, selection_method=method,
        alpha_curve=alphas, score_curve=scores, singular_values=fitted.singular_values,
        effective_rank=effective_rank,
        numerical_rank=int(np.sum(fitted.singular_values > fitted.rank_tolerance)),
        relative_residual=relative_residual, column_scale=fitted.column_scale,
        validation_relative_error=validation_error, training_count=ti.size,
        validation_count=0 if yv is None else yv.size, refit=did_refit,
    )
