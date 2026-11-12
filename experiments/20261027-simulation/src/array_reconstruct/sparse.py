"""Sparse fault diagnosis with an explicit known-nominal/mostly-healthy prior.

This is a different inverse problem from arbitrary coefficient reconstruction:
    c = gain * nominal + sparse_correction.
The common complex gain cannot be separated from a common array-wide fault.
No synthetic coefficient truth enters support selection or coefficient fitting.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq


@dataclass
class PreparedSparseFaults:
    matrix: np.ndarray
    nominal: np.ndarray
    weights: np.ndarray
    weighted_matrix: np.ndarray
    baseline_field: np.ndarray
    baseline_norm_squared: float
    projected_normalized_dictionary: np.ndarray
    observable_columns: np.ndarray


@dataclass
class SparseFaultResult:
    coefficients: np.ndarray
    baseline_gain: complex
    correction: np.ndarray
    support: np.ndarray
    support_size: int
    support_size_curve: np.ndarray
    validation_score_curve: np.ndarray
    training_score_curve: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    validation_relative_error: float
    relative_residual: float
    max_faults: int
    support_path: tuple
    gain_path: np.ndarray
    selection_method: str = "independent_validation_complex_omp"
    assumption: str = "known nominal excitation; common complex gain; a minority of changed elements"

    @property
    def gain(self):
        return self.baseline_gain

    @property
    def selected_fault_count(self):
        """Number of fitted correction atoms, not thresholded hardware faults."""
        return self.support_size

    @property
    def score_curve(self):
        return self.validation_score_curve

    def __getitem__(self, key):
        return getattr(self, key)


def _matrix(a, name):
    result = np.asarray(a, dtype=np.complex128)
    if result.ndim != 2 or min(result.shape) < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite complex matrix")
    return result


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite complex vector of length {size}")
    return result


def _weights(value, size, name):
    if value is None:
        return np.ones(size)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real inverse noise standard deviations")
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must be finite positive inverse noise standard deviations")
    return result


def prepare_sparse_faults(A, nominal, *, weights=None):
    """Cache the training-only dictionary and its common-gain projection.

    With Aw=W A and h=Aw nominal, OMP uses columns
    (I-h h^H/||h||^2) Aw[:,n], each normalized by its Euclidean norm.
    Projection is implemented as a rank-one update, never a dense M by M matrix.
    Weights are inverse noise standard deviations (square-root precision).
    """
    matrix = _matrix(A, "A")
    nominal = _vector(nominal, matrix.shape[1], "nominal").copy()
    if np.linalg.norm(nominal) == 0:
        raise ValueError("nominal must be nonzero to identify a common complex gain")
    weights = _weights(weights, matrix.shape[0], "weights")
    weighted = weights[:, None] * matrix
    baseline = weighted @ nominal
    baseline_norm_squared = float(np.vdot(baseline, baseline).real)
    if baseline_norm_squared <= 0:
        raise ValueError("The nominal excitation has no observable training field")
    projected = weighted - baseline[:, None] * ((baseline.conj() @ weighted) / baseline_norm_squared)[None, :]
    norms = np.linalg.norm(projected, axis=0)
    original_norms = np.linalg.norm(weighted, axis=0)
    tolerance = max(weighted.shape) * np.finfo(float).eps
    observable = norms > tolerance * original_norms
    normalized = np.divide(projected, norms[None, :], out=np.zeros_like(projected), where=observable[None, :])
    return PreparedSparseFaults(matrix, nominal, weights, weighted, baseline,
                                baseline_norm_squared, normalized, observable)


def solve_sparse_faults(
    A,
    y,
    nominal=None,
    validation_matrix=None,
    validation_y=None,
    max_faults=None,
    *,
    weights=None,
    validation_weights=None,
):
    """Complex OMP with jointly refitted gain and independent support selection.

    Training correlations select the next atom; an SVD least-squares solve
    simultaneously refits gain and all active corrections. Candidate support
    sizes include zero. An independently supplied validation field selects the
    support size by its weighted complex MSE. There is no coefficient-truth
    argument and no implicit fallback to training error or a known fault count.

    ``A`` may be a PreparedSparseFaults object for repeated cases. In that case
    omit nominal, or pass exactly its prepared value; weights are already fixed.
    The maximum permitted correction count is min(32, N//4), an explicit model
    scope restriction. Small, dense healthy variation is model error, not an
    exactly represented feature of the sparse prior. A fitted correction atom
    is not automatically a hardware fault: apply separate amplitude/phase
    thresholds after removing the returned common gain.
    """
    if isinstance(A, PreparedSparseFaults):
        prepared = A
        if weights is not None:
            raise ValueError("Specify weights when preparing the sparse dictionary")
        if nominal is not None and not np.array_equal(np.asarray(nominal), prepared.nominal):
            raise ValueError("nominal differs from the prepared sparse dictionary")
    else:
        if nominal is None:
            raise ValueError("A known nominal excitation is required")
        prepared = prepare_sparse_faults(A, nominal, weights=weights)
    matrix, nominal = prepared.matrix, prepared.nominal
    m, n = matrix.shape
    y = _vector(y, m, "y")
    if validation_matrix is None or validation_y is None:
        raise ValueError("Independent validation_matrix and validation_y are required")
    av = _matrix(validation_matrix, "validation_matrix")
    if av.shape[1] != n:
        raise ValueError("validation_matrix must have the same number of elements as A")
    yv = _vector(validation_y, av.shape[0], "validation_y")
    wv = _weights(validation_weights, av.shape[0], "validation_weights")
    cap = min(32, n // 4)
    if max_faults is None:
        max_faults = cap
    if (isinstance(max_faults, (bool, np.bool_)) or not isinstance(max_faults, (int, np.integer))
            or not 0 <= max_faults <= cap):
        raise ValueError(f"max_faults must be an integer between 0 and {cap}")
    max_faults = int(max_faults)

    yw = prepared.weights * y
    h = prepared.baseline_field
    gain = np.vdot(h, yw) / prepared.baseline_norm_squared
    train_residual = yw - h * gain
    current = gain * nominal
    support = []
    coefficients_path = [current.copy()]
    gains = [complex(gain)]
    supports = [np.empty(0, dtype=int)]
    training_scores = [float(np.mean(np.abs(train_residual) ** 2))]
    available = prepared.observable_columns.copy()
    dictionary_adjoint = prepared.projected_normalized_dictionary.conj().T
    # Correlations below this round-off bound contain no reliable training
    # direction. In particular pure common gain terminates with support zero.
    correlation_tolerance = max(m, n) * np.finfo(float).eps * np.linalg.norm(yw)
    for _ in range(max_faults):
        accepted = False
        while np.any(available):
            correlations = np.abs(dictionary_adjoint @ train_residual)
            correlations[~available] = -np.inf
            next_atom = int(np.argmax(correlations))
            if correlations[next_atom] <= correlation_tolerance:
                break
            available[next_atom] = False
            candidate_support = support + [next_atom]
            design = np.column_stack((h, prepared.weighted_matrix[:, candidate_support]))
            scales = np.linalg.norm(design, axis=0)
            # Column scaling is only a numerical preconditioner for this
            # unregularized active-set least squares, not a new prior.
            fitted_scaled, _, rank, _ = lstsq(design / scales[None, :], yw,
                                             cond=max(design.shape) * np.finfo(float).eps,
                                             lapack_driver="gelsd", check_finite=False)
            if rank < design.shape[1]:
                continue
            fitted = fitted_scaled / scales
            gain = fitted[0]
            support = candidate_support
            current = gain * nominal
            current[support] += fitted[1:]
            train_residual = yw - design @ fitted
            coefficients_path.append(current.copy())
            gains.append(complex(gain))
            supports.append(np.asarray(support, dtype=int))
            training_scores.append(float(np.mean(np.abs(train_residual) ** 2)))
            accepted = True
            break
        if not accepted:
            break

    candidates = np.column_stack(coefficients_path)
    validation_residuals = wv[:, None] * (av @ candidates - yv[:, None])
    validation_scores = np.mean(np.abs(validation_residuals) ** 2, axis=0)
    if not np.all(np.isfinite(validation_scores)):
        raise ValueError("Validation calculation overflowed; check physical scales")
    # In an exact noiseless fit, prefer the smaller support for differences at
    # numerical round-off only. This is not a noise-level or truth-derived rule.
    reference_power = float(np.mean(np.abs(wv * yv) ** 2))
    numerical_tie = 100 * max(av.shape) * np.finfo(float).eps ** 2 * reference_power
    best_score = float(np.min(validation_scores))
    chosen = int(np.flatnonzero(validation_scores <= best_score + numerical_tie)[0])
    coefficients = candidates[:, chosen]
    gain = gains[chosen]
    correction = coefficients - gain * nominal
    prediction = matrix @ coefficients
    residual = y - prediction
    tiny = np.finfo(float).tiny
    validation_relative_error = float(np.linalg.norm(validation_residuals[:, chosen]) /
                                      max(np.linalg.norm(wv * yv), tiny))
    relative_residual = float(np.linalg.norm(prepared.weights * residual) / max(np.linalg.norm(yw), tiny))
    return SparseFaultResult(
        coefficients=coefficients, baseline_gain=gain, correction=correction,
        support=supports[chosen], support_size=supports[chosen].size,
        support_size_curve=np.arange(len(supports)), validation_score_curve=validation_scores,
        training_score_curve=np.asarray(training_scores), predicted=prediction, residual=residual,
        validation_relative_error=validation_relative_error, relative_residual=relative_residual,
        max_faults=max_faults, support_path=tuple(supports), gain_path=np.asarray(gains),
    )
