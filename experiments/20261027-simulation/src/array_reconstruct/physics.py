"""Independent homogeneous-space kernels and a conservative angular-spectrum inverse.

All lengths are in metres, frequencies in hertz, and fields are complex phasors
with time dependence ``exp(-1j * omega * t)``. An outgoing wave therefore has
``exp(+1j * k * R)``. These kernels omit the common ``1/(4*pi)`` factor. The
electric-dipole kernel also omits ``k**2 / epsilon``: its transverse far-field
normalization is the same ``exp(+1j*k*R)/R`` as the scalar kernel.

These routines do not load measurement files. Neither regularization nor FFT
padding supplies information lost through spatial aliasing or a finite scan.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


C0_M_S = 299_792_458.0


def _positive_real(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Convert a scalar physical parameter, rejecting nonfinite/complex inputs."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real Cartesian coordinates in metres")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array of shape (count, 3)") from exc
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (count, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite coordinates")
    return result


def _polarization(value: tuple, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonzero three-vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonzero three-vector")
    norm = np.linalg.norm(result)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"{name} must be a finite nonzero three-vector")
    return result / norm


def rectangular_array(
    nx: int,
    ny: int,
    pitch_x_m: float,
    pitch_y_m: float | None = None,
    z_m: float = 0.0,
) -> np.ndarray:
    """Return a centred rectangular array as ``(ny*nx, 3)`` coordinates.

    ``meshgrid(indexing='xy')`` and C-order flattening make x vary fastest:
    rows run through x before advancing y. The array centre is ``(0, 0, z_m)``.
    """
    nx, ny = _positive_int(nx, "nx"), _positive_int(ny, "ny")
    px = _positive_real(pitch_x_m, "pitch_x_m")
    py = px if pitch_y_m is None else _positive_real(pitch_y_m, "pitch_y_m")
    z = _finite_real(z_m, "z_m")
    x = (np.arange(nx, dtype=float) - (nx - 1) / 2.0) * px
    y = (np.arange(ny, dtype=float) - (ny - 1) / 2.0) * py
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel(), np.full(nx * ny, z)))


def planar_scan(
    nx: int,
    ny: int,
    step_x_m: float,
    step_y_m: float | None = None,
    z_m: float = 0.06,
    offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return a centred x-fast planar scan; offsets translate its centre in x/y."""
    result = rectangular_array(nx, ny, step_x_m, step_y_m, z_m)
    raw = np.asarray(offset_xy_m)
    if np.iscomplexobj(raw):
        raise ValueError("offset_xy_m must contain two finite real lengths")
    try:
        offset = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset_xy_m must contain two finite real lengths") from exc
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset_xy_m must contain two finite real lengths")
    result[:, :2] += offset
    return result


def green_matrix(
    elements_m: np.ndarray,
    samples_m: np.ndarray,
    frequency_hz: float,
    model: str = "scalar",
    source_orientation: tuple = (1, 0, 0),
    receive_orientation: tuple = (1, 0, 0),
    chunk_size: int = 512,
    pattern_exponent: float = 0.0,
    *,
    directionality_power: float | None = None,
) -> np.ndarray:
    r"""Return the complex forward matrix ``A[sample, element]``.

    ``scalar`` is ``exp(+ikR)/R``. A nonzero pattern exponent multiplies either
    model by ``max(Rhat_z, 0)**q`` (a +z-facing *field-amplitude* pattern).
    ``directionality_power`` aliases ``pattern_exponent``; the alias may replace
    its zero default, while conflicting nonzero exponents are rejected.

    ``dipole`` (also ``transverse_dipole``) includes exact reactive terms:

    .. math::
       G={e^{ikR}\over R}\left[(I-\hat r\hat r^T)
          +(3\hat r\hat r^T-I)((kR)^{-2}-i(kR)^{-1})\right].

    The scalar output is ``q.conj().T @ G @ p``. Both complex polarization
    vectors are normalized in Hermitian norm. Here q denotes a receive analysis
    mode, not an unconjugated effective-length vector. The scalar model ignores
    polarization. No paraxial or far-field approximation is used in either model.
    This is an uncoupled point-source model, not an embedded-element calibration.
    """
    elements, samples = _points(elements_m, "elements_m"), _points(samples_m, "samples_m")
    frequency = _positive_real(frequency_hz, "frequency_hz")
    chunk = _positive_int(chunk_size, "chunk_size")
    exponent = _positive_real(pattern_exponent, "pattern_exponent", allow_zero=True)
    if directionality_power is not None:
        alias = _positive_real(directionality_power, "directionality_power", allow_zero=True)
        if exponent != 0.0 and exponent != alias:
            raise ValueError("pattern_exponent and directionality_power disagree")
        exponent = alias
    if model not in {"scalar", "dipole", "transverse_dipole"}:
        raise ValueError("model must be 'scalar', 'dipole', or 'transverse_dipole'")
    source = _polarization(source_orientation, "source_orientation")
    receive = _polarization(receive_orientation, "receive_orientation")
    k = 2.0 * np.pi * frequency / C0_M_S
    output = np.empty((samples.shape[0], elements.shape[0]), dtype=complex)
    for start in range(0, samples.shape[0], chunk):
        stop = min(start + chunk, samples.shape[0])
        displacement = samples[start:stop, None, :] - elements[None, :, :]
        distance = np.linalg.norm(displacement, axis=-1)
        if np.any(distance <= 0.0) or not np.all(np.isfinite(distance)):
            raise ValueError("every source-to-sample distance must be finite and > 0 metres")
        radial = displacement / distance[..., None]
        kernel = np.exp(1j * k * distance) / distance
        if model != "scalar":
            # Gp = (p-r(r.p)) + (3r(r.p)-p) * [(kR)^-2 - i/(kR)].
            radial_source = np.einsum("mni,i->mn", radial, source)
            radial_receive = np.einsum("i,mni->mn", receive.conj(), radial)
            direct = np.vdot(receive, source)
            outer_projection = radial_receive * radial_source
            inverse_kr = 1.0 / (k * distance)
            polarization = (
                direct - outer_projection
                + (3.0 * outer_projection - direct)
                * (inverse_kr**2 - 1j * inverse_kr)
            )
            kernel *= polarization
        if exponent != 0.0:
            kernel *= np.maximum(radial[..., 2], 0.0) ** exponent
        output[start:stop] = kernel
    if not np.all(np.isfinite(output)):
        raise ValueError("kernel overflow: frequency/distance is outside numerical range")
    return output


build_green_matrix = green_matrix


def angular_spectrum_backprop(
    field_2d: np.ndarray,
    dx_m: float,
    dy_m: float,
    frequency_hz: float,
    z_m: float,
    padding_factor: int = 2,
    window: str = "none",
    max_angle_deg: float = 85.0,
    probe_transfer: np.ndarray | complex | Callable | None = None,
    regularization: float = 1e-8,
) -> dict:
    """Backpropagate a planar scalar field through a homogeneous, source-free slab.

    Input shape is ``(ny, nx)``, x is axis 1, and z_m >= 0 is the distance from
    aperture to scan. Only upward, propagating plane waves are modeled. For the
    present phasor convention the inverse multiplier is ``exp(-1j*kz*z_m)``.
    Evanescent components are set to zero; none are exponentially amplified.

    Externally, ``spectrum``, all spectral masks and ``probe_transfer`` use
    fftshift-centred ordering on the *padded* grid. ``kx``/``ky`` are 1-D axes in
    rad/m; ``kx_grid``/``ky_grid`` are their xy meshgrid. A callable probe transfer
    receives these two grids and must return a broadcastable complex array T.
    Probe compensation is ``conj(T)/(|T|**2 + regularization)``; regularization is
    an absolute transfer-power parameter (T is normally normalized). Without a
    probe transfer it is unused. With zero regularization an exactly zero T is
    mapped to zero, while near-zero T may still amplify noise severely.

    Returned ``image`` is cropped back to the input shape; ``image_padded`` is
    the full inverse FFT. Windows: none, hann, hamming, blackman. A window reduces
    edge leakage while changing the measured field. Padding changes FFT grid
    interpolation and periodic replica separation, not physical resolution,
    scan coverage, sampling bandwidth, or missing information. Finite scans
    multiply the physical field by an aperture; their spectra are convolved.
    FFT propagation assumes a periodic padded field and may suffer wraparound.
    This diagnostic does not estimate discrete element excitations on its own.
    """
    try:
        field = np.asarray(field_2d, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("field_2d must be a finite nonempty complex 2-D array") from exc
    if field.ndim != 2 or 0 in field.shape or not np.all(np.isfinite(field)):
        raise ValueError("field_2d must be a finite nonempty complex 2-D array")
    dx, dy = _positive_real(dx_m, "dx_m"), _positive_real(dy_m, "dy_m")
    frequency = _positive_real(frequency_hz, "frequency_hz")
    distance = _positive_real(z_m, "z_m", allow_zero=True)
    padding = _positive_int(padding_factor, "padding_factor")
    angle = _positive_real(max_angle_deg, "max_angle_deg", allow_zero=True)
    if angle > 90.0:
        raise ValueError("max_angle_deg must be between 0 and 90 degrees")
    alpha = _positive_real(regularization, "regularization", allow_zero=True)
    windows = {
        "none": lambda n: np.ones(n),
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }
    if not isinstance(window, str) or window.lower() not in windows:
        raise ValueError("window must be 'none', 'hann', 'hamming', or 'blackman'")
    window_name = window.lower()
    ny, nx = field.shape
    padded_shape = (ny * padding, nx * padding)
    extra_y, extra_x = padded_shape[0] - ny, padded_shape[1] - nx
    before_y, before_x = extra_y // 2, extra_x // 2
    pad_width = ((before_y, extra_y - before_y), (before_x, extra_x - before_x))
    weights = np.outer(windows[window_name](ny), windows[window_name](nx))
    padded = np.pad(field * weights, pad_width)
    measured_spectrum = np.fft.fftshift(np.fft.fft2(padded))
    kx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(padded_shape[1], d=dx))
    ky = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(padded_shape[0], d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k = 2.0 * np.pi * frequency / C0_M_S
    kt_squared = kx_grid**2 + ky_grid**2
    propagating_mask = kt_squared <= k**2
    angle_mask = kt_squared <= (k * np.sin(np.deg2rad(angle))) ** 2
    retained_mask = propagating_mask & angle_mask
    kz = np.sqrt(np.maximum(k**2 - kt_squared, 0.0))
    backpropagator = np.zeros(padded_shape, dtype=complex)
    backpropagator[retained_mask] = np.exp(-1j * kz[retained_mask] * distance)
    if probe_transfer is None:
        transfer = np.ones(padded_shape, dtype=complex)
        compensation = np.ones(padded_shape, dtype=complex)
    else:
        raw_transfer = probe_transfer(kx_grid, ky_grid) if callable(probe_transfer) else probe_transfer
        try:
            transfer = np.broadcast_to(np.asarray(raw_transfer, dtype=complex), padded_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe_transfer must broadcast to the padded (ny, nx) spectrum") from exc
        if not np.all(np.isfinite(transfer)):
            raise ValueError("probe_transfer contains nonfinite values")
        denominator = np.abs(transfer) ** 2 + alpha
        compensation = np.zeros(padded_shape, dtype=complex)
        np.divide(transfer.conj(), denominator, out=compensation, where=denominator > 0)
    spectrum = measured_spectrum * compensation * backpropagator
    image_padded = np.fft.ifft2(np.fft.ifftshift(spectrum))
    crop = (slice(before_y, before_y + ny), slice(before_x, before_x + nx))
    return {
        "image": image_padded[crop].copy(),
        "image_padded": image_padded,
        "spectrum": spectrum,
        "measured_spectrum": measured_spectrum,
        "propagating_mask": propagating_mask,
        "angle_mask": angle_mask,
        "retained_mask": retained_mask,
        "kx": kx,
        "ky": ky,
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "kz": kz,
        "backpropagator": backpropagator,
        "probe_compensation": compensation,
        "probe_transfer": transfer,
        "window_weights": weights,
        "crop_slices": crop,
        "padding_factor": padding,
        "window": window_name,
        "frequency_hz": frequency,
        "z_m": distance,
        "dx_m": dx,
        "dy_m": dy,
        "max_angle_deg": angle,
        "regularization": alpha,
        "spectrum_order": "fftshift",
        "phasor_convention": "exp(-i*omega*t); outgoing exp(+i*k*R)",
        "notes": (
            "Only propagating, upward modes are retained; evanescent modes are discarded.",
            "Finite scan truncation, spatial aliasing and FFT periodic wraparound remain.",
            "Zero padding adds no measured information or physical resolution.",
            "A scalar probe transfer cannot represent full vector polarization mixing.",
        ),
    }


__all__ = [
    "C0_M_S",
    "rectangular_array",
    "planar_scan",
    "green_matrix",
    "build_green_matrix",
    "angular_spectrum_backprop",
]
