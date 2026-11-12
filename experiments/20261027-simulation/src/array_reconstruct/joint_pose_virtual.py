"""Independent, file-free scalar pose forward model and descriptive errors.

This implementation deliberately does not import any fitted pose kernel,
profile solver, or optimizer. It constructs three direct rotation matrices and
forms physical rays independently. Finite virtual poses may exceed fit bounds;
they are never clipped. The directional exponent split is a model assumption.
"""
from __future__ import annotations

import numpy as np

from .physics import C0_M_S


def _real(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real values") from error
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _elements(value):
    result = _real(value, "elements_m")
    if (result.ndim != 2 or not len(result) or result.shape[1] not in (2, 3)
            or (result.shape[1] == 3 and np.any(result[:, 2] != 0))):
        raise ValueError("elements_m must be nonempty planar (n,2) or (n,3), at z=0")
    if result.shape[1] == 2:
        result = np.column_stack((result, np.zeros(len(result))))
    return result


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _rotation(angles_deg):
    alpha, beta, gamma = np.deg2rad(_real(angles_deg, "angles_deg", (3,)))
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rotate_x = np.array(((1., 0., 0.), (0., ca, -sa), (0., sa, ca)), dtype=np.float64)
    rotate_y = np.array(((cb, 0., sb), (0., 1., 0.), (-sb, 0., cb)), dtype=np.float64)
    rotate_z = np.array(((cg, -sg, 0.), (sg, cg, 0.), (0., 0., 1.)), dtype=np.float64)
    return rotate_y@rotate_x@rotate_z


def independent_pose_field(elements_m, samples_xy_m, frequency_hz, geometry_mm,
                           angles_deg, effective_weights, *, q_array=2.,
                           q_probe=0., sign=-1, chunk_rows=256):
    """Return a complex128 field from independent, sample-chunked rays.

    ``geometry_mm`` is ``(d, dx, dy)`` and angles are ``(alpha,beta,gamma)``
    degrees. Element/sample coordinates are metres. ``effective_weights`` is
    already the elementwise product of local steering and complex response;
    no additional steering convention is imposed here. The pivot is the
    centroid of the complete supplied registered planar array.

    Finite truth poses need not lie within the inverse solver's bounds. A
    nonzero directional exponent requires a strictly positive matching ray
    projection. Each temporary ray/kernel block has at most ``chunk_rows``
    sample rows; there is no complete sample-by-element kernel allocation.
    """
    elements = _elements(elements_m)
    samples = _real(samples_xy_m, "samples_xy_m")
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
        raise ValueError("samples_xy_m must have nonempty shape (m,2)")
    geometry = _real(geometry_mm, "geometry_mm", (3,))
    rotation = _rotation(angles_deg)
    frequency = _scalar(frequency_hz, "frequency_hz", positive=True)
    qa, qp = _scalar(q_array, "q_array"), _scalar(q_probe, "q_probe")
    power = qa+qp+1.
    if not np.isfinite(power):
        raise ValueError("combined directional power must be finite")
    if (isinstance(sign, (bool, np.bool_)) or not isinstance(sign, (int, np.integer))
            or sign not in (-1, 1)):
        raise ValueError("sign must be integer -1 or +1")
    if (isinstance(chunk_rows, (bool, np.bool_)) or not isinstance(chunk_rows, (int, np.integer))
            or chunk_rows <= 0):
        raise ValueError("chunk_rows must be a positive integer")
    try:
        weights = np.array(effective_weights, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("effective_weights must be finite complex values") from error
    if weights.shape != (len(elements),) or not np.all(np.isfinite(weights)):
        raise ValueError("effective_weights must be a finite vector matching elements")
    center = elements.mean(axis=0)
    positions = center+(elements-center)@rotation.T
    positions[:, :2] -= geometry[1:]*.001
    normal = rotation[:, 2]
    wave = sign*2j*np.pi*(frequency/C0_M_S)
    result = np.empty(len(samples), dtype=np.complex128)
    for start in range(0, len(samples), int(chunk_rows)):
        stop = min(start+int(chunk_rows), len(samples))
        scan = np.column_stack((samples[start:stop], np.full(stop-start, geometry[0]*.001)))
        ray = scan[:, None, :]-positions[None, :, :]
        radius = np.sqrt(np.sum(ray*ray, axis=2))
        height_array, height_probe = ray@normal, ray[:, :, 2]
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("virtual pose requires a positive finite propagation radius")
        if qa and np.any(height_array <= 0):
            raise ValueError("q_array requires a positive array directional height")
        if qp and np.any(height_probe <= 0):
            raise ValueError("q_probe requires a positive probe directional height")
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            block = height_array**qa*height_probe**qp*np.exp(wave*radius)/radius**power
            result[start:stop] = block@weights
        if not np.all(np.isfinite(result[start:stop])):
            raise FloatingPointError("nonfinite independent virtual pose field")
    return result


def pose_error_metrics(elements_m, true_geometry_mm, true_angles_deg,
                       estimated_geometry_mm, estimated_angles_deg):
    """Return descriptive pose errors, without gates or identifiability claims.

    Geometry inputs are ``(d,dx,dy)`` millimetres, element inputs are metres,
    angles are degrees, and all point-displacement outputs are millimetres.
    Signed errors are estimate minus truth. Component angle errors are raw
    degree differences, while the geodesic compares the resulting rotations.

    In the fixed probe-plane frame, an element is
    ``c + R(e-c) - (dx,dy,d)``. The total displacement therefore includes all
    three translation errors; the rotation-only displacement uses the same
    registered centroid for both rotations and omits translations entirely.
    """
    elements = _elements(elements_m)
    true_geometry = _real(true_geometry_mm, "true_geometry_mm", (3,))
    estimate_geometry = _real(estimated_geometry_mm, "estimated_geometry_mm", (3,))
    true_angles = _real(true_angles_deg, "true_angles_deg", (3,))
    estimate_angles = _real(estimated_angles_deg, "estimated_angles_deg", (3,))
    true_rotation, estimate_rotation = _rotation(true_angles), _rotation(estimate_angles)
    center = elements.mean(axis=0)
    centered = elements-center
    # Form differences directly so a shared world-frame origin does not enter
    # the displacement arithmetic. Translation retains the required -d sign.
    rotation_delta = centered@(estimate_rotation-true_rotation).T
    translation_delta = (estimate_geometry-true_geometry)[[1, 2, 0]]*.001
    rotation_displacement = np.linalg.norm(rotation_delta, axis=1)*1000.
    total_displacement = np.linalg.norm(rotation_delta-translation_delta, axis=1)*1000.
    if not (np.all(np.isfinite(rotation_displacement)) and np.all(np.isfinite(total_displacement))):
        raise FloatingPointError("nonfinite virtual pose displacement metrics")
    relative = true_rotation.T@estimate_rotation
    cosine = np.clip((np.trace(relative)-1.)*.5, -1., 1.)
    skew = np.array((relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                     relative[1, 0]-relative[0, 1]))
    sine = .5*np.linalg.norm(skew)
    geodesic = float(np.rad2deg(np.arctan2(sine, cosine)))
    angle_error = estimate_angles-true_angles
    def summary(values):
        return {"max": float(np.max(values)), "p90": float(np.percentile(values, 90.))}
    return {"geometry_error_mm": (estimate_geometry-true_geometry).tolist(),
            "angle_error_deg": dict(zip(("alpha", "beta", "gamma"), angle_error.tolist())),
            "rotation_geodesic_angle_deg": geodesic,
            "rotation_only_displacement_mm": summary(rotation_displacement),
            "total_displacement_mm": summary(total_displacement),
            "element_count": len(elements), "pivot_m": center.tolist(),
            "frame": "fixed probe plane z=0: c+R(e-c)-(dx,dy,d)",
            "interpretation": "descriptive model pose errors; no pass threshold or identifiability conclusion"}
