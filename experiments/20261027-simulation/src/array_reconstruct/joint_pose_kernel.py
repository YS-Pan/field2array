"""Isolated six-coordinate scalar pose kernel and exact local derivatives.

This module has no file access. Registered elements are planar in antenna-local
metres, while scan coordinates are in the stationary device frame. The pivot is
the centroid of all supplied registered elements. With active column rotations,
``R = Ry(beta) @ Rx(alpha) @ Rz(gamma)`` and
``p_i = c + R @ (e_i-c) - (dx,dy,0)``. Thus positive ``dx,dy`` retain the legacy
``samples + offset - elements`` convention. Electronic steering stays local and
is deliberately outside this kernel.

The directional split ``q_array=2, q_probe=0`` is a working model assumption,
not a hardware fact. For n=R@ez, v=s-p and r=|v|, the kernel is
``(n.v)**q_array * v_z**q_probe * exp(sign*1j*k*r) / r**(q_array+q_probe+1)``.
All derivatives are with respect to dimensionless u6, ordered as
``((d-60)/10, dx/10, dy/10, alpha/1, beta/1, gamma/3)``; lengths in this tuple
are millimetres and angles are degrees. Nothing is rounded or clipped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import C0_M_S


BOUNDS_MM = np.array(((50., 70.), (-10., 10.), (-10., 10.)), dtype=np.float64)
ANGLE_BOUNDS_DEG = np.array(((-1., 1.), (-1., 1.), (-3., 3.)), dtype=np.float64)
BOUNDS_MM.flags.writeable = False
ANGLE_BOUNDS_DEG.flags.writeable = False
_ANGLE_SCALES = np.array((1., 1., 3.), dtype=np.float64)
_TRANSLATION_AXES = np.array(((0., 0., .01), (.01, 0., 0.), (0., .01, 0.)))


def _real_array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        out = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite real values") from exc
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def _bounded(value, name, bounds):
    out = _real_array(value, name, (3,))
    if np.any(out < bounds[:, 0]) or np.any(out > bounds[:, 1]):
        raise ValueError(f"{name} outside fixed pose bounds")
    return out


def _real_scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0
            or np.iscomplexobj(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(out) or (out <= 0 if positive else out < 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return out


def pose_to_parameters(geometry_mm, angles_deg):
    """Validate physical pose and return a new float64 dimensionless u6 array."""
    geometry = _bounded(geometry_mm, "geometry_mm", BOUNDS_MM)
    angles = _bounded(angles_deg, "angles_deg", ANGLE_BOUNDS_DEG)
    return np.concatenate(((geometry-np.array((60., 0., 0.)))/10.,
                           angles/_ANGLE_SCALES))


def parameters_to_pose(u6):
    """Return validated ``(geometry_mm3, angles_deg3)`` without clipping."""
    values = _real_array(u6, "normalized pose u6", (6,))
    geometry = _bounded(values[:3]*10.+np.array((60., 0., 0.)),
                        "geometry_mm", BOUNDS_MM)
    angles = _bounded(values[3:]*_ANGLE_SCALES,
                     "angles_deg", ANGLE_BOUNDS_DEG)
    return geometry, angles


def rotation_pose_derivatives(angles_deg):
    """Return ``R, dR, ddR`` for active ``Ry @ Rx @ Rz`` rotations.

    Shapes are ``(3,3)``, ``(6,3,3)``, and ``(6,6,3,3)``. The derivative axes
    use the full normalized u6 order; entries involving translations are zero.
    Degree-to-radian conversion and the gamma scale of three are included.
    """
    a, b, g = np.deg2rad(_bounded(angles_deg, "angles_deg", ANGLE_BOUNDS_DEG))
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    x = np.array(((1., 0., 0.), (0., ca, -sa), (0., sa, ca)))
    y = np.array(((cb, 0., sb), (0., 1., 0.), (-sb, 0., cb)))
    z = np.array(((cg, -sg, 0.), (sg, cg, 0.), (0., 0., 1.)))
    gx = np.array(((0., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    gy = np.array(((0., 0., 1.), (0., 0., 0.), (-1., 0., 0.)))
    gz = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 0.)))
    ax, ay, az = np.deg2rad(_ANGLE_SCALES)
    dx, dy, dz = ax*(gx@x), ay*(gy@y), az*(gz@z)
    ddx, ddy, ddz = ax*ax*(gx@gx@x), ay*ay*(gy@gy@y), az*az*(gz@gz@z)
    rotation = y@x@z
    first = np.zeros((6, 3, 3), dtype=np.float64)
    second = np.zeros((6, 6, 3, 3), dtype=np.float64)
    first[3], first[4], first[5] = y@dx@z, dy@x@z, y@x@dz
    second[3, 3], second[4, 4], second[5, 5] = y@ddx@z, ddy@x@z, y@x@ddz
    second[3, 4] = second[4, 3] = dy@dx@z
    second[3, 5] = second[5, 3] = y@dx@dz
    second[4, 5] = second[5, 4] = dy@x@dz
    return rotation, first, second


@dataclass
class _KernelState:
    centered: np.ndarray
    normal: np.ndarray
    d_rotation: np.ndarray
    dd_rotation: np.ndarray
    vector: np.ndarray
    radius: np.ndarray
    h_array: np.ndarray
    h_probe: np.ndarray
    value: np.ndarray
    radial_log: np.ndarray
    q_array: float
    q_probe: float
    radial_power: float


def _kernel_state(elements_m, samples_xy_m, f_hz, geometry_mm, angles_deg,
                  q_array, q_probe, sign):
    elements = _real_array(elements_m, "elements_m")
    samples = _real_array(samples_xy_m, "samples_xy_m")
    if (elements.ndim != 2 or not len(elements) or elements.shape[1] not in (2, 3)):
        raise ValueError("elements_m must have nonempty shape (n,2) or (n,3)")
    if elements.shape[1] == 3 and np.any(elements[:, 2] != 0):
        raise ValueError("registered elements_m must be planar at z=0")
    if elements.shape[1] == 2:
        elements = np.column_stack((elements, np.zeros(len(elements))))
    if samples.ndim != 2 or not len(samples) or samples.shape[1] != 2:
        raise ValueError("samples_xy_m must have nonempty shape (m,2)")
    frequency = _real_scalar(f_hz, "f_hz", positive=True)
    qa = _real_scalar(q_array, "q_array")
    qp = _real_scalar(q_probe, "q_probe")
    if (isinstance(sign, (bool, np.bool_)) or np.ndim(sign) != 0
            or np.iscomplexobj(sign) or sign not in (-1, 1)):
        raise ValueError("sign must be -1 or +1")
    geometry = _bounded(geometry_mm, "geometry_mm", BOUNDS_MM)
    angles = _bounded(angles_deg, "angles_deg", ANGLE_BOUNDS_DEG)
    d, dx, dy = geometry*1e-3
    center = elements.mean(axis=0)
    centered = elements-center
    rotation, dr, ddr = rotation_pose_derivatives(angles)
    # Use the registered coordinates exactly at zero pose, so the old offset
    # arithmetic and q-sum kernel are recovered without centroid roundoff.
    positions = elements if not np.any(angles) else center+centered@rotation.T
    vector = np.empty((len(samples), len(elements), 3), dtype=np.float64)
    vector[:, :, 0] = samples[:, None, 0]+dx-positions[None, :, 0]
    vector[:, :, 1] = samples[:, None, 1]+dy-positions[None, :, 1]
    vector[:, :, 2] = d-positions[None, :, 2]
    radius = np.sqrt(vector[:, :, 0]**2+vector[:, :, 1]**2+vector[:, :, 2]**2)
    if np.any(radius <= 0) or not np.all(np.isfinite(radius)):
        raise ValueError("pose produces nonpositive or nonfinite propagation radius")
    normal = rotation[:, 2]
    ha = np.einsum("mni,i->mn", vector, normal)
    hp = vector[:, :, 2]
    if qa and np.any(ha <= 0):
        raise ValueError("q_array requires strictly positive array directional height h_array")
    if qp and np.any(hp <= 0):
        raise ValueError("q_probe requires strictly positive probe directional height h_probe")
    power = qa+qp+1.
    wave = sign*1j*(2*np.pi*frequency/C0_M_S)
    if not np.any(angles):
        amplitude = d**(qa+qp)
    else:
        amplitude = ha**qa*hp**qp
    value = np.asarray(amplitude*np.exp(wave*radius)/radius**power, dtype=np.complex128)
    return _KernelState(centered, normal, dr, ddr, vector, radius, ha, hp,
                        value, wave-power/radius, qa, qp, power)


def _first_kinematics(state, j):
    # The element-only derivative broadcasts over samples. No six-coordinate
    # vector tensor is retained, including in the streaming second-order path.
    dv = -(state.centered@state.d_rotation[j].T)
    if j < 3:
        dv = dv+_TRANSLATION_AXES[j]
    dn = state.d_rotation[j, :, 2]
    rj = np.einsum("mni,ni->mn", state.vector, dv)/state.radius
    haj = (np.einsum("mni,i->mn", state.vector, dn)
           +np.einsum("ni,i->n", dv, state.normal)[None, :])
    hpj = dv[None, :, 2]
    lj = state.radial_log*rj
    if state.q_array:
        lj = lj+state.q_array*haj/state.h_array
    if state.q_probe:
        lj = lj+state.q_probe*hpj/state.h_probe
    return dv, dn, rj, haj, hpj, lj


def scalar_pose_kernel(elements_m, samples_xy_m, f_hz, geometry_mm, angles_deg,
                       *, q_array=2., q_probe=0., sign=-1, derivatives=False):
    """Return complex128 ``K`` or ``(K, dK6)`` with dK shape ``(6,m,n)``.

    Each derivative is with respect to one normalized u6 coordinate. The
    centroid belongs to the complete registered array passed in ``elements_m``;
    splitting that array into independent column batches changes the pivot.
    Nonzero directional exponents require strictly positive matching heights.
    """
    if not isinstance(derivatives, (bool, np.bool_)):
        raise ValueError("derivatives must be a boolean")
    state = _kernel_state(elements_m, samples_xy_m, f_hz, geometry_mm, angles_deg,
                          q_array, q_probe, sign)
    if not derivatives:
        return state.value
    first = np.empty((6, *state.value.shape), dtype=np.complex128)
    for j in range(6):
        first[j] = state.value*_first_kinematics(state, j)[-1]
    return state.value, first


def scalar_pose_kernel_second(elements_m, samples_xy_m, f_hz, geometry_mm, angles_deg,
                              *, j, k, q_array=2., q_probe=0., sign=-1):
    """Return only the complex128 ``(m,n)`` normalized derivative K_jk.

    This path constructs a single pair, rather than retaining all 21 symmetric
    second-order kernels. Angles, the rotating normal, and their mixed terms
    participate in the exact chain rule.
    """
    for index in (j, k):
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or not 0 <= index < 6):
            raise ValueError("second derivative indices j,k must be integers from 0 to 5")
    state = _kernel_state(elements_m, samples_xy_m, f_hz, geometry_mm, angles_deg,
                          q_array, q_probe, sign)
    vj, nj, rj, haj, hpj, lj = _first_kinematics(state, j)
    vk, nk, rk, hak, hpk, lk = _first_kinematics(state, k)
    vjk = -(state.centered@state.dd_rotation[j, k].T)
    njk = state.dd_rotation[j, k, :, 2]
    rjk = (np.einsum("ni,ni->n", vj, vk)[None, :]
           +np.einsum("mni,ni->mn", state.vector, vjk)-rj*rk)/state.radius
    hajk = (np.einsum("mni,i->mn", state.vector, njk)
            +(np.einsum("ni,i->n", vk, nj)
              +np.einsum("ni,i->n", vj, nk)
              +np.einsum("ni,i->n", vjk, state.normal))[None, :])
    hpjk = vjk[None, :, 2]
    ljk = (state.radial_log*rjk
           +state.radial_power*rj*rk/state.radius**2)
    if state.q_array:
        ljk = ljk+state.q_array*(hajk/state.h_array-haj*hak/state.h_array**2)
    if state.q_probe:
        ljk = ljk+state.q_probe*(hpjk/state.h_probe-hpj*hpk/state.h_probe**2)
    return np.asarray(state.value*(lj*lk+ljk), dtype=np.complex128)
