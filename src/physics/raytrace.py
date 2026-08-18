"""Linear gravity-wave ray tracing in a horizontally homogeneous background.

Physics (standard midlatitude f-plane internal gravity waves, e.g.
Fritts & Alexander 2003, Eq. 22-24):

    Intrinsic frequency:  omega_hat = omega - k*U(z) - l*V(z)
    Dispersion relation:  omega_hat^2 = (N^2 * kh^2 + f^2 * m^2) / (kh^2 + m^2)
    with kh^2 = k^2 + l^2  (hydrostatic and nonhydrostatic limits contained).

For a steady, horizontally uniform background, k, l and the absolute
frequency omega are conserved along a ray; m adapts via the dispersion
relation:

    m^2 = kh^2 * (N^2 - omega_hat^2) / (omega_hat^2 - f^2)

Vertical group velocity (from d(omega)/dm at fixed k,l):

    cg_z = - m * (omega_hat^2 - f^2) / (omega_hat * (kh^2 + m^2))

Horizontal group velocities:

    cg_x = U + k * (N^2 - omega_hat^2) / (omega_hat * (kh^2 + m^2))
    cg_y = V + l * (N^2 - omega_hat^2) / (omega_hat * (kh^2 + m^2))

A critical level is where omega_hat -> f (rotating) or 0 (f=0): m^2 -> inf,
cg_z -> 0, the ray stalls below it. Waves with N^2 < omega_hat^2 are
evanescent (reflected in full physics; here we terminate the ray and flag).

Integration is in HEIGHT, not time: for a steady, horizontally homogeneous
background, (k, l, omega) are conserved and m(z) is algebraic from the
dispersion relation, so the ray path reduces to pure quadratures

    dx/dz = cg_x / cg_z,   dy/dz = cg_y / cg_z,   dt/dz = 1 / cg_z,

integrated on a uniform z grid (Simpson per step). Time stepping is stiff
near critical levels -- cg_z ~ omega_hat^2 so the approach TIME diverges
while the path in z stays finite -- and z stepping removes that entirely:
the asymptotic fate of a ray (reaches a probe height vs. stalls below a
critical level vs. goes evanescent) is decided in O((z_top-z0)/dz) steps.

Neglected relative to full theory: the density scale-height term 1/(4H^2)
in the dispersion relation, wave transience, and horizontal background
variation. Acceptable for a screening baseline; documented limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RayResult:
    """Trajectory of one ray. Arrays are per-timestep."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    m: np.ndarray
    omega_hat: np.ndarray
    status: str = "ok"  # ok | critical_level | evanescent | left_domain
    meta: dict = field(default_factory=dict)


def intrinsic_frequency(omega, k, l, u, v):
    return omega - k * u - l * v


def m_squared(omega_hat, k, l, N2, f=0.0):
    """Vertical wavenumber squared from the dispersion relation."""
    kh2 = k * k + l * l
    denom = omega_hat**2 - f**2
    if np.any(np.abs(denom) < 1e-30):
        raise ValueError("omega_hat^2 == f^2: critical level, m^2 undefined")
    return kh2 * (N2 - omega_hat**2) / denom


def group_velocity(omega_hat, k, l, m, N2, u, v, f=0.0):
    """(cg_x, cg_y, cg_z) for the +/- branch consistent with sign(omega_hat)."""
    kh2m2 = k * k + l * l + m * m
    common = (N2 - omega_hat**2) / (omega_hat * kh2m2)
    cgx = u + k * common
    cgy = v + l * common
    cgz = -m * (omega_hat**2 - f**2) / (omega_hat * kh2m2)
    return cgx, cgy, cgz


def dispersion_omega_hat(k, l, m, N2, f=0.0, branch=+1):
    """omega_hat from (k,l,m): the wave-frequency branch (positive/negative)."""
    kh2 = k * k + l * l
    val = np.sqrt((N2 * kh2 + f * f * m * m) / (kh2 + m * m))
    return branch * val


def trace_ray(
    k,
    l,
    omega,
    z0,
    wind_u,
    wind_v,
    N2_of_z,
    f=0.0,
    x0=0.0,
    y0=0.0,
    dz=50.0,
    z_top=8.0e4,
    omega_floor=1.0e-5,
):
    """Integrate one ray upward from z0 on a uniform height grid.

    wind_u, wind_v, N2_of_z: callables of z (m) returning U (m/s), V, N^2 (s^-2).
    The wave is defined by conserved (k, l, omega). At each height, m is
    diagnosed algebraically from the dispersion relation, sign chosen for
    upward energy propagation (cg_z > 0), and the path follows the quadratures

        dx/dz = cg_x / cg_z,   dy/dz = cg_y / cg_z,   dt/dz = 1 / cg_z,

    each integrated with Simpson's rule per dz step (endpoints + midpoint).

    Termination: top of grid reached ("left_domain"); omega_hat^2 <=
    f^2*(1+eps) or |omega_hat| <= omega_floor at any evaluation point
    ("critical_level" -- the ray stalls below it, never crossing); m^2 <= 0
    ("evanescent"). Arrays hold every grid point where the wave still
    propagates. t is diagnostic only: the approach time to a critical level
    diverges physically, so late t values are large and floor-limited.
    """
    eps_cl = 1e-4  # relative margin flagging critical-level approach

    def diagnose(z):
        """(omega_hat, m, cg_x, cg_y, cg_z) at z, or a termination status."""
        u, v, N2 = wind_u(z), wind_v(z), N2_of_z(z)
        oh = intrinsic_frequency(omega, k, l, u, v)
        if oh * oh <= f * f * (1.0 + eps_cl) or abs(oh) <= omega_floor:
            return None, "critical_level"
        m2 = m_squared(oh, k, l, N2, f)
        if m2 <= 0:
            return None, "evanescent"
        # upward energy: cg_z > 0 requires sign(m) = -sign(omega_hat)
        m = -np.sign(oh) * np.sqrt(m2)
        cgx, cgy, cgz = group_velocity(oh, k, l, m, N2, u, v, f)
        return (oh, m, cgx, cgy, cgz), "ok"

    grid = np.append(np.arange(float(z0), float(z_top), float(dz)), float(z_top))
    ts, xs, ys, zs, ms, ohs = [], [], [], [], [], []
    x, y, t = float(x0), float(y0), 0.0
    status = "ok"
    for i, z in enumerate(grid):
        d, status = diagnose(z)
        if status != "ok":
            break
        ts.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ms.append(d[1])
        ohs.append(d[0])
        if i == grid.size - 1:
            status = "left_domain"
            break
        h = grid[i + 1] - z
        d_mid, st_mid = diagnose(z + 0.5 * h)
        if st_mid != "ok":
            status = st_mid
            break
        d_end, st_end = diagnose(grid[i + 1])
        if st_end != "ok":
            status = st_end
            break
        # Simpson over (dx/dz, dy/dz, dt/dz) = (cgx, cgy, 1)/cgz
        sl = [(dd[2] / dd[4], dd[3] / dd[4], 1.0 / dd[4]) for dd in (d, d_mid, d_end)]
        x += h / 6.0 * (sl[0][0] + 4.0 * sl[1][0] + sl[2][0])
        y += h / 6.0 * (sl[0][1] + 4.0 * sl[1][1] + sl[2][1])
        t += h / 6.0 * (sl[0][2] + 4.0 * sl[1][2] + sl[2][2])

    return RayResult(
        t=np.array(ts),
        x=np.array(xs),
        y=np.array(ys),
        z=np.array(zs),
        m=np.array(ms),
        omega_hat=np.array(ohs),
        status=status,
    )


def propagation_cone(
    wind_u,
    wind_v,
    N2_of_z,
    z_launch,
    z_probe,
    phase_speeds,
    azimuths,
    kh=2 * np.pi / 100e3,
    f=0.0,
    **trace_kw,
):
    """Horizontal displacement footprint at z_probe for a fan of rays.

    Launches one ray per (phase speed c, azimuth a): wave vector
    (k,l) = kh*(cos a, sin a), ground-relative omega = kh*c. Returns an array
    of rows [c, azimuth, dx, dy, reached(1/0), status_code] where dx, dy are
    horizontal displacements at the first crossing of z_probe (NaN if the ray
    never reaches it). This is the physical baseline for "attention follows
    the propagation cone" tests (T5).
    """
    rows = []
    codes = {"ok": 0, "left_domain": 0, "critical_level": 1, "evanescent": 2}
    for c in np.atleast_1d(phase_speeds):
        for a in np.atleast_1d(azimuths):
            k = kh * np.cos(a)
            l = kh * np.sin(a)
            omega = kh * c
            res = trace_ray(k, l, omega, z_launch, wind_u, wind_v, N2_of_z, f=f, **trace_kw)
            reached = res.z.size > 0 and res.z.max() >= z_probe
            if reached:
                i = int(np.argmax(res.z >= z_probe))
                dx, dy = res.x[i], res.y[i]
            else:
                dx, dy = np.nan, np.nan
            rows.append([c, a, dx, dy, float(reached), codes.get(res.status, 3)])
    return np.array(rows)
