"""A5 physics-baseline gate: every routine tested against analytic expectations
BEFORE it may judge any model. Run: pytest tests/test_physics.py -q
"""

import numpy as np
import pytest

from src.physics import profiles as pf
from src.physics import raytrace as rt


# ---------------------------------------------------------------- dispersion
def test_group_velocity_matches_numeric_derivative():
    """cg from closed forms == finite-difference d(omega)/d(k,l,m)."""
    N2, f = 4.0e-4, 1.0e-4
    k, l, m = 2 * np.pi / 150e3, 2 * np.pi / 400e3, -2 * np.pi / 6e3
    u = v = 0.0  # intrinsic frame

    def om(k_, l_, m_):
        return rt.dispersion_omega_hat(k_, l_, m_, N2, f, branch=+1)

    oh = om(k, l, m)
    h = 1e-9
    cgx_n = (om(k + h, l, m) - om(k - h, l, m)) / (2 * h)
    cgy_n = (om(k, l + h, m) - om(k, l - h, m)) / (2 * h)
    cgz_n = (om(k, l, m + h) - om(k, l, m - h)) / (2 * h)
    cgx, cgy, cgz = rt.group_velocity(oh, k, l, m, N2, u, v, f)
    assert np.isclose(cgx, cgx_n, rtol=1e-5)
    assert np.isclose(cgy, cgy_n, rtol=1e-5)
    assert np.isclose(cgz, cgz_n, rtol=1e-5)


def test_m_squared_consistent_with_dispersion():
    N2, f = 4.0e-4, 1.0e-4
    k, l, m = 2 * np.pi / 100e3, 0.0, -2 * np.pi / 4e3
    oh = rt.dispersion_omega_hat(k, l, m, N2, f)
    m2 = rt.m_squared(oh, k, l, N2, f)
    assert np.isclose(m2, m * m, rtol=1e-10)


# --------------------------------------------------- vertical propagation, uniform
def test_uniform_flow_straight_ray_constant_m():
    """Uniform U, constant N: m constant along ray, z(t) linear, cg_z analytic."""
    N2 = 4.0e-4
    U = 10.0
    k, l = 2 * np.pi / 50e3, 0.0
    c = 25.0  # ground-relative phase speed (eastward, faster than U)
    omega = k * c
    res = rt.trace_ray(
        k, l, omega, 1e3, pf.uniform_wind(U), pf.uniform_wind(0.0),
        pf.constant_N2(N2), f=0.0, dz=50.0, z_top=3e4,
    )
    assert res.status in ("ok", "left_domain")
    assert res.z[-1] > res.z[0]
    # m constant to numerical precision
    assert np.allclose(res.m, res.m[0], rtol=1e-8)
    # z(t) linear: residual from straight line is tiny
    p = np.polyfit(res.t, res.z, 1)
    assert np.max(np.abs(np.polyval(p, res.t) - res.z)) < 1.0  # < 1 m
    # analytic cg_z: oh = k(c-U); m^2 = k^2 (N^2-oh^2)/oh^2
    oh = k * (c - U)
    m = -np.sign(oh) * np.sqrt(rt.m_squared(oh, k, l, N2, 0.0))
    cgz = -m * oh / (k * k + m * m)
    assert np.isclose(p[0], cgz, rtol=1e-6)


def test_hydrostatic_mountain_wave_zero_horizontal_group_speed():
    """Textbook: 2D stationary hydrostatic mountain wave, ground-frame cg_x ~ 0."""
    N2 = 4.0e-4
    U = 10.0
    k = 2 * np.pi / 500e3  # long wave => hydrostatic (kh << m)
    res = rt.trace_ray(
        k, 0.0, 0.0, 1e3, pf.uniform_wind(U), pf.uniform_wind(0.0),
        pf.constant_N2(N2), f=0.0, dz=50.0, z_top=2e4,
    )
    assert res.z[-1] > 5e3  # propagates upward
    total_dx = abs(res.x[-1] - res.x[0])
    total_dz = res.z[-1] - res.z[0]
    # horizontal drift is a small fraction of vertical rise (exactly 0 in the
    # hydrostatic limit; finite k gives a small residual)
    assert total_dx / total_dz < 0.05


# ------------------------------------------------------------- critical level
def test_refraction_toward_critical_level():
    """U(z) linear with U(z_c)=c: ray stalls below z_c, |m| grows, flagged."""
    N2 = 4.0e-4
    c, z_c = 10.0, 10e3
    wind = pf.critical_level_profile(c, z_c, u_surface=0.0)
    k = 2 * np.pi / 100e3
    omega = k * c
    res = rt.trace_ray(
        k, 0.0, omega, 1e3, wind, pf.uniform_wind(0.0), pf.constant_N2(N2),
        f=0.0, dz=25.0, z_top=3e4,
    )
    assert res.status == "critical_level"
    assert res.z.max() < z_c  # never crosses
    assert res.z.max() > 0.9 * z_c  # but gets close
    assert abs(res.m[-1]) / abs(res.m[0]) > 10  # |m| -> large approaching z_c


def test_no_critical_level_when_wind_reversed():
    """Same shear magnitude but wind AWAY from c: no critical level, ray exits."""
    N2 = 4.0e-4
    c = 10.0
    wind = pf.linear_shear(0.0, -1e-3)  # U decreases with height, never = +10
    k = 2 * np.pi / 100e3
    res = rt.trace_ray(
        k, 0.0, k * c, 1e3, wind, pf.uniform_wind(0.0), pf.constant_N2(N2),
        f=0.0, dz=50.0, z_top=3e4,
    )
    assert res.status == "left_domain"
    assert res.z[-1] >= 3e4


def test_evanescent_flagged():
    """omega_hat^2 > N^2 => m^2 < 0 => evanescent status."""
    N2 = 1.0e-4  # N = 0.01 s^-1
    k = 2 * np.pi / 20e3
    c = 40.0  # oh = k*c = 0.0126 > N
    res = rt.trace_ray(
        k, 0.0, k * c, 1e3, pf.uniform_wind(0.0), pf.uniform_wind(0.0),
        pf.constant_N2(N2), f=0.0,
    )
    assert res.status == "evanescent"


# ------------------------------------------------- cone geometry under rotation
def test_cone_rotates_with_background_wind():
    """Rotating background wind and wavevector by theta rotates the footprint
    by exactly theta (rotational invariance of the ray equations)."""
    N2 = 4.0e-4
    speed = 15.0
    kh = 2 * np.pi / 80e3
    c = 5.0
    z_probe = 15e3

    def footprint(theta):
        wu = pf.uniform_wind(speed * np.cos(theta))
        wv = pf.uniform_wind(speed * np.sin(theta))
        k, l = kh * np.cos(theta), kh * np.sin(theta)
        res = rt.trace_ray(
            k, l, kh * c, 1e3, wu, wv, pf.constant_N2(N2),
            f=0.0, dz=50.0, z_top=20e3,
        )
        i = int(np.argmax(res.z >= z_probe))
        assert res.z.max() >= z_probe
        return np.array([res.x[i], res.y[i]])

    base = footprint(0.0)
    for theta in (np.pi / 6, np.pi / 2, 2.4):
        rotm = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        expect = rotm @ base
        got = footprint(theta)
        assert np.allclose(got, expect, rtol=1e-6, atol=1.0)  # 1 m tolerance


def test_propagation_cone_reports_critical_levels():
    """In shear flow, phase speeds matching the wind somewhere below z_probe
    are absorbed (critical level); faster ones pass."""
    N2 = 4.0e-4
    wind = pf.linear_shear(0.0, 2e-3)  # U = 0..40 m/s over 0..20 km
    rows = rt.propagation_cone(
        wind, pf.uniform_wind(0.0), pf.constant_N2(N2),
        z_launch=1e3, z_probe=15e3,
        phase_speeds=[5.0, 50.0], azimuths=[0.0],
        kh=2 * np.pi / 100e3, dz=25.0, z_top=20e3,
    )
    c5, c50 = rows[0], rows[1]
    assert c5[4] == 0.0 and c5[5] == 1.0   # c=5: absorbed at z=2.5 km
    assert c50[4] == 1.0                    # c=50: reaches 15 km


# ------------------------------------------------------------------- profiles
def test_isothermal_N2_analytic():
    Tfun, N2_analytic = pf.isothermal_T(250.0)
    z = np.linspace(0, 3e4, 301)
    N2 = pf.brunt_vaisala_from_T(Tfun(z), z)
    assert np.allclose(N2, N2_analytic, rtol=1e-6)


def test_richardson_number_linear_shear():
    shear, N2 = 2e-3, 4.0e-4
    z = np.linspace(0, 2e4, 201)
    u = pf.linear_shear(0.0, shear)(z)
    ri = pf.richardson_number(u, N2 * np.ones_like(z), z)
    assert np.allclose(ri, N2 / shear**2, rtol=1e-6)


def test_critical_level_profile_exact():
    wind = pf.critical_level_profile(c=12.0, z_c=8e3, u_surface=-3.0)
    assert np.isclose(wind(8e3), 12.0, rtol=1e-12)
    assert np.isclose(wind(0.0), -3.0, rtol=1e-12)


def test_gaussian_ridge_geometry():
    h = pf.gaussian_ridge(nx=65, ny=65, dx=5e3, h_max=1500.0, half_width=20e3)
    assert np.isclose(h.max(), 1500.0, rtol=1e-6)
    # value at one half-width across-ridge from center is h_max/e
    center = 32
    off = center + int(round(20e3 / 5e3))
    assert np.isclose(h[center, off], 1500.0 / np.e, rtol=1e-2)


def test_density_scale_height():
    rho, H = pf.density_isothermal(np.array([0.0, 7e3]), T0=250.0)
    assert np.isclose(H, 287.04 * 250.0 / 9.80665, rtol=1e-12)
    assert np.isclose(rho[1] / rho[0], np.exp(-7e3 / H), rtol=1e-12)
