"""Synthetic background-profile and forcing generators for controlled-input
experiments (T7) and physics baselines (A5).

All generators return plain callables or numpy arrays; deterministic given args.
"""

from __future__ import annotations

import numpy as np

G = 9.80665          # m s^-2
CP = 1004.0          # J kg^-1 K^-1
R_DRY = 287.04       # J kg^-1 K^-1


# ---------------------------------------------------------------- wind profiles
def uniform_wind(u0):
    return lambda z: u0 + 0.0 * np.asarray(z)


def linear_shear(u0, shear):
    """U(z) = u0 + shear * z."""
    return lambda z: u0 + shear * np.asarray(z)


def gaussian_jet(u_max, z_jet, half_width, u_base=0.0):
    """U(z) = u_base + u_max * exp(-((z - z_jet)/half_width)^2)."""
    return lambda z: u_base + u_max * np.exp(-(((np.asarray(z) - z_jet) / half_width) ** 2))


def critical_level_profile(c, z_c, u_surface=0.0):
    """Linear wind profile constructed so that U(z_c) == c exactly.

    A wave with ground-relative phase speed c meets its critical level at z_c.
    """
    shear = (c - u_surface) / z_c
    return linear_shear(u_surface, shear)


# ------------------------------------------------------------ stability profiles
def constant_N2(N2=4.0e-4):
    return lambda z: N2 + 0.0 * np.asarray(z)


def tropopause_step_N2(N2_trop=1.0e-4, N2_strat=4.0e-4, z_tropopause=12e3, width=1e3):
    """Smooth step from tropospheric to stratospheric stability."""

    def f(z):
        z = np.asarray(z)
        s = 0.5 * (1 + np.tanh((z - z_tropopause) / width))
        return N2_trop + (N2_strat - N2_trop) * s

    return f


def brunt_vaisala_from_T(T, z):
    """N^2 = (g/T) (dT/dz + g/cp) on the given grid (centered differences)."""
    T = np.asarray(T, dtype=float)
    z = np.asarray(z, dtype=float)
    dTdz = np.gradient(T, z)
    return (G / T) * (dTdz + G / CP)


def isothermal_T(T0=250.0):
    """Isothermal atmosphere: analytic N^2 = g^2 / (cp * T0)."""
    return (lambda z: T0 + 0.0 * np.asarray(z)), G * G / (CP * T0)


def richardson_number(u, N2, z):
    """Ri = N^2 / (dU/dz)^2 elementwise on grid z."""
    dudz = np.gradient(np.asarray(u, dtype=float), np.asarray(z, dtype=float))
    with np.errstate(divide="ignore"):
        return np.asarray(N2) / np.maximum(dudz**2, 1e-30)


def density_isothermal(z, T0=250.0, rho0=1.2):
    """rho(z) = rho0 exp(-z/H), H = R T0 / g."""
    H = R_DRY * T0 / G
    return rho0 * np.exp(-np.asarray(z) / H), H


# ------------------------------------------------------------------ topography
def gaussian_ridge(nx, ny, dx, h_max, half_width, x0=None, y0=None, axis_angle=0.0):
    """2D Gaussian ridge on an (ny, nx) grid with spacing dx (meters).

    axis_angle: orientation of the ridge long axis (radians, 0 = y-aligned
    ridge, i.e. elongated in y, blocking x-flow). The ridge is elongated by
    5x along its long axis.
    """
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dx
    X, Y = np.meshgrid(x, y)
    if x0 is not None:
        X = X - x0
    if y0 is not None:
        Y = Y - y0
    ca, sa = np.cos(axis_angle), np.sin(axis_angle)
    Xr = ca * X + sa * Y      # across-ridge coordinate
    Yr = -sa * X + ca * Y     # along-ridge coordinate
    return h_max * np.exp(-((Xr / half_width) ** 2) - (Yr / (5 * half_width)) ** 2)


def sinusoidal_perturbation(nx, ny, dx, amplitude, wavelength, angle=0.0):
    """Monochromatic horizontal perturbation field, wavevector at `angle`."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)
    kx = 2 * np.pi / wavelength * np.cos(angle)
    ky = 2 * np.pi / wavelength * np.sin(angle)
    return amplitude * np.sin(kx * X + ky * Y)
