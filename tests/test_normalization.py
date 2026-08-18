"""Gate: the WxC-month -> model-convention conversion must be exact and
physically consistent before any July-based experiment is trusted.

The decisive external check uses the fact that the July file's last hour
(5088) and the released August test file's first hour (5089) are CONSECUTIVE:
after conversion, per-level regression slopes between the two must be ~1
(fields one hour apart differ by weather evolution, not by scale)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data import normalization as nz

JUL = Path(r"C:\Users\sidi0\gwmi_data\era5_monthly\inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling07.nc")
AUG = Path(r"C:\Users\sidi0\gwmi_data\weights\nonlocal_gwfluxes\test_files\test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc")

pytestmark = pytest.mark.skipif(not (JUL.exists() and AUG.exists()),
                                reason="data files absent")


def test_parse_matches_registry():
    ds = xr.open_dataset(JUL)
    got = nz.parse_file_constants(ds["features"].attrs["long_name"],
                                  ds["output"].attrs["long_name"])
    assert got["u_printed"] == (nz.WXC_JULY2015["u"][0], nz.WXC_JULY2015["u"][1])
    assert got["uw"] == nz.WXC_JULY2015["uw"]
    assert got["vw"] == nz.WXC_JULY2015["vw"]
    ds.close()


def test_conversion_roundtrip_exactness():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(369, 8, 8)).astype(np.float32)
    o = rng.normal(size=(244, 8, 8)).astype(np.float32)
    g = nz.convert_inputs_to_model(f, nz.WXC_JULY2015)
    q = nz.convert_outputs_to_model(o, nz.WXC_JULY2015)
    # invert manually and compare
    d_m, mu_m = nz.MODEL_CONVENTION["u"]
    d_s, mu_s = nz.WXC_JULY2015["u"]
    back = (g[nz.U].astype(np.float64) * d_m + mu_m - mu_s) / d_s
    assert np.allclose(back, f[nz.U], atol=1e-6)
    d_m, mu_m = nz.MODEL_CONVENTION["uw"]
    d_s, mu_s = nz.WXC_JULY2015["uw"]
    back_o = np.cbrt((q[nz.UW].astype(np.float64) ** 3 * d_m + mu_m - mu_s) / d_s)
    # float32 storage floors near-zero cube-root values at ~1e-4 in normalized
    # units (= ~5e-15 Pa physically); the math itself is exact -- verified in
    # float64 below.
    assert np.allclose(back_o, o[nz.UW], atol=5e-4)
    q64 = nz.convert_outputs_to_model(o.astype(np.float64), nz.WXC_JULY2015)
    back64 = np.cbrt((q64[nz.UW] ** 3 * d_m + mu_m - mu_s) / d_s)
    assert np.allclose(back64, o[nz.UW], atol=1e-8)  # near-zero cbrt cancellation floor ~2e-9
    # theta and scalars untouched
    assert np.array_equal(g[nz.TH], f[nz.TH])
    assert np.array_equal(g[:3], f[:3])


def test_adjacent_hour_slope_unity():
    jul, aug = xr.open_dataset(JUL), xr.open_dataset(AUG)
    fj = nz.convert_inputs_to_model(
        jul["features"].isel(time=-1).values.astype(np.float64), nz.WXC_JULY2015)
    fa = aug["features"].isel(time=0).values.astype(np.float64)
    for sl in (nz.U, nz.V):
        x = fj[sl].ravel()
        y = fa[sl].ravel()
        slope = float(np.dot(x, y) / np.dot(x, x))
        corr = float(np.corrcoef(x, y)[0, 1])
        assert 0.95 < slope < 1.05, f"slope {slope}"
        assert corr > 0.98, f"corr {corr}"
    oj = nz.convert_outputs_to_model(
        jul["output"].isel(time=-1).values.astype(np.float64), nz.WXC_JULY2015)
    oa = aug["output"].isel(time=0).values.astype(np.float64)
    # fluxes decorrelate hour-to-hour (regression slope ~0.82 reflects weather,
    # not scale); the scale check is the std ratio.
    ratio = float(oj.std() / oa.std())
    assert 0.9 < ratio < 1.1, f"output std ratio {ratio}"
    jul.close(); aug.close()


def test_zs_channel_identical_across_files():
    jul, aug = xr.open_dataset(JUL), xr.open_dataset(AUG)
    zj = jul["features"].isel(time=0).values[2]
    za = aug["features"].isel(time=0).values[2]
    assert np.array_equal(zj, za)
    jul.close(); aug.close()
