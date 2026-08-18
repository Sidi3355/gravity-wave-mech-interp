"""Gate: the 3x3 constructor must reproduce the released nonlocal file
bitwise before it may generate M2 inputs for any experiment."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data.neighborhoods import columns_3x3, make_3x3

ROOT = Path(r"C:\Users\sidi0\gwmi_data\weights\nonlocal_gwfluxes\test_files")
F1 = ROOT / "test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc"
F3 = ROOT / "test_nonlocal_3x3_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc"

pytestmark = pytest.mark.skipif(not F1.exists(), reason="released test files absent")


def test_bitwise_match_released_file():
    a, b = xr.open_dataset(F1), xr.open_dataset(F3)
    for t in range(a.sizes["time"]):
        g = a["features"].isel(time=t).values
        stored = b["features"].isel(time=t).values
        assert np.array_equal(make_3x3(g), stored), f"mismatch at t={t}"
    a.close(); b.close()


def test_columns_layout():
    g = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    cols = columns_3x3(g)
    assert cols.shape == (24, 2, 3, 3)
    # center of column (j=2, i=3) is the original value
    assert cols[2 * 6 + 3, 1, 1, 1] == g[1, 2, 3]
    # zero padding at the lat edge
    assert cols[0, 0, 0, 1] == 0.0
    # lon wrap: (j=1, i=0) west neighbor equals g[:, 1, 5]
    assert cols[1 * 6 + 0, 0, 1, 0] == g[0, 1, 5]
