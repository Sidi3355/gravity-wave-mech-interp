"""D2-B graft battery, pre-January arms (pre-registered RESEARCH_LOG
2026-08-19 02:10):
  arm1: M1-uvtheta  P1' partial-graft reflection (July columns, beta rule)
  arm2: M1-uvthetaw P1' (August snapshot columns; w untouched) — robustness
  arm3: M3 spatially-coherent 10x10 patch reflection, suppression at center
  arm4: M1-uvthetaw P3' amplitude battery (snapshots) — robustness

Run: python experiments/21_d2_battery.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
JUL = "C:/Users/sidi0/gwmi_data/era5_monthly/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling07.nc"
AUG = "C:/Users/sidi0/gwmi_data/weights/nonlocal_gwfluxes/test_files/test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc"
ZC = 60
U_SL = slice(3, 125)
AMPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
REGIONS = {"1andes": (3, 21, 96, 113), "3himalaya": (41, 54, 26, 44),
           "5south_ocn": (8, 17, 10, 25), "7natlantic": (31, 44, 112, 124),
           "8npacific": (27, 47, 67, 87)}


def beta_reflect(col, mu, sd, zc=ZC, u_sl=U_SL):
    """Largest-beta partial reflection of u above zc (indices < zc; ERA5
    orientation: 0 = top). Returns (grafted, beta) or (None, 0)."""
    u = col[u_sl]
    for beta in np.arange(1.0, 0.0, -0.05):
        g = col.copy()
        gu = u.copy()
        gu[:zc] = u[:zc] + beta * (2 * u[zc] - 2 * u[:zc])
        g[u_sl] = gu
        if np.abs((g - mu) / sd).max() <= 4.0:
            return g, float(beta)
    return None, 0.0


def suppression(model, cols, mu, sd, idim):
    sup, betas, n_ood = [], [], 0
    for c in cols:
        g, beta = beta_reflect(c, mu, sd)
        if g is None or beta < 0.25:
            n_ood += 1
            continue
        with torch.no_grad():
            p0 = model(torch.from_numpy(c[None].astype(np.float32))).numpy()[0]
            p1 = model(torch.from_numpy(g[None].astype(np.float32))).numpy()[0]
        sup.append(1.0 - np.abs(p1[:ZC]).mean() / max(np.abs(p0[:ZC]).mean(), 1e-9))
        betas.append(beta)
    return {"median_suppression": float(np.median(sup)) if sup else None,
            "iqr": ([float(np.quantile(sup, .25)), float(np.quantile(sup, .75))]
                    if sup else None),
            "median_beta": float(np.median(betas)) if betas else None,
            "n_admissible": len(sup), "n_ood": n_ood}


def main():
    utils.set_seed(0)
    rng = np.random.default_rng(0)
    results = {}

    # ---------------- arm 1: M1-uvtheta, July ----------------
    ds = xr.open_dataset(JUL)
    src_conv = nz.detect_source_convention(ds)
    t_idx = np.unique(np.linspace(0, ds.sizes["time"] - 1, 24).astype(int))
    m1 = load_model("m1", "uvtheta", "global")
    sl = feature_slice("m1", "uvtheta", "global")
    stats, cols0 = [], None
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        c = g[sl].transpose(1, 2, 0).reshape(-1, 369)
        stats.append(c[rng.choice(8192, 1024, replace=False)])
        if t == t_idx[len(t_idx) // 2]:
            cols0 = c
    stats = np.concatenate(stats)
    mu, sd = stats.mean(0), stats.std(0) + 1e-8
    sel = rng.choice(8192, 300, replace=False)
    results["arm1_m1_uvtheta_july"] = suppression(m1, cols0[sel].astype(np.float64),
                                                  mu, sd, 369)

    # ---------------- arm 2 + 4: M1-uvthetaw, August snapshots ----------------
    ds_a = xr.open_dataset(AUG)
    m1w = load_model("m1", "uvthetaw", "global")
    slw = feature_slice("m1", "uvthetaw", "global")
    cols_a = np.concatenate([
        ds_a["features"][t].values.astype(np.float32)[slw]
        .transpose(1, 2, 0).reshape(-1, 491) for t in (0, 1)])
    mu_w, sd_w = cols_a.mean(0), cols_a.std(0) + 1e-8
    sel_w = rng.choice(cols_a.shape[0], 300, replace=False)
    results["arm2_m1_uvthetaw_aug"] = suppression(
        m1w, cols_a[sel_w].astype(np.float64), mu_w, sd_w, 491)

    spears, n_ood4 = [], 0
    clim_w = cols_a.mean(0)
    for i in rng.choice(cols_a.shape[0], 200, replace=False):
        c = cols_a[i]
        resp, ok = [], True
        for a in AMPS:
            g = clim_w + a * (c - clim_w)
            if np.abs((g - mu_w) / sd_w).max() > 4.0:
                ok = False
                break
            with torch.no_grad():
                p = m1w(torch.from_numpy(g[None].astype(np.float32))).numpy()[0]
            resp.append(float(np.abs(p[:122]).mean()))
        if not ok:
            n_ood4 += 1
            continue
        spears.append(float(spearmanr(AMPS, resp).statistic))
    results["arm4_p3_uvthetaw_aug"] = {
        "median_spearman": float(np.median(spears)),
        "n_admissible": len(spears), "n_ood": n_ood4}

    # ---------------- arm 3: M3 patch graft, July ----------------
    m3 = load_model("m3", "uvtheta", "global")
    sl3 = feature_slice("m3", "uvtheta", "global")
    t4 = np.unique(np.linspace(0, ds.sizes["time"] - 1, 4).astype(int))
    patch_sup, n_skip = [], 0
    for box, (y1, y2, x1, x2) in REGIONS.items():
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        for t in t4:
            g = ds["features"][t].values.astype(np.float32)
            if src_conv is not None:
                g = nz.convert_inputs_to_model(g, src_conv)
            gm = g.copy()
            betas = []
            for j in range(cy - 5, cy + 5):
                for i in range(cx - 5, cx + 5):
                    col = g[:, j, i % 128].astype(np.float64)
                    gcol, beta = beta_reflect(col, mu, sd)
                    if gcol is None or beta < 0.25:
                        betas = []
                        break
                    betas.append(beta)
                if not betas:
                    break
            if not betas:
                n_skip += 1
                continue
            beta_u = min(betas)
            for j in range(cy - 5, cy + 5):
                for i in range(cx - 5, cx + 5):
                    u = g[3:125, j, i % 128]
                    gm[3:125, j, i % 128][:ZC] = (
                        u[:ZC] + beta_u * (2 * u[ZC] - 2 * u[:ZC]))
            with torch.no_grad():
                p0 = m3(torch.from_numpy(g[sl3])[None]).numpy()[0]
                p1 = m3(torch.from_numpy(gm[sl3])[None]).numpy()[0]
            c0 = np.abs(p0[:ZC, cy - 2:cy + 2, cx - 2:cx + 2]).mean()
            c1 = np.abs(p1[:ZC, cy - 2:cy + 2, cx - 2:cx + 2]).mean()
            patch_sup.append({"box": box, "t": int(t), "beta": float(beta_u),
                              "suppression": float(1 - c1 / max(c0, 1e-9))})
    sups = [r["suppression"] for r in patch_sup]
    results["arm3_m3_patchgraft"] = {
        "median_suppression": float(np.median(sups)) if sups else None,
        "iqr": ([float(np.quantile(sups, .25)), float(np.quantile(sups, .75))]
                if sups else None),
        "n_patches": len(sups), "n_skipped_ood": n_skip, "rows": patch_sup}

    utils.save_results(REPO / "results/d2_battery/metrics.json", results,
                       {"zc": ZC}, seed=0)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
