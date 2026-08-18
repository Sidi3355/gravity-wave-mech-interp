"""J3: expanded M3 patch-graft battery with beta dose-response.
Ledger entry J3 (RESEARCH_LOG 2026-08-18 23:56); protocol details
pre-registered 2026-08-19 02:10 (arm3) extended here: 8 regions x
cfg n_timesteps, beta sweep {0.25, 0.5, 0.75, 1.0} with beta_eff =
min(beta, beta_admissible_patch); metric = suppression at the central 4x4
above z_c vs beta_eff; dose-response = per-patch Spearman, median across
patches. Paired M1 arm on the same center columns for contrast.

Run: CFG=<yaml> python experiments/23_j3_patchdose.py
"""

import json
import os
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
CFG = utils.load_config(os.environ["CFG"])
ZC = 60
U_SL = slice(3, 125)
BETAS = [0.25, 0.5, 0.75, 1.0]
REGIONS = {"1andes": (3, 21, 96, 113), "2scand": (45, 58, 0, 12),
           "3himalaya": (41, 54, 26, 44), "4newfound": (47, 58, 103, 119),
           "5south_ocn": (8, 17, 10, 25), "6se_asia": (33, 42, 32, 49),
           "7natlantic": (31, 44, 112, 124), "8npacific": (27, 47, 67, 87)}


def graft_patch(g, cy, cx, beta):
    gm = g.copy()
    for j in range(cy - 5, cy + 5):
        for i in range(cx - 5, cx + 5):
            u = g[U_SL, j, i % 128]
            gm[U_SL, j, i % 128][:ZC] = u[:ZC] + beta * (2 * u[ZC] - 2 * u[:ZC])
    return gm


def max_admissible_beta(g, cy, cx, mu, sd):
    for beta in np.arange(1.0, 0.0, -0.05):
        gm = graft_patch(g, cy, cx, beta)
        cols = gm[:, cy - 5:cy + 5, (np.arange(cx - 5, cx + 5)) % 128]
        z = np.abs((cols - mu[:, None, None]) / sd[:, None, None]).max()
        if z <= 4.0:
            return float(beta)
    return 0.0


def main():
    utils.set_seed(0)
    rng = np.random.default_rng(0)
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl3 = feature_slice("m3", "uvtheta", "global")
    sl1 = feature_slice("m1", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")
    m1 = load_model("m1", "uvtheta", "global")

    # column stats for admissibility (this month's own converted columns)
    stats = []
    for t in t_idx[:8]:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        c = g.transpose(1, 2, 0).reshape(-1, g.shape[0])
        stats.append(c[rng.choice(8192, 1024, replace=False)])
    stats = np.concatenate(stats)
    mu, sd = stats.mean(0), stats.std(0) + 1e-8

    rows = []
    for box, (y1, y2, x1, x2) in REGIONS.items():
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        for t in t_idx:
            g = ds["features"][t].values.astype(np.float32)
            if src_conv is not None:
                g = nz.convert_inputs_to_model(g, src_conv)
            b_adm = max_admissible_beta(g, cy, cx, mu, sd)
            if b_adm < 0.25:
                rows.append({"box": box, "t": int(t), "skipped": True})
                continue
            with torch.no_grad():
                p0 = m3(torch.from_numpy(g[sl3])[None]).numpy()[0]
                c0 = np.abs(p0[:ZC, cy - 2:cy + 2, cx - 2:cx + 2]).mean()
                m1_c0 = np.abs(m1(torch.from_numpy(
                    g[sl1][:, cy, cx][None].copy())).numpy()[0][:ZC]).mean()
            sups, sups_m1, beffs = [], [], []
            for b in BETAS:
                be = min(b, b_adm)
                gm = graft_patch(g, cy, cx, be)
                with torch.no_grad():
                    p1 = m3(torch.from_numpy(gm[sl3])[None]).numpy()[0]
                    c1 = np.abs(p1[:ZC, cy - 2:cy + 2, cx - 2:cx + 2]).mean()
                    m1_c1 = np.abs(m1(torch.from_numpy(
                        gm[sl1][:, cy, cx][None].copy())).numpy()[0][:ZC]).mean()
                sups.append(float(1 - c1 / max(c0, 1e-9)))
                sups_m1.append(float(1 - m1_c1 / max(m1_c0, 1e-9)))
                beffs.append(be)
            rho = (float(spearmanr(beffs, sups).statistic)
                   if len(set(beffs)) > 1 else None)
            rows.append({"box": box, "t": int(t), "skipped": False,
                         "beta_admissible": b_adm, "beta_eff": beffs,
                         "suppression_m3": sups, "suppression_m1": sups_m1,
                         "dose_spearman_m3": rho})

    ok = [r for r in rows if not r.get("skipped")]
    rhos = [r["dose_spearman_m3"] for r in ok if r["dose_spearman_m3"] is not None]
    sup_full = [r["suppression_m3"][-1] for r in ok]
    sup_full_m1 = [r["suppression_m1"][-1] for r in ok]
    results = {"experiment": "J3 patch dose-response",
               "month_file": CFG["source_file"],
               "n_patches_admissible": len(ok), "n_skipped": len(rows) - len(ok),
               "median_dose_spearman_m3": float(np.median(rhos)) if rhos else None,
               "median_suppression_m3_at_max_beta": float(np.median(sup_full)),
               "iqr_suppression_m3": [float(np.quantile(sup_full, .25)),
                                      float(np.quantile(sup_full, .75))],
               "median_suppression_m1_paired": float(np.median(sup_full_m1)),
               "rows": rows}
    utils.save_results(Path(CFG["out_dir"]) / "metrics.json", results, CFG, seed=0)
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
