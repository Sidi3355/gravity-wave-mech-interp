"""T7 physics-graft screens on real July columns (M1): H-P1, H-P2, H-P3.

Per critic S9/S10: grafts modify ONE physical aspect of real columns; every
grafted column gets an OOD score (max per-channel |z| vs July column stats);
columns/angles breaching |z| > 4 are inadmissible and excluded from verdicts
(counts reported).

P1 critical-level graft: above level index Lc (index 0 = model top), reflect
   u about u(Lc): u'(L) = 2 u(Lc) - u(L) for L < Lc — creates a directional
   reversal aloft, continuous at Lc; below Lc unchanged. Physical prior: flux
   above the reversal is filtered. Metric: suppression = 1 - |uw'|/|uw| mean
   over channels above Lc, vs matched unmodified control.
   KILL: median suppression < 0.30. Specificity: below-Lc change reported.
P2 wind rotation over orography: Andes columns, rotate (u,v) at all levels by
   phi in {0,45,...,315} deg; near-surface predicted flux vector should rotate
   with phi (drag opposes the low-level wind). Metric: circular correlation
   between phi and the flux-vector angle change (admissible angles only).
   KILL: circular corr < 0.5.
P3 amplitude scaling: u,v,theta deviations from the July climatological
   column scaled by a in {0.25..1.5}; response = column-mean |uw|.
   KILL: median Spearman(a, response) < 0.8 (in-distribution monotonicity).

Run: python experiments/15_p_grafts.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "p_grafts.yaml"))
U, V, TH = slice(3, 125), slice(125, 247), slice(247, 369)
ANDES = (3, 21, 96, 113)
LC = 60


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m1", "uvtheta", "global")
    m1 = load_model("m1", "uvtheta", "global")

    # July column stats (OOD envelope) + climatological column per location
    cols_all, clim_sum = [], None
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        c = g[sl]
        clim_sum = c if clim_sum is None else clim_sum + c
        cols_all.append(c.transpose(1, 2, 0).reshape(-1, 369)[
            rng.choice(8192, 1024, replace=False)])
    stats = np.concatenate(cols_all)
    mu, sd = stats.mean(0), stats.std(0) + 1e-8
    clim_map = clim_sum / t_idx.size                        # (369, 64, 128)

    def zmax(x):
        return float(np.abs((x - mu) / sd).max())

    def predict(X):
        with torch.no_grad():
            return m1(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

    # pick working columns: strong low-level westerlies, global sample + Andes
    g0 = ds["features"][t_idx[len(t_idx) // 2]].values.astype(np.float32)
    if src_conv is not None:
        g0 = nz.convert_inputs_to_model(g0, src_conv)
    cols0 = g0[sl].transpose(1, 2, 0).reshape(-1, 369)

    # ---------------- P1: critical-level graft ----------------
    idx = rng.choice(8192, CFG["p1_columns"], replace=False)
    sup, below_change, n_ood = [], [], 0
    for i in idx:
        c = cols0[i].copy()
        graft = c.copy()
        graft[U][:LC] = 2 * c[U][LC] - c[U][:LC]
        if zmax(graft) > 4.0:
            n_ood += 1
            continue
        p0 = predict(c[None])[0]
        p1_ = predict(graft[None])[0]
        above0 = np.abs(p0[:LC]).mean()
        above1 = np.abs(p1_[:LC]).mean()
        below0 = np.abs(p0[LC:122]).mean()
        below1 = np.abs(p1_[LC:122]).mean()
        sup.append(1.0 - above1 / max(above0, 1e-9))
        below_change.append(abs(below1 / max(below0, 1e-9) - 1.0))
    p1 = {"hypothesis": "H-P1",
          "median_suppression_above": float(np.median(sup)),
          "iqr": [float(np.quantile(sup, 0.25)), float(np.quantile(sup, 0.75))],
          "median_below_change": float(np.median(below_change)),
          "n_admissible": len(sup), "n_ood_excluded": n_ood}
    p1["verdict"] = "KILL" if p1["median_suppression_above"] < 0.30 else "PASS"

    # ---------------- P2: wind rotation over Andes ----------------
    y1, y2, x1, x2 = ANDES
    box_idx = [(j * 128 + i) for j in range(y1, y2) for i in range(x1, x2)]
    sel = rng.choice(len(box_idx), CFG["p2_columns"], replace=False)
    angles = np.arange(0, 360, 45)
    rows = []
    for k in sel:
        c = cols0[box_idx[k]].copy()
        base = predict(c[None])[0]
        base_ang = np.degrees(np.arctan2(base[122 + 110], base[110]))
        for phi in angles[1:]:
            r = np.radians(phi)
            u, v = c[U].copy(), c[V].copy()
            cg = c.copy()
            cg[U] = u * np.cos(r) - v * np.sin(r)
            cg[V] = u * np.sin(r) + v * np.cos(r)
            ood = zmax(cg) > 4.0
            p = predict(cg[None])[0]
            ang = np.degrees(np.arctan2(p[122 + 110], p[110]))
            dang = (ang - base_ang) % 360.0
            rows.append({"phi": int(phi), "flux_rot": float(dang), "ood": bool(ood)})
    adm = [r for r in rows if not r["ood"]]
    if adm:
        ph = np.radians([r["phi"] for r in adm])
        dm = np.radians([r["flux_rot"] for r in adm])
        ccorr = float(np.abs(np.mean(np.exp(1j * (dm - ph)))))
    else:
        ccorr = float("nan")
    p2 = {"hypothesis": "H-P2", "circular_alignment": ccorr,
          "n_admissible": len(adm), "n_ood_excluded": len(rows) - len(adm),
          "mean_abs_angle_error_deg": float(np.mean(
              [min(abs(r["flux_rot"] - r["phi"]), 360 - abs(r["flux_rot"] - r["phi"]))
               for r in adm])) if adm else float("nan")}
    p2["verdict"] = "KILL" if not (ccorr >= 0.5) else "PASS"

    # ---------------- P3: amplitude scaling ----------------
    amps = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    idx3 = rng.choice(8192, CFG["p3_columns"], replace=False)
    spears, n_ood3 = [], 0
    clim_cols = clim_map.transpose(1, 2, 0).reshape(-1, 369)
    for i in idx3:
        c = cols0[i]
        cl = clim_cols[i]
        resp, ok = [], True
        for a in amps:
            cg = cl + a * (c - cl)
            if zmax(cg) > 4.0:
                ok = False
                break
            resp.append(float(np.abs(predict(cg[None])[0][:122]).mean()))
        if not ok:
            n_ood3 += 1
            continue
        spears.append(float(spearmanr(amps, resp).statistic))
    p3 = {"hypothesis": "H-P3", "median_spearman": float(np.median(spears)),
          "frac_perfectly_monotone": float(np.mean(np.array(spears) > 0.99)),
          "n_admissible": len(spears), "n_ood_excluded": n_ood3}
    p3["verdict"] = "KILL" if p3["median_spearman"] < 0.8 else "PASS"

    results = {"p1": p1, "p2": p2, "p3": p3,
               "ood_rule": "max per-channel |z| vs July column stats > 4"}
    utils.save_results(REPO / "results/screen_p_grafts/metrics.json", results,
                       CFG, CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
