"""Stage-C wave 2b: four screens sharing one July data pass.

H-N2  Gate-flux alignment: corr(finest alpha_t, |flux|_t) vs two nulls —
      (a) |flux| climatology (static geography), (b) time-mismatched |flux|.
      Null refinement over the amendment wording logged in RESEARCH_LOG
      BEFORE this run. Kill: excess over max(null) < 0.1 (Fisher-z mean).
H-I1  Per-neighbor resample ablation of M2's 3x3 ring, composited by neighbor
      bearing RELATIVE to column-mean tropospheric wind. Kill: max-min
      relative-octant effect < 20% of mean effect.
H-A4  Low-rank context channel: PCA of act6 deltas (full - clamped);
      projection-out intervention. Kill: top-5 PCs < 50% of delta variance,
      or projection moves M2 predictions < 20% toward M1's.
H-F1  [MEASUREMENT] Failure co-occurrence: top-1% (column,timestep) error
      cells overlap across models vs 1% chance; error-map rank correlations.

Run: python experiments/11_wave2b_screens.py   (CFG env var overrides config)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz
from src.data.neighborhoods import columns_3x3
from src.interp import hooks
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "wave2b.yaml"))
U_TROP = slice(3 + 90, 3 + 122)      # lower-troposphere u channels
V_TROP = slice(125 + 90, 125 + 122)
# neighbor (ky,kx) -> bearing on the grid, radians, 0 = east, pi/2 = north.
# lat row index increases NORTHWARD (lat[0]=-87.9), so +ky = north.
NEIGHBORS = {(0, 0): 5, (0, 1): 6, (0, 2): 7,
             (1, 0): 4, (1, 2): 0,
             (2, 0): 3, (2, 1): 2, (2, 2): 1}
# octant index o corresponds to bearing o*45deg: 0=E,1=NE,2=N,3=NW,4=W,5=SW,6=S,7=SE


def batched(model, x, bs):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i:i + bs]).numpy())
    return np.concatenate(outs, 0)


def fisher_mean(rs):
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    return float(np.tanh(np.mean(z)))


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    m1 = load_model("m1", "uvtheta", "global")
    m2 = load_model("m2", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")
    sl12 = feature_slice("m1", "uvtheta", "global")
    sl3 = feature_slice("m3", "uvtheta", "global")

    alpha_list, fluxmag_list = [], []
    err_maps = {m: [] for m in ("m1", "m2", "m3")}
    ablate_delta = np.zeros((len(t_idx), 8, 8192))   # (t, neighbor, column)
    rel_wind_octant = np.zeros((len(t_idx), 8, 8192), dtype=int)
    d6_full_all, d6_clamp_all, pfull_all, pclamp_all, pm1_all = [], [], [], [], []

    for it, t in enumerate(t_idx):
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        truth = o.transpose(1, 2, 0).reshape(-1, 244)
        g12 = g[sl12]
        cols = g12.transpose(1, 2, 0).reshape(-1, 369)

        # ---- forward passes + error maps (F1) + M3 gate capture (N2)
        p1 = batched(m1, torch.from_numpy(cols.copy()), 8192)
        x3 = torch.from_numpy(columns_3x3(g12))
        with hooks.ActivationCapture(m2, ["act6"]) as cap, torch.no_grad():
            pf = []
            a6 = []
            for i in range(0, 8192, 1024):
                pf.append(m2(x3[i:i + 1024]).numpy())
                a6.append(cap.acts["act6"].numpy())
            p2 = np.concatenate(pf, 0)
            act6_full = np.concatenate(a6, 0)
        with hooks.ActivationCapture(m3, ["attn2.Psi"]) as cap, torch.no_grad():
            p3maps = m3(torch.from_numpy(g[sl3])[None]).numpy()[0]
            alpha = cap.acts["attn2.Psi"][0, 0].numpy()
        p3 = p3maps.transpose(1, 2, 0).reshape(-1, 244)
        for m, p in (("m1", p1), ("m2", p2), ("m3", p3)):
            err_maps[m].append(((p - truth) ** 2).mean(axis=1))  # (8192,)
        alpha_list.append(alpha)
        fluxmag_list.append(np.abs(o).mean(axis=0))              # (64,128) norm space

        # ---- I1: per-neighbor resample ablation
        base_mse = ((p2 - truth) ** 2).mean(axis=1)
        u_mean = g[3 + 90:3 + 122].mean(axis=0)
        v_mean = g[125 + 90:125 + 122].mean(axis=0)
        wind_bearing = np.arctan2(v_mean, u_mean).reshape(-1)     # (8192,)
        perm = rng.permutation(8192)
        x3np = x3.numpy()
        for nb, (pos, bearing_oct) in enumerate(NEIGHBORS.items()):
            ky, kx = pos
            xa = x3np.copy()
            xa[:, :, ky, kx] = x3np[perm, :, ky, kx]              # resample neighbor
            pa = batched(m2, torch.from_numpy(xa), 1024)
            ablate_delta[it, nb] = ((pa - truth) ** 2).mean(axis=1) - base_mse
            rel = (bearing_oct * 45.0 - np.degrees(wind_bearing)) % 360.0
            rel_wind_octant[it, nb] = (rel // 45.0).astype(int)

        # ---- A4: clamped forward + act6 capture (sampled columns)
        sub = rng.choice(8192, CFG["a4_cols_per_timestep"], replace=False)
        clamp = np.broadcast_to(cols[sub][:, :, None, None],
                                (sub.size, 369, 3, 3)).copy()
        with hooks.ActivationCapture(m2, ["act6"]) as cap, torch.no_grad():
            pc = m2(torch.from_numpy(clamp)).numpy()
            act6_clamp = cap.acts["act6"].numpy()
        d6_full_all.append(act6_full[sub])
        d6_clamp_all.append(act6_clamp)
        pfull_all.append(p2[sub])
        pclamp_all.append(pc)
        pm1_all.append(p1[sub])
        print(f"t={t} done", flush=True)

    out_root = REPO / "results"

    # ================= H-N2 =================
    clim = np.mean(fluxmag_list, axis=0)
    rs_real, rs_clim, rs_mismatch = [], [], []
    T = len(t_idx)
    for i in range(T):
        a = alpha_list[i].ravel()
        rs_real.append(np.corrcoef(a, fluxmag_list[i].ravel())[0, 1])
        rs_clim.append(np.corrcoef(a, clim.ravel())[0, 1])
        j = (i + T // 2) % T
        rs_mismatch.append(np.corrcoef(a, fluxmag_list[j].ravel())[0, 1])
    r_real, r_clim, r_mis = map(fisher_mean, (rs_real, rs_clim, rs_mismatch))
    excess = r_real - max(r_clim, r_mis)
    n2 = {"hypothesis": "H-N2", "verdict": "KILL" if excess < 0.1 else "PASS",
          "corr_alpha_flux_t": r_real, "null_climatology": r_clim,
          "null_time_mismatch": r_mis, "excess_over_max_null": float(excess),
          "n_timesteps": T,
          "note": "excess ~ 0 with high climatology corr = gates encode static "
                  "geography (position-encoding thread), reported either way"}
    utils.save_results(out_root / "screen_n2" / "metrics.json", n2, CFG, CFG["seed"])

    # ================= H-I1 =================
    mean_eff = float(ablate_delta.mean())
    oct_eff = np.zeros(8)
    for o_ in range(8):
        mask = rel_wind_octant == o_
        oct_eff[o_] = float(ablate_delta[mask].mean())
    spread = float(oct_eff.max() - oct_eff.min())
    i1 = {"hypothesis": "H-I1",
          "verdict": "KILL" if spread < 0.2 * mean_eff else "PASS",
          "mean_ablation_delta_mse": mean_eff,
          "relative_octant_effects": {f"oct{o_}_deg{o_*45}": oct_eff[o_] for o_ in range(8)},
          "octant_spread": spread, "spread_over_mean": float(spread / mean_eff),
          "octant_convention": "0 = neighbor bearing aligned with wind (downstream), "
                               "4 = opposed (upstream); bearings on grid, rows=north",
          "n_timesteps": T}
    utils.save_results(out_root / "screen_i1" / "metrics.json", i1, CFG, CFG["seed"])

    # ================= H-A4 =================
    D = np.concatenate(d6_full_all) - np.concatenate(d6_clamp_all)   # (N, 488)
    Dc = D - D.mean(axis=0)
    U_, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    var_frac5 = float((S[:5] ** 2).sum() / (S ** 2).sum())
    proj = Vt[:5]                                                    # (5, 488)
    pfull = np.concatenate(pfull_all)
    pm1c = np.concatenate(pm1_all)
    # intervention: patch act6 on full stencils, projecting out delta directions
    idx_all = np.arange(pfull.shape[0])
    sub2 = rng.choice(idx_all.size, min(4096, idx_all.size), replace=False)
    # rebuild inputs for patched forward from stored activations is not possible;
    # instead apply the projection analytically: act6 enters output layer linearly,
    # so pred_patched = pred_full - (proj components of (act6_full-mean)) @ W_out^T
    W = m2.output.weight.detach().numpy()                            # (244, 488)
    A = np.concatenate(d6_full_all)
    comp = (A - A.mean(axis=0)) @ proj.T @ proj                      # (N, 488)
    p_patch = pfull - comp @ W.T
    d_full = float(np.sqrt(((pfull - pm1c) ** 2).mean()))
    d_patch = float(np.sqrt(((p_patch - pm1c) ** 2).mean()))
    toward = 1.0 - d_patch / d_full
    a4 = {"hypothesis": "H-A4",
          "verdict": "KILL" if (var_frac5 < 0.5 or toward < 0.2) else "PASS",
          "top5_delta_variance_fraction": var_frac5,
          "singular_value_fractions_top10": [float(x) for x in (S[:10] ** 2 / (S ** 2).sum())],
          "distance_to_m1_full": d_full, "distance_to_m1_projected": d_patch,
          "moved_toward_m1_fraction": float(toward),
          "note": "projection applied analytically at act6 (linear into output "
                  "layer); columns sampled per timestep",
          "n_columns": int(pfull.shape[0]), "n_timesteps": T}
    utils.save_results(out_root / "screen_a4" / "metrics.json", a4, CFG, CFG["seed"])

    # ================= H-F1 =================
    E = {m: np.stack(err_maps[m]) for m in err_maps}                  # (T, 8192)
    flat = {m: E[m].ravel() for m in E}
    top = {m: flat[m] >= np.quantile(flat[m], 0.99) for m in E}
    overlap = {}
    for a, b in (("m1", "m2"), ("m1", "m3"), ("m2", "m3")):
        inter = float((top[a] & top[b]).mean())
        overlap[f"{a}_{b}"] = {"joint_top1pct_frac": inter,
                               "enrichment_over_chance": inter / (0.01 * 0.01)}
    from scipy.stats import spearmanr
    rank_corr = {f"{a}_{b}": float(spearmanr(flat[a], flat[b]).statistic)
                 for a, b in (("m1", "m2"), ("m1", "m3"), ("m2", "m3"))}
    f1 = {"hypothesis": "H-F1 [MEASUREMENT]",
          "top1pct_overlap": overlap, "error_rank_correlations": rank_corr,
          "n_cells": int(flat["m1"].size), "n_timesteps": T,
          "reading": "high enrichment = shared failure regimes (data-limited)"}
    utils.save_results(out_root / "screen_f1" / "metrics.json", f1, CFG, CFG["seed"])

    print(json.dumps({"n2": n2["verdict"], "i1": i1["verdict"], "a4": a4["verdict"],
                      "f1_overlap_m2_m3": overlap["m2_m3"]["enrichment_over_chance"]},
                     indent=1))


if __name__ == "__main__":
    main()
