"""Stage-D gate week, G2: two pre-registered D1 sanity gates
(RESEARCH_LOG 2026-08-18 23:56; run only after that entry was committed).

G2(i)  Triviality: per-channel variance-matching gains on M2 vs M3's ladder.
G2(ii) Calibration-under-roll: M3's ladder on rolled inputs (rolls 64, 32).

Run: python experiments/18_gateweek_g2.py   (CFG env var overrides config)
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
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "gateweek_g2.yaml"))
QS = [0.90, 0.99, 0.999]


def to_phys(y):
    uw = y[:, :122] ** 3 * nz.MODEL_CONVENTION["uw"][0] + nz.MODEL_CONVENTION["uw"][1]
    vw = y[:, 122:] ** 3 * nz.MODEL_CONVENTION["vw"][0] + nz.MODEL_CONVENTION["vw"][1]
    return np.concatenate([uw, vw], 1)


def hellinger(x, y, bins=200):
    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    e = np.linspace(lo, hi, bins + 1)
    p = np.histogram(np.clip(x, lo, hi), bins=e)[0].astype(float)
    q = np.histogram(np.clip(y, lo, hi), bins=e)[0].astype(float)
    return float(1 - np.sqrt(p / p.sum() * q / q.sum()).sum())


def batched(model, x, bs=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i:i + bs]).numpy())
    return np.concatenate(outs, 0)


def ladder(pred_list, true_list, rng):
    """Per-timestep quantile ratios with bootstrap CIs + pooled stats."""
    out = {}
    for q in QS:
        r = np.array([np.quantile(np.abs(p), q) / np.quantile(np.abs(t), q)
                      for p, t in zip(pred_list, true_list)])
        boots = [rng.choice(r, r.size, replace=True).mean() for _ in range(2000)]
        out[f"p{q}"] = {"ratio_mean": float(r.mean()),
                        "ci95": [float(np.quantile(boots, .025)),
                                 float(np.quantile(boots, .975))]}
    allp = np.concatenate([p.ravel() for p in pred_list])
    allt = np.concatenate([t.ravel() for t in true_list])
    out["variance_ratio"] = float(allp.var() / allt.var())
    out["hellinger_uw"] = hellinger(
        np.concatenate([p[:, :122].ravel() for p in pred_list]),
        np.concatenate([t[:, :122].ravel() for t in true_list]))
    return out


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_all = np.unique(np.linspace(0, n - 1, 24).astype(int))
    t_fit, t_eval = t_all[::2], t_all[1::2]
    src_conv = nz.detect_source_convention(ds)
    m1 = load_model("m1", "uvtheta", "global")
    m2 = load_model("m2", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")
    sl12 = feature_slice("m1", "uvtheta", "global")
    sl3 = feature_slice("m3", "uvtheta", "global")

    def load_t(t):
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        return g, o.transpose(1, 2, 0).reshape(-1, 244)

    # ---------- fit per-channel gains on M2 (normalized space) ----------
    sp, st = np.zeros(244), np.zeros(244)
    for t in t_fit:
        g, truth = load_t(t)
        p2 = batched(m2, torch.from_numpy(columns_3x3(g[sl12])))
        sp += p2.var(axis=0)
        st += truth.var(axis=0)
    gains = np.sqrt(st / np.maximum(sp, 1e-12))

    # ---------- evaluate all arms on eval timesteps ----------
    arms = {k: {"pred": [], "true": []} for k in
            ("m1", "m2", "m2_rescaled", "m3", "m3_roll64", "m3_roll32")}
    sse = {k: 0.0 for k in arms}
    nel = 0
    for t in t_eval:
        g, truth = load_t(t)
        tp = to_phys(truth)
        p1 = batched(m1, torch.from_numpy(
            g[sl12].transpose(1, 2, 0).reshape(-1, 369).copy()), 8192)
        p2 = batched(m2, torch.from_numpy(columns_3x3(g[sl12])))
        x3 = torch.from_numpy(g[sl3])
        with torch.no_grad():
            p3 = m3(x3[None]).numpy()[0]
            p3r64 = np.roll(m3(torch.roll(x3, 64, dims=-1)[None]).numpy()[0], -64, axis=-1)
            p3r32 = np.roll(m3(torch.roll(x3, 32, dims=-1)[None]).numpy()[0], -32, axis=-1)
        preds = {"m1": p1, "m2": p2, "m2_rescaled": p2 * gains,
                 "m3": p3.transpose(1, 2, 0).reshape(-1, 244),
                 "m3_roll64": p3r64.transpose(1, 2, 0).reshape(-1, 244),
                 "m3_roll32": p3r32.transpose(1, 2, 0).reshape(-1, 244)}
        for k, p in preds.items():
            arms[k]["pred"].append(to_phys(p))
            arms[k]["true"].append(tp)
            sse[k] += float(((p - truth) ** 2).sum())
        nel += truth.size

    res = {k: ladder(arms[k]["pred"], arms[k]["true"], rng) for k in arms}
    for k in res:
        res[k]["rmse_norm"] = float(np.sqrt(sse[k] / nel))

    # ---------- pre-registered criteria ----------
    trivial = (abs(res["m2_rescaled"]["p0.99"]["ratio_mean"]
                   - res["m3"]["p0.99"]["ratio_mean"]) <= 0.10
               and abs(res["m2_rescaled"]["hellinger_uw"]
                       - res["m3"]["hellinger_uw"]) <= 0.01)
    p99_m3, p99_m2 = res["m3"]["p0.99"]["ratio_mean"], res["m2"]["p0.99"]["ratio_mean"]
    moved = (p99_m3 - res["m3_roll64"]["p0.99"]["ratio_mean"]) / max(p99_m3 - p99_m2, 1e-9)
    layout_bound = bool(moved >= 0.5 or res["m3_roll64"]["variance_ratio"] < 0.70)

    results = {"gate": "G2", "arms": res, "gains_stats": {
                   "min": float(gains.min()), "median": float(np.median(gains)),
                   "max": float(gains.max())},
               "g2i_trivial_rescaling": bool(trivial),
               "g2ii_layout_bound": layout_bound,
               "g2ii_p99_fraction_moved_toward_m2": float(moved),
               "criteria": "pre-registered RESEARCH_LOG 2026-08-18 23:56"}
    utils.save_results(REPO / "results/gateweek_g2/metrics.json", results, CFG, CFG["seed"])
    brief = {k: {"p99": round(res[k]["p0.99"]["ratio_mean"], 3),
                 "p999": round(res[k]["p0.999"]["ratio_mean"], 3),
                 "var": round(res[k]["variance_ratio"], 3),
                 "H_uw": round(res[k]["hellinger_uw"], 4),
                 "rmse": round(res[k]["rmse_norm"], 4)} for k in res}
    print(json.dumps({"g2i_trivial": trivial, "g2ii_layout_bound": layout_bound,
                      "moved_frac": round(float(moved), 3), "arms": brief}, indent=1))


if __name__ == "__main__":
    main()
