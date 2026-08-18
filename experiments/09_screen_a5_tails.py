"""H-A5 (rewritten post-critic) formal screen: M2 under-disperses; M3 is
variance/tail-calibrated — and this drives the Hellinger ordering.

Evidence: (a) month-wide physical variance ratios (from experiments/02:
M1 0.360, M2 0.363, M3 1.066); (b) exact predicted/true quantile ratios
(P90/P99/P99.9 of |flux|, physical space) computed here on sampled timesteps
(month histograms are too coarse: 0.02 Pa bins vs P99 ~ 0.1 Pa).

KILL (pre-registered): |variance-ratio difference M3 vs M2| < 0.1, or P99
ratios indistinguishable (per-timestep bootstrap CIs overlap).

Run: python experiments/09_screen_a5_tails.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_a5.yaml"))
QS = [0.90, 0.99, 0.999]


def to_phys(y):
    uw = y[:, :122] ** 3 * nz.MODEL_CONVENTION["uw"][0] + nz.MODEL_CONVENTION["uw"][1]
    vw = y[:, 122:] ** 3 * nz.MODEL_CONVENTION["vw"][0] + nz.MODEL_CONVENTION["vw"][1]
    return np.concatenate([uw, vw], 1)


def batched(model, x, bs):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i:i + bs]).numpy())
    return np.concatenate(outs, 0)


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    models = {m: load_model(m, "uvtheta", "global") for m in ("m1", "m2", "m3")}
    slices = {m: feature_slice(m, "uvtheta", "global") for m in ("m1", "m2", "m3")}

    # per-timestep quantiles of |flux| (pred and true) per model
    qt = {m: {q: [] for q in QS} for m in models}
    qt_true = {q: [] for q in QS}
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        true_phys = np.abs(to_phys(o.transpose(1, 2, 0).reshape(-1, 244)))
        for q in QS:
            qt_true[q].append(float(np.quantile(true_phys, q)))
        for m in models:
            sl = slices[m]
            if m == "m1":
                x = torch.from_numpy(g[sl].transpose(1, 2, 0).reshape(-1, 369).copy())
                p = batched(models[m], x, 8192)
            elif m == "m2":
                p = batched(models[m], torch.from_numpy(columns_3x3(g[sl])), 1024)
            else:
                with torch.no_grad():
                    p = models[m](torch.from_numpy(g[sl])[None]).numpy()[0]
                p = p.transpose(1, 2, 0).reshape(-1, 244)
            pp = np.abs(to_phys(p))
            for q in QS:
                qt[m][q].append(float(np.quantile(pp, q)))

    rng = np.random.default_rng(CFG["seed"])
    def ratio_ci(pred_list, true_list):
        pred, true = np.array(pred_list), np.array(true_list)
        r = pred / true
        boots = [rng.choice(r, r.size, replace=True).mean() for _ in range(2000)]
        return float(r.mean()), [float(np.quantile(boots, 0.025)),
                                 float(np.quantile(boots, 0.975))]

    tails = {m: {} for m in models}
    for m in models:
        for q in QS:
            mean, ci = ratio_ci(qt[m][q], qt_true[q])
            tails[m][f"p{q}"] = {"ratio_mean": mean, "ci95": ci}

    var_ratio = {"m1": 0.360, "m2": 0.363, "m3": 1.066}  # from a4_fullmonth
    p99_m2, p99_m3 = tails["m2"]["p0.99"], tails["m3"]["p0.99"]
    distinguishable = (p99_m2["ci95"][1] < p99_m3["ci95"][0]
                       or p99_m3["ci95"][1] < p99_m2["ci95"][0])
    verdict = ("KILL" if (abs(var_ratio["m3"] - var_ratio["m2"]) < 0.1
                          or not distinguishable) else "PASS")

    results = {"hypothesis": "H-A5(rewritten)", "verdict": verdict,
               "variance_ratio_phys_month": var_ratio,
               "tail_ratios_pred_over_true": tails,
               "p99_m2_m3_distinguishable": bool(distinguishable),
               "n_timesteps": int(t_idx.size)}
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
