"""H-I2 screen (T3): is M2's gain over M1 mere spatial smoothing?

Intervention: clamp every 3x3 neighborhood to 9 copies of its center column
(pure-local information, zero horizontal structure) and measure how much of
the M1->M2 RMSE gap the clamped M2 retains.

  recovery = (RMSE_M1 - RMSE_M2clamp) / (RMSE_M1 - RMSE_M2full)

KILL (pre-registered): recovery > 0.70 (the gain is mostly smoothing/local).
C-3 diagnostic reported: Pearson r between M2clamp and M1 predictions.

Definition note (logged): clamping copies the center into ALL 9 positions,
including positions that in the true stencil would be zero-padded (pole rows)
or lon-wrapped; this is the cleanest "no horizontal information" arm.

Run: python experiments/04_screen_i2_clamp.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_i2.yaml"))


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
    m1 = load_model("m1", "uvtheta", "global")
    m2 = load_model("m2", "uvtheta", "global")
    sl = feature_slice("m1", "uvtheta", "global")  # same slice for m2

    sse = {"m1": 0.0, "m2_full": 0.0, "m2_clamp": 0.0}
    nel = 0
    corr_num = corr_m1sq = corr_clsq = 0.0
    m1_mean_acc = []
    per_t = {k: [] for k in sse}

    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        truth = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            truth = nz.convert_outputs_to_model(truth, src_conv)
        g = g[sl]
        true_cols = truth.transpose(1, 2, 0).reshape(-1, 244)

        cols = g.transpose(1, 2, 0).reshape(-1, 369)            # (8192, 369)
        p_m1 = batched(m1, torch.from_numpy(cols), 8192)

        x_full = torch.from_numpy(columns_3x3(g))
        p_full = batched(m2, x_full, 1024)

        clamp = np.broadcast_to(cols[:, :, None, None], (8192, 369, 3, 3)).copy()
        p_clamp = batched(m2, torch.from_numpy(clamp), 1024)

        for k, p in (("m1", p_m1), ("m2_full", p_full), ("m2_clamp", p_clamp)):
            se = float(((p - true_cols) ** 2).sum())
            sse[k] += se
            per_t[k].append(float(np.sqrt(se / true_cols.size)))
        nel += true_cols.size
        a = p_clamp - p_clamp.mean()
        b = p_m1 - p_m1.mean()
        corr_num += float((a * b).sum())
        corr_clsq += float((a * a).sum())
        corr_m1sq += float((b * b).sum())

    rmse = {k: float(np.sqrt(v / nel)) for k, v in sse.items()}
    gap = rmse["m1"] - rmse["m2_full"]
    recovery = (rmse["m1"] - rmse["m2_clamp"]) / gap if gap > 0 else float("nan")
    verdict = "KILL" if recovery > 0.70 else "PASS"
    # per-timestep consistency (3+ cases requirement)
    rec_t = [(a - c) / (a - b) if (a - b) > 0 else np.nan
             for a, b, c in zip(per_t["m1"], per_t["m2_full"], per_t["m2_clamp"])]

    results = {
        "hypothesis": "H-I2", "verdict": verdict,
        "rmse_norm": rmse, "m1_m2_gap": float(gap),
        "recovery_by_clamped_m2": float(recovery),
        "recovery_per_timestep": [float(r) for r in rec_t],
        "c3_corr_clamp_vs_m1": float(corr_num / np.sqrt(corr_clsq * corr_m1sq)),
        "n_timesteps_used": int(t_idx.size),
        "clamp_definition": "center copied into all 9 stencil positions",
    }
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
