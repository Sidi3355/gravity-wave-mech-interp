"""H-A1 screen (T6): does a LINEAR model on the 3x3 stencil close the M1->M2 gap?

Closed-form ridge regression from flattened stencils (369*9=3321 features) to
all 244 outputs, fitted by accumulated normal equations (memory-safe on 8 GB).
Train/eval split by TIMESTEP: even-index sampled timesteps train, odd eval.
Lambda chosen on an internal validation split of the training timesteps.

  closed_fraction = (RMSE_M1 - RMSE_ridge) / (RMSE_M1 - RMSE_M2)   [eval set]

KILL (pre-registered): closed_fraction < 0.20.
H-A1 states >= 0.50 closes; 0.20-0.50 = partial (reported as such).

Run: python experiments/05_screen_a1_ridge.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_a1.yaml"))
NF = 369 * 9  # stencil features


def batched(model, x, bs):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i:i + bs]).numpy())
    return np.concatenate(outs, 0)


def load_t(ds, t, sl, src_conv, rng=None, n_cols=None):
    g = ds["features"][t].values.astype(np.float32)
    o = ds["output"][t].values.astype(np.float32)
    if src_conv is not None:
        g = nz.convert_inputs_to_model(g, src_conv)
        o = nz.convert_outputs_to_model(o, src_conv)
    g = g[sl]
    X = columns_3x3(g).reshape(8192, -1)                     # (8192, 3321)
    Y = o.transpose(1, 2, 0).reshape(-1, 244)
    if n_cols is not None:
        idx = rng.choice(8192, n_cols, replace=False)
        X, Y = X[idx], Y[idx]
    return X.astype(np.float64), Y.astype(np.float64), g


def fit_ridge(xtx, xty, lam):
    A = xtx + lam * np.eye(xtx.shape[0])
    return np.linalg.solve(A, xty)


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    t_train, t_eval = t_idx[::2], t_idx[1::2]
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m1", "uvtheta", "global")

    # --- accumulate normal equations over training timesteps (with intercept)
    xtx = np.zeros((NF + 1, NF + 1))
    xty = np.zeros((NF + 1, 244))
    heldout = []  # last 2 training timesteps' data for lambda choice
    for i, t in enumerate(t_train):
        X, Y, _ = load_t(ds, t, sl, src_conv, rng, CFG["cols_per_timestep"])
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        if i >= len(t_train) - 2:
            heldout.append((X1, Y))
            continue
        xtx += X1.T @ X1
        xty += X1.T @ Y
        print(f"accumulated train t={t}", flush=True)

    lams = CFG["lambda_grid"]
    val_rmse = []
    for lam in lams:
        W = fit_ridge(xtx, xty, lam)
        se = sum(float(((X1 @ W - Y) ** 2).sum()) for X1, Y in heldout)
        cnt = sum(Y.size for _, Y in heldout)
        val_rmse.append(float(np.sqrt(se / cnt)))
    lam = float(lams[int(np.argmin(val_rmse))])
    for X1, Y in heldout:  # refit on all training data
        xtx += X1.T @ X1
        xty += X1.T @ Y
    W = fit_ridge(xtx, xty, lam)

    # --- evaluate ridge vs released M1/M2 on eval timesteps (all columns)
    m1 = load_model("m1", "uvtheta", "global")
    m2 = load_model("m2", "uvtheta", "global")
    sse = {"ridge": 0.0, "m1": 0.0, "m2": 0.0}
    nel = 0
    for t in t_eval:
        X, Y, g = load_t(ds, t, sl, src_conv)
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        sse["ridge"] += float(((X1 @ W - Y) ** 2).sum())
        cols = torch.from_numpy(g.transpose(1, 2, 0).reshape(-1, 369).copy())
        sse["m1"] += float(((batched(m1, cols, 8192) - Y) ** 2).sum())
        x3 = torch.from_numpy(columns_3x3(g))
        sse["m2"] += float(((batched(m2, x3, 1024) - Y) ** 2).sum())
        nel += Y.size
        print(f"evaluated t={t}", flush=True)

    rmse = {k: float(np.sqrt(v / nel)) for k, v in sse.items()}
    gap = rmse["m1"] - rmse["m2"]
    closed = (rmse["m1"] - rmse["ridge"]) / gap if gap > 0 else float("nan")
    verdict = "KILL" if closed < 0.20 else ("PASS" if closed >= 0.50 else "PARTIAL")

    results = {
        "hypothesis": "H-A1", "verdict": verdict,
        "rmse_norm": rmse, "m1_m2_gap": float(gap),
        "fraction_of_gap_closed_by_linear_stencil": float(closed),
        "lambda": lam, "lambda_val_rmse": dict(zip(map(str, lams), val_rmse)),
        "n_train_timesteps": int(len(t_train)), "n_eval_timesteps": int(len(t_eval)),
        "cols_per_train_timestep": CFG["cols_per_timestep"],
    }
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
