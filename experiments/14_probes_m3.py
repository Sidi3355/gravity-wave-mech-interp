"""M3 encoder probes (T1): H-R2 (orography inference) + H-R3 (regime coding).

H-R2 with the position-encoding discriminator (required by the H-P4 finding):
  - Probe zs from conv1 pixel vectors (64-d, full res), SPATIAL train/eval
    split by longitude sectors (train cols [0,32)+[64,96), eval the rest).
  - Input baseline: same probe from the raw 366-d column at the pixel.
  - ROLL ARM: run M3 on inputs rolled by 32 columns; apply the (unrolled-
    trained) probe to rolled activations at position (j,i), and score it
    against BOTH zs_source (the geography the features came from, at i-32)
    and zs_position (the map location, at i). Flow-derived orography info =>
    source wins; padding-derived position code => position wins.
  KILL: eval R^2(real) - R^2(input baseline) < 0.1.

H-R3: orographic-vs-nonorographic pixel classification from conv2 vectors
  (128-d, pooled grid): train Andes-vs-{SouthOcn,SEAsia}, test
  Himalaya-vs-{NAtl,NPac}. Controls: random-init UNet, input-baseline,
  shuffled labels. KILL: cross-region AUC < 0.65 (report whether
  within-region AUC > 0.8 indicates geographic-only coding).

Run: python experiments/14_probes_m3.py   (CFG env var overrides config)
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
from src.interp import hooks
from src.models.anchor.model_definition import Attention_UNet
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "probes_m3.yaml"))
BOXES = {"1andes": (3, 21, 96, 113), "3himalaya": (41, 54, 26, 44),
         "5south_ocn": (8, 17, 10, 25), "6se_asia": (33, 42, 32, 49),
         "7natlantic": (31, 44, 112, 124), "8npacific": (27, 47, 67, 87)}
TRAIN_COLS = np.r_[0:32, 64:96]
EVAL_COLS = np.r_[32:64, 96:128]


class RidgeProbe:
    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xs = np.hstack([(X - self.mu) / self.sd, np.ones((X.shape[0], 1))])
        A = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1])
        self.w = np.linalg.solve(A, Xs.T @ y)
        return self

    def predict(self, X):
        Xs = np.hstack([(X - self.mu) / self.sd, np.ones((X.shape[0], 1))])
        return Xs @ self.w


def r2(pred, y):
    return float(1.0 - ((pred - y) ** 2).mean() / y.var())


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(1, score.size + 1)
    pos = label > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def capture(model, ds, t_list, sl, src_conv, site, roll=0):
    """Return activations (T, C, H, W) at `site` for the given timesteps."""
    acts = []
    for t in t_list:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        x = torch.from_numpy(g[sl])
        if roll:
            x = torch.roll(x, roll, dims=-1)
        a = hooks.capture_unet_maps(model, x[None], sites=[site])[site][0]
        acts.append(a.numpy())
    return np.stack(acts)                     # (T, C, H, W)


def pixels(acts, rows, cols_idx):
    """(T, C, H, W) -> (T*len(rows)*len(cols), C) at the given row/col grids."""
    sub = acts[:, :, rows][:, :, :, cols_idx]           # (T, C, r, c)
    return sub.transpose(0, 2, 3, 1).reshape(-1, acts.shape[1])


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_all = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m3", "uvtheta", "global")
    real = load_model("m3", "uvtheta", "global")
    torch.manual_seed(0)
    rand = Attention_UNet(ch_in=366, ch_out=244, dropout=0.0).eval()

    zs = ds["features"][0].values[2].astype(np.float64)          # (64, 128) static
    rows = np.arange(4, 60)                                      # avoid pole pads

    # ---------------- H-R2 (conv1, full res) ----------------
    a_real = capture(real, ds, t_all, sl, src_conv, "conv1")
    a_rand = capture(rand, ds, t_all, sl, src_conv, "conv1")
    # raw-input baseline at pixel: the 366-d column
    raw = []
    for t in t_all:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        raw.append(g[sl])
    raw = np.stack(raw)

    def zsvec(cols_idx):
        return np.tile(zs[rows][:, cols_idx].ravel(), t_all.size)

    res2 = {}
    for name, A in (("real", a_real), ("randinit", a_rand), ("input", raw)):
        Xtr = pixels(A, rows, TRAIN_COLS)
        Xev = pixels(A, rows, EVAL_COLS)
        pr = RidgeProbe().fit(Xtr, zsvec(TRAIN_COLS))
        res2[name] = r2(pr.predict(Xev), zsvec(EVAL_COLS))
        if name == "real":
            probe_real = pr
    sel = res2["real"] - max(res2["input"], res2["randinit"])
    # ---- roll arm (probe trained unrolled, applied to rolled activations)
    a_roll = capture(real, ds, t_all, sl, src_conv, "conv1", roll=CFG["roll"])
    Xr = pixels(a_roll, rows, EVAL_COLS)
    zs_pos = zsvec(EVAL_COLS)
    src_cols = (EVAL_COLS - CFG["roll"]) % 128
    zs_src = zsvec(src_cols)
    r2_src = r2(probe_real.predict(Xr), zs_src)
    r2_pos = r2(probe_real.predict(Xr), zs_pos)
    r2_verdict = "KILL" if sel < 0.1 else "PASS"
    h_r2 = {"verdict": r2_verdict, "r2_eval": {k: round(v, 4) for k, v in res2.items()},
            "selectivity_over_baselines": round(float(sel), 4),
            "roll_arm": {"r2_vs_source_geography": round(r2_src, 4),
                         "r2_vs_map_position": round(r2_pos, 4),
                         "reading": "source >> position = flow-derived; "
                                    "position >> source = padding position code"}}

    # ---------------- H-R3 (conv2, pooled grid) ----------------
    def box_pixels(A2, box, pool=2):
        y1, y2, x1, x2 = BOXES[box]
        rr = np.arange(y1 // pool, y2 // pool)
        cc = np.arange(x1 // pool, x2 // pool)
        return pixels(A2, rr, cc)

    res3 = {}
    for name, model in (("real", real), ("randinit", rand)):
        A2 = capture(model, ds, t_all, sl, src_conv, "conv2")
        Xtr = np.vstack([box_pixels(A2, "1andes"),
                         box_pixels(A2, "5south_ocn"), box_pixels(A2, "6se_asia")])
        ytr = np.r_[np.ones(box_pixels(A2, "1andes").shape[0]),
                    np.zeros(Xtr.shape[0] - box_pixels(A2, "1andes").shape[0])]
        Xev = np.vstack([box_pixels(A2, "3himalaya"),
                         box_pixels(A2, "7natlantic"), box_pixels(A2, "8npacific")])
        yev = np.r_[np.ones(box_pixels(A2, "3himalaya").shape[0]),
                    np.zeros(Xev.shape[0] - box_pixels(A2, "3himalaya").shape[0])]
        pr = RidgeProbe().fit(Xtr, 2 * ytr - 1)
        res3[name] = {"cross_region_auc": auc(pr.predict(Xev), yev)}
        # within-region check: STRATIFIED interleaved split of the train set
        rng_l = np.random.default_rng(1)
        order = rng_l.permutation(Xtr.shape[0])
        tr_i, ev_i = order[::2], order[1::2]
        pr2 = RidgeProbe().fit(Xtr[tr_i], 2 * ytr[tr_i] - 1)
        res3[name]["within_region_auc"] = auc(pr2.predict(Xtr[ev_i]), ytr[ev_i])
        # shuffled-label control: 5 shuffles, report mean +- spread of eval AUC
        sh = []
        for k in range(5):
            ysh = np.random.default_rng(k).permutation(ytr)
            pr3 = RidgeProbe().fit(Xtr, 2 * ysh - 1)
            sh.append(auc(pr3.predict(Xev), yev))
        res3[name]["shuffled_auc_mean"] = float(np.mean(sh))
        res3[name]["shuffled_auc_range"] = [float(np.min(sh)), float(np.max(sh))]
    cr = res3["real"]["cross_region_auc"]
    r3_verdict = "KILL" if cr < 0.65 else "PASS"
    h_r3 = {"verdict": r3_verdict, **res3,
            "geographic_only_flag": bool(cr < 0.65 and res3["real"]["within_region_auc"] > 0.8)}

    results = {"r2": h_r2, "r3": h_r3, "n_timesteps": int(t_all.size),
               "spatial_split": "train lon cols [0,32)+[64,96), eval rest"}
    utils.save_results(REPO / "results/probes_m3/metrics.json", results, CFG, CFG["seed"])
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
