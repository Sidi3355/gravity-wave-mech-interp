"""H-I3 screen (T4->T3): are horizontal GRADIENTS the carriers of M2's gain?

Twin small-MLPs trained identically on July columns:
  A) center-only features (369)
  B) center + finite-difference gradients (E-W and N-S central differences of
     u, v, theta per level from the 3x3 stencil: +732 -> 1101)
3 seeds each; evaluated on time-disjoint July timesteps. Closure fraction =
(RMSE_A - RMSE_B) / (RMSE_M1 - RMSE_M2), all on the same eval columns
(released M1/M2 provide the reference gap).

KILL (pre-registered): mean closure < 0.30.

Run: python experiments/16_screen_i3.py   (CFG env var overrides config)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz
from src.data.neighborhoods import make_3x3
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_i3.yaml"))


def features_from_stencil(st):
    """st: (C, ny, nx, 3, 3) -> center (N, 369) and gradients (N, 732)."""
    c = st[:, :, :, 1, 1]
    dx = (st[:, :, :, 1, 2] - st[:, :, :, 1, 0]) / 2.0
    dy = (st[:, :, :, 2, 1] - st[:, :, :, 0, 1]) / 2.0
    def cols(a):
        return a.transpose(1, 2, 0).reshape(-1, a.shape[0])
    grads = np.concatenate([cols(dx)[:, 3:369], cols(dy)[:, 3:369]], axis=1)
    return cols(c), grads


class SmallMLP(nn.Module):
    def __init__(self, idim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(idim, 512), nn.LeakyReLU(),
            nn.Linear(512, 512), nn.LeakyReLU(),
            nn.Linear(512, 244))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, Y, seed, epochs, bs=512, lr=1e-3):
    utils.set_seed(seed)
    model = SmallMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ((model(Xt[idx]) - Yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    return model.eval()


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    all_t = np.unique(np.linspace(0, n - 1, 36).astype(int))
    t_train, t_eval = all_t[::3][:CFG["n_train_t"]], all_t[1::3][:CFG["n_eval_t"]]
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m1", "uvtheta", "global")
    m1 = load_model("m1", "uvtheta", "global")
    m2 = load_model("m2", "uvtheta", "global")

    def gather(t_list, n_cols):
        C, G, Y, ST = [], [], [], []
        for t in t_list:
            g = ds["features"][t].values.astype(np.float32)
            o = ds["output"][t].values.astype(np.float32)
            if src_conv is not None:
                g = nz.convert_inputs_to_model(g, src_conv)
                o = nz.convert_outputs_to_model(o, src_conv)
            st = make_3x3(g[sl])
            c, gr = features_from_stencil(st)
            y = o.transpose(1, 2, 0).reshape(-1, 244)
            idx = rng.choice(8192, n_cols, replace=False)
            C.append(c[idx]); G.append(gr[idx]); Y.append(y[idx])
            ST.append(st.transpose(1, 2, 0, 3, 4).reshape(-1, 369, 3, 3)[idx])
        return (np.concatenate(C), np.concatenate(G), np.concatenate(Y),
                np.concatenate(ST))

    Ctr, Gtr, Ytr, _ = gather(t_train, CFG["cols_train"])
    Cev, Gev, Yev, STev = gather(t_eval, CFG["cols_eval"])

    # reference gap from released models on the same eval columns
    with torch.no_grad():
        p1 = m1(torch.from_numpy(Cev)).numpy()
        p2 = []
        xe = torch.from_numpy(STev)
        for i in range(0, xe.shape[0], 1024):
            p2.append(m2(xe[i:i + 1024]).numpy())
        p2 = np.concatenate(p2)
    rmse_m1 = float(np.sqrt(((p1 - Yev) ** 2).mean()))
    rmse_m2 = float(np.sqrt(((p2 - Yev) ** 2).mean()))
    ref_gap = rmse_m1 - rmse_m2

    closures, details = [], []
    for seed in range(3):
        ma = train_mlp(Ctr, Ytr, seed, CFG["epochs"])
        mb = train_mlp(np.hstack([Ctr, Gtr]), Ytr, seed, CFG["epochs"])
        with torch.no_grad():
            ra = float(np.sqrt(((ma(torch.from_numpy(Cev)).numpy() - Yev) ** 2).mean()))
            rb = float(np.sqrt(((mb(torch.from_numpy(np.hstack([Cev, Gev]))).numpy()
                                 - Yev) ** 2).mean()))
        closures.append((ra - rb) / ref_gap)
        details.append({"seed": seed, "rmse_center": ra, "rmse_center_grad": rb})
        print(f"seed {seed}: center {ra:.4f} grad {rb:.4f} closure {closures[-1]:.3f}",
              flush=True)

    results = {"hypothesis": "H-I3",
               "verdict": "KILL" if float(np.mean(closures)) < 0.30 else "PASS",
               "mean_closure_of_m1_m2_gap": float(np.mean(closures)),
               "closures": [float(c) for c in closures],
               "reference": {"rmse_m1": rmse_m1, "rmse_m2": rmse_m2, "gap": ref_gap},
               "runs": details}
    utils.save_results(REPO / "results/screen_i3/metrics.json", results, CFG, CFG["seed"])
    print(json.dumps({k: results[k] for k in ("hypothesis", "verdict",
                                              "mean_closure_of_m1_m2_gap",
                                              "reference")}, indent=1))


if __name__ == "__main__":
    main()
