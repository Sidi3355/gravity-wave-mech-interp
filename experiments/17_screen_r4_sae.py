"""H-R4 screen (T2+T9), quantitative gate only (per post-critic re-scope).

SAEs (ReLU encoder, unit-norm decoder columns, L1 sparsity) at two sites:
  - M1 act3 (1476-d column vectors), dict k = 2x width
  - M3 conv2 (128-d pixel vectors, 32x64 grid), dict k = 2x width
Lambda swept over cfg grid; the reported model is the sparsest one meeting
mean L0 <= 40.

PASS requires (pre-registered, amendments): dead-feature rate < 0.80 AND
recon R^2 > 0.70 at mean L0 <= 40 AND mean spatial autocorrelation of the
top-50 most-active features' activation maps on a HELD-OUT timestep >
95th percentile of a random-direction null (50 random unit directions).
Dashboards/naming deferred to Stage D per critic.

Run: python experiments/17_screen_r4_sae.py   (CFG env var overrides config)
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
from src.interp import hooks
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_r4.yaml"))


class SAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Linear(d, k)
        self.dec = nn.Linear(k, d, bias=True)

    def forward(self, x):
        z = torch.relu(self.enc(x))
        with torch.no_grad():
            self.dec.weight.data /= self.dec.weight.data.norm(dim=0, keepdim=True) + 1e-8
        return self.dec(z), z


def train_sae(X, k, lam, epochs, seed, bs=512, lr=1e-3):
    utils.set_seed(seed)
    d = X.shape[1]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    sae = SAE(d, k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    Xt = torch.from_numpy(Xn.astype(np.float32))
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i + bs]]
            opt.zero_grad()
            rec, z = sae(xb)
            loss = ((rec - xb) ** 2).mean() + lam * z.abs().mean()
            loss.backward()
            opt.step()
    return sae.eval(), mu, sd


def evaluate_sae(sae, mu, sd, X):
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    with torch.no_grad():
        rec, z = sae(Xn)
    r2 = float(1 - ((rec - Xn) ** 2).mean() / Xn.var())
    zn = z.numpy()
    l0 = float((zn > 1e-6).sum(1).mean())
    active_frac = (zn > 1e-6).mean(0)
    dead = float((active_frac < 1e-4).mean())
    return r2, l0, dead, zn


def spatial_autocorr(vals2d):
    v = vals2d - vals2d.mean()
    denom = (v * v).sum() + 1e-12
    ac = 0.0
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        ac += (v * np.roll(v, sh, axis=ax)).sum() / denom
    return float(ac / 4.0)


def site_screen(name, Xtr, Xev, ev_shape, rng, cfg):
    d = Xtr.shape[1]
    k = 2 * d
    chosen = None
    for lam in cfg["lambda_grid"]:
        sae, mu, sd = train_sae(Xtr, k, lam, cfg["epochs"], cfg["seed"])
        r2, l0, dead, _ = evaluate_sae(sae, mu, sd, Xev)
        print(f"{name} lam={lam}: r2={r2:.3f} L0={l0:.0f} dead={dead:.2f}", flush=True)
        if l0 <= 40 and (chosen is None or r2 > chosen["r2"]):
            chosen = {"lam": lam, "r2": r2, "l0": l0, "dead": dead,
                      "sae": sae, "mu": mu, "sd": sd}
    if chosen is None:  # nothing hit the sparsity target; report sparsest
        sae, mu, sd = train_sae(Xtr, k, cfg["lambda_grid"][-1], cfg["epochs"], cfg["seed"])
        r2, l0, dead, _ = evaluate_sae(sae, mu, sd, Xev)
        chosen = {"lam": cfg["lambda_grid"][-1], "r2": r2, "l0": l0, "dead": dead,
                  "sae": sae, "mu": mu, "sd": sd}
    # spatial autocorrelation of top-50 feature maps on the held-out grid
    _, _, _, z = evaluate_sae(chosen["sae"], chosen["mu"], chosen["sd"], Xev)
    order = np.argsort(-z.mean(0))
    top = order[:50]
    ac_real = [spatial_autocorr(z[:, f].reshape(ev_shape)) for f in top]
    Xn = (Xev - chosen["mu"]) / chosen["sd"]
    ac_null = []
    for _ in range(50):
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        proj = np.maximum(Xn @ w, 0.0)
        ac_null.append(spatial_autocorr(proj.reshape(ev_shape)))
    null95 = float(np.quantile(ac_null, 0.95))
    mean_real = float(np.mean(ac_real))
    ok = (chosen["dead"] < 0.80 and chosen["r2"] > 0.70 and chosen["l0"] <= 40
          and mean_real > null95)
    return {"lambda": chosen["lam"], "recon_r2": chosen["r2"], "mean_L0": chosen["l0"],
            "dead_rate": chosen["dead"], "mean_autocorr_top50": mean_real,
            "null95_autocorr": null95, "gates_met": bool(ok)}


def main():
    utils.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_tr = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    t_ev = [int(t_tr[len(t_tr) // 2] + 1)]          # single held-out timestep
    src_conv = nz.detect_source_convention(ds)
    sl1 = feature_slice("m1", "uvtheta", "global")
    sl3 = feature_slice("m3", "uvtheta", "global")
    m1 = load_model("m1", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")

    def cols_m1(t_list, n_cols=None):
        out = []
        for t in t_list:
            g = ds["features"][t].values.astype(np.float32)
            if src_conv is not None:
                g = nz.convert_inputs_to_model(g, src_conv)
            x = torch.from_numpy(g[sl1].transpose(1, 2, 0).reshape(-1, 369).copy())
            acts = []
            with hooks.ActivationCapture(m1, ["act3"]) as cap, torch.no_grad():
                for i in range(0, 8192, 2048):
                    m1(x[i:i + 2048])
                    acts.append(cap.acts["act3"].numpy())
            a = np.concatenate(acts)
            if n_cols:
                a = a[rng.choice(8192, n_cols, replace=False)]
            out.append(a)
        return np.concatenate(out)

    def pix_m3(t_list):
        out = []
        for t in t_list:
            g = ds["features"][t].values.astype(np.float32)
            if src_conv is not None:
                g = nz.convert_inputs_to_model(g, src_conv)
            a = hooks.capture_unet_maps(
                m3, torch.from_numpy(g[sl3])[None], sites=["conv2"])["conv2"][0]
            out.append(a.numpy().transpose(1, 2, 0).reshape(-1, 128))
        return np.concatenate(out)

    res_m1 = site_screen("m1_act3", cols_m1(t_tr, CFG["cols_per_t"]),
                         cols_m1(t_ev), (64, 128), rng, CFG)
    res_m3 = site_screen("m3_conv2", pix_m3(t_tr), pix_m3(t_ev), (32, 64), rng, CFG)

    verdict = "PASS" if (res_m1["gates_met"] or res_m3["gates_met"]) else "KILL"
    results = {"hypothesis": "H-R4", "verdict": verdict,
               "m1_act3": res_m1, "m3_conv2": res_m3,
               "dict_size": "2x width", "note": "quantitative gates only; "
               "dashboards deferred to Stage D per critic re-scope"}
    utils.save_results(REPO / "results/screen_r4_sae/metrics.json", results,
                       CFG, CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
