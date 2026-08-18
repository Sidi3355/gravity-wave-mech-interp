"""R-family probes on M1's depth (T1 + T8): H-R1, H-R5, H-R6.

Streaming per-layer design: for each site in {input, act1..act6}, iterate the
sampled July timesteps, capture that site only, fit closed-form ridge probes,
evaluate on time-disjoint eval timesteps, discard activations.

Controls (mandatory, per table amendments):
  - random-init M1 (same architecture, seed 0) through the same pipeline;
  - input-baseline = the "input" site itself (raw 369 features);
  - selectivity = metric(real layer) - max(metric(rand-init layer), metric(input)).

Targets:
  R1: N2_proxy(L) = (1/theta) * dtheta/dlevel at L; Ri_proxy(L) =
      N2_proxy / (du/dlevel^2 + dv/dlevel^2 + eps), L in {30, 60, 90}
      (per-level index-derivatives are the physical quantities up to a fixed
      per-level layer-thickness factor, which ridge R^2 is invariant to).
  R5: sign(uw_norm(L)) (accuracy via thresholded ridge score) and
      |uw_norm(L)| (R^2), same levels.
      PRE-REGISTERED OPERATIONALIZATION (logged before run): compute each
      profile's selectivity across depth; "sign computed earlier" = the first
      layer reaching 90% of max sign-selectivity is >= 2 layers earlier than
      the first layer reaching 90% of max magnitude-selectivity. KILL if not.
  R6: full 244-output ridge flux lens per layer; R^2 profile.
      KILL if no layer-to-layer jump exceeds 2x the median increment.
  R1 KILL: peak selectivity < 0.05 R^2 for both N2 and Ri at every depth.

Run: python experiments/13_probes_m1.py   (CFG env var overrides config)
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
from src.models.anchor.model_definition import ANN_CNN
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "probes_m1.yaml"))
SITES = ["input", "act1", "act2", "act3", "act4", "act5", "act6"]
LEVELS = [30, 60, 90]
EPS = 1e-12


def targets_from_inputs(cols, out_cols):
    """cols: (N, 369) model-convention inputs; out_cols: (N, 244) truths."""
    u = cols[:, 3:125]
    v = cols[:, 125:247]
    th = cols[:, 247:369]
    dth = np.gradient(th, axis=1)
    du = np.gradient(u, axis=1)
    dv = np.gradient(v, axis=1)
    t = {}
    for L in LEVELS:
        n2 = dth[:, L] / (th[:, L] + EPS)
        t[f"N2_L{L}"] = n2
        t[f"Ri_L{L}"] = n2 / (du[:, L] ** 2 + dv[:, L] ** 2 + 1e-6)
        t[f"sign_uw_L{L}"] = np.sign(out_cols[:, L])
        t[f"mag_uw_L{L}"] = np.abs(out_cols[:, L])
    t["flux_all"] = out_cols
    return t


class RidgeProbe:
    """Closed-form ridge with feature standardization, multi-target."""

    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, Y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        A = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1])
        self.W = np.linalg.solve(A, Xs.T @ (Y if Y.ndim > 1 else Y[:, None]))
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        return Xs @ self.W


def collect(model, site, ds, t_list, sl, src_conv, rng, n_cols):
    """Return (acts, targets) for one site over the given timesteps."""
    A, C, O = [], [], []
    for t in t_list:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        cols = g[sl].transpose(1, 2, 0).reshape(-1, 369)
        outs = o.transpose(1, 2, 0).reshape(-1, 244)
        idx = rng.choice(8192, n_cols, replace=False)
        x = torch.from_numpy(cols[idx].copy())
        if site == "input":
            A.append(cols[idx])
        else:
            with hooks.ActivationCapture(model, [site]) as cap, torch.no_grad():
                model(x)
                A.append(cap.acts[site].numpy())
        C.append(cols[idx])
        O.append(outs[idx])
    return np.concatenate(A), np.concatenate(C), np.concatenate(O)


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    all_t = np.unique(np.linspace(0, n - 1, 36).astype(int))
    t_train, t_eval = all_t[::3][:CFG["n_train_t"]], all_t[1::3][:CFG["n_eval_t"]]
    assert not set(t_train) & set(t_eval)
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m1", "uvtheta", "global")
    real = load_model("m1", "uvtheta", "global")
    torch.manual_seed(0)
    rand = ANN_CNN(idim=369, odim=244, hdim=4 * 369, dropout=0.0, stencil=1).eval()

    metrics = {}  # metrics[model_name][site][target] = score
    for name, model in (("real", real), ("randinit", rand)):
        metrics[name] = {}
        for site in SITES:
            if name == "randinit" and site == "input":
                continue  # identical to real "input"
            rng = np.random.default_rng(CFG["seed"])   # same columns every site
            Atr, Ctr, Otr = collect(model, site, ds, t_train, sl, src_conv,
                                    rng, CFG["cols_train"])
            rng2 = np.random.default_rng(CFG["seed"] + 1)
            Aev, Cev, Oev = collect(model, site, ds, t_eval, sl, src_conv,
                                    rng2, CFG["cols_eval"])
            ttr = targets_from_inputs(Ctr, Otr)
            tev = targets_from_inputs(Cev, Oev)
            m = {}
            for key in ttr:
                if key.startswith("sign"):
                    pr = RidgeProbe().fit(Atr, ttr[key])
                    m[key] = float((np.sign(pr.predict(Aev)[:, 0]) == tev[key]).mean())
                else:
                    Y = ttr[key]
                    pr = RidgeProbe().fit(Atr, Y)
                    pred = pr.predict(Aev)
                    Yev = tev[key] if tev[key].ndim > 1 else tev[key][:, None]
                    m[key] = float(1.0 - ((pred - Yev) ** 2).mean() / Yev.var())
            metrics[name][site] = m
            print(f"{name}/{site} done", flush=True)

    inp = metrics["real"]["input"]

    def selectivity(site, key):
        base = max(inp[key], metrics["randinit"][site][key]) if site != "input" else inp[key]
        return metrics["real"][site][key] - base

    # ---- R1 verdict
    r1_peak = {}
    for q in ("N2", "Ri"):
        r1_peak[q] = max(selectivity(s, f"{q}_L{L}")
                         for s in SITES[1:] for L in LEVELS)
    r1_verdict = "KILL" if all(v < 0.05 for v in r1_peak.values()) else "PASS"

    # ---- R5 verdict (pre-registered depth-of-90%-max comparison)
    def first90(profile):
        arr = np.array(profile)
        if arr.max() <= 0:
            return len(arr)
        return int(np.argmax(arr >= 0.9 * arr.max()))
    depth_gap = {}
    for L in LEVELS:
        sp = [selectivity(s, f"sign_uw_L{L}") for s in SITES[1:]]
        mp = [selectivity(s, f"mag_uw_L{L}") for s in SITES[1:]]
        depth_gap[L] = first90(mp) - first90(sp)
    r5_verdict = "PASS" if np.median(list(depth_gap.values())) >= 2 else "KILL"

    # ---- R6 verdict
    lens = [metrics["real"][s]["flux_all"] for s in SITES]
    incs = np.diff(lens)
    r6_verdict = ("PASS" if (incs.size and incs.max() > 2 * max(np.median(incs), 1e-9))
                  else "KILL")

    results = {
        "r1": {"verdict": r1_verdict, "peak_selectivity": r1_peak,
               "profiles": {f"{q}_L{L}": [round(selectivity(s, f"{q}_L{L}"), 4)
                                          for s in SITES[1:]]
                            for q in ("N2", "Ri") for L in LEVELS}},
        "r5": {"verdict": r5_verdict, "depth_gap_mag_minus_sign": depth_gap,
               "sign_profiles": {L: [round(selectivity(s, f"sign_uw_L{L}"), 4)
                                     for s in SITES[1:]] for L in LEVELS},
               "mag_profiles": {L: [round(selectivity(s, f"mag_uw_L{L}"), 4)
                                    for s in SITES[1:]] for L in LEVELS}},
        "r6": {"verdict": r6_verdict, "flux_lens_r2_by_site": dict(zip(SITES, map(float, lens))),
               "increments": [float(x) for x in incs]},
        "sites": SITES, "n_train_cols": CFG["n_train_t"] * CFG["cols_train"],
    }
    utils.save_results(REPO / "results/probes_m1/metrics.json", results, CFG, CFG["seed"])
    print(json.dumps({k: results[k]["verdict"] for k in ("r1", "r5", "r6")}, indent=1))
    print("flux lens:", [round(x, 3) for x in lens])


if __name__ == "__main__":
    main()
