"""H-N1 screen (T5): are M3's skip-gates non-degenerate?

Computes, for the trained M3 (uvtheta) and a seeded random-init control of the
same architecture, per gate level (attn2..attn5 = finest..coarsest):
  spatial_std  — std of alpha over space, averaged over timesteps
  temporal_std — std of alpha over timesteps at fixed locations, averaged
  mean_alpha   — saturation diagnostic
PASS (pre-registered, post-critic): spatial_std ratio (real/control) >= 2 AND
temporal_std ratio >= 2 at the finest level. KILL otherwise.

Timesteps: cfg["n_timesteps"] spread evenly across the source file.
Run: python experiments/03_screen_n1_gates.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_n1.yaml"))
GATES = ["attn2.Psi", "attn3.Psi", "attn4.Psi", "attn5.Psi"]


def gate_stats(model, ds, sl, t_idx, src_conv):
    alphas = {g: [] for g in GATES}
    for t in t_idx:
        f = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            f = nz.convert_inputs_to_model(f, src_conv)
        x = torch.from_numpy(f[sl])[None]
        acts = hooks.capture_unet_maps(model, x, sites=GATES)
        for g in GATES:
            alphas[g].append(acts[g][0, 0].numpy())
    out = {}
    for g in GATES:
        a = np.stack(alphas[g])            # (T, H, W)
        out[g] = {
            "spatial_std": float(a.std(axis=(1, 2)).mean()),
            "temporal_std": float(a.std(axis=0).mean()),
            "mean_alpha": float(a.mean()),
            "shape": list(a.shape),
        }
    return out, alphas


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    sl = feature_slice("m3", "uvtheta", "global")
    src_conv = nz.detect_source_convention(ds)

    real = load_model("m3", "uvtheta", "global")
    torch.manual_seed(CFG["seed"])
    control = Attention_UNet(ch_in=366, ch_out=244, dropout=0.0).eval()

    stats_real, alphas_real = gate_stats(real, ds, sl, t_idx, src_conv)
    stats_ctrl, _ = gate_stats(control, ds, sl, t_idx, src_conv)

    finest = "attn2.Psi"
    r_spatial = stats_real[finest]["spatial_std"] / max(stats_ctrl[finest]["spatial_std"], 1e-9)
    r_temporal = stats_real[finest]["temporal_std"] / max(stats_ctrl[finest]["temporal_std"], 1e-9)
    verdict = "PASS" if (r_spatial >= 2.0 and r_temporal >= 2.0) else "KILL"

    results = {
        "hypothesis": "H-N1", "verdict": verdict,
        "ratio_spatial_finest": float(r_spatial),
        "ratio_temporal_finest": float(r_temporal),
        "n_timesteps_used": int(t_idx.size),
        "real": stats_real, "control": stats_ctrl,
    }
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    arr = out_dir / "arrays"
    arr.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arr / "alpha_sample.npz",
                        **{g.replace(".", "_"): np.stack(alphas_real[g][:4]) for g in GATES},
                        t_idx=t_idx[:4])
    print(json.dumps({k: results[k] for k in
                      ("hypothesis", "verdict", "ratio_spatial_finest",
                       "ratio_temporal_finest")}, indent=1))
    print(json.dumps(results["real"], indent=1))


if __name__ == "__main__":
    main()
