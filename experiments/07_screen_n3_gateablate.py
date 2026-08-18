"""H-N3 screen (T5->T3): are the skip-gates causally needed, scale-selectively?

Intervention: flatten one gate level's alpha field to its spatial mean (per
sample) via ActivationPatch and measure the RMSE change globally and over the
paper's 8 hotspot boxes, per level.

KILL (pre-registered): flattening EVERY single level changes global RMSE by
< 2% AND hotspot RMSE by < 5% (gates causally inert at screening resolution).

Run: python experiments/07_screen_n3_gateablate.py   (CFG overrides config)
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
from src.interp.hooks import ActivationPatch
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_n3.yaml"))
GATES = ["attn2.Psi", "attn3.Psi", "attn4.Psi", "attn5.Psi"]
HOTSPOTS = {
    "1andes": (3, 21, 96, 113), "2scand": (45, 58, 0, 12),
    "3himalaya": (41, 54, 26, 44), "4newfound": (47, 58, 103, 119),
    "5south_ocn": (8, 17, 10, 25), "6se_asia": (33, 42, 32, 49),
    "7natlantic": (31, 44, 112, 124), "8npacific": (27, 47, 67, 87),
}


def flatten_gate(alpha):
    return torch.ones_like(alpha) * alpha.mean(dim=(-2, -1), keepdim=True)


def rmse_parts(pred, truth):
    se = ((pred - truth) ** 2).mean(axis=0)          # (64, 128) over channels
    glob = float(np.sqrt(se.mean()))
    hot = float(np.sqrt(np.mean([se[y1:y2, x1:x2].mean()
                                 for y1, y2, x1, x2 in HOTSPOTS.values()])))
    return glob, hot


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m3", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")

    sums = {k: {"glob": [], "hot": []} for k in ["base"] + GATES}
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        x = torch.from_numpy(g[sl])[None]
        with torch.no_grad():
            base = m3(x).numpy()[0]
        gl, ho = rmse_parts(base, o)
        sums["base"]["glob"].append(gl)
        sums["base"]["hot"].append(ho)
        for gate in GATES:
            with ActivationPatch(m3, gate, flatten_gate), torch.no_grad():
                pred = m3(x).numpy()[0]
            gl, ho = rmse_parts(pred, o)
            sums[gate]["glob"].append(gl)
            sums[gate]["hot"].append(ho)

    base_g = float(np.mean(sums["base"]["glob"]))
    base_h = float(np.mean(sums["base"]["hot"]))
    per_level = {}
    any_effect = False
    for gate in GATES:
        dg = float(np.mean(sums[gate]["glob"])) / base_g - 1.0
        dh = float(np.mean(sums[gate]["hot"])) / base_h - 1.0
        per_level[gate] = {"delta_global_rmse_frac": dg, "delta_hotspot_rmse_frac": dh}
        if dg >= 0.02 or dh >= 0.05:
            any_effect = True
    verdict = "PASS" if any_effect else "KILL"

    results = {
        "hypothesis": "H-N3", "verdict": verdict,
        "base_rmse": {"global": base_g, "hotspot": base_h},
        "per_level": per_level, "n_timesteps": int(t_idx.size),
        "intervention": "alpha -> spatial mean (per sample), one level at a time",
    }
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
