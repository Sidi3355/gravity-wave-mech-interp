"""C-5 control (T4 calibration): M1's vertical input-Jacobian structure.

Computes d(uw_hat at output level L)/d(input channel) for M1, averaged over a
sample of columns, for a few output levels. This is the single-column analogue
of Pahlavan-style effective-receptive-field analysis and calibrates our T4
tooling against that prior art BEFORE any T4-based claims: we should see
coherent vertical structure (e.g., local + shear-layer influence), and the
result seeds H-R1/H-R6 site selection.

Purely descriptive — no verdict. Output: mean |Jacobian| per (output level,
input channel block), saved for figures.

Run: python experiments/08_c5_jacobian.py   (CFG env var overrides config)
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
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "c5_jacobian.yaml"))
BLOCKS = {"scalars": (0, 3), "u": (3, 125), "v": (125, 247), "theta": (247, 369)}


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    rng = np.random.default_rng(CFG["seed"])
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m1", "uvtheta", "global")
    m1 = load_model("m1", "uvtheta", "global")

    cols = []
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        c = g[sl].transpose(1, 2, 0).reshape(-1, 369)
        cols.append(c[rng.choice(8192, CFG["cols_per_timestep"], replace=False)])
    X = torch.from_numpy(np.concatenate(cols, 0)).requires_grad_(True)

    out_levels = CFG["output_levels"]  # indices into uw block (0..121)
    jac = {}
    y = m1(X)                                    # (N, 244)
    for L in out_levels:
        g_ = torch.autograd.grad(y[:, L].sum(), X, retain_graph=True)[0]
        jac[L] = g_.abs().mean(dim=0).numpy()    # (369,) mean |dy_L/dx|
    del y

    results = {"control": "C-5", "output_levels_uw": out_levels,
               "n_columns": int(X.shape[0]),
               "block_mean_abs_jacobian": {
                   str(L): {b: float(jac[L][s:e].mean()) for b, (s, e) in BLOCKS.items()}
                   for L in out_levels},
               "self_level_vs_block_mean": {
                   str(L): {"u_at_same_level": float(jac[L][3 + L]),
                            "u_block_mean": float(jac[L][3:125].mean())}
                   for L in out_levels}}
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    arr = out_dir / "arrays"
    arr.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arr / "jacobian_profiles.npz",
                        **{f"absjac_L{L}": jac[L] for L in out_levels})
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
