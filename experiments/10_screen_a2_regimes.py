"""H-A2 screen (T6): do nonlocality gains concentrate in high-shear/frontal
regimes rather than uniformly?

Composites the month-mean per-column RMSE gaps (M1-M2, M2-M3; from
experiments/02 artifacts) against July-mean horizontal wind-gradient magnitude
|grad_h u,v| (computed here from sampled timesteps, column-mean over levels)
and against orography gradient |grad zs|.

Metric (pre-registered): concentration = (share of total M1-M2 gap carried by
the top-decile |grad| columns) / (their column share = 0.1).
KILL: concentration < 1.5 for the M2-M1 gap. The M3-M2 composite is reported
with the input-set caveat (M3 lacks zs; S6 note).

Run: python experiments/10_screen_a2_regimes.py   (CFG env var overrides)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz

REPO = Path(__file__).resolve().parents[1]
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_a2.yaml"))


def grad_mag(field2d):
    """|grad| on the lon-periodic 64x128 grid (index units)."""
    dy = np.gradient(field2d, axis=0)
    dx = (np.roll(field2d, -1, axis=1) - np.roll(field2d, 1, axis=1)) / 2.0
    return np.sqrt(dx ** 2 + dy ** 2)


def concentration(gap_map, regime_map, q=0.9):
    """Share of total positive gap carried by top-(1-q) regime columns / share."""
    thresh = np.quantile(regime_map, q)
    mask = regime_map >= thresh
    total = gap_map.sum()
    if total <= 0:
        return float("nan")
    return float(gap_map[mask].sum() / total / mask.mean())


def main():
    utils.set_seed(CFG["seed"])
    art = np.load(REPO / CFG["month_artifacts"])
    # gaps in mean-squared-error units so shares are additive
    mse = {m: art[f"{m}_rmse_col"] ** 2 for m in ("m1", "m2", "m3")}
    gap_12 = mse["m1"] - mse["m2"]           # (64, 128), >0 where M2 helps
    gap_23 = mse["m2"] - mse["m3"]

    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    gmaps = []
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
        gm = np.zeros((64, 128))
        for ch0 in (3, 125):                          # u block, v block
            block = g[ch0:ch0 + 122]
            gm += np.mean([grad_mag(block[k]) for k in range(0, 122, 8)], axis=0)
        gmaps.append(gm)
    shear = np.mean(gmaps, axis=0)
    zs = ds["features"][0].values[2]
    orog_grad = grad_mag(zs.astype(np.float64))

    res = {
        "hypothesis": "H-A2", "n_timesteps_for_shear": int(t_idx.size),
        "concentration_gap12_by_shear_top10": concentration(gap_12, shear),
        "concentration_gap23_by_shear_top10": concentration(gap_23, shear),
        "concentration_gap12_by_oroggrad_top10": concentration(gap_12, orog_grad),
        "concentration_gap23_by_oroggrad_top10": concentration(gap_23, orog_grad),
        "corr_gap12_shear": float(np.corrcoef(gap_12.ravel(), shear.ravel())[0, 1]),
        "corr_gap23_shear": float(np.corrcoef(gap_23.ravel(), shear.ravel())[0, 1]),
        "note_m3": "M3 comparisons carry the input-set caveat (no zs/lat/lon)",
    }
    res["verdict"] = ("KILL" if (not np.isfinite(res["concentration_gap12_by_shear_top10"])
                                 or res["concentration_gap12_by_shear_top10"] < 1.5)
                      else "PASS")
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", res, CFG, seed=CFG["seed"])
    arr = out_dir / "arrays"
    arr.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arr / "maps.npz", gap_12=gap_12, gap_23=gap_23,
                        shear=shear, orog_grad=orog_grad)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
