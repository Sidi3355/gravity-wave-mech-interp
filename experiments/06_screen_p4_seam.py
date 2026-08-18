"""H-P4 screen (T3): does M3 have a longitude-seam artifact from zero-padded
convolutions on a periodic domain?  Plus C-4 (latitude-edge twin).

Part 1 (observational): per-longitude / per-latitude M3 RMSE profiles from the
full-month artifacts (experiments/02 output), compared with M1/M2 (seam-free
by construction: M1 is columnwise; M2's stencils lon-wrap).
Part 2 (causal): roll the input maps by half the domain (64 cells) so the
convolution seam lands at the date line instead of Greenwich; un-roll the
predictions; recompute the per-longitude error profile. If the excess error
moves with the roll, the seam is caused by the padding, not by geography.

KILL (pre-registered): no local excess >= 5% at seam columns (0-2, 125-127)
relative to interior AND the rolled profile shows no moved excess.

Run: python experiments/06_screen_p4_seam.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_p4.yaml"))
SEAM = [0, 1, 2, 125, 126, 127]
INTERIOR = list(range(10, 118))


def profile_excess(rmse_lon):
    seam = float(np.mean(rmse_lon[SEAM]))
    interior = float(np.mean(rmse_lon[INTERIOR]))
    return seam / interior - 1.0, seam, interior


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl = feature_slice("m3", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")
    roll = CFG["roll_cells"]

    # Part 1: month artifacts (all 744 timesteps, all three models)
    art = np.load(REPO / CFG["month_artifacts"])
    per_lon = {m: art[f"{m}_rmse_col"].mean(axis=0) for m in ("m1", "m2", "m3")}
    per_lat = {m: art[f"{m}_rmse_col"].mean(axis=1) for m in ("m1", "m2", "m3")}
    month_excess = {m: profile_excess(per_lon[m])[0] for m in per_lon}

    # Part 2: causal roll test on sampled timesteps
    sse_lon = {"base": np.zeros(128), "rolled": np.zeros(128)}
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        x = torch.from_numpy(g[sl])
        with torch.no_grad():
            p_base = m3(x[None]).numpy()[0]
            p_roll = m3(torch.roll(x, roll, dims=-1)[None]).numpy()[0]
        p_roll = np.roll(p_roll, -roll, axis=-1)          # back to earth frame
        se_base = ((p_base - o) ** 2).mean(axis=(0, 1))   # (128,)
        se_roll = ((p_roll - o) ** 2).mean(axis=(0, 1))
        sse_lon["base"] += se_base
        sse_lon["rolled"] += se_roll
    rmse_base = np.sqrt(sse_lon["base"] / t_idx.size)
    rmse_roll = np.sqrt(sse_lon["rolled"] / t_idx.size)
    exc_base, seam_b, int_b = profile_excess(rmse_base)
    # rolled seam sits at columns (SEAM + roll) mod 128 in earth frame
    seam_r_cols = [(c + roll) % 128 for c in SEAM]
    int_r_cols = [c for c in range(128)
                  if all(min((c - s) % 128, (s - c) % 128) > 7 for s in seam_r_cols)]
    exc_roll_at_moved = float(np.mean(rmse_roll[seam_r_cols])
                              / np.mean(rmse_roll[int_r_cols]) - 1.0)
    exc_roll_at_original = profile_excess(rmse_roll)[0]

    seam_moves = exc_roll_at_moved >= 0.05 and exc_roll_at_moved > exc_roll_at_original
    verdict = "KILL" if (month_excess["m3"] < 0.05 and not seam_moves) else "PASS"

    results = {
        "hypothesis": "H-P4", "verdict": verdict,
        "month_seam_excess": {m: float(v) for m, v in month_excess.items()},
        "roll_test": {
            "excess_base_at_seam": float(exc_base),
            "excess_rolled_at_moved_seam": exc_roll_at_moved,
            "excess_rolled_at_original_seam": float(exc_roll_at_original),
            "seam_moves_with_roll": bool(seam_moves),
            "roll_cells": roll, "n_timesteps": int(t_idx.size),
        },
        "c4_latitude_edges": {
            m: {"edge_rows_0_1": float(np.mean(per_lat[m][[0, 1]])),
                "edge_rows_62_63": float(np.mean(per_lat[m][[62, 63]])),
                "interior_rows": float(np.mean(per_lat[m][10:54]))}
            for m in per_lat},
    }
    out_dir = Path(CFG["out_dir"]) if os.path.isabs(str(CFG["out_dir"])) else REPO / CFG["out_dir"]
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    arr = out_dir / "arrays"
    arr.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arr / "profiles.npz", rmse_lon_base=rmse_base,
                        rmse_lon_rolled=rmse_roll,
                        **{f"month_lon_{m}": per_lon[m] for m in per_lon},
                        **{f"month_lat_{m}": per_lat[m] for m in per_lat})
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
