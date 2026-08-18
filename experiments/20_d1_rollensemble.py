"""D1-A: roll-ensemble decomposition of M3's calibration (pre-registered
RESEARCH_LOG 2026-08-19 02:10). July eval timesteps as in G2.

Outputs: base vs roll-ensemble calibration ladder; position shares; regional
(hotspot vs elsewhere) variance ratios base vs rolled.
Run: python experiments/20_d1_rollensemble.py
"""

import json
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
JUL = "C:/Users/sidi0/gwmi_data/era5_monthly/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling07.nc"
ROLLS = list(range(8, 128, 8))
QS = [0.90, 0.99, 0.999]
HOTSPOTS = {"1andes": (3, 21, 96, 113), "2scand": (45, 58, 0, 12),
            "3himalaya": (41, 54, 26, 44), "4newfound": (47, 58, 103, 119),
            "5south_ocn": (8, 17, 10, 25), "6se_asia": (33, 42, 32, 49),
            "7natlantic": (31, 44, 112, 124), "8npacific": (27, 47, 67, 87)}


def to_phys_maps(y):
    """y: (244, 64, 128) normalized -> physical, same shape."""
    uw = y[:122] ** 3 * nz.MODEL_CONVENTION["uw"][0] + nz.MODEL_CONVENTION["uw"][1]
    vw = y[122:] ** 3 * nz.MODEL_CONVENTION["vw"][0] + nz.MODEL_CONVENTION["vw"][1]
    return np.concatenate([uw, vw], 0)


def main():
    utils.set_seed(0)
    ds = xr.open_dataset(JUL)
    n = ds.sizes["time"]
    t_all = np.unique(np.linspace(0, n - 1, 24).astype(int))
    t_eval = t_all[1::2]
    src_conv = nz.detect_source_convention(ds)
    sl3 = feature_slice("m3", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")

    hot_mask = np.zeros((64, 128), dtype=bool)
    for y1, y2, x1, x2 in HOTSPOTS.values():
        hot_mask[y1:y2, x1:x2] = True

    preds = {r: [] for r in [0] + ROLLS}
    truths = []
    for t in t_eval:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        truths.append(to_phys_maps(o))
        x = torch.from_numpy(g[sl3])
        with torch.no_grad():
            preds[0].append(to_phys_maps(m3(x[None]).numpy()[0]))
            for r in ROLLS:
                p = m3(torch.roll(x, r, dims=-1)[None]).numpy()[0]
                preds[r].append(to_phys_maps(np.roll(p, -r, axis=-1)))
        print(f"t={t} done", flush=True)

    T = np.stack(truths)                                   # (12, 244, 64, 128)
    var_t = T.var()

    def ladder(P):
        out = {"variance_ratio": float(P.var() / var_t)}
        for q in QS:
            out[f"p{q}"] = float(np.quantile(np.abs(P), q) / np.quantile(np.abs(T), q))
        return out

    base = ladder(np.stack(preds[0]))
    rolled = {r: ladder(np.stack(preds[r])) for r in ROLLS}
    ens = {k: float(np.mean([rolled[r][k] for r in ROLLS])) for k in base}
    ens_sd = {k: float(np.std([rolled[r][k] for r in ROLLS])) for k in base}
    shares = {k: float((base[k] - ens[k]) / base[k]) for k in base}

    # regional conditioning (variance ratio in hotspot vs elsewhere columns)
    def var_ratio_region(P, mask):
        return float(P[:, :, mask].var() / T[:, :, mask].var())
    region = {
        "base_hotspot": var_ratio_region(np.stack(preds[0]), hot_mask),
        "base_elsewhere": var_ratio_region(np.stack(preds[0]), ~hot_mask),
        "rolled64_hotspot": var_ratio_region(np.stack(preds[64]), hot_mask),
        "rolled64_elsewhere": var_ratio_region(np.stack(preds[64]), ~hot_mask),
    }

    results = {"experiment": "D1-A roll-ensemble", "n_rolls": len(ROLLS),
               "base": base, "roll_ensemble_mean": ens, "roll_ensemble_sd": ens_sd,
               "position_shares": shares, "regional_variance_ratio": region,
               "per_roll_variance_ratio": {str(r): rolled[r]["variance_ratio"]
                                           for r in ROLLS}}
    utils.save_results(REPO / "results/d1_rollensemble/metrics.json", results,
                       {"rolls": ROLLS}, seed=0)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
