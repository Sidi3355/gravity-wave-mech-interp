"""D1-B: cumulative roll-patching localization of M3's position code.
Pre-registered RESEARCH_LOG 2026-08-19 03:15.

Run: python experiments/22_d1_rollpatch.py
"""

import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz
from src.interp.hooks import ActivationCapture, ActivationPatch
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
JUL = "C:/Users/sidi0/gwmi_data/era5_monthly/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling07.nc"
ROLL = 64
SITES = ["conv1", "conv2", "conv3", "conv4", "conv5"]
SITE_ROLL = {"conv1": 64, "conv2": 32, "conv3": 16, "conv4": 8, "conv5": 4}


def to_phys_maps(y):
    uw = y[:122] ** 3 * nz.MODEL_CONVENTION["uw"][0] + nz.MODEL_CONVENTION["uw"][1]
    vw = y[122:] ** 3 * nz.MODEL_CONVENTION["vw"][0] + nz.MODEL_CONVENTION["vw"][1]
    return np.concatenate([uw, vw], 0)


def main():
    utils.set_seed(0)
    ds = xr.open_dataset(JUL)
    n = ds.sizes["time"]
    t_eval = np.unique(np.linspace(0, n - 1, 24).astype(int))[1::2]
    src_conv = nz.detect_source_convention(ds)
    sl3 = feature_slice("m3", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")

    arms = {k: [] for k in ["base", "rolled"] + [f"patch_upto_{s}" for s in SITES]}
    truths = []
    for t in t_eval:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        truths.append(to_phys_maps(o))
        x = torch.from_numpy(g[sl3])
        xr_ = torch.roll(x, ROLL, dims=-1)

        # capture unrolled encoder activations, roll-align them
        with ActivationCapture(m3, SITES) as cap, torch.no_grad():
            base_pred = m3(x[None]).numpy()[0]
        aligned = {s: torch.roll(cap.acts[s], SITE_ROLL[s], dims=-1)
                   for s in SITES}
        arms["base"].append(to_phys_maps(base_pred))

        with torch.no_grad():
            p = m3(xr_[None]).numpy()[0]
        arms["rolled"].append(to_phys_maps(np.roll(p, -ROLL, axis=-1)))

        for k in range(1, len(SITES) + 1):
            patched_sites = SITES[:k]
            with ExitStack() as stack:
                for s in patched_sites:
                    stack.enter_context(ActivationPatch(
                        m3, s, (lambda a: (lambda out: a))(aligned[s])))
                with torch.no_grad():
                    p = m3(xr_[None]).numpy()[0]
            arms[f"patch_upto_{SITES[k-1]}"].append(
                to_phys_maps(np.roll(p, -ROLL, axis=-1)))
        print(f"t={t} done", flush=True)

    T = np.stack(truths)
    var_t = T.var()
    stats = {}
    for k, v in arms.items():
        P = np.stack(v)
        stats[k] = {"variance_ratio": float(P.var() / var_t),
                    "p999": float(np.quantile(np.abs(P), 0.999)
                                  / np.quantile(np.abs(T), 0.999))}
    vb, vr = stats["base"]["variance_ratio"], stats["rolled"]["variance_ratio"]
    rest = {s: float((stats[f"patch_upto_{s}"]["variance_ratio"] - vr) / (vb - vr))
            for s in SITES}
    # sanity endpoint: base prediction equality when all encoder sites patched
    sanity = abs(rest["conv5"] - 1.0) < 0.05

    results = {"experiment": "D1-B cumulative roll-patching", "roll": ROLL,
               "stats": stats, "restoration_fraction_variance": rest,
               "sanity_full_patch_restores": bool(sanity)}
    utils.save_results(REPO / "results/d1_rollpatch/metrics.json", results,
                       {"roll": ROLL}, seed=0)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
