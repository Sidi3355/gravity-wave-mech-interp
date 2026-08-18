"""A4 replication gate (Tier 1): evaluate the released M1/M2/M3 checkpoints on
the released held-out 2015 test snapshots and check the paper's skill ordering
M3 <= M2 <= M1 (RMSE in normalized space; Hellinger distance in physical flux
space). Tier-1 adaptation of the gate is logged in RESEARCH_LOG
[2026-08-18 18:55]. Caveat (logged): the released test files contain only 2
hourly snapshots (16,384 columns); transient-evolution metrics are not
testable here and are deferred to a full held-out month if a Stage-D claim
needs them.

Run:  python experiments/01_replication_a4.py
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.models.anchor_loader import feature_slice, load_model

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "a4_replication.yaml"


def parse_norm_constants(attr: str) -> dict:
    """Extract {uw: (mu, sigma), vw: (mu, sigma)} from the output long_name."""
    out = {}
    for var in ("uw", "vw"):
        m = re.search(var + r" = cuberoot\[\(" + var + r"-(-?[\d.eE-]+)\)/(-?[\d.eE-]+)\]", attr)
        out[var] = (float(m.group(1)), float(m.group(2)))
    return out


def to_physical(y: np.ndarray, consts: dict) -> np.ndarray:
    """Invert cube-root + mu/sigma scaling. y: (N, 244) normalized -> physical."""
    uw = y[:, :122] ** 3 * consts["uw"][1] + consts["uw"][0]
    vw = y[:, 122:] ** 3 * consts["vw"][1] + consts["vw"][0]
    return np.concatenate([uw, vw], axis=1)


def hellinger(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """H = 1 - sum(sqrt(p*q)) over a common histogram (paper's convention)."""
    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    q, _ = np.histogram(np.clip(y, lo, hi), bins=edges)
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - np.sqrt(p * q).sum())


def batched_forward(model, x: torch.Tensor, bs: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i : i + bs]).numpy())
    return np.concatenate(outs, axis=0)


def main():
    cfg = utils.load_config(CFG_PATH)
    utils.set_seed(cfg["seed"])
    t_start = time.time()

    ds1 = xr.open_dataset(cfg["test_file_1x1"])
    consts = parse_norm_constants(ds1["output"].attrs["long_name"])
    truth = ds1["output"].values.astype(np.float32)          # (T, 244, 64, 128)
    T = truth.shape[0]
    truth_cols = truth.transpose(0, 2, 3, 1).reshape(-1, 244)  # (T*8192, 244)
    truth_phys = to_physical(truth_cols, consts)
    feats1 = ds1["features"].values.astype(np.float32)       # (T, 491, 64, 128)
    ds1.close()

    results = {"models": {}, "norm_constants": {k: list(v) for k, v in consts.items()},
               "n_timesteps": int(T), "n_columns": int(truth_cols.shape[0])}

    for feat in cfg["feature_sets"]:
        for name in ("m1", "m2", "m3"):
            key = f"{name}_{feat}"
            t0 = time.time()
            model = load_model(name, feat, "global", root=Path(cfg["weights_root"]))
            sl = feature_slice(name, feat, "global")

            if name == "m1":
                x = torch.from_numpy(
                    feats1[:, sl].transpose(0, 2, 3, 1).reshape(-1, sl.stop - sl.start)
                )
                pred = batched_forward(model, x, cfg["batch_m1"])
            elif name == "m2":
                ds3 = xr.open_dataset(cfg["test_file_3x3"])
                preds = []
                for it in range(T):
                    f3 = ds3["features"][it, sl].values.astype(np.float32)  # (C,64,128,3,3)
                    x = torch.from_numpy(
                        f3.transpose(1, 2, 0, 3, 4).reshape(-1, sl.stop - sl.start, 3, 3)
                    )
                    preds.append(batched_forward(model, x, cfg["batch_m2"]))
                ds3.close()
                pred = np.concatenate(preds, axis=0)
            else:  # m3: global maps, UNet omits the 3 scalar channels
                x = torch.from_numpy(feats1[:, sl])           # (T, C, 64, 128)
                with torch.no_grad():
                    y = model(x).numpy()                       # (T, 244, 64, 128)
                pred = y.transpose(0, 2, 3, 1).reshape(-1, 244)

            assert np.isfinite(pred).all(), f"{key}: non-finite predictions"
            assert pred.shape == truth_cols.shape
            err = pred - truth_cols
            pred_phys = to_physical(pred, consts)
            res = {
                "rmse_norm": float(np.sqrt((err ** 2).mean())),
                "r2_norm": float(1.0 - (err ** 2).mean() / truth_cols.var()),
                "hellinger_uw_phys": hellinger(pred_phys[:, :122].ravel(),
                                               truth_phys[:, :122].ravel(),
                                               cfg["hellinger_bins"]),
                "hellinger_vw_phys": hellinger(pred_phys[:, 122:].ravel(),
                                               truth_phys[:, 122:].ravel(),
                                               cfg["hellinger_bins"]),
                "runtime_s": round(time.time() - t0, 1),
            }
            results["models"][key] = res
            print(f"{key}: rmse={res['rmse_norm']:.4f} r2={res['r2_norm']:.4f} "
                  f"H(uw)={res['hellinger_uw_phys']:.4f} H(vw)={res['hellinger_vw_phys']:.4f} "
                  f"[{res['runtime_s']}s]", flush=True)
            del model, pred, pred_phys, err

    for feat in cfg["feature_sets"]:
        r = {m: results["models"][f"{m}_{feat}"] for m in ("m1", "m2", "m3")}
        results[f"ordering_rmse_{feat}"] = bool(
            r["m3"]["rmse_norm"] <= r["m2"]["rmse_norm"] <= r["m1"]["rmse_norm"])
        results[f"ordering_hellinger_uw_{feat}"] = bool(
            r["m3"]["hellinger_uw_phys"] <= r["m2"]["hellinger_uw_phys"]
            <= r["m1"]["hellinger_uw_phys"])

    results["total_runtime_s"] = round(time.time() - t_start, 1)
    utils.save_results(Path(__file__).resolve().parents[1] / cfg["out"], results,
                       cfg, seed=cfg["seed"])
    print("orderings:", {k: v for k, v in results.items() if k.startswith("ordering")})


if __name__ == "__main__":
    main()
