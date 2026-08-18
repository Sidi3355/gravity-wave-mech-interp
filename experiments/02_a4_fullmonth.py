"""A4 full-month confirmation (July 2015, uvtheta trio) + shared artifacts.

Streams the WxC-Bench month file one timestep at a time through the released
M1/M2/M3 checkpoints, accumulating:
  - normalized-space squared error: total, per column (64x128), per channel
  - per-timestep RMSE series (transient skill) per model
  - per-hotspot per-timestep RMSE (the paper's 8 boxes)
  - physical-space histograms (fixed edges) of pred/true uw and vw for
    Hellinger, variance and tail (P99) diagnostics
  - truth variance accumulators for R^2
Checkpoints its state every N timesteps and resumes cleanly (master prompt:
nothing >2 h without checkpoint/resume; this run is ~1 h on this CPU).

Run:  python experiments/02_a4_fullmonth.py            (auto-resumes)
Outputs: results/a4_fullmonth/metrics.json (stamped) +
         results/a4_fullmonth/arrays/*.npz (gitignored, regenerable)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.data import normalization as nz
from src.data.neighborhoods import columns_3x3
from src.models.anchor_loader import feature_slice, load_model

REPO = Path(__file__).resolve().parents[1]
import os
CFG = utils.load_config(os.environ.get("A4_CFG", REPO / "configs" / "a4_fullmonth.yaml"))

# The paper's 8 evaluation boxes, hard-coded in the released dataloader
# (utils/dataloader_definition.py; indices into the 64x128 T42 grid).
HOTSPOTS = {
    "1andes": (3, 21, 96, 113), "2scand": (45, 58, 0, 12),
    "3himalaya": (41, 54, 26, 44), "4newfound": (47, 58, 103, 119),
    "5south_ocn": (8, 17, 10, 25), "6se_asia": (33, 42, 32, 49),
    "7natlantic": (31, 44, 112, 124), "8npacific": (27, 47, 67, 87),
}
MODELS = ("m1", "m2", "m3")


def parse_norm_constants(attr: str) -> dict:
    import re
    out = {}
    for var in ("uw", "vw"):
        m = re.search(var + r" = cuberoot\[\(" + var + r"-(-?[\d.eE-]+)\)/(-?[\d.eE-]+)\]", attr)
        out[var] = (float(m.group(1)), float(m.group(2)))
    return out


def to_physical(y: np.ndarray, consts: dict) -> np.ndarray:
    uw = y[:, :122] ** 3 * consts["uw"][1] + consts["uw"][0]
    vw = y[:, 122:] ** 3 * consts["vw"][1] + consts["vw"][0]
    return np.concatenate([uw, vw], axis=1)


class Accum:
    """Per-model accumulators, resumable via npz."""

    def __init__(self, n_time: int, edges: np.ndarray):
        nb = edges.size - 1
        self.sse = 0.0
        self.sse_aw = 0.0   # area-weighted (approx Gaussian weights ~ cos(lat))
        self.n = 0
        self.sse_col = np.zeros((64, 128))
        self.sse_chan = np.zeros(244)
        self.rmse_t = np.full(n_time, np.nan)
        self.rmse_hot = {k: np.full(n_time, np.nan) for k in HOTSPOTS}
        self.hist_pred = {v: np.zeros(nb, dtype=np.int64) for v in ("uw", "vw")}
        self.hist_true = {v: np.zeros(nb, dtype=np.int64) for v in ("uw", "vw")}
        self.sum_pred = 0.0
        self.sumsq_pred = 0.0
        self.sum_true = 0.0
        self.sumsq_true = 0.0

    def update(self, t: int, pred_cols: np.ndarray, true_cols: np.ndarray,
               pred_phys: np.ndarray, true_phys: np.ndarray, edges: np.ndarray,
               area_w: np.ndarray):
        err = pred_cols - true_cols                       # (8192, 244)
        se = err ** 2
        self.sse += float(se.sum())
        self.sse_aw += float((se.mean(axis=1) * area_w).sum())
        self.n += err.size
        self.sse_col += se.mean(axis=1).reshape(64, 128)
        self.sse_chan += se.mean(axis=0)
        self.rmse_t[t] = float(np.sqrt(se.mean()))
        se_map = se.mean(axis=1).reshape(64, 128)
        for k, (y1, y2, x1, x2) in HOTSPOTS.items():
            self.rmse_hot[k][t] = float(np.sqrt(se_map[y1:y2, x1:x2].mean()))
        lo, hi = edges[0], edges[-1]
        for name, arr_p, arr_t in (("uw", pred_phys[:, :122], true_phys[:, :122]),
                                   ("vw", pred_phys[:, 122:], true_phys[:, 122:])):
            self.hist_pred[name] += np.histogram(np.clip(arr_p, lo, hi), bins=edges)[0]
            self.hist_true[name] += np.histogram(np.clip(arr_t, lo, hi), bins=edges)[0]
        self.sum_pred += float(pred_phys.sum());  self.sumsq_pred += float((pred_phys ** 2).sum())
        self.sum_true += float(true_phys.sum());  self.sumsq_true += float((true_phys ** 2).sum())

    def state(self) -> dict:
        d = {"sse": self.sse, "sse_aw": self.sse_aw, "n": self.n, "sse_col": self.sse_col,
             "sse_chan": self.sse_chan, "rmse_t": self.rmse_t,
             "sum_pred": self.sum_pred, "sumsq_pred": self.sumsq_pred,
             "sum_true": self.sum_true, "sumsq_true": self.sumsq_true}
        for k in HOTSPOTS:
            d[f"rmse_hot_{k}"] = self.rmse_hot[k]
        for v in ("uw", "vw"):
            d[f"hist_pred_{v}"] = self.hist_pred[v]
            d[f"hist_true_{v}"] = self.hist_true[v]
        return d

    def load(self, z) -> None:
        self.sse = float(z["sse"]); self.sse_aw = float(z["sse_aw"]); self.n = int(z["n"])
        self.sse_col = z["sse_col"]; self.sse_chan = z["sse_chan"]
        self.rmse_t = z["rmse_t"]
        self.sum_pred = float(z["sum_pred"]); self.sumsq_pred = float(z["sumsq_pred"])
        self.sum_true = float(z["sum_true"]); self.sumsq_true = float(z["sumsq_true"])
        for k in HOTSPOTS:
            self.rmse_hot[k] = z[f"rmse_hot_{k}"]
        for v in ("uw", "vw"):
            self.hist_pred[v] = z[f"hist_pred_{v}"]
            self.hist_true[v] = z[f"hist_true_{v}"]


def hellinger_from_hist(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - np.sqrt(p * q).sum())


def batched(model, x: torch.Tensor, bs: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            outs.append(model(x[i:i + bs]).numpy())
    return np.concatenate(outs, 0)


def main():
    utils.set_seed(CFG["seed"])
    torch.set_num_threads(6)
    out_dir = REPO / CFG["out_dir"]
    arr_dir = REPO / CFG["arrays_dir"]
    arr_dir.mkdir(parents=True, exist_ok=True)
    state_path = arr_dir / "state.npz"

    ds = xr.open_dataset(CFG["month_file"])
    n_time = ds.sizes["time"]
    # Cross-file comparability gate (critic S6b, upgraded after the WxC-Bench
    # per-month scaling discovery): detect the file's convention and convert
    # every timestep to the model convention. Raises on unknown conventions.
    src_conv = nz.detect_source_convention(ds)
    print(f"source convention: {'model-native' if src_conv is None else 'WxC per-month (converting)'}",
          flush=True)
    consts = {"uw": (nz.MODEL_CONVENTION["uw"][1], nz.MODEL_CONVENTION["uw"][0]),
              "vw": (nz.MODEL_CONVENTION["vw"][1], nz.MODEL_CONVENTION["vw"][0])}
    edges = np.linspace(-CFG["hist_range"], CFG["hist_range"], CFG["hist_bins"] + 1)
    # Approx Gaussian-quadrature area weights ~ cos(lat), normalized to mean 1,
    # broadcast to the (8192,) column vector (lat-major reshape order).
    lat_deg = ds["lat"].values.astype(np.float64) * 90.0
    w_lat = np.cos(np.deg2rad(lat_deg))
    w_lat = w_lat / w_lat.mean()
    area_w = np.repeat(w_lat, 128)  # (64*128,), matches reshape(-1) order

    feat = CFG["features"]
    models = {m: load_model(m, feat, "global", root=Path(CFG["weights_root"]))
              for m in MODELS}
    slices = {m: feature_slice(m, feat, "global") for m in MODELS}

    acc = {m: Accum(n_time, edges) for m in MODELS}
    t0 = 0
    if state_path.exists():
        z = np.load(state_path, allow_pickle=False)
        t0 = int(z["next_t"])
        for m in MODELS:
            sub = {k[len(m) + 1:]: z[k] for k in z.files if k.startswith(m + "_")}
            acc[m].load(sub)
        print(f"resumed at t={t0}", flush=True)

    def save_state(next_t: int):
        payload = {"next_t": np.int64(next_t)}
        for m in MODELS:
            for k, v in acc[m].state().items():
                payload[f"{m}_{k}"] = v
        tmp = state_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(state_path)

    wall = time.time()
    for t in range(t0, n_time):
        g = ds["features"][t].values.astype(np.float32)      # (idim, 64, 128)
        truth = ds["output"][t].values.astype(np.float32)    # (244, 64, 128)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            truth = nz.convert_outputs_to_model(truth, src_conv)
        true_cols = truth.transpose(1, 2, 0).reshape(-1, 244)
        true_phys = to_physical(true_cols, consts)

        for m in MODELS:
            sl = slices[m]
            if m == "m1":
                x = torch.from_numpy(
                    g[sl].transpose(1, 2, 0).reshape(-1, sl.stop - sl.start))
                pred = batched(models[m], x, CFG["batch_m1"])
            elif m == "m2":
                x = torch.from_numpy(columns_3x3(g[sl]))
                pred = batched(models[m], x, CFG["batch_m2"])
            else:
                with torch.no_grad():
                    pred = models[m](torch.from_numpy(g[sl])[None]).numpy()[0]
                pred = pred.transpose(1, 2, 0).reshape(-1, 244)
            if not np.isfinite(pred).all():
                raise RuntimeError(f"non-finite prediction t={t} model={m}")
            acc[m].update(t, pred, true_cols, to_physical(pred, consts),
                          true_phys, edges, area_w)

        if (t + 1) % CFG["checkpoint_every"] == 0 or t == n_time - 1:
            save_state(t + 1)
            el = time.time() - wall
            done = t + 1 - t0
            print(f"t={t + 1}/{n_time}  {el / done:.1f}s/step  "
                  f"eta={(n_time - t - 1) * el / done / 60:.0f}min", flush=True)

    # ---- final metrics
    results = {"n_timesteps": int(n_time), "month": "2015-07",
               "features": feat, "hotspot_boxes": {k: list(v) for k, v in HOTSPOTS.items()},
               "models": {}}
    npz_out = {}
    for m in MODELS:
        a = acc[m]
        n_phys = a.n  # same element count in physical space
        var_true = a.sumsq_true / n_phys - (a.sum_true / n_phys) ** 2
        var_pred = a.sumsq_pred / n_phys - (a.sum_pred / n_phys) ** 2
        results["models"][m] = {
            "rmse_norm": float(np.sqrt(a.sse / a.n)),
            "rmse_norm_areaweighted": float(np.sqrt(a.sse_aw / (n_time * 8192))),
            "hellinger_uw_phys": hellinger_from_hist(a.hist_pred["uw"], a.hist_true["uw"]),
            "hellinger_vw_phys": hellinger_from_hist(a.hist_pred["vw"], a.hist_true["vw"]),
            "variance_ratio_phys": float(var_pred / var_true),
            "rmse_t_mean": float(np.nanmean(a.rmse_t)),
            "rmse_t_std": float(np.nanstd(a.rmse_t)),
        }
        npz_out[f"{m}_rmse_col"] = np.sqrt(a.sse_col / n_time)
        npz_out[f"{m}_rmse_chan"] = np.sqrt(a.sse_chan / n_time)
        npz_out[f"{m}_rmse_t"] = a.rmse_t
        for k in HOTSPOTS:
            npz_out[f"{m}_rmse_hot_{k}"] = a.rmse_hot[k]
        for v in ("uw", "vw"):
            npz_out[f"{m}_hist_pred_{v}"] = a.hist_pred[v]
            npz_out[f"{m}_hist_true_{v}"] = a.hist_true[v]
    npz_out["hist_edges"] = edges
    np.savez_compressed(arr_dir / "fullmonth_artifacts.npz", **npz_out)

    r = results["models"]
    results["ordering_rmse"] = bool(
        r["m3"]["rmse_norm"] <= r["m2"]["rmse_norm"] <= r["m1"]["rmse_norm"])
    results["ordering_hellinger_uw"] = bool(
        r["m3"]["hellinger_uw_phys"] <= r["m2"]["hellinger_uw_phys"]
        <= r["m1"]["hellinger_uw_phys"])
    # paired per-timestep comparison with a simple sign-flip bootstrap CI
    d = acc["m2"].rmse_t - acc["m3"].rmse_t   # >0 where M3 beats M2
    rng = np.random.default_rng(CFG["seed"])
    boots = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(2000)])
    results["m2_minus_m3_rmse_t"] = {
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "frac_timesteps_m3_better": float((d > 0).mean()),
    }
    utils.save_results(out_dir / "metrics.json", results, CFG, seed=CFG["seed"])
    print(json.dumps({k: v for k, v in results.items() if k != "hotspot_boxes"},
                     indent=1, default=str)[:2000])


if __name__ == "__main__":
    main()
