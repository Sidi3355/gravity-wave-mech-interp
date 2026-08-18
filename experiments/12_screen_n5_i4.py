"""H-N5 (flagship) + H-I4 screens, shared targets and machinery.

Targets (pre-registered): in each of 5 boxes (1andes, 3himalaya, 5south_ocn,
7natlantic, 8npacific — orographic + storm-track mix per H-A2b), the column
with peak July-climatological |flux|; at 4 timesteps spread across July
=> 20 (target, timestep) pairs.

H-N5: influence map I(j,i) = sum over input channels of |d s / d x(c,j,i)|,
s = predicted flux energy (sum of squared outputs) at the target column.
Metrics on the ring 1 < d <= 10 cells (the 3x3 core is excluded — that much
locality is M2-equivalent): (a) anisotropy = sqrt of inertia-tensor
eigenvalue ratio; (b) major-axis azimuth vs column-mean wind azimuth
(mod 180; aligned if |delta| <= 30 deg; chance = 1/3).
KILL: median axis ratio < 1.3 OR aligned targets < 12/20.
(Advection-vs-ray-cone discrimination is Stage-D scope; screening
establishes anisotropy + directional organization.)

H-I4: occlude input outside a box of half-width r in {1,2,4,8,16} cells
around the target (replace with July climatology, normalized space); forward
M3; MSE at the target column vs full-input MSE. Saturation radius = smallest
r with MSE(r) <= 1.05 * MSE(full).
KILL: median saturation radius <= 1 (M3 effectively local).

Run: python experiments/12_screen_n5_i4.py   (CFG env var overrides config)
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
CFG = utils.load_config(os.environ.get("CFG", REPO / "configs" / "screen_n5_i4.yaml"))
BOXES = {"1andes": (3, 21, 96, 113), "3himalaya": (41, 54, 26, 44),
         "5south_ocn": (8, 17, 10, 25), "7natlantic": (31, 44, 112, 124),
         "8npacific": (27, 47, 67, 87)}
RADII = [1, 2, 4, 8, 16]


def ring_inertia(I, j0, i0, rmax=10):
    """Anisotropy of influence in the ring 1 < d <= rmax (lon-wrapped)."""
    js = np.arange(I.shape[0])
    iis = np.arange(I.shape[1])
    JJ, II = np.meshgrid(js, iis, indexing="ij")
    dj = JJ - j0
    di = (II - i0 + 64) % 128 - 64
    d = np.sqrt(dj ** 2 + di ** 2)
    mask = (d > 1.5) & (d <= rmax)
    w = I[mask]
    x, y = di[mask], dj[mask]
    w = w / w.sum()
    cxx = float((w * x * x).sum()); cyy = float((w * y * y).sum())
    cxy = float((w * x * y).sum())
    C = np.array([[cxx, cxy], [cxy, cyy]])
    evals, evecs = np.linalg.eigh(C)
    ratio = float(np.sqrt(evals[1] / max(evals[0], 1e-12)))
    major = evecs[:, 1]                       # (x=east, y=north)
    azim = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return ratio, azim


def main():
    utils.set_seed(CFG["seed"])
    ds = xr.open_dataset(CFG["source_file"])
    n = ds.sizes["time"]
    t_idx = np.unique(np.linspace(0, n - 1, CFG["n_timesteps"]).astype(int))
    src_conv = nz.detect_source_convention(ds)
    sl3 = feature_slice("m3", "uvtheta", "global")
    m3 = load_model("m3", "uvtheta", "global")

    # climatology (normalized space) + peak-|flux| column per box
    clim_t = np.unique(np.linspace(0, n - 1, 24).astype(int))
    fsum = None
    flux_clim = None
    for t in clim_t:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        fsum = g if fsum is None else fsum + g
        fm = np.abs(o).mean(axis=0)
        flux_clim = fm if flux_clim is None else flux_clim + fm
    clim = (fsum / clim_t.size)
    flux_clim /= clim_t.size

    targets = []
    for box, (y1, y2, x1, x2) in BOXES.items():
        sub = flux_clim[y1:y2, x1:x2]
        jj, ii = np.unravel_index(np.argmax(sub), sub.shape)
        targets.append((box, y1 + jj, x1 + ii))

    n5_rows, i4_rows = [], []
    for t in t_idx:
        g = ds["features"][t].values.astype(np.float32)
        o = ds["output"][t].values.astype(np.float32)
        if src_conv is not None:
            g = nz.convert_inputs_to_model(g, src_conv)
            o = nz.convert_outputs_to_model(o, src_conv)
        x_np = g[sl3]
        truth = o
        u_mean = g[3 + 90:3 + 122].mean(axis=0)
        v_mean = g[125 + 90:125 + 122].mean(axis=0)

        for box, j0, i0 in targets:
            # ---------- H-N5 influence map
            x = torch.from_numpy(x_np)[None].requires_grad_(True)
            y = m3(x)
            s = (y[0, :, j0, i0] ** 2).sum()
            grad = torch.autograd.grad(s, x)[0][0].abs().sum(dim=0).numpy()
            ratio, azim = ring_inertia(grad, j0, i0, rmax=CFG["ring_rmax"])
            wind_az = float(np.degrees(np.arctan2(v_mean[j0, i0], u_mean[j0, i0])) % 180.0)
            dalign = min(abs(azim - wind_az), 180.0 - abs(azim - wind_az))
            n5_rows.append({"box": box, "t": int(t), "axis_ratio": ratio,
                            "azimuth": azim, "wind_azimuth": wind_az,
                            "align_delta_deg": float(dalign)})

            # ---------- H-I4 occlusion curve
            with torch.no_grad():
                p_full = m3(torch.from_numpy(x_np)[None]).numpy()[0][:, j0, i0]
            mse_full = float(((p_full - truth[:, j0, i0]) ** 2).mean())
            curve = {}
            for r in RADII:
                occ = clim[sl3].copy()
                jlo, jhi = max(0, j0 - r), min(64, j0 + r + 1)
                icols = [(i0 + di) % 128 for di in range(-r, r + 1)]
                occ[:, jlo:jhi, icols] = x_np[:, jlo:jhi, icols]
                with torch.no_grad():
                    p_r = m3(torch.from_numpy(occ)[None]).numpy()[0][:, j0, i0]
                curve[r] = float(((p_r - truth[:, j0, i0]) ** 2).mean())
            sat = next((r for r in RADII if curve[r] <= 1.05 * mse_full), ">16")
            i4_rows.append({"box": box, "t": int(t), "mse_full": mse_full,
                            "mse_by_radius": curve, "saturation_radius": sat})
        print(f"t={t} targets done", flush=True)

    ratios = [r["axis_ratio"] for r in n5_rows]
    aligned = sum(1 for r in n5_rows if r["align_delta_deg"] <= 30.0)
    n5_verdict = ("KILL" if (float(np.median(ratios)) < 1.3
                             or aligned < CFG["align_min_count"]) else "PASS")
    sat_num = [r["saturation_radius"] for r in i4_rows]
    sat_vals = [(17 if s == ">16" else s) for s in sat_num]
    i4_verdict = "KILL" if float(np.median(sat_vals)) <= 1 else "PASS"

    n5 = {"hypothesis": "H-N5", "verdict": n5_verdict,
          "median_axis_ratio": float(np.median(ratios)),
          "aligned_within_30deg": f"{aligned}/{len(n5_rows)}",
          "chance_alignment": "1/3", "rows": n5_rows}
    i4 = {"hypothesis": "H-I4", "verdict": i4_verdict,
          "median_saturation_radius": float(np.median(sat_vals)),
          "saturation_radii": [str(s) for s in sat_num], "rows": i4_rows}
    utils.save_results(REPO / "results/screen_n5/metrics.json", n5, CFG, CFG["seed"])
    utils.save_results(REPO / "results/screen_i4/metrics.json", i4, CFG, CFG["seed"])
    print(json.dumps({"n5": {k: n5[k] for k in ("verdict", "median_axis_ratio",
                                                "aligned_within_30deg")},
                      "i4": {k: i4[k] for k in ("verdict", "median_saturation_radius",
                                                "saturation_radii")}}, indent=1))


if __name__ == "__main__":
    main()
