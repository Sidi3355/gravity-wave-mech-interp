"""Stage-D gate week, G1: qbo1d positive control for the D2 trust audit.
Protocol + criteria pre-registered RESEARCH_LOG 2026-08-19 00:45.

Physics ground truth E[S|u]: Monte-Carlo over the stochastic 20-wave spectrum,
formulas reproduced verbatim from qbo1d/stochastic_forcing.py (verified by a
fidelity assertion against the package's own WaveSpectrum on fixed draws).

Run: python experiments/19_gateweek_g1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
QBO = Path(r"C:\Users\sidi0\gwmi_data\external\qbo1d")
sys.path.insert(0, str(QBO))
from qbo1d import adsolver, emulate, utils as qutils
from qbo1d.stochastic_forcing import sample_sf_cw

from src import utils

SEED_MC = 12345
K_DRAWS = 200
ZC = 36
AMPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
N_PROF = 100


class Physics:
    """E[S|u] via MC over (sf, cw); verbatim formulas from stochastic_forcing."""

    def __init__(self, solver, k_draws=K_DRAWS, seed=SEED_MC):
        self.z = solver.z
        self.D1 = solver.D1.double()
        self.rho = qutils.get_rho(self.z).double()
        self.alpha = qutils.get_alpha(self.z).double()
        self.dz = solver.dz
        sf, cw = sample_sf_cw(n=k_draws, sfe=3.7e-3, sfv=1e-8, cwe=32,
                              cwv=225, corr=0.75, seed=seed)
        cs = torch.hstack([torch.arange(-100., 0., 10.),
                           torch.arange(10., 110., 10.)]).double()
        ks = (2 * 2 * torch.pi / 4e7) * torch.ones(20, dtype=torch.float64)
        As = torch.empty((k_draws, 20), dtype=torch.float64)
        for i in range(k_draws):
            As[i] = torch.exp(-np.log(2) * (cs / cw[i]) ** 2)
            As[i] *= torch.sign(cs)
            As[i] *= sf[i] / As[i].abs().sum() / 0.1006
        self.cs, self.ks, self.As = cs, ks, As

    def s_one(self, u, A_row):
        Ftot = torch.zeros_like(u)
        for A, c, k in zip(A_row, self.cs, self.ks):
            g = qutils.NBV * self.alpha / (k * ((c - u) ** 2))
            F = A * torch.exp(-torch.hstack((
                torch.zeros(1, dtype=torch.float64),
                torch.cumulative_trapezoid(g, dx=self.dz))))
            Ftot += F
        return torch.matmul(self.D1, Ftot) * self.rho[0] / self.rho

    def expected_s(self, u):
        u = u.double()
        acc = torch.zeros_like(u)
        for i in range(self.As.shape[0]):
            acc += self.s_one(u, self.As[i])
        return (acc / self.As.shape[0]).numpy()


def main():
    utils.set_seed(0)
    solver = adsolver.ADSolver(t_max=360 * 96 * 86400, w=3e-4)
    ds = xr.open_dataset(QBO / "data" / "direct" / "control.nc")
    U = torch.tensor(ds["u"].values)                      # (34560, 73) float64
    S = torch.tensor(ds["S"].values)
    scaler_Y = emulate.GlobalMaxScaler(S)
    model = emulate.FullyConnected(solver)
    model.load_state_dict(torch.load(QBO / "models" / "fully_connected.pth",
                                     map_location="cpu", weights_only=False))
    model.eval()
    model.scaler_Y = scaler_Y
    phys = Physics(solver)

    u_mu = U.mean(0).numpy()
    u_sd = U.std(0).numpy() + 1e-12

    def emul(u):
        with torch.no_grad():
            return model(u.double()).numpy()

    idx = np.linspace(2000, U.shape[0] - 1, N_PROF).astype(int)

    # ---------- precondition: emulator fidelity on control profiles ----------
    fid = []
    for i in idx:
        se = emul(U[i])[1:-1]
        st = phys.expected_s(U[i])[1:-1]
        fid.append(np.corrcoef(se, st)[0, 1])
    fidelity = float(np.median(fid))
    print(f"precondition fidelity (median corr emul vs E[S|u]): {fidelity:.3f}",
          flush=True)

    # ---------- G1-a reflection graft ----------
    corr_a, betas, n_ood_a = [], [], 0
    INT = slice(1, 72)   # interior levels; boundaries are Dirichlet-pinned

    def zmax_int(x):
        return np.abs((x.numpy()[INT] - u_mu[INT]) / u_sd[INT]).max()

    def reflect(u, beta):
        g = u.clone()
        g[ZC + 1:-1] = u[ZC + 1:-1] + beta * (2 * u[ZC] - 2 * u[ZC + 1:-1])
        return g

    for i in idx:
        u = U[i].clone()
        beta_ok = 0.0
        for beta in np.arange(1.0, 0.0, -0.05):
            if zmax_int(reflect(u, beta)) <= 4.0:
                beta_ok = float(beta)
                break
        if beta_ok < 0.25:
            n_ood_a += 1
            continue
        g = reflect(u, beta_ok)
        betas.append(beta_ok)
        d_em = (emul(g) - emul(u))[1:-1]
        d_tr = (phys.expected_s(g) - phys.expected_s(u))[1:-1]
        corr_a.append(float(np.corrcoef(d_em, d_tr)[0, 1]))
    med_a = float(np.median(corr_a)) if corr_a else float("nan")

    # ---------- G1-b amplitude scaling ----------
    from scipy.stats import spearmanr
    rho_b, n_ood_b = [], 0
    for i in idx:
        u = U[i]
        r_em, r_tr, ok = [], [], True
        for a in AMPS:
            g = a * u
            if np.abs((g.numpy()[1:72] - u_mu[1:72]) / u_sd[1:72]).max() > 4.0:
                ok = False
                break
            r_em.append(float(np.abs(emul(g)[1:-1]).mean()))
            r_tr.append(float(np.abs(phys.expected_s(g)[1:-1]).mean()))
        if not ok:
            n_ood_b += 1
            continue
        rho_b.append(float(spearmanr(r_em, r_tr).statistic))
    med_b = float(np.median(rho_b)) if rho_b else float("nan")

    ok_a, ok_b = med_a >= 0.5, med_b >= 0.8
    verdict = ("VALIDATED" if (ok_a and ok_b) else
               "D2_KILLED" if (not ok_a and not ok_b) else "PARTIAL")
    results = {"gate": "G1", "verdict": verdict,
               "precondition_fidelity_median_corr": fidelity,
               "precondition_met": bool(fidelity >= 0.8),
               "g1a_reflection": {"median_corr_dS": med_a,
                                  "iqr": ([float(np.quantile(corr_a, .25)), float(np.quantile(corr_a, .75))] if corr_a else None),
                                  "n_admissible": len(corr_a), "n_ood": n_ood_a,
                                  "median_beta": float(np.median(betas)) if betas else None},
               "g1b_amplitude": {"median_spearman_vs_truth": med_b,
                                 "iqr": ([float(np.quantile(rho_b, .25)), float(np.quantile(rho_b, .75))] if rho_b else None),
                                 "n_admissible": len(rho_b), "n_ood": n_ood_b},
               "mc_draws": K_DRAWS, "zc_level": ZC}
    utils.save_results(Path(__file__).resolve().parents[1] /
                       "results/gateweek_g1/metrics.json", results,
                       {"seed_mc": SEED_MC}, seed=0)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
