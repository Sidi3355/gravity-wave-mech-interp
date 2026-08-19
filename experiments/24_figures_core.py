"""Core paper figures from stamped results (July-based; January panels added
after J1-J4). Print-first (light surface), CVD-validated palette, fixed
entity-color assignment across all figures:
  M1 #2a78d6 (blue) | M2 #eb6834 (orange) | M3 #1baf7a (aqua)
  M3-rolled = dashed aqua (same entity, altered state) | references gray.

Run: python experiments/24_figures_core.py  -> results/figures/*.{pdf,png}
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {"m1": "#2a78d6", "m2": "#eb6834", "m3": "#1baf7a",
     "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e5e4e0"}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
    "text.color": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
})


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load(p):
    return json.load(open(REPO / p, encoding="utf-8"))


# ---------------- F2: calibration ladder ----------------
def fig_ladder():
    months = [("July 2015", "results/screen_a5_tails/metrics.json", True),
              ("January 2015", "results/j1_tails_january/metrics.json", False)]
    g2 = load("results/gateweek_g2/metrics.json")["arms"]
    qs = ["p0.9", "p0.99", "p0.999"]
    xpos = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, (title, path, show_roll) in zip(axes, months):
        a5 = load(path)["tail_ratios_pred_over_true"]
        for off, m, label in ((-0.18, "m1", "M1 (column)"),
                              (0.0, "m2", "M2 (3x3)"), (0.18, "m3", "M3 (U-Net)")):
            y = [a5[m][q]["ratio_mean"] for q in qs]
            lo = [a5[m][q]["ratio_mean"] - a5[m][q]["ci95"][0] for q in qs]
            hi = [a5[m][q]["ci95"][1] - a5[m][q]["ratio_mean"] for q in qs]
            ax.errorbar(xpos + off, y, yerr=[lo, hi], fmt="o", ms=5,
                        color=C[m], capsize=2, lw=1.4, label=label)
        if show_roll:
            yr = [g2["m3_roll64"][q]["ratio_mean"] for q in qs]
            ax.plot(xpos + 0.18, yr, "s--", ms=4.5, color=C["m3"], alpha=0.55,
                    lw=1.2, label="M3, rolled input")
        ax.axhline(1.0, color=C["muted"], lw=1.0, ls=":")
        ax.set_xticks(xpos, ["P90", "P99", "P99.9"])
        ax.set_xlabel("|flux| quantile")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("predicted / true quantile ratio")
    axes[0].text(2.05, 1.03, "calibrated", fontsize=8, color=C["muted"])
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Tail calibration ladder (physical space)", fontsize=10, y=1.02)
    savefig(fig, "fig2_calibration_ladder")


# ---------------- F3: roll curve + regional ----------------
def fig_roll():
    d1 = load("results/d1_rollensemble/metrics.json")
    rolls = sorted(int(r) for r in d1["per_roll_variance_ratio"])
    vr = [d1["per_roll_variance_ratio"][str(r)] for r in rolls]
    base = d1["base"]["variance_ratio"]
    reg = d1["regional_variance_ratio"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    ax.plot(rolls, vr, "o-", color=C["m3"], lw=1.6, ms=4)
    ax.axhline(base, color=C["m3"], ls=":", lw=1.2)
    ax.text(2, base + 0.02, "M3, unrolled (1.10)", fontsize=8, color=C["m3"])
    ax.axhline(0.363, color=C["m2"], ls=":", lw=1.2)
    ax.text(2, 0.363 + 0.02, "M2 level (0.36)", fontsize=8, color=C["m2"])
    ax.set_xlabel("longitude roll of input map (grid cells)")
    ax.set_ylabel("flux variance ratio (pred/true)")
    ax.set_title("Variance calibration vs input roll")
    labels = ["hotspot\nbase", "hotspot\nrolled", "other\nbase", "other\nrolled"]
    vals = [reg["base_hotspot"], reg["rolled64_hotspot"],
            reg["base_elsewhere"], reg["rolled64_elsewhere"]]
    cols = [C["m3"], "#9fd9c4", C["m3"], "#9fd9c4"]
    b = ax2.bar(range(4), vals, color=cols, width=0.62, edgecolor="none")
    for r, v in zip(b, vals):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}",
                 ha="center", fontsize=8, color=C["ink"])
    ax2.axhline(1.0, color=C["muted"], lw=1.0, ls=":")
    ax2.set_xticks(range(4), labels, fontsize=7.5)
    ax2.set_title("Regional decomposition (roll 64)")
    ax2.set_ylabel("variance ratio")
    savefig(fig, "fig3_roll_position_prior")


# ---------------- F4: seam profile ----------------
def fig_seam():
    z = np.load(REPO / "results/screen_p4_seam/arrays/profiles.npz")
    lon = np.arange(128) * 2.8125
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m, label in (("m1", "M1"), ("m2", "M2"), ("m3", "M3")):
        ax.plot(lon, z[f"month_lon_{m}"], lw=1.5, color=C[m], label=label)
    jan = np.load(REPO / "results/a4_january/arrays/fullmonth_artifacts.npz")
    ax.plot(lon, jan["m3_rmse_col"].mean(axis=0), lw=1.2, ls="--", color=C["m3"],
            alpha=0.6, label="M3 (January)")
    for x0, x1 in ((0, 3 * 2.8125), (125 * 2.8125, 127 * 2.8125)):
        ax.axvspan(x0, x1, color=C["muted"], alpha=0.12, lw=0)
    ax.text(4, float(z["month_lon_m3"].max()) - 0.005, "conv padding seam",
            fontsize=8, color=C["muted"])
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("Per-longitude error: M3's boundary seam (July mean)")
    ax.legend(frameon=False, fontsize=8, ncols=3)
    savefig(fig, "fig4_seam_profile")


# ---------------- F5: roll-patch restoration ----------------
def fig_rollpatch():
    d = load("results/d1_rollpatch/metrics.json")
    rest = d["restoration_fraction_variance"]
    sites = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    vals = [rest[s] for s in sites] + [1.0]
    labels = sites + ["+ decoder\n(= unrolled)"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    cols = [C["m3"]] * 5 + [C["muted"]]
    b = ax.bar(range(6), vals, color=cols, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(6), labels, fontsize=8)
    ax.set_ylabel("variance-calibration restoration")
    ax.set_title("Where the position prior enters (cumulative roll-patching)")
    ax.set_ylim(-0.06, 1.1)
    savefig(fig, "fig5_rollpatch_localization")


# ---------------- F6: graft battery ----------------
def fig_grafts():
    g1c = load("results/gateweek_g1/effect_conditioned.json")
    d2 = load("results/d2_battery/metrics.json")
    pg = load("results/screen_p_grafts/metrics.json")
    rows = [
        ("qbo1d emulator\n(corr, effect-conditioned)", g1c["median_corr_conditioned"],
         g1c["iqr_conditioned"], C["muted"]),
        ("M1 reflection, July\n(uvtheta)", d2["arm1_m1_uvtheta_july"]["median_suppression"],
         d2["arm1_m1_uvtheta_july"]["iqr"], C["m1"]),
        ("M1 reflection, Aug\n(uvthetaw)", d2["arm2_m1_uvthetaw_aug"]["median_suppression"],
         d2["arm2_m1_uvthetaw_aug"]["iqr"], C["m1"]),
        ("M1 full reflection\n(screening)", pg["p1"]["median_suppression_above"],
         pg["p1"]["iqr"], C["m1"]),
        ("M3 patch graft\n(10x10)", d2["arm3_m3_patchgraft"]["median_suppression"],
         d2["arm3_m3_patchgraft"]["iqr"], C["m3"]),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for k, (label, med, iqr, col) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot(iqr, [y, y], lw=2.2, color=col, alpha=0.45, solid_capstyle="round")
        ax.plot(med, y, "o", ms=6, color=col)
        note = "corr with true response" if k == 0 else "flux suppression above graft"
        if k in (0, 1):
            ax.text(1.02, y, note, fontsize=7, color=C["muted"], va="center")
    ax.axvline(0, color=C["muted"], lw=1.0, ls=":")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    ax.set_xlabel("median response (dot) with IQR (bar)")
    ax.set_title("Critical-level graft responses: positive control vs climate models")
    savefig(fig, "fig6_graft_battery")


if __name__ == "__main__":
    fig_ladder()
    fig_roll()
    fig_seam()
    fig_rollpatch()
    fig_grafts()
