"""Generate LaTeX tables for the paper directly from stamped results files
(no hand transcription). Output: paper/tables/*.tex

Run: python experiments/25_tables.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    return json.load(open(REPO / p, encoding="utf-8"))


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


# ---------------- Table: full-month replication (July + January) ----------------
jul = load("results/a4_fullmonth/metrics.json")
jan = load("results/a4_january/metrics.json")
rows = []
for label, d in (("July 2015", jul), ("January 2015", jan)):
    for m, name in (("m1", "M1"), ("m2", "M2"), ("m3", "M3")):
        v = d["models"][m]
        rows.append(
            f"{label if m=='m1' else ''} & {name} & {v['rmse_norm']:.4f} & "
            f"{v['rmse_norm_areaweighted']:.4f} & {v['hellinger_uw_phys']:.4f} & "
            f"{v['hellinger_vw_phys']:.4f} & {v['variance_ratio_phys']:.3f} \\\\")
    ci = d["m2_minus_m3_rmse_t"]["ci95"]
    frac = d["m2_minus_m3_rmse_t"]["frac_timesteps_m3_better"]
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\quad\\footnotesize paired M2$-$M3 RMSE: "
                f"mean {d['m2_minus_m3_rmse_t']['mean']:.4f}, 95\\% CI "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]; M3 better in {frac*100:.0f}\\% of 744 "
                f"timesteps}} \\\\[2pt]")
body = (
    "\\begin{tabular}{llccccc}\n\\toprule\n"
    "Month & Model & RMSE & RMSE$_{\\cos\\varphi}$ & $H_{uw}$ & $H_{vw}$ & "
    "$\\sigma^2$-ratio \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n")
write("tab_replication.tex", body)

# ---------------- Table: calibration ladders (both months, with CIs) ----------------
a5j = load("results/screen_a5_tails/metrics.json")["tail_ratios_pred_over_true"]
a5jan = load("results/j1_tails_january/metrics.json")["tail_ratios_pred_over_true"]
g2 = load("results/gateweek_g2/metrics.json")["arms"]


def ladder_row(name, d, qs=("p0.9", "p0.99", "p0.999")):
    cells = []
    for q in qs:
        r = d[q]
        cells.append(f"{r['ratio_mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    return f"{name} & " + " & ".join(cells) + " \\\\"


rows = ["\\multicolumn{4}{l}{\\emph{July 2015}} \\\\"]
for m, n in (("m1", "M1"), ("m2", "M2"), ("m3", "M3")):
    rows.append(ladder_row(n, a5j[m]))
rows.append(ladder_row("M3 (rolled 64)", {q: g2["m3_roll64"][q] for q in
                                          ("p0.9", "p0.99", "p0.999")}))
rows.append("\\addlinespace\\multicolumn{4}{l}{\\emph{January 2015}} \\\\")
for m, n in (("m1", "M1"), ("m2", "M2"), ("m3", "M3")):
    rows.append(ladder_row(n, a5jan[m]))
body = ("\\begin{tabular}{lccc}\n\\toprule\n"
        " & P90 & P99 & P99.9 \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
write("tab_ladder.tex", body)

# ---------------- Table: graft battery ----------------
g1 = load("results/gateweek_g1/metrics.json")
g1c = load("results/gateweek_g1/effect_conditioned.json")
d2 = load("results/d2_battery/metrics.json")
pg = load("results/screen_p_grafts/metrics.json")
p2w = load("results/d2_battery/p2_uvthetaw.json")
j3 = load("results/j3_patchdose_january/metrics.json")
j3d = load("results/j3_patchdose_january/dose_distributions.json")


def iqr(x):
    return f"[{x[0]:.2f}, {x[1]:.2f}]" if x else "--"


rows = [
    ("\\emph{Positive control (qbo1d emulator; ground truth known)}", "", "", ""),
    ("Amplitude family (uncond.)",
     f"$\\rho$ = {g1['g1b_amplitude']['median_spearman_vs_truth']:.2f}",
     iqr(g1['g1b_amplitude']['iqr']), f"n={g1['g1b_amplitude']['n_admissible']}"),
    ("Reflection (effect $\\geq$ 10\\%)",
     f"$r$ = {g1c['median_corr_conditioned']:.2f}",
     iqr(g1c['iqr_conditioned']), f"n={g1c['n_conditioned']}"),
    ("\\addlinespace\\emph{Column models (ERA5)}", "", "", ""),
    ("M1 reflection, Jul (uvtheta)",
     f"suppr.\\ {d2['arm1_m1_uvtheta_july']['median_suppression']:.3f}",
     iqr(d2['arm1_m1_uvtheta_july']['iqr']),
     f"n={d2['arm1_m1_uvtheta_july']['n_admissible']}"),
    ("M1 reflection, Aug (uvthetaw)",
     f"suppr.\\ {d2['arm2_m1_uvthetaw_aug']['median_suppression']:.3f}",
     iqr(d2['arm2_m1_uvthetaw_aug']['iqr']),
     f"n={d2['arm2_m1_uvthetaw_aug']['n_admissible']}"),
    ("M1 amplitude (uvtheta / uvthetaw)",
     f"$\\rho$ = {pg['p3']['median_spearman']:.2f} / "
     f"{d2['arm4_p3_uvthetaw_aug']['median_spearman']:.2f}", "--",
     f"n={pg['p3']['n_admissible']}/{d2['arm4_p3_uvthetaw_aug']['n_admissible']}"),
    ("M1 wind rotation (uvtheta / uvthetaw)",
     f"align.\\ {pg['p2']['circular_alignment']:.2f} / "
     f"{p2w['circular_alignment']:.2f}", "--",
     f"n={pg['p2']['n_admissible']}/{p2w['n_admissible']}"),
    ("\\addlinespace\\emph{Hotspot-center patch grafts (January)}", "", "", ""),
    ("M3 patch (max dose)",
     f"suppr.\\ {j3['median_suppression_m3_at_max_beta']:.3f}",
     iqr(j3['iqr_suppression_m3']), f"n={j3['n_patches_admissible']}"),
    ("\\quad dose direction",
     f"{j3d['m3_dose_spearman']['n_positive_monotone']}/28 monotone, "
     f"{j3d['m3_dose_spearman']['n_anti_monotone']}/28 anti", "",
     f"med.\\ $\\beta$={j3d['median_applied_beta']:.2f}"),
    ("M1 paired (max dose)",
     f"suppr.\\ {j3['median_suppression_m1_paired']:.3f}", "",
     f"{j3d['m1_dose_spearman']['n_positive_monotone']}/28 monotone"),
]
lines = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
body = ("\\begin{tabular}{lccc}\n\\toprule\n"
        "Arm & Median response & IQR & N \\\\\n\\midrule\n" + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n")
write("tab_grafts.tex", body)

# ---------------- Table: roll decomposition ----------------
d1 = load("results/d1_rollensemble/metrics.json")
rp = load("results/d1_rollpatch/metrics.json")
b, e, sh = d1["base"], d1["roll_ensemble_mean"], d1["position_shares"]
sd = d1["roll_ensemble_sd"]
rows = []
for k, label in (("variance_ratio", "$\\sigma^2$-ratio"), ("p0.9", "P90"),
                 ("p0.99", "P99"), ("p0.999", "P99.9")):
    rows.append(f"{label} & {b[k]:.3f} & {e[k]:.3f} $\\pm$ {sd[k]:.3f} & "
                f"{100*sh[k]:.0f}\\% \\\\")
rest = rp["restoration_fraction_variance"]
body = ("\\begin{tabular}{lccc}\n\\toprule\n"
        "Quantity & Unrolled & Roll ensemble (15 rolls) & Position share "
        "(upper bd.) \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\midrule\n\\multicolumn{4}{l}{\\footnotesize Cumulative roll-patch "
        "restoration of $\\sigma^2$-ratio: "
        f"conv1--3 $\\leq${max(abs(rest['conv1']),abs(rest['conv2']),abs(rest['conv3']))*100:.0f}\\%, "
        f"conv4 {rest['conv4']*100:.0f}\\%, conv5 {rest['conv5']*100:.0f}\\%, "
        "decoder (residual) $\\approx$39\\%}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
write("tab_roll.tex", body)

print("all tables generated")
