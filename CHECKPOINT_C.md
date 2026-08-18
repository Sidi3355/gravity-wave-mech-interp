# CHECKPOINT_C — Screening complete; deep-dive selection (2026-08-18)

**Status: awaiting human review.** Per master prompt §6, Stage D begins after
you pick the deep-dive target(s) — or say "go with the recommendation".

## 1. The funnel at a glance

30 hypotheses generated (29 + H-A2b born from screening data). 25 resolved in
one day of screening on the released Tier-1 checkpoints against held-out July
2015 (744 timesteps for the shared artifacts; 12–24 timesteps per targeted
screen; every verdict pre-registered, thresholds in HYPOTHESIS_TABLE.md).

| Verdict | Count | Hypotheses (one-line evidence) |
|---|---|---|
| PASS | 9 | N1 gates structured (35x control); I2 context=information (clamp recovery −0.70); N3 gates causal, finest +4.3%; P4 seam artifact + global layout dependence; A5 variance/tail calibration ladder (CIs disjoint); F3 tails drive Hellinger (shared evidence w/ A5); I4 context radius ~4–16 cells; R3 regime code transfers (AUC 0.72 vs 0.40 control); R6 flux crystallizes late (act6 = 88% of skill) |
| KILL | 15 | A1 linear closes −47%; A2 shear concentration 1.38<1.5; F2 improvement uniform across hotspot types; N2 gates ⊥ flux (r=−0.015); I1 no directional ablation structure (0.165<0.20); A4 context high-rank, projection moves 2.8%; **N5 flagship: influence isotropic (1.07), alignment 9/20**; R1 negative probe selectivity everywhere; R2 encoder discards orography (−0.39); R5 no sign/magnitude separation; P1 no critical-level filtering (suppression −0.007); P2 drag ignores wind rotation (err 77°); P3 amplitude non-monotone (ρ=0.49); I3 gradients close 9.6%; R4 codes stay dense (L0≥106) |
| MEASURED | 2 | F1 failures co-occur 38–53x chance (shared data-limited floor); C-5 Jacobian calibration (geography-heavy M1; physical vertical structure) |
| DEFERRED | 5 | A3 (budget), N4 (conditional), A2b (needs January), L1a/L2 (asset verification) |

## 2. What the screening added up to (the emerging paper)

Five mutually consistent findings (screening-level evidence; causal
elements noted per item, full causal triangulation is Stage-D work):

1. **The nonlocality that matters is long-range, isotropic, and regime-
   indexed — not wave-propagation-shaped.** M3 causally uses context to
   ~1500+ km (I4) but its influence maps are isotropic (N5 kill) and its
   extra skill concentrates in high-shear storm-track regimes, avoiding
   orography (A2b — an observation pending January confirmation, not yet
   evidence). M2's 3×3 gain is genuine
   information (I2), but not gradients (I3), not directional (I1), not
   linear (A1), not low-rank (A4): the full multivariate neighborhood
   pattern is consumed.
2. **The U-Net's real distributional advantage is a variance/tail
   calibration mechanism.** Calibration ladder P90→P99.9: M1 0.07→0.59,
   M2 0.11→0.62, M3 0.28→1.02 (all CIs disjoint); month variance ratios
   0.36/0.36/1.07. This, not pointwise accuracy, drives the Hellinger
   ordering the anchor paper emphasizes (A5/F3).
3. **Implicit position encoding is a hidden geography channel.** M3 never
   receives lat/lon/zs, yet rolling the input map degrades it globally by
   15% and it carries an M3-specific longitude-seam artifact that moves with
   the padding boundary (P4). Its encoder *discards* linearly-readable
   orography (R2) while its mid-depth features carry a transferable
   orographic-regime code (R3). The network "knows where it is" partly via
   convolution boundary geometry — a deployment-relevant artifact and a
   confound for any "learned physics" narrative.
4. **Offline skill without causal physics.** Controlled grafts on real
   columns (OOD-policed): no critical-level filtering response (P1), drag
   that ignores wind rotation (P2), non-monotone amplitude response (P3).
   Nothing textbook is linearly readable inside (R1/R5); flux crystallizes
   only in the last layers (R6). These surrogates are regime pattern-matchers.
5. **All three architectures share a failure floor** (F1: top-1% errors
   co-occur at 38–53× chance) — consistent with an ERA5 data-limited ceiling.

## 3. LEAD ranking of deep-dive candidates

**D1 (recommended): The calibration mechanism (from A5/F3 + N3 + R6).**
Question: where and how does M3 implement variance/tail calibration that
M1/M2 lack — and can it be causally localized (gates? decoder scales?
last-layer statistics?) and transferred?
Why: strongest quantified positive; causal tools (gate/activation patching,
flux lens) already built and validated; direct link to the anchor's own headline metric; January 2015 available for confirmation.
Stage-D plan sketch: pre-register localization experiments (per-scale gate
flattening → tail response; decoder-stage lens on tail quantiles; last-layer
statistic matching between M2 and M3), 2 independent technique families
(T3 interventions + T8 lens), January held-out confirmation, epoch-pair
robustness (C-1).

**D2 (recommended): The physics-trust audit (from P1/P2/P3 + N5 + R1/R2).**
Question: do offline-skilled GW surrogates encode causal wave physics at all
— hardened into a rigorous, positive-controlled negative result.
Why: highest scientific stakes (trust in ML parameterizations); the missing
piece is a POSITIVE CONTROL, and we have one staged: the qbo1d testbed
(CPU-trivial, physics known) — run the same graft protocol on the released
Pahlavan CNN; if it passes where M1/M3 fail, the audit methodology is
validated and the negative becomes compelling. Plus epoch-pair robustness and
OOD-tightened grafts.
Stage-D plan sketch: pre-register graft familes + admissibility; qbo1d
positive control; M2/M3 uniform-context variants dropped (OOD); report as
trust audit with explicit scale-gap limitations.

**Fallback: The implicit-position-encoding thread (P4 + R2-roll + N2).**
Fully causal already (roll tests); novel for climate surrogates; engineering
value. Held back only because D1/D2 cover the two strongest narratives and
this thread's key results are already publishable at screening depth.

Recommendation (AMENDED after critic challenge, see section 4): keep D1 and
D2 as the two subjects, but resequenced and restructured:

**Gate week first (cheap, decides the plan):**
- G1: qbo1d positive control for D2 — run the graft protocol on the released
  Pahlavan CNN where the physics is known-learnable. If it FAILS the grafts
  too, the audit methodology is invalidated and D2 is killed (this is D2's
  own pre-registered kill criterion).
- G2: two D1 sanity gates: (i) triviality test — can a per-level output
  rescaling stitched onto M2 reproduce M3's calibration ladder? If yes, the
  "mechanism" is a scalar recalibration, reframe accordingly; (ii)
  calibration-under-roll — recompute A5's variance/tail ladder on rolled
  inputs. If M3's calibration collapses under roll, the advantage is
  layout-bound (padding-derived geographic variance climatology), which
  REFRAMES both D1 and the paper's headline.
**Then:** D2 primary (trust audit, positive-controlled), D1 secondary with
the position-encoding thread FUSED IN as a mandatory confound arm (it is not
a floating fallback — P4 showed M3 is uniquely layout-dependent, and D1's
calibration claim is confounded by it until tested).

January 2015 (scaling01) downloads at Stage-D start and serves: D1/D2
held-out confirmation, A2b, with a pre-registered multiplicity ledger (the
critic flagged January-comparison accumulation; every January test will be
listed before the file is opened).

## 4. CRITIC's challenge (verbatim summary; full text in
results/stageC_critic/critic_ranking.md)

Verdict: AMEND — keep D1+D2, resequence with a gate week, make D2 primary,
fuse position-encoding into D1 as a mandatory confound arm.
Top objections: (1) D1 had no position control although M3 is uniquely BOTH
layout-dependent (P4: +15% roll degradation) and variance-calibrated (A5),
on a single checkpoint — the "fallback" was actually D1's confound;
(2) D2 had no kill criterion for itself — the qbo1d positive control must be
a GATE, not garnish; P2's OOD exclusions (145/280 at the most diagnostic
angles) and P1's below-graft leakage already soften the negative;
(3) D1's only validated interventional handle (the gates) has a broad,
non-hotspot, flux-uncorrelated signature (N3/N2) — inconsistent with a tail-
calibration mechanism, risking a correlational-only Stage D below the rigor
bar. Also flagged: no M3 epoch-pair robustness control exists at Tier 1;
January multiplicity must be pre-registered; three over-claim wordings in
section 2 (now fixed).
Biggest risk: M3's calibration advantage is layout-bound — one Stage-D
result (calibration-under-roll) could gut findings 1 and 2 together; the
amended plan tests exactly this first.

LEAD response: AMENDMENTS ACCEPTED IN FULL. The gate week (G1, G2) is now
the first pre-registered Stage-D step; D2 primary / D1+position secondary;
section 2 wording fixed; January multiplicity ledger adopted.

## 5. Decision needed from you

Reply with one of:
- **"go with the recommendation"** → Stage D starts with the gate week,
  then D2 primary + D1(+position confound) secondary as amended above;
- **"deep-dive only D1"** / **"only D2"** / **"use the fallback"**;
- anything else you want changed (different targets, different priorities).
