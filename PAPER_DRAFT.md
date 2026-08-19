# PAPER_DRAFT.md — working draft (Stage D complete; prose expansion pending)

Status: results-complete draft (all January confirmations integrated).
Every number cited here exists in a stamped results/*.json.

## Title candidates
1. What Do Neural Gravity-Wave Parameterizations Actually Learn? A
   Mechanistic Audit of a Released Model Hierarchy
2. Skill Without Physics: Position Codes, Variance Priors, and the Limits of
   Neural Gravity-Wave Surrogates
3. A Mechanistic Audit of Nonlocality in Neural Gravity-Wave Parameterizations

## Abstract (draft v2, post final-critic F-fixes)
Neural-network parameterizations of atmospheric gravity-wave (GW) momentum
fluxes increasingly inform climate-model development, and horizontal
nonlocality is credited for much of their skill. We present the first
mechanistic interpretability study of a parameterization emulator, auditing
the released single-column / 3x3-stencil / Attention-U-Net hierarchy of
Gupta et al. (2025) with a pre-registered hypothesis funnel: 30 hypotheses
generated, 24 screened (15 killed), 1 measured, 5 deferred with reasons, and
1 further killed at held-out confirmation. Three findings emerge. (1) The
U-Net's variance and extreme-tail calibration rides substantially on a
position-indexed prior: convolutional padding lets the network infer
absolute position, and rolling the input map — which changes no physics —
collapses its flux-variance calibration from 1.10 to 0.33. Roll ensembles
attribute up to 62% of variance calibration and 37% of extreme-tail (P99.9)
calibration to position (an upper bound: rolling also induces off-manifold
degradation), while bulk-quantile calibration is flow-driven; the
calibration split replicates on a second held-out month (variance ratios
0.42/0.35/1.00). Notably, pooled-histogram Hellinger distance — the
literature's headline distributional metric — is insensitive to this
collapse and can even be improved by physically absurd variance inflation;
quantile-ratio ladders discriminate. (2) The U-Net's causally used context
is long-range (median 4 grid cells, ~1200 km at T42, up to >16) but
isotropic and unaligned with ray-traced propagation directions. (3) Under
single-aspect grafts of real columns — a protocol whose amplitude family is
validated without conditions on a 1D testbed emulator that demonstrably
learned its physics (response tracking 0.94 there vs 0.49 here) — the
column models show no filtering response at typical columns, drag that
ignores wind rotation, and non-monotone amplitude response, consistent
across the released uvtheta and uvthetaw checkpoints (a confounded
checkpoint-plus-input-variant axis; no seed ensembles exist). At
strong-forcing hotspot centers, weak partial-dose responses appear
(medians 0.09-0.13 at median applied dose beta ~ 0.4, with 12/28 U-Net
patches anti-monotone) — far short of the near-complete flux removal linear
theory prescribes at a full critical level, though we do not derive
per-patch quantitative expectations. These surrogates achieve much of their
offline skill as regime pattern-matchers with geographic priors rather than
by encoding wave physics, with direct implications for out-of-distribution
trust and for how distributional skill should be measured.

## 1. Introduction
- GW parameterization problem; ML emulators lineage (Espinosa 2022; Hardiman
  2023; Gupta 2024/2025 anchor); nonlocality claims.
- Interpretability gap: prior art = kernel Fourier analysis (Pahlavan 2024),
  gradient ERFs (Pahlavan 2025), SHAP (Haslauer 2026) — nothing
  activation-level or causal on any subgrid parameterization emulator
  (verified novelty scan, 2026-08).
- Contribution list (funnel + 3 findings + methods points).

## 2. Related work
(a) ML GW parameterizations; (b) interpretability in weather/climate models
(GraphCast SAEs, Aurora probes — cite arXiv:2512.24440, 2511.07787,
2606.11657, 2605.23778); (c) mech-interp methods being adapted (probing
controls, patching, SAEs); (d) CNN position-encoding literature (Islam et
al. 2020) — newly relevant to scientific surrogates.

## 3. Models, data, and the released-artifact setting
- Tier-1 setting: released checkpoints (M1/M2/M3), released code, held-out
  2015 months; one checkpoint per config (limitation).
- REPRODUCIBILITY FINDING (standalone value): all WxC-Bench monthly files
  share ONE alternative normalization (different constants AND a 1x-sigma
  u/v convention despite metadata claiming 3x); feeding them to the released
  checkpoints as-is yields R^2 of -4 to -81. A single exact conversion,
  validated by a consecutive-hour test and a skill gate, fixes all months.
- A4 replication: full-July ordering M3<M2<M1 on RMSE and Hellinger, M3
  better than M2 in 744/744 paired timesteps.

## 4. The hypothesis funnel (methods transparency)
- Full table (30 hypotheses, pre-registered kill criteria, verdicts,
  one-line evidence) — from HYPOTHESIS_TABLE.md.
- Screening discipline: <20 min/test, thresholds pre-registered, critic
  passes, deviations logged.
- Multiple comparisons: every test run is reported in the funnel table
  (winners and losers alike); confirmation-month tests were limited to a
  pre-registered four-entry ledger (J1-J4) logged before the January file
  was opened; no correction applied within screening (verdicts are
  per-hypothesis pre-registered thresholds, not p-values), and the full
  family of outcomes is disclosed.

## 5. Finding 1: a position-indexed variance prior
- P4 seam + roll degradation (+15% global RMSE under roll).
- G2(ii): variance calibration 1.10 -> 0.33 under roll; D1-A decomposition:
  position shares 62% (variance), 37% (P99.9), 17% (P99), 2.5% (P90);
  symmetric roll-distance curve; hotspot-concentrated (1.29 -> 0.24).
- R2: encoder DISCARDS linearly-readable orography (selectivity -0.39) —
  the geography channel is positional, not reconstructed from flow.
- Mechanism statement: variance and extreme-tail calibration implemented
  substantially (up to 62% / 37%, upper bounds) as a position-indexed
  variance prior over climatological hotspots; bulk calibration (P90 share
  2.5%) is flow-driven.
- January confirmation (J1/J4): variance ratios 0.419/0.351/0.996 (vs July
  0.360/0.363/1.066); seam excess +5.05% (vs +5.1%). Depth-resolved
  localization (roll-patching, exact identity control): position code enters
  at the bottleneck (conv5 ~48%) and decoder (~39%); early encoder ~0.

## 6. Finding 2: what the nonlocality is (and is not)
- I4 occlusion radius (median 4, up to >16 cells) vs N5 isotropy (ratio
  1.07, alignment 9/20). Two-month regime pattern: shear concentration 2.92
  (July) / 1.77 (January), orographic 0.27 / -0.73 — directionally
  consistent, quantitatively variable; A2b's pre-registered confirmation
  threshold (>=2.0) failed, so this is reported descriptively, not as a
  confirmed claim.
- I2 clamp (-0.70 recovery), I1/I3/A1/A4 kills: the full multivariate
  neighborhood is consumed; no simple carrier.
- Gates: causally live at finest scale (+4.3%) but flux-uncorrelated
  (r=-0.015) and non-saturated — modulation without source selection.
- Methods point: Hellinger gameable by inflation (rescaled-M2 beats M3 on
  Hellinger while physically absurd); quantile ladders discriminate.

## 7. Finding 3: the physics-trust audit
- Protocol: single-aspect grafts on real columns, OOD admissibility (|z|<=4),
  partial-graft beta rule; positive control on qbo1d emulator (fidelity
  0.983; amplitude Spearman 0.94; reflection corr 0.75 where true effect
  >= 10%).
- M1: suppression -0.008 (July uvtheta, n=289), +0.043 (Aug uvthetaw,
  n=284); amplitude Spearman 0.49 (both variants); drag-wind alignment
  circular corr 0.22 (135 admissible rotations).
- Patch grafts at hotspot centers (January, 28 admissible patches; stamped
  dose distributions in j3_patchdose_january/dose_distributions.json):
  median suppression M3 0.125 / paired M1 0.092 at max applied dose (median
  applied beta 0.40; 20/28 ladders have only 2 distinct doses). Dose
  direction: M1 22/28 positive-monotone (6 anti), M3 16/28 (12 anti) —
  a weak, partial-dose, direction-mixed response, far short of the
  near-complete flux removal linear theory prescribes at a full critical
  level (per-patch quantitative expectations not derived; limitation).
  July preliminary agreed (0.146, n=10). Typical columns: no response.
- R1/R5: no linearly-readable N^2/Ri/sign intermediates; R6: flux
  crystallizes in last layers.
- F1: cross-architecture failure co-occurrence 38-53x chance (shared
  data-limited floor).
- Framing: absence of mechanism, not method insensitivity (positive
  control); scope: offline, T42, this model family.

## 8. Limitations (mandatory content per protocol)
- Scale gap: T42 coarse-grained fluxes; ray-displacement sub-cell at this
  resolution (why propagation-shaped nonlocality was not necessarily
  expected).
- Offline only; no online/coupled claims (Pahlavan offline-online gap).
- One released checkpoint per configuration; robustness axes = input
  variants + (for probes) epoch pairs; no seed ensembles exist.
- ERA5's own unresolved GW fraction bounds achievable "physics".
- Synthetic/graft external validity: OOD-policed but grafts break physical
  balance; positive control is 1D and column-local, covers the column-model
  arms (amplitude family unconditionally; reflection family only where the
  true effect is detectable, a conditioning that cannot be applied in ERA5
  where ground truth is unknown), and does NOT cover the U-Net patch arm.
- Roll-based position attribution is an UPPER bound: rolling both removes
  the position code and moves inputs off-manifold; the two effects are not
  separable with one released checkpoint.
- J3 dose-response is partial-dose (admissibility caps; median applied
  beta 0.40) with 2-point ladders in 20/28 patches.
- 2015 months served as the anchor's validation year (epoch selection).

## 9. Reproducibility statement
All experiments config-driven, seeded, stamped with config hashes; released
data/checkpoints only; conversion utilities + funnel logs public; make
reproduce-figures [TODO Stage E].

## Reviewer-attack list (living)
1. "Grafts are OOD, nulls are meaningless" -> OOD policing + qbo1d positive
   control (amplitude family unconditional: 0.94 there vs 0.49 here) +
   partial-graft rule; P2's 145/280 exclusions disclosed.
1b. "Your 'weak response' has no quantitative physical benchmark" -> honest:
   per-patch expectations not derived; claims scoped to partial-dose grafts
   and the qualitative prescription of near-complete removal at full
   critical levels; deriving ray-tracer-based per-patch expectations is
   stated future work.
2. "One checkpoint, overgeneralization" -> variant robustness, scoped claims.
3. "Position encoding is trivial/known" -> known in vision (Islam 2020),
   unmeasured in scientific surrogates; here it carries up to 62% of
   variance calibration (the discriminating ladder metric) — while the
   literature's Hellinger headline is INSENSITIVE to the collapse
   (0.0745 -> 0.0744 under roll), which is itself our methods finding.
4. "Hellinger critique unfair" -> demonstrated constructively (rescaled-M2).
5. "Roll tests confound geography with position" -> zs channel identical
   across files; M1/M2 roll-invariant/equivariant controls carry geography;
   only M3 degrades. Position share stated as an UPPER bound (roll also
   induces off-manifold degradation); corroborated independently by the
   symmetric roll-distance curve, bottleneck/decoder localization with an
   exact identity control, and hotspot concentration.
6. "The positive control does not cover the U-Net patch arm" -> correct and
   disclosed; the patch-arm result is reported as bounded/descriptive, and
   the load-bearing audit null is the unconditionally-validated amplitude
   family on column models.

## Figures plan (scripts to be numbered in experiments/)
F1 funnel diagram; F2 calibration ladders + roll collapse; F3 roll-distance
curve + regional decomposition; F4 seam profile + roll test; F5 influence
maps vs ray cones (null example); F6 graft battery (M1 vs qbo1d positive
control vs M3 patches); F7 gate structure + ablation gradient.
