# PAPER_DRAFT.md — working draft (Stage D in progress)

Status: results-complete draft (all January confirmations integrated).
Every number cited here exists in a stamped results/*.json.

## Title candidates
1. What Do Neural Gravity-Wave Parameterizations Actually Learn? A
   Mechanistic Audit of a Released Model Hierarchy
2. Skill Without Physics: Position Codes, Variance Priors, and the Limits of
   Neural Gravity-Wave Surrogates
3. A Mechanistic Audit of Nonlocality in Neural Gravity-Wave Parameterizations

## Abstract (draft v1)
Neural-network parameterizations of atmospheric gravity-wave (GW) momentum
fluxes increasingly inform climate-model development, and horizontal
nonlocality is credited for much of their skill. We present the first
mechanistic interpretability study of a parameterization emulator, auditing
the released single-column / 3x3-stencil / Attention-U-Net hierarchy of
Gupta et al. (2025) with a pre-registered hypothesis funnel (30 hypotheses;
25 screened, 15 killed) combining probing, activation patching, controlled
physics grafts, and representation analysis. Three findings emerge. (1) The
U-Net's celebrated distributional skill is largely a position-indexed
variance prior: convolutional padding lets the network infer absolute
position, and rolling the input map — which leaves physics unchanged —
collapses its flux-variance calibration from 1.10 to 0.33, with 62% of
variance calibration attributable to position — replicated on a second
held-out month (January variance ratios 0.42/0.35/1.00).
(2) Its genuine nonlocality is long-range (~1500+ km), isotropic, and
regime-indexed, not wave-propagation-shaped: causal influence maps show no
alignment with ray-traced propagation, and the paper's headline Hellinger
metric is shown to be gameable by naive variance inflation, unlike
quantile-ratio calibration ladders. (3) Under controlled single-aspect
grafts of real columns — a protocol validated on a 1D testbed emulator that
demonstrably learned its physics — the column models exhibit no
critical-level filtering response, no drag-wind alignment, and non-monotone
amplitude response, robust across checkpoints and input variants; at
strong-forcing hotspot columns both models show weak but dose-monotone
responses an order of magnitude below physical expectation — a bounded,
not absolute, negative. These
surrogates achieve their offline skill substantially as regime
pattern-matchers with geographic priors rather than by encoding wave
physics, with direct implications for out-of-distribution trust and for how
distributional skill should be measured.

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
- REPRODUCIBILITY FINDING (standalone value): WxC-Bench monthly files are
  per-month re-normalized and 1x-sigma-scaled despite identical metadata
  wording; feeding them to the released checkpoints as-is yields R^2 of -4
  to -81. Exact conversion derived and validated (consecutive-hour test).
- A4 replication: full-July ordering M3<M2<M1 on RMSE and Hellinger, M3
  better than M2 in 744/744 paired timesteps.

## 4. The hypothesis funnel (methods transparency)
- Full table (30 hypotheses, pre-registered kill criteria, verdicts,
  one-line evidence) — from HYPOTHESIS_TABLE.md.
- Screening discipline: <20 min/test, thresholds pre-registered, critic
  passes, deviations logged.

## 5. Finding 1: a position-indexed variance prior
- P4 seam + roll degradation (+15% global RMSE under roll).
- G2(ii): variance calibration 1.10 -> 0.33 under roll; D1-A decomposition:
  position shares 62% (variance), 37% (P99.9), 17% (P99), 2.5% (P90);
  symmetric roll-distance curve; hotspot-concentrated (1.29 -> 0.24).
- R2: encoder DISCARDS linearly-readable orography (selectivity -0.39) —
  the geography channel is positional, not reconstructed from flow.
- Mechanism statement: tail calibration implemented largely as a
  position-indexed variance prior over climatological hotspots.
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
- Patch grafts at hotspot centers (January, 28 admissible patches):
  dose-monotone suppression in BOTH models (M3 0.125, Spearman 0.90; paired
  M1 0.092, Spearman 1.00) — an order of magnitude below the physical
  expectation that a critical level largely eliminates upward flux. July
  preliminary agreed (0.146, n=10). Typical columns: no response.
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
  balance; positive control is 1D and column-local.
- 2015 months served as the anchor's validation year (epoch selection).

## 9. Reproducibility statement
All experiments config-driven, seeded, stamped with config hashes; released
data/checkpoints only; conversion utilities + funnel logs public; make
reproduce-figures [TODO Stage E].

## Reviewer-attack list (living)
1. "Grafts are OOD, nulls are meaningless" -> OOD policing + qbo1d positive
   control + partial-graft rule; P2's 145/280 exclusions disclosed.
2. "One checkpoint, overgeneralization" -> variant robustness, scoped claims.
3. "Position encoding is trivial/known" -> known in vision (Islam 2020),
   unmeasured in scientific surrogates; here it carries 62% of the headline
   distributional-skill metric — a quantified, consequential finding.
4. "Hellinger critique unfair" -> demonstrated constructively (rescaled-M2).
5. "Roll tests confound geography with position" -> zs channel identical
   across files; M1/M2 roll-invariant/equivariant controls carry geography;
   only M3 degrades.

## Figures plan (scripts to be numbered in experiments/)
F1 funnel diagram; F2 calibration ladders + roll collapse; F3 roll-distance
curve + regional decomposition; F4 seam profile + roll test; F5 influence
maps vs ray cones (null example); F6 graft battery (M1 vs qbo1d positive
control vs M3 patches); F7 gate structure + ablation gradient.
