# HYPOTHESIS_TABLE.md — Stage B (generated 2026-08-18, zero filtering)

Status legend: GEN (generated) / SCREEN-PASS / SCREEN-KILL / DEFER / DEEP-DIVE.
Every entry: falsifiable statement; primary technique family (T1-T9) + cheapest
test; kill criterion at screening; payoff. Screening budget: <20 min compute,
<~150 new LOC, 3 seeds or 3 cases. Facts these build on (Stage A): M3's
"attention" is sigmoid skip-gating, not self-attention; M3 never receives
lat/lon/zs; M3's convs zero-pad longitude (physical fields are periodic);
published months are uvtheta-only; one released checkpoint per config.

## B-INFO — what the nonlocal context carries

**H-I1** [SCREEN-KILL 2026-08-18: octant spread 0.165 < 0.20; upstream hint logged] (T3+T4) M2's gain over M1 is carried disproportionately by the
UPSTREAM neighbors of the 3x3 stencil (w.r.t. local background wind).
- Cheapest: resample-ablate each of the 8 neighbor columns over held-out
  snapshots; composite skill drop by neighbor bearing relative to wind octant.
- Kill: ablation impact uniform across neighbors after wind conditioning.
- Payoff: information routing follows advection — mechanism, not just stencil.

**H-I2** [SCREEN-PASS 2026-08-18] (T3) M2's gain is not mere spatial smoothing: replacing all 8
neighbors with copies of the center column destroys most of the M1->M2 gap.
- Cheapest: forward pass with clamped neighborhoods (constructor exists).
- Kill: clamped M2 ~= full M2 (gain then IS local/smoothing — report as such).
- Payoff: separates "more context" from "averaged context".

**H-I3** [GEN] (T4->T3) The specific carriers are horizontal GRADIENTS (shear,
deformation) of u, v across the stencil, not raw neighbor values.
- Cheapest: small-MLP on center column + finite-difference gradient features
  vs center-only twin (1-week July subset, reduced width, 3 seeds); compare
  recovered fraction of the M1->M2 gap.
- Kill: gradient features close <30% of the gap.
- Payoff: names the physical quantity that nonlocality supplies (ties to
  frontogenesis/jet-imbalance GW sources).

**H-I4** [SCREEN-PASS 2026-08-18: median used-context radius 4 cells, up to >16] (T3) M3 causally uses context beyond the 3x3 ring: occluding
input outside radius r around a target column degrades skill for r >> 1.
- Cheapest: occlusion sweep r in {1,2,4,8,16} cells on snapshot subsets,
  skill-vs-r curve for hotspot vs quiet columns.
- Kill: skill saturates by r=1 (M3 is effectively local — strong negative
  worth reporting against the receptive-field narrative).
- Payoff: measures the physically USED nonlocality scale (vs theoretical RF).

## B-REPR — what is represented

**H-R1** [SCREEN-KILL 2026-08-18: negative selectivity at all depths] (T1) Derived stability quantities (N^2, Richardson number)
become linearly decodable with increasing selectivity across M1/M2 depth.
- Cheapest: linear probes per layer on ~50k columns; controls = random-init
  net + probes on raw inputs (N^2, Ri are nonlinear in theta, u profiles).
- Kill: selectivity (real minus controls) ~ 0 or non-monotone noise.
- Payoff: networks compute textbook GW intermediates.

**H-R2** [SCREEN-KILL 2026-08-18: encoder discards orography; selectivity -0.39] (T1) M3 internally reconstructs the orography it never receives:
zs is decodable from M3 encoder activations well above the raw-input baseline.
- Cheapest: linear probe for zs at 2 encoder depths vs same probe on inputs;
  natural control: M1 (receives zs directly).
- Kill: probe ~= input baseline.
- Payoff: surrogate infers boundary forcing from flow state — plus a clean
  natural experiment M1-with-zs vs M3-without.

**H-R3** [SCREEN-PASS 2026-08-18: cross-region AUC 0.72 vs 0.40 control] (T1) Source-regime coding is abstract, not geographic:
orographic-vs-nonorographic separability transfers across disjoint regions.
- Cheapest: probe trained Andes-vs-WPac, tested Himalaya-vs-storm-track
  (paper's 8 boxes as labels); selectivity vs random-init control.
- Kill: within-region separability high but cross-region transfer ~ chance
  (geographic coding — still reportable, flips the interpretation).
- Payoff: regime abstraction is the trust-relevant property for deployment.

**H-R4** [GEN] (T2+T9) Mid-depth SAE features align with nameable meteorology:
a nontrivial fraction of alive features have geographically/physically
coherent activation maps (mountain wakes, jet exits, convective clusters).
- Cheapest: SAE at one M3 encoder depth + one M1 mid-layer, ~100k vectors,
  feature dashboards on snapshots; report dead-feature rate + recon/sparsity
  honestly; superposition scored via probe interference on the dictionary.
- Kill: features dense, dashboards incoherent at all tested sparsities.
- Payoff: first feature dictionary for a parameterization emulator.

**H-R5** [SCREEN-KILL 2026-08-18: sign selectivity ~0 everywhere] (T1) Flux SIGN and MAGNITUDE are computed separably: sign is
decodable earlier in depth than magnitude bins.
- Cheapest: per-layer probes for sign(uw) and |uw| quantile bins at 3 levels.
- Kill: identical decodability profiles.
- Payoff: computational decomposition of the prediction (filtering vs
  amplitude), guides where to patch in Stage D.

**H-R6** [SCREEN-PASS 2026-08-18: act1 collapse then late crystallization, act6=88% of full skill] (T8) The flux estimate crystallizes late: a linear "flux lens"
readout per layer shows a sharp skill jump in the last two MLP layers.
- Cheapest: ridge readouts per layer on ~50k columns.
- Kill: skill accrues gradually/linearly with depth.
- Payoff: locates the computation for patching + explains why last-2-layer
  transfer learning suffices (link to B-TL).

## B-ARCH — where architecture gains live

**H-A1** [SCREEN-KILL 2026-08-18: linear closes -47% of gap] (T6) A LINEAR model on the 3x3 stencil closes >=50% of the
M1->M2 gap (the first nonlocality increment is largely linear context).
- Cheapest: closed-form ridge on flattened stencils, July subsample.
- Kill: linear closes <20%.
- Payoff: calibrates how much "deep" the nonlocality story needs to be.

**H-A2** [SCREEN-KILL 2026-08-18: shear concentration 1.38 < 1.5; spawned H-A2b] (T6) Nonlocality gains concentrate in high-horizontal-shear /
frontal / storm-track regimes rather than uniformly.
- Cheapest: per-column M2-M1 and M3-M2 error-gap maps from the full-month
  run, composited by |grad(u)|, orography, hotspot type.
- Kill: gap maps spatially/regime-wise flat.
- Payoff: extends Wang-Yuval-O'Gorman "where nonlocality pays" to GW fluxes.

**H-A3** [GEN,screening-heavy] (T6) Attention gating is not the source of M3's
edge: at reduced scale, a plain UNet (gates removed/frozen to 1) matches the
Attention UNet's Hellinger skill.
- Cheapest available: retrain BOTH at 1/4 width on 2 July weeks, 3 seeds
  (borderline vs 20-min budget — run as the one sanctioned "slow screen").
- Kill: plain UNet clearly and consistently worse.
- Payoff: mechanism-vs-capacity attribution for the headline architecture.

**H-A4** [SCREEN-KILL 2026-08-18: high-rank deltas, projection moves 2.8%] (T3+T6) The nonlocal information enters M2 through a LOW-RANK
channel: activation deltas (full vs clamped neighborhoods) concentrate in few
PCs, and projecting them out restores M1-like predictions.
- Cheapest: PCA of deltas on ~20k columns; projection intervention.
- Kill: deltas high-rank; projection does nothing targeted.
- Payoff: a compact, analyzable "context signal" object for Stage D.

**H-A5** [GEN] (T6) The A4-observed metric flip is systematic: M3 trades
pointwise RMSE for distributional fidelity (variance/tail calibration).
- Cheapest: falls out of the full-month A4 rerun (variance ratios, P99
  tail ratios per model).
- Kill: full-month RMSE ordering restores M3<=M2 (inversion was noise).
- Payoff: sharpens what "better" means for surrogate evaluation; anchors the
  paper's Hellinger emphasis mechanistically.

## B-ATTN — what the skip-gates do (alpha fields, 4 scales)

**H-N1** [SCREEN-PASS 2026-08-18] (T5) The gates are not degenerate: alpha maps are spatially
structured and input-dependent (vs near-uniform saturation).
- Cheapest: alpha statistics on snapshots vs random-init control net.
- Kill: alpha ~ uniform/static. (Then the gain lives in the trunk — pivotal
  negative that reroutes B-ATTN effort; cheap prerequisite for H-N2..N5.)
- Payoff: gatekeeper result for all attention claims in this domain.

**H-N2** [SCREEN-KILL 2026-08-18: corr with instantaneous flux = -0.015] (T5) Fine-scale gates localize GW sources: alpha correlates
with hotspot masks/flux magnitude beyond a distance-decay null.
- Cheapest: correlation of finest-level alpha with |flux| maps + null.
- Kill: no excess over null.
- Payoff: gates as source-selectors — interpretable model behavior.

**H-N3** [SCREEN-PASS 2026-08-18] (T5->T3) Gate structure is causally needed scale-selectively:
flattening alpha to its spatial mean at one level degrades hotspot skill most
at the level matching source scales; keeping alpha only near hotspots retains
most skill (necessity + sufficiency pair).
- Cheapest screening: uniform-alpha ablation per level, skill drop per level.
- Kill: no level-selective degradation.
- Payoff: causal, not correlational, role for gating (what T5-only papers lack).

**H-N4** [GEN] (T5) Gates track advection in time: hourly alpha displacement
cross-correlates with background-wind advection.
- Cheapest: lagged cross-correlation on July hourlies at hotspots.
- Kill: no coherent displacement signal.
- Payoff: dynamic gating mirrors transport — connects to transient-skill claim.

**H-N5** [SCREEN-KILL 2026-08-18: influence isotropic (ratio 1.07), alignment 9/20] (T3+T4+physics) FLAGSHIP CANDIDATE: M3's causal influence map
(which input columns move the flux at a target column, via occlusion/Jacobian)
is anisotropic along ray-traced horizontal group-propagation displacements,
beating distance-decay AND pure-advection nulls.
- Cheapest: Jacobian/occlusion influence maps for ~20 hotspot targets vs the
  validated ray-tracer's footprints (A5 gate green) + two nulls.
- Kill: influence isotropic, or advection null explains it fully.
- Payoff: "the network learned lateral GW propagation" — the headline claim,
  causally grounded; exactly what Fourier-kernel prior art cannot see.

## B-TL — transfer learning (locus is last-2-layers BY CONSTRUCTION; only
mechanism-inside-the-head questions are non-trivial at Tier 1)

**H-L1** [GEN, asset-check first] (T6) The ERA5->IFS fine-tune implements a
per-level GAIN (diagonal rescaling), not a rotation: a fitted scalar-per-level
gain on the frozen base recovers most of the TL benefit.
- Cheapest: SVD/diagonality of layer6+output weight deltas (released TL
  checkpoints, iccs_coupling_checkpoints) + 1-param-per-level gain fit,
  evaluated on 1-2 OSF T42 IFS files (~9 GB).
- Kill: deltas rotate subspaces; gain fit recovers little.
- Payoff: makes the anchor's "rescaling to compensate unresolved variance"
  story mechanistically precise.

**H-L2** [GEN, asset-check first] (T8+T6) The frozen trunk is already
IFS-sufficient: a fresh linear head on frozen base activations matches the
released fine-tuned head on IFS data.
- Cheapest: ridge head on cached activations for an IFS subset.
- Kill: fresh head falls well short (fine-tune exploited nonlinear coupling).
- Payoff: representation-reuse claim for cross-resolution transfer.

## B-FAIL — failure structure

**H-F1** [MEASURED 2026-08-18: co-occurrence 38-53x chance -> shared data-limited floor] (T6) Worst-case failures co-occur across all three
architectures (top-1% error overlap >> chance) — data-limited, not
mechanism-limited.
- Cheapest: error-map rank correlations + top-percentile overlap, full month.
- Kill: n/a (both outcomes informative; screening records which).
- Payoff: diagnosis of whether architecture or data caps current skill.

**H-F2** [SCREEN-KILL 2026-08-18: nonorog/orog improvement ratio 1.06] (T4) M1's worst transients are nonorographic (storm-track/
convective), and that is specifically what M3 fixes.
- Cheapest: composite top-error events by hotspot type from month run.
- Kill: no type asymmetry in the M1->M3 improvement.
- Payoff: names WHAT nonlocality fixes, in meteorological terms.

**H-F3** [SCREEN-PASS 2026-08-18, evidence shared with H-A5] (T6) All models compress flux tails (P99 pred/true < 1), M3
least — and this, not mean error, drives the Hellinger ordering.
- Cheapest: tail ratios + Hellinger-vs-tail decomposition from month run.
- Kill: tails calibrated or unrelated to Hellinger ordering.
- Payoff: locates the distributional advantage mechanically.

## B-PHYS — controlled-input physics tests (T7; generators A5-gated)

**H-P1** [SCREEN-KILL 2026-08-18: zero suppression above grafted reversal] (T7) Critical-level filtering is learned: directional wind
reversal at z_c suppresses predicted flux above z_c relative to matched
no-reversal profiles.
- Cheapest: paired synthetic columns through M1 (then M2/M3 with uniform
  context), flux-profile ratio above/below z_c.
- Kill: no systematic suppression.
- Payoff: the central GW mechanism verified in the surrogate — or violated
  (a trust-critical negative with publication value either way; dual-outcome
  but with a directional physical prior, unlike H-F1).

**H-P2** [SCREEN-KILL 2026-08-18: drag does not rotate with wind, err 77deg] (T7) Orographic source response rotates with low-level wind:
rotating the forcing wind rotates the predicted surface-flux vector
accordingly (drag opposes surface wind) over an idealized ridge (M1/M2, which
receive zs).
- Cheapest: wind-rotation sweep at fixed ridge, flux-vector angle response.
- Kill: no coherent rotation.
- Payoff: physically consistent source encoding; feeds H-N5 target selection.

**H-P3** [SCREEN-KILL 2026-08-18: median Spearman 0.49] (T7) Response amplitude scales physically in-distribution and
saturates/breaks out-of-distribution: flux vs forcing amplitude is monotone
(superlinear) within training range, degrading beyond it.
- Cheapest: amplitude sweep on synthetic columns, 3 profile families.
- Kill: flat or erratic response in-distribution.
- Payoff: extrapolation envelope of the surrogate on controlled physics.

**H-P4** [SCREEN-PASS 2026-08-18, story refined: seam moves + global layout-dependence -> implicit position encoding] (T3, discovered in code) M3 has a longitude SEAM artifact: its
convs zero-pad longitude although the field is periodic, so skill dips near
lon 0/360, and rolling the input map moves the dip with the seam.
- Cheapest: error-vs-longitude from month run + rolled-input causal check.
- Kill: no seam-localized error structure.
- Payoff: concrete architectural artifact with deployment implications;
  free-standing engineering finding (M1/M2 are seam-free by construction).

---
Count: 29 generated (4 I, 6 R, 5 A, 5 N, 2 L, 3 F, 4 P). Technique coverage:
T1 (R1,R2,R3,R5), T2 (R4), T3 (I1,I2,I4,A4,N3,N5,P4), T4 (I1,I3,F2,N5),
T5 (N1-N4), T6 (A1,A2,A3,A5,L1,L2,F1,F3), T7 (P1-P3), T8 (R6,L2), T9 (in R4)
— 9/9 families represented, >=6 required. CRITIC pass: pending (next step,
distinct adversarial pass; objections will be appended, not edited in).

---
# Post-critic amendments (2026-08-18; critic: results/stageB_critic/critic_pass.md,
# LEAD responses: RESEARCH_LOG same date). Statuses + sharpened kill thresholds.
# Metric-space rule: RMSE/R2 in normalized space; distributional/tail metrics in
# physical space. Screening month = July 2015 ONLY; confirmations = January 2015.

STATUS CHANGES
- H-A3 -> DEFER (training study over screening budget; causal half via H-N3).
- H-A5 -> REWRITTEN [SCREEN-PASS 2026-08-18: variance 0.36/0.36/1.07, P99.9 tails 0.59/0.62/1.02, CIs disjoint]: "M2 under-disperses (variance ratio < 1) and M3 is
  better variance/tail-calibrated in the uvtheta setting; this, not mean
  error, drives the Hellinger ordering." Kill: |var-ratio difference| < 0.1
  or tail (P99) ratios statistically indistinguishable (bootstrap CI overlap).
  The uvthetaw RMSE inversion is demoted to an unresolved snapshot observation.
- H-L1 -> L1a only (weight-delta diagonality analysis) until TL checkpoints
  are verified; H-L2 -> DEFER pending assets.
- H-N4 -> DEFER unless H-N1 passes and an input-displacement null fits budget.
- H-R4 -> quantitative gate only at screening: dead-feature rate < 80%, recon
  R2 > 0.7 at L0 <= 40, and mean spatial autocorrelation of top-50 feature maps
  > 95th percentile of a random-dictionary null. Dashboards deferred to Stage D.
- H-F1 -> relabeled [MEASUREMENT] (runs with the F-family artifacts).

SHARPENED KILL THRESHOLDS (screening; all with 3 cases/seeds where applicable)
- H-I1: kill if max-vs-min neighbor ablation RMSE delta, composited by wind
  octant, differs by < 20% of the mean ablation effect (i.e., no directional
  structure) across 3 disjoint July weeks.
- H-I2: kill if clamped-neighborhood M2 recovers > 70% of the M1->M2 RMSE gap
  (gain mostly smoothing). C-3 diagnostic reported alongside: correlation of
  clamped-M2 vs M1 predictions.
- H-I3: kill if center+gradient model closes < 30% of the M1->M2 gap (3 seeds,
  1-week subset, fixed width). Runs only if H-A1 leaves >= 30% of gap unexplained.
- H-I4: kill if RMSE(r=1 occlusion) within 5% of RMSE(full input) at hotspot
  targets (M3 effectively local). Budget: 5 radii x 24 timesteps x 1 fwd = ~1 min.
- H-R1: kill if peak selectivity (real minus max(random-init, input-baseline))
  < 0.05 R2 at every depth for both N2 and Ri.
- H-R2: spatial split (disjoint 90-deg longitude sectors); kill if M3 probe R2
  beats input-baseline probe by < 0.1, or fails the time-shuffle control.
- H-R3: kill if cross-region orographic/nonorographic AUC < 0.65 (chance 0.5)
  while within-region AUC > 0.8 signals geographic-only coding (report which).
- H-R5: kill if sign-vs-magnitude decodability profiles differ by < 0.05
  (probe metric) at every depth after input-baseline subtraction.
- H-R6: kill if no layer-to-layer jump in flux-lens R2 exceeds 2x the median
  inter-layer increment (no crystallization point).
- H-A1: kill if stencil ridge closes < 20% of the M1->M2 RMSE gap (normalized
  space, July subsample >= 10^5 columns, closed form).
- H-A2: kill if top-decile |grad(u)| columns' share of the M2-M1 skill gap is
  < 1.5x their column share (no concentration) — same test for M3-M2 with the
  input-set caveat noted.
- H-A4: kill if top-5 PCs of (full-minus-clamped) act6 deltas explain < 50% of
  delta variance, or projecting them out moves M2 predictions < 20% toward M1.
- H-N1: kill if spatial std of alpha (finest gate, 24 July timesteps) < 2x the
  random-init control's, or temporal input-sensitivity (std across timesteps at
  fixed location) < 2x control.
- H-N2: kill if corr(alpha, |flux|) minus distance-decay null < 0.1 (Fisher-z,
  24 timesteps).
- H-N3: kill if flattening any single gate level changes global RMSE by < 2%
  and hotspot RMSE by < 5% (gates causally inert at screening resolution).
- H-N5: kill if influence-map anisotropy (major/minor axis ratio) < 1.3, or
  alignment with ray-cone azimuth no better than the advection null (paired
  comparison over >= 20 hotspot targets). Budget: Jacobian rows 0.56 s each.
- H-F2: kill if the M1->M3 top-1%-error improvement ratio for nonorographic
  vs orographic hotspot columns is < 1.5 (with M3-lacks-zs interpretation note).
- H-F3: kill if P99(pred)/P99(true) ordering does not match Hellinger ordering
  across models, or all ratios within [0.9, 1.1] (tails calibrated).
- H-P1: real-profile-grafted reversals primary; kill if flux above the grafted
  critical level is suppressed by < 30% relative to matched no-reversal grafts
  (median over >= 100 columns) or if the OOD envelope flags the graft
  (any |z| > 4 channel) — then inadmissible, not negative.
- H-P2: kill if predicted surface-flux vector rotation tracks wind rotation
  with circular correlation < 0.5 over a 360-deg sweep (M1/M2 only).
- H-P3: kill if response is non-monotone (Spearman < 0.8) inside the observed
  amplitude range on grafted columns.
- H-P4: kill if longitudinal RMSE profile shows no local excess >= 5% at the
  seam (cols 0-2/125-127) vs interior, AND the rolled-input twin moves nothing.
  C-4 twin: same test at zero-padded latitude edges.

NEW CONTROL ENTRIES
- C-1 [CONTROL] Epoch-pair robustness: any probe/SAE/feature finding reaching
  Stage D must replicate across the strato uvtheta epoch pairs (M1 88/100,
  M2 38/93) or carry a fragility caveat.
- C-2 [CONTROL] Area-weighted metrics: implemented in experiments/02; any
  ordering that flips under area weighting is reported as weighting-sensitive.
- C-3 [CONTROL] Clamp-identity diagnostic inside H-I2 (M2-clamped vs M1).
- C-4 [CONTROL] Pole-edge twin of H-P4 (zero-padded latitude boundary).
- C-5 [CONTROL] M1 vertical input-Jacobian map — calibration against
  Pahlavan-style ERF analysis; establishes T4 baseline before any T4 claims.

NEW (generated during screening, 2026-08-18)
- H-A2b [GEN, confirmation deferred to January/Stage-D data — born from H-A2's
  screening data, no double-use]: M3's advantage over M2 concentrates in
  high-shear (storm-track/jet) regimes (July concentration 2.92) and is
  absent-to-negative over orography (0.27); mechanism candidate: lateral
  propagation/advection of nonorographic GWs. Kill at confirmation:
  January shear concentration < 2.0 or orographic concentration > 0.5.
