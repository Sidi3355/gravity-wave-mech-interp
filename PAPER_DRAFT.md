# Skill Without Physics: A Mechanistic Audit of Neural Gravity-Wave Parameterizations

Working title (alternates: "What Do Neural Gravity-Wave Parameterizations
Actually Learn?"; "A Mechanistic Audit of Nonlocality in Neural GW
Parameterizations"). ICML-format full draft, v3 (post final-critic F-fixes;
prose complete). Every number is traceable to a stamped file under results/.

## Abstract

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
strong-forcing hotspot centers, weak partial-dose responses appear (medians
0.09-0.13 at median applied dose beta ~ 0.4, with 12/28 U-Net patches
anti-monotone) — far short of the near-complete flux removal linear theory
prescribes at a full critical level, though we do not derive per-patch
quantitative expectations. These surrogates achieve much of their offline
skill as regime pattern-matchers with geographic priors rather than by
encoding wave physics, with direct implications for out-of-distribution
trust and for how distributional skill should be measured.

## 1. Introduction

Atmospheric gravity waves transport momentum from their sources — mountains,
convection, jets and fronts — into the middle atmosphere, where their
breaking drives circulations that global climate models cannot resolve and
must parameterize. Hand-built GW parameterizations embody strong simplifying
assumptions (single-column propagation, steady-state launch spectra), and a
growing literature replaces or augments them with neural networks trained on
high-resolution data (Espinosa et al., 2022; Hardiman et al., 2023; Gupta et
al., 2024; 2025). A headline result of this line is that *horizontal
nonlocality* — giving the network a view of neighboring columns or the whole
domain — substantially improves offline skill, with the natural and widely
voiced hope that networks so equipped are learning genuinely nonlocal wave
physics such as lateral propagation.

Whether that hope is warranted is a question about *mechanism*, and the
tools of the emerging mechanistic-interpretability field — probing with
controls, activation patching, controlled input interventions,
representation analysis — are built to answer exactly such questions. Yet
they have barely touched scientific surrogate models: prior interpretability
work on GW emulators stops at Fourier analysis of convolutional kernels
(Pahlavan et al., 2024), gradient-based receptive fields (Pahlavan et al.,
2025), and SHAP attributions (Haslauer et al., 2026). Nothing
activation-level or interventional has been published for any subgrid
parameterization emulator (we verified this by a systematic novelty scan in
August 2026; Section 2).

We conduct such a study on the *actual released artifacts* of Gupta et al.
(2025): their model hierarchy M1 (single-column MLP), M2 (3x3-stencil
convolution + MLP), and M3 (Attention U-Net), their released checkpoints,
and held-out months of their evaluation year. Because interpretability
studies are vulnerable to cherry-picking, we organize the entire
investigation as a pre-registered hypothesis funnel: 30 falsifiable
hypotheses with kill criteria fixed before each test, adversarial critic
passes at every stage gate, and held-out-month confirmation for the
surviving claims. The funnel itself (Table 1) is part of the contribution:
15 of 24 screened hypotheses were killed, several of them the most
attractive stories available (including our own pre-registered flagship,
"the U-Net's influence pattern follows wave-propagation geometry" — it does
not), and the findings that survived were in several cases discovered *by*
the kills.

Contributions:
1. **A position-indexed variance prior** (Section 5.2). M3 infers absolute
   map position through convolutional zero-padding and uses it as a de facto
   geography channel: its distinguishing skill — variance and extreme-tail
   calibration of the flux distribution — collapses under longitude rolling
   of the input (variance ratio 1.10 -> 0.33), with up to 62% of variance
   calibration attributable to position (upper bound), localized causally to
   the network's bottleneck (~48%) and decoder (~39%), concentrated over
   climatological hotspots, and replicated on a second held-out month. A
   visible corollary is a longitude "seam" of elevated error at the map
   boundary (+5% in both months, M3 only).
2. **The character of the learned nonlocality** (Section 5.3). Causal
   occlusion shows M3 uses context to a median radius of 4 grid cells
   (~1200 km at T42; >16 cells for a fifth of targets), but its influence
   maps are isotropic (median axis ratio 1.07) and unaligned with ray-traced
   propagation directions; the 3x3 model's gain is carried by the full
   multivariate neighborhood pattern, not by gradients, single directions,
   linear readouts, or any low-rank channel we could identify.
3. **A positive-controlled physics-trust audit** (Section 5.4). Under
   single-aspect grafts of real columns, the models show no critical-level
   filtering response at typical columns, wind-rotation-blind drag, and
   non-monotone amplitude scaling — while a testbed emulator that provably
   learned its (1D) physics passes the same battery. Hotspot centers show
   only weak, partial-dose, direction-mixed responses.
4. **Methods findings** with reach beyond this case study: pooled-histogram
   Hellinger distance is insensitive to — and gameable against — exactly the
   distributional failure it is used to detect (Section 5.5); and a
   reproducibility hazard in the public data stack (Section 3.2): the
   WxC-Bench monthly files are normalized incompatibly with the released
   checkpoints (feeding them as-is yields R^2 of -4 to -81), fixed by a
   single validated conversion.

We emphasize scope from the outset: this is an offline study of one released
model family at T42 resolution. It makes no claims about online/coupled
behavior, other architectures, or finer scales; Section 7 details these
limits and why some null results (e.g., propagation-shaped influence) may be
resolution-conditional.

## 2. Related Work

**ML parameterization of gravity waves.** Emulation of existing schemes
(Espinosa et al., 2022 — GRL 49(8); Hardiman et al., 2023 — AIES 2(4))
established feasibility and generalization to the QBO and CO2 response;
data-driven flux prediction from reanalysis (Gupta et al., 2024,
arXiv:2406.14775; Gupta et al., 2025, JAMES, 10.1029/2025MS004977 — our
anchor) introduced the M1/M2/M3 nonlocality hierarchy, trained on
ERA5-derived Helmholtz-decomposed fluxes (Gupta et al., 2024, Sci. Data,
10.1038/s41597-024-03699-x) and evaluated with Hellinger distances on flux
distributions; foundation-model fine-tuning followed (arXiv:2509.03816).
Nonlocal inputs for subgrid closures more broadly: Wang, Yuval & O'Gorman
(2022, JAMES 14(10)).

**Interpretability of scientific ML models.** For GW emulators specifically:
kernel-spectrum analysis (Pahlavan et al., 2024, GRL 51(2)), gradient
receptive fields (Pahlavan et al., 2025, GRL), SHAP on an orographic U-Net
(Haslauer et al., 2026, arXiv:2605.05052) — all weight- or
attribution-level. Deeper mechanistic tools have recently reached weather
foundation models — SAEs and feature steering on GraphCast
(arXiv:2512.24440), linear probes on Aurora (arXiv:2511.07787), SAEs on
Walrus (arXiv:2606.11657), CKA across models (arXiv:2605.23778) — but, per
our verified scan (Aug 2026), no probing, patching, or interventional study
existed for any subgrid parameterization emulator.

**CNN position encoding.** That padding leaks absolute position into CNNs is
known in computer vision (Islam et al., 2020; Kayhan & van Gemert, 2020).
Its scientific-surrogate consequences — a memorized geographic prior doing
the work attributed to physics — have to our knowledge not been measured
before, and our roll/patching decomposition quantifies them causally.

**Pre-registered, adversarial workflows.** Our funnel adapts registered-
report discipline to interpretability: every hypothesis carries a
falsifiable statement and kill criterion fixed before its test; critic
passes challenge tables, rankings and drafts; all outcomes are reported.

## 3. Setting: Models, Data, and a Reproducibility Hazard

### 3.1 The released hierarchy

We audit the released checkpoints of Gupta et al. (2025)
(huggingface.co/amangupta2/nonlocal_gwfluxes, MIT), with architectures from
the paper's released code (DataWaveProject/nonlocal_gwfluxes, tag 1.0.0,
Zenodo 10.5281/zenodo.16415113). Inputs are ERA5-derived columns on a T42
Gaussian grid (64x128), 122 model levels (surface to ~45 km): u, v, theta
(and optionally w), plus scalars [lat, lon, surface elevation zs] for M1/M2;
outputs are the two GW momentum-flux components at all levels (244
channels), cube-root transformed and globally standardized. M1 is a
6-hidden-layer MLP on single columns; M2 prepends one 3x3 valid convolution;
M3 is an attention-gated U-Net (Oktay-style additive *gating* on skip
connections — not self-attention) acting on the global map, which notably
*omits* the lat/lon/zs scalars from its inputs. Training used 2010/2012/2014
(all months); all of 2015 is the test year (from the released training
script), from which we use July (screening), January (confirmation), and two
released August snapshots. One checkpoint per configuration exists; no seed
ensembles were released (Section 7).

Two code-level facts matter downstream: the checkpoints contain vestigial,
never-executed BatchNorm parameters (verified: num_batches_tracked = 0;
the paper-era code defines but never applies them), and M3's convolutions
zero-pad a longitudinally periodic domain — the entry point for Finding 1.

### 3.2 A reproducibility hazard in the public data stack

The anchor's training-format monthly files are published within WxC-Bench
(nasa-impact/WxC-Bench, nonlocal_parameterization). We found that ALL these
monthly files share one alternative normalization — different constants and
a 1x-sigma u/v convention, despite file metadata claiming the same "3x"
scaling as the author-released test files. Fed to the released checkpoints
as-is, they yield R^2 of -57 to -81 (M1) and -4.1 to -4.4 (M3)
(results/normalization_forensics). We derived the exact affine conversion
and validated it three ways: algebraic round-trip; a consecutive-hour test
(the July file's last hour and the released August snapshot's first hour are
adjacent — converted fields match with regression slopes ~1); and a skill
gate (converted data reproduces paper-consistent R^2: M3 0.37-0.42 across
July and January). All our experiments auto-detect and convert; the
conversion utilities are released. Anyone combining these public checkpoints
with these public files needs this fix.

### 3.3 Replication gate

Before any interpretation we replicated the anchor's core offline result on
held-out data. Over all 744 July-2015 hours: RMSE ordering M3 < M2 < M1
(0.678/0.698/0.784 normalized; area-weighted the same), Hellinger ordering
likewise for both flux components, and M3 better than M2 in 744/744 paired
timesteps (bootstrap CI on the mean gap [0.0202, 0.0207]). January repeats
this (0.650/0.665/0.744; 744/744; CI [0.0150, 0.0155]). The nonlocality
skill benefit is unambiguous — the question this paper asks is what
implements it.

## 4. Methods: A Pre-Registered Audit

**The funnel.** Thirty falsifiable hypotheses were generated across seven
families (information content, representations, architecture, attention
gates, transfer learning, failure structure, physics consistency), each with
a technique family, cheapest test, and kill criterion; an adversarial critic
pass hardened thresholds and controls before screening (all pre-registered
thresholds appear in the released HYPOTHESIS_TABLE.md; deviations were
logged at the moment they occurred). Screening tests were budgeted under 20
minutes of (CPU) compute; survivors advanced to deep dives with pre-
registered analysis plans, and held-out-month confirmations were limited to
a four-entry ledger (J1-J4) logged before the January file was opened.

**Multiple comparisons.** Every test run is reported (Table 1), winners and
losers alike; verdicts are per-hypothesis pre-registered thresholds rather
than p-values, and no post-hoc selection is applied — the funnel table is
the family of outcomes.

**Techniques.** Linear/ridge probing with mandatory controls (random-init
network, raw-input baseline, shuffled labels, spatial splits); activation
capture and *patching* (intervening on internal activations, including
roll-aligned cross-run patching); controlled input interventions (occlusion,
neighborhood clamping/resampling, longitude rolling); physics grafts
(single-aspect modifications of real columns with an out-of-distribution
admissibility rule, max per-channel |z| <= 4 against monthly column
statistics, and partial-dose fallback); a unit-tested linear GW ray-tracing
baseline (14 analytic tests) for propagation comparisons; SAEs and
representational analyses where registered.

**Positive control.** Null results from graft tests are only meaningful if
the tests can detect physics where it exists. We therefore ran the same
graft families on the released qbo1d testbed emulator
(DataWaveProject/qbo1d) — a 10-layer MLP trained on a 1D QBO model whose
stochastic 20-wave source physics is exactly computable, so ground-truth
responses E[S|u] are available by Monte Carlo. The emulator is faithful
(median corr 0.983 to E[S|u]); the amplitude-graft family validates
unconditionally (median Spearman 0.94 between emulator and true response
curves); the reflection family validates where the true effect is
detectable (median corr 0.75 for relative effect >= 10%; agreement rises
with effect size, r = 0.93). Scope limits of this control (1D, column-local,
no analogue of the U-Net patch arm; the effect-size conditioning cannot be
applied in ERA5 where ground truth is unknown) are given in Section 7; the
unconditionally validated amplitude family is therefore the load-bearing
null for the audit.

## 5. Results

### 5.1 Where the distributional skill lives

Pointwise error tells only part of the story. In physical flux units, the
predicted/true quantile-ratio "ladder" (Fig. 2) shows all three models
compress typical amplitudes severely (P90 ratios 0.07/0.11/0.28 for
M1/M2/M3 in July), with calibration improving with quantile level and with
nonlocality until M3's extreme tail is calibrated (P99.9 = 1.02
[1.005, 1.043]); January replicates the entire pattern (M3: 0.32/0.77/1.04).
Month-scale variance ratios separate the hierarchy sharply: M1 0.36 / M2
0.36 / M3 1.07 (July); 0.42/0.35/1.00 (January). The U-Net's distinguishing
skill is variance/tail calibration — which the anchor's Hellinger metric
rewards only indirectly (Section 5.5).

### 5.2 Finding 1: a position-indexed variance prior

M3 never receives coordinates or orography, yet three causal probes show it
knows where it is and uses that knowledge for its calibration advantage.

**Roll test.** Rolling the input map in longitude changes no physics (the
domain is periodic; M1 is roll-invariant and M2 roll-equivariant by
construction, and both serve as controls). Rolled M3 degrades globally
(+15% RMSE) and its variance calibration collapses to 0.33 — *below M2* —
with P99.9 falling from 1.01 to 0.60 while pooled Hellinger stays unchanged
(0.0745 -> 0.0744). A 15-roll ensemble gives a symmetric, artifact-free
distance curve (variance ratio 0.54 at roll 8, minimum ~0.33 at half-domain,
recovering to 0.56 near full circle) and position shares of 62% (variance),
37% (P99.9), 17% (P99), 2.5% (P90): the position prior specifically supplies
extreme-tail variance, while bulk calibration is flow-driven. These shares
are upper bounds — rolling also moves inputs off-manifold — but four
independent observations tie the effect to position rather than generic
degradation: the symmetric distance curve; regional structure (base variance
ratio 1.29 over climatological hotspots vs 1.02 elsewhere, collapsing to
0.24/0.37 under roll — the prior lives where the hotspots are); the seam
(below); and causal localization (next).

**Causal localization.** Cumulative roll-patching — replacing rolled-run
encoder activations with roll-aligned unrolled ones, validated by an exact
identity control (max deviation 0.0) — restores essentially none of the
calibration through the first three encoder stages (-0.00/-0.00/-0.01),
+12% at conv4, +61% cumulative at the bottleneck; the remaining ~39% is
restored only when the decoder also sees consistent features. The position
computation forms where receptive fields become global and is applied
through decoder-stage boundary effects — consistent with padding-derived
position information accumulating with depth (Islam et al., 2020). (Our
pre-registered sanity endpoint assumed full encoder patching restores ~100%;
its measured 61% is not an implementation error — the identity control is
exact — but a finding: the decoder itself computes position-sensitively.
Logged as a corrected assumption.)

**The seam.** A visible corollary: M3 (only) carries +5.1% (July) / +5.05%
(January) excess error at the three columns adjacent to the map boundary
(M1: -1.3%, M2: -0.8%); under rolling, the absolute excess moves with the
padding boundary.

**Not reconstruction, and not rescaling.** Linear probes for surface
elevation from M3's first encoder stage read *worse* than from the raw
inputs (R^2 0.06 vs 0.45; even a random-init network gives 0.08) — the
encoder discards linearly-readable orography rather than reconstructing it;
what little zs information exists at that depth is flow-derived, not
position-derived (roll-arm probe: R^2 0.14 against source geography vs
-0.03 against map position). And the calibration is not a trivial output
rescaling: per-channel variance-matching gains applied to M2 explode
physical tails (P99 ratio 3.19, variance 11.9, RMSE 1.14) instead of
reproducing M3's ladder.

### 5.3 Finding 2: what the nonlocality is — and is not

**Causally long-range.** Occluding input outside radius r of a target
column (replacing with monthly climatology) shows M3's predictions need
context to a median saturation radius of 4 grid cells (~1200 km at T42),
with 4/20 targets needing >16 cells. The 3x3 model's context is genuine
information, not smoothing: clamping its neighborhood to nine copies of the
center column makes it *worse than M1* (recovery of the M1->M2 gap: -0.70,
consistent across 24 timesteps).

**But not propagation-shaped.** Our flagship pre-registered hypothesis held
that M3's causal influence maps (gradients of predicted flux energy at
hotspot targets w.r.t. all input columns) would be anisotropic and aligned
with ray-traced horizontal propagation directions from our unit-tested
tracer. It was killed at screening: median ring axis ratio 1.07 (threshold
1.3), wind alignment 9/20 (chance ~6.7/20). At T42, physically expected
horizontal group displacements are largely sub-cell, so this null is partly
resolution-conditioned (Section 7) — but the widely voiced "learned lateral
propagation" narrative finds no support at the scale where these models
operate.

**And not simple.** The M1->M2 gap is closed neither by a linear model on
the stencil (-47% of the gap: worse than M1) nor by gradient features
(+9.6%); per-neighbor ablation shows no wind-conditioned directional
structure passing threshold (spread/mean 0.165 < 0.20, an upstream-biased
hint at best); the context signal at the last hidden layer is high-rank
(top-5 PCs = 23% of clamp-delta variance; projecting them out moves
predictions 2.8% toward M1). The neighborhood is consumed as a full
multivariate pattern.

**Regime correlation (descriptive).** The M2->M3 gain concentrates in
high-shear regions in both months (top-decile shear columns carry 2.9x
(July) / 1.8x (January) their share of the gap) and avoids steep orography
(0.27 / -0.73) — but the pre-registered confirmation threshold (>= 2.0)
failed in January, so we report this as a directionally consistent,
quantitatively variable pattern, not a confirmed claim. The attention gates
are causally live (flattening the finest gate costs +4.3% RMSE, decaying
monotonically with scale) yet uncorrelated with instantaneous flux
(r = -0.015) — modulation, not source selection. Mid-encoder features do
carry a regime code that transfers across disjoint regions
(orographic-vs-nonorographic AUC 0.72 vs 0.40 for random-init, above the
entire shuffled-label range).

### 5.4 Finding 3: the physics-trust audit

The central mechanism of GW physics is critical-level filtering: where the
wind matches a wave's phase speed, upward flux is absorbed. We graft
directional wind reversals into real columns (reflection about u(z_c) above
z_c, partial-dose beta rule under the OOD-admissibility constraint), scale
perturbation amplitudes about climatology, and rotate column winds over
orography — each graft changing one physical aspect of an otherwise real
profile.

On the validated testbed (Section 4), the emulator that learned its physics
responds strongly and correctly. The audited models do not:

- **Typical columns, filtering:** median suppression above the grafted
  reversal -0.008 (July uvtheta, n=289) and +0.043 (August uvthetaw,
  n=284) — no response, replicated across month and across the released
  checkpoint/input-variant axis (which is confounded n=2; Section 7).
- **Amplitude scaling:** median Spearman of response vs amplitude 0.49
  (both variants; testbed: 0.94) — non-monotone for most columns.
- **Drag direction:** rotating column winds over Andes columns leaves the
  predicted drag vector unaligned (circular alignment 0.22/0.23, mean angle
  error 74-77 deg across both variants; n=121-135 admissible rotations,
  145-159 OOD-excluded — disclosed).
- **Hotspot centers, spatially coherent grafts:** 10x10-column patch
  reflections elicit weak partial-dose responses (January, 28 admissible
  patches: median suppression at max applied dose 0.125 for M3 vs paired M1
  0.092; median applied beta 0.40; 20/28 ladders have only two distinct
  doses; dose direction mixed — M1 22/28 positive-monotone, M3 16/28 with
  12 anti-monotone). This is far short of the near-complete flux removal
  linear theory prescribes at a full critical level, though we derive no
  per-patch quantitative expectation (Section 7).

Internally, the picture matches: no hidden layer carries linearly readable
N^2 or Richardson number beyond what raw inputs already provide (negative
probe selectivity at every depth, against random-init and input-baseline
controls); flux sign is barely decodable anywhere; the linear flux readout
collapses after layer 1 and crystallizes only in the last hidden layer (88%
of full-model skill at act6 vs 63% at act5) — consonant with the anchor's
own finding that last-2-layer fine-tuning suffices for transfer. And the
three architectures share a failure floor: their top-1% errors co-occur at
38-53x chance with error-map rank correlations 0.78-0.88, suggesting a
common, data-limited ceiling (ERA5's own unresolved GW fraction) rather
than mechanism-specific failures.

### 5.5 A methods result: Hellinger is gameable where ladders are not

Two of our arms make this concrete. Variance-matched M2 (Section 5.2) is
physically absurd (variance ratio 11.9) yet achieves *better* pooled
Hellinger (0.048) than M3 (0.075); rolled M3 loses its variance calibration
entirely while pooled Hellinger is unchanged (0.0745 -> 0.0744). Pooled
histograms reward marginal shape, not the joint calibration that matters;
quantile-ratio ladders (with per-timestep bootstrap CIs) discriminate in
both directions. We recommend ladders alongside — or instead of — pooled
Hellinger for distributional evaluation of flux surrogates.

## 6. The Funnel (Table 1)

30 hypotheses: 9 screen-passes (gates structured N1; context=information I2;
gates causal N3; seam P4; calibration ladder A5; tails drive Hellinger F3;
long-range context I4; regime code transfers R3; late crystallization R6);
15 screen-kills (linear stencil A1; regime concentration A2; hotspot-type
improvement F2; gate-flux correlation N2; directional ablation I1; low-rank
context A4; propagation-aligned influence N5 [flagship]; probed
intermediates R1; orography reconstruction R2; sign/magnitude separation
R5; critical-level response P1; drag rotation P2; amplitude monotonicity
P3; gradient carriers I3; sparse decomposition R4); 1 measured (failure
co-occurrence F1); 5 deferred with reasons (A3 compute; N4 conditional;
L1/L2 asset-verification; A2b successor); A2b killed at January
confirmation. Full statements, pre-registered kill criteria, and one-line
evidence: HYPOTHESIS_TABLE.md (released); every verdict's numbers are in
stamped results files.

## 7. Limitations

- **Scale gap.** T42 coarse-grained fluxes; expected horizontal group
  displacements are largely sub-cell, so propagation-shaped nonlocality
  (killed N5) may emerge at finer scales; conversely the position prior may
  be weaker in models trained on domains without fixed layouts.
- **Offline only.** No claims about online/coupled behavior (the
  offline-online gap of Pahlavan et al. is exactly why).
- **One checkpoint per configuration.** Robustness axes available were the
  released uvtheta/uvthetaw variants (a confounded checkpoint+input axis,
  n=2) and stratosphere-only epoch pairs (probe-level checks only); no seed
  ensembles exist for the released models.
- **Roll-based attribution is an upper bound**: rolling removes the
  position code *and* moves inputs off-manifold; the two effects are not
  separable with a single released checkpoint. Corroborating evidence
  (symmetric distance curve, localization with exact identity control,
  hotspot concentration, M1/M2 controls, unchanged Hellinger) ties the
  calibration collapse to position specifically.
- **Graft external validity.** Grafts break physical balance by design;
  admissibility (|z| <= 4) and partial dosing bound but do not eliminate
  off-manifold effects. The positive control is 1D and column-local; it
  covers the column-model arms (amplitude unconditionally; reflection only
  where true effects are detectable — a conditioning impossible in ERA5),
  and does NOT cover the U-Net patch arm, whose weak-response result is
  therefore bounded/descriptive. Per-patch quantitative expectations from
  linear theory are not derived (future work with the released ray tracer).
- **J3 dose-response caps**: median applied dose beta 0.40; 20/28 ladders
  had two distinct doses.
- **ERA5's unresolved GW fraction** bounds any achievable "physics" in
  these labels; the shared failure floor (F1) is consistent with this.
- **Test-year epoch selection.** Released checkpoint epochs were presumably
  selected on 2015 validation loss; our July/January months are within that
  year (standard caveat).

## 8. Reproducibility Statement

All experiments are config-driven and seeded; every results file embeds its
config hash, seed, and library versions. The repository releases: the
hypothesis table with pre-registered criteria; the append-only research log
(including failures, amendments, and two adversarial critic reports); the
WxC normalization conversion utilities with their validation tests; the
physics baselines with 14 analytic unit tests; 39 passing tests overall;
and `make reproduce-figures`, which regenerates every figure from stamped
results. Only released artifacts (checkpoints, code, public data) are used.

## 9. Conclusion

Audited mechanistically, the celebrated nonlocality of a released neural GW
parameterization hierarchy resolves into: a genuine but isotropic,
regime-reading use of long-range context; a variance-calibration advantage
carried substantially by an implicit, padding-derived geographic prior; and
an absence of the causal wave physics the skill is often taken to imply —
established with a positive-controlled intervention battery and a
pre-registered funnel in which most of our own hypotheses, including the
flagship, died. None of this diminishes the models' in-distribution utility;
it sharpens what their skill *is*, how to measure it (ladders, not pooled
Hellinger), which artifacts to fix (periodic padding; normalization
metadata), and how much trust to extend where the climate questions live —
out of distribution.

## Appendix pointers (repository)

A: funnel table (HYPOTHESIS_TABLE.md). B: normalization forensics and
conversion validation. C: physics-baseline tests. D: graft admissibility and
dose distributions (stamped). E: critic reports (screening, ranking, final
draft audit). F: figure scripts (experiments/24_figures_core.py).

## Reviewer-attack list (with honest responses)

1. "Grafts are OOD, nulls are meaningless" -> OOD policing + qbo1d positive
   control (amplitude family unconditional: 0.94 there vs 0.49 here) +
   partial-graft rule; P2's 145/280 exclusions disclosed.
1b. "Your 'weak response' has no quantitative physical benchmark" -> honest:
   per-patch expectations not derived; claims scoped to partial-dose grafts
   and the qualitative prescription of near-complete removal at full
   critical levels; deriving ray-tracer-based per-patch expectations is
   stated future work.
2. "One checkpoint, overgeneralization" -> variant robustness (confounded
   n=2, disclosed), scoped claims ("this released family").
3. "Position encoding is trivial/known" -> known in vision (Islam 2020),
   unmeasured in scientific surrogates; here it carries up to 62% of
   variance calibration (the discriminating ladder metric) — while the
   literature's Hellinger headline is INSENSITIVE to the collapse
   (0.0745 -> 0.0744 under roll), which is itself our methods finding.
4. "Hellinger critique unfair" -> demonstrated constructively (rescaled-M2
   beats M3 on Hellinger while physically absurd).
5. "Roll tests confound geography with position" -> zs channel identical
   across files; M1/M2 roll-invariant/equivariant controls carry geography;
   only M3 degrades. Position share stated as an UPPER bound; corroborated
   independently by the symmetric roll-distance curve, bottleneck/decoder
   localization with an exact identity control, and hotspot concentration.
6. "The positive control does not cover the U-Net patch arm" -> correct and
   disclosed; the patch-arm result is reported as bounded/descriptive, and
   the load-bearing audit null is the unconditionally-validated amplitude
   family on column models.
