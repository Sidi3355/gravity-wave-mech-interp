# CRITIC pass — Stage-C ranking and recommendation (2026-08-18)

Scope: attacking the CHOICE of deep-dive targets (CHECKPOINT_C §2–3), not the
pre-registered screen verdicts. Constraints honored: CPU-only, Tier 1 (one
checkpoint per config; strato epoch pairs M1 88/100 and M2 38/93 are the ONLY
multi-checkpoint control), January 2015 pre-committed, qbo1d staged, Stage-D
bar = 2+ technique families incl. one interventional, held-out generalization,
pre-registration, multiple-comparison honesty.

## O1. D1's central confound is the LEAD's own fallback — and D1's plan ignores it

The single most dangerous fact in the screening record for D1: M3 is
simultaneously (i) the only model with the implicit-position channel (P4:
+15% GLOBAL degradation under roll — layout dependence far beyond the seam)
and (ii) the only variance/tail-calibrated model (A5: var ratio 1.066 vs
0.360/0.363). One model, one checkpoint, two unique properties. If M3's
variance advantage is spatially pinned — a memorized geographic variance
climatology delivered through padding geometry — then "the calibration
mechanism" is P4 wearing a costume, and finding #2 of the emerging paper
collapses into finding #3. A2b makes this worse, not better: "advantage
concentrates in storm tracks" is indistinguishable at screening depth from
"advantage concentrates at fixed longitudes where storm tracks live."
R3's cross-region AUC 0.72 (vs shuffled up to 0.54) is the only datum
pushing back, and it is a correlational probe at one depth.

Yet D1's Stage-D sketch (gate flattening -> tail response; decoder lens;
last-layer statistic matching) contains NO position control. The
calibration-under-roll / position-vs-regime disentanglement is not optional
garnish — it is the experiment a hostile reviewer runs first. Demand: a
mandatory roll/pad-surgery arm inside D1 (circular-padding swap on the
released checkpoint is a CPU-trivial, retraining-free intervention that
directly manipulates the position channel), with a pre-registered outcome
tree for "calibration survives roll" vs "calibration is layout-bound."

## O2. D1 lacks a triviality gate and its named interventional handle points the wrong way

(a) Triviality gate missing. Before any localization, fit a per-level scalar
gain (or quantile map) on M2's OUTPUTS and ask whether post-hoc rescaling
closes the Hellinger/tail gap to M3. Costs ~nothing (saved month outputs
exist). If YES: the "mechanism" is a one-parameter-per-level gain — D1
bottoms out in "the loss/architecture under-disperses and M3 doesn't," a
description, not a mechanism (note the irony: H-L1 already hypothesizes TL
is exactly a diagonal gain). If NO: M3's extra variance is in the right
PLACES, and D1 sharpens into a question about spatial allocation of variance
— actually worth a deep dive. This gate must be pre-registered as D1's
go/no-go. The screening record cuts both ways here and the LEAD did not
engage: M2's variance ratio equals M1's (0.363 vs 0.360) despite 3x3 context
and better RMSE — evidence that calibration is NOT mere information gain
(good for D1) — but M2's P99 tail ratio DID improve (0.284 -> 0.403), so the
ladder partially tracks information after all (bad for the "distinct
mechanism" framing; "is D1 just I2/I4 restated?" is only half-answered).
Also mind metric spaces: RMSE lives in normalized space, variance/tails in
physical (cube-root) space; every "calibration" claim must survive the map.

(b) The pre-staged interventional evidence points AWAY from the sketch.
N3 found the finest gate matters BROADLY (global +4.3% > hotspot +2.4%) and
N2 found gates uncorrelated with instantaneous flux (-0.015). Tails live in
hotspots. So the one validated causal tool D1 leans on (gate manipulation)
already shows a signature inconsistent with being the tail-calibration
mechanism. If gates null out, D1 is left with T8 lenses — correlational —
and fails the master prompt's interventional requirement. And "last-layer
statistic matching between M2 and M3" across two different architectures at
Tier 1 (no retraining) is underspecified and likely budget burn.

(c) Robustness hole: C-1 epoch pairs exist for M1 and M2 ONLY. D1's headline
is a claim about M3. The plan sketch cites "epoch-pair robustness (C-1)" as
if it covers the finding; it cannot. Every D1 mechanism claim is n=1
training run with no within-run control. State it — or prefer targets that
are robust to it (see O3 on D2).

## O3. D2 has no kill criterion for itself, and its positive control is weaker than advertised

The recommendation's own logic — "the missing piece is a positive control" —
concedes D2 is currently unpublishable. So the qbo1d graft run is not a
Stage-D component, it is a GATE, and it must run FIRST (it is CPU-trivial;
there is no excuse for sequencing it after any ERA5 work). Pre-register now:
if the Pahlavan CNN fails the graft protocol (e.g., no critical-level
suppression where the 1D testbed's physics guarantees the mechanism exists),
D2 is dead as a trust audit — at most a cautionary methods note — and no
further D2 compute is spent. The LEAD provides no such decision rule.

Even if the control passes, claim strength is capped: qbo1d is 1D, 73
levels, u->S, daily timestep, different admissibility machinery. Passing
there validates the graft CONCEPT, not the ERA5 implementation. The ERA5
protocol carries its own unresolved anomalies the ranking glosses over:
P1's 20% below-graft change (grafts perturb where they shouldn't — the
paired-comparison logic is leaking); P2's 145/280 OOD exclusions
concentrated at large rotation angles (the most diagnostic region of the
sweep was discarded — "no rotation tracking" partly reflects admissibility,
not the model). "OOD-tightened grafts" tightens this bind: stricter
admissibility shrinks n and further truncates the diagnostic range. D2's
hardening plan must demonstrate a dose-response that survives policing, or
the negative stays soft. On the plus side (the LEAD undersells this): D2 is
the target MOST robust to the Tier-1 n=1 problem — a trust audit of the
released checkpoints is a valid claim about the artifacts people actually
deploy; no generality over training runs is required.

## O4. Best case for the fallback — and why "held back" is the wrong disposition

Position encoding is the only thread where Stage-D success is near-certain:
already causal (roll test), mechanism already identified (zero-pad boundary;
Islam et al. 2020 supplies prior-art legitimacy), an untouched clean
intervention available (circular-pad surgery at inference — no retraining),
two technique families for free (T3 roll/pad interventions + T1 position
probes by depth), January generalization trivial, deployment stakes real,
and it interlocks with C-5's finding that M1 spends ~25x Jacobian mass on
explicit geography (explicit vs implicit geography channels — a tidy
comparative story). Its weakness is ceiling, not floor: novelty is
incremental (known in vision) and screening already captured much of it.
Verdict: it should not float as a "fallback" — it should be FUSED into D1 as
the mandatory confound arm (O1). If D1 fails its triviality gate (O2a), the
fused thread IS the replacement deep dive at near-zero retooling cost. The
LEAD's reason for holding it back ("D1/D2 cover the two strongest
narratives") assumes D1 and D2 both succeed — exactly what a ranking is
supposed to hedge against.

## O5. Budget: two full deep dives is over-committed; sequence behind two cheap gates

Stage-D rigor (pre-registration, 5+ seeds where trainable, bootstrap CIs,
epoch pairs, January confirmation, deviations logged) roughly triples the
per-result cost of screening; the screening waves consumed a full day for
12–24-timestep tests, and the one full-month artifact took ~35 min per run
on this laptop. D1 needs new decoder-lens infrastructure; D2 needs a qbo1d
port plus graft rework; both need the January download. Running both in
parallel at rigor invites the classic failure mode: two half-hardened
results, neither at the bar. Amendment: week 1 runs ONLY the two gates —
(G1) qbo1d graft control for D2; (G2) M2-output-rescaling triviality test +
calibration-under-roll for D1. Both are cheap and use existing assets.
Commit full Stage-D resources to whichever survives. If both survive, D2 is
primary (stakes + n=1 robustness, O3) and D1 secondary. If both gates fail,
promote position encoding — the program still ships a causal paper.

## O6. January multiplicity is quietly accumulating

January 2015 is pre-committed as THE confirmation month, but it is now
booked for: A2b confirmation, D1 confirmation, D2 regime splits, and (§3)
"any" further splits. Each use is individually pre-registered; the ensemble
is a forking-paths liability, and "multiple-comparison honesty" is an
explicit Stage-D requirement. Demand: one frozen January analysis plan,
enumerating EVERY test that will touch January with family-wise correction
(or explicit per-family alpha spending), registered before the download.

## O7. §2 over-claims to strike before the human reads them as established

- "each already causally grounded at screening level" — false for finding 2
  (A5/F3 is observational: quantile and variance ratios; no intervention has
  ever touched calibration) and finding 5 (F1 co-occurrence is
  correlational; "data-limited ceiling" is interpretation).
- Finding 1's "regime-indexed" load-bears on A2b, which is explicitly
  unconfirmed (born from the same July data; January pending). A hypothesis
  cannot appear as evidence in the document that defers it.
- "direct rebuttal-proof link" (D1 rationale) — nothing at screening depth
  is rebuttal-proof; delete the phrase.
- The "9 PASS" topline counts A5 and F3 separately on shared evidence
  (logged honestly in the table, but the headline double-counts one
  dataset).
- "controlled grafts (OOD-policed)" hides that policing excluded the most
  diagnostic P2 samples (O3).

## Amended recommendation (what I would put in front of the human)

1. Gate week first: G1 (qbo1d graft control) and G2 (M2 rescaling
   triviality + M3 calibration-under-roll) BEFORE any other Stage-D spend.
   Pre-register both outcome trees now, including D2's death condition (O3)
   and D1's demotion condition (O2a).
2. D2 primary if G1 passes — highest stakes, and the only target robust to
   Tier-1 n=1 — with graft-admissibility dose-response analysis added and
   the qbo1d scale-gap limitation stated up front.
3. D1 secondary, restructured: position/roll confound arm mandatory (O1),
   rescaling gate as go/no-go (O2a), gate-flattening honestly labeled
   likely-null given N2/N3 signatures (O2b), M3 epoch-pair hole stated
   (O2c).
4. Position encoding fused into D1 and pre-designated (not "fallback") as
   the automatic replacement if either gate kills its parent.
5. One frozen January analysis plan covering A2b + all D1/D2 confirmations
   with explicit multiplicity accounting (O6).
6. §2 edits per O7 before human review.

Single biggest risk, named: M3's calibration advantage turns out to be
layout-bound — a padding-derived geographic variance climatology. That one
Stage-D result simultaneously guts findings 1 and 2 and converts the paper's
headline mechanism into its own artifact section. The current ranking does
not protect against it; the amended D1 does, by testing it first.
