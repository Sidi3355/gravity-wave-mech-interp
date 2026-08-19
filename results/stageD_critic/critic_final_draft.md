# CRITIC — Final hostile review of PAPER_DRAFT.md (pre-prose-expansion)

Date: 2026-08-19. Reviewer role: CRITIC (adversarial pass, distinct from LEAD/WRITER).
Scope: PAPER_DRAFT.md vs RESEARCH_LOG.md, HYPOTHESIS_TABLE.md, and stamped results/*.json.
Method: every number in the draft traced to its stamped source; J3 row-level data
recomputed independently from results/j3_patchdose_january/metrics.json.

VERDICT UP FRONT: NEEDS FIXES BEFORE PROSE EXPANSION. No fabricated numbers; the
evidence record is strong and unusually honest. But the draft contains (1) one
claim that is contradicted by its own stamped evidence (reviewer-attack list #3),
(2) one headline phrase with no quantitative basis ("an order of magnitude below
physical expectation"), (3) a "dose-monotone" characterization that a row-level
audit substantially undermines, and (4) two mandatory-content omissions. All are
fixable by rewording and one small optional computation; none require new
experimental campaigns.

---

## (a) NUMBER AUDIT

### Verified clean (draft value = stamped source)

| Draft claim | Stamped source | Value |
|---|---|---|
| Variance calibration 1.10 -> 0.33 under roll | gateweek_g2 m3 / m3_roll64 variance_ratio | 1.0984 -> 0.3335 |
| Position shares 62 / 37 / 17 / 2.5 % (var, P99.9, P99, P90) | d1_rollensemble position_shares | 0.6165 / 0.3653 / 0.1742 / 0.0247 |
| Hotspot variance 1.29 -> 0.24 | d1_rollensemble regional_variance_ratio | 1.2886 -> 0.2368 |
| Roll-distance curve symmetric (0.54 / 0.33 / 0.56) | d1_rollensemble per_roll_variance_ratio | 0.545 / 0.323–0.333 / 0.556 |
| +15% global RMSE under roll | gateweek_g2 rmse_norm | 0.6785 -> 0.7811 (+15.1%) |
| Rescaled-M2 beats M3 on Hellinger while absurd | gateweek_g2 m2_rescaled | Hell 0.0481 vs M3 0.0745; var ratio 11.86; P99 3.19; RMSE 1.139 |
| July variance ratios 0.360/0.363/1.066 | a4_fullmonth / screen_a5_tails | 0.3596 / 0.3632 / 1.0663 |
| January variance ratios 0.42/0.35/1.00 (0.419/0.351/0.996) | a4_january | 0.4192 / 0.3512 / 0.9956 |
| M3 > M2 in 744/744, July CI [.0202,.0207], Jan CI [.0150,.0155] | a4_fullmonth, a4_january m2_minus_m3_rmse_t | match |
| Seam excess Jan +5.05% (July +5.1%); M1 −1.3%, M2 −0.8% | j4_seam_january, screen_p4_seam | 0.05054 / 0.05103 / −0.0128 / −0.0082 |
| D1-B: conv5 ~48%, decoder ~39%, early encoder ~0 | d1_rollpatch restoration_fraction_variance | conv5 cum 0.6053 (incr 0.484); decoder 0.395; conv1–3 ≈ 0; identity control exact |
| R2 selectivity −0.39; roll arm flow 0.14 vs position −0.03 | probes_m3 r2 | −0.3867; 0.1362 / −0.0309 |
| R3 cross-region AUC 0.72 vs 0.40 randinit | probes_m3 r3 | 0.7203 / 0.3956 |
| N5 axis ratio 1.07, alignment 9/20 | screen_n5 | 1.0700, "9/20" |
| I4 median radius 4 cells, up to >16 | screen_i4 | 4.0; 4/20 rows ">16" |
| I2 recovery −0.70; C-3 corr 0.56 | screen_i2_clamp | −0.7027; 0.5640 |
| Gates +4.3% finest; r = −0.015 vs flux | screen_n3_gateablate, screen_n2 | 0.04282; −0.01525 |
| Shear conc 2.92 (Jul) / 1.77 (Jan); orog 0.27 / −0.73 | screen_a2_regimes, j2_regimes_january | 2.9190 / 1.7708; 0.2745 / −0.7269 |
| F1 co-occurrence 38–53x | screen_f1 | 37.9–53.2 |
| G1 fidelity 0.983; amplitude Spearman 0.94; reflection 0.75 @ effect ≥ 10% | gateweek_g1 metrics + effect_conditioned | 0.9826 / 0.9429 / 0.7472 (n=56) |
| M1 grafts: −0.008 (n=289), +0.043 (n=284); P3 Spearman 0.49/0.486; P2 0.22 (n=135), P2' 0.228 (n=121) | d2_battery, screen_p_grafts, d2_battery/p2_uvthetaw | all match |
| J3: 28 admissible; M3 suppression 0.125, median Spearman 0.90; M1 0.092; July 0.146 (n=10) | j3_patchdose_january, d2_battery arm3 | 0.1247 / 0.8999 / 0.0923 / 0.1460 |
| P2 exclusions 145/280 | screen_p_grafts p2 | n_adm 135 + n_ood 145 = 280 |
| A4 replication (2-snapshot) numbers in §3 | a4_replication | match (RMSE 0.796/0.704/0.684; uvthetaw inversion 0.615 < 0.627) |
| January conversion gate M3 0.394–0.416, M1 0.210–0.239 | january_gate | match |

### Mismatches / numbers WITHOUT a stamped source (5 items)

1. **M1 dose "Spearman 1.00" (§7) has no stamped summary.** j3_patchdose_january
   contains `median_dose_spearman_m3` but NO M1 analogue; the 1.00 must be
   recomputed from rows. I recomputed it: median is indeed 1.0 — but the
   distribution is 22 patches at +1.0 and 6 at −1.0, all trivially ±1 (see (b)-3).
   Fix: stamp an M1 summary (median + distribution) into the results file or an
   addendum before prose expansion; the draft's own preamble ("every number cited
   here exists in a stamped results/*.json") is currently false on this item.
2. **"R^2 of −4 to −81" (§3) has no stamped source.** The as-is (unconverted)
   evaluation exists only in the RESEARCH_LOG narrative (2026-08-18 20:05:
   −57..−81 M1, −4.1..−4.4 M3). Fix: stamp a small forensics JSON (the runs are
   seconds) or cite the log entry explicitly and drop the "every number stamped"
   preamble claim.
3. **"~1500+ km" (abstract, §6) is an unstamped unit conversion.** The stamped
   quantity is 4 grid cells (median). At T42 (~2.8°, ≈310 km at the equator,
   less poleward), 4 cells is ~1100–1250 km; the log's "~380 km cells" figure
   used for 4×380 ≈ 1520 km is itself generous. Fix: report "≥4 grid cells
   (~1200 km at T42)" or state the conversion.
4. **Reviewer-attack list #3: "carries 62% of the headline distributional-skill
   metric" is CONTRADICTED by stamped evidence.** The anchor's headline
   distributional metric is Hellinger, and gateweek_g2 shows Hellinger(uw) is
   UNCHANGED under roll (0.07450 -> 0.07443). Position carries 62% of the
   *variance ratio* (and 37% of P99.9), ~0% of Hellinger. This is not a typo —
   it is the exact conflation the paper's own methods point (Hellinger
   insensitivity/gameability) warns against. Fix: "62% of variance calibration
   (the discriminating ladder metric); the Hellinger headline is insensitive to
   the collapse — which is itself our methods finding."
5. **Funnel bookkeeping (abstract): "25 screened, 15 killed."** The log's final
   funnel is 9 pass + 15 kill = 24 screened, plus F1 measured = "25/30
   resolved"; its own kill-rate statement (62% = 15/24) counts 24 screened.
   Post-January, A2b was killed at confirmation, so total kills are 16/30.
   Fix: "30 generated; 24 screened (15 killed), 1 measured, 5 deferred; 1
   further killed at January confirmation" or similar — pick one accounting and
   state it once.

### Stamped-file blemishes (fix the files, not the draft)

- results/a4_january/metrics.json has `"month": "2015-07"` — a copied label; the
  config's month_file is scaling01 (January). An auditor will notice.
- results/j2_regimes_january/metrics.json says `"hypothesis": "H-A2",
  "verdict": "PASS"` while the log/table verdict is A2b KILLED at confirmation
  (1.77 < 2.0). The PASS apparently refers to the descriptive pattern
  replicating; as stamped it contradicts the funnel. Add a clarifying field or
  addendum.
- results/gateweek_g1/metrics.json retains `"verdict": "PARTIAL"` with the
  amended verdict only in the log + effect_conditioned.json — this one is
  actually GOOD practice (disclosed in the log); keep, but the paper must cite
  both files.

---

## (b) CLAIM-EVIDENCE GAPS

1. **"largely" (abstract, §5 mechanism statement).** "Distributional skill is
   largely a position-indexed variance prior" is supported only for the
   VARIANCE component (62%). The extreme tail is 37%, P99 is 17%, P90 is 2.5% —
   the draft's own §5 numbers. "Tail calibration implemented largely as a
   position-indexed variance prior" (§5) is unsupported as written: by D1-A's
   own decomposition, bulk and even P99 tail calibration are mostly
   flow-driven; the log's own reading was "the position code specifically
   injects EXTREME-TAIL variance; bulk calibration is flow-driven." Fix:
   "variance calibration is largely (62%) position-indexed, the extreme tail
   partially (37%), bulk quantiles mostly flow-driven." Note there is also an
   UNDERclaim available here, honestly stated with its own caveat: rolled-M3's
   variance ratio (ensemble mean 0.42) lands at the local models' floor
   (M2 0.396) — i.e., essentially ALL of M3's variance-calibration advantage
   over M2 is layout-derived. That is a stronger, defensible statement than
   "62%", provided the roll-damage confound caveat in (e)-2 is attached.
2. **"62% attributable to position" vs the D1-A definition (abstract).** The
   pre-registered D1-A definition is (var_base − mean_r var_rolled)/var_base
   with the 15-roll ENSEMBLE MEAN (0.421), not the roll-64 value (0.333) quoted
   in the same sentence ("1.10 to 0.33"). A reader computing 1 − 0.33/1.10 gets
   70%, not 62%. Also, the share is a share of the variance-ratio VALUE, not of
   "calibration" (distance from 1). Fix: quote the ensemble mean alongside
   ("roll-ensemble mean 0.42; single roll-64 0.33"), state the definition in §5,
   and say "of the variance ratio."
3. **"dose-monotone" (abstract, §7) — the largest quantitative gap found in this
   audit.** Row-level recomputation of j3_patchdose_january:
   - 20 of 28 admissible patches have only TWO distinct dose levels (admissible
     beta caps collapse the {0.25,0.5,0.75,1.0} ladder; median admissible
     beta = 0.45). A 2-point Spearman is a sign bit, not monotonicity.
   - M3 dose-Spearman distribution: 14x(+1.0), 1x(0.8), 1x(0.2), 1x(−0.78),
     11x(−1.0). TWELVE of 28 patches (43%) respond in the WRONG direction with
     dose. The median 0.90 sits on a bimodal ±1 distribution and is not a
     faithful summary. M1: 22x(+1.0), 6x(−1.0).
   - The claim that survives honestly: "median suppression is positive and
     correctly signed at hotspot centers (M3 0.125, M1 0.092), with the
     majority of patches (16/28 M3, 22/28 M1) responding in the correct
     direction; dose ladders are mostly two-point due to OOD admissibility
     caps." "Dose-monotone" as an unqualified property must go, or be scoped to
     the majority-of-patches sign statistic with the full distribution shown
     (F6 should show the distribution, not the median).
4. **"replicated on a second held-out month" (abstract).** The parenthetical
   correctly shows what replicated (variance ratios), but the sentence
   structure lets "62% attributable to position" inherit the replication. No
   January roll-ensemble was run (not in the J1–J4 ledger); the 62%
   decomposition is July-only. §5 states this correctly; the abstract must
   scope it: the calibration SPLIT and the seam replicate in January; the
   position-share decomposition does not (was not tested).
5. **"regime-indexed" (abstract, Finding 2).** As a property of the nonlocality
   this most naturally reads as A2b — which was KILLED at its pre-registered
   January confirmation (1.77 < 2.0) and is reported descriptively in §6
   (correctly). If "regime-indexed" is meant to rest on R3 (cross-region regime
   abstraction, AUC 0.72) plus I2, say so; otherwise soften to
   "regime-associated (descriptive; confirmation threshold not met)".
6. **"robust across checkpoints and input variants" (abstract, §7).** The
   robustness axis is n=2 (uvtheta/July vs uvthetaw/Aug), and checkpoint,
   input set, and month are CONFOUNDED across that pair — they cannot be
   claimed as two separate axes. Fix: "replicates across the two released
   input-variant checkpoints (which also differ in evaluation month)". §8
   already concedes one-checkpoint-per-config; the abstract wording outruns it.
7. **"the paper's headline Hellinger metric is shown to be gameable" (§6/abstract).**
   Our demonstration is on OUR pooled-histogram Hellinger implementation. The
   anchor PDF was never read (Cloudflare; logged open gap 2026-08-18 18:55), so
   the anchor's exact Hellinger construction is unverified. If the anchor pools
   differently (e.g., per-region or per-level), the constructive counterexample
   may not transfer verbatim. Fix: scope to "Hellinger on pooled flux
   histograms (as in our replication of the metric)" and close the PDF gap
   before submission — this was already flagged in the log as a to-do.

---

## (c) INTERNAL CONSISTENCY

1. **Abstract self-contradiction on Finding 3 (real, quotable).** The same
   sentence asserts "the column models exhibit no critical-level filtering
   response" and then "at strong-forcing hotspot columns BOTH models show weak
   but dose-monotone responses" — M1 is a column model, and its hotspot
   response (0.092) is a (weak) filtering-direction response. A reviewer will
   quote these clauses against each other. Fix: "no filtering response at
   typical columns; weak, correctly-signed responses at strong-forcing hotspot
   columns — a bounded negative." The bounded-negative framing itself is
   sound; the abstract's first clause just needs the "at typical columns"
   scope it already has in §7.
2. **"An order of magnitude below physical expectation" — the weakest link,
   confirmed.** The stated basis (§7) is qualitative: "a true critical level
   largely eliminates upward flux" (Booker–Bretherton-type linear absorption).
   No quantitative expectation is derived anywhere in the record. Two problems:
   (i) 0.125 vs ~1.0 is a factor of 8 — "order of magnitude" is already
   rounding up; (ii) far worse, the median APPLIED dose was beta = 0.45
   (admissibility-capped partial grafts). A partial shear reduction does not
   generally create a critical level for the flux-carrying spectrum, so the
   ~1.0-elimination expectation does NOT apply at beta 0.45; the honest
   expected response at the applied doses is UNKNOWN (the resolved-flux
   phase-speed spectrum is unobserved offline). The July arm3 (all beta ~1.0,
   n=10, median 0.146) is the only subset where "near-elimination expected"
   applies, and there the factor is ~7. Fixes, in order of preference:
   (a) derive a per-patch expected suppression under stated spectrum
   assumptions using the validated ray tracer (cheap, uses existing A5-gated
   code) and report observed/expected; (b) scope the "order of magnitude"
   claim to full-dose grafts (July n=10, plus the January beta=1 subset) and
   describe partial-dose results as "weak, correctly-signed, quantitatively
   unbenchmarked"; (c) drop the multiplier and say "far below the
   near-elimination expected for a full critical level."
3. **§5 R2 sentence misattributes.** "R2: encoder DISCARDS orography
   (selectivity −0.39) — the geography channel is positional, not
   reconstructed from flow." But R2's own roll arm (probes_m3) found the
   residual zs signal at conv1 is FLOW-derived (0.136) and NOT positional
   (−0.031). The positional geography channel is a DEEP phenomenon (D1-B:
   conv5/decoder). Correct statement: "the encoder does not reconstruct
   orography from flow at early depth (R2 kill); the geographic prior instead
   enters as implicit position at bottleneck/decoder depth (D1-B)." As written,
   a reviewer reading the R2 JSON will call the draft's compression wrong.
4. **Hellinger tension (see (a)-4).** The draft simultaneously argues
   (i) position carries the distributional skill and (ii) Hellinger doesn't
   move under roll. Both are true only if "distributional skill" is explicitly
   defined as ladder calibration, with Hellinger dismissed as insensitive.
   The draft must make that definitional move EXPLICITLY, once, and then use
   it consistently (abstract, §5, attack-list #3).
5. Minor: G1 raw verdict is PARTIAL with an amended (pre-declared) conditioned
   criterion — §7's "reflection corr 0.75 where true effect >= 10%" discloses
   the conditioning inline, which is honest; ensure the prose version keeps
   the amendment chain visible (raw 0.06 unconditioned -> 0.75 conditioned,
   IQR 0.10–0.83) rather than quoting only 0.75.

---

## (d) MISSING MANDATORY CONTENT

1. **Multiple-comparisons statement: ABSENT.** Nothing in the draft mentions
   the January multiplicity ledger (pre-registered J1–J4 before file open,
   RESEARCH_LOG 2026-08-18 23:56) or a screening-stage multiplicity position
   (30 hypotheses, pre-registered kill thresholds, no p-value shopping;
   descriptive decompositions carry no inferential claims). This is required
   content and is also one of the program's genuine strengths — write it into
   §4. (Suggested content: screening used pre-registered per-hypothesis kill
   thresholds, not significance tests; the only confirmatory inferences are
   the four pre-registered January tests; bootstrap CIs are per-metric
   descriptive intervals, uncorrected, and labeled as such.)
2. **Funnel-as-table: present as a plan (§4 references HYPOTHESIS_TABLE.md) —
   acceptable at this stage, but the table must carry the one-line evidence
   AND the numeric kill thresholds, including the two post-registration
   amendments (A5 rewrite, A2b birth/kill) with their dates.**
3. **Offline-only scope: PRESENT (§8, abstract). OK.**
4. **Limitations completeness: mostly present. Missing items:**
   - The roll-decomposition attribution caveat ((e)-2 below) — currently
     nowhere in §8.
   - The J3 admissibility-capped dose issue (median beta 0.45, two-point
     ladders) — belongs in §8 or §7.
   - The unread-anchor-PDF gap as it bears on the Hellinger-definition claim
     ((b)-7) — must be closed or disclosed.
   - The positive control's non-coverage of the M3 patch-graft protocol
     ((f) below) — §8 says "positive control is 1D and column-local" but does
     not say the patch arm has NO positive control.
5. **Negative results with equal care: largely satisfied** (kills are
   quantified in §5–§7 and the funnel). Exception: the J3 anti-monotone
   fraction (12/28) is currently invisible — reporting only median Spearman
   0.90 is NOT equal care for the unfavorable half of the distribution.
   Also report J3's 20 skipped-OOD patches (28/48) in the text, as arm3's
   10/20 was reported.

---

## (e) THE 3 WEAKEST POINTS (required deliverable) — with best honest responses

1. **"Order of magnitude below physical expectation" has no derived
   expectation, and the median applied dose (beta 0.45) voids the
   full-critical-level benchmark.**
   Honest response: concede and repair. The defensible finding is directional
   and bounded: correctly-signed median responses at hotspot centers
   (M3 0.125 / M1 0.092), absent at typical columns (−0.008/+0.043), majority
   correct-sign across patches — far below near-elimination WHERE
   near-elimination applies (full-dose subset, factor ~7). Either derive a
   per-patch linear-theory expectation with stated spectrum assumptions (the
   validated ray tracer makes this a day's work), or delete the multiplier and
   keep the bounded-negative language. The conclusion "no usable filtering
   mechanism" survives either way; the "10x" garnish does not.
2. **The roll-based position attribution conflates "position code removed"
   with "network degraded by an OOD layout."** Rolling both deletes the
   padding-derived position signal AND corrupts features near the relocated
   boundary (+15% global RMSE); D1-A charges ALL variance loss to "position,"
   so 62% is an upper bound on the clean position share.
   Honest response: four independent lines make position the dominant reading —
   (i) the roll-distance curve is symmetric and already drops at roll 8
   (0.54), far from seam-local; (ii) D1-B's cumulative roll-patching localizes
   restoration at conv5 (~48%) and decoder (~39%) with an EXACT identity
   control, the signature of a deep position code rather than local seam
   damage; (iii) M1 (roll-invariant) and M2 (roll-equivariant) carry geography
   explicitly and show no such collapse; (iv) the boost concentrates precisely
   at climatological hotspots (1.29 vs 1.02). State 62% as a
   "layout-sensitivity share (upper bound on position share)" and add the
   caveat to §8.
3. **The qbo1d positive control does not cover what it is quoted as covering.**
   It validates COLUMN grafts in a 1D, column-local emulator with a narrow
   discrete wave spectrum; the M3 PATCH protocol has no positive control at
   all; and G1-a only passed after conditioning on true-effect size — a
   conditioning that is IMPOSSIBLE in the ERA5 arms (no ground-truth
   response), so an unknown fraction of M1's reflection nulls may be
   small-true-effect columns.
   Honest response: re-weight the evidence hierarchy explicitly. The
   amplitude family is the load-bearing null: validated UNCONDITIONALLY in the
   testbed (Spearman 0.94, 100/100 admissible) and failed by M1 in both
   variants (0.49/0.486) — no conditioning asymmetry applies. The reflection
   and rotation nulls are corroborating, with the conditioning asymmetry
   disclosed; hotspot-center targeting (J3) is the partial proxy for
   large-expected-effect selection. Scope "protocol validated" to the column
   families and add the patch-arm gap to §8. Method-validation, not
   model-equivalence, is already the log's own framing (2026-08-19 01:50) —
   the draft should quote it.

---

## (f) QBO1D TRANSFER AUDIT (detail behind (e)-3)

What transfers: the LOGIC of the control — "these graft families elicit
detectable, physics-aligned responses in an emulator that demonstrably learned
its physics" — validates the graft-and-measure methodology as capable of
detecting learned filtering/amplitude physics when present. Fidelity 0.983
makes the testbed emulator a genuine physics-learner, and the OOD/partial-beta
admissibility machinery was mirrored across both settings (logged 01:05).

What does NOT transfer, and must be stated wherever "validated" appears:
1. Dimensionality/locality: 1D, single-column MLP; nothing about the U-Net
   patch protocol (spatial coherence, regime-statistics shifts under patch
   grafts — the log itself flagged "patch grafts also shift regime
   statistics") is validated. F6's juxtaposition (M1 vs qbo1d vs M3 patches)
   invites the reader to over-extend the validation to the patch arm — add an
   explicit marker in the figure/caption.
2. Wave spectrum and response sharpness: qbo1d's discrete two-wave/20-wave
   spectrum concentrates drag near critical lines, making reflection responses
   large and detectable; ERA5 T42 coarse-grained broadband fluxes dilute the
   expected response by an unquantified factor. Detection POWER does not
   transfer — only detection POSSIBILITY.
3. Outcome conditioning: G1-a's pass required restricting to profiles with
   relative true effect ≥ 0.10 (56/100; corr 0.75, IQR 0.10–0.83 — the lower
   quartile is weak even conditioned). The ERA5 arms cannot condition on true
   effect. Therefore the reflection nulls are one-sided evidence: consistent
   with absence-of-mechanism, but individually unable to exclude
   small-true-effect explanations column-by-column. The amplitude family
   (validated without conditioning) does not have this problem and should
   carry the abstract's weight.
Current scoping in the draft: §8 has "positive control is 1D and column-local"
(good start); abstract says "a protocol validated on a 1D testbed emulator"
(acceptable ONLY if the amplitude-first weighting and patch-arm gap are made
explicit in §7).

---

## REQUIRED FIXES BEFORE PROSE EXPANSION (consolidated)

F1. Rewrite attack-list #3 and any "62% of distributional skill" phrasing:
    62% is of the VARIANCE RATIO (ladder metric); Hellinger is unchanged under
    roll (0.0745->0.0744) — make the metric-definition move explicit. [(a)-4, (c)-4]
F2. Abstract Finding 3 sentence: scope "no critical-level filtering response"
    to typical columns; fix the "both models"/"column models" collision. [(c)-1]
F3. Replace/deriva-te "order of magnitude below physical expectation":
    derive per-patch expectation, or scope to full-dose grafts, or drop the
    multiplier. [(c)-2]
F4. Replace "dose-monotone" with the distribution-faithful statement
    (16/28 correct-sign M3, 22/28 M1; 2-point ladders; median beta 0.45);
    stamp the M1 J3 summary; show the distribution in F6. [(b)-3, (a)-1]
F5. Fix "largely": variance 62% only; tail 37%/17%; adjust §5 mechanism
    statement and abstract; optionally add the stronger
    "M3's variance-calibration edge over M2 is ~fully layout-derived" with
    the (e)-2 caveat. [(b)-1]
F6. Scope abstract's "replicated in January" to what J1/J4 tested (calibration
    split + seam), not the 62% decomposition. [(b)-4]
F7. Add the multiple-comparisons / multiplicity-ledger statement to §4. [(d)-1]
F8. Add to §8: roll-attribution upper-bound caveat; J3 dose caps; patch-arm
    has no positive control; unread-anchor-PDF status for the Hellinger
    definition (or read the PDF and close it). [(d)-4]
F9. Ground or soften "regime-indexed" (A2b killed at confirmation; R3 is the
    surviving basis). [(b)-5]
F10. Fix "robust across checkpoints and input variants" -> confounded n=2 axis
     wording. [(b)-6]
F11. Number hygiene: stamp or re-source "R^2 −4 to −81"; state the km
     conversion for "~1500 km"; unify funnel counts (24 screened + 1 measured,
     16 total kills incl. A2b); fix a4_january "month" label and j2 verdict
     field; correct §5's R2-sentence attribution ((c)-3). [(a)-2/3/5, (c)-3]

None of these threaten the core findings: the position-prior mechanism (D1-A/B,
G2, P4, J1/J4) is multiply-evidenced and month-robust; the Hellinger
gameability demonstration is constructive and stamped; the physics-trust
negatives (P1–P3, D2 battery) are pre-registered, replicated across variants,
and honestly bounded by J3. The program's discipline (pre-registration,
append-only log, adversarial passes, disclosed amendments) is its best defense
— the fixes above bring the DRAFT's language back inside what that discipline
actually purchased.

VERDICT: NEEDS FIXES FIRST (F1–F11). After F1–F8, ready for prose expansion.
