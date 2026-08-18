# CRITIC pass — Stage B hypothesis table (hostile review, 2026-08-18)

Reviewer role: CRITIC (adversarial, pre-screening). Inputs: HYPOTHESIS_TABLE.md
(29 entries), RESEARCH_LOG.md (through A4 first pass), ASSETS.md. Objections
below are to be logged verbatim and acted on. Nothing here is softened.

Verdict up front: the table is better than most first drafts — it has nulls,
controls, and several genuinely cheap causal tests — but it contains (1) one
hypothesis that is **mis-specified against data that does not exist** (H-A5
tests the uvtheta config for a phenomenon observed only in uvthetaw), (2) one
training run smuggled into a screening stage at 10–100x the budget (H-A3),
(3) two hypotheses gated on **unverified assets** (H-L1/L2), (4) at least four
kill criteria that are not decidable as written, and (5) a systemic
single-month / single-checkpoint double-dipping problem that no entry
acknowledges. Details follow.

---

## S. SYSTEMIC OBJECTIONS (apply to many hypotheses; must be resolved in
## Stage C pre-registration, not per-experiment ad hoc)

**S1 — July-only data, and screening/confirmation double-dipping.**
Every data-hungry test in the table (I1, I3, A1, A2, A4, A5, F1–F3, N4, P4,
R-probes' "50k columns") draws on the SAME July 2015 month that was bought for
the A4 confirmation gate. If Stage D "held-out confirmation" later uses July
2015 — the only month on disk — every screening pass-decision has been
conditioned on the confirmation set. That is textbook double-dipping. Demand:
pre-commit NOW a disjoint confirmation month (e.g., January, ~15 GB), or
pre-commit a within-July split (screen on weeks 1–2, confirm on weeks 3–4,
declared before any screening runs). Note the disk conflict with S8.
Additional bias: July = austral winter; Andes/Antarctic-peninsula orographic
hotspots are near seasonal maximum, NH convective sources are summer-mode.
Every "regime" conclusion (A2, F2, R3) is a JULY conclusion until a second
month exists. No entry says this.

**S2 — n = 1 checkpoint per config; no noise floor for ANY cross-model
effect size.** Tier 1 means one released checkpoint per configuration. Every
kill criterion of the form "gap present/absent" (A2, A5, F1–F3, I1) implicitly
assumes training noise is small relative to the effect — an assumption with
zero evidence. A 1.8% RMSE inversion was already observed to be
sample-dependent (A4). Mitigations that exist and the table ignores:
(a) ASSETS.md lists TWO epochs per stratosphere_only config (M1 e88/e100,
M2 e38/e93) — a free poor-man's robustness pair for probe/ablation stability
checks that nobody is using; (b) bootstrap over space/time within the month
gives a data-noise floor (not a training-noise floor — say so explicitly in
every write-up). Demand: every cross-architecture claim must carry the caveat
"single training run" in its pre-registration, and any effect smaller than the
strato epoch-pair discrepancy on the analogous statistic is uninterpretable.

**S3 — Normalized vs physical space is unspecified almost everywhere.**
Inputs are mu-sigma normalized; OUTPUTS are cube-root transformed with fixed
per-month constants. "RMSE", "skill", "|uw| bins", "P99 tail ratio", and
"Hellinger" change meaning (and can change ORDERING) between transformed and
physical space. A4 metrics were computed in — which space? Not stated in the
table. Demand: every hypothesis pre-registers the space of every metric.
Specifically: H-F3's P99 pred/true ratio in cube-root space is NOT the
physical tail ratio (cube-root compresses tails by construction); H-R5's
magnitude bins differ between spaces; H-A1/I3's "% of gap closed" depends on
the metric's space.

**S4 — Scaling constants are per-month ("scaling07"/"scaling08"), not
global.** The constraint sheet says "fixed constants," but the test files are
scaling08 (August) and the July month is scaling07. Which constants does the
July pipeline use, and is that what training used? Every July evaluation, all
T7 synthetics, and every probe dataset inherits this choice. A wrong-constant
run produces silently plausible garbage. Demand: a one-off harness gate
(assert the dataloader's constants for July against training code) BEFORE any
Stage B run; pre-register which month's constants T7 generators use.

**S5 — M3-lacks-lat/lon/zs contaminates every M3-vs-(M1,M2) comparison.**
The table knows this fact (it powers H-R2) but then ignores it in A2, A5, F1,
F2, F3, where any M3-vs-M2 difference NEAR OROGRAPHY is jointly caused by
(architecture) x (input set). H-F2 is worst: "M3 fixes nonorographic
transients" is exactly what you'd predict from the INPUT asymmetry alone
(M1/M2 get zs and can specialize orographically; M3 cannot). Demand: every
cross-architecture regime claim must be co-reported with the clean M1-vs-M2
contrast (same input family), and M3 claims near orography must be flagged
input-confounded.

**S6 — Hotspot boxes are an unverified asset.** R3, F2, N2, N5 all need the
paper's 8 hotspot boxes as labels/targets. The anchor PDF is UNREAD
(Cloudflare-blocked; open gap in the log). If the boxes are not recoverable
from arXiv:2406.14775 (which WAS read) or the code repo, four hypotheses lose
their labels. Demand: extract and pre-register box definitions (with source)
before Stage C; if unavailable, define boxes independently from climatological
flux magnitude and SAY they are ours, not the paper's.

**S7 — Unaudited compute budgets.** Not one entry states a measured cost. No
timing for a single M3 global forward (or backward) on this CPU appears
anywhere in the log. Claims like "occlusion sweep r in {1,2,4,8,16}" (I4),
"Jacobian maps for ~20 targets" (N5), "SAE on 100k vectors" (R4), "alpha on
July hourlies" (N4: 744 M3 forwards for a full month) are 20-min-budget
assertions made blind. Demand: a measured cost table (M1/M2 per-1k-column
forward; M3 global forward and backward; SAE epoch on 100k vectors) as the
FIRST Stage C action; re-scope every entry against it.

**S8 — Disk budget is already in conflict.** Weights 3.2 GB + July 14.9 GB
+ IFS 9 GB (L1/L2) + a second confirmation month 14.9 GB ≈ 42 GB > 30 GB
budget. The table implicitly commits to all of these. Decide now which gives:
my recommendation — defer IFS (L1/L2 are blocked anyway, see below) and buy
the second ERA5 month, because S1 threatens the validity of EVERYTHING while
L1/L2 threaten only two hypotheses.

**S9 — T7 external validity (OOD synthetics) is acknowledged nowhere in
B-PHYS.** The generators are validated against ANALYTIC PHYSICS (A5 gate),
not against the ERA5 training distribution. An idealized uniform-N,
tanh-shear column is far off the training manifold of a net trained
exclusively on ERA5 columns; "no critical-level suppression" could mean
"synthetic input is OOD garbage-in," not "physics not learned." This
asymmetry makes P1–P3 one-directional: positives are meaningful, negatives
are ambiguous — the table sells P1 as "publication value either way," which
is overstated. Demand: (a) report an OOD score per synthetic (e.g.,
Mahalanobis or kNN distance to training columns in normalized input space);
(b) add a REAL-profile variant (graft wind reversals / rotate winds on real
ERA5 columns) as the primary evidence, keeping fully-idealized profiles as a
secondary sweep; (c) drop "M2/M3 with uniform context" from screening —
a spatially uniform global field into a U-Net trained on ERA5 is maximally
OOD and uninterpretable (P1 as written includes this).

---

## Per-hypothesis review

Format: (a) testable at tier/budget; (b) kill sharpness; (c) redundancy;
(d) confounds. Only non-empty sections listed.

### H-I1 — upstream neighbors carry M2 gain
(a) Testable; cheap (column-model forwards). Fine.
(b) NOT SHARP. "Ablation impact uniform across neighbors after wind
conditioning" has no threshold, no test statistic, and 8 neighbors x
wind-octant conditioning is a multiple-comparisons machine. Demand: preregister
one scalar — e.g., upstream-vs-downstream ablation-impact ratio (impact
projected on the local wind bearing), kill if ratio < 1.2 or CI includes 1.
(d) Three confounds unaddressed: (i) "upstream w.r.t. background wind" —
wind at WHICH level? u rotates/shears with height; pick and preregister (e.g.,
launch-level or column-mean wind) or the result is analyst-dof. (ii) Latitude
confound: midlatitude westerlies make "upstream" ≈ "west", which is also the
direction of grid anisotropy (physical zonal spacing shrinks poleward); a
west-east asymmetry could be geometric, not advective. Control: composite
within latitude bands. (iii) "Resample-ablate" — resampled FROM WHAT? A
neighbor drawn from another location breaks spatial coherence and is OOD; you
may be measuring OOD sensitivity, not information. State the resampling
distribution (recommend: same-latitude, same-time shuffle) in the prereg.
(c) Shares harness with I2/A4; I2 is logically prior (if clamping all
neighbors changes nothing, per-neighbor ablation is moot). Order: I2 → I1.

### H-I2 — not mere smoothing (clamp neighbors to center)
(a) Testable; cheapest entry in B-INFO. Good.
(b) SOFT. "Clamped M2 ~= full M2" needs a number: preregister "clamped M2
retains ≥ (100−X)% of the M1→M2 gap ⇒ smoothing" with X stated, and in which
metric/space (S3). Also note this is quietly dual-outcome ("report as such")
— acceptable as a gatekeeper, but LABEL it dual-outcome like F1, or the
"zero filtering" bookkeeping is inconsistent.
(d) Useful free fact the table misses: on a constant 3x3 stencil the conv
collapses to a linear map of the center column, so clamped-M2 is an
M1-architecture model with M2's weights — compare it to ACTUAL M1 as well
(three-way: M1, clamped-M2, full-M2). Costs nothing, sharpens interpretation.

### H-I3 — gradients are the carriers (train small twin MLPs)
(a) BORDERLINE-OVER BUDGET. Two twins x 3 seeds = 6 CPU trainings on ~1M+
columns (1 week hourly x 8,192 columns) with ~350+ input dims. On 4 cores
this is not <20 min unless subsampled hard; no measured basis (S7). Rescope:
≤100k-column subsample, fixed 2-layer width, and TIME IT first.
(b) The 30% threshold is good but the DENOMINATOR is ill-defined. The twins
(reduced width, 1 week) will not reproduce the released M1→M2 gap; "fraction
of gap closed" must be defined twin-internally: gap = full-stencil-twin −
center-twin, with the full-stencil twin trained as the ceiling. As written,
the ceiling twin is absent from the design — without it, "gradient features
close <30%" is uninterpretable (maybe NOTHING closes 30% at this scale).
Demand: add the full-stencil twin (raises cost — see (a)).
(d) Surrogate-inference gap: a reduced twin failing to use gradients does not
show the RELEASED M2 doesn't. A negative here kills only the twin, not the
hypothesis about M2. State this scope limit in the prereg or the kill
criterion overclaims.
(c) Overlaps A1: run A1 (ridge with/without gradient features — gradients
are linear in stencil values, so a RIDGE on raw stencil already spans them;
to isolate gradients linearly, compare ridge-on-gradients-only vs
ridge-on-full-stencil) BEFORE spending training compute. If ridge-on-gradients
≈ ridge-on-stencil, the linear version of I3 is settled for free.

### H-I4 — M3 used-nonlocality radius (occlusion sweep)
(a) Budget unproven: targets x radii x snapshots x full M3 forwards. At a
guessed 2–5 s/forward, 40 targets x 5 r x 5 snapshots = 1000 forwards =
33–83 min. Over budget. Rescope after S7 timing (fewer targets or shared-mask
batching: occlude around ONE target per forward is the naive way; consider
evaluating skill only at the target column but batching targets far apart in
one masked forward — cuts cost ~10x; preregister the minimum separation).
(b) "Skill saturates by r=1" — define saturation: e.g., skill(r=1) ≥ 95% of
full-field skill. Otherwise a 4% residual becomes a Rorschach test.
(d) Occlude WITH WHAT? Zeros in normalized space = climatological mean state
with a hard discontinuity ring at radius r — the conv sees an artificial
edge; skill-vs-r conflates information loss with edge-artifact injection, and
the artifact's strength itself varies with r. Control required: a "replace with
same-location different-time field" occlusion (spatially coherent, in-
distribution) as the primary, mean-fill as secondary. Also: near the lon-0
seam this interacts with H-P4 — exclude near-seam targets until P4 is run.
(c) Same machinery as N5 (occlusion/influence); build ONE masked-forward
harness for both.

### H-R1 — N², Ri decodable with increasing selectivity
(a) Testable, cheap. Controls (random-init, raw-input probe) are the right
ones — best-controlled probe entry in the table.
(b) "Selectivity ~0 or non-monotone noise" — no threshold. Preregister:
selectivity(depth) slope > 0 with bootstrap CI excluding 0, and peak
selectivity ≥ some minimum R² margin (state it).
(d) (i) Layer widths differ across depth → probe capacity confound; fix by
probing equal-dim random projections of each layer (state dim). (ii) Ri is
heavy-tailed (÷ shear²): probe log-Ri or a bounded transform, else R² is
outlier-dominated; preregister the transform and the SPACE (S3). (iii) Ri/N²
at WHICH levels — full profile, per-level, or column-reduced? Unspecified;
preregister.

### H-R2 — M3 reconstructs zs it never receives
(a) Testable, cheap. BUT:
(d) NEAR-GUARANTEED POSITIVE AS DESIGNED — flag as the table's clearest
"probe decodes trivially-present info" trap. zs is a FIXED map; with a
time-based train/test split, the probe memorizes location→zs from ANY
location-identifying signal in the activations (conv activations at a fixed
gridpoint vary little across a few snapshots relative to across-location
variance; zero-padding at the lon seam and pole edges injects absolute
position into conv features — a known CNN effect — so "position" IS in the
activations for architectural reasons, no "inference of boundary forcing"
needed). Demand: SPATIAL held-out split (train probe on a set of longitudinal
sectors, test on disjoint sectors), plus the raw-input baseline computed under
the SAME spatial split. Also beware the baseline being near-ceiling: near-
surface u,v,theta on terrain-following/pressure levels encode orography
strongly; if input-baseline R² is already 0.9, "well above baseline" is
undecidable — preregister the margin.
(b) "Probe ~= input baseline" — no margin stated. With the spatial split,
preregister: kill if (probe R² − input-baseline R²) < 0.05 (or chosen value).
(c) The M1-vs-M3 "natural experiment" is the salvageable core; keep it.

### H-R3 — regime coding is abstract (cross-region transfer)
(a) BLOCKED ON S6 (hotspot boxes unverified). Also: WHICH MODEL? The entry
never says. This matters enormously: M1/M2 RECEIVE lat/lon/zs as inputs, so
their activations decode geography trivially and "geographic coding" is
guaranteed present; M3 is the only clean subject. Preregister: M3 encoder
(primary), M1 as contrast.
(b) Dual-outcome in disguise: the kill outcome is explicitly "still
reportable, flips the interpretation" — so nothing can fail. Either justify
(as for P1) or restate the kill as a real kill: e.g., "if WITHIN-region
separability is at chance, the probe target itself is broken — kill."
(d) Cross-region transfer can succeed via latitude/season proxies (lapse
rate, wind climatology) rather than regime abstraction. Control: a
latitude-matched region pairing, or regress out latitude from features before
transfer. July-only seasonality (S1) bites here hardest.

### H-R4 — SAE features nameable
(a) OVER BUDGET AS WRITTEN: SAE training (two sites: M3 encoder + M1 mid)
x "all tested sparsities" (a sweep!) x dashboards is a Stage-D workload
wearing a screening costume. No timing exists (S7).
(b) UNFALSIFIABLE AS WRITTEN. "Nontrivial fraction" (no number) of features
"coherent" (no rubric, human judgment) "at all tested sparsities" (unbounded).
This is the classic SAE-paper failure mode the program should be above.
Demand: preregister (i) N features inspected (fixed, random sample),
(ii) a written coherence rubric decided BEFORE looking, (iii) a NULL: the
same rubric applied to PCA directions and random orthogonal directions of the
same activations — atmospheric fields make almost ANY linear feature's
activation map look "geographically coherent"; the claim is only meaningful
as an EXCESS over that null, and the table has no null here (it has them
elsewhere, so the authors know better).
(c) Verdict: DEFER to Stage D, run a 1-site pilot only if slack exists.

### H-R5 — sign decodable earlier than magnitude
(a) Testable, cheap, piggybacks R1's activation cache. Good.
(b) "Identical decodability profiles" is not a kill anyone can adjudicate —
profiles are never identical. Preregister: depth-at-half-max-skill difference
≥ 1 layer (or similar scalar).
(d) (i) NO INPUT-BASELINE CONTROL LISTED (R1 has one; R5 forgot it): sign(uw)
correlates with hemisphere/season/launch-level wind sign and is substantially
decodable from raw inputs; without the baseline, "early sign decodability"
is trivially-present input info. (ii) Sign-AUC vs magnitude-bin-accuracy are
incommensurable tasks (different chance levels, different difficulty); use
matched formulations (e.g., binary sign vs binary above/below-median |uw|,
both as AUC). (iii) Class imbalance by region — stratify or AUC-only.
(iv) Space of magnitude bins (S3): cube-root or physical — preregister.

### H-R6 — flux crystallizes in last two layers
(a) Testable, cheap, shares R1/R5 cache. Fine (M1/M2 only; a "per-layer"
readout for M3's decoder is ill-posed as columns — restrict to M1/M2 or
define M3 readout sites explicitly).
(b) "Skill accrues gradually/linearly" — quantify: preregister e.g. "kill if
last-2-layer skill gain < 40% of total input→output gain."
(d) (i) The final layer's readout is (affinely) the model's own head — a
late jump is partially BUILT IN; the informative comparison is layers 1..L-1,
excluding the head, and against (ii) an equal-capacity control: layer widths
differ, so ridge on a wider layer wins for free — probe fixed-dim random
projections (same fix as R1). Without (i)+(ii) a "sharp late jump" is close
to guaranteed and the link to "why last-2-layer TL suffices" is circular
(TL froze everything else BY CONSTRUCTION — the table admits this in B-TL's
header and then re-derives it as a finding here; don't).

### H-A1 — linear stencil model closes ≥50% of M1→M2 gap
(a) Testable, closed-form, genuinely cheap. Among the best entries.
(b) Kill (<20%) is sharp — BUT (i) the 20–50% dead zone has no verdict
(claim says ≥50, kill says <20; what does 35% mean? preregister three-way:
confirm ≥50 / ambiguous / kill <20), and (ii) "% of gap" in which metric and
space (S3)? A4 already showed the ORDERING is metric-dependent; the gap
fraction certainly is. Preregister metric+space.
(d) Missing baseline: linear-on-center-only ridge. Without it, "linear
stencil closes X%" conflates (linearity of the local map) with (linear
context information). Demand the 2x2: {ridge, released net} x {1x1, 3x3}.
Costs nothing (two more closed-form fits).

### H-A2 — nonlocality gains concentrate in high-shear regimes
(a) Piggybacks the month run; cheap. Fine.
(b) "Gap maps spatially/regime-wise flat" — nothing is exactly flat.
Preregister: kill if top-vs-bottom |∇u|-quintile gap ratio < 1.2 (or chosen).
(d) REGION-VS-REGIME, the exact confound the task warns about: |∇u| is
geographically pinned (storm tracks, jet exits); a shear composite is also a
latitude/land-sea/orography composite. Controls: (i) composite within
latitude bands; (ii) partial-out a static climatological gap map (does the
TRANSIENT shear anomaly predict the gap, or only the mean map?); (iii)
area-weight all composites (grid convergence poleward — unweighted column
counts overweight the poles by ~an order of magnitude). Also S5: report
M2−M1 (clean) separately from M3−M2 (input-confounded).

### H-A3 — attention gating vs plain UNet (reduced-scale retrain)
(a) NOT TESTABLE WITHIN SCREENING BUDGET, and the table knows it
("borderline... sanctioned slow screen" is self-granted absolution). 2 archs
x 3 seeds = 6 UNet trainings on 2 weeks of global hourly data on a 4-core
laptop: this is many hours to days, i.e., 10–100x the 20-min cap, competing
with ALL other screening for the same 4 cores. It is also scientifically the
weakest form of the question: a 1/4-width, 2-week retrain's gate-vs-no-gate
outcome does not transfer to the released full-scale M3 (capacity-dependent),
so even a "clean" result attributes little about the actual model under
study.
(c) H-N3 (uniform-alpha ablation on the RELEASED M3) answers the causal
half of this question — "do the learned gates matter?" — for the cost of a
forward hook. It cannot answer "would training without gates have matched?"
but that question is Tier-2 by nature. Demand: DEFER H-A3 to Stage D,
contingent on H-N1/N3 results; do not spend screening compute on it.
(b) If it ever runs: "clearly and consistently worse" — preregister effect
size vs the 3-seed spread.

### H-A4 — low-rank context channel
(a) Testable; PCA + projection is cheap. Layer(s) unspecified — preregister.
(b) Kill is a conjunction of two vague clauses ("high-rank", "nothing
targeted"). Preregister BOTH: (i) rank criterion: top-k (k≤5) PCs of the
full-vs-clamped activation delta explain ≥X% variance AND exceed a matched
null (see (d)); (ii) intervention criterion: projecting out those PCs moves
M2's predictions ≥Y% of the way toward clamped-M2's predictions while a
random same-dim subspace projection moves <Z%.
(d) (i) Smooth spatially-correlated perturbations concentrate in top PCs for
ANY network — "low-rank delta" is weak evidence without a null spectrum
(e.g., deltas from neighbor-shuffled inputs). (ii) Projection ablation can
degrade generically; the random-subspace control is mandatory, not optional.
(c) Hard dependency: if H-I2 finds clamped≈full (smoothing), deltas are ~0
and A4 is moot. Order: I2 → A4. Shares I2's clamping harness.

### H-A5 — metric flip is systematic (M3 trades RMSE for tails)
(a) **MIS-SPECIFIED — UNTESTABLE AS WRITTEN AT THIS TIER.** The A4-observed
RMSE inversion (M2 0.615 < M3 0.627) occurred in the UVTHETAW config. The
full-month data is uvtheta-ONLY (no w in published months — hard
constraint); the promised "falls out of the full-month rerun" therefore
tests a DIFFERENT config from the one where the phenomenon was observed. In
uvtheta, A4 showed NO inversion (M3 0.684 < M2 0.704). The kill criterion
("full-month RMSE ordering restores M3≤M2") would be evaluated on a config
where it already held at 2 snapshots. The uvthetaw inversion is stuck at
n=2 snapshots FOREVER at this tier unless w-inputs are sourced (ARCO-ERA5
Colab workaround — a real cost, not a screening freebie).
Demand: REWRITE as an honest uvtheta hypothesis ("in uvtheta, M3's RMSE edge
over M2 is small/unstable while its distributional edge is large/stable —
quantified on the month with bootstrap CIs") and RELABEL the uvthetaw
inversion as an explicitly snapshot-limited observation, or DEFER pending a
w-data acquisition decision. As it stands this entry promises what the data
cannot deliver.
(b) Moot until re-specified. (c) Piggybacks the same month run as A2/F1–F3.

### H-N1 — gates not degenerate (gatekeeper)
(a) Testable, near-free. Correct that it gates N2–N5 — run FIRST.
(b) "alpha ~ uniform/static" — cheap to sharpen, so sharpen: preregister
(i) spatial: variance of alpha vs random-init-net alpha variance (ratio
threshold); (ii) temporal: correlation of alpha across distinct inputs <
threshold (e.g., mean pattern corr < 0.95 across snapshots — a static mask
would sit at ~1); (iii) saturation: fraction of alpha within [0.05, 0.95].
(d) Note the random-init null is an EMPIRICAL null (random convs produce
structured alpha too); that's fine, but the comparison is then
"more/differently structured than random init," not "structured vs not." OK.

### H-N2 — fine-scale gates localize sources
(a) Cheap; has a null (distance-decay). Good instinct.
(b) "No excess over null" — with ONE checkpoint and heavy spatial
autocorrelation, a naive significance test is anticonservative. Preregister a
field-aware null (longitude-roll / phase-randomized surrogates of the alpha
map), not i.i.d. permutation.
(d) MISSING NULL: amplitude coupling. Gates tend to open where activations
are large; |flux| is large where |input wind/shear| is large; alpha–|flux|
correlation may be pure input-magnitude coupling with zero source-selectivity.
Add an input-magnitude-conditioned null (correlate alpha with |flux| AFTER
partialling out local |u|,|shear| — or match columns on input magnitude).
(c) Dependent on N1. Uses hotspot masks → S6.

### H-N3 — gate structure causally needed, scale-selectively
(a) Testable on the released M3 via forward hooks; cheap. One of the best
causal designs in the table.
(b) "No level-selective degradation" — preregister: kill if max-over-levels
hotspot skill drop < X% AND the drop profile across levels is flat within
bootstrap CI. Also preregister what replaces alpha (spatial mean per level —
fine; but note mean-alpha ≠ alpha=1 ≠ alpha=0.5; run the alpha=1 variant too,
it is one more hook config and disambiguates "gating matters" from "gating
level matters").
(c) Partially answers H-A3 for free — say so and let it carry that load.
Dependent on N1.

### H-N4 — gates track advection in time
(a) Budget: alpha time series needs an M3 forward per hour; a full July =
744 forwards — likely 30–60+ min (S7). Restrict to ≤1 week / hotspot
subwindows and it fits.
(d) FATAL-ISH CONFOUND, unaddressed: alpha is a deterministic function of the
inputs, and the INPUTS advect (the atmosphere advects). ANY input-dependent
field — including trivial ones like local wind speed, or the alpha of a
RANDOM-INIT net — will show lagged cross-correlation with advection. As
written this measures the atmosphere, not the mechanism. To have content it
must beat (i) a random-init-net alpha advection null and (ii) a trivial
input-functional null (e.g., advected |shear|). If it only matches those
nulls, the finding is "inputs advect" — worthless. My expectation is it will
not beat them, which is why this is a weakest-5 entry.
(b) "No coherent displacement signal" — vague; if kept, preregister the
cross-correlation statistic, lag window, and the two nulls above.

### H-N5 — FLAGSHIP: influence maps anisotropic along ray-traced propagation
(a) Testable in principle; budget unproven (20 targets x Jacobian/occlusion
on global M3 — time it, S7). Ray tracer is validated (A5 green) — good.
(b) Two nulls specified — best-nulled entry. But "beats both nulls" needs a
preregistered statistic: e.g., mean pattern correlation of influence maps
with ray footprints minus max(null correlations), bootstrap over the ~20
targets, threshold stated. With 20 targets and spatial autocorrelation,
naive p-values will flatter you; bootstrap over TARGETS, not pixels.
(d) Four confounds to handle in the prereg:
(i) Ray footprints depend on ASSUMED launch spectra (phase speeds,
directions) — analyst degrees of freedom; preregister the spectrum and show
footprint robustness to it, or the test can be tuned into significance.
(ii) Gradients are w.r.t. NORMALIZED inputs — per-level sigma scales the
Jacobian rows; for any cross-level statement, de-normalize (S3).
(iii) H-P4's seam: zero-padded lon convs give M3 position-dependent influence
near lon 0/360 that could masquerade as anisotropy — run P4 FIRST and exclude
near-seam targets (dependency: P4 → N5). Pole edges likewise.
(iv) Advection and group propagation are correlated (waves ride the flow);
with 20 targets the advection null and ray null may be statistically
inseparable — preregister the discriminating statistic (e.g., meridional
displacement components, where advection and propagation decorrelate most).
(c) Shares the occlusion harness with I4. Depends on hotspot targets (S6).

### H-L1 / H-L2 — transfer-learning mechanism
(a) **BLOCKED — both gated on assets that are explicitly UNVERIFIED.** The
iccs_coupling_checkpoints have not been downloaded/verified (hard
constraint), and the OSF IFS files are a 9 GB acquisition that collides with
the disk budget (S8). Additional unexamined risks the table skips:
(i) LINEAGE: weight-delta analysis (L1) is meaningless unless the released
TL checkpoint is verifiably fine-tuned FROM the exact released ERA5 base we
hold (same init, frozen layers) — if the TL run started from a different
base or epoch, "deltas" compare unrelated weights. Verify lineage from the
TL scripts/checkpoint metadata BEFORE any analysis. (ii) ARCHITECTURE: the
coupling checkpoints are L93 — a different vertical grid than the ERA5
models; the loader may not even apply. (iii) IFS normalization constants —
which, from where?
(b) Kills are double-vague ("recovers most", "falls well short") — no
numbers. If/when unblocked, preregister recovery fractions.
(c) Salvageable cheap core: SPLIT L1 into L1a (weight-delta
diagonality/SVD — needs ONLY the checkpoints, ~zero compute, no IFS data)
and L1b (gain-fit evaluated on IFS data — needs the 9 GB). L1a can screen the
moment checkpoints are verified; L1b and L2 should be DEFER with an explicit
asset-verification gate. Do not let "asset-check first" quietly become
"asset-check never happened but we screened anyway."

### H-F1 — failures co-occur (dual-outcome)
(a) Cheap piggyback. Fine as a MEASUREMENT — but it is not a hypothesis: kill
"n/a" means it cannot fail, and the table's own legend promised falsifiable
statements. Relabel as a descriptive measurement so the count "29 hypotheses"
stops overclaiming (it is really 27–28 falsifiable + measurements).
(d) ">> chance" is broken as stated: chance for top-1% overlap between
spatially autocorrelated error fields that share the SAME heavy-tailed target
is NOT 1%. All three models will fail where |flux| is extreme — overlap is
guaranteed high, and "data-limited" is baked in. Fix: condition on per-column
difficulty (overlap of DIFFICULTY-ADJUSTED errors, e.g., residuals after
regressing error on |true flux|), plus a spatial-null (rolled-field) overlap
baseline. Without this, F1's conclusion is predetermined.

### H-F2 — M1's worst transients are nonorographic; M3 fixes them
(a) Cheap piggyback; needs hotspot-type labels → S6.
(b) "No type asymmetry" — preregister a ratio threshold (e.g., M1→M3 error
reduction in nonorographic hotspots ≥ 1.5x that in orographic ones).
(d) TWO confounds: (i) S5 in its sharpest form — M1/M2 receive zs, M3 does
not; "M3 fixes nonorographic best" is exactly the input-asymmetry
prediction, no nonlocality needed. The clean version of this hypothesis is
M1→M2 (same inputs); the M1→M3 version is confounded BY DESIGN and must be
labeled so. (ii) Orographic vs nonorographic hotspots differ in flux
magnitude/variance; "type" asymmetry may be amplitude asymmetry — match or
normalize by local flux variance.

### H-F3 — tail compression drives the Hellinger ordering
(a) Cheap piggyback. Fine.
(b) Two problems: (i) "Hellinger-vs-tail decomposition" is not a defined
method — no such standard decomposition exists; specify the actual
computation (e.g., recompute Hellinger with tails winsorized at P99 and
report the ordering's sensitivity) BEFORE screening or the analysis will be
invented post hoc. (ii) "tails calibrated or unrelated" — vague; preregister
thresholds.
(d) S3 with teeth: P99 ratios in cube-root space vs physical space differ
mechanically (cube root compresses tails); the paper's Hellinger is computed
in — which space? Pin BOTH to preregistered spaces or the "decomposition"
mixes spaces invalidly. Note P99 estimates from one month, one checkpoint:
bootstrap the tail ratios; and area-weight (tropical columns dominate counts).

### H-P1 — critical-level filtering learned
(a) Testable, cheap forwards. S9 applies in full: synthetic columns are OOD;
negatives ambiguous. The "M2/M3 with uniform context" arm is maximally OOD
for M3 and should be CUT from screening (M1-only screen; real-profile-grafted
variant as primary evidence).
(b) "No systematic suppression" — preregister: flux-above-z_c / flux-below-z_c
ratio for reversal vs matched no-reversal profiles, kill if ratio difference
< preregistered margin across ≥N profile families. Dual-outcome framing is
ACCEPTED here (directional physical prior, explicitly justified — this is
what F1 failed to do), but S9 weakens the negative branch: a negative on
purely idealized profiles is NOT "trust-critical violation," it is
"idealized-OOD failure" — only a negative on REAL profiles with grafted
reversals carries the publication-worthy trust claim. Preregister that
distinction now, or the negative will be oversold later.
(d) Verify the generator normalizes with the correct month's constants (S4).

### H-P2 — orographic source response rotates with wind
(a) Testable for M1/M2 (they receive zs). But the entry's mental model is
suspicious: M1 sees zs as a per-column SCALAR — there is no "ridge" in its
input, only local elevation plus the column profile; the wind-rotation test
is really "flux vector direction tracks low-level wind direction at fixed
zs/lat/lon." Fine and worth testing — but drop the "idealized ridge" language
(it implies spatial structure M1 cannot see; for M2 the 3x3 stencil could
encode ridge orientation only if zs varies per stencil cell — VERIFY from the
dataloader whether M2's stencil includes per-cell zs before claiming an
orientation test; if zs is center-only, M2's version collapses to M1's).
(b) "No coherent rotation" — preregister circular correlation between wind
bearing and predicted surface-flux bearing, threshold, and the physical
expectation (drag OPPOSES surface wind: correlation near −1 in bearing terms).
(d) S9 (OOD) applies; use real high-orography columns with rotated winds as
the primary variant. S4 (constants) applies.

### H-P3 — amplitude scaling in/out of distribution
(a) Cheap sweep. Fine.
(b) SPLIT the claim: the in-distribution half ("monotone, superlinear within
training range") is falsifiable — preregister monotonicity fraction and the
superlinearity exponent test. The OOD half ("saturates/breaks beyond") is
NOT falsifiable — there is no ground truth OOD, and ANY behavior
(saturation, blow-up, flatness) can be narrated as "degrading." Relabel the
OOD half as descriptive envelope-mapping, not a hypothesis. "Flat or erratic"
kill: "erratic" undefined — preregister.
(d) S9: "within training range" must be defined in NORMALIZED input space
against the actual July/August input distribution, not by physical intuition.

### H-P4 — longitude seam artifact in M3
(a) Cheap, piggybacks month run, and the causal check (roll the input,
watch the dip move) is genuinely decisive — M3 receives no lat/lon, so
rolling is a clean intervention with rolled truth targets. Best
value-per-compute in the table.
(b) Nearly sharp; finish it: preregister the seam-window width (the 4x8
bottleneck means the receptive field is huge — the "dip" may be tens of
degrees wide and shallow; a ±2-cell window would miss it) and the statistic
(error vs longitude, seam-window mean vs rolled-control same-window mean,
bootstrap over time).
(d) Also check the POLE edges (same zero-pad logic, meridional direction) —
free while you're there; see Missing M2 below.

---

## MISSING (cheap, high-value, absent from the table)

**M-1. Epoch-pair robustness control (free).** The strato configs have TWO
released epochs each (M1 e88/e100; M2 e38/e93) — the only multi-checkpoint
configs in existence. Run the cheapest probe (R1-style) and one ablation
(I2-style) on both epochs of one config and report statistic stability. This
is the ONLY empirical handle on training-run noise available at Tier 1 (S2),
it costs minutes, and no entry uses it. Add it.

**M-2. Pole-edge artifact twin of P4.** The neighborhood constructor
zero-pads beyond the poles and M3's convs zero-pad latitude edges; an
error-vs-latitude edge analysis is free alongside P4 and closes the "boundary
artifacts" story in one figure instead of half of it.

**M-3. Vertical influence structure of M1 (input-level → output-level
Jacobian).** One backward pass per output level on a column batch; directly
comparable to Pahlavan's ERF results (the closest prior art, which MUST be
engaged) and calibrates the T4 machinery on the cheapest model before N5
spends it on M3. Novelty care: frame as replication-then-extension
(horizontal is ours; vertical is theirs — cite arXiv:2407.05224).

**M-4. Evaluation hygiene hypothesis: area-weighting changes the story (or
not).** All A4 metrics were (apparently) computed unweighted over a lat-lon
grid — poleward columns are overrepresented ~cos(lat)^-1. One afternoon:
recompute the A4/month orderings area-weighted. If orderings flip, EVERY
downstream composite inherits it. If not, one sentence kills the concern.
Either way it must be settled before Stage C, and no entry mentions it.

**M-5. The clamped-M2-vs-M1 identity check** (noted under I2): on constant
stencils M2's conv reduces to a linear center map — comparing clamped-M2 to
M1 directly tests whether M2 "contains" an M1 and makes the I2 result
interpretable in absolute terms. Free.

---

## RANKINGS

**WEAKEST 5 (defer or rewrite before Stage C):**
1. **H-A5** — mis-specified: cites a uvthetaw phenomenon, tests uvtheta;
   the confirming data (w months) does not exist at this tier. Rewrite or defer.
2. **H-A3** — a 6-run training study inside a 20-min screening budget;
   reduced-scale result wouldn't transfer to the released M3 anyway; H-N3
   answers the causal half free. Defer to Stage D.
3. **H-L2** — blocked on unverified TL checkpoints + 9 GB IFS data + lineage
   and normalization unknowns; not screenable this stage. (H-L1: same block,
   but salvage L1a weight-delta part once checkpoints verify.)
4. **H-N4** — as designed it measures that the ATMOSPHERE advects, not that
   the network does anything: any input-dependent gate passes; needs two
   nulls it doesn't have, and will likely not beat them.
5. **H-R4** — unfalsifiable coherence criterion ("nontrivial fraction",
   human judgment, unbounded sparsity sweep) and over budget; a Stage-D
   deep-dive masquerading as a screen.
   (Dishonorable mentions: H-F1 is a measurement, not a hypothesis — relabel;
   H-R2 is near-guaranteed-positive without a spatial probe split — fix
   before running.)

**STRONGEST 5 (screening value per compute):**
1. **H-P4** — free from the month run, clean causal roll test (M3 has no
   lat/lon inputs), decisive either way, standalone engineering finding.
2. **H-N1** — minutes of compute, gates four downstream hypotheses; either
   outcome redirects the whole B-ATTN family. Run first.
3. **H-I2** — one clamped forward pass separates "context" from "smoothing";
   logical prerequisite for I1 and A4; with M-5 it's interpretable absolutely.
4. **H-A1** — closed-form ridge with a numeric threshold calibrates how deep
   the entire nonlocality narrative needs to be; add the 2x2 baselines.
5. **H-N3** — causal gate ablation on the released model; level-resolved;
   absorbs most of H-A3's value at ~zero cost.
   (H-N5 is the highest-VALUE entry but not highest value-per-compute at
   screening: it depends on P4, hotspot boxes (S6), a preregistered launch
   spectrum, and unmeasured Jacobian costs. Screen it AFTER P4/N1 with a
   trimmed target set.)

**UNTESTABLE AT THIS TIER (as currently written):**
- **H-A5** — the uvthetaw inversion cannot be tested on uvtheta-only months;
  w-bearing data beyond 2 snapshots does not exist in our holdings.
- **H-L1/H-L2** — conditionally untestable: blocked until TL checkpoints are
  downloaded, lineage-verified against our exact base checkpoints, and IFS
  data acquired within a disk budget that currently doesn't close (S8).
- **H-A3** — untestable within the screening budget (not at the tier per se;
  Tier-1's retrain clause is a Stage-D supplement, not a screen).
- **P1/P2 M3-with-uniform-context arms** — runnable but uninterpretable
  (maximal OOD for a U-Net); results from those arms should be inadmissible.

---

## DEMANDS BEFORE STAGE C (actionable, in order)

1. Resolve S1: pre-commit the screening/confirmation data split (second month
   or declared within-July split) BEFORE any screening run; log it.
2. Resolve S8 jointly: decide second-month vs IFS download; my recommendation
   is second month (S1 protects everything; L1/L2 protect two entries).
3. Rewrite H-A5 (uvtheta-honest version); relabel the uvthetaw inversion as
   snapshot-limited. Defer H-A3, H-R4, H-L2 (and H-L1b) with explicit
   unblock conditions. Relabel H-F1 as measurement.
4. Fix H-R2 (spatial probe split + margin) and H-R5 (input baseline) before
   they run; both are otherwise guaranteed-positive probe traps.
5. Publish the measured cost table (S7) and re-scope I3, I4, N4, N5, R4
   against real timings.
6. Add preregistered numeric kill thresholds where flagged (I1, I2, R1, R2,
   R3, R5, R6, A2, A4, F2, F3, N1–N4, P1–P3, P4 window) — a kill criterion
   without a number is a negotiation, not a criterion.
7. Verify hotspot boxes (S6) and the July scaling constants (S4); add the
   free controls M-1 (epoch pairs), M-4 (area weighting), M-5 (clamped-M2 vs
   M1); adopt M-2, M-3 as riders on existing runs.
8. Enforce dependency ordering in the screening schedule:
   N1 → {N2, N3, N4}; I2 → {I1, A4}; P4 → {N5, I4 target selection};
   A1 → I3; asset-verify → L1a.

— CRITIC, 2026-08-18. Objections stand until each is answered in writing in
the RESEARCH_LOG (append-only), not by silent table edits.
