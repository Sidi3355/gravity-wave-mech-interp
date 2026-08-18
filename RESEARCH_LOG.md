# RESEARCH_LOG.md — append-only

Program: Mechanistic Interpretability of Deep Learning Gravity Wave Parameterizations
Anchor: Gupta, Sheshadri, Roy, Anantharaj (2025), JAMES, 10.1029/2025MS004977

Rules: append-only; every hypothesis, screening result, kill decision, pre-registration,
deviation, and failure is logged with timestamp and reasoning. History is never edited.

---

## [2026-08-18 16:47] LEAD — Program start

Session begins from an empty directory. Roles instantiated (LEAD, DATA ENGINEER,
EXPERIMENTALIST, PHYSICS VERIFIER, CRITIC, WRITER); CRITIC will always run as a
distinct adversarial pass. Stage A begins now: environment, asset discovery,
replication gates.

## [2026-08-18 16:48] DATA ENGINEER — Environment survey

Hardware/software found:
- Windows 11 Home, Python 3.11.9 (Microsoft Store build), pip 24.0.
- PyTorch 2.6.0+cpu already present; numpy 2.2.2; xarray absent.
- **No NVIDIA GPU** (nvidia-smi absent; Intel Iris Plus integrated graphics only).
- CPU: Intel i7-1065G7 (4 cores / 8 threads, low-power laptop class). RAM: 7.8 GB.
  Disk: 148 GB free on C:.
- No conda. No ~/.cdsapirc (no Copernicus CDS credentials for direct ERA5 download).

Compute assumptions decided (per master prompt §9, degrade-to-CPU clause):
- ALL training and interp experiments sized for CPU: regional patches (target 32–64
  grid cells square), small channel counts, minutes-to-tens-of-minutes runs, hard cap
  ~2 h with checkpoint/resume.
- Dataset budget: ≤ ~30 GB on disk; prefer public no-credential mirrors
  (ARCO-ERA5 / WeatherBench2 zarr on GCS) or the anchor's released data if it exists.
- RAM budget 7.8 GB forces chunked/zarr-backed pipelines; never load full fields.

Deviation from master prompt A1: no conda available, so version pinning goes into
`requirements.txt` + a stdlib `venv` instead of `environment.yml`. Reason: conda not
installed; installing miniconda adds risk/time with no scientific benefit. Logged as
a deviation, functionally equivalent pinning.

Two placement decisions (OneDrive constraint): the project directory is inside
OneDrive; a venv (~10k small files) and multi-GB datasets would thrash sync. The venv
lives at `C:\Users\sidi0\venvs\gwmi`, bulk data at `C:\Users\sidi0\gwmi_data\`
(both outside OneDrive, referenced via config), with paths recorded in configs and
README. Code, configs, results metrics, and figures stay in-repo and git-tracked.

## [2026-08-18 16:50] LEAD — Stage A discovery plan

Launching a parallel discovery workflow (8 agents) covering:
1. Deep read of the anchor paper (architectures, data, metrics, availability stmt).
2. Code/weights hunt (GitHub, Zenodo; authors' orgs; DataWave project).
3. WINDSET dataset hunt (Sci Data 2024 paper; DOI; hosting; sizes).
4. Pahlavan et al. 2024 verification + 1D QBO testbed code hunt.
5. Lineage verification (Espinosa 2022, Hardiman, Wang/Yuval/O'Gorman 2022,
   Fritts & Alexander 2003).
6. Gupta-group related work (arXiv:2406.14775 for M1/M2/M3 specs; arXiv:2509.03816).
7. Novelty scan: existing interpretability work on climate/weather surrogates.
8. ERA5 practical access paths without CDS credentials.

Every URL must be fetched to count as verified; unfound assets are recorded as such
(honesty protocol §10). Output feeds ASSETS.md and the tier decision (A3).

## [2026-08-18 17:10] LEAD — Session interruption and resume

The host process exited mid-Stage-A, killing the discovery workflow (6/8 agents
started, none finished) and the venv package install (torch 2.13.0+cpu landed;
rest cut off). Both relaunched; workflow resumed from run wf_1b17b2c2-deb.
Physics baselines (raytrace.py, profiles.py, tests/test_physics.py) were written
before the interruption; tests pending until numpy lands in the venv.
Note: venv torch is 2.13.0+cpu (latest at install time), not system 2.6.0 —
version will be pinned in requirements.txt after install completes.

## [2026-08-18 18:30] PHYSICS VERIFIER — A5 gate: ray tracer fixed, all tests green

First full run of tests/test_physics.py (previous session never got to run them):
11/14 passed, 3 failed. Diagnosis: trace_ray still integrated in TIME (RK4, t_max
cutoff) while the module docstring already specified the intended HEIGHT-based
scheme. Time stepping is stiff near critical levels — cg_z ~ omega_hat^2, so the
approach time diverges physically — and rays timed out with status "ok" instead
of stalling and being flagged "critical_level". The hydrostatic mountain-wave case
(cg_z ≈ 0.063 m/s) confirmed it exactly: predicted stall height 1e3 + cg_z*t_max
= 2256 m matched the observed failure value 2256.6 m.

Fix: rewrote trace_ray to integrate in height, as the docstring specified. m(z) is
algebraic from the dispersion relation; x, y, t follow quadratures dx/dz = cg_x/cg_z
etc., Simpson's rule per dz step (default dz = 50 m). Termination checks run at every
Simpson evaluation point. Tests updated ONLY in the integrator kwargs they pass
(dt/t_max -> dz); every physics assertion is unchanged. Result: 14/14 pass, suite
runtime 48 s -> 0.5 s. A5 physics gate GREEN: ray tracer + profile generators now
validated against analytic cases (uniform-flow propagation, hydrostatic mountain
wave, critical-level refraction and absorption, evanescence, cone rotation
invariance, isothermal N^2, Ri, density scale height).

Also this session: venv install completed and pinned to requirements.txt
(torch 2.13.0+cpu, numpy 2.4.6, xarray 2026.7.0, zarr 3.1.6, full list in file);
data root C:/Users/sidi0/gwmi_data created. Stage A discovery relaunched as 7
parallel background agents (anchor deep-read; code/weights hunt; WINDSET hunt;
Pahlavan+QBO testbed; lineage citation verification; novelty scan; ERA5 access
paths) — the prior session's workflow died without writing any output. Two agents
hit connection errors just before writing findings; both resumed with context
intact (their intermediate artifacts, including a cloned QBO-1D repo and the
Pahlavan full text, are on disk).

## [2026-08-18 18:55] LEAD — Stage A discovery complete; TIER 1 DECISION

All 7 discovery agents returned (full reports in results/stageA_discovery/,
inventory in ASSETS.md). Highlights:
- Anchor's code (MIT, Zenodo 10.5281/zenodo.16415113) AND all 12 released
  checkpoints (M1/M2/M3 x uvtheta/uvthetaw x global/stratosphere_only, HF
  amangupta2/nonlocal_gwfluxes) verified and downloaded. Two training-format
  test files (hourly 2015) included with exact normalization constants.
- Training data distribution (WxC-Bench monthly files, ~15-21 GB/month) public
  on HF; 1-2 months affordable if retraining controls are needed.
- Architectures read from code: M1/M2 share ANN_CNN (6 hidden LeakyReLU
  layers; M2 prepends one valid 3x3 Conv2d, channel-preserving); M3 is an
  Oktay-style Attention U-Net (64->1024 ch, 4 down/up levels) whose
  "attention" is sigmoid GATING on skip connections — NOT self-attention.
  B-ATTN hypotheses must target gate fields alpha(x,y) at 4 scales; M3
  nonlocality lives in the encoder-decoder hierarchy (global at 4x8
  bottleneck). Paper-vs-code discrepancy noted: arXiv:2406.14775 says
  "4-layer" MLP; code has 6 hidden layers. Code + checkpoints are ground truth.

DECISION (A3): **Tier 1** — pure interpretability study on the actual released
models. Reasoning: released checkpoints + released eval data + released code
= zero replication risk of the models themselves; our CPU-only machine is far
better matched to inference-heavy interp than to training. Supplements:
(a) Tier-2-style reduced-scale retrains remain available (WxC-Bench months)
for seed-robustness controls and architecture ablations where a claim needs
them; (b) qbo1d testbed (CPU-trivial, differentiable) available as a
mechanistically-transparent complement if needed.

A4 ADAPTATION (logged deviation): the master prompt's "3 seeds per model"
replication gate assumed we train the ladder ourselves. At Tier 1 there is
exactly one released checkpoint per configuration, so the gate becomes:
evaluate released M1/M2/M3 (global uvtheta + uvthetaw variants) on the
released held-out 2015 test data and confirm the paper's skill ordering
M3 <= M2 <= M1 on RMSE and Hellinger distance. Seed variance enters later
via reduced-scale retrain controls only where a Stage-D claim depends on it.

NOVELTY REFRAME (from agent 6): the broad claim "mech interp of climate
surrogates is unexplored" is FALSE as of 2025-26 (SAEs+steering on GraphCast
arXiv:2512.24440; probes on Aurora arXiv:2511.07787; SAEs on Walrus
arXiv:2606.11657; CKA GraphCast-Aurora arXiv:2605.23778). The NARROW claim
survives and is verified: no probing, SAE, causal-intervention, or
representation-level interp exists for GW (or any subgrid) parameterization
emulators; deepest GW-specific work is SHAP (Haslauer/Eyring arXiv:2605.05052)
and kernel-Fourier (Pahlavan 2024) + gradient ERF (Pahlavan 2025, overlaps our
T4 -> cite). Framing: "first mechanistic interpretability study of a
parameterization emulator, on the actual released models of the anchor paper."
The window is PERISHABLE (Eyring group active on GW-NN explainability).

OPEN GAP: anchor PDF itself unreadable by agents (Cloudflare). All technical
facts recovered from code/data/companions. Flagged for optional manual browser
download by the human (CC BY, free) to verify hotspot boxes + limitations
wording before paper writing.

Next: A4 replication gate on the released checkpoints.

## [2026-08-18 19:40] EXPERIMENTALIST — A4 replication gate, first pass (released test snapshots)

Infrastructure built and gated first:
- Vendored the release-tag model code (src/models/anchor/, provenance header).
- Checkpoint forensics: released ANN checkpoints contain bnorm1..6 entries that
  the current release code lacks. Resolved via the author's paper-era dev repo:
  ANN_CNN DEFINED BatchNorm1d modules but never applied them in forward()
  (calls commented out; bnorm4 even has an impossible width 2*hdim=2952 that
  would crash if used). Empirical proof: all num_batches_tracked == 0. Loader
  (src/models/anchor_loader.py) strips them after asserting untrained, loads
  the rest strictly, infers dims from checkpoint shapes. Discovered en route:
  the Attention UNet is trained WITHOUT the 3 scalar inputs lat/lon/zs
  (dataloader comment confirms) -> M1/M2 see orography, M3 does not. Logged as
  a confound to respect in cross-architecture comparisons.
- tests/test_anchor_loader.py: all 6 global checkpoints load with exact
  key/shape agreement, finite outputs, deterministic eval. 7/7 green.
- Released 1x1 and 3x3 test files verified to contain bitwise-identical
  instants (Aug 1 2015, hours 5089-5090 of year; "scaling08" = month 8).
  File-attr level metadata is stale strato text; channel semantics taken from
  dataloader code. idim/odim coords are fill values.

RESULT (experiments/01_replication_a4.py, 2 snapshots = 16,384 columns,
results/a4_replication/metrics.json):
  uvtheta : RMSE 0.796 (M1) > 0.704 (M2) > 0.684 (M3); R2 .18/.36/.39;
            Hellinger(uw) .111/.091/.052, (vw) .127/.091/.068  -> FULL ordering PASS
  uvthetaw: RMSE 0.732 (M1) > 0.615 (M2) < 0.627 (M3)  -> M2/M3 RMSE INVERTED;
            Hellinger(uw) .084/.055/.035 -> ordering PASS; R2 .31/.51/.49
Verdict: qualitative replication PASSES (M1 clearly worst on every metric;
distributional ordering M3<=M2<=M1 holds 4/4). The uvthetaw M2-vs-M3 RMSE
inversion (1.8% relative, 2 snapshots) matches the paper's own "M2 close
second" and is not called either way at this sample size. Action per master
prompt: (a) full-month confirmation queued — July 2015 (scaling07) WxC-Bench
month downloading (~14.9 GB; u,v,theta only, so confirmation covers the
uvtheta trio; uvthetaw stays snapshot-only and will be reported as such);
(b) "metric- and sample-dependence of the M2/M3 ranking" promoted to a
Stage-B hypothesis seed (B-ARCH family).

Enabler for the month eval: published months exist only in 1x1 format, so M2
needs reconstructed neighborhoods. src/data/neighborhoods.py reverse-engineers
the stored convention (centered 3x3, lon wrap, ZERO padding beyond poles) and
tests/test_neighborhoods.py verifies bitwise equality against the released 3x3
file at both timesteps. Gate green before use.

## [2026-08-18 19:35] LEAD — Response to CRITIC pass (results/stageB_critic/critic_pass.md)

Each objection answered in writing per protocol; table amendments appended to
HYPOTHESIS_TABLE.md (statuses changed openly, history in git).

S1 (H-A5 mis-specified: RMSE inversion is uvthetaw-only, months are
uvtheta-only). ACCEPTED. H-A5 REWRITTEN: full-month test now targets the
testable core — variance/tail calibration differences between M2 and M3 in the
uvtheta setting (dry run already shows M2 variance ratio 0.53 vs M3 1.10 on
snapshots). The uvthetaw inversion itself is demoted to "unresolved,
snapshot-only observation" in all future reporting.

S2 (double-dipping July). ACCEPTED. PRE-COMMITTED SPLIT, logged now, before
any screening run: July 2015 (scaling07) = SCREENING ONLY. Stage-D
confirmations use January 2015 (scaling01, download deferred to Stage D).
Disk budget revised 30 -> 35 GB to accommodate (148 GB free; OneDrive
unaffected). IFS data for B-TL, if pursued after asset verification, will
displace the July file (screening artifacts are small and kept; raw month is
re-downloadable). 

S3 (H-A3 training study 10-100x over screening budget, result would not
transfer to released M3). ACCEPTED. H-A3 -> DEFER (Stage-D option at most,
only if gate-ablation evidence from H-N3 makes it decisive); its causal half
is covered free by H-N3.

S4 (guaranteed-positive probe traps). ACCEPTED. H-R2 amended: spatial
train/test split (disjoint longitude sectors), must beat the raw-input probe
baseline by a pre-set margin (see amendments), plus time-shuffled control.
H-R5 amended: input-probe baseline mandatory; claim only depth-DIFFERENTIAL
decodability (sign-vs-magnitude divergence), not raw decodability.

S5 (unsharp kills; no cost audit). ACCEPTED. (a) Numeric kill thresholds
added for every Stage-C-runnable hypothesis (amendments section). (b) Cost
table measured and committed (results/cost_table.json): M1 0.82 s, M2 1.37 s,
M3 0.19 s per timestep fwd; M3 Jacobian row 0.56 s; hooks ~free; M3 acts
16 MB/timestep. Full-month eval ~29 min. Re-scoped budgets for I3/I4/N5/R4
noted in amendments; nothing exceeds 20 min except the (already sanctioned)
full-month artifact run.

S6 (cross-cutting confounds). ACCEPTED with one REBUTTAL:
- Metric-space rule declared: RMSE/R2-type metrics in NORMALIZED space
  (training-loss space); distributional/tail metrics in PHYSICAL space.
  Every results file states its space. 
- Normalization-constant equality across files is now a hard assertion in
  experiments/02 (run aborts on mismatch).
- M3-lacks-zs contamination of F2/A2/A5: interpretation notes added — any
  M1-vs-M3 regime difference may reflect input-set differences, not
  architecture alone; M2-vs-M1 comparisons (same inputs) carry the clean
  architectural contrast.
- REBUTTAL on hotspot boxes: they do NOT come from the unread PDF; they are
  hard-coded in the released dataloader (region registry, verified in code)
  and cross-validated against companion papers by the deep-read. Usable for
  screening. The unread-PDF gap affects only paper-wording verification.
- Epoch-pair robustness control ADOPTED (new C-1): strato uvtheta pairs
  (M1 ep88/100, M2 ep38/93) re-run for any probe/feature finding that reaches
  Stage D, as the only free multi-checkpoint robustness check at Tier 1.

S7 (weakest five). H-A5 rewritten (above); H-A3 deferred; H-L2 DEFER pending
asset verification (H-L1 reduced to its weight-delta component L1a, runnable
after checkpoint verification, before committing to IFS data); H-N4 DEFER
unless H-N1 passes AND the new input-displacement null (alpha displacement vs
displacement of the input fields themselves) is implementable within budget;
H-R4 re-scoped to a quantitative screening gate: recon/sparsity frontier +
dead-rate + spatial-autocorrelation of feature maps vs a random-dictionary
null (numeric), with human-legible dashboards deferred to Stage D. H-F1
relabeled [MEASUREMENT] (still run; informs F-family interpretation).

S8 (strongest five: P4, N1, I2, A1, N3). ADOPTED as Stage-C wave 1, with
dependency ordering (S12): N1 gates N2/N3(/N4); I2 gates I1/A4; P4 informs
N5/I4 target placement; A1 gates I3; asset-verify gates L1a.

S9/S10 (T7 external validity). ACCEPTED. All synthetic-input screens must
report an OOD score (per-channel z-score envelope vs training distribution)
and include real-profile-GRAFTED variants (modify one physical aspect of a
real column) as the primary evidence; fully synthetic profiles are secondary.
P1's "publication value either way" claim withdrawn for OOD-ambiguous
negatives. M3/M2 uniform-context arms of P1/P2 dropped (inadmissibly OOD).

S11 (missing controls). ADOPTED as C-1..C-5: epoch-pair robustness (C-1);
area-weighted metric recheck (C-2 — already implemented in experiments/02);
clamped-M2-vs-M1 prediction-identity diagnostic inside H-I2 (C-3); pole-edge
twin of P4 (C-4, zero-padded lat edges); M1 vertical-Jacobian calibration
against Pahlavan-style ERF prior art (C-5).

Verdict: table amended, 27 active + 2 deferred-pending-assets + 2 deferred;
Stage C may begin once the July artifacts run completes.

## [2026-08-18 20:05] DATA ENGINEER — DISCOVERY: WxC-Bench monthly files are not
## checkpoint-compatible as stored; exact conversion built and validated

The critic-mandated constants gate fired on the July 2015 month file. Forensics:
1. The released checkpoints expect ONE fixed normalization convention — that of
   the author's HF test files: u,v mu-sigma scaled with a 3x factor
   (u: (u-6.3955)/66.527), theta/1000, fluxes cube-root scaled with global
   constants (uw sigma 0.005077).
2. The WxC-Bench nonlocal_parameterization 2015 monthly files use PER-MONTH
   constants AND a 1x-sigma u/v convention, despite attribute text still
   claiming "3x<sigma>" and "[-2,2]" (July u_norm std is 1.35, not ~0.33).
   Outputs use per-month sigmas with honest labels (verified via the
   adjacent-hour test: July hour 5088 vs Aug snapshot hour 5089 are
   consecutive; converted slopes/std-ratios ~1).
3. Feeding July data as-is: R2 of -57..-81 (M1), -4.1..-4.4 (M3). After exact
   affine conversion to the model convention: M1 R2 ~ 0.17, M3 R2 ~ 0.38 —
   matching the Aug-snapshot baseline. Conversion is exact (float64 to ~1e-9;
   float32 storage floors near-zero cube-root values at ~1e-4 normalized
   = ~5e-15 Pa, negligible).
COMMUNITY-RELEVANT: anyone evaluating the HF checkpoints on WxC-Bench monthly
files without this conversion gets garbage. Candidate for a data note /
upstream issue after the paper.

Implementation: src/data/normalization.py (constants registry, convention
detector that fails loud on unknown files, exact converters);
tests/test_normalization.py (4 tests: registry parse, roundtrip exactness,
adjacent-hour physical consistency, zs channel identity across files).
All experiments (02/03/04) auto-detect and convert. Suite 39/39 green.

Also CONFIRMED from the released training script (era5_training/training.py):
train years = 2010, 2012, 2014 (all 12 months each); test years = 2015 (all
months). So July 2015 and January 2015 are both legitimately held out, with
the standard caveat that released checkpoint epochs were presumably selected
on 2015 validation loss. utils/files.py contains a dev-leftover list that
includes 2015 months in training — superseded by training.py's explicit
year arrays; noted to avoid future confusion.

Launching the full-month A4 confirmation now (~35 min with conversion).

## [2026-08-18 21:15] EXPERIMENTALIST — A4 FULL-MONTH CONFIRMATION: GATE GREEN

July 2015, all 744 held-out hourly timesteps, uvtheta trio, WxC->model
conversion applied (results/a4_fullmonth/metrics.json):
- RMSE(norm): M3 0.6775 < M2 0.6980 < M1 0.7844. Area-weighted: same ordering
  (0.6764/0.6958/0.7882) -> not weighting-sensitive (C-2 satisfied).
- Hellinger(phys): uw 0.0399/0.0493/0.0644, vw 0.0433/0.0970/0.1264 — full
  ordering M3 < M2 < M1 on both components.
- Paired per-timestep M2-minus-M3 RMSE: mean +0.0204, bootstrap 95% CI
  [0.0202, 0.0207], M3 better in 744/744 timesteps. The 2-snapshot uvthetaw
  RMSE inversion (logged 2026-08-18 19:40) does NOT generalize to the uvtheta
  setting at month scale; it remains an unresolved snapshot-only observation
  for uvthetaw (untestable at month scale, no public w months).
- H-A5 (rewritten) preview: physical variance ratios M1 0.360, M2 0.363,
  M3 1.066. The single-column and 3x3 models under-disperse flux variance
  ~2.75x; the UNet is nearly variance-calibrated. Formal H-A5 verdict when
  tail ratios are extracted from the histograms.
A4 (Tier-1 adapted) REPLICATION GATE: PASS. Stage A fully complete.

Ops note: the background run was killed once and hit a OneDrive file-lock
PermissionError on the state file at t=743; save_state now retries and falls
back to non-atomic write. Resume-from-checkpoint worked as designed both times.

Stage C wave 1 launching now (pre-registered thresholds in HYPOTHESIS_TABLE
amendments): H-N1, H-I2, H-N3, C-5, H-P4, H-A1 on July data.

## [2026-08-18 21:40] EXPERIMENTALIST + LEAD — Stage C wave 1 complete: 4 pass, 1 kill, 1 control

All on July 2015 (screening month), pre-registered thresholds (table
amendments 2026-08-18). Full metrics in results/screen_*/metrics.json.

H-N1 (gates degenerate?) SCREEN-PASS. Trained-vs-random-init ratios at the
finest gate: spatial structure 35x, temporal input-sensitivity 20x; mean
alpha 0.34-0.64 (no saturation). Structure INCREASES with gate coarseness
(spat_std 0.10 -> 0.28). B-ATTN family unlocked.

H-I2 (context = smoothing?) SCREEN-PASS, decisive. Clamped-neighborhood M2:
RMSE 0.845 > M1 0.784 (recovery of the M1->M2 gap = -0.70; per-timestep
range -0.83..-0.55, 24/24 consistent). Horizontal structure is INFORMATION,
not averaging; M2 has no local fallback mode. C-3: corr(M2clamp, M1) = 0.56.
Caveat (logged): uniform patches are mildly OOD for M2; the negative recovery
overstates information loss by an unknown margin — the kill test is one-sided
against this caveat (it could only have inflated recovery, not suppressed it).

H-N3 (gates causally needed?) SCREEN-PASS with clean scale gradient:
flattening alpha to spatial mean costs global RMSE +4.3% (finest gate),
+1.4%, +0.3%, +0.1% (coarsest). The causal action is at full resolution.
NOTE: hotspot delta (+2.4%) < global (+4.3%) — the finest gate matters
BROADLY, not just at hotspots; refines H-N2's expectation.

H-P4 (longitude seam) SCREEN-PASS, story refined by the causal test.
Month-scale: M3-specific seam excess +5.1% (M1 -1.2%, M2 -0.3%). Roll test
ABSOLUTE-terms analysis (the screen's coded relative-to-own-interior metric
was confounded and is superseded — deviation logged here, analysis from saved
profiles.npz): (a) the seam excess MOVES with the padding boundary (date-line
columns spike +24% under roll; Greenwich edge relief -5%); (b) ADDITIONALLY
rolling degrades M3 globally +15% RMSE — the network is layout-dependent far
beyond the seam. Since convs are translation-equivariant except padding, M3
evidently derives IMPLICIT ABSOLUTE POSITION from boundary geometry (cf.
Islam et al. 2020) and uses it as a de facto geography channel. Consequences:
(i) new mechanism thread — "how much of M3's advantage is implicit positional
encoding?" (folds into H-R2/H-R3 interpretation: position-from-padding is a
CONFOUND for 'M3 infers orography from flow'; H-R2 design must add a
rolled-input probe arm to separate position from flow-state information);
(ii) deployment-relevant artifact finding stands.
C-4 (lat edges): south-edge excess exists for ALL models (+11-18% vs own
interior; Antarctic winter physics), north edge better for all — no
M3-specific latitude artifact.

H-A1 (linear stencil closes gap?) SCREEN-KILL. Ridge on flattened 3x3
stencils: RMSE 0.827 vs M1 0.786, M2 0.700 — closes -47% of the gap (worse
than the column DNN). Even the first nonlocality increment is deeply
nonlinear. Evidence caveat: ridge fit on ~40k July columns vs models' 3-year
training set; a stronger linear baseline could narrow but plausibly not
reverse a -47% deficit. Consequence: H-I3 (gradient features) runs with a
NONLINEAR small-MLP carrier as specified, not linear.

C-5 (M1 vertical Jacobian calibration) complete, two results:
(1) M1's |Jacobian| is dominated ~25x by the 3 scalar inputs (lat/lon/zs)
    over u/v/theta blocks at all tested output levels — heavy explicit
    geography reliance (sharpens H-R3; contrast with M3's implicit position
    channel from H-P4).
(2) Vertical influence structure is physical: near-surface flux is
    level-local (same-level u sensitivity 0.50 vs block mean 0.18); upper
    flux depends on the whole column below (0.009 vs 0.038) — the filtering
    integral of GW theory. T4 tooling calibrated.

H-A5 formal verdict pending (tail extraction from saved histograms; variance
ratios M1 0.360 / M2 0.363 / M3 1.066 already recorded). Funnel: 29 generated,
5 screened (4 pass, 1 kill), 1 measurement done, 2 deferred, 2 asset-blocked.

## [2026-08-18 21:45] EXPERIMENTALIST + LEAD — Stage C wave 2a: A5/F3 pass, A2/F2 killed, A2b born

H-A5 (rewritten) SCREEN-PASS (results/screen_a5_tails/): exact quantile ratios
pred/true of |flux|, physical space, 24 July timesteps, per-timestep bootstrap:
  M1  P90 0.065 [.064,.066]   P99 0.284 [.272,.297]   P99.9 0.593 [.576,.611]
  M2  P90 0.114 [.112,.116]   P99 0.403 [.394,.412]   P99.9 0.615 [.605,.626]
  M3  P90 0.278 [.274,.283]   P99 0.705 [.693,.716]   P99.9 1.024 [1.005,1.043]
All CIs disjoint. Reading: ALL models regress typical amplitudes toward zero
(even M3's P90 is 3.6x low); calibration improves with quantile level and with
nonlocality; M3's extreme tail is calibrated. With month variance ratios
(0.360/0.363/1.066) this mechanically explains the Hellinger ordering.
H-F3 SCREEN-PASS on the SAME evidence (shared, not independent — logged to
avoid double-counting): all tail ratios < 1 at P99 with M3 largest; ordering
matches Hellinger ordering.

H-A2 SCREEN-KILL per pre-registered threshold: M1->M2 gap concentration in
top-decile |grad_h(u,v)| columns = 1.38 < 1.5 (orog-gradient concentration
1.65). BUT the M2->M3 gap is strongly regime-structured: shear concentration
2.92, orographic-gradient concentration 0.27 (anti-concentrated over
mountains). NEW HYPOTHESIS H-A2b [GEN]: "M3's advantage over M2 is a
shear-regime (storm-track/jet) phenomenon and is absent-to-negative over
orography; consistent with lateral propagation of nonorographic GWs requiring
horizontal context while orographic sources are locally determined."
Screening data IS this observation, so confirmation is deferred to
independent data (January month, Stage D) — no HARKing. Consequence: H-N5
Jacobian targets must include storm-track columns, not only Andes.

H-F2 SCREEN-KILL: M3's improvement on M1's top-decile-error timesteps is
uniform across hotspot types (nonorog/orog ratio 1.06; per-box 0.13-0.23).
Tension with H-A2b noted (time-mean column-wise concentration vs box-level
event-conditional improvement measure different things); January adjudicates.

FUNNEL: 30 hypotheses total (29 + A2b). Screened 9: PASS N1, I2, N3, P4, A5,
F3(shared); KILL A1, A2, F2. Measurement C-5 done. Remaining cheap screens:
N2, I1, A4, F1 (wave 2b, need fresh forward passes); R-family + N5 need the
activation/probe infrastructure (wave 3).

## [2026-08-18 21:55] EXPERIMENTALIST + LEAD — Wave 2b: N2, I1, A4 killed; F1 measured

Null refinement pre-registered for H-N2 (gate fields don't admit the T5
"distance-decay" null): nulls = flux CLIMATOLOGY alignment + time-mismatched
flux maps. Run AFTER logging this refinement.

H-N2 SCREEN-KILL (decisive): corr(finest alpha_t, |flux|_t) = -0.015;
climatology null 0.068; excess -0.083. Gates DO NOT localize sources — yet
they are causally important (H-N3: +4.3%). Open wave-3 question: what do the
gates encode? (First look suggests neither instantaneous flux nor pure
climatology; candidates: input-field structure, implicit position.)

H-I1 SCREEN-KILL per threshold: relative-octant ablation spread/mean = 0.165
< 0.20. Suggestive below-threshold structure logged honestly: the UPSTREAM
octant has the largest effect (0.515 vs mean 0.476), smallest at 315deg
(0.437) — direction consistent with advection, magnitude insufficient at
screening power. Not promoted; may be revisited only if a Stage-D result
needs it. Secondary observation: resampling ONE of eight neighbors roughly
DOUBLES M2's error (mean dMSE +0.476 on base ~0.49) — M2 demands spatially
coherent neighborhoods (same OOD family as the I2 caveat; both interventions
break physical coherence).

H-A4 SCREEN-KILL: act6 context deltas are high-rank (top-5 PCs = 22.9% of
variance; spectrum flat at ~4-5% each) and analytic projection-out moves M2
predictions only 2.8% toward M1 (threshold 20%). No compact context channel
at act6; context is integrated diffusely. (Untested at earlier layers —
noted, not pursued without independent motivation.)

H-F1 [MEASUREMENT] complete: top-1% error cells co-occur at 38-53x chance
(m2-m3 strongest, 53x); error rank correlations 0.78-0.88. All three
architectures share failure regimes -> consistent with a common data-limited
error floor (ERA5 unresolved variance), for the paper's discussion section.

FUNNEL: 13/30 screened. PASS: N1, I2, N3, P4, A5, F3. KILL: A1, A2, F2, N2,
I1, A4. MEASUREMENTS: F1, C-5. Kill rate 46% — the funnel is doing its job.
Next: flagship H-N5 + H-I4 (shared machinery), then R-family probe wave,
T7 grafts (P1-P3), I3; then CHECKPOINT_C.

## [2026-08-18 22:05] EXPERIMENTALIST + LEAD — FLAGSHIP H-N5 KILLED; H-I4 PASSES

H-N5 SCREEN-KILL (results/screen_n5/): influence maps (|grad| of predicted
flux energy at 20 hotspot targets w.r.t. all input columns), ring 1<d<=10
cells: median axis ratio 1.07 (threshold 1.3) — essentially ISOTROPIC; wind
alignment 9/20 (chance 6.7/20, threshold 12/20). The "network learned lateral
GW propagation" headline hypothesis dies at screening. Post-hoc caveat logged
for honesty: at T42 (~380 km cells), physical horizontal group displacements
during vertical propagation are typically sub-cell, so the prior for
propagation-shaped anisotropy at this resolution was weaker than the
hypothesis assumed. The kill stands (screening criteria were pre-registered);
the caveat shapes the PAPER framing (offline nonlocality at T42 is not
propagation-shaped), not the verdict.

H-I4 SCREEN-PASS (results/screen_i4/): occlusion-radius curves at the same 20
targets: median saturation radius 4 cells (~1500 km); spread 1..>16 with 4/20
targets needing >16 cells. M3 causally uses FAR more than the 3x3 ring.

EMERGING SYNTHESIS (for CHECKPOINT_C): M3's nonlocality is real and
large-radius (I4) but ISOTROPIC (N5), not flux-localizing (N2), concentrated
in shear regimes (A2b), variance/tail-calibrating (A5/F3), and entangled with
implicit position encoding from padding geometry (P4). Picture: a broad
"regime encoder" using regional context + implicit geography — NOT a wave-path
tracer. The kills are doing as much narrative work as the passes.

FUNNEL: 15/30 screened. PASS: N1, I2, N3, P4, A5, F3, I4 (7). KILL: A1, A2,
F2, N2, I1, A4, N5 (7). MEASURED: F1, C-5. Remaining before CHECKPOINT_C:
R-family probes (R1, R2, R3, R5, R6), SAE gate (R4), T7 grafts (P1-P3), I3;
deferred: A3, N4, A2b (January), L1a/L2 (asset check).
