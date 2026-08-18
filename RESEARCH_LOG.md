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
