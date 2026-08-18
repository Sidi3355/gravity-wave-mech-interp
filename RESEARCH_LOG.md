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
