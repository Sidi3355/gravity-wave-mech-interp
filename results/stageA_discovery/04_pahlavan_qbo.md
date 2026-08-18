# Pahlavan et al. 2024 (GRL, 10.1029/2023GL106324) — verification and testbed hunt

Date of investigation: 2026-08-18. Role: DATA ENGINEER. All content below was fetched during this session; sources and dead ends recorded explicitly.

## 1. Paper verification: VERIFIED

- **Exact title** (Crossref + arXiv, identical): *"Explainable Offline-Online Training of Neural Networks for Parameterizations: A 1D Gravity Wave-QBO Testbed in the Small-Data Regime"*
- **Authors** (Crossref, with affiliations): Hamid A. Pahlavan (Rice University, Houston TX), Pedram Hassanzadeh (Rice University, Houston TX), M. Joan Alexander (NorthWest Research Associates, Boulder CO)
- **Journal/vol/issue/date** (Crossref): Geophysical Research Letters, Vol. 51, Issue 2, published 2024-01-27. License: CC BY-NC 4.0. Crossref cited-by count: 21 (Semantic Scholar: 27 citations).
- **DOI**: 10.1029/2023GL106324. **arXiv preprint**: [2309.09024](https://arxiv.org/abs/2309.09024) (v1 submitted 16 Sep 2023).

### Sources used (and one dead end)
- **DEAD END**: `https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL106324` returned **HTTP 403 Forbidden** (bot-blocked), as did the Wiley page of the 2025 follow-up (`/doi/full/10.1029/2024GL114136`). The published Wiley HTML was therefore NOT read directly.
- Full text obtained instead from **ar5iv HTML of arXiv:2309.09024** — saved locally as `pahlavan_ar5iv.html` and plain text `pahlavan_fulltext.txt` in this discovery directory. Metadata cross-checked against Crossref API (`crossref.json`, saved) and the arXiv abstract page.
- Caveat: verbatim quotes below are from the **arXiv version**, which may differ in minor wording from the published GRL version (unverifiable due to Wiley 403). The abstract on arXiv matches the paper's claims as summarized by AMS/search-index abstracts.

### Abstract (verbatim, arXiv v1)
> "There are different strategies for training neural networks (NNs) as subgrid-scale parameterizations. Here, we use a 1D model of the quasi-biennial oscillation (QBO) and gravity wave (GW) parameterizations as testbeds. A 12-layer convolutional NN that predicts GW forcings for given wind profiles, when trained offline in a big-data regime (100-years), produces realistic QBOs once coupled to the 1D model. In contrast, offline training of this NN in a small-data regime (18-months) yields unrealistic QBOs. However, online re-training of just two layers of this NN using ensemble Kalman inversion and only time-averaged QBO statistics leads to parameterizations that yield realistic QBOs. Fourier analysis of these three NNs' kernels suggests why/how re-training works and reveals that these NNs primarily learn low-pass, high-pass, and a combination of band-pass filters, consistent with the importance of both local and non-local dynamics in GW propagation/dissipation. These findings/strategies apply to data-driven parameterizations of other climate processes generally."

## 2. What model they interpret (architecture)

- A **CNN emulator of a physics-based gravity-wave parameterization** inside a 1D QBO model: input = zonal wind profile u(z), output = GW drag profile G_CNN(u; theta).
- Architecture (verbatim from full text): "This CNN consists of 12 sequential 1D convolutional layers. Each hidden layer has 15 channels, each with 15 kernels with a size of 5, resulting in ~11,600 learnable parameters. The activation function is hyperbolic tangent (tanh)."
- Kernels are size 5, operating on activations of size 37 (37 vertical levels — consistent with the 500 m grid version of the 1D model; note the qbo1d repo default is dz=250 m giving 73 levels, but the paper's dataset filenames say "500m").
- Three trained variants compared: **CNN-BD** (offline, 100-yr "big data"), **CNN-SD** (offline, 18-month "small data"), **CNN-EKI** (CNN-SD with layers 2 and 11 re-trained online). A fourth, **CNN-S3D** (strategically sampled 18 months = 72 weeks spaced a month apart), appears in the Discussion/SI.

### The 1D QBO model
- Holton & Lindzen (1972) / Plumb (1977)-type forced advection-diffusion model of the tropical stratosphere: du/dt + w du/dz − kappa d2u/dz2 = G(u) + eta(t), with w=0 (no upwelling in this configuration), kappa = 0.3 m2/s, stochastic forcing eta.
- Driven by two vertically propagating GWs, phase speeds (c1, c2) = (−30, +30) m/s. Control QBO: period 28.0 ± 0.7 months, amplitude 21 ± 0.3 m/s at 25 km.

## 3. Interpretability methods used (precise)

**Single interpretability method: the SpArK framework — "Spectral Analysis of Regression Kernels and Activations", introduced in Subel et al. (2023).** Verbatim from the paper:
> "We also use the Spectral Analysis of Regression Kernels and Activations (SpArK) framework (introduced in Subel et al. (2023)) to provide physically interpretable insights into what these three CNNs learn."

Procedure (Section 3.3, "Explainable Learning using SpArK"):
1. Each convolutional kernel (size 5) is **zero-padded to size 37** to match the activation size.
2. Padded kernels are **Fourier-transformed** into spectral space (using the convolution theorem applied to the CNN's governing equations, per Subel et al. 2023).
3. Kernels are **categorized by dominant wavenumber k\*** (wavenumber where the spectrum peaks in magnitude).

Findings (verbatim excerpts): "The dominant spectra have k\*=0 (low-pass filters), followed by k\*=18 (high-pass filters). k\*=5 and k\*=13 come next, each representing band-pass filters. While Figure 4 showcases the composited kernels from layers 2 and 11, similar patterns are observed across other layers. Collectively, these four wavenumbers account for ~65% of all the kernels, with k\*=0 and k\*=18 together constituting ~45% of the total." Low-pass filters "extract large scales / perform averaging" (non-local dynamics); high-pass filters "capture more local dynamics" — connected to physics of GW propagation (non-local: critical-level filtering by winds below) and dissipation (local).

**Which layers**: kernels of ALL layers are analyzed/categorized; the composited spectra shown in Fig. 4 are from **layers 2 and 11** (the two layers that were re-trained online), chosen because re-training only these layers "simplif[ies] the analysis and enhanc[es] interpretability". SpArK is used to compare CNN-BD vs CNN-SD vs CNN-EKI kernels to explain why re-training works.

### Online re-training method (context for the offline-online gap)
- **Ensemble Kalman Inversion (EKI)** via `EnsembleKalmanProcesses.jl` (CliMA project): 200 ensemble members, 10 iterations, matching 85 time-averaged QBO statistics from 10-year runs (not instantaneous profiles).
- Verbatim: "By online re-training of only the shallowest and deepest hidden layers of the CNN-SD (i.e., layers 2 and 11), we obtain results that are on par with full re-training."
- Offline-online skill gap: CNN-SD has high offline (a-priori) skill (R2 = 0.9) but fails a-posteriori (unrealistic QBO, ~32% error vs ~3% for CNN-BD); the tails of the GWD PDF (top 1% magnitudes) are where CNN-SD fails offline (R2 at tails 0.14, vs 0.7 for CNN-S3D). Closing quote: "…between a-priori metrics and a-posteriori performances further emphasize this need."

## 4. What they explicitly did NOT do

Keyword search of the full text (`pahlavan_fulltext.txt`) — **zero hits** for: saliency, probe/probing, sparse autoencoder (SAE), attribution, ablation, intervention, Shapley/SHAP, layer-wise relevance propagation, receptive field. So:
- **No probing** (no trained probes on activations).
- **No causal interventions / ablations / activation patching.**
- **No SAEs or feature-dictionary methods.**
- **No saliency/attribution maps.**
- Activations: SpArK's name includes "Activations", but in this paper the analysis presented is of **kernels only** (kernel spectra by dominant wavenumber); no separate activation-space analysis is shown.
- Interpretability is purely **weight-space spectral analysis** (Fourier of conv kernels), plus physical reasoning connecting filter types to GW dynamics.

## 5. Code/data availability — verbatim, and the testbed hunt

### Open Research statement (verbatim, arXiv v1; published version unverified due to Wiley 403)
> "We use the open source software EnsembleKalmanProcesses.jl Oliver et al. (2022) for EKI analysis and the qbo1d code for the 1D-QBO model simulations, accessible at https://github.com/DataWaveProject/qbo1d.git."

### Zenodo dataset (found via search; fetched record page)
- **"Dataset for 'Explainable Offline-Online Training of Neural Networks for Parameterizations: …' by Pahlavan et al. (2023)"**, DOI [10.5281/zenodo.10278373](https://zenodo.org/records/10278373), published 2023-12-06, CC BY 4.0, author "Alizadeh Pahlavan, Hamid (Rice University)".
- Files include: `coeffs_CNN-BD.nc`, `coeffs_CNN-SD.nc`, `coeffs_EKI.nc` (52.4/52.4/24.4 kB — these look like the **trained CNN weights/coefficients**), offline/online forcing and wind fields (`f_*`, `u_*` ~100–213 MB each), the noise realization, and the training dataset `u_f_flux_true_forNN_500m_consalpha_noise_0.5x_1000yrs.nc` (319.7 MB) and `..._100yrs.nc` (32.0 MB).
- NOTE: paper-specific **training scripts** for the 2024 paper were not found as a standalone repo; the Zenodo record carries data + coefficients, and the 1D model + an NN-emulator example live in qbo1d (below). The follow-up paper's repo (Nonlocality, below) has NN training notebooks.

### TESTBED FOUND: DataWaveProject/qbo1d — cloned and inspected
- URL: **https://github.com/DataWaveProject/qbo1d** (Apache-2.0; 3 stars/2 forks; a DataWave-project fork of **ofershamir/qbo1d** — original author Ofer Shamir, NYU/Gerber group; Hamid Pahlavan also maintains a fork). Local clone: `qbo1d_repo/` in this discovery directory.
- "A PyTorch 1D QBO model." Pure-Python package, **1,271 lines total**:
  - `qbo1d/adsolver.py` (360 lines): `ADSolver` class — 1D forced advection-diffusion solver. Defaults: z from 17 to 35 km, **dz=250 m (73 levels)**, **dt=86400 s (1 day)**, 12-year default integration, kappa=0.3 m2/s; second-order centered differentiation matrices; supports constant/time/height-dependent vertical advection w.
  - `qbo1d/emulate.py` (112 lines): `QBODataset` (xarray→torch Dataset mapping u→S), `relative_MSELoss`, `GlobalStandardScaler` — the NN-emulator plumbing.
  - `qbo1d/stochastic_forcing.py`, `stochastic_2wave_forcing.py`, `deterministic_forcing.py`, `utils.py` (`utils.load_model` loads a PyTorch model as the source term — this is how an NN parameterization is coupled online).
  - Example notebooks: `example.ipynb`, **`example_nn_emulator.ipynb`**, `example_stochastic.ipynb`; `control.py`; trained models in `models/` (`fully_connected.pth`, `deterministic_forcing.pth`, `stochastic_forcing.pth`, ~19 MB); sample data in `data/` (~59 MB, included in repo); Sphinx docs at https://datawaveproject.github.io/qbo1d/ (model description, stochastic forcing, numerical scheme, code).
  - Dependencies: PyTorch, NumPy, Matplotlib, SciPy (+ Jupyter for examples); conda spec file included.
- **CPU cost: trivial.** A 96-year integration = ~34,560 timesteps of a 73-level (or 37-level at 500 m) linear solve plus a small NN forward pass; README's example figure is a 96-year run. Differentiable (backprop through the solver is an advertised feature). Entirely feasible on a laptop CPU in minutes; ideal for our interpretability experiments (training data generation AND online coupling both included).

## 6. Follow-up papers citing it (novelty check)

Semantic Scholar lists **27 citing papers** (full list saved in `citations.json`). The ones relevant to GW/QBO emulator interpretability:

1. **Pahlavan, Hassanzadeh & Alexander 2025**, *"On the Importance of Learning Non-Local Dynamics for Stable Data-Driven Climate Modeling: A 1D Gravity Wave-QBO Testbed"*, GRL, DOI 10.1029/2024GL114136 ([arXiv:2407.05224](https://arxiv.org/abs/2407.05224)). Same testbed; compares **CNN vs FNO vs MLP**; interpretability = **receptive-field (RF) analysis + effective receptive field (ERF) via gradient backpropagation** ("similar to the layer-wise relevance propagation (LRP) method"), plus drag-wind correlation matrices. Key claims: NNs with 99% offline accuracy can still be unstable online if the RF is too small; "For a CNN to remain stable and accurate, its RF must exceed the number of model levels." Code availability (verbatim from arXiv HTML): "The code for the 1D-QBO model is available on GitHub at https://github.com/DataWaveProject/qbo1d. You can also access the code for training the neural networks via this link: https://github.com/HamidPahlavan/Nonlocality". The **Nonlocality** repo (MIT, 6 commits) contains `CNN.ipynb`, `FNO.ipynb`, `MLP.ipynb` and dataset `QBO_0.5x_100yrs_500m.nc`.
2. **Shamir et al. 2024**, *"The graft-versus-host problem for data-driven gravity-wave parameterizations in a one-dimensional quasibiennial oscillation model"*, QJRMS, DOI 10.1002/qj.4707 — same qbo1d testbed, about online-coupling degradation (offline-online gap), not deep interpretability.
3. **"Fourier analysis of the physics of transfer learning for data-driven subgrid-scale models of ocean turbulence"** (2025, Machine Learning: Earth, DOI 10.1088/3049-4753/ae510d) — extends the SpArK-style spectral kernel analysis line (Hassanzadeh group), but for **ocean turbulence**, not GW/QBO.
4. Others (UQ for GW emulators 10.1029/2024MS004292; WACCM GW emulation lessons 10.1029/2023MS004145; reviews in Nature Reviews Physics and Annu. Rev. Condens. Matter Phys.) touch on GW emulators but do **not** perform deeper mechanistic interpretability.

**Novelty conclusion: no citing paper applies probing, causal interventions/patching, SAEs, or any activation-space mechanistic analysis to GW/QBO emulators. Prior interp art = kernel Fourier spectra (2024 paper) and receptive-field/ERF analysis (2025 follow-up). Activation-level and causal-intervention interpretability of this testbed appears open.**

## Local artifacts saved in this directory
- `pahlavan_ar5iv.html`, `pahlavan_fulltext.txt` — full text of arXiv:2309.09024 (ar5iv render)
- `crossref.json` — Crossref metadata for 10.1029/2023GL106324
- `citations.json` — Semantic Scholar citing papers (27 entries)
- `qbo1d_repo/` — shallow clone (depth 5) of https://github.com/DataWaveProject/qbo1d
