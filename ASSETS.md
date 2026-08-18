# ASSETS.md — Stage A asset inventory

Compiled 2026-08-18 from 7 parallel discovery agents. Every entry below was
verified by actually fetching the resource (or noted otherwise). Detailed
per-topic reports with access logs: `results/stageA_discovery/01..07_*.md`.

## 1. Anchor paper and companions (all citations verified to exist)

| Paper | Venue / ID | Status |
|---|---|---|
| Gupta, Sheshadri, Roy, Anantharaj, "Offline Performance of a Nonlocal Deep Learning Parameterization for Climate Model Representation of Atmospheric Gravity Waves" (ANCHOR) | JAMES 17(10), e2025MS004977, 2025, DOI 10.1029/2025MS004977, CC BY | Full text NOT directly readable by agents (Wiley/ESSOAr Cloudflare-blocked; access log in 01_anchor_paper.md §9). All technical content recovered from its published code, data files, and companion papers. **Manual browser PDF download recommended to close the gap.** |
| Gupta, Sheshadri, Anantharaj, "Gravity Wave Momentum Fluxes from 1 km Global ECMWF Integrated Forecast System" | Scientific Data, 2024, DOI 10.1038/s41597-024-03699-x | Full text READ (Europe PMC XML). Defines Helmholtz-based flux extraction + IFS-1km dataset. |
| Gupta et al., "Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation" | arXiv:2406.14775 (v2, Nov 2024) | Full text READ. Same M1/M2/M3 ladder. |
| Pahlavan, Hassanzadeh, Alexander, "Explainable Offline-Online Training of Neural Networks for Parameterizations: A 1D Gravity Wave-QBO Testbed in the Small-Data Regime" | GRL 51(2), 2024, DOI 10.1029/2023GL106324 | Full text READ (arXiv:2309.09024). Interp = SpArK kernel-Fourier only. Closest prior art. |
| Pahlavan et al. (follow-up: receptive fields / nonlocality of CNN/FNO/MLP emulators) | GRL 2025, DOI 10.1029/2024GL114136, arXiv:2407.05224 | Verified. Gradient-based ERF analysis — overlaps our T4 family; must be cited. Code: github.com/HamidPahlavan/Nonlocality |
| Espinosa, Sheshadri, et al. (GW param. generalizes: QBO + CO2 response) | GRL 49(8), e2022GL098174, 2022, DOI 10.1029/2022GL098174 | Verified. Caveat: abstract never says "WaveNet"/"AD99" — confirm from PDF before using those names. |
| Hardiman et al., "Machine Learning for Nonorographic Gravity Waves in a Climate Model" | AIES 2(4), **2023**, DOI 10.1175/AIES-D-22-0081.1 | Verified (year correction vs master prompt: 2023, not 2022). |
| Wang, Yuval, O'Gorman, "Non-local parameterization of atmospheric subgrid processes with neural networks" | JAMES 14(10), e2022MS002984, 2022, DOI 10.1029/2022MS002984 | Verified. 3x3 nonlocal inputs; attribution shows where nonlocality pays. |
| Fritts & Alexander, "Gravity wave dynamics and effects in the middle atmosphere" | Rev. Geophys. 41(**1**), 1003, 2003, DOI 10.1029/2001RG000106 | Verified (issue 1, not 3 as some aggregators claim; erratum exists). |
| Gupta et al., "Finetuning AI Foundation Models to Develop Subgrid-Scale Parameterizations: A Case Study on Atmospheric Gravity Waves" | arXiv:2509.03816 | Verified (real title differs from master prompt). Hellinger 0.06 vs 0.11 baseline. Contains NO interpretability — do not cite it for that. |

## 2. Code (all fetched/cloned)

- **Anchor training/model code**: https://github.com/DataWaveProject/nonlocal_gwfluxes — MIT, tag `1.0.0` = paper release, archived as Zenodo DOI 10.5281/zenodo.16415113. `ANN_CNN` (stencil 1/3/5) + `Attention_UNet` classes, ERA5 training, IFS transfer-learning scripts. **Cloned to `C:/Users/sidi0/gwmi_data/external/nonlocal_gwfluxes`.**
- Author dev repo (paper-era layout + 5x5 stencil + probabilistic extras): https://github.com/amangupta2/nonlocal_gwfluxes (author GitHub is `amangupta2`).
- **qbo1d testbed** (Pahlavan lineage): https://github.com/DataWaveProject/qbo1d — Apache-2.0, differentiable PyTorch 1D QBO (73 levels), CPU-trivial; trained CNN coeffs + data: Zenodo 10.5281/zenodo.10278373.
- Prithvi WxC GW fine-tune: https://github.com/NASA-IMPACT/gravity-wave-finetuning (checkpoint 9.9 GB at ibm-nasa-geospatial — too heavy for our CPU budget; reference only).
- Flux-extraction scripts (Helmholtz, coarse-graining, PySpharm mods): OSF DOI 10.17605/OSF.IO/GX32S, `python_scripts/`.

## 3. Pretrained weights (Tier 1 enabler)

**https://huggingface.co/amangupta2/nonlocal_gwfluxes** — MIT, ~3.2 GB total, HEAD-verified and downloaded to `C:/Users/sidi0/gwmi_data/weights/nonlocal_gwfluxes/`:

- M1 `ANN_1x1/`: global uvtheta (epoch94, 121.5 MB), global uvthetaw (epoch94, 210 MB), stratosphere_only uvtheta (epochs 88 & 100, 29.9 MB each)
- M2 `ANN_3x3/`: global uvtheta (epoch52, 136.2 MB), global uvthetaw (epoch80, 236.1 MB), stratosphere_only uvtheta (epochs 38 & 93, 33.6 MB each)
- M3 `AttentionUNet/`: global uvtheta (epoch100), global uvthetaw (epoch119), stratosphere_only uvtheta (epoch131), stratosphere_only uvthetaw (epoch138) — ~455 MB each
- Aux: https://huggingface.co/amangupta2/iccs_coupling_checkpoints (IFS/L93 transfer-learned; for the Fortran coupling MWE)

Note: one released checkpoint per configuration (no seed ensemble published).

## 4. Data

| Asset | Where | Size / access | Fits budget? |
|---|---|---|---|
| Test ERA5 samples, training-format (hourly 2015, inputs u,v,theta,w + outputs uw,vw, with exact normalization constants in attrs) | HF repo above, `test_files/` | 48 MB (1x1) + 434 MB (3x3/global); downloaded | YES — primary eval data |
| ERA5-derived training months (the anchor's training distribution; part of WxC-Bench) | HF `nasa-impact/WxC-Bench` (`nonlocal_parameterization/`, 4 monthly files ~14.9 GB) and HF `ibm-nasa-geospatial/gravity-wave-parameterization` (1 month, 20.9 GB) | Public HTTPS, no auth, verified | PARTIAL — 1-2 months max within 30 GB budget |
| Coarse-grained (T42) IFS-1km GW fluxes (transfer-learning target) | OSF, DOI 10.17605/OSF.IO/GX32S | 10 netCDF x ~4.49 GB = ~43 GB, plain HTTPS, byte-range OK | PARTIAL — 2-3 files feasible |
| Native 1.4 km IFS fluxes ("XNR1K", TCo7999, Nov 2018-Feb 2019) | ORNL, DOI 10.13139/OLCF/2308755, Globus endpoint 57618e0a-2c99-45ff-9694-24141b92fa17 | Globus-only | NO — out of scope |
| Raw ERA5 (auxiliary fields only, if needed) | ARCO-ERA5 `gs://gcp-public-data-arco-era5` (anon-verified) / WB2 | Global chunking => ~123x transfer overhead from home; Colab-subset workaround documented in 07_era5_access.md | Only via Colab workaround |
| Full 48-month anchor training set | — | NOT PUBLISHED anywhere found | — |

## 5. Not found / dead ends (honest record)

- Anchor paper full text: all programmatic routes blocked (Wiley 403, ESSOAr Cloudflare, NSF PAR 404, no PMC/OSTI/NTRS/Wayback). Details: 01_anchor_paper.md §9.
- `get-model-and-data.sh` epoch8/Globus URLs in the code repo: 404 (stale).
- Zenodo search for anchor-DOI-citing records: 0 hits. Legacy AWS `era5-pds`: dead (403).
- GitHub user `ag4680` (guessed handle): 404.

## 6. Local holdings after Stage A

```
C:/Users/sidi0/gwmi_data/
  weights/nonlocal_gwfluxes/   # 12 checkpoints + 2 test .nc + README (~3.2 GB)
  external/nonlocal_gwfluxes/  # code clone, tag 1.0.0
```

Tier decision and reasoning: see RESEARCH_LOG.md entry [2026-08-18 18:55].
