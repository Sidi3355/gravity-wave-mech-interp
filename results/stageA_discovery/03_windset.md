# 03 — WINDSET / GW momentum-flux dataset discovery (Data Engineer)

Date: 2026-08-18. All claims below were verified by direct fetch (URLs given). Failures recorded explicitly.

## 1. The Scientific Data paper (found and fetched via PMC mirror)

- **Exact title:** "Gravity Wave Momentum Fluxes from 1 km Global ECMWF Integrated Forecast System"
- **Authors:** Gupta, Sheshadri, Anantharaj
- **DOI:** 10.1038/s41597-024-03699-x
- **Published:** 21 August 2024, Scientific Data
- **Fetched from:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11339449/ (nature.com redirected to an IDP authorize page; PMC open-access mirror used instead)
- **Naming note:** the paper does NOT call this dataset "WINDSET" (see section 3 — WINDSET is a different, ERA5-derived dataset). The OSF record uses the earlier working title "Extracting Mesoscale Gravity Wave Momentum Fluxes from Kilometer-Scale ECMWF Integrated Forecast System".

### What the data is
- Source: ECMWF IFS 1.4 km experimental nature run "XNR1K" (~1.5 PB raw output), Nov 1 2018 – Mar 1 2019 (120 days).
- GW fluxes extracted by Helmholtz decomposition.
- Variables: u, v, omega, T, N^2, vertical flux of zonal momentum (u'omega'/g), vertical flux of meridional momentum (v'omega'/g); 3-hourly instantaneous; 137 model levels (hybrid near surface, pure pressure above ~75 hPa).
- Two products:
  1. **Coarse-grained to T42 (~2.8 deg) Gaussian grid** — hosted on OSF (downloadable — details in 2a).
  2. **Native ~1.4 km reduced Gaussian grid** — hosted at ORNL Constellation, Globus only (2b).

## 2. Hosting and access (verified)

### 2a. OSF — coarse-grained fluxes (PRACTICAL, no credentials)
- Record: **DOI 10.17605/OSF.IO/GX32S**, https://osf.io/gx32s/ (public: true, verified via OSF API `https://api.osf.io/v2/nodes/gx32s/`)
- Title: Data for "Extracting Mesoscale Gravity Wave Momentum Fluxes from Kilometer-Scale ECMWF Integrated Forecast System"
- **10 netCDF4 files** `cg_ifs_3hourly_gwmf_helmholtz_ndjf_01.nc` ... `_10.nc`; files 01–09 = 4,492,669,798–4,492,669,801 bytes (~4.49 GB) each holding 100 three-hourly records (~12.5 days); file 10 = 2,740,518,217 bytes (~2.74 GB, 61 records). **Total ~43.2 GB** (961 timesteps).
- Also a `python_scripts` folder (OSF folder id 65c460be35be200bd5a50420).
- **Direct HTTPS download links (per-file, from OSF API):**
  - 01: https://osf.io/download/b2qvr/  02: https://osf.io/download/dc5ku/  03: https://osf.io/download/n5zyv/
  - 04: https://osf.io/download/rwnbz/  05: https://osf.io/download/p6de2/  06: https://osf.io/download/cqyr7/
  - 07: https://osf.io/download/hqzk4/  08: https://osf.io/download/xm5v3/  09: https://osf.io/download/pbkjm/
  - 10: https://osf.io/download/zp8dw/
- **Downloadability test (performed 2026-08-18):** `curl -I https://osf.io/download/b2qvr/` -> 302 -> files.osf.io -> HTTP 200, Content-Length 4492669798, final URL is a signed Google Cloud Storage URL; **byte-range request returned HTTP 206** (100 bytes fetched). No login, no API key. Per-file granularity = ~4.5 GB.

### 2b. ORNL Constellation — native 1.4 km fluxes (NOT practical for us)
- Record: https://doi.ccs.ornl.gov/dataset/7f3421d3-d3a6-58ed-b47b-6a4ac62729af (linked from paper as https://doi.ccs.ornl.gov/ui/doi/475)
- **DOI 10.13139/OLCF/2308755**, "Mesoscale Gravity Wave Momentum Fluxes from Kilometer-Scale ECMWF Experimental Nature Run", released 2024-06-25, NetCDF4.
- **Access is via Globus file transfer** (Globus login required); no plain-HTTPS file listing or file sizes visible on the landing page. Not pursued further — out of scope for a 30 GB budget.
- Paper also mentions interactive access through a Stanford **Redivis** project (Jupyter environment; account required — not tested).

## 3. WINDSET — what it actually is (task 4: the ERA5 training data)

**WINDSET is not the 1 km IFS dataset.** Per the ICML'24 ML4ESM paper (fetched: https://arxiv.org/html/2406.14775v1, "Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation", Gupta, Sheshadri, Roy, Gaur, Maskey, Ramachandran):
- WINDSET = "Weather Insights and Novel Data for Systematic Evaluation and Testing", introduced by Shinde et al. 2024 = **WxC-Bench** (arXiv 2412.02780, NASA-IMPACT).
- The GW component: background fields + resolved GW momentum fluxes derived from **ERA5**, hourly, coarse-grained to a **64x128 (T42) Gaussian grid**, **122 vertical levels**, years **2010, 2012, 2014, 2015** (48 months computed). This is the training data of the anchor JAMES paper (3 years train + 1 year validation).

### Where the ERA5 training data is published (verified on HuggingFace, all public, ungated)
1. **nasa-impact/WxC-Bench** — https://huggingface.co/datasets/nasa-impact/WxC-Bench (HF DOI 10.57967/hf/7711, MIT license). Folder `nonlocal_parameterization/` contains **4 of the 48 months** ("provided here for testing"):
   - `inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling01.nc` — 14,944,689,158 B (~14.9 GB)
   - `..._scaling04.nc` — 14,462,606,346 B (~14.5 GB)
   - `..._scaling07.nc` — 14,944,689,162 B (~14.9 GB)
   - `..._scaling10.nc` — 14,944,689,162 B (~14.9 GB)
   - The 30-day/31-day size ratio (0.9677) confirms monthly files: Jan/Apr/Jul/Oct 2015, hourly (744/720 steps, ~20 MB per timestep).
   - Format (from folder README, fetched from HF): netCDF; `features` TIME x IDIM(369 = 3 + 3x122: lat, lon, surface elevation, u, v, theta on 122 levels) x 64 x 128; `output` TIME x ODIM(244 = u'w' and v'w' on 122 levels each) x 64 x 128; pre-normalized (constant mu/sigma).
   - Example direct URL pattern: `https://huggingface.co/datasets/nasa-impact/WxC-Bench/resolve/main/nonlocal_parameterization/<filename>.nc`
2. **ibm-nasa-geospatial/gravity-wave-parameterization** — https://huggingface.co/datasets/ibm-nasa-geospatial/gravity-wave-parameterization (Apache-2.0; identical content mirrored at **Prithvi-WxC/Gravity_wave_Parameterization**). Single file:
   - `wxc_input_u_v_t_p_output_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling05.nc` — **20,893,258,746 B (~20.9 GB)** — one month (May 2015, stated to be from the validation set). Richer channels: input IDIM=491 (lat, lon, elev + u, v, T, P on 122 lvls), output ODIM=366 (theta, u'w', v'w' on 122 lvls).
   - **Download test (2026-08-18):** `curl -I https://huggingface.co/datasets/ibm-nasa-geospatial/gravity-wave-parameterization/resolve/main/wxc_input_u_v_t_p_output_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling05.nc` -> 302 to `us.aws.cdn.hf.co` signed URL with `user_id=public` -> **HTTP 200, content-length 20893258746. No auth.** HF CDN supports byte ranges.
3. **simonpf/WINDSET** — https://huggingface.co/datasets/simonpf/WINDSET — checked: contains only `long_term_precipitation_forecast/`; no GW data. Not useful.

- **The full 48-month training set is NOT published anywhere I could find** — only the 4+1 months above. Contact per README: amangupta2@gmail.com / https://github.com/amangupta2. It can in principle be regenerated: the WxC-Bench GitHub repo (https://github.com/NASA-IMPACT/WxC-Bench, `nonlocal_parameterization/`) contains the full ERA5-download (CDS API) + Helmholtz-decomposition (windspharm) + T42 conservative-regridding (xESMF) pipeline.

## 4. Anchor JAMES paper (10.1029/2025MS004977)

- **Exact title:** "Offline Performance of a Nonlocal Deep Learning Parameterization for Climate Model Representation of Atmospheric Gravity Waves", Gupta, Sheshadri, Roy, Anantharaj. JAMES, published 2025-10-22 (issue Oct 2025), CC-BY gold OA (per Crossref/OpenAlex APIs). Preprint DOI: 10.22541/essoar.173869599.90237060.
- **FAILURE (recorded honestly):** the verbatim Data Availability Statement could not be fetched — agupubs.onlinelibrary.wiley.com returned HTTP 403 (WebFetch and curl with browser UA, both HTML and pdfdirect routes), essopenarchive.org PDF returned 403, DOAJ article page returned 403. Metadata came from OpenAlex, Crossref, and Semantic Scholar APIs (all fetched successfully). Training-data locations above were established independently of the DAS.
- **Training-code repo (README fetched):** https://github.com/amangupta2/nonlocal_gwfluxes — implements M1 (1x1 ANN), M2 (3x3 CNN+ANN), M3 (Attention UNet); trained on ERA5 hourly (3 yr train + 1 yr val); includes **transfer-learning functionality onto the 1 km IFS fluxes** — exactly the anchor paper's pipeline connecting the datasets in sections 2 and 3.
- Related: https://github.com/amangupta2/gravity-wave-finetuning-james (Prithvi-WxC finetuning for GW flux, the companion 10.1029/2025MS005075 line of work).

## 5. Fit to our constraints (148 GB disk, ~30 GB data budget, no GPU)

| Option | Size | Fits budget? |
|---|---|---|
| HF `ibm-nasa-geospatial/gravity-wave-parameterization`, 1 month ERA5 (u,v,T,P -> theta,u'w',v'w'), May 2015 | 20.9 GB | Yes (single file) |
| HF WxC-Bench `nonlocal_parameterization`, 1 month ERA5 (u,v,theta -> u'w',v'w') | 14.5–14.9 GB per month | Yes; 2 months ~29.4 GB borderline |
| OSF coarse-grained 1 km IFS fluxes, per file (~12.5 days, 3-hourly, T42, 137 lvls) | ~4.49 GB per file | Yes — 1 to 6 files; full set 43.2 GB is OVER budget |
| ORNL native 1.4 km | petascale, Globus login | No |
| Full 48-month ERA5 WINDSET | not published | N/A (regeneration via CDS possible but far over budget) |

**Recommended combo under 30 GB:** one WxC-Bench monthly ERA5 file (~14.9 GB) as training/analysis data + 2–3 OSF IFS files (~9–13.5 GB) for the transfer-learning/eval side. All plain HTTPS, no credentials. Both hosts honor byte-range requests (verified HTTP 206 on OSF; HF CDN ranges), so partial/streamed reads (e.g., fsspec + h5netcdf/kerchunk) are possible without downloading whole files.
