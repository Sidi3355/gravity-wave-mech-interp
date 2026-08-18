# Anchor paper deep-read: Gupta, Sheshadri, Roy, Anantharaj (2025), JAMES

**Paper:** "Offline Performance of a Nonlocal Deep Learning Parameterization for Climate Model Representation of Atmospheric Gravity Waves"
**DOI:** 10.1029/2025MS004977 — JAMES Vol. 17, Issue 10, article e2025MS004977, published online 2025-10-22 (print Oct 2025), CC BY 4.0, Gold OA.
**Preprint:** ESS Open Archive, DOI 10.22541/essoar.173869599.90237060/v1 (confirmed to exist via OpenAIRE; not readable, see access log).
**Funding (Crossref):** Schmidt Futures (VESRI/DataWave), NSF OAC-2004492, DOE DE-AC05-00OR22725.

---

## 0. SOURCE PROVENANCE — read this first

I could NOT read the published full text or the preprint PDF directly (both Wiley and ESS Open Archive are behind Cloudflare bot protection; every alternate full-text mirror failed — full access log in §9). Everything below is therefore tagged by source tier:

- **[CODE]** — read directly from the paper's own published code, `github.com/DataWaveProject/nonlocal_gwfluxes` (cloned; this repo is what the paper's Zenodo code DOI 10.5281/zenodo.16415113 archives). Highest reliability for replication; it is the literal training/model code.
- **[DATA]** — read directly from the paper's published artifacts: HuggingFace checkpoint repo file listing (HF API) and an actual training-format NetCDF test file (downloaded from HF and inspected with netCDF4).
- **[SCIDATA]** — read directly from the full text (Europe PMC XML, PMC11339449) of the companion dataset paper: Gupta, Sheshadri, Anantharaj, "Gravity Wave Momentum Fluxes from 1 km Global ECMWF Integrated Forecast System", Scientific Data (2024), DOI 10.1038/s41597-024-03699-x. This paper defines the flux-extraction method and the IFS-1km dataset the anchor paper uses.
- **[ICML]** — read via WebFetch of arXiv HTML 2406.14775v2 ("Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation", ICML 2024): the precursor paper with the same M1/M2/M3 taxonomy (earlier model variants — differs from the JAMES code in places; noted where relevant).
- **[FT-PAPER]** — read via WebFetch of arXiv HTML 2509.03816v1 (companion JAMES paper 10.1029/2025MS005075, "Finetuning AI Foundation Models..."), which describes the anchor's Attention UNet as its baseline and re-states dataset construction.
- **[SNIPPET]** — fragments of the anchor paper's indexed full text recovered through web-search snippets of the Wiley page. Real indexed content but relayed through a search tool; treat as medium confidence and verify wording against the PDF when access is obtained.
- **[ABSTRACT]** — the anchor's abstract, read verbatim from NSF PAR's TEI record (par.nsf.gov/biblio/10653528/media/xml).
- **[INFERENCE]** — my computation/deduction (e.g., parameter counts computed by instantiating the repo's model classes; grid-index-to-lat/lon conversions). Explicitly marked.

---

## 1. Abstract (verbatim) [ABSTRACT]

> "Gravity waves (GWs) make crucial contributions to the middle atmospheric circulation. Yet, their climate model representation remains inaccurate, leading to key circulation biases. This study introduces a set of three neural networks (NNs) that learn to predict GW fluxes (GWFs) from multiple years of high-resolution ERA5 reanalysis. The three NNs: a ANN, a ANN-CNN, and an Attention UNet embed different levels of horizontal nonlocality in their architecture and are capable of representing nonlocal GW effects that are missing from current operational GW parameterizations. The NNs are evaluated offline on both time-averaged statistics and time-evolving flux variability. All NNs, especially the Attention UNet, accurately recreate the global GWF distribution in both the troposphere and the stratosphere. Moreover, the Attention UNet most skillfully predicts the transient evolution of GWFs over prominent orographic and nonorographic hotspots, with the model being a close second. Since even ERA5 does not resolve a substantial portion of GWFs, this deficiency is compensated by subsequently applying transfer learning on the ERA5-trained ML models for GWFs from a 1.4km global climate model. It is found that the re-trained models both (a) preserve their learning from ERA5, and (b) learn to appropriately scale the predicted fluxes to account for ERA5's limited resolution. Our results highlight the importance of embedding nonlocal information for a more accurate GWF prediction and establish strategies to complement abundant reanalysis data with limited high-resolution data to develop machine learning-driven parameterizations for missing mesoscale processes in climate models."

Note: the TEI text has typographic gaps ("a ANN, a ANN-CNN") where the original contains "a 1×1 ANN, a 3×3 ANN-CNN" — the Semantic Scholar abstract confirms the models as "a 1×1 ANN, 3×3 ANN-CNN, and Attention UNet". [SNIPPET] A Wiley-indexed snippet describes them as "a local single-column artificial neural network M1, a locally nonlocal 3x3 columns artificial neural network M2 with one preceding convolutional layer, and a globally nonlocal Attention UNet convolutional neural network M3".

---

## 2. Model architectures (exact, from the paper's own code) [CODE]

Source file: `utils/model_definition.py` in DataWaveProject/nonlocal_gwfluxes (local clone: `scratchpad/discovery/nonlocal_gwfluxes/`).

### 2.1 M1 — single-column ANN (`ANN_CNN` class with `stencil=1`)

Pure MLP (the conv branch is skipped for stencil=1). Layer stack, all Linear:

| layer | in → out | activation | dropout after |
|---|---|---|---|
| layer1 | idim → hdim | LeakyReLU | p=0.1 |
| layer2 | hdim → hdim | LeakyReLU | p=0.1 |
| layer3 | hdim → hdim | LeakyReLU | p=0.1 |
| layer4 | hdim → hdim | LeakyReLU | p=0.1 |
| layer5 | hdim → hdim | LeakyReLU | p=0.1 |
| layer6 | hdim → 2·odim | LeakyReLU | p=0.1 |
| output | 2·odim → odim | none (linear) | — |

- `hdim = 4 * idim` (set in `training.py`).
- `dropout = 0.1` during training; 0.0 at inference.
- No batch norm in this published version. (Caveat: `Model_Freeze_Transfer_Learning` in `function_training.py` references `model.bnorm6.*` for the ANN, implying a batchnorm-bearing variant existed at some stage; the shipped `ANN_CNN` has no `bnorm6` — a code inconsistency worth noting for replication.)
- There is also an `ANN_CNN10` "ABLATION_10hiddenlayers" variant (layers 1–9 at hdim, layer10 → 2·odim, output) — used for an ablation, toggled by `ablation=False` flag.

Dimensions for the main ("vertical=global", troposphere+stratosphere) configuration [CODE+DATA]:
- features `uvtheta`: idim=369 (=3 static + 3×122 levels), hdim=1476, odim=244 → **10,106,420 params** [INFERENCE: computed by instantiating the class; consistent with HF checkpoint size 121,460,352 B ≈ 3×params×4 B for Adam(m,v)+weights]
- features `uvthetaw`: idim=491 (=3 + 4×122), hdim=1964, odim=244 → **17,481,564 params** [INFERENCE; checkpoint 210,010,138 B ✓]
- stratosphere_only uvtheta: idim=183 (=3+3×60), odim=120 → 2,485,752 params [INFERENCE; checkpoint ~29.9 MB ✓]

### 2.2 M2 — 3×3 ANN-CNN (`ANN_CNN` with `stencil=3`)

Same MLP as M1, preceded by ONE convolution block:
- `nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0)` — collapses the 3×3 neighborhood to 1×1 (per-column image-to-pixel regression), ReLU activation, dropout p=0.5×0.1=0.05 after conv.
- For stencil=5, two consecutive 3×3 convs (code comment: "Applying multiple 3x3 conv layers than just one stencil×stencil layer performs better"). stencil=5 is supported in code; the paper's M2 is the 3×3 model.
- Params: uvtheta 369ch: **11,332,238**; uvthetaw 491ch: **19,651,784** [INFERENCE from code; HF ckpt sizes 136,172,760 B and 236,055,226 B ✓]

### 2.3 M3 — Attention UNet (`Attention_UNet` class)

Classic 4-down/4-up Attention U-Net (Oktay-style additive attention gates, but with 3×3 convs inside the gates):

- Encoder: `Conv_block` channels ch_in→64→128→256→512→1024, each block = [Conv2d 3×3 pad 1 → BatchNorm2d → ReLU(inplace)] ×2; downsampling via MaxPool2d(kernel=2, stride=2); dropout p=0.1 after every encoder block during training ("applying dropout only during downsampling").
- Decoder: `Upsample` block = nn.Upsample(scale_factor=2) → Conv2d 3×3 → BatchNorm2d → ReLU; channels 1024→512→256→128→64; after each upsample, attention-gated skip concat then `Conv_block` (1024→512, 512→256, 256→128, 128→64).
- Attention gate `Attention_block(F_x, F_g, F_int)` at each skip: Wx = Conv2d(F_x→F_int, 3×3, pad 1)+BN; Wg = Conv2d(F_g→F_int, 3×3, pad 1)+BN; α = Sigmoid(BN(Conv2d(F_int→1, 3×3)))( ReLU(Wx(x)+Wg(g)) ); skip scaled as x·α. Gates: (512,512,256), (256,256,128), (128,128,64), (64,64,32). `attn_3d=False` default → single-channel attention map (F_attn=1).
- Head: `nn.Conv2d(64 → ch_out, kernel_size=1)`.
- Input channels: the UNet drops the 3 static channels (lat/lon/zs): `np.arange(3, …)` in the dataloader → ch_in = 366 (uvtheta) or 488 (uvthetaw); ch_out = 244.
- Params [INFERENCE from code]: uvtheta **37,892,576**; uvthetaw **37,962,848**. [FT-PAPER] states the baseline UNet has "over 35 million learnable parameters" — consistent.

---

## 3. Inputs, outputs, grid, normalization

### 3.1 Grid and channel layout [DATA — inspected `test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc`]

- Horizontal grid: **64 Gaussian latitudes × 128 longitudes** (T42-like, ≈2.8°). lat from −87.86° to +87.86° (south→north), lon 0→357.19° in 2.8125° steps. Stored scaled: lat/90, lon/360.
- Dimensions: `time` (hourly; e.g., month file for Aug), `idim=491`, `odim=244`, `lat=64`, `lon=128`. time variable = hour-of-year index (e.g., 5089=1 Aug 2015 00 UTC).
- `features(time, idim, lat, lon)` float32. Channel order: **[lat map, lon map, zs (surface elevation), u×122, v×122, θ×122, w×122]** — verified: ch0 varies only with lat, ch1 only with lon, ch2 orography-like in [−0.005, 1.010].
- `output(time, odim, lat, lon)` float32 = **[u′ω′ ×122 levels, v′ω′ ×122 levels]**.
- Vertical levels: 122 model levels. [FT-PAPER] "122 levels (top 15 of 137 removed due to damping; surface to ~45 km)" — i.e., ERA5's 137 model levels minus the top 15 sponge levels. (The .nc `units` attribute strings say "levels 15 to 74 (60 levels), 1 to 200 hPa" — that text is stale metadata copied from the stratosphere_only files; the actual channel counts (122/feature, 244 out) are authoritative. Flagged honestly as a metadata inconsistency.)
- Stratosphere_only variant files: 60 levels/feature (ERA5 model levels 15–74, 1–200 hPa), features {u,v,θ,w,N²}, outputs 120ch.
- `stratosphere_update` variant [CODE]: full 122-level inputs, outputs restricted to the top 60 levels of each flux (channels 0–59 and 122–181) → odim 120.

### 3.2 Normalization (exact constants, verbatim from file attributes) [DATA]

Inputs ("scaled to ~[-2,2] using mu-sigma scaling"):
- `u = (u − 6.395471175756457) / (3 × 22.175504140184618)`
- `v = (v − 0.020313991225046) / (3 × 9.84143148277375)`
- `theta = theta / 1000`
- `w = cuberoot[ (w − 0.0016040905945274022) / 0.017021397434040318 ]`
- lat = degrees_north/90, lon = degrees_east/360; zs scaled to ≈[0,1].

Outputs ("scaled to [-6,6] using mu-sigma scaling", cube-root transformed):
- `uw = cuberoot[ (uw − (−0.0005112474139891424)) / 0.0050768547492663395 ]`
- `vw = cuberoot[ (vw − (−0.0002982954242187403)) / 0.003792741148955207 ]`

Constants are GLOBAL (single mu/sigma per variable — "constant_mu_sigma_scaling" in every filename), not per-level.

### 3.3 Feature-set variants [CODE]

`uvtheta` (369/366ch), `uvthetaw` (491/488ch), `uvw`; stratosphere_only additionally `uvthetaN2`, `uvthetawN2`. Published checkpoints exist for uvtheta and uvthetaw (both vertical domains).

---

## 4. Training setup [CODE — `era5_training/training.py`]

- Loss: `nn.MSELoss()` (on normalized, cube-root-scaled outputs).
- Optimizer: Adam, initialized lr=1e-4.
- LR schedule: `CyclicLR(base_lr=1e-4, max_lr=5e-4, step_size_up=50, step_size_down=50, cycle_momentum=False)` — stepped per BATCH.
- Epochs: `nepochs = 100` for ANNs, `150` for Attention UNet. Published checkpoints at epochs 94 (M1 uvtheta/uvthetaw), 52/80 (M2), 100/119/131/138 (M3 variants) → best-epoch selection, not final-epoch. [INFERENCE from HF filenames]
- Batch size: 40 (stencil 1, = 40 hourly global samples → 40×64×128 columns per batch for the ANN after reshape), 20 (stencil 3); UNet trained with the same `bs_train` variable (40) as image batch.
- Dropout 0.1; `torch.manual_seed(123)`; no input shuffling in the shipped config (`manual_shuffle=False`, `shuffle=False`; a seed-51 permutation option exists).
- Train/val split [CODE]: train = **2010, 2012, 2014** (36 monthly files), validation/test = **2015** (12 monthly files); README: "the models are trained on three years of ERA5 data, and a fourth year is used for validation". An alternative split in `utils/files.py` uses all four years except May 2015 for training and **May 2015** as the test month (this matches [ICML] "four years: 2010, 2012, 2014, and 2015 at an hourly resolution ... May 2015 reserved for testing"). [FT-PAPER] says "4 years, i.e., 48 months, of ERA5 background conditions" ≈ "roughly 35k training+validation samples" (35,064 hours in 4 years [INFERENCE]).
- Hardware [FT-PAPER, describing the anchor's UNet]: "around 110 hours to complete 100 epochs" on a single A100 GPU; lr 1e-4.
- Data files: monthly NetCDF, hourly snapshots, naming `1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_{YYYY}_constant_mu_sigma_scaling{MM}.nc` (nonlocal 3×3 files have a `nonlocal_3x3_` prefix and per-pixel 3×3 patches as extra trailing dims).

---

## 5. Flux extraction from ERA5 / IFS (Helmholtz decomposition) [SCIDATA — full text read]

Method (defined for IFS-1km; "The optimized code was also used to compute momentum fluxes for ERA5"):

1. Helmholtz decomposition of horizontal wind (u,v) into rotational + divergent parts via spherical harmonics (WindSpharm/PySpharm; heavily optimized custom PySpharm).
2. GW perturbations = divergent flow minus its **T21 truncation**: `(u′_div, v′_div) = (u_div − u_div,T21, v_div − v_div,T21)` — i.e., total horizontal wavenumbers >21 of the divergent component are "GW".
3. Vertical-velocity perturbation: `ω′ = ω − ω̄` (zonal mean removed).
4. Fluxes: `F = (Fx, Fy) = g⁻¹ (u′_div ω′, v′_div ω′)`, g = −9.81 m/s² (sign convention as printed), units Pa.
5. Coarse-graining: **first-order conservative regridding (xESMF)** from native grid to **T42 Gaussian (~2.8°, 64×128)** — "averaging the fluxes over wavenumbers 42 and above", which averages over wave cycles and removes phase dependence.
6. For IFS-1km specifically, T7999 spherical harmonics were truncated to T3999 before decomposition (compute optimization).

ERA5 native resolution quoted as ~25–30 km ([FT-PAPER] "coarse-grained from 25 km"; [ICML] "ERA5 at its native 30 km resolution"). ERA5 fields: hourly, 137 model levels; anchor uses u, v, θ (from T), ω→w, on 122 retained levels.

IFS-1km ("1.4 km global model") [SCIDATA]:
- ECMWF IFS Experimental Nature Run "XNR1K": TCo7999 cubic-octahedral grid, Δx ≈ 1.4 km, 137 levels, initialized 1 Nov 2018 00 UTC from ECMWF operational analysis, timestep 60 s, cycle 45r1, OSTIA SST/sea-ice forcing, run on Summit (INCITE).
- Free-running **4 months: Nov 2018 – Feb 2019**; deep convection AND gravity-wave parameterizations switched OFF; effective resolution ~6–8Δx (resolves wavelengths ≳10 km).
- 3-hourly instantaneous model-level fields; 961 timeframes.
- Coarse-grained flux dataset variables (Table 1): zsurf, p, u, v, ω, T, ∂u/∂p, ∂v/∂p, N², Fx=u′ω′/g, Fy=v′ω′/g at 3-hourly resolution on the T42 grid.

---

## 6. Evaluation protocol

### 6.1 Hellinger distance [FT-PAPER — definition as printed there; the anchor uses the same metric per its abstract/snippets]

> "Given two probability densities, p and q, their Hellinger distance, ℋ, is defined as: ℋ(p,q) = 1 − ∫ₓ√(p(x)q(x))dx. By definition, ℋ ∈ [0,1]" — 0 = identical distributions a.e., 1 = disjoint supports.

(Note [INFERENCE]: this is the squared Hellinger distance in standard terminology; replicate as written.)

Anchor-paper results recovered [SNIPPET]: seasonally-averaged GWF distributions match with Hellinger distances < 0.01; for daily averages the NNs tend to underestimate GWFs (distribution tails), and "For daily averages, the Attention UNet model consistently has the lowest Hellinger distances and offers better predictions over both the distribution bulk and tails."

### 6.2 Hotspot regions

[ABSTRACT/SNIPPET] The paper evaluates transient (time-evolving) flux prediction over "prominent orographic and nonorographic hotspots" — one snippet says "six distinct orographic and nonorographic hotspots" are analyzed.

[CODE] `utils/dataloader_definition.py` defines 8 named regions as grid-index boxes (y measured south→north from lat −87.9°, x east from 0°; 2.79° lat, 2.8125° lon spacing):

| code name | y1:y2, x1:x2 | ≈ lat/lon box [INFERENCE from grid] |
|---|---|---|
| 1andes | 3:21, 96:113 | 79°S–29°S, 90°W–42°W (Andes/Drake Passage) |
| 2scand | 45:58, 0:12 | 38°N–74°N, 0°–34°E (Scandinavia/European mountains) |
| 3himalaya | 41:54, 26:44 | 27°N–63°N, 73°E–124°E |
| 4newfound | 47:58, 103:119 | 43°N–74°N, 70°W–25°W (Newfoundland) |
| 5south_ocn | 8:17, 10:25 | 65°S–40°S, 28°E–70°E (Southern Ocean) |
| 6se_asia | 33:42, 32:49 | 5°N–30°N, 90°E–138°E (Southeast Asia) |
| 7natlantic | 31:44, 112:124 | 1°S–35°N, 45°W–11°W (North Atlantic ITCZ) |
| 8npacific | 27:47, 67:87 | 12°S–43°N, 188°E–245°E (tropical/North Pacific) |

[FT-PAPER] (its Figure 9, same research group, may mirror the anchor's regions) lists boxes: Tropical Pacific 170°W–130°W × 10°S–40°N; Newfoundland 70°W–30°W × 45°N–70°N; European Mountains 0°–30°E × 40°N–70°N; Himalayas/East Asia 75°E–120°E × 28°N–58°N; North Atlantic 45°W–15°W × 0°–30°N; Southeast Asia 90°E–135°E × 5°N–25°N; Drake Passage 90°W–45°W × 78°S–33°S; Southern Ocean 30°E–65°E × 65°S–45°S. These agree with the [CODE] index boxes to within one grid cell — strong cross-validation. The anchor's exact six chosen boxes/wording must be confirmed against the PDF.

### 6.3 Transient evolution [ABSTRACT/SNIPPET]

Offline evaluation on (a) time-averaged statistics (global GWF distributions, troposphere + stratosphere) and (b) time-evolving flux variability over hotspots; Attention UNet best, M2 "a close second". Exact transient metrics (correlations vs time, etc.) not recoverable without the PDF.

---

## 7. Transfer-learning experiment

[CODE — `ifs_transfer_learning/training_ifs_transfer_learning.py`, `utils/function_training.py`]:
- Source: ERA5-trained checkpoints (M1/M2/M3). Target: 4 months of IFS XNR1K 1.4-km fluxes (single consolidated NetCDF in ERA5-style format).
- `Model_Freeze_Transfer_Learning`: freeze ALL parameters, then unfreeze **the last two layers**: ANN → `layer6` + `output` (+ `bnorm6` in the referenced-but-absent variant); Attention UNet → the final decoder `Conv_block` (`upconv2`, 128→64) + final 1×1 conv (`conv1x1`). Code comment: "unfreezeing just the last layer might not be enough since it is linear and there is no nonlinearity."
- TL hyperparameters: nepochs=200, Adam, CyclicLR base 1e-4 → max **9e-4**, step_size 10/10 ("since low IFS data, step size is small"), MSE loss, dropout 0.1, batch 20 (ANN) / 80 (UNet), NO validation set ("Since not a lot of IFS data, opting for no validation set").
- Results [ABSTRACT]: re-trained models "(a) preserve their learning from ERA5, and (b) learn to appropriately scale the predicted fluxes to account for ERA5's limited resolution". [SNIPPET] "The ERA5 + IFS-1 km approach highlights a pathway to use heterogeneous high-resolution GW data sets to develop future data-driven parameterizations."

---

## 8. Data & code availability

### 8.1 Anchor paper's statements [SNIPPET — recovered from indexed Wiley text via search snippets; wording near-verbatim but NOT read directly from the article; verify against PDF]

- "The code to compute the GW momentum fluxes and to conservatively coarse-grain the fluxes, along with the modified PySpharm functions can be accessed at https://doi.org/10.17605/OSF.IO/GX32S in the python_scripts folder."
- "Gravity wave fluxes extracted from ECMWF's IFS-1 km run are available at native grid resolution at https://doi.ccs.ornl.gov/ui/doi/475."
- "The code for all the machine learning models, along with the jobscripts and inference scripts, is publicly available at https://doi.org/10.5281/zenodo.16415113, and the ML model checkpoints can be accessed at https://huggingface.co/amangupta2/nonlocal_gwfluxes."

Every one of these targets was independently verified to exist and match:

| artifact | verified content |
|---|---|
| Zenodo 10.5281/zenodo.16415113 | "DataWaveProject/nonlocal_gwfluxes: Publication share", v1.0.0, 2025-07-24, MIT, archives github.com/DataWaveProject/nonlocal_gwfluxes (44.6 kB zip) [read: Zenodo page] |
| github.com/DataWaveProject/nonlocal_gwfluxes | cloned; era5_training/, ifs_transfer_learning/, utils/ as documented above [CODE] |
| huggingface.co/amangupta2/nonlocal_gwfluxes | MIT; checkpoints: ANN_1x1 (uvtheta e94 121 MB, uvthetaw e94 210 MB, strat uvtheta e88/e100 ~30 MB), ANN_3x3 (uvtheta e52 136 MB, uvthetaw e80 236 MB, strat e38/e93 ~33.5 MB), AttentionUNet (global uvtheta e100 455 MB, uvthetaw e119 456 MB, strat uvtheta e131 / uvthetaw e138 ~454 MB), plus test_files/ 1x1 (48 MB) and 3x3 (434 MB) Aug-2015 samples [DATA] |
| doi.ccs.ornl.gov/ui/doi/475 | "Mesoscale Gravity Wave Momentum Fluxes from Kilometer-Scale ECMWF Experimental Nature Run", Gupta/Sheshadri/Anantharaj, DOI 10.13139/OLCF/2308755, released 2024-06-25, netCDF4, via Globus endpoint 57618e0a-2c99-45ff-9694-24141b92fa17 [read: ORNL landing page] |
| OSF 10.17605/OSF.IO/GX32S | "Data for 'Gravity Wave Momentum Fluxes from 1.4 km Global ECMWF Integrated Forecast System'" (coarse-grained fluxes + python_scripts) [cited in SCIDATA Data Records; OSF page itself not fetched] |

Auxiliary checkpoint repo (not in the availability statement): huggingface.co/amangupta2/iccs_coupling_checkpoints (3 checkpoints incl. `attnunet_era5_global_global_uvthetaw_mseloss_train_epoch119.pt`, used by the FTorch/Fortran coupling MWE in `era5_training/`; the MWE test input was hosted at `https://g-b56e81.7a577b.6fbd.data.globus.org/...` which now returns 404).

### 8.2 Companion SciData paper's statements (verbatim, full text read) [SCIDATA]

> "The data files for the coarse grained momentum fluxes, in netCDF format, can be accessed at: 10.17605/OSF.IO/GX32S. The high-resolution gravity wave momentum fluxes in netCDF4 format can be accessed at: https://doi.ccs.ornl.gov/ui/doi/475."

Code availability (as rendered in the PMC XML; leading text truncated in the XML at "python_scripts folder"):
> "... python_scripts folder. Raw climate model output from XNR1K at native grid resolution is available at https://doi.ccs.ornl.gov/ui/doi/475. A Stanford Redivis project that allows users to work with the data interactively in a Jupyter notebook environment has been staged at: 10.57761/rc6v-hf22. The default WindSpharm Python package is publicly available at https://ajdawson.github.io/windspharm/latest/, and the PySpharm Python package is publicly available at: https://pypi.org/project/pyspharm/. The xESMF package used for conservative coarsegraining is publicly available at: https://xesmf.readthedocs.io/en/stable/."

ERA5: Hersbach et al., ERA5 hourly data, DOI 10.24381/CDS.BD0915C6, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels [SCIDATA refs / FT-PAPER].

---

## 9. Access log (what I tried for the anchor full text; all honest failures)

| URL | result |
|---|---|
| https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS004977 (WebFetch) | HTTP 403 (Cloudflare) |
| https://onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2025MS004977 (WebFetch + local curl, browser UA) | 403 / Cloudflare "Just a moment" challenge page |
| https://essopenarchive.org/doi/pdf/10.22541/essoar.173869599.90237060 (curl ×2 UAs, WebFetch) | Cloudflare managed JS challenge / 403 |
| https://par.nsf.gov/servlets/purl/10653528 | "Not Found Error" page (record exists but no served full text); /biblio/10653528 loads (metadata+abstract only); /media/xml = TEI header with abstract |
| https://d197for5662m48.cloudfront.net/.../145940/preprint_pdf/... (search-indexed ESSOAr CDN) | DNS ENOTFOUND (host retired) |
| Europe PMC, OSTI API, NTRS API (DOI + title queries) | 0 records |
| OpenAlex / Unpaywall / Semantic Scholar / Crossref / DOAJ | metadata only; best_oa pdf = the blocked Wiley pdfdirect |
| Wayback Machine (Wiley page, ESSOAr pdf) | no snapshots |
| r.jina.ai proxy | 401 (now requires API key) |
| scholar.archive.org search | HTTP 500 |
| eddy.stanford.edu/publications | 403 |
| amangupta2.github.io | loads; links only to the (blocked) Wiley DOI, no self-hosted PDF |
| Browser automation (claude-in-chrome) | extension not installed in this session |

**Recommended manual step:** the user can download the PDF in an ordinary browser from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS004977 (it is CC BY / free) — that would let us verify the [SNIPPET] items, the exact hotspot boxes, transient metrics, and the Limitations section wording.

---

## 10. Stated limitations & open questions (what could be recovered)

- [ABSTRACT] "even ERA5 does not resolve a substantial portion of GWFs" — motivates the IFS-1km transfer learning; ERA5-trained models under-predict absolute flux magnitudes.
- [SNIPPET] NNs underestimate daily-averaged GWFs in the tails (small/extreme values problem), while seasonal averages are near-perfect (H < 0.01).
- [SCIDATA] IFS-1km data limited to 4 months (Nov 2018–Feb 2019) → "low-volume, high-fidelity" regime; no validation set used in TL. Effective resolution of "1.4 km" run is really ~8–11 km.
- [CODE] TL retrains only the last two layers; code comments flag that unfreezing only the final linear layer is insufficient ("error reduction in TL training is low and not good enough" — sic).
- [INFERENCE] Offline-only evaluation (title: "Offline Performance") — online/coupled testing left as future work; the repo contains an FTorch/Fortran coupling MWE (`infer.f90`, pt2ts.py) pointing toward online coupling work with ICCS.
- Full verbatim Limitations/Discussion section: NOT recovered — requires the PDF.

## 11. Quick-reference replication card (reduced-scale)

- Data: 64×128 T42 Gaussian grid, hourly ERA5; inputs 366–491 channels (lat/lon/zs + u,v,θ[,w] on 122 model levels, normalized as §3.2 with global constants + cube-root on w and fluxes); outputs 244 channels (u′ω′, v′ω′ × 122 levels).
- M1: MLP 5×(4·idim) hidden + (2·odim) + linear head, LeakyReLU, dropout 0.1 (~10–17.5M params).
- M2: 1× Conv2d(idim→idim, 3×3, valid) + ReLU then the same MLP (~11–19.7M params).
- M3: Attention U-Net 64-128-256-512-1024, BN+ReLU double-conv blocks, additive 3×3 attention gates, 1×1 head, dropout 0.1 encoder-side (~38M params); drop static channels.
- Train: MSE, Adam, CyclicLR 1e-4↔5e-4 (per-batch, 50/50), 100 (ANN) / 150 (UNet) epochs, batch 40/20, train 2010+2012+2014, validate 2015, select best epoch.
- TL: freeze all but last two layers, CyclicLR 1e-4↔9e-4 (10/10), 200 epochs, MSE, on 4 months of 3-hourly XNR1K fluxes.
