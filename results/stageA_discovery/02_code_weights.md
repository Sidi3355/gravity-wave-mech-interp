# Code & Pretrained Weights Hunt — Gupta et al. (2025), JAMES 10.1029/2025MS004977

Role: DATA ENGINEER. Date of search: 2026-08-18. All items below were fetched and verified (GitHub API, HuggingFace API, Zenodo API, curl HEAD checks) unless marked as a dead end.

## Verdict up front

**Training code: FOUND.** **Pretrained weights for M1/M2/M3: FOUND** (full set, ~3.1 GB, HuggingFace, MIT). **Paper-tagged Zenodo code archive: FOUND.** The author's GitHub username is **`amangupta2`** (NOT ag4680 — that 404s).

---

## 1. PRIMARY CODE REPO — DataWaveProject/nonlocal_gwfluxes

- URL: https://github.com/DataWaveProject/nonlocal_gwfluxes
- Description: "A set of NNs for predicting Nonlocal Gravity Wave Fluxes"
- License: **MIT**. Last push: **2026-02-04** (last commit 940d9515, 2026-02-02, "chore: add MIT license to pyproject.toml"). Repo size 369 KB (code only, no weights in git).
- Release/tag **1.0.0** (2025-07-24) = the publication share archived on Zenodo (below).
- Verified contents (main tree, fetched via GitHub API):
  - `era5_training/` — `training.py` (M1/M2 ANN-CNN training; CLI: `python training.py <horizontal_domain> <vertical_domain> <features> <stencil>`; stencil 1 = M1, 3 = M2), `inference.py`, `infer.py`/`infer.f90` (FTorch Fortran coupling MWE), `pt2ts.py` (TorchScript export), `get-model-and-data.sh` (downloads pretrained weights from HuggingFace), batch scripts.
  - `ifs_transfer_learning/` — `training_ifs_transfer_learning.py`, `ANN_inference.py`, `attn_inference.py` (transfer-learning ERA5 models onto 1-km IFS fluxes).
  - `utils/model_definition.py` (15.9 KB) — **verified class definitions**: `ANN_CNN(idim, odim, hdim, stencil, dropout)` (= M1 with stencil=1, M2 with stencil=3), `Conv_block`, `Upsample`, `Attention_block`, `Attention_UNet(ch_in, ch_out, dropout, attn_3d)` (= M3).
  - `utils/dataloader_definition.py`, `utils/function_training.py`, `utils/files.py`.
- Note: the top-level README's usage section refers to `ann_cnn_training/` and `attention_unet/` directories which no longer exist; current layout is `era5_training/` + `utils/`. The paper-era layout survives in the personal dev repo (Sec. 4).
- Branches of interest: `ablation_10layers`, `circular-unet`, `running_on_derecho`, `updates-for-high-resolution`.
- Contributors per Zenodo: @TomMelt, @amangupta2, @omarjamil (ICCS Cambridge), Aditi Sheshadri, @j-emberton.
- Training data basis: 3 years of ERA5 hourly (validation on a 4th year); features `uvtheta`/`uvthetaw`/`uvw` (+N2 variants for stratosphere_only).

## 2. PRETRAINED WEIGHTS (M1/M2/M3) — HuggingFace amangupta2/nonlocal_gwfluxes

- URL: https://huggingface.co/amangupta2/nonlocal_gwfluxes
- License: **MIT**. Last modified: 2025-06-04. Total ~**3.13 GB**. Files verified via HF API (`?blobs=true`), sizes in bytes:
  - `ANN_1x1/` (M1): `ann_cnn_1x1_global_global_era5_uvtheta__train_epoch94.pt` (121,460,352); `ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch94.pt` (210,010,138); `ann_cnn_1x1_global_stratosphere_only_era5_uvtheta__train_epoch88.pt` (29,939,486); `..._epoch100.pt` (29,939,640)
  - `ANN_3x3/` (M2): `ann_cnn_3x3_global_global_era5_uvtheta__train_epoch52.pt` (136,172,760); `ann_cnn_3x3_global_global_era5_uvthetaw__train_epoch80.pt` (236,055,226); `ann_cnn_3x3_global_stratosphere_only_era5_uvtheta__train_epoch38.pt` (33,561,166); `..._epoch93.pt` (33,560,206)
  - `AttentionUNet/` (M3): `attnunet_era5_global_global_uvtheta_mseloss_train_epoch100.pt` (455,007,084); `attnunet_era5_global_global_uvthetaw_mseloss_train_epoch119.pt` (455,851,134); `attnunet_era5_global_stratosphere_only_uvtheta_mseloss_train_epoch131.pt` (453,679,170); `attnunet_era5_global_stratosphere_only_uvthetaw_mseloss_train_epoch138.pt` (454,094,548)
  - `test_files/`: `test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc` (48,198,176); `test_nonlocal_3x3_inputfeatures_..._scaling08.nc` (433,538,979)
- **Download URL verified live** (HEAD 302 -> 200, content-length matches): `https://huggingface.co/amangupta2/nonlocal_gwfluxes/resolve/main/AttentionUNet/attnunet_era5_global_global_uvthetaw_mseloss_train_epoch119.pt`
- The .pt files are full PyTorch checkpoints; the repo's `inference.py` loads them and can export TorchScript via `--script`.

### 2b. Secondary weights repo — amangupta2/iccs_coupling_checkpoints
- URL: https://huggingface.co/amangupta2/iccs_coupling_checkpoints — MIT, last modified 2025-10-17.
- Holds the epoch94 ANN + epoch119 AttnUNet checkpoints PLUS **transfer-learned/retrained** ones: `retrained_L93_ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch70.pt` (122,560,966); `retrained_L93_attnunet_era5_global_global_uvthetaw_mseloss_train_epoch100.pt` (455,054,906); `retrained_ann_cnn_1x1_..._epoch45.pt` / `_epoch85.pt` (~264 MB each). "L93" = 93-level vertical grid (climate-model/IFS levels) — these are the IFS-transfer-learned versions.
- CAUTION (verified 404s): `.../resolve/main/ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch8.pt` referenced in `get-model-and-data.sh` returns **404** (removed), and the Globus test-input URL in that script (`https://g-b56e81.7a577b.6fbd.data.globus.org/1x1_inputfeatures_..._2010_...nc`) also returns **404**. Use `test_files/` in HF amangupta2/nonlocal_gwfluxes instead.

## 3. ZENODO ARCHIVES

- **10.5281/zenodo.16415113** — "DataWaveProject/nonlocal_gwfluxes: Publication share", 2025-07-24, MIT. File: `DataWaveProject/nonlocal_gwfluxes-1.0.0.zip` (44,554 bytes — code only, no weights). isSupplementTo -> https://github.com/DataWaveProject/nonlocal_gwfluxes/tree/1.0.0. Description confirms: "train single-column (M1), multiple columns (M2), and global Attention U-Net (M3) ... using resolved ERA5 gravity waves fluxes, and then transfer learn on high-resolution climate model output." Timing ("Publication share", 2025-07) makes this almost certainly the code DOI cited in the JAMES paper's Open Research statement.
- **10.5281/zenodo.16666812** — "amangupta2/gravity-wave-finetuning-james: Release v1.0.0 — Gravity Wave Flux Fine-Tuning with Prithvi WxC", 2025-08-01, MIT. File: zip, 11,454,667 bytes. isSupplementTo -> https://github.com/amangupta2/gravity-wave-finetuning-james/tree/1.0.0. Code archive for the companion paper arXiv:2509.03816.

## 4. PERSONAL / DEV REPOS (amangupta2)

- Profile: https://github.com/amangupta2 — Aman Gupta, "Stanford University", blog https://amangupta2.github.io/ (site source cites 2025MS004977). 9 public repos.
- **amangupta2/nonlocal_gwfluxes** (non-fork, MIT, last push 2024-11-13): original dev version with the paper-era layout: `ann_cnn_training/` (training.py, model_definition.py, inference.py, inference2.py, inference_ifs.py + epoch-log `.txt` files for 1x1/3x3/**5x5** × global/stratosphere_only uvtheta), `attention_unet/` (training_attention_unet.py, model_attention_unet.py, probabilistic_inference.py, inference_ifs.py), `ifs_transfer_learning/`, `quick_inference_script/two_sample_inference.py`. Useful for exact-paper reproduction and for the 5x5-stencil + probabilistic extras.
- **amangupta2/gravity-wave-finetuning** (non-fork, MIT, push 2025-02-14): personal copy of the Prithvi fine-tune code (+ test.py).
- **amangupta2/gravity-wave-finetuning-james** (fork of NASA-IMPACT/gravity-wave-finetuning, MIT, push 2025-08-01): the JAMES-companion release; verified contents: finetune_gravity_wave.py, gravity_wave_model.py (12,805 B), datamodule.py, inference.py, config.py, distributed.py, scripts/train.{sh,pbs}.

## 5. PRITHVI WxC GRAVITY-WAVE FINE-TUNE (companion work: arXiv:2509.03816, arXiv:2406.14775)

- Code: https://github.com/NASA-IMPACT/gravity-wave-finetuning — MIT; verified contents: finetune_gravity_wave.py (single/multi-node), gravity_wave_model.py, inference.py, datamodule.py, config.py, distributed.py, environment.yml, Prithvi-WxC submodule, scripts/train.pbs.
- Weights: https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300m-gravity-wave-parameterization — **Apache-2.0**, last modified 2025-09-05. Checkpoint **`magnet-flux-uvtp122-epoch-99-loss-0.1022.pt` = 9,919,872,996 bytes (~9.9 GB)** + `config.yaml`. Tags link base model `ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M` and arXiv 2406.14775.
- Dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/gravity-wave-parameterization — Apache-2.0; one validation-month ERA5 pair file `wxc_input_u_v_t_p_output_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling05.nc` = **20,893,258,746 bytes (~20.9 GB)**; 64 lat x 128 lon grid; years 2010/2012/2014/2015 used for the full dataset. (HF id `Prithvi-WxC/Gravity_wave_Parameterization` redirects here.)
- arXiv:2406.14775 (ICML 2024 ML4ESM workshop; Gupta, Sheshadri, Roy, Gaur, Maskey, Ramachandran; CC-BY 4.0): abstract page lists NO code/data links; the code trail runs through the repos above (the DataWave repo README cites this paper as reference [1]; it introduces the WINDSET dataset).

## 6. DATAWAVE PROJECT

- Website: https://datawaveproject.github.io/ ("international consortium ... gravity waves"; Schmidt Sciences/VESRI). Links to GitHub org https://github.com/DataWaveProject.
- Org repos most relevant (of 26 total): `nonlocal_gwfluxes` (THE paper repo), `FTorch-coupling-examples` (Fortran NN-GW coupling tests, push 2025-06-09), `CAM_GW_pytorch_emulator` (Python<->CAM coupling, push 2025-04-24), `MiMA-machine-learning`, `MiMAv1.0_Wavenet`, `WaveNet_UQ`, `qbo1d`, `EKI_QBO`, `Loon-momentum-fluxes`, `GWMF_healpix`, `CAM` (fork, push 2026-05-20).

## 7. DEAD ENDS (recorded honestly)

- `github.com/ag4680` — **404**; correct username is `amangupta2` (found via GitHub code search hit on his homepage citing 2025MS004977, and via Zenodo record 16666812).
- Wiley article page and pdfdirect for 10.1029/2025MS004977: **HTTP 403** (bot-blocked, including with browser UA) — could NOT read the paper's verbatim Open Research/Data Availability statement. Crossref confirms: published 2025-10, CC-BY 4.0, funders = Schmidt Futures, NSF, DOE.
- ESS Open Archive preprint (10.22541/essoar.173869599.90237060): **403** via WebFetch and via curl with browser UA.
- web.archive.org: blocked by the fetch tool. Europe PMC: paper not indexed. Semantic Scholar: has record, OA PDF points back to blocked Wiley URL.
- Zenodo full-text search for records citing "2025MS004977": 0 hits.
- GitHub repo searches for "gravity wave attention unet" and "WINDSET gravity wave": 0 hits (repo names/descriptions do not carry those terms).
- HF datasets under author amangupta2: none exist.

## 8. TOP ASSETS FOR MECH-INTERP WORK (ranked)

1. **HF amangupta2/nonlocal_gwfluxes** — all 12 M1/M2/M3 checkpoints (uvtheta and uvthetaw, global and stratosphere_only) + 2 test .nc inputs. Everything needed for offline forward passes and probing.
2. **GitHub DataWaveProject/nonlocal_gwfluxes** (+ tag 1.0.0 / Zenodo 10.5281/zenodo.16415113) — model classes (`ANN_CNN`, `Attention_UNet`), dataloaders, training + inference scripts that load those exact checkpoints, TorchScript/FTorch export path.
3. **GitHub amangupta2/nonlocal_gwfluxes** — paper-era code layout incl. 5x5-stencil variants, probabilistic inference, and per-epoch training logs (.txt) for training-dynamics context.
4. (Companion) Prithvi WxC GW fine-tune: 9.9 GB checkpoint (ibm-nasa-geospatial) + 20.9 GB ERA5 sample month + NASA-IMPACT/gravity-wave-finetuning code — same group, useful cross-model comparison.
