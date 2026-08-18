# Citation verification: neural GW parameterization lineage (6 papers)

Role: WRITER (citation verification). Date checked: 2026-08-18.
Method: every paper verified against a fetched primary source (arXiv abs/html pages, Crossref API, Semantic Scholar API, NASA GISS abstract page, web search of publisher listings). No details below are from memory alone; caveats are flagged inline.

Status legend: VERIFIED = existence + bibliographic data confirmed from fetched primary/authoritative source.

---

## 1. Espinosa et al. 2022 (GRL) — ML emulation of a physics-based GW scheme; QBO + increased-CO2 generalization

**Status: VERIFIED** (NASA GISS abstract page + Wiley/AGU listing via search; full citation from GISS pubs page)

- **Title:** Machine Learning Gravity Wave Parameterization Generalizes to Capture the QBO and Response to Increased CO2
- **Authors:** Z. I. Espinosa, A. Sheshadri, G. R. Cain, E. P. Gerber, K. J. DallaSanta
- **Venue:** Geophysical Research Letters, 49(8), e2022GL098174, 2022
- **DOI:** 10.1029/2022GL098174
- **URLs:** https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL098174 ; https://www.giss.nasa.gov/pubs/abs/es06000g.html

**Summary (3-5 sentences):**
Espinosa et al. (2022) present single-column machine-learning gravity wave parameterizations (GWPs) that emulate the non-orographic gravity wave momentum forcing of a conventional physics-based GWP in an idealized climate model. The artificial neural networks are trained on deliberately limited data — one view of the annual cycle and only one phase of the Quasi-Biennial Oscillation (QBO). When coupled online to the climate model, moderately complex ANNs nonetheless generate full QBO cycles, i.e., they generalize out-of-sample to the unseen QBO phase. Under increased CO2 concentrations, the ANN-driven model's climate response matches that of the original physics-based parameterization. The authors conclude that ANNs can accurately emulate an existing scheme and generalize to new regimes given limited data, motivating future parameterizations learned directly from observational constraints on GW momentum transport.

**Caveats/flags:**
- The names "WaveNet" and "AD99" (Alexander & Dunkerton 1999, the emulated scheme) do NOT appear in the fetched GISS abstract text; the abstract only says the ANNs emulate a conventional GWP in an idealized climate model. The WaveNet/AD99 framing is how follow-on literature (e.g., Mansfield & Sheshadri 2024, JAMES, 10.1029/2024MS004292) refers to this lineage. If citing this paper for "WaveNet emulates AD99," confirm against the full PDF (author copy: https://edwinpgerber.github.io/files/espinosa_etal-GRL-2022.pdf) before using those specific names.

---

## 2. Hardiman et al. 2023 (AIES) — Met Office NN emulation of non-orographic GW scheme

**Status: VERIFIED** (AMS journal listing via search + Semantic Scholar API + Crossref API). **Year is 2023, not 2022** (manuscript ID AIES-D-22-0081 reflects 2022 submission).

- **Title:** Machine Learning for Nonorographic Gravity Waves in a Climate Model
  (AMS prints "Nonorographic" unhyphenated; Semantic Scholar renders it "non-orographic" — use the AMS form.)
- **Authors (Crossref, in order):** Steven C. Hardiman, Adam A. Scaife, Annelize van Niekerk, Rachel Prudden, Aled Owen, Samantha V. Adams, Tom Dunstan, Nick J. Dunstone, Sam Madge
- **Venue:** Artificial Intelligence for the Earth Systems (AIES), Vol. 2, Issue 4, October 2023 (published online 4 Oct 2023)
- **DOI:** 10.1175/AIES-D-22-0081.1
- **URL:** https://journals.ametsoc.org/view/journals/aies/2/4/AIES-D-22-0081.1.xml

**Summary (3-5 sentences):**
Hardiman et al. (2023) train a neural network to mimic the behavior of the non-orographic gravity wave drag scheme used in the Met Office climate model, a scheme important for stratospheric climate and variability. Notably, the network achieves accurate emulation using only two of the six inputs used by the original parameterization scheme, suggesting the potential for greater efficiency in the scheme. Hyperparameters were selected with the aid of a one-dimensional mechanistic model, chosen on the basis of emergent features of the coupled system rather than offline skill alone. Coupled into climate simulations, the emulator reproduces a quasi-biennial oscillation of the tropical stratospheric winds and correctly simulates gravity wave variability associated with ENSO and with the stratospheric polar vortex, despite not being explicitly trained on these modes of variability.

**Caveats/flags:**
- Listed in the task as "~2022-2023"; the publication year is definitively 2023.
- No second/companion Hardiman ML-GW paper surfaced in searches; this appears to be the single Met Office paper matching the description. (The superficially similar ECMWF emulation paper is Chantry et al. 2021, arXiv:2101.08195 — different group, do not conflate.)
- AMS e-locator (article-number) not confirmed from fetched sources; cite by volume/issue/DOI.

---

## 3. Wang, Yuval & O'Gorman 2022 (JAMES) — non-local inputs for subgrid parameterization

**Status: VERIFIED** (arXiv abs page 2201.00417 with journal reference + Wiley/AGU listing via search)

- **Title:** Non-Local Parameterization of Atmospheric Subgrid Processes With Neural Networks
- **Authors:** Peidong Wang, Janni Yuval, Paul A. O'Gorman
- **Venue:** Journal of Advances in Modeling Earth Systems (JAMES), 14(10), e2022MS002984, October 2022
- **DOI:** 10.1029/2022MS002984 (preprint: arXiv:2201.00417, v1 Jan 2022, v2 Dec 2022)
- **URLs:** https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022MS002984 ; https://arxiv.org/abs/2201.00417

**Summary (3-5 sentences):**
Wang, Yuval & O'Gorman (2022) test whether machine-learning parameterizations of atmospheric subgrid processes benefit from relaxing the single-column assumption, training neural networks on non-local inputs spanning 3x3 grid columns from high-resolution model output. Including the non-local inputs improves the offline prediction of a range of subgrid processes, with the improvement especially notable for subgrid momentum transport and for atmospheric conditions associated with mid-latitude fronts and convective instability. Using an interpretability (attribution) method, they find the NN improvements partly rely on the horizontal wind divergence, and they show that including the divergence or vertical velocity as a separate input substantially improves offline performance. Even with vertical velocity included, non-local winds remain useful inputs for parameterizing subgrid momentum transport. The authors conclude that non-local variables and the vertical velocity could improve ML parameterizations and should be tested in online simulations in future work.

**Caveats/flags:** none — matches the task description exactly (venue JAMES, year 2022, attribution analysis showing where nonlocality pays).

---

## 4. Fritts & Alexander 2003 (Rev. Geophys.) — the standard middle-atmosphere GW review

**Status: VERIFIED** (Crossref API; Wiley page 403-blocked but listing confirmed via search; authors' hosted PDF exists at NWRA but exceeded fetch size limit)

- **Title:** Gravity wave dynamics and effects in the middle atmosphere
- **Authors:** David C. Fritts, M. Joan Alexander
- **Venue:** Reviews of Geophysics, 41(1), 1003, 2003 (print March 2003; online 16 April 2003)
- **DOI:** 10.1029/2001RG000106
- **URLs:** https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2001RG000106 ; author copy: https://www.cora.nwra.com/~alexand/publications/FrittsAlexander03.pdf

**Summary (3-5 sentences):**
Fritts & Alexander (2003) is the comprehensive review of atmospheric gravity wave dynamics and their effects, motivated by the waves' myriad contributions to atmospheric circulation, structure, and variability. The major wave influences occur in the middle atmosphere, between roughly 10 and 110 km altitude, because of decreasing density and increasing wave amplitudes with altitude. The review synthesizes theoretical, numerical, and observational advances since the mid-1980s, covering gravity wave sources, propagation, spectral evolution, climatologies, and wave-mean-flow interactions including momentum flux deposition. It also emphasizes developments in gravity wave parameterizations enabling more realistic descriptions of gravity wave forcing in large-scale models, while identifying remaining knowledge gaps. It remains the standard citation for GW sources, critical-level filtering, and momentum-flux physics underpinning parameterization work.

**Caveats/flags:**
- Correct issue is 1 (Crossref: vol 41, issue 1, article citation number 1003). Some third-party citation aggregators (e.g., SCIRP) list "No. 3, pp. 1-64" — that is wrong; do not copy it. Standard citation form: Rev. Geophys., 41(1), 1003.
- A published Correction/Erratum to this paper exists (also Fritts & Alexander, Rev. Geophys.); relevant if quoting equations exactly.
- Abstract-level claims above come from the Crossref-indexed abstract and search snippets of the Wiley abstract, not the full text; the "critical-level filtering" phrase in the last sentence is a standard characterization of the review's content, not a fetched quote.

---

## 5. Gupta et al. 2024 (arXiv:2406.14775) — global ML simulation of nonlocal GW propagation; M1/M2/M3 ladder

**Status: VERIFIED** (arXiv abs page + arXiv HTML full text v2)

- **Title:** Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation
- **Authors:** Aman Gupta, Aditi Sheshadri, Sujit Roy, Vishal Gaur, Manil Maskey, Rahul Ramachandran
- **Venue:** arXiv preprint arXiv:2406.14775 (v1: 20 Jun 2024; v2: 13 Nov 2024). Categories: physics.ao-ph, cs.LG, physics.flu-dyn, physics.geo-ph. **No journal reference listed on the arXiv abs page** as of fetch date.
- **DOI:** 10.48550/arXiv.2406.14775
- **URL:** https://arxiv.org/abs/2406.14775

**Summary (3-5 sentences):**
Gupta et al. (2024) present what they describe as the first global simulation of atmospheric gravity wave momentum fluxes using machine learning, trained on the WINDSET dataset (four years — 2010, 2012, 2014, 2015 — of hourly ERA5-derived u, v, and potential temperature on a 64x128 global grid with 122 vertical levels) as an alternative to traditional single-column parameterizations. The central target is the "single-column approximation," which completely neglects horizontal evolution of subgrid processes and is blamed for key biases in current climate models. Using an Attention U-Net trained on globally resolved GW momentum fluxes (zonal and meridional vertical-flux components u'w' and v'w'), they illustrate the importance and effectiveness of global nonlocality when simulating GWs with data-driven schemes. Predictive skill is highest in the midlatitudes (R^2 ~ 0.6 for zonal flux), lower in the tropics (R^2 ~ 0.3-0.4), and the stratosphere remains challenging (negative R^2 in places).

**M1/M2/M3 architecture ladder (from arXiv HTML full text):**
- **M1 — single-column ANN:** 4 hidden layers, each twice the input size (366 inputs = u, v, theta on 122 levels), ReLU activations, Adam optimizer with cyclic learning rates; replicates the conventional single-column setting (column in, column out).
- **M2 — nonlocal 3x3 ANN-CNN:** predicts fluxes in a given column from the surrounding 3x3 grid of columns; a 3x3 convolution layer pools neighboring spatial data into a single-column representation (regional nonlocality).
- **M3 — global Attention U-Net:** encoder-decoder U-Net backbone with residual connections and attention multipliers at skip connections; ingests full global maps (64x128 resolution, 366 input channels) and decodes 244 output channels (the two flux components across 122 levels); global receptive field.
- Ladder result: skill increases from M1 to M2 to M3, which the authors read as direct evidence of "the importance of nonlocality and model complexity in learning the nonlinear evolution of atmospheric waves."

**Caveats/flags:**
- The task note that this shares the M1/M2/M3 architecture ladder with the anchor JAMES paper (10.1029/2025MS004977) is consistent with what is in this preprint, but the arXiv abs page does not (yet) cross-reference that JAMES DOI — confirm the correspondence from the JAMES paper's own text before asserting it in print.
- Fetched full text of this preprint says WINDSET derives from ERA5 at "30 km native resolution," while the companion preprint (2509.03816) fetch said 25 km; minor discrepancy between fetch summaries — check the source PDFs if the exact number matters.

---

## 6. Gupta et al. 2025 (arXiv:2509.03816) — Prithvi WxC foundation model fine-tuned for GW fluxes

**Status: VERIFIED — but the title in the task list was informal; actual title differs** (arXiv abs page + arXiv HTML full text v1)

- **Title:** Finetuning AI Foundation Models to Develop Subgrid-Scale Parameterizations: A Case Study on Atmospheric Gravity Waves
- **Authors:** Aman Gupta, Aditi Sheshadri, Sujit Roy, Johannes Schmude, Vishal Gaur, Wei Ji Leong, Manil Maskey, Rahul Ramachandran
- **Venue:** arXiv preprint arXiv:2509.03816 (v1: 4 Sep 2025). Categories: physics.ao-ph, cs.LG. No journal reference listed.
- **DOI:** 10.48550/arXiv.2509.03816
- **URL:** https://arxiv.org/abs/2509.03816

**Summary (3-5 sentences):**
Gupta et al. (2025) develop a machine-learning gravity wave parameterization by fine-tuning NASA/IBM's Prithvi WxC, a 2.3-billion-parameter transformer-based weather-climate foundation model: the pre-trained encoder-decoder is frozen, and four learnable convolutional blocks (160/320/640/1280 channels) are added before the encoder and after the decoder. The model learns GW momentum fluxes (u'w' and v'w', plus potential temperature for validation) from ERA5 reanalysis (25 km, 137 levels, hourly; fluxes extracted via Helmholtz decomposition) coarse-grained to ~280 km on a 64x128 grid — i.e., capturing GW effects for a coarse-resolution climate model by learning fluxes from a reanalysis with ~10x finer resolution — using training years 2010/2012/2014/2015 (~35,000 samples). Against an Attention U-Net baseline (>35M parameters; the M3-style architecture of arXiv:2406.14775), the fine-tuned foundation model shows superior predictive performance throughout the atmosphere, even in regions excluded from pre-training. The headline metric is the Hellinger distance between predicted and ERA5 flux distributions: 0.11 (baseline) vs 0.06 (fine-tuned FM) for global monthly averages, and 0.116 vs 0.062 for daily averages; Pearson correlations (e.g., 0.99 over the Drake Passage) and RMSE are also reported. The authors argue this demonstrates the versatility and reusability of foundation models for building observation-driven, physically accurate parameterizations of more Earth-system processes.

**Interpretability note (important for our paper):** per the fetched full text, the paper contains NO dedicated interpretability/explainability analysis — no attention visualization, feature attribution, or mechanistic analysis. It offers only qualitative claims that the model captures three-dimensional propagation/dissipation and lateral-propagation effects absent from traditional schemes. Do not cite it as containing interpretability content; cite it for the FM-finetuning approach and the Hellinger-distance evaluation.

**Caveats/flags:**
- Working description in task list ("Prithvi WxC foundation model fine-tuned for GW flux prediction") is accurate as a description but is not the title — use the exact title above.
- Hellinger distance confirmed as the headline metric (values above from abstract and full text).

---

## Cross-cutting notes for the paper

1. **Lineage structure confirmed by the fetched sources:** (a) emulation-of-existing-scheme lineage: Espinosa 2022 (GRL, idealized model) and Hardiman 2023 (AIES, Met Office full climate model) — both emulate physics-based non-orographic schemes and both demonstrate coupled-model QBO reproduction; Hardiman additionally finds only 2 of 6 scheme inputs are needed. (b) Learn-from-resolved-fluxes, nonlocal lineage: Wang et al. 2022 (JAMES, 3x3 nonlocality + attribution), Gupta et al. 2024 (global nonlocality, M1/M2/M3 ladder), Gupta et al. 2025 (foundation-model transfer, Hellinger metric). Fritts & Alexander 2003 is the physics anchor for all of it.
2. **Corrections vs. task list:** Hardiman is 2023 (not 2022) in AIES; Gupta 2025's actual title is "Finetuning AI Foundation Models to Develop Subgrid-Scale Parameterizations: A Case Study on Atmospheric Gravity Waves"; Gupta 2025 has no interpretability content despite the task's query. Everything else matches as listed.
3. **All six papers exist.** None failed verification.
