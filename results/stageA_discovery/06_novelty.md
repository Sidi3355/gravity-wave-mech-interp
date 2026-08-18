# 06 — Novelty audit (CRITIC role): can the mech-interp-for-GW-parameterization claim be broken?

Date: 2026-08-18. Searches: web (arXiv, AGU/Wiley, PNAS, Semantic Scholar citation graph of Pahlavan et al. 2024 GRL, DataWave site, CCAI workshops).

Depth codes: (a) attribution/saliency, (b) kernel/weight analysis, (c) probing, (d) feature dictionaries/SAEs, (e) causal interventions/patching/steering, (f) representation similarity (CKA etc.).

Evidence basis: papers marked [fetched] were fetched (at least abstract/landing page read via WebFetch). Papers marked [snippet] were classified from substantive search-result abstract text only (fetch blocked or not attempted); treat those classifications as provisional.

---

## 1. Verdict up front

- **The NARROW claim survives (for now):** I found NO work in categories (c)-(f) applied to gravity-wave flux parameterizations / GW drag emulators. Deepest existing GW-NN interpretability remains (a) SHAP and (b) kernel/Fourier analysis.
- **The BROAD claim is DEAD as stated:** "prior interpretability in this domain stops at Fourier kernel analysis and saliency" is only true if "domain" is restricted to GW parameterizations. For climate/weather NN surrogates generally, 2025-2026 produced a rapidly growing body of genuine mechanistic interpretability: SAEs + causal feature steering on GraphCast, linear probing of Aurora, CKA between GraphCast and Aurora, SAEs on geophysical toy models and on a continuum-dynamics foundation model. Any novelty statement of the form "first mech interp in earth-system ML" or "little exists for climate surrogates" will be refereed to death.
- **Required reframing:** claim novelty as "first mechanistic interpretability (probing/SAE/patching/CKA with physical baselines) of *subgrid parameterization emulators*, specifically GW momentum flux," and position against the 2025-2026 weather-foundation-model mech-interp wave as methods-transfer neighbors, not as an empty field.
- **Threat level / urgency:** HIGH. Three independent groups published earth-system SAE/probing papers between Nov 2025 and Jun 2026, and the Eyring group (Haslauer et al. 2026) is already publishing interpretability of GW-flux NNs (currently SHAP-only). A GW-specific SAE/probing paper could appear within months.

---

## 2. Papers that most directly threaten the claim

### 2.1 MacMillan & Ouellette (2025) — CLOSEST PRIOR WORK [fetched]
- T. MacMillan, N. T. Ouellette, "Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features," arXiv:2512.24440 (submitted 30 Dec 2025). https://arxiv.org/abs/2512.24440
- SAEs on GraphCast internal activations; features for tropical cyclones, atmospheric rivers, diurnal/seasonal signals, geography, sea ice; **causal case study: steering a tropical-cyclone feature produces physically consistent modifications to evolving hurricanes**.
- Depth: **(d) + (e)** on a weather foundation model. Not GW, not a parameterization. This is the single paper that most constrains our novelty narrative: "LLM-style SAE + intervention applied to an earth-system NN" already exists.

### 2.2 Rosenfeld & Sonnewald (2026) [fetched]
- K. Rosenfeld, M. Sonnewald, "Sparse probes and murky physics: a case study of interpretability challenges in a foundation model for continuum dynamics," arXiv:2606.11657 (10 Jun 2026). https://arxiv.org/abs/2606.11657
- SAEs/sparse probing of the Walrus (Polymathic) continuum-dynamics foundation model; >20k features scored against physical baselines (enstrophy); finds intermittent, only piecewise-physical structure; explicitly discusses artifact-vs-structure pitfalls.
- Depth: **(d)** with physical-grounding methodology. Not climate-specific but fluid dynamics; Sonnewald is an ocean/climate ML figure. Directly anticipates our "physical baselines" framing AND supplies a cautionary-tale narrative we must engage with.

### 2.3 Craig et al. (2026) — CKA already done for weather models [fetched]
- G. Craig, T. Selz, M. Beylich, K. I. Tempest, "The physics of AI weather models," arXiv:2605.23778 (22 May 2026). https://arxiv.org/abs/2605.23778
- Uses Centered Kernel Alignment to compare internal representations of GraphCast and Aurora (CKA > 0.73 reported in related search text).
- Depth: **(f)** for weather foundation models. Kills any claim that representation-similarity analysis is new to atmospheric ML.

### 2.4 Richards & Balan (2025) — linear probing of Aurora [fetched]
- B. Richards, P. K. Balan, "Physical Consistency of Aurora's Encoder: A Quantitative Study," arXiv:2511.07787 (11 Nov 2025). https://arxiv.org/abs/2511.07787
- Linear classifiers on Aurora encoder embeddings for land-sea boundary, extreme temperature events, atmospheric instability.
- Depth: **(c)** on a weather foundation model. No interventions/SAEs.

### 2.5 Cheon (2026) [fetched — PDF metadata + abstract]
- M. Cheon, "Beyond Linear Superposition: Discovering Climate Features in AI Weather Models with KAN-SAE," arXiv:2605.17493 (19 May 2026). https://arxiv.org/abs/2605.17493
- KAN-augmented SAEs on AI weather model activations (heatwave detector, tropical-cyclone tracker features); explicitly tests the linear representation hypothesis in weather models.
- Depth: **(d)**. No GW/parameterization content found.

### 2.6 King et al. (2025) [snippet — Wiley returned 403; classified from publisher/search abstract text]
- King et al., "Leveraging Sparse Autoencoders to Reveal Interpretable Features in Geophysical Models," J. Geophys. Res.: Machine Learning and Computation, 2025, doi:10.1029/2025JH000769. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000769
- SAEs to untangle polysemantic neurons in toy geophysical/precipitation models; links neurons to known physical phenomena; ~33% reduction in sensitive inputs per neuron.
- Depth: **(d)** for geophysical NNs (toy scale). Peer-reviewed AGU venue — hardest to wave away as "just arXiv."

### 2.7 Aurora latent analysis (2026) [snippet]
- "Does Aurora Encode Atmospheric Structure? Latent Regime Analysis and Attribution," arXiv:2606.26361. https://arxiv.org/abs/2606.26361
- Spatially pooled PCA of Aurora latents + layer-wise relevance propagation; latent space organized by seasonality; storms not linearly separable.
- Depth: **(a)/(c)-adjacent** (unsupervised latent analysis + attribution).

---

## 3. Interpretability in the PARAMETERIZATION domain (convection etc.) — deeper than saliency, predating the LLM-style wave

These break any implication that parameterization-NN interpretability is saliency-only, even though none are GW and none use the modern probing/SAE/patching toolkit on hidden activations of a flux emulator.

### 3.1 Brenowitz et al. (2020) [snippet — arXiv abstract text via search]
- N. D. Brenowitz, T. Beucler, M. Pritchard, C. S. Bretherton, "Interpreting and Stabilizing Machine-learning Parametrizations of Convection," J. Atmos. Sci. (2020); arXiv:2003.06549. https://arxiv.org/abs/2003.06549
- Jacobians / linearized response functions of convection NNs **coupled to simplified (2-D linearized gravity-wave) dynamics**; input-ablation interventions (removing upper-atmospheric humidity) to locate a spurious learned causal shortcut that destabilized coupled runs.
- Depth: input/output-level **causal analysis, (e)-adjacent** (not internal-activation patching). NOTE the irony: the canonical convection-interpretability paper already couples NN response functions to gravity-wave dynamics. We must cite and clearly differentiate (internal circuits vs input-output Jacobians).

### 3.2 Behrens et al. (2022) [snippet — abstract text via search]
- G. Behrens, T. Beucler, P. Gentine, F. Iglesias-Suarez, M. Pritchard, V. Eyring, "Non-Linear Dimensionality Reduction With a Variational Encoder Decoder to Understand Convective Processes in Climate Models," JAMES 14, e2022MS003130 (2022); arXiv:2204.08708. https://doi.org/10.1029/2022MS003130
- VED with 5 latent nodes for superparameterized convection; latent space identifies convective regimes; **direct manipulation of the latent manifold** (generative interventions) to connect regimes to large-scale drivers.
- Depth: **(d)-adjacent latent feature analysis + (e)-adjacent latent manipulation**, in the convection-parameterization domain. Differentiator for us: the VED is a purpose-built interpretable surrogate, not a post-hoc dissection of an operational emulator's hidden layers.

### 3.3 Shamekh & Gentine et al. (2023) [snippet]
- S. Shamekh et al., "Implicit learning of convective organization explains precipitation stochasticity," PNAS 120, e2216158120 (2023). https://www.pnas.org/doi/10.1073/pnas.2216158120
- Trained encoder bottleneck yields an interpretable learned "organization" latent; decoder + rotation-invariant loss used to interpret what the latent encodes.
- Depth: **(c)/(d)-adjacent** (interpreting a learned bottleneck variable), convection/precipitation domain.

### 3.4 Song & Kuang (2025) [fetched]
- Q. Song, Z. Kuang, "Physically Interpretable Emulation of a Moist Convecting Atmosphere With a Recurrent Neural Network," GRL (2025); arXiv:2501.08513. https://arxiv.org/abs/2501.08513
- Interpretable-by-design: explicit linear state-space component inside the RNN cell; linear response analysis of the coupled emulation (convection + 2-D gravity waves).
- Depth: interpretable-by-design + linear-response, **(b)/(e)-adjacent**, not post-hoc mech interp.

---

## 4. GW-parameterization interpretability specifically (the direct competition) — all (a)/(b)

- **Pahlavan, Hassanzadeh & Alexander (2024)** [snippet; the paper our claim already cites as the frontier]: "Explainable Offline-Online Training of Neural Networks for Parameterizations: A 1D Gravity Wave-QBO Testbed in the Small-Data Regime," GRL 51, e2023GL106324 (2024). https://doi.org/10.1029/2023GL106324 — Fourier/spectral analysis of learned CNN kernels. Depth: **(b)**.
- **Haslauer et al. (2026)** [fetched]: E. Haslauer, M. Schwabe, A. Doernbrack, E. P. Gerber, M. Rapp, N. Zagar, V. Eyring, "Interpretable Neural Networks to Predict Momentum Fluxes of Orographic Gravity Waves," arXiv:2605.05052 (6 May 2026); also at IOPscience doi:10.1088/3049-4753/ae7df2. https://arxiv.org/abs/2605.05052 — U-Net on ERA5 orographic GW momentum fluxes; **SHAP only** (confirmed by fetch: no probing, SAEs, patching, or interventions). Depth: **(a)**. Confirms active competitor interest (Eyring/DLR + Gerber) in interpreting GW-flux NNs.
- **Gupta et al. (2024/2025)** [snippet]: "Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation" (arXiv:2406.14775) and "Offline Performance of a Nonlocal Deep Learning Parameterization for Climate Model Representation of Atmospheric Gravity Waves," JAMES doi:10.1029/2025MS004977 — Attention U-Net used architecturally; any attention-map discussion is qualitative. Depth: **at most (a)**.
- **Connelly & Gerber (2024)** [snippet]: "Regression Forest Approaches to Gravity Wave Parameterization for Climate Projection" — tree feature importances. Depth: **(a)**.
- **Sun et al. (2024) JAMES; Mansfield & Sheshadri (2024) JAMES doi:10.1029/2024MS004292; Yang et al. (2026) JAMES doi:10.1029/2024MS004313** [snippet]: data imbalance / UQ / transfer learning — not interpretability of internals.
- **Gupta et al. (2025)** [snippet]: "Finetuning AI Foundation Models to Develop Subgrid-Scale Parameterizations: A Case Study on Atmospheric Gravity Waves," arXiv:2509.03816 / JAMES doi:10.1029/2025MS005075 — Prithvi-WxC finetuning for GW fluxes; no internal-representation analysis surfaced in searches. (If we probe finetuned foundation models, this is the substrate paper, not competing interp.)
- **Citation sweep** [fetched]: all 25 papers citing Pahlavan et al. 2024 GRL (Semantic Scholar, 2026-08-18) checked by title/abstract — none apply (c)-(f) to GW or parameterization NNs (closest: "Interpretable Structural Model Error Discovery From Sparse Assimilation Increments," QG turbulence, equation-discovery, not internal interp).
- **DataWave site sweep** [fetched, https://datawaveproject.github.io/]: no interpretability-focused outputs listed beyond the above.

---

## 5. What I did NOT find (searched, negative results)

- No probing classifiers, SAEs, activation patching, causal tracing, or CKA applied to any GW drag / GW flux emulator (direct keyword searches; citation graph of Pahlavan 2024; DataWave publication list; Sheshadri-group 2025-2026 outputs).
- No mech-interp dissection of ClimSim/CBRAIN emulator hidden layers with the modern toolkit (VAE-latent work in Sec. 3 is the closest).
- No mech-interp work on NeuralGCM's or ACE2's learned-physics internals.
- No circuits-style (ACDC etc.) analysis anywhere in earth-system ML.
- Caveats: several classifications rest on abstracts only; full texts may contain more than abstracts advertise (esp. Craig et al. 2026 and the Aurora papers). Wiley blocked direct fetches (403) for AGU journals, so King 2025 and several JAMES papers were classified from abstract snippets.

---

## 6. Implications for the research program

1. **Keep**: "first (c)-(f) mechanistic interpretability of GW flux parameterizations" — defensible today, perishable. Move fast; prioritize GW-specific results over toolkit breadth.
2. **Drop/reframe**: any "first in climate ML / earth-system ML" phrasing. Related work must lead with MacMillan & Ouellette 2025, Rosenfeld & Sonnewald 2026, Craig et al. 2026, Richards & Balan 2025, King et al. 2025, Cheon 2026, and the convection lineage (Brenowitz 2020; Behrens 2022; Shamekh 2023; Song & Kuang 2025).
3. **Differentiators to emphasize**: (i) parameterization emulators are small, physically constrained regression models — superposition/probing behave differently than in giant forecast transformers (Rosenfeld & Sonnewald's murky results make this a live scientific question, not gap-filling); (ii) online/coupled causal validation (patch an internal feature, run the coupled QBO testbed) — nobody has closed the loop from internal feature to coupled-climate response; (iii) physical baselines specific to GW theory (critical levels, wave breaking, intermittency) rather than generic enstrophy/cyclone features; (iv) input-output Jacobian analysis (Brenowitz) vs internal-circuit analysis (ours) is a clean, citable distinction.
4. **Risk register**: Eyring group (Haslauer et al.) is one methods-upgrade away from (c)/(d) on GW NNs; MacMillan & Ouellette or Cheon could pivot SAEs to parameterizations; a JGR-MLC follow-up of King et al. on a real parameterization is plausible. Re-run this audit before any submission.
