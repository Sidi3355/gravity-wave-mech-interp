# Mechanistic Interpretability of Deep Learning Gravity Wave Parameterizations

An autonomous research program auditing the neural gravity-wave parameterization
hierarchy of Gupta, Sheshadri, Roy & Anantharaj (2025), *Offline Performance of a
Nonlocal Deep Learning Parameterization for Climate Model Representation of
Atmospheric Gravity Waves*, JAMES, [doi:10.1029/2025MS004977](https://doi.org/10.1029/2025MS004977).

**Goal:** a causal, mechanistic account of why horizontal nonlocality improves
neural gravity-wave flux prediction and what these networks internally represent.

**Paper draft:** *Skill Without Physics: A Mechanistic Audit of Neural
Gravity-Wave Parameterizations* — [`PAPER_DRAFT.md`](PAPER_DRAFT.md) (prose),
[`paper/main.pdf`](paper/main.pdf) (compiled ICML-format draft).

## Headline findings

1. **Position, not physics, carries much of the U-Net's calibration.** Convolutional
   padding lets the network infer absolute position; rolling the input map (which
   changes no physics) collapses flux-variance calibration from 1.10 to 0.33. Roll
   ensembles attribute up to 62% of variance calibration and 37% of extreme-tail
   calibration to a position-indexed prior; the split replicates on a second
   held-out month. Pooled-histogram Hellinger distance — the literature's headline
   metric — is insensitive to this collapse.
2. **Causally used context is long-range but physically unaligned.** Median 4 grid
   cells (~1200 km at T42), isotropic, and uncorrelated with ray-traced
   propagation directions.
3. **Column models fail single-aspect physics grafts** (no filtering response at
   typical columns, drag insensitive to wind rotation, non-monotone amplitude
   response) under a protocol validated on a 1D testbed emulator that demonstrably
   learned its physics.

The full hypothesis funnel: 30 generated, 24 screened (15 killed), 1 measured,
5 deferred with reasons, 1 killed at held-out confirmation. See
[`HYPOTHESIS_TABLE.md`](HYPOTHESIS_TABLE.md) and the append-only
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) for the true story, including failures.
A non-specialist summary is in [`PLAIN_SUMMARY.md`](PLAIN_SUMMARY.md).

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Numbered, self-contained experiment scripts (01–25, chronological) |
| `src/` | Shared library: data pipeline, model loading, interp tools, physics baselines |
| `configs/` | YAML configs; every results file is stamped with its config hash |
| `results/` | Git-tracked metrics (JSON) and figures; large arrays are gitignored and regenerated |
| `paper/` | LaTeX source, programmatic tables, compiled `main.pdf` |
| `tests/` | Physics baselines, data-pipeline checks, probe controls (`make test`) |
| `references/` | The audited paper's open-access (CC-BY) preprint |
| `RESEARCH_LOG.md` | Append-only program log — decisions, pre-registrations, kills |
| `HYPOTHESIS_TABLE.md` | Live status of all 30 hypotheses with kill/confirm criteria |
| `ASSETS.md` | Data and checkpoint provenance (what to download, from where) |
| `CHECKPOINT_C.md` | Mid-program checkpoint review |

## Reproduction

Environment: CPU-only, Python 3.11, PyTorch CPU; pinned deps in
`requirements.txt`. Bulk data and released checkpoints are **not** in this repo —
see [`ASSETS.md`](ASSETS.md) for provenance and `configs/paths.yaml` for where
they are expected on disk.

```
python -m venv <venv-path>
<venv-path>/Scripts/pip install -r requirements.txt

make test               # physics baselines + pipeline checks
make reproduce-figures  # regenerate paper figures from stamped metrics (fast)
make reproduce-results  # full experiment regeneration (hours; requires data)
make paper              # rebuild tables + compile paper/main.pdf
```

Every metrics file under `results/` carries the config hash and git state it was
produced from, so figures and tables are traceable end-to-end.
