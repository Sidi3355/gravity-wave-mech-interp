# Mechanistic Interpretability of Deep Learning Gravity Wave Parameterizations

Autonomous research program building on Gupta, Sheshadri, Roy & Anantharaj (2025),
*Offline Performance of a Nonlocal Deep Learning Parameterization for Climate Model
Representation of Atmospheric Gravity Waves*, JAMES, doi:10.1029/2025MS004977.

Goal: a causal, mechanistic account of why horizontal nonlocality improves neural
gravity-wave flux prediction and what these networks internally represent.

## Layout
- `experiments/` — numbered, self-contained experiment scripts
- `src/` — shared library (data, models, interp tools, physics baselines)
- `configs/` — YAML configs; every results file is stamped with its config hash
- `results/` — git-tracked metrics and figures (large arrays gitignored)
- `tests/` — physics baselines, data pipeline, probe controls
- `RESEARCH_LOG.md` — append-only program log (the true story, including failures)
- `ASSETS.md`, `HYPOTHESIS_TABLE.md`, `CHECKPOINT_*.md`, `PAPER_DRAFT.md`

## Environment
CPU-only (no GPU). Python 3.11, PyTorch 2.13 CPU. Pinned deps in `requirements.txt`.
Venv lives outside OneDrive at `C:\Users\sidi0\venvs\gwmi`; bulk data at
`C:\Users\sidi0\gwmi_data\` (see `configs/paths.yaml`).

Setup:
```
python -m venv C:\Users\sidi0\venvs\gwmi
C:\Users\sidi0\venvs\gwmi\Scripts\pip install -r requirements.txt
```
