# Reproduction targets. PY points at the pinned venv (see README).
PY ?= C:/Users/sidi0/venvs/gwmi/Scripts/python.exe

reproduce-figures:
	$(PY) experiments/24_figures_core.py

test:
	$(PY) -m pytest tests/ -q

# Full result regeneration (hours; requires data per ASSETS.md):
reproduce-results:
	$(PY) experiments/01_replication_a4.py
	$(PY) experiments/02_a4_fullmonth.py
	$(PY) experiments/03_screen_n1_gates.py
	$(PY) experiments/04_screen_i2_clamp.py
	$(PY) experiments/05_screen_a1_ridge.py
	$(PY) experiments/06_screen_p4_seam.py
	$(PY) experiments/07_screen_n3_gateablate.py
	$(PY) experiments/08_c5_jacobian.py
	$(PY) experiments/09_screen_a5_tails.py
	$(PY) experiments/10_screen_a2_regimes.py
	$(PY) experiments/11_wave2b_screens.py
	$(PY) experiments/12_screen_n5_i4.py
	$(PY) experiments/13_probes_m1.py
	$(PY) experiments/14_probes_m3.py
	$(PY) experiments/15_p_grafts.py
	$(PY) experiments/16_screen_i3.py
	$(PY) experiments/17_screen_r4_sae.py
	$(PY) experiments/18_gateweek_g2.py
	$(PY) experiments/19_gateweek_g1.py
	$(PY) experiments/20_d1_rollensemble.py
	$(PY) experiments/22_d1_rollpatch.py
	$(PY) experiments/21_d2_battery.py
