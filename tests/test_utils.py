"""Harness gate: determinism, config hashing, and results stamping must behave
before any experiment output is trusted."""

import json

import numpy as np
import torch

from src import utils


def test_config_hash_order_independent():
    a = {"x": 1, "nested": {"b": 2, "a": [1, 2]}}
    b = {"nested": {"a": [1, 2], "b": 2}, "x": 1}
    assert utils.config_hash(a) == utils.config_hash(b)
    assert utils.config_hash(a) != utils.config_hash({"x": 2})


def test_set_seed_reproducible_draws():
    utils.set_seed(123)
    n1, t1 = np.random.rand(5), torch.rand(5)
    utils.set_seed(123)
    n2, t2 = np.random.rand(5), torch.rand(5)
    assert np.allclose(n1, n2)
    assert torch.equal(t1, t2)


def test_set_seed_different_seeds_differ():
    utils.set_seed(1)
    a = torch.rand(5)
    utils.set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_save_results_roundtrip(tmp_path):
    cfg = {"lr": 1e-3, "model": "m1"}
    out = tmp_path / "sub" / "metrics.json"
    utils.save_results(out, {"rmse": 0.5}, cfg, seed=7)
    loaded = json.loads(out.read_text())
    assert loaded["rmse"] == 0.5
    assert loaded["config"] == {"lr": 0.001, "model": "m1"}
    st = loaded["stamp"]
    assert st["config_hash"] == utils.config_hash(cfg)
    assert st["seed"] == 7
    assert st["torch"] and st["numpy"] and st["python"]
