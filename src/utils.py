"""Shared utilities: determinism, config loading, config hashing, results stamping.

Every experiment loads a YAML config, seeds everything through `set_seed`, and
stamps outputs with `config_hash` + library versions via `results_stamp`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Global determinism: python, numpy, torch (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config dict (order-independent)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def results_stamp(cfg: dict, seed: int | None = None) -> dict:
    """Provenance block embedded in every results file."""
    return {
        "config_hash": config_hash(cfg),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def save_results(path: str | Path, payload: dict, cfg: dict, seed: int | None = None) -> None:
    """Write a JSON results file with provenance stamp. Small metrics only —
    large arrays go to results/**/arrays/ (gitignored) as .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"stamp": results_stamp(cfg, seed), "config": cfg, **payload}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)


def data_root() -> Path:
    """Bulk-data directory (outside OneDrive); override via GWMI_DATA env var."""
    root = Path(os.environ.get("GWMI_DATA", r"C:\Users\sidi0\gwmi_data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
