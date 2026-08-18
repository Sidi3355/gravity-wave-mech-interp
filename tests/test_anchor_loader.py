"""Gate: released checkpoints must load into the vendored architectures with
exact key/shape agreement and produce correctly-shaped finite outputs."""

import pytest
import torch

from src.models.anchor_loader import WEIGHTS_ROOT, load_model

pytestmark = pytest.mark.skipif(
    not WEIGHTS_ROOT.exists(), reason="released weights not downloaded"
)


@pytest.mark.parametrize("features,idim", [("uvtheta", 369), ("uvthetaw", 491)])
def test_m1_loads_and_runs(features, idim):
    m = load_model("m1", features)
    x = torch.randn(8, idim)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (8, 244) and torch.isfinite(y).all()


@pytest.mark.parametrize("features,idim", [("uvtheta", 369), ("uvthetaw", 491)])
def test_m2_loads_and_runs(features, idim):
    m = load_model("m2", features)
    x = torch.randn(8, idim, 3, 3)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (8, 244) and torch.isfinite(y).all()


@pytest.mark.parametrize("features,idim", [("uvtheta", 366), ("uvthetaw", 488)])
def test_m3_loads_and_runs(features, idim):
    # UNet omits the 3 scalar channels (lat, lon, zs) per released dataloader
    m = load_model("m3", features)
    x = torch.randn(1, idim, 64, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 244, 64, 128) and torch.isfinite(y).all()


def test_deterministic_eval():
    m = load_model("m1", "uvtheta")
    x = torch.randn(4, 369)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))  # dropout must be inert in eval
