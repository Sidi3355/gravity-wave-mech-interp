"""Gate: capture/patch hooks must be shape-correct, leak-free, and causally
effective before any interp experiment uses them."""

import pytest
import torch

from src.interp import hooks
from src.models.anchor.model_definition import ANN_CNN, Attention_UNet


@pytest.fixture(scope="module")
def m1():
    torch.manual_seed(0)
    return ANN_CNN(idim=20, odim=10, hdim=80, dropout=0.0, stencil=1).eval()


@pytest.fixture(scope="module")
def m3():
    torch.manual_seed(0)
    return Attention_UNet(ch_in=8, ch_out=6, dropout=0.0).eval()


def test_ann_capture_shapes(m1):
    x = torch.randn(16, 20)
    acts = hooks.capture_ann_columns(m1, x)
    assert acts["act1"].shape == (16, 80)
    assert acts["act6"].shape == (16, 20)   # 2*odim
    assert acts["output"].shape == (16, 10)
    assert "act_cnn" not in acts            # stencil=1 has no conv


def test_unet_capture_shapes(m3):
    x = torch.randn(2, 8, 64, 128)
    acts = hooks.capture_unet_maps(m3, x)
    assert acts["conv1"].shape == (2, 64, 64, 128)
    assert acts["conv5"].shape == (2, 1024, 4, 8)      # bottleneck
    assert acts["attn2.Psi"].shape == (2, 1, 64, 128)  # finest gate field
    assert acts["conv1x1"].shape == (2, 6, 64, 128)
    assert acts["attn2.Psi"].min() >= 0 and acts["attn2.Psi"].max() <= 1


def test_hooks_deregister(m1):
    x = torch.randn(4, 20)
    with hooks.ActivationCapture(m1, ["act1"]) as cap, torch.no_grad():
        m1(x)
    n_before = len(cap.acts)
    with torch.no_grad():
        m1(x)  # after exit: no new captures
    assert len(cap.acts) == n_before
    assert all(len(m._forward_hooks) == 0 for m in m1.modules())


def test_unknown_site_raises_and_cleans(m1):
    with pytest.raises(AttributeError):
        with hooks.ActivationCapture(m1, ["act1", "nope"]):
            pass
    assert all(len(m._forward_hooks) == 0 for m in m1.modules())


def test_patch_identity_is_noop(m1):
    x = torch.randn(8, 20)
    with torch.no_grad():
        base = m1(x)
        with hooks.ActivationPatch(m1, "act3", lambda t: t):
            same = m1(x)
    assert torch.equal(base, same)


def test_patch_zero_ablation_changes_output(m1):
    x = torch.randn(8, 20)
    with torch.no_grad():
        base = m1(x)
        with hooks.ActivationPatch(m1, "act3", torch.zeros_like):
            ablated = m1(x)
        after = m1(x)
    assert not torch.equal(base, ablated)   # intervention is causally live
    assert torch.equal(base, after)         # and fully removed afterwards


def test_patch_shape_guard(m1):
    x = torch.randn(8, 20)
    with pytest.raises(ValueError):
        with hooks.ActivationPatch(m1, "act3", lambda t: t[:, :5]):
            with torch.no_grad():
                m1(x)


def test_gate_patch_on_unet(m3):
    """Flattening the finest gate to its spatial mean must change predictions
    (the primitive needed by H-N3)."""
    x = torch.randn(1, 8, 64, 128)

    def flatten_gate(alpha):
        return torch.ones_like(alpha) * alpha.mean(dim=(-2, -1), keepdim=True)

    with torch.no_grad():
        base = m3(x)
        with hooks.ActivationPatch(m3, "attn2.Psi", flatten_gate):
            patched = m3(x)
    assert not torch.equal(base, patched)
