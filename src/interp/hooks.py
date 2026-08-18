"""Activation capture and intervention hooks for the anchor architectures.

Capture sites (module names as in the vendored classes):
  ANN_CNN (M1/M2): act_cnn (M2 post-conv ReLU), act1..act6 (post-LeakyReLU
    of each hidden layer), output (final linear).
  Attention_UNet (M3): conv1..conv5 (encoder blocks), up5..up2 (decoder
    upsamples), attn5..attn2 (gated skip outputs x*alpha), attn5.Psi..attn2.Psi
    (the sigmoid gate fields alpha themselves), upconv5..upconv2, conv1x1.

Two modes:
  ActivationCapture — record module outputs during forward (detached).
  ActivationPatch   — replace/transform a module's output during forward
                      (the editing primitive for T3 causal interventions).
Both are context managers that always deregister their hooks.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

ANN_SITES = ["act_cnn", "act1", "act2", "act3", "act4", "act5", "act6", "output"]
UNET_ENCODER_SITES = ["conv1", "conv2", "conv3", "conv4", "conv5"]
UNET_GATE_SITES = ["attn5.Psi", "attn4.Psi", "attn3.Psi", "attn2.Psi"]
UNET_DECODER_SITES = ["upconv5", "upconv4", "upconv3", "upconv2", "conv1x1"]


def _module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


class ActivationCapture:
    """Record outputs of named submodules during forward passes.

    with ActivationCapture(model, ["act3", "act5"]) as cap:
        model(x)
    cap.acts["act3"]  # tensor (detached, CPU)
    """

    def __init__(self, model: torch.nn.Module, sites: list[str]):
        self.model = model
        self.sites = list(sites)
        self.acts: dict[str, torch.Tensor] = {}
        self._handles = []

    def _hook(self, site):
        def fn(module, inputs, out):
            self.acts[site] = out.detach()
        return fn

    def __enter__(self):
        missing = []
        for s in self.sites:
            try:
                m = _module_by_name(self.model, s)
            except AttributeError:
                missing.append(s)
                continue
            self._handles.append(m.register_forward_hook(self._hook(s)))
        if missing:
            self.__exit__(None, None, None)
            raise AttributeError(f"capture sites not found: {missing}")
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


@contextmanager
def ActivationPatch(model: torch.nn.Module, site: str, transform):
    """Replace the output of `site` with transform(output) during forward.

    transform: callable tensor -> tensor of the SAME shape (checked).
    Example zero-ablation:  ActivationPatch(m, "act3", torch.zeros_like)
    """
    mod = _module_by_name(model, site)

    def fn(module, inputs, out):
        new = transform(out)
        if new.shape != out.shape:
            raise ValueError(
                f"patch at {site} changed shape {tuple(out.shape)} -> {tuple(new.shape)}")
        return new

    handle = mod.register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def capture_ann_columns(model, x: torch.Tensor, sites=None) -> dict[str, torch.Tensor]:
    """One-shot convenience: run M1/M2 on a batch, return {site: activations}."""
    sites = [s for s in (sites or ANN_SITES) if s != "act_cnn" or model.fac >= 1]
    with ActivationCapture(model, sites) as cap, torch.no_grad():
        model(x)
        return dict(cap.acts)


def capture_unet_maps(model, x: torch.Tensor, sites=None) -> dict[str, torch.Tensor]:
    """One-shot convenience: run M3 on (B, C, 64, 128), return {site: maps}."""
    sites = sites or (UNET_ENCODER_SITES + UNET_GATE_SITES + UNET_DECODER_SITES)
    with ActivationCapture(model, sites) as cap, torch.no_grad():
        model(x)
        return dict(cap.acts)
