"""Load the anchor paper's released checkpoints into the vendored architecture.

Checkpoint provenance: https://huggingface.co/amangupta2/nonlocal_gwfluxes
(MIT), trained with the paper-era code (github.com/amangupta2/nonlocal_gwfluxes)
whose ANN_CNN class DEFINES bnorm1..6 BatchNorm1d modules in __init__ but never
applies them in forward() (the calls are commented out; empirically all
bnorm*.num_batches_tracked == 0 in the checkpoints, and bnorm4 was constructed
with the wrong width 2*hdim, which would crash if ever applied). The cleaned-up
release code (DataWaveProject/nonlocal_gwfluxes, tag 1.0.0, vendored under
src/models/anchor/) dropped these dead modules; its forward is functionally
identical. load_ann() therefore strips bnorm* keys after asserting they are
untrained, and loads everything else strictly.

Channel conventions (global vertical, from the released dataloader):
  inputs:  [lat, lon, zs] + u(122) + v(122) + theta(122) [+ w(122)]
           M1/M2 uvtheta -> idim 369, uvthetaw -> 491; the UNet (M3) omits the
           3 scalars -> 366 / 488 (see feature_slice)
  outputs: uw(122) + vw(122) -> odim 244
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.anchor.model_definition import ANN_CNN, Attention_UNet

WEIGHTS_ROOT = Path(r"C:\Users\sidi0\gwmi_data\weights\nonlocal_gwfluxes")

CHECKPOINTS = {
    # (model, features, vertical) -> relative path (released epoch choices)
    ("m1", "uvtheta", "global"): "ANN_1x1/ann_cnn_1x1_global_global_era5_uvtheta__train_epoch94.pt",
    ("m1", "uvthetaw", "global"): "ANN_1x1/ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch94.pt",
    ("m2", "uvtheta", "global"): "ANN_3x3/ann_cnn_3x3_global_global_era5_uvtheta__train_epoch52.pt",
    ("m2", "uvthetaw", "global"): "ANN_3x3/ann_cnn_3x3_global_global_era5_uvthetaw__train_epoch80.pt",
    ("m3", "uvtheta", "global"): "AttentionUNet/attnunet_era5_global_global_uvtheta_mseloss_train_epoch100.pt",
    ("m3", "uvthetaw", "global"): "AttentionUNet/attnunet_era5_global_global_uvthetaw_mseloss_train_epoch119.pt",
    ("m1", "uvtheta", "stratosphere_only"): "ANN_1x1/ann_cnn_1x1_global_stratosphere_only_era5_uvtheta__train_epoch100.pt",
    ("m2", "uvtheta", "stratosphere_only"): "ANN_3x3/ann_cnn_3x3_global_stratosphere_only_era5_uvtheta__train_epoch93.pt",
    ("m3", "uvtheta", "stratosphere_only"): "AttentionUNet/attnunet_era5_global_stratosphere_only_uvtheta_mseloss_train_epoch131.pt",
    ("m3", "uvthetaw", "stratosphere_only"): "AttentionUNet/attnunet_era5_global_stratosphere_only_uvthetaw_mseloss_train_epoch138.pt",
}

def feature_slice(model: str, features: str, vertical: str = "global") -> slice:
    """Channel range into the stored feature vector, per released dataloaders.
    M1/M2 include the 3 scalars [lat, lon, zs]; the Attention UNet omits them
    ("omitting lat, lon, zs for attention unet" -- dataloader_definition.py)."""
    ends = {("uvtheta", "global"): 369, ("uvthetaw", "global"): 491,
            ("uvtheta", "stratosphere_only"): 183, ("uvthetaw", "stratosphere_only"): 243}
    start = 3 if model == "m3" else 0
    return slice(start, ends[(features, vertical)])


def _load_state(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_state_dict"] if "model_state_dict" in ck else ck


def load_ann(path: str | Path, stencil: int) -> ANN_CNN:
    """M1 (stencil=1) / M2 (stencil=3): dims inferred from the checkpoint."""
    sd = _load_state(Path(path))
    bnorm = {k: v for k, v in sd.items() if k.startswith("bnorm")}
    for k, v in bnorm.items():
        if k.endswith("num_batches_tracked") and int(v) != 0:
            raise ValueError(f"{path}: {k}={int(v)} — batchnorm WAS trained; "
                             "vendored (bnorm-free) forward would be wrong")
    sd = {k: v for k, v in sd.items() if not k.startswith("bnorm")}
    hdim, idim = sd["layer1.weight"].shape
    odim = sd["output.weight"].shape[0]
    if hdim != 4 * idim:
        raise ValueError(f"{path}: hdim {hdim} != 4*idim {idim}")
    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=0.0, stencil=stencil)
    missing = set(model.state_dict()) - set(sd)
    extra = set(sd) - set(model.state_dict())
    if missing or extra:
        raise ValueError(f"{path}: key mismatch missing={missing} extra={extra}")
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_unet(path: str | Path) -> Attention_UNet:
    sd = _load_state(Path(path))
    ch_in = sd["conv1.conv.0.weight"].shape[1]
    ch_out = sd["conv1x1.weight"].shape[0]
    model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=0.0)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_model(model: str, features: str, vertical: str = "global",
               root: Path | None = None):
    """Load a released checkpoint by (m1|m2|m3, uvtheta|uvthetaw, vertical)."""
    rel = CHECKPOINTS[(model, features, vertical)]
    path = (root or WEIGHTS_ROOT) / rel
    if model == "m1":
        return load_ann(path, stencil=1)
    if model == "m2":
        return load_ann(path, stencil=3)
    if model == "m3":
        return load_unet(path)
    raise ValueError(model)
