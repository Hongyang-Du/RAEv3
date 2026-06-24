import sys; sys.path.insert(0, "src")
import torch
from omegaconf import OmegaConf
from configs.stage2 import Stage2Config
from utils.model_utils import instantiate_from_config

def strip(sd):
    out={}
    for k,v in sd.items():
        out[k[10:] if k.startswith("_orig_mod.") else k]=v
    return out

cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config),
        OmegaConf.load("configs/stage2/training/imagenet-dinov3l-k7.yaml")))
cfg.post_process(); cfg.prepare_model_params()
model = instantiate_from_config(cfg.stage_2).eval()
ck = torch.load("pretrained_models/stage2/imagenet/dinov3l-k7/checkpoint.pt", map_location="cpu", weights_only=False)
sd = strip(ck["ema"])
r = model.load_state_dict(sd, strict=False)
print("missing:", len(r.missing_keys), r.missing_keys[:5])
print("unexpected:", len(r.unexpected_keys), r.unexpected_keys[:5])
np = sum(p.numel() for p in model.parameters())
print("model params:", np, " ckpt-ema tensors:", len(sd))
print("epoch:", ck.get("epoch"), "step:", ck.get("step"))
