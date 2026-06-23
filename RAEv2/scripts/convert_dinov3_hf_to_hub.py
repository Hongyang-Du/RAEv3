#!/usr/bin/env python3
"""Convert a HuggingFace DINOv3 ViT checkpoint (model.safetensors, transformers
naming) into the original facebookresearch/dinov3 torch.hub state_dict (.pth) that
this repo's encoder loader (encoders/models/dinov3_loader.py) expects.

Usage:
    python scripts/convert_dinov3_hf_to_hub.py \
        --hf-safetensors /path/to/model.safetensors \
        --out pretrained_models/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

The conversion is verified by building the actual hub model (pretrained=False) and
calling load_state_dict(strict=True): any missing/unexpected/shape-mismatched key
aborts the script, so a successful run guarantees the .pth is directly loadable.
"""
import argparse
import os
import re

import torch
from safetensors.torch import load_file

HUB_REF_DIR = os.environ.get(
    "DINOV3_HUB_DIR",
    "/mnt/localssd/.cache/torch/hub/facebookresearch_dinov3_94a96ac83c2446f15f9bdcfae23cad3c6a9d4988",
)


def convert(hf_sd: dict) -> dict:
    """Map transformers DINOv3ViT keys -> facebookresearch/dinov3 hub keys."""
    out = {}
    # token / embedding level
    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    # mask_token in hub is (1, C); HF stores (1,1,C) -> squeeze handled below.

    # gather per-layer q/k/v to fuse into qkv
    layer_qkv = {}  # idx -> {q.w,q.b,k.w,v.w,v.b}
    n_layers = 0

    for k, v in hf_sd.items():
        if k in direct:
            t = v
            if k == "embeddings.mask_token" and t.dim() == 3:
                t = t.reshape(1, -1)
            out[direct[k]] = t
            continue
        m = re.match(r"layer\.(\d+)\.(.+)", k)
        if not m:
            # rope periods etc. — try a couple of known extras
            if k.endswith("rope_embeddings.periods") or k.endswith("rope_embed.periods"):
                out["rope_embed.periods"] = v
            else:
                raise KeyError(f"Unmapped top-level key: {k}  shape={tuple(v.shape)}")
            continue
        idx = int(m.group(1)); rest = m.group(2)
        n_layers = max(n_layers, idx + 1)
        p = f"blocks.{idx}"
        if rest == "norm1.weight": out[f"{p}.norm1.weight"] = v
        elif rest == "norm1.bias": out[f"{p}.norm1.bias"] = v
        elif rest == "norm2.weight": out[f"{p}.norm2.weight"] = v
        elif rest == "norm2.bias": out[f"{p}.norm2.bias"] = v
        elif rest == "layer_scale1.lambda1": out[f"{p}.ls1.gamma"] = v
        elif rest == "layer_scale2.lambda1": out[f"{p}.ls2.gamma"] = v
        elif rest == "mlp.up_proj.weight": out[f"{p}.mlp.fc1.weight"] = v
        elif rest == "mlp.up_proj.bias": out[f"{p}.mlp.fc1.bias"] = v
        elif rest == "mlp.down_proj.weight": out[f"{p}.mlp.fc2.weight"] = v
        elif rest == "mlp.down_proj.bias": out[f"{p}.mlp.fc2.bias"] = v
        elif rest == "attention.o_proj.weight": out[f"{p}.attn.proj.weight"] = v
        elif rest == "attention.o_proj.bias": out[f"{p}.attn.proj.bias"] = v
        elif rest.startswith("attention."):
            d = layer_qkv.setdefault(idx, {})
            d[rest[len("attention."):]] = v
        else:
            raise KeyError(f"Unmapped layer key: {k}  shape={tuple(v.shape)}")

    # fuse q/k/v -> qkv.weight / qkv.bias  (+ bias_mask: 1 where bias is real, 0 for k)
    for idx, d in layer_qkv.items():
        p = f"blocks.{idx}"
        qw, kw, vw = d["q_proj.weight"], d["k_proj.weight"], d["v_proj.weight"]
        out[f"{p}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        C = qw.shape[0]
        qb = d.get("q_proj.bias", torch.zeros(C, dtype=qw.dtype))
        kb = d.get("k_proj.bias", torch.zeros(C, dtype=qw.dtype))  # HF DINOv3: k has no bias
        vb = d.get("v_proj.bias", torch.zeros(C, dtype=qw.dtype))
        out[f"{p}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        # bias_mask: hub applies bias only to q and v (k-bias forced 0). 1=keep, 0=mask.
        mask = torch.cat([torch.ones(C), torch.zeros(C), torch.ones(C)]).to(qw.dtype)
        out[f"{p}.attn.qkv.bias_mask"] = mask
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-safetensors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-dir", default=HUB_REF_DIR)
    ap.add_argument("--model", default="dinov3_vitl16")
    args = ap.parse_args()

    print(f"Loading HF safetensors: {args.hf_safetensors}")
    hf_sd = load_file(args.hf_safetensors)
    print(f"  {len(hf_sd)} HF tensors")

    converted = convert(hf_sd)
    print(f"Converted -> {len(converted)} hub tensors")

    # Build the real hub model and verify a strict load.
    print(f"Building hub model {args.model} (pretrained=False) for strict verification...")
    model = torch.hub.load(args.hub_dir, args.model, source="local",
                           trust_repo=True, skip_validation=True, pretrained=False)
    ref = model.state_dict()

    # rope_embed.periods is a non-persistent buffer in some builds; if the model
    # has it but conversion lacks it, copy the model's own (it's deterministic).
    for k in ref:
        if k not in converted and ("periods" in k):
            converted[k] = ref[k]
            print(f"  filled deterministic buffer from model: {k}")

    missing = [k for k in ref if k not in converted]
    unexpected = [k for k in converted if k not in ref]
    if missing:
        raise SystemExit(f"MISSING {len(missing)} keys, e.g. {missing[:8]}")
    if unexpected:
        raise SystemExit(f"UNEXPECTED {len(unexpected)} keys, e.g. {unexpected[:8]}")
    for k in ref:
        if tuple(ref[k].shape) != tuple(converted[k].shape):
            raise SystemExit(f"SHAPE MISMATCH {k}: model {tuple(ref[k].shape)} vs conv {tuple(converted[k].shape)}")

    model.load_state_dict(converted, strict=True)
    print("strict load_state_dict OK.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(converted, args.out)
    print(f"Saved hub-format checkpoint -> {args.out}  ({os.path.getsize(args.out)//1024//1024} MB)")


if __name__ == "__main__":
    main()
