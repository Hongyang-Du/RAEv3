"""
Convert the HuggingFace transformers DINOv3 checkpoint (model.safetensors) to the
original facebookresearch/dinov3 torch.hub backbone format expected by
src/encoders/models/dinov3_loader.py (which loads a local .pth via torch.hub).

The gated HF repo `facebook/dinov3-vitl16-pretrain-lvd1689m` only ships the
transformers-format safetensors; the torch.hub loader needs the original backbone
state_dict (strict=True). This script bridges the two and verifies the result
numerically against the HF model.

Run inside the container (needs HF_TOKEN with gated access):
    HF_TOKEN=hf_xxx python scripts/data/convert_dinov3_hf_to_orig.py
"""
import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

DEFAULT_HUB_DIR = (
    "/root/.cache/torch/hub/"
    "facebookresearch_dinov3_94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
)
DEFAULT_HF_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_OUT = (
    "pretrained_models/encoders/dinov3/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)


def build_orig_model(hub_dir: str):
    return torch.hub.load(
        hub_dir, "dinov3_vitl16", source="local",
        pretrained=False, trust_repo=True, skip_validation=True,
    )


def convert(hsd: dict, ref: dict) -> dict:
    """hsd = HF safetensors state_dict, ref = original-model state_dict (for buffers)."""
    out = {k: v.clone() for k, v in ref.items()}  # keep buffers (rope periods, qkv.bias_mask)

    out["cls_token"] = hsd["embeddings.cls_token"]
    out["mask_token"] = hsd["embeddings.mask_token"].reshape(ref["mask_token"].shape)
    out["storage_tokens"] = hsd["embeddings.register_tokens"]
    out["patch_embed.proj.weight"] = hsd["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hsd["embeddings.patch_embeddings.bias"]
    out["norm.weight"] = hsd["norm.weight"]
    out["norm.bias"] = hsd["norm.bias"]

    n_layers = 1 + max(int(k.split(".")[1]) for k in hsd if k.startswith("layer."))
    for i in range(n_layers):
        p, b = f"layer.{i}.", f"blocks.{i}."
        q = hsd[p + "attention.q_proj.weight"]
        k = hsd[p + "attention.k_proj.weight"]
        v = hsd[p + "attention.v_proj.weight"]
        out[b + "attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = hsd[p + "attention.q_proj.bias"]
        vb = hsd[p + "attention.v_proj.bias"]
        kb = torch.zeros_like(qb)  # HF has no k bias; original masks it to zero
        out[b + "attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        # bias_mask is a buffer init'd to NaN in the arch; it must come from the
        # checkpoint. It zeros the K-bias via masked_bias = bias * bias_mask:
        # ones for q and v, zeros for the middle (k) third.
        dim = qb.shape[0]
        mask = torch.ones(3 * dim, dtype=out[b + "attn.qkv.bias"].dtype)
        mask[dim:2 * dim] = 0.0
        out[b + "attn.qkv.bias_mask"] = mask
        out[b + "attn.proj.weight"] = hsd[p + "attention.o_proj.weight"]
        out[b + "attn.proj.bias"] = hsd[p + "attention.o_proj.bias"]
        out[b + "ls1.gamma"] = hsd[p + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = hsd[p + "layer_scale2.lambda1"]
        out[b + "mlp.fc1.weight"] = hsd[p + "mlp.up_proj.weight"]
        out[b + "mlp.fc1.bias"] = hsd[p + "mlp.up_proj.bias"]
        out[b + "mlp.fc2.weight"] = hsd[p + "mlp.down_proj.weight"]
        out[b + "mlp.fc2.bias"] = hsd[p + "mlp.down_proj.bias"]
        out[b + "norm1.weight"] = hsd[p + "norm1.weight"]
        out[b + "norm1.bias"] = hsd[p + "norm1.bias"]
        out[b + "norm2.weight"] = hsd[p + "norm2.weight"]
        out[b + "norm2.bias"] = hsd[p + "norm2.bias"]
    return out


@torch.no_grad()
def verify(orig_model, hf_repo: str, hub_dir: str):
    """Numerically compare converted original backbone vs HF transformers model."""
    from transformers import AutoModel

    hf = AutoModel.from_pretrained(hf_repo).eval()
    x = torch.randn(1, 3, 224, 224)

    feats = orig_model.get_intermediate_layers(
        x, n=[23], reshape=False, return_class_token=False, norm=True
    )[0]  # (1, 196, 1024) patch tokens, final-normed

    hf_out = hf(pixel_values=x).last_hidden_state  # (1, 1+4+196, 1024)
    n_patch = feats.shape[1]
    hf_patch = hf_out[:, -n_patch:, :]

    cos = torch.nn.functional.cosine_similarity(feats, hf_patch, dim=-1)  # (1,196)
    max_abs = (feats - hf_patch).abs().max().item()
    return cos.mean().item(), cos.min().item(), max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-dir", default=DEFAULT_HUB_DIR)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    print(">> loading HF safetensors ...")
    hsd = load_file(hf_hub_download(args.hf_repo, "model.safetensors"))

    print(">> building original backbone (random init) for reference keys/buffers ...")
    model = build_orig_model(args.hub_dir)
    ref = model.state_dict()

    print(">> converting ...")
    converted = convert(hsd, ref)

    print(">> strict load check ...")
    model.load_state_dict(converted, strict=True)  # raises on any mismatch
    print("   strict=True load OK")

    if not args.no_verify:
        print(">> numerical cross-check vs HF transformers model ...")
        mean_cos, min_cos, max_abs = verify(model, args.hf_repo, args.hub_dir)
        print(f"   patch-token cosine: mean={mean_cos:.5f} min={min_cos:.5f} | max_abs_diff={max_abs:.4f}")
        # `not (>= 0.99)` also catches NaN (NaN comparisons are always False)
        if not (mean_cos >= 0.99):
            raise SystemExit(f"VERIFY FAILED: mean cosine {mean_cos} not >= 0.99 — conversion likely wrong")
        print("   verify PASS")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(converted, args.out)
    print(f">> saved: {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
