"""Precompute h1 statistics for RAEDecoderH1.

  - mean / var : per-channel stats of the decoder block-0 PATCH tokens, used to
                 normalize the DiT diffusion target. shape [1, Cdec, 1, 1].
  - mean_cls   : dataset-mean of the block-0 CLS token, substituted at decode time for
                 the image-dependent CLS the spatial DiT cannot generate. shape [1, 1, Cdec].

  torchrun --nproc_per_node=1 src/compute_h1_stats.py \
      --config configs/stage2/training/imagenet-dinov3l-h1decoder-k23.yaml \
      --num-samples 50000 --out output_full/h1_decoder_k23_stats.pt
"""
import argparse
import dataclasses

import torch
from omegaconf import OmegaConf

from configs.stage2 import Stage2Config
from data import prepare_unified_dataloader
from utils.dist_utils import setup_distributed
from utils.model_utils import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("opts", nargs="*", default=[],
                    help="optional dotlist config overrides, e.g. stage_1.params.block_idx=7")
    args = ap.parse_args()

    rank, world_size, device = setup_distributed()
    _merged = OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    if args.opts:
        _merged = OmegaConf.merge(_merged, OmegaConf.from_dotlist(list(args.opts)))
    config: Stage2Config = OmegaConf.to_object(_merged)
    config.post_process()

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    cdec = rae.decoder.decoder_pred.in_features

    loader = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=args.batch,
        num_workers=config.training.num_workers,
        rank=rank, world_size=world_size,
        transform=None, condition_type=config.conditioning.type,
        virtual_epoch_steps=None,
    )

    # running sums (float64 for stability)
    n_patch = 0
    s = torch.zeros(cdec, dtype=torch.float64, device=device)     # sum of patch tokens
    ss = torch.zeros(cdec, dtype=torch.float64, device=device)    # sum of squares
    n_img = 0
    s_cls = torch.zeros(cdec, dtype=torch.float64, device=device)  # sum of CLS tokens

    seen = 0
    for images, _ in loader:
        images = images.to(device)
        h1 = rae._block0(images)                  # [B, N+1, Cdec]
        cls = h1[:, 0, :].double()                # [B, Cdec]
        patch = h1[:, 1:, :].double()             # [B, N, Cdec]
        bN = patch.shape[0] * patch.shape[1]
        s += patch.sum(dim=(0, 1)); ss += (patch * patch).sum(dim=(0, 1)); n_patch += bN
        s_cls += cls.sum(dim=0); n_img += cls.shape[0]
        seen += images.shape[0]
        if rank == 0 and seen % (args.batch * 20) == 0:
            print(f"  {seen}/{args.num_samples}", flush=True)
        if seen >= args.num_samples:
            break

    mean = (s / n_patch)
    var = (ss / n_patch) - mean * mean
    mean_cls = (s_cls / n_img)

    if rank == 0:
        out = {
            "mean": mean.float().view(1, cdec, 1, 1).cpu(),
            "var": var.clamp_min(0).float().view(1, cdec, 1, 1).cpu(),
            "mean_cls": mean_cls.float().view(1, 1, cdec).cpu(),
            "num_samples": seen,
        }
        torch.save(out, args.out)
        print(f"saved -> {args.out}  | mean {mean.mean():.4f} var {var.mean():.4f} "
              f"cls |.| {mean_cls.norm():.3f}  (Cdec={cdec}, N_patch={n_patch})")


if __name__ == "__main__":
    main()
