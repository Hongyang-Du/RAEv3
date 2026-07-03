"""Precompute PER-POSITION h1 statistics for RAEDecoderH1 (like the official encoder stats).

  - mean / var : per-(channel, spatial-position) stats of decoder block-0 PATCH tokens.
                 shape [Cdec, 16, 16]  (vs the old per-channel [1, Cdec, 1, 1] which pooled
                 the 16x16 positions -> under-normalizes the positionally-structured latent).
  - mean_cls   : dataset-mean of the block-0 CLS token. shape [1, 1, Cdec] (unchanged).

  Multi-GPU safe: sums are all_reduced across ranks, so `--nproc_per_node=8` gives full-data
  stats and stays symmetric (no single-rank stall -> safe to run alongside another DDP job).

  torchrun --nproc_per_node=8 src/compute_h1_stats_perpos.py \
      --config <h1 training yaml> --num-samples 50000 --out <...>_perpos.pt
"""
import argparse
import dataclasses

import torch
import torch.distributed as dist
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
    args = ap.parse_args()

    rank, world_size, device = setup_distributed()
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
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

    N = None
    s = ss = None                                             # per-position sums [N, Cdec]
    n_img = torch.zeros((), dtype=torch.float64, device=device)
    s_cls = torch.zeros(cdec, dtype=torch.float64, device=device)

    seen_local = 0
    target_per_rank = args.num_samples // world_size
    for images, _ in loader:
        images = images.to(device)
        h1 = rae._block0(images)                             # [B, N+1, Cdec]
        cls = h1[:, 0, :].double()                           # [B, Cdec]
        patch = h1[:, 1:, :].double()                        # [B, N, Cdec]
        if s is None:
            N = patch.shape[1]
            s = torch.zeros(N, cdec, dtype=torch.float64, device=device)
            ss = torch.zeros(N, cdec, dtype=torch.float64, device=device)
        s += patch.sum(dim=0)                                # sum over BATCH only -> keep positions
        ss += (patch * patch).sum(dim=0)
        n_img += cls.shape[0]
        s_cls += cls.sum(dim=0)
        seen_local += images.shape[0]
        if rank == 0 and seen_local % (args.batch * 20) == 0:
            print(f"  ~{seen_local*world_size}/{args.num_samples}", flush=True)
        if seen_local >= target_per_rank:
            break

    # reduce across ranks -> full-data stats
    if world_size > 1:
        for t in (s, ss, s_cls, n_img):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    n_patch = n_img                                          # each image contributes 1 count per position
    mean = s / n_patch                                       # [N, Cdec]
    var = (ss / n_patch) - mean * mean
    mean_cls = s_cls / n_img

    if rank == 0:
        hw = int(N ** 0.5)
        # [N,Cdec] -> [hw,hw,Cdec] -> [Cdec,hw,hw]  (matches RAEDecoderH1.encode z layout)
        mean_pp = mean.view(hw, hw, cdec).permute(2, 0, 1).contiguous()
        var_pp = var.clamp_min(0).view(hw, hw, cdec).permute(2, 0, 1).contiguous()
        out = {
            "mean": mean_pp.float().cpu(),                   # [Cdec, 16, 16]
            "var": var_pp.float().cpu(),                     # [Cdec, 16, 16]
            "mean_cls": mean_cls.float().view(1, 1, cdec).cpu(),
            "num_samples": int(n_img.item()),
        }
        torch.save(out, args.out)
        print(f"saved -> {args.out}  | shape mean {tuple(mean_pp.shape)}  "
              f"mean[range {mean_pp.min():.3f},{mean_pp.max():.3f}]  var[range {var_pp.min():.4f},{var_pp.max():.4f}]  "
              f"(Cdec={cdec}, N={N}, n_img={int(n_img.item())})")


if __name__ == "__main__":
    main()
