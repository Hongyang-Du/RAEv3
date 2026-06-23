"""2-GPU smoke test for the grad-accum + noise_tau decoder training path.

Exercises the exact loop logic edited in train_decoder.py (micro-batch accum,
optimizer/EMA/disc stepping only on accum boundaries, GAN on) under real DDP,
real models, and the real config — without iterating the full 1.28M-image epoch.
Run: torchrun --nproc_per_node=2 scripts/smoke_decoder_accum.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from omegaconf import OmegaConf

from configs.stage1_decoder import DecoderConfig
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug
from stage1.disc.lpips import LPIPS as LPIPS_
from stage1.disc.utils import RandomWindowCrop
from utils.model_utils import get_obj_from_str

CFG = "configs/stage1/decoder/random-drop-layer-mls-plain-cls-k23.yaml"


def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank(); local = int(os.environ["LOCAL_RANK"])
    dev = torch.device(f"cuda:{local}"); torch.cuda.set_device(dev)
    is_main = rank == 0
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DecoderConfig), OmegaConf.load(CFG)))
    C, L, D, T = cfg.combine, cfg.loss, cfg.data, cfg.training
    BS = T.batch_size; accum = max(1, T.grad_accum_steps)
    if is_main:
        print(f"[cfg] batch/GPU={BS} accum={accum} eff_global={BS*dist.get_world_size()*accum} "
              f"noise_tau={C.params.get('noise_tau')} disc_weight={L.gan.disc_weight}", flush=True)

    layers = list(C.params["layers"]); ls = ".".join(map(str, layers))
    enc = create_encoder(f"dinov3mls-vit-l16[layers={ls}]", device=dev, resolution=D.image_size).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    mean = torch.tensor([0.485,0.456,0.406], device=dev).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=dev).view(1,3,1,1)
    def encode(x): return list(enc.model.get_intermediate_layers((x-mean)/std, n=enc.layer_indices,
                              reshape=False, return_class_token=False, norm=True))

    ld = cfg.decoder.latent_dim
    combine = get_obj_from_str(C.target)(**C.params).to(dev)
    decoder = _load_decoder(cfg.decoder.config_path, hidden_size=ld, patch_size=cfg.decoder.patch_size,
                            num_patches=cfg.decoder.num_patches, pretrained_path=None).to(dev)
    combine_has = any(True for _ in combine.parameters())
    combine_ddp = DDP(combine, device_ids=[local]) if combine_has else combine
    decoder_ddp = DDP(decoder, device_ids=[local])
    ema_combine = deepcopy(combine).eval(); ema_dec = deepcopy(decoder).eval()
    for m in (ema_combine, ema_dec):
        for p in m.parameters(): p.requires_grad_(False)
    trainable = list(combine.parameters()) + list(decoder.parameters())

    lpips = LPIPS_().to(dev).eval()
    for p in lpips.parameters(): p.requires_grad_(False)
    disc = DinoDiscriminator(device=dev, dino_ckpt_path=L.gan.disc_ckpt, ks=1, recipe="S_8",
                             norm_type="bn", using_spec_norm=True).to(dev)
    disc.dino_proxy[0].crop = RandomWindowCrop(D.image_size, 224, 9, False)
    disc.dino_proxy[0].original_input_size = D.image_size
    disc_ddp = DDP(disc, device_ids=[local])
    disc_aug = DiffAug(prob=0.5, cutout=True)
    opt = torch.optim.AdamW(trainable, lr=T.lr, betas=(0.9,0.95))
    dopt = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5,0.9))
    ac = dict(device_type="cuda", dtype=torch.bfloat16)

    def decode_imgs(dec, z):
        out = dec(z, drop_cls_token=False).logits
        m = dec.module if isinstance(dec, DDP) else dec
        return (m.unpatchify(out) * std + mean).clamp(0, 1)

    # snapshot ALL decoder weights + ema to check stepping cadence (aggregate
    # norm, not a single param — the first param can have ~zero grad).
    w0 = [p.detach().clone() for p in decoder.parameters()]
    e0 = next(ema_dec.parameters()).detach().clone()

    combine_ddp.train() if combine_has else None; decoder_ddp.train()
    opt.zero_grad(set_to_none=True); dopt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(dev)
    N_MICRO = 6  # 3 optimizer steps at accum=2; GAN on the whole time
    opt_steps = 0
    for micro_idx in range(N_MICRO):
        imgs = torch.rand(BS, 3, D.image_size, D.image_size, device=dev)
        with torch.no_grad(): lt = encode(imgs)
        is_accum = ((micro_idx + 1) % accum == 0)
        with torch.autocast(**ac):
            z = combine_ddp(lt); x_rec = decode_imgs(decoder_ddp, z)
            l1 = F.l1_loss(x_rec, imgs); lp = lpips(x_rec*2-1, imgs*2-1).mean()
        lrec = l1 + L.lpips_w * lp
        disc_ddp.eval(); half = x_rec.shape[0]
        with torch.autocast(**ac):
            fa = disc_aug.aug(x_rec[:half]*2-1); lf,_ = disc_ddp(fa, None)
        lg = vanilla_g_loss(lf)
        last = next(reversed(list(decoder_ddp.module.parameters())))
        aw = calculate_adaptive_weight(lrec, lg, last).clamp(0,1e4).detach()
        loss = lrec + L.gan.disc_weight * aw * lg
        (loss / accum).backward()
        if is_accum:
            torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
            opt.step(); opt.zero_grad(set_to_none=True); opt_steps += 1
        disc_ddp.train()
        with torch.autocast(**ac):
            ra = disc_aug.aug(imgs[:half]*2-1); fa2 = disc_aug.aug(x_rec[:half].detach()*2-1)
            lr_,_ = disc_ddp(ra, None); lf2,_ = disc_ddp(fa2, None); ld_ = hinge_d_loss(lr_, lf2)
        (ld_ / accum).backward()
        if is_accum:
            dopt.step(); dopt.zero_grad(set_to_none=True)
            update_ema(ema_combine, combine, T.ema_decay)
            update_ema(ema_dec, decoder, T.ema_decay)
        if is_main:
            print(f"  micro {micro_idx} accum_boundary={is_accum} loss={loss.item():.4f} "
                  f"l1={l1.item():.4f} lpips={lp.item():.4f} gan={lg.item():.4f} finite={torch.isfinite(loss).item()}",
                  flush=True)

    torch.cuda.synchronize()
    total_delta = sum((p.detach() - w).abs().sum().item() for p, w in zip(decoder.parameters(), w0))
    w_moved = total_delta > 0
    e_moved = not torch.equal(e0, next(ema_dec.parameters()).detach())
    peak = torch.cuda.max_memory_allocated(dev)/1e9
    if is_main:
        print(f"\n[RESULT] opt_steps={opt_steps} (expected {N_MICRO//accum}) "
              f"decoder_updated={w_moved} (|Δ|={total_delta:.3e}) ema_updated={e_moved} peak={peak:.1f}GB", flush=True)
        ok = (opt_steps == N_MICRO//accum) and w_moved and e_moved
        print("[SMOKE]", "PASS" if ok else "FAIL", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
