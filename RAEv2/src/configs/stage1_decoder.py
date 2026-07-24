"""Config dataclasses for the unified stage-1 MLS-decoder trainer
(src/train_decoder.py). Mirrors the OmegaConf structured-config style of
src/configs/stage2.py: load with

    cfg = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(DecoderConfig), OmegaConf.load(path)))

The `combine` block is a generic ModelConfig instantiated via
instantiate_from_config (target: stage1.combine.MLSCombine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .shared import ModelConfig


@dataclass
class DecoderModuleConfig:
    """ViT decoder built from scratch (configs/decoder/ViTXL)."""
    config_path: str = "configs/decoder/ViTXL"
    latent_dim: int = 1024
    patch_size: int = 16
    num_patches: int = 256


@dataclass
class SigregConfig:
    weight: float = 0.02
    distributed: bool = True
    scale_by_n: bool = True


@dataclass
class KLConfig:
    """VAE-style KL(N(mu,sigma^2) || N(0,I)) weight. Only has effect when the combine
    was built with variational=true (stage1.combine.VAEBottleneck); ignored otherwise
    (train_decoder.py warns if set without a variational combine)."""
    weight: float = 1.0e-6          # LDM-scale default (train_decoder_mls_kl.py precedent)


@dataclass
class GanConfig:
    disc_weight: float = 0.75
    disc_start: int = 1            # epoch GAN turns on
    disc_ckpt: str = "pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth"


@dataclass
class LossConfig:
    lpips_w: float = 1.0
    sigreg: Optional[SigregConfig] = None          # None -> SIGReg off
    kl: Optional[KLConfig] = None                  # None -> KL off (needs combine.variational=true)
    gan: GanConfig = field(default_factory=GanConfig)


@dataclass
class DataConfig:
    data_dir: str = "/datasets/imagenet-256-full"
    image_size: int = 256
    num_workers: int = 4
    val_npz: str = "data_eval/imagenet-256-val.npz"
    val_n: int = 1000
    # Optional multi-source weighted mix (official general recipe). When set, the
    # trainer builds the loader via prepare_unified_dataloader instead of the single
    # data_dir loader. Each entry: {target, name, weight, ... + per-source loader keys
    # (data_dir/splits/subsets/hf_repo/hf_paths/cache_dir/cache_size)}.
    mix: Optional[List[Any]] = None
    virtual_epoch_steps: Optional[int] = None   # cap mix epoch length (steps); None = full


@dataclass
class MaskCondConfig:
    """Mask conditioning (Variant A): the trainer samples stratified per-sample
    layer masks, builds z from the masked mean AND feeds the same mask to the
    decoder via MaskEmbedder + zero-init AdaLN (stage1/mask_cond.py). None ->
    everything unchanged (unconditional random-drop training)."""
    d_c: int = 256                # condition dim (small: shared AdaLN head keeps params <1%)
    cond_drop: float = 0.1        # CFG-style: replace c with learned null embedding
    full_frac: float = 1 / 3      # stratified sampler: all-ones (full feed) group
    uniform_frac: float = 1 / 3   # |S| ~ Uniform{1..K} group; remainder = iid Bernoulli(p_drop)


@dataclass
class PDropScheduleConfig:
    """Anneal MLSCombine.p_drop from `start` to `end` over training. Supports both
    a<b (0.1->0.9) and a>b (0.9->0.1). When p_drop_schedule is None the static
    combine.p_drop is used. (Ported from the main-repo trainer.)"""
    start: float = 0.3
    end: float = 0.3
    type: str = "linear"        # linear | cosine
    warmup_frac: float = 0.0    # hold p_drop at `start` for the first warmup_frac of training


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32          # per GPU (micro-batch when grad_accum_steps > 1)
    grad_accum_steps: int = 1     # accumulate N micro-batches per optimizer step
    lr: float = 8.0e-4
    warmup_epochs: int = 2
    ema_decay: float = 0.9995
    clip_grad: float = 1.0
    precision: str = "bf16"       # fp32 | bf16
    ckpt_every: int = 1
    log_every: int = 50
    seed: int = 42
    out_dir: str = "output_full/decoder_run"
    init_from: Optional[str] = None   # warm-start weights from an external ckpt
                                      # (combine+decoder+ema+disc); fresh optimizer/epoch
    p_drop_schedule: Optional[PDropScheduleConfig] = None   # anneal MLSCombine.p_drop start->end


@dataclass
class ProbeConfig:
    loo_solo: str = "final"       # off | final | every
    val_image: Optional[str] = "assets/samples/sample_1.png"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "raev3-full"
    entity: str = "uscgvl"
    name: Optional[str] = None


@dataclass
class DecoderConfig:
    combine: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderModuleConfig = field(default_factory=DecoderModuleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    mask_cond: Optional[MaskCondConfig] = None   # Variant A mask conditioning; None = off
