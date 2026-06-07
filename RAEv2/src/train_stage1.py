"""Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
from copy import deepcopy

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from configs import Stage1Config
from data import prepare_unified_dataloader
from stage1.disc import LPIPS, build_discriminator
from eval.datasets import normalize_eval_datasets, prepare_eval_datasets
from stage1.engine import train_one_epoch
from stage1.utils import validate_stage1_config
from utils.checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.sync_utils import sync_checkpoint_blocking, sync_evals_blocking
from utils.train_utils import get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store outputs.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument('--wandb', action='store_true', help='Use W&B for logging.')
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--sync-checkpoints", action="store_true", help="Sync checkpoints to S3.")
    # Optional overrides so run_train_stage1.sh can set these from bash (like run_train_attnres.sh).
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Override training.checkpoint_interval (epochs between checkpoints).")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="Override training.sample_every (steps between val-recon dumps).")
    parser.add_argument("--val-image", type=str, default="assets/samples/sample_1.png",
                        help="Fixed val image (or any image in its dir) for recon viz, like run_train_attnres.sh.")
    return parser.parse_args()


def main():
    args = parse_args()

    #########################################################
    # Distributed + Config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage1Config), OmegaConf.load(args.config)))
    validate_stage1_config(config)

    # CLI overrides (let run_train_stage1.sh drive these from bash, like run_train_attnres.sh).
    if args.epochs is not None:
        config.training.epochs = args.epochs
        # Keep cosine decay spanning the full run (YAML convention: decay_end_epoch == epochs).
        if config.training.scheduler is not None:
            config.training.scheduler.decay_end_epoch = args.epochs
        if config.gan.scheduler is not None:
            config.gan.scheduler.decay_end_epoch = args.epochs
    if args.checkpoint_interval is not None:
        config.training.checkpoint_interval = args.checkpoint_interval
    if args.sample_every is not None:
        config.training.sample_every = args.sample_every

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)

    #########################################################
    # Dataset and dataloader (unified)
    #########################################################
    batch_size = config.training.global_batch_size // world_size if config.training.global_batch_size else config.training.batch_size
    dataloader_result = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    dataloader = dataloader_result.loader

    steps_per_epoch = config.training.virtual_epoch_steps if config.training.virtual_epoch_steps else len(dataloader_result)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches.")

    # Eval datasets (unified format)
    eval_datasets = None
    if config.eval is not None:
        eval_datasets_config = normalize_eval_datasets(config.eval.datasets)
        if eval_datasets_config:
            eval_datasets = prepare_eval_datasets(
                eval_datasets_config,
                image_size=config.training.image_size,
                batch_size=batch_size,
                num_workers=config.training.num_workers,
                rank=rank,
                world_size=world_size,
            )

    #########################################################
    # Model and DDP setup
    #########################################################
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)

    ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    if args.compile:
        ddp_model = torch.compile(ddp_model)

    # Discriminator
    discriminator, disc_aug = build_discriminator(config.gan.arch, device, config.gan.augment)
    ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    discriminator.train()

    lpips_model = LPIPS().to(device).eval()

    #########################################################
    # Optimizer and scheduler
    #########################################################
    optimizer, _ = build_optimizer(rae.decoder.parameters(), config.training.optimizer)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, _ = build_optimizer(disc_params, config.gan.optimizer)

    scheduler = None
    disc_scheduler = None
    if config.training.scheduler is not None:
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)
    if config.gan.scheduler is not None:
        disc_scheduler, _ = build_scheduler(disc_optimizer, steps_per_epoch, config.gan.scheduler)

    autocast_kwargs = get_autocast_kwargs(args)

    #########################################################
    # Resume
    #########################################################
    start_epoch, global_step = 0, 0
    maybe_ckpt = find_resume_checkpoint(experiment_dir)
    if maybe_ckpt:
        logger.info(f"Resuming from {maybe_ckpt}...")
        start_epoch, global_step = load_stage1_checkpoint(
            maybe_ckpt, ddp_model, ema_model, optimizer, scheduler,
            discriminator, disc_optimizer, disc_scheduler,
        )
        logger.info(f"Resumed epoch={start_epoch}, step={global_step}.")
    else:
        if rank == 0:
            save_worktree(experiment_dir, config, {"cmd_args": vars(args)})

    # Fixed validation images for recon viz — same set as run_train_attnres.sh
    # (assets/samples/*.png, "concat" grids skipped). Encoder normalization is
    # handled inside RAE.encode, so we feed plain [0, 1] images.
    viz_samples = None
    if args.val_image:
        from torchvision import transforms
        from PIL import Image as PILImage
        val_tf = transforms.Compose([
            transforms.Resize(config.training.image_size + 32),
            transforms.CenterCrop(config.training.image_size),
            transforms.ToTensor(),
        ])
        val_dir = os.path.dirname(os.path.abspath(args.val_image))
        val_paths = sorted(
            p for p in (glob.glob(os.path.join(val_dir, "*.png")) +
                        glob.glob(os.path.join(val_dir, "*.jpg")))
            if "concat" not in os.path.basename(p).lower()
        )
        if not val_paths:
            val_paths = [args.val_image]
        viz_samples = torch.stack([val_tf(PILImage.open(p).convert("RGB")) for p in val_paths]).to(device)
        logger.info(f"Loaded {len(val_paths)} fixed val images from {val_dir}")

    # Progress bar
    total_steps = config.training.epochs * steps_per_epoch
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training", disable=rank != 0)

    #########################################################
    # Train loop
    #########################################################
    dist.barrier()
    for epoch in range(start_epoch, config.training.epochs):
        dataloader_result.set_epoch(epoch)
        global_step = train_one_epoch(
            ddp_model, ema_model, ddp_disc, disc_aug, lpips_model,
            dataloader, optimizer, disc_optimizer, scheduler, disc_scheduler,
            autocast_kwargs, device, epoch, global_step, batch_size,
            config, args, logger, rank, world_size, checkpoint_dir, experiment_dir,
            progress_bar, eval_datasets, viz_samples,
        )
    progress_bar.close()

    #########################################################
    # Final checkpoint and cleanup
    #########################################################
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {config.training.epochs}...")
        save_stage1_checkpoint(
            f"{checkpoint_dir}/ep-{config.training.epochs:07d}.pt", global_step, config.training.epochs,
            ddp_model, ema_model, optimizer, scheduler, discriminator, disc_optimizer, disc_scheduler,
        )
        if args.sync_checkpoints:
            sync_checkpoint_blocking(checkpoint_dir, logger)
            sync_evals_blocking("evals/stage1", logger)

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
