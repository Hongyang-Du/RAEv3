"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def _prune_old_checkpoints(path: str) -> None:
    """Retention for ep-*.pt siblings of `path`, guarded by env so default is keep-all.

    CKPT_KEEP_RECENT=N  -> keep the N highest-epoch checkpoints.
    CKPT_KEEP_EVERY=K   -> additionally keep every-K-epoch milestones.
    Prevents the per-user disk quota from filling up (which truncates the next save
    into a corrupt zip and sends resume into a crash loop).
    """
    keep_recent = int(os.environ.get("CKPT_KEEP_RECENT", "0") or "0")
    keep_every = int(os.environ.get("CKPT_KEEP_EVERY", "0") or "0")
    if keep_recent <= 0:
        return
    d = os.path.dirname(path)
    cks = []
    for f in glob.glob(os.path.join(d, "ep-*.pt")):
        m = re.search(r"ep-(\d+)\.pt$", f)
        if m:
            cks.append((int(m.group(1)), f))
    cks.sort()
    keep = {f for _, f in cks[-keep_recent:]}
    if keep_every > 0:
        keep |= {f for e, f in cks if e > 0 and e % keep_every == 0}
    for _, f in cks:
        if f not in keep:
            try:
                os.remove(f)
            except OSError:
                pass


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator)."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic write: save to a tmp then rename, so an interrupted save (preemption /
    # quota-exceeded mid-write) never leaves a truncated/corrupt ep-*.pt that resume
    # would then pick as "latest" and crash-loop on.
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    _prune_old_checkpoints(path)


def load_stage1_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic write: a crash mid-save leaves the .tmp (ignored by resume), never a
    # half-written ep-*.pt — so the previous complete checkpoint stays usable.
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    # tolerate older checkpoints saved without EMA: re-init EMA from model
    ema_state = checkpoint.get("ema")
    ema_model.load_state_dict(ema_state if ema_state is not None else checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
]
