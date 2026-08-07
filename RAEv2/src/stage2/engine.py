"""Stage 2 training engine: train_one_epoch and helpers."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import wandb
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.stage2 import Stage2Config
from stage2 import nwm_cond
from stage2.utils import (
    denoise_probe,
    encode_text,
    get_fixed_viz_batch_conditions,
    get_null_cond,
    sample_and_decode,
)
from utils import wandb_utils
from utils.checkpoint import save_stage2_checkpoint
from utils.guidance_utils import get_model_forward_fn
from utils.logging import save_eval_to_csv
from utils.sync_utils import sync_checkpoint_async, sync_evals_async
from utils.train_utils import update_ema

logger = logging.getLogger("rae")


#########################################################
# Main training function
#########################################################
def train_one_epoch(
    *, # * forces all arguments to be passed as keyword arguments
    ddp_model: DDP,
    ema_model: torch.nn.Module,
    rae,
    transport,
    eval_sampler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    gate_optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: Stage2Config,
    args,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    text_encoder=None,
    repa_target_encoder=None,
    eval_datasets: Optional[Dict] = None,
    viz_fixed: Optional[Dict] = None,
) -> int:
    """Run one epoch of Stage 2 training. Returns updated global_step.

    Args:
        viz_fixed: Mutable dict with keys 'zs', 'y', 'encoder_hidden_states',
            'encoder_attention_mask'. Populated from first batch, persists across epochs.
    """
    #########################################################
    # Setup
    #########################################################
    model = ddp_model.module

    # Guidance: derive model_fn / ema_model_fn / sample_kwargs from config
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    ema_model_fn, _ = get_model_forward_fn(ema_model, config.guidance)
    use_guidance = config.guidance.any_guidance_active

    # Eval settings
    do_eval = config.eval is not None and eval_datasets is not None
    if do_eval: eval_dir = config.eval.eval_dir
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    # dataset.target == "latent_cache": dataloader already yields (latent, label) from
    # scripts/stage1/precompute_latents.py -- skip the per-step stage_1.encode() forward
    # entirely. Only valid for a deterministic frozen latent (no gate grad, no raw-pixel
    # REPA target, no raw-pixel denoise-probe capture).
    use_cached_latents = getattr(config.dataset, "target", None) == "latent_cache"
    if use_cached_latents:
        assert not (gate_optimizer is not None and getattr(rae, "has_learnable_gate", False)), \
            "latent_cache dataset + learned_gate is incompatible: the gate needs a grad-enabled " \
            "encode_train() over raw images, which cached latents cannot provide."
        assert not config.repa.use_repa, \
            "latent_cache dataset + REPA is incompatible: REPA needs the raw image for its own " \
            "target encoder, which isn't stored in the cache."

    # Get null conditions for CFG dropout
    if config.conditioning.type == "nwm":
        model_kwargs_null = nwm_cond.null_context(config, micro_batch_size, device)
    else:
        model_kwargs_null = get_null_cond(text_encoder, config.conditioning.type, config.misc.num_classes, micro_batch_size, device)

    # per-epoch state
    num_viz_samples = viz_fixed['zs'].shape[0] if viz_fixed is not None else 0
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0
    optimizer.zero_grad()

    # save checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_model, optimizer, scheduler)
        # optional retention (CKPT_KEEP_RECENT>0): keep every-CKPT_KEEP_EVERY-epoch
        # milestones + the most recent N, prune the rest to bound disk. Default unset =
        # keep everything (unchanged for other users).
        keep_recent = int(os.environ.get("CKPT_KEEP_RECENT", "0"))
        if keep_recent > 0:
            import re
            keep_every = int(os.environ.get("CKPT_KEEP_EVERY", "10"))
            eps = sorted(int(m.group(1)) for f in os.listdir(checkpoint_dir)
                         if (m := re.fullmatch(r"ep-(\d+)\.pt", f)))
            keep = set(eps[-keep_recent:]) | {e for e in eps if keep_every > 0 and e > 0 and e % keep_every == 0}
            for e in eps:
                if e not in keep:
                    try: os.remove(f"{checkpoint_dir}/ep-{e:07d}.pt")
                    except OSError: pass
        if args.sync_checkpoints:
            sync_checkpoint_async(checkpoint_dir, logger)
            if do_eval: sync_evals_async(eval_dir, logger)

    #########################################################
    # Training loop
    #########################################################
    dataloader.set_epoch(epoch)
    # learned_gate: accumulate the gate gradient (diffusion + small-batch recon) over
    # gate_accum_steps DiT-steps before stepping -> larger effective batch, stable gate.
    gate_accum = getattr(config.training, "gate_accum_steps", 1) if gate_optimizer is not None else 1
    if gate_optimizer is not None and not hasattr(gate_optimizer, "_accum_i"):
        gate_optimizer._accum_i = 0
    # STEP_TIMING=1: env-gated per-step breakdown (data-wait / fwd / bwd+all-reduce / opt),
    # rank0, gstep 5-35, then quiet. Locates the multi-node cached-latent slowdown. The
    # cuda.syncs slightly inflate absolute numbers (they remove step-to-step overlap) but
    # the point is which section dominates. Calibrated for grad_accum=1. No cost when unset.
    _st = bool(int(os.environ.get("STEP_TIMING", "0"))) and rank == 0
    _t_end = time.perf_counter()
    for step, (images, y) in enumerate(dataloader):
        _t_data = time.perf_counter() - _t_end   # time blocked waiting for this batch
        images = images.to(device)

        # Encode images to latents and compute REPA targets.
        # learned_gate: the gate-weighted latent must carry gradient (grad-enabled
        # encode_train); otherwise the latent is the frozen, detached diffusion target.
        use_gate = gate_optimizer is not None and getattr(rae, "has_learnable_gate", False)
        # Decoupled full-mean target: x_t is built from the random-drop latent (z), but the
        # FM loss regresses to the deterministic full-mean latent (z_target). Requires the
        # per-step raw-image encode -> incompatible with cached latents / learned gate.
        decoupled_full_target = getattr(config.transport, "decoupled_full_target", False)
        z_target = None
        z_tokens = None
        if use_cached_latents:
            assert not decoupled_full_target, \
                "transport.decoupled_full_target needs a raw-image encode (z_drop + z_full); " \
                "it is incompatible with the latent_cache dataset."
            z = images  # dataloader already yielded the normalized post-combine latent
        elif use_gate:
            assert not decoupled_full_target, \
                "transport.decoupled_full_target is incompatible with learned_gate."
            z, z_tokens = rae.encode_train(images)
        elif decoupled_full_target:
            with torch.no_grad():
                z, z_target = rae.encode_cond_target(images)
        else:
            with torch.no_grad():
                z = rae.encode(images)
        with torch.no_grad():
            if repa_target_encoder is not None:
                raw_images = images.clone() * 255.0
                raw_img_preprocessed = repa_target_encoder.preprocess(raw_images)
                z_clean = repa_target_encoder.forward_features(raw_img_preprocessed)['x_norm_patchtokens']
            else:
                z_clean = None

        # Capture fixed conditions from first batch
        if viz_fixed is not None:
            if config.conditioning.type == "nwm":
                if viz_fixed['context'] is None:
                    viz_fixed['context'] = nwm_cond.viz_context(y, viz_fixed['zs'].shape[0], rae, device)
            else:
                first_capture = viz_fixed['context'] is None
                viz_fixed = get_fixed_viz_batch_conditions(viz_fixed, y, config.conditioning.type, text_encoder, device)
                # fixed images (aligned with the captured labels) for the denoise probe --
                # unavailable in latent_cache mode (no raw pixels in the cache), so the
                # probe_imgs-gated denoise-probe logging block later just stays skipped.
                if first_capture and config.conditioning.type == "label" and not use_cached_latents:
                    viz_fixed['probe_imgs'] = images[:viz_fixed['zs'].shape[0]].clone()

        # Encode conditions
        if config.conditioning.type == "text":
            context, context_attn_mask = encode_text(text_encoder, y)
        elif config.conditioning.type == "nwm":
            context = nwm_cond.encode_train_context(y, rae, device)
            context_attn_mask = None
        else:
            context, context_attn_mask = y.to(device), None

        #########################################################
        # Forward + backward
        #########################################################
        model_kwargs = dict(context=context, attn_mask=context_attn_mask)

        if _st:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        with autocast(**autocast_kwargs):
            loss_dict = transport.training_losses(
                ddp_model, z, model_kwargs, model_kwargs_null,
                z_clean=z_clean,
                repa_coeff=config.repa.repa_coeff if config.repa.use_repa else None,
                base_model_coeff=config.internal_guidance.base_model_coeff,
                cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                x1_target=z_target,
            )
            loss_diff = loss_dict["loss"].mean()
            loss_repa = loss_dict.get("loss_repa", torch.tensor(0.0, device=device)).mean()
            loss = loss_diff + loss_repa if config.repa.use_repa else loss_diff

            # learned_gate recon anchor: decode the RAW gate-weighted latent with the
            # FROZEN decoder -> L1 recon loss whose gradient reaches ONLY the gate.
            loss_recon = None
            recon_coeff = getattr(config.training, "gate_recon_coeff", 0.0) if use_gate else 0.0
            gbal = getattr(config.training, "gate_recon_balance", 0.0) if use_gate else 0.0
            if recon_coeff > 0 or gbal > 0:
                rbs = getattr(config.training, "gate_recon_bs", 16)
                rec = rae.recon_from_tokens(z_tokens[:rbs])
                loss_recon = (rec - images[:rbs]).abs().mean()

        # ---- learned_gate regularizers on the gate, each MAGNITUDE-MATCHED to the
        # diffusion gate-gradient so neither overpowers it:
        #   recon anchor : ||grad_gate(recon)||   = gbal * ||grad_gate(diffusion)||
        #   entropy hinge: ||grad_gate(H_pen)||   = gent * ||grad_gate(diffusion)||
        # H_pen = relu(log k - H(gate))^2 fires ONLY when the softmax starts collapsing
        # below ~k effective layers (keeps the kept-set spread, not winner-take-all).
        # Norms re-measured every gate_balance_every steps via autograd.grad, held between.
        gent = getattr(config.training, "gate_entropy_balance", 0.0) if use_gate else 0.0
        topk = getattr(rae.combine, "topk", 0) if use_gate else 0
        H_pen = None
        if gent > 0 and topk:
            wg = torch.softmax(rae.combine.gate_logits / rae.combine.tau, dim=0)
            H = -(wg * (wg + 1e-9).log()).sum()
            H_pen = torch.relu(math.log(topk) - H).pow(2)

        if use_gate and (gbal > 0 or H_pen is not None):
            g = rae.combine.gate_logits
            if step % max(1, getattr(config.training, "gate_balance_every", 50)) == 0:
                gd = torch.autograd.grad(loss_diff, g, retain_graph=True)[0].norm()
                gate_optimizer._gd = gd.item()
                if loss_recon is not None and gbal > 0:
                    gr = torch.autograd.grad(loss_recon, g, retain_graph=True)[0].norm()
                    gate_optimizer._recon_w = float((gbal * gd / (gr + 1e-8)).clamp(0, 1e4))
                    gate_optimizer._gr = gr.item()
                if H_pen is not None and H_pen.item() > 1e-8:
                    ge = torch.autograd.grad(H_pen, g, retain_graph=True)[0].norm()
                    gate_optimizer._ent_w = float((gent * gd / (ge + 1e-8)).clamp(0, 1e4))
        if loss_recon is not None:
            loss = loss + getattr(gate_optimizer, "_recon_w", recon_coeff) * loss_recon
        if H_pen is not None:
            loss = loss + getattr(gate_optimizer, "_ent_w", 0.0) * H_pen

        loss = loss / config.training.grad_accum_steps

        if _st:
            torch.cuda.synchronize(); _t_fwd = time.perf_counter() - _t0; _t0 = time.perf_counter()

        is_accum_step = (step + 1) % config.training.grad_accum_steps != 0
        if is_accum_step:
            with ddp_model.no_sync():
                loss.backward()
        else:
            loss.backward()  # DDP auto-syncs gradients on final micro-step

        if _st:
            torch.cuda.synchronize(); _t_bwd = time.perf_counter() - _t0; _t0 = time.perf_counter()

        if not is_accum_step:
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if gate_optimizer is not None:
                # accumulate the gate grad over gate_accum DiT-steps, then sync + step.
                # (grad keeps accumulating on gate_logits across these steps; we do NOT
                # zero it until we actually step.)
                gate_optimizer._accum_i += 1
                if gate_optimizer._accum_i >= gate_accum:
                    gp = rae.combine.gate_logits
                    if gp.grad is not None and world_size > 1:
                        # rae is NOT DDP-wrapped -> manually average the gate grad across ranks
                        dist.all_reduce(gp.grad)
                        gp.grad /= world_size
                    gate_optimizer.step()
                    gate_optimizer.zero_grad(set_to_none=True)
                    gate_optimizer._accum_i = 0
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, ddp_model.module, decay=config.training.ema_decay)
            global_step += 1

            # CKPT_EVERY_STEPS>0: roll a single ckpt_latest.pt every N optimizer steps so a
            # mid-epoch preemption loses at most N steps instead of the whole epoch. Reuses
            # save_stage2_checkpoint's atomic tmp+os.replace (a kill mid-write keeps the old
            # complete file); resume ranks it by (epoch, step) via get_checkpoint_sort_key.
            # Default 0 (off) preserves prior behavior; set in the launch script. Each save
            # blocks rank0 (other ranks wait at the next all-reduce), so size N vs that cost.
            ckpt_every_steps = int(os.environ.get("CKPT_EVERY_STEPS", "0") or "0")
            if ckpt_every_steps > 0 and rank == 0 and global_step % ckpt_every_steps == 0:
                logger.info(f"Rolling checkpoint at step {global_step} (epoch {epoch})...")
                save_stage2_checkpoint(f"{checkpoint_dir}/ckpt_latest.pt", global_step, epoch,
                                       ddp_model, ema_model, optimizer, scheduler)

            if _st:
                torch.cuda.synchronize()
                _t_opt = time.perf_counter() - _t0
                _t_total = time.perf_counter() - _t_end
                if 5 <= global_step <= 35:
                    logger.info(
                        f"[step-timing] gstep={global_step} total={_t_total:.3f}s | "
                        f"data={_t_data:.3f} fwd={_t_fwd:.3f} bwd+ar={_t_bwd:.3f} opt={_t_opt:.3f}"
                    )
                _t_end = time.perf_counter()

        epoch_metrics['loss'] += loss_diff.detach()
        num_batches += 1
        progress_bar.update(1)

        # Skip logging/viz/eval on non-boundary micro-steps
        if is_accum_step:
            continue

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            cur_loss = loss_diff.item()
            stats = {"train/loss": cur_loss, "train/lr": optimizer.param_groups[0]["lr"]}
            if config.repa.use_repa:
                stats["train/loss_repa"] = loss_repa.item()
            if "loss_base" in loss_dict:
                stats["train/loss_base"] = loss_dict["loss_base"].mean().item()
            if gate_optimizer is not None:
                w = rae.gate_weights()                       # [K] current softmax gate
                ent = -(w * (w + 1e-9).log()).sum().item()   # entropy: log(K) -> 0 = polarized
                stats["gate/entropy"] = ent
                stats["gate/max"] = w.max().item()
                stats["gate/n_active"] = (w > 0.5 / w.numel()).sum().item()
                if loss_recon is not None:
                    stats["gate/recon_loss"] = loss_recon.item()
                if getattr(gate_optimizer, "_gd", None) is not None:
                    stats["gate/recon_w"] = getattr(gate_optimizer, "_recon_w", 0.0)
                    stats["gate/ent_w"] = getattr(gate_optimizer, "_ent_w", 0.0)
                    stats["gate_grad/diff_norm"] = gate_optimizer._gd        # ||grad_gate(diffusion)||
                    stats["gate_grad/recon_unit_norm"] = getattr(gate_optimizer, "_gr", 0.0)
                topk = getattr(rae.combine, "topk", 0)
                if topk and 0 < topk < w.numel():
                    top = torch.topk(w, topk).indices.sort().values.tolist()
                    sel = [rae.combine.layers[i] for i in top]
                    logger.info(f"[gate top-{topk} layers] {sel}")
                for li, lyr in enumerate(rae.combine.layers):
                    stats[f"gate/L{lyr}"] = w[li].item()
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            )
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=cur_loss, lr=optimizer.param_groups[0]["lr"])

        # Sampling visualization
        if global_step % config.training.sample_every == 0:
            model.eval()
            logger.info("Generating EMA samples...")
            sample_args = dict(
                eval_sampler=eval_sampler, model_fn=ema_model_fn,
                sample_model_kwargs=sample_model_kwargs, rae=rae,
                use_guidance=use_guidance, condition_type=config.conditioning.type,
                text_encoder=text_encoder, num_classes=config.misc.num_classes,
                device=device, autocast_kwargs=autocast_kwargs,
            )
            if rank == 0:
                with torch.no_grad():
                    samples_dict = {}
                    # 1. Batch samples (from current batch conditions)
                    is_dict_ctx = isinstance(context, dict)
                    batch_n = min(num_viz_samples, nwm_cond.batch_size(context) if is_dict_ctx else context.shape[0])
                    zs_batch = torch.randn(batch_n, *config.misc.latent_size, device=device, dtype=torch.float32)
                    samples_dict["samples/batch"] = sample_and_decode(
                        zs_batch, nwm_cond.slice(context, batch_n) if is_dict_ctx else context[:batch_n],
                        context_attn_mask[:batch_n] if context_attn_mask is not None else None,
                        **sample_args,
                    )
                    # 2. Fixed samples (consistent across epochs)
                    if viz_fixed is not None and viz_fixed['context'] is not None:
                        fixed_ctx = viz_fixed['context']
                        fixed_ctx_clone = nwm_cond.clone_context(fixed_ctx) if isinstance(fixed_ctx, dict) else fixed_ctx.clone()
                        samples_dict["samples/fixed"] = sample_and_decode(
                            viz_fixed['zs'].clone(), fixed_ctx_clone,
                            viz_fixed['attn_mask'].clone() if viz_fixed['attn_mask'] is not None else None,
                            **sample_args,
                        )
                    # save sample grids locally (wandb-independent visualization)
                    from torchvision.utils import save_image as _save_image
                    sample_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    for name, samples in samples_dict.items():
                        out_png = os.path.join(sample_dir, f"{name.replace('/', '_')}_s{global_step:07d}.png")
                        _save_image(samples.clamp(0, 1), out_png, nrow=round(samples.shape[0] ** 0.5))
                    logger.info(f"Saved sample grids to {sample_dir} (step {global_step})")
                    if args.wandb: # log samples to wandb
                        for name, samples in samples_dict.items():
                            grid = wandb_utils.array2grid(samples)
                            wandb.log({name: wandb.Image(grid)}, step=global_step)
            dist.barrier()
            logger.info("Generating EMA samples done.")
            model.train() # set model back to train mode

        #########################################################
        # Evaluation; distributed evaluation
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step % config.eval.eval_interval == 0:
            from eval import evaluate_generation_distributed
            logger.info("Starting evaluation...")
            model.eval()
            # eval ema or both ema and running model if eval_model is True
            eval_models = [(ema_model_fn, "ema")] if not config.eval.eval_model else [(ema_model_fn, "ema"), (model_fn, "model")]
            for fn, mod_name in eval_models:
                for ds_name, ds_info in eval_datasets.items():
                    logger.info(f"Evaluating {mod_name} on {ds_name}...")
                    eval_n = min(ds_info.num_samples or len(ds_info.dataset), len(ds_info.dataset))
                    eval_stats = evaluate_generation_distributed(
                        fn, eval_sampler, tuple(config.misc.latent_size), sample_model_kwargs,
                        use_guidance, rae, ds_info.dataset, eval_n,
                        rank=rank, world_size=world_size, device=device,
                        batch_size=micro_batch_size, experiment_dir=experiment_dir,
                        global_step=global_step, autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz,
                        shared_tmpdir=config.dataset.shared_tmpdir,
                        condition_type=ds_info.condition_type,
                        null_label=config.misc.num_classes,
                        text_encoder=text_encoder if ds_info.condition_type == "text" else None,
                        metrics_to_compute=ds_info.metrics,
                        data_dir=ds_info.data_dir,
                    )
                    if eval_stats is not None and rank == 0:
                        save_eval_to_csv(experiment_name, mod_name, global_step, {'dataset': ds_name, **eval_stats}, eval_dir)
                        if args.wandb:
                            wandb_utils.log({f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}, step=global_step)
            model.train() # set model back to train mode
            logger.info("Evaluation done.")


    #########################################################
    # Denoise probe — pixel-space PSNR of EMA x-prediction at a fixed t grid.
    # The cross-run comparable progress metric (latent losses are not comparable
    # across different stage-1 latent spaces; pixel space is shared).
    #########################################################
    if (rank == 0 and viz_fixed is not None and config.conditioning.type == "label"
            and config.transport.prediction == "x"
            and viz_fixed.get('probe_imgs') is not None):
        probe = denoise_probe(ema_model, rae, viz_fixed, autocast_kwargs)
        if probe is not None:
            t_grid = [k for k in probe if k != 'ceiling']
            grid = "  ".join(f"t{int(t*100)}={probe[t]:.2f}" for t in t_grid)
            logger.info(f"[Epoch {epoch}] Denoise PSNR (EMA): {grid}  ceil={probe['ceiling']:.2f} dB")
            if args.wandb:
                wandb_utils.log({**{f"val/denoise_psnr_t{int(t*100)}": probe[t] for t in t_grid},
                                 "val/denoise_psnr_ceiling": probe['ceiling']}, step=global_step)

    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        avg_loss = epoch_metrics['loss'].item() / num_batches
        epoch_stats = {"epoch/loss": avg_loss}
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
