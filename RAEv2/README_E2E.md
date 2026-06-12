# End-to-End Training: Projector (SIGReg) + Decoder + DiT

Joint training of the representation, the decoder, and the diffusion model in a
single loop — no frozen stage-1, no EMA target, no stop-grad. The bet (validated
by [LeWorldModel](https://arxiv.org/abs/2603.19312) in the world-model setting):
**SIGReg pins the latent's distributional geometry and the recon loss pins its
information content, so the generative branch cannot collapse the
representation** even though its target is fully live.

Motivation from the two-stage results (see STATUS.md): freezing a
SIGReg-gaussianized latent did NOT improve DiT FID (16.5 / 16.9 / 17.3 — a tie),
so the latent's *static* geometry is not the bottleneck. E2E tests whether
*joint adaptation* — the latent actively reshaping toward what the DiT can
predict — is where the gain lives (the REPA-E hypothesis, transplanted to RAE).

## Architecture

```
imgs [B,3,256,256] ∈ [0,1]
  │
  ▼  frozen DINOv3-L, layers (11,13,15,17,19,21,23), no grad
7 × patch tokens [B,256,1024]
  │
  ▼  MLSProjector  (~8.4M, trainable)
  │    z0 = mean over 7 layers                  (nodrop)
  │         or random-subset mean, p_drop=0.3   (drop variant, train-time only)
  │    z  = skip(z0) + fc2( GELU( BN( fc1(z0) ) ) )
  │         LeWM recipe: BatchNorm on the HIDDEN dim (4096) over B·256 token
  │         samples; output is a bare Linear — no norm constrains the
  │         distribution SIGReg shapes
  ▼
z [B,256,1024]  ──► ViT-XL decoder (~450M, trainable) ──► x_rec
  │
  ▼  xt = (1−t)·z + t·ε,  t ~ logit-normal + shift 8   (z NOT detached)
DDT-XL DiT (875M, trainable, same config as stage-2 baselines)
  ──► ẑ (x-prediction)  +  ẑ_base (Internal-Guidance head @ block 8)
```

## Loss

```
L = w_rec · [ L1(dec(z), GT) + LPIPS(dec(z), GT) ]                 # information anchor
  + sigreg_w · SIGReg(z)                                           # geometry anchor
  + w_fm · (1/max(t,.05)²) · [ (ẑ−z)² + base_coeff·(ẑ_base−z)² ]   # flow matching
  + w_pix · [ L1(dec(ẑ), dec(z)) + LPIPS(dec(ẑ), dec(z)) ]         # OPTIONAL, default 0
```

Defaults: `w_rec=1, sigreg_w=1, w_fm=1, w_pix=0` (minimal design — the live FM
target already routes generation gradients into the projector; the pixel branch
only adds perceptual reweighting at the cost of a 2nd decoder forward + 2nd
LPIPS per step).

### Gradient flow

| loss      | projector | decoder | DiT |
|-----------|-----------|---------|-----|
| `L_rec`   | ✓ (info anchor) | ✓ | |
| `SIGReg`  | ✓ (geometry anchor) | | |
| `L_fm`    | ✓ via target side (z pulled toward ẑ) **and** via xt through the whole DiT | | ✓ |
| `L_pix` (if on) | ✓ | ✓ | ✓ |

The FM target side is the collapse-flavored direction ("target chases
prediction"); recon + SIGReg are the two anchors holding it. **Collapse alarm =
the per-epoch `Val PSNR` line falling.** Safety valves (all default OFF):

- `--detach-fm-target`  stop-grad z in the FM target (REPA-E style)
- `--detach-xt`         stop-grad z in the xt interpolation
- `--detach-pix-target` stop-grad dec(z) in L_pix
- `--pix-t-weight`      weight L_pix per-sample by (1−t)

### SIGReg is GLOBAL across GPUs

`sigreg_loss(..., distributed=True)`: projection directions are broadcast from
rank 0 and the per-projection cos/sin ECF means are all-reduced
**differentiably** (`torch.distributed.nn`), so the Epps-Pulley statistic is
computed on the pooled 8-GPU batch (8·24·256 ≈ 49k tokens), not per rank.
Per-rank statistics would keep an O(1/B_local) sampling-noise floor (the ECF
test is nonlinear in the sample means). Cost: 8 tiny `[1024]` all-reduces per
step. BatchNorm inside the projector stays per-GPU on purpose — it is an
internal conditioner (6144 token samples/GPU ⇒ ~1.3% stat error), not the
distribution objective; DDP `broadcast_buffers` keeps eval stats consistent.

### EMA policy

Only the DiT keeps an EMA copy (`ema_dit`) — the raw-vs-EMA sampling gap in
diffusion is large and the stage-2 baseline FIDs are EMA, so dropping it would
handicap the comparison. EMA weights are accumulated during training (a running
average of the whole trajectory cannot be reconstructed afterwards). Projector
and decoder are evaluated **live** (eval mode: BN running stats, layer-drop
off) — a more responsive collapse alarm with no 1/(1−decay)-step lag.
`update_ema` copies buffers (BN running stats) instead of EMA-ing them.

## The two variants

| variant | layer dropout | warm-start (proj Linears + decoder) | out dir |
|---------|---------------|--------------------------------------|---------|
| `nodrop` | 0.0 — deterministic FM target | nogate stage-1 ckpt | `output_full/train_e2e_nodrop` |
| `drop`   | 0.3 — stochastic z ⇒ FM target jitters (raises the FM loss floor; buys balanced layer reliance) | dropmean stage-1 ckpt | `output_full/train_e2e_drop` |

Warm-start is approximate for the projector (Linear weights remap from the old
LN recipe via `load_ln_ckpt`, BN stats start fresh) and exact for the decoder.
DiT trains from scratch (`--init-dit` optional).

## How to run

```bash
# full queue: nodrop → FID → drop → FID → summary (auto-resumes; FID skipped if done)
docker exec -d rae bash -c "cd /workspace/RAEv2 && \
    nohup bash run_all_e2e.sh > output_full/run_all_e2e.log 2>&1"

# single variant
bash run_train_e2e_sigreg_dit.sh nodrop   # or: drop

# offline FID for any e2e ckpt (1 GPU, ~15 min at 5k/batch 256)
python src/eval_fid_e2e.py --ckpt output_full/train_e2e_nodrop/ckpt_latest.pt \
    --num-samples 5000 --out output_full/train_e2e_nodrop/fid.json
```

Stop: `pkill -9 -f run_all_e2e.sh ; pkill -9 -f '[t]rain_e2e_sigreg_dit'`.
Re-running the queue resumes: the ckpt stores all three models + ema_dit + both
optimizers + both schedulers + epoch/step.

Schedule: 10 epochs, global batch 192 (24/GPU × 8), bf16, AdamW(1e-4) for
projector+decoder, gmuon(2e-4) for DiT, 1-epoch warmup → cosine.

## Monitoring

Per 50 steps (stdout + wandb `uscgvl/raev3-full`, run `e2e-sigreg-dit-k7-bn[-drop0.3]`):
`train/loss·rec·fm·sigreg·pix`, batch recon `psnr`, latent stats
(`z_var_mean` ≈ 0.9 and `z_std` ≈ 1.0 mean SIGReg is in control), both lrs.

Per epoch (fixed val images, live proj/dec + EMA DiT):

- `Val PSNR (EMA): xx.xx dB` — recon quality, **the collapse alarm**; compare
  with stage-1 finals (25.8–26.0 dB)
- `Denoise PSNR (EMA): t25=…  t50=…  t75=…  t95=…  ceil=…` — single-step
  x-pred PSNR at 4 noise levels, same protocol as the stage-2 baselines
  (dropmean final: 23.10 / 22.02 / 19.82 / 13.75)

Per 2500 steps: fixed-noise 50-step Euler sample grid → `samples_s*.png` + wandb.

## Evaluation vs the two-stage baselines

`eval_fid_e2e.py` mirrors the stage-2 sweep protocol exactly: N class-balanced
EMA samples, seed 42, 50-step Euler with the official **shift-8 time grid**,
vs N real train images (torch-fidelity). Reference numbers (5k, no guidance,
10-epoch screening): raev2 16.47, nogate 16.85, dropmean 17.26
(`ckpts_full/stage2/dit_fid_compare.png`). No guidance is applied anywhere
(this repo's sampler implements neither CFG nor IG at inference; the IG base
head is trained and sits in the ckpt for later use).

## Files

| file | role |
|------|------|
| `src/train_e2e_sigreg_dit.py` | the training loop (everything above) |
| `src/overfit_sigreg.py` | `sigreg_loss` (per-rank / global), `gaussian_diag`, `psnr` |
| `src/eval_fid_e2e.py` | offline generation FID for e2e ckpts |
| `run_train_e2e_sigreg_dit.sh` | single-variant launcher (`nodrop` / `drop`) |
| `run_all_e2e.sh` | queue: both variants + FID + summary |
| `third_party/le-wm/` | LeWorldModel reference implementation (BN-projector recipe, live-target + SIGReg precedent) |

## Open questions this run should answer

1. Does joint adaptation beat the frozen-latent tie? (e2e FID vs 16.5/16.9/17.3)
2. Does the live FM target stay stable with only recon+SIGReg as anchors?
   (`Val PSNR` trajectory; if it falls → `--detach-fm-target`)
3. Does layer dropout's FM-target jitter cost more than its balanced-reliance
   benefit? (nodrop vs drop, same seeds)
4. Does the BN projector + global SIGReg change latent stats vs the LN stage-1
   numbers (var_mean 0.89–0.94)?
