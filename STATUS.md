# Project: RAEv3 — MLS latent decoder (DINOv3 multi-layer fusion + SIGReg)

## Meta
- **Phase**: Experiment
- **Target**: TBD
- **Last Updated**: 2026-06-11
- **Updated By**: dgx session

## Current Focus
How to combine K=7 frozen DINOv3-L layers (11,13,15,17,19,21,23) into one decodable
latent without the combine collapsing to the shallowest layer (L11). Any *learnable*
per-layer weight collapses under recon+GAN; now testing a param-free random-subset
combine (dropmean).

## Experiment Status
| Run | Config | Status | Key Metric (val PSNR, EMA) | Notes |
|-----|--------|--------|-----------|-------|
| mls_raev2 | fixed mean, no SIGReg | done (5 ep) | 25.98 | baseline |
| mls_nogate_sigreg | fixed mean + projector + SIGReg | done (5 ep) | 26.02 | SIGReg ~free vs baseline |
| mls_softgate_sigreg | softmax gate + per-step layer-drop 0.2 | KILLED at ep3 | 27.57 (ep3) | gate collapsed to [0.98 0.02 0 ...] = L11 only; higher PSNR is the shallow-layer hack, defeats fusion |
| mls_dropmean_sigreg | NO gate; per-sample random-subset equal-weight mean, layer_drop 0.3 | done (5 ep) | 25.80 | collapse-free; only 0.22 dB below nogate |

All runs: imagenet-256-full, 8 GPU, batch 32/GPU, 5 epochs, bf16, lr 8e-4, ViT-XL decoder,
L1+LPIPS+GAN+SIGReg(w=1). Logs: `RAEv2/output_full/<run>/train.log`; compare plot:
`RAEv2/output_full/val_psnr_compare.png` (plot_val_psnr.py).

## Key Results (stage-2 / semantics, 2026-06-12)
- DiT 10-ep FID (5k, no guidance): raev2 16.5 / nogate 16.9 / dropmean 17.3 — a TIE
  (gaps within 5k-FID noise). Denoise-probe PSNR advantage of SIGReg runs did NOT
  translate into FID at this schedule. Curves: ckpts_full/stage2/dit_fid_compare.png.
- Linear probe top-1 (25k held-out, 5 ep): DINOv3 layers monotonic 57.5 (L11) -> 87.3
  (L23); mls_mean 85.4; raev2 combine 87.1 (CLS surrogate +1.7); nogate 84.3 /
  dropmean 84.3 (SIGReg projector costs ~1.1 vs its input). Figure:
  output_full/linear_probe_compare.png; runner: run_linear_probes.sh.

## Key Results
- SIGReg costs nothing on recon (26.02 vs 25.98) while shaping z toward N(0,I).
- Learnable gates (free sigmoid AND softmax+dropout-0.2) both collapse to L11.
- Shallow-collapse *raises* val PSNR (27.57) — recon PSNR alone cannot detect this
  failure mode; check gate weights / per-subset decoding instead.
- Gate-free collapse readout (`src/probe_layer_usage.py`, works on any nogate/dropmean
  ckpt): LOO dPSNR per layer (reliance) + solo PSNR per layer (sufficiency).
- nogate ep5 probe: decoder reliance is shallow-heavy even with fixed mean —
  LOO dPSNR = [+3.80, +0.89, +0.25, +0.33, +0.34, +0.57, +0.19] (L11..L23),
  solo = [20.9, 21.9, 23.3, 16.1, 13.3, 11.1, 10.0]; deep-only 22.22 dB,
  deepest-3 only 12.88 dB. Baseline for dropmean to beat on deep-layer usage.

## Decisions Made
- 2026-06-10: Killed softgate at ep3 — gate=[0.98 0.02 0...] confirmed collapse; remaining epochs uninformative.
- 2026-06-10: New variant dropmean: remove the gate parameter entirely; per-SAMPLE
  Bernoulli layer dropout (p=0.3, >=1 kept), equal-weight mean over kept subset,
  eval = full mean. Rationale: with no learnable weight there is nothing for the
  objective to bias toward L11; decoder is forced to reconstruct from deep-only subsets.
- 2026-06-10: Per-sample (not per-step) masks — diverse subsets within each batch.

## Open Questions
- [ ] Does dropmean match nogate's full-mean PSNR while actually using deep layers?
- [ ] Post-hoc: decode from deep-only subset (drop L11 at inference) — PSNR gap vs full mean quantifies deep-layer usage (combine is param-free, any subset works with the same ckpt).
- [ ] Is p_drop=0.3 the right strength? (0.5 = more aggressive subset diversity.)

## Stage-2 (DiT) — ready to launch
- Pipeline understood & adapted (2026-06-10): flow matching (x-prediction, logit-normal t,
  shift 8), DDT-XL 875M w/ internal guidance (base head @ depth 8), gmuon lr 2e-4,
  global batch 1024, online encoding of [B,1024,16,16] latents from frozen stage-1.
- New: `src/stage1/rae_variants.py` (RAEMLSBaseline / RAEProjected — local decoders
  output ImageNet-normalized space so decode() de-normalizes; SIGReg variants' latent
  is the EMA projector output), 3 configs `imagenet-dinov3l-k7-{raev2mls,nogate-sigreg,
  dropmean-sigreg}.yaml`, `scripts/stage1/compute_latent_stats.py`,
  `run_train_dit.sh <variant>` (auto-computes stats), `run_all_dits.sh` queue.
- Sanity-verified in container: encode→decode roundtrip ~24 dB both wrappers; nogate
  latent already ~N(0,1) (mean -0.002, std 1.020) vs raev2 std 1.32 → SIGReg removes
  the need for stats normalization (still applied for safety/parity).
- eval (FID) disabled in configs: no val split / reference npz on this box yet.
- Stage-2 progress metric (2026-06-10): per-epoch DENOISE PROBE — pixel-space PSNR of
  EMA x-prediction at t∈{.25,.5,.75,.95} on fixed images+noise (identical across runs,
  seeded loader), + stage-1 decode ceiling. Cross-run comparable (latent losses are
  not). In stage2/engine.py + stage2/utils.denoise_probe; plot_dit_progress.py compares
  the 3 logs. SIGReg advantage, if real, should show first at large t.

## L11-only control (shallowest layer, decoder + DiT) — built, not launched
- 2026-06-11: control for "are the deeper layers / multi-layer mix actually needed?"
  Same raev2 recipe end to end, layers=[11] only (dinov3mls combine on one layer =
  L11 tokens + L11 token-mean surrogate). Records decoder val PSNR, DiT denoise
  probes, and generation FID.
- New: `run_train_decoder_mls_l11.sh` (stage-1, reuses train_decoder_mls.py --layers 11,
  -> output_full/train_decoder_mls_l11), config `imagenet-dinov3l-l11-raev2mls.yaml`,
  `l11` variant in run_train_dit.sh (EXPERIMENT_NAME=dit-l11), `src/eval_fid_dit.py`
  (offline generation FID: N class-balanced EMA samples vs N real train images,
  torch-fidelity, fixed seed — use same N/seed across variants for comparison),
  master `run_l11_ablation.sh` (decoder -> DiT -> FID -> summary; auto-resumes).
- Queue AFTER the current 3-run DiT queue finishes (uses all 8 GPUs).

## E2E joint training (projector + decoder + DiT) — built, queued behind baselines
- 2026-06-10: `src/train_e2e_sigreg_dit.py` + `run_train_e2e_sigreg_dit.sh`. User's
  design: NO stop-grad / NO EMA target — SIGReg pins latent geometry, recon pins
  information; generation gradients reach the projector via (a) FM target, (b) xt
  interpolation, (c) L_pix = d(dec(zhat), dec(z)) [target = reconstruction, NOT GT].
  L = w_rec·(L1+LPIPS)(dec(z),GT) + SIGReg(z) + w_fm·|zhat-z|²/t² (+IG base head)
      + w_pix·(L1+LPIPS)(dec(zhat), dec(z))
- Safety valves (flags, default OFF): --detach-fm-target / --detach-xt /
  --detach-pix-target / --pix-t-weight. Collapse alarm = falling per-epoch Val PSNR.
- Verified in container: full backward grads proj=1564/dec=4165/dit=1025; isolated
  pixel-FM path alone delivers projector grad 420.8 (>0, the design's core claim);
  Euler sampler OK. Probe line formats match plot_val_psnr.py + plot_dit_progress.py.
- Warm-start: nogate stage-1 by default (INIT_STAGE1 var; switch to dropmean when done).
  DiT from scratch (INIT_DIT optional). 10 ep, batch 24/GPU, AdamW(pd)+gmuon(DiT).

## Next Steps
- [ ] Monitor dropmean stage-1 (ep2-5 w/ layer-usage probe): `RAEv2/output_full/train_decoder_mls_dropmean_sigreg/train.log`
- [ ] Compare stage-1 val PSNR + LOO/solo profiles across runs.
- [ ] Launch DiT queue after dropmean stage-1 finishes: `bash run_all_dits.sh` (or per-variant `run_train_dit.sh`).
- [ ] FID eval setup (val split + reference npz) for stage-2.
