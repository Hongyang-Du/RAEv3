# Runbook — eupe-b-k11 / siglip2-l-k23 random-layer-drop sweep

Everything is driven by one script, `RAEv2/run_sweep_eupe_siglip_s3.sh`. It stages the
data, fetches every weight it needs, patches the configs for your paths, and runs the
jobs in order. Nothing has to be prepared by hand.

## What it runs

16 jobs, all from configs already in the repo — the same experiment as the DINOv3 runs,
on two other encoders:

| stage | n | what |
|---|---|---|
| `dec` | 8 | stage-1 ViT-XL decoder, 16 ep, per-sample random layer drop `p ∈ {0, 0.3, 0.6, 0.9}`, `cls_surrogate: true` |
| `dit` | 8 | stage-2 DiT-B, 40 ep. `x_t` is built from the layer-**dropped** latent, the flow-matching target is the deterministic **full-layer** mean + cls surrogate (`stage_1.params.drop` + `transport.decoupled_full_target` + `combine.cls_surrogate`), same four `p` |

Encoders: `eupe-b-k11` (EUPE ViT-B/16, layers 1–11, latent 768) and `siglip-l-k23`
(SigLIP2-L/16-256, layers 1–23, latent 1024). The `p0` DiT row is the clean baseline
(`drop: false`, coupled target).

Configs (never edited — patched copies go to `$CKPT_ROOT/_gen_configs/`):
- `configs/stage1/decoder/random-drop-layer-mls-plain-<enc>-nano-p*-oldnorm.yaml`
- `configs/stage2/training/imagenet-<enc>-encoder-cls-drop-p*-oldnorm.yaml`

## Requirements

- 8×80GB per node (the recipe is dec 32 img/GPU, DiT global batch 2048 / accum 1)
- ~25 GB for ImageNet-256 + room for checkpoints under `$CKPT_ROOT`
- Network access to huggingface.co and pypi. An S3 copy of ImageNet is optional —
  without one the script pulls the official `nanovisionx/RAEv2-data` build.

## 0. Code + environment

```bash
git clone -b oldnorm git@github.com:Hongyang-Du/RAEv3.git
cd RAEv3/RAEv2

# in a container, e.g.
#   docker run -d --name rae --gpus all --ipc=host --shm-size=64g \
#     -v /datasets:/datasets -v $PWD/..:/workspace/RAEv3 \
#     nvcr.io/nvidia/pytorch:25.04-py3 sleep infinity
bash setup_rae_env.sh                 # ~15 min, idempotent: /opt/conda/envs/rae
export CONDA=/opt/conda/envs/rae      # or set PY= / TR= yourself
```

`gram-newton-schulz` (the DiT's gmuon optimizer) is installed by the script if missing.

## 1. Smoke test first (~15 min — do not skip)

```bash
SMOKE=1 FG=1 NGPU=2 CKPT_ROOT=/big/disk/rae_ckpt \
  bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/ all eupe_p03 siglip_p03
```

Drop the `s3://...` argument to pull ImageNet from `nanovisionx/RAEv2-data` instead.
This stages the data, downloads every weight, and runs a few minutes of each job with
tiny batches. Expect four lines like `rc=0 ... (SMOKE timeout -> pass)` in the summary.

## 2. Launch

### 4 nodes (recommended): 4 independent 8-GPU jobs

Same command on each node, only `SHARD` differs. The 16-job queue is dealt round-robin,
so each node gets 2 decoders then 2 DiTs, and node 0 owns both `p0.3` decoders.

```bash
SHARD=0 NUM_SHARDS=4 CKPT_ROOT=/shared/rae_ckpt WANDB=1 \
  bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/
# node 1: SHARD=1 ... node 2: SHARD=2 ... node 3: SHARD=3 ...
```

| node | jobs |
|---|---|
| 0 | dec eupe_p03, dec siglip_p03, dit eupe_p03, dit siglip_p03 |
| 1 | the same four at p0 |
| 2 | p0.6 |
| 3 | p0.9 |

Per-job recipe is untouched, so results stay comparable to the DINOv3 sweep.

### One node

```bash
CKPT_ROOT=/big/disk/rae_ckpt bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/
```

Runs all 16 sequentially (detached; add `FG=1` to keep it in the foreground).

### One job spread over N nodes

Use this only to finish a single run sooner:

```bash
NNODES=4 NODE_RANK=$RANK MASTER_ADDR=<rank0-host> \
  bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/ dit siglip_p03
```

The DiT's `global_batch_size: 2048` is absolute, so it just splits (64/GPU on 32 GPUs) —
recipe unchanged. The decoder's `batch_size` is per-GPU, so the script sets
`BATCH_SIZE_OVERRIDE` to keep the 256-image global batch of the 8-GPU reference. With
`NNODES>1` every rank stages its own `DATA` (right for node-local SSD); on a shared
filesystem run the `data` stage once first.

### Subsets

```bash
bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/ data           # stage only
bash run_sweep_eupe_siglip_s3.sh dec  eupe_p03 eupe_p0                      # pick jobs
bash run_sweep_eupe_siglip_s3.sh dit  siglip_p09
DRY=1 bash run_sweep_eupe_siglip_s3.sh s3://<bucket>/imagenet-256/          # print the plan
```

## What it fetches (S3 first, official repos as fallback)

| component | where from |
|---|---|
| ImageNet-256 arrow | `S3_DATA` argument → hf `nanovisionx/RAEv2-data` `imagenet-256/` |
| `EUPE-ViT-B.pt` | `S3_ENC` → hf `nyu-visionx/RAEv2-models` `encoders/eupe/` → hf `facebook/EUPE-ViT-B` |
| DINO ViT-S/8 (stage-1 GAN discriminator, on from epoch 8) | `S3_ENC` → hf `nyu-visionx/RAEv2-models` `encoders/dino/` → `dl.fbaipublicfiles.com/dino` |
| siglip2-l | hf `google/siglip2-large-patch16-256` |
| LPIPS vgg + torchvision VGG16 | taming-transformers / torchvision |
| gmuon | `pip install gram-newton-schulz` |
| val npz (optional) | `S3_EVAL`, or `VAL_NPZ=1` to pull the 9.8 GB one from RAEv2-data |

## Monitoring, resume, outputs

- Logs: `$CKPT_ROOT/logs/sweep_<stage>_<timestamp>.log` (`tail -f`), one summary table at the end
- Resume: re-run the identical command. The decoder resumes from `<out_dir>/ckpt_latest.pt`
  (also written every `CKPT_EVERY_STEPS=2500` steps), the DiT via `find_resume_checkpoint`
- Decoders: `$CKPT_ROOT/omni-randomdrop-plain-<enc>-nano-p<P>-oldnorm/`
- DiTs: `$CKPT_ROOT/dit-b-drop-<enc>-p<P>-oldnorm/`
- Latent stats: `$CKPT_ROOT/dit-drop-<enc>-oldnorm/latent_stats_p<P>.pt`
  (computed once per encoder — the eval combine is a deterministic full-layer mean, so
  the stats do not depend on `p`; `STATS_REUSE=0` recomputes per rate)

## Troubleshooting

- **DiT dies in the first minute** with `torch._dynamo.exc.FailOnRecompileLimitHit` in
  `gram_newton_schulz/muon` → relaunch with `TORCHDYNAMO_DISABLE=1` (verified fix; that
  build compiles its Newton–Schulz with `fullgraph=True` over varying parameter shapes).
- **Fewer than 8 GPUs** → `GRAD_ACCUM_OVERRIDE=<n>` (DiT) / `BATCH_SIZE_OVERRIDE=<n>` (dec).
- **`stage1_ckpt_path=null` warning** on a DiT → the `p0.3` decoder has not reached
  `ckpt_ep016.pt` yet. Training and latents are unaffected, only the sample images are.
- **No val PSNR/SSIM in the stage-1 log** → no `data_eval/imagenet-256-val.npz`. Set
  `S3_EVAL=` or `VAL_NPZ=1`.
- **S3 sync fails** → the node needs credentials/role for that bucket; the script retries
  three times, then exits. Dropping the `s3://` argument falls back to HuggingFace.

Full option list is in the header of `run_sweep_eupe_siglip_s3.sh`.
