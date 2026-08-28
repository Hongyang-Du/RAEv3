#!/usr/bin/env bash
# =============================================================================
#  eupe-b-k11 / siglip2-l-k23 random-layer-drop sweep, S3-portable.
#
#  Same experiment we already ran on DINOv3 (k7/k23 nano decoders +
#  imagenet-dinov3l-omni-randomdrop-fulltarget-plain-*-ditb DiTs), re-run on the two
#  other encoders. Portable version of run_siglip_eupe_oldnorm.sh: ImageNet is staged
#  from an S3 prefix you pass on the command line, and the /sensei-fs-3 paths baked
#  into the 16 configs are rewritten to $CKPT_ROOT / $DATA before launch (originals
#  are never edited; patched copies land in $CKPT_ROOT/_gen_configs/).
#
#  16 jobs, all from existing configs -- nothing new is invented here:
#    dec  x8  stage-1 ViT-XL decoder, 16 ep, per-sample random layer drop
#             p in {0, 0.3, 0.6, 0.9}, cls_surrogate on
#             configs/stage1/decoder/random-drop-layer-mls-plain-<enc>-nano-p*-oldnorm.yaml
#    dit  x8  stage-2 DiT-B, 40 ep. Input x_t is built from the layer-DROPPED latent,
#             the flow-matching target is the deterministic FULL-layer mean + cls
#             surrogate  (stage_1.params.drop: true  +  transport.decoupled_full_target:
#             true  +  combine.cls_surrogate: true), p in {0, 0.3, 0.6, 0.9}.
#             p0 row is the clean baseline: drop false, coupled target (z_cond==z_target).
#             configs/stage2/training/imagenet-<enc>-encoder-cls-drop-p*-oldnorm.yaml
#
#  encoders:  eupe   -> eupe-b-k11    (EUPE ViT-B/16, layers 1..11,  latent 768)
#             siglip -> siglip-l-k23  (SigLIP2-L/16-256, layers 1..23, latent 1024)
#
#  Everything the two stages need is fetched before the first job -- S3 first, the
#  official RAEv2 / upstream repos as fallback, so nothing has to be hand-staged:
#    ImageNet-256 arrow   S3_DATA -> hf nanovisionx/RAEv2-data   imagenet-256/
#    val npz (val PSNR)   S3_EVAL -> hf nanovisionx/RAEv2-data   (9.8G, VAL_NPZ=1 only)
#    EUPE-ViT-B.pt        S3_ENC  -> hf nyu-visionx/RAEv2-models encoders/eupe/
#                                 -> hf facebook/EUPE-ViT-B      (last resort)
#    DINO ViT-S/8 (GAN)   S3_ENC  -> hf nyu-visionx/RAEv2-models encoders/dino/
#                                 -> dl.fbaipublicfiles.com/dino (last resort)
#    siglip2-l            hf google/siglip2-large-patch16-256
#    LPIPS vgg + VGG16    taming-transformers / torchvision
#    gram-newton-schulz   pip (the DiT's gmuon optimizer, a RAEv2 pyproject dep)
#
#  ---------------------------------------------------------------------------
#  usage
#    bash run_sweep_eupe_siglip_s3.sh <s3://bucket/prefix/imagenet-256/> [stage] [tag ...]
#    bash run_sweep_eupe_siglip_s3.sh <stage> [tag ...]        # no s3: use data already
#                                                              # staged, else RAEv2-data
#
#    stage : all (default) | data | dec | dit
#    tag   : eupe_p0 eupe_p03 eupe_p06 eupe_p09
#            siglip_p0 siglip_p03 siglip_p06 siglip_p09   (default: all 8)
#
#  examples
#    # everything on this node, detached, logs under $LOGDIR
#    bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/
#    # just stage the data
#    bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/ data
#    # 4 nodes, 16 jobs dealt 4-per-node (run this on each node, only SHARD differs)
#    SHARD=0 NUM_SHARDS=4 bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/
#    SHARD=1 NUM_SHARDS=4 bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/
#    SHARD=2 ... SHARD=3 ...
#    # or hand-pick what runs where
#    bash run_sweep_eupe_siglip_s3.sh dec  eupe_p03 eupe_p0
#    bash run_sweep_eupe_siglip_s3.sh dit  siglip_p03
#    # ONE job across 4 nodes (32 GPUs) instead: same command on every node
#    NNODES=4 NODE_RANK=$RANK MASTER_ADDR=<rank0-host> \
#      bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/ dit siglip_p03
#    # see the plan without running anything
#    DRY=1 bash run_sweep_eupe_siglip_s3.sh s3://my-bucket/imagenet-256/
#
#  env
#    DATA=/datasets/imagenet-256-full   where ImageNet-256 is staged (node-local SSD)
#    CKPT_ROOT=$HOME/rae_ckpt           all decoder / DiT / latent-stats output
#    LOGDIR=$CKPT_ROOT/logs             per-job logs
#    CONDA=<env>                        use <env>/bin/{python,torchrun} (else: PATH)
#    PY= TR=                            explicit python / torchrun binaries
#    NGPU / CUDA_VISIBLE_DEVICES        default: all visible GPUs (recipe assumes 8)
#    SHARD=0 NUM_SHARDS=1               deal the 16-job queue across N independent nodes
#    NNODES=1 NODE_RANK=$RANK           or put ONE job on N nodes (torchrun rendezvous)
#    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
#    DEC_GLOBAL_BATCH=256               8x32, the reference the decoder is kept at when
#                                       NNODES>1 (auto BATCH_SIZE_OVERRIDE)
#    DEC_GRAD_ACCUM=<n>                 decoder grad accumulation -- on a node with fewer
#                                       than 8 GPUs, n = 8/NGPU restores that 256 global
#                                       batch at unchanged per-GPU memory
#    S3_EVAL=s3://.../data_eval/         optional: val npz for the stage-1 val PSNR
#    VAL_NPZ=0                           1 = pull that npz (9.8G) from RAEv2-data instead
#    S3_ENC=s3://.../encoders/           optional: encoder weights (else RAEv2-models)
#    ASSET_DIR=<repo>/pretrained_models  encoder + GAN-discriminator weights land here
#    EUPE_CKPT_DIR=$ASSET_DIR/encoders/eupe
#    DINO_DISC_CKPT=$ASSET_DIR/encoders/dino/dino_vit_small_patch8_224.pth
#    HF_MODELS_REPO=nyu-visionx/RAEv2-models    official weights (ungated)
#    HF_DATA_REPO=nanovisionx/RAEv2-data        official data    (ungated)
#    NUM_STATS_SAMPLES=250000           images used for the latent mean/var
#    STATS_REUSE=1                      copy the encoder's first latent_stats to the
#                                       other p (the eval combine is a plain full mean,
#                                       so the stats are p-independent -- 0 = recompute)
#    SKIP_VIZ=0                         1 = never load the p0.3 decoder for DiT sample viz
#    FAIL_FAST=0                        1 = stop the queue on the first failing job
#    DIT_OPTS="a.b=c ..."               extra dotlist overrides for src/train.py
#    SMOKE=0                            1 = plumbing test, NOT a real run: tiny batches,
#                                       1 epoch, GAN from step 0, few stats samples, and
#                                       every job killed after SMOKE_SECS (rc 124 = pass)
#    SMOKE_SECS=600                     per-job wall clock in SMOKE mode
#    WANDB=0  FG=0  DRY=0  FORCE_SYNC=0
#
#  notes
#    * Both stages auto-resume, so a preempted/failed job is restarted with the exact
#      same command (dec -> <out_dir>/ckpt_latest.pt, dit -> find_resume_checkpoint).
#    * Job order puts p0.3 first per encoder: the DiT configs point their sample-viz
#      decoder at the p0.3 stage-1 ckpt. If it is not there yet the DiT still runs --
#      stage1_ckpt_path is set to null and only the sample images are meaningless.
#    * The recipe is 8x80GB (dec 32 img/GPU, DiT global batch 2048 / accum 1). On fewer
#      GPUs pass GRAD_ACCUM_OVERRIDE=<n> (DiT) / BATCH_SIZE_OVERRIDE=<n> (dec).
#    * 4 nodes: prefer SHARD/NUM_SHARDS (4 independent 8-GPU jobs per node -- no cross-node
#      traffic and the recipe is bit-identical to the DINOv3 runs) over NNODES=4. With
#      NUM_SHARDS=4 the deal is: node0 = both p0.3 (dec+dit), node1 = p0, node2 = p0.6,
#      node3 = p0.9 -- i.e. 2 decoders then 2 DiTs each. NNODES=4 is for finishing ONE
#      run 4x sooner: the DiT keeps its 2048 global batch (64/GPU), and the decoder gets
#      BATCH_SIZE_OVERRIDE=8 so its global batch stays 256. With NNODES>1 every rank
#      stages its own DATA (right for node-local SSD); on a shared FS run the `data`
#      stage once first, then launch with the data already in place.
#    * If the DiT dies in the first minute with
#        torch._dynamo.exc.FailOnRecompileLimitHit ... gram_newton_schulz/muon
#      the gmuon build on that box compiles its Newton-Schulz with fullgraph=True over
#      varying param shapes. Relaunch with TORCHDYNAMO_DISABLE=1 (verified fix, eager
#      optimizer step), or install the gmuon you trained the DINOv3 runs with.
#    * On the DGX box run this inside the project container, not on the host.
#    * Before burning 8x40 ep on a new cluster, check the plumbing first:
#        SMOKE=1 FG=1 NGPU=2 bash run_sweep_eupe_siglip_s3.sh <s3://...> all eupe_p03
#      -- stages the data, pulls both encoders, and runs a few steps of dec + stats +
#      DiT per tag. Every job ends in `rc=124 (SMOKE timeout -> pass)`.
# =============================================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
REPO="$(pwd)"

# ---- args -------------------------------------------------------------------
S3_DATA=""
if [[ "${1:-}" == s3://* ]]; then S3_DATA="$1"; shift; fi
STAGE="${1:-all}"; [[ $# -gt 0 ]] && shift
case "${STAGE}" in
  all|data|dec|dit) ;;
  *) echo "usage: bash $(basename "$0") [s3://bucket/prefix/] <all|data|dec|dit> [tag ...]"; exit 1 ;;
esac
ALL_TAGS=(eupe_p03 eupe_p0 eupe_p06 eupe_p09 siglip_p03 siglip_p0 siglip_p06 siglip_p09)
TAGS=("$@"); [[ ${#TAGS[@]} -eq 0 ]] && TAGS=("${ALL_TAGS[@]}")
for t in "${TAGS[@]}"; do
  [[ " ${ALL_TAGS[*]} " == *" ${t} "* ]] || { echo "unknown tag '${t}'; valid: ${ALL_TAGS[*]}"; exit 1; }
done

tag_enc  () { case "$1" in eupe_*) echo eupe-b-k11 ;; siglip_*) echo siglip-l-k23 ;; esac; }
tag_pcode() { echo "${1#*_}"; }
tag_pval () { case "${1#*_}" in p0) echo 0.0 ;; p03) echo 0.3 ;; p06) echo 0.6 ;; p09) echo 0.9 ;; esac; }

# ---- env --------------------------------------------------------------------
export DATA=${DATA:-/datasets/imagenet-256-full}
export CKPT_ROOT=${CKPT_ROOT:-${HOME}/rae_ckpt}
export GEN_DIR=${GEN_DIR:-${CKPT_ROOT}/_gen_configs}
export LOGDIR=${LOGDIR:-${CKPT_ROOT}/logs}
export NUM_STATS_SAMPLES=${NUM_STATS_SAMPLES:-250000}
export STATS_REUSE=${STATS_REUSE:-1}
export SKIP_VIZ=${SKIP_VIZ:-0}
export FAIL_FAST=${FAIL_FAST:-0}
export FORCE_SYNC=${FORCE_SYNC:-0}
export DRY=${DRY:-0}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export SMOKE=${SMOKE:-0}
export SMOKE_SECS=${SMOKE_SECS:-600}
export DIT_OPTS=${DIT_OPTS:-}

if [[ -n "${CONDA:-}" ]]; then
  PY=${PY:-${CONDA}/bin/python}; TR=${TR:-${CONDA}/bin/torchrun}; unset VIRTUAL_ENV
else
  PY=${PY:-$(command -v python3 || command -v python)}; TR=${TR:-$(command -v torchrun)}
fi
export PY TR
[[ -x "${PY}" ]] || { echo "### FATAL: python not found (set PY= or CONDA=)"; exit 1; }
[[ -x "${TR}" ]] || { echo "### FATAL: torchrun not found (set TR= or CONDA=)"; exit 1; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(${PY} - <<'EOF' 2>/dev/null || echo 0,1,2,3,4,5,6,7
import torch; print(",".join(str(i) for i in range(torch.cuda.device_count())) or "0")
EOF
)}
export NGPU=${NGPU:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')}

# ---- multi-node -------------------------------------------------------------
# Two ways to use N nodes, pick one:
#  (a) SHARD/NUM_SHARDS  -- N independent single-node jobs, each 8 GPUs. The 16-job
#      queue is dealt round-robin, so with NUM_SHARDS=4 every node gets 2 dec + 2 dit
#      and node 0 owns both p0.3 decoders. Recipe untouched -> comparable to DINOv3.
#  (b) NNODES/NODE_RANK  -- ONE job across all nodes (torchrun rendezvous). The DiT
#      recipe is node-count agnostic (training.global_batch_size=2048 is absolute, it
#      just splits: 256/GPU on 8 GPUs -> 64/GPU on 32). The DECODER's batch_size is
#      PER-GPU (32), so on >8 GPUs BATCH_SIZE_OVERRIDE is auto-set to keep the same
#      256 global batch as the 8-GPU reference; export it yourself to opt out.
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export SHARD=${SHARD:-0}
export NUM_SHARDS=${NUM_SHARDS:-1}
export DEC_GLOBAL_BATCH=${DEC_GLOBAL_BATCH:-256}     # 8 GPUs x 32 img, the reference
if [[ "${NNODES}" -gt 1 ]]; then
  TR_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${NGPU} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
  _world=$((NNODES * NGPU))
  if [[ -z "${BATCH_SIZE_OVERRIDE:-}" && $((DEC_GLOBAL_BATCH % _world)) -eq 0 && $((DEC_GLOBAL_BATCH / _world)) -ge 1 ]]; then
    export BATCH_SIZE_OVERRIDE=$((DEC_GLOBAL_BATCH / _world))
  fi
else
  TR_ARGS=""
fi
export TR_ARGS

# SMOKE: shrink every knob the trainers expose so one job is a few minutes of real
# forward/backward instead of a 16/40-epoch run. Nothing here is a training recipe.
if [[ "${SMOKE}" == "1" ]]; then
  export BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE:-4}      # dec: img/GPU
  export EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE:-1}
  export WARMUP_OVERRIDE=${WARMUP_OVERRIDE:-0}
  export DISC_START_OVERRIDE=${DISC_START_OVERRIDE:-0}      # exercise the GAN branch too
  export NUM_STATS_SAMPLES=512
  export CKPT_EVERY_STEPS=${CKPT_EVERY_STEPS:-50}           # prove ckpt_latest/resume works
  DIT_OPTS="training.global_batch_size=$((NGPU * 4)) training.epochs=1 training.log_interval=1 \
            training.checkpoint_interval=1 training.sample_every=100 ${DIT_OPTS}"
  export DIT_OPTS
  RUNNER="timeout -s INT -k 60 ${SMOKE_SECS}"
else
  RUNNER=""
fi
export RUNNER

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCH_HOME=${TORCH_HOME:-${CKPT_ROOT}/.cache/torch}          # facebookresearch_EUPE_*
export HF_HOME=${HF_HOME:-${CKPT_ROOT}/.cache/huggingface}          # google/siglip2-large-patch16-256
export CKPT_EVERY_STEPS=${CKPT_EVERY_STEPS:-2500}                   # mid-epoch ckpt_latest
export CKPT_KEEP_RECENT=${CKPT_KEEP_RECENT:-2}                      # bound disk (40ep/interval2)
export CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-10}
if [[ "${WANDB:-0}" == "1" ]]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-uscgvl} WANDB_PROJECT=${WANDB_PROJECT:-omnirae}
  export WANDB_FLAG=--wandb
else
  export WANDB_MODE=disabled; export WANDB_FLAG=
fi
mkdir -p "${LOGDIR}" "${GEN_DIR}"

log () { echo "##### $(date '+%F %T')  $*"; }
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }
# single node: any free port. multi node: every rank must agree -> fixed MASTER_PORT.
tr_args () { if [[ -n "${TR_ARGS}" ]]; then echo "${TR_ARGS}"; else echo "--nproc_per_node=${NGPU} --master-port=$(freeport)"; fi; }

# ---- 0. asset locations + fetch helpers -------------------------------------
export ASSET_DIR=${ASSET_DIR:-${REPO}/pretrained_models}
export EUPE_CKPT_DIR=${EUPE_CKPT_DIR:-${ASSET_DIR}/encoders/eupe}
export DINO_DISC_CKPT=${DINO_DISC_CKPT:-${ASSET_DIR}/encoders/dino/dino_vit_small_patch8_224.pth}
export HF_MODELS_REPO=${HF_MODELS_REPO:-nyu-visionx/RAEv2-models}
export HF_DATA_REPO=${HF_DATA_REPO:-nanovisionx/RAEv2-data}
DINO_DISC_URL=https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth

# hf_download <repo> <model|dataset> <local_dir> <pattern> [pattern ...]
hf_download () {
  local repo=$1 rtype=$2 dest=$3; shift 3
  ${PY} - "$repo" "$rtype" "$dest" "$@" <<'HFPY'
import sys, os
from huggingface_hub import snapshot_download
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
repo, rtype, dest, patterns = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
snapshot_download(repo_id=repo, repo_type=rtype, local_dir=dest, max_workers=8,
                  allow_patterns=patterns or None, ignore_patterns=[".gitattributes"])
print("hf ok:", repo, "->", dest)
HFPY
}

# S3_ENC is the optional fast path for the encoder weights (same layout as the HF
# RAEv2-models repo: <prefix>/eupe/EUPE-ViT-B.pt, <prefix>/dino/*.pth). Synced once.
s3_enc_sync () {
  [[ -n "${S3_ENC:-}" && "${_S3_ENC_DONE:-0}" == "0" ]] || return 0
  _S3_ENC_DONE=1
  mkdir -p "${ASSET_DIR}/encoders"
  log "staging encoder weights: ${S3_ENC} -> ${ASSET_DIR}/encoders/"
  aws s3 sync "${S3_ENC}" "${ASSET_DIR}/encoders/" --no-progress || log "WARNING: S3_ENC sync failed"
}

# ---- 1. data ----------------------------------------------------------------
stage_data () {
  if [[ "${DRY}" == "1" ]]; then
    log "DRY: stage imagenet-256 (${S3_DATA:-hf ${HF_DATA_REPO}}) -> ${DATA}, symlink data/imagenet-256"
    return 0
  fi
  if [[ -d "${DATA}/imagenet-latents-images" && "${FORCE_SYNC}" != "1" ]]; then
    log "imagenet already staged: ${DATA} ($(du -sh "${DATA}" 2>/dev/null | cut -f1))"
  else
    mkdir -p "${DATA}"
    local free_gb; free_gb=$(df -PBG "${DATA}" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
    [[ -n "${free_gb}" && "${free_gb}" -lt "${MIN_FREE_GB}" ]] && \
      log "WARNING: only ${free_gb}G free on $(df -P "${DATA}" | awk 'NR==2{print $6}') (< ${MIN_FREE_GB}G)"
    if [[ -n "${S3_DATA}" ]]; then
      command -v aws >/dev/null || { echo "### FATAL: aws cli not found"; exit 1; }
      local ok=0
      for try in 1 2 3; do
        log "aws s3 sync (try ${try}/3)  ${S3_DATA} -> ${DATA}"
        aws s3 sync "${S3_DATA}" "${DATA}/" --no-progress && { ok=1; break; }
        log "sync attempt ${try} failed, retrying in 60s"; sleep 60
      done
      [[ "${ok}" == "1" ]] || { echo "### FATAL: S3 sync failed (creds/role on this node?)"; exit 1; }
    else
      # official fallback: the RAEv2 data repo ships the same arrow build + the val npz
      log "no S3 path -> pulling imagenet-256 from ${HF_DATA_REPO} (official RAEv2 data)"
      hf_download "${HF_DATA_REPO}" dataset "${DATA}" "imagenet-256/**" \
        || { echo "### FATAL: hf download of ${HF_DATA_REPO} imagenet-256 failed"; exit 1; }
      [[ -d "${DATA}/imagenet-256/imagenet-latents-images" ]] && \
        ln -sfn "${DATA}/imagenet-256/imagenet-latents-images" "${DATA}/imagenet-latents-images"
    fi
    touch "${DATA}/.SYNC_DONE" 2>/dev/null || true   # marker other launchers wait on
    log "staged: $(du -sh "${DATA}" 2>/dev/null | cut -f1)"
  fi
  [[ -d "${DATA}/imagenet-latents-images" ]] || {
    echo "### FATAL: ${DATA}/imagenet-latents-images missing -- the HF arrow dir the loaders"
    echo "###        expect. Point the S3 prefix at the dir that CONTAINS it."; exit 1; }
  mkdir -p data && ln -sfn "${DATA}" data/imagenet-256 && ln -sfn "${DATA}" data/imagenet-256-full

  # stage-1 val PSNR/SSIM reads data_eval/imagenet-256-val.npz (trainer skips the val
  # block if absent). S3_EVAL -> already-staged copy -> official RAEv2 data repo.
  if [[ ! -f data_eval/imagenet-256-val.npz ]]; then
    mkdir -p data_eval
    [[ -n "${S3_EVAL:-}" ]] && { aws s3 sync "${S3_EVAL}" data_eval/ --no-progress || log "WARNING: S3_EVAL sync failed"; }
    # RAEv2-data ships the same npz but it is 9.8 GB for a val_n=100 readout -> opt-in.
    if [[ ! -f data_eval/imagenet-256-val.npz && "${VAL_NPZ:-0}" == "1" ]]; then
      [[ -f "${ASSET_DIR}/imagenet-256/imagenet-256-val.npz" ]] || \
        hf_download "${HF_DATA_REPO}" dataset "${ASSET_DIR}" "imagenet-256/imagenet-256-val.npz" \
        || log "WARNING: val npz download failed"
      [[ -f "${ASSET_DIR}/imagenet-256/imagenet-256-val.npz" ]] && \
        ln -sfn "${ASSET_DIR}/imagenet-256/imagenet-256-val.npz" data_eval/imagenet-256-val.npz
    fi
  fi
  [[ -f data_eval/imagenet-256-val.npz ]] \
    && log "val npz: $(readlink -f data_eval/imagenet-256-val.npz)" \
    || log "note: no val npz -> stage-1 val PSNR/SSIM skipped (VAL_NPZ=1 pulls the 9.8G one)"
}

# ---- 2. encoders ------------------------------------------------------------
# siglip2-l pulls itself from HF. EUPE does not: eupe_loader.load_eupe() takes the ARCH
# from the pinned github hub ref and the WEIGHTS from ${EUPE_CKPT_DIR}/EUPE-ViT-B.pt
# (Meta's own dl.fbaipublicfiles.com/eupe URL answers 403), so fetch that file from the
# official RAEv2 model repo, or from facebook/EUPE-ViT-B as a last resort.
prep_encoders () {
  [[ "${DRY}" == "1" ]] && { log "DRY: fetch/verify EUPE-ViT-B.pt + siglip2-l"; return 0; }
  local want_eupe=0 want_siglip=0 t
  for t in "${TAGS[@]}"; do [[ "${t}" == eupe_* ]] && want_eupe=1; [[ "${t}" == siglip_* ]] && want_siglip=1; done
  mkdir -p "${TORCH_HOME}" "${HF_HOME}"

  if [[ "${want_eupe}" == "1" ]]; then
    if [[ ! -f "${EUPE_CKPT_DIR}/EUPE-ViT-B.pt" ]]; then
      s3_enc_sync
      [[ -f "${EUPE_CKPT_DIR}/EUPE-ViT-B.pt" ]] || {
        log "fetching EUPE-ViT-B.pt from ${HF_MODELS_REPO} (official)"
        hf_download "${HF_MODELS_REPO}" model "${ASSET_DIR}" "encoders/eupe/EUPE-ViT-B.pt" || true; }
      [[ -f "${EUPE_CKPT_DIR}/EUPE-ViT-B.pt" ]] || {
        log "fetching EUPE-ViT-B.pt from facebook/EUPE-ViT-B (upstream)"
        mkdir -p "${EUPE_CKPT_DIR}"
        hf_download facebook/EUPE-ViT-B model "${EUPE_CKPT_DIR}" "EUPE-ViT-B.pt" || true; }
    fi
    [[ -f "${EUPE_CKPT_DIR}/EUPE-ViT-B.pt" ]] || {
      echo "### FATAL: could not obtain ${EUPE_CKPT_DIR}/EUPE-ViT-B.pt (no S3_ENC, no HF access?)"; exit 1; }
    log "loading EUPE ViT-B/16 (arch from hub ref, weights from ${EUPE_CKPT_DIR})"
    ${PY} -c "import sys;sys.path.insert(0,'src');from encoders.models.eupe_loader import load_eupe;load_eupe('eupe_vitb16');print('eupe ok')" \
      || { echo "### FATAL: EUPE load failed (hub ref unreachable, or bad checkpoint)"; exit 1; }
  fi

  if [[ "${want_siglip}" == "1" ]]; then
    log "loading google/siglip2-large-patch16-256 (HF cache: ${HF_HOME})"
    ${PY} -c "from transformers import SiglipVisionModel as M;M.from_pretrained('google/siglip2-large-patch16-256');print('siglip ok')" \
      || { echo "### FATAL: siglip2 download/load failed"; exit 1; }
  fi
  log "encoders ok"
}

# ---- 2b. stage-1 GAN / LPIPS assets -----------------------------------------
# The GAN turns on at loss.gan.disc_start (epoch 8 here) and needs a DINO ViT-S/8 ckpt;
# LPIPS pulls a vgg ckpt + torchvision VGG16 on first use. Fetch all three up front so
# hour 6 of a 16-epoch run never dies on a download.
prep_assets () {
  [[ "${DRY}" == "1" ]] && { log "DRY: fetch/verify DINO disc ckpt + LPIPS/VGG16"; return 0; }
  if [[ ! -f "${DINO_DISC_CKPT}" ]]; then
    s3_enc_sync
    [[ -f "${DINO_DISC_CKPT}" ]] || {
      log "fetching DINO ViT-S/8 (GAN discriminator) from ${HF_MODELS_REPO} (official)"
      hf_download "${HF_MODELS_REPO}" model "${ASSET_DIR}" "encoders/dino/*" || true; }
    if [[ ! -f "${DINO_DISC_CKPT}" ]]; then
      mkdir -p "$(dirname "${DINO_DISC_CKPT}")"
      log "fetching DINO ViT-S/8 from ${DINO_DISC_URL} (upstream)"
      curl -fL --retry 3 -o "${DINO_DISC_CKPT}" "${DINO_DISC_URL}" \
        || { rm -f "${DINO_DISC_CKPT}"; echo "### FATAL: DINO disc ckpt download failed"; exit 1; }
    fi
  fi
  ${PY} -c "
import sys; sys.path.insert(0,'src')
from stage1.disc.lpips_utils import get_ckpt_path
print('lpips ckpt:', get_ckpt_path('vgg_lpips'))
import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1'); print('vgg16 ok')" \
    || { echo "### FATAL: LPIPS/VGG16 weights unavailable"; exit 1; }
  log "stage-1 assets ok (disc=${DINO_DISC_CKPT})"
}

# ---- 2c. stage-2 optimizer --------------------------------------------------
# The DiT configs use optimizer.type=gmuon -> `from gram_newton_schulz import Muon`
# (a RAEv2 pyproject dependency). Install it now rather than 30s into the first DiT.
prep_gmuon () {
  [[ "${DRY}" == "1" ]] && { log "DRY: verify gram_newton_schulz (gmuon) import"; return 0; }
  ${PY} -c "from gram_newton_schulz import Muon" 2>/dev/null && { log "gmuon ok"; return 0; }
  log "installing gram-newton-schulz (gmuon optimizer)"
  PIP_CONSTRAINT= PIP_CONSTRAINTS= ${PY} -m pip install --no-deps gram-newton-schulz \
    || { echo "### FATAL: gram-newton-schulz install failed (needed by optimizer.type=gmuon)"; exit 1; }
  ${PY} -c "from gram_newton_schulz import Muon; print('gmuon ok')" \
    || { echo "### FATAL: gram_newton_schulz still not importable"; exit 1; }
}

# ---- 3. configs -------------------------------------------------------------
# The 16 configs hardcode exactly two roots: /sensei-fs-3/users/hongyangd/ckpt and
# /datasets/imagenet-256-full. Rewrite both into a copy; leave everything else alone.
gen_cfg () {  # $1 src, $2 dst
  [[ -f "$1" ]] || { echo "### FATAL: config not found: $1"; return 1; }
  sed -e "s|/sensei-fs-3/users/hongyangd/ckpt|${CKPT_ROOT}|g" \
      -e "s|/datasets/imagenet-256-full|${DATA}|g" "$1" > "$2" || return 1
  # loss.gan.disc_ckpt is a dataclass default (a repo-relative path), not a yaml key ->
  # pin it explicitly so the run works from any ASSET_DIR.
  grep -q "disc_ckpt:" "$2" || \
    sed -i "s|^\([[:space:]]*\)disc_weight:|\1disc_ckpt: ${DINO_DISC_CKPT}\n\1disc_weight:|" "$2"
}

# ---- 4. jobs ----------------------------------------------------------------
run_dec () {
  local tag=$1 enc pcode pval cfg out
  enc=$(tag_enc "${tag}"); pcode=$(tag_pcode "${tag}"); pval=$(tag_pval "${tag}")
  cfg=${GEN_DIR}/dec-${enc}-${pcode}-oldnorm.yaml
  out=${CKPT_ROOT}/omni-randomdrop-plain-${enc}-nano-p${pval}-oldnorm
  gen_cfg "configs/stage1/decoder/random-drop-layer-mls-plain-${enc}-nano-${pcode}-oldnorm.yaml" "${cfg}" || return 1
  # fewer GPUs than the 8x32 reference: accumulate to keep the 256-image global batch
  # (same per-GPU memory as the reference, unlike raising batch_size).
  if [[ -n "${DEC_GRAD_ACCUM:-}" ]] && ! grep -q "grad_accum_steps:" "${cfg}"; then
    sed -i "s|^\([[:space:]]*\)batch_size:\(.*\)|\1batch_size:\2\n\1grad_accum_steps: ${DEC_GRAD_ACCUM}|" "${cfg}"
    log "dec grad_accum_steps=${DEC_GRAD_ACCUM} (global batch = 32 x ${NGPU} x ${DEC_GRAD_ACCUM})"
  fi
  mkdir -p "${out}"
  log "START dec ${tag}  p_drop=${pval}  ngpu=${NGPU}  -> ${out}"
  [[ "${DRY}" == "1" ]] && { echo "  DRY: ${TR} $(tr_args) src/train_decoder.py --config ${cfg}"; return 0; }
  [[ -f "${out}/ckpt_latest.pt" ]] && log "resuming from ${out}/ckpt_latest.pt"
  ${RUNNER} ${TR} $(tr_args) src/train_decoder.py --config "${cfg}"
}

run_dit () {
  local tag=$1 enc pcode pval cfg viz stats exp first_stats
  enc=$(tag_enc "${tag}"); pcode=$(tag_pcode "${tag}"); pval=$(tag_pval "${tag}")
  cfg=${GEN_DIR}/dit-${enc}-${pcode}-oldnorm.yaml
  viz=${CKPT_ROOT}/omni-randomdrop-plain-${enc}-nano-p0.3-oldnorm/ckpt_ep016.pt
  stats=${CKPT_ROOT}/dit-drop-${enc}-oldnorm/latent_stats_p${pval}.pt
  exp=dit-b-drop-${enc}-p${pval}-oldnorm
  gen_cfg "configs/stage2/training/imagenet-${enc}-encoder-cls-drop-${pcode}-oldnorm.yaml" "${cfg}" || return 1

  # sample-viz decoder is optional: null it out rather than blocking the DiT on stage 1.
  if [[ "${SKIP_VIZ}" == "1" || ! -f "${viz}" ]]; then
    sed -i -E "s|^([[:space:]]*stage1_ckpt_path:).*|\1 null   # p0.3 decoder unavailable -> sample viz is meaningless|" "${cfg}"
    log "WARNING: ${viz} missing -> stage1_ckpt_path=null (training/latents unaffected, sample images are not)"
  fi

  if [[ ! -f "${stats}" ]]; then
    # The eval combine is a deterministic full-layer mean (+ cls surrogate), independent
    # of p_drop -> one stats file per ENCODER is enough. STATS_REUSE=0 recomputes each.
    first_stats=$(ls "${CKPT_ROOT}/dit-drop-${enc}-oldnorm"/latent_stats_p*.pt 2>/dev/null | head -1)
    mkdir -p "$(dirname "${stats}")"
    if [[ "${STATS_REUSE}" == "1" && -n "${first_stats}" ]]; then
      log "latent stats: reusing ${first_stats} -> ${stats} (p-independent eval combine)"
      [[ "${DRY}" == "1" ]] || cp "${first_stats}" "${stats}"
    else
      log "latent stats (${NUM_STATS_SAMPLES} imgs) -> ${stats}"
      if [[ "${DRY}" == "1" ]]; then
        echo "  DRY: ${TR} $(tr_args) scripts/stage1/compute_latent_stats.py --config ${cfg} ..."
      else
        ${RUNNER} ${TR} $(tr_args) \
          scripts/stage1/compute_latent_stats.py --config "${cfg}" --data-dir "${DATA}" \
          --output-path "${stats}" --num-samples "${NUM_STATS_SAMPLES}" \
          || { echo "### latent stats failed for ${tag}"; return 1; }
      fi
    fi
  else
    log "latent stats present: ${stats}"
  fi

  log "START dit ${tag}  p_drop=${pval}  ngpu=${NGPU}  -> ${CKPT_ROOT}/${exp}"
  [[ "${DRY}" == "1" ]] && { echo "  DRY: EXPERIMENT_NAME=${exp} ${TR} $(tr_args) src/train.py --config ${cfg} --results-dir ${CKPT_ROOT} --precision bf16 ${WANDB_FLAG} ${DIT_OPTS}"; return 0; }
  EXPERIMENT_NAME=${exp} ${RUNNER} ${TR} $(tr_args) \
      src/train.py --config "${cfg}" --results-dir "${CKPT_ROOT}" \
      --precision bf16 ${WANDB_FLAG} ${DIT_OPTS}
}

# ---- 5. queue ---------------------------------------------------------------
main () {
  log "sweep ${STAGE}  tags=[${TAGS[*]}]  ngpu=${NGPU}  data=${DATA}  ckpt=${CKPT_ROOT}"
  [[ "${NUM_SHARDS}" -gt 1 ]] && log "shard ${SHARD}/${NUM_SHARDS} (this node runs its slice of the queue)"
  [[ "${NNODES}" -gt 1 ]] && log "multi-node: ${NNODES} nodes x ${NGPU} gpu, node_rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}${BATCH_SIZE_OVERRIDE:+, dec batch/GPU=${BATCH_SIZE_OVERRIDE} (global ${DEC_GLOBAL_BATCH})}"
  [[ "${NGPU}" != "8" && "${NNODES}" == "1" ]] && log "WARNING: recipe is tuned for 8x80GB; on ${NGPU} GPUs consider GRAD_ACCUM_OVERRIDE / BATCH_SIZE_OVERRIDE"
  stage_data
  [[ "${STAGE}" == "data" ]] && { log "data stage done"; return 0; }
  prep_encoders
  [[ "${STAGE}" == "all" || "${STAGE}" == "dec" ]] && prep_assets
  [[ "${STAGE}" == "all" || "${STAGE}" == "dit" ]] && prep_gmuon

  local jobs=() all=() tag st rc t0 note i
  for tag in "${TAGS[@]}"; do [[ "${STAGE}" == "all" || "${STAGE}" == "dec" ]] && all+=("dec:${tag}"); done
  for tag in "${TAGS[@]}"; do [[ "${STAGE}" == "all" || "${STAGE}" == "dit" ]] && all+=("dit:${tag}"); done
  # round-robin deal: node k takes jobs k, k+NUM_SHARDS, ... (NUM_SHARDS=1 -> all of them)
  for i in "${!all[@]}"; do [[ $((i % NUM_SHARDS)) -eq "${SHARD}" ]] && jobs+=("${all[$i]}"); done

  local summary=()
  for job in "${jobs[@]}"; do
    st=${job%%:*}; tag=${job##*:}; t0=$SECONDS
    if [[ "${st}" == "dec" ]]; then run_dec "${tag}"; else run_dit "${tag}"; fi
    rc=$?
    note=""
    # timeout escalation: 124 = SIGINT honoured, 130/143/137 = INT/TERM/KILL of torchrun
    [[ "${SMOKE}" == "1" && ( "${rc}" == "124" || "${rc}" == "130" || "${rc}" == "137" || "${rc}" == "143" ) ]] \
      && { note=" (SMOKE timeout -> pass)"; rc=0; }
    log "DONE ${st} ${tag} rc=${rc}${note} ($(( (SECONDS-t0)/60 )) min)"
    summary+=("$(printf '%-4s %-11s rc=%-3s %4d min%s' "${st}" "${tag}" "${rc}" $(( (SECONDS-t0)/60 )) "${note}")")
    if [[ "${rc}" -ne 0 && "${FAIL_FAST}" == "1" ]]; then
      log "FAIL_FAST: stopping after ${st} ${tag}"; break
    fi
  done

  echo; log "==== summary ===="
  printf '  %s\n' "${summary[@]}"
  log "ckpts: ${CKPT_ROOT}   logs: ${LOGDIR}   configs: ${GEN_DIR}"
}

LOG=${LOGDIR}/sweep_${STAGE}_$(date +%Y%m%d-%H%M%S).log
if [[ "${FG:-0}" == "1" || "${DRY}" == "1" ]]; then
  main 2>&1 | tee "${LOG}"
else
  # setsid: survive SIGHUP when the launching ssh/VSCode session ends
  setsid nohup bash -c "$(declare -f main stage_data prep_encoders prep_assets prep_gmuon hf_download s3_enc_sync gen_cfg run_dec run_dit log freeport tr_args tag_enc tag_pcode tag_pval)
                        STAGE=${STAGE}; S3_DATA='${S3_DATA}'; TAGS=(${TAGS[*]}); main" \
      < /dev/null > "${LOG}" 2>&1 &
  disown
  echo "launched sweep ${STAGE} [${TAGS[*]}]  pid=$!"
  echo "log: ${LOG}    (tail -f ${LOG})"
fi
