#!/usr/bin/env bash
# Upload the precomputed depthattn-k23-nano-p03 latent cache from local SSD to S3,
# to be run AFTER scripts/stage1/precompute_latents.py finishes (check for
# <out>/train/manifest.json first -- its "num_samples" should match the full
# ImageNet train set, 1281167). Safe to re-run (sync is incremental).
#
#   bash upload_latents_depthattn.sh
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export AWS_PROFILE=raev3
LOCAL_DIR=/mnt/localssd/latents-depthattn-k23-nano-p03
S3_DEST=s3://hongyangd-raev2-backup/raev2-data/latents-depthattn-k23-nano-p03
S3_REGION=us-west-2

MANIFEST="${LOCAL_DIR}/train/manifest.json"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST -- precompute_latents.py hasn't finished (or failed)"; exit 1; }

echo "=== Verifying AWS creds ==="
aws sts get-caller-identity --query Arn --output text --region "${S3_REGION}" || {
    echo "AWS creds invalid/expired -- refresh ~/.aws/credentials [raev3] and re-run."; exit 1; }

echo "=== manifest summary ==="
python3 -c "import json; m=json.load(open('${MANIFEST}')); print('num_samples=',m['num_samples'],'shards=',len(m['shards']),'dtype=',m['dtype'])"

echo "=== syncing ${LOCAL_DIR} -> ${S3_DEST} ==="
aws s3 sync "${LOCAL_DIR}" "${S3_DEST}" --region "${S3_REGION}" --only-show-errors

echo "=== done. remote listing (top level) ==="
aws s3 ls "${S3_DEST}/train/" --region "${S3_REGION}" --summarize --human-readable | tail -5
