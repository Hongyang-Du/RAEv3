#!/usr/bin/env bash
# Manually upload the final decoder ckpts to S3, to be run AFTER training finishes
# (and after refreshing AWS creds in ~/.aws/credentials [raev3]). Safe to re-run.
#
#   bash upload_final_ckpts.sh
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export AWS_PROFILE=raev3
S3_DEST=s3://hongyang-du/raev3_decoders
S3_REGION=ap-southeast-2

declare -A OUTDIRS=(
  [plain]=output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep_gan8
  [sigreg]=output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23
)

echo "=== Verifying AWS creds ==="
aws sts get-caller-identity --query Arn --output text || {
    echo "AWS creds invalid/expired — refresh ~/.aws/credentials [raev3] and re-run."; exit 1; }

for name in "${!OUTDIRS[@]}"; do
    out=${OUTDIRS[$name]}
    echo "=== ${name} (${out}) ==="
    if [ ! -f "${out}/ckpt_latest.pt" ]; then
        echo "  no ckpt_latest.pt yet — skipping (training may be unfinished)"
        continue
    fi
    aws s3 cp "${out}/ckpt_latest.pt" "${S3_DEST}/${name}/ckpt_latest.pt" --region "${S3_REGION}" \
        && echo "  -> ${S3_DEST}/${name}/ckpt_latest.pt"
    # upload every epoch-archive ckpt too (provenance)
    for ep in $(ls -1 ${out}/ckpt_ep*.pt 2>/dev/null | sort); do
        aws s3 cp "${ep}" "${S3_DEST}/${name}/$(basename ${ep})" --region "${S3_REGION}" \
            && echo "  -> ${S3_DEST}/${name}/$(basename ${ep})"
    done
done

echo "=== Done. Listing S3 dest ==="
aws s3 ls "${S3_DEST}/" --recursive --region "${S3_REGION}" --human-readable
