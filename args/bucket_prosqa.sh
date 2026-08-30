#!/bin/bash
set -euo pipefail

########################################
EXPERIMENT="bucket_prosqa"
WALLTIME="00:20:00"
MODE="${1:-full}"          # matches eval mode: probe | full

LOG_DIR="runs/train_prosqa_ddmsft"
IN_FILE="data/prosqa_eval_${MODE}.jsonl"
OUT_FILE="outputs/prosqa/prosqa_buckets_${MODE}.jsonl"
########################################

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# --- OAR self-submit (cpu-only, no gpu needed) ---
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}_${MODE}" \
        -p "network_address='lig-gpu2.imag.fr'" \
        -l /host=1,walltime=${WALLTIME} \
        -O "${LOG_DIR}/${EXPERIMENT}_${MODE}.txt" \
        -E "${LOG_DIR}/${EXPERIMENT}_${MODE}.txt" \
        "bash ${SCRIPT_PATH} ${MODE}"
    exit 0
fi

cd "$(dirname "${SCRIPT_PATH}")"
source diffu/bin/activate

echo "[$(date +'%F %T')] bucketing ${IN_FILE}"
python -u -m experiments.prosqa.bucket_prosqa \
    --in_file  "${IN_FILE}" \
    --out_file "${OUT_FILE}"
echo "[$(date +'%F %T')] done -> ${OUT_FILE}"