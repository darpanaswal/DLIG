#!/bin/bash
set -euo pipefail

EXPERIMENT="prosqa_dlig"
WALLTIME="00:05:00"
VENV="dlig/bin/activate" 
OUT_DIR="outputs/prosqa"
GRAPH="${OUT_DIR}/prosqa_graph_labels.jsonl"
OUT_DLIG="${OUT_DIR}/prosqa_dlig.jsonl"

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/analyze_only.log"

# Submit if not already inside an OAR job
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"

    oarsub \
        -n "git_push" \
        -l /host=1/core=1,walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"

    exit 0
fi

source "${VENV}"
mkdir -p "${OUT_DIR}"

> "${LOG_FILE}"
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

python -u -m helpers.analyze_prosqa \
    --dlig_file "${OUT_DLIG}" \
    --graph_labels "${GRAPH}" \
    --out_dir "${OUT_DIR}"