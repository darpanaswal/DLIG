#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="verify_completeness"
N_GPUS=1
WALLTIME="02:00:00"

# --- Model selection ---
FAMILY="diffugpt"                            # dream | diffugpt

# --- Python Script Arguments ---
TORCH_DTYPE="float32"

LAYER="6"                                    # GPT2-medium has 24 layers; pick mid-stack
PROMPT="Explain how photosynthesis works."
SYSTEM_PROMPT="You are a helpful assistant."

GEN_STEPS=8
MAX_NEW_TOKENS=64

INTEGRATION_BATCH_SIZE=5
RTOL="1e-2"
SEED=0

CHECK_STEPS=(7)
M_LIST=(50 100 200 500 1000)                 # convergence sweep; rel_err must shrink
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}.txt"

if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='lig-gpu4.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

source dlig/bin/activate

> "${LOG_FILE}"
echo "Experiment        : ${EXPERIMENT}"
echo "Family            : ${FAMILY}"
echo "Dtype             : ${TORCH_DTYPE}"
echo "Layer             : ${LAYER}"
echo "Gen Steps         : ${GEN_STEPS}"
echo "Max New Tokens    : ${MAX_NEW_TOKENS}"
echo "Integ Batch Size  : ${INTEGRATION_BATCH_SIZE}"
echo "rtol              : ${RTOL}"
echo "Check Steps       : ${CHECK_STEPS[*]:-[last recorded]}"
echo "m_list            : ${M_LIST[*]}"
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

CMD=(
    python -u -m experiments.theorems.verify_completeness
    --family "${FAMILY}"
    --torch_dtype "${TORCH_DTYPE}"
    --layer "${LAYER}"
    --prompt "${PROMPT}"
    --system "${SYSTEM_PROMPT}"
    --generation_steps "${GEN_STEPS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --integration_batch_size "${INTEGRATION_BATCH_SIZE}"
    --rtol "${RTOL}"
    --seed "${SEED}"
    --score_mode meancentered
)

if [ "${#CHECK_STEPS[@]}" -gt 0 ]; then
    CMD+=(--check_steps "${CHECK_STEPS[@]}")
fi

CMD+=(--m_list "${M_LIST[@]}")

"${CMD[@]}" > "${LOG_FILE}" 2>&1