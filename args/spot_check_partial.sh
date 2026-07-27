#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="spot_check_partial"
N_GPUS=1          # single-GPU: no DDP, builds one DLIGAttribution
WALLTIME="02:00:00"

# --- Model selection ---
FAMILY="diffugpt"                            # dream | diffugpt
MODEL_PATH="models/diffugpt-m-prosqa"        # override checkpoint dir; leave "" for the
                                              # family default (DREAM_PATH/GPT_PATH)
TORCH_DTYPE="bfloat16"                       # MATCH production dtype

LAYERS=(0 6 11)                              # GPT2-medium has 24 layers; span shallow->deep
STEP=2
M=12
CHUNK=12
GEN_STEPS=8
MAX_NEW_TOKENS=64
PROMPT="Explain how photosynthesis works."
SYSTEM_PROMPT="You are a helpful assistant."
SCORE_MODE="logprob"
SEED=0

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='lig-gpu8.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

# Inside OAR job → run experiment
source dlig/bin/activate

> "${LOG_FILE}"
echo "Experiment        : ${EXPERIMENT}"
echo "GPUs              : ${N_GPUS}"
echo "Family            : ${FAMILY}"
echo "Model Path        : ${MODEL_PATH:-[family default]}"
echo "Dtype             : ${TORCH_DTYPE}"
echo "Layers            : ${LAYERS[*]}"
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

# Build the base command array
CMD=(
    python -u -m experiments.theorems.spot_check_partial
    --family "${FAMILY}"
    --torch_dtype "${TORCH_DTYPE}"
)

if [ -n "${MODEL_PATH:-}" ]; then
    CMD+=(--model_path "${MODEL_PATH}")
fi

CMD+=(
    --layers "${LAYERS[@]}"
    --step "${STEP}"
    --m "${M}"
    --chunk "${CHUNK}"
    --gen_steps "${GEN_STEPS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --prompt "${PROMPT}"
    --system "${SYSTEM_PROMPT}"
    --score_mode "${SCORE_MODE}"
    --seed "${SEED}"
)

# Execute and pipe to log file
"${CMD[@]}" > "${LOG_FILE}" 2>&1