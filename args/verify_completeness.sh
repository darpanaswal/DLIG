#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="verify_completeness"
N_GPUS=1
WALLTIME="24:00:00"

# --- Model selection ---
FAMILY="diffugpt"                            # dream | diffugpt
MODEL_PATH="models/diffugpt-m-prosqa"        # override checkpoint dir; leave "" for the
                                              # family default (DREAM_PATH/GPT_PATH)

# --- Python Script Arguments ---
TORCH_DTYPE="float32"

PROMPT="Explain how photosynthesis works."
SYSTEM_PROMPT="You are a helpful assistant."

GEN_STEPS=12                                 # paper's stated setting for this check
MAX_NEW_TOKENS=64

INTEGRATION_BATCH_SIZE=5
RTOL="1e-2"
ATOL="5e-3"
SEED=0                                       # fixed seed -> identical trajectory
                                             # across all invocations -> grid consistent

# --- Full grid (analyzed config) ---
GRID_LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22)
GRID_STEPS=(1 3 5 7 9 11)
GRID_M_LIST=(200 1000)                        # per-cell convergence + smaller abs err at max m
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
echo "Model Path        : ${MODEL_PATH:-[family default]}"
echo "Dtype             : ${TORCH_DTYPE}"
echo "Gen Steps         : ${GEN_STEPS}"
echo "Max New Tokens    : ${MAX_NEW_TOKENS}"
echo "Integ Batch Size  : ${INTEGRATION_BATCH_SIZE}"
echo "rtol              : ${RTOL}"
echo "atol              : ${ATOL}"
echo "Grid              : layers ${GRID_LAYERS[*]} x steps ${GRID_STEPS[*]}, m_list ${GRID_M_LIST[*]}"
echo "Log dir           : ${LOG_DIR}"
echo "----------------------------------------"

# Common args shared by both phases
COMMON=(
    python -u -m experiments.theorems.verify_completeness
    --family "${FAMILY}"
    --torch_dtype "${TORCH_DTYPE}"
    --prompt "${PROMPT}"
    --system "${SYSTEM_PROMPT}"
    --generation_steps "${GEN_STEPS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --integration_batch_size "${INTEGRATION_BATCH_SIZE}"
    --rtol "${RTOL}"
    --atol "${ATOL}"
    --seed "${SEED}"
    --score_mode meancentered
)
if [ -n "${MODEL_PATH:-}" ]; then
    COMMON+=(--model_path "${MODEL_PATH}")
fi

# --- Grid, one invocation per layer (all steps internal) ---
for LAYER in "${GRID_LAYERS[@]}"; do
    GRID_LOG="${LOG_DIR}/grid_layer${LAYER}.txt"
    echo "[GRID] layer ${LAYER} -> ${GRID_LOG}"
    "${COMMON[@]}" \
        --layer "${LAYER}" \
        --check_steps "${GRID_STEPS[@]}" \
        --m_list "${GRID_M_LIST[@]}" \
        > "${GRID_LOG}" 2>&1
done

# --- Summary: verdict lines from every log, plus max rel_err over the grid ---
echo ""
echo "########################################"
echo "SUMMARY"
echo "########################################"
for LAYER in "${GRID_LAYERS[@]}"; do
    echo "[grid: layer ${LAYER}]"
    grep -E "step [0-9]+: transparency" "${LOG_DIR}/grid_layer${LAYER}.txt" || true
done
echo ""
echo "[max abs_err across grid (paper number)]"
grep -hE "step [0-9]+: transparency" "${LOG_DIR}"/grid_layer*.txt \
    | grep -oE "abs_err=[0-9.e+-]+" \
    | cut -d= -f2 \
    | sort -g \
    | tail -1
echo "[max rel_err across grid]"
grep -hE "step [0-9]+: transparency" "${LOG_DIR}"/grid_layer*.txt \
    | grep -oE "rel_err=[0-9.e+-]+" \
    | cut -d= -f2 \
    | sort -g \
    | tail -1
echo "[convergence failures across grid, if any]"
grep -hE "step [0-9]+: transparency" "${LOG_DIR}"/grid_layer*.txt \
    | grep "conv=FAIL" || echo "none"