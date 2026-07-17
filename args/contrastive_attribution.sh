#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="contrastive_attribution"
N_GPUS=4                               # Updated to request 4 GPUs
WALLTIME="12:00:00"

# --- Model selection ---
FAMILY="diffugpt"

# --- Python Script Arguments ---
N_PER_CLASS=250
SYSTEM_PROMPT=""                       
REFUSAL_TARGET="I cannot fulfill this request."

# DLIG Hyperparameters
M_STEPS=8
CHUNK=12
GEN_STEPS=12
SEED=42

# Multi-value args
TARGET_STEPS=(1 5 9 11)
LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22)

OUT_UNFORCED="outputs/contrast/diffugpt_attribution_unforced.jsonl"
OUT_FORCED="outputs/contrast/diffugpt_attribution_forced_refusal.jsonl"
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
echo "GPUs              : ${N_GPUS} (Sharded execution)"
echo "N Per Class       : ${N_PER_CLASS}"
echo "Refusal Target    : ${REFUSAL_TARGET}"
echo "Integration (M)   : ${M_STEPS}"
echo "Chunk Size        : ${CHUNK}"
echo "Gen Steps         : ${GEN_STEPS}"
echo "Target Steps      : ${TARGET_STEPS[*]}"
echo "Layers            : ${LAYERS[*]}"
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

COMMON_ARGS=(
    --family "${FAMILY}"
    --n_per_class "${N_PER_CLASS}"
    --system "${SYSTEM_PROMPT}"
    --m "${M_STEPS}"
    --chunk "${CHUNK}"
    --gen_steps "${GEN_STEPS}"
    --seed "${SEED}"
    --target_steps "${TARGET_STEPS[@]}"
    --layers "${LAYERS[@]}"
)

# Helper variables for concatenation and cleanup
BASE_UNFORCED="${OUT_UNFORCED%.jsonl}"
BASE_FORCED="${OUT_FORCED%.jsonl}"

# ---- Run A: UNFORCED (self-generated target) ----
echo "[$(date +'%Y-%m-%d %H:%M:%S')] [RUN A] Starting Unforced Shards..." | tee -a "${LOG_FILE}"
PIDS=()
for i in $(seq 0 $((N_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u -m experiments.contrastive_attribution \
        "${COMMON_ARGS[@]}" \
        --num_shards $N_GPUS \
        --shard_id $i \
        --out_file "${OUT_UNFORCED}" \
        > "${LOG_DIR}/unforced_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all Unforced shards to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] [RUN A] Combining Unforced Shards -> ${OUT_UNFORCED}" | tee -a "${LOG_FILE}"
cat "${BASE_UNFORCED}_shard"*.jsonl > "${OUT_UNFORCED}"
rm "${BASE_UNFORCED}_shard"*.jsonl


# ---- Run B: FORCED refusal target ----
echo "[$(date +'%Y-%m-%d %H:%M:%S')] [RUN B] Starting Forced Shards..." | tee -a "${LOG_FILE}"
PIDS=()
for i in $(seq 0 $((N_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u -m experiments.contrastive_attribution \
        "${COMMON_ARGS[@]}" \
        --num_shards $N_GPUS \
        --shard_id $i \
        --target "${REFUSAL_TARGET}" \
        --out_file "${OUT_FORCED}" \
        > "${LOG_DIR}/forced_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all Forced shards to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] [RUN B] Combining Forced Shards -> ${OUT_FORCED}" | tee -a "${LOG_FILE}"
cat "${BASE_FORCED}_shard"*.jsonl > "${OUT_FORCED}"
rm "${BASE_FORCED}_shard"*.jsonl

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "DONE. Final Unified Outputs:" | tee -a "${LOG_FILE}"
echo "  Unforced : ${OUT_UNFORCED}" | tee -a "${LOG_FILE}"
echo "  Forced   : ${OUT_FORCED}" | tee -a "${LOG_FILE}"