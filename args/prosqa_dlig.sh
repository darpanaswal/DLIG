#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="prosqa_dlig"
N_GPUS=3
WALLTIME="12:00:00"
HOST="lig-gpu9.imag.fr"                 # attribution host (dlig venv)
VENV="dlig/bin/activate"                # switch to diffu/bin/activate if running on lig-gpu2

# --- Model / data ---
MODEL_PATH="models/diffugpt-m-prosqa"   # finetuned checkpoint dir
BUCKETS="outputs/prosqa/prosqa_buckets_full.jsonl"

# --- Groups ---
DLIG_GROUPS=(success fail)                   # off dropped (paper filtering)
N_PER_GROUP=-1                          # -1 => all (286 + 69)

# DLIG Hyperparameters (paper setting)
M_STEPS=12
CHUNK=12
GEN_STEPS=64
MAX_NEW_TOKENS=64
SEED=42

# Multi-value args
TARGET_STEPS=(1 3 5 7 9 11)
LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22)

OUT_DIR="outputs/prosqa"
GRAPH="${OUT_DIR}/prosqa_graph_labels.jsonl"
OUT_DLIG="${OUT_DIR}/prosqa_dlig.jsonl"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}.txt"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# --- OAR self-submit ---
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='${HOST}'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "bash ${SCRIPT_PATH}"
    exit 0
fi

source "${VENV}"
mkdir -p "${OUT_DIR}"

> "${LOG_FILE}"
echo "Experiment        : ${EXPERIMENT}"
echo "Model Path        : ${MODEL_PATH}"
echo "Buckets           : ${BUCKETS}"
echo "Groups            : ${DLIG_GROUPS[*]} (n_per_group=${N_PER_GROUP})"
echo "GPUs              : ${N_GPUS} (Sharded execution)"
echo "Integration (M)   : ${M_STEPS}"
echo "Chunk Size        : ${CHUNK}"
echo "Gen Steps         : ${GEN_STEPS}"
echo "Target Steps      : ${TARGET_STEPS[*]}"
echo "Layers            : ${LAYERS[*]}"
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

# ---- Step 1: graph labels (CPU, seconds, idempotent) ----
echo "[$(date +'%F %T')] [STEP 1] Graph labels -> ${GRAPH}" | tee -a "${LOG_FILE}"
python -u -m experiments.prosqa.prosqa_graph_labels \
    --buckets "${BUCKETS}" \
    --out_file "${GRAPH}"

# ---- Step 2: sharded contrastive DLIG ----
echo "[$(date +'%F %T')] [STEP 2] Starting DLIG shards..." | tee -a "${LOG_FILE}"
BASE_DLIG="${OUT_DLIG%.jsonl}"
PIDS=()
for i in $(seq 0 $((N_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u -m experiments.prosqa.prosqa_contrastive_dlig \
        --graph_labels "${GRAPH}" \
        --out_file "${OUT_DLIG}" \
        --model_path "${MODEL_PATH}" \
        --groups "${DLIG_GROUPS[@]}" \
        --n_per_group "${N_PER_GROUP}" \
        --m "${M_STEPS}" \
        --chunk "${CHUNK}" \
        --gen_steps "${GEN_STEPS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --target_steps "${TARGET_STEPS[@]}" \
        --layers "${LAYERS[@]}" \
        --num_shards $N_GPUS \
        --shard_id $i \
        > "${LOG_DIR}/dlig_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "[$(date +'%F %T')] [STEP 2] Combining shards -> ${OUT_DLIG}" | tee -a "${LOG_FILE}"
cat "${BASE_DLIG}_shard"*.jsonl >> "${OUT_DLIG}"
rm "${BASE_DLIG}_shard"*.jsonl

# ---- Step 3: analysis A-D (CPU) ----
echo "[$(date +'%F %T')] [STEP 3] Analysis..." | tee -a "${LOG_FILE}"
python -u -m helpers.analyze_prosqa \
    --dlig_file "${OUT_DLIG}" \
    --graph_labels "${GRAPH}" \
    --out_dir "${OUT_DIR}"

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "DONE. Outputs:" | tee -a "${LOG_FILE}"
echo "  DLIG    : ${OUT_DLIG}" | tee -a "${LOG_FILE}"
echo "  Report  : ${OUT_DIR}/prosqa_dlig_report.txt" | tee -a "${LOG_FILE}"
echo "  Figures : ${OUT_DIR}/{A_path_precision,B_failure_faithfulness,C_mass_predicts_correctness,D_hops_and_localization}.png" | tee -a "${LOG_FILE}"