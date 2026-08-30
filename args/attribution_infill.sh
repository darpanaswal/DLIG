#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE RUN PROPERTIES
########################################
TARGET_MODE="gold"                    # self (primary) | gold (appendix robustness)
EXPERIMENT="run_infill_attribution_${TARGET_MODE}"
N_GPUS=4
WALLTIME="12:00:00"
FAMILY="diffugpt"  # Swap to 'dream' later

# --- Data Configuration ---
N_SAMPLES=1000                        # paper uses first 1000 ROCStories cases
DATASET="data/rocstories_test.jsonl"  # local JSONL (run download_rocstories.py first)
                                      # OR an HF id, e.g. "Ximing/ROCStories"
MAX_SIDE_TOKENS=120                   # cap tokens kept per side for the position axis

# DLIG Hyperparameters
M_STEPS=8
CHUNK=12
GEN_STEPS=12                          # diffusion denoising steps T for the span
SEED=42
SCORE_MODE="meancentered"             # n_t-normalized, comparable across timesteps

TARGET_STEPS=(1 3 5 7 9 11)           # which recorded denoising steps to attribute at
LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22)

OUT_FILE="outputs/infill_attribution/diffugpt_${TARGET_MODE}.jsonl"
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
mkdir -p "$(dirname "${OUT_FILE}")"

echo "Experiment        : ${EXPERIMENT}"
echo "Family            : ${FAMILY}"
echo "Dataset           : ${DATASET}"
echo "Total Samples     : ${N_SAMPLES}"
echo "Integration (M)   : ${M_STEPS}"
echo "Gen Steps (T)     : ${GEN_STEPS}"
echo "Target Steps      : ${TARGET_STEPS[*]}"
echo "Score Mode        : ${SCORE_MODE}"
echo "Target Mode       : ${TARGET_MODE}"
echo "Layers            : ${LAYERS[*]}"
echo "----------------------------------------"

COMMON_ARGS=(
    --family "${FAMILY}"
    --dataset "${DATASET}"
    --n_samples "${N_SAMPLES}"
    --max_side_tokens "${MAX_SIDE_TOKENS}"
    --m "${M_STEPS}"
    --chunk "${CHUNK}"
    --gen_steps "${GEN_STEPS}"
    --seed "${SEED}"
    --score_mode "${SCORE_MODE}"
    --target_mode "${TARGET_MODE}"
    --target_steps "${TARGET_STEPS[@]}"
    --layers "${LAYERS[@]}"
)

BASE_NAME="${OUT_FILE%.jsonl}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Executing ROCStories Infilling Attribution Across Shards..."
PIDS=()
for i in $(seq 0 $((N_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u -m experiments.infill.attribution_infill \
        "${COMMON_ARGS[@]}" \
        --num_shards $N_GPUS \
        --shard_id $i \
        --out_file "${OUT_FILE}" \
        > "${LOG_DIR}/infill_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Consolidation Phase -> Gathering shards into unified log..."
cat "${BASE_NAME}_shard"*.jsonl >> "${OUT_FILE}"
python -c "
import json, sys
path = '${OUT_FILE}'
seen, rows = set(), []
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        sid = json.loads(line)['story_id']
        if sid not in seen:
            seen.add(sid)
            rows.append(line)
with open(path, 'w') as f:
    f.writelines(rows)
print(f'[MERGE] {len(rows)} unique stories in {path}')
"
rm "${BASE_NAME}_shard"*.jsonl

echo "----------------------------------------"
echo "PROCESS COMPLETED SUCCESSFULLY."
echo "Destination Record: ${OUT_FILE}"