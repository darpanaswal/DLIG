#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE RUN PROPERTIES
########################################
EXPERIMENT="run_wic_dlig"
N_GPUS=4
WALLTIME="06:00:00"
HOST="lig-gpu4.imag.fr"

# --- Model / data ---
MODEL_PATH="models/diffugpt-m-wic"        # finetuned WiC checkpoint
WIC_JSONL="data/wic_test_raw.jsonl"       # held-out test set (raw fields)
N=-1                                      # -1 => all rows
ONLY_CORRECT=1                            # 1 => attribute only correct examples

# --- DLIG hyperparameters (paper setting) ---
M_STEPS=12
CHUNK=12
GEN_STEPS=64
MAX_NEW_TOKENS=6
SEED=42
SCORE_MODE="meancentered"
TARGET_STEPS=(1 3 5 7 9 11)
LAYERS=(0 2 4 6 8 10 12 14 16 18 20 22)

OUT_FILE="outputs/wic/wic_dlig.jsonl"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}.txt"

if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='${HOST}'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

source dlig/bin/activate
mkdir -p "$(dirname "${OUT_FILE}")"

echo "Experiment      : ${EXPERIMENT}"
echo "Model Path      : ${MODEL_PATH}"
echo "WiC jsonl       : ${WIC_JSONL}"
echo "Only correct    : ${ONLY_CORRECT}"
echo "Integration (M) : ${M_STEPS}"
echo "Gen Steps (T)   : ${GEN_STEPS}"
echo "Target Steps    : ${TARGET_STEPS[*]}"
echo "Score Mode      : ${SCORE_MODE}"
echo "Layers          : ${LAYERS[*]}"
echo "----------------------------------------"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --wic_jsonl "${WIC_JSONL}"
    --n "${N}"
    --m "${M_STEPS}"
    --chunk "${CHUNK}"
    --gen_steps "${GEN_STEPS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
    --score_mode "${SCORE_MODE}"
    --target_steps "${TARGET_STEPS[@]}"
    --layers "${LAYERS[@]}"
)
if [ "${ONLY_CORRECT}" -eq 1 ]; then
    COMMON_ARGS+=(--only_correct)
fi

BASE_NAME="${OUT_FILE%.jsonl}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running WiC self-generated DLIG across shards..."
PIDS=()
for i in $(seq 0 $((N_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u -m experiments.wic.wic \
        "${COMMON_ARGS[@]}" \
        --num_shards $N_GPUS \
        --shard_id $i \
        --out_file "${OUT_FILE}" \
        > "${LOG_DIR}/wic_dlig_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Consolidation -> merging shards..."
cat "${BASE_NAME}_shard"*.jsonl > "${OUT_FILE}"
rm "${BASE_NAME}_shard"*.jsonl

echo "----------------------------------------"
echo "DONE. Output: ${OUT_FILE}"
echo "Next: plot per-example panels, e.g."
echo "  python -m helpers.analyze_wic --panel --dlig ${OUT_FILE} --panel_pick correct_no --panel_out_file outputs/wic/figs/correct_no.png"
echo "  python -m helpers.analyze_wic --panel --dlig ${OUT_FILE} --panel_pick correct_yes --panel_out_file outputs/wic/figs/correct_yes.png"
echo "  python -m helpers.analyze_wic --panel --dlig ${OUT_FILE} --panel_idx <N> --panel_per_step --panel_out_file outputs/wic/figs/evolution.png"