#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="eval_wic"
N_GPUS=1
WALLTIME="02:00:00"
HOST="lig-gpu4.imag.fr"
VENV="dlig/bin/activate"

MODEL_PATH="models/diffugpt-m-wic"
WIC_JSONL="data/wic_test_raw.jsonl"
N=-1

GEN_STEPS=12
MAX_NEW_TOKENS=6
SEED=42

OUT_DIR="outputs/wic"
OUT_FILE="${OUT_DIR}/eval.json"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}.txt"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

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

########################################
# JOB ENVIRONMENT
########################################

source "${VENV}"

# Prefer the Python interpreter from the activated environment.
if [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -x "${VIRTUAL_ENV:-}/bin/python3" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] No Python interpreter found after activating: ${VENV}" >&2
    echo "[ERROR] VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}" >&2
    echo "[ERROR] PATH=${PATH}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Truncate only after the OAR job has started.
> "${LOG_FILE}"

echo "Experiment     : ${EXPERIMENT}"
echo "Model Path     : ${MODEL_PATH}"
echo "WiC jsonl      : ${WIC_JSONL}"
echo "N rows         : ${N} (-1 => all)"
echo "Gen Steps      : ${GEN_STEPS}"
echo "Max new tokens : ${MAX_NEW_TOKENS}"
echo "Out file       : ${OUT_FILE}"
echo "Log file       : ${LOG_FILE}"
echo "Python         : ${PYTHON}"
echo "Python version : $(${PYTHON} --version 2>&1)"
echo "VIRTUAL_ENV    : ${VIRTUAL_ENV:-<unset>}"
echo "----------------------------------------"

echo "[$(date +'%F %T')] [RUN] WiC accuracy gate" | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0 \
    "${PYTHON}" -u -m experiments.wic.eval_wic \
    --model_path "${MODEL_PATH}" \
    --wic_jsonl "${WIC_JSONL}" \
    --out_file "${OUT_FILE}" \
    --n "${N}" \
    --gen_steps "${GEN_STEPS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "DONE. Eval JSON: ${OUT_FILE}" | tee -a "${LOG_FILE}"
echo "Accuracy printed above; chance = 0.500." | tee -a "${LOG_FILE}"