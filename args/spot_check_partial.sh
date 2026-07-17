#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="spot_check_partial"
N_GPUS=1          # single-GPU: no DDP, builds one DLIGAttribution
WALLTIME="02:00:00"

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
echo "Log file          : ${LOG_FILE}"
echo "----------------------------------------"

# Build the base command array
CMD=(
    python -u -m experiments.theorems.spot_check_partial
)

# Execute and pipe to log file
"${CMD[@]}" > "${LOG_FILE}" 2>&1