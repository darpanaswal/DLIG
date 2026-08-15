#!/bin/bash
set -euo pipefail

WALLTIME="00:50:00"
LOG_DIR="runs/diagnose"
LOG_FILE="${LOG_DIR}/diagnose.log"

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

# Resolve repository/script directory so paths don't depend on cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# echo "Running on $(hostname)"
# echo "Repository: $(pwd)"

# git status

python3 diagnose.py --eval_json outputs/wic/eval.json