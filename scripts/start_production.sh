#!/usr/bin/env bash
# ==============================================================================
# Enterprise Knowledge Navigator (EKN) — Production Startup Script (POSIX)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "======================================================================"
echo " Starting Enterprise Knowledge Navigator Production Services"
echo " Project Root: ${PROJECT_ROOT}"
echo "======================================================================"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Check virtual environment
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

python3 "${PROJECT_ROOT}/scripts/start_production.py" "$@"
