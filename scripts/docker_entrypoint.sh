#!/usr/bin/env bash
# ==============================================================================
# Enterprise Knowledge Navigator (EKN) — Container Entrypoint
# ==============================================================================
# Ensures storage directories exist, optimizes volume ownership without
# recursive traversal, and executes the supervisor via gosu if running as root.
# ==============================================================================
set -euo pipefail

STORAGE_ROOT="${STORAGE_LOCAL_ROOT:-/app/data/storage}"

# Ensure required directory structure exists on persistent volume
mkdir -p "${STORAGE_ROOT}/uploads" \
         "${STORAGE_ROOT}/processed" \
         "${STORAGE_ROOT}/reports" \
         "${STORAGE_ROOT}/temp" \
         /app/data/chroma_db

# Configure model cache location to use the build-time baked cache
export HF_HOME="${HF_HOME:-/app/model_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/app/model_cache}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/app/model_cache}"

# Railway mounts persistent volumes as root:root (or when RAILWAY_RUN_UID=0 is set).
# When running as root, initialize volume permissions once at the top-level without
# recursive chown of all stored documents to keep startup instantaneous.
if [ "$(id -u)" = "0" ]; then
    INIT_SENTINEL="${STORAGE_ROOT}/.initialized"
    if [ ! -f "${INIT_SENTINEL}" ]; then
        echo "[EKN Entrypoint] Initializing persistent storage permissions for appuser (UID 1000)..."
        chown appuser:appuser "${STORAGE_ROOT}" \
                              "${STORAGE_ROOT}/uploads" \
                              "${STORAGE_ROOT}/processed" \
                              "${STORAGE_ROOT}/reports" \
                              "${STORAGE_ROOT}/temp" \
                              /app/data/chroma_db
        touch "${INIT_SENTINEL}"
        chown appuser:appuser "${INIT_SENTINEL}"
    fi
    exec gosu appuser "$@"
else
    exec "$@"
fi
