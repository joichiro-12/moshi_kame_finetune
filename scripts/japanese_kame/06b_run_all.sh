#!/usr/bin/env bash
# Drive 06b_tokenize_batch.sh over all remaining matched names, in fixed-size
# batches, until none are left. Each batch produces one parquet (easy to rsync
# to ABCI incrementally). Resumable: already-claimed batches are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/japanese_kame/06b_run_all.sh
#   START_ID=1 BATCH_SIZE=10000 MAX_BATCHES=10 bash scripts/japanese_kame/06b_run_all.sh
set -euo pipefail

START_ID="${START_ID:-1}"           # batch_0000 is the validation batch; start at 1
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_BATCHES="${MAX_BATCHES:-20}"     # safety cap
NUM_AUDIO_WORKERS="${NUM_AUDIO_WORKERS:-1}"
HERE="$(dirname "$0")"

for k in $(seq "${START_ID}" $((START_ID + MAX_BATCHES - 1))); do
    echo "######## driver: batch ${k} ########"
    out="$(NUM_AUDIO_WORKERS="${NUM_AUDIO_WORKERS}" BATCH_ID="${k}" BATCH_SIZE="${BATCH_SIZE}" \
        bash "${HERE}/06b_tokenize_batch.sh" 2>&1 | grep -vE 'it/s|UserWarning|_handle_alignment')"
    echo "${out}"
    if echo "${out}" | grep -q "No available matched names left"; then
        echo "######## driver: all matched names consumed. Done. ########"
        break
    fi
done
