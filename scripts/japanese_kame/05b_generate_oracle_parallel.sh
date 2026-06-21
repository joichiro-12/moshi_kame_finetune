#!/usr/bin/env bash
# Launch N parallel sharded workers for oracle generation against local vLLM.
# Each worker handles a disjoint slice of the input files (stride sharding),
# so they never collide on output files. All workers share --resume, so the
# job is safely restartable.
#
# Usage:
#   bash scripts/japanese_kame/05b_generate_oracle_parallel.sh
#   NUM_SHARDS=64 bash scripts/japanese_kame/05b_generate_oracle_parallel.sh
set -euo pipefail

TEXT_DIR="${TEXT_DIR:-data/japanese_kame/0610/text}"
OUTPUT_DIR="${OUTPUT_DIR:-data/japanese_kame/0610/oracle_raw}"
NUM_SHARDS="${NUM_SHARDS:-32}"
TIME_INTERVAL="${TIME_INTERVAL:-0.5}"
# Comma-separated list of vLLM endpoints. Shards are round-robin assigned to
# them, so multiple GPUs (one server each) share the load.
LLM_BASE_URLS="${LLM_BASE_URLS:-http://localhost:8000/v1,http://localhost:8001/v1}"
LOG_DIR="${LOG_DIR:-logs/oracle_0610}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# `read` returns non-zero at EOF (no trailing newline); guard against set -e.
IFS=',' read -r -a URLS <<< "${LLM_BASE_URLS}" || true
NUM_URLS=${#URLS[@]}

echo "=== Launching ${NUM_SHARDS} parallel oracle workers ==="
echo "  Text dir   : ${TEXT_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Endpoints  : ${LLM_BASE_URLS} (${NUM_URLS})"
echo "  Interval   : ${TIME_INTERVAL}"
echo "  Logs       : ${LOG_DIR}/shard_*.log"

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    url="${URLS[$((shard_id % NUM_URLS))]}"
    nohup uv run --extra oracle -m tools.generate_oracle_from_text \
        --text_dir "${TEXT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --language ja \
        --time_interval "${TIME_INTERVAL}" \
        --llm_base_url "${url}" \
        --num_shards "${NUM_SHARDS}" \
        --shard_id "${shard_id}" \
        --fallback_to_hint_on_error \
        --resume \
        > "${LOG_DIR}/shard_${shard_id}.log" 2>&1 &
    echo "  started shard ${shard_id} -> ${url} (PID $!)"
done

echo "=== All ${NUM_SHARDS} workers launched. ==="
echo "Monitor:  ls ${OUTPUT_DIR}/*.json | wc -l   (target: $(ls ${TEXT_DIR}/*.json | wc -l))"
# Keep this launcher alive so it (and its tmux window) holds the worker group.
wait
