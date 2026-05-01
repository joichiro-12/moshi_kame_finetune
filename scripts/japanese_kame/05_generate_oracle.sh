#!/usr/bin/env bash
# Generate oracle_raw JSON files from canonical Japanese text transcripts.
# Uses gpt-4.1-mini with Japanese prompts (--language ja).
set -euo pipefail

TEXT_DIR="${TEXT_DIR:-data/japanese_kame/text}"
OUTPUT_DIR="${OUTPUT_DIR:-data/japanese_kame/oracle_raw}"
MODEL="${MODEL:-gpt-4.1-mini}"
TIME_INTERVAL="${TIME_INTERVAL:-0.5}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
EXTRA_ARGS=()

if [ -n "${LLM_BASE_URL}" ]; then
    EXTRA_ARGS+=(--llm_base_url "${LLM_BASE_URL}")
fi

echo "=== Generating Japanese oracle predictions ==="
echo "  Text dir   : ${TEXT_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${MODEL}"

uv run --extra oracle -m tools.generate_oracle_from_text \
    --text_dir "${TEXT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --language ja \
    --model "${MODEL}" \
    --time_interval "${TIME_INTERVAL}" \
    --fallback_to_hint_on_error \
    --resume \
    "${EXTRA_ARGS[@]}"

echo "=== Done. Oracle records written to ${OUTPUT_DIR} ==="
