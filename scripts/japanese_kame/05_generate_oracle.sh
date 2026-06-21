#!/usr/bin/env bash
# Generate oracle_raw JSON files from canonical Japanese text transcripts.
# Defaults to a local vLLM server (http://localhost:8000/v1) with Japanese
# prompts (--language ja). To use OpenAI instead, set MODEL and LLM_BASE_URL,
# e.g. MODEL=gpt-4.1-mini LLM_BASE_URL= bash scripts/japanese_kame/05_generate_oracle.sh
set -euo pipefail

TEXT_DIR="${TEXT_DIR:-data/japanese_kame/0610/text}"
OUTPUT_DIR="${OUTPUT_DIR:-data/japanese_kame/0610/oracle_raw}"
# Empty MODEL -> let the script auto-detect the model served by vLLM.
MODEL="${MODEL:-}"
TIME_INTERVAL="${TIME_INTERVAL:-0.5}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
EXTRA_ARGS=()

if [ -n "${LLM_BASE_URL}" ]; then
    EXTRA_ARGS+=(--llm_base_url "${LLM_BASE_URL}")
fi
if [ -n "${MODEL}" ]; then
    EXTRA_ARGS+=(--model "${MODEL}")
fi

echo "=== Generating Japanese oracle predictions ==="
echo "  Text dir   : ${TEXT_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${MODEL:-<auto-detect from vLLM>}"
echo "  Base URL   : ${LLM_BASE_URL:-<OpenAI default>}"

uv run --extra oracle -m tools.generate_oracle_from_text \
    --text_dir "${TEXT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --language ja \
    --time_interval "${TIME_INTERVAL}" \
    --fallback_to_hint_on_error \
    --resume \
    "${EXTRA_ARGS[@]}"

echo "=== Done. Oracle records written to ${OUTPUT_DIR} ==="
