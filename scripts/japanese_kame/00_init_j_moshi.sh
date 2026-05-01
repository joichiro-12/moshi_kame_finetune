#!/usr/bin/env bash
# Initialize J-Moshi (nu-dialogue/j-moshi-ext) for KAME fine-tuning.
# Adds the oracle stream (4th stream) by backfilling oracle_emb from text_emb.
# J-Moshi already uses rinna/japanese-gpt2-medium SentencePiece, so text embeddings
# are preserved as-is (no --init_text_embeddings).
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-init_models/j-moshi-ext-four_streams-bfloat16}"
MOSHI_REPO="${MOSHI_REPO:-nu-dialogue/j-moshi-ext}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

echo "=== Initializing J-Moshi for KAME fine-tuning ==="
echo "  Source repo : ${MOSHI_REPO}"
echo "  Save dir    : ${SAVE_DIR}"
echo "  Dtype       : ${MODEL_DTYPE}"

uv run -m tools.init_moshi_for_ft \
    --moshi_lm_repo "${MOSHI_REPO}" \
    --save_dir "${SAVE_DIR}" \
    --model_dtype "${MODEL_DTYPE}"

echo "=== Done. Init model saved to ${SAVE_DIR} ==="
