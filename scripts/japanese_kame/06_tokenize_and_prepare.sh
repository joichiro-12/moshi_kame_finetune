#!/usr/bin/env bash
# Full tokenization and dataset assembly pipeline for Japanese KAME data.
# Wraps tools/tokenize_audio.py, tokenize_text.py, tokenize_oracle.py,
# and prepare_dataset.py with the correct Japanese parameters.
#
# Environment variables (all have defaults):
#   AUDIO_DIR, TEXT_DIR, ORACLE_DIR
#   TOKENIZED_AUDIO_DIR, TOKENIZED_TEXT_DIR, TOKENIZED_ORACLE_DIR
#   OUTPUT_PREFIX
#   TEXT_TOKENIZER_REPO, TEXT_TOKENIZER_NAME
#   NUM_WORKERS
set -euo pipefail

AUDIO_DIR="${AUDIO_DIR:-data/japanese_kame/audio}"
TEXT_DIR="${TEXT_DIR:-data/japanese_kame/text}"
ORACLE_DIR="${ORACLE_DIR:-data/japanese_kame/oracle_raw}"

TOKENIZED_AUDIO_DIR="${TOKENIZED_AUDIO_DIR:-data/japanese_kame/tokenized_audio}"
TOKENIZED_TEXT_DIR="${TOKENIZED_TEXT_DIR:-data/japanese_kame/tokenized_text}"
TOKENIZED_ORACLE_DIR="${TOKENIZED_ORACLE_DIR:-data/japanese_kame/tokenized_oracle_a0b1_events}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-processed_data/japanese_kame/train_text_oracle_a0b1_events}"

# J-Moshi uses rinna/japanese-gpt2-medium SentencePiece.
# Confirm the exact filename by checking nu-dialogue/j-moshi-ext on HuggingFace.
TEXT_TOKENIZER_REPO="${TEXT_TOKENIZER_REPO:-nu-dialogue/j-moshi-ext}"
TEXT_TOKENIZER_NAME="${TEXT_TOKENIZER_NAME:-tokenizer_spm_32k.model}"

NUM_WORKERS="${NUM_WORKERS:-8}"

echo "========================================"
echo "Step 1: Audio tokenization"
echo "========================================"
uv run -m tools.tokenize_audio \
    --audio_dir "${AUDIO_DIR}" \
    --output_dir "${TOKENIZED_AUDIO_DIR}"

echo "========================================"
echo "Step 2: Text tokenization (Japanese)"
echo "========================================"
uv run -m tools.tokenize_text \
    --word_transcript_dir "${TEXT_DIR}" \
    --output_dir "${TOKENIZED_TEXT_DIR}" \
    --text_tokenizer_repo "${TEXT_TOKENIZER_REPO}" \
    --text_tokenizer_name "${TEXT_TOKENIZER_NAME}" \
    --no_whitespace_before_word \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --num_workers "${NUM_WORKERS}" \
    --resume \
    --allow_alignment_warnings

echo "========================================"
echo "Step 3: Oracle tokenization"
echo "========================================"
uv run -m tools.tokenize_oracle \
    --oracle_dir "${ORACLE_DIR}" \
    --oracle_suffix ".json" \
    --tokenized_audio_dir "${TOKENIZED_AUDIO_DIR}" \
    --output_dir "${TOKENIZED_ORACLE_DIR}" \
    --A_channel 0 \
    --B_channel 1

echo "========================================"
echo "Step 4: Dataset assembly"
echo "========================================"
uv run -m tools.prepare_dataset \
    --tokenized_text_dir "${TOKENIZED_TEXT_DIR}" \
    --tokenized_audio_dir "${TOKENIZED_AUDIO_DIR}" \
    --tokenized_oracle_dir "${TOKENIZED_ORACLE_DIR}" \
    --output_prefix "${OUTPUT_PREFIX}"

echo "========================================"
echo "Done. Parquet files written to:"
echo "  ${OUTPUT_PREFIX}-*.parquet"
echo "========================================"
