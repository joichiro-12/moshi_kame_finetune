#!/usr/bin/env bash
# Per-batch tokenize → parquet for incremental data prep (0610).
#
# Picks the next BATCH_SIZE dialogue names that have ALL of oracle+audio+text
# (so prepare_dataset's exact-set-match constraint is satisfied), stages them
# into a per-batch input dir via symlinks, runs the 4 tokenization tools, and
# writes ONE self-contained parquet for the batch.
#
# Idempotent:
#   - Each batch's chosen names are frozen in <batch>/names.txt. Re-running the
#     same BATCH_ID reuses that list (resumes tokenization, no reselection).
#   - Other batches' names.txt are excluded, so batches never overlap.
#
# Usage:
#   BATCH_ID=0 BATCH_SIZE=200  bash scripts/japanese_kame/06b_tokenize_batch.sh
#   BATCH_ID=1 BATCH_SIZE=5000 bash scripts/japanese_kame/06b_tokenize_batch.sh
#   # Pin to a GPU:
#   CUDA_VISIBLE_DEVICES=1 BATCH_ID=2 bash scripts/japanese_kame/06b_tokenize_batch.sh
set -euo pipefail

# --- config ---
BATCH_ID="${BATCH_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-200}"

ROOT="${ROOT:-data/japanese_kame/0610}"
AUDIO_DIR="${AUDIO_DIR:-data/japanese_kame/audio_gemini}"
TEXT_DIR="${TEXT_DIR:-${ROOT}/text}"
ORACLE_DIR="${ORACLE_DIR:-${ROOT}/oracle_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/japanese_kame/0610}"

TEXT_TOKENIZER_REPO="${TEXT_TOKENIZER_REPO:-nu-dialogue/j-moshi-ext}"
TEXT_TOKENIZER_NAME="${TEXT_TOKENIZER_NAME:-tokenizer_spm_32k_3.model}"
NUM_AUDIO_WORKERS="${NUM_AUDIO_WORKERS:-2}"
NUM_TEXT_WORKERS="${NUM_TEXT_WORKERS:-8}"

# Resolve to absolute paths so symlinks resolve regardless of cwd.
AUDIO_ABS="$(readlink -f "${AUDIO_DIR}")"
TEXT_ABS="$(readlink -f "${TEXT_DIR}")"
ORACLE_ABS="$(readlink -f "${ORACLE_DIR}")"

BATCHES_ROOT="${ROOT}/_batches"
BATCH_TAG="batch_$(printf '%04d' "${BATCH_ID}")"
BATCH="${BATCHES_ROOT}/${BATCH_TAG}"
NAMES_FILE="${BATCH}/names.txt"

mkdir -p "${BATCH}" "${OUTPUT_DIR}"

echo "============================================================"
echo " Batch ${BATCH_TAG}  (size=${BATCH_SIZE})"
echo "   audio : ${AUDIO_ABS}"
echo "   text  : ${TEXT_ABS}"
echo "   oracle: ${ORACLE_ABS}"
echo "   out   : ${OUTPUT_DIR}/${BATCH_TAG}-*.parquet"
echo "============================================================"

# --- 1) select names (only if not already frozen for this batch) ---
if [ -s "${NAMES_FILE}" ]; then
    echo "Reusing existing ${NAMES_FILE} ($(wc -l < "${NAMES_FILE}") names)."
else
    echo "Selecting matched names (oracle ∩ audio ∩ text) minus already-claimed..."
    tmp="$(mktemp -d)"
    find "${ORACLE_ABS}" -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort -u > "${tmp}/oracle"
    find "${TEXT_ABS}"   -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort -u > "${tmp}/text"
    find "${AUDIO_ABS}"  -maxdepth 1 -name '*.wav'  -printf '%f\n' | sed 's/\.wav$//'  | sort -u > "${tmp}/audio"
    comm -12 "${tmp}/oracle" "${tmp}/text" | comm -12 - "${tmp}/audio" > "${tmp}/matched_raw"

    # Keep only dialogues whose text has BOTH speakers A and B. Some texts are
    # single-speaker (yet still got an oracle), which would make tokenize_text
    # fail (it requires A and B). Cache the valid set since the scan is O(all).
    VALID_FILE="${BATCHES_ROOT}/_valid_two_speaker.txt"
    if [ ! -s "${VALID_FILE}" ]; then
        echo "  Building two-speaker validity cache (one-time scan of text dir)..."
        python3 - "${TEXT_ABS}" "${VALID_FILE}" <<'PY'
import json, os, sys
text_dir, out = sys.argv[1], sys.argv[2]
valid = []
for fn in os.listdir(text_dir):
    if not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(text_dir, fn), encoding="utf-8") as f:
            spk = {seg.get("speaker") for seg in json.load(f)}
    except Exception:
        continue
    if "A" in spk and "B" in spk:
        valid.append(fn[:-5])
with open(out, "w") as f:
    f.write("\n".join(sorted(valid)) + ("\n" if valid else ""))
print(f"  valid two-speaker dialogues: {len(valid)}")
PY
    fi
    comm -12 "${tmp}/matched_raw" "${VALID_FILE}" > "${tmp}/matched"

    # names already claimed by any OTHER batch
    cat "${BATCHES_ROOT}"/*/names.txt 2>/dev/null | sort -u > "${tmp}/claimed" || true
    [ -s "${tmp}/claimed" ] || : > "${tmp}/claimed"

    # awk (not head) to take the first N: awk reads the whole stream, avoiding
    # SIGPIPE that would trip `set -o pipefail`.
    comm -23 "${tmp}/matched" "${tmp}/claimed" | awk -v n="${BATCH_SIZE}" 'NR<=n' > "${NAMES_FILE}"
    echo "  matched=$(wc -l < "${tmp}/matched")  claimed=$(wc -l < "${tmp}/claimed")  selected=$(wc -l < "${NAMES_FILE}")"
    rm -rf "${tmp}"
fi

N="$(wc -l < "${NAMES_FILE}")"
if [ "${N}" -eq 0 ]; then
    echo "No available matched names left. Nothing to do."
    exit 0
fi

# --- 2) stage symlinks (re-staged each run; cheap & keeps inputs in sync) ---
rm -rf "${BATCH}/audio" "${BATCH}/text" "${BATCH}/oracle"
mkdir -p "${BATCH}/audio" "${BATCH}/text" "${BATCH}/oracle" \
         "${BATCH}/tok_audio" "${BATCH}/tok_text" "${BATCH}/tok_oracle"
while IFS= read -r n; do
    [ -n "${n}" ] || continue
    ln -sf "${AUDIO_ABS}/${n}.wav"   "${BATCH}/audio/${n}.wav"
    ln -sf "${TEXT_ABS}/${n}.json"   "${BATCH}/text/${n}.json"
    ln -sf "${ORACLE_ABS}/${n}.json" "${BATCH}/oracle/${n}.json"
done < "${NAMES_FILE}"
echo "Staged ${N} dialogues."

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: stopping after staging (no tokenization)."
    exit 0
fi

# --- 3) tokenize ---
echo "── Step 1/4: audio tokenize (Mimi, GPU) ──"
uv run -m tools.tokenize_audio \
    --audio_dir "${BATCH}/audio" \
    --output_dir "${BATCH}/tok_audio" \
    --num_workers "${NUM_AUDIO_WORKERS}" \
    --resume

echo "── Step 2/4: text tokenize (CPU) ──"
uv run -m tools.tokenize_text \
    --word_transcript_dir "${BATCH}/text" \
    --output_dir "${BATCH}/tok_text" \
    --text_tokenizer_repo "${TEXT_TOKENIZER_REPO}" \
    --text_tokenizer_name "${TEXT_TOKENIZER_NAME}" \
    --no_whitespace_before_word \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --num_workers "${NUM_TEXT_WORKERS}" \
    --resume \
    --allow_alignment_warnings

echo "── Step 3/4: oracle tokenize (CPU) ──"
uv run -m tools.tokenize_oracle \
    --oracle_dir "${BATCH}/oracle" \
    --oracle_suffix ".json" \
    --tokenized_audio_dir "${BATCH}/tok_audio" \
    --output_dir "${BATCH}/tok_oracle" \
    --A_channel 0 \
    --B_channel 1 \
    --num_workers "${NUM_TEXT_WORKERS}" \
    --resume

echo "── Step 4/4: assemble parquet ──"
uv run -m tools.prepare_dataset \
    --tokenized_text_dir "${BATCH}/tok_text" \
    --tokenized_audio_dir "${BATCH}/tok_audio" \
    --tokenized_oracle_dir "${BATCH}/tok_oracle" \
    --output_prefix "${OUTPUT_DIR}/${BATCH_TAG}"

echo "============================================================"
echo " Done: ${OUTPUT_DIR}/${BATCH_TAG}-*.parquet"
ls -lh "${OUTPUT_DIR}/${BATCH_TAG}"-*.parquet
echo "============================================================"
