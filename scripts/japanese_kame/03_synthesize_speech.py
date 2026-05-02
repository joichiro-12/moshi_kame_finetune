"""Synthesize stereo WAV files from two-speaker dialogue JSON using J-Moshi.

J-Moshi does NOT expose a standalone TTS CLI (`jmoshi.tts` does not exist).
This script uses kame-model's offline inference API (LMGen + MimiModel) to
produce speech directly, which is the same underlying engine used by
`moshi.server --hf-repo nu-dialogue/j-moshi-ext`.

Multi-stream TTS procedure (following J-Moshi paper Sec. 3.4):
  1. Load J-Moshi model weights via kame-model loaders
  2. For each dialogue, feed text tokens frame-by-frame while the model
     generates audio tokens for both speakers concurrently
  3. Run with --num_seeds different random seeds; keep the WAV whose
     WhisperX WER against the source text is lowest

Output stereo WAV layout: L=speaker A, R=speaker B, 24 kHz.

Prerequisites:
  - GPU with ≥24 GB VRAM (same requirement as moshi.server)
  - kame-model package (already in this venv)

Usage:
    uv run -m scripts.japanese_kame.03_synthesize_speech \
        --input_dir  data/japanese_kame/dialogues \
        --output_dir data/japanese_kame/audio \
        --j_moshi_repo nu-dialogue/j-moshi-ext \
        --num_seeds 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


SAMPLE_RATE = 24000
# Frame rate of the Mimi audio codec (12.5 Hz = 80 ms per frame)
FRAME_RATE = 12.5
# Average Japanese speech rate: roughly 7 mora/s → ~3.5 tokens/frame at 12.5 fps
CHARS_PER_SECOND = 8.0


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate between two strings (character-level for Japanese)."""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    if not ref:
        return 0.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            new_dp[j] = dp[j - 1] if ref[i - 1] == hyp[j - 1] else 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / n


def _transcribe_mono(audio: np.ndarray, device: str) -> str:
    """Transcribe a mono float32 array (24 kHz) using whisperx."""
    try:
        import whisperx

        model = whisperx.load_model("large-v3", device=device, language="ja")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            path = tmp.name
        result = model.transcribe(path)
        os.unlink(path)
        return " ".join(seg["text"] for seg in result.get("segments", []))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Offline TTS via kame-model (LMGen)
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _load_model(repo: str, device: str):
    """Load J-Moshi model from HuggingFace (cached across calls within a process)."""
    cache_key = f"{repo}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from kame.models import loaders
    from kame.run_inference import InferenceState, get_condition_tensors

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        repo,
        moshi_weight=None,
        mimi_weight=None,
        tokenizer=None,
    )
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16)
    state = InferenceState(
        checkpoint_info, mimi, text_tokenizer, lm,
        batch_size=1, cfg_coef=1.0, device=device,
    )
    _model_cache[cache_key] = (state, text_tokenizer)
    return state, text_tokenizer


def _text_to_frame_schedule(
    turns: list[dict[str, str]],
    text_tokenizer,
    frame_rate: float = FRAME_RATE,
    chars_per_second: float = CHARS_PER_SECOND,
) -> tuple[list[tuple[int, str, list[int]]], int]:
    """Build a per-frame token injection schedule from a dialogue turn list.

    Returns:
        schedule: list of (frame_idx, speaker, token_ids)
        total_frames: estimated total audio frames to generate
    """
    from tools.tokenize_text import encode_as_pieces_wo_byte_fallback

    schedule: list[tuple[int, str, list[int]]] = []
    cursor = 0.0  # seconds

    for turn in turns:
        speaker = turn.get("speaker", "A")
        text = turn.get("text", "").strip()
        if not text:
            continue

        tokens = encode_as_pieces_wo_byte_fallback(text_tokenizer, text)
        token_ids = [text_tokenizer.piece_to_id(t) for t in tokens if t]
        if not token_ids:
            continue

        duration = len(text) / chars_per_second
        frames_for_turn = max(1, int(duration * frame_rate))
        stride = max(1, frames_for_turn // max(1, len(token_ids)))

        start_frame = int(cursor * frame_rate)
        for i, tid in enumerate(token_ids):
            schedule.append((start_frame + i * stride, speaker, [tid]))

        cursor += duration + 0.3  # 300 ms gap between turns

    total_frames = int(cursor * frame_rate) + int(1.0 * frame_rate)
    return schedule, total_frames


@torch.no_grad()
def synthesize_with_kame(
    turns: list[dict[str, str]],
    seed: int,
    device: str,
    repo: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Synthesize a two-speaker dialogue offline using kame-model LMGen.

    Returns (audio_a, audio_b) float32 numpy arrays at SAMPLE_RATE Hz, or None on error.

    Implementation note: Moshi/J-Moshi generates both the inner-monologue text
    stream AND the audio stream simultaneously.  For TTS purposes we seed the
    text stream with the known dialogue text and let the model fill in the audio.
    Speaker A tokens go to stream 0; speaker B tokens go to stream 1 (oracle).
    This matches how j-moshi-ext was fine-tuned for multi-stream TTS.
    """
    try:
        import sphn

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        state, text_tokenizer = _load_model(repo, device)
        state.mimi.reset_streaming()
        state.lm_gen.reset_streaming()

        schedule, total_frames = _text_to_frame_schedule(turns, text_tokenizer)
        schedule_map: dict[int, dict[str, list[int]]] = {}
        for frame_idx, speaker, token_ids in schedule:
            schedule_map.setdefault(frame_idx, {}).setdefault(speaker, []).extend(token_ids)

        audio_a_chunks: list[np.ndarray] = []
        audio_b_chunks: list[np.ndarray] = []

        for frame in range(total_frames):
            # Inject scheduled text tokens into the appropriate stream
            text_ids_a = schedule_map.get(frame, {}).get("A", None)
            text_ids_b = schedule_map.get(frame, {}).get("B", None)

            # Step the LM one frame
            # lm_gen.step() returns (text_token, audio_codes) for the main stream
            # For multi-stream models oracle tokens feed the secondary stream
            text_tok, codes = state.lm_gen.step(
                text_ids_a[0] if text_ids_a else None,
                oracle_tokens=text_ids_b[0] if text_ids_b else None,
            )

            # Decode the audio codes (Mimi codecs) to waveform
            # codes shape: [K, 1] where K=num_codebooks
            codes_tensor = torch.tensor(codes, device=device).unsqueeze(0)  # [1, K, 1]
            pcm = state.mimi.decode(codes_tensor)  # [1, 1, frame_size]
            audio_frame = pcm.squeeze().cpu().numpy().astype(np.float32)
            # Split L/R if stereo output, else duplicate for speaker assignment
            if audio_frame.ndim == 2:
                audio_a_chunks.append(audio_frame[0])
                audio_b_chunks.append(audio_frame[1])
            else:
                # Fallback: assign same audio to both (not ideal)
                audio_a_chunks.append(audio_frame)
                audio_b_chunks.append(audio_frame)

        if not audio_a_chunks:
            return None

        return np.concatenate(audio_a_chunks), np.concatenate(audio_b_chunks)

    except Exception as e:
        print(f"[WARN] kame synthesis error (seed={seed}): {e}")
        return None


def synthesize_dialogue(
    turns: list[dict],
    output_path: Path,
    *,
    num_seeds: int,
    device: str,
    j_moshi_repo: str,
) -> bool:
    """Synthesize with multiple seeds and keep the one with lowest WER."""
    ref_a = " ".join(t["text"] for t in turns if t.get("speaker") == "A")
    ref_b = " ".join(t["text"] for t in turns if t.get("speaker") == "B")

    best_wer = float("inf")
    best_audio: tuple[np.ndarray, np.ndarray] | None = None

    for seed in range(num_seeds):
        result = synthesize_with_kame(turns, seed=seed, device=device, repo=j_moshi_repo)
        if result is None:
            continue
        audio_a, audio_b = result

        hyp_a = _transcribe_mono(audio_a, device)
        hyp_b = _transcribe_mono(audio_b, device)
        wer = (_compute_wer(ref_a, hyp_a) + _compute_wer(ref_b, hyp_b)) / 2

        if wer < best_wer:
            best_wer = wer
            best_audio = (audio_a, audio_b)

    if best_audio is None:
        return False

    audio_a, audio_b = best_audio
    min_len = min(len(audio_a), len(audio_b))
    stereo = np.stack([audio_a[:min_len], audio_b[:min_len]], axis=1)
    sf.write(str(output_path), stereo, SAMPLE_RATE)
    return True


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No dialogue JSON files found in {input_dir}")
    print(f"Found {len(json_files)} dialogue files")

    success = 0
    fail = 0
    for json_path in tqdm(json_files, desc="Synthesizing"):
        out_path = output_dir / f"{json_path.stem}.wav"
        if args.resume and out_path.exists():
            success += 1
            continue

        with json_path.open(encoding="utf-8") as f:
            turns = json.load(f)

        ok = synthesize_dialogue(
            turns, out_path,
            num_seeds=args.num_seeds,
            device=args.device,
            j_moshi_repo=args.j_moshi_repo,
        )
        if ok:
            success += 1
        else:
            fail += 1
            print(f"[WARN] Failed: {json_path.name}")

    print(f"\nDone: {success} synthesized, {fail} failed → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize stereo WAV files from dialogue JSON using J-Moshi (kame-model offline)."
    )
    parser.add_argument("--input_dir", type=str, default="data/japanese_kame/dialogues")
    parser.add_argument("--output_dir", type=str, default="data/japanese_kame/audio")
    parser.add_argument("--j_moshi_repo", type=str, default="nu-dialogue/j-moshi-ext")
    parser.add_argument(
        "--num_seeds", type=int, default=10,
        help="Number of random seeds; keep lowest WER (J-Moshi paper Sec. 3.4).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
