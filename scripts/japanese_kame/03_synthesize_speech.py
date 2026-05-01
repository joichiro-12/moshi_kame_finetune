"""Synthesize stereo WAV files from two-speaker dialogue JSON using J-Moshi TTS.

For each dialogue JSON, speaker A audio goes to the left channel and speaker B
to the right channel.  Following J-Moshi paper Sec. 3.4, we run synthesis with
--num_seeds different random seeds and keep the one with the lowest WER
(measured by whisperx transcription against the dialogue text).

Prerequisites:
    - J-Moshi (nu-dialogue/j-moshi) must be installed or available as a server
    - GPU recommended for reasonable throughput

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
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


SAMPLE_RATE = 24000


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between two strings."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    # simple dynamic programming WER
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / n


def _transcribe_channel(audio: np.ndarray, device: str) -> str:
    """Transcribe a mono audio array using whisperx."""
    try:
        import whisperx

        model = whisperx.load_model("large-v3", device=device, language="ja")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp_path = tmp.name
        result = model.transcribe(tmp_path)
        os.unlink(tmp_path)
        return " ".join(seg["text"] for seg in result.get("segments", []))
    except Exception:
        return ""


def _dialogue_reference(turns: list[dict], speaker: str) -> str:
    return " ".join(t["text"] for t in turns if t.get("speaker") == speaker)


def _synthesize_with_j_moshi(
    turns: list[dict],
    seed: int,
    device: str,
    j_moshi_repo: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Synthesize a two-speaker dialogue via J-Moshi TTS.

    Returns (audio_a, audio_b) numpy arrays (mono, SAMPLE_RATE) or None on failure.

    J-Moshi's multi-stream TTS is invoked via the CLI entry point
    `python -m jmoshi.tts` (exact module path may differ; see nu-dialogue/j-moshi).
    The CLI writes a stereo WAV to a temporary path; we split it here.
    """
    dialogue_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "dialogue.txt")
        output_path = os.path.join(tmpdir, "output.wav")
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(dialogue_text)

        # J-Moshi TTS CLI – adjust module path to match the installed package.
        # nu-dialogue/j-moshi exposes multi-stream TTS at `jmoshi.server` or similar.
        # The command below is a best-effort approximation; update to match the
        # actual j-moshi package CLI once the repository is confirmed.
        cmd = [
            "python", "-m", "jmoshi.tts",
            "--repo", j_moshi_repo,
            "--input", input_path,
            "--output", output_path,
            "--seed", str(seed),
            "--device", device,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output_path):
            return None

        stereo, sr = sf.read(output_path, always_2d=True)
        if stereo.shape[1] < 2:
            return None
        audio_a = stereo[:, 0]
        audio_b = stereo[:, 1]
        return audio_a, audio_b


def synthesize_dialogue(
    turns: list[dict],
    output_path: Path,
    *,
    num_seeds: int,
    device: str,
    j_moshi_repo: str,
) -> bool:
    """Synthesize with multiple seeds and keep the one with lowest WER."""
    ref_a = _dialogue_reference(turns, "A")
    ref_b = _dialogue_reference(turns, "B")

    best_wer = float("inf")
    best_audio: tuple[np.ndarray, np.ndarray] | None = None

    for seed in range(num_seeds):
        result = _synthesize_with_j_moshi(turns, seed=seed, device=device, j_moshi_repo=j_moshi_repo)
        if result is None:
            continue
        audio_a, audio_b = result

        hyp_a = _transcribe_channel(audio_a, device)
        hyp_b = _transcribe_channel(audio_b, device)
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
            turns,
            out_path,
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
        description="Synthesize stereo WAV files from dialogue JSON using J-Moshi TTS."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/japanese_kame/dialogues",
        help="Directory containing dialogue JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/japanese_kame/audio",
        help="Directory to write stereo WAV files (L=A, R=B).",
    )
    parser.add_argument(
        "--j_moshi_repo",
        type=str,
        default="nu-dialogue/j-moshi-ext",
        help="HuggingFace repo ID for J-Moshi (used by the TTS CLI).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of random seeds to try; keep lowest WER (J-Moshi paper Sec. 3.4).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Skip already-synthesized files.")
    main(parser.parse_args())
