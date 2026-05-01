"""Generate word-level timestamps for stereo WAV files using WhisperX.

Reads stereo WAVs (L=speaker A, R=speaker B), processes each channel
independently, and writes canonical text/*.json files:

    [{"speaker": "A", "word": "こんにちは", "start": 0.46, "end": 1.02}, ...]

Usage:
    uv run --extra data -m scripts.japanese_kame.04_word_alignment \
        --audio_dir  data/japanese_kame/audio \
        --output_dir data/japanese_kame/text \
        --device cuda \
        --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


SAMPLE_RATE = 16000  # WhisperX expects 16 kHz
WHISPERX_MODEL = "large-v3"


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio

        import torch

        wav = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(wav).squeeze(0).numpy()
    except Exception:
        # fallback: linear interpolation
        factor = target_sr / orig_sr
        new_len = int(len(audio) * factor)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(audio.dtype)


def align_audio(
    audio_mono: np.ndarray,
    orig_sr: int,
    speaker: str,
    device: str,
) -> list[dict]:
    """Run WhisperX on a mono audio array and return word-level transcript."""
    import whisperx

    audio_16k = _resample(audio_mono, orig_sr, SAMPLE_RATE).astype(np.float32)

    model = whisperx.load_model(WHISPERX_MODEL, device=device, language="ja")
    result = model.transcribe(audio_16k, batch_size=16)

    align_model, metadata = whisperx.load_align_model(
        language_code="ja", device=device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio_16k,
        device,
        return_char_alignments=False,
    )

    words = []
    for seg in aligned.get("word_segments", []):
        word = seg.get("word", "").strip()
        start = seg.get("start")
        end = seg.get("end")
        if word and start is not None and end is not None:
            words.append(
                {
                    "speaker": speaker,
                    "word": word,
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                }
            )
    return words


def process_file(audio_path: Path, output_dir: Path, device: str) -> bool:
    out_path = output_dir / f"{audio_path.stem}.json"
    try:
        stereo, sr = sf.read(str(audio_path), always_2d=True)
    except Exception as e:
        print(f"[WARN] Cannot read {audio_path.name}: {e}")
        return False

    if stereo.shape[1] < 2:
        print(f"[WARN] {audio_path.name} is not stereo, skipping")
        return False

    audio_a = stereo[:, 0]
    audio_b = stereo[:, 1]

    words_a = align_audio(audio_a, sr, "A", device)
    words_b = align_audio(audio_b, sr, "B", device)

    # merge and sort by start time
    all_words = sorted(words_a + words_b, key=lambda w: w["start"])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
    return True


def worker(audio_paths: list[Path], output_dir: Path, device: str, resume: bool) -> None:
    for audio_path in tqdm(audio_paths, desc=f"[{device}]", dynamic_ncols=True):
        out_path = output_dir / f"{audio_path.stem}.json"
        if resume and out_path.exists():
            continue
        ok = process_file(audio_path, output_dir, device)
        if not ok:
            print(f"[WARN] Skipped: {audio_path.name}")


def main(args: argparse.Namespace) -> None:
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No WAV files found in {audio_dir}")
    print(f"Found {len(audio_files)} WAV files")

    if args.num_workers <= 1:
        worker(audio_files, output_dir, args.device, args.resume)
    else:
        chunks = np.array_split(audio_files, args.num_workers)
        processes = []
        for i, chunk in enumerate(chunks):
            device = f"cuda:{i}" if args.device == "cuda" else args.device
            p = mp.Process(target=worker, args=(list(chunk), output_dir, device, args.resume))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed")

    print(f"Word transcripts written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate word-level timestamps from stereo WAV files using WhisperX."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/japanese_kame/audio",
        help="Directory containing stereo WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/japanese_kame/text",
        help="Directory to write canonical text/*.json files.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (each gets its own CUDA device).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-aligned files.")
    main(parser.parse_args())
