"""Generate word-level timestamps for stereo WAV files using faster-whisper.

Reads stereo WAVs (L=speaker A, R=speaker B), processes each channel
independently, and writes canonical text/*.json files:

    [{"speaker": "A", "word": "こんにちは", "start": 0.46, "end": 1.02}, ...]

Usage:
    uv run --extra data -m scripts.japanese_kame.04_word_alignment \
        --audio_dir  data/japanese_kame/test_audio \
        --output_dir data/japanese_kame/test_text \
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


SAMPLE_RATE = 16000  # Whisper expects 16 kHz
WHISPER_MODEL = "large-v3"

_model_cache: dict[str, object] = {}


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
    compute_type: str = "float16",
) -> list[dict]:
    """Run faster-whisper on a mono audio array and return word-level transcript."""
    from faster_whisper import WhisperModel

    cache_key = f"{WHISPER_MODEL}:{device}:{compute_type}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = WhisperModel(
            WHISPER_MODEL, device=device, compute_type=compute_type
        )
    model = _model_cache[cache_key]

    audio_16k = _resample(audio_mono, orig_sr, SAMPLE_RATE).astype(np.float32)
    segments, _ = model.transcribe(audio_16k, language="ja", word_timestamps=True, beam_size=5)

    words = []
    for seg in segments:
        for w in seg.words or []:
            word = w.word.strip()
            if word and w.start is not None and w.end is not None:
                words.append(
                    {
                        "speaker": speaker,
                        "word": word,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                    }
                )
    return words


def process_file(audio_path: Path, output_dir: Path, device: str, compute_type: str) -> bool:
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

    words_a = align_audio(audio_a, sr, "A", device, compute_type)
    words_b = align_audio(audio_b, sr, "B", device, compute_type)

    all_words = sorted(words_a + words_b, key=lambda w: w["start"])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
    return True


def worker(
    audio_paths: list[Path], output_dir: Path, device: str, resume: bool, compute_type: str
) -> None:
    for audio_path in tqdm(audio_paths, desc=f"[{device}]", dynamic_ncols=True):
        out_path = output_dir / f"{audio_path.stem}.json"
        if resume and out_path.exists():
            continue
        ok = process_file(audio_path, output_dir, device, compute_type)
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

    compute_type = args.compute_type or ("float16" if args.device != "cpu" else "int8")

    if args.num_workers <= 1:
        worker(audio_files, output_dir, args.device, args.resume, compute_type)
    else:
        chunks = np.array_split(audio_files, args.num_workers)
        processes = []
        for i, chunk in enumerate(chunks):
            device = f"cuda:{i}" if args.device == "cuda" else args.device
            p = mp.Process(
                target=worker, args=(list(chunk), output_dir, device, args.resume, compute_type)
            )
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
        default="data/japanese_kame/test_audio",
        help="Directory containing stereo WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/japanese_kame/test_text",
        help="Directory to write canonical text/*.json files.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="",
        help="CTranslate2 compute type (float16, int8_float16, int8). Default: float16 on GPU, int8 on CPU.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (each gets its own CUDA device).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-aligned files.")
    main(parser.parse_args())
