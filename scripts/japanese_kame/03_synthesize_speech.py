"""Synthesize stereo WAV files from two-speaker dialogue JSON via TTS server.

Sends each dialogue turn to the TTS server (POST /synthesize) and assembles
the responses into a stereo WAV: L=speaker A, R=speaker B.

Start the TTS server first:
    cd /mnt/kiso-qnap4/jsato/tts_server
    DEVICE=cuda:1 uv run uvicorn main:app --host 0.0.0.0 --port 8001

Usage (sarashina — default, zero-shot voice cloning):
    uv run -m scripts.japanese_kame.03_synthesize_speech \
        --input_dir  data/japanese_kame/test/test_dialogues \
        --output_dir data/japanese_kame/test/test_audio_sarashina \
        --engine sarashina \
        --ref_audio_a /mnt/kiso-qnap4/jsato/moshi_kame_finetune/data/sample_audio/male_v1.wav \
        --ref_audio_b /mnt/kiso-qnap4/jsato/moshi_kame_finetune/data/sample_audio/female_v2.wav

Usage (qwen_custom — no ref audio needed):
    uv run -m scripts.japanese_kame.03_synthesize_speech \
        --input_dir  data/japanese_kame/dialogues \
        --output_dir data/japanese_kame/audio \
        --engine qwen_custom

Usage (qwen_clone — voice cloning with ref audio):
    uv run -m scripts.japanese_kame.03_synthesize_speech \
        --input_dir  data/japanese_kame/test/test_dialogues \
        --output_dir data/japanese_kame/test/test_audio \
        --engine qwen_clone \
        --ref_audio_a /mnt/kiso-qnap4/jsato/moshi_kame_finetune/data/sample_audio/man.mp3 \
        --ref_audio_b /mnt/kiso-qnap4/jsato/moshi_kame_finetune/data/sample_audio/woman_sample.mp3
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SILENCE_SEC = 0.3

DEFAULT_SERVER_URL = "http://localhost:8001"

_print_lock = threading.Lock()


def _encode_audio(path: str) -> str:
    """Read a WAV file and return base64-encoded bytes."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def _build_payload(text: str, speaker: str, args: argparse.Namespace) -> dict:
    """Build the /synthesize request payload for the given speaker (A or B)."""
    base: dict = {"text": text, "engine": args.engine}

    if args.engine == "qwen_custom":
        base["instruct"] = args.voice_a if speaker == "A" else args.voice_b
    elif args.engine in ("qwen_clone", "sarashina"):
        ref_path = args.ref_audio_a if speaker == "A" else args.ref_audio_b
        ref_text = args.ref_text_a if speaker == "A" else args.ref_text_b
        if ref_path:
            base["ref_audio"] = _encode_audio(ref_path)
        if ref_text:
            base["ref_text"] = ref_text

    return base


def _synthesize_turn(payload: dict, server_url: str) -> tuple[np.ndarray, int]:
    """POST to TTS server, return (float32 mono PCM, sample_rate)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url}/synthesize",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio, sr = sf.read(io.BytesIO(resp.read()))
    except urllib.error.URLError as e:
        raise RuntimeError(f"TTS server unreachable at {server_url}: {e}") from e
    return np.asarray(audio, dtype=np.float32), int(sr)


def synthesize_dialogue(
    turns: list[dict],
    output_path: Path,
    *,
    server_url: str,
    args: argparse.Namespace,
) -> None:
    """Synthesize each turn and assemble stereo WAV (L=A, R=B)."""
    ch_a: list[np.ndarray] = []
    ch_b: list[np.ndarray] = []
    sample_rate: int | None = None

    for turn in turns:
        speaker = turn.get("speaker", "A")
        text = turn.get("text", "").strip()
        if not text:
            continue

        payload = _build_payload(text, speaker, args)
        audio, sr = _synthesize_turn(payload, server_url)

        if sample_rate is None:
            sample_rate = sr
        silence = np.zeros(int(SILENCE_SEC * sr), dtype=np.float32)
        n = len(audio)

        if speaker == "A":
            ch_a.append(audio)
            ch_b.append(np.zeros(n, dtype=np.float32))
        else:
            ch_a.append(np.zeros(n, dtype=np.float32))
            ch_b.append(audio)

        ch_a.append(silence)
        ch_b.append(silence)

    if not ch_a:
        raise ValueError("No audio generated (empty turns)")

    stereo = np.stack([np.concatenate(ch_a), np.concatenate(ch_b)], axis=1)
    sf.write(str(output_path), stereo, sample_rate)


def _process_file(
    json_path: Path,
    output_dir: Path,
    server_url: str,
    args: argparse.Namespace,
) -> bool:
    out_path = output_dir / f"{json_path.stem}.wav"
    if args.resume and out_path.exists():
        return True
    with json_path.open(encoding="utf-8") as f:
        turns = json.load(f)
    synthesize_dialogue(turns, out_path, server_url=server_url, args=args)
    return True


def main(args: argparse.Namespace) -> None:
    try:
        with urllib.request.urlopen(f"{args.server_url}/health", timeout=5) as resp:
            info = json.loads(resp.read())
        print(f"TTS server: {info}")
        engine_key = args.engine
        if not info.get("engines", {}).get(engine_key):
            raise RuntimeError(f"{engine_key} engine is not available on the server")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach TTS server at {args.server_url}: {e}") from e

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No dialogue JSON files found in {input_dir}")
    print(f"Found {len(json_files)} dialogue files  (engine={args.engine}, max_workers={args.max_workers})")

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_process_file, p, output_dir, args.server_url, args): p
            for p in json_files
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Synthesizing")
        for future in pbar:
            path = futures[future]
            try:
                future.result()
                success += 1
            except Exception as e:
                fail += 1
                with _print_lock:
                    print(f"\n[WARN] Failed {path.name}: {e}")

    print(f"\nDone: {success} synthesized, {fail} failed → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize stereo WAV from dialogue JSON via TTS server."
    )
    parser.add_argument("--input_dir", type=str, default="data/japanese_kame/dialogues")
    parser.add_argument("--output_dir", type=str, default="data/japanese_kame/audio")
    parser.add_argument("--server_url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument(
        "--engine",
        type=str,
        default="sarashina",
        choices=["qwen_custom", "qwen_clone", "sarashina"],
        help="TTS engine to use (default: sarashina).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel synthesis threads.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-generated files.")

    # Qwen3 CustomVoice options
    qwen = parser.add_argument_group("Qwen3 CustomVoice options")
    qwen.add_argument("--voice_a", type=str, default="", help="Instruct style for speaker A.")
    qwen.add_argument("--voice_b", type=str, default="", help="Instruct style for speaker B.")

    # Qwen3 voice clone options
    clone = parser.add_argument_group("Qwen3 voice clone options")
    clone.add_argument("--ref_audio_a", type=str, default="", help="Reference WAV for speaker A.")
    clone.add_argument("--ref_audio_b", type=str, default="", help="Reference WAV for speaker B.")
    clone.add_argument("--ref_text_a", type=str, default="", help="Reference transcript for speaker A.")
    clone.add_argument("--ref_text_b", type=str, default="", help="Reference transcript for speaker B.")

    main(parser.parse_args())
