"""Synthesize stereo WAV files from two-speaker dialogue JSON using Google Gemini TTS.

Calls Gemini TTS once per dialogue turn (speaker A and B synthesized separately),
then assembles them into a stereo WAV: L=speaker A, R=speaker B, 24 kHz.
Authenticated via a Google Cloud service account JSON file.

Usage:
    uv run --extra tts -m scripts.japanese_kame.03_synthesize_speech \
        --service_account research-494900-52bc0aa22136.json \
        --model gemini-3.1-flash-tts-preview
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 24000
SILENCE_SAMPLES = int(0.3 * SAMPLE_RATE)  # 300 ms gap between turns


def _build_client(service_account_path: str, location: str):
    from google import genai
    from google.oauth2 import service_account

    with open(service_account_path) as f:
        sa_info = json.load(f)
    project_id = sa_info["project_id"]

    credentials = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )


def _synthesize_turn(text: str, voice: str, model: str, client) -> np.ndarray:
    """Call Gemini TTS for a single utterance, return float32 mono PCM at SAMPLE_RATE."""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    part = response.candidates[0].content.parts[0]
    audio_bytes = part.inline_data.data
    mime = part.inline_data.mime_type or ""

    if "wav" in mime:
        audio, _ = sf.read(io.BytesIO(audio_bytes))
        return audio.astype(np.float32)
    # Default: raw PCM int16 at SAMPLE_RATE
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def synthesize_dialogue(
    turns: list[dict],
    output_path: Path,
    *,
    model: str,
    client,
    voice_a: str,
    voice_b: str,
) -> None:
    """Synthesize each turn and assemble stereo WAV (L=A, R=B)."""
    silence = np.zeros(SILENCE_SAMPLES, dtype=np.float32)

    ch_a: list[np.ndarray] = []
    ch_b: list[np.ndarray] = []

    for turn in turns:
        speaker = turn.get("speaker", "A")
        text = turn.get("text", "").strip()
        if not text:
            continue
        voice = voice_a if speaker == "A" else voice_b
        audio = _synthesize_turn(text, voice, model, client)
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
    sf.write(str(output_path), stereo, SAMPLE_RATE)


def main(args: argparse.Namespace) -> None:
    client = _build_client(args.service_account, location=args.location)

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

        try:
            synthesize_dialogue(
                turns, out_path,
                model=args.model,
                client=client,
                voice_a=args.voice_a,
                voice_b=args.voice_b,
            )
            success += 1
        except Exception as e:
            fail += 1
            print(f"[WARN] Failed {json_path.name}: {e}")

    print(f"\nDone: {success} synthesized, {fail} failed → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize stereo WAV from dialogue JSON using Google Gemini TTS."
    )
    parser.add_argument("--input_dir", type=str, default="data/japanese_kame/test_dialogues")
    parser.add_argument("--output_dir", type=str, default="data/japanese_kame/test_audio")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-tts-preview")
    parser.add_argument(
        "--service_account",
        type=str,
        default="research-494900-52bc0aa22136.json",
        help="Path to Google Cloud service account JSON.",
    )
    parser.add_argument("--location", type=str, default="us-central1")
    parser.add_argument("--voice_a", type=str, default="Kore", help="Gemini TTS voice for speaker A.")
    parser.add_argument("--voice_b", type=str, default="Puck", help="Gemini TTS voice for speaker B.")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated files.")
    main(parser.parse_args())
