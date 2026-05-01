"""Japanese Speech MT-Bench evaluation for J-KAME.

Evaluates the following systems on a speech-synthesized variant of a Japanese
MT-Bench subset (reasoning, STEM, humanities categories):

  (a) J-Moshi-ext baseline (original, no oracle)
  (b) Cascaded system (Whisper + LLM-jp-4 + TTS)
  (c) J-KAME (this work)

Evaluation pipeline:
  1. Load Japanese MT-Bench questions
  2. Synthesize questions as audio via J-Moshi TTS
  3. Send audio to the target system; capture response audio
  4. ASR the response audio back to text (Whisper)
  5. Score with LLM-as-a-Judge (GPT-4o, following MT-Bench procedure)
  6. Write results to JSON + CSV

Usage:
    uv run --extra oracle -m scripts.japanese_kame.evaluate \
        --systems baseline kame_jp \
        --baseline_url  ws://localhost:8998/api/chat \
        --kame_jp_url   ws://localhost:9000/api/chat \
        --output_dir    eval_results/japanese_kame \
        --num_runs 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

JUDGE_MODEL = "gpt-4o"
MT_BENCH_JA_CATEGORIES = ["reasoning", "stem", "humanities"]

# Default Japanese MT-Bench question set (Nejumi / lm-sys subset for speech).
# Coding, Extraction, Math, Roleplay, Writing are excluded as they are not
# suitable for speech interaction (following KAME paper setup).
DEFAULT_QUESTIONS: list[dict] = [
    # reasoning
    {
        "category": "reasoning",
        "question_id": "r01",
        "turn1": "地球温暖化が進むと海面上昇が起きる主な理由を教えてください。",
        "turn2": "その影響を最も受けやすい地域はどこですか？",
    },
    {
        "category": "reasoning",
        "question_id": "r02",
        "turn1": "三段論法の例を使って論理的推論を説明してください。",
        "turn2": "その推論が誤りになる場合はありますか？",
    },
    # stem
    {
        "category": "stem",
        "question_id": "s01",
        "turn1": "光合成とはどういうプロセスですか？",
        "turn2": "光合成に必要な条件を三つ挙げてください。",
    },
    {
        "category": "stem",
        "question_id": "s02",
        "turn1": "ニュートンの第二法則をわかりやすく説明してください。",
        "turn2": "その法則が日常生活でどう使われているか例を挙げてください。",
    },
    # humanities
    {
        "category": "humanities",
        "question_id": "h01",
        "turn1": "明治維新が日本社会に与えた最大の変化は何ですか？",
        "turn2": "その変化は現代の日本にどう影響していますか？",
    },
    {
        "category": "humanities",
        "question_id": "h02",
        "turn1": "俳句の特徴を教えてください。",
        "turn2": "有名な俳句を一つ挙げて、その意味を説明してください。",
    },
]

JUDGE_SYSTEM_PROMPT = """あなたは公正なAIアシスタントの評価者です。
ユーザーの質問と、あるアシスタントの応答が提示されます。
応答の正確性・有益性・簡潔さを 1〜10 のスコアで評価してください。
評価根拠を1〜2文で述べ、最後に「スコア: X」という形式で数値だけを出力してください。"""


@dataclass
class TurnResult:
    question_id: str
    category: str
    turn: int
    system: str
    question_text: str
    response_text: str
    score: float
    latency_ms: float


@dataclass
class EvalConfig:
    output_dir: Path
    num_runs: int = 6
    asr_model: str = "large-v3"
    judge_model: str = JUDGE_MODEL
    device: str = "cuda"


def _transcribe(audio_path: str, device: str, model_name: str) -> str:
    try:
        import whisper

        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(audio_path, language="ja")
        return result.get("text", "").strip()
    except Exception as e:
        return f"[ASR ERROR: {e}]"


def _synthesize_question(text: str, output_path: str, j_moshi_repo: str, device: str) -> bool:
    """Synthesize a single utterance as speaker A using J-Moshi TTS."""
    import subprocess

    turns = [{"speaker": "A", "text": text}]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(turns, f, ensure_ascii=False)
        tmp_input = f.name

    cmd = [
        "python", "-m", "jmoshi.tts",
        "--repo", j_moshi_repo,
        "--input", tmp_input,
        "--output", output_path,
        "--seed", "0",
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp_input)
    return result.returncode == 0


async def _query_system_ws(
    ws_url: str,
    question_audio_path: str,
    timeout: float = 30.0,
) -> tuple[bytes, float]:
    """Send question audio over WebSocket and collect response audio.

    Returns (response_audio_bytes, latency_ms).
    This is a simplified stub; the actual KAME WebSocket protocol is handled
    by the kame package client utilities.
    """
    try:
        import aiohttp
        import soundfile as sf

        audio, sr = sf.read(question_audio_path)
        audio_bytes = audio.tobytes()

        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Handshake
                await ws.receive_bytes(timeout=5)
                # Send audio
                await ws.send_bytes(audio_bytes)
                # Collect response
                response_chunks: list[bytes] = []
                deadline = time.time() + timeout
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.BINARY and msg.data:
                            response_chunks.append(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                    except asyncio.TimeoutError:
                        if response_chunks:
                            break

        latency_ms = (time.time() - t0) * 1000
        return b"".join(response_chunks), latency_ms
    except Exception as e:
        return b"", 0.0


def _judge_response(
    question: str,
    response: str,
    client,
    judge_model: str,
) -> float:
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"質問: {question}\n\n応答: {response}",
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        import re

        m = re.search(r"スコア[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            return float(m.group(1))
        nums = re.findall(r"\b([0-9]|10)\b", text)
        if nums:
            return float(nums[-1])
    except Exception as e:
        print(f"[WARN] Judge failed: {e}")
    return 0.0


def run_evaluation(
    systems: dict[str, str],
    questions: list[dict],
    cfg: EvalConfig,
    j_moshi_repo: str,
) -> list[TurnResult]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for LLM-as-a-Judge")
    judge_client = OpenAI(api_key=api_key)

    results: list[TurnResult] = []

    for q in questions:
        for turn_num, turn_key in enumerate(["turn1", "turn2"], start=1):
            question_text = q[turn_key]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                q_audio_path = tmp.name

            ok = _synthesize_question(question_text, q_audio_path, j_moshi_repo, cfg.device)
            if not ok:
                print(f"[WARN] TTS failed for {q['question_id']} turn{turn_num}")
                os.unlink(q_audio_path)
                continue

            for system_name, ws_url in systems.items():
                run_scores: list[float] = []
                run_latencies: list[float] = []

                for run_idx in range(cfg.num_runs):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        resp_audio_path = tmp.name

                    resp_bytes, latency_ms = asyncio.run(
                        _query_system_ws(ws_url, q_audio_path, timeout=30.0)
                    )

                    if resp_bytes:
                        with open(resp_audio_path, "wb") as f:
                            f.write(resp_bytes)
                        resp_text = _transcribe(resp_audio_path, cfg.device, cfg.asr_model)
                    else:
                        resp_text = ""
                        latency_ms = 30000.0
                    os.unlink(resp_audio_path)

                    score = _judge_response(
                        question_text, resp_text, judge_client, cfg.judge_model
                    )
                    run_scores.append(score)
                    run_latencies.append(latency_ms)
                    print(
                        f"  [{system_name}] {q['question_id']} t{turn_num} run{run_idx}: "
                        f"score={score:.1f} lat={latency_ms:.0f}ms text={resp_text[:60]!r}"
                    )

                result = TurnResult(
                    question_id=q["question_id"],
                    category=q["category"],
                    turn=turn_num,
                    system=system_name,
                    question_text=question_text,
                    response_text=resp_text,
                    score=statistics.mean(run_scores),
                    latency_ms=statistics.median(run_latencies),
                )
                results.append(result)

            os.unlink(q_audio_path)

    return results


def _print_summary(results: list[TurnResult]) -> None:
    from collections import defaultdict

    by_system: dict[str, list[float]] = defaultdict(list)
    by_system_cat: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_system_latency: dict[str, list[float]] = defaultdict(list)

    for r in results:
        by_system[r.system].append(r.score)
        by_system_cat[(r.system, r.category)].append(r.score)
        by_system_latency[r.system].append(r.latency_ms)

    print("\n=== MT-Bench Score Summary ===")
    header = f"{'System':<20} {'reasoning':>10} {'stem':>10} {'humanities':>10} {'avg':>8} {'latency(ms)':>12}"
    print(header)
    print("-" * len(header))
    for system in sorted(by_system):
        scores_by_cat = {
            cat: statistics.mean(by_system_cat.get((system, cat), [0.0]))
            for cat in MT_BENCH_JA_CATEGORIES
        }
        avg = statistics.mean(by_system[system])
        lat = statistics.median(by_system_latency[system])
        print(
            f"{system:<20} "
            f"{scores_by_cat['reasoning']:>10.2f} "
            f"{scores_by_cat['stem']:>10.2f} "
            f"{scores_by_cat['humanities']:>10.2f} "
            f"{avg:>8.2f} "
            f"{lat:>12.0f}"
        )


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    if args.questions_json:
        with open(args.questions_json, encoding="utf-8") as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS
    print(f"Evaluating {len(questions)} questions x {len(args.systems)} systems x {args.num_runs} runs")

    # Build system map
    systems: dict[str, str] = {}
    for system_name in args.systems:
        url_attr = f"{system_name}_url"
        url = getattr(args, url_attr, None)
        if not url:
            raise ValueError(f"--{url_attr} is required when evaluating system '{system_name}'")
        systems[system_name] = url

    cfg = EvalConfig(
        output_dir=output_dir,
        num_runs=args.num_runs,
        asr_model=args.asr_model,
        judge_model=args.judge_model,
        device=args.device,
    )

    results = run_evaluation(systems, questions, cfg, j_moshi_repo=args.j_moshi_repo)

    # Save results
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Save CSV
    import csv

    csv_path = output_dir / "results.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
    print(f"CSV saved to {csv_path}")

    _print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Japanese Speech MT-Bench evaluation for J-KAME."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["baseline", "kame_jp"],
        help="System names to evaluate.",
    )
    parser.add_argument("--baseline_url", type=str, default="ws://localhost:8998/api/chat")
    parser.add_argument("--kame_jp_url", type=str, default="ws://localhost:9000/api/chat")
    parser.add_argument(
        "--questions_json",
        type=str,
        default=None,
        help="Path to a custom questions JSON file. Uses built-in questions if omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/japanese_kame",
    )
    parser.add_argument("--num_runs", type=int, default=6)
    parser.add_argument("--asr_model", type=str, default="large-v3")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--j_moshi_repo", type=str, default="nu-dialogue/j-moshi-ext")
    main(parser.parse_args())
