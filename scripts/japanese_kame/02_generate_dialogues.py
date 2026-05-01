"""Convert Japanese QA pairs into two-speaker conversational dialogues using an LLM.

Reads JSONL files from a QA pairs directory and produces dialogue JSON files
in data/japanese_kame/dialogues/.  Each output file contains a list of turn
dicts: [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}, ...]

Speaker A is the questioner; speaker B is the answerer.

Usage:
    export OPENAI_API_KEY=...
    uv run --extra oracle -m scripts.japanese_kame.02_generate_dialogues \
        --input_dir  data/japanese_kame/qa_pairs \
        --output_dir data/japanese_kame/dialogues \
        --model gpt-4.1-mini \
        --max_workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYSTEM_PROMPT = """あなたは自然な日本語の会話を生成するアシスタントです。
Q&Aペアを受け取り、2人の話者（AとB）による口語的な会話に変換してください。
- AはBに質問する（カジュアルで自然な言い方で）
- Bは質問に答える（口語的・わかりやすく）
- 必要に応じてAがフォローアップの質問をして2ターンに展開してください
- 書き言葉は使わず、実際の会話らしい表現を使うこと
- 各発話は1〜3文程度に収める
出力形式: JSON配列 [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}, ...]
JSON以外のテキストは出力しないこと。"""

USER_PROMPT_TEMPLATE = """以下のQ&Aペアを2人の話者による自然な日本語会話に変換してください。

質問: {question}
回答: {answer}

JSON配列のみを出力してください。"""

_print_lock = threading.Lock()


def generate_dialogue(
    qa: dict,
    client,
    model: str,
) -> list[dict[str, str]]:
    prompt = USER_PROMPT_TEMPLATE.format(
        question=qa["question"],
        answer=qa["answer"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        stream=False,
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    # some models wrap the array in a key
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []


def process_file(
    jsonl_path: Path,
    output_dir: Path,
    client,
    model: str,
    resume: bool,
) -> tuple[int, int]:
    success = 0
    fail = 0
    with jsonl_path.open(encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    for i, qa in enumerate(qa_pairs):
        out_id = f"{jsonl_path.stem}_{i:06d}"
        out_path = output_dir / f"{out_id}.json"
        if resume and out_path.exists():
            success += 1
            continue
        try:
            turns = generate_dialogue(qa, client, model)
            if not turns:
                fail += 1
                continue
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(turns, f, ensure_ascii=False, indent=2)
            success += 1
        except Exception as e:
            with _print_lock:
                print(f"[WARN] {out_id}: {e}")
            fail += 1
    return success, fail


def main(args: argparse.Namespace) -> None:
    from openai import OpenAI

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(
        api_key=api_key,
        base_url=args.llm_base_url or None,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    print(f"Found {len(jsonl_files)} JSONL source files")

    total_success = 0
    total_fail = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(process_file, p, output_dir, client, args.model, args.resume): p
            for p in jsonl_files
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                s, f = future.result()
                total_success += s
                total_fail += f
                with _print_lock:
                    print(f"[OK] {path.name}: {s} ok, {f} failed")
            except Exception as e:
                with _print_lock:
                    print(f"[ERROR] {path.name}: {e}")

    print(f"\nTotal: {total_success} dialogues generated, {total_fail} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Japanese QA pairs to two-speaker dialogues using an LLM."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/japanese_kame/qa_pairs",
        help="Directory containing JSONL QA pair files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/japanese_kame/dialogues",
        help="Directory to write dialogue JSON files.",
    )
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of parallel LLM API threads.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-generated files.")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key override.")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="",
        help="Custom LLM base URL (e.g. vLLM endpoint).",
    )
    main(parser.parse_args())
