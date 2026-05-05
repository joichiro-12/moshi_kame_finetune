"""Convert Japanese QA pairs into two-speaker conversational dialogues using an LLM.

Reads JSONL files from a QA pairs directory and produces dialogue JSON files
in data/japanese_kame/dialogues/.  Each output file contains a list of turn
dicts: [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}, ...]

Speaker A is the questioner; speaker B is the answerer.

Usage:
    export OPENAI_API_KEY=...
    uv run --extra oracle -m scripts.japanese_kame.02_generate_dialogues \
        --model    llm-jp/llm-jp-4-8b-thinking \
        --llm_base_url http://localhost:8000/v1 \
        --no_strict_format \
        --max_workers 2 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYSTEM_PROMPT = """あなたは自然な日本語の会話を生成するアシスタントです。
Q&Aペアを受け取り、2人の話者（AとB）による口語的な会話に変換してください。
- AはBに質問する（カジュアルで自然な言い方で）
- Bは質問に答える（口語的・わかりやすく）
- 必要に応じてAがフォローアップの質問をして2ターンに展開してください
- 書き言葉は使わず、実際の会話らしい表現を使うこと
- 入力が「{ }」「空欄」「当てはまるもの」などの穴埋め問題でも、その表現を会話に出さないこと
- 穴埋め問題は、空欄に入る語を自然に尋ねる質問へ言い換えること
  例: 「1252年，６代将軍に{ }が就任した」なら「1252年に6代将軍になったのって誰？」のようにする
- 各発話は1〜3文程度に収める"""

USER_PROMPT_TEMPLATE = """以下のQ&Aペアを2人の話者による自然な日本語会話に変換してください。

質問: {question}
回答: {answer}"""

USER_PROMPT_TEMPLATE_JSON = """以下のQ&Aペアを2人の話者による自然な日本語会話に変換してください。

質問: {question}
回答: {answer}

必ず以下のJSON形式で出力してください（他のテキストは不要）:
{{"turns": [{{"speaker": "A", "text": "発話内容"}}, {{"speaker": "B", "text": "発話内容"}}]}}"""

_RESPONSE_FORMAT_STRICT = {
    "type": "json_schema",
    "json_schema": {
        "name": "dialogue",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["speaker", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["turns"],
            "additionalProperties": False,
        },
    },
}

_RESPONSE_FORMAT_JSON = {"type": "json_object"}

_print_lock = threading.Lock()


def _extract_json(text: str) -> str:
    """Extract the JSON object containing 'turns' from model output.

    Handles models that prepend free-form thinking text to the JSON answer,
    with or without <think>...</think> tags.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find the JSON object that contains "turns" using brace-depth tracking
    m = re.search(r'\{[^{]*"turns"', text, re.DOTALL)
    if m:
        start = m.start()
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    # Fallback: find any JSON-like object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def generate_dialogue(
    qa: dict,
    client,
    model: str,
    use_strict_format: bool = True,
) -> list[dict[str, str]]:
    template = USER_PROMPT_TEMPLATE if use_strict_format else USER_PROMPT_TEMPLATE_JSON
    prompt = template.format(
        question=qa["question"],
        answer=qa["answer"],
    )
    if use_strict_format:
        kwargs: dict = {"response_format": _RESPONSE_FORMAT_STRICT}
    else:
        # json_object guided decoding on this model produces garbled output;
        # omit response_format and extract JSON from free-text response instead.
        kwargs = {"extra_body": {"reasoning_effort": "low"}}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        **kwargs,
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return []
    return json.loads(_extract_json(raw))["turns"]


def process_file(
    jsonl_path: Path,
    output_dir: Path,
    client,
    model: str,
    resume: bool,
    use_strict_format: bool = True,
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
        for attempt in range(3):
            try:
                turns = generate_dialogue(qa, client, model, use_strict_format=use_strict_format)
                if not turns:
                    raise ValueError("empty turns")
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(turns, f, ensure_ascii=False, indent=2)
                success += 1
                break
            except Exception as e:
                if attempt < 2:
                    with _print_lock:
                        print(f"[RETRY {attempt + 1}/3] {out_id}: {e}")
                else:
                    with _print_lock:
                        print(f"[WARN] {out_id}: {e}")
                    fail += 1
    return success, fail


def main(args: argparse.Namespace) -> None:
    from openai import OpenAI

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    use_strict_format = not args.no_strict_format

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
            pool.submit(process_file, p, output_dir, client, args.model, args.resume, use_strict_format): p
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
        default=2,
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
    parser.add_argument(
        "--no_strict_format",
        action="store_true",
        help="Use json_object instead of json_schema (needed for vLLM / local models).",
    )
    main(parser.parse_args())
