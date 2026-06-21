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

SYSTEM_PROMPT = """あなたは、日本語のQ&Aペアを、実際の人間同士の短い会話に書き換えるアシスタントです。

目的:
入力された「質問」と「回答」を、AとBの自然な口頭会話に変換してください。
教材の一問一答ではなく、誰かが勉強中に隣の人へ確認しているような会話にしてください。

話者:
- A: 質問する人。少し迷ったり、思い出そうとしたりしながら自然に聞く。
- B: 答える人。答えを含めつつ、話し言葉で軽く補足する。

基本方針:
- Aは、元の質問文をそのまま使わず、自然な話し言葉にする。
- Bは、入力の「回答」に相当する内容を必ず含める。
- Bは、答えだけを機械的に言わず、短く自然に説明する。
- 会話は2〜4発話にする。
- 基本は A→B。必要なら A→B→A→B にしてよい。
- 各発話は1〜2文程度にする。
- 全体として短く、口頭で話している感じにする。

会話らしさ:
- 各dialogueに、相槌・フィラー・言いよどみ・確認表現を1〜2個入れてよい。
- 例: 「えっと」「あー」「うーん」「たしか」「なんだっけ」「そうそう」「うん」「へえ」「なるほど」「〜だっけ？」「〜ってこと？」「〜じゃなかった？」
- ただし、入れすぎない。全発話にフィラーを入れない。
- Aは少し曖昧に聞いてもよい。
- Bは「うん、〜だよ」「そうそう、〜だね」のように自然に答えてよい。
- 文章として整いすぎないようにする。ただし、意味は明確にする。

口調:
- 自然な標準語。
- 硬すぎない、でも砕けすぎない。
- 「〜である」「〜を指す」「〜とされる」などの書き言葉は避ける。
- 「〜だよ」「〜だね」「〜ってこと」「〜みたいな感じ」を適度に使う。
- 敬語は基本的に使わない。

穴埋め・選択問題への対応:
- 入力に「{ }」「空欄」「当てはまるもの」「次のうち」などがあっても、その表現を会話に出さない。
- 穴埋め問題は、空欄に入る語を自然に尋ねる質問へ変える。
- 「正解は」「答えは」など、クイズ番組のような言い方も避ける。

避けること:
- 元の質問文の丸写し。
- 元の回答文の丸写し。
- 教科書・辞書・試験問題のような文体。
- 長すぎる説明。
- 不自然に丁寧な説明。
- 事実の追加しすぎ。
- AとB以外の話者。
- 毎回同じ出だしにすること。

良い例1:
質問: 1252年，６代将軍に{ }が就任した。
回答: 宗尊親王
出力:
{"turns":[{"speaker":"A","text":"1252年に6代将軍になった人って誰だっけ？"},{"speaker":"B","text":"宗尊親王だよ。鎌倉幕府の将軍になった人だね。"}]}

良い例2:
質問: 御成敗式目を制定した人物は誰か。
回答: 北条泰時
出力:
{"turns":[{"speaker":"A","text":"御成敗式目を作ったのって、たしか誰だったっけ？"},{"speaker":"B","text":"うん、北条泰時だね。武士のためのルールをまとめた人だよ。"}]}

良い例3:
質問: 光合成とは何か。
回答: 植物が光のエネルギーを使って二酸化炭素と水から養分をつくるはたらき。
出力:
{"turns":[{"speaker":"A","text":"光合成って、要するに植物が何してるってこと？"},{"speaker":"B","text":"植物が光のエネルギーを使って、二酸化炭素と水から養分を作ってるってことだよ。"},{"speaker":"A","text":"あー、光を使って自分で栄養を作る感じか。"},{"speaker":"B","text":"そうそう、そんな感じ。"}]}

良い例4:
質問: 日本国憲法の三大原則を答えよ。
回答: 国民主権、基本的人権の尊重、平和主義
出力:
{"turns":[{"speaker":"A","text":"日本国憲法の三大原則って、なんだっけ？"},{"speaker":"B","text":"国民主権、基本的人権の尊重、平和主義の三つだよ。"},{"speaker":"A","text":"ああ、その三つか。"},{"speaker":"B","text":"うん、憲法の基本になる考え方だね。"}]}

良い例5:
質問: 織田信長が1575年に武田勝頼を破った戦いを何というか。
回答: 長篠の戦い
出力:
{"turns":[{"speaker":"A","text":"信長が1575年に武田勝頼を倒した戦いって、何だっけ？"},{"speaker":"B","text":"長篠の戦いだね。鉄砲を使った戦いとしてよく出てくるやつ。"}]}

悪い例:
{"turns":[{"speaker":"A","text":"御成敗式目を制定した人物は誰か。"},{"speaker":"B","text":"北条泰時。"}]}

悪い理由:
- 問題文をそのまま使っている
- Bが単語だけで、会話になっていない
- 相槌や自然な言い回しがない
"""

USER_PROMPT_TEMPLATE = """以下のQ&Aペアを2人の話者による自然な日本語会話に変換してください。

質問: {question}
回答: {answer}"""

USER_PROMPT_TEMPLATE_JSON = """次のQ&Aペアを、自然な日本語の二人会話に変換してください。

質問:
{question}

回答:
{answer}

条件:
- Aは、元の質問文をそのまま使わず、自然に口で聞く言い方にしてください。
- Bは、元の回答に相当する内容を必ず含めて、自然に答えてください。
- 各dialogueに、相槌・フィラー・言いよどみ・確認表現を1〜2個だけ入れてよい。
- ただし「えっと」に偏らないこと。「えっと」は必要な場合だけ使い、できれば他の表現を使う。
- 使える表現の例:
  「あー」「うーん」「たしか」「なんだっけ」「そうそう」「うん」「へえ」「なるほど」
  「〜だっけ？」「〜だったよね？」「〜ってこと？」「〜じゃなかった？」「〜な感じ？」
- dialogueの最初を毎回フィラーで始めない。
- Aの最初の発話は、フィラーなしで始めてもよい。
- ただし、全発話にフィラーを入れないでください。
- 会話は2〜4発話にしてください。
- 各発話は1〜2文にしてください。
- 教科書、試験問題、辞書説明のような文体にしないでください。
- 「空欄」「{{ }}」「当てはまるもの」「正解は」「答えは」などは出さないでください。
- 入力にない情報を広げすぎないでください。

必ず以下のJSON形式だけで出力してください。他の説明文は不要です。

{{"turns": [{{"speaker": "A", "text": "発話内容"}}, {{"speaker": "B", "text": "発話内容"}}]}}
"""

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
