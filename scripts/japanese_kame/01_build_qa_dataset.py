"""Build a Japanese Q&A dataset from public HuggingFace datasets.

Collects QA pairs from JMMLU, MGSM-ja, JCommonsenseQA, JaQuAD, and AIO, then
writes them to data/japanese_kame/qa_pairs/ as JSONL files.

Each line: {"question": "...", "answer": "...", "source": "...", "category": "..."}

Usage:
    uv run --extra data -m scripts.japanese_kame.01_build_qa_dataset \
        --output_dir data/japanese_kame/qa_pairs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_jmmlu(output_dir: Path) -> int:
    from datasets import get_dataset_config_names, load_dataset

    out_path = output_dir / "jmmlu.jsonl"
    count = 0
    try:
        subjects = get_dataset_config_names("nlp-waseda/JMMLU")
    except Exception as e:
        print(f"[JMMLU] Failed to get config names: {e}")
        print(f"[JMMLU] 0 records -> {out_path}")
        out_path.write_text("")
        return 0

    with out_path.open("w", encoding="utf-8") as f:
        for subject in subjects:
            try:
                ds = load_dataset("nlp-waseda/JMMLU", subject, split="test", trust_remote_code=True)
            except Exception as e:
                print(f"[JMMLU] Skipping subject {subject}: {e}")
                continue
            for row in ds:
                answer_key = row["answer"]
                answer_text = row.get(answer_key, answer_key)
                record = {
                    "question": row["question"],
                    "answer": answer_text,
                    "source": "JMMLU",
                    "category": subject,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"[JMMLU] {count} records -> {out_path}")
    return count


def _load_mgsm_ja(output_dir: Path) -> int:
    from datasets import load_dataset

    out_path = output_dir / "mgsm_ja.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        try:
            ds = load_dataset("juletxara/mgsm", "ja", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"[MGSM-ja] Failed: {e}")
            return 0
        for row in ds:
            record = {
                "question": row["question"],
                "answer": str(row["answer_number"]),
                "source": "MGSM-ja",
                "category": "math",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"[MGSM-ja] {count} records -> {out_path}")
    return count


def _load_jcommonsenseqa(output_dir: Path) -> int:
    from datasets import load_dataset

    out_path = output_dir / "jcommonsenseqa.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split in ("train", "validation"):
            try:
                ds = load_dataset("sbintuitions/JCommonsenseQA", split=split)
            except Exception as e:
                print(f"[JCommonsenseQA] Skipping split {split}: {e}")
                continue
            for row in ds:
                label = int(row["label"])
                choices = [row[f"choice{i}"] for i in range(5)]
                record = {
                    "question": row["question"],
                    "answer": choices[label] if label < len(choices) else str(label),
                    "source": "JCommonsenseQA",
                    "category": "commonsense",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"[JCommonsenseQA] {count} records -> {out_path}")
    return count


def _load_jaquad(output_dir: Path) -> int:
    from datasets import load_dataset

    out_path = output_dir / "jaquad.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split in ("train", "validation"):
            try:
                ds = load_dataset(
                    "SkelterLabsInc/JaQuAD", split=split, trust_remote_code=True
                )
            except Exception as e:
                print(f"[JaQuAD] Skipping split {split}: {e}")
                continue
            for row in ds:
                question = row["question"]
                answers = row.get("answers", {})
                answer_texts = answers.get("text", [])
                answer = answer_texts[0] if answer_texts else ""
                if not answer:
                    continue
                record = {
                    "question": question,
                    "answer": answer,
                    "source": "JaQuAD",
                    "category": "reading_comprehension",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"[JaQuAD] {count} records -> {out_path}")
    return count



LOADERS = {
    "jmmlu": _load_jmmlu,
    "mgsm_ja": _load_mgsm_ja,
    "jcommonsenseqa": _load_jcommonsenseqa,
    "jaquad": _load_jaquad,
}


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources if args.sources else list(LOADERS)
    total = 0
    for name in sources:
        if name not in LOADERS:
            print(f"[WARN] Unknown source: {name}, skipping")
            continue
        total += LOADERS[name](output_dir)

    print(f"\nTotal QA pairs collected: {total}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Japanese QA datasets from HuggingFace Hub."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/japanese_kame/qa_pairs",
        help="Directory to write JSONL output files.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=list(LOADERS),
        default=None,
        help=f"Datasets to load. Default: all ({', '.join(LOADERS)}).",
    )
    main(parser.parse_args())
