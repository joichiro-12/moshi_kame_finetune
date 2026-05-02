# Japanese KAME (J-KAME) Pipeline

End-to-end pipeline for training and serving a Japanese KAME system using
`nu-dialogue/j-moshi-ext` as the front-end S2S model and
`llm-jp/llm-jp-4-8b-thinking` as the back-end oracle LLM.

---

## スクリプト一覧

| ステップ | スクリプト | 内容 |
|---------|-----------|------|
| 0 | `00_init_j_moshi.sh` | J-Moshi を 4-stream モデルとして初期化 |
| 1 | `01_build_qa_dataset.py` | 日本語 Q&A データ収集 (JMMLU / MGSM-ja / JCommonsenseQA / JaQuAD / AIO) |
| 2 | `02_generate_dialogues.py` | Q&A → 2話者対話変換 (gpt-4.1-mini) |
| 3 | `03_synthesize_speech.py` | 対話テキスト → ステレオ WAV (J-Moshi kame-model offline TTS) |
| 4 | `04_word_alignment.py` | WhisperX word-level timestamp 生成 |
| 5 | `05_generate_oracle.sh` | oracle_raw 生成 (generate_oracle_from_text --language ja) |
| 6 | `06_tokenize_and_prepare.sh` | トークナイズ + parquet 組み立て |
| - | `evaluate.py` | Japanese Speech MT-Bench 評価 (LLM-as-a-Judge) |

Fine-tuning: `examples/finetune_j_moshi.sh`
推論サーバー: `kame_jp/server_oracle_jp.py`

---

## 確認済み事項（実行前に要確認だったもの）

### 1. テキストトークナイザーファイル名

```
nu-dialogue/j-moshi-ext のファイル一覧:
  .gitattributes
  README-en.md
  README.md
  model.safetensors
  tokenizer-e351c8d8-checkpoint125.safetensors
  tokenizer_spm_32k_3.model   ← ← ← これが正しいファイル名
```

`06_tokenize_and_prepare.sh` の `TEXT_TOKENIZER_NAME` は
**`tokenizer_spm_32k_3.model`** に設定済み（末尾の `_3` が重要）。
計画書の `tokenizer_spm_32k.model` は誤りだったため修正した。

### 2. J-Moshi の TTS インタフェース

`nu-dialogue/j-moshi` リポジトリにはスタンドアロンの `jmoshi.tts` CLI **は存在しない**。

- 公式のインタラクティブデモ起動は `python -m moshi.server --hf-repo nu-dialogue/j-moshi-ext`
- オフライン TTS（データ生成用）は kame-model パッケージの `LMGen` + `MimiModel` を
  直接呼び出すことで実現する（`03_synthesize_speech.py` がこの方式で実装済み）

`03_synthesize_speech.py` の概要:
```
_load_model()              # kame.models.loaders でモデルロード（プロセス内キャッシュ）
_text_to_frame_schedule()  # 対話テキスト → フレーム別トークン注入スケジュール
synthesize_with_kame()     # state.lm_gen.step() + state.mimi.decode() でフレーム毎に合成
```

### 3. vLLM の `--reasoning-parser` フラグ（LLM-jp-4 用）

**vLLM 0.19.1 には `llmjp4` reasoning parser は存在しない。**

```
$ python -c "import vllm; print(vllm.__version__)"
0.19.1
```

vLLM 0.19.1 で利用可能な reasoning parser: `deepseek_r1`, `qwen3` など。

対処方法: `--reasoning-parser` フラグは使わずに vLLM を起動し、
Harmony フォーマットのパースは **クライアント側** (`kame_jp/server_oracle_jp.py` の
`parse_harmony_response()`) で行う。

LLM-jp-4-thinking の Harmony 出力形式:
```
<|start|>...<|channel|>analysis<|message|>[Chain-of-Thought]<|channel|>final<|message|>[最終回答]<|return|>
```

`parse_harmony_response()` は `final` チャネルのみ抽出し、`analysis` チャネル（CoT）は破棄する。
`<|return|>` タグがない途中ストリームにも対応するため正規表現の末尾は `(?:<\|return\|>|$)`。

---

## vLLM 起動コマンド（参考）

```bash
# --reasoning-parser は不要（Harmony パースはクライアント側で実施）
vllm serve llm-jp/llm-jp-4-8b-thinking \
    --max-model-len 8192 \
    --port 8000
```

---

## J-KAME サーバー起動コマンド（参考）

```bash
uv run -m kame_jp.server_oracle_jp \
    --moshi-weight output/j-moshi-kame-finetuned/step_XXXXX_fp32_cleaned/model.safetensors \
    --config-path  output/j-moshi-kame-finetuned/step_XXXXX_fp32_cleaned/moshi_lm_kwargs.json \
    --tokenizer    /path/to/j-moshi-ext/tokenizer_spm_32k_3.model \
    --llm-base-url http://localhost:8000/v1 \
    --llm-model    llm-jp/llm-jp-4-8b-thinking \
    --port 8998
```

---

## 依存パッケージ

```bash
# データ生成用
uv sync --extra data

# oracle 生成用
uv sync --extra oracle

# 推論サーバー（日本語 ASR）用
uv sync --extra server-jp
```
