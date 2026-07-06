# CLAUDE.md — 日本語版 KAME 開発の作業ガイド

このリポジトリは **KAME（oracle 付き全二重 S2S 対話モデル）の日本語版**を作るためのファインチューニング・ワークフロー。
あなた（Claude Code）は**研究助手**として、データ作成・学習（ABCI）・変換・推論（kiso）・デバッグ・資料作成を一貫して支援する。

> 詳細な推論/デバッグ手順は [docs/kame_inference_debug.md](docs/kame_inference_debug.md) に分離。本書は全体像と運用規約。

---

## 0. 何を作っているか
- ベース：`nu-dialogue/j-moshi-ext`（日本語 Moshi、text+音声 Mimi 8 コードブック、全二重）。
- KAME 拡張：**第4ストリーム = oracle**（外部 LLM が「次に言うべき発話」を実時間予測し注入）。`tools.init_moshi_for_ft` で `j-moshi-ext` を 4 ストリーム化（`oracle_emb` を `text_emb` からバックフィル）。
- ゴール：QA 由来の知識を、oracle 経由で実時間注入できる日本語音声対話モデル。

## 1. 実行環境（2台体制・最重要）
- **kiso**（このマシン、hostname `g21`）：`/mnt/kiso-qnap4/jsato/moshi_kame_finetune`。GPU = A5000 24GB ×2。
  - 用途：**データ作成（oracle 生成・tokenize）と推論サーバ**。
  - 既存の vLLM `:8000`（`llm-jp/llm-jp-4-8b-thinking`、ユーザ所有）が常駐＝oracle バックエンドに流用。
  - `uv` は `~/.local/bin/uv`。Python 3.12、torch cu121。
- **ABCI 3.0**（LLM-jp）：**学習専用**。group `gcg51557`、実験番号 **0374**、作業ディレクトリ `/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune`。1 ノード = H200×8。
  - 予約キュー `R9920261000`（年度依存、`qrstat -f` で確認）。`qsub` 共通: `-P gcg51557 -q R9920261000 -v RTYPE=rt_HF`。ジョブ名は `0374_...`。
  - 詳細・ハマりどころは `~/.claude` のメモ「ABCI training workflow」を参照。
- **ファイル転送 kiso↔ABCI**：直結不可（kiso に ABCI 鍵なし、ABCI 計算ノードは直 SSH 不可）。**手元 Mac を中継**して 2 段 rsync（Mac の ssh エイリアス: `g21`=kiso, `abci`=ABCI）。

## 2. データ作成パイプライン（kiso、`scripts/japanese_kame/`）
00 init → 01 QA収集 → 02 対話化 → 03 音声合成 → 04 アライン → 05 oracle → 06/06b tokenize。
- 01 QA収集：JaQuAD / JCommonsenseQA / JMMLU / MGSM-ja → `data/japanese_kame/qa_pairs/`。
- 02 対話化：LLM（`llm-jp-4-8b-thinking` 既定 / `gpt-4.1-mini`）で 2 話者対話化 → `dialogues/`。
- 03 音声合成：**Google Gemini TTS**（`gemini-3.1-flash-tts-preview`、`03_synthesize_speech_via_api.py`）。話者 A/B 個別合成 → ステレオ WAV（L=A,R=B,24kHz）→ `audio_gemini/`。要 `GEMINI_API_KEY`。
- 04 アライン：強制アラインメントで単語タイムスタンプ → `0610/text/`。
- 05 oracle：ローカル vLLM（`llm-jp-4-8b-thinking`）で 0.5 秒間隔に次発話予測 → `0610/oracle_raw/`。thinking の reasoning は除去（`tools/generate_oracle_from_text.py` の `strip_reasoning`）。
- 06b tokenize（段階運用）：[scripts/japanese_kame/06b_tokenize_batch.sh](scripts/japanese_kame/06b_tokenize_batch.sh)（symlink ステージング・冪等・**A/B両話者フィルタ**）と `06b_run_all.sh`。
  - 制約：`tools/prepare_dataset.py` は **text==audio 集合一致 & oracle⊇text** を要求 → バッチごとに揃えて parquet 化。
  - 出力：`processed_data/japanese_kame/0610/batch_*-*.parquet`。各行 = A/B それぞれ `[9,T]`（1 text+8 音声）+ oracle イベント列。
- GPU が要るのは音声 tokenize（Mimi）のみ。text/oracle/assemble は CPU。

## 3. 学習（ABCI）
- 初期化：`bash scripts/japanese_kame/00_init_j_moshi.sh`（CPU/オンライン）。
- 起動はバッチジョブ（`train.pbs`、`qsub`）。ジョブスクリプトに必ず：
  ```bash
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load cuda/12.1                 # DeepSpeed の CUDA_HOME 用（torch cu121 と一致）
  export PATH="$HOME/.local/bin:$PATH"  # uv
  export WANDB_MODE=offline
  ```
- 本体：`examples/finetune_j_moshi.sh`（`finetune.py` を accelerate+DeepSpeed ZeRO-3 bf16 で起動）。env override：`TRAIN_DATA_GLOB / OUTPUT_DIR / NUM_PROCESSES=8 / NUM_EPOCHS / PER_DEVICE_BATCH_SIZE / SAVE_STEPS / USE_ORACLE`。
- 主要ハイパラ：lr 3e-5、max_length 512、loss 重み semantic 100 / acoustic 1 / text_padding 0.5、oracle augmentation（`--oracle_skip_prob_min/max` 既定 0.1/0.7、shift、jitter、`--oracle_embedding_mode separate`）。
- 注意：`examples/finetune_j_moshi.sh` は `MAX_TRAIN_STEPS` 非対応。スモークは**小さい parquet を 1 つ**だけ対象にして短く回す。
- 監視：`qstat`（`R` で実行中）、標準出力は完了時に `<job>.o<id>` / 進捗は `.e<id>`、または出力ディレクトリの `step_*` 増加。バッチジョブは PC を閉じてよい。

## 4. 変換 → 推論
- 変換（ABCI、計算ノードで `module load cuda/12.1` 後）：`tools.zero_to_fp32`（ZeRO3→fp32）→ `tools.clean_moshi`（`--model_dtype bfloat16`）→ `<step>_cleaned/`（`model.safetensors`+`moshi_lm_kwargs.json`、約 15GB）。
- 転送：cleaned を Mac 経由で kiso へ。
- 推論（kiso、GPU1）：`kame_jp.server_oracle_jp`。env `OPENAI_API_KEY=EMPTY`・`GOOGLE_APPLICATION_CREDENTIALS=<llmjp-*.json>`・`CUDA_VISIBLE_DEVICES=1`。oracle は kiso の vLLM `:8000` を利用。`127.0.0.1` バインドなので **Mac から `ssh -L 8998:localhost:8998 g21`** してブラウザで対話。
- ベース比較：`kame.server`（oracle/ASR/vLLM 不要）で素の `j-moshi-ext` を起動可能。
- 具体コマンド・ログの読み方は [docs/kame_inference_debug.md](docs/kame_inference_debug.md)。

## 5. ハマりどころ（再発防止）
- thinking モデルは `analysis … assistant final …` 形式 → reasoning を必ず除去（生成側 `strip_reasoning`、推論側 `parse_harmony_response`）。
- 推論サーバは `OPENAI_API_KEY` env 必須（ローカル vLLM でも）。
- ASR は GCP **Cloud Speech-to-Text API 有効化**必須（無効だと `SERVICE_DISABLED`・0 語）。
- ABCI：GPU 処理は**計算ノード（`hnode*`）**で（ログインノードは GPU 無し＆重い処理を SIGKILL）。インタラクティブは `qsub -I ...`、入ったら `hostname` で確認。毎回 `module load cuda/12.1` と uv PATH。
- 推論は localhost バインド＋SSHトンネル。繋がらない時はまず kiso 側 `ss -ltn|grep 8998`・`curl localhost:8998`→生きていればトンネル張り直し。
- ツール経由で起動した長時間プロセス/サーバ/tmux はセッション終了で回収され得る。**長期運用はユーザ自身の tmux** で。

## 6. 現状（2026-06 時点の成果物）
- データ：39,043 アライン済み → 2 話者フィルタで **38,783 対話** → parquet（5 本、~223MB）。
- 学習：実験 0374。①1 epoch（2,336 step, loss≈3.1, `output/0374_kame-0610`）→出力崩壊。②5 epoch（11,680 step, loss≈0.99, `output/0374_kame-0610-e5/step_11680`）→流暢化。cleaned 済み。
- 推論：配管完成。oracle は正しく注入（例 3+3→6 / 鎌倉→1185 年）。
- 既知課題：**oracle 非追従**（ISSUE-02）、**開幕の自発話/役割ミスマッチ**（ISSUE-01）。詳細・根本原因・候補手は [docs/issues.md](docs/issues.md)。
- 進行中：EXP-002（`--moshi_speakers B` 再学習で ISSUE-01 に対処）。run の状態は [docs/experiments.md](docs/experiments.md)。
- 次の打ち手：oracle を speaking 区間へ重畳（EXP-003, ISSUE-02 最重要手）。

## 7. 進め方の規約（ユーザの好み）
- **ABCI 操作は一歩ずつ**確認しながら（初学者前提で `hostname`・`qstat` 等の確認を挟む）。
- **破壊的・外部影響のある操作は事前確認**：特にユーザ所有の vLLM `:8000` や他人の ABCI ジョブには触れない。GPU を空ける時は対象を明示して確認。
- **長時間ジョブ**：学習は `qsub`（PC 閉じてよい）、推論サーバはユーザ tmux。
- **資料作成**：ゼミ用は**事実ベースで簡潔**。推測・誇張を避け、数値は確定値を使う。スライド骨子は「メインメッセージ／サブメッセージ／ボディ／伝えたいこと（口頭）」形式。
- 機密：`llmjp-*.json`（GCP 鍵）はコミットしない（`.gitignore` 済みが望ましい）。
- 既存ツールは無改変で再利用を優先。段階運用が要る所だけ薄いラッパー（例 `06b_*`）を足す。

## 8. 参照
- **課題台帳**：[docs/issues.md](docs/issues.md)（症状→根本原因→候補手。PDCA の Plan/Act 本体）
- **実験台帳**：[docs/experiments.md](docs/experiments.md)（学習 run を仮説→予測→結果→次手で管理）
- **学習パイプライン**：[docs/training_pipeline.md](docs/training_pipeline.md)（parquet 1 行 → 1 ステップを ①〜⑩ で解説。課題の発生箇所つき）
- 推論・分析の詳細：[docs/kame_inference_debug.md](docs/kame_inference_debug.md)
- パイプライン：[scripts/japanese_kame/](scripts/japanese_kame/)、`README.md`
- `~/.claude` メモ：「ABCI training workflow」「0610 dataset」
