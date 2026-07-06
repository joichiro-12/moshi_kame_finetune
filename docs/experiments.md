# 実験台帳 (Experiments)

KAME 日本語版の学習 run を PDCA で管理する。課題は [issues.md](issues.md) と相互参照。

**使い方（PDCA 規律）**
- **Plan**: run を起こす前に「対象 Issue・仮説・変更点（baseline 差分）・予測（成功条件）」を書く。**予測は Do の前に確定する**。
- **Do**: コマンドを記録（再現可能に）。
- **Check**: 結果を予測と突き合わせ判定（◯解消 / △部分 / ✗未解消）。
- **Act**: 分かったことを [issues.md](issues.md) に反映し、次の `EXP` を起こす。

**共通設定（baseline, 特記なき限り共通）**
- ベースモデル: `init_models/j-moshi-ext-four_streams-bfloat16`（4 ストリーム化済み）
- データ: `processed_data/japanese_kame/0610/batch_*.parquet`（5 本, ~223MB, 38,783 行）
- 学習: accelerate + DeepSpeed ZeRO-3 bf16, `NUM_PROCESSES=8`, `PER_DEVICE_BATCH_SIZE=2`
- ハイパラ: lr 3e-5, max_length 512, loss 重み semantic 100 / acoustic 1 / text_padding 0.5
- oracle aug: `oracle_skip_prob` 0.1–0.7, shift, jitter, `oracle_embedding_mode separate`
- 環境: ABCI 実験 0374, queue `R9920261000`, 1 ノード H200×8

## 一覧
| ID | 日付 | 対象 | 仮説 / 変更（baseline 差分） | 予測（成功条件） | 結果 | 判定 | 次 |
|---|---|---|---|---|---|---|---|
| EXP-000 | (初期) | baseline | A-run 1 epoch | 流暢な対話 | 2,336 step, loss≈3.1, 出力崩壊 | ✗ | epoch 増 |
| EXP-001 | (e5) | baseline | A-run 5 epoch | 流暢化 | 11,680 step, loss≈0.99, 流暢／配管OK／開幕自発話・oracle非追従 | △ | →spkB, oracle改修 |
| EXP-002 | 2026-06-30 | ISSUE-01 | `--moshi_speakers B`, 5 epoch | 開幕自発話が消え、答える側になる | ✗ 開幕自発話・出題/相槌が残存・非応答, oracle非追従, 流暢性も低下。ASR/micは正常 | ✗ | データ見直し(冒頭無音・ユーザ先行)→EXP-003 |
| EXP-003 | 計画 | ISSUE-02 | oracle を speaking 区間へ重畳/持続 | oracle 追従↑（新事実が発話に出る） | – | – | – |

---

## EXP-000　baseline 1 epoch
- **対象**: —（最初の動作確認）
- **変更点**: なし（A-run, 1 epoch）
- **結果**: `output/0374_kame-0610`, 2,336 step, loss≈3.1。出力崩壊（流暢でない）。
- **判定**: ✗ 学習不足
- **Act**: epoch を増やす → EXP-001

## EXP-001　baseline 5 epoch (e5, A-run)
- **対象**: —（baseline 確立）
- **変更点**: `NUM_EPOCHS=5`（他は baseline 通り、`moshi_speakers` 未指定＝A）
- **コマンド**: `examples/finetune_j_moshi.sh`（既定）
- **結果**: `output/0374_kame-0610-e5/step_11680`, loss≈0.99, 流暢化。oracle 注入の配管は機能（例 3+3→6 / 鎌倉→1185 年）。cleaned 済み・kiso へ転送・推論確認済み。
- **判定**: △ 流暢だが 2 課題が顕在化 → ISSUE-01（開幕自発話/役割）, ISSUE-02（oracle 非追従）
- **Act**: ISSUE-01 に spkB（EXP-002）、ISSUE-02 に oracle 改修（EXP-003）

## EXP-002　spkB 5 epoch (ISSUE-01)
- **対象**: ISSUE-01（役割ミスマッチ）
- **仮説**: 話者 B（99% が答える側）で学習すれば、開幕自発話が消え「聞き終えてから答える」アシスタント挙動になる。oracle（B 側=答え）も推論時の役割と一致。
- **変更点（EXP-001 差分）**: `MOSHI_SPEAKERS=B` のみ。データ・ハイパラ・モデルは同一（再トークナイズ不要）。
- **実装**: `examples/finetune_j_moshi.sh` に `MOSHI_SPEAKERS` env → `--moshi_speakers` を追加（未指定なら従来＝A、後方互換）。`train_0374_kame_spkB.pbs`。
- **出力先**: `output/0374_kame-0610-e5-spkB`（本番）, `output/0374_kame-spkB-smoke`（スモーク）
- **予測（成功条件）**:
  - 接続直後に自発話しない（無音で待つ／ユーザ先行で始まる）
  - 質問されてから答える挙動
  - loss が EXP-001 同等（≈1.0）に収束
- **Do**:
  - スモーク（2026-06-30, job `1984116.pbs1`）: `NUM_EPOCHS=1`, 1 parquet。配管と B 切替の確認 → **完了**。
  - 本番（2026-06-30 投入, 学習中）: `qsub train_0374_kame_spkB.pbs`（5ep ≈ step_11680, 約 2.5h）。job id: `1984288.pbs1`。
- **変換・転送（完了, 2026-06-30）**:
  - `tools.zero_to_fp32`（fp32 化, 7,818,805,248 elements）→ `tools.clean_moshi --model_dtype bfloat16`（**`--remove_modules_for_user_stream` は不要**: 本モデルは元から dep_q=8）。
  - cleaned を Mac 経由で kiso へ（Mac→kiso は openrsync push が `unexpected end of file` で失敗 → **scp で成功**）。
  - kiso 着弾検証: `output/0374_kame-0610-e5-spkB/step_11680_cleaned/`、`model.safetensors`=15,637,663,072B（e5 と同一）、`dep_q:8/depformer_context:8/n_q:16`、owner 読み可。
- **結果（2026-06-30, kiso GPU1 でマイク対話）**:
  - ASR/mic は正常（`final_transcripts` 増・`[oracle:in]` 多数。`words_detected:0` は未集計ダミーで判定に使えない）。
  - **開幕自発話は解消せず**：無音中にモデルが「サッカーで最も強い国っては?」等を自発的に出題。
  - **役割も未解消**：質問に答えず「ありがとう/なるほど/納得した」の相槌に回る（出題者+受け手の A 的挙動）。
  - **oracle 非追従**（ISSUE-02 健在）：「鎌倉幕府は何年?」に答えを渡しても年号を言わない。
  - **流暢性低下**：発話が断片化・非文（例「アサケス朝から1日後に、3,8,2・2)って誰が作った川だったかな?」）。oracle LLM すら解釈不能。
- **判定**: ✗ 仮説不成立。`--moshi_speakers B` だけでは ISSUE-01 は解消しない。
- **話者選択の検証（2026-06-30）**: 「B が効いていないのでは」の疑いを潰した。①ローカルで `main_speaker_streams` を A/B で回すと別ターゲット（B 選択時は text 冒頭が全 padding=無音、`utils/data.py:9`）。②本番 job 1984288 ログに `main speaker = B`、`config.json` に `moshi_speakers` 記録。→ **話者選択は正しく機能。開幕質問は話者フラグの取り違えではない。**
- **Act**: ISSUE-01 の真因を **turn-taking（データに「開始＝全員無音」の precedent が皆無）** に確定。モデルが学んだ無音は「P(B=無音 | A=喋っている)」に限られ、推論開幕の「全員無音」が OOD → 生成に走る。次は EXP-003 を **oracle 重畳から「冒頭無音・ユーザ先行」データ改修へ差し替え**。spkB の流暢性低下も併せて要観察。

## EXP-003　oracle speaking 重畳 (ISSUE-02) — 計画
- **対象**: ISSUE-02（oracle 非追従）
- **仮説**: oracle を speaking 区間まで重畳/持続させ、フレーム単位で「oracle→発話」を直接教えれば、推論時に新事実（oracle）を追従する。
- **変更点（候補）**: データ生成側で oracle イベントを答え区間に 1 フレーム先行で並べ持続（`tools/generate_oracle_from_text.py` / `tools/prepare_dataset.py`）。＋ `oracle_skip_prob` 低減。
- **予測（成功条件）**: 新規質問でも oracle の答えが発話に反映される。
- **注意**: EXP-002 の結果を見てから設計確定（役割ミスマッチ解消後の方が oracle 経路の評価がきれい）。
- **状態**: 未着手
