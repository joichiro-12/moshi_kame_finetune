# Kame / J-Moshi 推論・分析ガイド

J-KAME（oracle 付き日本語 Moshi）のファインチューニング済みモデルを**動かす・観察する・デバッグする**ための実務メモ。
本ドキュメントは実際の立ち上げで踏んだ事項をまとめたもの。

---

## 1. しくみ（最小限の理解）

### Moshi（ベース）
- **全二重（full-duplex）S2S**：ユーザとモデルが同時に話せる。モデルは **常時トークンを生成し続ける**。
- ストリーム構成：**text(1) + 音声 Mimi コードブック(8)**。音声は 12.5 Hz フレーム、各フレーム 8 コードブック。
- 重要な含意：**接続した瞬間 t=0 から自分のチャンネルに生成**するため、「開始直後に自分から喋り出すか／黙って待つか」は**学習データの会話開始パターン依存**。

### Kame（oracle 拡張）
- Moshi に **第4ストリーム = oracle** を足した tandem 構成。
- 外部 LLM が「**次にモデル（話者）が言うべき発話**」を**実時間で予測**し、ヒントとしてモデルへ注入する。
- 狙い：軽量な S2S 本体に、外部 LLM の**知識・次発話の指針**をリアルタイムに載せる。
- 初期化：`tools.init_moshi_for_ft` で `j-moshi-ext` に oracle ストリームを追加（`oracle_emb` を `text_emb` からバックフィル）。

---

## 2. 推論システム構成（J-KAME server）

エントリ：[kame_jp/server_oracle_jp.py](../kame_jp/server_oracle_jp.py)（`kame.server_oracle` の日本語版）。3経路が並行して動く：

1. **マイク音声 → ASR**：Google Cloud Speech-to-Text（`ja-JP`）でユーザ発話を文字起こし → `pending_user_text` / 確定で会話に追加。
2. **会話テキスト → oracle LLM**：vLLM の `llm-jp/llm-jp-4-8b-thinking`（OpenAI 互換 API）へ問い合わせ。
   - **0.5 秒間隔のポーリングで、pending（ASR結果）がある時だけ起動**（[upstream `run_periodic_streaming`]）。
   - thinking モデルなので **reasoning（analysis）を除去**し、最終回答のみを `oracle_queue` に流す（`parse_harmony_response`）。
3. **oracle_queue → Moshi 本体**：第4ストリームとして注入 → 音声＋テキストを生成。

### 必要なもの
- cleaned モデル：`model.safetensors` + `moshi_lm_kwargs.json`（学習チェックポイントを `tools.zero_to_fp32` → `tools.clean_moshi` で変換したもの）。
- Mimi 音声トークナイザ（HF から自動DL、キャッシュ済み）。
- テキスト tokenizer：`tokenizer_spm_32k_3.model`（`nu-dialogue/j-moshi-ext`）。
- oracle 用 vLLM（`:8000` で `llm-jp/llm-jp-4-8b-thinking`）。
- ASR 用 GCP サービスアカウント鍵（`GOOGLE_APPLICATION_CREDENTIALS`）。
- GPU（本体 7B 級 ≈ 15〜18GB）。

---

## 3. 起動方法

### KAME（FT + oracle）— kiso 例
```bash
cd /mnt/kiso-qnap4/jsato/moshi_kame_finetune
mkdir -p logs; LOG=logs/kame_$(date +%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 ORACLE_DEBUG=1 \
OPENAI_API_KEY=EMPTY \
GOOGLE_APPLICATION_CREDENTIALS=$PWD/llmjp-497505-118510c984bd.json \
CUDA_VISIBLE_DEVICES=1 \
uv run -m kame_jp.server_oracle_jp \
  --moshi-weight output/0374_kame-0610-e5/step_11680_cleaned/model.safetensors \
  --config-path  output/0374_kame-0610-e5/step_11680_cleaned/moshi_lm_kwargs.json \
  --tokenizer    ~/.cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/*/tokenizer_spm_32k_3.model \
  --llm-base-url http://localhost:8000/v1 --llm-model llm-jp/llm-jp-4-8b-thinking \
  --port 8998 --log-dir logs/session 2>&1 | tee "$LOG"
```

### ベース j-moshi-ext（oracle 無し）— 比較用
`kame.server`（純 S2S。ASR/LLM/oracle 不要、vLLM も GCP も不要）：
```bash
SNAP=~/.cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/<hash>
CUDA_VISIBLE_DEVICES=1 uv run -m kame.server \
  --hf-repo nu-dialogue/j-moshi-ext \
  --moshi-weight "$SNAP/model.safetensors" \
  --mimi-weight  "$SNAP/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer    "$SNAP/tokenizer_spm_32k_3.model" \
  --device cuda --port 8998
```
（`j-moshi-ext` には `config.json` が無く、`from_hf_repo` が既定設定にフォールバックして読める。）

### クライアント接続（マイク対話）
サーバは `127.0.0.1` バインド。手元から**SSHトンネル**して、ブラウザ＋マイクで使う：
```bash
# 手元(Mac)で。トンネル用ターミナルは開いたままにする
ssh -L 8998:localhost:8998 g21
# → ブラウザ http://localhost:8998 → マイク許可 → 日本語で話す
```
`localhost` 扱いなのでブラウザはマイクを許可する（HTTPS不要）。

### 永続運用
- サーバは**自分の tmux** 上で起動する（PC を閉じても残る・ログも見える）。
- ツール経由のバックグラウンド/サブプロセスは終了時に回収されることがあるため、長期運用は tmux 推奨。

---

## 4. ログ・分析方法

起動時に `2>&1 | tee logs/...`、`--log-dir logs/session`、`ORACLE_DEBUG=1`、`PYTHONUNBUFFERED=1` を付けると追いやすい。

### 構造化ログ `logs/session/`
| ファイル | 中身 |
|---|---|
| `asr_partial.txt` | マイク→ASR の途中認識（タイムスタンプ付き） |
| `user_words.txt` | 確定したユーザ発話 |
| `conversation.txt` | user / moshi の会話全体 |
| `llm_stream_words.txt` | oracle LLM が流した予測語（[CANCELLED]/[ERROR] 注記あり） |
| `oracle_stream.txt` | **モデルへ実際に注入された** oracle（`[RESET]` 付き） |
| `moshi_words.txt` | Moshi 本体が実際に発話した語 |

### コンソールのログタグ
| タグ | 意味 | モデルに入る？ |
|---|---|---|
| `[ASR Partial +N] …` | マイク→ASR の認識。**出る＝マイクOK** | – |
| `[oracle:in] pending=… context_tail=…` | oracle への入力（ASR 文脈） | – |
| `[oracle:raw] …` | LLM 生出力（thinking 込み、`ORACLE_DEBUG=1` 時のみ） | ❌ 参考用 |
| `[oracle:out] …` | reasoning 除去後の最終回答 | ✅ 注入される |
| `[LLM-jp4] <word>` | oracle が流した語（逐次） | ✅ |

> `[oracle:in]/out` と関連ログは [kame_jp/server_oracle_jp.py](../kame_jp/server_oracle_jp.py) の `_stream_llm_response` で出力。

### 切り分けフロー（マイク対話が動かない時）
1. **`[ASR Partial …]` が出るか？**
   - 出ない → マイク/ASR の問題：ブラウザのマイク許可、入力デバイス/ミュート、`localhost` 接続か、**GCP Speech-to-Text API が有効か**（無効だと `SERVICE_DISABLED`）。
   - サーバログに `words_detected: 0` が続く → 音声が無音／届いていない。
2. **`[oracle:in]` / `[oracle:out]` の中身**：ASR 文脈が入っているか、最終回答が出ているか。
3. **`oracle_stream.txt`**：`[RESET]`＋語が出ていれば**注入はできている**。
4. **`moshi_words.txt` と `[oracle:out]` を突き合わせ**：内容が一致していれば追従、ズレていれば**oracle 非追従**。
5. **Moshi が声で応答するか**：応答するのにマイクが効かない＝上り（マイク→ASR）だけの問題。

### 接続できなくなった時（`localhost:8998` に繋がらない）
- まず kiso 側：`ss -ltn | grep 8998`（LISTENING か）、`curl -s -o /dev/null -w '%{http_code}' localhost:8998`（200 か）。
- サーバが生きていれば、原因は**SSHトンネル切れ**。手元で `ssh -L 8998:localhost:8998 g21` を張り直す。
- ポート競合時は別ローカルポート：`ssh -L 9998:localhost:8998 g21` → `http://localhost:9998`。

---

## 5. ハマりどころ（既知の対処）

- **thinking モデルの reasoning 混入**：`llm-jp-4-8b-thinking` は `analysis … assistant final …` 形式（Harmony タグ無し）で返す。`parse_harmony_response` が `assistant final` 以降のみ抽出。これを怠ると思考過程が oracle に丸ごと流れ込む。
- **`OPENAI_API_KEY` env が必須**：ローカル vLLM 利用時でも未設定だと起動失敗。`OPENAI_API_KEY=EMPTY` を付与。
- **GCP Cloud Speech-to-Text API の有効化**：未有効だと ASR が `SERVICE_DISABLED` で 0 語。プロジェクトで API 有効化＋課金設定が必要。
- **`127.0.0.1` バインド + SSHトンネル**：手元から直接は届かない。トンネル必須、切れると接続不可。
- **（ABCI 学習側）**：GPU 処理は**計算ノード（hnode）**で。ログインノードは GPU 無し＆重い処理を SIGKILL。計算ノードでは毎回 `module load cuda/12.1`（DeepSpeed の CUDA_HOME 用）と `export PATH=$HOME/.local/bin:$PATH`（uv）が必要。詳細は別途 ABCI ワークフローのメモ参照。

---

## 6. これまでの観測と解釈

- **配管は完成**：ASR → oracle(LLM) → Moshi 注入は機能。oracle の知識応答は正しい（例：「3+3」→「6/六です」、「鎌倉時代は何年から」→「1185年です」）。`oracle_stream.txt` に注入も確認済み。
- **学習量の効果**：1 epoch（loss ≈3.1）は出力が崩壊（反復）。5 epoch（loss ≈0.99）で**流暢な日本語対話**になり、QA(クイズ)調の話し方を獲得。
- **残課題**：
  - **oracle 非追従**：oracle が「6です」でも本体は別内容を話すことがある。設計上の augmentation（`oracle_skip_prob` 0.1–0.7 等）が oracle 依存を弱めている側面＋学習・データの偏り。
  - **開幕の自発話**：接続直後にユーザより先に喋り出す。全二重の性質＋学習データの会話開始パターン（モデル側が口火を切る）に起因。
- **今後の打ち手**：`oracle_skip_prob` 低減＋ hint curriculum で追従強化／データ多様化（雑談・指示応答）＋冒頭無音・ユーザ先行の例で turn-taking 改善／追加学習。

---

## 関連
- データ作成パイプライン：[scripts/japanese_kame/](../scripts/japanese_kame/)（00 init →01 QA収集→02 対話化→03 Gemini TTS→04 アライン→05 oracle→06/06b tokenize）。
- 学習：`examples/finetune_j_moshi.sh` + `finetune.py`（ABCI, DeepSpeed ZeRO-3）。
- 変換：`tools.zero_to_fp32` → `tools.clean_moshi`。
