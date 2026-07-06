# 学習パイプライン（parquet → 1 ステップ）

`parquet 1 行(=1 対話)` が学習 1 ステップに変換されるまでを、実コードに即して順に追う。
フレームは Mimi（約 12.5Hz＝1 フレーム 80ms）単位。課題は [issues.md](issues.md)、実験は [experiments.md](experiments.md) と相互参照。

## 全体像（データの流れ）
```
parquet 1行(=1対話)
  └ A:[9,T], B:[9,T], A_oracle_*, B_oracle_*
        │  preprocess_function  [CPU, datasets.map で前計算]
        ▼
  ① 主話者選択&合成 → ② 遅延/パディング → ③ oracle整列 → ④ max_length分割 → ⑤ ラベル化
        │  1対話 → 複数チャンク {streams:[17,T'], labels:[17,T'], oracleイベント}
        ▼  DataCollator [バッチ生成時にランダム性]
  ⑥ ミニバッチ化 + oracleをフレーム展開(+augmentation)
        │  Batch{ input_ids:[B,17,T], labels:[B,17,T], oracle_tokens:[B,1,T] }
        ▼  forward() [GPU]
  ⑦ Temporal Transformer(時間方向) → text損失
  ⑧ Depth Transformer(フレーム内8コードブック) → 音声損失
  ⑨ 損失合算(semantic100/acoustic1/text_pad0.5)
        ▼
  ⑩ 逆伝播 (accelerate + DeepSpeed ZeRO-3, bf16, 8GPU)
```

---

## ① 主話者の選択とストリーム合成 — `utils/data.py:9` (`main_speaker_streams`)
parquet には A・B が各 `[9,T]`（**1 text + 8 音声コードブック**）。`--moshi_speakers` で「自分（モデル）」を選び、残りを「相手」に:

```
streams = [ 自分の 1 text + 8 音声 ]  ⊕  [ 相手の 8 音声 ]   = 17 行 × T
            └ 生成ターゲットになる           └ 入力専用（相手の text は捨てる）
```
> **ISSUE-01 の発生点**。A を選ぶと「質問者A」を生成学習、B を選ぶと「回答者B」。EXP-002 はここを `B` にしただけ。

## ② 遅延とパディング — `utils/data.py:33` (`delay_and_pad_streams`)
`delays = [0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]`（text=0、各音声群は cb0=0・cb1〜7=1）。各行を遅延分だけ右へずらし、先頭に initial token を 1 個前置。
→ Moshi の acoustic delay。同一フレームで「semantic（cb0）を先に、acoustic（cb1-7）を 1 フレーム遅れで」予測できるよう時間軸をずらす仕掛け。

## ③ oracle イベントをタイムラインに整列 — `utils/data.py:164` (`align_oracle_events_to_delayed_text_timeline`)
oracle イベントの `frame_pos` を `+1+text_delay` ずらして、②後の text タイムラインに合わせる。範囲外イベントは捨てる。

## ④ max_length 分割 + 短い系列除去 — `utils/data.py:224` / `:263`
長い対話を `max_length=512` フレームのチャンクに切る（`min_length=128` 未満は捨てる）。oracle イベントもチャンク境界で分配し、チャンク内ローカル座標へ。**1 対話 → 複数の学習サンプル**になる（「行数 < ステップ数」の理由）。

## ⑤ ラベル作成 — `utils/data.py:287` (`make_streams_labels`)
`labels = streams` のコピー。ただし initial token の位置は `zero_token_id` に置換 → **損失で無視**（cross_entropy の `ignore_index`）。

> ①〜⑤は `preprocess_function`（`utils/data.py:311`）。**CPU で事前計算**され、出力 1 件 = `{streams[17,T'], labels[17,T'], num_frames, oracle イベント列}`。

## ⑥ ミニバッチ化 + oracle フレーム展開 + augmentation — `utils/data.py:725` (`DataCollator.__call__`)
バッチ生成のたびに走る（**ここがランダム性の源**）:
- B 件を `max_frames` まで zero パディング → `input_ids [B,17,T]`、`labels [B,17,T]`、`text_attention_mask`。
- **oracle をイベント → フレーム列に展開**（`utils/data.py:648` `_events_to_oracle_1d`）：各イベントの `frame_pos` に `oracle_start_id`、続く位置に予測トークン列（hint or pred）を書き込み → `oracle_tokens [B,1,T]`。
- **augmentation**（ISSUE-02 で「記憶を壊す」と言った部分）:
  - **skip**：確率 `0.1〜0.7`（例ごとにサンプル）でイベントを脱落（`skip_forbid=0` のみ）。
  - **jitter**：`±max_time_jitter_frames`。
  - **shift**：oracle 列全体を左右シフト（`utils/data.py:605`）。
  - hint→pred カリキュラム（warmup で切替）も実装あり。

## ⑦ Temporal Transformer（時間方向） — `finetune.py:654` (`tempformer_forward`)
1 フレーム = 1 トークン位置として、時間方向の自己回帰本体:
```
text_emb = text_emb(text)
★ oracle 注入:  oracle_emb = oracle_emb(oracle_tokens);  text_emb += oracle_emb   # separate モード
audio_emb = Σ_{16cb} emb[cb](audio)      # 自分8+相手8 を合算（cb0 は semantic_emb_dropout 対象）
tempformer_in = text_emb + audio_emb  →  Transformer  →  text_logits
```
- **text 損失**（teacher forcing, 1 つ先予測 `logits[:-1] vs labels[1:]`）を、`non_pad`（実テキスト, 重み 1）と `pad`（無音 padding, 重み 0.5）に分けて集計。
- > **ISSUE-02 の核**：oracle は `text_emb` に**足すだけで、自前の予測ターゲットを持たない**（入力条件付け専用）。しかも oracle が濃いフレーム（相手を聞いている listening 区間）では text/音声ラベルが pad/無音 → 「oracle=X → 次に X と言え」のフレーム対応を直接は学べない。EXP-003 はこの⑥/⑦の oracle 配置を speaking 区間に重ねる話。

## ⑧ Depth Transformer（フレーム内 8 コードブック） — `finetune.py:751` (`depformer_forward`)
各フレームの隠れ状態から、**自分の 8 音声コードブックをフレーム内で逐次**予測（RQ-transformer）。入力に「直前フレームの正解トークン」埋め込みを足す（teacher forcing）。予測対象は `labels[1:9]`（自分の 8 音声）**のみ**＝`dep_q=8`（相手 9..16 は入力専用 → cleaned で dep_q=8 になる理由）。
- 損失を **cb0 = semantic / cb1-7 = acoustic** に分離。

## ⑨ 損失の合算と重み — `finetune.py:885`（`forward`）
```
text_loss  = mean(非padテキスト) + 0.5 * mean(padテキスト)
audio_loss = semantic と acoustic を 100:1 で重み付けした加重平均
             （semantic 1トークン ≒ acoustic 100トークン分の重み）
total_loss = text_loss + audio_loss
```
→ **意味/内容を担う cb0(semantic) を最重視**、テキストの無音は軽視、という設計。
ハイパラ既定：`semantic_loss_weight=100` / `acoustic_loss_weight=1` / `text_padding_loss_weight=0.5`。

## ⑩ 逆伝播・最適化
accelerate + DeepSpeed **ZeRO-3 / bf16** で 8 GPU に param/grad/optimizer をシャーディング。`lr 3e-5`・warmup 100・`save_steps 500` ごとに `step_*` 保存。`PER_DEVICE_BATCH_SIZE=2 × 8GPU` で、38,783 対話＋④のチャンク分割 → **1 epoch ≈ 2,336 step**（×5ep = 11,680）。

---

## 2 つの課題がパイプラインのどこに効くか
| 課題 | 発生ステップ | 打ち手 |
|---|---|---|
| **ISSUE-01** 役割ミスマッチ | **①** 自分=A→質問者を生成学習 | `--moshi_speakers B`（EXP-002, 実行済） |
| **ISSUE-02** oracle 非追従 | **⑥/⑦** oracle は入力加算のみ＋濃いフレームのラベルが pad＋aug が破壊 | oracle を speaking 区間へ重畳（EXP-003） |

## 主要ファイル早見
| 役割 | 場所 |
|---|---|
| 起動・ハイパラ | `examples/finetune_j_moshi.sh` |
| 引数・forward・loss・学習ループ | `finetune.py`（`tempformer_forward:654` / `depformer_forward:751` / `forward:850`） |
| 前処理・collate・augmentation | `utils/data.py`（`preprocess_function:311` / `DataCollator:509`） |
| モデル本体（4-stream 化, ZeRO-3 対応） | `models/moshi_for_finetuning.py` |
| oracle 埋め込み backfill | `models/oracle_embedding_utils.py` |
