# MixJudge CLAP Text Pipeline

利用 MixAssist 真實混音對話，為合成訓練資料補上更自然的文字層（L2 / L3）。本 repo 目前聚焦 **Phase 2a：對話標註 → problem pool → L2 style transfer**。

合成資料的 L1 標籤正確但像教科書；MixAssist 則是 Amateur / Expert 在 session 裡真實討論問題與處置。本專案先把 MixAssist 標成可檢索的 axis–dimension 結構，再據此生成接近真人語氣的問題描述。

---

## 整體架構

```text
MixAssist CSV (train / validation / test)
        │
        ▼
┌───────────────────────┐
│  Labeling (Phase 2a)  │  topic + history(≤4) + current turn
│  Gemma / Groq LLM     │  → JSON labels → normalize + expand
└───────────────────────┘
        │
        ▼
  labeled_turns_*.csv  (+ _raw.jsonl)
        │
        ▼
┌───────────────────────┐
│  export_problem_pool  │  has_problem ∧ dimension≠none
│                       │  → concat splits
└───────────────────────┘
        │
        ▼
  *_all_problems.csv  (L2 retrieval pool)
        │
        ├──► analyze_labels  → 統計 / 抽查
        └──► generate_l2_style → Amateur / Expert 風格對話
```

流程圖詳見 [`labeling_pipeline.md`](labeling_pipeline.md) 與下圖：

![Labeling pipeline](Labeling_Flow_Chart.png)

**三層文字系統（專案目標）：**

| Layer | 內容 | 狀態 |
|-------|------|------|
| **L1** | 結構化模板（axis / dim / subject…），訓練錨點，不修改 | 既有 |
| **L2** | 真實混音對話風格的問題描述（retrieval + style transfer） | 進行中 |
| **L3** | L2 + 同 turn 的自然 fix 用語 | 後續 |

標註詞彙為 MixJudge **7 axes / 12 classes**（11 problem + `clean`；見 [`LEXICON_BRIEF.md`](LEXICON_BRIEF.md)）。`none` 表示無法從 CURRENT TURN 區分。

---

## Repository 結構（GitHub 上可見）

```text
.
├── README.md
├── .gitignore
├── Labeling_Flow_Chart.png
├── labeling_pipeline.md
├── LEXICON_BRIEF.md
├── Lexicon_and_L2_Generation.md
├── label_stats.md
├── generation_note.md
├── intern_chores.md
├── scripts/
│   ├── export_problem_pool.py
│   ├── analyze_labels.py
│   ├── analyze_vocal.py
│   ├── compare_l2_labels.py
│   ├── filter_l2_by_relabel.py
│   └── run_l2_raw_variants.sh
└── src/
    ├── labeling/
    │   ├── labeling_common.py    # schema / I/O / parse / normalize
    │   ├── labeling_hf.py        # Hugging Face transformers
    │   ├── labeling_gguf.py      # llama.cpp GGUF (single GPU)
    │   ├── labeling_groq.py      # Groq API
    │   └── label_l2_generated.py # re-label L2 dialogues (consistency check)
    ├── prompts/
    │   ├── labeling_prompt.py
    │   ├── l2_generation_prompt.py
    │   ├── lexicon_slots.py
    │   └── term_extract_prompt.py
    ├── generate_l2_style.py
    └── term_extract.py
```

> 大型資料（`*.csv`、`outputs/`、`models/*.gguf`、音訊等）已列在 `.gitignore`，不會出現在 GitHub。本地需自行準備 MixAssist split CSV 與模型權重。

---

## 程式模組

| 檔案 | 角色 |
|------|------|
| `src/labeling/labeling_common.py` | 共用：`DIMENSION_TO_AXIS`、JSON 解析、增量寫入 CSV / JSONL |
| `src/prompts/labeling_prompt.py` | `SYSTEM_PROMPT`、user message 組裝 |
| `src/labeling/labeling_gguf.py` | 本地 GGUF 推論；預設每 split 輸出 `outputs/labeled_turns_gguf_{split}.csv` |
| `src/labeling/labeling_hf.py` | Hugging Face 本地推論 |
| `src/labeling/labeling_groq.py` | 雲端 API 推論，方便快速試 prompt / 比模型 |
| `scripts/export_problem_pool.py` | 過濾有效 problem 列，寫出 `*_problems.csv` 與 `*_all_problems.csv` |
| `scripts/analyze_labels.py` | 覆蓋率、axis / dimension 分佈、抽樣檢視 |
| `src/generate_l2_style.py` | LEXICON_BRIEF L1 captions + MixAssist style pool → Amateur / Expert 對話 |
| `src/labeling/label_l2_generated.py` | 對 L2 生成對話再跑 labeling（同 GGUF / prompt），輸出 `labeled_l2_gguf_{mode}_raw.csv` |
| `scripts/compare_l2_labels.py` | 比對 L1 gold vs 再標結果，輸出 dim / axis 一致率 |
| `scripts/filter_l2_by_relabel.py` | 留下 dim_any 命中列 → `l2_from_l1_{mode}_raw_kept.csv` |

---

## 文件

| 文件 | 說明 |
|------|------|
| [`labeling_pipeline.md`](labeling_pipeline.md) | 標註管線設計：上下文、JSON schema、正規化、L2 pool 匯出 |
| [`LEXICON_BRIEF.md`](LEXICON_BRIEF.md) | MixJudge 11 dim 本體、必須可區分的 pair、caption QUALITY 用詞 |
| [`Lexicon_and_L2_Generation.md`](Lexicon_and_L2_Generation.md) | lexicon 抽取 → Quality Lexicon → L1/L2 生成與 relabel filter |
| [`label_stats.md`](label_stats.md) | 全量標註後的 coverage 與 axis / dimension 統計 |
| [`generation_note.md`](generation_note.md) | L2 生成的輸入輸出與 style prompt 構想 |
| [`intern_chores.md`](intern_chores.md) | 背景目標、L1/L2/L3、早期 9-key 規劃與 Phase checklist |

---

## 本地執行（概要）

資料與模型需放在 repo 根目錄旁（皆不進 git）：

- 輸入：`train.csv` / `validation.csv` / `test.csv`
- 模型（GGUF 路徑）：`models/google_gemma-4-31B-it-Q4_K_M.gguf`
- 輸出：`outputs/`

```bash
# 全套（label → pool → generate → relabel → compare → filter）
# MixAssist 用 --overwrite，因為 prompt 已換成 LEXICON_BRIEF 12-class。
bash scripts/run_l2_raw_variants.sh
# 已用新 prompt 標過 MixAssist 時：SKIP_LABEL=1 bash scripts/run_l2_raw_variants.sh

# 或分步：
# 1) MixAssist 標註（src/prompts/labeling_prompt.py）
python src/labeling/labeling_gguf.py --n-batch 512 --overwrite

# 2) 匯出 L2 problem pool
python scripts/export_problem_pool.py

# 3) 統計
python scripts/analyze_labels.py

# 4) L2 生成
python src/generate_l2_style.py \
  --l1 new_outputs/l1_lexicon_captions_lex.jsonl \
  --pool outputs/labeled_turns_gguf_all_problems.csv \
  --modes retarget strict free \
  --exemplar-content raw --n-exemplars 1 --n-batch 512 --overwrite

# 5) 對生成對話再標註（同一套 labeling prompt）
python src/labeling/label_l2_generated.py \
  --inputs outputs/l2_from_l1_retarget_raw.csv \
           outputs/l2_from_l1_strict_raw.csv \
           outputs/l2_from_l1_free_raw.csv \
  --output-dir outputs --n-batch 512 --overwrite

# 6) 比對 gold vs 再標一致率
for mode in retarget strict free; do
  tag="${mode}_raw"
  python scripts/compare_l2_labels.py \
    --l2 "outputs/l2_from_l1_${tag}.csv" \
    --labeled "outputs/labeled_l2_gguf_${tag}.csv" \
    --report "outputs/l2_label_agreement_${tag}.md"
done

# 7) 過濾 dim_any 命中列
python scripts/filter_l2_by_relabel.py --output-dir outputs --min-keep 10
```

Groq 版需設定 `GROQ_API_KEY` 後執行 `python src/labeling/labeling_groq.py`。

標註支援中斷續跑（增量寫入）；換 prompt / schema 後可用 `--overwrite` 重標。

---

## 設計要點

1. **標註與篩選分離**：labeling 保留含 `none` 的完整分佈；進入 L2 前才由 `export_problem_pool.py` 過濾。
2. **模型只填 dimension**：`problem_axis` 由對照表決定，降低 axis / dimension 混淆。
3. **一 turn 可多列**：同一回合若有多個明確問題，展開為多筆 CSV 列。
4. **歷史輔助、當前為主**：最多帶入前 4 turns，用於消歧義與延續問題，不作為獨立標註來源。
