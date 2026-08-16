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

標註詞彙為 **8 axes × signed dimensions**（如 `level` / `too_loud`、`body` / `muddy`），外加 `phase` 與 `none`。詳見 [`labeling_pipeline.md`](labeling_pipeline.md)。

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
├── scripts/
│   ├── export_problem_pool.py
│   ├── analyze_labels.py
│   ├── analyze_vocal.py
│   └── compare_l2_labels.py
└── src/
    ├── labeling/
    │   ├── labeling_common.py    # schema / I/O / parse / normalize
    │   ├── labeling_hf.py        # Hugging Face transformers
    │   ├── labeling_gguf.py      # llama.cpp GGUF
    │   ├── labeling_groq.py      # Groq API
    │   └── label_l2_generated.py # re-label L2 dialogues (consistency check)
    ├── prompts/
    │   ├── labeling_prompt.py
    │   └── l2_generation_prompt.py
    └── generate_l2_style.py
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
| `src/generate_l2_style.py` | 以 problem pool 為 gold + few-shot，生成 Amateur / Expert 對話 |
| `src/labeling/label_l2_generated.py` | 對 L2 生成對話再跑 labeling（同 GGUF / prompt），輸出 `labeled_l2_gguf_{mode}.csv` |
| `scripts/compare_l2_labels.py` | 比對 L1 gold vs 再標結果，輸出 dim / axis 一致率 |

---

## 文件

| 文件 | 說明 |
|------|------|
| [`labeling_pipeline.md`](labeling_pipeline.md) | 標註管線設計：上下文、JSON schema、正規化、L2 pool 匯出 |
| [`LEXICON_BRIEF.md`](LEXICON_BRIEF.md) | MixJudge caption ontology：11 dims、slot grammar、QUALITY 用詞 |
| [`Lexicon_and_L2_Generation.md`](Lexicon_and_L2_Generation.md) | 現行 lexicon 抽取 → Quality Lexicon → L2 生成流程 |

---

## 本地執行（概要）

資料與模型需放在 repo 根目錄旁（皆不進 git）：

- 輸入：`train.csv` / `validation.csv` / `test.csv`
- 模型（GGUF 路徑）：`models/google_gemma-4-31B-it-Q4_K_M.gguf`
- 輸出：`outputs/`

```bash
# 1) 標註（本地 GGUF）
python src/labeling/labeling_gguf.py --splits train --limit 5   # 小量測試
python src/labeling/labeling_gguf.py                            # 全量

# 2) 匯出 L2 problem pool
python scripts/export_problem_pool.py

# 3) 統計
python scripts/analyze_labels.py

# 4) L2 生成（pilot）— 固定 temp=0.1；只用 MixAssist raw；鬆緊靠 mode
python src/generate_l2_style.py --modes retarget strict free \
  --exemplar-content raw --n-exemplars 1 --overwrite
# 或一次跑完 generate + re-label + compare:
bash scripts/run_l2_raw_variants.sh

# 5) 對生成對話再標註（驗證是否仍符合 L1 axis/dim）
python src/labeling/label_l2_generated.py \
  --inputs outputs/l2_from_l1_retarget_raw.csv \
           outputs/l2_from_l1_strict_raw.csv \
           outputs/l2_from_l1_free_raw.csv \
  --overwrite

# 6) 比對 gold vs 再標一致率
for mode in retarget strict free; do
  tag="${mode}_raw"
  python scripts/compare_l2_labels.py \
    --l2 "outputs/l2_from_l1_${tag}.csv" \
    --labeled "outputs/labeled_l2_gguf_${tag}.csv" \
    --report "outputs/l2_label_agreement_${tag}.md"
done
```

Groq 版需設定 `GROQ_API_KEY` 後執行 `python src/labeling/labeling_groq.py`。

標註支援中斷續跑（增量寫入）；換 prompt / schema 後可用 `--overwrite` 重標。

---

## 設計要點

1. **標註與篩選分離**：labeling 保留含 `none` 的完整分佈；進入 L2 前才由 `export_problem_pool.py` 過濾。
2. **模型只填 dimension**：`problem_axis` 由對照表決定，降低 axis / dimension 混淆。
3. **一 turn 可多列**：同一回合若有多個明確問題，展開為多筆 CSV 列。
4. **歷史輔助、當前為主**：最多帶入前 4 turns，用於消歧義與延續問題，不作為獨立標註來源。
