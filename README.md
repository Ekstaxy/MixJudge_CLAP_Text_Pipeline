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
├── README.md                 # 本說明
├── .gitignore
├── Labeling_Flow_Chart.png   # 標註管線流程圖
├── labeling_pipeline.md      # 標註方法說明（論文式）
├── label_stats.md            # 標註結果統計摘要
├── generation_note.md        # L2 style transfer 構想筆記
├── intern_chores.md          # 實習任務與 Phase 規劃
└── src/
    ├── labeling.py           # 共用 prompt / 解析 / 寫檔；HF 本地推論
    ├── labeling_gguf.py      # llama.cpp GGUF（Gemma Q4）全量標註
    ├── labeling_groq.py      # Groq API 標註（同一 schema）
    ├── export_problem_pool.py# 匯出 problem-only pool、合併 splits
    ├── analyze_labels.py     # 標註品質統計與抽查
    └── generate_l2_style.py  # L2 style transfer 生成
```

> 大型資料（`*.csv`、`outputs/`、`models/*.gguf`、音訊等）已列在 `.gitignore`，不會出現在 GitHub。本地需自行準備 MixAssist split CSV 與模型權重。

---

## 程式模組

| 檔案 | 角色 |
|------|------|
| `src/labeling.py` | 核心：`SYSTEM_PROMPT`、user message 組裝、JSON 解析、`DIMENSION_TO_AXIS`、增量寫入 CSV / JSONL |
| `src/labeling_gguf.py` | 本地 GGUF 推論；預設每 split 輸出 `outputs/labeled_turns_gguf_{split}.csv` |
| `src/labeling_groq.py` | 雲端 API 推論，方便快速試 prompt / 比模型 |
| `src/export_problem_pool.py` | 過濾有效 problem 列，寫出 `*_problems.csv` 與 `*_all_problems.csv` |
| `src/analyze_labels.py` | 覆蓋率、axis / dimension 分佈、抽樣檢視 |
| `src/generate_l2_style.py` | 以 problem pool 為 gold + few-shot，生成 Amateur / Expert 對話 |

---

## 文件

| 文件 | 說明 |
|------|------|
| [`labeling_pipeline.md`](labeling_pipeline.md) | 標註管線設計：上下文、JSON schema、正規化、L2 pool 匯出 |
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
# 1) 標註（本地 GGUF）
python src/labeling_gguf.py --splits train --limit 5   # 小量測試
python src/labeling_gguf.py                            # 全量

# 2) 匯出 L2 problem pool
python src/export_problem_pool.py

# 3) 統計
python src/analyze_labels.py

# 4) L2 生成（pilot）
python src/generate_l2_style.py --limit 3
```

Groq 版需設定 `GROQ_API_KEY` 後執行 `python src/labeling_groq.py`。

標註支援中斷續跑（增量寫入）；換 prompt / schema 後可用 `--overwrite` 重標。

---

## 設計要點

1. **標註與篩選分離**：labeling 保留含 `none` 的完整分佈；進入 L2 前才由 `export_problem_pool.py` 過濾。
2. **模型只填 dimension**：`problem_axis` 由對照表決定，降低 axis / dimension 混淆。
3. **一 turn 可多列**：同一回合若有多個明確問題，展開為多筆 CSV 列。
4. **歷史輔助、當前為主**：最多帶入前 4 turns，用於消歧義與延續問題，不作為獨立標註來源。
