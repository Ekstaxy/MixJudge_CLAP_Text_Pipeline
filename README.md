# MixJudge CLAP Text Pipeline

Use real MixAssist mixing dialogues to give synthetic training data a more natural text layer (L2 / L3). This repo currently focuses on **Phase 2a: dialogue labeling → problem pool → L2 style transfer**.

Synthetic L1 labels are correct but read like a textbook. MixAssist is Amateur / Expert talk from real sessions about problems and fixes. This project first labels MixAssist into a searchable axis–dimension structure, then generates problem descriptions that sound closer to real engineers.

---

## Architecture

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
        ├──► analyze_labels  → stats / spot checks
        └──► generate_l2_style → Amateur / Expert style dialogues
```

See [`labeling_pipeline.md`](labeling_pipeline.md) and the diagram below:

![Labeling pipeline](Labeling_Flow_Chart.png)

**Three text layers (project goal):**

| Layer | Content | Status |
|-------|---------|--------|
| **L1** | Structured templates (axis / dim / subject…); training anchors; not rewritten | Existing |
| **L2** | Problem descriptions in real mixing-dialogue style (retrieval + style transfer) | In progress |
| **L3** | L2 plus natural fix wording from the same turn | Later |

The labeling vocabulary is MixJudge **7 axes / 12 classes** (11 problem + `clean`; see [`LEXICON_BRIEF.md`](LEXICON_BRIEF.md)). `none` means the CURRENT TURN cannot be distinguished.

---

## Repository layout (what is visible on GitHub)

```text
.
├── README.md
├── .gitignore
├── Labeling_Flow_Chart.png
├── labeling_pipeline.md
├── LEXICON_BRIEF.md
├── L1_Quality_and_L2_Generation.md
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

> Large data (`*.csv`, `outputs/`, `models/*.gguf`, audio, etc.) is listed in `.gitignore` and will not appear on GitHub. Prepare MixAssist split CSVs and model weights locally.

---

## Modules

| File | Role |
|------|------|
| `src/labeling/labeling_common.py` | Shared: `DIMENSION_TO_AXIS`, JSON parsing, incremental CSV / JSONL writes |
| `src/prompts/labeling_prompt.py` | `SYSTEM_PROMPT` and user-message assembly |
| `src/labeling/labeling_gguf.py` | Local GGUF inference; default per-split output `outputs/labeled_turns_gguf_{split}.csv` |
| `src/labeling/labeling_hf.py` | Hugging Face local inference |
| `src/labeling/labeling_groq.py` | Cloud API inference for fast prompt / model trials |
| `scripts/export_problem_pool.py` | Filter valid problem rows; write `*_problems.csv` and `*_all_problems.csv` |
| `scripts/analyze_labels.py` | Coverage, axis / dimension distribution, sampled review |
| `src/generate_l2_style.py` | LEXICON_BRIEF L1 captions + MixAssist style pool → Amateur / Expert dialogues |
| `src/labeling/label_l2_generated.py` | Re-run labeling on generated L2 (same GGUF / prompt); write `labeled_l2_gguf_{mode}_raw.csv` |
| `scripts/compare_l2_labels.py` | Compare L1 gold vs re-labels; report dim / axis agreement |
| `scripts/filter_l2_by_relabel.py` | Keep dim_any hits → `l2_from_l1_{mode}_raw_kept.csv` |

---

## Docs

| Doc | Description |
|-----|-------------|
| [`labeling_pipeline.md`](labeling_pipeline.md) | Labeling pipeline: context, JSON schema, normalization, L2 pool export |
| [`LEXICON_BRIEF.md`](LEXICON_BRIEF.md) | MixJudge 11-dim ontology, pairs that must stay distinct, caption QUALITY wording |
| [`L1_Quality_and_L2_Generation.md`](L1_Quality_and_L2_Generation.md) | Lexicon extraction → Quality lexicon → L1/L2 generation and relabel filter |
| [`label_stats.md`](label_stats.md) | Coverage and axis / dimension stats after full labeling |
| [`generation_note.md`](generation_note.md) | L2 generation I/O and style-prompt notes |
| [`intern_chores.md`](intern_chores.md) | Background goals, L1/L2/L3, early 9-key plan, and Phase checklist |

---

## Running locally (overview)

Data and models live next to the repo root (none of this is in git):

- Input: `train.csv` / `validation.csv` / `test.csv`
- Model (GGUF path): `models/google_gemma-4-31B-it-Q4_K_M.gguf`
- Output: `outputs/`

```bash
# Full pipeline (label → pool → generate → relabel → compare → filter)
# MixAssist uses --overwrite because the prompt is now LEXICON_BRIEF 12-class.
bash scripts/run_l2_raw_variants.sh
# If MixAssist is already labeled with the current prompt:
# SKIP_LABEL=1 bash scripts/run_l2_raw_variants.sh

# Or step by step:
# 1) MixAssist labeling (src/prompts/labeling_prompt.py)
python src/labeling/labeling_gguf.py --n-batch 512 --overwrite

# 2) Export L2 problem pool
python scripts/export_problem_pool.py

# 3) Stats
python scripts/analyze_labels.py

# 4) L2 generation
python src/generate_l2_style.py \
  --l1 new_outputs/l1_lexicon_captions_lex.jsonl \
  --pool outputs/labeled_turns_gguf_all_problems.csv \
  --modes retarget strict free \
  --exemplar-content raw --n-exemplars 1 --n-batch 512 --overwrite

# 5) Re-label generated dialogues (same labeling prompt)
python src/labeling/label_l2_generated.py \
  --inputs outputs/l2_from_l1_retarget_raw.csv \
           outputs/l2_from_l1_strict_raw.csv \
           outputs/l2_from_l1_free_raw.csv \
  --output-dir outputs --n-batch 512 --overwrite

# 6) Compare gold vs re-label agreement
for mode in retarget strict free; do
  tag="${mode}_raw"
  python scripts/compare_l2_labels.py \
    --l2 "outputs/l2_from_l1_${tag}.csv" \
    --labeled "outputs/labeled_l2_gguf_${tag}.csv" \
    --report "outputs/l2_label_agreement_${tag}.md"
done

# 7) Filter dim_any hits
python scripts/filter_l2_by_relabel.py --output-dir outputs --min-keep 10
```

For Groq, set `GROQ_API_KEY` and run `python src/labeling/labeling_groq.py`.

Labeling can resume after interrupt (incremental writes). After a prompt / schema change, re-label with `--overwrite`.

---

## Design notes

1. **Labeling and filtering are separate.** Labeling keeps the full distribution including `none`; `export_problem_pool.py` filters only before L2.
2. **The model fills dimension only.** `problem_axis` comes from a lookup table so axis / dimension do not get mixed up.
3. **One turn can become several rows.** If a turn has several distinct problems, they expand to multiple CSV rows.
4. **History helps; the current turn is primary.** Up to 4 previous turns are included for disambiguation and continuing problems, not as a standalone labeling source.
