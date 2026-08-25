#!/usr/bin/env bash
# Full pipeline with the LEXICON_BRIEF labeling prompt:
#   MixAssist label → export pool → generate L2 → relabel L2 → compare → filter
# Single-GPU defaults (n_batch=512, no tensor_split). Run from repo root.
#
# Labeling prompt: src/prompts/labeling_prompt.py
# Relabel uses the same SYSTEM_PROMPT.
#
# SKIP_LABEL=1  — skip MixAssist re-label + pool export (only if you already
#                 ran labeling_gguf.py with the current prompt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUTPUT_DIR:-outputs}"
POOL="${POOL:-outputs/labeled_turns_gguf_all_problems.csv}"
L1="${L1:-new_outputs/l1_lexicon_captions_lex.jsonl}"
N_BATCH="${N_BATCH:-512}"
MIN_KEEP="${MIN_KEEP:-10}"
MODEL_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model-path "$MODEL_PATH")
fi

if [[ "${SKIP_LABEL:-0}" != "1" ]]; then
  echo "=== 1/6 MixAssist label (prompt=src/prompts/labeling_prompt.py, overwrite) ==="
  python src/labeling/labeling_gguf.py \
    --output-dir "$OUT" \
    --n-batch "$N_BATCH" \
    --overwrite \
    "${MODEL_ARGS[@]}"

  echo "=== 2/6 Export problem pool (incl. clean) ==="
  python scripts/export_problem_pool.py --output-dir "$OUT"
else
  echo "=== skip MixAssist label + pool export (SKIP_LABEL=1) ==="
fi

echo "=== 3/6 Generate L2 (l1=${L1}, pool=${POOL}) ==="
# Drop leftover mock_* rows from older CSVs so this run is only the L1 batch.
rm -f \
  "$OUT"/l2_from_l1_retarget_raw.csv \
  "$OUT"/l2_from_l1_strict_raw.csv \
  "$OUT"/l2_from_l1_free_raw.csv \
  "$OUT"/l2_from_l1_retarget_raw_raw.jsonl \
  "$OUT"/l2_from_l1_strict_raw_raw.jsonl \
  "$OUT"/l2_from_l1_free_raw_raw.jsonl
python src/generate_l2_style.py \
  --l1 "$L1" \
  --pool "$POOL" \
  --output-dir "$OUT" \
  --modes retarget strict free \
  --exemplar-content raw \
  --n-exemplars 1 \
  --n-batch "$N_BATCH" \
  --overwrite \
  "${MODEL_ARGS[@]}"

echo "=== 4/6 Re-label generated L2 (same labeling prompt) ==="
python src/labeling/label_l2_generated.py \
  --inputs \
    "$OUT/l2_from_l1_retarget_raw.csv" \
    "$OUT/l2_from_l1_strict_raw.csv" \
    "$OUT/l2_from_l1_free_raw.csv" \
  --output-dir "$OUT" \
  --n-batch "$N_BATCH" \
  --overwrite \
  "${MODEL_ARGS[@]}"

echo "=== 5/6 Compare agreement ==="
for mode in retarget strict free; do
  tag="${mode}_raw"
  python scripts/compare_l2_labels.py \
    --l2 "$OUT/l2_from_l1_${tag}.csv" \
    --labeled "$OUT/labeled_l2_gguf_${tag}.csv" \
    --report "$OUT/l2_label_agreement_${tag}.md"
done

echo "=== 6/6 Filter dim_any matches (min-keep=${MIN_KEEP}) ==="
python scripts/filter_l2_by_relabel.py \
  --output-dir "$OUT" \
  --min-keep "$MIN_KEEP"

echo "Done. Kept CSVs: $OUT/l2_from_l1_{retarget,strict,free}_raw_kept.csv"
echo "Reports: $OUT/l2_label_agreement_*_raw.md  $OUT/l2_relabel_keep_report.md"
