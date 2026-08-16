#!/usr/bin/env bash
# Generate L2 from LEXICON_BRIEF captions, re-label, keep dim_any matches.
# Target: 20 generated per dim × mode; ≥10 kept after relabel (one top-up round).
# Run from repo root. Requires GGUF. Kaggle T4x2: auto tensor_split; OOM → N_BATCH=256.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUTPUT_DIR:-outputs}"
POOL="${POOL:-outputs/labeled_turns_gguf_all_problems.csv}"
N_BATCH="${N_BATCH:-256}"
N_PER_DIM="${N_PER_DIM:-20}"
MIN_KEEP="${MIN_KEEP:-10}"
MODEL_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model-path "$MODEL_PATH")
fi

echo "=== 1/4 Generate 3 modes (n_per_dim=${N_PER_DIM}, pool=${POOL}) ==="
python src/generate_l2_style.py \
  --pool "$POOL" \
  --output-dir "$OUT" \
  --modes retarget strict free \
  --exemplar-content raw \
  --n-per-dim "$N_PER_DIM" \
  --n-exemplars 1 \
  --n-batch "$N_BATCH" \
  --overwrite \
  "${MODEL_ARGS[@]}"

echo "=== 2/4 Re-label ==="
python src/labeling/label_l2_generated.py \
  --inputs \
    "$OUT/l2_from_l1_retarget_raw.csv" \
    "$OUT/l2_from_l1_strict_raw.csv" \
    "$OUT/l2_from_l1_free_raw.csv" \
  --output-dir "$OUT" \
  --n-batch "$N_BATCH" \
  --overwrite \
  "${MODEL_ARGS[@]}"

echo "=== 3/4 Filter dim_any matches ==="
FILTER_LOG="$(mktemp)"
python scripts/filter_l2_by_relabel.py \
  --output-dir "$OUT" \
  --min-keep "$MIN_KEEP" | tee "$FILTER_LOG"
TOPUP_DIMS="$(sed -n 's/^TOPUP_DIMS //p' "$FILTER_LOG" | tail -n1 | xargs || true)"
rm -f "$FILTER_LOG"

if [[ -z "${TOPUP_DIMS}" ]]; then
  echo "All dims met min-keep=${MIN_KEEP}. No top-up."
  echo "Done. Kept CSVs: $OUT/l2_from_l1_{retarget,strict,free}_raw_kept.csv"
  exit 0
fi

echo "=== 4/4 Top-up shortfall dims (one extra round of ${N_PER_DIM}): ${TOPUP_DIMS} ==="
# Do not --overwrite: append new segment_ids; relabel skips already-done keys.
python src/generate_l2_style.py \
  --pool "$POOL" \
  --output-dir "$OUT" \
  --modes retarget strict free \
  --exemplar-content raw \
  --n-per-dim "$N_PER_DIM" \
  --n-exemplars 1 \
  --id-prefix lex_topup \
  --index-offset 0 \
  --dims ${TOPUP_DIMS} \
  --n-batch "$N_BATCH" \
  "${MODEL_ARGS[@]}"

python src/labeling/label_l2_generated.py \
  --inputs \
    "$OUT/l2_from_l1_retarget_raw.csv" \
    "$OUT/l2_from_l1_strict_raw.csv" \
    "$OUT/l2_from_l1_free_raw.csv" \
  --output-dir "$OUT" \
  --n-batch "$N_BATCH" \
  "${MODEL_ARGS[@]}"

python scripts/filter_l2_by_relabel.py \
  --output-dir "$OUT" \
  --min-keep "$MIN_KEEP"

echo "Done. Kept CSVs: $OUT/l2_from_l1_{retarget,strict,free}_raw_kept.csv"
echo "Report: $OUT/l2_relabel_keep_report.md"
