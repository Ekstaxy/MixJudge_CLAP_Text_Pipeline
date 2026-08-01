#!/usr/bin/env bash
# Generate L2 raw-text variants only (3 modes), re-label, and compare.
# Run from repo root. Requires GGUF model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1/3 Generate 3 CSVs (raw only, temp=0.1, n_exemplars=1, seed=123) ==="
python src/generate_l2_style.py \
  --modes retarget strict free \
  --exemplar-content raw \
  --n-exemplars 1 \
  --seed 123 \
  --overwrite

echo "=== 2/3 Re-label raw variants ==="
python src/labeling/label_l2_generated.py \
  --inputs \
    outputs/l2_from_l1_retarget_raw.csv \
    outputs/l2_from_l1_strict_raw.csv \
    outputs/l2_from_l1_free_raw.csv \
  --overwrite

echo "=== 3/3 Compare agreement ==="
for mode in retarget strict free; do
  tag="${mode}_raw"
  python scripts/compare_l2_labels.py \
    --l2 "outputs/l2_from_l1_${tag}.csv" \
    --labeled "outputs/labeled_l2_gguf_${tag}.csv" \
    --report "outputs/l2_label_agreement_${tag}.md"
done

echo "Done. Reports: outputs/l2_label_agreement_*_raw.md"
