#!/usr/bin/env bash
# Remap lexicon hierarchically (8 axes → then 2 dims within axis; phase=1),
# then export top-N JSON for centroid/clap.
# Does NOT re-run Gemma extract. Run from repo root (or anywhere; script cds).
#
# Usage:
#   bash scripts/run_lexicon_map_top.sh
#   bash scripts/run_lexicon_map_top.sh 20          # top-n
#   bash scripts/run_lexicon_map_top.sh 10 cpu both
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TOP_N="${1:-10}"
DEVICE="${2:-cpu}"
METHOD="${3:-both}"

echo "=== 1/3 map-and-export (method=${METHOD}, device=${DEVICE}) ==="
python src/term_extract.py map-and-export --method "${METHOD}" --device "${DEVICE}"

echo "=== 2/3 top-N axis (n=${TOP_N}) ==="
python scripts/extract_lexicon.py --level axis \
  --centroid outputs/lexicon_review_centroid_axis.csv \
  --centroid-out outputs/lexicon_top_centroid_axis.json \
  --clap outputs/lexicon_review_clap_axis.csv \
  --clap-out outputs/lexicon_top_clap_axis.json \
  --top-n "${TOP_N}"

echo "=== 3/3 top-N dim (n=${TOP_N}) ==="
python scripts/extract_lexicon.py --level dim \
  --centroid outputs/lexicon_review_centroid_dim.csv \
  --centroid-out outputs/lexicon_top_centroid_dim.json \
  --clap outputs/lexicon_review_clap_dim.csv \
  --clap-out outputs/lexicon_top_clap_dim.json \
  --top-n "${TOP_N}"

echo "Done."
echo "  outputs/lexicon_top_centroid_axis.json"
echo "  outputs/lexicon_top_clap_axis.json"
echo "  outputs/lexicon_top_centroid_dim.json"
echo "  outputs/lexicon_top_clap_dim.json"
