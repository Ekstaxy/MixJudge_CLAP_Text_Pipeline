# -*- coding: utf-8 -*-
"""Top-N unique descs per axis or dimension (exact-desc dedupe).

JSON: only desc lists, padded with "" to top_n.

Label priority: corrected_* (human) if non-empty, else assigned_*.

Taxonomy: MixJudge 7 axes / 12 classes (11 problem + clean; same as term_extract).

Usage:
  # 7 axes
  python scripts/extract_lexicon.py \
    --level axis \
    --centroid outputs/lexicon_review_centroid_axis.csv \
    --centroid-out outputs/lexicon_top_centroid_axis.json \
    --clap outputs/lexicon_review_clap_axis.csv \
    --clap-out outputs/lexicon_top_clap_axis.json \
    --top-n 10

  # 12 classes (11 problem + clean)
  python scripts/extract_lexicon.py \
    --level dim \
    --centroid outputs/lexicon_review_centroid_dim.csv \
    --centroid-out outputs/lexicon_top_centroid_dim.json \
    --clap outputs/lexicon_review_clap_dim.csv \
    --clap-out outputs/lexicon_top_clap_dim.json \
    --top-n 10
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

AXES = (
    "level",
    "body",
    "brightness",
    "space",
    "dynamic",
    "masking",
    "clean",
)

DIMENSIONS = (
    "too_quiet",
    "too_loud",
    "muddy",
    "thin",
    "harsh",
    "dull",
    "too_wet",
    "too_dry",
    "over_compressed",
    "under_compressed",
    "masking",
    "clean",
)


def effective_label(row: dict, assigned_col: str, corrected_col: str) -> str:
    """Human corrected_* wins when non-empty; else machine assigned_*."""
    corr = (row.get(corrected_col) or "").strip()
    if corr:
        return corr
    return (row.get(assigned_col) or "").strip()


def top_descs(
    csv_path: Path,
    top_n: int,
    assigned_col: str,
    corrected_col: str,
    labels: tuple[str, ...],
) -> tuple[dict[str, list[str]], int]:
    by_lab: dict[str, list[tuple[float, str]]] = defaultdict(list)
    n_corr = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lab = effective_label(row, assigned_col, corrected_col)
            if lab not in labels:
                continue
            if (row.get(corrected_col) or "").strip():
                n_corr += 1
            desc = row.get("desc") or ""
            if not desc:
                continue
            sim = float(row.get("similarity") or 0.0)
            by_lab[lab].append((sim, desc))

    out: dict[str, list[str]] = {}
    for lab in labels:
        seen: set[str] = set()
        picked: list[str] = []
        for _, desc in sorted(by_lab.get(lab, []), key=lambda x: -x[0]):
            if desc in seen:
                continue
            seen.add(desc)
            picked.append(desc)
            if len(picked) >= top_n:
                break
        while len(picked) < top_n:
            picked.append("")
        out[lab] = picked
    return out, n_corr


def write_one(
    csv_path: Path,
    out_path: Path,
    top_n: int,
    label: str,
    assigned_col: str,
    corrected_col: str,
    labels: tuple[str, ...],
) -> None:
    data, n_corr = top_descs(csv_path, top_n, assigned_col, corrected_col, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    filled = {a: sum(1 for d in data[a] if d) for a in labels}
    print(f"[{label}] filled={filled} (rows_with_corrected={n_corr})")
    print(f"  → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Top-N unique descs per axis/dim → JSON")
    p.add_argument("--level", choices=("axis", "dim"), required=True, help="axis (7) or dim (12=11+clean)")
    p.add_argument("--centroid", type=Path, required=True)
    p.add_argument("--centroid-out", type=Path, required=True)
    p.add_argument("--clap", type=Path, required=True)
    p.add_argument("--clap-out", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    if args.level == "axis":
        assigned_col, corrected_col, labels = "assigned_axis", "corrected_axis", AXES
    else:
        assigned_col, corrected_col, labels = "assigned_dimension", "corrected_dimension", DIMENSIONS

    write_one(
        args.centroid,
        args.centroid_out,
        args.top_n,
        f"centroid-{args.level}",
        assigned_col,
        corrected_col,
        labels,
    )
    write_one(
        args.clap,
        args.clap_out,
        args.top_n,
        f"clap-{args.level}",
        assigned_col,
        corrected_col,
        labels,
    )


if __name__ == "__main__":
    main()
