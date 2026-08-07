#!/usr/bin/env python3
"""Build a self-contained HTML viewer for L2 retarget/strict/free examples.

Usage (from repo root):
  python scripts/build_l2_examples_viewer.py
  # then open outputs/l2_examples_viewer.html in a browser
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
VARIANTS = ("retarget_raw", "strict_raw", "free_raw")


def main() -> None:
    l2: dict[str, dict[str, dict]] = {}
    for v in VARIANTS:
        path = OUT / f"l2_from_l1_{v}.csv"
        l2[v] = {
            r["l1_segment_id"]: r
            for r in csv.DictReader(path.open(encoding="utf-8-sig"))
        }

    labs: dict[str, dict[str, list]] = {}
    for v in VARIANTS:
        by: dict[str, list] = defaultdict(list)
        path = OUT / f"labeled_l2_gguf_{v}.csv"
        if path.exists():
            for r in csv.DictReader(path.open(encoding="utf-8-sig")):
                by[r["conversation_id"]].append(r)
        labs[v] = by

    base = l2["retarget_raw"]
    items = []
    for sid, row in base.items():
        item = {
            "id": sid,
            "gold_dim": row["gold_dim"],
            "gold_axis": row["gold_axis"],
            "l1": row["l1_text"],
            "source": row["source_instrument"],
            "style_turn": row["style_turn_ids"],
            "style_pt": row.get("style_problem_texts") or "",
            "style_u": row.get("style_user_raw_contents") or "",
            "style_a": row.get("style_assistant_raw_contents") or "",
            "modes": {},
        }
        for v in VARIANTS:
            r = l2[v][sid]
            preds = labs[v].get(sid, [])
            dims = [p.get("problem_dimension") or "none" for p in preds]
            mode = v.replace("_raw", "")
            item["modes"][mode] = {
                "amateur": r.get("amateur_text") or "",
                "expert": r.get("expert_text") or "",
                "pred_dims": dims,
                "hit": item["gold_dim"] in dims,
                "style_turn": r.get("style_turn_ids") or "",
            }
        items.append(item)

    (OUT / "l2_examples_viewer_data.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html_path = OUT / "l2_examples_viewer.html"
    template = (Path(__file__).with_name("l2_examples_viewer_template.html")).read_text(
        encoding="utf-8"
    )
    html_path.write_text(
        template.replace("__DATA__", json.dumps(items, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"wrote {html_path}")
    print(f"open: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
