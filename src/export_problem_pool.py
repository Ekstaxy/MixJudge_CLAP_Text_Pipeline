# -*- coding: utf-8 -*-
"""
從 labeling_gguf 全量 CSV 匯出「只有 problem state」的 L2 retrieval pool。

保留原檔不動；另存精簡檔，供 style transfer / L2 當 exemplar。

篩選條件 (預設):
  - 無 error
  - has_problem = True
  - problem_dimension 非空且 != none
  - problem_text 非空

用法:
  python src/export_problem_pool.py
  python src/export_problem_pool.py --splits train validation test
  python src/export_problem_pool.py --min-confidence mid
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PREFIX = "labeled_turns_gguf"
SPLITS = ("train", "validation", "test")

# L2 用欄位 (精簡,仍保留對齊 audio / 驗證所需)
L2_FIELDS = [
    "conversation_id",
    "split",
    "topic",
    "turn_id",
    "has_content",
    "audio_file",
    "problem_stem",
    "problem_axis",
    "problem_dimension",
    "problem_speaker",
    "problem_text",
    "vocal_lead",
    "confidence",
    # 原文保留,方便抽查 style;生成時可只用 problem_text
    "user_raw_content",
    "assistant_raw_content",
]

CONF_RANK = {"low": 0, "mid": 1, "high": 2}


def is_true(v: str) -> bool:
    return str(v or "").strip().lower() == "true"


def keep_row(row: dict, min_confidence: str | None) -> bool:
    if str(row.get("error") or "").strip():
        return False
    if not is_true(row.get("has_problem", "")):
        return False
    dim = str(row.get("problem_dimension") or "").strip().lower()
    if dim in ("", "none"):
        return False
    if not str(row.get("problem_text") or "").strip():
        return False
    if min_confidence:
        conf = str(row.get("confidence") or "").strip().lower()
        if CONF_RANK.get(conf, -1) < CONF_RANK.get(min_confidence, 0):
            return False
    return True


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def summarize(name: str, rows: list[dict]) -> None:
    print(f"\n[{name}] problem rows: {len(rows)}")
    if not rows:
        return
    print("  by axis:")
    for axis, n in Counter(r.get("problem_axis") or "(empty)" for r in rows).most_common():
        print(f"    {axis:<12} {n:>4}")
    print("  by dimension:")
    for dim, n in Counter(r.get("problem_dimension") or "(empty)" for r in rows).most_common():
        print(f"    {dim:<22} {n:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export L2 problem-only retrieval pools")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument(
        "--min-confidence",
        choices=["low", "mid", "high"],
        default=None,
        help="optional: drop rows below this confidence",
    )
    parser.add_argument(
        "--also-combined",
        action="store_true",
        default=True,
        help="also write combined all-splits file (default on)",
    )
    parser.add_argument("--no-combined", action="store_true", help="skip combined file")
    args = parser.parse_args()

    combined: list[dict] = []
    for split in args.splits:
        src = args.output_dir / f"{args.prefix}_{split}.csv"
        rows = load_csv(src)
        kept = [r for r in rows if keep_row(r, args.min_confidence)]
        for r in kept:
            if not r.get("split"):
                r["split"] = split
        out = args.output_dir / f"{args.prefix}_{split}_problems.csv"
        write_csv(out, kept, L2_FIELDS)
        print(f"Wrote {out} ({len(kept)} / {len(rows)} rows)")
        summarize(split, kept)
        combined.extend(kept)

    if args.also_combined and not args.no_combined and len(args.splits) > 1:
        out = args.output_dir / f"{args.prefix}_all_problems.csv"
        write_csv(out, combined, L2_FIELDS)
        print(f"\nWrote {out} ({len(combined)} rows)")
        summarize("all", combined)


if __name__ == "__main__":
    main()
