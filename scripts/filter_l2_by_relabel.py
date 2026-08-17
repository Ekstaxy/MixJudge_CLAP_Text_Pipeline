# -*- coding: utf-8 -*-
"""Keep L2 rows whose re-label includes gold_dim (dim_any).

Writes:
  {output-dir}/l2_from_l1_{mode}_raw_kept.csv
  {output-dir}/l2_relabel_keep_report.md
  {output-dir}/l2_shortfall.json

Usage (from repo root):
  python scripts/filter_l2_by_relabel.py --output-dir outputs --min-keep 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compare_l2_labels import compare, load_csv  # noqa: E402
from prompts.lexicon_slots import PROBLEM_DIMS  # noqa: E402

MODES = ("retarget", "strict", "free")
CONTENT = "raw"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def filter_mode(
    output_dir: Path,
    mode: str,
    min_keep: int,
    mismatch_limit: int,
) -> dict:
    l2_path = output_dir / f"l2_from_l1_{mode}_{CONTENT}.csv"
    labeled_path = output_dir / f"labeled_l2_gguf_{mode}_{CONTENT}.csv"
    if not l2_path.is_file():
        raise SystemExit(f"找不到 L2: {l2_path}")
    if not labeled_path.is_file():
        raise SystemExit(f"找不到 labeled: {labeled_path}")

    l2_rows = load_csv(l2_path)
    labeled_rows = load_csv(labeled_path)
    stats = compare(l2_rows, labeled_rows, mismatch_limit)
    keep_ids = {
        r["l1_segment_id"]
        for r in stats["results"]
        if r.get("dim_any_match")
    }
    fields = list(l2_rows[0].keys()) if l2_rows else []
    kept_rows = [
        r
        for r in l2_rows
        if str(r.get("l1_segment_id") or "").strip() in keep_ids
        and not str(r.get("error") or "").strip()
    ]
    kept_path = output_dir / f"l2_from_l1_{mode}_{CONTENT}_kept.csv"
    if fields:
        write_csv(kept_path, kept_rows, fields)

    by_dim: Counter[str] = Counter(
        str(r.get("gold_dim") or "").strip().lower() for r in kept_rows
    )
    gold_in_file = {
        str(r.get("gold_dim") or "").strip().lower()
        for r in l2_rows
        if str(r.get("gold_dim") or "").strip().lower()
    }
    ordered = [d for d in PROBLEM_DIMS if d in gold_in_file]
    dim_counts = {d: int(by_dim.get(d, 0)) for d in ordered}
    shortfall = {
        d: max(0, min_keep - n) for d, n in dim_counts.items() if n < min_keep
    }
    return {
        "mode": mode,
        "l2_path": str(l2_path),
        "labeled_path": str(labeled_path),
        "kept_path": str(kept_path),
        "n_l2": len(l2_rows),
        "n_scored": stats["n_scored"],
        "n_kept": len(kept_rows),
        "dim_any": list(stats["dim_any"]),
        "dim_counts": dim_counts,
        "shortfall": shortfall,
        "shortfall_dims": sorted(shortfall),
    }


def format_report(summaries: list[dict], min_keep: int) -> str:
    lines = [
        "# L2 relabel keep report",
        "",
        f"Keep criterion: **dim_any** (gold_dim in any predicted problem label).",
        f"Target: ≥ {min_keep} kept rows per dimension per mode.",
        "",
    ]
    header = "| dim | " + " | ".join(s["mode"] for s in summaries) + " |"
    sep = "|---|" + "|".join("---:" for _ in summaries) + "|"
    lines += ["## Kept counts", "", header, sep]
    dims = []
    seen: set[str] = set()
    for s in summaries:
        for d in s["dim_counts"]:
            if d not in seen:
                seen.add(d)
                dims.append(d)
    for dim in dims:
        cells = []
        for s in summaries:
            n = s["dim_counts"].get(dim, 0)
            mark = "" if n >= min_keep else " *"
            cells.append(f"{n}{mark}")
        lines.append(f"| `{dim}` | " + " | ".join(cells) + " |")
    lines += ["", "\\* below min-keep", "", "## Totals", ""]
    for s in summaries:
        hits, total, rate = s["dim_any"]
        lines.append(
            f"- **{s['mode']}**: kept {s['n_kept']} / scored {s['n_scored']} "
            f"(dim_any {hits}/{total} = {rate}) → `{s['kept_path']}`"
        )
        if s["shortfall_dims"]:
            bits = [f"{d}={s['dim_counts'][d]}" for d in s["shortfall_dims"]]
            lines.append(f"  - shortfall: {', '.join(bits)}")
        else:
            lines.append("  - all dims met min-keep")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter L2 CSVs to rows whose re-label matches gold_dim"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=None,
        help="modes to filter (default: whichever l2_from_l1_{mode}_raw.csv exist)",
    )
    parser.add_argument("--min-keep", type=int, default=10)
    parser.add_argument("--mismatch-limit", type=int, default=15)
    args = parser.parse_args()

    output_dir = args.output_dir
    modes = list(args.modes) if args.modes else list(MODES)
    present = []
    missing = []
    for mode in modes:
        l2_path = output_dir / f"l2_from_l1_{mode}_{CONTENT}.csv"
        labeled_path = output_dir / f"labeled_l2_gguf_{mode}_{CONTENT}.csv"
        if l2_path.is_file() and labeled_path.is_file():
            present.append(mode)
        else:
            missing.append(mode)
            print(f"[skip] {mode}: missing {l2_path.name if not l2_path.is_file() else labeled_path.name}")
    if not present:
        raise SystemExit(
            f"找不到可過濾的 mode。需要 l2_from_l1_{{mode}}_raw.csv "
            f"+ labeled_l2_gguf_{{mode}}_raw.csv under {output_dir}"
        )
    summaries = [
        filter_mode(output_dir, mode, args.min_keep, args.mismatch_limit)
        for mode in present
    ]

    union_dims = sorted({d for s in summaries for d in s["shortfall_dims"]})
    payload = {
        "min_keep": args.min_keep,
        "modes": {s["mode"]: s for s in summaries},
        "union_shortfall_dims": union_dims,
    }
    shortfall_path = output_dir / "l2_shortfall.json"
    shortfall_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "l2_relabel_keep_report.md"
    report_path.write_text(
        format_report(summaries, args.min_keep), encoding="utf-8"
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {shortfall_path}")
    for s in summaries:
        print(
            f"  {s['mode']:<10} kept {s['n_kept']:>4}  "
            f"shortfall dims={s['shortfall_dims'] or 'none'}"
        )
        print(f"    {s['kept_path']}")
    if union_dims:
        print("TOPUP_DIMS " + " ".join(union_dims))
    else:
        print("TOPUP_DIMS")
    # exit 0 even with shortfall — driver decides whether to refill


if __name__ == "__main__":
    main()
