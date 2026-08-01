# -*- coding: utf-8 -*-
"""
Compare L1 gold labels vs re-labeled L2 generation outputs.

Joins:
  L2 CSV   (gold_axis / gold_dim / source_instrument / amateur_text)
  labeled  (problem_axis / problem_dimension / problem_stem — same schema
            as MixAssist labeled_turns_*.csv)

A turn may expand to multiple label rows. Success criterion (default):
  dim_any  — gold_dim appears in ANY predicted problem label
  axis_any — gold_axis appears in ANY predicted problem label

Primary (first has_problem label) is reported only as a secondary diagnostic.

Usage (from repo root):
  python scripts/compare_l2_labels.py \\
    --l2 outputs/l2_from_l1_retarget_raw.csv \\
    --labeled outputs/labeled_l2_gguf_retarget_raw.csv \\
    --report outputs/l2_label_agreement_retarget_raw.md
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pct(n: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * n / total:.1f}%"


def is_true(v: str) -> bool:
    return str(v or "").strip().lower() == "true"


def has_error(row: dict) -> bool:
    return bool(str(row.get("error") or "").strip())


def is_problem_row(row: dict) -> bool:
    dim = str(row.get("problem_dimension") or "").strip().lower()
    return is_true(row.get("has_problem", "")) and dim not in ("", "none")


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"找不到檔案: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_primary(pred_rows: list[dict]) -> dict | None:
    """First has_problem + dim!=none; else first non-error row; else None."""
    usable = [r for r in pred_rows if not has_error(r)]
    for r in usable:
        if is_problem_row(r):
            return r
    return usable[0] if usable else None


def summarize_preds(pred_rows: list[dict]) -> dict:
    usable = [r for r in pred_rows if not has_error(r)]
    errored = [r for r in pred_rows if has_error(r)]
    problem_rows = [r for r in usable if is_problem_row(r)]
    dims = [
        str(r.get("problem_dimension") or "").strip().lower() for r in problem_rows
    ]
    axes = [str(r.get("problem_axis") or "").strip().lower() for r in problem_rows]
    primary = select_primary(pred_rows)
    if primary and is_problem_row(primary):
        primary_dim = str(primary.get("problem_dimension") or "").strip().lower()
        primary_axis = str(primary.get("problem_axis") or "").strip().lower()
        primary_stem = str(primary.get("problem_stem") or "").strip().lower()
    elif primary:
        primary_dim = "none"
        primary_axis = "none"
        primary_stem = str(primary.get("problem_stem") or "").strip().lower()
    else:
        primary_dim = ""
        primary_axis = ""
        primary_stem = ""

    return {
        "n_pred_rows": len(pred_rows),
        "n_usable": len(usable),
        "n_error_rows": len(errored),
        "has_error": bool(errored) and not usable,
        "has_problem": bool(problem_rows),
        "dims": dims,
        "axes": axes,
        "primary_dim": primary_dim,
        "primary_axis": primary_axis,
        "primary_stem": primary_stem,
        "error_msg": str(errored[0].get("error") or "") if errored else "",
    }


def truncate(text: str, n: int = 80) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def compare(
    l2_rows: list[dict],
    labeled_rows: list[dict],
    mismatch_limit: int,
) -> dict:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in labeled_rows:
        cid = str(r.get("conversation_id") or "").strip()
        if cid:
            by_id[cid].append(r)

    results = []
    skipped = 0
    for row in l2_rows:
        if str(row.get("error") or "").strip():
            skipped += 1
            continue
        seg = str(row.get("l1_segment_id") or "").strip()
        amateur = str(row.get("amateur_text") or "").strip()
        expert = str(row.get("expert_text") or "").strip()
        if not seg or (not amateur and not expert):
            skipped += 1
            continue

        gold_dim = str(row.get("gold_dim") or "").strip().lower()
        gold_axis = str(row.get("gold_axis") or "").strip().lower()
        source = str(row.get("source_instrument") or "").strip().lower()
        pred = summarize_preds(by_id.get(seg, []))

        missing = seg not in by_id
        dim_primary = (not missing) and pred["primary_dim"] == gold_dim
        axis_primary = (not missing) and pred["primary_axis"] == gold_axis
        dim_any = (not missing) and gold_dim in pred["dims"]
        axis_any = (not missing) and gold_axis in pred["axes"]
        stem_match = (
            (not missing)
            and bool(source)
            and pred["primary_stem"] == source
        )

        results.append(
            {
                "l1_segment_id": seg,
                "gold_dim": gold_dim,
                "gold_axis": gold_axis,
                "source_instrument": source,
                "amateur_text": amateur,
                "missing": missing,
                "has_error": pred["has_error"],
                "error_msg": pred["error_msg"],
                "has_problem": pred["has_problem"],
                "primary_dim": pred["primary_dim"],
                "primary_axis": pred["primary_axis"],
                "primary_stem": pred["primary_stem"],
                "pred_dims": pred["dims"],
                "dim_primary_match": dim_primary,
                "dim_any_match": dim_any,
                "axis_primary_match": axis_primary,
                "axis_any_match": axis_any,
                "stem_match": stem_match,
            }
        )

    n = len(results)
    scored = [r for r in results if not r["missing"] and not r["has_error"]]
    n_scored = len(scored)
    n_missing = sum(1 for r in results if r["missing"])
    n_error = sum(1 for r in results if r["has_error"])
    n_has_problem = sum(1 for r in scored if r["has_problem"])

    def rate(key: str) -> tuple[int, int, str]:
        hits = sum(1 for r in scored if r[key])
        return hits, n_scored, pct(hits, n_scored)

    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_dim[r["gold_dim"]].append(r)

    dim_breakdown = []
    for dim in sorted(by_dim):
        group = by_dim[dim]
        g_n = len(group)
        dim_breakdown.append(
            {
                "gold_dim": dim,
                "n": g_n,
                "dim_primary": pct(sum(1 for r in group if r["dim_primary_match"]), g_n),
                "dim_any": pct(sum(1 for r in group if r["dim_any_match"]), g_n),
                "axis_primary": pct(
                    sum(1 for r in group if r["axis_primary_match"]), g_n
                ),
                "axis_any": pct(sum(1 for r in group if r["axis_any_match"]), g_n),
            }
        )

    mismatches = [
        r for r in scored if not r["dim_any_match"]
    ][:mismatch_limit]

    # Confusions only when gold is missing from ALL predicted dims
    confusion = Counter()
    for r in scored:
        if r["dim_any_match"]:
            continue
        preds = r["pred_dims"] or ["(none)"]
        for p in preds:
            confusion[(r["gold_dim"], p)] += 1

    return {
        "n_l2_joined": n,
        "n_skipped": skipped,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "n_error": n_error,
        "n_has_problem": n_has_problem,
        "has_problem_rate": pct(n_has_problem, n_scored),
        "dim_primary": rate("dim_primary_match"),
        "dim_any": rate("dim_any_match"),
        "axis_primary": rate("axis_primary_match"),
        "axis_any": rate("axis_any_match"),
        "stem": rate("stem_match"),
        "dim_breakdown": dim_breakdown,
        "mismatches": mismatches,
        "confusion": confusion.most_common(20),
        "results": results,
    }


def format_report(stats: dict, l2_path: Path, labeled_path: Path) -> str:
    dim_hits, dim_total, dim_rate = stats["dim_any"]
    axis_hits, axis_total, axis_rate = stats["axis_any"]
    lines = [
        "# L2 Label Agreement",
        "",
        f"- L2 (gold): `{l2_path}`",
        f"- Labeled: `{labeled_path}`",
        f"- Joined rows: {stats['n_l2_joined']} "
        f"(skipped invalid L2: {stats['n_skipped']})",
        f"- Scored (has pred, no error): {stats['n_scored']}",
        f"- Missing pred: {stats['n_missing']}",
        f"- Pred error: {stats['n_error']}",
        "",
        "## Pass criterion: any-match",
        "",
        "One L2 dialogue may re-label to **multiple** problem rows "
        "(same as MixAssist). Gold counts as a hit if **any** predicted "
        "`problem_dimension` / `problem_axis` matches.",
        "",
        f"- **dim_any (PASS): {dim_hits}/{dim_total} ({dim_rate})**",
        f"- **axis_any (PASS): {axis_hits}/{axis_total} ({axis_rate})**",
        "",
        "## Overall rates (scored only)",
        "",
        "| Metric | Hits | Total | Rate |",
        "|--------|------|-------|------|",
    ]

    for label, key in [
        ("dim_any_match (PASS)", "dim_any"),
        ("axis_any_match (PASS)", "axis_any"),
        ("dim_primary_match (diagnostic)", "dim_primary"),
        ("axis_primary_match (diagnostic)", "axis_primary"),
        ("stem_match (vs source_instrument)", "stem"),
    ]:
        hits, total, rate = stats[key]
        lines.append(f"| {label} | {hits} | {total} | {rate} |")

    lines.extend(
        [
            "",
            f"- has_problem_rate: {stats['n_has_problem']}/{stats['n_scored']} "
            f"({stats['has_problem_rate']})",
            "",
            "## By gold_dim",
            "",
            "| gold_dim | n | dim_any (PASS) | axis_any | dim_primary | axis_primary |",
            "|----------|---|---------------|----------|-------------|--------------|",
        ]
    )
    for row in stats["dim_breakdown"]:
        lines.append(
            f"| {row['gold_dim']} | {row['n']} | {row['dim_any']} | "
            f"{row['axis_any']} | {row['dim_primary']} | {row['axis_primary']} |"
        )

    if stats["confusion"]:
        lines.extend(
            [
                "",
                "## Top gold→pred dim confusions (dim_any misses only)",
                "",
                "| gold_dim | pred_dim | count |",
                "|----------|----------|-------|",
            ]
        )
        for (g, p), c in stats["confusion"]:
            lines.append(f"| {g} | {p or '(empty)'} | {c} |")

    if stats["mismatches"]:
        lines.extend(
            [
                "",
                "## Mismatch samples (dim_any miss: gold not in any pred label)",
                "",
            ]
        )
        for r in stats["mismatches"]:
            lines.append(
                f"- `{r['l1_segment_id']}`: gold `{r['gold_axis']}/{r['gold_dim']}` "
                f"→ all pred dims={r['pred_dims'] or ['none']} "
                f"(primary `{r['primary_axis']}/{r['primary_dim']}`)"
            )
            lines.append(f"  - amateur: {truncate(r['amateur_text'])}")

    lines.append("")
    return "\n".join(lines)


def print_console(stats: dict) -> None:
    print(f"Joined: {stats['n_l2_joined']}  scored: {stats['n_scored']}  "
          f"missing: {stats['n_missing']}  error: {stats['n_error']}")
    print("Pass = any predicted label matches gold (multi-label OK):")
    for label, key in [
        ("dim_any", "dim_any"),
        ("axis_any", "axis_any"),
    ]:
        hits, total, rate = stats[key]
        print(f"  {label:14s}  {hits}/{total}  ({rate})")
    print("Diagnostic (first label only):")
    for label, key in [
        ("dim_primary", "dim_primary"),
        ("axis_primary", "axis_primary"),
        ("stem", "stem"),
    ]:
        hits, total, rate = stats[key]
        print(f"  {label:14s}  {hits}/{total}  ({rate})")
    print(
        f"  has_problem    {stats['n_has_problem']}/{stats['n_scored']}  "
        f"({stats['has_problem_rate']})"
    )
    if stats["mismatches"]:
        print("\nMismatch samples (gold missing from ALL pred dims):")
        for r in stats["mismatches"][:5]:
            print(
                f"  {r['l1_segment_id']}: "
                f"{r['gold_dim']} not in {r['pred_dims'] or ['none']} | "
                f"{truncate(r['amateur_text'], 60)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare L1 gold vs re-labeled L2 dialogues"
    )
    parser.add_argument(
        "--l2",
        type=Path,
        required=True,
        help="L2 generation CSV with gold_axis / gold_dim",
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        required=True,
        help="Re-labeled CSV (labeled_l2_gguf_*.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write markdown report (default: outputs/l2_label_agreement_<mode>.md)",
    )
    parser.add_argument(
        "--mismatch-limit",
        type=int,
        default=15,
        help="Max mismatch samples in the report (default 15)",
    )
    args = parser.parse_args()

    l2_path = args.l2 if args.l2.is_absolute() else PROJECT_ROOT / args.l2
    labeled_path = (
        args.labeled if args.labeled.is_absolute() else PROJECT_ROOT / args.labeled
    )

    if args.report is None:
        # Derive tag from labeled filename: labeled_l2_gguf_retarget_raw → retarget_raw
        stem = labeled_path.stem
        prefix = "labeled_l2_gguf_"
        if stem.startswith(prefix):
            tag = stem[len(prefix) :]
        else:
            tag = "unknown"
            for m in ("retarget", "strict", "free"):
                if stem.endswith(m):
                    tag = m
                    break
        report_path = PROJECT_ROOT / "outputs" / f"l2_label_agreement_{tag}.md"
    else:
        report_path = (
            args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
        )

    l2_rows = load_csv(l2_path)
    labeled_rows = load_csv(labeled_path)
    stats = compare(l2_rows, labeled_rows, args.mismatch_limit)

    print_console(stats)
    report = format_report(stats, l2_path, labeled_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
