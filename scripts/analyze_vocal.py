# -*- coding: utf-8 -*-
"""
Vocal-related distribution analysis on labeling CSV outputs.

預設讀 labeling_gguf 三個 split:
  outputs/labeled_turns_gguf_{train,validation,test}.csv

也可指定單一檔:
  python scripts/analyze_vocal.py --csv outputs/labeled_turns_gguf_all_problems.csv
  python scripts/analyze_vocal.py --csv outputs/labeled_turns_groq_llama-3.3-70b-versatile.csv

輸出:
  outputs/vocal_stats.md
  outputs/vocal_stats.json
  (可另存 repo 根目錄 vocal_stats.md)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PREFIX = "labeled_turns_gguf"
SPLITS = ("train", "validation", "test")

VOCAL_STEMS = {"vocal"}  # problem_stem / fix_stem 正規化後的 lead/backing 共用 stem


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def pct_s(n: int, total: int) -> str:
    return f"{pct(n, total):.1f}%" if total else "n/a"


def is_true(v: str) -> bool:
    return str(v or "").strip().lower() == "true"


def has_error(row: dict) -> bool:
    return bool(str(row.get("error") or "").strip())


def is_problem(row: dict) -> bool:
    dim = str(row.get("problem_dimension") or "").strip().lower()
    return is_true(row.get("has_problem", "")) and dim not in ("", "none")


def is_fix(row: dict) -> bool:
    if not is_true(row.get("has_fix", "")):
        return False
    return bool(
        str(row.get("fix_text") or "").strip()
        or str(row.get("fix_action") or "").strip()
    )


def stem(row: dict, key: str = "problem_stem") -> str:
    return str(row.get(key) or "").strip().lower()


def is_vocal_stem(row: dict, key: str = "problem_stem") -> bool:
    return stem(row, key) in VOCAL_STEMS


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def counter_rows(rows: list[dict], key: str) -> Counter:
    return Counter(str(r.get(key) or "").strip() or "(empty)" for r in rows)


def counter_to_list(c: Counter, total: int | None = None, top: int | None = None) -> list[dict]:
    items = c.most_common(top)
    denom = total if total is not None else sum(c.values())
    return [
        {"key": str(k), "n": n, "pct": pct(n, denom)}
        for k, n in items
    ]


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def analyze(rows: list[dict]) -> dict:
    ok = [r for r in rows if not has_error(r)]
    problems = [r for r in ok if is_problem(r)]

    vocal_prob = [r for r in problems if is_vocal_stem(r, "problem_stem")]
    lead = [r for r in vocal_prob if is_true(r.get("vocal_lead", ""))]
    backing = [r for r in vocal_prob if not is_true(r.get("vocal_lead", ""))]
    # vocal_lead=True but stem not vocal (inconsistent / rare)
    lead_non_vocal_stem = [
        r for r in problems if is_true(r.get("vocal_lead", "")) and not is_vocal_stem(r)
    ]

    vocal_fix = [r for r in ok if is_fix(r) and is_vocal_stem(r, "fix_stem")]
    vocal_prob_with_fix = [r for r in vocal_prob if is_fix(r)]

    by_split_all = Counter(str(r.get("split") or "").strip() or "(empty)" for r in problems)
    by_split_vocal = Counter(str(r.get("split") or "").strip() or "(empty)" for r in vocal_prob)

    def axis_dim_breakdown(subset: list[dict]) -> dict:
        axis = counter_rows(subset, "problem_axis")
        dim = counter_rows(subset, "problem_dimension")
        axis_dim = Counter(
            f"{str(r.get('problem_axis') or '').strip() or '(empty)'}/"
            f"{str(r.get('problem_dimension') or '').strip() or '(empty)'}"
            for r in subset
        )
        return {
            "n": len(subset),
            "axis": counter_to_list(axis, len(subset)),
            "dimension": counter_to_list(dim, len(subset)),
            "axis_dimension": counter_to_list(axis_dim, len(subset)),
            "confidence": counter_to_list(counter_rows(subset, "confidence"), len(subset)),
            "problem_speaker": counter_to_list(
                counter_rows(subset, "problem_speaker"), len(subset)
            ),
            "topic": counter_to_list(counter_rows(subset, "topic"), len(subset), top=20),
        }

    # Share of each axis that is vocal
    axis_total: Counter = Counter()
    axis_vocal: Counter = Counter()
    for r in problems:
        a = str(r.get("problem_axis") or "").strip() or "(empty)"
        axis_total[a] += 1
        if is_vocal_stem(r):
            axis_vocal[a] += 1
    axis_vocal_share = [
        {
            "axis": a,
            "vocal_n": axis_vocal[a],
            "all_n": axis_total[a],
            "vocal_pct_of_axis": pct(axis_vocal[a], axis_total[a]),
            "pct_of_all_vocal": pct(axis_vocal[a], len(vocal_prob)),
        }
        for a, _ in axis_total.most_common()
    ]

    # When problem is vocal, what gets fixed?
    fix_stem_when_vocal = counter_rows(vocal_prob_with_fix, "fix_stem")
    fix_action_when_vocal = counter_rows(vocal_prob_with_fix, "fix_action")

    # When fix targets vocal, what was the problem stem?
    prob_stem_when_fix_vocal = counter_rows(
        [r for r in ok if is_fix(r) and is_vocal_stem(r, "fix_stem") and is_problem(r)],
        "problem_stem",
    )

    # Turn-level: turns that mention vocal problem
    by_turn: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        key = (
            str(r.get("split") or ""),
            str(r.get("conversation_id") or ""),
            str(r.get("turn_id") or ""),
        )
        by_turn[key].append(r)

    turns_with_problem = 0
    turns_with_vocal_problem = 0
    turns_with_lead = 0
    for labels in by_turn.values():
        if any(is_problem(r) for r in labels):
            turns_with_problem += 1
        if any(is_problem(r) and is_vocal_stem(r) for r in labels):
            turns_with_vocal_problem += 1
        if any(is_problem(r) and is_true(r.get("vocal_lead", "")) for r in labels):
            turns_with_lead += 1

    # Examples for inspection
    def samples(subset: list[dict], k: int = 5) -> list[dict]:
        out = []
        for r in subset[:k]:
            out.append(
                {
                    "split": r.get("split", ""),
                    "conversation_id": r.get("conversation_id", ""),
                    "turn_id": r.get("turn_id", ""),
                    "problem_axis": r.get("problem_axis", ""),
                    "problem_dimension": r.get("problem_dimension", ""),
                    "vocal_lead": r.get("vocal_lead", ""),
                    "problem_text": (r.get("problem_text") or "")[:160],
                    "confidence": r.get("confidence", ""),
                }
            )
        return out

    return {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_problem_rows": len(problems),
        "n_vocal_problem_rows": len(vocal_prob),
        "vocal_pct_of_problems": pct(len(vocal_prob), len(problems)),
        "n_lead": len(lead),
        "n_backing_or_unmarked": len(backing),
        "lead_pct_of_vocal": pct(len(lead), len(vocal_prob)),
        "backing_pct_of_vocal": pct(len(backing), len(vocal_prob)),
        "n_lead_but_non_vocal_stem": len(lead_non_vocal_stem),
        "n_vocal_fix_rows": len(vocal_fix),
        "n_vocal_problem_with_fix": len(vocal_prob_with_fix),
        "vocal_problem_with_fix_pct": pct(len(vocal_prob_with_fix), len(vocal_prob)),
        "n_turns": len(by_turn),
        "turns_with_problem": turns_with_problem,
        "turns_with_vocal_problem": turns_with_vocal_problem,
        "turns_with_lead": turns_with_lead,
        "vocal_turn_pct_of_problem_turns": pct(
            turns_with_vocal_problem, turns_with_problem
        ),
        "by_split_all_problems": counter_to_list(by_split_all, len(problems)),
        "by_split_vocal": counter_to_list(by_split_vocal, len(vocal_prob)),
        "all_problems": axis_dim_breakdown(problems),
        "vocal_problems": axis_dim_breakdown(vocal_prob),
        "lead_vocal": axis_dim_breakdown(lead),
        "non_lead_vocal": axis_dim_breakdown(backing),
        "axis_vocal_share": axis_vocal_share,
        "fix_stem_when_vocal_problem": counter_to_list(
            fix_stem_when_vocal, len(vocal_prob_with_fix), top=20
        ),
        "fix_action_when_vocal_problem": counter_to_list(
            fix_action_when_vocal, len(vocal_prob_with_fix), top=20
        ),
        "problem_stem_when_fix_vocal": counter_to_list(
            prob_stem_when_fix_vocal, sum(prob_stem_when_fix_vocal.values()), top=20
        ),
        "samples_lead": samples(lead),
        "samples_non_lead_vocal": samples(backing),
    }


def write_markdown(path: Path, stats: dict, sources: list[str]) -> None:
    vp = stats["vocal_problems"]
    lead = stats["lead_vocal"]
    non_lead = stats["non_lead_vocal"]

    lines: list[str] = [
        "# Vocal-related label distributions",
        "",
        f"Sources: {', '.join(f'`{s}`' for s in sources)}",
        "",
        "## Coverage",
        "",
        md_table(
            ["metric", "n", "%"],
            [
                ["all label rows", str(stats["n_rows"]), ""],
                ["ok rows (no error)", str(stats["n_ok"]), ""],
                ["problem rows", str(stats["n_problem_rows"]), "100% of problems"],
                [
                    "vocal problem rows (`problem_stem=vocal`)",
                    str(stats["n_vocal_problem_rows"]),
                    f"{stats['vocal_pct_of_problems']}% of problems",
                ],
                [
                    "  └ lead (`vocal_lead=True`)",
                    str(stats["n_lead"]),
                    f"{stats['lead_pct_of_vocal']}% of vocal",
                ],
                [
                    "  └ non-lead vocal (`vocal_lead=False`)",
                    str(stats["n_backing_or_unmarked"]),
                    f"{stats['backing_pct_of_vocal']}% of vocal",
                ],
                [
                    "vocal_lead=True but stem≠vocal",
                    str(stats["n_lead_but_non_vocal_stem"]),
                    "(inconsistent)",
                ],
                [
                    "vocal as fix_stem",
                    str(stats["n_vocal_fix_rows"]),
                    "",
                ],
                [
                    "vocal problem ∧ has fix",
                    str(stats["n_vocal_problem_with_fix"]),
                    f"{stats['vocal_problem_with_fix_pct']}% of vocal problems",
                ],
            ],
        ),
        "",
        "### Turn-level",
        "",
        md_table(
            ["metric", "n", "%"],
            [
                ["turns", str(stats["n_turns"]), ""],
                ["turns with any problem", str(stats["turns_with_problem"]), ""],
                [
                    "turns with vocal problem",
                    str(stats["turns_with_vocal_problem"]),
                    f"{stats['vocal_turn_pct_of_problem_turns']}% of problem turns",
                ],
                ["turns with lead vocal problem", str(stats["turns_with_lead"]), ""],
            ],
        ),
        "",
        "## By split (vocal problem rows)",
        "",
    ]

    split_rows = []
    all_map = {x["key"]: x for x in stats["by_split_all_problems"]}
    for item in stats["by_split_vocal"]:
        all_n = all_map.get(item["key"], {}).get("n", 0)
        split_rows.append(
            [
                item["key"],
                str(item["n"]),
                f"{item['pct']}%",
                str(all_n),
                f"{pct(item['n'], all_n)}% of split problems" if all_n else "n/a",
            ]
        )
    lines.append(
        md_table(
            ["split", "vocal n", "% of vocal", "all problems", "vocal share"],
            split_rows or [["(none)", "0", "", "0", ""]],
        )
    )

    def section_dist(title: str, block: dict) -> None:
        lines.extend(["", f"## {title}", "", f"n = {block['n']}", ""])
        if block["n"] == 0:
            lines.append("(empty)")
            return
        lines.append("### Axis")
        lines.append("")
        lines.append(
            md_table(
                ["axis", "n", "%"],
                [[x["key"], str(x["n"]), f"{x['pct']}%"] for x in block["axis"]],
            )
        )
        lines.append("")
        lines.append("### Dimension")
        lines.append("")
        lines.append(
            md_table(
                ["dimension", "n", "%"],
                [[x["key"], str(x["n"]), f"{x['pct']}%"] for x in block["dimension"]],
            )
        )
        lines.append("")
        lines.append("### Axis × Dimension")
        lines.append("")
        lines.append(
            md_table(
                ["axis/dimension", "n", "%"],
                [[x["key"], str(x["n"]), f"{x['pct']}%"] for x in block["axis_dimension"]],
            )
        )
        lines.append("")
        lines.append("### Confidence")
        lines.append("")
        lines.append(
            md_table(
                ["confidence", "n", "%"],
                [[x["key"], str(x["n"]), f"{x['pct']}%"] for x in block["confidence"]],
            )
        )

    section_dist("Vocal problems (`problem_stem=vocal`)", vp)
    section_dist("Lead vocal only (`vocal_lead=True`)", lead)
    section_dist("Non-lead vocal (`problem_stem=vocal`, `vocal_lead=False`)", non_lead)

    lines.extend(
        [
            "",
            "## Vocal share within each axis (all problem rows)",
            "",
            md_table(
                ["axis", "vocal n", "all n", "vocal % of axis", "% of all vocal"],
                [
                    [
                        x["axis"],
                        str(x["vocal_n"]),
                        str(x["all_n"]),
                        f"{x['vocal_pct_of_axis']}%",
                        f"{x['pct_of_all_vocal']}%",
                    ]
                    for x in stats["axis_vocal_share"]
                ],
            ),
            "",
            "## When problem is vocal: fix_stem / fix_action",
            "",
            f"Rows with vocal problem and a fix: {stats['n_vocal_problem_with_fix']}",
            "",
            "### fix_stem",
            "",
            md_table(
                ["fix_stem", "n", "%"],
                [
                    [x["key"], str(x["n"]), f"{x['pct']}%"]
                    for x in stats["fix_stem_when_vocal_problem"]
                ]
                or [["(none)", "0", ""]],
            ),
            "",
            "### fix_action",
            "",
            md_table(
                ["fix_action", "n", "%"],
                [
                    [x["key"], str(x["n"]), f"{x['pct']}%"]
                    for x in stats["fix_action_when_vocal_problem"]
                ]
                or [["(none)", "0", ""]],
            ),
            "",
            "## When fix_stem is vocal: problem_stem",
            "",
            md_table(
                ["problem_stem", "n", "%"],
                [
                    [x["key"], str(x["n"]), f"{x['pct']}%"]
                    for x in stats["problem_stem_when_fix_vocal"]
                ]
                or [["(none)", "0", ""]],
            ),
            "",
            "## Sample rows",
            "",
            "### Lead vocal",
            "",
        ]
    )
    for s in stats["samples_lead"]:
        lines.append(
            f"- [{s['split']}] {s['conversation_id']} turn {s['turn_id']}: "
            f"`{s['problem_axis']}/{s['problem_dimension']}` "
            f"(conf={s['confidence']}) — {s['problem_text']!r}"
        )
    if not stats["samples_lead"]:
        lines.append("(none)")

    lines.extend(["", "### Non-lead vocal", ""])
    for s in stats["samples_non_lead_vocal"]:
        lines.append(
            f"- [{s['split']}] {s['conversation_id']} turn {s['turn_id']}: "
            f"`{s['problem_axis']}/{s['problem_dimension']}` "
            f"(conf={s['confidence']}) — {s['problem_text']!r}"
        )
    if not stats["samples_non_lead_vocal"]:
        lines.append("(none)")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def print_console(stats: dict) -> None:
    print("\n" + "=" * 72)
    print("VOCAL DISTRIBUTION SUMMARY")
    print("=" * 72)
    print(
        f"problem rows: {stats['n_problem_rows']}  |  "
        f"vocal: {stats['n_vocal_problem_rows']} ({stats['vocal_pct_of_problems']}%)  |  "
        f"lead: {stats['n_lead']}  |  non-lead vocal: {stats['n_backing_or_unmarked']}"
    )
    print(
        f"turns with vocal problem: {stats['turns_with_vocal_problem']} / "
        f"{stats['turns_with_problem']} problem turns "
        f"({stats['vocal_turn_pct_of_problem_turns']}%)"
    )
    print("\nAxis among vocal problems:")
    for x in stats["vocal_problems"]["axis"]:
        print(f"  {x['key']:<12} {x['n']:>4}  ({x['pct']}%)")
    print("\nDimension among vocal problems:")
    for x in stats["vocal_problems"]["dimension"]:
        print(f"  {x['key']:<22} {x['n']:>4}  ({x['pct']}%)")
    print("\nLead vs non-lead axis:")
    print("  [lead]")
    for x in stats["lead_vocal"]["axis"]:
        print(f"    {x['key']:<12} {x['n']:>4}  ({x['pct']}%)")
    print("  [non-lead vocal]")
    for x in stats["non_lead_vocal"]["axis"]:
        print(f"    {x['key']:<12} {x['n']:>4}  ({x['pct']}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze vocal-related label distributions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        help="one or more CSV paths (overrides --prefix/--splits)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="markdown path (default: <output-dir>/vocal_stats.md)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="JSON path (default: <output-dir>/vocal_stats.json)",
    )
    parser.add_argument(
        "--also-root",
        action="store_true",
        help="also copy markdown to repo root vocal_stats.md",
    )
    args = parser.parse_args()

    sources: list[Path] = []
    if args.csv:
        sources = list(args.csv)
    else:
        for split in args.splits:
            sources.append(args.output_dir / f"{args.prefix}_{split}.csv")

    all_rows: list[dict] = []
    used: list[str] = []
    for path in sources:
        rows = load_csv(path)
        print(f"Loading {path} ... ({len(rows)} rows)")
        if not rows:
            continue
        # infer split from filename if missing
        for r in rows:
            if not str(r.get("split") or "").strip():
                name = path.stem
                for sp in SPLITS:
                    if name.endswith(f"_{sp}") or f"_{sp}_" in name:
                        r["split"] = sp
                        break
        all_rows.extend(rows)
        used.append(str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path))

    if not all_rows:
        print(
            "No rows loaded. Provide labeled CSVs, e.g.\n"
            "  python scripts/analyze_vocal.py\n"
            "  python scripts/analyze_vocal.py --csv outputs/labeled_turns_gguf_all_problems.csv"
        )
        raise SystemExit(1)

    stats = analyze(all_rows)
    print_console(stats)

    md_path = args.md_out or (args.output_dir / "vocal_stats.md")
    json_path = args.json_out or (args.output_dir / "vocal_stats.json")
    write_markdown(md_path, stats, used)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    if args.also_root:
        root_md = PROJECT_ROOT / "vocal_stats.md"
        root_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {root_md}")


if __name__ == "__main__":
    main()
