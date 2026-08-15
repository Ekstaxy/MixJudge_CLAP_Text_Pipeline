# -*- coding: utf-8 -*-
"""
分析 labeling_gguf 輸出 CSV (train / validation / test)。

預設讀:
  outputs/labeled_turns_gguf_train.csv
  outputs/labeled_turns_gguf_validation.csv
  outputs/labeled_turns_gguf_test.csv

用法:
  python scripts/analyze_labels.py
  python scripts/analyze_labels.py --splits train validation
  python scripts/analyze_labels.py --output-dir outputs --prefix labeled_turns_gguf
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


def pct(n: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * n / total:.1f}%"


def is_true(v: str) -> bool:
    return str(v or "").strip().lower() == "true"


def has_error(row: dict) -> bool:
    return bool(str(row.get("error") or "").strip())


def is_problem(row: dict) -> bool:
    """有問題狀態: has_problem 且 dimension 不是 none。"""
    dim = str(row.get("problem_dimension") or "").strip().lower()
    return is_true(row.get("has_problem", "")) and dim not in ("", "none")


def is_fix(row: dict) -> bool:
    """有 fix 狀態: has_fix 且有 fix_text 或 fix_action。"""
    if not is_true(row.get("has_fix", "")):
        return False
    text = str(row.get("fix_text") or "").strip()
    action = str(row.get("fix_action") or "").strip()
    return bool(text or action)


def load_split(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def print_counter(title: str, counter: Counter, total: int | None = None, top: int | None = None) -> None:
    print(f"  {title}")
    items = counter.most_common(top)
    if not items:
        print("    (empty)")
        return
    width = max(len(str(k)) for k, _ in items)
    denom = total if total is not None else sum(counter.values())
    for key, n in items:
        label = str(key) if str(key) else "(empty)"
        print(f"    {label:<{width}}  {n:>5}  ({pct(n, denom)})")


def turn_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("conversation_id") or ""),
        str(row.get("turn_id") or ""),
    )


def analyze_rows(name: str, rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"SPLIT: {name}")
    print("=" * 72)

    if not rows:
        print("  (file missing or empty)")
        return

    n_labels = len(rows)
    err_rows = [r for r in rows if has_error(r)]
    ok_rows = [r for r in rows if not has_error(r)]

    # --- turn aggregation ---
    by_turn: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok_rows:
        by_turn[turn_key(r)].append(r)
    n_turns = len(by_turn)

    turn_has_problem = 0
    turn_has_fix = 0
    turn_both = 0
    turn_problem_only = 0
    turn_fix_only = 0
    turn_empty = 0
    for labels in by_turn.values():
        p = any(is_problem(r) for r in labels)
        f = any(is_fix(r) for r in labels)
        if p and f:
            turn_both += 1
        elif p:
            turn_problem_only += 1
        elif f:
            turn_fix_only += 1
        else:
            turn_empty += 1
        if p:
            turn_has_problem += 1
        if f:
            turn_has_fix += 1

    print("\n[0] Coverage")
    print(f"  label rows          : {n_labels}")
    print(f"  ok label rows       : {len(ok_rows)}")
    print(f"  error label rows    : {len(err_rows)}  ({pct(len(err_rows), n_labels)})")
    print(f"  unique turns (ok)   : {n_turns}")
    if err_rows:
        print("  error messages:")
        for msg, n in Counter(str(r.get("error") or "").strip() for r in err_rows).most_common():
            print(f"    {n:>3}x  {msg[:100]}")

    # --- 1. problem state ---
    problem_rows = [r for r in ok_rows if is_problem(r)]
    none_problem_rows = [r for r in ok_rows if not is_problem(r)]
    print("\n[1] Problem state (label-row)")
    print(f"  has problem (non-none) : {len(problem_rows):>5} / {len(ok_rows)}  ({pct(len(problem_rows), len(ok_rows))})")
    print(f"  no problem / none      : {len(none_problem_rows):>5} / {len(ok_rows)}  ({pct(len(none_problem_rows), len(ok_rows))})")
    print("\n[1b] Problem state (turn-level, any label in turn)")
    print(f"  turns with problem     : {turn_has_problem:>5} / {n_turns}  ({pct(turn_has_problem, n_turns)})")
    print(f"  turns without problem  : {n_turns - turn_has_problem:>5} / {n_turns}  ({pct(n_turns - turn_has_problem, n_turns)})")

    dim_counter = Counter(
        str(r.get("problem_dimension") or "").strip() or "(empty)" for r in problem_rows
    )
    print_counter("problem_dimension among problem rows", dim_counter, total=len(problem_rows))

    # --- 2. fix state ---
    fix_rows = [r for r in ok_rows if is_fix(r)]
    print("\n[2] Fix state (label-row)")
    print(f"  has fix                : {len(fix_rows):>5} / {len(ok_rows)}  ({pct(len(fix_rows), len(ok_rows))})")
    print(f"  no fix                 : {len(ok_rows) - len(fix_rows):>5} / {len(ok_rows)}  ({pct(len(ok_rows) - len(fix_rows), len(ok_rows))})")
    print("\n[2b] Fix state (turn-level)")
    print(f"  turns with fix         : {turn_has_fix:>5} / {n_turns}  ({pct(turn_has_fix, n_turns)})")
    print(f"  turns without fix      : {n_turns - turn_has_fix:>5} / {n_turns}  ({pct(n_turns - turn_has_fix, n_turns)})")

    action_counter = Counter(
        str(r.get("fix_action") or "").strip() or "(empty)" for r in fix_rows
    )
    print_counter("fix_action among fix rows", action_counter, total=len(fix_rows))

    fix_stem_counter = Counter(
        str(r.get("fix_stem") or "").strip() or "(empty)" for r in fix_rows
    )
    print_counter("fix_stem among fix rows", fix_stem_counter, total=len(fix_rows), top=15)

    # --- 3. both ---
    both_rows = [r for r in ok_rows if is_problem(r) and is_fix(r)]
    print("\n[3] Problem AND fix")
    print(f"  label rows with both   : {len(both_rows):>5} / {len(ok_rows)}  ({pct(len(both_rows), len(ok_rows))})")
    print(f"  turns with both        : {turn_both:>5} / {n_turns}  ({pct(turn_both, n_turns)})")
    print(f"  turns problem-only     : {turn_problem_only:>5} / {n_turns}  ({pct(turn_problem_only, n_turns)})")
    print(f"  turns fix-only         : {turn_fix_only:>5} / {n_turns}  ({pct(turn_fix_only, n_turns)})")
    print(f"  turns empty (neither)  : {turn_empty:>5} / {n_turns}  ({pct(turn_empty, n_turns)})")

    # --- 4. axis / dimension ---
    axis_counter = Counter(
        str(r.get("problem_axis") or "").strip() or "(empty)" for r in problem_rows
    )
    print("\n[4] Problem axis distribution (among problem rows)")
    print_counter("problem_axis", axis_counter, total=len(problem_rows))

    axis_dim = Counter(
        f"{str(r.get('problem_axis') or '').strip() or '(empty)'}/"
        f"{str(r.get('problem_dimension') or '').strip() or '(empty)'}"
        for r in problem_rows
    )
    print_counter("problem_axis/dimension", axis_dim, total=len(problem_rows))

    # --- 5. errors (already in [0], keep explicit section) ---
    print("\n[5] Errors")
    print(f"  error rows             : {len(err_rows)}")
    err_turns = {(r.get("conversation_id"), r.get("turn_id")) for r in err_rows}
    print(f"  error turns            : {len(err_turns)}")

    # --- 6. extras ---
    print("\n[6] Extra breakdowns")

    stem_counter = Counter(
        str(r.get("problem_stem") or "").strip() or "(empty)" for r in problem_rows
    )
    print_counter("problem_stem among problem rows", stem_counter, total=len(problem_rows), top=15)

    speaker_p = Counter(
        str(r.get("problem_speaker") or "").strip() or "(empty)" for r in problem_rows
    )
    print_counter("problem_speaker among problem rows", speaker_p, total=len(problem_rows))

    speaker_f = Counter(
        str(r.get("fix_speaker") or "").strip() or "(empty)" for r in fix_rows
    )
    print_counter("fix_speaker among fix rows", speaker_f, total=len(fix_rows))

    vocal = Counter(
        "True" if is_true(r.get("vocal_lead", "")) else "False" for r in problem_rows
    )
    print_counter("vocal_lead among problem rows", vocal, total=len(problem_rows))

    conf = Counter(
        str(r.get("confidence") or "").strip() or "(empty)" for r in problem_rows
    )
    print_counter("confidence among problem rows", conf, total=len(problem_rows))

    # has_content among turns
    hc = Counter()
    for labels in by_turn.values():
        hc[str(labels[0].get("has_content") or "").strip() or "(empty)"] += 1
    print_counter("has_content among turns", hc, total=n_turns)

    # multi-label turns
    multi = sum(1 for labels in by_turn.values() if len(labels) > 1)
    print(f"  multi-label turns      : {multi:>5} / {n_turns}  ({pct(multi, n_turns)})")
    labels_per_turn = Counter(len(labels) for labels in by_turn.values())
    print_counter("labels per turn", labels_per_turn, total=n_turns)

    # topic among problem turns
    topic = Counter()
    for labels in by_turn.values():
        if any(is_problem(r) for r in labels):
            topic[str(labels[0].get("topic") or "").strip() or "(empty)"] += 1
    print_counter("topic among problem turns", topic, total=turn_has_problem or None)

    # has_problem=True but dimension none (inconsistent)
    inconsistent = [
        r
        for r in ok_rows
        if is_true(r.get("has_problem", ""))
        and str(r.get("problem_dimension") or "").strip().lower() in ("", "none")
    ]
    print(f"  has_problem=True but dim none/empty : {len(inconsistent)}")

    # has_fix=True but empty text+action
    weak_fix = [
        r
        for r in ok_rows
        if is_true(r.get("has_fix", ""))
        and not str(r.get("fix_text") or "").strip()
        and not str(r.get("fix_action") or "").strip()
    ]
    print(f"  has_fix=True but empty text/action : {len(weak_fix)}")


AXIS_ORDER = [
    "level",
    "body",
    "brightness",
    "space",
    "dynamic",
    "masking",
    "clean",
]
DIM_BY_AXIS = {
    "level": ["too_quiet", "too_loud"],
    "body": ["muddy", "thin"],
    "brightness": ["harsh", "dull"],
    "space": ["too_wet", "too_dry"],
    "dynamic": ["over_compressed", "under_compressed"],
    "masking": ["masking"],
    "clean": ["clean"],
}


def empty_matrix() -> dict[str, dict[str, int]]:
    return {a: {d: 0 for d in dims} for a, dims in DIM_BY_AXIS.items()}


def collect_split_stats(name: str, rows: list[dict]) -> dict:
    err_rows = [r for r in rows if has_error(r)]
    ok_rows = [r for r in rows if not has_error(r)]
    problem_rows = [r for r in ok_rows if is_problem(r)]
    fix_rows = [r for r in ok_rows if is_fix(r)]

    by_turn: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok_rows:
        by_turn[turn_key(r)].append(r)
    n_turns = len(by_turn)

    turn_problem = turn_fix = turn_both = turn_empty = 0
    for labels in by_turn.values():
        p = any(is_problem(r) for r in labels)
        f = any(is_fix(r) for r in labels)
        if p and f:
            turn_both += 1
        elif p:
            turn_problem += 1  # will recount below properly
        elif f:
            turn_fix += 1
        else:
            turn_empty += 1

    # recount cleanly
    turn_has_problem = sum(1 for ls in by_turn.values() if any(is_problem(r) for r in ls))
    turn_has_fix = sum(1 for ls in by_turn.values() if any(is_fix(r) for r in ls))
    turn_both = sum(
        1
        for ls in by_turn.values()
        if any(is_problem(r) for r in ls) and any(is_fix(r) for r in ls)
    )
    turn_problem_only = turn_has_problem - turn_both
    turn_fix_only = turn_has_fix - turn_both
    turn_empty = n_turns - turn_has_problem - turn_fix_only

    axis = Counter(str(r.get("problem_axis") or "").strip() or "(empty)" for r in problem_rows)
    dim = Counter(
        str(r.get("problem_dimension") or "").strip() or "(empty)" for r in problem_rows
    )
    matrix = empty_matrix()
    for r in problem_rows:
        a = str(r.get("problem_axis") or "").strip()
        d = str(r.get("problem_dimension") or "").strip()
        if a in matrix and d in matrix[a]:
            matrix[a][d] += 1

    vocal_lead_true = sum(1 for r in problem_rows if is_true(r.get("vocal_lead", "")))
    n_prob = len(problem_rows)

    return {
        "split": name,
        "label_rows": len(ok_rows),
        "error_rows": len(err_rows),
        "problem_rows": n_prob,
        "problem_row_pct": round(100 * n_prob / len(ok_rows), 1) if ok_rows else 0.0,
        "fix_rows": len(fix_rows),
        "turns": n_turns,
        "problem_turns": turn_has_problem,
        "problem_turn_pct": round(100 * turn_has_problem / n_turns, 1) if n_turns else 0.0,
        "fix_turns": turn_has_fix,
        "both_turns": turn_both,
        "problem_only_turns": turn_problem_only,
        "fix_only_turns": turn_fix_only,
        "empty_turns": turn_empty,
        "vocal_lead_true": vocal_lead_true,
        "vocal_lead_pct": round(100 * vocal_lead_true / n_prob, 1) if n_prob else 0.0,
        "axis": {a: axis.get(a, 0) for a in AXIS_ORDER},
        "dimension": dict(dim.most_common()),
        "matrix": matrix,
    }


def print_dashboard(stats_list: list[dict], overall: dict) -> None:
    """Compact view focused on counts + axis/dimension distribution."""
    print("\n" + "#" * 72)
    print("# DASHBOARD — counts & axis/dimension distribution")
    print("#" * 72)

    print("\n## Coverage (problem-centric)")
    print(
        f"{'split':<12} {'turns':>6} {'prob_t':>7} {'%':>6} "
        f"{'prob_rows':>9} {'%rows':>6} {'vocal%':>7} {'both_t':>6} {'err':>4}"
    )
    for s in stats_list:
        print(
            f"{s['split']:<12} {s['turns']:>6} {s['problem_turns']:>7} "
            f"{s['problem_turn_pct']:>5.1f}% {s['problem_rows']:>9} "
            f"{s['problem_row_pct']:>5.1f}% {s['vocal_lead_pct']:>6.1f}% "
            f"{s['both_turns']:>6} {s['error_rows']:>4}"
        )
    print(
        f"{'ALL':<12} {overall['turns']:>6} {overall['problem_turns']:>7} "
        f"{overall['problem_turn_pct']:>5.1f}% {overall['problem_rows']:>9} "
        f"{overall['problem_row_pct']:>5.1f}% {overall['vocal_lead_pct']:>6.1f}% "
        f"{overall['both_turns']:>6} {overall['error_rows']:>4}"
    )
    print(
        f"  vocal_lead=True among problem rows: "
        f"{overall['vocal_lead_true']}/{overall['problem_rows']} "
        f"({overall['vocal_lead_pct']}%)"
    )

    print("\n## Axis distribution (problem rows, all splits)")
    total_p = overall["problem_rows"] or 1
    print(f"{'axis':<12} {'n':>6} {'%':>7}  bar")
    for a in AXIS_ORDER:
        n = overall["axis"].get(a, 0)
        p = 100.0 * n / total_p
        bar = "#" * int(round(p / 2))
        print(f"{a:<12} {n:>6} {p:>6.1f}%  {bar}")

    print("\n## Axis × Dimension matrix (all splits)")
    # header: all dims in schema order
    dims_flat = [d for a in AXIS_ORDER for d in DIM_BY_AXIS[a]]
    # print per-axis blocks (clearer than huge wide table)
    for a in AXIS_ORDER:
        cells = overall["matrix"][a]
        parts = [f"{d}={cells.get(d, 0)}" for d in DIM_BY_AXIS[a]]
        axis_n = sum(cells.values())
        print(f"  {a:<12} (n={axis_n:>3})  " + "  |  ".join(parts))

    print("\n## Dimension ranking (all splits)")
    print(f"{'dimension':<22} {'n':>6} {'%':>7}")
    for d, n in overall["dimension"].items():
        print(f"{d:<22} {n:>6} {100.0 * n / total_p:>6.1f}%")

    sparse = [a for a in AXIS_ORDER if overall["axis"].get(a, 0) < 15]
    if sparse:
        print("\n## Sparse axes (< 15 problem rows) — watch for L2 retrieval")
        for a in sparse:
            print(f"  - {a}: {overall['axis'].get(a, 0)}")


def write_stats_json(path: Path, stats_list: list[dict], overall: dict) -> None:
    payload = {
        "source": "labeled_turns_gguf_{train,validation,test}.csv",
        "overall": overall,
        "by_split": {s["split"]: s for s in stats_list},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


def write_stats_markdown(path: Path, stats_list: list[dict], overall: dict) -> None:
    lines = [
        "# Labeling stats (problem-centric)",
        "",
        "## Coverage",
        "",
        "| split | turns | problem turns | % | problem rows | % rows | vocal_lead=True | vocal % | both turns | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats_list + [overall]:
        name = s.get("split", "ALL")
        lines.append(
            f"| {name} | {s['turns']} | {s['problem_turns']} | {s['problem_turn_pct']}% "
            f"| {s['problem_rows']} | {s['problem_row_pct']}% "
            f"| {s['vocal_lead_true']} | {s['vocal_lead_pct']}% "
            f"| {s['both_turns']} | {s['error_rows']} |"
        )
    lines += [
        "",
        f"**vocal_lead=True** among all problem rows: "
        f"{overall['vocal_lead_true']}/{overall['problem_rows']} "
        f"({overall['vocal_lead_pct']}%). "
        "Lead vocal only; backing vocals are counted as False.",
        "",
        "## Axis distribution (all)",
        "",
        "| axis | n | % |",
        "|---|---:|---:|",
    ]
    total_p = overall["problem_rows"] or 1
    for a in AXIS_ORDER:
        n = overall["axis"].get(a, 0)
        lines.append(f"| {a} | {n} | {100.0 * n / total_p:.1f}% |")
    lines += ["", "## Axis × Dimension", ""]
    for a in AXIS_ORDER:
        cells = overall["matrix"][a]
        parts = ", ".join(f"`{d}`={cells.get(d, 0)}" for d in DIM_BY_AXIS[a])
        lines.append(f"- **{a}** (n={sum(cells.values())}): {parts}")
    lines += ["", "## Dimension ranking", "", "| dimension | n | % |", "|---|---:|---:|"]
    for d, n in overall["dimension"].items():
        lines.append(f"| {d} | {n} | {100.0 * n / total_p:.1f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def merge_overall(stats_list: list[dict]) -> dict:
    overall = {
        "split": "ALL",
        "label_rows": sum(s["label_rows"] for s in stats_list),
        "error_rows": sum(s["error_rows"] for s in stats_list),
        "problem_rows": sum(s["problem_rows"] for s in stats_list),
        "fix_rows": sum(s["fix_rows"] for s in stats_list),
        "turns": sum(s["turns"] for s in stats_list),
        "problem_turns": sum(s["problem_turns"] for s in stats_list),
        "fix_turns": sum(s["fix_turns"] for s in stats_list),
        "both_turns": sum(s["both_turns"] for s in stats_list),
        "problem_only_turns": sum(s["problem_only_turns"] for s in stats_list),
        "fix_only_turns": sum(s["fix_only_turns"] for s in stats_list),
        "empty_turns": sum(s["empty_turns"] for s in stats_list),
        "vocal_lead_true": sum(s["vocal_lead_true"] for s in stats_list),
        "axis": {a: 0 for a in AXIS_ORDER},
        "dimension": Counter(),
        "matrix": empty_matrix(),
    }
    for s in stats_list:
        for a in AXIS_ORDER:
            overall["axis"][a] += s["axis"].get(a, 0)
            for d, n in s["matrix"][a].items():
                overall["matrix"][a][d] += n
        overall["dimension"].update(s["dimension"])
    overall["dimension"] = dict(Counter(overall["dimension"]).most_common())
    pr = overall["problem_rows"]
    tr = overall["turns"]
    lr = overall["label_rows"]
    overall["problem_row_pct"] = round(100 * pr / lr, 1) if lr else 0.0
    overall["problem_turn_pct"] = round(100 * overall["problem_turns"] / tr, 1) if tr else 0.0
    overall["vocal_lead_pct"] = (
        round(100 * overall["vocal_lead_true"] / pr, 1) if pr else 0.0
    )
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MixAssist labeling CSV outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory with labeled CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"filename prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
        help="splits to analyze",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also print the verbose per-split breakdown (legacy)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write machine-readable stats JSON (default: <output-dir>/label_stats.json)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="write markdown summary (default: <output-dir>/label_stats.md)",
    )
    parser.add_argument("--no-export", action="store_true", help="skip JSON/MD export")
    args = parser.parse_args()

    stats_list: list[dict] = []
    all_rows: list[dict] = []
    for split in args.splits:
        path = args.output_dir / f"{args.prefix}_{split}.csv"
        print(f"Loading {path} ...")
        rows = load_split(path)
        for r in rows:
            if not r.get("split"):
                r["split"] = split
        all_rows.extend(rows)
        stats_list.append(collect_split_stats(split, rows))
        if args.full:
            analyze_rows(split, rows)

    overall = merge_overall(stats_list)
    print_dashboard(stats_list, overall)

    if args.full and len(args.splits) > 1:
        analyze_overall(all_rows)

    if not args.no_export:
        json_path = args.json_out or (args.output_dir / "label_stats.json")
        md_path = args.md_out or (args.output_dir / "label_stats.md")
        write_stats_json(json_path, stats_list, overall)
        write_stats_markdown(md_path, stats_list, overall)


if __name__ == "__main__":
    main()
