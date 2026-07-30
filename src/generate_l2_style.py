# -*- coding: utf-8 -*-
"""
L2 style transfer — structured mix-problem label → Amateur/Expert dialogue.

Uses MixAssist labeled problem rows as:
  - structured input gold (axis / dim / subject stub)
  - same-dim few-shot style exemplars (excludes the current turn)

Usage:
  python src/generate_l2_style.py --limit 3
  python src/generate_l2_style.py
  python src/generate_l2_style.py --per-dim 5 --seed 42

Output:
  outputs/l2_gen_pilot.csv
  outputs/l2_gen_pilot_raw.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Reuse GGUF loader / CUDA prep from labeling package
from labeling.labeling_gguf import (  # noqa: E402
    DEFAULT_GGUF,
    load_llm,
)
from prompts.l2_generation_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    stem_to_subject,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL = PROJECT_ROOT / "outputs" / "labeled_turns_gguf_all_problems.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "l2_gen_pilot.csv"

# Generation: more variety than labeling
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 64
MAX_TOKENS = 768
N_CTX = 4096

OUTPUT_FIELDS = [
    "pilot_id",
    "source_conversation_id",
    "source_turn_id",
    "source_split",
    "gold_axis",
    "gold_dim",
    "gold_stem",
    "subject",
    "severity",
    "vocal_lead",
    "source_problem_text",
    "exemplar_ids",
    "input_json",
    "amateur_text",
    "expert_text",
    "generated_dialogue",
    "error",
]


def load_pool(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        dim = (r.get("problem_dimension") or "").strip().lower()
        text = (r.get("problem_text") or "").strip()
        conf = (r.get("confidence") or "").strip().lower()
        if not dim or dim == "none" or not text:
            continue
        if conf and conf != "high":
            continue
        kept.append(r)
    return kept


def turn_key(row: dict) -> str:
    return f"{row.get('conversation_id', '')}::{row.get('turn_id', '')}"


def sample_pilot(pool: list[dict], per_dim: int, seed: int | None) -> list[dict]:
    rng = random.Random(seed)
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_dim[(r.get("problem_dimension") or "").strip().lower()].append(r)

    pilot: list[dict] = []
    for dim in sorted(by_dim):
        rows = by_dim[dim][:]
        rng.shuffle(rows)
        # Prefer vocal_lead when available, but still fill quota
        vocal = [r for r in rows if str(r.get("vocal_lead", "")).lower() == "true"]
        other = [r for r in rows if str(r.get("vocal_lead", "")).lower() != "true"]
        ordered = vocal + other
        # Dedupe by turn within dim
        seen = set()
        picked = []
        for r in ordered:
            k = turn_key(r)
            if k in seen:
                continue
            seen.add(k)
            picked.append(r)
            if len(picked) >= per_dim:
                break
        pilot.extend(picked)

    rng.shuffle(pilot)
    return pilot


def pick_exemplars(
    pool: list[dict],
    dim: str,
    exclude_key: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    cands = [
        r
        for r in pool
        if (r.get("problem_dimension") or "").strip().lower() == dim
        and turn_key(r) != exclude_key
        and (r.get("problem_text") or "").strip()
    ]
    rng.shuffle(cands)
    return cands[:n]


def parse_dialogue(raw: str) -> tuple[str, str]:
    """Extract Amateur / Expert quoted or plain lines from model output."""
    text = (raw or "").strip()
    amateur = ""
    expert = ""

    # Prefer quoted forms
    m_a = re.search(
        r'Amateur\s*:\s*"([^"]*)"',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    m_e = re.search(
        r'Expert\s*:\s*"([^"]*)"',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_a:
        amateur = m_a.group(1).strip()
    if m_e:
        expert = m_e.group(1).strip()

    if not amateur or not expert:
        # Fallback: line-based
        for line in text.splitlines():
            s = line.strip()
            if not amateur and re.match(r"(?i)^Amateur\s*:", s):
                amateur = re.sub(r"(?i)^Amateur\s*:\s*", "", s).strip().strip('"')
            elif not expert and re.match(r"(?i)^Expert\s*:", s):
                expert = re.sub(r"(?i)^Expert\s*:\s*", "", s).strip().strip('"')

    return amateur, expert


def call_generate(llm, user_message: str) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def purge_error_rows(path: Path) -> int:
    """Drop rows with error so a re-run can retry them (keep successful rows)."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if not (r.get("error") or "").strip()]
    removed = len(rows) - len(kept)
    if removed:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in kept:
                w.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})
    return removed


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("pilot_id") or "").strip()
            err = (row.get("error") or "").strip()
            if pid and not err:
                done.add(pid)
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})


def append_raw(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 style-transfer generation (pilot)")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-dim", type=int, default=4, help="max pilot rows per dimension")
    parser.add_argument("--n-exemplars", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only first N pilot rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete existing output and regenerate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sample pilot and print prompts only (no model load)",
    )
    args = parser.parse_args()

    if not args.pool.is_file():
        sys.exit(f"找不到 pool: {args.pool}")

    pool = load_pool(args.pool)
    print(f"Pool (confidence=high): {len(pool)} rows from {args.pool.name}")

    pilot = sample_pilot(pool, per_dim=args.per_dim, seed=args.seed)
    if args.limit is not None:
        pilot = pilot[: args.limit]
    print(f"Pilot size: {len(pilot)} (per_dim={args.per_dim}, seed={args.seed})")

    by_dim: dict[str, int] = defaultdict(int)
    for r in pilot:
        by_dim[(r.get("problem_dimension") or "").strip().lower()] += 1
    for dim, n in sorted(by_dim.items()):
        print(f"  {dim:<22} {n}")

    out_csv: Path = args.output
    raw_jsonl = out_csv.with_name(out_csv.stem + "_raw.jsonl")

    if args.overwrite:
        for p in (out_csv, raw_jsonl):
            if p.exists():
                p.unlink()
                print(f"--overwrite: deleted {p.name}")
    else:
        removed = purge_error_rows(out_csv)
        if removed:
            print(f"Purged {removed} error row(s) for retry")

    done = load_done_ids(out_csv)
    if done:
        print(f"Resume: skipping {len(done)} finished pilot_id(s)")

    rng = random.Random(args.seed)
    llm = None
    if not args.dry_run:
        llm = load_llm(args.model_path, args.n_ctx, args.n_gpu_layers, args.verbose)

    failures = 0
    for i, row in enumerate(pilot, 1):
        src_key = turn_key(row)
        pilot_id = f"{src_key}::{(row.get('problem_dimension') or '').strip().lower()}"
        tag = f"[{i}/{len(pilot)}] {pilot_id}"

        if pilot_id in done:
            print(f"{tag} done, skip")
            continue

        dim = (row.get("problem_dimension") or "").strip().lower()
        axis = (row.get("problem_axis") or "").strip().lower()
        stem = (row.get("problem_stem") or "").strip().lower()
        subject = stem_to_subject(stem)
        severity = "medium"
        input_obj = {
            "axis": axis,
            "dim": dim,
            "subject": subject,
            "severity": severity,
        }

        exemplars = pick_exemplars(pool, dim, src_key, args.n_exemplars, rng)
        user_prompt = build_user_prompt(input_obj, exemplars)
        exemplar_ids = ";".join(turn_key(e) for e in exemplars)

        base = {
            "pilot_id": pilot_id,
            "source_conversation_id": row.get("conversation_id", ""),
            "source_turn_id": row.get("turn_id", ""),
            "source_split": row.get("split", ""),
            "gold_axis": axis,
            "gold_dim": dim,
            "gold_stem": stem,
            "subject": subject,
            "severity": severity,
            "vocal_lead": row.get("vocal_lead", ""),
            "source_problem_text": row.get("problem_text", ""),
            "exemplar_ids": exemplar_ids,
            "input_json": json.dumps(input_obj, ensure_ascii=False),
            "amateur_text": "",
            "expert_text": "",
            "generated_dialogue": "",
            "error": "",
        }

        if args.dry_run:
            print(f"\n===== {tag} =====")
            print(user_prompt[:800], "..." if len(user_prompt) > 800 else "")
            continue

        raw_reply = ""
        try:
            assert llm is not None
            t0 = time.perf_counter()
            raw_reply = call_generate(llm, user_prompt)
            elapsed = time.perf_counter() - t0
            amateur, expert = parse_dialogue(raw_reply)
            if not amateur or not expert:
                raise ValueError("failed to parse Amateur/Expert from model output")
            dialogue = f'Amateur: "{amateur}"\nExpert: "{expert}"'
            base["amateur_text"] = amateur
            base["expert_text"] = expert
            base["generated_dialogue"] = dialogue
            append_row(out_csv, base)
            append_raw(
                raw_jsonl,
                {
                    "pilot_id": pilot_id,
                    "input_json": input_obj,
                    "exemplar_ids": exemplar_ids,
                    "user_prompt": user_prompt,
                    "raw_reply": raw_reply,
                    "elapsed_s": round(elapsed, 2),
                },
            )
            print(f"{tag} ok ({elapsed:.1f}s) A={amateur[:60]!r}...")
        except Exception as e:
            failures += 1
            base["error"] = str(e)
            base["generated_dialogue"] = raw_reply
            append_row(out_csv, base)
            append_raw(
                raw_jsonl,
                {
                    "pilot_id": pilot_id,
                    "error": str(e),
                    "user_prompt": user_prompt,
                    "raw_reply": raw_reply,
                },
            )
            print(f"{tag} ERROR: {e}")

    if args.dry_run:
        print("\nDry-run only — no model calls.")
        return

    print(f"\nDone → {out_csv}")
    print(f"Raw  → {raw_jsonl}")
    if failures:
        print(f"Failures: {failures} (re-run same command to retry error rows)")


if __name__ == "__main__":
    main()
