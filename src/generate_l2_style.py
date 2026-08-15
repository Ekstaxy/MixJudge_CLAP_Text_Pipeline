# -*- coding: utf-8 -*-
"""
L2 style transfer — L1 structured records → Amateur/Expert dialogue (+ problem_state_text).

Pipeline (current):
  1. Read L1 JSONL (mock or future export): axis / dimension / subject / source / texts.L1
  2. Style pool = outputs/labeled_turns_gguf_all_problems.csv (FULL pool; no confidence filter)
  3. For each L1 row, randomly sample same-dimension MixAssist turns as style exemplars
     (prefer single-dim turns; skip turns that also carry competing dims when possible)
  4. Generate under two orthogonal axes (fixed low temperature for prompt adherence):
       mode:             retarget | strict | free   (rewrite policy)
       exemplar_content: problem_text | raw         (what MixAssist text is fed)
  5. Model returns JSON: amateur, expert, problem_state_text

Caveats:
  - Dimension/axis follow MixJudge note + all_problems (14 signed dims). No phase.
  - Do NOT use labeled_turns_gguf_train.csv as the style pool (includes none/fix noise).
  - Exemplars often say guitar/drums; prompts must retarget to L1 source/subject.
  - muddy L1 should be bed-subject (carried-by), not vocal-only anchors.
  - Temperature is fixed low; looseness comes from --modes, not from temp.

Usage (from repo root):
  for content in problem_text raw; do
    python src/generate_l2_style.py --modes retarget strict free \\
      --exemplar-content "$content" --overwrite
  done
  python src/generate_l2_style.py --modes retarget --exemplar-content raw --dry-run --limit 1

Output:
  outputs/l2_from_l1_{mode}_{exemplar_content}.csv (+ _raw.jsonl)
  e.g. l2_from_l1_retarget_raw.csv, l2_from_l1_free_problem_text.csv
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

from labeling.labeling_gguf import (  # noqa: E402
    DEFAULT_GGUF,
    load_llm,
    parse_tensor_split,
)
from prompts.l2_generation_prompt import (  # noqa: E402
    VALID_EXEMPLAR_CONTENTS,
    VALID_MODES,
    build_user_prompt,
    system_prompt_for_mode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_L1 = PROJECT_ROOT / "data" / "l1_mock_records.jsonl"
DEFAULT_POOL = PROJECT_ROOT / "outputs" / "labeled_turns_gguf_all_problems.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs"

TOP_P = 0.95
TOP_K = 64
MAX_TOKENS = 768
N_CTX = 4096

# Single fixed temperature so all modes follow prompt rules (not a third experiment axis).
FIXED_TEMPERATURE = 0.1

OUTPUT_FIELDS = [
    "l1_segment_id",
    "l1_text",
    "gold_axis",
    "gold_dim",
    "gold_subject",
    "source_instrument",
    "severity",
    "style_mode",
    "exemplar_content",
    "temperature",
    "style_turn_ids",
    "style_problem_texts",
    # MixAssist reference turns (full dialogue; multi-exemplar joined by " ||| ")
    "style_splits",
    "style_audio_files",
    "style_user_raw_contents",
    "style_assistant_raw_contents",
    "style_exemplars_json",
    "input_json",
    "amateur_text",
    "expert_text",
    "generated_dialogue",
    "problem_state_text",
    "error",
]

# Separator when packing multiple exemplars into one CSV cell
EXEMPLAR_SEP = " ||| "


def variant_tag(mode: str, exemplar_content: str) -> str:
    return f"{mode}_{exemplar_content}"


def output_csv_path(output_dir: Path, mode: str, exemplar_content: str) -> Path:
    return output_dir / f"l2_from_l1_{variant_tag(mode, exemplar_content)}.csv"


def pack_exemplar_fields(exemplars: list[dict]) -> dict:
    """Serialize MixAssist style exemplars for CSV + JSON audit."""
    if not exemplars:
        return {
            "style_turn_ids": "",
            "style_problem_texts": "",
            "style_splits": "",
            "style_audio_files": "",
            "style_user_raw_contents": "",
            "style_assistant_raw_contents": "",
            "style_exemplars_json": "[]",
        }
    payloads = []
    for e in exemplars:
        payloads.append(
            {
                "conversation_id": e.get("conversation_id", ""),
                "turn_id": e.get("turn_id", ""),
                "split": e.get("split", ""),
                "audio_file": e.get("audio_file", ""),
                "problem_stem": e.get("problem_stem", ""),
                "problem_dimension": e.get("problem_dimension", ""),
                "problem_text": e.get("problem_text", ""),
                "user_raw_content": e.get("user_raw_content", ""),
                "assistant_raw_content": e.get("assistant_raw_content", ""),
            }
        )
    return {
        "style_turn_ids": ";".join(turn_key(e) for e in exemplars),
        "style_problem_texts": EXEMPLAR_SEP.join(
            (e.get("problem_text") or "").strip() for e in exemplars
        ),
        "style_splits": EXEMPLAR_SEP.join(
            str(e.get("split") or "") for e in exemplars
        ),
        "style_audio_files": EXEMPLAR_SEP.join(
            str(e.get("audio_file") or "") for e in exemplars
        ),
        "style_user_raw_contents": EXEMPLAR_SEP.join(
            str(e.get("user_raw_content") or "") for e in exemplars
        ),
        "style_assistant_raw_contents": EXEMPLAR_SEP.join(
            str(e.get("assistant_raw_content") or "") for e in exemplars
        ),
        "style_exemplars_json": json.dumps(payloads, ensure_ascii=False),
    }


def load_pool(path: Path) -> list[dict]:
    """Full problem pool — no confidence filter (all_problems is already curated)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        dim = (r.get("problem_dimension") or "").strip().lower()
        text = (r.get("problem_text") or "").strip()
        if not dim or dim == "none" or not text:
            continue
        if dim == "phase":
            continue
        kept.append(r)
    return kept


def load_l1_records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dim = (obj.get("dimension") or "").strip().lower()
            if not dim or dim == "phase":
                continue
            rows.append(obj)
    return rows


def turn_key(row: dict) -> str:
    return f"{row.get('conversation_id', '')}::{row.get('turn_id', '')}"


def index_pool_by_dim(pool: list[dict]) -> dict[str, list[dict]]:
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        dim = (r.get("problem_dimension") or "").strip().lower()
        by_dim[dim].append(r)
    return by_dim


def index_dims_by_turn(pool: list[dict]) -> dict[str, set[str]]:
    """All problem dims labeled on each MixAssist turn (for competing-dim gate)."""
    out: dict[str, set[str]] = defaultdict(set)
    for r in pool:
        dim = (r.get("problem_dimension") or "").strip().lower()
        if not dim or dim == "none":
            continue
        out[turn_key(r)].add(dim)
    return out


def pick_exemplars(
    candidates: list[dict],
    n: int,
    rng: random.Random,
    *,
    target_source: str | None = None,
    target_dim: str | None = None,
    top_k: int = 12,
    dims_by_turn: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Random same-dim exemplars. Prefer turns with only target_dim (no competing labels).

    Gate A: if the MixAssist turn also carries other problem_dimensions, skip it when
    any single-dim turn exists for target_dim. Falls back to multi-dim turns only when
    the pool has no clean turn for that dim. Still random among the allowed set (--seed).
    """
    del target_source, top_k  # kept for call-site compat; unused
    if not candidates:
        return []
    cands = candidates[:]
    td = (target_dim or "").strip().lower()
    if td and dims_by_turn is not None:
        clean = [
            r for r in cands if dims_by_turn.get(turn_key(r), set()) == {td}
        ]
        if clean:
            cands = clean
    rng.shuffle(cands)
    seen: set[str] = set()
    picked: list[dict] = []
    for r in cands:
        k = turn_key(r)
        if k in seen:
            continue
        seen.add(k)
        picked.append(r)
        if len(picked) >= n:
            break
    return picked


def l1_to_input_obj(rec: dict) -> dict:
    texts = rec.get("texts") or {}
    l1 = ""
    if isinstance(texts, dict):
        l1 = (texts.get("L1") or "").strip()
    return {
        "axis": (rec.get("axis") or "").strip().lower(),
        "dim": (rec.get("dimension") or "").strip().lower(),
        "subject": (rec.get("subject") or "").strip().lower(),
        "source": (rec.get("source") or "").strip().lower(),
        "severity": (rec.get("severity") or "medium").strip().lower(),
        "l1": l1,
    }


def extract_json_obj(raw: str) -> dict:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise json.JSONDecodeError("Expecting JSON object", text, 0)


def parse_generation(raw: str) -> tuple[str, str, str]:
    """Return amateur, expert, problem_state_text."""
    obj = extract_json_obj(raw)
    amateur = str(obj.get("amateur") or "").strip()
    expert = str(obj.get("expert") or "").strip()
    pst = str(obj.get("problem_state_text") or "").strip()
    if not amateur or not expert:
        raise ValueError("JSON missing amateur/expert")
    if not pst:
        pst = amateur
    return amateur, expert, pst


def call_generate(llm, system_prompt: str, user_message: str, temperature: float) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def purge_error_rows(path: Path) -> int:
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
            pid = (row.get("l1_segment_id") or "").strip()
            err = (row.get("error") or "").strip()
            mode = (row.get("style_mode") or "").strip()
            content = (row.get("exemplar_content") or "").strip()
            if pid and mode and not err:
                done.add(f"{pid}::{mode}::{content}" if content else f"{pid}::{mode}")
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


def run_mode(
    *,
    mode: str,
    exemplar_content: str,
    temperature: float,
    l1_rows: list[dict],
    by_dim: dict[str, list[dict]],
    dims_by_turn: dict[str, set[str]],
    n_exemplars: int,
    seed: int,
    out_csv: Path,
    overwrite: bool,
    dry_run: bool,
    llm,
) -> int:
    tag_prefix = variant_tag(mode, exemplar_content)
    raw_jsonl = out_csv.with_name(out_csv.stem + "_raw.jsonl")
    if overwrite:
        for p in (out_csv, raw_jsonl):
            if p.exists():
                p.unlink()
                print(f"[{tag_prefix}] --overwrite: deleted {p.name}")
    else:
        removed = purge_error_rows(out_csv)
        if removed:
            print(f"[{tag_prefix}] purged {removed} error row(s)")

    done = load_done_ids(out_csv)
    if done:
        print(f"[{tag_prefix}] resume: skip {len(done)} done ids")

    system_prompt = system_prompt_for_mode(mode)
    failures = 0

    for i, rec in enumerate(l1_rows, 1):
        seg_id = (rec.get("segment_id") or f"l1_{i}").strip()
        done_key = f"{seg_id}::{mode}::{exemplar_content}"
        tag = f"[{tag_prefix} {i}/{len(l1_rows)}] {seg_id}"
        if done_key in done:
            print(f"{tag} done, skip")
            continue

        input_obj = l1_to_input_obj(rec)
        dim = input_obj["dim"]
        l1_text = input_obj.get("l1") or ""
        cands = by_dim.get(dim, [])
        if not cands:
            failures += 1
            row = {
                "l1_segment_id": seg_id,
                "l1_text": l1_text,
                "gold_axis": input_obj["axis"],
                "gold_dim": dim,
                "gold_subject": input_obj["subject"],
                "source_instrument": input_obj["source"],
                "severity": input_obj["severity"],
                "style_mode": mode,
                "exemplar_content": exemplar_content,
                "temperature": temperature,
                **pack_exemplar_fields([]),
                "input_json": json.dumps(input_obj, ensure_ascii=False),
                "amateur_text": "",
                "expert_text": "",
                "generated_dialogue": "",
                "problem_state_text": "",
                "error": f"no_style_pool for dim={dim}",
            }
            if not dry_run:
                append_row(out_csv, row)
            print(f"{tag} ERROR: no style pool for dim={dim}")
            continue

        # Per-record RNG: same seed + segment → same exemplars across modes.
        row_rng = random.Random(f"{seed}|{seg_id}|{dim}|{n_exemplars}")
        exemplars = pick_exemplars(
            cands,
            n_exemplars,
            row_rng,
            target_source=input_obj.get("source"),
            target_dim=dim,
            dims_by_turn=dims_by_turn,
        )
        user_prompt = build_user_prompt(
            input_obj,
            exemplars,
            l1_text=l1_text,
            mode=mode,
            exemplar_content=exemplar_content,
        )
        style_packed = pack_exemplar_fields(exemplars)

        base = {
            "l1_segment_id": seg_id,
            "l1_text": l1_text,
            "gold_axis": input_obj["axis"],
            "gold_dim": dim,
            "gold_subject": input_obj["subject"],
            "source_instrument": input_obj["source"],
            "severity": input_obj["severity"],
            "style_mode": mode,
            "exemplar_content": exemplar_content,
            "temperature": temperature,
            **style_packed,
            "input_json": json.dumps(input_obj, ensure_ascii=False),
            "amateur_text": "",
            "expert_text": "",
            "generated_dialogue": "",
            "problem_state_text": "",
            "error": "",
        }

        if dry_run:
            print(f"\n===== {tag} temp={temperature} =====")
            print(user_prompt[:900], "..." if len(user_prompt) > 900 else "")
            continue

        raw_reply = ""
        try:
            t0 = time.perf_counter()
            raw_reply = call_generate(llm, system_prompt, user_prompt, temperature)
            elapsed = time.perf_counter() - t0
            amateur, expert, pst = parse_generation(raw_reply)
            dialogue = f'Amateur: "{amateur}"\nExpert: "{expert}"'
            base["amateur_text"] = amateur
            base["expert_text"] = expert
            base["generated_dialogue"] = dialogue
            base["problem_state_text"] = pst
            append_row(out_csv, base)
            append_raw(
                raw_jsonl,
                {
                    "l1_segment_id": seg_id,
                    "style_mode": mode,
                    "exemplar_content": exemplar_content,
                    "temperature": temperature,
                    "input_json": input_obj,
                    "style_turn_ids": style_packed["style_turn_ids"],
                    "style_exemplars": json.loads(style_packed["style_exemplars_json"]),
                    "user_prompt": user_prompt,
                    "raw_reply": raw_reply,
                    "elapsed_s": round(elapsed, 2),
                },
            )
            print(f"{tag} ok ({elapsed:.1f}s) pst={pst[:50]!r}...")
        except Exception as e:
            failures += 1
            base["error"] = str(e)
            base["generated_dialogue"] = raw_reply
            append_row(out_csv, base)
            append_raw(
                raw_jsonl,
                {
                    "l1_segment_id": seg_id,
                    "style_mode": mode,
                    "exemplar_content": exemplar_content,
                    "error": str(e),
                    "style_turn_ids": style_packed["style_turn_ids"],
                    "style_exemplars": json.loads(style_packed["style_exemplars_json"]),
                    "user_prompt": user_prompt,
                    "raw_reply": raw_reply,
                },
            )
            print(f"{tag} ERROR: {e}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "L2 from L1 + MixAssist style pool "
            "(modes × exemplar_content; fixed low temperature)"
        )
    )
    parser.add_argument("--l1", type=Path, default=DEFAULT_L1)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(VALID_MODES),
        default=["retarget", "strict", "free"],
        help="Rewrite policy: retarget | strict | free",
    )
    parser.add_argument(
        "--exemplar-content",
        choices=list(VALID_EXEMPLAR_CONTENTS),
        default="raw",
        help="What MixAssist text to feed: problem_text | raw (default raw)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=FIXED_TEMPERATURE,
        help=f"Shared temperature for all modes (default {FIXED_TEMPERATURE})",
    )
    parser.add_argument(
        "--n-exemplars",
        type=int,
        default=1,
        help="Style exemplars per L1 row (default 1; retarget always uses 1)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Exemplar sampling seed (default 123; change to reshuffle MixAssist refs)",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="llama.cpp n_batch (T4x2 OOM → try 256)",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--tensor-split",
        type=parse_tensor_split,
        default=None,
        help="Multi-GPU proportions, e.g. 0.5,0.5 (Kaggle T4x2). Auto when >=2 GPUs.",
    )
    parser.add_argument(
        "--no-tensor-split",
        action="store_true",
        help="Force single-GPU (disable auto multi-GPU split)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print prompts only; no model load",
    )
    args = parser.parse_args()

    if not args.l1.is_file():
        sys.exit(f"找不到 L1: {args.l1}")
    if not args.pool.is_file():
        sys.exit(f"找不到 pool: {args.pool}")

    l1_rows = load_l1_records(args.l1)
    if args.limit is not None:
        l1_rows = l1_rows[: args.limit]
    pool = load_pool(args.pool)
    by_dim = index_pool_by_dim(pool)
    dims_by_turn = index_dims_by_turn(pool)

    print(f"L1 records: {len(l1_rows)} from {args.l1.name}")
    print(f"Style pool: {len(pool)} rows from {args.pool.name} (no confidence filter)")
    print(f"exemplar_content={args.exemplar_content}  temperature={args.temperature}")
    print("Pool coverage (rows / single-dim turns preferred by gate A):")
    for dim in sorted(by_dim):
        turns = {turn_key(r) for r in by_dim[dim]}
        single = sum(1 for t in turns if dims_by_turn.get(t) == {dim})
        print(f"  {dim:<22} {len(by_dim[dim]):>4} rows  {single:>3}/{len(turns)} single-dim turns")
    missing = sorted(
        {(r.get("dimension") or "").strip().lower() for r in l1_rows} - set(by_dim)
    )
    if missing:
        print(f"WARNING: L1 dims with empty pool: {missing}")

    llm = None
    if not args.dry_run:
        llm = load_llm(
            args.model_path,
            args.n_ctx,
            args.n_gpu_layers,
            args.verbose,
            n_batch=args.n_batch,
            tensor_split=args.tensor_split,
            disable_tensor_split=args.no_tensor_split,
        )

    total_fail = 0
    written: list[Path] = []
    for mode in args.modes:
        out_csv = output_csv_path(args.output_dir, mode, args.exemplar_content)
        written.append(out_csv)
        n_ex = args.n_exemplars
        if mode == "retarget" and n_ex != 1:
            print(
                f"[{mode}] forcing n_exemplars=1 "
                f"(was {n_ex}; retarget rewrites a single MixAssist turn)"
            )
            n_ex = 1
        print(
            f"\n=== mode={mode} exemplar_content={args.exemplar_content} "
            f"temp={args.temperature} → {out_csv.name} ==="
        )
        total_fail += run_mode(
            mode=mode,
            exemplar_content=args.exemplar_content,
            temperature=args.temperature,
            l1_rows=l1_rows,
            by_dim=by_dim,
            dims_by_turn=dims_by_turn,
            n_exemplars=n_ex,
            seed=args.seed,
            out_csv=out_csv,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            llm=llm,
        )

    if args.dry_run:
        print("\nDry-run only — no model calls.")
        return

    print(f"\nDone. Failures (rows): {total_fail}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
