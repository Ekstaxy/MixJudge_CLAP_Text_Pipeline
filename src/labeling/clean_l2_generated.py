# -*- coding: utf-8 -*-
"""LLM-clean generated L2: flip reversed polarity; drop wrong_source / malformed.

Does not regenerate dialogue. Uses the same GGUF as labeling (single GPU).

Usage (from repo root):
  python src/labeling/clean_l2_generated.py --overwrite --n-batch 512
  python src/labeling/clean_l2_generated.py \\
    --inputs outputs/l2_from_l1_retarget_raw.csv \\
             outputs/l2_from_l1_strict_raw.csv \\
             outputs/l2_from_l1_free_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from labeling.labeling_common import (  # noqa: E402
    PROJECT_ROOT,
    DIMENSION_TO_AXIS,
    extract_json,
    normalize_dimension,
)
from labeling.labeling_gguf import (  # noqa: E402
    DEFAULT_GGUF,
    N_BATCH,
    N_CTX,
    N_GPU_LAYERS,
    call_llm,
    load_llm,
    parse_tensor_split,
)
from prompts.l2_clean_prompt import (  # noqa: E402
    OPPOSITE_DIM,
    SYSTEM_PROMPT,
    build_user_message,
)

VALID_MODES = ("retarget", "strict", "free")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DROP_REASONS = frozenset({"wrong_source", "malformed"})
EXTRA_FIELDS = (
    "gold_dim_original",
    "clean_action",
    "clean_flipped",
    "clean_drop_reason",
    "clean_reasoning",
    "clean_error",
)


def infer_mode(path: Path, row: dict) -> str:
    mode = str(row.get("style_mode") or "").strip().lower()
    if mode in VALID_MODES:
        return mode
    name = path.stem.lower()
    for m in VALID_MODES:
        if f"_{m}_" in f"_{name}_" or name.endswith(f"_{m}_raw") or name.endswith(f"_{m}"):
            return m
    return "unknown"


def output_csv(output_dir: Path, mode: str) -> Path:
    return output_dir / f"l2_from_l1_{mode}_raw_llm_clean.csv"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("clean_error") or "").strip():
                continue
            sid = str(row.get("l1_segment_id") or "").strip()
            if sid:
                done.add(sid)
    return done


def append_row(path: Path, row: dict, fields: list[str]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def apply_decision(row: dict, obj: dict) -> dict:
    out = dict(row)
    gold0 = str(row.get("gold_dim") or "").strip().lower()
    out["gold_dim_original"] = gold0
    action = str(obj.get("action") or "keep").strip().lower()
    if action not in ("keep", "drop"):
        action = "keep"
    reason = str(obj.get("drop_reason") or "").strip().lower()
    if action == "drop" and reason not in DROP_REASONS:
        reason = "malformed" if reason else "wrong_source"

    proposed = normalize_dimension(obj.get("gold_dim") or gold0)
    flipped_flag = str(obj.get("flipped") or "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    gold_out = gold0
    flipped = False
    if action == "keep":
        opp = OPPOSITE_DIM.get(gold0)
        if proposed == gold0:
            flipped = False
        elif opp and proposed == opp:
            gold_out = opp
            flipped = True
        elif flipped_flag and opp:
            gold_out = opp
            flipped = True
        else:
            gold_out = gold0
            flipped = False

    out["gold_dim"] = gold_out if action == "keep" else gold0
    out["gold_axis"] = DIMENSION_TO_AXIS.get(
        str(out["gold_dim"]), str(row.get("gold_axis") or "")
    )
    out["clean_action"] = action
    out["clean_flipped"] = "true" if flipped else "false"
    out["clean_drop_reason"] = reason if action == "drop" else ""
    out["clean_reasoning"] = str(obj.get("label_reasoning") or "").strip()
    out["clean_error"] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-clean L2 CSVs (flip polarity / drop wrong source & malformed)"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--n-batch", type=int, default=N_BATCH)
    parser.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS)
    parser.add_argument("--tensor-split", type=parse_tensor_split, default=None)
    parser.add_argument("--no-tensor-split", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = args.inputs or [
        PROJECT_ROOT / "outputs" / f"l2_from_l1_{m}_raw.csv" for m in VALID_MODES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, list[dict]]] = []
    for path in inputs:
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        rows = load_rows(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        mode = infer_mode(path, rows[0] if rows else {})
        jobs.append((mode, path, rows))
    if not jobs:
        sys.exit("沒有可清洗的 L2 CSV。")

    llm = load_llm(
        args.model_path,
        args.n_ctx,
        args.n_gpu_layers,
        args.verbose,
        n_batch=args.n_batch,
        tensor_split=args.tensor_split,
        disable_tensor_split=args.no_tensor_split,
    )
    system_prompt = SYSTEM_PROMPT.strip()

    for mode, src, rows in jobs:
        out_path = output_csv(args.output_dir, mode)
        raw_jsonl = out_path.with_name(out_path.stem + "_raw.jsonl")
        fields = list(rows[0].keys()) if rows else []
        for extra in EXTRA_FIELDS:
            if extra not in fields:
                fields.append(extra)

        if args.overwrite:
            for p in (out_path, raw_jsonl):
                if p.exists():
                    p.unlink()
                    print(f"[{mode}] --overwrite: deleted {p.name}")
        done = load_done_ids(out_path)
        if done:
            print(f"[{mode}] resume: skip {len(done)} done ids")

        n_keep = n_drop = n_flip = n_fail = 0
        t0 = time.time()
        for i, row in enumerate(rows, 1):
            sid = str(row.get("l1_segment_id") or "").strip()
            tag = f"[{mode} {i}/{len(rows)}] {sid}"
            if not sid:
                continue
            if sid in done:
                print(f"{tag} done, skip")
                continue
            if str(row.get("error") or "").strip():
                n_fail += 1
                rec = dict(row)
                rec["clean_action"] = "drop"
                rec["clean_drop_reason"] = "malformed"
                rec["clean_error"] = "source_error"
                rec["gold_dim_original"] = rec.get("gold_dim", "")
                rec["clean_flipped"] = "false"
                rec["clean_reasoning"] = ""
                append_row(out_path, rec, fields)
                continue

            user_msg = build_user_message(row)
            try:
                raw = call_llm(llm, system_prompt, user_msg)
                obj = extract_json(raw)
                rec = apply_decision(row, obj if isinstance(obj, dict) else {})
            except Exception as e:
                n_fail += 1
                rec = dict(row)
                rec["gold_dim_original"] = rec.get("gold_dim", "")
                rec["clean_action"] = "keep"
                rec["clean_flipped"] = "false"
                rec["clean_drop_reason"] = ""
                rec["clean_reasoning"] = ""
                rec["clean_error"] = str(e)
                append_row(out_path, rec, fields)
                with raw_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"l1_segment_id": sid, "error": str(e)}) + "\n")
                print(f"{tag} ERROR {e}")
                continue

            if rec["clean_action"] == "drop":
                n_drop += 1
            else:
                n_keep += 1
            if rec["clean_flipped"] == "true":
                n_flip += 1
            append_row(out_path, rec, fields)
            with raw_jsonl.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "l1_segment_id": sid,
                            "clean_action": rec["clean_action"],
                            "gold_dim": rec["gold_dim"],
                            "gold_dim_original": rec["gold_dim_original"],
                            "raw": raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"{tag} {rec['clean_action']}"
                f"{' flip→'+rec['gold_dim'] if rec['clean_flipped']=='true' else ''}"
                f"{' '+rec['clean_drop_reason'] if rec['clean_action']=='drop' else ''}"
            )

        elapsed = time.time() - t0
        print(
            f"[{mode}] wrote {out_path}  keep={n_keep} drop={n_drop} "
            f"flip={n_flip} fail={n_fail} ({elapsed:.0f}s)"
        )
        print(
            f"[{mode}] kept-only file: filter clean_action==keep when you use it"
        )


if __name__ == "__main__":
    main()
