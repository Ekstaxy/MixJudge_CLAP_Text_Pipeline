# -*- coding: utf-8 -*-
"""
L2 generation consistency check — re-label generated Amateur/Expert dialogue
with the same MixAssist labeling prompt + GGUF model.

Input:  new_outputs/l2_from_l1_{mode}_raw.csv
Output: new_outputs/labeled_l2_gguf_{mode}_raw.csv (+ _raw.jsonl)

Uses the same SYSTEM_PROMPT / OUTPUT_FIELDS as MixAssist labeling.
Gold axis/dim are NOT fed to the model (only source_instrument as TOPIC).

Usage (from repo root):
  python src/labeling/label_l2_generated.py --overwrite --n-batch 256

  python src/labeling/label_l2_generated.py \\
    --inputs new_outputs/l2_from_l1_retarget_raw.csv \\
             new_outputs/l2_from_l1_strict_raw.csv \\
             new_outputs/l2_from_l1_free_raw.csv \\
    --output-dir new_outputs --n-batch 256

Then compare:
  python scripts/compare_l2_labels.py \\
    --l2 outputs/l2_from_l1_retarget_raw.csv \\
    --labeled outputs/labeled_l2_gguf_retarget_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from labeling.labeling_gguf import (  # noqa: E402
    DEFAULT_GGUF,
    N_BATCH,
    N_CTX,
    N_GPU_LAYERS,
    call_llm,
    load_llm,
    parse_tensor_split,
)
from labeling.labeling_common import (  # noqa: E402
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    append_raw,
    append_rows,
    build_user_message,
    error_row,
    extract_json,
    labels_to_rows,
    load_done_keys,
    purge_error_rows,
)

VALID_MODES = ("retarget", "strict", "free")
VALID_CONTENTS = ("problem_text", "raw")

DEFAULT_INPUTS = [
    PROJECT_ROOT / "new_outputs" / f"l2_from_l1_{mode}_raw.csv"
    for mode in VALID_MODES
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "new_outputs"


def variant_output(output_dir: Path, variant: str) -> Path:
    return output_dir / f"labeled_l2_gguf_{variant}.csv"


def infer_variant(path: Path, row: dict) -> str:
    """Return variant tag like retarget_raw / strict_problem_text."""
    mode = str(row.get("style_mode") or "").strip().lower()
    content = str(row.get("exemplar_content") or "").strip().lower()
    if mode in VALID_MODES and content in VALID_CONTENTS:
        return f"{mode}_{content}"

    name = path.stem.lower()  # l2_from_l1_retarget_raw
    for m in VALID_MODES:
        for c in VALID_CONTENTS:
            tag = f"{m}_{c}"
            if name.endswith(tag) or f"_{tag}" in name:
                return tag
    # legacy: l2_from_l1_strict.csv
    for m in VALID_MODES:
        if name.endswith(f"_{m}") or name == f"l2_from_l1_{m}":
            return m
    return mode or "unknown"


def l2_row_to_turn(row: dict, variant: str) -> dict | None:
    if str(row.get("error") or "").strip():
        return None
    amateur = str(row.get("amateur_text") or "").strip()
    expert = str(row.get("expert_text") or "").strip()
    if not amateur and not expert:
        return None

    segment_id = str(row.get("l1_segment_id") or "").strip()
    if not segment_id:
        return None

    return {
        "conversation_id": segment_id,
        "split": variant,
        "topic": str(row.get("source_instrument") or "").strip(),
        "turn_id": "0",
        "has_content": "True",
        "audio_file": "",
        "input_history": "[]",
        "user": amateur,
        "assistant": expert,
    }


def load_l2_turns(path: Path, limit: int | None) -> tuple[str, list[dict]]:
    if not path.is_file():
        sys.exit(f"找不到 L2 輸入: {path}")

    with open(path, encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    turns: list[dict] = []
    variant = "unknown"
    for row in raw_rows:
        variant = infer_variant(path, row)
        turn = l2_row_to_turn(row, variant)
        if turn is None:
            continue
        turns.append(turn)
        if limit is not None and len(turns) >= limit:
            break

    if variant == "unknown" and turns:
        variant = turns[0]["split"]
    return variant, turns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-label L2 generated dialogues with MixAssist labeling (GGUF)"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="L2 CSV paths (default: existing new_outputs/l2_from_l1_{mode}_raw.csv)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument(
        "--n-batch",
        type=int,
        default=N_BATCH,
        help=f"llama.cpp n_batch (default {N_BATCH}; T4x2 OOM → try 256)",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS)
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
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        sys.exit("SYSTEM_PROMPT 為空,請在 src/prompts/labeling_prompt.py 填入 prompt。")

    inputs = args.inputs
    if inputs is None:
        inputs = [p for p in DEFAULT_INPUTS if p.is_file()]
        if not inputs:
            # legacy fallbacks
            legacy = [
                PROJECT_ROOT / "outputs" / "l2_from_l1_strict.csv",
                PROJECT_ROOT / "outputs" / "l2_from_l1_free.csv",
                PROJECT_ROOT / "outputs" / "l2_from_l1_retarget.csv",
            ]
            inputs = [p for p in legacy if p.is_file()]
        if not inputs:
            sys.exit(
                "找不到 L2 輸入。請先 generate，或用 --inputs 指定 "
                "l2_from_l1_{mode}_{exemplar_content}.csv"
            )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    by_variant: dict[str, list[dict]] = {}
    for inp in inputs:
        variant, turns = load_l2_turns(inp, args.limit)
        if not turns:
            print(f"[skip] {inp}: 無有效對話列")
            continue
        by_variant.setdefault(variant, []).extend(turns)
        print(f"[load] {inp.name}: {len(turns)} turns → variant={variant}")

    if not by_variant:
        sys.exit("沒有可標註的 L2 對話。")

    llm = load_llm(
        args.model_path,
        args.n_ctx,
        args.n_gpu_layers,
        args.verbose,
        n_batch=args.n_batch,
        tensor_split=args.tensor_split,
        disable_tensor_split=args.no_tensor_split,
    )
    failures: list[tuple] = []
    written: list[Path] = []

    for variant, rows in by_variant.items():
        output_csv = variant_output(output_dir, variant)
        raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
        written.append(output_csv)

        if args.overwrite:
            for p in (output_csv, raw_jsonl):
                if p.exists():
                    p.unlink()
                    print(f"[{variant}] --overwrite: 已刪除 {p.name}")

        removed = purge_error_rows(output_csv)
        if removed:
            print(f"[{variant}] 已清除 {removed} 筆舊 error 列,將重試。")

        done = load_done_keys(output_csv)
        if done:
            print(
                f"[{variant}] 偵測到既有輸出 {output_csv.name},"
                f"已完成 {len(done)} 個 turn,將跳過。"
            )

        print(f"=== {variant}: {len(rows)} turns → {output_csv.name} ===")

        for i, row in enumerate(rows, 1):
            key = (variant, row.get("conversation_id", ""), row.get("turn_id", ""))
            tag = f"[{variant} {i}/{len(rows)}] {key[1]} turn {key[2]}"
            if key in done:
                print(f"{tag} 已標過,跳過")
                continue

            user_message = build_user_message(row)
            raw_reply = ""
            try:
                t0 = time.perf_counter()
                raw_reply = call_llm(llm, system_prompt, user_message)
                elapsed = time.perf_counter() - t0
                parsed = extract_json(raw_reply)
                labels = parsed.get("labels", [])
                if not isinstance(labels, list):
                    labels = [labels]
                out_rows = labels_to_rows(row, labels)
                append_rows(output_csv, out_rows)
                append_raw(
                    raw_jsonl,
                    {
                        "key": list(key),
                        "model_path": str(args.model_path),
                        "user_message": user_message,
                        "raw_reply": raw_reply,
                    },
                )
                dims = [
                    f"{r['problem_axis']}/{r['problem_dimension']}" for r in out_rows
                ]
                print(f"{tag} -> {len(out_rows)} label(s): {dims} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"{tag} -> 錯誤: {e}")
                failures.append((key, str(e)))
                append_rows(output_csv, [error_row(row, str(e))])
                append_raw(
                    raw_jsonl,
                    {
                        "key": list(key),
                        "error": str(e),
                        "user_message": user_message,
                        "raw_reply": raw_reply,
                    },
                )

    print("\n完成。輸出:")
    for p in written:
        print(f"  {p}")
        print(f"  {p.with_name(p.stem + '_raw.jsonl')}")
    if failures:
        print(f"\n失敗 {len(failures)} 筆 (重跑同一指令會自動重試):")
        for key, err in failures:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
