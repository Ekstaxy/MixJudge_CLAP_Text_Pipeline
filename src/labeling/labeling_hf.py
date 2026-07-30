# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a) — Hugging Face / transformers 版。

對 train / validation / test 每個 turn 用 Gemma 4 31B 標註。
共用 schema / I/O / parse 見 labeling_common.py；prompt 見 prompts/labeling_prompt.py。

本地 GPU 推論版 (V100 32GB):
  - V100 不支援 bfloat16,一律用 float16 計算。
  - 31B fp16 ~62GB,放不進 32GB,預設 4-bit NF4 (~18GB)。
  - --precision: 4bit / 8bit / fp16 / auto (預設 auto)。
  - torch 請用 cu126 (cu130 不支援 V100 CC 7.0)。

用法 (在專案根目錄):
  python src/labeling/labeling_hf.py --splits train --limit 5
  python src/labeling/labeling_hf.py
  python src/labeling/labeling_hf.py --precision 4bit
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

# V100: CUDA 13 ptxas 不支援 sm_70,關掉 torch native Triton JIT
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from labeling.labeling_common import (  # noqa: E402
    DEFAULT_OUTPUT,
    MAX_NEW_TOKENS,
    SPLIT_FILES,
    SYSTEM_PROMPT,
    append_raw,
    append_rows,
    build_user_message,
    error_row,
    extract_json,
    labels_to_rows,
    load_done_keys,
    load_split,
    purge_error_rows,
)

# ====================== HF 專用 ======================
HF_TOKEN = ""  # <-- Hugging Face token (已下載過可留空)
MODEL_ID = "google/gemma-4-31B-it"


def pick_precision(requested: str) -> str:
    """依 VRAM 決定精度。V100 32GB: 31B fp16 放不下,預設 4bit。"""
    import torch

    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        sys.exit("找不到 CUDA GPU。請確認 torch 有裝 CUDA 版 (cu126) 且 GPU 可用。")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 70:
        return "fp16"
    if total_gb >= 40:
        return "8bit"
    return "4bit"


def load_model(precision: str):
    """載入本地模型。V100 (compute 7.0) 不支援 bf16,計算一律 float16。"""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    if precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:  # fp16
        kwargs["dtype"] = torch.float16

    print(f"載入模型 {MODEL_ID} (precision={precision}) 到 GPU,請稍候...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print("模型載入完成。")
    return processor, model


def call_llm(processor, model, system_prompt: str, user_message: str) -> str:
    import torch

    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    reply = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    reply = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", reply, flags=re.DOTALL)
    return reply.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="MixAssist LLM labeling pipeline (HF)")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLIT_FILES),
        default=list(SPLIT_FILES),
        help="要處理的 split (預設全部)",
    )
    parser.add_argument("--limit", type=int, default=None, help="每個 split 只處理前 N 個 turn")
    parser.add_argument(
        "--random-window",
        action="store_true",
        help="搭配 --limit: 隨機取連續 N 筆",
    )
    parser.add_argument("--seed", type=int, default=None, help="隨機視窗 seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--precision",
        choices=["auto", "4bit", "8bit", "fp16"],
        default="auto",
    )
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        sys.exit("SYSTEM_PROMPT 為空,請在 src/prompts/labeling_prompt.py 填入 prompt。")

    output_csv: Path = args.output
    raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    removed = purge_error_rows(output_csv)
    if removed:
        print(f"已清除 {removed} 筆舊 error 列,將重試。")

    done = load_done_keys(output_csv)
    if done:
        print(f"偵測到既有輸出,已完成 {len(done)} 個 turn,將跳過。")

    precision = pick_precision(args.precision)
    processor, model = load_model(precision)
    failures = []

    if args.seed is not None:
        random.seed(args.seed)

    for split in args.splits:
        rows = load_split(split, args.limit, random_window=args.random_window)
        print(f"=== {split}: {len(rows)} turns ===")

        for i, row in enumerate(rows, 1):
            key = (split, row.get("conversation_id", ""), row.get("turn_id", ""))
            tag = f"[{split} {i}/{len(rows)}] {key[1]} turn {key[2]}"
            if key in done:
                print(f"{tag} 已標過,跳過")
                continue

            user_message = build_user_message(row)
            raw_reply = ""
            try:
                raw_reply = call_llm(processor, model, system_prompt, user_message)
                parsed = extract_json(raw_reply)
                labels = parsed.get("labels", [])
                if not isinstance(labels, list):
                    labels = [labels]
                out_rows = labels_to_rows(row, labels)
                append_rows(output_csv, out_rows)
                append_raw(
                    raw_jsonl,
                    {"key": list(key), "user_message": user_message, "raw_reply": raw_reply},
                )
                dims = [f"{r['problem_axis']}/{r['problem_dimension']}" for r in out_rows]
                print(f"{tag} -> {len(out_rows)} label(s): {dims}")
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

    print(f"\n完成。輸出: {output_csv}")
    print(f"原始回覆: {raw_jsonl}")
    if failures:
        print(f"\n失敗 {len(failures)} 筆 (重跑同一指令會自動重試):")
        for key, err in failures:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
