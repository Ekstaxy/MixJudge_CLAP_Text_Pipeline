# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a) — 本地 GGUF / llama.cpp 版。

功能與 src/labeling.py 相同 (同一套 prompt / schema / 輸出格式),
改用 llama-cpp-python 載入 Q4_K_M GGUF,在 V100 上把全部層丟 GPU。

預設模型檔:
  models/google_gemma-4-31B-it-Q4_K_M.gguf

用法:
  python src/labeling_gguf.py --splits train --limit 5
  python src/labeling_gguf.py
  python src/labeling_gguf.py --model-path models/xxx.gguf

輸出 (每個 split 各一檔):
  outputs/labeled_turns_gguf_train.csv
  outputs/labeled_turns_gguf_validation.csv
  outputs/labeled_turns_gguf_test.csv

需要套件:
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

中斷後重跑會自動跳過已標過的 turn (增量寫入)。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path


def _prepare_cuda_libs() -> None:
    """預載 libllama.so 所需的 CUDA 12 動態庫。

    cu124 wheel 的 libllama 依賴 libcudart / libcublas 等,但本機預設
    搜尋路徑找不到 pip 裝在 ~/.local/.../nvidia/*/lib 的檔案;且若 so 有
    RUNPATH,事後改 LD_LIBRARY_PATH 也無效。因此直接 CDLL 全部預載。
    """
    import ctypes

    nvidia_root = Path.home() / ".local/lib/python3.12/site-packages/nvidia"
    # 順序重要: Lt 先於 cublas;cudart / cuda 先於 ggml-cuda
    must = [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        nvidia_root / "cuda_runtime/lib/libcudart.so.12",
        nvidia_root / "nvjitlink/lib/libnvJitLink.so.12",
        nvidia_root / "cublas/lib/libcublasLt.so.12",
        nvidia_root / "cublas/lib/libcublas.so.12",
    ]
    missing = [p for p in must if not p.exists()]
    if missing:
        sys.exit(
            "缺少 CUDA 12 動態庫:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n請安裝: pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu126"
        )

    extras = []
    for lib_dir in nvidia_root.glob("*/lib"):
        extras.append(str(lib_dir))
    host = ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
    old = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "cuda/compat" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(host + extras + old))

    for path in must:
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            sys.exit(f"無法載入 {path}: {e}")


_prepare_cuda_libs()

from llama_cpp import Llama  # noqa: E402

from labeling import (  # noqa: E402
    SYSTEM_PROMPT,
    SPLIT_FILES,
    build_user_message,
    extract_json,
    labels_to_rows,
    error_row,
    load_split,
    load_done_keys,
    append_rows,
    append_raw,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = PROJECT_ROOT / "models" / "google_gemma-4-31B-it-Q4_K_M.gguf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def split_output(output_dir: Path, split: str) -> Path:
    return output_dir / f"labeled_turns_gguf_{split}.csv"

# ----- 影響品質 vs 速度 -----
# temperature : 越低越穩定、適合標註;越高越有創意但不一致。標註建議 0.05~0.2
# n_ctx       : 能看多長上下文。太小會砍掉 HISTORY → stem/dimension 變差;太大更慢、更吃 VRAM
# max_tokens  : 單次回覆長度上限。太小 JSON 被截斷;太大幾乎不影響品質只是上限
# n_batch     : 只影響速度,不影響標註品質
# n_gpu_layers: -1=全上 GPU。降到 CPU 會更慢,品質幾乎無好處
# 模型量化檔  : Q4→Q5/Q6/Q8 對品質影響最大 (要另下載,VRAM 更吃)
TEMPERATURE = 0.1
N_CTX = 6144
N_BATCH = 512
MAX_TOKENS = 768
N_GPU_LAYERS = -1
TOP_P = 0.95
TOP_K = 64


def load_llm(model_path: Path, n_ctx: int, n_gpu_layers: int, verbose: bool) -> Llama:
    if not model_path.is_file():
        sys.exit(
            f"找不到 GGUF: {model_path}\n"
            "請先下載,例如:\n"
            "  hf download bartowski/google_gemma-4-31B-it-GGUF "
            "--include '*Q4_K_M*.gguf' --local-dir models"
        )
    print(f"載入 GGUF → GPU: {model_path} (n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=N_BATCH,
        n_ubatch=N_BATCH,
        verbose=verbose,
    )
    print("模型載入完成。")
    return llm


def call_llm(llm: Llama, system_prompt: str, user_message: str) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MixAssist LLM labeling pipeline (local GGUF / llama.cpp)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLIT_FILES),
        default=list(SPLIT_FILES),
        help="要處理的 split (預設全部)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每個 split 只處理前 N 個 turn (測試用)",
    )
    parser.add_argument(
        "--random-window",
        action="store_true",
        help="搭配 --limit: 隨機取連續 N 筆",
    )
    parser.add_argument("--seed", type=int, default=None, help="隨機視窗 seed")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_GGUF,
        help=f"GGUF 路徑 (預設 {DEFAULT_GGUF})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"輸出目錄,每個 split 一檔 labeled_turns_gguf_<split>.csv (預設 {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--n-ctx", type=int, default=N_CTX, help="上下文長度")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=N_GPU_LAYERS,
        help="丟到 GPU 的層數 (-1 = 全部)",
    )
    parser.add_argument("--verbose", action="store_true", help="顯示 llama.cpp 載入細節")
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        sys.exit("SYSTEM_PROMPT 為空,請在 src/labeling.py 填入 prompt。")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm(args.model_path, args.n_ctx, args.n_gpu_layers, args.verbose)
    failures = []
    written: list[Path] = []

    if args.seed is not None:
        random.seed(args.seed)

    for split in args.splits:
        output_csv = split_output(output_dir, split)
        raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
        written.append(output_csv)

        done = load_done_keys(output_csv)
        if done:
            print(f"[{split}] 偵測到既有輸出 {output_csv.name},已完成 {len(done)} 個 turn,將跳過。")

        rows = load_split(split, args.limit, random_window=args.random_window)
        print(f"=== {split}: {len(rows)} turns → {output_csv.name} ===")

        for i, row in enumerate(rows, 1):
            key = (split, row.get("conversation_id", ""), row.get("turn_id", ""))
            tag = f"[{split} {i}/{len(rows)}] {key[1]} turn {key[2]}"
            if key in done:
                print(f"{tag} 已標過,跳過")
                continue

            user_message = build_user_message(row)
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
                dims = [r["problem_dimension"] for r in out_rows]
                print(f"{tag} -> {len(out_rows)} label(s): {dims} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"{tag} -> 錯誤: {e}")
                failures.append((key, str(e)))
                append_rows(output_csv, [error_row(row, str(e))])
                append_raw(raw_jsonl, {"key": list(key), "error": str(e)})

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
