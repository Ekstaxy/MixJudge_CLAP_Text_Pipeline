# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a) — 本地 GGUF / llama.cpp 版。

功能與 labeling_hf.py 相同 (同一套 prompt / schema / 輸出格式),
共用邏輯在 labeling_common.py；prompt 在 prompts/labeling_prompt.py。
改用 llama-cpp-python 載入 Q4_K_M GGUF,在本機單卡把全部層丟 GPU（預設不做 tensor_split）。

預設模型檔:
  models/google_gemma-4-31B-it-Q4_K_M.gguf

用法 (在專案根目錄):
  python src/labeling/labeling_gguf.py --splits train --limit 5
  python src/labeling/labeling_gguf.py
  python src/labeling/labeling_gguf.py --model-path models/xxx.gguf

或:
  cd src && python -m labeling.labeling_gguf

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

# Bootstrap src/ onto path when run as a file (python src/labeling/labeling_gguf.py)
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _nvidia_package_roots() -> list[Path]:
    """Locate pip-installed nvidia/*/lib trees (version-agnostic)."""
    roots: list[Path] = []
    home_local = Path.home() / ".local/lib"
    if home_local.is_dir():
        for p in sorted(home_local.glob("python*/site-packages/nvidia")):
            if p.is_dir():
                roots.append(p)
    try:
        import site

        candidates = list(site.getsitepackages())
        user = site.getusersitepackages()
        if user:
            candidates.append(user)
        for sp in candidates:
            p = Path(sp) / "nvidia"
            if p.is_dir():
                roots.append(p)
    except Exception:
        pass
    # preserve order, drop dupes
    return list(dict.fromkeys(roots))


def _prepare_cuda_libs() -> None:
    """預載 libllama.so 所需的 CUDA 12 動態庫（本機 pip nvidia 套件）。

    cu124 wheel 的 libllama 依賴 libcudart / libcublas 等;本機若 so 有
    RUNPATH,事後改 LD_LIBRARY_PATH 也無效,因此直接 CDLL 預載。

    Kaggle / 系統已帶 CUDA 的環境通常沒有 ~/.local/.../nvidia;此時略過,
    交給動態連結器找系統庫,不要硬 exit。
    """
    import ctypes

    libcuda_candidates = [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
        Path("/usr/local/cuda/lib64/libcuda.so.1"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
    ]
    libcuda = next((p for p in libcuda_candidates if p.exists()), None)

    nvidia_roots = _nvidia_package_roots()
    if not nvidia_roots:
        if libcuda is not None:
            try:
                ctypes.CDLL(str(libcuda), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"[cuda] warn: cannot preload {libcuda}: {e}")
        print("[cuda] no pip nvidia packages; using system CUDA (e.g. Kaggle)")
        return

    nvidia_root = None
    pip_libs: list[Path] = []
    for root in nvidia_roots:
        candidate = [
            root / "cuda_runtime/lib/libcudart.so.12",
            root / "nvjitlink/lib/libnvJitLink.so.12",
            root / "cublas/lib/libcublasLt.so.12",
            root / "cublas/lib/libcublas.so.12",
        ]
        if all(p.exists() for p in candidate):
            nvidia_root = root
            pip_libs = candidate
            break

    if nvidia_root is None:
        print(
            "[cuda] pip nvidia tree(s) incomplete; skip preload "
            "(system CUDA may still work). Tried:\n  "
            + "\n  ".join(str(r) for r in nvidia_roots)
        )
        return

    # 順序重要: Lt 先於 cublas;cudart / cuda 先於 ggml-cuda
    must = ([libcuda] if libcuda is not None else []) + pip_libs

    extras: list[str] = []
    for root in nvidia_roots:
        for lib_dir in root.glob("*/lib"):
            extras.append(str(lib_dir))
    host = [
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib64",
    ]
    old = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "cuda/compat" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(host + extras + old))

    for path in must:
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            print(f"[cuda] warn: cannot preload {path}: {e}")


from labeling.labeling_common import (  # noqa: E402
    PROJECT_ROOT,
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

DEFAULT_GGUF = PROJECT_ROOT / "models" / "google_gemma-4-31B-it-Q4_K_M.gguf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def split_output(output_dir: Path, split: str) -> Path:
    return output_dir / f"labeled_turns_gguf_{split}.csv"

# ----- 影響品質 vs 速度 -----
# temperature : 越低越穩定、適合標註;越高越有創意但不一致。標註建議 0.05~0.2
# n_ctx       : 能看多長上下文。太小會砍掉 HISTORY → stem/dimension 變差;太大更慢、更吃 VRAM
# max_tokens  : 單次回覆長度上限。太小 JSON 被截斷;太大幾乎不影響品質只是上限
# n_batch     : 只影響速度,不影響標註品質。OOM 可降到 256
# n_gpu_layers: -1=全上 GPU。降到 CPU 會更慢,品質幾乎無好處
# tensor_split: 預設關閉。僅在顯式 --tensor-split 時才跨卡（本機單卡不要開）
# 模型量化檔  : Q4→Q5/Q6/Q8 對品質影響最大 (要另下載,VRAM 更吃)
TEMPERATURE = 0.1
N_CTX = 6144
N_BATCH = 512
MAX_TOKENS = 1536
N_GPU_LAYERS = -1
TOP_P = 0.95
TOP_K = 64


def cuda_device_count() -> int:
    """Best-effort GPU count (nvidia-smi, then torch)."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if n:
            return n
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def parse_tensor_split(value: str | None) -> list[float] | None:
    """Parse '0.5,0.5' → [0.5, 0.5]. Empty/None → None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    try:
        splits = [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid --tensor-split {value!r}; expected comma-separated floats"
        ) from e
    if any(s < 0 for s in splits):
        raise argparse.ArgumentTypeError("--tensor-split values must be >= 0")
    total = sum(splits)
    if total <= 0:
        raise argparse.ArgumentTypeError("--tensor-split must sum to > 0")
    return splits


def resolve_tensor_split(
    tensor_split: list[float] | None,
    *,
    disable: bool = False,
) -> list[float] | None:
    """Use tensor_split only when the caller passes it. Never auto-split on N GPUs."""
    if disable:
        return None
    return tensor_split


def load_llm(
    model_path: Path,
    n_ctx: int,
    n_gpu_layers: int,
    verbose: bool,
    *,
    n_batch: int | None = None,
    tensor_split: list[float] | None = None,
    disable_tensor_split: bool = False,
):
    # Preload must happen before importing llama_cpp (RUNPATH).
    _prepare_cuda_libs()
    from llama_cpp import Llama

    if not model_path.is_file():
        sys.exit(
            f"找不到 GGUF: {model_path}\n"
            "請先下載,例如:\n"
            "  hf download bartowski/google_gemma-4-31B-it-GGUF "
            "--include '*Q4_K_M*.gguf' --local-dir models"
        )
    batch = N_BATCH if n_batch is None else n_batch
    split = resolve_tensor_split(tensor_split, disable=disable_tensor_split)
    n_dev = cuda_device_count()
    split_msg = f", tensor_split={split}" if split else ""
    print(
        f"載入 GGUF → GPU: {model_path} "
        f"(devices={n_dev}, n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, "
        f"n_batch={batch}{split_msg})"
    )
    kwargs: dict = {
        "model_path": str(model_path),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_batch": batch,
        "n_ubatch": batch,
        "verbose": verbose,
    }
    if split is not None:
        kwargs["tensor_split"] = split
    llm = Llama(**kwargs)
    print("模型載入完成。")
    return llm


def call_llm(llm, system_prompt: str, user_message: str) -> str:
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
        "--n-batch",
        type=int,
        default=N_BATCH,
        help=f"llama.cpp n_batch / n_ubatch (預設 {N_BATCH}; OOM 可降到 256)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=N_GPU_LAYERS,
        help="丟到 GPU 的層數 (-1 = 全部)",
    )
    parser.add_argument(
        "--tensor-split",
        type=parse_tensor_split,
        default=None,
        help="多卡比例,例如 0.5,0.5。預設關閉（本機單卡）",
    )
    parser.add_argument(
        "--no-tensor-split",
        action="store_true",
        help="強制不使用 tensor_split（預設已是單卡）",
    )
    parser.add_argument("--verbose", action="store_true", help="顯示 llama.cpp 載入細節")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="刪除既有輸出後重跑 (換 schema/prompt 後重新標註用)",
    )
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        sys.exit("SYSTEM_PROMPT 為空,請在 src/prompts/labeling_prompt.py 填入 prompt。")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm(
        args.model_path,
        args.n_ctx,
        args.n_gpu_layers,
        args.verbose,
        n_batch=args.n_batch,
        tensor_split=args.tensor_split,
        disable_tensor_split=args.no_tensor_split,
    )
    failures = []
    written: list[Path] = []

    if args.seed is not None:
        random.seed(args.seed)

    for split in args.splits:
        output_csv = split_output(output_dir, split)
        raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
        written.append(output_csv)

        if args.overwrite:
            for p in (output_csv, raw_jsonl):
                if p.exists():
                    p.unlink()
                    print(f"[{split}] --overwrite: 已刪除 {p.name}")

        removed = purge_error_rows(output_csv)
        if removed:
            print(f"[{split}] 已清除 {removed} 筆舊 error 列,將重試。")

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
                dims = [f"{r['problem_axis']}/{r['problem_dimension']}" for r in out_rows]
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
