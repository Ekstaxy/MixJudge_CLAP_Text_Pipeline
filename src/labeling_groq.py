# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a) — Groq API 版。

功能與 src/labeling.py 相同 (同一套 prompt / schema / 輸出格式),
但改用 Groq API 雲端推論,不需要本地 GPU。

可用模型 (--model 切換,兩個都試比較品質):
  llama-3.3-70b-versatile   品質較好 (預設)
  llama-3.1-8b-instant      速度快、便宜,品質較弱

輸出 (預設依模型分開,方便比較):
  outputs/labeled_turns_groq_llama-3.3-70b-versatile.csv
  outputs/labeled_turns_groq_llama-3.3-70b-versatile_raw.jsonl
  outputs/labeled_turns_groq_llama-3.1-8b-instant.csv
  outputs/labeled_turns_groq_llama-3.1-8b-instant_raw.jsonl

用法:
  export GROQ_API_KEY=<your-key>            # 或填在下方 GROQ_API_KEY
  python src/labeling_groq.py --splits train --limit 5    # 測試: train 前 5 個 turn
  python src/labeling_groq.py                             # 全量 (70B 模型)
  python src/labeling_groq.py --model llama-3.1-8b-instant
  python src/labeling_groq.py --splits validation test

需要套件:
  pip install -U groq

中斷後重跑會自動跳過已標過的 turn (增量寫入)。
遇到 rate limit (429) 會自動指數退避重試。
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

from groq import Groq, RateLimitError, APIError

# 與本地版共用資料處理 / 解析 / 輸出邏輯
from labeling import (
    SYSTEM_PROMPT,
    SPLIT_FILES,
    MAX_NEW_TOKENS,
    build_user_message,
    extract_json,
    labels_to_rows,
    error_row,
    load_split,
    load_done_keys,
    append_rows,
    append_raw,
)

# ====================== 設定 ======================
GROQ_API_KEY = ""  # <-- 可直接填 API key;留空則讀環境變數 GROQ_API_KEY

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b"
]
DEFAULT_MODEL = MODELS[0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPERATURE = 0.2       # 標註任務要穩定,低溫
MAX_RETRIES = 5         # rate limit / 暫時性錯誤的重試次數
RETRY_BASE_SECONDS = 2  # 指數退避基數: 2, 4, 8, 16, 32 秒


def default_output(model: str) -> Path:
    safe = model.replace("/", "_")
    return PROJECT_ROOT / "outputs" / f"labeled_turns_groq_{safe}.csv"


# ====================== LLM 呼叫 ======================
def make_client() -> Groq:
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        sys.exit(
            "找不到 Groq API key。請 export GROQ_API_KEY=<key>,"
            "或在 src/labeling_groq.py 的 GROQ_API_KEY 填入。"
        )
    return Groq(api_key=api_key)


def call_llm(client: Groq, model: str, system_prompt: str, user_message: str) -> str:
    """呼叫 Groq chat completions,遇 429 / 5xx 指數退避重試。"""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                # JSON mode: 強制回傳合法 JSON (prompt 已要求 JSON,雙保險)
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError as e:
            last_err = e
            wait = RETRY_BASE_SECONDS * (2**attempt)
            print(f"  rate limit,{wait}s 後重試 ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except APIError as e:
            # 5xx 等暫時性錯誤重試;4xx (除 429) 直接丟出
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_err = e
            wait = RETRY_BASE_SECONDS * (2**attempt)
            print(f"  API 錯誤 ({e}),{wait}s 後重試 ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError(f"重試 {MAX_RETRIES} 次仍失敗: {last_err}")


# ====================== 主流程 ======================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MixAssist LLM labeling pipeline (Groq API)"
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
        help="每個 split 只處理前 N 個 turn (測試用,預設全部)",
    )
    parser.add_argument(
        "--random-window",
        action="store_true",
        help="搭配 --limit: 隨機取連續 N 筆 (如 10-14),而非固定前 N 筆",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="隨機視窗的 seed (要重現同一段時指定)",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=DEFAULT_MODEL,
        help=f"Groq 模型 (預設 {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出 CSV 路徑 (預設依模型放在 outputs/ 下)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每個 turn 之間額外 sleep 秒數 (free tier 怕撞 rate limit 可設 1~2)",
    )
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        sys.exit("SYSTEM_PROMPT 為空,請在 src/labeling.py 填入 prompt。")

    output_csv: Path = args.output or default_output(args.model)
    raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(output_csv)
    if done:
        print(f"偵測到既有輸出,已完成 {len(done)} 個 turn,將跳過。")

    client = make_client()
    print(f"使用 Groq 模型: {args.model}")
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
                raw_reply = call_llm(client, args.model, system_prompt, user_message)
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
                        "model": args.model,
                        "user_message": user_message,
                        "raw_reply": raw_reply,
                    },
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

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\n完成。輸出: {output_csv}")
    print(f"原始回覆: {raw_jsonl}")
    if failures:
        print(f"\n失敗 {len(failures)} 筆 (重跑同一指令會自動重試):")
        for key, err in failures:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
