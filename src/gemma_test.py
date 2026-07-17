# -*- coding: utf-8 -*-
"""
Gemma 4 31B (google/gemma-4-31B-it) 標註測試腳本。

功能:
  1. 透過 Hugging Face Inference API 傳送 prompt
  2. 要求模型輸出 JSON 格式的混音問題標註
  3. 將結果同時存成 JSON 與 CSV,方便之後擴充成批次標註 pipeline

使用前:
  pip install -U huggingface_hub
  在下方 HF_TOKEN 填入你的 token (hf_xxx...)
"""

import csv
import json
import re
import time
from pathlib import Path

from huggingface_hub import InferenceClient

# ====================== 設定 ======================
HF_TOKEN = ""  # <-- 在這裡填入你的 Hugging Face token
MODEL_ID = "google/gemma-4-31B-it"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_JSON = OUTPUT_DIR / "gemma_test_results.json"
OUTPUT_CSV = OUTPUT_DIR / "gemma_test_results.csv"

SYSTEM_PROMPT = (
    "你是一位專業的混音工程師,負責標註文本。"
    "請分析文本中的混音問題,並以 JSON 格式輸出,包含以下欄位:"
    "'issue_type' (如 EQ, 動態, 空間系), 'severity' (High, Medium, Low), "
    "'explanation' (一句話說明)。只能輸出 JSON,不要有其他文字。"
)

# 測試資料,之後可改成從 CSV 讀入
DATA_TO_LABEL = [
    "這個 Vocal 的 Reverb tail 太長,會吃掉底下的 Bass 頻段",
    "吉他 Tone 太乾了,需要加一點 Chorus 和 Delay",
    "整體 LUFS 已經推到 -8,但聽起來還是不夠 punchy",
]
# ==================================================


def extract_json(text: str) -> dict:
    """從模型回覆中抽出 JSON(容錯:模型有時會包 ```json ... ``` 或多餘文字)。"""
    text = text.strip()
    # 去掉 markdown code fence
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 嘗試抓第一個 {...} 區塊
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return json.loads(brace.group(0))
        raise


def label_text(client: InferenceClient, text: str) -> dict:
    """傳一筆文本給 Gemma,回傳解析後的標註 dict。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    response = client.chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=1.0,  # Gemma 4 官方建議的 sampling 設定
        top_p=0.95,
    )
    raw = response.choices[0].message.content
    return extract_json(raw)


def save_results(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV:固定欄位 + 保留原始 label JSON 字串,方便之後擴充欄位
    fieldnames = ["text", "issue_type", "severity", "explanation", "raw_label_json"]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            label = r.get("label", {})
            writer.writerow(
                {
                    "text": r["text"],
                    "issue_type": label.get("issue_type", ""),
                    "severity": label.get("severity", ""),
                    "explanation": label.get("explanation", ""),
                    "raw_label_json": json.dumps(label, ensure_ascii=False),
                }
            )

    print(f"已寫入 JSON: {OUTPUT_JSON}")
    print(f"已寫入 CSV:  {OUTPUT_CSV}")


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("請先在 HF_TOKEN 填入你的 Hugging Face token!")

    client = InferenceClient(model=MODEL_ID, api_key=HF_TOKEN)
    results = []

    for i, text in enumerate(DATA_TO_LABEL, 1):
        print(f"[{i}/{len(DATA_TO_LABEL)}] 標註中: {text}")
        try:
            label = label_text(client, text)
            results.append({"text": text, "label": label})
            print(f"  -> {json.dumps(label, ensure_ascii=False)}")
            time.sleep(2)  # 免費 API 有 rate limit
        except Exception as e:
            print(f"  -> 發生錯誤: {e}")
            results.append({"text": text, "label": {}, "error": str(e)})
            time.sleep(10)

    save_results(results)
    print("全部標註完成!")


if __name__ == "__main__":
    main()
