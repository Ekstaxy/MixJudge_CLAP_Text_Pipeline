# -*- coding: utf-8 -*-
"""Shared MixAssist labeling: schema, I/O, parse, normalize.

Used by labeling_hf.py, labeling_gguf.py, labeling_groq.py.
Prompt text lives in prompts/labeling_prompt.py.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

# src/ = parent of this package; repo root = parent of src/
_SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from prompts.labeling_prompt import SYSTEM_PROMPT, build_user_message

SPLIT_FILES = {
    "train": PROJECT_ROOT / "train.csv",
    "validation": PROJECT_ROOT / "validation.csv",
    "test": PROJECT_ROOT / "test.csv",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "labeled_turns.csv"
MAX_NEW_TOKENS = 1024

# MixJudge caption ontology (LEXICON_BRIEF): 7 axes, 12 classes
# = 11 problem dims + clean. LLM 只標 problem_dimension; problem_axis 由 code 反查.
DIMENSION_TO_AXIS = {
    "too_quiet": "level",
    "too_loud": "level",
    "muddy": "body",
    "thin": "body",
    "harsh": "brightness",
    "dull": "brightness",
    "too_wet": "space",
    "too_dry": "space",
    "over_compressed": "dynamic",
    "under_compressed": "dynamic",
    "masking": "masking",
    "clean": "clean",
}

# Old MixAssist labels used a two-way masking split. Map them onto the
# current single dim so an un-remapped pool still has masking exemplars.
DIMENSION_ALIASES = {
    "swamping": "masking",
    "invading": "masking",
}

VALID_DIMENSIONS = set(DIMENSION_TO_AXIS) | {"none"}
VALID_CONFIDENCE = {"low", "mid", "high"}

# problem_stem / fix_stem 只允許固定值 (樂器 + mix),其他一律 normalize 成 other
VALID_STEMS = {
    "vocal", "guitar", "bass", "drums", "kick", "snare", "hats", "toms",
    "overheads", "cymbals", "percussion", "keys", "piano", "synth",
    "strings", "horns", "ambience", "mix", "",
}
STEM_ALIASES = {
    "vocals": "vocal", "vox": "vocal", "lead vocal": "vocal",
    "lead vocals": "vocal", "main vocal": "vocal",
    "backing vocal": "vocal", "backing vocals": "vocal",
    "bgv": "vocal", "bgvs": "vocal", "harmony": "vocal",
    "harmonies": "vocal", "doubles": "vocal", "double": "vocal",
    "guitars": "guitar", "electric guitar": "guitar", "acoustic guitar": "guitar",
    "drum": "drums", "tom": "toms", "hat": "hats", "hi-hat": "hats",
    "hihat": "hats", "hi-hats": "hats", "overhead": "overheads",
    "oh": "overheads", "o.h.": "overheads",
    "cymbal": "cymbals", "keyboard": "keys", "full track": "mix",
    "overall mix": "mix", "master": "mix", "song": "mix", "track": "mix",
    "room": "ambience", "room mic": "ambience", "rooms": "ambience",
    "amb": "ambience", "ambience track": "ambience", "amb tracks": "ambience",
}

# ====================== 輸出欄位 ======================
OUTPUT_FIELDS = [
    "conversation_id",
    "split",
    "topic",
    "turn_id",
    "has_content",
    "audio_file",
    "user_raw_content",
    "assistant_raw_content",
    "has_problem",
    "problem_stem",
    "problem_axis",
    "problem_dimension",
    "problem_speaker",
    "problem_text",
    "vocal_lead",
    "has_fix",
    "fix_text",
    "fix_stem",
    "fix_action",
    "fix_speaker",
    "has_speaker",
    "label_reasoning",
    "confidence",
    "error",
]


# ====================== 資料讀取 ======================
def load_split(split: str, limit: int | None, random_window: bool = False) -> list[dict]:
    """讀取 split。limit + random_window=True 時隨機取「連續」limit 筆
    (例如 10-14 或 100-104),方便抽查不同段落的標註品質。"""
    path = SPLIT_FILES[split]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None and limit < len(rows):
        start = random.randint(0, len(rows) - limit) if random_window else 0
        rows = rows[start : start + limit]
        if random_window:
            print(f"[{split}] 隨機視窗: 第 {start}~{start + limit - 1} 列 (0-based)")
    for r in rows:
        r["split"] = split
    return rows


# ====================== LLM 呼叫與解析 ======================
def extract_json(text: str) -> dict:
    """從模型回覆中抽出 JSON (容錯: code fence、多餘文字、常見語法瑕疵、截斷)。"""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    last_err: Exception | None = None
    for cand in candidates:
        for variant in (cand, _repair_json_text(cand)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as e:
                last_err = e

    # Truncated multi-label JSON: keep complete label objects.
    salvaged = _salvage_labels(text)
    if salvaged:
        return {"labels": salvaged}

    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("Expecting value", text, 0)


def _repair_json_text(text: str) -> str:
    """修常見 LLM JSON 瑕疵: 尾逗號、單引號 key/字串。"""
    # trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 'key': → "key":  (粗修,只處理簡單情況)
    text = re.sub(r"'([^'\\]*)'\s*:", r'"\1":', text)
    return text


def _salvage_labels(text: str) -> list[dict]:
    """從截斷的 labels JSON 中救回已完整的 label 物件。"""
    out: list[dict] = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (
            "problem_dimension" in obj or "has_problem" in obj or "has_fix" in obj
        ):
            out.append(obj)
    return out


def pick_precision(requested: str) -> str:
    """依 VRAM 決定精度。V100 32GB: 31B fp16 放不下,預設 4bit。"""
    import torch

    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        sys.exit("找不到 CUDA GPU。請確認 torch 有裝 CUDA 版 (cu126) 且 GPU 可用。")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 31B: fp16 ~62GB, 8bit ~31GB, 4bit ~18GB (不含 KV cache)
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
            temperature=1.0,  # Gemma 4 官方建議
            top_p=0.95,
            top_k=64,
        )
    reply = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    # 去掉 Gemma 4 的空 thought 區塊 (<|channel>thought ... <channel|>)
    reply = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", reply, flags=re.DOTALL)
    return reply.strip()


def normalize_dimension(value) -> str:
    v = str(value or "none").strip().lower()
    v = DIMENSION_ALIASES.get(v, v)
    return v if v in VALID_DIMENSIONS else "none"


def dimension_axis(dimension: str) -> str:
    return DIMENSION_TO_AXIS.get(dimension, "none")


def normalize_stem(value) -> str:
    v = str(value or "").strip().lower()
    v = STEM_ALIASES.get(v, v)
    return v if v in VALID_STEMS else "other"


def normalize_confidence(value) -> str:
    v = str(value or "").strip().lower()
    aliases = {"medium": "mid", "middle": "mid"}
    v = aliases.get(v, v)
    return v if v in VALID_CONFIDENCE else "low"


def normalize_bool(value) -> str:
    if isinstance(value, bool):
        return str(value)
    v = str(value or "").strip().lower()
    return str(v in ("true", "yes", "1"))


def labels_to_rows(row: dict, labels: list[dict]) -> list[dict]:
    """把 LLM 回傳的 labels list 轉成輸出列 (一個 label 一列)。"""
    base = {
        "conversation_id": row.get("conversation_id", ""),
        "split": row.get("split", ""),
        "topic": row.get("topic", ""),
        "turn_id": row.get("turn_id", ""),
        "has_content": row.get("has_content", ""),
        "audio_file": row.get("audio_file", ""),
        "user_raw_content": row.get("user", ""),
        "assistant_raw_content": row.get("assistant", ""),
        "error": "",
    }
    if not labels:
        labels = [{}]

    out = []
    for label in labels:
        if not isinstance(label, dict):
            label = {}
        r = dict(base)
        dimension = normalize_dimension(label.get("problem_dimension"))
        r.update(
            {
                "has_problem": normalize_bool(label.get("has_problem")),
                "problem_stem": normalize_stem(label.get("problem_stem")),
                "problem_axis": dimension_axis(dimension),
                "problem_dimension": dimension,
                "vocal_lead": normalize_bool(label.get("vocal_lead")),
                "problem_speaker": str(label.get("problem_speaker", "") or ""),
                "problem_text": str(label.get("problem_text", "") or ""),
                "has_fix": normalize_bool(label.get("has_fix")),
                "fix_text": str(label.get("fix_text", "") or ""),
                "fix_stem": normalize_stem(label.get("fix_stem")),
                "fix_action": str(label.get("fix_action", "") or ""),
                "fix_speaker": str(label.get("fix_speaker", "") or ""),
                "has_speaker": normalize_bool(label.get("has_speaker")),
                "label_reasoning": str(label.get("label_reasoning", "") or ""),
                "confidence": normalize_confidence(label.get("confidence")),
            }
        )
        out.append(r)
    return out


def error_row(row: dict, error: str) -> dict:
    r = labels_to_rows(row, [{}])[0]
    r["error"] = error
    r["confidence"] = ""
    r["problem_axis"] = ""
    r["problem_dimension"] = ""
    return r


# ====================== 增量寫入 ======================
def purge_error_rows(output_csv: Path) -> int:
    """刪除 CSV 裡帶 error 的列,讓失敗 turn 可干净重試且不留下 stale error。

    Returns:
        刪除的列數。
    """
    if not output_csv.exists():
        return 0
    with open(output_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if not str(r.get("error") or "").strip()]
    removed = len(rows) - len(keep)
    if removed == 0:
        return 0
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(keep)
    return removed


def load_done_keys(output_csv: Path) -> set[tuple]:
    """讀取已輸出的 (split, conversation_id, turn_id),供中斷重跑時跳過。"""
    if not output_csv.exists():
        return set()
    with open(output_csv, encoding="utf-8-sig") as f:
        return {
            (r.get("split", ""), r.get("conversation_id", ""), r.get("turn_id", ""))
            for r in csv.DictReader(f)
            if not r.get("error")  # 之前失敗的重跑
        }


def append_rows(output_csv: Path, rows: list[dict]) -> None:
    write_header = not output_csv.exists()
    if not write_header:
        with open(output_csv, encoding="utf-8-sig") as f:
            existing = next(csv.reader(f), [])
        if existing != OUTPUT_FIELDS:
            sys.exit(
                f"既有輸出 {output_csv} 的欄位與目前 schema 不符 "
                "(可能是舊版 9-key 輸出)。請先改名或刪除舊檔再重跑。"
            )
    with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_raw(raw_jsonl: Path, record: dict) -> None:
    with open(raw_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


