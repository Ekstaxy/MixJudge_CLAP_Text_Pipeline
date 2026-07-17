# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a)。

對 train / validation / test 每個 turn 用 Gemma 4 31B 標註:
  problem / fix / 9 個 retrieval keys + none。

輸出 (仿 problem_fix_gold_pairs.csv 作法,單一檔案用 split 欄區分):
  outputs/labeled_turns.csv         標註結果 (一個 problem 一列,一 turn 可多列)
  outputs/labeled_turns_raw.jsonl   每個 turn 的原始 LLM 回覆 (debug 用)

本地 GPU 推論版 (V100 32GB+):
  - V100 不支援 bfloat16,一律用 float16 計算。
  - 31B 模型 fp16 約需 62GB VRAM,放不進 36GB,因此預設 4-bit NF4 量化 (~18GB)。
  - 可用 --precision 強制指定: 4bit / 8bit / fp16 / auto (預設 auto,依 VRAM 自動選)。

用法:
  python src/labeling.py --splits train --limit 5       # 測試: 只標 train 前 5 個 turn
  python src/labeling.py                                # 全量: train + validation + test
  python src/labeling.py --splits validation test       # 指定 split
  python src/labeling.py --precision 4bit               # 強制 4-bit 量化

需要套件:
  pip install -U transformers torch accelerate bitsandbytes

中斷後重跑會自動跳過已標過的 turn (增量寫入)。
"""

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

# ====================== 設定 ======================
HF_TOKEN = ""  # <-- Hugging Face token (模型下載用,已下載過可留空)
MODEL_ID = "google/gemma-4-31B-it"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILES = {
    "train": PROJECT_ROOT / "train.csv",
    "validation": PROJECT_ROOT / "validation.csv",
    "test": PROJECT_ROOT / "test.csv",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "labeled_turns.csv"

MAX_NEW_TOKENS = 1024
MAX_HISTORY_MESSAGES = 4  # 帶給 LLM 的 input_history 上限 (4 = 前兩個完整對話)

# 特殊 turn 的固定句
PLACEHOLDER_USER = "Please analyze this audio segment."
PLACEHOLDER_ASSISTANT = "I need more information before I can respond. Please elaborate."

# 9 個 retrieval keys (dim x direction),參考 intern_chores.md
VALID_DIMENSIONS = {
    "level_low",       # Stem too quiet — buried, can't hear it
    "level_high",      # Stem too loud — dominant, overpowering
    "mud",             # Too much low-mid energy — thick, cloudy, boomy
    "harshness",       # Too harsh / bright / gritty — fatiguing, piercing
    "space_wet",       # Too much reverb — washed-out, distant
    "space_dry",       # Too dry / dead — no air, no room
    "masking",         # Stem obscured in mix — can't cut through
    "dynamics_over",   # Over-processed dynamics — squashed, flat, no punch
    "dynamics_under",  # Under-controlled dynamics — inconsistent, lurching
    "none",
}
VALID_CONFIDENCE = {"low", "mid", "high"}

# ====================== Prompt ======================
# 在這裡填入你的 system prompt。
#
# code 端的解析邏輯依賴以下 JSON schema,你的 prompt 必須要求模型
# 「只輸出 JSON」且結構如下 (labels 為 list,一個 problem 一個元素;
#  一個 turn 有兩個清楚的 problem 就回傳兩個元素,即使共用同一個 fix):
#
# {
#   "labels": [
#     {
#       "has_problem": true/false,
#       "problem_text": "原文句子,沒有則空字串",
#       "problem_stem": "問題的 stem (如 drums, vocal, bass...),或 mix,沒有則空字串",
#       "problem_dimension": "9 keys 之一或 none",
#       "problem_dimension_alt": "第二可能的 dimension;確定只有一個時填 none",
#       "problem_speaker": "amateur / expert / 空字串",
#       "has_fix": true/false,
#       "fix_text": "原文句子,沒有則空字串",
#       "fix_stem": "fix 針對的 stem,沒有則空字串",
#       "fix_action": "如 lower_level, raise_level, eq_cut...,沒有則空字串",
#       "fix_speaker": "amateur / expert / 空字串",
#       "has_speaker": true/false,
#       "label_reasoning": "20-30 字內,為什麼標這個 problem_dimension",
#       "confidence": "low / mid / high"
#     }
#   ]
# }
#
# dimension 允許值:
#   level_low, level_high, mud, harshness, space_wet, space_dry,
#   masking, dynamics_over, dynamics_under, none
#
# 建議 prompt 要涵蓋的規則 (對應 user message 的組成,見 build_user_message):
#   - problem 和 fix 必須來自同一個 turn
#   - user 若是 "Please analyze this audio segment." 代表 amateur 該 turn 沒說話,
#     視為 expert 接續發言,參考 CONTEXT (input_history) 判斷有沒有順接的 problem
#   - assistant 若是 "I need more information before I can respond. Please elaborate."
#     代表 expert 沒說話,由 amateur 當次發言 + CONTEXT 判斷能否形成一筆 data
SYSTEM_PROMPT = """
You are annotating raw turns from MixAssist, a dialogue between an AMATEUR
(user) and an EXPERT (assistant) about audio mixing.
Your purpose is to build high-precision retrieval data for nine perceptual
mixing-problem categories. Be conservative: a turn is useful only when its
language supports a specific mixing problem or a specific repair action.
Do not infer a problem merely because a speaker is discussing workflow,
musical preference, an instrument, a plugin, or an unrelated production idea.
You receive:
- TOPIC: a broad session topic only; it is not proof of a stem.
- CONTEXT: earlier messages, used only to resolve references such as "it",
  "that one", or the two special placeholder cases below.
- CURRENT TURN: the messages that normally supply the problem and fix.
IMPORTANT EVIDENCE RULES
1. Extract quoted or minimally trimmed wording from the dialogue. Do not
   invent a problem, stem, fix, or action that is not supported by the text.
2. Normally, problem_text and fix_text must come from CURRENT TURN.
3. A turn may produce several labels only for clearly distinct problems.
   Return one label per distinct problem. If one fix addresses two distinct
   problems, repeat that same fix in both corresponding label objects.
4. A turn with only a fix and no explicit or context-resolvable problem may
   have has_problem=false and has_fix=true. A turn with only a problem may
   have has_problem=true and has_fix=false.
5. If no useful problem or fix exists, return exactly one empty label with
   has_problem=false, has_fix=false, and both dimensions set to "none".
6. Do not turn neutral statements such as "that sounds great", "let's choose
   a section", or a general description of workflow into a problem or fix.
SPECIAL PLACEHOLDER CASES
- If CURRENT TURN says AMATEUR: "Please analyze this audio segment." or short sentences like "okay", "yeah", the
  amateur did not contribute language in this turn. Treat the expert message
  as a continuation. Use CONTEXT only to resolve the continuing target or
  explicitly stated earlier problem. If this makes a problem recoverable,
  problem_text may quote the relevant CONTEXT wording.
- If CURRENT TURN says EXPERT:
  "I need more information before I can respond. Please elaborate." or short sentences like "okay", "yeah", the
  expert did not contribute a meaningful response. Judge the amateur message
  using CONTEXT only when needed to resolve its target.
- Outside these two exact placeholder cases, do not copy problem_text or
  fix_text from CONTEXT.
DIMENSION DEFINITIONS
Choose exactly one primary problem_dimension from:
- level_low: a source is too quiet, buried, weak, or cannot be heard.
- level_high: a source is too loud, dominant, overpowering, or excessive.
- mud: excessive low or low-mid energy; boomy, thick, cloudy, muddy.
- harshness: excessive upper-mid/high energy; harsh, bright, piercing,
  brittle, gritty, fatiguing.
- space_wet: excessive ambience, reverb, delay, room, wash, or distance.
- space_dry: insufficient ambience/room; dry, dead, no air, no space.
- masking: one source is obscured by another or cannot cut through because
  of a competing source/frequency range.
- dynamics_over: too compressed, limited, squashed, flat, over-processed,
  or lacking punch because of dynamic processing.
- dynamics_under: dynamics are uncontrolled, uneven, inconsistent, or
  lurching because compression/control is insufficient.
- none: no supported problem dimension.
DIMENSION DECISION RULES
- Prefer masking over level_low only when another source explicitly causes
  the lack of audibility or cut-through.
- Prefer level_low/level_high for a simple volume balance statement.
- Prefer dynamics_over for "flat" or "no punch" only when dynamics,
  compression, limiting, or squashing is actually implied. Otherwise use
  none rather than guessing.
- "Dry" normally means space_dry; "wet", "washed out", or too much reverb
  normally means space_wet.
- A preference is not automatically a defect. Mark it only if the speaker
  presents it as something to change or improve.
- problem_dimension_alt is only for a genuinely plausible second category
  supported by the same evidence. Otherwise set it to "none". Never use
  "none" as primary when has_problem=true unless the problem is real but
  outside all nine categories.
FIELD RULES
- problem_stem and fix_stem: use the most specific stated target, such as
  vocal, bass, kick, snare, guitar, overheads, ambience. Use
  "mix" only for a genuinely overall-mix issue. Use "" if not inferable.
- problem_speaker and fix_speaker: exactly "amateur", "expert", or "".
- has_speaker: true only if the relevant problem or fix can be attributed
  to amateur or expert; otherwise false.
- fix_action: use a concise normalized action when explicitly supported,
  e.g. lower_level, raise_level, eq_cut, eq_boost, add_reverb,
  reduce_reverb, add_compression, reduce_compression, pan, add_saturation,
  reduce_saturation. Use "" if there is no identifiable action.
- label_reasoning: maximum 25 words. State why the primary dimension follows
  from the quoted evidence; do not give generic explanations.
- confidence: high = direct explicit evidence; mid = target or category needs
  limited contextual resolution; low = ambiguous but still plausibly useful.
- Return valid JSON only. Do not use Markdown fences or any prose outside JSON.
Return exactly this schema:
{
  "labels": [
    {
      "has_problem": true,
      "problem_text": "",
      "problem_stem": "",
      "problem_dimension": "none",
      "problem_dimension_alt": "none",
      "problem_speaker": "",
      "has_fix": false,
      "fix_text": "",
      "fix_stem": "",
      "fix_action": "",
      "fix_speaker": "",
      "has_speaker": false,
      "label_reasoning": "",
      "confidence": "low"
    }
  ]
}
"""

# SYSTEM_PROMPT 留空時使用的暫代 prompt (讓 --limit 測試可以先跑起來)。
FALLBACK_SYSTEM_PROMPT = """\
You are a professional mixing engineer annotating a mixing-session dialogue corpus.

You will receive one dialogue turn between an amateur and an expert (and possibly
recent CONTEXT from earlier turns). Identify mixing PROBLEM statement(s) and FIX
statement(s) in this turn. Problem and fix must come from THIS turn (except for the
placeholder cases described below). If the turn contains two clearly distinct
problems, output two elements in "labels" (even if they share the same fix).

Placeholder rules:
- If AMATEUR said "Please analyze this audio segment." the amateur did not speak;
  treat the expert as continuing. Use CONTEXT to check for a continued problem.
- If EXPERT said "I need more information before I can respond. Please elaborate."
  the expert did not speak; judge from the amateur turn plus CONTEXT.

problem_dimension must be one of:
level_low (stem too quiet), level_high (stem too loud), mud (too much low-mid,
boomy/cloudy), harshness (too bright/gritty/piercing), space_wet (too much reverb),
space_dry (too dry/dead), masking (stem obscured, can't cut through),
dynamics_over (squashed, no punch), dynamics_under (inconsistent dynamics), none.

Output ONLY a JSON object, no other text, with this exact structure:
{"labels": [{"has_problem": bool, "problem_text": str, "problem_stem": str,
"problem_dimension": str, "problem_dimension_alt": str (second most likely
dimension, or "none" if you are sure there is only one), "problem_speaker":
"amateur"|"expert"|"", "has_fix": bool, "fix_text": str, "fix_stem": str,
"fix_action": str (e.g. lower_level, raise_level, eq_cut, add_reverb),
"fix_speaker": "amateur"|"expert"|"", "has_speaker": bool, "label_reasoning":
str (under 30 words, why this dimension), "confidence": "low"|"mid"|"high"}]}
If there is no problem and no fix, return one element with has_problem=false,
has_fix=false and problem_dimension="none".
"""

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
    "problem_dimension",
    "problem_dimension_alt",
    "problem_speaker",
    "problem_text",
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
def parse_input_history(raw: str) -> list[dict]:
    """解析 input_history 欄位 (python repr 的 list,元素間可能沒有逗號)。"""
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    for candidate in (raw, re.sub(r"\}\s*\n\s*\{", "}, {", raw)):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
        except (ValueError, SyntaxError):
            continue
    return []


def load_split(split: str, limit: int | None) -> list[dict]:
    path = SPLIT_FILES[split]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    for r in rows:
        r["split"] = split
    return rows


# ====================== Payload 組裝 ======================
def is_placeholder_user(text: str) -> bool:
    return (text or "").strip() == PLACEHOLDER_USER


def is_placeholder_assistant(text: str) -> bool:
    return (text or "").strip() == PLACEHOLDER_ASSISTANT


def build_user_message(row: dict) -> str:
    """組出給 LLM 的 user message: 當前 turn + 必要的 context + 特殊 turn 提示。"""
    user_text = (row.get("user") or "").strip()
    assistant_text = (row.get("assistant") or "").strip()

    parts = [
        f"TOPIC: {row.get('topic', '')}",
        f"TURN_ID: {row.get('turn_id', '')}",
    ]

    history = parse_input_history(row.get("input_history", ""))
    if history:
        recent = history[-MAX_HISTORY_MESSAGES:]
        lines = []
        for msg in recent:
            role = "AMATEUR" if msg.get("role") == "user" else "EXPERT"
            lines.append(f"  {role}: {msg.get('content', '')}")
        parts.append("CONTEXT (earlier turns, for reference only):\n" + "\n".join(lines))

    parts.append("CURRENT TURN:")
    parts.append(f"  AMATEUR: {user_text}")
    parts.append(f"  EXPERT: {assistant_text}")

    if is_placeholder_user(user_text):
        parts.append(
            "NOTE: The amateur message is a placeholder — the amateur did not speak "
            "in this turn. Treat the expert as continuing from CONTEXT."
        )
    if is_placeholder_assistant(assistant_text):
        parts.append(
            "NOTE: The expert message is a placeholder — the expert did not speak "
            "in this turn. Judge from the amateur message plus CONTEXT."
        )

    return "\n".join(parts)


# ====================== LLM 呼叫與解析 ======================
def extract_json(text: str) -> dict:
    """從模型回覆中抽出 JSON (容錯: code fence 或多餘文字)。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return json.loads(brace.group(0))
        raise


def pick_precision(requested: str) -> str:
    """依 VRAM 決定精度。V100 36GB: 31B fp16 放不下,預設 4bit。"""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        sys.exit("找不到 CUDA GPU。請確認 torch 有裝 CUDA 版且 GPU 可用。")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 31B 參數: fp16 ~62GB, 8bit ~31GB, 4bit ~18GB (皆不含 KV cache)
    if total_gb >= 70:
        return "fp16"
    if total_gb >= 40:
        return "8bit"
    return "4bit"


def load_model(precision: str):
    """載入本地模型。V100 (compute 7.0) 不支援 bf16,計算一律 float16。"""
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

    print(f"載入模型 {MODEL_ID} (precision={precision}),第一次會下載權重,請稍候...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print("模型載入完成。")
    return processor, model


@torch.inference_mode()
def call_llm(processor, model, system_prompt: str, user_message: str) -> str:
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
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

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
    return v if v in VALID_DIMENSIONS else "none"


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
        r.update(
            {
                "has_problem": normalize_bool(label.get("has_problem")),
                "problem_stem": str(label.get("problem_stem", "") or ""),
                "problem_dimension": normalize_dimension(label.get("problem_dimension")),
                "problem_dimension_alt": normalize_dimension(label.get("problem_dimension_alt")),
                "problem_speaker": str(label.get("problem_speaker", "") or ""),
                "problem_text": str(label.get("problem_text", "") or ""),
                "has_fix": normalize_bool(label.get("has_fix")),
                "fix_text": str(label.get("fix_text", "") or ""),
                "fix_stem": str(label.get("fix_stem", "") or ""),
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
    r["problem_dimension"] = ""
    r["problem_dimension_alt"] = ""
    return r


# ====================== 增量寫入 ======================
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
    with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_raw(raw_jsonl: Path, record: dict) -> None:
    with open(raw_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ====================== 主流程 ======================
def main() -> None:
    parser = argparse.ArgumentParser(description="MixAssist LLM labeling pipeline")
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"輸出 CSV 路徑 (預設 {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "4bit", "8bit", "fp16"],
        default="auto",
        help="模型精度 (預設 auto: 依 VRAM 自動選,36GB V100 會選 4bit)",
    )
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        print("[警告] SYSTEM_PROMPT 未填,使用 FALLBACK_SYSTEM_PROMPT 暫代。\n")
        system_prompt = FALLBACK_SYSTEM_PROMPT

    output_csv: Path = args.output
    raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(output_csv)
    if done:
        print(f"偵測到既有輸出,已完成 {len(done)} 個 turn,將跳過。")

    precision = pick_precision(args.precision)
    processor, model = load_model(precision)
    failures = []

    for split in args.splits:
        rows = load_split(split, args.limit)
        print(f"=== {split}: {len(rows)} turns ===")

        for i, row in enumerate(rows, 1):
            key = (split, row.get("conversation_id", ""), row.get("turn_id", ""))
            tag = f"[{split} {i}/{len(rows)}] {key[1]} turn {key[2]}"
            if key in done:
                print(f"{tag} 已標過,跳過")
                continue

            user_message = build_user_message(row)
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
                dims = [r["problem_dimension"] for r in out_rows]
                print(f"{tag} -> {len(out_rows)} label(s): {dims}")
            except Exception as e:
                print(f"{tag} -> 錯誤: {e}")
                failures.append((key, str(e)))
                append_rows(output_csv, [error_row(row, str(e))])
                append_raw(raw_jsonl, {"key": list(key), "error": str(e)})

    print(f"\n完成。輸出: {output_csv}")
    print(f"原始回覆: {raw_jsonl}")
    if failures:
        print(f"\n失敗 {len(failures)} 筆 (重跑同一指令會自動重試):")
        for key, err in failures:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
