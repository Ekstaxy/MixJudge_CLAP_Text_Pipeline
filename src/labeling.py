# -*- coding: utf-8 -*-
"""
MixAssist 標記 pipeline (Phase 2a)。

對 train / validation / test 每個 turn 用 Gemma 4 31B 標註:
  problem / fix / 15 dimensions (8 axes x 2 方向,phase 除外) + none。

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
import random
import re
import sys
from pathlib import Path

# torch / transformers 延遲載入 (在 pick_precision / load_model 內 import),
# 讓 labeling_groq.py 等 API 版可以 import 本檔的共用函式而不需要 GPU 套件。

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
MAX_HISTORY_MESSAGES = 8  # 帶給 LLM 的 input_history 上限 (4 = 前兩個完整對話)

# 特殊 turn 的固定句
PLACEHOLDER_USER = "Please analyze this audio segment."
PLACEHOLDER_ASSISTANT = "I need more information before I can respond. Please elaborate."

# 8 axes x 2 方向 (phase 不分方向) = 15 dimensions + none
# axis 欄位由 dimension 反查 (DIMENSION_TO_AXIS),LLM 只需要標 dimension
DIMENSION_TO_AXIS = {
    "level_low": "level",         # Stem too quiet — buried, can't hear it
    "level_high": "level",        # Stem too loud — dominant, overpowering
    "mud_high": "mud",            # Too much low-mid — thick, cloudy, boomy
    "mud_low": "mud",             # Too little low-mid — thin, hollow, no body
    "harsh_high": "harshness",    # Too harsh/bright — piercing, fatiguing
    "harsh_low": "harshness",     # Too dull/dark — no presence, lifeless top
    "space_wet": "space",         # Too much reverb — washed-out, distant
    "space_dry": "space",         # Too dry/dead — no air, no room
    "masking_high": "masking",    # Stem obscured by another — can't cut through
    "masking_low": "masking",     # Overly separated — doesn't blend/glue
    "stereo_wide": "stereo",      # Too wide — unfocused, hollow center
    "stereo_narrow": "stereo",    # Too narrow/mono — no width, crowded center
    "dynamics_over": "dynamics",  # Over-processed — squashed, flat, no punch
    "dynamics_under": "dynamics", # Under-controlled — inconsistent, lurching
    "phase": "phase",             # Phase problem — comb filtering, cancellation
}
VALID_DIMENSIONS = set(DIMENSION_TO_AXIS) | {"none"}

# 舊 9-key 名稱的相容對應 (mud / harshness / masking 未分方向的舊標)
DIMENSION_ALIASES = {
    "mud": "mud_high",
    "harshness": "harsh_high",
    "masking": "masking_high",
}
VALID_CONFIDENCE = {"low", "mid", "high"}

# problem_stem / fix_stem 只允許固定值 (樂器 + mix),其他一律 normalize 成 other
VALID_STEMS = {
    "vocal", "guitar", "bass", "drums", "kick", "snare", "hats", "toms",
    "overheads", "cymbals", "percussion", "keys", "piano", "synth",
    "strings", "horns", "mix", "",
}
STEM_ALIASES = {
    "vocals": "vocal", "vox": "vocal", "lead vocal": "vocal",
    "guitars": "guitar", "electric guitar": "guitar", "acoustic guitar": "guitar",
    "drum": "drums", "tom": "toms", "hat": "hats", "hi-hat": "hats",
    "hihat": "hats", "hi-hats": "hats", "overhead": "overheads",
    "oh": "overheads", "o.h.": "overheads",
    "cymbal": "cymbals", "keyboard": "keys", "full track": "mix",
    "overall mix": "mix", "master": "mix", "song": "mix", "track": "mix",
    "room": "ambience", "room mic": "ambience", "rooms": "ambience",
    "amb": "ambience", "ambience track": "ambience", "amb tracks": "ambience",
}

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
#       "problem_stem": "問題的 stem (固定清單,見 VALID_STEMS),沒有則空字串",
#       "problem_dimension": "15 dimensions 之一或 none (axis 由 code 反查)",
#       "problem_speaker": "amateur / expert / 空字串",
#       "vocal_lead": true/false (有討論到 vocal 就 true),
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
# dimension 允許值 (8 axes x 2 方向,phase 只有一個):
#   level_low, level_high, mud_high, mud_low, harsh_high, harsh_low,
#   space_wet, space_dry, masking_high, masking_low, stereo_wide,
#   stereo_narrow, dynamics_over, dynamics_under, phase, none
#
# 建議 prompt 要涵蓋的規則 (對應 user message 的組成,見 build_user_message):
#   - problem 和 fix 必須來自同一個 turn
#   - user 若是 "Please analyze this audio segment." 代表 amateur 該 turn 沒說話,
#     視為 expert 接續發言;INPUT HISTORY 輔助判斷 stem/axis/dimension,
#     僅在當次 turn 沒有可用 problem_text 時才從 HISTORY 取 problem_text
#   - assistant 若是 "I need more information before I can respond. Please elaborate."
#     代表 expert 沒說話,由 amateur 當次發言 + INPUT HISTORY 判斷能否形成一筆 data
SYSTEM_PROMPT = """
You annotate turns from MixAssist, a mixing-session dialogue between an
AMATEUR (user) and an EXPERT (assistant). Goal: high-precision labels of
mixing problems and fixes. Be conservative — label a problem only when the
words clearly support it. Workflow talk, musical preference, or production
ideas are not problems.
You receive:
- TOPIC: broad session topic; not proof of a stem.
- INPUT HISTORY: earlier turns (readable, auxiliary).
- CURRENT TURN: primary source for problem_text and fix_text.
EVIDENCE RULES
1. Prefer quotes (minimally trimmed) from CURRENT TURN. Never invent wording.
2. One label per distinct problem (different problem_text spans). Pick
   exactly ONE problem_dimension — do not invent a second/alternate
   dimension. If one fix addresses two distinct problems, repeat the fix
   in both labels.
3. has_problem and has_fix are independent: a fix-only turn is
   has_problem=false, has_fix=true, and vice versa. If neither exists,
   return exactly one empty label (has_problem=false, has_fix=false,
   problem_dimension "none").
4. Neutral statements ("that sounds great", "let's pick a section") are
   neither problems nor fixes.
INPUT HISTORY (auxiliary)
- CURRENT TURN is primary. HISTORY helps judge stem, axis, and
  problem_dimension when the current wording is vague (pronouns, unfinished
  thought, continuing the same mix decision).
- Incomplete turns are common. Vague intents like "I wanna adjust that",
  "pull it back", or "where is that coming from" are NOT problem statements
  by themselves — they are usually continuing an earlier problem. In that
  case: keep labeling THIS turn's action as the fix, but recover the
  perceptual problem (what is wrong / what is missing) from HISTORY.
- problem_text: use CURRENT TURN first. If CURRENT TURN has no usable
  problem-state raw text (placeholder, filler, or only a vague adjust/
  locate intent with no stated defect) AND HISTORY clearly continues the
  same issue, take problem_text from HISTORY. Do not re-report an old
  HISTORY problem the current turn is not working on.
- fix_text: always from CURRENT TURN when present. Do not copy fix_text
  from HISTORY.
- Placeholders: AMATEUR "Please analyze this audio segment." = amateur did
  not speak; EXPERT "I need more information before I can respond. Please
  elaborate." = expert did not speak. Same for filler-only messages.
AFFECTED STEM vs ACTION TARGET (critical)
- problem_stem = the instrument that is suffering / the perceptual problem
  is about (what you cannot hear, what sounds wrong).
- fix_stem = the instrument you operate on (fader, EQ, mute, etc.).
- These often DIFFER. Example: HISTORY says snare is lost; CURRENT says
  the cymbal is filling highs and "pull that back" → problem_stem=snare,
  problem_dimension=masking_high, fix_stem=cymbals, fix_action=lower_level
  or eq_cut. Do NOT set problem_stem=cymbals just because that is what
  you turn down — the cymbal is the cause/masker, not the problem stem.
- For masking_high: problem_stem is always the obscured source. Name the
  competing source in label_reasoning (and usually as fix_stem if that is
  what gets adjusted).
- If CURRENT only names the cause ("that cymbal") but HISTORY already
  established the affected source (snare lost), keep problem_stem as the
  affected source from HISTORY.
DIMENSIONS (8 axes x 2 directions; phase has 1)
- level_low: too quiet, buried, weak. | level_high: too loud, overpowering.
- mud_high: too much low/low-mid; boomy, thick, muddy. | mud_low: thin,
  hollow, no body, lacks low end.
- harsh_high: too much upper-mid/high; harsh, piercing, gritty. |
  harsh_low: dull, dark, muffled, no presence.
- space_wet: too much reverb/ambience/room; washed out, distant. |
  space_dry: too dry, dead, no air (includes too-quiet drum room/amb).
- masking_high: obscured by a competing source, can't cut through. |
  masking_low: overly separated, doesn't blend or glue.
- stereo_wide: too wide, hollow center. | stereo_narrow: too narrow/mono,
  crowded center.
- dynamics_over: squashed, flat, no punch from over-processing. |
  dynamics_under: uncontrolled, uneven, lurching.
- phase: cancellation, comb filtering, polarity issues.
- none: no supported problem.
DECISION RULES
- POLARITY CHECK (mandatory): the dimension direction must match the
  complaint. "too dry / needs ambience" = space_dry, NEVER space_wet;
  "too wet / washed out" = space_wet; "too quiet" = level_low; "too loud"
  = level_high. Verify problem_text and dimension point the same way.
- masking_high over level_low only when another source explicitly causes
  the inaudibility (or HISTORY already established that causal link).
  Simple volume complaints with no competing source are level_low/level_high.
- DRUM ROOM vs OVERHEADS (mandatory):
  - Drum room / room mic / ambience / amb tracks: level or send amount of
    the room is SPACE (space_wet if too much room, space_dry if too little).
    Prefer stem "ambience" (or "other" only if clearly not ambience).
  - Overheads: treat as a drum instrument (like snare, hats, toms). Volume
    of overheads is LEVEL (level_high / level_low), NOT space. Stem
    "overheads".
- "flat"/"no punch" is dynamics_over only when compression/limiting is
  implied; otherwise none.
- Arrangement/composition comments (how many parts play, what would be a
  cool effect) and pure preferences are none, unless presented as something
  to change.
FIELD RULES
- problem_stem / fix_stem MUST be one of: vocal, guitar, bass, drums, kick,
  snare, hats, toms, overheads, cymbals, percussion, keys, piano, synth,
  strings, horns, ambience, mix, other, "". Most specific instrument;
  "mix" only for overall-mix issues; "other" for a stated target not
  listed; "" if unresolvable. Never invent values like "dry signal" or
  "low end". Resolve pronouns / "that" from HISTORY. Remember:
  problem_stem = affected; fix_stem = action target (may differ).
- problem_speaker / fix_speaker: "amateur", "expert", or "".
- vocal_lead: true if the label's problem or fix involves the vocal at all.
- has_speaker: true only if the problem or fix is attributable to a speaker.
- fix_action: concise normalized action when explicit (lower_level,
  raise_level, eq_cut, eq_boost, add_reverb, reduce_reverb, add_compression,
  reduce_compression, pan, add_saturation, reduce_saturation); else "".
  Prefer lower_level / eq_cut over vague "adjust".
- label_reasoning: max 25 words. State affected stem, dimension, and if
  relevant the cause/masker (e.g. "snare lost; cymbal masking highs").
- confidence: high = explicit direct evidence; mid = needed HISTORY to
  recover affected stem or continued problem; low = ambiguous but useful.
- Return valid JSON only. No Markdown fences, no prose outside JSON.
Return exactly this schema:
{
  "labels": [
    {
      "has_problem": true,
      "problem_text": "",
      "problem_stem": "",
      "problem_dimension": "none",
      "problem_speaker": "",
      "vocal_lead": false,
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


# ====================== Payload 組裝 ======================
def is_placeholder_user(text: str) -> bool:
    return (text or "").strip() == PLACEHOLDER_USER


def is_placeholder_assistant(text: str) -> bool:
    return (text or "").strip() == PLACEHOLDER_ASSISTANT


def format_history_readable(history: list[dict]) -> str:
    """把 input_history 排成可讀的 earlier-turn 區塊 (amateur/expert 成對)。"""
    if not history:
        return ""
    # 成對: user 後面接 assistant 算一 turn;孤立訊息各成一塊
    blocks: list[list[dict]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if msg.get("role") == "user" and i + 1 < len(history) and history[i + 1].get("role") != "user":
            blocks.append([msg, history[i + 1]])
            i += 2
        else:
            blocks.append([msg])
            i += 1

    # 只保留最近幾輪 (以 turn 數計,不是訊息數)
    max_turns = max(1, MAX_HISTORY_MESSAGES // 2)
    blocks = blocks[-max_turns:]

    lines = [
        "INPUT HISTORY (earlier turns — auxiliary for stem / axis / dimension;",
        "and for problem_text ONLY if CURRENT TURN has no usable problem statement.",
        "If CURRENT only says 'adjust/pull that', recover the continued defect from HISTORY;",
        "problem_stem = affected instrument; fix_stem = what you operate on — often different):",
    ]
    for t, block in enumerate(blocks, 1):
        lines.append(f"--- earlier turn {t} ---")
        for msg in block:
            role = "AMATEUR" if msg.get("role") == "user" else "EXPERT"
            content = (msg.get("content") or "").strip()
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_user_message(row: dict) -> str:
    """組出給 LLM 的 user message: 當前 turn + 可讀 history + 特殊 turn 提示。"""
    user_text = (row.get("user") or "").strip()
    assistant_text = (row.get("assistant") or "").strip()

    parts = [
        f"TOPIC: {row.get('topic', '')}",
        f"TURN_ID: {row.get('turn_id', '')}",
    ]

    history = parse_input_history(row.get("input_history", ""))
    history_block = format_history_readable(history)
    if history_block:
        parts.append(history_block)

    parts.append("CURRENT TURN (primary):")
    parts.append(f"AMATEUR: {user_text}")
    parts.append(f"EXPERT: {assistant_text}")

    if is_placeholder_user(user_text):
        parts.append(
            "NOTE: Amateur message is a placeholder — they did not speak this turn. "
            "Treat EXPERT as continuing. Prefer CURRENT TURN for fix_text; "
            "use INPUT HISTORY for problem_text only if CURRENT TURN has none."
        )
    if is_placeholder_assistant(assistant_text):
        parts.append(
            "NOTE: Expert message is a placeholder — they did not speak this turn. "
            "Judge from AMATEUR plus INPUT HISTORY for stem/axis/dimension."
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
    import torch

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

    print(f"載入模型 {MODEL_ID} (precision={precision}),第一次會下載權重,請稍候...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print("模型載入完成。")
    return processor, model


def call_llm(processor, model, system_prompt: str, user_message: str) -> str:
    import torch

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
        sys.exit("SYSTEM_PROMPT 為空,請在 src/labeling.py 填入 prompt。")

    output_csv: Path = args.output
    raw_jsonl = output_csv.with_name(output_csv.stem + "_raw.jsonl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
