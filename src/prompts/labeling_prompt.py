# -*- coding: utf-8 -*-
"""MixAssist labeling prompts: system text + user-message assembly."""

from __future__ import annotations

import ast
import re

# History window for INPUT HISTORY in the user message (message count; //2 = turns).
MAX_HISTORY_MESSAGES = 8

PLACEHOLDER_USER = "Please analyze this audio segment."
PLACEHOLDER_ASSISTANT = "I need more information before I can respond. Please elaborate."

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
#       "problem_dimension": "12 classes 之一或 none (axis 由 code 反查)",
#       "problem_speaker": "amateur / expert / 空字串",
#       "vocal_lead": true/false (僅 lead vocal 為 true; backing vocals 為 false),
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
# 7 axes (problem_axis 由 code 反查, LLM 只填 problem_dimension):
#   level, body, brightness, space, dynamic, masking, clean
#
# problem_dimension 允許值 (11 problem + clean + none):
#   too_quiet, too_loud, muddy, thin, harsh, dull, too_wet, too_dry,
#   over_compressed, under_compressed, masking, clean, none
#
# 禁止把 dimension 名稱當 axis (例如 harsh/muddy 是 dimension, 不是 axis)。
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
4. Vague approval ("that sounds great", "this is fine", "perfect") with no
   concrete sonic claim → none (neither problem nor clean). Explicit
   clean/balanced/fault-free sonic state → clean (has_problem=true).
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
  is about (what you cannot hear, what sounds wrong). For clean: the source
  described as sitting cleanly / balanced.
- fix_stem = the instrument you operate on (fader, EQ, mute, etc.).
- They can differ. Example: HISTORY says snare is lost; CURRENT says
  the cymbal is filling highs and "pull that back" → problem_stem=snare,
  problem_dimension=masking, fix_stem=cymbals, fix_action=lower_level
  or eq_cut. Do NOT set problem_stem=cymbals just because that is what
  you turn down — the cymbal is the cause/masker, not the problem stem.
- For masking: problem_stem is the victim (the source being covered up /
  unable to cut through). Name the aggressor/masker in label_reasoning
  (and usually as fix_stem if that is what gets adjusted).
- If CURRENT only names the cause ("that cymbal") but HISTORY already
  established the affected source (snare lost), keep problem_stem as the
  affected source from HISTORY.
FIGURE vs BAND (vocal_lead)
- MixAssist often talks about a foreground source (often lead vocal) versus
  the backing bed / band, but this is a heuristic, not a hard schema rule.
- vocal_lead: true ONLY if THIS label involves the LEAD / main vocal
  (problem or fix about the lead vocal's level, tone, space, etc.).
- vocal_lead: false for backing vocals, doubles, harmonies, ad-libs, and
  any non-lead vocal layer — even if problem_stem is still "vocal".
- Use HISTORY to tell lead vs backing when CURRENT is vague ("it", "that",
  "wanted it more in the background"). If HISTORY established backing
  vocals and CURRENT continues that thread, keep vocal_lead=false.
- vocal_lead is a separate boolean only. Do NOT use it to override the
  chosen axis or problem_dimension.
- Do not hard-code dimensions to vocal-only or band-only. A dimension
  should be chosen from the wording of the complaint, not from whether
  the source is vocal or accompaniment.
AXES AND DIMENSIONS (CRITICAL — use EXACT strings)
There are exactly 7 axes and 12 classes (11 problem + clean). You output
problem_dimension ONLY; code maps to problem_axis. NEVER output an axis
name as problem_dimension. Only use the strings in the table below (or none).
| AXIS        | problem_dimension (pick ONE)              |
| level       | too_quiet | too_loud                        |
| body        | muddy     | thin                            |
| brightness  | harsh     | dull                            |
| space       | too_wet   | too_dry                         |
| dynamic     | over_compressed | under_compressed          |
| masking     | masking (only one; no opposite)         |
| clean       | clean (only one; fault-free state)      |
| (none)      | none                                      |
Meanings:
- too_quiet: source level is down — buried, weak, can't hear (no competing
  source named as the cause). | too_loud: dominant, overpowering, sticking
  out / too forward in level (not competition-masking).
- muddy: too much low/low-mid — boomy, thick, boxy, congested.
  | thin: lacks body/warmth, hollow, weedy.
- harsh: piercing, gritty, fatiguing highs / presence. | dull: dark,
  muffled, veiled, no air up top.
- too_wet: too much reverb/room/ambience; washed out, interrupts phrases.
  | too_dry: too little reverb/room/ambience — dry, disconnected, reverb
  tail too quiet (includes drum room too quiet; backing-vocal reverb that
  needs to be audible / more "in the background" via wetness).
- over_compressed: too much compression/limiting — squashed, flat, lifeless,
  pumping, energy lost BECAUSE of compression (fix often reduce_compression
  / slower attack / less ratio).
  | under_compressed: uncontrolled / uneven dynamics OR lacks punch, snap,
  transient impact (needs compression or transient shaping; "could use more
  punch", "snappier", peaking/clicky inconsistent levels).
- masking: a competing source covers the victim; victim can't cut through /
  is obscured / lost in a frequency clash. Do NOT use bare loudness words
  alone ("too quiet", "too soft") — those are too_quiet.
- clean: explicit fault-free / balanced / sits cleanly claim (concrete sonic
  state). Not vague "sounds good".
- none: no supported problem and no clean claim.
DECISION RULES
- POLARITY CHECK (mandatory): dimension direction must match the complaint.
  "needs more ambience / too dry" = too_dry, NEVER too_wet; "too wet /
  washed out" = too_wet; "too quiet" = too_quiet; "too loud" = too_loud.
  Verify problem_text and dimension point the same way.
- PUNCH / COMPRESSION (mandatory — easy to reverse):
  * "needs more punch / snappier / could use punch / add transient shaping
    to get hit back" → under_compressed (lack of punch). NEVER over_compressed.
  * "too punchy" (peaks/transients too aggressive, uncontrolled hit) →
    under_compressed (needs compression / dynamic control). NOT too_loud
    and NOT over_compressed — even if someone lowers a fader as a workaround.
  * "squashed / flat from compression / losing energy because compressor
    attack/release/ratio is too aggressive" → over_compressed.
    over_compressed means TOO LITTLE punch left after over-processing.
- REVERB vs LEVEL (mandatory): if the talk is about reverb / send / room /
  delay / reverb-tail amount or audibility, use SPACE — not LEVEL on the
  dry stem. Examples:
  * reverb too quiet / can't hear the tail / need more air / want it more
    "in the background" via reverb → too_dry (raise reverb / send / tail).
  * reverb too loud / washes out / interrupts → too_wet (lower reverb /
    send / tail).
  Do NOT label these as too_loud / too_quiet on vocal just because a
  fader or "volume" word appears on the reverb return.
- masking vs too_quiet (mandatory): use masking ONLY when another source
  explicitly causes the inaudibility / covering (or HISTORY established
  that link). Simple volume with no competing source → too_quiet /
  too_loud. "Sticking out / too forward" with no competition framing →
  too_loud (not masking).
- If the complaint is not one of the listed dimensions → none.
- DRUM ROOM vs OVERHEADS (mandatory):
  - Drum room / room mic / ambience / amb tracks: level or send amount of
    the room is SPACE (too_wet if too much room, too_dry if too little).
    Stem "ambience".
  - Overheads: drum instrument (like snare, hats). Volume of overheads is
    LEVEL (too_loud / too_quiet), NOT space. Stem "overheads".
- Enhancement suggestions without a stated defect ("add saturation for
  warmth", "try distortion") are NOT problems — has_problem=false unless
  the speaker clearly says something sounds wrong.
- Arrangement/composition comments and pure stylistic preferences are none,
  unless presented as something to change in this mix.
FIELD RULES
- problem_stem / fix_stem MUST be one of: vocal, guitar, bass, drums, kick,
  snare, hats, toms, overheads, cymbals, percussion, keys, piano, synth,
  strings, horns, ambience, mix, other, "". Most specific instrument;
  "mix" only for overall-mix issues; "other" for a stated target not
  listed; "" if unresolvable. Never invent values like "dry signal" or
  "low end". Resolve pronouns / "that" from HISTORY. Remember:
  problem_stem = affected; fix_stem = action target (may differ).
- problem_speaker / fix_speaker: "amateur", "expert", or "".
- vocal_lead: true ONLY for lead/main vocal; false for backing vocals /
  doubles / harmonies (see FIGURE vs BAND).
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
        "problem_stem = affected instrument; fix_stem = what you operate on — may differ):",
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


