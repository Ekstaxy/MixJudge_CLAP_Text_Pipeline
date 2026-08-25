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
#     視為 expert 接續發言。INPUT HISTORY 只輔助 (代詞 / stem),不能單獨決定 dimension。
#     當次 CURRENT TURN 自己看不出是哪一個 dimension → 標 none,不要從 HISTORY 補 dim。
#   - assistant 若是 "I need more information before I can respond. Please elaborate."
#     代表 expert 沒說話,由 amateur 當次發言判斷 dimension;HISTORY 只解代詞/stem。
SYSTEM_PROMPT = """
You annotate turns from MixAssist, a mixing-session dialogue between an
AMATEUR (user) and an EXPERT (assistant). Goal: high-precision labels of
mixing problems and fixes. Be conservative — label a problem only when the
words clearly support it. Workflow talk, musical preference, or production
ideas are not problems.
You receive:
- TOPIC: broad session topic; not proof of a stem.
- INPUT HISTORY: earlier turns (readable, auxiliary only).
- CURRENT TURN: primary source for problem_text, fix_text, AND which
  problem_dimension. There are 12 classes (11 problems + clean).
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
4. Praise / approval counts as clean: "that sounds great", "this is fine",
   "perfect", "sounds good", "I like it", "yeah that's better" with no
   stated defect → clean (has_problem=true). Explicit balanced / sits
   cleanly / fault-free wording → clean as well. Always keep clean as a
   labelable class. Do not drop praise into none.
CURRENT TURN vs HISTORY (dimension gate — mandatory)
- Decide problem_dimension from CURRENT TURN wording ALONE.
- If this turn, ignoring HISTORY, is compatible with more than one of the
  12 classes — or with none of them — output none. Do not guess. Do not
  use HISTORY to break a tie between dims (too_quiet vs masking, muddy vs
  dull, too_loud vs over_compressed, etc.).
- HISTORY is auxiliary only: pronouns ("it"/"that"), which stem, lead vs
  backing, and whether a CURRENT TURN fix is still the same job. It must
  not supply the dimension when CURRENT TURN does not name a distinguishable
  fault.
- Do NOT recover problem_text / problem_dimension from HISTORY just because
  CURRENT TURN is a vague adjust ("pull it back", "I wanna adjust that",
  "where is that coming from") with no stated defect. Those are none for
  has_problem (you may still label a CURRENT TURN fix as has_fix).
- Do not re-report an old HISTORY problem the current turn is not stating.
- Placeholders: AMATEUR "Please analyze this audio segment." = amateur did
  not speak; EXPERT "I need more information before I can respond. Please
  elaborate." = expert did not speak. Same for filler-only messages. Judge
  dimension from whoever actually spoke in CURRENT TURN.
AFFECTED STEM vs ACTION TARGET (critical)
- problem_stem = the instrument that is suffering / the perceptual problem
  is about (what you cannot hear, what sounds wrong). For clean: the source
  described as sitting cleanly / balanced.
- fix_stem = the instrument you operate on (fader, EQ, mute, etc.).
- They can differ. Example: CURRENT says the snare is lost and the cymbal
  is filling highs so "pull that back" → problem_stem=snare,
  problem_dimension=masking, fix_stem=cymbals, fix_action=lower_level
  or eq_cut. The covering/lost claim must be in CURRENT TURN. Do NOT set
  problem_stem=cymbals just because that is what you turn down — the
  cymbal is the cause/masker, not the problem stem.
- For masking: problem_stem is the victim (the source being covered up /
  unable to cut through). Name the aggressor/masker in label_reasoning
  (and usually as fix_stem if that is what gets adjusted).
- HISTORY may resolve which stem "it/that" refers to. It may NOT invent
  a missing masking/too_quiet/muddy/... claim that CURRENT TURN never made.
vocal_lead
- true only if THIS label is about the lead / main vocal.
- false for backing vocals, doubles, harmonies, and any non-lead source
  (even if problem_stem is "vocal").
- Do not use vocal_lead to choose problem_dimension.
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
| clean       | clean (fault-free or praise; no opposite) |
| (none)      | none                                      |
Meanings (what the class is — MixAssist source can be any stem, not only vocal):
- too_loud: source sits way above the rest; drowning the mix; pure LEVEL
  (gain), no tonal or dynamic claim.
- too_quiet: source is under the mix; you strain to hear it; the source's
  own level is down. No competing source named as the cause.
- muddy: the source itself is thick / boxy / congested in the low mids.
- thin: no body, weedy, hollow; opposite end of the low mids from muddy.
- harsh: piercing, edgy, fatiguing highs / presence ("up top").
- dull: no air, no shine, blanket over the source; too little top. Opposite
  of harsh. NOT the same as muddy (muddy = too much low-mid; dull = too
  little top — engineers mix these words; we must not).
- too_wet: drowned in reverb / too much room / wash / tail clutter.
- too_dry: dead, close, disconnected; too little reverb/room/ambience.
  Opposite of too_wet.
- over_compressed: squashed, flat, lifeless; dynamics flattened; no punch
  left (robbed of impact). NOT the same as too_loud (too_loud is gain only).
- under_compressed: uneven, jumping around in level, uncontrolled /
  unpredictable dynamics. Opposite of over_compressed. NOT "lacks punch" —
  lack of punch is over_compressed. Never "loud" or "quiet" as the dim
  for either compression class.
- masking: swallowed / covered / can't cut through because something else
  is in the way. The victim may be untouched while a competing source
  rises. Competition/obstruction words only — never bare "quiet"/"too soft"
  (those are too_quiet). Masking is not muddy: muddy = the source itself
  sounds thick; masking = another source is in the way.
- clean: fault-free / balanced / sits cleanly, OR praise/approval with no
  defect ("sounds good", "perfect", "that's great", "I like it"). One of
  the 12 classes. Do not send praise to none.
- none: no supported problem and no clean claim, OR the current turn
  cannot distinguish which of the 12 it is.
WHAT MUST STAY DISTINGUISHABLE (if CURRENT TURN could be either side → none)
These pairs must stay apart (LEXICON_BRIEF). If the wording could describe
either side, label none — do not pick the closer class.
- too_quiet vs masking: too_quiet = this source's OWN level is down (strain
  to hear it). masking = the victim may be untouched; a COMPETING source
  covers it / it cannot cut through. Never use bare loudness words for
  masking (no "quiet", "too soft"). Competition/obstruction only: masked,
  obscured, covered up, lost in a frequency clash. Bare "buried" / "can't
  hear it" with no cause named in CURRENT TURN → none.
- masking vs muddy: masking = something else is in the way. muddy = the
  source ITSELF is thick / boxy / congested in the low mids (not a
  competing stem).
- too_loud vs over_compressed: too_loud = pure gain, drowning the mix; no
  tonal or dynamic claim. over_compressed = dynamics flattened, loudness
  held. No brightness / thickness / space words for too_loud or too_quiet.
- over_compressed vs under_compressed: over = squashed / lifeless / no
  punch left. under = uneven / inconsistent / unpredictable (phrases too
  loud AND too quiet). Never use "loud" or "quiet" as the dim for either.
  "Needs more punch" / "snappier" / robbed of punch = over, not under.
- muddy vs dull: muddy = too much low-mid. dull = too little top ("up top"
  / no air). Engineers mix these words; we must not.
- harsh vs dull: opposite ends of the top (piercing vs no shine).
- too_wet vs too_dry: opposite amounts of ambience.
- thin vs muddy: opposite ends of the low mids (weedy/hollow vs thick).
If an adjective could plausibly describe two of these classes, it belongs
to neither → none.
DECISION RULES
- POLARITY CHECK (mandatory): dimension direction must match the complaint.
  "needs more ambience / too dry" = too_dry, NEVER too_wet; "too wet /
  washed out" = too_wet; "too quiet" = too_quiet; "too loud" = too_loud.
  Verify problem_text and dimension point the same way.
- PUNCH / COMPRESSION (mandatory — easy to reverse; follow LEXICON_BRIEF):
  * over_compressed = TOO LITTLE punch left: squashed / flat / lifeless /
    crushed / pumping / "losing energy because the compressor is too hard" /
    "robbed of punch" / "needs more punch" / "snappier" / "add punch back"
    when the complaint is that it has no hit left. NEVER under_compressed.
  * under_compressed = dynamics too VARIABLE: jumping around in level /
    uneven / inconsistent / "too punchy" (peaks poke out, uncontrolled hit) /
    needs compression or transient control to tame peaks. NOT too_loud.
    NOT over_compressed.
  * "Add a compressor" / "add punch" / "transient shaping" with NO stated
    defect (not jumpy, not squashed) → none (enhancement, not a problem).
  * Do not use too_loud / too_quiet for either compression class.
- REVERB vs LEVEL (mandatory): if the talk is about reverb / send / room /
  delay / reverb-tail amount or audibility, use SPACE — not LEVEL on the
  dry stem. Examples:
  * reverb too quiet / can't hear the tail / need more air / want it more
    "in the background" via reverb → too_dry (raise reverb / send / tail).
  * reverb too loud / washes out / interrupts → too_wet (lower reverb /
    send / tail).
  Do NOT label these as too_loud / too_quiet on vocal just because a
  fader or "volume" word appears on the reverb return.
- masking vs too_quiet (mandatory): use masking ONLY when CURRENT TURN names
  a competing source as the cause of covering / not cutting through. Simple
  volume with no competing source in CURRENT TURN → too_quiet / too_loud.
  "Sticking out / too forward" with no competition framing → too_loud.
  If CURRENT TURN is only "buried" / "can't hear it" with no cause, that
  could be too_quiet OR masking → none (do not let HISTORY pick).
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
- vocal_lead: true ONLY for lead/main vocal; false otherwise.
- has_speaker: true only if the problem or fix is attributable to a speaker.
- fix_action: concise normalized action when explicit (lower_level,
  raise_level, eq_cut, eq_boost, add_reverb, reduce_reverb, add_compression,
  reduce_compression, pan, add_saturation, reduce_saturation); else "".
  Prefer lower_level / eq_cut over vague "adjust".
- label_reasoning: max 25 words. State affected stem, dimension, and if
  relevant the cause/masker (e.g. "snare lost; cymbal masking highs").
- confidence: high = CURRENT TURN names the dim directly; mid = CURRENT
  TURN is enough for the dim but needed HISTORY for stem; low = still
  useful but wording is thin. If dim is not decidable from CURRENT TURN
  → none, not low-confidence guess.
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
        "INPUT HISTORY (earlier turns — auxiliary ONLY for pronouns / stem;",
        "problem_dimension must be decidable from CURRENT TURN alone.",
        "If CURRENT TURN without history could be more than one class, label none.",
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
            "Treat EXPERT as continuing. Dimension still must be clear from CURRENT TURN "
            "(expert's words). HISTORY cannot supply the dimension."
        )
    if is_placeholder_assistant(assistant_text):
        parts.append(
            "NOTE: Expert message is a placeholder — they did not speak this turn. "
            "Judge from AMATEUR in CURRENT TURN. HISTORY only for stem/pronouns, "
            "not for choosing problem_dimension."
        )

    return "\n".join(parts)


