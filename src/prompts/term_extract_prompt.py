# -*- coding: utf-8 -*-
"""Blind term extraction prompts for MixAssist lexicon building.

IMPORTANT (for human review):
  This prompt must NOT mention the 8 mixing axes (level / body / brightness /
  space / dynamic / masking / stereo / phase) or any signed dimensions.
  The model only dumps free-form (target, desc) pairs.
"""

from __future__ import annotations

from prompts.labeling_prompt import (
    MAX_HISTORY_MESSAGES,
    is_placeholder_assistant,
    is_placeholder_user,
    parse_input_history,
)

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — edit here; re-run extract after changes.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You extract sonic descriptors from MixAssist mixing-session dialogue
between an AMATEUR (user) and an EXPERT (assistant).

TASK
Find EVERY adjective, verb, or short phrase that describes a sound's
state, problem, or listening impression, and the instrument / mix
element it refers to. Dump freely. Do NOT classify, bucket, or rename
terms into any taxonomy.

WHAT TO EXTRACT
- Sonic quality / problem / feel: e.g. "too harsh", "muddy", "buried",
  "punchy", "washed out", "lacks punch", "bleeding into the kick",
  "too quiet", "sibilant", "hollow".
- Keep the speaker's wording when possible (minimally trimmed).
- Include both Amateur and Expert wording when they describe sound.

WHAT NOT TO EXTRACT
- Pure workflow / DAW talk with no sonic description ("open the EQ",
  "let's solo that", "save the session").
- Musical arrangement / songwriting preference with no mix-sound claim.
- Vague acknowledgements ("yeah", "sounds good") unless they name a
  concrete sonic quality.
- Any descriptions that only state general approval or disapproval (e.g., "sounds good", "this is fine", "perfect", "balanced", "not loving") 
  lacking concrete sonic details.

TARGET
- Prefer a concrete instrument or bus: vocal, kick, snare, bass, guitar,
  hats, cymbals, overheads, ambience, mix, etc.
- Use "mix" for overall-mix impressions.
- Resolve pronouns ("it", "that", "this") from INPUT HISTORY + CURRENT
  TURN when possible; if still unclear, use "unknown".

CONTEXT
- TOPIC: session topic; background only.
- INPUT HISTORY: earlier turns; use to resolve what "it/that" refers to.
- CURRENT TURN: primary source for descriptors.

OUTPUT
Return valid JSON only. No Markdown fences, no prose outside JSON.
Schema:
{
  "terms": [
    {"target": "kick", "desc": "not punchy enough"},
    {"target": "vocal", "desc": "too harsh"}
  ]
}

If there are no sonic descriptors in this turn, return {"terms": []}.
One object per distinct (target, desc) pair. Do not invent wording that
is not supported by the dialogue.
""".strip()


def format_history_for_extract(history: list[dict]) -> str:
    """Readable earlier-turn block (no axis/dimension wording)."""
    if not history:
        return ""

    blocks: list[list[dict]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if (
            msg.get("role") == "user"
            and i + 1 < len(history)
            and history[i + 1].get("role") != "user"
        ):
            blocks.append([msg, history[i + 1]])
            i += 2
        else:
            blocks.append([msg])
            i += 1

    max_turns = max(1, MAX_HISTORY_MESSAGES // 2)
    blocks = blocks[-max_turns:]

    lines = [
        "INPUT HISTORY (earlier turns — resolve pronouns / continued referents only):",
    ]
    for t, block in enumerate(blocks, 1):
        lines.append(f"--- earlier turn {t} ---")
        for msg in block:
            role = "AMATEUR" if msg.get("role") == "user" else "EXPERT"
            content = (msg.get("content") or "").strip()
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_extract_user_message(row: dict) -> str:
    """Assemble user message for blind term extraction."""
    user_text = (row.get("user") or "").strip()
    assistant_text = (row.get("assistant") or "").strip()

    parts = [
        f"TOPIC: {row.get('topic', '')}",
        f"TURN_ID: {row.get('turn_id', '')}",
    ]

    history = parse_input_history(row.get("input_history", ""))
    history_block = format_history_for_extract(history)
    if history_block:
        parts.append(history_block)

    parts.append("CURRENT TURN (primary source for descriptors):")
    parts.append(f"AMATEUR: {user_text}")
    parts.append(f"EXPERT: {assistant_text}")

    if is_placeholder_user(user_text):
        parts.append(
            "NOTE: Amateur message is a placeholder — they did not speak this turn. "
            "Extract from EXPERT and HISTORY only."
        )
    if is_placeholder_assistant(assistant_text):
        parts.append(
            "NOTE: Expert message is a placeholder — they did not speak this turn. "
            "Extract from AMATEUR and HISTORY only."
        )

    parts.append(
        'Extract all sonic (target, desc) pairs from CURRENT TURN. '
        'Return JSON: {"terms": [{"target": "...", "desc": "..."}, ...]}'
    )
    return "\n".join(parts)
