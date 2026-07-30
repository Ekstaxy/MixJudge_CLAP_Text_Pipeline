# -*- coding: utf-8 -*-
"""L2 style-transfer prompts: system text + user-prompt assembly."""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a professional audio-engineering dialogue generator.
Convert a structured mixing-problem label into a short, natural studio dialogue
between Amateur (client / novice) and Expert (mixer). This is style transfer.

Roles:
- Amateur: can hear something is wrong, uses perceptual language only
  (e.g. muddy, piercing, buried, washed out, flat). No mixing jargon.
  About 30% hesitation / small talk. English, like a real mixing session.
- Expert: professional, friendly, precise. Must name the problem axis/dimension
  clearly, and may give a brief plausible fix. Keep fix short (L2 focuses on problem).

Hard rules:
1. Strictly follow the Input JSON: axis, dim, subject, severity. Do not invent
   a different mixing problem.
2. Subject mapping: figure = lead vocal; bed = backing band / instruments.
   Do not copy instrument names from style exemplars if they conflict with subject.
3. Verbosity: medium — natural but not long-winded.
4. Output format EXACTLY two lines (no other preamble):
Amateur: "..."
Expert: "..."
"""


def stem_to_subject(stem: str) -> str:
    """Map MixAssist stem → MixJudge-style subject stub (figure / bed)."""
    s = (stem or "").strip().lower()
    return "figure" if s == "vocal" else "bed"


def build_user_prompt(input_obj: dict, exemplars: list[dict]) -> str:
    """Assemble the generation user message from structured input + style exemplars."""
    parts = [
        "Style exemplars from real mixing sessions (same problem dimension).",
        "Mimic their tone and perceptual wording; do NOT copy them verbatim.",
        "Ignore exemplar instrument names if they conflict with Input.subject.\n",
    ]
    if not exemplars:
        parts.append("(No exemplars available for this dim — rely on role rules.)\n")
    for i, ex in enumerate(exemplars, 1):
        stem = (ex.get("problem_stem") or "").strip() or "?"
        text = (ex.get("problem_text") or "").strip()
        parts.append(f'Exemplar {i} [stem={stem}]: "{text}"')

    parts.append("\n# Fixed format example (content is illustrative only)")
    parts.append(
        'Input: {"axis": "body", "dim": "muddy", "subject": "bed", "severity": "medium"}'
    )
    parts.append("Output:")
    parts.append(
        'Amateur: "The backing tracks feel kinda thick and cloudy down there — '
        'like everything is glued together?"'
    )
    parts.append(
        'Expert: "Yeah — the bed is muddy in the low-mids. Let\'s ease a bit around '
        '250 Hz on the band bus so the vocal has room."'
    )
    parts.append("\n# Now convert this Input:")
    parts.append(f"Input: {json.dumps(input_obj, ensure_ascii=False)}")
    parts.append("Output:")
    return "\n".join(parts)
