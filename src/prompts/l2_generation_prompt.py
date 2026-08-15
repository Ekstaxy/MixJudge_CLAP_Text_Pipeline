# -*- coding: utf-8 -*-
"""L2 style-transfer prompts: retarget / strict / free + problem_text|raw exemplars.

Modes (rewrite policy — from most conservative to freest):
- retarget: almost copy one MixAssist turn; only swap instrument/subject words
- strict:   stay close to exemplar tone/shape; may drop filler; retarget party
- free:     paraphrase freely; exemplars are tone hints only

Exemplar content (orthogonal axis):
- problem_text: feed only labeled problem_text spans
- raw:          also feed full Amateur/Expert MixAssist lines

L1 dims follow MixJudge 12 classes (11 problem + clean). Exemplar pool
dims must match L1.dim.

Temperature is fixed low in the generator so the model follows these rules;
do not use temp as a substitute for mode.
"""

from __future__ import annotations

import json

# MixJudge classes (must match labeling_common.DIMENSION_TO_AXIS keys).
VALID_L1_DIMS = (
    "too_quiet",
    "too_loud",
    "muddy",
    "thin",
    "harsh",
    "dull",
    "too_wet",
    "too_dry",
    "over_compressed",
    "under_compressed",
    "masking",
    "clean",
)

OUTPUT_SCHEMA = """\
Return valid JSON only (no markdown fences), exactly:
{
  "amateur": "client / opinion-giver line (conversational English)",
  "expert": "mixer / engineer line (conversational English)",
  "problem_state_text": "short standalone span of the L1 dim meaning (fault OR clean/balanced state); may come from either speaker — do not assume only expert or only amateur states it"
}
"""

_ROLES = """\
Roles (session dialogue only — do NOT assign who must state the problem):
- Amateur: the person giving opinions / feedback about the mix.
- Expert: the mixer / engineer in the session.
The mixing problem (problem_state_text) may appear in either line.
"""

SYSTEM_PROMPT_RETARGET = f"""\
You are a professional audio-engineering dialogue editor doing RETARGET rewrite.

Goal: near-copy ONE MixAssist turn. Keep the original wording, hedges, fillers, and
sentence shape. Change ONLY the instrument / party words so the turn is about
L1.source / L1.subject and still means L1.dim.

{_ROLES}
RETARGET rules (highest priority first):
1. Almost verbatim copy of Exemplar 1. Prefer the raw AMATEUR / EXPERT lines when
   provided; otherwise copy problem_text into the speaker who carries the fault.
2. Swap ONLY stems / party names that conflict with L1.source / L1.subject
   (e.g. keys→lead vocal, toms→band, cymbals→lead vocal). Keep everything else:
   fillers ("yeah", "I guess", "like"), rhythm, length, confirmations.
3. Do NOT invent a new short dialogue from L1 caption. Do NOT paraphrase into
   generic "I don't know, it feels like…". If the exemplar is long, the output
   should stay similarly long.
4. Keep L1 axis/dim/subject/severity meaning. Stem-swap so the labeled fault still
   reads as L1.dim about L1.source (not the exemplar's original stem).
5. Both amateur and expert must be non-empty. If one raw side is empty/trivial,
   keep that side's wording and put the stem-swapped fault on the side that has it;
   if needed, add a minimal confirmation on the empty side — do not rewrite both sides.
6. problem_state_text: short span of the stem-swapped L1.dim fault after retarget.

{OUTPUT_SCHEMA}
"""

SYSTEM_PROMPT_STRICT = f"""\
You are a professional audio-engineering dialogue generator doing STRICT style transfer.

Convert a structured L1 mixing-problem label into a short Amateur/Expert studio dialogue
that closely follows the style exemplars' tone and sentence shape.

{_ROLES}
STRICT mode rules:
1. Keep L1 axis, dim, subject, severity EXACTLY. Do not invent another fault.
2. Stay close to exemplar wording/rhythm; mainly REPLACE the target instrument / party
   to match L1.source and L1.subject (figure = foreground/lead source named by L1;
   bed = backing / the other party — follow L1 fields, do not invent a different source).
3. When raw exemplars are provided: use problem_text as the fault anchor. Do not copy
   unrelated competing defects from the surrounding MixAssist chatter.
4. Do NOT copy exemplar stem names when they conflict with L1.source / L1.subject.
5. Verbosity: medium, not long-winded. Light cleanup of filler is OK (unlike retarget).
6. Both amateur and expert must be non-empty.
7. problem_state_text must reflect the L1 axis/dim meaning only (short span); do not
   copy the whole L1 caption verbatim if a shorter in-dialogue span works.

{OUTPUT_SCHEMA}
"""

SYSTEM_PROMPT_FREE = f"""\
You are a professional audio-engineering dialogue generator doing FREE style transfer.

Convert a structured L1 mixing-problem label into a natural Amateur/Expert studio dialogue.
You may paraphrase freely and vary surface form, as long as the mixing problem stays true.

{_ROLES}
FREE mode rules:
1. Preserve L1 axis, dim, subject, severity meaning. Do not change the fault type.
2. You may reword heavily vs exemplars; exemplars are tone hints only.
3. Target party must match L1.subject / L1.source (figure = foreground/lead source
   named by L1; bed = backing / the other party — follow L1 fields).
4. If raw exemplars contain extra MixAssist chatter, ignore competing faults; only keep
   tone. The output must clearly support L1.dim under labeling.
5. Verbosity: medium.
6. Both amateur and expert must be non-empty.
7. problem_state_text must reflect the L1 axis/dim meaning only (short span); may come
   from either speaker; avoid copying the full L1 caption verbatim when possible.

{OUTPUT_SCHEMA}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_FREE

VALID_MODES = ("retarget", "strict", "free")
VALID_EXEMPLAR_CONTENTS = ("problem_text", "raw")


def system_prompt_for_mode(mode: str) -> str:
    m = (mode or "free").strip().lower()
    if m == "retarget":
        return SYSTEM_PROMPT_RETARGET
    if m == "strict":
        return SYSTEM_PROMPT_STRICT
    return SYSTEM_PROMPT_FREE


def stem_to_subject(stem: str) -> str:
    s = (stem or "").strip().lower()
    return "figure" if s == "vocal" else "bed"


def _format_exemplar_block(ex: dict, index: int, *, include_raw: bool) -> str:
    stem = (ex.get("problem_stem") or "").strip() or "?"
    dim = (ex.get("problem_dimension") or "").strip() or "?"
    text = (ex.get("problem_text") or "").strip()
    lines = [f"Exemplar {index} [stem={stem}, dim={dim}]:"]
    lines.append(f'  problem_text: "{text}"')
    if include_raw:
        user = (ex.get("user_raw_content") or "").strip()
        asst = (ex.get("assistant_raw_content") or "").strip()
        lines.append(f'  AMATEUR (raw): "{user}"')
        lines.append(f'  EXPERT (raw): "{asst}"')
    return "\n".join(lines)


def build_user_prompt(
    input_obj: dict,
    exemplars: list[dict],
    *,
    l1_text: str = "",
    mode: str = "free",
    exemplar_content: str = "raw",
) -> str:
    """Assemble the generation user message from L1 + style exemplars."""
    mode = (mode or "free").strip().lower()
    exemplar_content = (exemplar_content or "raw").strip().lower()
    if exemplar_content not in VALID_EXEMPLAR_CONTENTS:
        exemplar_content = "raw"
    include_raw = exemplar_content == "raw"

    parts = [
        f"Style mode: {mode}",
        f"Exemplar content: {exemplar_content}",
        "Style exemplars from real MixAssist turns (same problem dimension; random).",
    ]

    if mode == "retarget":
        if include_raw:
            parts.append(
                "RETARGET + RAW:\n"
                "- COPY Exemplar 1 AMATEUR/EXPERT nearly verbatim (keep length & fillers).\n"
                "- Swap ONLY instrument/party words to L1.source / L1.subject.\n"
                "- Stem-swap problem_text the same way so the fault still means L1.dim.\n"
                "- Do NOT invent a new dialogue from the L1 caption.\n"
                "- Do NOT shorten into a generic 1–2 sentence rewrite.\n"
            )
        else:
            parts.append(
                "RETARGET: rewrite Exemplar 1 problem_text nearly verbatim; swap only the "
                "instrument/party to L1.source / L1.subject. Expert may be a short confirm "
                "but must be non-empty.\n"
            )
    elif mode == "strict":
        if include_raw:
            parts.append(
                "STRICT + RAW: stay close to exemplar phrasing/rhythm; swap target party. "
                "Anchor on problem_text + L1.dim; do not copy unrelated faults from raw "
                "chatter.\n"
            )
        else:
            parts.append(
                "STRICT: stay close to exemplar problem_text phrasing/rhythm; mainly swap "
                "the target instrument/party. Only problem_text is provided (no raw lines).\n"
            )
    else:
        if include_raw:
            parts.append(
                "FREE + RAW: paraphrase freely; raw exemplars are tone only. Output must "
                "clearly express L1.dim about L1.source (ignore competing raw chatter).\n"
            )
        else:
            parts.append(
                "FREE: paraphrase freely; exemplars are optional tone references only.\n"
            )

    if l1_text:
        parts.append(f'L1 caption (must preserve meaning): "{l1_text}"')

    parts.append(
        f"Must express dim={input_obj.get('dim', '')!r} on "
        f"source={input_obj.get('source', '')!r} "
        f"(subject={input_obj.get('subject', '')!r})."
    )

    if include_raw and mode != "retarget":
        parts.append(
            "RAW SAFETY: surrounding MixAssist talk may mention other instruments/faults; "
            "those are NOT to be preserved. Only the labeled problem_text fault (same dim "
            "as L1) should remain after rewrite."
        )
    elif include_raw and mode == "retarget":
        parts.append(
            "RETARGET RAW NOTE: keep surrounding MixAssist wording; only stem-swap party "
            "names to L1.source / L1.subject so the turn still reads as L1.dim."
        )


    if not exemplars:
        parts.append("(No exemplars for this dim — rely on L1 + role rules.)\n")
    else:
        shown = exemplars[:1] if mode == "retarget" else exemplars
        for i, ex in enumerate(shown, 1):
            parts.append(_format_exemplar_block(ex, i, include_raw=include_raw))
        if mode == "retarget" and len(exemplars) > 1:
            parts.append(
                f"(Note: {len(exemplars) - 1} extra exemplar(s) ignored in retarget; "
                "only Exemplar 1 is the rewrite template.)"
            )

    parts.append("\n# Example JSON shape (content illustrative only)")
    if mode == "retarget":
        if include_raw:
            parts.append(
                'MixAssist problem_text: "too cymbal-y / harsh up top"'
            )
            parts.append(
                "MixAssist raw Amateur: \"It seems like too cymbal-y to me, right? "
                "Like, do you feel that too?\""
            )
            parts.append(
                "MixAssist raw Expert: \"Yeah — I'd turn those cymbals down a bit. "
                "Too harsh up top.\""
            )
        else:
            parts.append('MixAssist problem_text: "too cymbal-y"')
        parts.append(
            'L1 Input: {"axis": "brightness", "dim": "harsh", "subject": "figure", '
            '"source": "vocal", "severity": "medium"}'
        )
        parts.append("Output (near-copy; stem cymbals→lead vocal only):")
        if include_raw:
            out = {
                "amateur": (
                    "It seems like the lead vocal is too harsh up top to me, right? "
                    "Like, do you feel that too?"
                ),
                "expert": (
                    "Yeah — I'd turn the lead vocal down a bit. Too harsh up top."
                ),
                "problem_state_text": "lead vocal is too harsh up top",
            }
        else:
            out = {
                "amateur": "too vocal-y / harsh up top",
                "expert": "Yeah, the lead vocal is too harsh.",
                "problem_state_text": "lead vocal is too harsh up top",
            }
        parts.append(json.dumps(out, ensure_ascii=False))
    else:
        parts.append(
            'Input: {"axis": "body", "dim": "muddy", "subject": "bed", '
            '"source": "band", "severity": "medium"}'
        )
        parts.append("Output:")
        parts.append(
            json.dumps(
                {
                    "amateur": (
                        "The backing tracks feel kinda thick and cloudy — glued together?"
                    ),
                    "expert": "Yeah — let's ease a bit around 250 Hz on the band bus.",
                    "problem_state_text": (
                        "backing tracks feel thick and cloudy in the low mids"
                    ),
                },
                ensure_ascii=False,
            )
        )

    parts.append("\n# Now convert this Input:")
    parts.append(f"Input: {json.dumps(input_obj, ensure_ascii=False)}")
    if l1_text:
        parts.append(f'L1: "{l1_text}"')
    parts.append("Output JSON:")
    return "\n".join(parts)
