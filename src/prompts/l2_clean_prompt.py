# -*- coding: utf-8 -*-
"""LLM pass over generated L2: flip reversed polarity, drop wrong source / malformed."""

from __future__ import annotations

OPPOSITE_DIM = {
    "too_loud": "too_quiet",
    "too_quiet": "too_loud",
    "muddy": "thin",
    "thin": "muddy",
    "harsh": "dull",
    "dull": "harsh",
    "too_wet": "too_dry",
    "too_dry": "too_wet",
    "over_compressed": "under_compressed",
    "under_compressed": "over_compressed",
}

SYSTEM_PROMPT = """\
You review one generated Amateur/Expert mix-dialogue that is supposed to be
about the VOCAL (the lead vocal / the singer / the vocal / the voice).

GOLD_DIM is the intended MixJudge class. Decide keep vs drop, and whether
GOLD_DIM's polarity is reversed in the dialogue.

DIMENSIONS (exact strings). Each class is a perceptual STATE of the source:
- too_loud: source sits way above the rest; drowning the mix; pure LEVEL
  (gain), no tonal or dynamic claim.
- too_quiet: source is under the mix; you strain to hear it; the source's
  own level is down. No competing source named as the cause.
- muddy: the source itself is thick / boxy / congested in the low mids.
- thin: no body, weedy, hollow; opposite end of the low mids from muddy.
- harsh: piercing, edgy, fatiguing highs / presence ("up top").
- dull: no air, no shine, blanket over the source; too little top. Opposite
  of harsh. NOT muddy (muddy = too much low-mid; dull = too little top).
- too_wet: drowned in reverb / too much room / wash / tail clutter.
- too_dry: dead, close, disconnected; too little reverb/room/ambience.
  Opposite of too_wet.
- over_compressed: squashed, flat, lifeless; dynamics flattened; no punch
  left. NOT too_loud (too_loud is gain only).
- under_compressed: uneven, jumping around in level; unpredictable dynamics.
  Opposite of over_compressed. Lack of punch is over_compressed, not under.
- masking: swallowed / covered because a competing source is in the way.
  Competition/obstruction words only — not bare "quiet"/"too soft".
- clean: fault-free / balanced / sits cleanly, or praise with no defect.

WHAT MUST STAY DISTINGUISHABLE (direction pairs — these are opposites):
- too_loud vs too_quiet: too much own level vs too little own level.
- muddy vs thin: too much low-mid body vs too little.
- harsh vs dull: too much top vs too little top.
- too_wet vs too_dry: too much ambience vs too little.
- over_compressed vs under_compressed: dynamics flattened / no punch left
  vs dynamics too variable / jumping.
Other close pairs (not polarity flips — do NOT retarget GOLD_DIM to these):
- too_quiet vs masking: own level down vs a competing source covering it.
- masking vs muddy: something else in the way vs the source itself thick.
- too_loud vs over_compressed: pure gain vs dynamics flattened.

ACTIONS
1. DROP (action=drop) only for:
   - wrong_source: the complaint is about a non-vocal instrument (snare,
     guitar, drums, keys, bass, cymbals, …) as the affected party. Mentioning
     the band as context while the vocal is still the fault is NOT wrong_source.
   - malformed: the Amateur/Expert English is broken beyond casual speech
     (collapsed syntax, duplicated function words that wreck the sentence,
     fragments that are not a readable turn). Casual fillers and hedges are
     fine.
2. KEEP (action=keep) otherwise, including when someone states a fix or an
   imperative ("needs to come down", "pull it back") — that is allowed.
3. POLARITY: if you KEEP, and the dialogue clearly means the OPPOSITE of
   GOLD_DIM along the direction pairs above, set gold_dim to that opposite.
   Example: GOLD_DIM=too_loud but the talk is that the vocal is too quiet /
   buried by lack of its own level → gold_dim=too_quiet.
   If GOLD_DIM has no opposite (masking, clean), never flip.
   Do not change GOLD_DIM to a different axis. If unsure whether it is a
   polarity flip, keep the original GOLD_DIM.

Return valid JSON only (no markdown):
{
  "action": "keep" or "drop",
  "drop_reason": "" or "wrong_source" or "malformed",
  "gold_dim": "one exact dimension string (original or its opposite if flipped)",
  "flipped": true or false,
  "label_reasoning": "max 25 words"
}
"""


def build_user_message(row: dict) -> str:
    gold = str(row.get("gold_dim") or "").strip()
    opposite = OPPOSITE_DIM.get(gold, "")
    opp_line = (
        f"Opposite of GOLD_DIM (only legal flip): {opposite}"
        if opposite
        else "GOLD_DIM has no opposite — never flip."
    )
    return "\n".join(
        [
            f"GOLD_DIM: {gold}",
            f"GOLD_AXIS: {str(row.get('gold_axis') or '').strip()}",
            opp_line,
            f"L1 caption: {str(row.get('l1_text') or '').strip()}",
            "DIALOGUE:",
            f"AMATEUR: {str(row.get('amateur_text') or '').strip()}",
            f"EXPERT: {str(row.get('expert_text') or '').strip()}",
            f"problem_state_text: {str(row.get('problem_state_text') or '').strip()}",
        ]
    )
