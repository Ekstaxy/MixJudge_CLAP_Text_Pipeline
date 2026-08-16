# -*- coding: utf-8 -*-
"""LEXICON_BRIEF §4 slots for assembling L1 captions.

Frame: The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? .
Only SUBJECT and COPULA are sampled. QUALITY is the first degree-compatible
phrase per dim; SCOPE is attached when the table lists one.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# 11 problem dims (no clean) — generation L1 set.
PROBLEM_DIMS = (
    "too_loud",
    "too_quiet",
    "muddy",
    "thin",
    "harsh",
    "dull",
    "too_wet",
    "too_dry",
    "over_compressed",
    "under_compressed",
    "masking",
)

DIM_TO_AXIS = {
    "too_loud": "level",
    "too_quiet": "level",
    "muddy": "body",
    "thin": "body",
    "harsh": "brightness",
    "dull": "brightness",
    "too_wet": "space",
    "too_dry": "space",
    "over_compressed": "dynamic",
    "under_compressed": "dynamic",
    "masking": "masking",
}

SUBJECTS = (
    "the lead vocal",
    "the singer",
    "the vocal",
    "the voice",
)

COPULAS = ("is", "sounds")

# First degree-compatible QUALITY from LEXICON_BRIEF §4.
QUALITY_BY_DIM = {
    "too_loud": "too loud",
    "too_quiet": "too quiet",
    "muddy": "muddy",
    "thin": "thin",
    "harsh": "harsh",
    "dull": "dull",
    "too_wet": "drowned in reverb",
    "too_dry": "bone dry",
    "over_compressed": "squashed",
    "under_compressed": "dynamically uncontrolled",
    "masking": "masked",
}

# First listed SCOPE when the table has one; else "".
SCOPE_BY_DIM = {
    "too_loud": "",
    "too_quiet": "",
    "muddy": "in the low mids",
    "thin": "down low",
    "harsh": "up top",
    "dull": "up top",
    "too_wet": "",
    "too_dry": "",
    "over_compressed": "",
    "under_compressed": "",
    "masking": "",
}


def capitalize_subject(subject: str) -> str:
    s = (subject or "").strip()
    if not s:
        return "The vocal"
    return s[0].upper() + s[1:]


def assemble_caption(
    dim: str,
    *,
    subject: str | None = None,
    copula: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? ."""
    dim = (dim or "").strip().lower()
    if dim not in QUALITY_BY_DIM:
        raise ValueError(f"unknown lexicon dim: {dim}")
    rng = rng or random.Random()
    subj = subject if subject is not None else rng.choice(SUBJECTS)
    cop = copula if copula is not None else rng.choice(COPULAS)
    quality = QUALITY_BY_DIM[dim]
    scope = SCOPE_BY_DIM.get(dim) or ""
    head = f"{capitalize_subject(subj)} {cop} {quality}"
    if scope:
        head = f"{head} {scope}"
    return f"{head}."


def l1_record(
    dim: str,
    index: int,
    *,
    rng: random.Random,
    id_prefix: str = "lex",
) -> dict:
    caption = assemble_caption(dim, rng=rng)
    axis = DIM_TO_AXIS[dim]
    return {
        "segment_id": f"{id_prefix}_{dim}_{index:02d}",
        "source": "vocal",
        "subject": "figure",
        "dimension": dim,
        "axis": axis,
        "severity": "",
        "texts": {"L1": caption},
        "split": "train",
        "validity_type": "A",
    }


def build_l1_records(
    n_per_dim: int,
    seed: int,
    *,
    dims: tuple[str, ...] | list[str] | None = None,
    id_prefix: str = "lex",
    index_offset: int = 0,
) -> list[dict]:
    """n_per_dim captions per dim; SUBJECT/COPULA sampled per row."""
    chosen = tuple(dims) if dims else PROBLEM_DIMS
    unknown = [d for d in chosen if d not in QUALITY_BY_DIM]
    if unknown:
        raise ValueError(f"unknown lexicon dims: {unknown}")
    rows: list[dict] = []
    for dim in chosen:
        for i in range(n_per_dim):
            idx = index_offset + i
            rng = random.Random(f"{seed}|{id_prefix}|{dim}|{idx}")
            rows.append(l1_record(dim, idx, rng=rng, id_prefix=id_prefix))
    return rows


def write_l1_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
