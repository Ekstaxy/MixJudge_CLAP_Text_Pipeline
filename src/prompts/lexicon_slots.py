# -*- coding: utf-8 -*-
"""L1 captions: The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? .

SUBJECT / COPULA are sampled. QUALITY is sampled from
outputs/quality_from_manual/quality_from_manual.json (degree_compatible or
standalone). SCOPE comes from that same file when the dim lists one.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_QUALITY_JSON = (
    PROJECT_ROOT / "outputs" / "quality_from_manual" / "quality_from_manual.json"
)
# HF `--include "outputs/*"` only gets one directory level, so Kaggle may
# land the lexicon at outputs/quality_from_manual.json instead of the subdir.
QUALITY_JSON_CANDIDATES = (
    DEFAULT_QUALITY_JSON,
    PROJECT_ROOT / "outputs" / "quality_from_manual.json",
    PROJECT_ROOT / "new_outputs" / "quality_from_manual" / "quality_from_manual.json",
    PROJECT_ROOT / "new_outputs" / "quality_from_manual.json",
)

# 12 classes = 11 problem dims + clean. `--dims clean` generates only clean.
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
    "clean",
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
    "clean": "clean",
}

SUBJECTS = (
    "the lead vocal",
    "the singer",
    "the vocal",
    "the voice",
)

COPULAS = ("is", "sounds")

# Fallback if a dim is missing from the manual json.
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
    "clean": "clean",
}

_QUALITY_CACHE: dict[str, dict] = {}


def resolve_quality_json(path: Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"quality lexicon not found: {p}")
    for candidate in QUALITY_JSON_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(p) for p in QUALITY_JSON_CANDIDATES)
    raise FileNotFoundError(f"quality lexicon not found. tried:\n  {tried}")


def load_quality_lexicon(path: Path | None = None) -> dict:
    p = resolve_quality_json(path)
    key = str(p)
    if key not in _QUALITY_CACHE:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"quality lexicon must be a JSON object: {p}")
        _QUALITY_CACHE[key] = data
    return _QUALITY_CACHE[key]


def _clean_phrases(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def parse_scopes(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return [""]
    parts = [p.strip() for p in raw.split("·") if p.strip()]
    return parts or [""]


def fit_after_copula(quality: str) -> str:
    """Lowercase a leading capital so it sits after is/sounds."""
    q = (quality or "").strip().strip('"')
    if len(q) >= 2 and q[0].isupper() and q[1].islower():
        return q[0].lower() + q[1:]
    return q


def sample_quality(
    dim: str,
    rng: random.Random,
    *,
    lexicon: dict | None = None,
) -> tuple[str, str, str]:
    """Return (quality, kind, scope). kind is degree_compatible | standalone | fallback."""
    dim = (dim or "").strip().lower()
    data = lexicon if lexicon is not None else load_quality_lexicon()
    entry = data.get(dim) or {}
    degree = _clean_phrases(entry.get("degree_compatible"))
    standalone = _clean_phrases(entry.get("standalone"))
    kinds: list[str] = []
    if degree:
        kinds.append("degree_compatible")
    if standalone:
        kinds.append("standalone")
    if not kinds:
        return QUALITY_BY_DIM[dim], "fallback", ""
    kind = rng.choice(kinds)
    pool = degree if kind == "degree_compatible" else standalone
    quality = rng.choice(pool)
    scope = rng.choice(parse_scopes(str(entry.get("scope") or "")))
    return quality, kind, scope


def capitalize_subject(subject: str) -> str:
    s = (subject or "").strip()
    if not s:
        return "The vocal"
    return s[0].upper() + s[1:]


def _standalone_needs_copula(quality: str) -> bool:
    q = fit_after_copula(quality).lower()
    prefixes = (
        "is ",
        "sounds ",
        "sound ",
        "can't ",
        "cannot ",
        "doesn't ",
        "does not ",
        "didn't ",
        "did not ",
        "needs ",
        "need to ",
        "could ",
        "can ",
        "turn ",
        "pull ",
        "bring ",
        "add ",
        "give ",
        "warm ",
        "brighten ",
        "sing ",
    )
    return not q.startswith(prefixes)


def format_caption(
    subject: str,
    copula: str,
    quality: str,
    scope: str,
    kind: str,
    dim: str = "",
) -> str:
    q = fit_after_copula(quality)
    dim = (dim or "").strip().lower()
    if dim == "clean" and kind == "standalone":
        head = f"{capitalize_subject(subject)} — {q}"
    elif kind == "standalone" and not _standalone_needs_copula(q):
        head = f"{capitalize_subject(subject)} {q}"
    else:
        head = f"{capitalize_subject(subject)} {copula} {q}"
    if scope and scope.lower() not in q.lower():
        head = f"{head} {scope}"
    return f"{head}."


def assemble_caption(
    dim: str,
    *,
    subject: str | None = None,
    copula: str | None = None,
    rng: random.Random | None = None,
    quality_json: Path | None = None,
) -> str:
    """The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? ."""
    dim = (dim or "").strip().lower()
    if dim not in DIM_TO_AXIS:
        raise ValueError(f"unknown lexicon dim: {dim}")
    rng = rng or random.Random()
    lexicon = load_quality_lexicon(quality_json)
    subj = subject if subject is not None else rng.choice(SUBJECTS)
    cop = copula if copula is not None else rng.choice(COPULAS)
    quality, kind, scope = sample_quality(dim, rng, lexicon=lexicon)
    return format_caption(subj, cop, quality, scope, kind, dim=dim)


def l1_record(
    dim: str,
    index: int,
    *,
    rng: random.Random,
    id_prefix: str = "lex",
    quality_json: Path | None = None,
) -> dict:
    dim = (dim or "").strip().lower()
    lexicon = load_quality_lexicon(quality_json)
    subj = rng.choice(SUBJECTS)
    cop = rng.choice(COPULAS)
    quality, kind, scope = sample_quality(dim, rng, lexicon=lexicon)
    caption = format_caption(subj, cop, quality, scope, kind, dim=dim)
    axis = DIM_TO_AXIS[dim]
    return {
        "segment_id": f"{id_prefix}_{dim}_{index:02d}",
        "source": "vocal",
        "subject": "figure",
        "dimension": dim,
        "axis": axis,
        "severity": "",
        "quality": fit_after_copula(quality),
        "quality_kind": kind,
        "scope": scope,
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
    quality_json: Path | None = None,
) -> list[dict]:
    """n_per_dim captions per dim; SUBJECT/COPULA/QUALITY sampled per row."""
    chosen = tuple(dims) if dims else PROBLEM_DIMS
    unknown = [d for d in chosen if d not in DIM_TO_AXIS]
    if unknown:
        raise ValueError(f"unknown lexicon dims: {unknown}")
    load_quality_lexicon(quality_json)
    rows: list[dict] = []
    for dim in chosen:
        for i in range(n_per_dim):
            idx = index_offset + i
            rng = random.Random(f"{seed}|{id_prefix}|{dim}|{idx}")
            rows.append(
                l1_record(
                    dim,
                    idx,
                    rng=rng,
                    id_prefix=id_prefix,
                    quality_json=quality_json,
                )
            )
    return rows


def write_l1_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
