# -*- coding: utf-8 -*-
"""
MixAssist blind lexicon pipeline.

Stages:
  1) extract  — Gemma (GGUF) dumps free-form (target, desc) pairs
  2) map      — hierarchical assign (centroid / clap):
                  first 7 axes, then within that axis split into dims
                  (masking / clean have only 1 dim each)
  3) export   — review CSVs for axis and dimension (corrected_* empty)

Lexicon taxonomy follows MixJudge caption ontology (same DIMENSION_TO_AXIS
as labeling): 12 classes = 11 problem dims + clean; 7 axes.

Usage (from repo root):
  python src/term_extract.py extract --splits train --limit 5
  python src/term_extract.py map-and-export --method both --device cpu

Outputs (under outputs/):
  lexicon_raw_{split}.jsonl / lexicon_raw_all.json
  lexicon_mapped_{centroid|clap}.json
  lexicon_review_{centroid|clap}_axis.csv
  lexicon_review_{centroid|clap}_dim.csv
  lexicon_review_compare_axis.csv / lexicon_review_compare_dim.csv

Deps:
  extract:  llama-cpp-python
  centroid: sentence-transformers
  clap:     transformers + torch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from labeling.labeling_common import (  # noqa: E402
    DIMENSION_TO_AXIS,
    PROJECT_ROOT,
    SPLIT_FILES,
    STEM_ALIASES,
    append_raw,
    extract_json,
    load_split,
)
from prompts.term_extract_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_extract_user_message,
)

DEFAULT_GGUF = PROJECT_ROOT / "models" / "google_gemma-4-31B-it-Q4_K_M.gguf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"

# Axes / dims shared with labeling (MixJudge 12 classes). Seeds stay local.
AXES = (
    "level",
    "body",
    "brightness",
    "space",
    "dynamic",
    "masking",
    "clean",
)

DIMENSIONS = tuple(DIMENSION_TO_AXIS.keys())

# axis → dims (pairs of 2; masking / clean → 1)
AXIS_TO_DIMS: dict[str, tuple[str, ...]] = {}
for _dim, _axis in DIMENSION_TO_AXIS.items():
    AXIS_TO_DIMS.setdefault(_axis, [])
    AXIS_TO_DIMS[_axis].append(_dim)
AXIS_TO_DIMS = {a: tuple(AXIS_TO_DIMS[a]) for a in AXES}

# 7-axis seed anchors
# Adjustments below are driven only by misplaced descs in
# lexicon_review_centroid_dim.csv (remove magnets / move wrong-dim hits).
SEED_WORDS_AXIS = {
    "level": [
        "loud", "quiet", "too loud", "too quiet",
        "overpowering", "inaudible", "can't hear", "volume",
    ],
    "body": [
        # removed "full" (pulled wide/wider → muddy), "lacks body" (pulled gives it some body → thin)
        "muddy", "muddiness", "mud", "thin", "boomy",
        "low-end mud", "thick",
    ],
    "brightness": [
        # removed sparkly/shimmer (pulled sparkliness / add some sparkle → dull)
        "harsh", "bright", "too bright", "dull", "dark",
        "sibilant", "piercing", "edgy", "fatiguing", "high end",
    ],
    "space": [
        # removed airy / in a space (pulled airiness / give it some air → too_dry)
        "dry", "too dry", "wet", "too wet", "washed out",
        "distant", "closely miked", "too much reverb", "no reverb",
    ],
    "dynamic": [
        # keep punchy here so punchier hits dynamic not brightness/harsh
        "punchy", "punchier", "flat", "squashed", "pumping", "tight",
        "pinched", "over compressed", "no dynamics", "transient", "clicky",
    ],
    # competition / obstruction only — avoid bare loudness words (vs level)
    "masking": [
        "masked", "overcrowded", "drowned", "gets lost", "buried",
        "bleeding", "getting in the way", "covered by", "swamping", "muffled",
    ],
    # fault-free / no degradation (wet stem untouched)
    "clean": [
        "clean", "balanced", "sits cleanly", "well balanced",
        "clear and balanced", "no issues", "sounds good in the mix",
        "natural and clear",
    ],
}

# 12 classes = 11 problem dims + clean
# Seed edits from misplaced CSV descs only (not newly invented phrases).
SEED_WORDS_DIM: dict[str, list[str]] = {
    "too_quiet": [
        "too quiet", "quiet", "barely audible", "inaudible",
        "too soft", "can't hear", "lost under the band",
    ],
    "too_loud": [
        "too loud", "loud", "overpowering", "way too loud",
        "blasting over the band", "far too dominant",
    ],
    "muddy": [
        "muddy", "boomy", "boxy", "congested", "woolly", "thick in the low mids",
    ],
    "thin": [
        # removed "lacks body" / "small and lacking body" — magnet for "gives it some body"
        "thin", "hollow", "weedy", "no weight",
    ],
    "harsh": [
        # CSV: bright / so bright / more bright were wrongly → dull; anchor them here
        # (brighten / add some sparkle are wish phrases — not added as seeds)
        "harsh", "piercing", "brittle", "edgy", "fatiguing", "sibilant",
        "bright", "too bright", "so bright", "more bright",
    ],
    "dull": [
        # removed "lacking air" / "no shine" — magnets for brighten / add some sparkle / sparkliness
        "dull", "dark", "veiled", "lidded",
    ],
    "too_wet": [
        # CSV correctly had these on too_wet — reinforce
        "too wet", "washed out", "drowned in reverb", "swimming in ambience",
        "too much reverb", "distant and diffuse",
        "super wet", "100% wet",
    ],
    "too_dry": [
        # removed "without any ambience" — magnet for airiness / give it some air / airy
        "too dry", "bone dry", "no reverb",
        "disconnected from the room", "pasted on top of the mix",
    ],
    "over_compressed": [
        "over compressed", "squashed", "flattened", "lifeless",
        "pumping", "no dynamics", "crushed flat",
    ],
    "under_compressed": [
        # CSV: punchier was wrongly → harsh; "needs compression" pulled wish phrasing
        "under compressed", "uneven", "jumping around in level",
        "uncontrolled", "wildly inconsistent",
        "punchier", "a little bit punchier",
    ],
    # no "quiet" / "soft" / bare "buried" — those belong to too_quiet
    "masking": [
        "masked", "overcrowded", "drowned", "gets lost", "buried",
        "bleeding", "getting in the way", "covered by", "swamping", "muffled",
    ],
    # no fault; brief: "sits cleanly", "mix is balanced" (no slot grammar)
    "clean": [
        "clean", "balanced", "sits cleanly", "the mix is balanced",
        "clear and balanced", "natural and clear", "sits well in the mix",
        "no mix problems",
    ],
}

TEMPERATURE = 0.1
N_CTX = 6144
N_BATCH = 512
MAX_TOKENS = 1536
N_GPU_LAYERS = -1
TOP_P = 0.95
TOP_K = 64

METHODS = ("centroid", "clap", "both")


def resolve_methods(method: str) -> list[str]:
    return ["centroid", "clap"] if method == "both" else [method]


# ====================== post-filter (after nearest-seed map) ======================
# Lightweight rules: drop retired/noise phrasing; fix common polarity / wish mistakes.
# Edits assigned_* only. Human overrides stay in corrected_*.

_DROP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(pan|panned|panning)\b", re.I), "stereo"),
    (re.compile(r"\b(left ear|right ear)\b", re.I), "stereo"),
    (re.compile(r"\b(sounds?\s+)?mono\b", re.I), "stereo"),
    (re.compile(r"\b(too wide|widen|wider)\b", re.I), "stereo"),
    (re.compile(r"^wide$", re.I), "stereo"),
    (re.compile(r"\bsound better\b", re.I), "noise"),
    (re.compile(r"\bsoulful\b", re.I), "noise"),
    (re.compile(r"^(weird|sounds weird|sounds funky|bit strange|sucks)$", re.I), "noise"),
    (re.compile(r"^more$", re.I), "noise"),
    (re.compile(r"^better$", re.I), "noise"),
    (re.compile(r"^simple$", re.I), "noise"),
    (re.compile(r"^characteristic$", re.I), "noise"),
    (re.compile(r"^not loving$", re.I), "noise"),
]

# (pattern, new_dimension, note_tag) — first match wins
_RELABEL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\b(brighten(s| it up)?|add(ing)? some sparkle|shimmer a little bit more|"
            r"more air|give(s| it)? some air|giving it the airiness|clearer tone)\b",
            re.I,
        ),
        "dull",
        "wish→dull",
    ),
    (
        re.compile(
            r"\b(could be a little quieter|a little bit quieter|"
            r"needs? to come down|need to go .+ down|can come down|go down|"
            r"turn(ing)? (it |them |that \w+ )?down|"
            r"does not need nearly as much volume|should be lowered|"
            r"could be turned down)\b",
            re.I,
        ),
        "too_loud",
        "wish→too_loud",
    ),
    (
        re.compile(
            r"\b(needs? to come up|need to come up|sing louder|pull the voice up|"
            r"bring .+ up a little)\b",
            re.I,
        ),
        "too_quiet",
        "wish→too_quiet",
    ),
]


def _append_note(existing: str, tag: str) -> str:
    tag = f"post:{tag}"
    if not existing:
        return tag
    if tag in existing:
        return existing
    return f"{existing}; {tag}"


def decide_post_rule(desc: str, assigned_dimension: str) -> tuple[str, str | None, str | None]:
    """Return (action, new_dim_or_None, note_tag).
    action: 'keep' | 'drop' | 'relabel'
    """
    text = (desc or "").strip()
    if not text:
        return "drop", None, "empty"

    for pat, tag in _DROP_PATTERNS:
        if pat.search(text):
            return "drop", None, tag

    for pat, new_dim, tag in _RELABEL_PATTERNS:
        if pat.search(text):
            if new_dim != assigned_dimension:
                return "relabel", new_dim, tag
            return "keep", None, None

    # bare loudness wish stuck on masking → too_quiet family already handled;
    # if masking but text is pure "come up" etc., covered above.
    return "keep", None, None


def apply_post_rules(terms: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Filter/relabel mapped terms in place-style; returns (kept, stats)."""
    stats = {"kept": 0, "dropped": 0, "relabeled": 0}
    out: list[dict] = []
    for t in terms:
        desc = t.get("desc", "")
        old_dim = t.get("assigned_dimension", "")
        action, new_dim, tag = decide_post_rule(desc, old_dim)
        if action == "drop":
            stats["dropped"] += 1
            continue
        if action == "relabel" and new_dim:
            t = dict(t)
            t["assigned_dimension"] = new_dim
            t["assigned_axis"] = DIMENSION_TO_AXIS.get(new_dim, t.get("assigned_axis", ""))
            t["dimension_axis"] = t["assigned_axis"]
            t["post_note"] = tag or ""
            # keep dimension_scores but mark note for export
            stats["relabeled"] += 1
        else:
            stats["kept"] += 1
        out.append(t)
    return out, stats


def apply_post_rules_csv_rows(
    rows: list[dict],
    *,
    label_key: str,
) -> tuple[list[dict], list[dict]]:
    """Apply rules to review CSV rows. Returns (new_rows, change_records)."""
    new_rows: list[dict] = []
    changes: list[dict] = []
    for row in rows:
        desc = row.get("desc", "")
        if label_key == "assigned_dimension":
            old = (row.get("assigned_dimension") or "").strip()
            action, new_dim, tag = decide_post_rule(desc, old)
            if action == "drop":
                changes.append(
                    {
                        "action": "drop",
                        "desc": desc,
                        "from": old,
                        "to": "",
                        "note": tag or "",
                    }
                )
                continue
            row = dict(row)
            if action == "relabel" and new_dim:
                row["assigned_dimension"] = new_dim
                row["dimension_axis"] = DIMENSION_TO_AXIS.get(new_dim, row.get("dimension_axis", ""))
                row["notes"] = _append_note(row.get("notes") or "", tag or "relabel")
                changes.append(
                    {
                        "action": "relabel",
                        "desc": desc,
                        "from": old,
                        "to": new_dim,
                        "note": tag or "",
                    }
                )
            new_rows.append(row)
        else:
            # axis CSV: drop stereo/noise; if dim-side relabel known, sync axis
            old_ax = (row.get("assigned_axis") or "").strip()
            # Use dimension_axis hint if present; else treat via desc-only dim rules
            # Infer by running dim decision against empty old dim only for drop;
            # for relabel, map new dim → axis.
            action, new_dim, tag = decide_post_rule(desc, "")
            if action == "drop":
                changes.append(
                    {
                        "action": "drop",
                        "desc": desc,
                        "from": old_ax,
                        "to": "",
                        "note": tag or "",
                    }
                )
                continue
            row = dict(row)
            if action == "relabel" and new_dim:
                new_ax = DIMENSION_TO_AXIS[new_dim]
                if new_ax != old_ax:
                    row["assigned_axis"] = new_ax
                    row["dimension_axis"] = new_ax
                    row["notes"] = _append_note(row.get("notes") or "", tag or "relabel")
                    changes.append(
                        {
                            "action": "relabel",
                            "desc": desc,
                            "from": old_ax,
                            "to": new_ax,
                            "note": tag or "",
                        }
                    )
            new_rows.append(row)
    return new_rows, changes


def _load_csv_corrections(path: Path, corrected_key: str) -> dict[str, dict[str, str]]:
    """Load human corrected_* / notes keyed by term_key(target, desc)."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = term_key(row.get("target", ""), row.get("desc", ""))
            corr = (row.get(corrected_key) or "").strip()
            notes = (row.get("notes") or "").strip()
            if corr or notes:
                out[k] = {"corrected": corr, "notes": notes}
    return out


# ====================== paths ======================
def raw_jsonl_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"lexicon_raw_{split}.jsonl"


def aggregate_path(output_dir: Path) -> Path:
    return output_dir / "lexicon_raw_all.json"


def mapped_path(output_dir: Path, method: str) -> Path:
    return output_dir / f"lexicon_mapped_{method}.json"


def review_csv_path(output_dir: Path, method: str, level: str) -> Path:
    return output_dir / f"lexicon_review_{method}_{level}.csv"


def compare_csv_path(output_dir: Path, level: str) -> Path:
    return output_dir / f"lexicon_review_compare_{level}.csv"


# ====================== GGUF (delegates to labeling_gguf for Kaggle T4x2) ======================
# load_llm / CUDA preload live in labeling.labeling_gguf (multi-GPU tensor_split).


def call_llm(llm, system_prompt: str, user_message: str) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


# ====================== normalize / aggregate ======================
def normalize_target(value: str) -> str:
    v = str(value or "").strip().lower() or "unknown"
    return STEM_ALIASES.get(v, v)


def normalize_desc(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def term_key(target: str, desc: str) -> str:
    return f"{normalize_target(target)}||{normalize_desc(desc)}"


def parse_terms(parsed: dict) -> list[dict]:
    out = []
    for item in parsed.get("terms") or []:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("desc") or "").strip()
        if not desc:
            continue
        target = str(item.get("target") or "").strip() or "unknown"
        out.append({"target": target, "desc": desc})
    return out


def load_done_extract_keys(jsonl_path: Path) -> set[tuple]:
    if not jsonl_path.exists():
        return set()
    done: set[tuple] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            key = rec.get("key")
            if isinstance(key, list) and len(key) == 3:
                done.add(tuple(str(x) for x in key))
    return done


def aggregate_raw(output_dir: Path, splits: list[str]) -> dict:
    buckets: dict[str, dict] = {}
    for split in splits:
        path = raw_jsonl_path(output_dir, split)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                key_parts = rec.get("key") or []
                example = (
                    f"{key_parts[1]}:{key_parts[2]}"
                    if len(key_parts) > 2
                    else ""
                )
                for term in rec.get("terms") or []:
                    if not isinstance(term, dict):
                        continue
                    target = str(term.get("target") or "unknown")
                    desc = str(term.get("desc") or "").strip()
                    if not desc:
                        continue
                    k = term_key(target, desc)
                    if k not in buckets:
                        buckets[k] = {
                            "target": normalize_target(target),
                            "desc": desc,
                            "desc_norm": normalize_desc(desc),
                            "count": 0,
                            "examples": [],
                        }
                    buckets[k]["count"] += 1
                    ex = buckets[k]["examples"]
                    if example and example not in ex and len(ex) < 5:
                        ex.append(example)

    terms = sorted(buckets.values(), key=lambda t: (-t["count"], t["target"], t["desc_norm"]))
    payload = {
        "n_unique": len(terms),
        "n_mentions": sum(t["count"] for t in terms),
        "splits": splits,
        "terms": terms,
    }
    out = aggregate_path(output_dir)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aggregate] {payload['n_unique']} unique / {payload['n_mentions']} mentions → {out}")
    return payload


# ====================== extract ======================
def cmd_extract(args: argparse.Namespace) -> None:
    # Lazy import so map/export do not pull llama-cpp / CUDA preload.
    from labeling.labeling_gguf import load_llm, parse_tensor_split

    tensor_split = (
        args.tensor_split
        if isinstance(args.tensor_split, list)
        else parse_tensor_split(args.tensor_split)
    )
    llm = load_llm(
        args.model_path,
        args.n_ctx,
        args.n_gpu_layers,
        args.verbose,
        n_batch=args.n_batch,
        tensor_split=tensor_split,
        disable_tensor_split=args.no_tensor_split,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)

    system_prompt = SYSTEM_PROMPT.strip()
    for split in args.splits:
        jsonl = raw_jsonl_path(output_dir, split)
        if args.overwrite and jsonl.exists():
            jsonl.unlink()
            print(f"[{split}] --overwrite: 已刪除 {jsonl.name}")

        done = load_done_extract_keys(jsonl)
        if done:
            print(f"[{split}] 既有 {jsonl.name},已完成 {len(done)} turns,將跳過。")

        rows = load_split(split, args.limit, random_window=args.random_window)
        print(f"=== extract {split}: {len(rows)} turns → {jsonl.name} ===")

        for i, row in enumerate(rows, 1):
            key = (split, row.get("conversation_id", ""), row.get("turn_id", ""))
            tag = f"[{split} {i}/{len(rows)}] {key[1]} turn {key[2]}"
            if key in done:
                print(f"{tag} 已抽過,跳過")
                continue

            user_message = build_extract_user_message(row)
            try:
                t0 = time.perf_counter()
                raw_reply = call_llm(llm, system_prompt, user_message)
                terms = parse_terms(extract_json(raw_reply))
                append_raw(
                    jsonl,
                    {
                        "key": list(key),
                        "topic": row.get("topic", ""),
                        "terms": terms,
                        "user_message": user_message,
                        "raw_reply": raw_reply,
                    },
                )
                preview = [f"{t['target']}:{t['desc']}" for t in terms[:4]]
                more = f" +{len(terms) - 4}" if len(terms) > 4 else ""
                print(f"{tag} -> {len(terms)} term(s) {preview}{more} ({time.perf_counter() - t0:.1f}s)")
            except Exception as e:
                print(f"{tag} -> 錯誤: {e}")
                append_raw(jsonl, {"key": list(key), "error": str(e), "terms": []})

    existing = [s for s in SPLIT_FILES if raw_jsonl_path(output_dir, s).exists()]
    if existing:
        aggregate_raw(output_dir, existing)


# ====================== device ======================
def resolve_torch_device(requested: str) -> str:
    """auto: probe real CUDA alloc (is_available alone is unreliable here)."""
    if requested == "cpu":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        print("[device] CUDA not available → cpu")
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        print("[device] CUDA alloc ok → cuda")
        return "cuda"
    except Exception as e:
        if requested == "cuda":
            raise RuntimeError(f"CUDA unusable: {e}") from e
        print(f"[device] CUDA unusable → cpu ({e})")
        return "cpu"


def _centroid_vec(encode_fn, seeds: list[str]):
    import numpy as np

    vecs = np.asarray(encode_fn(seeds), dtype=np.float32)
    c = np.mean(vecs, axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


def _stack_centroids(encode_fn, seed_map: dict[str, list[str]], labels: tuple[str, ...]):
    import numpy as np

    return np.stack([_centroid_vec(encode_fn, seed_map[lab]) for lab in labels], axis=0)


def _hierarchical_assign(
    terms: list[dict],
    emb,
    axis_mat,
    dim_centroids: dict[str, object],
    method: str,
    min_sim: float,
) -> list[dict]:
    """Step1: nearest of 7 axes. Step2: nearest dim within that axis (masking/clean: 1)."""
    import numpy as np

    axis_sims = emb @ axis_mat.T  # (N, n_axes)
    mapped = []
    for i, term in enumerate(terms):
        axis_scores = {ax: float(axis_sims[i, j]) for j, ax in enumerate(AXES)}
        axis = max(axis_scores, key=axis_scores.get)
        axis_sim = axis_scores[axis]

        cand_dims = AXIS_TO_DIMS[axis]
        dim_scores: dict[str, float] = {}
        for dim in cand_dims:
            dim_scores[dim] = float(np.dot(emb[i], dim_centroids[dim]))
        # also record 0 for non-candidate dims (readable in CSV)
        for dim in DIMENSIONS:
            dim_scores.setdefault(dim, 0.0)
        dim = max((d for d in cand_dims), key=lambda d: dim_scores[d])
        dim_sim = dim_scores[dim]

        mapped.append(
            {
                **term,
                "method": method,
                "assigned_axis": axis,
                "axis_similarity": round(axis_sim, 4),
                "axis_scores": {k: round(v, 4) for k, v in axis_scores.items()},
                "axis_low_confidence": bool(axis_sim < min_sim),
                "assigned_dimension": dim,
                "dimension_similarity": round(dim_sim, 4),
                "dimension_scores": {k: round(v, 4) for k, v in dim_scores.items()},
                "dimension_low_confidence": bool(dim_sim < min_sim),
                "dimension_axis": axis,  # always matches assigned_axis (hierarchical)
            }
        )
    return mapped


def _build_label_centroids(encode_fn):
    axis_mat = _stack_centroids(encode_fn, SEED_WORDS_AXIS, AXES)
    dim_centroids = {dim: _centroid_vec(encode_fn, SEED_WORDS_DIM[dim]) for dim in DIMENSIONS}
    return axis_mat, dim_centroids


# ====================== map: centroid ======================
def map_centroid(
    terms: list[dict],
    embed_model: str,
    min_sim: float,
    batch_size: int,
    device: str,
) -> list[dict]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"[centroid] loading {embed_model} on {device} ...")
    model = SentenceTransformer(embed_model, device=device)

    def encode(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)

    print("[centroid] hierarchical: 7 axes → then dims within axis (12 classes)")
    axis_mat, dim_centroids = _build_label_centroids(encode)
    descs = [t["desc"] for t in terms]
    print(f"[centroid] embedding {len(descs)} unique descs ...")
    emb = np.asarray(encode(descs), dtype=np.float32)
    return _hierarchical_assign(terms, emb, axis_mat, dim_centroids, "centroid", min_sim)


# ====================== map: CLAP ======================
def _clap_encode_texts(model, processor, texts: list[str], device: str, batch_size: int):
    import numpy as np
    import torch

    outs = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
            feats = features.pooler_output if hasattr(features, "pooler_output") else features
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        outs.append(feats.cpu().numpy().astype(np.float32))
        print(f"[clap] encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(outs, axis=0)


def map_clap(
    terms: list[dict],
    clap_model: str,
    min_sim: float,
    batch_size: int,
    device: str,
) -> list[dict]:
    from transformers import ClapModel, ClapProcessor

    print(f"[clap] loading {clap_model} on {device} ...")
    processor = ClapProcessor.from_pretrained(clap_model)
    model = ClapModel.from_pretrained(clap_model).to(device)

    def encode(texts: list[str]):
        return _clap_encode_texts(model, processor, texts, device, batch_size)

    print("[clap] hierarchical: 7 axes → then dims within axis (12 classes)")
    axis_mat, dim_centroids = _build_label_centroids(encode)
    descs = [t["desc"] for t in terms]
    print(f"[clap] embedding {len(descs)} unique descs ...")
    emb = encode(descs)
    return _hierarchical_assign(terms, emb, axis_mat, dim_centroids, "clap", min_sim)


def cmd_map(args: argparse.Namespace) -> dict[str, list[dict]]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [s for s in SPLIT_FILES if raw_jsonl_path(output_dir, s).exists()]
    if not existing:
        sys.exit("找不到任何 lexicon_raw_*.jsonl。請先跑 extract。")

    terms = aggregate_raw(output_dir, existing)["terms"]
    if not terms:
        sys.exit("aggregate 為空（沒有 terms）。")

    device = resolve_torch_device(args.device)
    results: dict[str, list[dict]] = {}
    for method in resolve_methods(args.method):
        if method == "centroid":
            mapped = map_centroid(terms, args.embed_model, args.min_sim, args.batch_size, device)
            model_name = args.embed_model
        else:
            mapped = map_clap(terms, args.clap_model, args.min_sim, args.batch_size, device)
            model_name = args.clap_model

        mapped, post_stats = apply_post_rules(mapped)
        print(
            f"[post] {method}: kept={post_stats['kept']} "
            f"relabeled={post_stats['relabeled']} dropped={post_stats['dropped']}"
        )

        path = mapped_path(output_dir, method)
        path.write_text(
            json.dumps(
                {
                    "method": method,
                    "n_terms": len(mapped),
                    "model": model_name,
                    "seed_words_axis": SEED_WORDS_AXIS,
                    "seed_words_dim": SEED_WORDS_DIM,
                    "terms": mapped,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[map] wrote {path} ({len(mapped)} terms)")
        results[method] = mapped
    return results


# ====================== export ======================
REVIEW_FIELDS_AXIS = [
    "target",
    "desc",
    "count",
    "assigned_axis",
    "similarity",
    "low_confidence",
    "dimension_axis",
    "corrected_axis",
    "notes",
    "example_turn",
    "scores_json",
]

REVIEW_FIELDS_DIM = [
    "target",
    "desc",
    "count",
    "assigned_dimension",
    "similarity",
    "low_confidence",
    "dimension_axis",
    "corrected_dimension",
    "notes",
    "example_turn",
    "scores_json",
]


def _example_turn(term: dict) -> str:
    ex = term.get("examples") or []
    return ex[0] if ex else ""


def export_axis(mapped: list[dict], path: Path) -> None:
    prev = _load_csv_corrections(path, "corrected_axis")
    rows = []
    for t in sorted(
        mapped,
        key=lambda x: (x.get("assigned_axis", ""), -float(x.get("axis_similarity") or 0), x.get("desc", "")),
    ):
        k = term_key(t.get("target", ""), t.get("desc", ""))
        human = prev.get(k, {})
        note = human.get("notes") or ""
        if t.get("post_note"):
            note = _append_note(note, t["post_note"])
        rows.append(
            {
                "target": t.get("target", ""),
                "desc": t.get("desc", ""),
                "count": t.get("count", 0),
                "assigned_axis": t.get("assigned_axis", ""),
                "similarity": t.get("axis_similarity", ""),
                "low_confidence": t.get("axis_low_confidence", False),
                "dimension_axis": t.get("dimension_axis", ""),
                "corrected_axis": human.get("corrected", ""),
                "notes": note,
                "example_turn": _example_turn(t),
                "scores_json": json.dumps(t.get("axis_scores") or {}, ensure_ascii=False),
            }
        )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS_AXIS)
        w.writeheader()
        w.writerows(rows)
    n_corr = sum(1 for r in rows if r["corrected_axis"])
    print(f"[export] axis {len(rows)} rows (merged corrected={n_corr}) → {path}")


def export_dim(mapped: list[dict], path: Path) -> None:
    prev = _load_csv_corrections(path, "corrected_dimension")
    rows = []
    for t in sorted(
        mapped,
        key=lambda x: (
            x.get("assigned_dimension", ""),
            -float(x.get("dimension_similarity") or 0),
            x.get("desc", ""),
        ),
    ):
        k = term_key(t.get("target", ""), t.get("desc", ""))
        human = prev.get(k, {})
        note = human.get("notes") or ""
        if t.get("post_note"):
            note = _append_note(note, t["post_note"])
        rows.append(
            {
                "target": t.get("target", ""),
                "desc": t.get("desc", ""),
                "count": t.get("count", 0),
                "assigned_dimension": t.get("assigned_dimension", ""),
                "similarity": t.get("dimension_similarity", ""),
                "low_confidence": t.get("dimension_low_confidence", False),
                "dimension_axis": t.get("dimension_axis", ""),
                "corrected_dimension": human.get("corrected", ""),
                "notes": note,
                "example_turn": _example_turn(t),
                "scores_json": json.dumps(t.get("dimension_scores") or {}, ensure_ascii=False),
            }
        )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS_DIM)
        w.writeheader()
        w.writerows(rows)
    n_corr = sum(1 for r in rows if r["corrected_dimension"])
    print(f"[export] dim {len(rows)} rows (merged corrected={n_corr}) → {path}")


def export_compare(
    centroid_terms: list[dict],
    clap_terms: list[dict],
    path: Path,
    *,
    label_key: str,
    sim_key: str,
    level: str,
) -> None:
    by_c = {term_key(t["target"], t["desc"]): t for t in centroid_terms}
    by_p = {term_key(t["target"], t["desc"]): t for t in clap_terms}
    label_c = f"centroid_{level}"
    label_p = f"clap_{level}"
    corr_key = f"corrected_{level}"
    prev = _load_csv_corrections(path, corr_key)
    fields = [
        "target",
        "desc",
        "count",
        label_c,
        "centroid_sim",
        label_p,
        "clap_sim",
        "agree",
        corr_key,
        "notes",
        "example_turn",
    ]
    rows = []
    agree_n = 0
    for k in sorted(set(by_c) | set(by_p)):
        c, p = by_c.get(k), by_p.get(k)
        base = c or p or {}
        c_lab = (c or {}).get(label_key, "")
        p_lab = (p or {}).get(label_key, "")
        agree = bool(c_lab and p_lab and c_lab == p_lab)
        if agree:
            agree_n += 1
        human = prev.get(k, {})
        rows.append(
            {
                "target": base.get("target", ""),
                "desc": base.get("desc", ""),
                "count": base.get("count", 0),
                label_c: c_lab,
                "centroid_sim": (c or {}).get(sim_key, ""),
                label_p: p_lab,
                "clap_sim": (p or {}).get(sim_key, ""),
                "agree": agree,
                corr_key: human.get("corrected", ""),
                "notes": human.get("notes", ""),
                "example_turn": _example_turn(base),
            }
        )
    rows.sort(key=lambda r: (r["agree"], r.get(label_c, ""), -int(r.get("count") or 0), r.get("desc", "")))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(
        f"[export] compare_{level} {len(rows)} rows, agree={agree_n}/{len(rows)} "
        f"({100.0 * agree_n / max(len(rows), 1):.1f}%) → {path}"
    )


def _write_postfilter_diff(path: Path, changes: list[dict], level: str) -> None:
    lines = [f"# postfilter {level} changes: {len(changes)}", ""]
    by_act: dict[str, list[dict]] = {}
    for ch in changes:
        by_act.setdefault(ch["action"], []).append(ch)
    for act in ("drop", "relabel"):
        items = by_act.get(act, [])
        lines.append(f"## {act} ({len(items)})")
        for ch in items:
            if act == "drop":
                lines.append(f"- [{ch['from']}] {ch['desc']!r}  ({ch['note']})")
            else:
                lines.append(
                    f"- {ch['from']} → {ch['to']}: {ch['desc']!r}  ({ch['note']})"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[post] wrote diff → {path}")


def cmd_postfilter(args: argparse.Namespace) -> None:
    """Apply post-rules to existing review CSVs (no remap). Preserves corrected_*."""
    output_dir: Path = args.output_dir
    for method in resolve_methods(args.method):
        for level, fields, label_key in (
            ("dim", REVIEW_FIELDS_DIM, "assigned_dimension"),
            ("axis", REVIEW_FIELDS_AXIS, "assigned_axis"),
        ):
            path = review_csv_path(output_dir, method, level)
            if not path.exists():
                print(f"[post] skip missing {path}")
                continue
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            bak = path.with_suffix(".before_postfilter.csv")
            if not bak.exists():
                bak.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
                print(f"[post] backup → {bak}")
            new_rows, changes = apply_post_rules_csv_rows(rows, label_key=label_key)
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(new_rows)
            diff_path = output_dir / f"lexicon_postfilter_diff_{method}_{level}.md"
            _write_postfilter_diff(diff_path, changes, f"{method}/{level}")
            print(
                f"[post] {method}/{level}: {len(rows)} → {len(new_rows)} "
                f"(changes={len(changes)}) → {path}"
            )


def _load_mapped(output_dir: Path, method: str, cache: dict[str, list[dict]] | None) -> list[dict]:
    if cache and method in cache:
        return cache[method]
    path = mapped_path(output_dir, method)
    if not path.exists():
        sys.exit(f"找不到 {path}。請先跑 map --method {method}")
    return json.loads(path.read_text(encoding="utf-8"))["terms"]


def cmd_export(args: argparse.Namespace, mapped_cache: dict[str, list[dict]] | None = None) -> None:
    output_dir: Path = args.output_dir
    methods = resolve_methods(args.method)
    loaded = {m: _load_mapped(output_dir, m, mapped_cache) for m in methods}

    for method, terms in loaded.items():
        export_axis(terms, review_csv_path(output_dir, method, "axis"))
        export_dim(terms, review_csv_path(output_dir, method, "dim"))

    for m in ("centroid", "clap"):
        if m not in loaded and mapped_path(output_dir, m).exists():
            loaded[m] = _load_mapped(output_dir, m, None)

    if "centroid" in loaded and "clap" in loaded:
        export_compare(
            loaded["centroid"],
            loaded["clap"],
            compare_csv_path(output_dir, "axis"),
            label_key="assigned_axis",
            sim_key="axis_similarity",
            level="axis",
        )
        export_compare(
            loaded["centroid"],
            loaded["clap"],
            compare_csv_path(output_dir, "dim"),
            label_key="assigned_dimension",
            sim_key="dimension_similarity",
            level="dimension",
        )


def cmd_map_and_export(args: argparse.Namespace) -> None:
    cmd_export(args, mapped_cache=cmd_map(args))


# ====================== CLI ======================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MixAssist blind lexicon: extract → map → export")
    sub = p.add_subparsers(dest="command", required=True)

    def add_io(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    pe = sub.add_parser("extract", help="Gemma blind (target, desc) extraction")
    add_io(pe)
    pe.add_argument("--splits", nargs="+", choices=list(SPLIT_FILES), default=list(SPLIT_FILES))
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--random-window", action="store_true")
    pe.add_argument("--seed", type=int, default=None)
    pe.add_argument("--model-path", type=Path, default=DEFAULT_GGUF)
    pe.add_argument("--n-ctx", type=int, default=N_CTX)
    pe.add_argument("--n-batch", type=int, default=N_BATCH)
    pe.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS)
    pe.add_argument(
        "--tensor-split",
        type=str,
        default=None,
        help="Multi-GPU proportions, e.g. 0.5,0.5 (Kaggle T4x2). Auto when >=2 GPUs.",
    )
    pe.add_argument(
        "--no-tensor-split",
        action="store_true",
        help="Force single-GPU (disable auto multi-GPU split)",
    )
    pe.add_argument("--verbose", action="store_true")
    pe.add_argument("--overwrite", action="store_true")

    def add_map_args(sp: argparse.ArgumentParser) -> None:
        add_io(sp)
        sp.add_argument(
            "--method",
            choices=METHODS,
            default="both",
            help="centroid | clap | both (default both)",
        )
        sp.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
        sp.add_argument("--clap-model", default=DEFAULT_CLAP_MODEL)
        sp.add_argument("--min-sim", type=float, default=0.25, help="low_confidence cosine threshold")
        sp.add_argument("--batch-size", type=int, default=32)
        sp.add_argument(
            "--device",
            choices=("auto", "cuda", "cpu"),
            default="auto",
            help="map device (auto falls back to cpu if CUDA alloc fails)",
        )

    add_map_args(sub.add_parser("map", help="Map descs: 7 axes → 12 MixJudge classes (11+clean)"))
    px = sub.add_parser("export", help="Write axis/dim review CSV(s)")
    add_io(px)
    px.add_argument("--method", choices=METHODS, default="both")
    add_map_args(sub.add_parser("map-and-export", help="map then export"))

    pp = sub.add_parser(
        "postfilter",
        help="Apply drop/relabel rules to existing review CSVs (no remap)",
    )
    add_io(pp)
    pp.add_argument("--method", choices=METHODS, default="centroid")
    return p


def main() -> None:
    args = build_parser().parse_args()
    {
        "extract": cmd_extract,
        "map": cmd_map,
        "export": cmd_export,
        "map-and-export": cmd_map_and_export,
        "postfilter": cmd_postfilter,
    }[args.command](args)


if __name__ == "__main__":
    main()

