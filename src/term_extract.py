# -*- coding: utf-8 -*-
"""
MixAssist blind lexicon pipeline.

Stages:
  1) extract  — Gemma (GGUF) dumps free-form (target, desc) pairs
  2) map      — hierarchical assign (centroid / clap):
                  first 8 axes, then within that axis split into its 2 dims
                  (phase has only 1 dim)
  3) export   — review CSVs for axis and dimension (corrected_* empty)

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
import os
import random
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

AXES = (
    "level",
    "body",
    "brightness",
    "space",
    "dynamic",
    "masking",
    "stereo",
    "phase",
)

# Same 15 signed dims as labeling (order stable for scoring matrices).
DIMENSIONS = tuple(DIMENSION_TO_AXIS.keys())

# axis → its signed dims (usually 2; phase → 1)
AXIS_TO_DIMS: dict[str, tuple[str, ...]] = {}
for _dim, _axis in DIMENSION_TO_AXIS.items():
    AXIS_TO_DIMS.setdefault(_axis, [])
    AXIS_TO_DIMS[_axis].append(_dim)
AXIS_TO_DIMS = {a: tuple(AXIS_TO_DIMS[a]) for a in AXES}

# 8-axis seed anchors
SEED_WORDS_AXIS = {
    "level": [
        "loud", "quiet", "too loud", "too quiet",
        "overpowering", "inaudible", "can't hear", "volume",
    ],
    "body": [
        "muddy", "muddiness", "mud", "thin", "boomy",
        "full", "lacks body", "low-end mud", "thick",
    ],
    "brightness": [
        "harsh", "bright", "too bright", "dull", "dark",
        "sibilant", "piercing", "sparkly", "shimmer", "high end",
    ],
    "space": [
        "dry", "too dry", "wet", "too wet", "washed out",
        "distant", "airy", "in a space", "closely miked", "too much reverb",
    ],
    "dynamic": [
        "punchy", "flat", "squashed", "pumping", "tight",
        "pinched", "over compressed", "no dynamics", "transient", "clicky",
    ],
    "masking": [
        "masked", "overcrowded", "drowned", "gets lost", "buried"
        "bleeding", "getting in the way", "covered by", "swamping", "muffled",
    ],
    "stereo": [
        "wide", "narrow", "too wide", "too narrow",
        "panned", "mono", "centered", "to the side", "left and right",
    ],
    "phase": [
        "phasing", "phasing issues", "phasey", "out of phase",
        "phase cancellation", "not time aligned", "canceled", "polarity",
    ],
}

# 15-dimension seed anchors (aligned with labeling DIMENSION_TO_AXIS)
SEED_WORDS_DIM: dict[str, list[str]] = {
    "too_quiet": ["too quiet", "quiet", "buried", "inaudible", "too soft", "can't hear"],
    "too_loud": ["too loud", "loud", "overpowering", "way too loud", "front"],
    "muddy": ["muddy", "muddiness", "mud", "boomy", "cloudy", "thick low end"],
    "thin": ["thin", "skinny", "anemic", "lacks body", "no weight"],
    "harsh": ["harsh", "harshness", "sibilant", "piercing", "too bright", "presence", "sparkle"],
    "dull": ["dull", "dark", "lacks air", "lifeless top", "no sparkle"],
    "too_wet": ["too wet", "washed out", "too much reverb", "big reverb"],
    "too_dry": ["too dry", "dry", "no reverb", "boxy", "needs space"],
    "over_compressed": ["over compressed", "squashed", "pumping", "no dynamics", "flat"],
    "under_compressed": ["under compressed", "uncontrolled", "jumpy", "too dynamic", "needs compression"],
    "swamping": ["swamping", "drowned", "covered by", "overcrowded", "masking the vocal"],
    "invading": ["invading", "sticking out", "poking through", "cutting through too much"],
    "too_wide": ["too wide", "wide", "overspread", "spread too much", "spacey"],
    "too_narrow": ["too narrow", "narrow", "mono", "too centered"],
    "phase": ["phasey", "hollow", "canceled", "phase cancellation", "polarity"],
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


# ====================== GGUF ======================
def _prepare_cuda_libs() -> None:
    """Preload CUDA 12 libs for llama-cpp (same as labeling_gguf)."""
    import ctypes

    nvidia_root = Path.home() / ".local/lib/python3.12/site-packages/nvidia"
    must = [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        nvidia_root / "cuda_runtime/lib/libcudart.so.12",
        nvidia_root / "nvjitlink/lib/libnvJitLink.so.12",
        nvidia_root / "cublas/lib/libcublasLt.so.12",
        nvidia_root / "cublas/lib/libcublas.so.12",
    ]
    extras = [str(p) for p in nvidia_root.glob("*/lib")]
    host = ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
    old = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "cuda/compat" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(host + extras + old))
    for path in must:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load_llm(model_path: Path, n_ctx: int, n_gpu_layers: int, verbose: bool):
    from llama_cpp import Llama

    print(f"載入 GGUF → GPU: {model_path} (n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=N_BATCH,
        n_ubatch=N_BATCH,
        verbose=verbose,
    )
    print("模型載入完成。")
    return llm


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
    _prepare_cuda_libs()
    llm = load_llm(args.model_path, args.n_ctx, args.n_gpu_layers, args.verbose)
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
    """Step1: nearest of 8 axes. Step2: nearest of that axis's 2 dims (phase: 1)."""
    import numpy as np

    axis_sims = emb @ axis_mat.T  # (N, 8)
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

    print("[centroid] hierarchical: 8 axes → then 2 dims within axis")
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

    print("[clap] hierarchical: 8 axes → then 2 dims within axis")
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
    rows = [
        {
            "target": t.get("target", ""),
            "desc": t.get("desc", ""),
            "count": t.get("count", 0),
            "assigned_axis": t.get("assigned_axis", ""),
            "similarity": t.get("axis_similarity", ""),
            "low_confidence": t.get("axis_low_confidence", False),
            "dimension_axis": t.get("dimension_axis", ""),
            "corrected_axis": "",
            "notes": "",
            "example_turn": _example_turn(t),
            "scores_json": json.dumps(t.get("axis_scores") or {}, ensure_ascii=False),
        }
        for t in sorted(
            mapped,
            key=lambda x: (x.get("assigned_axis", ""), -float(x.get("axis_similarity") or 0), x.get("desc", "")),
        )
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS_AXIS)
        w.writeheader()
        w.writerows(rows)
    print(f"[export] axis {len(rows)} rows → {path}")


def export_dim(mapped: list[dict], path: Path) -> None:
    rows = [
        {
            "target": t.get("target", ""),
            "desc": t.get("desc", ""),
            "count": t.get("count", 0),
            "assigned_dimension": t.get("assigned_dimension", ""),
            "similarity": t.get("dimension_similarity", ""),
            "low_confidence": t.get("dimension_low_confidence", False),
            "dimension_axis": t.get("dimension_axis", ""),
            "corrected_dimension": "",
            "notes": "",
            "example_turn": _example_turn(t),
            "scores_json": json.dumps(t.get("dimension_scores") or {}, ensure_ascii=False),
        }
        for t in sorted(
            mapped,
            key=lambda x: (
                x.get("assigned_dimension", ""),
                -float(x.get("dimension_similarity") or 0),
                x.get("desc", ""),
            ),
        )
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS_DIM)
        w.writeheader()
        w.writerows(rows)
    print(f"[export] dim {len(rows)} rows → {path}")


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
    fields = [
        "target",
        "desc",
        "count",
        label_c,
        "centroid_sim",
        label_p,
        "clap_sim",
        "agree",
        f"corrected_{level}",
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
                f"corrected_{level}": "",
                "notes": "",
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
    pe.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS)
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

    add_map_args(sub.add_parser("map", help="Map descs: 8 axes, then 2 dims within axis"))
    px = sub.add_parser("export", help="Write axis/dim review CSV(s)")
    add_io(px)
    px.add_argument("--method", choices=METHODS, default="both")
    add_map_args(sub.add_parser("map-and-export", help="map then export"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    {"extract": cmd_extract, "map": cmd_map, "export": cmd_export, "map-and-export": cmd_map_and_export}[
        args.command
    ](args)


if __name__ == "__main__":
    main()

