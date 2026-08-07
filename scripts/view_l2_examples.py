#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a local HTML side-by-side viewer for L2 raw variants.

Usage (from repo root):
  python scripts/view_l2_examples.py
  # writes outputs/l2_examples_viewer.html — open in browser

  python scripts/view_l2_examples.py --open   # also try xdg-open
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import webbrowser
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("retarget_raw", "strict_raw", "free_raw")
MODES = ("retarget", "strict", "free")


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def jaccard(a: str, b: str) -> float:
    A, B = toks(a), toks(b)
    return round(len(A & B) / len(A | B), 3) if A | B else 0.0


def load_rows(tag: str) -> dict[str, dict]:
    path = ROOT / "outputs" / f"l2_from_l1_{tag}.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["l1_segment_id"]: r for r in csv.DictReader(f)}


def load_labels(tag: str) -> dict[str, list[dict]]:
    path = ROOT / "outputs" / f"labeled_l2_gguf_{tag}.csv"
    out: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["conversation_id"]].append(r)
    return out


def build_payload() -> dict:
    rows = {v: load_rows(v) for v in VARIANTS}
    labs = {v: load_labels(v) for v in VARIANTS}
    ids = list(rows["retarget_raw"].keys())
    out_rows = []
    for sid in ids:
        r0 = rows["retarget_raw"][sid]
        item = {
            "id": sid,
            "gold_dim": r0["gold_dim"],
            "gold_axis": r0["gold_axis"],
            "l1_text": r0["l1_text"],
            "source": r0["source_instrument"],
            "style_turn": r0["style_turn_ids"],
            "style_problem_text": r0.get("style_problem_texts") or "",
            "style_user": r0.get("style_user_raw_contents") or "",
            "style_asst": r0.get("style_assistant_raw_contents") or "",
            "modes": {},
        }
        for v in VARIANTS:
            r = rows[v][sid]
            mode = v.replace("_raw", "")
            gen_u = r.get("amateur_text") or ""
            gen_a = r.get("expert_text") or ""
            style = (
                f"{r.get('style_user_raw_contents', '')} "
                f"{r.get('style_assistant_raw_contents', '')}"
            )
            preds = labs[v].get(sid, [])
            dims = [
                (p.get("problem_dimension") or "")
                for p in preds
                if (p.get("problem_dimension") or "")
                and p.get("problem_dimension") != "none"
            ]
            if not dims and preds:
                dims = ["none"]
            item["modes"][mode] = {
                "amateur": gen_u,
                "expert": gen_a,
                "jaccard": jaccard(f"{gen_u} {gen_a}", style),
                "pred_dims": dims,
                "dim_any": r0["gold_dim"]
                in [p.get("problem_dimension") for p in preds],
            }
        out_rows.append(item)

    summary = {}
    for mode in MODES:
        hits = sum(1 for x in out_rows if x["modes"][mode]["dim_any"])
        js = [x["modes"][mode]["jaccard"] for x in out_rows]
        summary[mode] = {
            "dim_any": f"{hits}/{len(out_rows)}",
            "jaccard_mean": round(sum(js) / len(js), 2) if js else 0.0,
        }
    return {"summary": summary, "rows": out_rows}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>L2 examples viewer</title>
<style>
  :root {
    --bg: #f7f6f3;
    --fg: #1a1a1a;
    --muted: #5c5c5c;
    --line: #d9d6cf;
    --card: #fff;
    --ok: #1b6b3a;
    --bad: #8b1e1e;
    --accent: #0b3d5c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 14px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--fg);
  }
  header {
    padding: 16px 20px; border-bottom: 1px solid var(--line);
    background: var(--card); position: sticky; top: 0; z-index: 2;
  }
  h1 { margin: 0 0 6px; font-size: 20px; color: var(--accent); }
  .meta { color: var(--muted); font-size: 12px; }
  .stats { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
  .stat {
    border: 1px solid var(--line); background: var(--bg);
    padding: 8px 12px; min-width: 160px;
  }
  .stat b { display: block; font-size: 16px; }
  .controls {
    display: flex; gap: 10px; flex-wrap: wrap; padding: 12px 20px;
    align-items: center;
  }
  select, button {
    font: inherit; padding: 6px 10px; border: 1px solid var(--line);
    background: var(--card);
  }
  main { padding: 0 20px 40px; max-width: 1400px; }
  .l1 { margin: 8px 0 14px; font-weight: 600; }
  .grid { display: grid; gap: 12px; }
  .grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
  .grid.cols-1 { grid-template-columns: 1fr; }
  @media (max-width: 980px) {
    .grid.cols-3 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card); border: 1px solid var(--line); padding: 12px;
  }
  .card h3 {
    margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted);
    display: flex; justify-content: space-between; gap: 8px; align-items: center;
  }
  .pill {
    font-size: 11px; padding: 2px 7px; border: 1px solid var(--line);
    white-space: nowrap;
  }
  .pill.ok { color: var(--ok); border-color: var(--ok); }
  .pill.bad { color: var(--bad); border-color: var(--bad); }
  .label { color: var(--muted); font-size: 11px; margin-top: 8px; font-weight: 600; }
  pre {
    white-space: pre-wrap; word-break: break-word; margin: 4px 0 0;
    font: 13px/1.4 "IBM Plex Mono", ui-monospace, monospace;
  }
  .style { margin-bottom: 14px; }
</style>
</head>
<body>
<header>
  <h1>L2 examples: retarget / strict / free</h1>
  <div class="meta">Generated locally from outputs/l2_from_l1_*_raw.csv · click segment to compare</div>
  <div class="stats" id="stats"></div>
</header>
<div class="controls">
  <label>Filter
    <select id="filter">
      <option value="all">All segments</option>
      <option value="retarget_fail">retarget dim miss</option>
      <option value="any_fail">any mode miss</option>
    </select>
  </label>
  <label>Segment <select id="seg"></select></label>
  <label>Modes
    <select id="modeView">
      <option value="all">All modes</option>
      <option value="retarget">retarget</option>
      <option value="strict">strict</option>
      <option value="free">free</option>
    </select>
  </label>
  <button type="button" id="prev">Prev</button>
  <button type="button" id="next">Next</button>
</div>
<main id="main"></main>
<script>
const DATA = __DATA__;
const MODES = ["retarget", "strict", "free"];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));
}

function filtered() {
  const f = document.getElementById("filter").value;
  return DATA.rows.filter(r => {
    if (f === "retarget_fail") return !r.modes.retarget.dim_any;
    if (f === "any_fail") return !(r.modes.retarget.dim_any && r.modes.strict.dim_any && r.modes.free.dim_any);
    return true;
  });
}

function fillStats() {
  const el = document.getElementById("stats");
  el.innerHTML = MODES.map(m => {
    const s = DATA.summary[m];
    return `<div class="stat"><span>${m}</span><b>${esc(s.dim_any)} · j=${s.jaccard_mean}</b></div>`;
  }).join("");
}

function fillSegSelect() {
  const rows = filtered();
  const sel = document.getElementById("seg");
  const cur = sel.value;
  sel.innerHTML = rows.map(r =>
    `<option value="${esc(r.id)}">${esc(r.gold_dim)} · ${esc(r.id.replace("mock_","").replace("_001",""))}</option>`
  ).join("");
  if (rows.some(r => r.id === cur)) sel.value = cur;
  else if (rows[0]) sel.value = rows[0].id;
}

function modeCard(mode, out, gold) {
  const hit = out.dim_any ? "ok" : "bad";
  return `<div class="card">
    <h3>${esc(mode)}
      <span>
        <span class="pill ${hit}">${out.dim_any ? "dim hit" : "dim miss"}</span>
        <span class="pill">j=${out.jaccard.toFixed(2)}</span>
      </span>
    </h3>
    <div class="label">Amateur</div><pre>${esc(out.amateur)}</pre>
    <div class="label">Expert</div><pre>${esc(out.expert)}</pre>
    <div class="label">pred: ${esc((out.pred_dims||[]).join(", ") || "(none)")} · gold: ${esc(gold)}</div>
  </div>`;
}

function render() {
  fillSegSelect();
  const rows = filtered();
  const id = document.getElementById("seg").value;
  const active = rows.find(r => r.id === id) || rows[0];
  const main = document.getElementById("main");
  if (!active) { main.innerHTML = "<p>No rows.</p>"; return; }
  const mv = document.getElementById("modeView").value;
  const modes = mv === "all" ? MODES : [mv];
  main.innerHTML = `
    <h2>${esc(active.gold_axis)}/${esc(active.gold_dim)} · ${esc(active.source)}</h2>
    <div class="l1">${esc(active.l1_text)}</div>
    <div class="meta">style turn: ${esc(active.style_turn)}</div>
    <div class="card style">
      <h3>MixAssist style exemplar (shared)</h3>
      <div class="label">problem_text</div><pre>${esc(active.style_problem_text)}</pre>
      <div class="label">Amateur (raw)</div><pre>${esc(active.style_user)}</pre>
      <div class="label">Expert (raw)</div><pre>${esc(active.style_asst)}</pre>
    </div>
    <div class="grid cols-${modes.length === 1 ? 1 : 3}">
      ${modes.map(m => modeCard(m, active.modes[m], active.gold_dim)).join("")}
    </div>`;
}

function shift(delta) {
  const rows = filtered();
  const sel = document.getElementById("seg");
  const i = rows.findIndex(r => r.id === sel.value);
  if (i < 0) return;
  const n = (i + delta + rows.length) % rows.length;
  sel.value = rows[n].id;
  render();
}

fillStats();
["filter","seg","modeView"].forEach(id =>
  document.getElementById(id).addEventListener("change", render)
);
document.getElementById("prev").onclick = () => shift(-1);
document.getElementById("next").onclick = () => shift(1);
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") shift(-1);
  if (e.key === "ArrowRight") shift(1);
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "l2_examples_viewer.html",
    )
    ap.add_argument("--open", action="store_true", help="Open in default browser")
    args = ap.parse_args()

    payload = build_payload()
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")
    print("Summary:", json.dumps(payload["summary"], ensure_ascii=False))
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
