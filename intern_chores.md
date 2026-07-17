# Intern Chores — MixJudge-CLAP Text Pipeline

**Ultimate goal:** Exploit MixAssist to bridge the gap between our synthesised mixing data and real mixing conversation — and where applicable, the real DAW and audio evidence behind it.

---

## Background

Our training data is synthetically generated: DSP recipes degrade audio stems, and structured L1 text describes the problem. The descriptions are correct and consistent, but they sound like a textbook — not like a real mixing session.

MixAssist is a real mixing dialogue corpus: amateurs and experts discussing mixing problems and fixes in session, with accompanying audio and DAW operations. It is the closest available resource to how mixing problems are actually described in practice.

**The intern's job** is to design a system that retrieves relevant MixAssist dialogue turns per mixing dimension and direction, and uses them to generate richer, more natural L2 and L3 text fields for our training records.

---

## Reference: Jason's proof-of-concept retrieval

Before handing this to the intern, a small free LLaMA model was used to retrieve (problem, fix) pairs from MixAssist dialogues. The outputs are:

- `problem_fix_meaningful.csv` — retrieved turns with at least one meaningful statement
- `problem_fix_gold_pairs.csv` — retrieved turns with both a problem and a fix statement

These CSVs are **given to the intern as reference** — showing what relevant retrieved turns look like and demonstrating that retrieval from MixAssist is feasible. They are not a data source to pull from directly. The intern's task is to design a more rigorous, scalable retrieval system.

---

## The Text Layer System

| Layer | Who | What |
|---|---|---|
| **L1** | Jason | Structured template. Simple, deterministic, correct. Training anchor — never modified. |
| **L2** | Intern | Problem description in real mixing dialogue style. Produced via retrieval from MixAssist + style-matched generation. |
| **L3** | Intern | L2 + natural fix language from the same retrieved dialogue turn. |

L2 and L3 are new fields added to `records.jsonl`. L1 is never touched.

---

## The 9 Retrieval Keys (dim × direction)

Retrieval is done per key — each key corresponds to a specific perceptual concept in our data:

| Key | Recipes covered | Perceptual concept |
|---|---|---|
| `level_low` | `gain_static/low`, `phrase_envelope/low` | Stem too quiet — buried, can't hear it |
| `level_high` | `gain_static/high`, `phrase_envelope/high` | Stem too loud — dominant, overpowering |
| `mud` | `low_shelf_boost`, `low_mid_bell`, `proximity_buildup`, `dark_reverb` | Too much low-mid energy — thick, cloudy, boomy |
| `harshness` | `upper_mid_bell`, `high_shelf`, `soft_clip` | Too harsh / bright / gritty — fatiguing, piercing |
| `space_wet` | `reverb_send`, `large_room` | Too much reverb — washed-out, distant |
| `space_dry` | `spectral_dereverb` | Too dry / dead — no air, no room |
| `masking` | `submix_clash`, `presence_cut` | Stem obscured in mix — can't cut through |
| `dynamics_over` | `over_compress`, `over_limit`, `parallel_compress` | Over-processed dynamics — squashed, flat, no punch |
| `dynamics_under` | `under_compress` | Under-controlled dynamics — inconsistent, lurching |

---

## Phase 1 — Study MixAssist (1–2 weeks)

**What to study:**
- Full MixAssist dataset: raw dialogue turns, audio files, DAW operation logs
- Reference CSVs: read to understand what retrieved turns look like and what makes a turn useful

**Goals:**
- Deeply understand how people describe mixing problems and fixes in natural language
- Understand the amateur vs expert vocabulary divide
- Understand how the MixAssist audio and DAW data is structured — what operations are logged, how they relate to the dialogue
- Form a clear sense of which dialogue turns are relevant to each of our 9 keys

---

## Phase 2 — Design and Build the Retrieval System

**Task:** Given each of our 9 (dim × direction) keys, retrieve a pool of relevant raw MixAssist dialogue turns.

**What to retrieve:** Raw dialogue turns — not pre-extracted sentences. Each turn may contain a problem statement, a fix statement, or both. Problem and fix always come from the same turn.

**What to propose and justify:**
- Indexing strategy: how to represent MixAssist turns for retrieval
- Retrieval method: embedding model choice, similarity metric, ranking
- How to handle turn quality — not all retrieved turns will be useful
- Evaluation: how to verify the retrieved pool is actually relevant to each key

Jason's LLaMA retrieval is the baseline to beat or compare against.

**Output:** A retrieval pool per key — a ranked list of MixAssist turns relevant to that perceptual concept.

---

## Phase 3 — Design and Build L2/L3 Generation

**Task:** Use retrieved dialogue turns to generate L2 and L3 text for each training record.

**L2:** Given a record's L1 text and a pool of retrieved turns for its key, generate a natural problem description in real mixing dialogue style. The retrieved turns ground the style — the output should sound like it came from a real session.

**L3:** Extend L2 with a fix statement, drawn from the same retrieved turn as the problem. Problem and fix are always paired from the same source turn.

**What to propose and justify:**
- Generation method: how retrieved turns are used (few-shot prompting, style transfer, etc.)
- How to preserve L1 facts (instrument, direction, severity) while adopting dialogue style
- Evaluation: does L2 actually close the gap toward MixAssist language?

**Output:** Updated `records.jsonl` with `l2_text` and `l3_text` fields populated.

---

## Phase 4 — Audio and DAW Evidence (if applicable)

Study the audio and DAW operation parts of MixAssist. If the structure supports it, use real audio or operation evidence to further inform or validate the retrieval — for example, confirming that retrieved turns correspond to actual mixing decisions in the session.

This is a bonus contribution, not a hard deliverable. It depends on how the MixAssist audio/DAW data is structured and accessible.

---

## Out of Scope for Intern

- L1 template writing (`mixstate/data/templates.py`) — Jason owns, DSP-coupled
- DSP recipe implementation and parameters
- Applicability logic (`mixstate/data/applicability.py`)
- Dataset generation script (`scripts/generate_dataset.py`)
- Model architecture, training loop, loss function
