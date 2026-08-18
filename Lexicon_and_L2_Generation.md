# Task

## Lexicon / Quality Extraction

The final result lives in `outputs/quality_from_manual/quality_from_manual.csv` and `quality_from_manual.json`. Both files split each dimension into **degree compatible** vs **standalone** wording (plus a scope slot where one exists). That's the actual wording menu we care about.

How it gets there, starting from MixAssist turns:

```text
MixAssist turns (train / validation / test)
        │
        │  Google Gemma 4 dumps (target, desc) pairs
        │  NO axis/dim names in this prompt — just "what sounds like what"
        ▼
lexicon_raw_all.json
        │
        │  map: centroid (sentence-transformers)
        │  first the 7 axes, then split into dims inside that axis
        ▼
lexicon_review_centroid_dim.csv   (and the axis-level version)
        │
        │  extract_lexicon.py — top-N unique descs per dim
        ▼
lexicon_top_centroid_dim.json
        │
        │  human pass → pick / rewrite / throw out junk
        ▼
lexicon_top_centroid_manual.json
        │
        │  another human pass → degree_compatible vs standalone
        ▼
quality_from_manual.json / .csv
```

`lexicon_raw_all` is extracted from MixAssist by pulling every subject + description of that subject, regardless of which dimension or axis it belongs to, or even whether it's a real problem at all. Workflow talk ("open the EQ") and empty "sounds good" get dropped. The extraction prompt is deliberately kept blind to our taxonomy so it doesn't just parrot back `too_loud` / `muddy`.

The next step is mapping. Each free-form MixAssist phrase gets assigned to one of the 12 classes (11 problems + `clean`) by nearest-seed similarity.

Principle (`src/term_extract.py`):

- Every axis and every dim has a hand-written **seed list** (`SEED_WORDS_AXIS` / `SEED_WORDS_DIM`). These are short prototype phrases, not the raw MixAssist dumps — e.g. level seeds are `loud` / `too quiet` / `can't hear`; masking seeds are competition words (`masked`, `covered by`, `swamping`) and deliberately skip bare loudness words so they don't collapse into `too_quiet`. Seeds got edited whenever the review CSV showed a "magnet" — a seed pulling in the wrong kind of desc (e.g. `lacks body` dragging "gives it some body" into `thin`).
- A sentence-transformers model embeds those seeds. For each label, the **centroid** is the mean of its seed vectors, then L2-normalized — that's the class prototype in embedding space.
- Assignment is hierarchical: embed the MixAssist `desc` → find the nearest of the **7 axis** centroids (cosine / dot product on unit vectors) → then find the nearest **dim centroid inside that axis only**. `masking` / `clean` each have one dim, so picking the axis is the same as picking the dim. Similarity below `min_sim` gets flagged low-confidence, not dropped.
- After that, a small regex post-filter fixes obvious polarity / "wish" mistakes (`needs to come up` → `too_quiet`) and drops stereo/noise phrasing. Human overrides live in `corrected_*`; the machine's guess is `assigned_*`.

That dump is `lexicon_review_centroid_dim.csv`. `extract_lexicon.py` then takes the human `corrected_*` value if it's filled in, otherwise falls back to `assigned_*`, dedupes exact desc strings, and keeps the top-N unique descs per dim.

From there, I manually pulled a cleaner list into `lexicon_top_centroid_manual.json`, and from that into `quality_from_manual` — that's the actual Quality Lexicon.

**quality_from_manual.json — wording counts per dimension**


| dimension        | degree_compatible | standalone | total |
| ---------------- | ----------------- | ---------- | ----- |
| too_quiet        | 12                | 11         | 23    |
| too_loud         | 20                | 25         | 45    |
| muddy            | 10                | 5          | 15    |
| thin             | 4                 | 5          | 9     |
| harsh            | 16                | 5          | 21    |
| dull             | 8                 | 9          | 17    |
| too_wet          | 11                | 11         | 22    |
| too_dry          | 5                 | 3          | 8     |
| over_compressed  | 10                | 10         | 20    |
| under_compressed | 15                | 11         | 26    |
| masking          | 3                 | 27         | 30    |
| clean            | 12                | 1          | 13    |
| Total            | 126               | 123        | 249   |


The Quality Lexicon feeds L1 captions. L1 captions then feed L2 generation.

---

## Labeling MixAssist (before generation)

MixAssist turns get labeled with the **7 axes / 12 classes** (11 problems + `clean`). The model only outputs `problem_dimension`; axis is looked up in code.


| axis       | dimensions                                        |
| ---------- | ------------------------------------------------- |
| level      | too_quiet, too_loud                               |
| body       | muddy, thin                                       |
| brightness | harsh, dull                                       |
| space      | too_wet, too_dry                                  |
| dynamic    | over_compressed, under_compressed                 |
| masking    | masking (no opposite)                             |
| clean      | clean (fault-free state, not vague "sounds good") |


`none` = no supported problem. One turn can explode into multiple rows if there are two clear problems.

The key rules baked into the prompt:

- Be conservative. Workflow / preference talk / "add saturation for warmth" with no stated defect → not a problem.
- CURRENT TURN must be enough to pick the dimension by itself. If this turn without history could be more than one class (or none), label `none`. HISTORY only helps pronouns / which stem — it cannot break a dim tie.
- `problem_stem` = the thing that sounds wrong / is getting covered up. `fix_stem` = what actually gets adjusted. These can differ (the snare's lost, but you pull the cymbal).
- The subject of the complaint in MixAssist can be anything. For MixJudge captions later, the vocal is always the subject.
- Polarity: "needs more reverb" is `too_dry`, never `too_wet`. Wanting more punch/snap is `under_compressed`, not over. Reverb send talk is SPACE, not LEVEL.
- `too_quiet` vs `masking`: no named competing source → `too_quiet`. Don't use bare "quiet/soft" for masking.
- `vocal_lead` is only true for the lead vocal, not backing vocals / doubles.

The pool we actually sample from: `outputs/labeled_turns_gguf_all_problems.csv` (`has_problem` true, `dim` ≠ `none`, non-empty `problem_text`).

**labeled_turns_gguf_all_problems.csv — counts per dimension**


| dimension        | count |
| ---------------- | ----- |
| too_quiet        | 44    |
| too_loud         | 54    |
| muddy            | 41    |
| thin             | 7     |
| harsh            | 5     |
| dull             | 7     |
| too_wet          | 13    |
| too_dry          | 26    |
| over_compressed  | 4     |
| under_compressed | 33    |
| masking          | 34    |
| clean            | 15    |
| Total            | 283   |


---

## Generation

L1 is composed as:

```text
The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? .
```

Always about the vocal. `SUBJECT` is randomly picked from `the lead vocal` / `the singer` / `the vocal` / `the voice`. `COPULA` is randomly `is` / `sounds`. `QUALITY` is randomly picked from `quality_from_manual` **degree_compatible** only (not standalone). `SCOPE` is attached when that dim has one.

No `clean` in this L1 set. 11 dims × 20 captions × 3 modes.

Generation (`src/generate_l2_style.py` + `src/prompts/l2_generation_prompt.py`) also gets a random same-dim MixAssist turn as a style exemplar (raw Amateur/Expert lines). If a dim has no pool rows, it still generates from the L1 caption alone.

It then writes Amateur / Expert JSON under three rewrite policies:

- **retarget** — near-copy of the MixAssist turn; only swap the instrument words onto the L1 vocal. Keep fillers, length, hedges. Don't invent a cute short dialogue from the L1 sentence if an exemplar exists.  
The temperature of the inferencing is set to 0.1.
- **strict** — stay close to the exemplar's rhythm/tone; swap the party to the vocal; don't copy extra faults from the surrounding chatter.  
The temperature of the inferencing is set to 0.1.
- **free** — paraphrase however; exemplars are tone hints only. The dim / QUALITY and "it's the vocal" still have to hold.  
The temperature of the inferencing is set to 0.5.

Output schema is always:

```json
{
  "amateur": "...",
  "expert": "...",
  "problem_state_text": "short span of the L1 dim"
}
```

After that, we relabel the generated dialogue with the **same** MixAssist labeling prompt. If the predicted labels don't contain the original dimension, the row gets thrown out. Target is at least 10 kept per dim per mode; if a dim comes up short, we top it up once.