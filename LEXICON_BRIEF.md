# Caption lexicon brief — ontology + recipe guide

**For:** whoever is expanding the caption/prompt wording.
**Goal:** more ways of saying each fault, without ever describing a fault we
don't actually render, or blurring two faults the model has to keep apart.

**The job in one line:** §4 lists the wording choices the caption generator draws
from — add more of them, in the QUALITY columns only.

Read §1–§3 first: they say what each fault actually is, and which pairs must
never be described the same way. §5 is the deliverable.

---

## 0. Why the words matter more than they look

MixJudge scores *(sentence, audio)* pairs. The audio encoder is trained; **the
text encoder is frozen** — it is CLAP's stock RoBERTa tower and never updates
(except in one experimental arm). So the model can only learn to move its
*audio* embeddings toward wherever the frozen text tower already puts our
sentences.

Two consequences:

1. **Our captions are the entire text world the model ever sees.** If every
   `too_loud` caption in training says "too loud", the model grounds the fault
   on that exact phrasing and degrades on anything else. Measured: retrieval
   holds up under rewording, but rank consistency between two phrasings of the
   same fault drops to ~0.4.
2. **Wording that overlaps between two dimensions is not a cosmetic problem.**
   If `too_quiet` and `masking` both say "buried", the two classes are
   partially merged in text space before training even starts. Keeping them
   apart in wording is your main job, alongside adding variety.

---

## 1. The ontology — 11 dimensions

**11 fault dimensions, rendered by 18 recipes — all of them listed below.** A
dimension may have more than one recipe: they are different ways of producing
the *same* complaint, and they share the same captions. Every caption is
**about the vocal**, no matter which stem the DSP actually touched (Rule 1).

| dim | what a listener would say | what the DSP actually does | stem touched |
|---|---|---|---|
| `too_loud` | vocal sits way above the band, drowning the kick/snare | **2 recipes.** `too_loud_level`: flat gain **+3 / +6 / +10.5 dB** on the vocal. `too_loud_level_clipped`: the same push, but far enough to genuinely clip — the other real way this happens | vocal |
| `too_quiet` | vocal is under the band, you strain to hear it | **1 recipe.** `too_quiet_level`: flat gain **down**, solved per clip until the vocal's contribution to the mix goes "underwater" | vocal |
| `muddy` | thick, boxy, congested in the low mids | **2 recipes.** `muddy_tone`: boost the vocal's own low mids (220 Hz, +500 Hz and a −3 dB presence tilt at severe). `muddy_clash`: boost the **vocal** at 350 Hz into the pocket the band already occupies | vocal (both) |
| `thin` | no body, weedy, hollow | **2 recipes.** `thin_presence_tilt`: presence band made dominant (2.7 kHz up, 300 Hz down). `thin_body_deficit`: cut the vocal's own low register at 136 / 225 Hz | vocal (both) |
| `harsh` | piercing, edgy, fatiguing | **2 recipes.** `harsh_presence`: narrow resonant boost (Q=3) at a frequency **sampled per example** (mostly 2.5–4 kHz, rare draws to 13 kHz). `harsh_sibilance`: expander that boosts **only** the s/t/sh bursts | vocal (both) |
| `dull` | no air, no shine, blanket over the mic | **1 recipe.** `dull_air`: high-shelf cut at a **fixed 12 kHz**, −8 / −12 / −16 dB | vocal |
| `too_wet` | drowned in reverb | **2 recipes.** `too_wet_quantity`: constant wash — long decay, high mix. `too_wet_timing`: tail clutter — no pre-delay, no ducking, so one syllable rings into the next | vocal |
| `too_dry` | dead, close, disconnected from the mix | **2 recipes.** `too_dry_dereverb`: real ML dereverb strips the vocal's own room. `too_dry_band_contrast`: **vocal untouched** — reverb is added to the *band*, so the vocal reads dry by contrast | vocal / band |
| `over_compressed` | squashed, flat, lifeless | **2 recipes.** `over_compressed`: limiter driven by a solver until measured crest factor hits a ceiling (16 / 13 / 10 dB). `over_compressed_parallel`: blends a crushed copy back in with the dry | vocal |
| `under_compressed` | uneven, jumping around in level | **1 recipe.** `under_compressed_fader`: slow random-walk gain wobble — **bidirectional**, some phrases too loud *and* some too quiet, with the clip's overall level restored | vocal |
| `masking` | swallowed by the band, can't cut through | **1 recipe.** `masking_static`: **vocal untouched** — one real backing stem (whichever competes hardest at the vocal's own measured peak) is boosted +3 / +4.5 / +6 dB there | band |

**Severity is always three rungs: `mild`, `medium`, `severe`** (plus `clean`).

**Not in the taxonomy, don't write captions for them:** stereo/width (the
backbone is mono — it averages L+R, so we literally cannot render it),
phase/polarity (no data), and `too_loud_presence`, a "brightness reads as
loudness" idea that was built three times and never once sounded loud — it was
retired.

---

## 2. What must stay distinguishable

These pairs are deliberately close in audio. The text must not make them closer.
This table is the one to keep open while writing.

| pair | how the audio differs | rule for your wording |
|---|---|---|
| `too_quiet` **vs** `masking` | too_quiet = the **vocal's level** drops. masking = the vocal is untouched, the **band** rises around it | **Never use bare loudness words for `masking`** — no "quiet", no "too soft". Use competition/obstruction words: masked, obscured, covered up, lost in a frequency clash |
| `masking` **vs** `muddy_clash` | masking = band elevated **broadly** across the vocal's range. muddy_clash = the **vocal** pushed into one narrow 200–500 Hz pocket | muddy = the vocal itself sounds thick. masking = something else is in the way |
| `too_loud` **vs** `over_compressed` | too_loud = pure gain, no tonal or dynamic change. over_compressed = dynamics flattened, loudness held | **No tonal words for `too_loud`/`too_quiet`** — they are level-only. Nothing about brightness, thickness or space |
| `over_compressed` **vs** `under_compressed` | flattened vs excessively variable | over = squashed/lifeless/no dynamics. under = uneven/inconsistent/unpredictable. Never "loud" or "quiet" for either |
| `muddy` **vs** `dull` | muddy = too much low-mid. dull = too little top | real engineers use these interchangeably; we cannot. muddy stays in the low mids, dull stays "up top" |
| `harsh` **vs** `dull` | opposite ends of the top | direction pair — keep them opposite |
| `too_wet` **vs** `too_dry` | opposite amounts of ambience | direction pair |
| `thin` **vs** `muddy` | opposite ends of the low mids | direction pair |

Every dim above except `masking` has a **direction pair** — its opposite. The
model is explicitly trained to separate a fault from its opposite, and a
paraphrase that could describe either one damages that.

---

## 3. Rules a caption must obey

1. **Subject is the vocal, always.** Even when the DSP touched the band
   (`masking`, `too_dry_band_contrast`), the sentence is about the vocal. Allowed
   subjects: *the lead vocal · the singer · the vocal · the voice*.
2. **No negation.** Write "lacking air", not "doesn't have air".
3. **Adjectives belong to exactly one dimension.** If a word could plausibly
   describe two dims, it belongs to neither.
4. **Only factually-safe frequency claims.** You may say "in the low mids"
   (muddy), "down low" (thin), "up top" / "in the presence range" (harsh, dull).
   **No Hz numbers, ever**, and **no band phrase at all for `masking`** — its
   target frequency is measured per clip, so any fixed claim would be false.
5. **Severity is carried by a degree word, not by a stronger adjective.**
   Degrees: mild → *slightly, a bit*; medium → *noticeably, clearly*; severe →
   *severely, completely*. Roughly half of generated captions carry one; the
   rest state the fault plainly. So we need both: short adjectives that accept a
   degree word, and standalone phrases that stand alone.
6. **Two registers, both wanted.** Plain adjectives ("muddy", "squashed") and
   natural-sounding descriptions ("stuck at one flat level with no life to it").
   The second kind is what makes the model robust to real user phrasing.

---

## 4. The grammar, and the choices in every slot

Captions are not written one at a time — they are **assembled** from a fixed
sentence frame. Your job is to widen the menu each slot draws from.

```
The [SUBJECT]  [COPULA]  [DEGREE]?  [QUALITY]  [SCOPE]?  .

The lead vocal   sounds    severely     thin      down low  .
The singer         is                   washed out          .
```

`[DEGREE]` and `[SCOPE]` are optional; `[SUBJECT] [COPULA] [QUALITY]` are always
present. Roughly **half** of generated captions carry a degree word — that is
deliberate, so severity is signalled by the *presence* of a degree word and not
only by which adjective was chosen.

### The four shared slots

These are the same for every dimension.

| slot | current choices | notes |
|---|---|---|
| **SUBJECT** | the lead vocal · the singer · the vocal · the voice | always the vocal (Rule 1). A parallel set for the band — *the backing band · the band · the instruments · the backing tracks* — exists **only** to build wrong-subject hard negatives, never for real captions |
| **COPULA** | is · sounds | plural forms *are / sound* exist for the band set |
| **DEGREE** — mild | slightly · a bit | |
| **DEGREE** — medium | noticeably · clearly | |
| **DEGREE** — severe | severely · completely | |
| **SCOPE** | per dimension, see below | may be empty; most dimensions have none |

### The two per-dimension slots

**QUALITY splits into two kinds, and the split matters.**

- **Degree-compatible** — must still read correctly after "slightly" /
  "noticeably" / "severely". Short adjectives. *"severely muddy"* ✅
- **Standalone** — used without any degree word, so they can be longer and more
  natural. *"stuck at one flat level with no life to it"* ✅, but
  *"severely stuck at one flat level…"* ✗

| dim | QUALITY — degree-compatible | QUALITY — standalone | SCOPE |
|---|---|---|---|
| `too_loud` | too loud · excessively loud | blasting over the band · much louder than everything else · far too dominant in the mix · stuck out uncomfortably in front of the mix · loud enough to drown out the rest of the band · way out in front of everything else | — |
| `too_quiet` | too quiet · buried in the mix | barely audible · lost under the band · unable to cut through the mix · left behind by the rest of the band · far behind the rest of the band · impossible to hear once the band comes in | — |
| `muddy` | muddy · boomy | boxy and congested · woolly · thick in the low mids | in the low mids |
| `thin` | thin · hollow | weedy · small and lacking body | down low |
| `harsh` | harsh · piercing | brittle · edgy and fatiguing | up top · in the presence range |
| `dull` | dull · dark and lidded | veiled · lacking air | up top |
| `too_wet` | drowned in reverb · washed out | swimming in ambience · distant and diffuse | — |
| `too_dry` | bone dry · disconnected from the room | pasted on top of the mix · without any ambience | — |
| `over_compressed` | squashed · flattened and lifeless | choked · pumping unnaturally · crushed flat with no dynamics left · robbed of all its natural punch · airless from being squeezed so hard · stuck at one flat level with no life to it | — |
| `under_compressed` | dynamically uncontrolled · uneven | jumping around in level · wild and unsteady · unpredictable from quiet to loud · wildly inconsistent between loud and soft · all over the place from note to note · impossible to pin down at one steady level | — |
| `masking` | masked · obscured | buried under competing frequencies · swallowed by the rest of the mix · covered up by a competing instrument · lost in a frequency clash with a competing instrument | — |

**This table is the deliverable surface.** Every phrase you add goes into one of
these three columns, for one dimension. Nothing else in the sentence changes.

A separate small set describes a clean, fault-free mix — *"The lead vocal sits
cleanly and the mix is balanced."* Four variants exist; more are welcome, and
they follow no slot structure.

---

## 5. The job

**Add more QUALITY choices.** That is the whole task. SUBJECT, COPULA, DEGREE
and SCOPE stay exactly as they are.

For each of the 11 dimensions, add to both QUALITY columns of the §4 table:

| column | how many | what it must be |
|---|---|---|
| QUALITY — degree-compatible | **+2 to 4** | short; must read correctly after "slightly / noticeably / severely" |
| QUALITY — standalone | **+4 to 8** | longer, natural, how someone would really complain. Never used with a degree word |

**Order to work in** — four dimensions were already expanded, six were not:

| dimensions | current | do what |
|---|---|---|
| `muddy`, `thin`, `harsh`, `dull`, `too_wet`, `too_dry` | 4–5 entries | **start here** |
| `masking` | 6 entries | light top-up |
| `too_loud`, `too_quiet`, `over_compressed`, `under_compressed` | 8 entries | already deep — match this as the target |

**How to hand it back:** one markdown table, same three columns as §4, new
entries marked. No code changes.

**Check every phrase before submitting:**

1. Could it also describe this dimension's opposite? → rewrite
2. For `masking` — does it use a loudness word ("quiet", "soft")? → rewrite
3. Does it claim a frequency range we don't render? → rewrite
4. Does it contain a negation ("doesn't", "no ...")? → rewrite
