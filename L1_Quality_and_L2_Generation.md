# L2 Dataset Generation

The L2 dataset is generated from L1 problem statements based on the MixAssist dataset, which is a collection of mixing session dialogue turns. The L2 dataset consists of two generated styles: Retarget and Strict. The Retarget style checks the problem dimension of the L1 problem statement, selects an aligned conversation turn from the MixAssist dataset, and simply substitutes the subject. The Strict style provides more freedom to paraphrase, utilizing materials from both the L1 problem statement and the aligned MixAssist turn to generate an L2 conversation.

The following table outlines the 7 axes and 12 dimensions (11 problem dimensions + 1 clean) with brief descriptions that are used for labeling, extraction, and generation throughout the experiment:


| dim                | what a listener would say                        |
| ------------------ | ------------------------------------------------ |
| `too_loud`         | too loud · sticking out too much                 |
| `too_quiet`        | quiet · barely be heard                          |
| `muddy`            | muddy · too much mud                             |
| `thin`             | a little bit thin · could be a little bit fuller |
| `harsh`            | bright · too much high end                       |
| `dull`             | a little low · warming up the sound              |
| `too_wet`          | super wet · unwanted reverb                      |
| `too_dry`          | dry · lacks the feeling of being in a space      |
| `over_compressed`  | a bit flat · lack of the dynamics                |
| `under_compressed` | punchy · adding too much punch                   |
| `masking`          | lost · masked by other elements                  |
| `clean`            | clean · tone didn't sound bad at all             |


The following table specifies confusing dimensions that need to be carefully distinguished. These pairs are deliberately close in audio, and the text must not make them closer. This table serves as a reference during writing:


| pair                                        | how the audio differs                                                                                                              | rule for your wording                                                                                                                                                   |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `too_quiet` **vs** `masking`                | too_quiet = the **vocal's level** drops. masking = the vocal is untouched, the **band** rises around it                            | **Never use bare loudness words for** `masking` — no "quiet", no "too soft". Use competition/obstruction words: masked, obscured, covered up, lost in a frequency clash |
| `masking` **vs** `muddy_clash`              | masking = band elevated **broadly** across the vocal's range. muddy_clash = the **vocal** pushed into one narrow 200–500 Hz pocket | muddy = the vocal itself sounds thick. masking = something else is in the way                                                                                           |
| `too_loud` **vs** `over_compressed`         | too_loud = pure gain, no tonal or dynamic change. over_compressed = dynamics flattened, loudness held                              | **No tonal words for** `too_loud`**/**`too_quiet` — they are level-only. Nothing about brightness, thickness or space                                                   |
| `over_compressed` **vs** `under_compressed` | flattened vs excessively variable                                                                                                  | over = squashed/lifeless/no dynamics. under = uneven/inconsistent/unpredictable. Never "loud" or "quiet" for either                                                     |
| `muddy` **vs** `dull`                       | muddy = too much low-mid. dull = too little top                                                                                    | real engineers use these interchangeably; we cannot. muddy stays in the low mids, dull stays "up top"                                                                   |
| `harsh` **vs** `dull`                       | opposite ends of the top                                                                                                           | direction pair — keep them opposite                                                                                                                                     |
| `too_wet` **vs** `too_dry`                  | opposite amounts of ambience                                                                                                       | direction pair                                                                                                                                                          |
| `thin` **vs** `muddy`                       | opposite ends of the low mids                                                                                                      | direction pair                                                                                                                                                          |


Every dimension above except `masking` and `clean` has a **direction pair** — its opposite. The model is explicitly trained to separate a fault from its opposite, and a paraphrase that could describe either one damages that separation.

The generation follows this pipeline:

```text
MixAssist Dataset
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
Quality / Lexicon Extraction    MixAssist Turn Labeling
(blind extract → map → filter)   (dim + speaker + vocal?)
       │                              │
       ▼                              │
L1 Caption Generation                 │
(Subject + Copula + Quality + Scope)  │
       │                              │
       └──────────────┬───────────────┘
                      │
                      ▼
               Raw L2 Generation
               (retarget / strict)
          L1 + labeled MixAssist turn
                      │
                      ▼
                  Relabeling
         (same as MixAssist labeling)
                      │
                      ▼
           Post Filtering & Cleaning
        (keep if gold_dim in dim_any;
         polarity / subject / malform checks)
                      │
                      ▼
                  L2 Dataset
              (retarget / strict)
```

or

```text
Quality Extraction → L1 Captions ─┐
                                  ├→ Raw L2 → Relabel → Filter/Clean → L2 Dataset
MixAssist Labeling ───────────────┘
```



## Quality / Lexicon Extraction

To form the L1 problem statement, the quality or the lexicon must first be constructed. Since terms or adjectives generated directly from an LLM or a random website might lack precision, the terms, phrases, and adjectives forming the Quality or Lexicon library are extracted directly from the MixAssist Dataset.

1. **Blind Extraction (term_extract_prompt):**

Using the Google-Gemma 4 LLM, we first extract any adjectives, terms, verbs, and phrases that describe a sound's state or an instrument—regardless of the specific problem dimension or whether it is simply praising something well done. This creates a large pool of "subject + description" combinations. We then use the descriptions for finer term separation based on our dimension table.

1. **Hierarchical Mapping:**

To map the large term pool to its corresponding dimensions, we manually construct a small seed lexicon pool. For instance, obvious terms describing a track that is too loud include "loud" or "overpowering", while "harsh" or "bright" are used for harshness. We first map the lexicon into the 7 axes, and then into the dimensions within each axis (except for `masking` and `clean`, which only have one dimension). 

The mapping is done using embedding similarity. For each extracted term (description), we encode it and every seed phrase into the same embedding space. Specifically, we first build one centroid vector per label by averaging the normalized seed embeddings, and then score each term based on the cosine similarity between the term embedding and that centroid. The mapping is hierarchical—first finding the best of the 7 axes, then the best dimension inside that axis. Terms under each label are then ranked by this score (highest first) using a sentence embedding model encoder (`all-mpnet-base-v2`). *Note: We also experimented with using a CLAP model for mapping, but the results were not satisfactory.*

1. **Top-N and Manual Filtering:**

After mapping, we filter and retain only the top 30 terms for each dimension (if a dimension has fewer, all are included). Given the manageable size of the raw lexicon, we perform manual filtering and grammar correction to fit the following format:

```text
The [SUBJECT]  [COPULA]  [DEGREE]?  [QUALITY]  [SCOPE]?  .

The lead vocal   sounds    severely      thin      down low  .
The singer         is                 washed out             .
```

Note: `[DEGREE]?` here is only a conceptual slot for editing the lexicon (e.g. whether a phrase can take "slightly / severely"). L1 Construction does **not** sample a separate DEGREE token — any degree wording is already folded into the QUALITY phrase itself (e.g. "a little muddy", "very quiet").

The final qualities are separated into two categories for each dimension: Degree-Compatible and Standalone. Each quality must adhere to these rules:

- No negations (e.g., "doesn't have air" is invalid, but "lacking air" or "failed to cut through" is acceptable).
- No specific mixing numbers (e.g., Hz values).
- The grammar of the quality should be mostly correct.

The final statistics for each dimension are listed below:


| dimension        | degree_compatible | standalone | total   | scope                          |
| ---------------- | ----------------- | ---------- | ------- | ------------------------------ |
| too_loud         | 17                | 15         | 32      | —                              |
| too_quiet        | 10                | 6          | 16      | —                              |
| muddy            | 9                 | 5          | 14      | in the low mids                |
| thin             | 4                 | 5          | 9       | down low                       |
| harsh            | 13                | 10         | 23      | up top · in the presence range |
| dull             | 5                 | 4          | 9       | up top                         |
| too_wet          | 11                | 11         | 22      | —                              |
| too_dry          | 5                 | 3          | 8       | —                              |
| over_compressed  | 9                 | 10         | 19      | —                              |
| under_compressed | 14                | 11         | 25      | —                              |
| masking          | 4                 | 19         | 23      | —                              |
| clean            | 11                | 1          | 12      | —                              |
| **Total**        | **112**           | **100**    | **212** |                                |


Thin, dull, and too_dry have the smallest degree_compatible lists (4–5), while Clean has 11 degree_compatible phrases for 20 L1 slots.

**Settings (this stage):**
- **Prompt:** blind term-extraction prompt (full text in the prompts appendix)
- **LLM (blind extract):** Gemma 4 31B-IT, Q4_K_M GGUF, local `llama.cpp`; temperature = 0.1; top_p = 0.95 (nucleus sampling); top_k = 64; max_tokens = 1536 (max generation length); n_ctx = 6144 (context window)
- **Input:** MixAssist train / validation / test
- **Embedding map:** `all-mpnet-base-v2`; hierarchical centroid cosine similarity; hand-written seed phrases per axis/dimension; min-sim = 0.25 (low-confidence flag only — does not auto-drop)
- **CLAP map:** tried (`clap-htsat-unfused`) but not used for the final lexicon

## L1 Construction

The L1 captions are constructed using the following template: 
`The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? .`

The components are generated according to the following rules:


| Component   | Source / Rule                                                                           |
| ----------- | --------------------------------------------------------------------------------------- |
| `[SUBJECT]` | Randomly picked from: *the lead vocal*, *the singer*, *the vocal*, or *the voice*.      |
| `[COPULA]`  | Randomly picked from: *is* or *sounds*.                                                 |
| `[QUALITY]` | Randomly picked exclusively from the `degree_compatible` list (standalone is not used). |
| `[SCOPE]`   | Sampled and added only if the dimension includes a scope.                               |


The constructed L1 statements are then used as inputs to generate the L2 dataset.
Note that L1 used for L2 generation samples only degree_compatible; standalone may be used if treating L1 itself as a dataset.

**Settings (this stage):**
- **No LLM** — deterministic slot sampling from the manual lexicon
- **n_per_dim = 20** (captions per dimension → 240 L1 captions total)
- **seed = 123** (reproducible SUBJECT / COPULA / QUALITY / SCOPE draws)

## MixAssist Turn Labeling

The MixAssist dataset consists of 640 conversation turns. We analyze each turn; if it mentions a problem statement or describes a mix issue, we label it with the corresponding dimension. The LLM determines the meaning and differences of each dimension based on the descriptions provided earlier. 

Crucially, the labeling follows strict evidence rules: 

- The dimension is decided from the **CURRENT TURN alone**. 
- The dialogue history is only used as auxiliary context to resolve pronouns or identify the affected stem. If the current turn is ambiguous without relying on the history, it is labeled as `none`. 
- We strictly distinguish between the `problem_stem` (the instrument suffering the issue) and the `fix_stem` (the action target). 
- Additionally, `clean` is a valid class used for praise or fault-free statements, and a `vocal_lead` boolean specifies whether the issue pertains specifically to the lead/main vocal.

If the LLM determines a statement is merely a personal preference or a generic DAW operation without clearly pointing out a defect, the turn is discarded.

**Settings (this stage):**
- **Prompt:** MixAssist turn-labeling prompt (full text in the prompts appendix); history window = 8 messages (auxiliary context only)
- **LLM:** Gemma 4 31B-IT, Q4_K_M GGUF; temperature = 0.1; top_p = 0.95; top_k = 64; max_tokens = 1536; n_ctx = 6144
- **Input:** MixAssist train (340) / validation (50) / test (250) = 640 turns
- **L2 exemplar pool:** all three splits combined; keep turns with a clear problem (or `clean`) and non-empty problem text; no confidence filter

## Raw L2 Generation

Generation is performed by the same LLM used for labeling, but with a different prompt. The raw L2 generation produces two dataset styles: Retarget and Strict. Each style generates 20 conversation turns for each of the 12 dimensions.

For each dimension:

1. Construct an L1 based on the L1 construction rules.
2. Randomly select a labeled conversation turn from the MixAssist dataset that matches the target dimension. (If all turns for a dimension have been used, they are randomly picked again).
3. Generate the L2 conversation using the L1 description, the selected MixAssist turn, and the system prompt. Both styles operate at a temperature of 0.1. The output is a conversation turn featuring an "amateur" and an "expert."

For the **Retarget** style, nearly everything from the original MixAssist dialogue is preserved; only the instrument or party words are altered so that the subject becomes the vocal. For the **Strict** style, the L1 problem caption is converted into a studio dialogue referencing the selected MixAssist dialogue. It allows more freedom to paraphrase and merge materials from the L1 caption and the MixAssist dialogue, and it may ignore unimportant chatting sentences from the original turn. Both system prompts ensure that the generated L2 conversation strictly aligns with the `L1.dim`.

**Settings (this stage):**
- **Prompt:** Retarget / Strict generation prompts (full text in the prompts appendix)
- **LLM:** same Gemma 4 31B-IT Q4_K_M GGUF; temperature = 0.1; top_p = 0.95; top_k = 64; max_tokens = 768; n_ctx = 4096
- **Exemplar:** 1 same-dimension MixAssist turn from the combined train+val+test problem pool; content = **raw** (full Amateur/Expert dialogue, not only the short problem span)
- **seed = 123** (shared with L1 sampling for reproducibility)
- **Scale:** 12 dimensions × 20 = 240 raw turns per style (Retarget, Strict)

## Relabel and Filtering

After generation, we obtain a total of 240 raw conversation turns for each Retarget and Strict styles. We then perform a relabeling step identical to the original MixAssist conversation turn labeling. Following the relabeling, we check whether the assigned label of the generated dialogue matches the L1 `gold_dim` (`dim_any`, which is any predicted problem label). If it does not match, the generated dialogue is discarded.

**Settings (this stage):**
- **Prompt:** same MixAssist turn-labeling prompt (prompts appendix); gold dimension is **not** shown to the model
- **LLM:** same as MixAssist labeling (temperature = 0.1; top_p = 0.95; top_k = 64; max_tokens = 1536; n_ctx = 6144)
- **Keep rule:** `dim_any` — gold dimension appears in any predicted problem label on that turn

## Final LLM Cleaning

Following the relabeling, a final cleaning is performed on the generated dataset. The LLM ensures the following:

- The polarity is correct for the axis (e.g., ensuring turns meant to be "too loud" are not accidentally labeled "too quiet" due to negations or confusing wording).
- The dialogues are not malformed.
- The subject is correct and accurately pertains to the vocal.

**Settings (this stage):**
- **Prompt:** L2 cleaning prompt (full text in the prompts appendix)
- **LLM:** same as labeling (temperature = 0.1; top_p = 0.95; top_k = 64; max_tokens = 1536; n_ctx = 6144)
- **Actions:** keep / drop (malformed or wrong source) / polarity flip of gold dimension along direction pairs
- **Note:** dropped rows may still sit in the CSV; downstream use should keep only `clean_action = keep`

The final statistics of the generated L2 dataset are listed below:


| dim              | retarget | strict  | Total   |
| ---------------- | -------- | ------- | ------- |
| too_loud         | 16       | 18      | 34      |
| too_quiet        | 16       | 18      | 34      |
| muddy            | 16       | 19      | 35      |
| thin             | 19       | 20      | 39      |
| harsh            | 13       | 18      | 31      |
| dull             | 15       | 20      | 35      |
| too_wet          | 16       | 16      | 32      |
| too_dry          | 15       | 19      | 34      |
| over_compressed  | 20       | 19      | 39      |
| under_compressed | 10       | 15      | 25      |
| masking          | 19       | 15      | 34      |
| clean            | 19       | 20      | 39      |
| **Total**        | **194**  | **217** | **411** |




# How to use it

## Use L1 as Dataset

To construct the L1 caption, you can directly generate the caption by using the `quality_from_manual.json` file to populate the `[QUALITY]` slot, while randomly assigning usable words to other slots like `[SUBJECT]` and `[COPULA]`. Alternatively, you can directly use the generated `l1_lexicon_captions_lex.jsonl` file by extracting the `L1` text from the `texts` field.

## Use L2 Dataset

There are two styles of the generated dataset: Retarget and Strict (`l2_from_l1_retarget_raw_llm_clean.csv` and `l2_from_l1_strict_raw_llm_clean.csv`). Both can be used directly as datasets. Extract the desired turns by referencing the `gold_dim` (the dimension the L2 generated dialogue belongs to) and pulling the corresponding `generated_dialogue` field. 

Additionally, the CSV structure contains separate, independent fields for `amateur_text` and `expert_text`. Users have the flexibility to extract the combined dialogue string or to extract just the specific amateur or expert fields depending on the requirements of the downstream task.