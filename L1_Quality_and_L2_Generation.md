# L2 Dataset Generation
The L2 dataset is generated from L1 problem statement refer to the MixAssist dataset, which is a dataset that collects the dialogue or conversation turn. The L2 dataset consist of two kinds of generated style, which are Retarget and Strict two type. Retarget type dataset basically just check which problem dimension it is of the L1 problem statement, and then pick one of the aligned conversation turn from the MixAssist dataset, and then just simply substitute the subject of the MixAssist conversation turn. Strict type dataset has a higher freedom of doing paraphrase or using the materials in the L1 problem statement and one of the aligned MixAssist conversation turn, then generate a L2 conversation.

The following table is the 7 axis, 12 dimensions (11 problem dimension + 1 clean) brief descriptions that would be used in the labeling, extraction and generation through out the experiment:
| dim | what a listener would say |
|---|---|
| `too_loud` | vocal sits way above the band, drowning the kick/snare |
| `too_quiet` | vocal is under the band, you strain to hear it |
| `muddy` | thick, boxy, congested in the low mids |
| `thin` | no body, weedy, hollow |
| `harsh` | piercing, edgy, fatiguing |
| `dull` | no air, no shine, blanket over the mic |
| `too_wet` | drowned in reverb |
| `too_dry` | dead, close, disconnected from the mix |
| `over_compressed` | squashed, flat, lifeless |
| `under_compressed` | uneven, jumping around in level |
| `masking` | swallowed by the band, can't cut through |

And this following table specified confusing dimensions that need to be distinguished:
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

The generation method is done in the following pipeline or sequence:

## Quality / Lexicon Extraction
To form the L1 problem statement, the quality or the lexicon should be first constructed. However, it is doubted that the terms or adjective that is generated from LLM or searching on a random website would be not precise. Therefore, the terms / phrases / adjectives that will form the Quality or the Lexicon library would be also extracted from the MixAssist Dataset.

1. Blind Extraction (term_extract_prompt):  
By using Google-Gemma 4 LLM, first extract any adjectives, terms, verbs, phrases that describes a sound's state or an instrument, no matter which problem dimension it is or it is just a praise of something is doing well. By doing so, we could first get a large pool of subject + description and we could take the description to do finer term separation according to the dimension table we want.

2. Hierarchical Mapping
To map the large term pool to its dimension, we first glimpse through the pool and manually construect a small seed lexicon pool. For instance, there are some obvious term that could describe a track is too loud such as "loud", "overpowering", or when describe a track is too harsh we could use "harsh", "bright". We first map the lexicon into 7 axis, then map each axis into 2 dimensions (except masked and clean, which only has one dimension in the axis). The seed is as the following. 

The mapping is done by embedding similarity. For each extracted term (description), we encode it and every seed phrase into the same embedding space, then score the term against each axis (and later each dimension) as follows: compute the cosine similarity between the term embedding and each seed under that label, average those similarities into one score for the label, and assign the term to the label with the highest score. Mapping is hierarchical — first the best of the 7 axes, then the best dimension inside that axis (one choice only for `masking` and `clean`). After assignment, terms under each label are ranked by that score (highest first). We run this with a sentence embedding model encoder (`all-mpnet-base-v2`). 

3. Top-N and Manual Filtering
After the mapping, according to distribution of each dimension, we first filter out and leave only top 30 terms for each dimensions (if any dimension isn't enough, then just includes all). Since the total amount of the extracted raw lexicon is not that much. We decide to do manual filtering and grammer correction to fit to the following form:
```
The [SUBJECT]  [COPULA]  [DEGREE]?  [QUALITY]  [SCOPE]?  .

The lead vocal   sounds    severely     thin      down low  .
The singer         is                   washed out          .
```
The final quality will be separate to two categories in each dimension: Degree-Compatible and Standalone.
Each quality should follow the following rules:
- The term can't have any negation such as "doesn't have air", but "lacking air" or "failed to cut through" is ok
- No actual mixing number such as Hz number
- The grammer of the quality should be mostly correct.

The final statistic of the number of the each dimension is list at the following table:
| dimension | degree_compatible | standalone | total | scope |
| --- | ---: | ---: | ---: | --- |
| too_loud | 17 | 15 | 32 | — |
| too_quiet | 10 | 6 | 16 | — |
| muddy | 9 | 5 | 14 | in the low mids |
| thin | 4 | 5 | 9 | down low |
| harsh | 13 | 10 | 23 | up top · in the presence range |
| dull | 5 | 4 | 9 | up top |
| too_wet | 11 | 11 | 22 | — |
| too_dry | 5 | 3 | 8 | — |
| over_compressed | 9 | 10 | 19 | — |
| under_compressed | 14 | 11 | 25 | — |
| masking | 4 | 19 | 23 | — |
| clean | 11 | 1 | 12 | — |
| **Total** | **112** | **100** | **212** | |

Thin / dull / too_dry still have the smallest degree_compatible lists (4–5). Clean has 11 degree_compatible phrases for 20 L1 slots.

## L1 Construction
The L1 would be constructed in the following form:
The [SUBJECT] [COPULA] [QUALITY] [SCOPE]? .
TBD: Make following a table
Subject is randomly picked from the lead vocal / the singer / the vocal / the voice
Copula is randomly picked from is / sounds
QUALITY would be randomly picked from degree_compatible only (standalone is not used).
Scope would be sampled and added if the dimension has it.
The L1 that is constructed will be used as a input to generate L2.

## MixAssist Turn Labeling
MixAssist dataset consist totally 500.. conversation turns. We first try to analyze each conversation turn and if the turn mentions a problem statement or describe a problem in a mix, then we will label with the corresponding dimension.
The LLM will specify the what does each dimension mean and what are the differences between them, based on the description table above. The input of the labeling inference also consist of the dialogue history to help the LLM decide which dimension it is, but if solely read the conversation turn and couldn't tell which lable it is or the problem statement is in the history not in the turn, then the conversation turn would be discarded.
The labeling prompt is specified at the labeling_prompt. The output of the labeling not only includes the label but also additional information such as problem_speaker and if the subject is vocal or not (also separate the difference between lead vocal and background vocal). The LLM also will analyze if the statement is about personal preference or just a DAW operation and discard it if it is not a clear problem statement that actually points out the problem and not the fix.

## Raw L2 Generation
The generation is done by the same LLM as the labeling but with different prompt.
The raw L2 generates 2 types of style of dataset which is retarget and strict.
Both types of raw generated dataset will consists of each 20 generated turns of each 12 dimensions (11 problem state dimension + 1 clean dimension).
For each dimension:
1. Construct a L1 based on the rule that is specified in the L1 construction.
2. Randomly pick one of the conversation turn from the labeled MixAssist dataset that matches the label that want to be generated. If the whole conversation turns of one of the dimensions have all been picked once, then picked randomly again.
3. Do the L2 generation with the input of: L1 description, picked MixAssist conversation turn, system prompt. The generation should follow these rules, and strict / retarget has a different system prompt while both has the temperature of 0.1. The output should also be in a conversation turn of one person is amateur and the other person is expert.
For the retarget, nearly preserve everything from the MixAssit dialogue but jsut change only the instrument or the party word so that the subject become vocal.
For the strict, it converts the L1 problem caption into a studio dialogue refer to the picked MixAssist dialogue. It has a slightly higher freedom of doing paraphrase and using the materials in the L1 caption and the picked MixAssist dialogue. It could also ignore or discarded the not important infromation or just chatting sentences in the original MixAssist conversation.
Both system prompt specified that the generated L2 conversation turn should also align with the L1.dim.

## Relabel and Filtering
After the generation is finished we could get totally 240 generated raw conversation turns for both retarget and strict. Then we do the labeling just like what we did to the original MixAssist conversation turns in the previous step. The relabeling's prompt, method is the exactly same thing as what it did in the previous labeling step. After doing relabeling, we will check whether the re-label of the generated dialogue includes the L1 gold_dim (dim_any). If not, then it would be discarded.

## Final LLM Cleaning
After the relabel is finished, we'll do a final cleaning for the whole generated dataset.
The LLM will do these following things:
- Make sure the polarity is correct for axis, for instance, some turns that should be "too loud" will be labeled as "too quiet" because of the negation term or confusing wording.
- Maks sure that the dialogue is not malformed
- Check if the subject is right and its actually talking about the vocal.

The final statics of the numbers of generated L2 dataset is list at the following table:
| dim | retarget | strict | free |
| --- | ---: | ---: | ---: |
| too_loud | 16 | 18 | 20 |
| too_quiet | 16 | 18 | 16 |
| muddy | 16 | 19 | 20 |
| thin | 19 | 20 | 20 |
| harsh | 13 | 18 | 20 |
| dull | 15 | 20 | 20 |
| too_wet | 16 | 16 | 20 |
| too_dry | 15 | 19 | 20 |
| over_compressed | 20 | 19 | 18 |
| under_compressed | 10 | 15 | 20 |
| masking | 19 | 15 | 19 |
| clean | 19 | 20 | 20 |
| **Total** | **194** | **217** | **233** |





