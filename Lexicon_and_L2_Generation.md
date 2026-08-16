# Task

## Lexicon / Quality Extraction
Final result is in the `quality_from_manual.csv` and `quality_from_manual.json`, both of them separate each dimension to degree compatability and standalone.
The result is manually extracted from the `lexicon_top_centroid_manual.json` which is also manually extracted from the `lexicon_top_centroid_dim.json`.
The `lexicon_top_centroid_dim.json` is extracted from the `lexcicon_review_centroid_dim.csv`, which is generated from `lexicon_raw_all` by doing (centroid, embedding, blah, blah, blah)
The `lexicon_raw_all` is extracted from the turns of MixAssist by extracting all the subject + description of the subject, no matter which dimension or axis, or if it's a problem.
The Quality Lexicon is going to be used as L1 text. And these L1 text is going to be used as a input to generate L2 text.

(It needs a text flow chart of how the quality/lexicon is extracted.)

## Generation
The turns of the MixAssit is first labeled by 12 dimension with 6 axis:
(list all of them)
The label prompt is ... (with some crucial regulations), and every turn could be labeled with different dim or axis if possible.

