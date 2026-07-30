# Vocal-related label distributions

Sources: `/mnt/d/Downloads/labeled_turns_gguf_train.csv`, `/mnt/d/Downloads/labeled_turns_gguf_validation.csv`, `/mnt/d/Downloads/labeled_turns_gguf_test.csv`

## Coverage

| metric | n | % |
| --- | ---: | ---: |
| all label rows | 695 |  |
| ok rows (no error) | 695 |  |
| problem rows | 276 | 100% of problems |
| vocal problem rows (`problem_stem=vocal`) | 53 | 19.2% of problems |
|   └ lead (`vocal_lead=True`) | 39 | 73.6% of vocal |
|   └ non-lead vocal (`vocal_lead=False`) | 14 | 26.4% of vocal |
| vocal_lead=True but stem≠vocal | 0 | (inconsistent) |
| vocal as fix_stem | 31 |  |
| vocal problem ∧ has fix | 34 | 64.2% of vocal problems |

### Turn-level

| metric | n | % |
| --- | ---: | ---: |
| turns | 640 |  |
| turns with any problem | 238 |  |
| turns with vocal problem | 49 | 20.6% of problem turns |
| turns with lead vocal problem | 36 |  |

## By split (vocal problem rows)

| split | vocal n | % of vocal | all problems | vocal share |
| --- | ---: | ---: | ---: | ---: |
| test | 24 | 45.3% | 110 | 21.8% of split problems |
| train | 23 | 43.4% | 148 | 15.5% of split problems |
| validation | 6 | 11.3% | 18 | 33.3% of split problems |

## Vocal problems (`problem_stem=vocal`)

n = 53

### Axis

| axis | n | % |
| --- | ---: | ---: |
| level | 22 | 41.5% |
| space | 12 | 22.6% |
| masking | 9 | 17.0% |
| dynamic | 5 | 9.4% |
| stereo | 3 | 5.7% |
| brightness | 1 | 1.9% |
| body | 1 | 1.9% |

### Dimension

| dimension | n | % |
| --- | ---: | ---: |
| too_quiet | 19 | 35.8% |
| too_wet | 6 | 11.3% |
| swamping | 6 | 11.3% |
| too_dry | 6 | 11.3% |
| under_compressed | 4 | 7.5% |
| too_loud | 3 | 5.7% |
| invading | 3 | 5.7% |
| too_narrow | 2 | 3.8% |
| harsh | 1 | 1.9% |
| too_wide | 1 | 1.9% |
| over_compressed | 1 | 1.9% |
| thin | 1 | 1.9% |

### Axis × Dimension

| axis/dimension | n | % |
| --- | ---: | ---: |
| level/too_quiet | 19 | 35.8% |
| space/too_wet | 6 | 11.3% |
| masking/swamping | 6 | 11.3% |
| space/too_dry | 6 | 11.3% |
| dynamic/under_compressed | 4 | 7.5% |
| level/too_loud | 3 | 5.7% |
| masking/invading | 3 | 5.7% |
| stereo/too_narrow | 2 | 3.8% |
| brightness/harsh | 1 | 1.9% |
| stereo/too_wide | 1 | 1.9% |
| dynamic/over_compressed | 1 | 1.9% |
| body/thin | 1 | 1.9% |

### Confidence

| confidence | n | % |
| --- | ---: | ---: |
| high | 45 | 84.9% |
| mid | 8 | 15.1% |

## Lead vocal only (`vocal_lead=True`)

n = 39

### Axis

| axis | n | % |
| --- | ---: | ---: |
| level | 16 | 41.0% |
| masking | 9 | 23.1% |
| space | 7 | 17.9% |
| dynamic | 5 | 12.8% |
| brightness | 1 | 2.6% |
| body | 1 | 2.6% |

### Dimension

| dimension | n | % |
| --- | ---: | ---: |
| too_quiet | 15 | 38.5% |
| too_wet | 6 | 15.4% |
| swamping | 6 | 15.4% |
| under_compressed | 4 | 10.3% |
| invading | 3 | 7.7% |
| too_dry | 1 | 2.6% |
| harsh | 1 | 2.6% |
| over_compressed | 1 | 2.6% |
| too_loud | 1 | 2.6% |
| thin | 1 | 2.6% |

### Axis × Dimension

| axis/dimension | n | % |
| --- | ---: | ---: |
| level/too_quiet | 15 | 38.5% |
| space/too_wet | 6 | 15.4% |
| masking/swamping | 6 | 15.4% |
| dynamic/under_compressed | 4 | 10.3% |
| masking/invading | 3 | 7.7% |
| space/too_dry | 1 | 2.6% |
| brightness/harsh | 1 | 2.6% |
| dynamic/over_compressed | 1 | 2.6% |
| level/too_loud | 1 | 2.6% |
| body/thin | 1 | 2.6% |

### Confidence

| confidence | n | % |
| --- | ---: | ---: |
| high | 36 | 92.3% |
| mid | 3 | 7.7% |

## Non-lead vocal (`problem_stem=vocal`, `vocal_lead=False`)

n = 14

### Axis

| axis | n | % |
| --- | ---: | ---: |
| level | 6 | 42.9% |
| space | 5 | 35.7% |
| stereo | 3 | 21.4% |

### Dimension

| dimension | n | % |
| --- | ---: | ---: |
| too_dry | 5 | 35.7% |
| too_quiet | 4 | 28.6% |
| too_loud | 2 | 14.3% |
| too_narrow | 2 | 14.3% |
| too_wide | 1 | 7.1% |

### Axis × Dimension

| axis/dimension | n | % |
| --- | ---: | ---: |
| space/too_dry | 5 | 35.7% |
| level/too_quiet | 4 | 28.6% |
| level/too_loud | 2 | 14.3% |
| stereo/too_narrow | 2 | 14.3% |
| stereo/too_wide | 1 | 7.1% |

### Confidence

| confidence | n | % |
| --- | ---: | ---: |
| high | 9 | 64.3% |
| mid | 5 | 35.7% |

## Vocal share within each axis (all problem rows)

| axis | vocal n | all n | vocal % of axis | % of all vocal |
| --- | ---: | ---: | ---: | ---: |
| level | 22 | 110 | 20.0% | 41.5% |
| space | 12 | 42 | 28.6% | 22.6% |
| body | 1 | 38 | 2.6% | 1.9% |
| masking | 9 | 30 | 30.0% | 17.0% |
| dynamic | 5 | 25 | 20.0% | 9.4% |
| stereo | 3 | 20 | 15.0% | 5.7% |
| brightness | 1 | 11 | 9.1% | 1.9% |

## When problem is vocal: fix_stem / fix_action

Rows with vocal problem and a fix: 34

### fix_stem

| fix_stem | n | % |
| --- | ---: | ---: |
| vocal | 26 | 76.5% |
| guitar | 4 | 11.8% |
| ambience | 2 | 5.9% |
| bass | 2 | 5.9% |

### fix_action

| fix_action | n | % |
| --- | ---: | ---: |
| raise_level | 11 | 32.4% |
| lower_level | 9 | 26.5% |
| add_reverb | 5 | 14.7% |
| reduce_reverb | 3 | 8.8% |
| pan | 3 | 8.8% |
| add_compression | 2 | 5.9% |
| reduce_compression | 1 | 2.9% |

## When fix_stem is vocal: problem_stem

| problem_stem | n | % |
| --- | ---: | ---: |
| vocal | 26 | 100.0% |

## Sample rows

### Lead vocal

- [train] group1_conv8 turn 1: `space/too_wet` (conf=high) — "It's probably super wet right now"
- [train] group1_conv8 turn 4: `space/too_wet` (conf=high) — 'need it to be way less drastic'
- [train] group1_conv8 turn 5: `space/too_wet` (conf=high) — 'bring the reverb down a little more'
- [train] group6_conv6 turn 1: `level/too_quiet` (conf=high) — "what the heck is he saying, I can't understand him"
- [train] group6_conv6 turn 2: `level/too_quiet` (conf=high) — "what the heck is he saying, I can't understand him"

### Non-lead vocal

- [train] group3_conv1 turn 0: `level/too_loud` (conf=high) — "backing vocals... they're probably maybe a little loud"
- [train] group4_conv4 turn 0: `stereo/too_narrow` (conf=high) — 'they are doubled'
- [train] group4_conv4 turn 1: `space/too_dry` (conf=mid) — 'put the backing vocals in the back of the mix'
- [train] group4_conv4 turn 2: `level/too_quiet` (conf=high) — 'Way, way quiet'
- [train] group4_conv4 turn 3: `space/too_dry` (conf=high) — 'I feel like maybe a little reverb on these vocals?'
