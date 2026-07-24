# Labeling stats (problem-centric)

## Coverage

| split | turns | problem turns | % | problem rows | % rows | both turns | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 340 | 128 | 37.6% | 148 | 39.8% | 91 | 0 |
| validation | 50 | 16 | 32.0% | 18 | 33.3% | 11 | 0 |
| test | 250 | 94 | 37.6% | 110 | 40.9% | 67 | 0 |
| ALL | 640 | 238 | 37.2% | 276 | 39.7% | 169 | 0 |

## Axis distribution (all)

| axis | n | % |
|---|---:|---:|
| level | 110 | 39.9% |
| body | 38 | 13.8% |
| brightness | 11 | 4.0% |
| space | 42 | 15.2% |
| dynamic | 25 | 9.1% |
| masking | 30 | 10.9% |
| stereo | 20 | 7.2% |
| phase | 0 | 0.0% |

## Axis × Dimension

- **level** (n=110): `too_quiet`=54, `too_loud`=56
- **body** (n=38): `muddy`=29, `thin`=9
- **brightness** (n=11): `harsh`=5, `dull`=6
- **space** (n=42): `too_wet`=15, `too_dry`=27
- **dynamic** (n=25): `over_compressed`=5, `under_compressed`=20
- **masking** (n=30): `swamping`=16, `invading`=14
- **stereo** (n=20): `too_wide`=5, `too_narrow`=15
- **phase** (n=0): `phase`=0

## Dimension ranking

| dimension | n | % |
|---|---:|---:|
| too_loud | 56 | 20.3% |
| too_quiet | 54 | 19.6% |
| muddy | 29 | 10.5% |
| too_dry | 27 | 9.8% |
| under_compressed | 20 | 7.2% |
| swamping | 16 | 5.8% |
| too_narrow | 15 | 5.4% |
| too_wet | 15 | 5.4% |
| invading | 14 | 5.1% |
| thin | 9 | 3.3% |
| dull | 6 | 2.2% |
| harsh | 5 | 1.8% |
| too_wide | 5 | 1.8% |
| over_compressed | 5 | 1.8% |
