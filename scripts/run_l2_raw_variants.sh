#!/usr/bin/env bash
# Back-compat wrapper: lexicon L1 generation + relabel + keep filter.
exec "$(dirname "$0")/run_l2_lexicon_kaggle.sh" "$@"
