# -*- coding: utf-8 -*-
"""MixAssist labeling backends (HF / GGUF / Groq) + shared common."""

from __future__ import annotations

# Ensure `src/` is on sys.path so `prompts` and `labeling` resolve when scripts
# are run as `python src/labeling/labeling_gguf.py` or via `python -m`.
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
