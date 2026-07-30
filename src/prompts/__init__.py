# -*- coding: utf-8 -*-
"""Prompt modules for MixAssist labeling and L2 style transfer."""

from .labeling_prompt import (  # noqa: F401
    MAX_HISTORY_MESSAGES,
    PLACEHOLDER_ASSISTANT,
    PLACEHOLDER_USER,
    SYSTEM_PROMPT as LABELING_SYSTEM_PROMPT,
    build_user_message,
)
from .l2_generation_prompt import (  # noqa: F401
    SYSTEM_PROMPT as L2_SYSTEM_PROMPT,
    build_user_prompt,
    stem_to_subject,
)
