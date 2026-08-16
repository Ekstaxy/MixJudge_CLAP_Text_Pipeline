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
    SYSTEM_PROMPT_FREE,
    SYSTEM_PROMPT_RETARGET,
    SYSTEM_PROMPT_STRICT,
    VALID_EXEMPLAR_CONTENTS,
    VALID_L1_DIMS,
    VALID_MODES,
    build_user_prompt,
    stem_to_subject,
    system_prompt_for_mode,
)
from .lexicon_slots import (  # noqa: F401
    PROBLEM_DIMS,
    assemble_caption,
    build_l1_records,
    resolve_quality_json,
)
from .term_extract_prompt import (  # noqa: F401
    SYSTEM_PROMPT as TERM_EXTRACT_SYSTEM_PROMPT,
    build_extract_user_message,
)
