"""
Gotcha — Score Combiner Module
Blends deterministic Track 1 scoring and LLM Track 2 scoring,
and applies the honeypot penalty.
"""

import logging
from typing import Optional

from src.config import TRACK2_BLEND_WEIGHT, HONEYPOT_PENALTY

logger = logging.getLogger(__name__)


def combine_scores(
    track1_score: float,
    track2_score: Optional[float] = None,
    is_honeypot: bool = False,
    blend_weight: float = TRACK2_BLEND_WEIGHT,
    penalty: float = HONEYPOT_PENALTY,
) -> float:
    """Blend Track 1 and Track 2 scores and apply penalty if honeypot is flagged.

    Formula:
        If track2_score is not None:
            combined = (1.0 - blend_weight) * track1_score + blend_weight * track2_score
        Else:
            combined = track1_score

        If is_honeypot is True:
            final = combined * penalty
        Else:
            final = combined

    Args:
        track1_score: Float in [0, 1].
        track2_score: Optional float in [0, 1].
        is_honeypot: Boolean flag.
        blend_weight: Weight given to Track 2 score.
        penalty: Multiplier applied if candidate is a honeypot.

    Returns:
        Float in [0, 1] representing the final rank score.
    """
    #decide base combined score
    if track2_score is not None:
        combined = (1.0 - blend_weight) * track1_score + blend_weight * track2_score
    else:
        combined = track1_score

    # apply penalty if honeypot
    if is_honeypot:
        final = combined * penalty
        logger.info("Applying honeypot penalty: %.4f -> %.4f", combined, final)
    else:
        final = combined

    final = max(0.0, min(final, 1.0))
    return final
