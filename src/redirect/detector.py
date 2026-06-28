"""
Gotcha — Redirect Detector Module
Identifies when a candidate is a mismatch for the current JD,
but has strong skills suitable for other company roles (archetypes).
"""

import logging
from typing import Optional

from src.config import REDIRECT_LOW_THRESHOLD, REDIRECT_HIGH_THRESHOLD, CandidateResult

logger = logging.getLogger(__name__)


def detect_redirect(
    result: CandidateResult,
    low_threshold: float = REDIRECT_LOW_THRESHOLD,
    high_threshold: float = REDIRECT_HIGH_THRESHOLD,
) -> None:
    """Evaluate if a candidate should be redirected to a different position.

    If candidate matches the current JD poorly (final_score < low_threshold)
    but has a well-defined archetype profile, recommend a redirect.

    Args:
        result: CandidateResult object (mutated in place).
        low_threshold: Under this score, candidate is considered a mismatch.
        high_threshold: Over this score, candidate is considered a strong fit.
    """
    score = result.final_score
    archetype = result.redirect_suggestion or "Other Roles"

    # Check if mismatch
    if score < low_threshold:
        result.redirect_suggestion = f"Redirect to: {archetype}"
        result.redirect_reason = (
            f"Candidate score for current JD is low ({score:.1%} < {low_threshold:.1%}), "
            f"but their profile profile aligns with the '{archetype}' talent pool."
        )
        logger.debug("Candidate %s flagged for redirect to %s", result.candidate_id, archetype)
    else:
        # Candidate is a good fit for the current role
        result.redirect_suggestion = None
        result.redirect_reason = "Strong direct match for the current job description."
