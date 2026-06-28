"""
Gotcha — Data Confidence Scoring Module
Scores how trustworthy the candidate's data is based on profile completeness,
verifications, and assessment availability.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def score_data_confidence(candidate: dict) -> float:
    """Score data confidence / profile trustworthiness.

    Components:
        - profile_completeness_score / 100  (weight 0.40)
        - verified_email:  +0.15
        - verified_phone:  +0.15
        - linkedin_connected: +0.10
        - has non-empty skill_assessment_scores: +0.20

    This is essentially the same as normalizer.compute_data_confidence,
    provided here as the canonical scoring entry point.

    Args:
        candidate: Normalized candidate dict.

    Returns:
        Float in [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.0

    score = 0.0

    # Profile completeness (40% weight)
    pcs = signals.get("profile_completeness_score")
    if pcs is not None:
        score += 0.40 * min(float(pcs) / 100.0, 1.0)

    # Verifications
    if signals.get("verified_email"):
        score += 0.15

    if signals.get("verified_phone"):
        score += 0.15

    if signals.get("linkedin_connected"):
        score += 0.10

    # Assessment scores available and non-empty
    assessments = signals.get("skill_assessment_scores")
    if isinstance(assessments, dict) and len(assessments) > 0:
        score += 0.20

    score = max(0.0, min(score, 1.0))
    logger.debug("Data confidence: %.3f", score)
    return score
