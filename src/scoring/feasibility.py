"""
Gotcha — Feasibility Scoring Module
Scores how feasible it is to hire this candidate: notice period, work mode,
relocation, interview completion rate, offer acceptance rate.
"""

import logging
from typing import Optional

from src.config import (
    NOTICE_PERIOD_IDEAL_MAX,
    NOTICE_PERIOD_ACCEPTABLE_MAX,
    NOTICE_PERIOD_PENALTY_MAX,
    JDProfile,
)

logger = logging.getLogger(__name__)


def _score_notice_period(days: Optional[int]) -> float:
    """Score notice period: lower is better.

    0-30 days:   1.0
    31-90 days:  linear decay 1.0 → 0.5
    91-180 days: linear decay 0.5 → 0.1
    >180 days:   0.1
    """
    if days is None:
        return 0.5  # Unknown — neutral

    try:
        days = max(int(days), 0)
    except (TypeError, ValueError):
        return 0.5

    if days <= NOTICE_PERIOD_IDEAL_MAX:
        return 1.0
    elif days <= NOTICE_PERIOD_ACCEPTABLE_MAX:
        # Linear decay from 1.0 to 0.5
        range_size = NOTICE_PERIOD_ACCEPTABLE_MAX - NOTICE_PERIOD_IDEAL_MAX
        progress = (days - NOTICE_PERIOD_IDEAL_MAX) / range_size
        return 1.0 - 0.5 * progress
    elif days <= NOTICE_PERIOD_PENALTY_MAX:
        # Linear decay from 0.5 to 0.1
        range_size = NOTICE_PERIOD_PENALTY_MAX - NOTICE_PERIOD_ACCEPTABLE_MAX
        progress = (days - NOTICE_PERIOD_ACCEPTABLE_MAX) / range_size
        return 0.5 - 0.4 * progress
    else:
        return 0.1


def _score_work_mode(
    preferred: Optional[str],
    jd_mode: Optional[str] = None,
) -> float:
    """Score work mode preference match.

    If JD mode is known, exact match = 1.0, hybrid = 0.7, mismatch = 0.3.
    If JD mode is unknown, any preference = 0.7 (neutral).
    """
    if preferred is None:
        return 0.5

    preferred = str(preferred).lower().strip()

    if jd_mode is None:
        return 0.7  # Can't evaluate — neutral-positive

    jd_mode = str(jd_mode).lower().strip()

    if preferred == jd_mode:
        return 1.0
    elif preferred == "hybrid" or jd_mode == "hybrid":
        return 0.7
    else:
        return 0.3


def score_feasibility(
    redrob_signals: Optional[dict],
    jd_profile: Optional[JDProfile] = None,
) -> float:
    """Score overall hiring feasibility.

    Components and weights:
        - notice_period_days: 0.30  (lower = better)
        - preferred_work_mode: 0.15 (match JD if available)
        - willing_to_relocate: 0.10 (bonus if True)
        - interview_completion_rate: 0.25 (direct quality signal)
        - offer_acceptance_rate: 0.20 (if not sentinel/None)

    Args:
        redrob_signals: Normalized signals dict.
        jd_profile: Optional parsed JD for work-mode matching.

    Returns:
        Float in [0, 1].
    """
    if not redrob_signals or not isinstance(redrob_signals, dict):
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    # Notice period (weight 0.30)
    notice = redrob_signals.get("notice_period_days")
    np_score = _score_notice_period(notice)
    weighted_sum += 0.30 * np_score
    total_weight += 0.30

    # Work mode (weight 0.15)
    preferred_mode = redrob_signals.get("preferred_work_mode")
    jd_mode = None
    if jd_profile:
        # Check if jd_profile is a dictionary or object with logistics attribute
        logistics = getattr(jd_profile, "logistics", None)
        if logistics is None and isinstance(jd_profile, dict):
            logistics = jd_profile.get("logistics")
        if isinstance(logistics, dict):
            jd_mode = logistics.get("work_mode")
    wm_score = _score_work_mode(preferred_mode, jd_mode)
    weighted_sum += 0.15 * wm_score
    total_weight += 0.15

    # Willing to relocate (weight 0.10)
    relocate = redrob_signals.get("willing_to_relocate")
    if relocate is not None:
        weighted_sum += 0.10 * (1.0 if relocate else 0.4)
        total_weight += 0.10

    # Interview completion rate (weight 0.25)
    icr = redrob_signals.get("interview_completion_rate")
    if icr is not None:
        try:
            weighted_sum += 0.25 * min(float(icr), 1.0)
            total_weight += 0.25
        except (ValueError, TypeError):
            pass

    # Offer acceptance rate (weight 0.20)
    oar = redrob_signals.get("offer_acceptance_rate")
    if oar is not None:
        try:
            weighted_sum += 0.20 * min(float(oar), 1.0)
            total_weight += 0.20
        except (ValueError, TypeError):
            pass

    if total_weight == 0.0:
        return 0.0

    score = weighted_sum / total_weight
    score = max(0.0, min(score, 1.0))

    logger.debug("Feasibility score: %.3f", score)
    return score
