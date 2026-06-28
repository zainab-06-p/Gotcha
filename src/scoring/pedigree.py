"""
Gotcha — Pedigree Scoring Module
Scores candidate's educational background and company prestige.
Deliberately capped low (×0.5) to avoid bias dominating.
"""

import logging
from typing import Optional

from src.config import TIER_SCORES

logger = logging.getLogger(__name__)

# Company size bonus mapping
COMPANY_SIZE_BONUS = {
    "10001+": 0.15,
    "5001-10000": 0.12,
    "1001-5000": 0.10,
    "501-1000": 0.08,
    "201-500": 0.05,
    "51-200": 0.03,
    "11-50": 0.02,
    "2-10": 0.01,
    "1": 0.00,
}

# Hard cap multiplier — pedigree must never dominate
PEDIGREE_CAP = 0.5


def _best_education_tier(candidate: dict) -> float:
    """Find the highest education tier score from all education entries."""
    if not candidate or not isinstance(candidate, dict):
        return TIER_SCORES.get("unknown", 0.30)
    education = candidate.get("education") or []
    if not education or not isinstance(education, list):
        return TIER_SCORES.get("unknown", 0.30)

    best = 0.0
    for edu in education:
        if not edu or not isinstance(edu, dict):
            continue
        tier = edu.get("tier")
        if tier is None:
            tier = "unknown"
        tier_score = TIER_SCORES.get(str(tier).lower().strip(), TIER_SCORES.get("unknown", 0.30))
        best = max(best, tier_score)

    return best


def _company_size_bonus(candidate: dict) -> float:
    """Compute a small bonus based on current company size.

    Larger companies provide slightly more signal about process/scale exposure.
    """
    if not candidate or not isinstance(candidate, dict):
        return 0.0
    profile = candidate.get("profile") or {}
    if not profile or not isinstance(profile, dict):
        return 0.0
    size = profile.get("current_company_size") or ""
    return COMPANY_SIZE_BONUS.get(str(size).strip(), 0.0)


def score_pedigree(candidate: dict) -> float:
    """Score candidate pedigree from education tier and company size.

    Formula:
        raw = 0.70 * best_education_tier + 0.30 * company_size_bonus_scaled
        final = raw * PEDIGREE_CAP  (hard cap at 0.5)

    This ensures pedigree contributes some signal but never dominates
    the ranking over actual skill/career evidence.

    Args:
        candidate: Normalized candidate dict.

    Returns:
        Float in [0, 1].
    """
    if not candidate or not isinstance(candidate, dict):
        return 0.0
    edu_score = _best_education_tier(candidate)
    co_bonus = _company_size_bonus(candidate)

    # Scale company bonus to 0-1 range (max bonus is 0.15)
    co_score = min(co_bonus / 0.15, 1.0) if co_bonus > 0 else 0.0

    raw = 0.70 * edu_score + 0.30 * co_score
    final = raw * PEDIGREE_CAP

    final = max(0.0, min(final, 1.0))
    logger.debug("Pedigree score: %.3f (edu=%.2f, co=%.2f, raw=%.2f)",
                 final, edu_score, co_score, raw)
    return final
