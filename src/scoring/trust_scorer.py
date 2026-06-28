"""
Gotcha — Skill Trust Scoring Module
Computes trust scores per skill using verified signals (assessments, duration,
proficiency, endorsements) and aggregates across JD-matched skills.
"""

import logging
from typing import Optional

from src.config import (
    PROFICIENCY_SCORES,
    SKILL_TRUST_WEIGHTS,
    SKILL_TRUST_WEIGHTS_NO_ASSESSMENT,
    SkillTrust,
)
from src.ingestion.normalizer import build_synonym_lookup

logger = logging.getLogger(__name__)

# Caps for normalization
DURATION_CAP_MONTHS = 60
ENDORSEMENTS_CAP = 50


def score_skill_trust(
    skill: dict,
    assessment_scores: Optional[dict],
    synonym_lookup: Optional[dict[str, str]] = None,
) -> SkillTrust:
    """Compute trust score for a single candidate skill.

    Formula:
        If assessment available for this skill:
            trust = W_assess * (score/100) + W_dur * (dur/60) + W_prof * prof + W_end * (end/50)
        Else:
            trust = W_dur' * (dur/60) + W_prof' * prof + W_end' * (end/50)

    Args:
        skill: Skill dict with name, proficiency, duration_months, endorsements.
        assessment_scores: Dict of skill_name → score (0-100), or None.
        synonym_lookup: Alias→canonical mapping.

    Returns:
        SkillTrust dataclass with computed trust_score.
    """
    if not skill or not isinstance(skill, dict):
        return SkillTrust(
            skill_name="unknown",
            canonical_name="unknown",
            trust_score=0.0,
        )

    if synonym_lookup is None:
        synonym_lookup = build_synonym_lookup()

    raw_name = skill.get("name") or "unknown"
    raw_name_str = str(raw_name).lower().strip()
    canonical = skill.get("canonical_name")
    if not canonical:
        canonical = synonym_lookup.get(raw_name_str, raw_name_str)

    # Safely parse duration
    duration = 0
    duration_val = skill.get("duration_months")
    if duration_val is not None:
        try:
            duration = max(int(duration_val), 0)
        except (TypeError, ValueError):
            duration = 0

    # Safely parse proficiency
    proficiency_str = str(skill.get("proficiency") or "beginner").lower().strip()

    # Safely parse endorsements
    endorsements = 0
    endorsements_val = skill.get("endorsements")
    if endorsements_val is not None:
        try:
            endorsements = max(int(endorsements_val), 0)
        except (TypeError, ValueError):
            endorsements = 0

    # Normalize components
    norm_duration = min(duration / DURATION_CAP_MONTHS, 1.0)
    norm_proficiency = PROFICIENCY_SCORES.get(proficiency_str, 0.25)
    norm_endorsements = min(endorsements / ENDORSEMENTS_CAP, 1.0)

    # Check for assessment score — match by raw name or canonical name
    assessment_val = None
    if assessment_scores and isinstance(assessment_scores, dict):
        # Try raw name first, then canonical
        assessment_val = assessment_scores.get(raw_name)
        if assessment_val is None:
            assessment_val = assessment_scores.get(canonical)
        # Also try case-insensitive matching
        if assessment_val is None:
            for k, v in assessment_scores.items():
                if str(k).lower().strip() == raw_name_str:
                    try:
                        assessment_val = float(v)
                    except (TypeError, ValueError):
                        pass
                    break

    if assessment_val is not None:
        try:
            norm_assessment = min(float(assessment_val) / 100.0, 1.0)
        except (TypeError, ValueError):
            norm_assessment = 0.0
        weights = SKILL_TRUST_WEIGHTS
        trust = (
            weights["assessment_score"] * norm_assessment
            + weights["duration_months"] * norm_duration
            + weights["proficiency"] * norm_proficiency
            + weights["endorsements"] * norm_endorsements
        )
    else:
        weights = SKILL_TRUST_WEIGHTS_NO_ASSESSMENT
        trust = (
            weights["duration_months"] * norm_duration
            + weights["proficiency"] * norm_proficiency
            + weights["endorsements"] * norm_endorsements
        )

    trust = max(0.0, min(trust, 1.0))

    return SkillTrust(
        skill_name=str(raw_name),
        canonical_name=str(canonical),
        trust_score=trust,
        assessment_score=assessment_val,
        duration_months=duration,
        proficiency=proficiency_str,
        endorsements=endorsements,
    )


def score_candidate_trust(
    candidate: dict,
    jd_skills: list[str],
    synonym_lookup: Optional[dict[str, str]] = None,
) -> float:
    """Compute trust score across JD-matched skills.

    Uses TOP-3 average (rewards depth, not breadth).
    A candidate with 3 deep retrieval skills beats one with 15 shallow keywords.

    Matching is broadened: substring match + synonym canonical match.
    This means 'sentence transformers' matches 'sentence-transformers', etc.

    Args:
        candidate: Normalized candidate dict.
        jd_skills: List of JD must-have skill names.
        synonym_lookup: Alias→canonical mapping.

    Returns:
        Float in [0, 1].
    """
    if not candidate or not isinstance(candidate, dict):
        return 0.0
    if not jd_skills:
        return 0.0
    if synonym_lookup is None:
        synonym_lookup = build_synonym_lookup()

    skills = candidate.get("skills")
    if not skills or not isinstance(skills, list):
        return 0.0

    signals = candidate.get("redrob_signals")
    assessment_scores = (
        signals.get("skill_assessment_scores")
        if (signals and isinstance(signals, dict))
        else None
    )

    # Build normalised JD skill set for matching
    jd_skills_lower = {str(s).lower().strip() for s in jd_skills}

    matched_trusts: list[float] = []

    for skill in skills:
        if not skill or not isinstance(skill, dict):
            continue

        raw_name = str(skill.get("name") or "").lower().strip()
        canonical = str(
            skill.get("canonical_name")
            or synonym_lookup.get(raw_name, raw_name)
        ).lower().strip()

        # Broadened matching: exact canonical match OR substring in either direction
        is_match = (
            canonical in jd_skills_lower
            or raw_name in jd_skills_lower
            or any(jd_s in raw_name or jd_s in canonical or raw_name in jd_s for jd_s in jd_skills_lower)
        )

        if is_match:
            st = score_skill_trust(skill, assessment_scores, synonym_lookup)
            matched_trusts.append(st.trust_score)

    if not matched_trusts:
        return 0.0

    # TOP-3 AVERAGE — rewards depth, penalises keyword stuffing
    top3 = sorted(matched_trusts, reverse=True)[:3]
    result = sum(top3) / len(top3)
    logger.debug(
        "Candidate trust: %.3f (top-%d of %d matched skills)",
        result, len(top3), len(matched_trusts),
    )
    return min(result, 1.0)

