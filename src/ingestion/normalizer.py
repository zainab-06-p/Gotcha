"""
Gotcha — Candidate Normalization Module
Maps sentinels to None, normalizes skill names via synonym dictionary,
and computes a data-confidence score per candidate.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.config import SENTINEL_VALUE, DATA_DIR

logger = logging.getLogger(__name__)

# Module-level cache for synonym lookup
_synonym_lookup_cache: Optional[dict] = None


def build_synonym_lookup(synonyms_path: Optional[str] = None) -> dict[str, str]:
    """Build a flat lookup: any alias (lowercased) → canonical name.

    Args:
        synonyms_path: Path to skill_synonyms.json. Defaults to data/skill_synonyms.json.

    Returns:
        Dict mapping every known alias (lowercase) to its canonical group name.
    """
    global _synonym_lookup_cache
    if _synonym_lookup_cache is not None:
        return _synonym_lookup_cache

    if synonyms_path is None:
        synonyms_path = DATA_DIR / "skill_synonyms.json"

    synonyms_path = Path(synonyms_path)
    if not synonyms_path.exists():
        logger.warning("Synonym file not found: %s", synonyms_path)
        return {}

    try:
        with open(synonyms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load synonyms: %s", e)
        return {}

    lookup: dict[str, str] = {}
    canonical_groups = data.get("canonical_groups", {})
    if not isinstance(canonical_groups, dict):
        canonical_groups = {}
    for canonical, aliases in canonical_groups.items():
        canonical_str = str(canonical).lower().strip()
        if not canonical_str:
            continue
        if isinstance(aliases, list):
            for alias in aliases:
                alias_str = str(alias).lower().strip()
                if alias_str:
                    lookup[alias_str] = str(canonical)
        # Also map the canonical name to itself
        lookup[canonical_str] = str(canonical)

    logger.info("Built synonym lookup with %d entries across %d groups",
                len(lookup), len(canonical_groups))
    _synonym_lookup_cache = lookup
    return lookup


def _is_sentinel(value) -> bool:
    """Check if a value is the -1 sentinel."""
    if value is None:
        return False
    try:
        return float(value) == SENTINEL_VALUE
    except (TypeError, ValueError):
        return False


def _normalize_sentinels(signals: dict) -> dict:
    """Replace all -1 sentinel values with None in redrob_signals."""
    if not isinstance(signals, dict):
        return {}
    sentinel_fields = [
        "github_activity_score",
        "offer_acceptance_rate",
        "interview_completion_rate",
        "recruiter_response_rate",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
        "profile_completeness_score",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
    ]
    for field in sentinel_fields:
        if field in signals and _is_sentinel(signals[field]):
            signals[field] = None
    return signals


def _normalize_skills(skills: list[dict], synonym_lookup: dict[str, str]) -> list[dict]:
    """Normalize skill names using the synonym dictionary.

    Adds a `canonical_name` field to each skill dict.
    """
    if not skills or not isinstance(skills, list):
        return []

    valid_skills = []
    for skill in skills:
        if not skill or not isinstance(skill, dict):
            continue
        raw_name = skill.get("name") or ""
        lower_name = str(raw_name).lower().strip()
        canonical = synonym_lookup.get(lower_name, lower_name)
        skill["canonical_name"] = str(canonical)
        valid_skills.append(skill)
    return valid_skills


def normalize_candidate(
    candidate: dict,
    synonym_lookup: Optional[dict[str, str]] = None,
) -> dict:
    """Main normalization entry point for a single candidate.

    Maps all alternate keys/formats to the canonical schema to ensure
    cross-dataset resiliency. Maps -1 sentinels to None.
    """
    if not candidate or not isinstance(candidate, dict):
        return {}

    # Idempotent guard
    if candidate.get("_normalized"):
        return candidate

    if synonym_lookup is None:
        synonym_lookup = build_synonym_lookup()

    # 1. Map alternate top-level keys to canonical names
    if "candidate_id" not in candidate:
        for alt in ["id", "candidateId", "cid"]:
            if alt in candidate:
                candidate["candidate_id"] = candidate[alt]
                break
        if "candidate_id" not in candidate:
            candidate["candidate_id"] = "CAND_0000000"

    if "profile" not in candidate:
        for alt in ["profile_info", "profileInfo", "about"]:
            if alt in candidate and isinstance(candidate[alt], dict):
                candidate["profile"] = candidate[alt]
                break
        if "profile" not in candidate:
            candidate["profile"] = {}

    profile = candidate["profile"]
    if isinstance(profile, dict):
        profile_mappings = {
            "anonymized_name": ["name", "full_name", "fullName", "anonymizedName"],
            "headline": ["headline_info", "professional_headline", "title"],
            "summary": ["bio", "about_me", "aboutMe"],
            "location": ["city", "address"],
            "years_of_experience": ["years_exp", "experience", "total_exp", "total_experience", "yearsOfExperience"],
            "current_title": ["current_role", "currentTitle"],
            "current_company": ["currentCompany"],
            "current_industry": ["industry", "currentIndustry"],
        }
        for canonical, alts in profile_mappings.items():
            if canonical not in profile:
                for alt in alts:
                    if alt in profile:
                        profile[canonical] = profile[alt]
                        break
                if canonical not in profile:
                    for alt in alts + [canonical]:
                        if alt in candidate and not isinstance(candidate[alt], (dict, list)):
                            profile[canonical] = candidate[alt]
                            break
        if "anonymized_name" not in profile:
            profile["anonymized_name"] = "Anonymized Candidate"
        if "years_of_experience" not in profile:
            profile["years_of_experience"] = 0.0
        else:
            try:
                profile["years_of_experience"] = float(profile["years_of_experience"])
            except (ValueError, TypeError):
                profile["years_of_experience"] = 0.0

    # Map alternate keys/formats to ensure robustness against malformed/null lists
    career_history = candidate.get("career_history")
    if not isinstance(career_history, list):
        found_alt = False
        for alt in ["careerHistory", "experience_history", "jobs", "work_history", "workHistory"]:
            if alt in candidate and isinstance(candidate[alt], list):
                candidate["career_history"] = candidate[alt]
                found_alt = True
                break
        if not found_alt:
            candidate["career_history"] = []

    education = candidate.get("education")
    if not isinstance(education, list):
        found_alt = False
        for alt in ["educationHistory", "schools", "academic_history"]:
            if alt in candidate and isinstance(candidate[alt], list):
                candidate["education"] = candidate[alt]
                found_alt = True
                break
        if not found_alt:
            candidate["education"] = []

    skills = candidate.get("skills")
    if not isinstance(skills, list):
        found_alt = False
        for alt in ["skills_list", "skillset", "skill_list"]:
            if alt in candidate and isinstance(candidate[alt], list):
                candidate["skills"] = candidate[alt]
                found_alt = True
                break
        if not found_alt:
            candidate["skills"] = []

    # Map skill string list to list of dicts if needed
    skills = candidate.get("skills", [])
    normalized_skills = []
    for s in skills:
        if isinstance(s, str):
            normalized_skills.append({
                "name": s,
                "proficiency": "intermediate",
                "endorsements": 0,
                "duration_months": 12
            })
        elif isinstance(s, dict):
            normalized_skills.append(s)
    candidate["skills"] = normalized_skills

    signals = candidate.get("redrob_signals")
    if not isinstance(signals, dict):
        found_alt = False
        for alt in ["signals", "redrobSignals", "activity_signals"]:
            if alt in candidate and isinstance(candidate[alt], dict):
                candidate["redrob_signals"] = candidate[alt]
                found_alt = True
                break
        if not found_alt:
            candidate["redrob_signals"] = {}

    # 2. Normalize redrob_signals sentinels
    signals = candidate.get("redrob_signals")
    if signals and isinstance(signals, dict):
        _normalize_sentinels(signals)
        assessments = signals.get("skill_assessment_scores")
        if isinstance(assessments, dict) and len(assessments) == 0:
            signals["skill_assessment_scores"] = None

    # 3. Normalize skill names
    skills = candidate.get("skills")
    if skills and isinstance(skills, list):
        _normalize_skills(skills, synonym_lookup)

    candidate["_normalized"] = True
    return candidate


def compute_data_confidence(candidate: dict) -> float:
    """Compute a 0–1 data-confidence score based on profile completeness
    and verification signals.

    Components:
        - profile_completeness_score / 100   (weighted 0.40)
        - verified_email                     (+0.15)
        - verified_phone                     (+0.15)
        - linkedin_connected                 (+0.10)
        - has non-empty skill_assessment_scores (+0.20)

    Args:
        candidate: Normalized candidate dict.

    Returns:
        Float in [0, 1].
    """
    if not candidate or not isinstance(candidate, dict):
        return 0.0

    signals = candidate.get("redrob_signals", {})
    if not signals or not isinstance(signals, dict):
        return 0.0

    score = 0.0

    # Profile completeness (40% weight)
    pcs = signals.get("profile_completeness_score")
    if pcs is not None:
        try:
            score += 0.40 * min(float(pcs) / 100.0, 1.0)
        except (ValueError, TypeError):
            pass

    # Verifications
    if signals.get("verified_email"):
        score += 0.15
    if signals.get("verified_phone"):
        score += 0.15
    if signals.get("linkedin_connected"):
        score += 0.10

    # Assessment scores available
    assessments = signals.get("skill_assessment_scores")
    if isinstance(assessments, dict) and len(assessments) > 0:
        score += 0.20

    return min(score, 1.0)
