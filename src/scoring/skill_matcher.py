"""
Gotcha — Skill Matching Module
Extracts required skills from JD text and fuzzy-matches them against candidate skills.
Uses synonym normalization + rapidfuzz for robust matching.
"""

import logging
import re
from typing import Optional

from src.ingestion.normalizer import build_synonym_lookup

logger = logging.getLogger(__name__)

# Fuzzy match threshold (0-100 scale for rapidfuzz)
FUZZY_THRESHOLD = 80


def extract_jd_skills(
    jd_text: str,
    synonym_lookup: Optional[dict[str, str]] = None,
) -> list[str]:
    """Extract skill names from JD text by matching against the synonym dictionary.

    Strategy: tokenize the JD into n-grams (1-4 words), check each against
    the synonym lookup. Returns canonical skill names found.

    Args:
        jd_text: Raw job description text.
        synonym_lookup: Alias→canonical mapping.

    Returns:
        De-duplicated list of canonical skill names found in the JD.
    """
    if not jd_text:
        return []

    if synonym_lookup is None:
        synonym_lookup = build_synonym_lookup()

    text_lower = jd_text.lower()
    # Clean the text for matching
    text_clean = re.sub(r"[^a-z0-9\s\-\+\#\./&]", " ", text_lower)

    found_skills: set[str] = set()
    words = text_clean.split()

    # Check n-grams from 1 to 4 words
    for ngram_size in range(1, 5):
        for i in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[i : i + ngram_size])
            ngram = ngram.strip()
            if ngram in synonym_lookup:
                found_skills.add(synonym_lookup[ngram])

    # Also do direct substring matching for multi-word skill names
    for alias, canonical in synonym_lookup.items():
        if len(alias) > 2 and alias in text_lower:
            found_skills.add(canonical)

    result = sorted(found_skills)
    logger.info("Extracted %d JD skills: %s", len(result), result[:10])
    return result


def match_skills(
    candidate_skills: list[dict],
    jd_skills: list[str],
    synonym_lookup: Optional[dict[str, str]] = None,
) -> tuple[float, list[str]]:
    """Match candidate skills against JD-required skills.

    1. Normalize both sides via synonym lookup.
    2. Exact match on canonical names.
    3. Fuzzy match (rapidfuzz) as fallback for unmatched skills.

    Args:
        candidate_skills: List of skill dicts from candidate (each has 'name'
                          and optionally 'canonical_name').
        jd_skills: List of canonical JD skill names.
        synonym_lookup: Alias→canonical dict.

    Returns:
        (relevance_score 0-1, list of matched canonical skill names)
    """
    if not jd_skills:
        return 0.0, []

    if not candidate_skills or not isinstance(candidate_skills, list):
        return 0.0, []

    if synonym_lookup is None:
        synonym_lookup = build_synonym_lookup()

    # Build candidate canonical skill set
    candidate_canonical: dict[str, str] = {}  # canonical → original name
    for skill in candidate_skills:
        if not skill or not isinstance(skill, dict):
            continue
        name = skill.get("name")
        if name is None:
            continue
        name_str = str(name).lower().strip()
        if not name_str:
            continue
        canonical = skill.get("canonical_name")
        if canonical is None:
            canonical = synonym_lookup.get(name_str, name_str)
        canonical_str = str(canonical).lower().strip()
        if canonical_str:
            candidate_canonical[canonical_str] = str(name)

    # Phase 1: exact match on canonical names
    matched: set[str] = set()
    unmatched_jd: list[str] = []

    for jd_skill in jd_skills:
        if jd_skill in candidate_canonical:
            matched.add(jd_skill)
        else:
            unmatched_jd.append(jd_skill)

    # Phase 2: fuzzy match remaining with rapidfuzz
    if unmatched_jd and candidate_canonical:
        try:
            from rapidfuzz import fuzz

            candidate_names = list(candidate_canonical.keys())
            for jd_skill in unmatched_jd:
                best_score = 0.0
                best_match = None
                for cand_skill in candidate_names:
                    score = fuzz.token_sort_ratio(jd_skill, cand_skill)
                    if score > best_score:
                        best_score = score
                        best_match = cand_skill
                if best_score >= FUZZY_THRESHOLD and best_match is not None:
                    matched.add(jd_skill)
        except ImportError:
            logger.warning("rapidfuzz not installed — skipping fuzzy matching")

    # Compute score: matched / total JD skills
    relevance_score = len(matched) / len(jd_skills) if jd_skills else 0.0
    relevance_score = min(relevance_score, 1.0)

    matched_list = sorted(matched)
    logger.debug("Skill match: %d/%d JD skills matched (%.2f)",
                 len(matched), len(jd_skills), relevance_score)

    return relevance_score, matched_list
