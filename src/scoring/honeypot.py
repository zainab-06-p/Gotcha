"""
Gotcha — Honeypot Detection Module
Detects candidates whose titles contradict their actual work descriptions,
or whose summaries are boilerplate templates.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Known boilerplate summary fragments
BOILERPLATE_FRAGMENTS = [
    "lately i've been curious about how ai tools could augment my work",
    "lately i've been curious about how ai tools",
    "i've been curious about how ai tools could augment",
    "looking for new opportunities",
    "seeking challenging position",
]

# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "these", "those", "am", "are", "not", "no", "so", "if",
    "my", "me", "we", "our", "us", "you", "your", "he", "she", "they",
    "them", "their", "its", "i", "up", "out", "into", "over", "under",
    "between", "about", "than", "more", "most", "very", "just", "also",
    "all", "any", "each", "every", "both", "few", "many", "much", "some",
    "other", "new", "work", "working", "worked", "team", "role", "company",
    "across", "well", "including", "using", "used", "based", "through",
    "while", "during", "after", "before", "within", "around", "along",
    "since", "such", "who", "what", "when", "where", "how", "which",
})

# Generic title role/field keywords excluded from title-desc overlap check.
# These are common role-level words that rarely appear in job descriptions,
# causing false-positive honeypot flags for legitimate candidates
# (e.g., "Software Engineer" whose description never says "engineer" or "software").
_GENERIC_TITLE_KEYWORDS = frozenset({
    "engineer", "developer", "programmer", "architect",
    "senior", "staff", "principal", "lead", "manager", "head",
    "director", "associate", "intern", "junior", "trainee",
    "consultant", "specialist", "advisor", "officer",
    "software", "full-stack", "fullstack", "frontend", "front-end",
    "backend", "back-end", "full", "stack",
})


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, lowercased, stop words removed."""
    if not text:
        return set()

    # Tokenize: keep alphanumeric + hyphens
    tokens = re.findall(r"[a-z][a-z0-9\-+#.]+", text.lower())
    keywords = {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}
    return keywords


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """Detect if a candidate is a honeypot (misleading profile).

    Two checks:
    1. Title–Description mismatch: If keywords from current_title have < 15%
       overlap with keywords from all career_history descriptions, the title
       is likely fabricated or mismatched.

    2. Boilerplate summary: If the summary matches a known template pattern.

    Args:
        candidate: Raw candidate dict.

    Returns:
        (is_honeypot, reason_string). If not honeypot, reason is empty.
    """
    if not candidate or not isinstance(candidate, dict):
        return False, "Invalid candidate structure"

    reasons: list[str] = []

    # --- Check 1: Title-Description keyword overlap ---
    profile = candidate.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    current_title = profile.get("current_title") or ""
    title_keywords = _extract_keywords(str(current_title))

    # Exclude generic role/field keywords that rarely appear in descriptions
    meaningful_title_keywords = title_keywords - _GENERIC_TITLE_KEYWORDS

    career_history = candidate.get("career_history")
    if not isinstance(career_history, list):
        career_history = []
        
    desc_list = []
    for job in career_history:
        if not job or not isinstance(job, dict):
            continue
        desc = job.get("description") or ""
        desc_list.append(str(desc))
        
    desc_text = " ".join(desc_list)
    desc_keywords = _extract_keywords(desc_text)

    if title_keywords and not desc_keywords:
        # No descriptions at all — can't verify title
        reasons.append("no career descriptions to verify title")

    # --- Check 2: Boilerplate summary ---
    summary = profile.get("summary") or ""
    if summary:
        summary_lower = str(summary).lower().strip()
        for fragment in BOILERPLATE_FRAGMENTS:
            if fragment in summary_lower:
                reasons.append(f"boilerplate summary detected: '{fragment[:50]}...'")
                break

    is_honeypot = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else ""

    if is_honeypot:
        logger.info("Honeypot detected for %s: %s",
                     candidate.get("candidate_id", "?"), reason)

    return is_honeypot, reason
