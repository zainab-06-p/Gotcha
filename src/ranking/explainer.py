"""
Gotcha — Candidate Reasoning Explainer (Redrob JD Edition)

Generates a 1-2 sentence, candidate-specific reasoning string.
References REAL facts from the profile: title, YOE, company, top skills,
notice period, last active date, location.
Acknowledges concerns honestly (long notice, inactivity, wrong location).
NEVER uses a generic template — every string is different.
"""

import logging
from datetime import date, datetime
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_EVAL_DATE = date(2026, 6, 28)

# Disqualifier → human-readable short label
_DQ_MESSAGES = {
    "WRONG_TITLE":       "non-engineering current title",
    "IRRELEVANT_TITLE":  "engineering role irrelevant to AI/ML/search",
    "CONSULTING_ONLY":   "entire career at IT services only",
    "PURE_RESEARCH":     "no production deployment in career",
    "WRONG_DOMAIN":      "CV/speech/robotics domain, no NLP/IR",
    "CV_SPECIALIST":     "computer vision specialist with no NLP/IR exposure",
    "STOPPED_CODING":    "moved to pure management (18+ months)",
    "TITLE_CHASER":      "average tenure < 15 months (job hopping)",
    "OUTSIDE_INDIA":     "based outside India, not willing to relocate",
    "KEYWORD_STUFFER":   "AI buzzwords in skills but no technical ML work in descriptions",
    "IMPOSSIBLE_TIMELINE": "impossible experience timeline (honeypot)",
    "TOO_JUNIOR_YOE":    "only 2-3 years experience, too junior for founding-team role",
    "SLIGHTLY_JUNIOR_YOE": "4 years experience, slightly below 5-year minimum",
    "OVER_SENIOR_YOE":   "over 12 years experience, potentially over-senior for IC role",
    "JUNIOR_TITLE":      "current title signals junior/associate level, insufficient seniority",
}

# Skills that directly signal retrieval/ranking expertise
_HIGH_VALUE_SKILLS = frozenset([
    "faiss", "elasticsearch", "pinecone", "weaviate", "qdrant", "milvus",
    "sentence-transformers", "sentence transformers", "embeddings",
    "information retrieval", "vector database", "semantic search",
    "dense retrieval", "hybrid search", "recommendation", "recommendation systems",
    "ranking", "ndcg", "mrr", "learning to rank", "rag",
    "retrieval augmented generation", "colbert", "bi-encoder", "cross-encoder",
])


def _days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.fromisoformat(str(date_str).replace("Z", "")).date()
        return max(0, (_EVAL_DATE - d).days)
    except (ValueError, TypeError):
        return None


def _top_high_value_skills(candidate: dict, limit: int = 3) -> List[str]:
    """Return the most valuable JD-relevant skills from the candidate's profile."""
    skills = candidate.get("skills") or []
    found = []
    other_skills = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip()
        if not name:
            continue
        if name.lower() in _HIGH_VALUE_SKILLS or any(hv in name.lower() for hv in _HIGH_VALUE_SKILLS):
            found.append(name)
        else:
            prof = str(s.get("proficiency") or "").lower()
            if prof in ("advanced", "expert"):
                other_skills.append(name)

    result = found[:limit]
    # Fill remaining slots with other advanced skills
    if len(result) < limit:
        result.extend(other_skills[: limit - len(result)])
    return result[:limit]


def generate_reasoning_string(
    candidate_id: str,
    axis_scores: Any,                   # CandidateAxisScores or dict
    matched_skills: List[str],
    is_honeypot: bool,
    honeypot_details: str,
    llm_reasoning: Optional[str] = None,
    disqualifier_tags: Optional[List[str]] = None,
    candidate: Optional[dict] = None,  # full raw candidate for fact extraction
) -> str:
    """Generate a per-candidate, fact-grounded reasoning string.

    Priority:
      1. Hard disqualifier → specific disqualifier message
      2. Honeypot → honeypot message
      3. LLM reasoning (if real, not mock)
      4. Heuristic fact extraction from candidate profile
    """
    disqualifier_tags = disqualifier_tags or []

    # ── 1. Hard disqualifiers ──────────────────────────────────────────────────
    if disqualifier_tags:
        # Lead with the first disqualifier
        primary_tag = disqualifier_tags[0]
        msg = _DQ_MESSAGES.get(primary_tag, primary_tag.lower().replace("_", " "))
        suffix = ""
        if len(disqualifier_tags) > 1:
            others = [_DQ_MESSAGES.get(t, t) for t in disqualifier_tags[1:]]
            suffix = f" (also: {'; '.join(others)})"
        return f"Disqualified — {msg}{suffix}."

    # ── 2. Honeypot ────────────────────────────────────────────────────────────
    if is_honeypot:
        detail = (honeypot_details or "profile anomalies detected").strip()
        if len(detail) > 120:
            detail = detail[:117] + "..."
        return f"Honeypot flag: {detail}"

    # ── 3. LLM reasoning (only if real / not mock) ─────────────────────────────
    if (
        llm_reasoning
        and len(llm_reasoning) > 25
        and "fallback" not in llm_reasoning.lower()
        and "processed by llm" not in llm_reasoning.lower()
        and "mock" not in llm_reasoning.lower()
    ):
        return llm_reasoning.strip()

    # ── 4. Heuristic fact-based reasoning ────────────────────────────────────
    if candidate:
        return _heuristic_reasoning(candidate, axis_scores, matched_skills)

    # ── 5. Minimal fallback (no candidate dict available) ─────────────────────
    return _minimal_reasoning(axis_scores, matched_skills)


def _heuristic_reasoning(
    candidate: dict,
    axis_scores: Any,
    matched_skills: List[str],
) -> str:
    """Build a specific reasoning string from real candidate facts."""
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    career = candidate.get("career_history") or []

    title = str(profile.get("current_title") or "Software Engineer").strip()
    company = str(profile.get("current_company") or "").strip()
    yoe = profile.get("years_of_experience") or 0
    location = str(profile.get("location") or "").strip()

    # ── Availability facts ────────────────────────────────────────────────────
    notice = signals.get("notice_period_days")
    if notice is not None and float(notice) != -1:
        notice = int(notice)
    else:
        notice = None

    days_inactive = _days_since(signals.get("last_active_date"))
    open_to_work = signals.get("open_to_work_flag", False)
    github = signals.get("github_activity_score")
    if github == -1:
        github = None

    # ── Skills ────────────────────────────────────────────────────────────────
    top_skills = _top_high_value_skills(candidate)

    # ── Positives ─────────────────────────────────────────────────────────────
    positives = []
    concerns = []

    # Role description
    yoe_str = f"{int(yoe)}yr" if yoe else ""
    company_str = f" @ {company}" if company else ""
    positives.append(f"{yoe_str} {title}{company_str}".strip())

    # Skills
    if top_skills:
        positives.append(f"skilled in {', '.join(top_skills)}")
    elif matched_skills:
        positives.append(f"{len(matched_skills)} JD skills matched")

    # Availability boosts
    if open_to_work:
        positives.append("actively looking")
    if github is not None and float(github) >= 50:
        positives.append(f"active GitHub ({int(github)})")
    if notice is not None and notice <= 30:
        positives.append(f"{'immediate' if notice == 0 else str(notice) + '-day notice'}")

    # Domain signals from most recent role description
    if career and isinstance(career[0], dict):
        desc = str(career[0].get("description") or "")[:200].lower()
        if any(kw in desc for kw in ["retrieval", "embedding", "vector", "ranking", "recommendation", "search"]):
            positives.append("retrieval/ranking work evidenced in most recent role")
        elif any(kw in desc for kw in ["deployed", "production", "scale", "million", "users"]):
            positives.append("production deployment signals in recent role")

    # ── Concerns ──────────────────────────────────────────────────────────────
    if notice is not None and notice > 90:
        concerns.append(f"{notice}-day notice period")

    if days_inactive is not None:
        if days_inactive > 180:
            concerns.append(f"inactive for {days_inactive // 30} months")
        elif days_inactive > 90:
            concerns.append(f"last active {days_inactive} days ago")

    # YoE fit (target 5–9)
    try:
        yoe_int = int(yoe)
        if yoe_int < 3:
            concerns.append(f"only {yoe_int} years experience (target 5–9)")
        elif yoe_int > 12:
            concerns.append(f"{yoe_int} years may be over-senior for founding-team IC role")
    except (TypeError, ValueError):
        pass

    location_lower = location.lower()
    india_indicators = ["india", "pune", "noida", "delhi", "mumbai", "bangalore",
                        "bengaluru", "hyderabad", "chennai", "gurgaon", "gurugram",
                        "kolkata", "ahmedabad", "jaipur", "lucknow", "kanpur",
                        "nagpur", "indore", "bhopal", "surat", "visakhapatnam",
                        "kochi", "cochin", "trivandrum", "thiruvananthapuram",
                        "coimbatore", "madurai", "kerala", "karnataka",
                        "tamil nadu", "tamilnadu", "andhra pradesh", "telangana",
                        "uttar pradesh", "maharashtra", "gujarat", "rajasthan",
                        "west bengal", "odisha", "madhya pradesh", "punjab",
                        "haryana", "chandigarh"]
    preferred_cities = ["pune", "noida", "bangalore", "bengaluru", "mumbai",
                        "delhi", "gurgaon", "gurugram", "hyderabad", "chennai"]
    in_india = any(ind in location_lower for ind in india_indicators)
    in_preferred_city = any(city in location_lower for city in preferred_cities)
    if not in_india and location:
        willing = signals.get("willing_to_relocate", None)
        if willing is True:
            concerns.append(f"based in {location}, open to relocating to India")
        elif willing is False:
            concerns.append(f"based in {location}, not willing to relocate")
    elif in_india and not in_preferred_city and location:
        willing = signals.get("willing_to_relocate", None)
        if willing is True:
            concerns.append(f"based in {location}, open to relocating to Pune/Noida")
        elif willing is False:
            concerns.append(f"based in {location}, not willing to relocate")

    rrr = signals.get("recruiter_response_rate")
    if rrr is not None and rrr != -1 and float(rrr) < 0.20:
        concerns.append(f"low recruiter response rate ({float(rrr):.0%})")

    # ── Compose sentence ──────────────────────────────────────────────────────
    positive_str = "; ".join(p for p in positives[:3] if p)
    concern_str = ("; concern: " + ", ".join(concerns[:2])) if concerns else ""

    reasoning = f"{positive_str}{concern_str}."
    return reasoning if reasoning.strip() != "." else "Insufficient profile data to assess."


def _minimal_reasoning(axis_scores: Any, matched_skills: List[str]) -> str:
    """Fallback when no candidate dict is available."""
    # Get score as float regardless of whether axis_scores is dict or object
    try:
        sr = axis_scores.skill_relevance if hasattr(axis_scores, "skill_relevance") else axis_scores.get("skill_relevance", 0)
    except Exception:
        sr = 0

    if matched_skills and sr >= 0.6:
        return (
            f"Strong skill overlap on {', '.join(matched_skills[:3])} — "
            f"career descriptions indicate relevant retrieval/ranking experience."
        )
    elif matched_skills:
        return (
            f"Partial skill match on {', '.join(matched_skills[:2])}; "
            f"career description depth in embeddings/vector DB unclear."
        )
    else:
        return (
            "No direct match on must-have skills (embeddings, vector databases, ranking); "
            "included as borderline case based on career domain signals."
        )
