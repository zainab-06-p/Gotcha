"""
Gotcha — Hard Disqualifier Detection Module (v3)
Implements a single get_disqualifier_multiplier() used in the pipeline.
The multiplier is applied to the ENTIRE final score after all components.
"""

import logging

logger = logging.getLogger(__name__)


def _lower(text) -> str:
    if not text:
        return ""
    return str(text).lower().strip()


def yoe_fit_score(candidate: dict) -> float:
    """
    Returns a 0.0-1.0 score for years of experience fit.
    Not a disqualifier — a SCORED component added to the weighted sum.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe is None:
        return 0.50
    try:
        yoe = int(yoe)
    except (TypeError, ValueError):
        return 0.50

    if 5 <= yoe <= 9:
        return 1.00
    elif 4 <= yoe < 5:
        return 0.75
    elif 9 < yoe <= 12:
        return 0.85
    elif 3 <= yoe < 4:
        return 0.40
    elif yoe > 12:
        return 0.65
    elif yoe < 3:
        return 0.15
    else:
        return 0.50


def get_disqualifier_multiplier(candidate: dict) -> tuple[float, list[str], str]:
    """
    Returns a multiplier (0.0-1.0) that is applied to the final score AFTER
    all other components are computed.

    1.0 = no disqualifier, full score retained
    0.0 = honeypot / impossible timeline, score zeroed

    Returns:
        (multiplier: float, triggered_tags: list[str], reason: str)
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    current_title = _lower(profile.get("current_title", ""))
    country = _lower(profile.get("country", ""))

    # ── Shared signal lists ─────────────────────────────────────────────────
    CV_TITLE_SIGNALS = [
        "computer vision", "cv engineer", "vision engineer",
        "image processing", "robotics engineer", "speech engineer",
        "speech recognition", "asr engineer", "ocr engineer",
    ]

    CV_SKILL_SIGNALS = [
        "computer vision", "opencv", "object detection", "image classification",
        "yolo", "cnn", "image segmentation", "pose estimation", "face recognition",
        "optical flow", "depth estimation", "gan", "diffusion models",
        "speech recognition", "asr", "tts", "text to speech", "robotics",
    ]

    NLP_IR_SIGNALS = [
        "nlp", "information retrieval", "embeddings", "sentence transformers",
        "faiss", "elasticsearch", "qdrant", "weaviate", "pinecone", "milvus",
        "vector", "retrieval", "ranking", "recommendation", "rag",
        "bert", "transformers", "semantic search", "learning to rank",
        "text classification", "named entity", "question answering",
    ]

    # If candidate's title contains any of these ML/AI keywords, they are
    # NOT a CV specialist even if they happen to have CV skills listed.
    ML_TITLE_SIGNALS = [
        "ai engineer", "ml engineer", "machine learning", "deep learning",
        "data scientist", "data science", "nlp engineer", "llm engineer",
        "recommendation", "search engineer", "ranking engineer",
        "ai/ml", "artificial intelligence",
    ]

    NON_ML_TITLES = [
        "data analyst",
        "data engineer",
        "analytics engineer",
        "backend engineer",
        "frontend engineer",
        "full stack",
        "devops",
        "marketing",
        "sales",
        "accountant",
        "operations manager",
        "civil engineer",
        "mechanical engineer",
        "hr ",
        "human resources",
        "recruiter",
        "finance",
        "legal",
        "customer support",
        "content writer",
        "project manager",
        "business analyst",
        "office manager",
        "product manager",
        "software engineer",
        "software developer",
        "cloud engineer",
        "graphic designer",
        "ui designer",
        "ux designer",
        "network engineer",
        "systems engineer",
        "site reliability",
        "security engineer",
        "qa engineer",
        "test engineer",
        "support engineer",
    ]

    CONSULTING_FIRMS = [
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "mindtree", "mphasis", "hexaware", "ltimindtree", "tech mahindra",
        "hcl technologies", "hcl tech", "birlasoft", "niit technologies",
        "zensar", "mastech", "sonata software",
    ]

    skill_names_lower = [_lower(s.get("name", "")) for s in skills if isinstance(s, dict)]

    # Current multiplier: start at 1.0, take MIN as we go
    multiplier = 1.0
    tags = []
    reasons = []

    def _apply(new_mult: float, tag: str, reason: str):
        nonlocal multiplier, tags, reasons
        if new_mult < multiplier:
            multiplier = new_mult
        tags.append(tag)
        reasons.append(reason)

    # ═════════════════════════════════════════════════════════════════════════
    # DISQUALIFIER 1: CV / Speech / Robotics specialist
    # ═════════════════════════════════════════════════════════════════════════
    title_is_cv = any(cv in current_title for cv in CV_TITLE_SIGNALS)

    cv_skill_count = sum(
        1 for sn in skill_names_lower
        if any(cv in sn for cv in CV_SKILL_SIGNALS)
    )
    nlp_skill_count = sum(
        1 for sn in skill_names_lower
        if any(nlp in sn for nlp in NLP_IR_SIGNALS)
    )

    all_descriptions = " ".join(
        _lower(j.get("description", ""))
        for j in history if isinstance(j, dict)
    )
    nlp_in_descriptions = sum(
        1 for nlp in NLP_IR_SIGNALS
        if nlp in all_descriptions
    )

    title_is_ml = any(ml in current_title for ml in ML_TITLE_SIGNALS)
    is_cv_specialist = (title_is_cv or cv_skill_count >= 2) and not title_is_ml
    has_nlp_ir = nlp_skill_count >= 2 or nlp_in_descriptions >= 3

    if is_cv_specialist and not has_nlp_ir:
        _apply(0.12, "CV_SPECIALIST",
               f"CV specialist (title='{profile.get('current_title','')}', {cv_skill_count} CV skills), no NLP/IR exposure")

    elif is_cv_specialist and has_nlp_ir:
        _apply(0.55, "CV_SPECIALIST_PARTIAL",
               f"CV specialist with some NLP/IR ({nlp_skill_count} skills, {nlp_in_descriptions} desc signals) — partial penalty")

    # ═════════════════════════════════════════════════════════════════════════
    # DISQUALIFIER 2: Non-ML engineering / non-engineering titles
    # ═════════════════════════════════════════════════════════════════════════
    is_non_ml = any(t in current_title for t in NON_ML_TITLES)

    if is_non_ml:
        # Titles that could plausibly involve ML/AI work despite the name
        ml_adjacent_titles = (
            "data engineer" in current_title or "data analyst" in current_title
            or "software engineer" in current_title or "software developer" in current_title
            or "cloud engineer" in current_title
        )
        if ml_adjacent_titles:
            ml_skill_count = sum(
                1 for sn in skill_names_lower
                if any(nlp in sn for nlp in NLP_IR_SIGNALS)
            )
            if ml_skill_count >= 3:
                _apply(0.40, "NON_ML_TITLE",
                       f"'{profile.get('current_title','')}' but has {ml_skill_count} ML skills — partial penalty")
            else:
                _apply(0.15, "NON_ML_TITLE",
                       f"'{profile.get('current_title','')}' with only {ml_skill_count} ML skills — near-disqualified")
        else:
            _apply(0.08, "NON_ML_TITLE",
                   f"'{profile.get('current_title','')}' is not an ML/AI engineering role")

    # ═════════════════════════════════════════════════════════════════════════
    # DISQUALIFIER 3: Consulting-only career
    # ═════════════════════════════════════════════════════════════════════════
    all_companies_lower = [_lower(j.get("company", "")) for j in history if isinstance(j, dict)]

    if len(all_companies_lower) > 0:
        consulting_only = all(
            any(cf in company for cf in CONSULTING_FIRMS)
            for company in all_companies_lower
        )
        if consulting_only:
            companies_str = ", ".join(set(all_companies_lower[:3]))
            _apply(0.20, "CONSULTING_ONLY",
                   f"entire career at IT services firms ({companies_str})")

    # ═════════════════════════════════════════════════════════════════════════
    # DISQUALIFIER 4: Location — outside India, not willing to relocate
    # ═════════════════════════════════════════════════════════════════════════
    if country not in ("india", "") and not signals.get("willing_to_relocate", False):
        _apply(0.10, "OUTSIDE_INDIA",
               f"based in {profile.get('country','')}, not willing to relocate to India")

    # ═════════════════════════════════════════════════════════════════════════
    # DISQUALIFIER 5: Honeypot — impossible timeline
    # ═════════════════════════════════════════════════════════════════════════
    total_career_months = sum(
        int(j.get("duration_months", 0) or 0)
        for j in history if isinstance(j, dict)
    )
    claimed_yoe = profile.get("years_of_experience", 0)
    if claimed_yoe is not None:
        try:
            claimed_yoe_months = int(claimed_yoe) * 12
            if claimed_yoe_months > 0 and total_career_months > 0 and claimed_yoe_months > total_career_months * 3:
                _apply(0.0, "IMPOSSIBLE_TIMELINE",
                       f"claims {claimed_yoe} YoE but career history only sums to {total_career_months/12:.1f}y — impossible timeline")
        except (TypeError, ValueError):
            pass

    # ── Honeypot: expert with zero duration ─────────────────────────────────
    expert_with_zero_duration = sum(
        1 for s in skills
        if _lower(s.get("proficiency", "")) == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_with_zero_duration >= 4:
        _apply(0.0, "EXPERT_ZERO_DURATION",
               f"claims expert in {expert_with_zero_duration} skills with 0 months usage — honeypot")

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOUND PENALTIES — multiply into current multiplier (not min-based)
    # ═════════════════════════════════════════════════════════════════════════

    # Penalty 1: Junior title — steep experience discount
    if "junior" in current_title:
        multiplier *= 0.60
        tags.append("JUNIOR_TITLE")
        reasons.append(f"junior title '{profile.get('current_title','')}' — experience discount")

    # Penalty 2: Strict YOE scoring
    yoe_raw = candidate.get("profile", {}).get("years_of_experience", 0)
    yoe_val = 0
    if yoe_raw is not None:
        try:
            yoe_val = int(yoe_raw)
        except (TypeError, ValueError):
            pass
    if yoe_val < 3:
        multiplier *= 0.25
        tags.append("YOE_LOW")
        reasons.append(f"only {yoe_val} years experience (< 3) — major penalty")
    elif yoe_val < 4:
        multiplier *= 0.50
        tags.append("YOE_LOW")
        reasons.append(f"only {yoe_val} years experience (< 4) — substantial penalty")
    elif yoe_val < 5:
        multiplier *= 0.75
        tags.append("YOE_LOW")
        reasons.append(f"only {yoe_val} years experience (< 5) — moderate penalty")
    elif yoe_val > 12:
        multiplier *= 0.80
        tags.append("YOE_HIGH")
        reasons.append(f"{yoe_val} years experience (> 12) — seniority mismatch")

    # Penalty 3: Vision-only skill profile (no NLP/IR skills)
    VISION_ONLY_SIGNALS = [
        "yolo", "opencv", "cnn", "gan", "speech recognition", "asr",
    ]
    has_vision = any(v in sn for v in VISION_ONLY_SIGNALS for sn in skill_names_lower)
    has_retrieval_nlp = any(nlp in sn for nlp in NLP_IR_SIGNALS for sn in skill_names_lower)
    if has_vision and not has_retrieval_nlp:
        multiplier *= 0.30
        tags.append("VISION_ONLY")
        reasons.append("vision-only skills (YOLO/OpenCV/CNN/GAN/ASR) without NLP/retrieval — major penalty")

    # ═════════════════════════════════════════════════════════════════════════
    # Build final result
    # ═════════════════════════════════════════════════════════════════════════
    reason_str = "; ".join(reasons)
    if multiplier < 1.0 and tags:
        logger.info("Candidate %s disqualified by: %s (multiplier=%.2f)",
                     candidate.get("candidate_id", "?"), tags, multiplier)

    return multiplier, tags, reason_str


def run_all_disqualifiers(candidate: dict) -> dict:
    """
    Legacy wrapper — returns the same dict format expected by the pipeline.

    Returns:
        {
            "is_disqualified": bool,
            "disqualifiers_triggered": [str],
            "score_cap": float,
            "reasons": str,
        }
    """
    multiplier, tags, reason = get_disqualifier_multiplier(candidate)
    return {
        "is_disqualified": multiplier < 1.0,
        "disqualifiers_triggered": tags,
        "score_cap": multiplier,
        "reasons": reason,
    }
