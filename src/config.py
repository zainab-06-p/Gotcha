"""
Gotcha — Central Configuration
All weights, thresholds, data structures, and constants live here.
Every module imports from this file — single source of truth.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"

# Challenge data paths (adjust if your data is elsewhere)
CHALLENGE_DATA_DIR = PROJECT_ROOT.parent / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
CANDIDATES_JSONL = CHALLENGE_DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON = CHALLENGE_DATA_DIR / "sample_candidates.json"
JOB_DESCRIPTION_DOCX = CHALLENGE_DATA_DIR / "job_description.docx"

# ============================================================================
# Scoring Weights — Track 1 (Deterministic)
# These sum to 1.0
# ============================================================================
TRACK1_WEIGHTS = {
    "skill_relevance": 0.25,      # JD skill ↔ candidate skill fuzzy match
    "skill_trust": 0.20,          # Assessment scores, duration, endorsements
    "career_relevance": 0.20,     # TF-IDF keyword match on descriptions
    "behavioral": 0.10,           # github, recruiter saves, response rate
    "feasibility": 0.10,          # notice period, salary, work mode
    "data_confidence": 0.10,      # profile completeness, verifications
    "pedigree": 0.05,             # education tier (deliberately capped low)
}

# ============================================================================
# Skill Trust Sub-Weights
# How we compute trust for each individual skill
# ============================================================================
SKILL_TRUST_WEIGHTS = {
    "assessment_score": 0.45,     # Only verified signal — highest weight
    "duration_months": 0.25,      # Time using the skill
    "proficiency": 0.15,          # Self-reported level
    "endorsements": 0.15,         # Social proof (gameable, but some signal)
}

# When assessment score is missing, redistribute its weight
SKILL_TRUST_WEIGHTS_NO_ASSESSMENT = {
    "duration_months": 0.45,
    "proficiency": 0.30,
    "endorsements": 0.25,
}

# ============================================================================
# Proficiency Mapping
# ============================================================================
PROFICIENCY_SCORES = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.0,
}

# ============================================================================
# Education Tier Mapping
# ============================================================================
TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.30,
}

# ============================================================================
# Behavioral Signal Normalization
# These are the max values used for 0-1 normalization
# ============================================================================
BEHAVIORAL_NORMS = {
    "github_activity_score_max": 100.0,
    "search_appearance_30d_max": 500.0,
    "saved_by_recruiters_30d_max": 50.0,
    "recruiter_response_rate_max": 1.0,  # already 0-1
}

# Sentinel value in the dataset meaning "not applicable"
SENTINEL_VALUE = -1

# ============================================================================
# Feasibility Thresholds
# ============================================================================
NOTICE_PERIOD_IDEAL_MAX = 30     # days — immediate joiners preferred
NOTICE_PERIOD_ACCEPTABLE_MAX = 90
NOTICE_PERIOD_PENALTY_MAX = 180  # 6 months — heavy penalty

# ============================================================================
# Honeypot Detection
# ============================================================================
HONEYPOT_KEYWORD_OVERLAP_THRESHOLD = 0.15  # If title keywords overlap < 15% with description keywords, flag it
HONEYPOT_PENALTY = 0.3  # Multiply score by this if flagged as honeypot

# ============================================================================
# Track 2 — LLM Configuration
# ============================================================================
LLM_PROVIDER = "gemini"  # "gemini", "groq", or "ollama"
LLM_MODEL = "gemini-2.0-flash"
LLM_TOP_N_CANDIDATES = 300  # Only run LLM on top N from Track 1
TRACK2_BLEND_WEIGHT = 0.5  # final = (1-w)*track1 + w*track2

# ============================================================================
# Clustering
# ============================================================================
N_ARCHETYPES = 8  # Number of role archetypes to discover
REDIRECT_LOW_THRESHOLD = 0.35   # Match score below this → candidate doesn't fit this job
REDIRECT_HIGH_THRESHOLD = 0.65  # Archetype match above this → strong fit elsewhere

# ============================================================================
# Output
# ============================================================================
SUBMISSION_TOP_N = 100  # Required by challenge: exactly 100 ranked candidates

# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class SkillTrust:
    """Trust score for a single skill on a candidate's profile."""
    skill_name: str
    canonical_name: str  # After synonym normalization
    trust_score: float   # 0-1, weighted blend
    assessment_score: Optional[float] = None  # 0-100 if available
    duration_months: int = 0
    proficiency: str = "beginner"
    endorsements: int = 0
    evidenced_in_career: bool = False  # True if found in career_history descriptions


@dataclass
class CandidateAxisScores:
    """The 7-axis profile vector for a candidate."""
    skill_relevance: float = 0.0       # Axis 1: recomputed per JD
    experience_impact: float = 0.0     # Axis 2: seniority, scope
    domain_coherence: float = 0.0      # Axis 3: career thread consistency
    narrative_credibility: float = 0.0 # Axis 4: summary specificity
    behavioral_validation: float = 0.0 # Axis 5: github, recruiter signals
    engagement_feasibility: float = 0.0# Axis 6: notice, salary, mode
    pedigree: float = 0.0             # Axis 7: education tier (capped)

    def to_dict(self):
        return {
            "skill_relevance": round(self.skill_relevance, 4),
            "experience_impact": round(self.experience_impact, 4),
            "domain_coherence": round(self.domain_coherence, 4),
            "narrative_credibility": round(self.narrative_credibility, 4),
            "behavioral_validation": round(self.behavioral_validation, 4),
            "engagement_feasibility": round(self.engagement_feasibility, 4),
            "pedigree": round(self.pedigree, 4),
        }

    def to_vector(self):
        """Return as a list for numpy operations."""
        return [
            self.skill_relevance,
            self.experience_impact,
            self.domain_coherence,
            self.narrative_credibility,
            self.behavioral_validation,
            self.engagement_feasibility,
            self.pedigree,
        ]


@dataclass
class JDProfile:
    """Parsed job description → ideal candidate profile."""
    title: str = ""
    must_have_skills: list = field(default_factory=list)   # [{"skill": str, "weight": float}]
    nice_to_have_skills: list = field(default_factory=list)
    axis_weights: dict = field(default_factory=lambda: {
        "skill_relevance": 0.30,
        "experience_impact": 0.20,
        "domain_coherence": 0.15,
        "narrative_credibility": 0.10,
        "behavioral_validation": 0.10,
        "engagement_feasibility": 0.10,
        "pedigree": 0.05,
    })
    seniority_expected: str = "mid"
    logistics: dict = field(default_factory=dict)
    raw_text: str = ""
    all_keywords: list = field(default_factory=list)  # Extracted keywords for TF-IDF matching


@dataclass
class CandidateResult:
    """Final scored result for one candidate."""
    candidate_id: str
    track1_score: float = 0.0
    track2_score: Optional[float] = None
    final_score: float = 0.0
    axis_scores: CandidateAxisScores = field(default_factory=CandidateAxisScores)
    reasoning: str = ""
    is_honeypot: bool = False
    honeypot_details: str = ""
    redirect_suggestion: Optional[str] = None
    redirect_reason: str = ""
    top_matching_skills: list = field(default_factory=list)
    data_confidence: float = 0.0
