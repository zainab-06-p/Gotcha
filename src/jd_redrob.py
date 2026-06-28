"""
Gotcha — Hardcoded JD Profile: Senior AI Engineer at Redrob AI
This module returns the pre-parsed JDProfile for the specific challenge role
so the pipeline can be run without any LLM JD-parsing step.

Role: Senior AI Engineer — Founding Team at Redrob AI (Series A, Pune/Noida)
"""

from src.config import JDProfile

# ─────────────────────────────────────────────────────────────────────────────
# Hard requirements mapped to skills (weight = relative importance)
# ─────────────────────────────────────────────────────────────────────────────
MUST_HAVE_SKILLS = [
    {"skill": "sentence-transformers",   "weight": 0.12},
    {"skill": "embeddings",              "weight": 0.12},
    {"skill": "vector database",         "weight": 0.12},
    {"skill": "information retrieval",   "weight": 0.10},
    {"skill": "python",                  "weight": 0.10},
    {"skill": "ranking systems",         "weight": 0.10},
    {"skill": "NDCG",                    "weight": 0.08},  # evaluation frameworks
    {"skill": "faiss",                   "weight": 0.07},
    {"skill": "elasticsearch",           "weight": 0.07},
    {"skill": "recommendation systems",  "weight": 0.07},
    {"skill": "search",                  "weight": 0.05},
]

NICE_TO_HAVE_SKILLS = [
    {"skill": "lora",                    "weight": 0.15},
    {"skill": "qlora",                   "weight": 0.10},
    {"skill": "peft",                    "weight": 0.10},
    {"skill": "learning to rank",        "weight": 0.15},
    {"skill": "xgboost",                 "weight": 0.10},
    {"skill": "hr tech",                 "weight": 0.10},
    {"skill": "distributed systems",     "weight": 0.15},
    {"skill": "open source",             "weight": 0.05},
    {"skill": "github",                  "weight": 0.05},
    {"skill": "qdrant",                  "weight": 0.05},
]

# JD-specific axis weights (sum to 1.0)
# Overrides the default 7-axis weights for this specific role
JD_AXIS_WEIGHTS = {
    "skill_relevance":          0.30,   # Retrieval/ranking/search skills — most critical
    "experience_impact":        0.25,   # Production deployment + seniority
    "domain_coherence":         0.15,   # Career narrative is search/ML focused
    "narrative_credibility":    0.10,   # Descriptions show real technical work
    "behavioral_validation":    0.10,   # GitHub, recruiter response, open-to-work
    "engagement_feasibility":   0.07,   # Notice period, India location
    "pedigree":                 0.03,   # Education (very low weight)
}

# All keywords used for TF-IDF matching against career descriptions
ALL_KEYWORDS = [
    "embedding", "embeddings", "vector", "retrieval", "ranking", "search",
    "recommendation", "recommendation system", "sentence transformer",
    "semantic search", "dense retrieval", "sparse retrieval", "hybrid search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "annoy", "hnsw", "bm25", "ndcg", "mrr", "map", "precision",
    "recall", "a/b test", "evaluation", "offline evaluation", "online evaluation",
    "python", "pytorch", "tensorflow", "hugging face", "transformers", "bert",
    "llm", "fine-tuning", "lora", "qlora", "peft", "rag", "retrieval augmented",
    "information retrieval", "learning to rank", "lambdamart", "xgboost",
    "neural ranking", "cross-encoder", "bi-encoder", "colbert",
    "distributed", "inference optimization", "scalability", "real users",
    "production", "deployed", "shipped", "platform", "ai engineer",
    "ml engineer", "applied scientist",
]

RAW_JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI

Redrob AI is a Series A AI-native talent intelligence platform based in Pune/Noida, India.
They are building their AI engineering org from scratch and need someone to own the intelligence 
layer — the ranking, retrieval, and matching systems of their product.

HARD REQUIREMENTS:
1. Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
2. Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch)
3. Strong Python — code quality matters
4. Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing)
5. 5–9 years total experience
6. Has shipped at least one end-to-end ranking, search, or recommendation system to real users at scale
7. Product company experience — not purely IT services
8. Located in India (Pune, Noida, Delhi NCR, Mumbai, Hyderabad, Bangalore preferred) OR willing to relocate

NICE TO HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural LTR)
- HR-tech, recruiting tech, or marketplace product background
- Distributed systems or large-scale inference optimization
- Open-source contributions
- Active GitHub

LOCATION: India (Pune/Noida preferred)
SENIORITY: Senior (5-9 years)
NOTICE PERIOD: Sub-30 days preferred
"""


def get_redrob_jd_profile() -> JDProfile:
    """Return the hardcoded JD profile for Senior AI Engineer at Redrob AI.
    
    This bypasses the LLM parsing step and returns the exact requirements
    as defined by the challenge judges.
    
    Returns:
        JDProfile with all required fields populated.
    """
    profile = JDProfile()
    profile.title = "Senior AI Engineer — Founding Team at Redrob AI"
    profile.must_have_skills = MUST_HAVE_SKILLS
    profile.nice_to_have_skills = NICE_TO_HAVE_SKILLS
    profile.axis_weights = JD_AXIS_WEIGHTS
    profile.seniority_expected = "senior"
    profile.logistics = {
        "work_mode": "hybrid",
        "min_experience_years": 5,
        "max_experience_years": 9,
        "location": "India",
        "notice_period_preferred_days": 30,
    }
    profile.raw_text = RAW_JD_TEXT
    profile.all_keywords = ALL_KEYWORDS
    return profile
