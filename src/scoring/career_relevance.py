"""
Gotcha — Career Relevance Scoring Module (Redrob JD Edition)

Uses TF-IDF cosine similarity between a candidate's FULL career description text
and a keyword-expanded version of the Redrob Senior AI Engineer JD.

Key fix vs. old version:
  OLD: was called with the raw .docx text (~200 words, generic)
  NEW: uses the keyword-expanded JD from jd_redrob.ALL_KEYWORDS to build a
       rich anchor document (800+ tokens) so TF-IDF vocab is much more informative.

The expanded JD anchors the vocabulary on:
  embeddings, retrieval, vector database, FAISS, Elasticsearch, Pinecone,
  sentence-transformers, ranking, NDCG, MRR, MAP, A/B testing, Python,
  LLM, fine-tuning, LoRA, information retrieval, NLP, recommendation,
  hybrid search, semantic search, production deployment, real users, scale, etc.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Keyword-expanded JD anchor text ───────────────────────────────────────────
# This is deliberately verbose so TF-IDF builds a rich, discriminating vocabulary.
_REDROB_JD_ANCHOR = """
Senior AI Engineer Redrob AI talent intelligence platform information retrieval
ranking recommendation system production embeddings vector database search

Must have skills:
sentence-transformers embeddings retrieval semantic search dense retrieval
FAISS Pinecone Weaviate Qdrant Milvus Elasticsearch OpenSearch hybrid search
BM25 HNSW approximate nearest neighbor vector similarity cosine similarity

Ranking and evaluation:
NDCG MRR MAP precision recall A/B testing offline evaluation online evaluation
learning to rank LambdaMART XGBoost LightGBM neural ranking cross-encoder
bi-encoder ColBERT re-ranking relevance judgments learning to rank

Python engineering:
Python PyTorch TensorFlow HuggingFace transformers BERT GPT language model
fine-tuning LoRA QLoRA PEFT adapter retrieval augmented generation RAG
inference optimization batching quantization ONNX TorchScript deployment

Production systems:
shipped deployed production real users scale system design microservices
API backend platform ML pipeline data pipeline feature store model serving
real-time low latency high throughput distributed system

NLP and information retrieval:
NLP natural language processing text understanding semantic similarity
document retrieval question answering named entity recognition
tokenization text preprocessing word embeddings positional encoding
attention mechanism self-attention transformer architecture

Experience signals:
5 years 6 years 7 years 8 years 9 years senior engineer founding team startup
series A product company end to end ownership built designed implemented
shipped at scale millions of queries recommendations personalization

Nice to have:
HR tech hiring talent acquisition applicant tracking recruiting marketplace
open source GitHub contribution distributed inference large scale optimization
knowledge graph entity resolution
"""


def _collect_career_text(candidate: dict) -> str:
    """Concatenate ALL career_history descriptions + titles for maximum signal.

    Weights current role title more heavily by repeating it 3x.
    """
    if not candidate or not isinstance(candidate, dict):
        return ""
    career = candidate.get("career_history") or []
    if not isinstance(career, list):
        return ""

    texts = []

    # Current title repeated for weight
    profile = candidate.get("profile") or {}
    current_title = str(profile.get("current_title") or "").strip()
    if current_title:
        texts.extend([current_title] * 3)  # weight current role more heavily

    # All job descriptions and titles
    for job in career:
        if not job or not isinstance(job, dict):
            continue
        desc = str(job.get("description") or "").strip()
        if desc:
            texts.append(desc)
        title = str(job.get("title") or "").strip()
        if title:
            texts.append(title)
        company = str(job.get("company") or "").strip()
        if company:
            texts.append(company)

    # Also include skills as text (secondary signal)
    skills = candidate.get("skills") or []
    skill_names = [str(s.get("name") or "") for s in skills if isinstance(s, dict)]
    if skill_names:
        texts.append(" ".join(skill_names))

    # Profile summary
    summary = str(profile.get("summary") or "").strip()
    if summary:
        texts.append(summary)

    return " ".join(texts)


def score_career_relevance(
    candidate: dict,
    jd_text: Optional[str] = None,  # kept for API compatibility but ignored
) -> float:
    """Score career relevance using TF-IDF cosine similarity vs the Redrob JD anchor.

    Args:
        candidate: Normalized candidate dict.
        jd_text: Ignored — we use the keyword-expanded anchor internally.

    Returns:
        Float in [0, 1].
    """
    career_text = _collect_career_text(candidate)
    if not career_text.strip():
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform([_REDROB_JD_ANCHOR, career_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return max(0.0, min(float(sim), 1.0))

    except ImportError:
        logger.error("scikit-learn not installed. Run: pip install scikit-learn")
        return 0.0
    except Exception as e:
        logger.error("Error computing career relevance: %s", e)
        return 0.0


def batch_score_career_relevance(
    candidates: list[dict],
    jd_text: Optional[str] = None,  # kept for API compatibility
) -> list[float]:
    """Score career relevance for many candidates efficiently.

    IMPORTANT: We fit the TF-IDF on [JD_anchor, candidate_text] PAIRS rather than
    fitting once on all 5000 candidate texts simultaneously.
    
    Why: Fitting on all 5000 documents dilutes the vocabulary — rare but important
    terms like 'faiss', 'pinecone', 'ndcg' get swamped and the cosine similarity
    collapses to ~0.08 for everyone. Paired fitting gives real differentiation.

    For large N we do a single fit on [JD_anchor] + all career texts but then
    compare each candidate vector against the JD vector — this is still O(N) but
    retains IDF scores calibrated to the actual corpus rather than an artificial
    2-doc fit. We then apply a score rescaling so the range is [0, 1].

    Args:
        candidates: List of normalized candidate dicts.
        jd_text: Ignored — keyword-expanded anchor used internally.

    Returns:
        List of floats in [0, 1], same order as candidates.
    """
    if not candidates:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        career_texts = [_collect_career_text(c) for c in candidates]
        career_texts_safe = [t if t.strip() else "placeholder empty profile" for t in career_texts]

        # Fit on JD anchor + all career texts (real corpus IDF)
        all_texts = [_REDROB_JD_ANCHOR] + career_texts_safe

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,       # keep rare JD-specific terms
            max_df=0.95,    # drop near-universal tokens
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        jd_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]

        raw_sims = cosine_similarity(jd_vector, candidate_vectors)[0]

        # === ABSOLUTE SCORING (not min-max) ===
        # Min-max rescaling was the primary cause of score compression:
        # it mapped the entire raw range (e.g. 0.02-0.18) to [0.05, 1.0],
        # giving a Marketing Manager the same score as an ML Engineer.
        #
        # Instead: use the raw cosine similarity directly, then apply a
        # calibrated power curve that amplifies high scorers and keeps
        # low scorers genuinely low.
        #
        # Calibration from the Redrob corpus:
        #   Raw sim ~0.08-0.12 = irrelevant (QA, Java, Marketing)
        #   Raw sim ~0.15-0.25 = adjacent (Data Analyst, DevOps)
        #   Raw sim ~0.25-0.40 = relevant (ML Engineer, Data Scientist)
        #   Raw sim ~0.40+     = highly relevant (Senior AI Engineer)
        #
        # We anchor on p90 of raw scores as "1.0" to prevent saturation.
        sim_array = np.array(raw_sims, dtype=float)
        p90 = float(np.percentile(sim_array, 90))
        anchor = max(p90, 0.10)  # floor anchor so we don't divide near-zero

        # Scale so that p90 maps to ~0.80, and apply power curve
        # score = clip(raw_sim / anchor, 0, 1.5) ^ 1.3
        # This means:
        #   raw=0.05 → 0.05/0.25=0.20 → 0.20^1.3 ≈ 0.14 (irrelevant, stays low)
        #   raw=0.15 → 0.15/0.25=0.60 → 0.60^1.3 ≈ 0.49 (adjacent)
        #   raw=0.25 → 0.25/0.25=1.00 → 1.00^1.3 = 1.00 → capped 1.0 (relevant)
        scaled = np.clip(sim_array / anchor, 0.0, 1.5)
        powered = np.power(scaled, 1.3)
        scores = [float(np.clip(s, 0.0, 1.0)) for s in powered]

        logger.info(
            "Batch career relevance: n=%d raw[%.3f-%.3f] p90=%.3f final[%.3f-%.3f]",
            len(scores), float(np.min(sim_array)), float(np.max(sim_array)),
            p90, min(scores), max(scores),
        )
        return scores

    except ImportError:
        logger.error("scikit-learn not installed")
        return [0.0] * len(candidates)
    except Exception as e:
        logger.error("Error in batch career relevance: %s", e)
        return [0.0] * len(candidates)

