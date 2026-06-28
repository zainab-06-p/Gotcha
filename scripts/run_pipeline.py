"""
Gotcha — End-to-End Pipeline Execution Script (Redrob JD Edition)
Runs Track 1 scoring on all candidates using the hardcoded Senior AI Engineer JD,
applies 8 hard disqualifiers before scoring, runs Track 2 LLM on top 300,
clusters candidates into archetypes, flags redirects, and outputs the final ranked CSV.
"""

import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHALLENGE_DATA_DIR,
    CANDIDATES_JSONL,
    SAMPLE_CANDIDATES_JSON,
    JOB_DESCRIPTION_DOCX,
    OUTPUTS_DIR,
    LLM_TOP_N_CANDIDATES,
    TRACK1_WEIGHTS,
    CandidateResult,
    CandidateAxisScores,
)
from src.ingestion.loader import (
    load_job_description,
    stream_candidates_jsonl,
    load_sample_candidates,
)
from src.ingestion.normalizer import (
    build_synonym_lookup,
    normalize_candidate,
    compute_data_confidence,
)
from src.scoring.skill_matcher import match_skills
from src.scoring.trust_scorer import score_candidate_trust
from src.scoring.career_relevance import batch_score_career_relevance
from src.scoring.behavioral import score_behavioral
from src.scoring.feasibility import score_feasibility
from src.scoring.pedigree import score_pedigree
from src.scoring.honeypot import detect_honeypot
from src.scoring.disqualifier import run_all_disqualifiers, yoe_fit_score
from src.jd_redrob import get_redrob_jd_profile, ALL_KEYWORDS
from src.llm.extractor import evaluate_candidate
from src.ranking.combiner import combine_scores
from src.ranking.ranker import rank_candidates, write_submission_csv
from src.ranking.explainer import generate_reasoning_string
from src.clustering.archetypes import fit_archetypes
from src.redirect.detector import detect_redirect

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "pipeline.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("run_pipeline")


def _nice_to_have_bonus(candidate: dict) -> float:
    """Compute a bonus score [0, 0.15] for nice-to-have skills.

    Checks skills and career descriptions against the nice-to-have list:
    - LLM fine-tuning (LoRA, QLoRA, PEFT)
    - Learning-to-rank
    - HR-tech background
    - Distributed systems
    - Active GitHub / open-source
    """
    NICE_KEYWORDS = frozenset([
        "lora", "qlora", "peft", "fine-tun", "learning to rank", "lambdamart",
        "hr tech", "hrtech", "talent", "recruiting", "marketplace",
        "distributed", "inference optimization", "large-scale inference",
        "open source", "github", "open-source contribution",
    ])

    skills = candidate.get("skills", []) or []
    skill_text = " ".join([str(s.get("name", "")).lower() for s in skills if isinstance(s, dict)])
    career = candidate.get("career_history", []) or []
    desc_text = " ".join([str(j.get("description", "")).lower() for j in career if isinstance(j, dict)])
    combined = f"{skill_text} {desc_text}"

    matched = sum(1 for kw in NICE_KEYWORDS if kw in combined)
    # Up to 5 matches = full 0.15 bonus
    bonus = min(matched / 5.0, 1.0) * 0.15
    return bonus



def run_pipeline(
    jd_path: Path,
    candidates_path: Path,
    output_path: Path,
    is_jsonl: bool,
    limit: Optional[int] = None,
    sample_only: bool = False,
) -> None:
    logger.info("=" * 68)
    logger.info("Gotcha — Redrob Senior AI Engineer JD Edition")
    logger.info("=" * 68)
    logger.info("Candidates: %s", candidates_path)
    logger.info("Output:     %s", output_path)

    # 1. Load hardcoded JD profile (no LLM call needed for JD parsing)
    logger.info("Step 1: Loading hardcoded Redrob JD profile...")
    jd_profile = get_redrob_jd_profile()
    jd_skills = [s.get("skill") for s in jd_profile.must_have_skills if s.get("skill")]
    logger.info("JD: %s | Must-have skills: %s", jd_profile.title, jd_skills)

    # 2. Build Synonym Lookup
    logger.info("Step 2: Building skill synonym lookup...")
    synonym_lookup = build_synonym_lookup()

    # 3. Load Candidates
    logger.info("Step 3: Loading candidates from %s...", candidates_path)
    candidates = []
    if is_jsonl:
        for c in tqdm(stream_candidates_jsonl(candidates_path, limit=limit), desc="Streaming JSONL"):
            candidates.append(c)
    else:
        cands = load_sample_candidates(candidates_path)
        if limit:
            cands = cands[:limit]
        candidates = cands

    total_candidates = len(candidates)
    logger.info("Loaded %d candidates total.", total_candidates)

    if total_candidates == 0:
        logger.error("No candidates loaded. Exiting.")
        sys.exit(1)

    # 4. Normalize all candidates
    logger.info("Step 4: Normalizing candidate profiles...")
    normalized_candidates = []
    for c in tqdm(candidates, desc="Normalizing Profiles"):
        normalized_candidates.append(normalize_candidate(c, synonym_lookup))

    # 5. Pre-compute batch career relevance (TF-IDF)
    logger.info("Step 5: TF-IDF career relevance (batch)...")
    career_relevance_scores = batch_score_career_relevance(normalized_candidates, jd_profile.raw_text)

    # 6. Track 1 Deterministic Scoring + Disqualifier Check
    logger.info("Step 6: Track 1 scoring + hard disqualifier checks...")
    results = []
    disqualified_count = 0
    honeypot_count = 0
    # Store disqualifier tags here during step 6 so step 8 doesn't re-run them
    disqualifier_map: dict[str, list] = {}

    for idx, c in enumerate(tqdm(normalized_candidates, desc="Track 1 Axis Scoring")):
        candidate_id = c.get("candidate_id", "CAND_0000000")

        # ── Hard disqualifier check (run FIRST, cheapest filter) ─────────────
        disq_result = run_all_disqualifiers(c)
        disq_multiplier = disq_result["score_cap"]   # 0.0/0.05/0.10/0.15 or 1.0
        disq_tags = disq_result["disqualifiers_triggered"]
        disqualifier_map[candidate_id] = disq_tags
        if disq_result["is_disqualified"]:
            disqualified_count += 1
            if disq_multiplier == 0.0:
                honeypot_count += 1

        # ── Core axis scoring ─────────────────────────────────────────────────
        skill_relevance, matched_list = match_skills(c.get("skills", []), jd_skills, synonym_lookup)
        skill_trust        = score_candidate_trust(c, jd_skills, synonym_lookup)
        career_relevance   = career_relevance_scores[idx]
        behavioral         = score_behavioral(c.get("redrob_signals", {}))
        feasibility        = score_feasibility(c.get("redrob_signals", {}), jd_profile)
        data_confidence    = compute_data_confidence(c)
        pedigree           = score_pedigree(c)

        # YoE fit (independent scored component)
        profile = c.get("profile", {}) or {}
        yoe_fit = yoe_fit_score(c)

        # Nice-to-have bonus (additive, before disqualifier)
        nih_bonus = _nice_to_have_bonus(c)

        # Compose axis scores
        axis_scores = CandidateAxisScores(
            skill_relevance=skill_relevance,
            experience_impact=skill_trust,
            domain_coherence=career_relevance,
            narrative_credibility=data_confidence,
            behavioral_validation=behavioral,
            engagement_feasibility=feasibility,
            pedigree=pedigree,
        )

        # Weighted Track 1 score
        t1_raw = (
            0.25 * skill_relevance
            + 0.20 * skill_trust
            + 0.18 * career_relevance
            + 0.10 * behavioral
            + 0.10 * feasibility
            + 0.10 * yoe_fit
            + 0.05 * data_confidence
            + 0.02 * pedigree
        )
        t1_raw = min(t1_raw + nih_bonus, 1.0)

        # ── Apply disqualifier as a direct multiplier ─────────────────────────
        # No power stretch — raw * mult gives the full spread naturally.
        t1_score = t1_raw * disq_multiplier

        # ── Honeypot check ────────────────────────────────────────────────────
        is_honeypot, honeypot_details = detect_honeypot(c)
        if is_honeypot and disq_result["reasons"]:
            honeypot_details = disq_result["reasons"] + " | " + honeypot_details

        # ── Combine scores (NO double-penalty: honeypot penalty only for
        #    genuine honeypots detected by detect_honeypot, not disqualified) ──
        final_score = combine_scores(t1_score, None, is_honeypot)

        res = CandidateResult(
            candidate_id=candidate_id,
            track1_score=t1_score,
            final_score=final_score,
            axis_scores=axis_scores,
            is_honeypot=is_honeypot or ("KEYWORD_STUFFER" in disq_tags),
            honeypot_details=honeypot_details or disq_result.get("reasons", ""),
            top_matching_skills=matched_list,
            data_confidence=data_confidence,
        )
        results.append((res, c))

    logger.info("Disqualified: %d candidates | Honeypots: %d", disqualified_count, honeypot_count)

    # 7. Track 2: LLM Enhancement on top N candidates
    logger.info("Step 7: Track 2 LLM enhancement on top %d candidates...", LLM_TOP_N_CANDIDATES)
    results.sort(key=lambda x: x[0].final_score, reverse=True)
    top_n_to_enhance = min(LLM_TOP_N_CANDIDATES, len(results))

    for i in tqdm(range(top_n_to_enhance), desc="Track 2 LLM Processing"):
        res, c = results[i]
        if res.is_honeypot or res.final_score < 0.05:
            continue  # Skip disqualified in LLM pass

        eval_data = evaluate_candidate(c, jd_profile)
        is_mock = eval_data.get("__is_mock__", True)

        # Skip score blending if LLM is mock — mock returns 0.5 for everyone,
        # which compresses the score distribution and inflates weak candidates.
        # Keep track1_score as-is when using mock.
        if is_mock:
            if eval_data.get("reasoning", ""):
                res.reasoning = eval_data["reasoning"]
            continue

        llm_exp  = eval_data.get("experience_impact", 0.5)
        llm_coh  = eval_data.get("domain_coherence", 0.5)
        llm_cred = eval_data.get("narrative_credibility", 0.5)
        reasoning = eval_data.get("reasoning", "")

        res.axis_scores.experience_impact = (res.axis_scores.experience_impact + llm_exp) / 2.0
        res.axis_scores.domain_coherence  = (res.axis_scores.domain_coherence + llm_coh) / 2.0
        res.axis_scores.narrative_credibility = (res.axis_scores.narrative_credibility + llm_cred) / 2.0

        t2_score = (llm_exp + llm_coh + llm_cred) / 3.0
        res.track2_score = t2_score
        res.final_score = combine_scores(res.track1_score, t2_score, res.is_honeypot)
        # Re-apply cap after blending
        res.final_score = min(res.final_score, results[i][0].track1_score / res.track1_score * res.final_score
                              if res.track1_score > 0 else res.final_score)
        # Only use reasoning from LLM if it's real (not a mock/offline fallback)
        if reasoning and len(reasoning) > 20 and not is_mock:
            res.reasoning = reasoning

    # 8. Clustering + Redirects + Reasoning
    logger.info("Step 8: Archetype clustering, redirects, and per-candidate reasoning...")
    candidate_results = [r[0] for r in results]
    candidate_map = {c.get("candidate_id", ""): c for _, c in results}

    fit_archetypes(candidate_results)

    for res in candidate_results:
        detect_redirect(res)
        if not res.reasoning:
            raw_c = candidate_map.get(res.candidate_id)
            res.reasoning = generate_reasoning_string(
                candidate_id=res.candidate_id,
                axis_scores=res.axis_scores,
                matched_skills=res.top_matching_skills,
                is_honeypot=res.is_honeypot,
                honeypot_details=res.honeypot_details,
                disqualifier_tags=disqualifier_map.get(res.candidate_id, []),
                candidate=raw_c,   # ← passes full profile for fact-based reasoning
            )

    # 9. Sort by final_score (disqualifier multiplier already applied) and take top 100
    logger.info("Step 9: Sorting all candidates by final score and taking top 100...")

    for res in candidate_results:
        res.final_score = round(res.final_score, 4)
    candidate_results.sort(key=lambda x: (-x.final_score, x.candidate_id))
    top_candidates = candidate_results[:100]

    write_submission_csv(top_candidates, output_path)
    logger.info("Done! Output: %s", output_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gotcha End-to-End Pipeline (Redrob JD)")
    parser.add_argument("--jd", type=str, default=str(JOB_DESCRIPTION_DOCX), help="Path to JD docx (unused — JD is hardcoded)")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidates json/jsonl")
    parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "team_infinity_and_beyond.csv"), help="Path to output csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates processed")
    parser.add_argument("--sample-only", action="store_true", help="Process only sample candidates JSON")

    args = parser.parse_args()

    if args.sample_only:
        cands_file = Path(args.candidates or SAMPLE_CANDIDATES_JSON)
        is_jsonl = False
    else:
        cands_file = Path(args.candidates or CANDIDATES_JSONL)
        if not cands_file.exists():
            cands_file = Path(SAMPLE_CANDIDATES_JSON)
            is_jsonl = False
            logger.info("candidates.jsonl not found. Falling back to sample_candidates.json")
        else:
            is_jsonl = cands_file.suffix.lower() == ".jsonl"

    out_file = Path(args.output)

    run_pipeline(
        jd_path=Path(args.jd),
        candidates_path=cands_file,
        output_path=out_file,
        is_jsonl=is_jsonl,
        limit=args.limit,
    )
