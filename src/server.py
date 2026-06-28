"""
Gotcha — FastAPI Backend Server
Exposes candidate profiles, rankings, job description parsing, and pipeline orchestration
via a REST API to support the Next.js frontend.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CANDIDATES_JSONL,
    SAMPLE_CANDIDATES_JSON,
    JOB_DESCRIPTION_DOCX,
    OUTPUTS_DIR,
    TIER_SCORES,
)
from src.ingestion.loader import load_job_description
from src.llm.extractor import parse_jd
from src.ingestion.normalizer import build_synonym_lookup, normalize_candidate, compute_data_confidence
from src.scoring.skill_matcher import match_skills, extract_jd_skills
from src.scoring.trust_scorer import score_candidate_trust, score_skill_trust
from src.scoring.career_relevance import score_career_relevance
from src.scoring.behavioral import score_behavioral
from src.scoring.feasibility import score_feasibility
from src.scoring.pedigree import score_pedigree
from src.scoring.honeypot import detect_honeypot
from src.clustering.archetypes import fit_archetypes
from src.redirect.detector import detect_redirect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="Gotcha API Backend", version="1.0.0")

# Enable CORS for Next.js dev server on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# --- Active Job Description State & Caching ---
ACTIVE_JD_PATH = str(JOB_DESCRIPTION_DOCX)
ACTIVE_JD_TEXT = None
_cached_jd_profile = None

def get_active_jd_profile() -> Any:
    """Retrieve and parse the currently active job description, using caching."""
    global _cached_jd_profile
    if _cached_jd_profile is not None:
        return _cached_jd_profile

    try:
        custom_txt_path = PROJECT_ROOT / "data" / "active_jd.txt"
        if ACTIVE_JD_TEXT is not None:
            _cached_jd_profile = parse_jd(ACTIVE_JD_TEXT)
        elif custom_txt_path.exists():
            text = load_job_description(str(custom_txt_path))
            _cached_jd_profile = parse_jd(text)
        else:
            text = load_job_description(ACTIVE_JD_PATH)
            _cached_jd_profile = parse_jd(text)
    except Exception as e:
        logger.error(f"Failed to load/parse active JD: {e}")
        from src.config import JDProfile
        _cached_jd_profile = JDProfile()

    return _cached_jd_profile

# --- Helper functions ---

def load_candidate_profiles(cids: List[str]) -> Dict[str, Dict]:
    """Retrieve full profile data for a list of candidate IDs.
    Duplicate of the UI helper, but adapted cleanly for FastAPI.
    """
    target_ids = set(cids)
    profiles = {}

    # Try candidates.jsonl first
    if CANDIDATES_JSONL.exists():
        try:
            synonym_lookup = build_synonym_lookup()
            with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if any(cid in line for cid in target_ids):
                        try:
                            c = json.loads(line)
                            actual_cid = c.get("candidate_id")
                            if actual_cid in target_ids:
                                normalize_candidate(c, synonym_lookup)
                                profiles[actual_cid] = c
                                target_ids.remove(actual_cid)
                        except Exception:
                            pass
                    if not target_ids:
                        break
        except Exception as e:
            logger.error(f"Error loading candidates.jsonl: {e}")

    # Try sample_candidates.json for any remaining IDs
    if target_ids and SAMPLE_CANDIDATES_JSON.exists():
        try:
            synonym_lookup = build_synonym_lookup()
            with open(SAMPLE_CANDIDATES_JSON, "r", encoding="utf-8") as f:
                raw_cands = json.load(f)
            for c in raw_cands:
                cid = c.get("candidate_id")
                if cid in target_ids:
                    normalize_candidate(c, synonym_lookup)
                    profiles[cid] = c
                    target_ids.remove(cid)
                    if not target_ids:
                        break
        except Exception as e:
            logger.error(f"Error loading sample_candidates.json: {e}")

    return profiles

def get_base_candidates() -> List[Dict]:
    """Return candidates from output files or fallback to sample dataset."""
    csv_path = OUTPUTS_DIR / "submission.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "gotcha.csv"
        
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            ranked_cids = list(df["candidate_id"].unique())
            cand_map = load_candidate_profiles(ranked_cids)
            
            merged_list = []
            for _, row in df.iterrows():
                cid = row["candidate_id"]
                raw_c = cand_map.get(cid, {})
                profile = raw_c.get("profile", {})
                raw_skills = raw_c.get("skills") or []
                skills = [s.get("name") for s in raw_skills if s and isinstance(s, dict)]
                
                is_hp, hp_details = detect_honeypot(raw_c)
                
                merged_list.append({
                    "rank": int(row["rank"]),
                    "candidate_id": cid,
                    "name": profile.get("anonymized_name", "Anonymized Candidate"),
                    "title": profile.get("current_title", "Software Engineer"),
                    "company": profile.get("current_company", "N/A"),
                    "location": profile.get("location", "India"),
                    "years_exp": profile.get("years_of_experience", 0),
                    "final_score": float(row["score"]),
                    "is_honeypot": is_hp,
                    "top_skills": skills[:5],
                    "reasoning": row["reasoning"],
                })
            return merged_list
        except Exception as e:
            logger.error(f"Failed to load actual pipeline output: {e}")
 
    # Fallback: Load and return sample candidates
    if SAMPLE_CANDIDATES_JSON.exists():
        try:
            with open(SAMPLE_CANDIDATES_JSON, "r", encoding="utf-8") as f:
                raw_cands = json.load(f)
            synonym_lookup = build_synonym_lookup()
            
            merged_list = []
            for i, c in enumerate(raw_cands):
                normalize_candidate(c, synonym_lookup)
                profile = c.get("profile", {})
                raw_skills = c.get("skills") or []
                skills = [s.get("name") for s in raw_skills if s and isinstance(s, dict)]
                is_hp, hp_details = detect_honeypot(c)
                
                merged_list.append({
                    "rank": i + 1,
                    "candidate_id": c.get("candidate_id"),
                    "name": profile.get("anonymized_name", "Anonymized Candidate"),
                    "title": profile.get("current_title", "Software Engineer"),
                    "company": profile.get("current_company", "N/A"),
                    "location": profile.get("location", "India"),
                    "years_exp": profile.get("years_of_experience", 0),
                    "final_score": 0.5 + (i * 0.005) % 0.45,  # Dummy score for fallback
                    "is_honeypot": is_hp,
                    "top_skills": skills[:5],
                    "reasoning": "Fallback profile from sample dataset.",
                })
            # Sort by score desc
            merged_list.sort(key=lambda x: x["final_score"], reverse=True)
            for idx, c in enumerate(merged_list):
                c["rank"] = idx + 1
            return merged_list
        except Exception as e:
            logger.error(f"Failed to load sample candidates: {e}")
            
    return []

# --- API Models ---

class JDInput(BaseModel):
    jd_text: str

class PipelineRunConfig(BaseModel):
    limit: Optional[int] = 1000
    jd_path: Optional[str] = None
    candidates_path: Optional[str] = None
    output_path: Optional[str] = None

# --- API Endpoints ---

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    candidates = get_base_candidates()
    if not candidates:
        return {
            "total_candidates": 0,
            "avg_score": 0,
            "honeypots": 0,
            "redirects": 0,
            "score_distribution": [],
            "archetypes": []
        }
        
    total_cands = len(candidates)
    avg_score = sum(c["final_score"] for c in candidates) / total_cands
    honeypots = sum(1 for c in candidates if c.get("is_honeypot", False))
    
    # Load full profiles to check redirects
    cids = [c["candidate_id"] for c in candidates]
    cand_map = load_candidate_profiles(cids)
    
    redirects = 0
    archetype_counts = {}
    
    # Load and parse JD Profile once outside the loop
    jd_profile = get_active_jd_profile()

    for c in candidates:
        cid = c["candidate_id"]
        raw_c = cand_map.get(cid, {})
        
        # Determine redirect
        suggested_role = None
        if raw_c and jd_profile:
            try:
                # Fit archetypes & detect redirect
                from src.redirect.detector import detect_redirect
                from src.config import CandidateResult
                
                res = CandidateResult(
                    candidate_id=cid,
                    track1_score=c["final_score"],
                    final_score=c["final_score"]
                )
                detect_redirect(res)
                if res.redirect_suggestion:
                    redirects += 1
                    suggested_role = res.redirect_suggestion
            except Exception:
                pass
                
        role = suggested_role or "Direct JD Match"
        if "Redirect to:" in role:
            role = role.replace("Redirect to: ", "")
        archetype_counts[role] = archetype_counts.get(role, 0) + 1
        
    # Generate score distribution bins
    score_bins = [0] * 10
    for c in candidates:
        idx = min(int(c["final_score"] * 10), 9)
        score_bins[idx] += 1
        
    score_dist = []
    for idx, count in enumerate(score_bins):
        score_dist.append({
            "range": f"{idx*10}-{(idx+1)*10}%",
            "count": count
        })
        
    archetypes = [{"archetype": k, "count": v} for k, v in archetype_counts.items()]
    # Sort archetypes by count desc
    archetypes.sort(key=lambda x: x["count"], reverse=True)
    
    # Mock PCA points for scatter plot (50 points)
    pca_data = []
    import random
    random.seed(42)
    archs = ["Backend", "Frontend", "DevOps", "Data Eng", "ML/AI", "Full Stack", "SRE", "Platform"]
    cx = {"Backend": -2, "Frontend": 2, "DevOps": -1, "Data Eng": 1, "ML/AI": 3, "Full Stack": 0, "SRE": -3, "Platform": -0.5}
    cy = {"Backend": 1, "Frontend": 1, "DevOps": -2, "Data Eng": 2, "ML/AI": -1, "Full Stack": 0, "SRE": -1, "Platform": 2}
    
    for i in range(min(total_cands, 100)):
        arch = random.choice(archs)
        pca_data.append({
            "candidate_id": candidates[i]["candidate_id"],
            "x": cx[arch] + random.gauss(0, 0.6),
            "y": cy[arch] + random.gauss(0, 0.6),
            "archetype": arch,
            "score": candidates[i]["final_score"]
        })
        
    return {
        "total_candidates": total_cands,
        "avg_score": round(avg_score, 4),
        "honeypots": honeypots,
        "redirects": redirects,
        "score_distribution": score_dist,
        "archetypes": archetypes,
        "pca_data": pca_data
    }

@app.post("/api/jd/parse")
def parse_job_description_endpoint(data: JDInput):
    if not data.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text is empty.")
    try:
        global ACTIVE_JD_TEXT, _cached_jd_profile
        parsed = parse_jd(data.jd_text)
        ACTIVE_JD_TEXT = data.jd_text
        _cached_jd_profile = parsed
        
        # Save it to data/active_jd.txt so the rank.py pipeline subprocess can read it!
        try:
            custom_txt_path = PROJECT_ROOT / "data" / "active_jd.txt"
            custom_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(custom_txt_path, "w", encoding="utf-8") as f:
                f.write(data.jd_text)
        except Exception as e:
            logger.error(f"Failed to save active_jd.txt: {e}")

        return {
            "title": parsed.title,
            "seniority_expected": parsed.seniority_expected,
            "must_have_skills": parsed.must_have_skills,
            "nice_to_have_skills": parsed.nice_to_have_skills,
            "axis_weights": parsed.axis_weights,
            "logistics": parsed.logistics,
            "all_keywords": parsed.all_keywords
        }
    except Exception as e:
        logger.error(f"Failed to parse JD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rankings")
def get_rankings(
    min_experience: int = Query(0),
    max_experience: int = Query(20),
    min_score: float = Query(0.0),
    exclude_honeypots: bool = Query(False)
):
    candidates = get_base_candidates()
    filtered = []
    for c in candidates:
        if not (min_experience <= c.get("years_exp", 0) <= max_experience):
            continue
        if c["final_score"] < min_score:
            continue
        if exclude_honeypots and c.get("is_honeypot"):
            continue
        filtered.append(c)
        
    return filtered

@app.get("/api/candidates/{candidate_id}")
def get_candidate_details(candidate_id: str):
    cand_map = load_candidate_profiles([candidate_id])
    raw_c = cand_map.get(candidate_id)
    if not raw_c:
        raise HTTPException(status_code=404, detail="Candidate profile not found.")
        
    try:
        synonym_lookup = build_synonym_lookup()
        
        # Load JD Profile
        jd_profile = get_active_jd_profile()
        
        # Extract skills
        jd_skills = [s.get("skill") for s in jd_profile.must_have_skills if s.get("skill")]
        if not jd_skills:
            jd_skills = extract_jd_skills(jd_profile.raw_text, synonym_lookup)
            
        skill_relevance, matched_list = match_skills(raw_c.get("skills", []), jd_skills, synonym_lookup)
        skill_trust = score_candidate_trust(raw_c, jd_skills, synonym_lookup)
        career_rel = score_career_relevance(raw_c, jd_profile.raw_text)
        behavioral = score_behavioral(raw_c.get("redrob_signals", {}))
        feasibility = score_feasibility(raw_c.get("redrob_signals", {}), jd_profile)
        data_confidence = compute_data_confidence(raw_c)
        pedigree = score_pedigree(raw_c)
        is_hp, hp_details = detect_honeypot(raw_c)
        
        # Determine redirects
        from src.config import CandidateResult
        res = CandidateResult(
            candidate_id=candidate_id,
            track1_score=skill_relevance,
            final_score=skill_relevance
        )
        detect_redirect(res)
        
        axis_scores = {
            "skill_relevance": round(skill_relevance, 4),
            "experience_impact": round(skill_trust, 4),
            "domain_coherence": round(career_rel, 4),
            "narrative_credibility": round(data_confidence, 4),
            "behavioral_validation": round(behavioral, 4),
            "engagement_feasibility": round(feasibility, 4),
            "pedigree": round(pedigree, 4),
        }
        
        # Populate timeline jobs
        timeline_jobs = []
        raw_career = raw_c.get("career_history")
        if isinstance(raw_career, list):
            for job in raw_career:
                if not job or not isinstance(job, dict):
                    continue
                timeline_jobs.append({
                    "title": job.get("title", "Job Title"),
                    "company": job.get("company", "Company"),
                    "start": job.get("start_date", "N/A"),
                    "end": "Present" if job.get("is_current") else job.get("end_date", "N/A"),
                    "duration_months": job.get("duration_months", 0),
                    "domain": job.get("industry", "N/A"),
                    "mismatch": False,
                })
                
        # Skill trust lists
        skill_trusts = []
        raw_skills = raw_c.get("skills")
        if isinstance(raw_skills, list):
            signals = raw_c.get("redrob_signals") or {}
            assessment_scores = signals.get("skill_assessment_scores") or {}
            for skill in raw_skills:
                if not skill or not isinstance(skill, dict):
                    continue
                st_obj = score_skill_trust(skill, assessment_scores, synonym_lookup)
                skill_trusts.append({
                    "skill_name": st_obj.skill_name,
                    "proficiency": st_obj.proficiency,
                    "trust_score": round(st_obj.trust_score, 4),
                    "duration_months": st_obj.duration_months,
                    "endorsements": st_obj.endorsements,
                    "has_assessment": st_obj.assessment_score is not None
                })
                
        profile = raw_c.get("profile") or {}
        
        # Find score reasoning
        final_score = 0.0
        reasoning = ""
        csv_path = OUTPUTS_DIR / "gotcha.csv"
        if not csv_path.exists():
            csv_path = OUTPUTS_DIR / "submission.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df_csv = pd.read_csv(csv_path)
                match_row = df_csv[df_csv["candidate_id"] == candidate_id]
                if not match_row.empty:
                    final_score = float(match_row.iloc[0]["score"])
                    reasoning = str(match_row.iloc[0]["reasoning"])
            except Exception:
                pass

        return {
            "candidate_id": candidate_id,
            "name": profile.get("anonymized_name", "Anonymized Candidate"),
            "title": profile.get("current_title", "Software Engineer"),
            "company": profile.get("current_company", "N/A"),
            "location": profile.get("location", "India"),
            "years_exp": profile.get("years_of_experience", 0),
            "final_score": final_score or skill_relevance,
            "is_honeypot": is_hp,
            "honeypot_details": hp_details,
            "redirect_suggestion": res.redirect_suggestion,
            "redirect_reason": res.redirect_reason,
            "axis_scores": axis_scores,
            "timeline_jobs": timeline_jobs,
            "skill_trusts": skill_trusts,
            "reasoning": reasoning,
            "data_confidence": round(data_confidence, 4),
            "signals": {
                "profile_completeness": signals.get("profile_completeness_score"),
                "verified_email": signals.get("verified_email", False),
                "verified_phone": signals.get("verified_phone", False),
                "linkedin_connected": signals.get("linkedin_connected", False),
            }
        }
    except Exception as e:
        logger.error(f"Error computing candidate deep-dive for {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/redirects")
def get_redirects(suggested_role: str = Query("All")):
    candidates = get_base_candidates()
    cids = [c["candidate_id"] for c in candidates]
    cand_map = load_candidate_profiles(cids)
    
    redirects = []
    
    for c in candidates:
        cid = c["candidate_id"]
        raw_c = cand_map.get(cid, {})
        if not raw_c:
            continue
            
        try:
            # Detect redirects
            from src.config import CandidateResult
            res = CandidateResult(
                candidate_id=cid,
                track1_score=c["final_score"],
                final_score=c["final_score"]
            )
            detect_redirect(res)
            
            if res.redirect_suggestion:
                role = res.redirect_suggestion
                if "Redirect to:" in role:
                    role = role.replace("Redirect to: ", "")
                    
                if suggested_role != "All" and role != suggested_role:
                    continue
                    
                redirects.append({
                    "candidate_id": cid,
                    "name": c["name"],
                    "current_title": c["title"],
                    "current_score": c["final_score"],
                    "suggested_role": role,
                    "archetype_score": round(c["final_score"] + 0.12, 4),  # Archetype fit proxy
                    "reason": res.redirect_reason or "Strong signals suggest fit for this role."
                })
        except Exception:
            pass
            
    return redirects

@app.post("/api/pipeline/run")
def trigger_pipeline(config: PipelineRunConfig):
    custom_txt_path = PROJECT_ROOT / "data" / "active_jd.txt"
    default_jd = str(custom_txt_path) if custom_txt_path.exists() else str(JOB_DESCRIPTION_DOCX)
    jd = config.jd_path or default_jd
    
    global ACTIVE_JD_PATH, ACTIVE_JD_TEXT, _cached_jd_profile
    ACTIVE_JD_PATH = jd
    ACTIVE_JD_TEXT = None
    _cached_jd_profile = None

    candidates = config.candidates_path or str(CANDIDATES_JSONL)
    if not Path(candidates).exists():
        candidates = str(SAMPLE_CANDIDATES_JSON)
    output = config.output_path or str(OUTPUTS_DIR / "gotcha.csv")
    limit = config.limit or 1000
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "rank.py"),
        "--jd", jd,
        "--candidates", candidates,
        "--output", output,
        "--limit", str(limit)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT)
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "detail": f"Execution failed with return code {e.returncode}"
        }

@app.post("/api/pipeline/validate")
def trigger_validation(output_path: str = Query(None)):
    output = output_path or str(OUTPUTS_DIR / "submission.csv")
    if not Path(output).exists():
        output = str(OUTPUTS_DIR / "gotcha.csv")
        
    val_script = PROJECT_ROOT.parent / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "validate_submission.py"
    if not val_script.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Validator script not found at {val_script}"
        }
        
    cmd = [
        sys.executable,
        str(val_script),
        output
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
