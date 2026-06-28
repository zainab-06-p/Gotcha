"""
Gotcha — LLM Extraction and Parsing Module
Interfaces between the raw text data and LLM client to extract structured information.
"""

import json
import logging
import re
from typing import Any, Optional

from src.config import JDProfile, CandidateAxisScores
from src.llm.client import call_llm
from src.llm.prompts import (
    JD_PARSING_SYSTEM_PROMPT,
    JD_PARSING_USER_PROMPT,
    CANDIDATE_EVALUATION_SYSTEM_PROMPT,
    CANDIDATE_EVALUATION_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def clean_json_response(raw_text: str) -> str:
    """Clean markdown backticks or extra text from LLM response to get raw JSON."""
    cleaned = raw_text.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```"):
        # Match ```json ... ``` or just ``` ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            
    # Try finding the first '{' and last '}'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]
        
    return cleaned


def parse_jd(jd_text: str) -> JDProfile:
    """Parse a job description string into a structured JDProfile.
    
    Args:
        jd_text: Raw job description text.
        
    Returns:
        JDProfile dataclass.
    """
    if not jd_text:
        return JDProfile()
        
    prompt = JD_PARSING_USER_PROMPT.format(jd_text=jd_text)
    
    try:
        response_text = call_llm(
            prompt=prompt,
            system_instruction=JD_PARSING_SYSTEM_PROMPT,
            temperature=0.1
        )
        
        json_str = clean_json_response(response_text)
        data = json.loads(json_str)
        
        # Build JDProfile
        profile = JDProfile()
        profile.title = data.get("title", "Job Description")
        profile.must_have_skills = data.get("must_have_skills", [])
        profile.nice_to_have_skills = data.get("nice_to_have_skills", [])
        profile.seniority_expected = data.get("seniority_expected", "mid")
        profile.raw_text = jd_text
        
        # Extract logistics
        profile.logistics = {
            "work_mode": data.get("work_mode", "hybrid"),
            "min_experience_years": data.get("min_experience_years", 0),
        }
        
        # Collect keywords for TF-IDF
        keywords = [profile.title]
        for skill_info in profile.must_have_skills:
            keywords.append(skill_info.get("skill", ""))
        for skill_info in profile.nice_to_have_skills:
            keywords.append(skill_info.get("skill", ""))
        profile.all_keywords = list(set([k for k in keywords if k]))
        
        logger.info("Successfully parsed JD '%s' with %d must-have skills", 
                    profile.title, len(profile.must_have_skills))
        return profile
        
    except Exception as e:
        logger.error("Failed to parse JD with LLM: %s. Using heuristic fallback.", e)
        # Simple heuristic fallback
        from src.scoring.skill_matcher import extract_jd_skills
        fallback_skills = extract_jd_skills(jd_text)
        
        profile = JDProfile()
        profile.title = "Job Description"
        profile.must_have_skills = [{"skill": s, "weight": 1.0/max(len(fallback_skills), 1)} for s in fallback_skills]
        profile.raw_text = jd_text
        profile.all_keywords = fallback_skills
        return profile


def evaluate_candidate(candidate: dict, jd_profile: JDProfile) -> dict[str, Any]:
    """Evaluate a single candidate against a JD profile using LLM.
    
    Args:
        candidate: Normalized candidate dict.
        jd_profile: Parsed JDProfile.
        
    Returns:
        Dict with experience_impact, domain_coherence, narrative_credibility, and reasoning.
    """
    profile = candidate.get("profile", {})
    name = profile.get("anonymized_name", "Candidate")
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")
    yoe = profile.get("years_of_experience", 0)
    
    # Format career history
    career_list = []
    for job in candidate.get("career_history", []):
        career_list.append(
            f"- Job Title: {job.get('title')}\n"
            f"  Company: {job.get('company')} (Size: {job.get('company_size')})\n"
            f"  Duration: {job.get('duration_months')} months\n"
            f"  Description: {job.get('description')}\n"
        )
    career_str = "\n".join(career_list)
    
    # Format JD requirements
    must_have_skills = ", ".join([s.get("skill", "") for s in jd_profile.must_have_skills])
    
    prompt = CANDIDATE_EVALUATION_USER_PROMPT.format(
        candidate_name=name,
        candidate_summary=summary,
        candidate_headline=headline,
        candidate_yoe=yoe,
        candidate_career=career_str,
        jd_title=jd_profile.title,
        must_have_skills=must_have_skills,
        expected_seniority=jd_profile.seniority_expected,
        jd_context=jd_profile.raw_text[:1000] # Truncated to fit context
    )
    
    try:
        response_text = call_llm(
            prompt=prompt,
            system_instruction=CANDIDATE_EVALUATION_SYSTEM_PROMPT,
            temperature=0.2
        )
        
        json_str = clean_json_response(response_text)
        data = json.loads(json_str)
        
        # Validate values in range
        return {
            "experience_impact": max(0.0, min(float(data.get("experience_impact", 0.5)), 1.0)),
            "domain_coherence": max(0.0, min(float(data.get("domain_coherence", 0.5)), 1.0)),
            "narrative_credibility": max(0.0, min(float(data.get("narrative_credibility", 0.5)), 1.0)),
            # Only include reasoning if it's real (not the mock placeholder)
            "reasoning": str(data.get("reasoning", "")).strip(),
            "__is_mock__": False,
        }
    except Exception as e:
        logger.error("Failed to evaluate candidate %s with LLM: %s. Using default baseline.", 
                     candidate.get("candidate_id"), e)
        # Default fallback — mark as mock so explainer heuristic takes over
        return {
            "experience_impact": 0.5,
            "domain_coherence": 0.5,
            "narrative_credibility": 0.5,
            "reasoning": "",
            "__is_mock__": True,
        }
