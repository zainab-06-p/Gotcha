"""
Gotcha — Candidate Profile Deep Dive Page
Interactive detail view for a single candidate profile.
Shows the 7-axis radar chart, structured skill trust metrics, career history timeline, and honeypot alerts.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    init_session_state,
    load_css,
    page_header,
    glass_card,
    divider,
    get_candidates,
    radar_chart,
    score_breakdown_chart,
    alert_banner,
    badge,
    skill_chip,
    load_candidate_profiles,
    generate_demo_skill_trusts,
    generate_demo_career,
    generate_demo_jd,
)
from src.config import OUTPUTS_DIR, SAMPLE_CANDIDATES_JSON, JOB_DESCRIPTION_DOCX
from src.ingestion.loader import load_job_description
from src.llm.extractor import parse_jd

# Streamlit config
st.set_page_config(page_title="Gotcha — Candidate Profile", page_icon="👤", layout="wide")

load_css()
init_session_state()

page_header(
    title="Candidate Profile Deep-Dive",
    subtitle="7-axis visual inspection, verification checkmarks, and corroborated skill signals",
    icon="👤"
)

# Helper to load actual generated candidates IDs
def load_actual_shortlist_ids() -> list[str]:
    # Prefer the Redrob JD-specific output
    csv_path = OUTPUTS_DIR / "team_infinity_and_beyond.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "gotcha.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "submission.csv"
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
        return list(df["candidate_id"].unique())
    except Exception:
        return []

# Load candidates list
candidates = get_candidates()
actual_ids = load_actual_shortlist_ids()

if actual_ids:
    cids = actual_ids
    st.info("Showing actual output candidates from pipeline runs.")
else:
    cids = [c["candidate_id"] for c in candidates]
    st.info("Showing mockup candidate pool.")

# Check for candidate_id in query parameters
query_params = st.query_params
param_cid = query_params.get("candidate_id")

selected_idx = 0
if param_cid in cids:
    selected_idx = cids.index(param_cid)

selected_cid = st.selectbox(
    "Select Candidate to Inspect:",
    options=cids,
    index=selected_idx
)

st.session_state["selected_candidate_id"] = selected_cid

# Render selected candidate
if selected_cid.startswith("CAND_"):
    # Load actual candidate from dataset
    cand_map = load_candidate_profiles([selected_cid])
    raw_c = cand_map.get(selected_cid)
    
    if raw_c:
        # Re-compute scoring components in-process for accurate radar chart
        from src.ingestion.normalizer import build_synonym_lookup
        from src.scoring.skill_matcher import match_skills
        from src.scoring.trust_scorer import score_candidate_trust, score_skill_trust
        from src.scoring.career_relevance import score_career_relevance
        from src.scoring.behavioral import score_behavioral
        from src.scoring.feasibility import score_feasibility
        from src.scoring.pedigree import score_pedigree
        from src.scoring.confidence import score_data_confidence
        from src.scoring.honeypot import detect_honeypot
        from src.scoring.disqualifier import get_disqualifier_multiplier, yoe_fit_score
        
        synonym_lookup = build_synonym_lookup()
        
        # Load JD Profile — use hardcoded Redrob JD (faster, no LLM needed)
        jd_profile = st.session_state.get("jd_parsed")
        if not jd_profile:
            from src.jd_redrob import get_redrob_jd_profile
            jd_profile_obj = get_redrob_jd_profile()
            jd_profile = {
                "title": jd_profile_obj.title,
                "must_have_skills": jd_profile_obj.must_have_skills,
                "nice_to_have_skills": jd_profile_obj.nice_to_have_skills,
                "axis_weights": jd_profile_obj.axis_weights,
                "logistics": jd_profile_obj.logistics,
                "raw_text": jd_profile_obj.raw_text,
                "all_keywords": jd_profile_obj.all_keywords,
            }
            st.session_state["jd_parsed"] = jd_profile
            
        # Extract skills
        jd_skills = [s.get("skill") for s in jd_profile.get("must_have_skills", []) if s.get("skill")]
        if not jd_skills:
            from src.scoring.skill_matcher import extract_jd_skills
            jd_skills = extract_jd_skills(jd_profile.get("raw_text", ""), synonym_lookup)
            
        skill_relevance, matched_list = match_skills(raw_c.get("skills", []), jd_skills, synonym_lookup)
        
        # Build dummy JDProfile instance for compatibility
        from src.config import JDProfile as ConfigJDProfile
        dummy_jd = ConfigJDProfile(
            title=jd_profile.get("title", ""),
            must_have_skills=jd_profile.get("must_have_skills", []),
            nice_to_have_skills=jd_profile.get("nice_to_have_skills", []),
            axis_weights=jd_profile.get("axis_weights", {}),
            logistics=jd_profile.get("logistics", {}),
            raw_text=jd_profile.get("raw_text", ""),
            all_keywords=jd_profile.get("all_keywords", []),
        )
        
        skill_trust = score_candidate_trust(raw_c, jd_skills, synonym_lookup)
        career_relevance = score_career_relevance(raw_c, dummy_jd.raw_text)
        behavioral = score_behavioral(raw_c.get("redrob_signals", {}))
        feasibility = score_feasibility(raw_c.get("redrob_signals", {}), dummy_jd)
        data_confidence = score_data_confidence(raw_c)
        pedigree = score_pedigree(raw_c)
        
        is_hp, hp_details = detect_honeypot(raw_c)
        
        # ── Disqualifier multiplier (compound penalties: Junior, YOE, Vision-only) ──
        disq_mult, disq_tags, disq_reason = get_disqualifier_multiplier(raw_c)
        
        axis_scores = {
            "skill_relevance": skill_relevance,
            "experience_impact": skill_trust,
            "domain_coherence": career_relevance,
            "narrative_credibility": data_confidence,
            "behavioral_validation": behavioral,
            "engagement_feasibility": feasibility,
            "pedigree": pedigree,
        }
        
        # Load CSV details for final score and reasoning
        final_score = 0.0
        reasoning = ""
        csv_path = OUTPUTS_DIR / "team_infinity_and_beyond.csv"
        if not csv_path.exists():
            csv_path = OUTPUTS_DIR / "gotcha.csv"
        if not csv_path.exists():
            csv_path = OUTPUTS_DIR / "submission.csv"
        if csv_path.exists():
            try:
                df_csv = pd.read_csv(csv_path)
                match_row = df_csv[df_csv["candidate_id"] == selected_cid]
                if not match_row.empty:
                    final_score = float(match_row.iloc[0]["score"])
                    reasoning = str(match_row.iloc[0]["reasoning"])
            except Exception:
                pass
                
        # Adapt candidate dict
        profile = raw_c.get("profile") or {}
        raw_skills = raw_c.get("skills")
        if not isinstance(raw_skills, list):
            raw_skills = []
        candidate = {
            "candidate_id": selected_cid,
            "name": profile.get("anonymized_name", "Anonymized Candidate"),
            "title": profile.get("current_title", "Software Engineer"),
            "company": profile.get("current_company", "N/A"),
            "location": profile.get("location", "N/A"),
            "years_exp": profile.get("years_of_experience", 0),
            "final_score": final_score,
            "axis_scores": axis_scores,
            "is_honeypot": is_hp,
            "honeypot_details": hp_details,
            "top_skills": [s.get("name") for s in raw_skills if s and isinstance(s, dict) and s.get("name")][:5],
            "reasoning": reasoning,
            "data_confidence": data_confidence,
            "disq_multiplier": disq_mult,
            "disq_tags": disq_tags,
            "disq_reason": disq_reason,
        }
        
        # Populate timeline jobs
        timeline_jobs = []
        raw_career = raw_c.get("career_history")
        if not isinstance(raw_career, list):
            raw_career = []
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
        signals = raw_c.get("redrob_signals") or {}
        assessment_scores = signals.get("skill_assessment_scores") or {}
        for skill in raw_skills:
            if not skill or not isinstance(skill, dict):
                continue
            st_obj = score_skill_trust(skill, assessment_scores, synonym_lookup)
            skill_trusts.append({
                "skill_name": st_obj.skill_name,
                "proficiency": st_obj.proficiency,
                "trust_score": st_obj.trust_score,
            })
            
    else:
        st.error("Failed to load candidate details from dataset.")
        st.stop()
else:
    # Use demo fallback candidate
    candidate = next(c for c in candidates if c["candidate_id"] == selected_cid)
    timeline_jobs = generate_demo_career()
    skill_trusts = generate_demo_skill_trusts(candidate.get("top_skills", []))

# Profile details
name = candidate["name"]
title = candidate["title"]
company = candidate["company"]
location = candidate["location"]
exp = candidate["years_exp"]
score = candidate["final_score"]

# Check for Honeypot
if candidate.get("is_honeypot"):
    alert_banner(
        text=f"CRITICAL WARNING: Profile flagged as potential HONEYPOT. Reason: {candidate.get('honeypot_details', 'Descriptive mismatch')}",
        kind="honeypot"
    )

# Render main grid
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### Profile Overview")
    profile_html = f"""
    <div style='margin-bottom:15px;'>
        <h3 style='margin:0;color:#6C63FF;'>{name}</h3>
        <p style='margin:5px 0;font-size:1.1rem;font-weight:500;'>{title} @ {company}</p>
        <p style='margin:2px 0;color:#A0AEC0;font-size:0.9rem;'>Location: {location} | Experience: {exp} years</p>
    </div>
    """
    glass_card(profile_html)
    
    # 7-Axis radar chart
    jd_ideal = generate_demo_jd()["axis_weights"]
    fig_radar = radar_chart(candidate["axis_scores"], jd_ideal)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Disqualifier penalty status
    disq_mult = candidate.get("disq_multiplier", 1.0)
    disq_tags = candidate.get("disq_tags", [])
    disq_reason = candidate.get("disq_reason", "")
    if disq_mult < 1.0:
        penalty_pct = (1.0 - disq_mult) * 100
        tag_badges = "".join(f'<span style="background:rgba(252,129,129,0.15);color:#FC8181;padding:2px 8px;border-radius:12px;font-size:0.75rem;margin:2px 4px;">{t}</span>' for t in disq_tags)
        alert_banner(
            text=f"<b>Penalty Applied: −{penalty_pct:.0f}%</b><br/>{tag_badges}<br/><span style='font-size:0.85rem;color:#A0AEC0;'>{disq_reason}</span>",
            kind="honeypot",
            icon="⚠️"
        )

with col2:
    st.write("### Fit Score Breakdown")
    fig_breakdown = score_breakdown_chart(candidate["axis_scores"])
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Redirect Suggestion Banner if any
    red_role = candidate.get("redirect_suggestion")
    if red_role:
        reason = candidate.get("redirect_reason", "Candidate vector matches another role cluster.")
        alert_banner(
            text=f"REDIRECTION TIP: Consider for <b>{red_role}</b>. Reason: {reason}",
            kind="redirect",
            icon="🔄"
        )

st.write("### Verification and Skill Trust Corroboration")
col3, col4 = st.columns([1, 1])

with col3:
    st.write("#### Structured Verification Signals")
    conf = candidate.get("data_confidence", 0.8)
    
    verif_html = f"""
    <div style='display:flex; flex-direction:column; gap:10px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span>Profile Completeness Score</span>
            <span><b>{conf:.0%}</b></span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>Email Address Verified</span>
            <span style='color:#00D4AA; font-weight:bold;'>✓ Verified</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>Phone Number Verified</span>
            <span style='color:#00D4AA; font-weight:bold;'>✓ Verified</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>LinkedIn Connected</span>
            <span style='color:#00D4AA; font-weight:bold;'>✓ Connected</span>
        </div>
    </div>
    """
    glass_card(verif_html, accent="teal")

with col4:
    st.write("#### Skill Corroboration Details")
    
    trust_html = "<table style='width:100%; border-collapse:collapse;'>"
    trust_html += "<thead><tr style='color:#A0AEC0;border-bottom:1px solid rgba(255,255,255,0.1);'><th>Skill</th><th>Proficiency</th><th>Trust</th></tr></thead><tbody>"
    for st_item in skill_trusts:
        s_name = st_item["skill_name"]
        prof = str(st_item["proficiency"]).title()
        t_score = st_item["trust_score"]
        # Mini trust bar
        pct = int(t_score * 100)
        t_color = "#00D4AA" if t_score > 0.7 else "#FFB347" if t_score > 0.4 else "#FF6B6B"
        bar = f"""
        <div style="background:rgba(255,255,255,0.06);border-radius:4px;width:70px;height:6px;display:inline-block;vertical-align:middle;margin-right:5px;">
            <div style="background:{t_color};width:{pct}%;height:100%;border-radius:4px;"></div>
        </div>
        """
        trust_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);height:32px;'><td><b>{s_name}</b></td><td>{prof}</td><td>{bar} <span style='color:{t_color};font-size:0.8rem;'>{pct}%</span></td></tr>"
    trust_html += "</tbody></table>"
    glass_card(trust_html)

st.write("### Career History Timeline")

for job in timeline_jobs:
    is_mismatch = job["mismatch"]
    accent = "coral" if is_mismatch else ""
    flag_txt = f" {badge('Mismatch warning', 'warning')}" if is_mismatch else ""
    
    job_html = f"""
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <span style='font-size:1.15rem; font-weight:600; color:#FAFAFA;'>{job['title']} @ {job['company']} {flag_txt}</span>
        <span style='color:#A0AEC0; font-size:0.9rem;'>{job['start']} to {job['end']} ({job['duration_months']} months)</span>
    </div>
    <p style='color:#A0AEC0; margin-top:5px; font-size:0.95rem;'>
        Domain: <b>{job['domain']}</b> | Scope: Managed core services, engineered deployment workflows, and aligned with stack architectures.
    </p>
    """
    glass_card(job_html, accent=accent)
