"""
Gotcha — Job Description Page (Redrob Senior AI Engineer Edition)
Displays the exact JD for the India Runs by Redrob challenge:
Senior AI Engineer — Founding Team at Redrob AI.
Shows hard requirements, disqualifiers, scoring hierarchy, and behavioral signals.
"""

import sys
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
    badge,
    skill_chip,
)
from src.jd_redrob import (
    get_redrob_jd_profile,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    JD_AXIS_WEIGHTS,
    RAW_JD_TEXT,
)

# Streamlit config
st.set_page_config(page_title="Gotcha — Job Description", page_icon="📄", layout="wide")
load_css()
init_session_state()

page_header(
    title="Senior AI Engineer — Founding Team at Redrob AI",
    subtitle="India Runs by Redrob AI Challenge · Track 01: Data & AI",
    icon="📄"
)

# ─── Company + Role Banner ────────────────────────────────────────────────────
banner_html = """
<div style='display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start;'>
    <div style='flex:2; min-width:260px;'>
        <h3 style='margin:0 0 8px 0; color:#6C63FF;'>Redrob AI</h3>
        <p style='color:#A0AEC0; margin:0 0 8px 0;'>
            Series A · AI-native talent intelligence platform · Pune / Noida, India
        </p>
        <p style='color:#E2E8F0; font-size:0.97rem;'>
            Building an AI engineering org from scratch. This person will own the
            <b>intelligence layer</b> — ranking, retrieval, and matching systems —
            the core of the product.
        </p>
    </div>
    <div style='flex:1; min-width:180px; display:flex; flex-direction:column; gap:8px;'>
        <span style='background:rgba(108,99,255,0.18); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
            🏢 <b>Type:</b> Product Company (Series A)
        </span>
        <span style='background:rgba(108,99,255,0.18); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
            📍 <b>Location:</b> Pune / Noida (India)
        </span>
        <span style='background:rgba(108,99,255,0.18); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
            ⏳ <b>Experience:</b> 5–9 years
        </span>
        <span style='background:rgba(108,99,255,0.18); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
            🚀 <b>Seniority:</b> Senior · Founding Team
        </span>
        <span style='background:rgba(72,209,164,0.18); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
            📅 <b>Notice:</b> Sub-30 days preferred
        </span>
    </div>
</div>
"""
glass_card(banner_html, accent="purple")

divider()

col1, col2 = st.columns([1, 1])

# ─── LEFT: Requirements ───────────────────────────────────────────────────────
with col1:

    st.write("### ✅ Hard Requirements (Must Have ALL)")
    reqs_html = """
    <ol style='color:#A0AEC0; line-height:2.0; margin:0; padding-left:18px;'>
        <li>Production experience with <b style='color:#48D1A4;'>embeddings-based retrieval</b>
            (sentence-transformers, OpenAI embeddings, BGE, E5 — deployed to real users)</li>
        <li>Production experience with <b style='color:#48D1A4;'>vector databases / hybrid search</b>
            (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch)</li>
        <li><b style='color:#48D1A4;'>Strong Python</b> — code quality matters</li>
        <li>Designed <b style='color:#48D1A4;'>evaluation frameworks</b> for ranking
            (NDCG, MRR, MAP, A/B testing, offline-to-online correlation)</li>
        <li><b style='color:#48D1A4;'>5–9 years</b> total experience</li>
        <li>Shipped ≥1 end-to-end <b style='color:#48D1A4;'>ranking / search / recommendation</b>
            to real users at scale</li>
        <li><b style='color:#48D1A4;'>Product company</b> experience (not purely IT services)</li>
        <li>Located in <b style='color:#48D1A4;'>India</b> (Pune, Noida, Delhi NCR, Mumbai,
            Hyderabad, Bangalore) OR willing to relocate</li>
    </ol>
    """
    glass_card(reqs_html, accent="teal")

    st.write("### 🌟 Nice-to-Have (Score Boosters)")
    nih_items = [
        "LLM fine-tuning (LoRA, QLoRA, PEFT)",
        "Learning-to-rank models (XGBoost-based or neural LTR)",
        "HR-tech, recruiting tech, or marketplace background",
        "Distributed systems / large-scale inference optimization",
        "Open-source AI/ML contributions",
        "Active GitHub (github_activity_score > 50)",
    ]
    nih_html = "<ul style='color:#A0AEC0; line-height:1.9; margin:0; padding-left:18px;'>" + \
               "".join(f"<li>{item}</li>" for item in nih_items) + "</ul>"
    glass_card(nih_html, accent="amber")

    st.write("### 🔑 JD Skills Matched in Scoring")
    must_chips = " ".join([
        f"<span class='skill-chip skill-chip-matched'>{s['skill']} "
        f"<span style='font-size:0.72rem;opacity:0.7;'>w={s['weight']:.2f}</span></span>"
        for s in MUST_HAVE_SKILLS
    ])
    nih_chips = " ".join([
        f"<span class='skill-chip'>{s['skill']} "
        f"<span style='font-size:0.72rem;opacity:0.7;'>bonus</span></span>"
        for s in NICE_TO_HAVE_SKILLS
    ])
    st.markdown(f"**Must-have:** {must_chips}", unsafe_allow_html=True)
    st.markdown(f"**Nice-to-have:** {nih_chips}", unsafe_allow_html=True)


# ─── RIGHT: Disqualifiers + Scoring Hierarchy ─────────────────────────────────
with col2:

    st.write("### 🚫 Disqualifier Tags")
    dq_items = [
        {"Tag": "CV_SPECIALIST", "Description": "Title/Role is CV-only (computer vision engineer) — no NLP/retrieval/ranking in current role"},
        {"Tag": "NON_ML_TITLE", "Description": "Current title is non-ML/AI (engineering manager, data analyst, software dev not in AI)"},
        {"Tag": "CONSULTING_ONLY", "Description": "Entire career at IT services (TCS, Infosys, Wipro, Accenture…) — no product company ever"},
        {"Tag": "OUTSIDE_INDIA", "Description": "Based outside India and not willing to relocate (no visa sponsorship)"},
        {"Tag": "IMPOSSIBLE_TIMELINE", "Description": "Timeline contradiction — e.g. 8yr experience at a 3yr old company (honeypot)"},
        {"Tag": "EXPERT_ZERO_DURATION", "Description": "Expert in 5+ skills with 0 months usage each — buzzword padding"},
    ]
    st.dataframe(pd.DataFrame(dq_items), hide_index=True, use_container_width=True, column_config={
        "Tag": st.column_config.TextColumn("Tag", width=150),
        "Description": st.column_config.TextColumn("Description"),
    })

    st.write("### 🔻 Compound Penalties (Multiplier × raw score)")
    cp_items = [
        {"Rule": "JUNIOR_TITLE", "Penalty": "×0.60", "Description": 'Title contains "Junior", "Associate", "Trainee", "Fresher"'},
        {"Rule": "YOE_LOW_3", "Penalty": "×0.25", "Description": "Experience < 3 years"},
        {"Rule": "YOE_LOW_4", "Penalty": "×0.50", "Description": "Experience 3–3.99 years"},
        {"Rule": "YOE_LOW_5", "Penalty": "×0.75", "Description": "Experience 4–4.99 years"},
        {"Rule": "YOE_HIGH", "Penalty": "×0.80", "Description": "Experience > 12 years"},
        {"Rule": "VISION_ONLY", "Penalty": "×0.30", "Description": "Skills are CV-only (YOLO, OpenCV, CNN, GAN, ASR) with zero NLP/retrieval"},
    ]
    st.dataframe(pd.DataFrame(cp_items), hide_index=True, use_container_width=True, column_config={
        "Rule": st.column_config.TextColumn("Rule", width=130),
        "Penalty": st.column_config.TextColumn("Multiplier", width=90),
        "Description": st.column_config.TextColumn("Description"),
    })

    st.write("### ⚖️ Axis Weights (8-Axis Model)")
    cols = st.columns(2)
    weight_labels = {
        "skill_relevance":         "Skill Relevance (0.25)",
        "trust_signals":           "Trust Signals (0.20)",
        "career_relevance":        "Career Relevance / TF-IDF (0.18)",
        "behavioral_validation":   "Behavioral Validation (0.10)",
        "engagement_feasibility":  "Engagement Feasibility (0.10)",
        "yoe_fit":                 "YoE Fit — 5–9 yr ideal (0.10)",
        "data_confidence":         "Data Confidence (0.05)",
        "pedigree":                "Pedigree / Education (0.02)",
    }
    for idx, (key, val) in enumerate(JD_AXIS_WEIGHTS.items()):
        with cols[idx % 2]:
            st.slider(
                label=weight_labels.get(key, key),
                min_value=0.0,
                max_value=0.40,
                value=float(val),
                key=f"jd_weight_{key}",
                disabled=True,
                help="Weights tuned specifically for this Senior AI Engineer role."
            )

divider()

# ─── Behavioral Signals Section ──────────────────────────────────────────────
st.write("### 📡 Behavioral Signals from Platform Data")
beh_col1, beh_col2 = st.columns(2)

with beh_col1:
    boost_html = """
    <h4 style='color:#48D1A4; margin-bottom:10px;'>⬆️ Boost Signals</h4>
    <ul style='color:#A0AEC0; line-height:1.85; margin:0; padding-left:18px;'>
        <li>last_active_date <b>within 30 days</b></li>
        <li>open_to_work_flag = <b>true</b></li>
        <li>notice_period_days ≤ <b>30</b> (preferred)</li>
        <li>recruiter_response_rate ≥ <b>0.60</b></li>
        <li>github_activity_score ≥ <b>50</b></li>
        <li>verified_email <b>AND</b> verified_phone = true</li>
    </ul>
    """
    glass_card(boost_html, accent="teal")

with beh_col2:
    downweight_html = """
    <h4 style='color:#FC8181; margin-bottom:10px;'>⬇️ Down-weight Signals</h4>
    <ul style='color:#A0AEC0; line-height:1.85; margin:0; padding-left:18px;'>
        <li>last_active_date <b>more than 6 months ago</b></li>
        <li>recruiter_response_rate &lt; <b>0.20</b></li>
        <li>notice_period_days &gt; <b>90</b></li>
        <li>open_to_work_flag = <b>false</b> AND no recent applications</li>
        <li>interview_completion_rate &lt; <b>0.50</b></li>
    </ul>
    """
    glass_card(downweight_html, accent="coral")

divider()

# ─── Scoring Formula ──────────────────────────────────────────────────────────
st.write("### 🎯 Evaluation Formula")
formula_html = """
<div style='font-family: monospace; background:rgba(0,0,0,0.3); border-radius:10px; padding:18px; color:#E2E8F0; font-size:0.95rem;'>
    <b>Final Score</b> = raw_score × disqualifier_multiplier<br><br>
    <b>raw_score</b> (weighted sum, 0→1) =<br>
    &nbsp;&nbsp;0.25 × skill_relevance       (embedding, vector DB, search, eval skills)<br>
    + 0.20 × trust_signals         (github_activity, verified profiles, complete data)<br>
    + 0.18 × career_relevance      (TF-IDF career description → JD keyword similarity)<br>
    + 0.10 × behavioral            (recruiter response, open-to-work, interview rate)<br>
    + 0.10 × feasibility           (notice period, India location, relocation)<br>
    + 0.10 × yoe_fit               (YoE 5–9 = 1.0; 10–12 = 0.8; &lt;5 = 0.5; &gt;12 = 0.3)<br>
    + 0.05 × data_confidence       (info completeness — skip if sparse)<br>
    + 0.02 × pedigree              (education tier — intentionally near-zero)<br><br>
    <b>disqualifier_multiplier</b> (product of all triggered penalties) =<br>
    &nbsp;&nbsp;(no tag → 1.0)<br>
    &nbsp;&nbsp;× 0.60 if JUNIOR_TITLE<br>
    &nbsp;&nbsp;× 0.25 / 0.50 / 0.75 if YOE &lt;3 / &lt;4 / &lt;5<br>
    &nbsp;&nbsp;× 0.80 if YOE &gt;12<br>
    &nbsp;&nbsp;× 0.30 if VISION_ONLY<br>
    &nbsp;&nbsp;→ <b>0.0</b> if any hard disqualifier (CV_SPECIALIST, NON_ML_TITLE, CONSULTING_ONLY, etc.)<br><br>
    <span style='color:#48D1A4;'>No power stretch — score = raw × mult (linear)</span>
</div>
"""
glass_card(formula_html, accent="teal")

divider()

# ─── Traps to Avoid ──────────────────────────────────────────────────────────
st.write("### ⚠️ Evaluation Traps (Anti-patterns)")
traps = [
    ("CV Specialist Trap",   "CV/robotics/speech engineers without ANY NLP or retrieval experience → VISION_ONLY penalty (×0.30). Still scorable but heavily penalized."),
    ("Junior Title Trap",    "Titles like 'Junior ML Engineer' or 'Associate Data Scientist' → ×0.60 multiplier regardless of actual ability."),
    ("YoE Bands",            "5–9 years is ideal (score 1.0 for fit). Under 3 years → ×0.25; 3–4 yr → ×0.50; 4–5 yr → ×0.75; Over 12 yr → ×0.80."),
    ("Consulting-Only",      "Entire career at TCS/Wipro/Accenture without a single product company → hard disqualifier (score zero)."),
    ("Outside India + Won't Relocate", "Based abroad and unwilling to move → hard disqualifier (score zero)."),
]
for trap_name, trap_desc in traps:
    st.markdown(
        f"<div style='padding:8px 14px; margin-bottom:6px; border-left:3px solid #F6AD55; background:rgba(246,173,85,0.07); border-radius:0 8px 8px 0;'>"
        f"<b style='color:#F6AD55;'>⚠️ {trap_name}:</b> "
        f"<span style='color:#A0AEC0; font-size:0.9rem;'>{trap_desc}</span></div>",
        unsafe_allow_html=True
    )
