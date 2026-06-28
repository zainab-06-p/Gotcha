"""
Gotcha — Main Landing Page
Premium welcome portal introducing Gotcha: Graph-Native Candidate Discovery & Ranking Engine.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_css, page_header, glass_card, divider, render_metric_row

# Streamlit page configuration
st.set_page_config(
    page_title="Gotcha — AI Candidate Discovery",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load styles
load_css()

# Header
page_header(
    title="Gotcha AI",
    subtitle="Graph-Native Candidate Discovery & Corroboration-Based Ranking Engine",
    icon="🎯"
)

# Hero Section
hero_content = """
<h2 style='margin-top:0;'>Welcome to the Future of Talent Discovery</h2>
<p style='font-size:1.1rem;color:#A0AEC0;'>
    Gotcha is an advanced, corroboration-weighted AI search and ranking engine designed for the 
    <b>Redrob AI Challenge</b>. Unlike traditional systems that read profiles at face value, 
    Gotcha builds a multi-relational graph per candidate and mathematically cross-references 
    self-reported skills against assessments, career descriptions, and behavioral signals.
</p>
"""
glass_card(hero_content, accent="teal")

# Key Metrics Overview
st.write("### Engine Status")
metrics = [
    {"icon": "👥", "value": "100,000+", "label": "Candidate Database", "delta": "", "accent": "purple"},
    {"icon": "⚡", "value": "Track 1 & 2", "label": "Decoupled Processing", "delta": "Active", "accent": "teal"},
    {"icon": "📊", "value": "7-Axis", "label": "Profile Modeling", "delta": "", "accent": "amber"},
    {"icon": "🎯", "value": "Top 100", "label": "Ranked Shortlist", "delta": "CSV ready", "accent": "coral"},
]
render_metric_row(metrics)

st.write("### System Architecture")

col1, col2 = st.columns([2, 1])

with col1:
    arch_explanation = """
    <h4>The Gotcha Pipeline</h4>
    <ol style='color:#A0AEC0; line-height: 1.6;'>
        <li><b>Fuzzy Synonym Ingestion:</b> Resolves 100+ raw skills to canonical names (e.g. <i>py3, python, Python3</i> -> <i>Python</i>).</li>
        <li><b>Deterministic Track 1 Scoring:</b> Evaluates 7 independent scoring axes (Skill Trust, TF-IDF career relevance, behavioral platforms, feasibility, and pedigree).</li>
        <li><b>Honeypot Detection:</b> Flags candidates with descriptive inconsistencies (e.g. title-to-description keyword overlap < 15%) or boilerplate summaries.</li>
        <li><b>Track 2 LLM Reranking:</b> Automatically triggers Gemini to evaluate the narrative credibility and domain coherence of the top 300 candidates.</li>
        <li><b>K-Means Archetype Discovery:</b> Clusters candidate vectors into 8 distinct roles for talent pool analytics.</li>
        <li><b>Mismatch Redirects:</b> Identifies high-quality candidates who mismatch the current job but are perfect fits for other internal positions.</li>
    </ol>
    """
    glass_card(arch_explanation)

with col2:
    navigation_card = """
    <h4>Navigate Engine Pages</h4>
    <div style='display:flex; flex-direction:column; gap:10px; margin-top:15px;'>
        <p style='color:#A0AEC0;'>Use the sidebar to explore features:</p>
        <span style='font-size:0.9rem;'>📊 <b>1_Dashboard:</b> Candidate pool analytics</span>
        <span style='font-size:0.9rem;'>📄 <b>2_Job_Description:</b> Parse JDs & extract requirements</span>
        <span style='font-size:0.9rem;'>🏆 <b>3_Rankings:</b> Inspect the final submission shortlist</span>
        <span style='font-size:0.9rem;'>👤 <b>4_Candidate_Profile:</b> Radar charts and trust breakdown</span>
        <span style='font-size:0.9rem;'>🔄 <b>5_Redirects:</b> Archetypes & redirect matches</span>
        <span style='font-size:0.9rem;'>⚙️ <b>6_Pipeline:</b> Configure, scale & run the pipeline</span>
    </div>
    """
    glass_card(navigation_card, accent="purple")
