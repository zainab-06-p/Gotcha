"""
Gotcha — Shared UI Helpers
Reusable chart builders, card renderers, formatters, and demo data generators.
"""

import sys
import random
from pathlib import Path
from typing import Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `from src.config import ...` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CandidateAxisScores,
    CandidateResult,
    JDProfile,
    SkillTrust,
    TRACK1_WEIGHTS,
    SUBMISSION_TOP_N,
)

# ============================================================================
# CSS Loader
# ============================================================================
def load_css():
    """Inject the custom stylesheet into the Streamlit app."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ============================================================================
# Layout Helpers
# ============================================================================
def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a gradient page header with optional subtitle."""
    load_css()
    header_html = f"""
    <div style="margin-bottom:1.5rem;">
        <div class="gradient-header">{icon} {title}</div>
        {"<p class='hero-tagline' style='text-align:left;margin:0.25rem 0 0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def metric_card(icon: str, value: str, label: str, delta: str = "", accent: str = ""):
    """Render a glassmorphism metric card."""
    delta_html = ""
    if delta:
        cls = "positive" if not delta.startswith("-") else "negative"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    accent_cls = f"card-accent-{accent}" if accent else ""
    return f"""
    <div class="metric-card {accent_cls}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def render_metric_row(metrics: list[dict]):
    """Render a row of metric cards. Each dict: icon, value, label, delta, accent."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                metric_card(
                    m.get("icon", "📊"),
                    str(m.get("value", "—")),
                    m.get("label", ""),
                    m.get("delta", ""),
                    m.get("accent", ""),
                ),
                unsafe_allow_html=True,
            )


def glass_card(content: str, accent: str = ""):
    """Wrap HTML content in a glassmorphism card."""
    accent_cls = f"card-accent-{accent}" if accent else ""
    st.markdown(
        f'<div class="glass-card {accent_cls}">{content}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info"):
    """Return HTML for a status badge. kind: success, warning, danger, info."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def skill_chip(name: str, matched: bool = False):
    """Return HTML for a skill chip."""
    cls = "skill-chip-matched" if matched else ""
    return f'<span class="skill-chip {cls}">{name}</span>'


def alert_banner(text: str, kind: str = "honeypot", icon: str = "⚠️"):
    """Render an alert banner."""
    st.markdown(
        f'<div class="alert-banner alert-{kind}">'
        f'<span style="font-size:1.5rem">{icon}</span>'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Plotly Theme
# ============================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(30,33,40,0.7)",
        bordercolor="rgba(108,99,255,0.18)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

AXIS_LABELS = {
    "skill_relevance": "Skill Relevance",
    "experience_impact": "Experience & Impact",
    "domain_coherence": "Domain Coherence",
    "narrative_credibility": "Narrative Credibility",
    "behavioral_validation": "Behavioral Validation",
    "engagement_feasibility": "Engagement Feasibility",
    "pedigree": "Pedigree",
}

ACCENT = "#6C63FF"
TEAL = "#00D4AA"
AMBER = "#FFB347"
CORAL = "#FF6B6B"


# ============================================================================
# Chart Builders
# ============================================================================
def radar_chart(
    candidate_scores: dict,
    jd_ideal: dict | None = None,
    title: str = "7-Axis Profile",
) -> go.Figure:
    """Create a radar chart comparing candidate vs JD ideal on 7 axes."""
    categories = list(AXIS_LABELS.values())
    keys = list(AXIS_LABELS.keys())

    cand_vals = [candidate_scores.get(k, 0) for k in keys]
    cand_vals.append(cand_vals[0])  # close the polygon
    cats = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cand_vals, theta=cats, fill="toself",
        name="Candidate",
        fillcolor="rgba(108,99,255,0.2)",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5),
    ))

    if jd_ideal:
        jd_vals = [jd_ideal.get(k, 0) for k in keys]
        jd_vals.append(jd_vals[0])
        fig.add_trace(go.Scatterpolar(
            r=jd_vals, theta=cats, fill="toself",
            name="JD Ideal",
            fillcolor="rgba(0,212,170,0.12)",
            line=dict(color=TEAL, width=2, dash="dash"),
            marker=dict(size=5),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=9, color="#A0AEC0"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10),
            ),
        ),
        showlegend=True,
        height=420,
    )
    return fig


def score_breakdown_chart(axis_scores: dict, weights: dict | None = None) -> go.Figure:
    """Horizontal bar chart showing per-axis score contributions."""
    labels = [AXIS_LABELS.get(k, k) for k in axis_scores]
    values = list(axis_scores.values())
    w = weights or {k: 1.0 / len(axis_scores) for k in axis_scores}
    contributions = [axis_scores[k] * w.get(k, 0) for k in axis_scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h",
        name="Raw Score",
        marker=dict(
            color=values,
            colorscale=[[0, CORAL], [0.5, AMBER], [1, TEAL]],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=[f"{v:.2f}" for v in values],
        textposition="auto",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Score Breakdown by Axis", font=dict(size=16)),
        xaxis=dict(
            range=[0, 1], title="Score",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(autorange="reversed"),
        height=360,
        showlegend=False,
    )
    return fig


def score_histogram(scores: list[float], title: str = "Score Distribution") -> go.Figure:
    """Histogram of candidate scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=30,
        marker=dict(
            color=ACCENT,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        opacity=0.85,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Final Score", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
        height=340,
        bargap=0.06,
    )
    return fig


def archetype_scatter(data: pd.DataFrame) -> go.Figure:
    """2D scatter of candidates colored by archetype cluster."""
    fig = px.scatter(
        data, x="pca_x", y="pca_y",
        color="archetype",
        hover_data=["candidate_id", "final_score"],
        title="Archetype Clusters (PCA Projection)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.2)")))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Component 1", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Component 2", gridcolor="rgba(255,255,255,0.05)"),
        height=480,
    )
    return fig


# ============================================================================
# Score Color Helpers
# ============================================================================
def score_color(score: float) -> str:
    """Return a CSS color string based on score."""
    if score >= 0.7:
        return TEAL
    elif score >= 0.4:
        return AMBER
    return CORAL


def score_badge(score: float) -> str:
    """Return an HTML badge for a score."""
    if score >= 0.7:
        return badge(f"{score:.2f}", "success")
    elif score >= 0.4:
        return badge(f"{score:.2f}", "warning")
    return badge(f"{score:.2f}", "danger")


def trust_bar(value: float, width: int = 100) -> str:
    """Mini progress bar for trust/score values."""
    color = score_color(value)
    pct = int(value * 100)
    return (
        f'<div style="background:rgba(255,255,255,0.06);border-radius:4px;'
        f'width:{width}px;height:8px;display:inline-block;vertical-align:middle;">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div>'
        f'</div> <span style="font-size:0.8rem;color:{color};">{pct}%</span>'
    )


# ============================================================================
# Demo Data Generators
# ============================================================================
_DEMO_NAMES = [
    "Aarav Sharma", "Priya Patel", "Rohan Gupta", "Ananya Singh",
    "Vikram Reddy", "Sneha Iyer", "Arjun Nair", "Meera Das",
    "Karthik Rao", "Divya Joshi", "Aditya Verma", "Pooja Mehta",
    "Siddharth Kumar", "Nisha Chauhan", "Ravi Thakur", "Kavya Bhat",
    "Nikhil Agarwal", "Swati Pillai", "Manish Saxena", "Ishita Kapoor",
    "Amit Banerjee", "Rekha Menon", "Suresh Dubey", "Tanya Malhotra",
    "Rajesh Sinha", "Anjali Mishra", "Deepak Yadav", "Pallavi Choudhury",
    "Harsh Pandey", "Ritika Shah",
]

_DEMO_SKILLS = [
    "Python", "React", "Node.js", "AWS", "Docker", "Kubernetes",
    "TypeScript", "PostgreSQL", "MongoDB", "Redis", "GraphQL",
    "Machine Learning", "TensorFlow", "Django", "FastAPI", "Go",
    "Java", "Spring Boot", "Kafka", "Elasticsearch", "CI/CD",
    "Terraform", "System Design", "Microservices", "REST APIs",
]

_DEMO_TITLES = [
    "Senior Software Engineer", "Full Stack Developer", "Backend Engineer",
    "Data Engineer", "ML Engineer", "DevOps Engineer", "Frontend Developer",
    "Platform Engineer", "SRE", "Tech Lead",
]

_DEMO_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Flipkart", "Razorpay",
    "Swiggy", "PhonePe", "Infosys", "TCS", "Wipro",
    "Paytm", "Zerodha", "Freshworks", "Zoho", "Ola",
]


def generate_demo_candidates(n: int = 100) -> list[dict]:
    """Generate n demo candidate result dicts for UI testing."""
    random.seed(42)
    candidates = []
    for i in range(n):
        name = _DEMO_NAMES[i % len(_DEMO_NAMES)]
        if i >= len(_DEMO_NAMES):
            name = f"{name} ({i // len(_DEMO_NAMES) + 1})"

        axis = CandidateAxisScores(
            skill_relevance=random.uniform(0.3, 1.0),
            experience_impact=random.uniform(0.2, 0.95),
            domain_coherence=random.uniform(0.25, 0.9),
            narrative_credibility=random.uniform(0.3, 0.85),
            behavioral_validation=random.uniform(0.1, 0.8),
            engagement_feasibility=random.uniform(0.4, 1.0),
            pedigree=random.uniform(0.2, 1.0),
        )

        track1 = sum(
            axis.to_dict()[k] * w
            for k, w in TRACK1_WEIGHTS.items()
            if k in axis.to_dict()
        )
        # Adjust for axes that map differently
        track1 = random.uniform(0.25, 0.92)
        is_hp = random.random() < 0.08
        skills = random.sample(_DEMO_SKILLS, k=random.randint(3, 7))

        candidates.append({
            "rank": i + 1,
            "candidate_id": f"CAND-{1000 + i:04d}",
            "name": name,
            "title": random.choice(_DEMO_TITLES),
            "company": random.choice(_DEMO_COMPANIES),
            "location": random.choice(["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]),
            "years_exp": random.randint(2, 18),
            "final_score": round(track1, 4),
            "track1_score": round(track1, 4),
            "track2_score": round(track1 + random.uniform(-0.05, 0.05), 4) if random.random() > 0.5 else None,
            "axis_scores": axis.to_dict(),
            "top_skills": skills,
            "is_honeypot": is_hp,
            "honeypot_details": "Title keywords do not match career description keywords" if is_hp else "",
            "redirect_suggestion": random.choice(["Data Engineer", "DevOps Engineer", None, None, None]),
            "redirect_reason": "Strong infra skills suggest better fit for DevOps role" if random.random() < 0.15 else "",
            "data_confidence": round(random.uniform(0.5, 1.0), 3),
            "reasoning": f"Strong match on {skills[0]} and {skills[1]}. "
                         f"{'Honeypot flag: title mismatch detected. ' if is_hp else ''}"
                         f"Data confidence: {random.uniform(0.5, 1.0):.0%}.",
        })

    # Sort by score descending and re-rank
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    return candidates


def generate_demo_jd() -> dict:
    """Generate a demo parsed JD profile."""
    return {
        "title": "Senior Backend Engineer",
        "seniority_expected": "senior",
        "must_have_skills": [
            {"skill": "Python", "weight": 1.0},
            {"skill": "Django", "weight": 0.9},
            {"skill": "PostgreSQL", "weight": 0.85},
            {"skill": "REST APIs", "weight": 0.8},
            {"skill": "Docker", "weight": 0.75},
            {"skill": "AWS", "weight": 0.7},
        ],
        "nice_to_have_skills": [
            {"skill": "Kubernetes", "weight": 0.5},
            {"skill": "Redis", "weight": 0.45},
            {"skill": "GraphQL", "weight": 0.4},
            {"skill": "Kafka", "weight": 0.35},
        ],
        "axis_weights": {
            "skill_relevance": 0.30,
            "experience_impact": 0.20,
            "domain_coherence": 0.15,
            "narrative_credibility": 0.10,
            "behavioral_validation": 0.10,
            "engagement_feasibility": 0.10,
            "pedigree": 0.05,
        },
        "logistics": {
            "work_mode": "hybrid",
            "location": "Bangalore",
            "notice_period_max": 60,
        },
        "raw_text": (
            "We are looking for a Senior Backend Engineer to join our platform team. "
            "You will design and build scalable microservices using Python/Django, "
            "manage PostgreSQL databases, and deploy on AWS using Docker and Kubernetes. "
            "5+ years of experience required. Strong REST API design skills essential."
        ),
        "all_keywords": [
            "python", "django", "postgresql", "rest", "api", "docker",
            "aws", "kubernetes", "microservices", "backend", "scalable",
        ],
    }


def generate_demo_skill_trusts(skills: list[str] | None = None) -> list[dict]:
    """Generate demo skill trust data."""
    random.seed(123)
    if not skills:
        skills = random.sample(_DEMO_SKILLS, k=random.randint(5, 10))
    trusts = []
    for s in skills:
        has_assessment = random.random() > 0.3
        trusts.append({
            "skill_name": s,
            "canonical_name": s.lower().replace(" ", "_"),
            "trust_score": round(random.uniform(0.3, 1.0), 3),
            "assessment_score": round(random.uniform(40, 98), 1) if has_assessment else None,
            "duration_months": random.randint(6, 84),
            "proficiency": random.choice(["beginner", "intermediate", "advanced", "expert"]),
            "endorsements": random.randint(0, 45),
            "evidenced_in_career": random.random() > 0.3,
        })
    trusts.sort(key=lambda t: t["trust_score"], reverse=True)
    return trusts


def generate_demo_career() -> list[dict]:
    """Generate a demo career timeline."""
    return [
        {"title": "Senior Backend Engineer", "company": "Razorpay", "start": "2022-03", "end": "Present", "duration_months": 28, "domain": "FinTech", "mismatch": False},
        {"title": "Software Engineer II", "company": "Flipkart", "start": "2019-08", "end": "2022-02", "duration_months": 30, "domain": "E-Commerce", "mismatch": False},
        {"title": "Backend Developer", "company": "Freshworks", "start": "2017-06", "end": "2019-07", "duration_months": 25, "domain": "SaaS", "mismatch": False},
        {"title": "Junior Developer", "company": "TCS", "start": "2015-07", "end": "2017-05", "duration_months": 22, "domain": "IT Services", "mismatch": True},
    ]



# Role-specific redirect reason templates used by generate_demo_redirects
_REDIRECT_REASONS: dict[str, list[str]] = {
    "Data Engineer": [
        "Strong SQL and pipeline experience in career history; weak embedding/ranking signals for AI Engineer role.",
        "Data warehousing and ETL background aligns better with Data Engineering archetype than ML/search systems.",
        "Worked heavily with Spark, Airflow, and dbt — high cluster similarity to Data Engineer archetype (0.78).",
        "Career descriptions mention data pipelines and schema design, not retrieval or ranking systems.",
    ],
    "DevOps Engineer": [
        "CI/CD, Docker, and Kubernetes dominate the skill list; no retrieval or ML system signals found.",
        "Infrastructure-heavy background (Terraform, AWS, monitoring) maps to DevOps archetype, not AI Engineer.",
        "3 of 4 roles focused on deployment automation and SRE work — strong DevOps cluster fit.",
        "Github activity is high but commits are infra-related, not ML/NLP code.",
    ],
    "SRE": [
        "On-call, incident response, and reliability engineering language found across all job descriptions.",
        "Observability stack experience (Prometheus, Grafana, PagerDuty) matches SRE archetype closely.",
        "Latency budgets and SLA ownership language detected — SRE cluster score 0.81, AI Engineer 0.31.",
        "Strong systems debugging and reliability focus; no evidence of ML model ownership.",
    ],
    "ML Engineer": [
        "Has model training experience (PyTorch/TensorFlow) but focuses on CV/speech — not NLP/retrieval.",
        "MLOps and experiment tracking skills present, but domain is image classification, not search/ranking.",
        "Solid ML infrastructure skills; career context is recommendation pipelines in e-commerce, not hiring tech.",
        "Feature engineering and model serving experience maps to ML Engineer, but no vector DB or ranking work.",
    ],
    "Backend Engineer": [
        "Python and system design skills are strong, but no ML or retrieval experience found in descriptions.",
        "Career is entirely API design and microservices — backend engineering archetype fit is 0.82.",
        "REST APIs, PostgreSQL, and distributed systems dominate; AI/ML skills are absent or superficial.",
        "Strong product engineering background without the embedding/ranking depth needed for this AI role.",
    ],
    "Full Stack Developer": [
        "React + Node.js across multiple roles — no AI/ML signal in career history.",
        "Frontend and backend breadth is notable, but retrieval and ranking systems are absent from the profile.",
        "UI and product engineering experience is strong; AI Engineer requires specialized IR/search depth.",
    ],
}

_DEFAULT_REDIRECT_REASONS = [
    "Cluster analysis shows higher cosine similarity to this archetype than to the Senior AI Engineer centroid.",
    "Career trajectory and skill distribution do not match the retrieval/ranking requirements of this role.",
    "Profile signals (skills + descriptions) align more strongly with this archetype based on 7-axis scoring.",
    "Strong generalist engineering background without the specialized embedding/vector DB depth required.",
]


def _make_redirect_reason(suggested_role: str, candidate: dict) -> str:
    """Generate a role-specific, non-empty redirect reason."""
    pool = _REDIRECT_REASONS.get(suggested_role, _DEFAULT_REDIRECT_REASONS)
    # Pick deterministically based on candidate_id so the same candidate always gets the same reason
    cid_hash = abs(hash(candidate.get("candidate_id", "x"))) % len(pool)
    return pool[cid_hash]


def generate_demo_redirects(candidates: list[dict] | None = None) -> pd.DataFrame:
    """Generate demo redirect data with guaranteed non-empty reason strings."""
    random.seed(99)
    if candidates is None:
        candidates = generate_demo_candidates(100)

    redirects = [c for c in candidates if c.get("redirect_suggestion")]

    if not redirects:
        # If no candidates had redirect_suggestion set, pick a random sample
        for c in random.sample(candidates, k=min(15, len(candidates))):
            c["redirect_suggestion"] = random.choice(
                ["Data Engineer", "DevOps Engineer", "SRE", "ML Engineer", "Backend Engineer"]
            )
            redirects.append(c)

    rows = []
    for c in redirects:
        suggested_role = c["redirect_suggestion"]
        # Use stored reason if non-empty, otherwise generate a role-specific one
        reason = c.get("redirect_reason", "").strip()
        if not reason:
            reason = _make_redirect_reason(suggested_role, c)

        rows.append({
            "candidate_id": c["candidate_id"],
            "name": c["name"],
            "current_title": c.get("title", "Unknown"),
            "current_score": c["final_score"],
            "suggested_role": suggested_role,
            "archetype_score": round(random.uniform(0.55, 0.92), 3),
            "reason": reason,
        })
    return pd.DataFrame(rows)


def generate_demo_archetype_scatter(n: int = 100) -> pd.DataFrame:
    """Generate demo PCA scatter data for archetype clusters."""
    random.seed(77)
    archetypes = ["Backend", "Frontend", "DevOps", "Data Eng", "ML/AI", "Full Stack", "SRE", "Platform"]
    rows = []
    for i in range(n):
        arch = random.choice(archetypes)
        # cluster centers
        cx = {"Backend": -2, "Frontend": 2, "DevOps": -1, "Data Eng": 1, "ML/AI": 3, "Full Stack": 0, "SRE": -3, "Platform": -0.5}
        cy = {"Backend": 1, "Frontend": 1, "DevOps": -2, "Data Eng": 2, "ML/AI": -1, "Full Stack": 0, "SRE": -1, "Platform": 2}
        rows.append({
            "candidate_id": f"CAND-{1000 + i:04d}",
            "pca_x": cx[arch] + random.gauss(0, 0.8),
            "pca_y": cy[arch] + random.gauss(0, 0.8),
            "archetype": arch,
            "final_score": round(random.uniform(0.3, 0.9), 3),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Session State Helpers
# ============================================================================
def init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        "jd_raw": "",
        "jd_parsed": None,
        "pipeline_run": False,
        "pipeline_step": 0,
        "candidates": None,
        "selected_candidate_id": None,
        "demo_mode": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_candidates() -> list[dict]:
    """Return candidates from session state or demo data."""
    if st.session_state.get("candidates"):
        return st.session_state["candidates"]
    return generate_demo_candidates()


def get_jd_profile() -> dict | None:
    """Return parsed JD from session state or demo."""
    if st.session_state.get("jd_parsed"):
        return st.session_state["jd_parsed"]
    return None


def load_candidate_profiles(cids: list[str]) -> dict[str, dict]:
    """Retrieve full profile data for a list of candidate IDs.

    Uses a highly optimized single-pass substring scan over candidates.jsonl first.
    Falls back to sample_candidates.json or demo generation if needed.
    """
    import json
    from src.config import CANDIDATES_JSONL, SAMPLE_CANDIDATES_JSON
    from src.ingestion.normalizer import normalize_candidate, build_synonym_lookup

    target_ids = set(cids)
    profiles = {}

    # Try candidates.jsonl first
    if CANDIDATES_JSONL.exists():
        try:
            synonym_lookup = build_synonym_lookup()
            with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    # Fast check before parsing JSON
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
        except Exception:
            pass

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
        except Exception:
            pass

    return profiles
