"""
Gotcha — Redirect Suggestions Page
Presents redirected candidate matches using actual pipeline output.
"""

import sys
import re
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    init_session_state,
    load_css,
    page_header,
    glass_card,
    divider,
)
from src.config import REDIRECT_LOW_THRESHOLD

st.set_page_config(page_title="Gotcha — Redirects", page_icon="🔄", layout="wide")
load_css()
init_session_state()

page_header(
    title="Candidate Redirect Suggestions",
    subtitle="Candidates who score below the JD threshold — potential fits for other archetypes",
    icon="🔄"
)

glass_card("""
<h4 style='margin:0 0 8px 0;'>What is a Redirect?</h4>
<p style='color:#A0AEC0; margin:0; font-size:0.95rem; line-height:1.6;'>
Candidates listed here scored <b>below the redirect threshold</b> for the Senior AI Engineer JD.
Rather than discarding them, the system flags them as potential pipeline additions for other
open roles based on their skill and career profile.
</p>
""", accent="teal")

CSV_PATH = PROJECT_ROOT / "outputs" / "team_infinity_and_beyond.csv"

@st.cache_data
def load_redirect_csv(path: Path, threshold: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["score"] < threshold].copy()
    df = df.sort_values("score", ascending=False)
    return df

df = load_redirect_csv(CSV_PATH, REDIRECT_LOW_THRESHOLD)

if df.empty:
    st.info("No redirect candidates found. Run the pipeline first or check the threshold.")
    st.stop()

_ARCHETYPE_KEYWORDS = {
    "Data Engineer":          ["sql", "pipeline", "etl", "warehouse", "spark", "airflow", "dbt", "data engineer"],
    "DevOps Engineer":        ["ci/cd", "docker", "kubernetes", "terraform", "jenkins", "ansible", "devops"],
    "SRE":                    ["on-call", "incident", "reliability", "prometheus", "grafana", "sre", "pagerduty", "sla"],
    "Backend Engineer":       ["backend", "api", "microservice", "rest", "grpc", "postgresql", "system design"],
    "Full Stack Developer":   ["react", "node.js", "frontend", "vue", "angular", "full stack", "ui"],
    "ML Engineer":            ["pytorch", "tensorflow", "mlops", "model training", "feature engineering", "ml engineer"],
    "Mobile Developer":       ["android", "ios", "kotlin", "swift", "mobile", "flutter", "react native"],
    "Platform Engineer":      ["platform", "infrastructure", "helm", "istio", "cloud", "aws", "gcp", "azure"],
}

def infer_archetype(reasoning: str) -> str:
    if not isinstance(reasoning, str):
        return "Other Roles"
    r_lower = reasoning.lower()
    scores = {}
    for arch, keywords in _ARCHETYPE_KEYWORDS.items():
        scores[arch] = sum(1 for kw in keywords if kw in r_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other Roles"

def make_reason(row) -> str:
    cid = row["candidate_id"]
    score = row["score"]
    arch = row.get("suggested_role", "Other Roles")
    return (
        f"Candidate score for Senior AI Engineer is {score:.1%} (< threshold). "
        f"Profile aligns with '{arch}' archetype based on career description signals."
    )

df["suggested_role"] = df["reasoning"].apply(infer_archetype)
df["archetype_score"] = 0.0
df["name"] = df["candidate_id"].apply(lambda c: c.replace("CAND_", ""))
df["reason"] = df.apply(make_reason, axis=1)

with st.sidebar:
    st.markdown("### 🔍 Filter Redirects")
    role_opts = sorted(df["suggested_role"].unique().tolist())
    selected_role = st.selectbox("Filter by Suggested Role:", options=["All"] + role_opts)
    min_score = st.slider("Min Current Score", 0.0, 0.50, 0.0, 0.01)

df_filtered = df.copy()
if selected_role != "All":
    df_filtered = df_filtered[df_filtered["suggested_role"] == selected_role]
df_filtered = df_filtered[df_filtered["score"] >= min_score]

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Redirect Candidates", len(df_filtered))
with m2:
    avg_arch = df_filtered["score"].mean() if len(df_filtered) > 0 else 0
    st.metric("Avg JD Score", f"{avg_arch:.3f}")
with m3:
    st.metric("Distinct Roles Suggested", df_filtered["suggested_role"].nunique())

divider()

# ── Main table ────────────────────────────────────────────────────────────────
st.write(f"### 🔄 {len(df_filtered)} Redirect Suggestions")

if not df_filtered.empty:
    display_rows = []
    for _, row in df_filtered.iterrows():
        cs = row["score"]
        tier = "🔴 Low" if cs < 0.3 else "🟡 Mid"

        display_rows.append({
            "Candidate ID":  row["candidate_id"],
            "JD Score":      f"{cs:.2f}  {tier}",
            "Suggested Role":row["suggested_role"],
            "Reason":        row["reason"],
        })

    df_display = pd.DataFrame(display_rows)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(520, 50 + len(display_rows) * 38),
        column_config={
            "Candidate ID":  st.column_config.TextColumn("Candidate ID", width=150),
            "JD Score":      st.column_config.TextColumn("JD Score", width=120),
            "Suggested Role":st.column_config.TextColumn("Suggested Role →", width=200),
            "Reason":        st.column_config.TextColumn("Reason for Redirect", width=480),
        }
    )

    divider()

    # ── Per-role breakdown chart ───────────────────────────────────────────────
    st.write("### 📊 Breakdown by Suggested Role")
    role_counts = (
        df_filtered.groupby("suggested_role")
        .agg(Count=("candidate_id", "count"), Avg_Score=("score", "mean"))
        .reset_index()
        .rename(columns={"suggested_role": "Role", "Avg_Score": "Avg JD Score"})
    )
    role_counts["Avg JD Score"] = role_counts["Avg JD Score"].round(3)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(role_counts, use_container_width=True, hide_index=True)
    with col2:
        import plotly.express as px
        fig = px.bar(
            role_counts, x="Role", y="Count",
            color="Avg JD Score",
            color_continuous_scale=["#FF6B6B", "#FFB347", "#00D4AA"],
            title="Redirect Candidates per Role",
            text="Count",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=300,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Individual candidate reason cards ────────────────────────────────────
    st.write("### 🔍 Inspect a Redirect Candidate")
    sel_cid = st.selectbox(
        "Select candidate:",
        options=df_filtered["candidate_id"].tolist(),
        format_func=lambda c: f"{c}  —  {df_filtered[df_filtered['candidate_id']==c]['suggested_role'].values[0]}"
    )
    sel = df_filtered[df_filtered["candidate_id"] == sel_cid].iloc[0]
    glass_card(f"""
    <h4 style='margin:0 0 6px; color:#6C63FF;'>{sel['candidate_id']} &nbsp;·&nbsp; <span style='color:#A0AEC0;font-size:0.95rem;'>{sel['reasoning'][:80]}</span></h4>
    <p style='margin:4px 0; font-size:0.9rem; color:#A0AEC0;'>
        Current JD score: <b style='color:#FC8181;'>{sel['score']:.2f}</b> &nbsp;|&nbsp;
        Suggested role: <b style='color:#00D4AA;'>{sel['suggested_role']}</b>
    </p>
    <p style='margin:10px 0 0; color:#E2E8F0; font-size:0.95rem; line-height:1.65;'>
        <b>Reason:</b> {sel['reason']}
    </p>
    """, accent="teal")

    divider()

    # ── Recruiter Tips ────────────────────────────────────────────────────────
    st.write("### 💡 Recruiter Tips: How to Use Redirects")
    tip_cols = st.columns(3)
    tips = [
        ("🚀 Fast-Track Them",
         "If archetype fit ≥ 0.75 (computed during full pipeline clustering), skip initial screening. Route directly to the engineering manager for that role."),
        ("🗃️ ATS Re-tagging",
         "Re-categorize their profiles under the suggested role in your ATS to reduce future sourcing costs and keep them in the active pipeline."),
        ("📧 Honest Outreach",
         "Contact them with the specific role suggestion. Candidates appreciate transparency — 'You're a stronger fit for X than Y' is more compelling than silence."),
    ]
    for col, (title, desc) in zip(tip_cols, tips):
        with col:
            glass_card(
                f"<h4 style='margin:0 0 8px; color:#00D4AA;'>{title}</h4>"
                f"<p style='color:#A0AEC0; margin:0; font-size:0.9rem; line-height:1.6;'>{desc}</p>",
                accent="teal"
            )
else:
    st.info("No redirect candidates match the current filters.")

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter Redirects")
    suggested_roles_opts = sorted(df["suggested_role"].unique().tolist())
    selected_role = st.selectbox("Filter by Suggested Role:", options=["All"] + suggested_roles_opts)
    min_score = st.slider("Min Current Score", 0.0, 0.50, 0.0, 0.01)

df_filtered = df.copy()
if selected_role != "All":
    df_filtered = df_filtered[df_filtered["suggested_role"] == selected_role]
df_filtered = df_filtered[df_filtered["score"] >= min_score]

# ── Metrics ────────────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Redirect Candidates", len(df_filtered))
with m2:
    avg_arch = df_filtered["archetype_score"].mean() if len(df_filtered) > 0 else 0
    st.metric("Avg Archetype Score", f"{avg_arch:.3f}")
with m3:
    st.metric("Distinct Roles Suggested", df_filtered["suggested_role"].nunique())

divider()

# ── Main table ────────────────────────────────────────────────────────────────
st.write(f"### 🔄 {len(df_filtered)} Redirect Suggestions")

if not df_filtered.empty:
    display_rows = []
    for _, row in df_filtered.iterrows():
        cs = row["current_score"]
        ar = row["archetype_score"]
        cs_tier = "🔴 Low" if cs < 0.3 else "🟡 Mid"
        ar_tier  = "🟢 Strong" if ar >= 0.75 else ("🟡 Moderate" if ar >= 0.55 else "🟠 Weak")

        display_rows.append({
            "Candidate ID":     row["candidate_id"],
            "Name":             row["name"],
            "Current Title":    row["current_title"],
            "JD Score":         f"{cs:.2f}  {cs_tier}",
            "Suggested Role":   row["suggested_role"],
            "Archetype Fit":    f"{ar:.2f}  {ar_tier}",
            "Reason":           row["reason"],
        })

    df_display = pd.DataFrame(display_rows)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(520, 50 + len(display_rows) * 38),
        column_config={
            "Candidate ID":  st.column_config.TextColumn("Candidate ID", width=150),
            "Name":          st.column_config.TextColumn("Name", width=150),
            "Current Title": st.column_config.TextColumn("Current Title", width=180),
            "JD Score":      st.column_config.TextColumn("JD Score", width=120),
            "Suggested Role":st.column_config.TextColumn("Suggested Role →", width=160),
            "Archetype Fit": st.column_config.TextColumn("Archetype Fit", width=140),
            "Reason":        st.column_config.TextColumn("Reason for Redirect", width=420),
        }
    )

    divider()

    # ── Per-role breakdown chart ───────────────────────────────────────────────
    st.write("### 📊 Breakdown by Suggested Role")
    role_counts = (
        df_filtered.groupby("suggested_role")
        .agg(Count=("candidate_id", "count"), Avg_Score=("archetype_score", "mean"))
        .reset_index()
        .rename(columns={"suggested_role": "Role", "Avg_Score": "Avg Archetype Score"})
    )
    role_counts["Avg Archetype Score"] = role_counts["Avg Archetype Score"].round(3)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(role_counts, use_container_width=True, hide_index=True)
    with col2:
        import plotly.express as px
        fig = px.bar(
            role_counts, x="Role", y="Count",
            color="Avg Archetype Score",
            color_continuous_scale=["#FF6B6B", "#FFB347", "#00D4AA"],
            title="Redirect Candidates per Role",
            text="Count",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=300,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Individual candidate reason cards ────────────────────────────────────
    st.write("### 🔍 Inspect a Redirect Candidate")
    sel_cid = st.selectbox(
        "Select candidate:",
        options=df_filtered["candidate_id"].tolist(),
        format_func=lambda c: f"{c}  —  {df_filtered[df_filtered['candidate_id']==c]['suggested_role'].values[0]}"
    )
    sel = df_filtered[df_filtered["candidate_id"] == sel_cid].iloc[0]
    glass_card(f"""
    <h4 style='margin:0 0 6px; color:#6C63FF;'>{sel['name']} &nbsp;·&nbsp; <span style='color:#A0AEC0;font-size:0.95rem;'>{sel['current_title']}</span></h4>
    <p style='margin:4px 0; font-size:0.9rem; color:#A0AEC0;'>
        Current JD score: <b style='color:#FC8181;'>{sel['current_score']:.2f}</b> &nbsp;|&nbsp;
        Suggested role: <b style='color:#00D4AA;'>{sel['suggested_role']}</b> &nbsp;|&nbsp;
        Archetype match: <b style='color:#00D4AA;'>{sel['archetype_score']:.2f}</b>
    </p>
    <p style='margin:10px 0 0; color:#E2E8F0; font-size:0.95rem; line-height:1.65;'>
        <b>Reason:</b> {sel['reason']}
    </p>
    """, accent="teal")

    divider()

    # ── Recruiter Tips ────────────────────────────────────────────────────────
    st.write("### 💡 Recruiter Tips: How to Use Redirects")
    tip_cols = st.columns(3)
    tips = [
        ("🚀 Fast-Track Them",
         "If archetype fit ≥ 0.75, skip initial screening. Route directly to the engineering manager for that role — they have already passed candidate quality checks."),
        ("🗃️ ATS Re-tagging",
         "Re-categorize their profiles under the suggested role in your ATS to reduce future sourcing costs and keep them in the active pipeline."),
        ("📧 Honest Outreach",
         "Contact them with the specific role suggestion. Candidates appreciate transparency — 'You're a stronger fit for X than Y' is more compelling than silence."),
    ]
    for col, (title, desc) in zip(tip_cols, tips):
        with col:
            glass_card(
                f"<h4 style='margin:0 0 8px; color:#00D4AA;'>{title}</h4>"
                f"<p style='color:#A0AEC0; margin:0; font-size:0.9rem; line-height:1.6;'>{desc}</p>",
                accent="teal"
            )
else:
    st.info("No redirect candidates match the current filters.")
