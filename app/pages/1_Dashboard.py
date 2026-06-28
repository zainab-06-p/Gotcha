"""
Gotcha — Dashboard Page
Visual analytics dashboard displaying pool-wide candidate metrics, archetype distributions, and component correlation charts.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    render_metric_row,
    get_candidates,
    score_histogram,
    archetype_scatter,
)

# Streamlit page config
st.set_page_config(page_title="Gotcha — Dashboard", page_icon="📊", layout="wide")

# Load CSS and init state
load_css()
init_session_state()

# Page title
page_header(
    title="Pool Analytics Dashboard",
    subtitle="Global insights and statistical overview of the candidate talent pool",
    icon="📊"
)

# Get candidates
candidates = get_candidates()
total_cands = len(candidates)
avg_score = sum(c["final_score"] for c in candidates) / total_cands
honeypots = sum(1 for c in candidates if c.get("is_honeypot", False))
redirects = sum(1 for c in candidates if c.get("redirect_suggestion"))

# Metrics Row
metrics = [
    {"icon": "👥", "value": f"{total_cands:,}", "label": "Processed Candidates", "delta": "", "accent": "purple"},
    {"icon": "📈", "value": f"{avg_score:.2%}", "label": "Average Match Score", "delta": "", "accent": "teal"},
    {"icon": "⚠️", "value": f"{honeypots} ({honeypots/total_cands:.1%})", "label": "Flagged Honeypots", "delta": "Critical Alert", "accent": "coral"},
    {"icon": "🔄", "value": f"{redirects}", "label": "Redirect Candidates", "delta": "Suggested", "accent": "amber"},
]
render_metric_row(metrics)

st.write("### Pool Distribution")

col1, col2 = st.columns(2)

with col1:
    # Score distribution histogram
    scores = [c["final_score"] for c in candidates]
    fig_hist = score_histogram(scores, "Score Distribution (Final Blended Scores)")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Archetype breakdown bar chart
    archetype_counts = {}
    for c in candidates:
        arch = c.get("redirect_suggestion") or "Direct JD Match"
        if "Redirect to:" in arch:
            arch = arch.replace("Redirect to: ", "")
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        
    df_arch = pd.DataFrame([
        {"Archetype": k, "Count": v} for k, v in archetype_counts.items()
    ]).sort_values(by="Count", ascending=False)
    
    fig_bar = px.bar(
        df_arch, x="Archetype", y="Count",
        color="Count",
        color_continuous_scale="Purples",
        title="Candidate Talent Pool Archetypes",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.write("### Multi-Dimensional Clustering")

col3, col4 = st.columns([2, 1])

with col3:
    comp_cols = [k for k in candidates[0].get("axis_scores", {})] if candidates and candidates[0].get("axis_scores") else []
    if len(comp_cols) >= 3:
        comp_data = [c["axis_scores"] for c in candidates]
        df_comp = pd.DataFrame(comp_data)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_comp[comp_cols])
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(scaled)
        archetype_labels = []
        for c in candidates:
            arch = c.get("redirect_suggestion") or "Direct JD Match"
            if "Redirect to:" in arch:
                arch = arch.replace("Redirect to: ", "")
            archetype_labels.append(arch)
        df_scatter = pd.DataFrame({
            "pca_x": coords[:, 0], "pca_y": coords[:, 1],
            "final_score": [c["final_score"] for c in candidates],
            "archetype": archetype_labels,
            "candidate_id": [c.get("candidate_id", f"CAND-{i:04d}") for i, c in enumerate(candidates)],
        })
        fig_scatter = archetype_scatter(df_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("### 🧭 Candidate Embedding Map")
        st.info("Component-level data not available for PCA projection.")
        st.plotly_chart(px.scatter(title="No data — run pipeline first"), use_container_width=True)

with col4:
    explain_clusters = """
    <h4>Score Component PCA (2D Projection)</h4>
    <p style='color:#A0AEC0; font-size: 0.95rem; line-height: 1.5;'>
        Pipeline generates <b>8 component scores</b> per candidate (skill, trust, career, behavioral,
        feasibility, yoe_fit, data_confidence, pedigree), reduced via <b>PCA → 2D</b>.
    </p>
    <p style='color:#A0AEC0; font-size: 0.95rem; line-height: 1.5;'>
        Point color shows final blended score tier and marker shape flags honeypots.
    </p>
    <ul style='color:#A0AEC0; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.5;'>
        <li><b>Outliers</b> are penalized candidates (disqualifiers, low YOE, vision-only).</li>
        <li><b>Tight clusters near the center</b> are ideal-fit candidates (no penalties, 5–9yr YoE).</li>
    </ul>
    """
    glass_card(explain_clusters, accent="purple")
