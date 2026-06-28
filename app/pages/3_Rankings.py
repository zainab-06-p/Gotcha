"""
Gotcha — Rankings Shortlist Page
Displays the official ranked shortlist (ranks 1-100) using native Streamlit components.
Includes filtering by experience, score, and flags, with full CSV export option.
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
    score_badge,
    badge,
    skill_chip,
    get_candidates,
    load_candidate_profiles,
    score_color,
)
from src.config import OUTPUTS_DIR, SAMPLE_CANDIDATES_JSON

# Streamlit config
st.set_page_config(page_title="Gotcha — Rankings", page_icon="🏆", layout="wide")

load_css()
init_session_state()

page_header(
    title="Intelligent Shortlist Rankings",
    subtitle="Verified top candidates ranked by corroboration-weighted match engine | Senior AI Engineer — Redrob AI",
    icon="🏆"
)


@st.cache_data
def load_actual_shortlist() -> list[dict]:
    """Load and merge CSV output with profile data."""
    csv_path = OUTPUTS_DIR / "team_infinity_and_beyond.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "gotcha.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "submission.csv"
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        ranked_cids = list(df["candidate_id"].unique())
        cand_map = load_candidate_profiles(ranked_cids)

        from src.scoring.honeypot import detect_honeypot

        merged = []
        for _, row in df.iterrows():
            cid = row["candidate_id"]
            raw_c = cand_map.get(cid, {})
            profile = raw_c.get("profile", {})
            skills = [s.get("name", "") for s in raw_c.get("skills", []) if s]
            is_hp, hp_details = detect_honeypot(raw_c)

            merged.append({
                "rank": int(row["rank"]),
                "candidate_id": cid,
                "name": profile.get("anonymized_name", "Candidate"),
                "title": profile.get("current_title", "N/A"),
                "company": profile.get("current_company", "N/A"),
                "location": profile.get("location", "India"),
                "years_exp": profile.get("years_of_experience", 0),
                "final_score": float(row["score"]),
                "is_honeypot": is_hp,
                "top_skills": [s for s in skills[:5] if s],
                "reasoning": str(row["reasoning"]),
            })
        return merged
    except Exception as e:
        st.warning(f"Failed to load pipeline output: {e}")
        return []


# ── Load data ──────────────────────────────────────────────────────────────────
actual_candidates = load_actual_shortlist()
if actual_candidates:
    candidates = actual_candidates
    source_label = "✅ Showing live pipeline output: `outputs/team_infinity_and_beyond.csv`"
else:
    candidates = get_candidates()
    source_label = "ℹ️ No pipeline output found — showing demo candidate pool."

st.info(source_label)

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter Candidates")
    min_exp = st.slider("Min Years of Experience", 0, 20, 0)
    max_exp = st.slider("Max Years of Experience", 0, 20, 20)
    min_score = st.slider("Min Score", 0.0, 1.0, 0.0, 0.05)
    exclude_honeypots = st.checkbox("Exclude flagged honeypots", value=True)
    st.markdown("---")
    st.markdown("### 📥 Export")
    if actual_candidates:
        csv_path = OUTPUTS_DIR / "team_infinity_and_beyond.csv"
        if not csv_path.exists():
            csv_path = OUTPUTS_DIR / "gotcha.csv"
        if csv_path.exists():
            csv_bytes = csv_path.read_bytes()
            st.download_button(
                label="⬇️ Download Submission CSV",
                data=csv_bytes,
                file_name="team_infinity_and_beyond.csv",
                mime="text/csv",
            )

# ── Apply Filters ──────────────────────────────────────────────────────────────
filtered = [
    c for c in candidates
    if min_exp <= c.get("years_exp", 0) <= max_exp
    and c["final_score"] >= min_score
    and (not exclude_honeypots or not c.get("is_honeypot", False))
]

# ── Summary metrics row ────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Candidates Shown", len(filtered))
with m2:
    avg_score = sum(c["final_score"] for c in filtered) / len(filtered) if filtered else 0
    st.metric("Avg Score", f"{avg_score:.3f}")
with m3:
    top10_avg = sum(c["final_score"] for c in filtered[:10]) / 10 if len(filtered) >= 10 else 0
    st.metric("Top-10 Avg Score", f"{top10_avg:.3f}")
with m4:
    hp_count = sum(1 for c in candidates if c.get("is_honeypot"))
    st.metric("Honeypots Flagged", hp_count)

divider()

# ── Main Rankings Table ────────────────────────────────────────────────────────
st.write(f"### 🏆 Ranked Shortlist — {len(filtered)} candidates")

if filtered:
    # Build DataFrame for display
    rows = []
    for c in filtered[:100]:
        score = c["final_score"]
        score_pct = f"{score:.1%}"
        if score >= 0.70:
            tier = "🟢 Tier 1"
        elif score >= 0.65:
            tier = "🔵 Tier 2"
        elif score >= 0.60:
            tier = "🟡 Tier 3"
        elif score >= 0.40:
            tier = "🟠 Tier 4"
        else:
            tier = "🔴 Tier 5"

        flag = " ⚠️" if c.get("is_honeypot") else ""
        skills_str = " · ".join(c["top_skills"][:3]) if c["top_skills"] else "—"

        rows.append({
            "Rank": f"#{c['rank']}",
            "Candidate ID": c["candidate_id"] + flag,
            "Title": c["title"],
            "Company": c["company"],
            "Exp": f"{c['years_exp']} yrs",
            "Score": score_pct,
            "Tier": tier,
            "Top Skills": skills_str,
            "Location": c["location"],
        })

    df_display = pd.DataFrame(rows)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(600, 40 + len(rows) * 36),
        column_config={
            "Rank":         st.column_config.TextColumn("Rank", width=60),
            "Candidate ID": st.column_config.TextColumn("Candidate ID", width=160),
            "Title":        st.column_config.TextColumn("Title", width=220),
            "Company":      st.column_config.TextColumn("Company", width=160),
            "Exp":          st.column_config.TextColumn("Exp", width=70),
            "Score":        st.column_config.TextColumn("Score", width=80),
            "Tier":         st.column_config.TextColumn("Tier", width=100),
            "Top Skills":   st.column_config.TextColumn("Top Skills", width=240),
            "Location":     st.column_config.TextColumn("Location", width=130),
        }
    )

    divider()

    # ── Reasoning Detail Expander ──────────────────────────────────────────────
    st.write("### 🔍 Inspect Candidate Reasoning")

    selected_cid = st.selectbox(
        "Select a candidate to read detailed reasoning:",
        options=[c["candidate_id"] for c in filtered[:100]],
        format_func=lambda cid: f"#{next(c['rank'] for c in filtered if c['candidate_id'] == cid)} — {cid}"
    )

    selected = next(c for c in filtered if c["candidate_id"] == selected_cid)
    score = selected["final_score"]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Final Score", f"{score:.4f}", delta=f"Rank #{selected['rank']}")
    with col_b:
        st.metric("Experience", f"{selected['years_exp']} years")
    with col_c:
        st.metric("Location", selected["location"])

    glass_card(f"""
    <h4 style='margin:0 0 8px 0; color:#6C63FF;'>{selected['name']}</h4>
    <p style='margin:4px 0; color:#E2E8F0; font-weight:500;'>{selected['title']} @ {selected['company']}</p>
    <p style='margin:10px 0 4px; color:#A0AEC0; font-size:0.85rem;'><b>Skills:</b> {' · '.join(selected['top_skills']) if selected['top_skills'] else '—'}</p>
    <p style='margin:10px 0 4px; color:#A0AEC0; font-size:0.85rem;'><b>Reasoning:</b></p>
    <p style='margin:0; color:#E2E8F0; font-size:0.95rem; line-height:1.6;'>"{selected['reasoning']}"</p>
    {"<p style='color:#FC8181; margin-top:10px; font-size:0.85rem;'>⚠️ Honeypot flag active on this profile</p>" if selected.get("is_honeypot") else ""}
    """, accent="purple")

    # Link to profile page
    st.page_link(
        "pages/4_Candidate_Profile.py",
        label=f"👤 Open Full Profile for {selected_cid}",
        icon="👤"
    )

else:
    st.info("No candidates match the current filters. Adjust the sidebar sliders.")
