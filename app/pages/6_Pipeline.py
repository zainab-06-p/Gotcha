"""
Gotcha — Pipeline Execution Page
Interactive control panel to configure, run, and validate the end-to-end candidate ranking pipeline.
Shows execution logs and provides direct verification status checks.
"""

import sys
import re
import subprocess
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
    alert_banner,
)
from src.config import JOB_DESCRIPTION_DOCX, CANDIDATES_JSONL, SAMPLE_CANDIDATES_JSON, OUTPUTS_DIR

# Streamlit config
st.set_page_config(page_title="Gotcha — Pipeline", page_icon="⚙️", layout="wide")

load_css()
init_session_state()

page_header(
    title="Pipeline Orchestrator",
    subtitle="Configure, execute, and validate the Gotcha discovery pipeline on candidate datasets",
    icon="⚙️"
)

col1, col2 = st.columns([1, 1])

# Determine default paths
default_jd = str(JOB_DESCRIPTION_DOCX)
default_cands = str(CANDIDATES_JSONL)
if not Path(default_cands).exists():
    default_cands = str(SAMPLE_CANDIDATES_JSON)

with col1:
    st.write("### Pipeline Configuration")

    # ── JD notice ───────────────────────────────────────────────────────────
    st.info(
        "**JD is hardcoded:** The pipeline uses the **Senior AI Engineer — Redrob AI** "
        "job description with all 8 hard disqualifiers, YoE fit scoring (5–9 yrs target), "
        "and nice-to-have bonuses. No LLM call is needed for JD parsing.",
        icon="📌"
    )

    jd_path = st.text_input(
        "Job Description File Path (.docx):",
        value=default_jd,
        help="Note: JD is hardcoded in src/jd_redrob.py — this path is used for reference only."
    )

    # ── File uploader for custom candidates dataset ──
    uploaded_file = st.file_uploader(
        "Upload a candidates file (.jsonl or .json):",
        type=["jsonl", "json"],
        help="Upload your own candidates dataset. Overrides the path below."
    )

    if uploaded_file is not None:
        import tempfile
        suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        candidates_path = tmp.name
        st.success(f"Using uploaded file: {uploaded_file.name}")
    else:
        candidates_path = st.text_input("Candidates Dataset File Path (.json or .jsonl):", value=default_cands)

    output_csv = st.text_input(
        "Output CSV Destination Path:",
        value=str(OUTPUTS_DIR / "team_infinity_and_beyond.csv")
    )

    limit_size = st.number_input(
        "Candidate Batch Limit (0 = all; recommended 1000–5000 for PoC):",
        min_value=0,
        max_value=100000,
        value=1000,
        step=500
    )

    run_btn = st.button("🚀 Run Gotcha Pipeline (Redrob JD)", type="primary")

    if run_btn:
        st.write("---")
        st.write("### Execution Logs")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "rank.py"),
            "--jd", jd_path,
            "--candidates", candidates_path,
            "--output", output_csv,
        ]
        if int(limit_size) > 0:
            cmd += ["--limit", str(int(limit_size))]
        
        with st.spinner("Executing pipeline in background..."):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(PROJECT_ROOT)
                )

                log = result.stdout

                total_match = re.search(r"Loaded (\d+) candidates total", log)
                disq_match = re.search(r"Disqualified: (\d+) candidates", log)
                hp_match = re.search(r"Honeypots: (\d+)", log)
                cluster_match = re.search(r"clustered (\d+) candidates into (\d+) archetypes", log)
                csv_match = re.search(r"Successfully wrote (\d+) ranked candidates", log)

                total = total_match.group(1) if total_match else "?"
                disq = disq_match.group(1) if disq_match else "?"
                hp = hp_match.group(1) if hp_match else "?"
                clustered = cluster_match.group(1) if cluster_match else "?"
                archetypes = cluster_match.group(2) if cluster_match else "?"
                written = csv_match.group(1) if csv_match else "?"

                tag_counts = {}
                for tag_line in re.findall(r"disqualified by:\s*(\[[^\]]+\])", log):
                    for t in re.findall(r"'([^']+)'", tag_line):
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]

                st.success("✅ Pipeline completed successfully!")

                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("📦 Candidates Loaded", total)
                with summary_cols[1]:
                    st.metric("🚫 Disqualified", disq)
                with summary_cols[2]:
                    st.metric("⚠️ Honeypots", hp)
                with summary_cols[3]:
                    st.metric("🏛️ Archetypes Found", archetypes)

                st.write("#### 📊 Disqualifier Breakdown (Top 5)")
                if top_tags:
                    tag_df = pd.DataFrame(top_tags, columns=["Rule", "Count"])
                    st.dataframe(tag_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No disqualifiers triggered in this run.")

                st.write("#### 🏆 Ranked Output Preview")
                csv_out = Path(output_csv)
                if csv_out.exists():
                    try:
                        preview = pd.read_csv(csv_out)
                        cols = ["rank", "candidate_id", "score"]
                        st.dataframe(
                            preview[cols].head(10),
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "rank": st.column_config.TextColumn("Rank", width=50),
                                "candidate_id": st.column_config.TextColumn("Candidate ID", width=140),
                                "score": st.column_config.NumberColumn("Score", width=80, format="%.4f"),
                            }
                        )
                        st.caption(f"Showing top 10 of {written} ranked candidates.")
                    except Exception:
                        st.code(log[:2000], language="text")
                else:
                    st.code(log[:2000], language="text")

                st.session_state["pipeline_run"] = True

            except subprocess.CalledProcessError as e:
                st.error("❌ Pipeline execution failed!")
                st.code(e.stderr, language="text")

with col2:
    st.write("### Challenge Validation Status")
    
    validation_status_card = """
    <h4>Verify Output Compliance</h4>
    <p style='color:#A0AEC0;'>
        The challenge portal runs a strict validator (<code>validate_submission.py</code>) requiring exactly 100 data rows, non-increasing scores, and ascending candidate_id tie-breaking.
    </p>
    """
    glass_card(validation_status_card, accent="purple")
    
    val_btn = st.button("Run Challenge Validator on Output CSV", type="secondary")
    
    if val_btn:
        val_script = PROJECT_ROOT.parent / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "validate_submission.py"
        
        if not val_script.exists():
            st.error(f"Validator script not found at {val_script}")
        else:
            # Build validation command
            val_cmd = [
                sys.executable,
                str(val_script),
                output_csv
            ]
            
            try:
                val_result = subprocess.run(
                    val_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT)
                )
                
                if val_result.returncode == 0:
                    alert_banner(
                        text="VALIDATOR PASSED: Your submission CSV is fully compliant with the challenge rules!",
                        kind="success",
                        icon="✅"
                    )
                else:
                    alert_banner(
                        text=f"VALIDATOR FAILED: Please fix the errors below:<br/><br/>{val_result.stdout}",
                        kind="honeypot",
                        icon="❌"
                    )
            except Exception as ex:
                st.error(f"Error running validator: {ex}")
