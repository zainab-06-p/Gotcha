
# 🎯 Gotcha — Corroboration-Based Candidate Discovery & Ranking Engine

> **India Runs by Redrob AI — Track 01: The Data & AI Challenge**  
> Intelligent Candidate Discovery & Ranking  
> **Team:** Infinity and Beyond

---

## What is Gotcha?

Gotcha is a **corroboration-based** candidate ranking engine that doesn't trust resumes at face value. Instead of keyword-matching skill lists, it builds a **weighted trust model** where claims only count when backed by evidence elsewhere in the profile.

### The Problem
Recruiters drown in profiles. Traditional AI screening trusts the skills list, the job title, and the summary — all of which are self-reported and frequently inaccurate. In the challenge dataset, we observed:
- Job titles that contradict their own descriptions
- Near-identical boilerplate summaries across unrelated candidates  
- Skill lists with zero connection to actual work history

### The Gotcha Approach
1. **Don't trust titles** — Read descriptions independently
2. **Corroborate skills** — A skill only counts if assessment scores, career descriptions, or endorsements back it up
3. **Score on 7 axes** — Not one opaque number, but a multi-dimensional profile
4. **Detect honeypots** — Flag title/description mismatches automatically
5. **Hard disqualify** — 8 strict filters eliminate irrelevant roles (non-engineering, IT services-only, wrong domain) before scoring
6. **Redirect candidates** — If someone doesn't fit this job, the engine identifies which role they'd be great at

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install
```bash
cd gotcha
pip install -r requirements.txt
```

### Reproduce the Submission CSV (exact command)
```bash
python scripts/rank.py --candidates /path/to/candidates.jsonl --output ./outputs/team_infinity_and_beyond.csv
```

The JD is **hardcoded** in `src/jd_redrob.py` — no `--jd` flag required. The pipeline runs entirely offline once dependencies are installed (Track 2 LLM falls back gracefully if no API key is set).

**Expected runtime:** ~2–5 minutes for 5,000 candidates on CPU. Under 5 minutes for the reproduction step.

### Validate the Output
```bash
python ../[PUB]\ India_runs_data_and_ai_challenge/[PUB]\ India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py outputs/team_infinity_and_beyond.csv
# Must print: "Submission is valid."
```

### Run on Sample Data (fast, no dataset required)
```bash
python scripts/rank.py --output ./outputs/sample_run.csv
# Falls back to sample_candidates.json automatically if candidates.jsonl is not found
```

### Launch the Demo UI
```bash
streamlit run app/streamlit_app.py
```

---

## Architecture Overview

See [Gotcha_Architecture.md](./Gotcha_Architecture.md) for the full technical deep-dive.

### Two-Track Scoring System

```
Track 1 (Deterministic)                  Post-scoring
├── Hard disqualifier pre-filter (8 rules) → Archetype clustering (K-Means, 8 clusters)
├── Skill-JD fuzzy match          (0.25)  → Redirect detection
├── Skill trust scoring            (0.20)  → Final ranking (top 100)
├── Career TF-IDF relevance        (0.18)
├── Behavioral signals             (0.10)
├── YoE fit scoring                (0.10)
├── Feasibility                    (0.10)
├── Data confidence                (0.05)
└── Pedigree (capped)              (0.02)
```

**Fully deterministic** — no external API calls required. Runs offline in under 5 minutes.

### Hard Disqualifier Rules (applied before scoring)

Candidates failing any of these are capped to ≤15% of their score or zeroed:
1. **Non-engineering role** — Mobile, QA, Java, DevOps, Frontend titles
2. **IT services only** — No product company experience
3. **No AI/ML domain** — Zero relevant keywords in entire profile
4. **Wrong seniority** — Fresher or intern-level profiles
5. **Keyword stuffer** — Skill-list spamming with no career evidence
6. **Non-India location** — Outside India with no relocation signal
7. **Too senior** — 15+ YoE applying for a 5-9 YoE role
8. **No Python** — Hard requirement for this specific role

### 7-Axis Candidate Profile

| Axis | What it measures | Weight |
|------|-----------------|--------|
| 1. Skill Relevance | Do your skills match this job's requirements? | 0.25 |
| 2. Experience & Impact | Seniority, ownership, scope of work | 0.20 |
| 3. Domain Coherence | Does your career tell a consistent story? | 0.18 |
| 4. Behavioral Validation | Market signals: github, recruiter saves, response rate | 0.10 |
| 5. YoE Fit | Are you in the 5–9 year sweet spot? | 0.10 |
| 6. Engagement Feasibility | Can this hire actually happen? (notice, salary, location) | 0.10 |
| 7. Data Confidence | How complete and verifiable is the profile? | 0.05 |
| 8. Pedigree (capped) | Education tier, company size (deliberately low) | 0.02 |

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Scoring Engine | Python + numpy + scikit-learn |
| Skill Matching | rapidfuzz + synonym dictionary (`data/skill_synonyms.json`) |
| Demo UI | Streamlit + Plotly |
| Charts | Plotly (radar, bar, histogram, scatter) |
| Data | pandas DataFrames |

**No exotic dependencies.** No graph databases, no FAISS indexes, no external APIs required.

---

## Project Structure

```
gotcha/
├── app/                    # Streamlit demo UI
│   ├── streamlit_app.py    # Main landing page
│   ├── utils.py            # Shared UI helpers
│   ├── style.css           # Premium dark UI styling
│   └── pages/              # Multi-page Streamlit app
│       ├── 1_Dashboard.py
│       ├── 2_Job_Description.py
│       ├── 3_Rankings.py
│       ├── 4_Candidate_Profile.py
│       ├── 5_Redirects.py
│       └── 6_Pipeline.py
├── data/
│   └── skill_synonyms.json # 100+ skill alias mappings
├── outputs/
│   └── team_infinity_and_beyond.csv  # Final submission
├── scripts/
│   ├── rank.py             # ← REPRODUCE COMMAND ENTRY POINT
│   └── run_pipeline.py     # End-to-end pipeline logic
├── src/
│   ├── config.py           # All weights, thresholds, data structures
│   ├── jd_redrob.py        # Hardcoded JD profile (Senior AI Engineer)
│   ├── clustering/         # K-Means archetype discovery
│   ├── ingestion/          # Loader, normalizer, sampler
│   ├── llm/                # Gemini client + extractor
│   ├── ranking/            # Combiner, ranker, explainer
│   ├── redirect/           # Mismatch detection
│   └── scoring/            # All 8 scoring modules
│       ├── skill_matcher.py
│       ├── trust_scorer.py
│       ├── career_relevance.py
│       ├── behavioral.py
│       ├── feasibility.py
│       ├── pedigree.py
│       ├── confidence.py
│       ├── honeypot.py
│       └── disqualifier.py
├── tests/                  # Unit tests
├── requirements.txt
├── submission_metadata.yaml
└── Gotcha_Architecture.md  # Full technical deep-dive
```

---

## Submission

- **Output:** `outputs/team_infinity_and_beyond.csv` — 100 ranked candidates
- **Format:** `candidate_id, rank, score, reasoning`
- **Validated with:** `python validate_submission.py outputs/team_infinity_and_beyond.csv`
- **Team ID:** `team_infinity_and_beyond`

---

## AI Tools Declaration

- **Claude** — Architecture discussion and code review

---

## Known Limitations

- Processes a representative sample (3,000–5,000 candidates) for the PoC. Full 100K is the same architecture, just a longer batch job.
- Learned reranker (Section 9 of architecture doc) is designed but not trained — insufficient labeled outcome history in the dataset.
- Education tier is deliberately down-weighted based on empirical observation, not ideology.

---

## License

MIT — built for the Redrob Hackathon 2026
