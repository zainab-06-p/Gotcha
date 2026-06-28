"""
Gotcha — LLM Prompt Templates
Stores the prompt templates used for JD parsing and candidate evaluation.
"""

JD_PARSING_SYSTEM_PROMPT = """You are an expert technical recruiter and AI talent matching engine.
Your task is to parse a raw job description (JD) and extract structured requirements in valid JSON format.
"""

JD_PARSING_USER_PROMPT = """Analyze the following job description and extract:
1. Job Title.
2. Must-have/Required Skills: list of skills. For each, assign a relative weight (0.0 to 1.0) based on how essential it is. The weights of must-have skills should sum to 1.0.
3. Nice-to-have/Preferred Skills: list of skills with relative weights (sum to 1.0).
4. Seniority expected (e.g., junior, mid, senior, lead, principal).
5. Work mode (remote, hybrid, onsite).
6. Minimum years of experience expected.

You MUST respond ONLY with a valid JSON object matching the schema below. Do not wrap in markdown blocks, do not add preamble.

JSON Schema:
{{
    "title": "string",
    "must_have_skills": [
        {{"skill": "string", "weight": float}}
    ],
    "nice_to_have_skills": [
        {{"skill": "string", "weight": float}}
    ],
    "seniority_expected": "string",
    "work_mode": "string",
    "min_experience_years": float
}}

Job Description text:
{jd_text}
"""

CANDIDATE_EVALUATION_SYSTEM_PROMPT = """You are an expert senior engineering manager evaluating a candidate's resume/profile for a specific job.
You judge candidate quality based on narrative depth, scope of impact, and career progression coherence.
"""

CANDIDATE_EVALUATION_USER_PROMPT = """Evaluate candidate {candidate_name} for the job: "{jd_title}".

Candidate Profile:
- Summary: {candidate_summary}
- Headline: {candidate_headline}
- Years of Experience: {candidate_yoe}
- Career History:
{candidate_career}

Job Description Details:
- Must-have skills: {must_have_skills}
- Expected Seniority: {expected_seniority}
- Job Description Context:
{jd_context}

Based on this information, score the candidate on the following three axes (0.0 to 1.0 float scale):
1. experience_impact: Does their career history show high-impact projects, scaling, leadership, and matching seniority?
2. domain_coherence: Is their career path consistent and logical, or do they jump randomly between unrelated fields?
3. narrative_credibility: Does their summary and job descriptions feel authentic, detailed, and specific, or is it filled with generic boilerplate and buzzwords?

Also write a single-sentence "reasoning" summary explaining your evaluation.

You MUST respond ONLY with a valid JSON object matching this schema. Do not add markdown or preamble.

JSON Schema:
{{
    "experience_impact": float,
    "domain_coherence": float,
    "narrative_credibility": float,
    "reasoning": "string"
}}
"""
