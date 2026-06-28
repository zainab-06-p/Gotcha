"""
Gotcha — Reproduce Command Entry Point (Redrob Senior AI Engineer JD Edition)
Usage: python scripts/rank.py --candidates <path_to_candidates> --output <output_path>

The JD is hardcoded (src/jd_redrob.py) — no --jd file needed for the challenge role.
All 8 hard disqualifiers, YoE fit scoring, and nice-to-have bonuses are applied.
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import run_pipeline
from src.config import JOB_DESCRIPTION_DOCX, CANDIDATES_JSONL, SAMPLE_CANDIDATES_JSON, OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Gotcha Reproducible Ranking Entry Point")
    parser.add_argument(
        "--jd",
        type=str,
        default=str(JOB_DESCRIPTION_DOCX),
        help="Path to the Job Description (.docx) file"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to the candidate profile dataset (.json or .jsonl)"
    )
    parser.add_argument(
        "--output", "--out",
        type=str,
        default=str(OUTPUTS_DIR / "team_infinity_and_beyond.csv"),
        help="Path where the final ranked CSV file should be written"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of candidates to process (useful for testing)"
    )

    args = parser.parse_args()

    jd_path = Path(args.jd)
    
    # Determine candidates path
    if args.candidates:
        cands_path = Path(args.candidates)
    else:
        # Check if full jsonl exists, if not use sample
        cands_path = Path(CANDIDATES_JSONL)
        if not cands_path.exists():
            cands_path = Path(SAMPLE_CANDIDATES_JSON)
            print(f"candidates.jsonl not found. Falling back to {SAMPLE_CANDIDATES_JSON}")

    output_path = Path(args.output)
    is_jsonl = cands_path.suffix.lower() == ".jsonl"

    print("====================================================================")
    print("Gotcha Candidate Discovery and Ranking Engine")
    print("====================================================================")
    print(f"JD Path:         {jd_path}")
    print(f"Candidates Path: {cands_path}")
    print(f"Output Path:     {output_path}")
    print("--------------------------------------------------------------------")

    # Run the pipeline
    run_pipeline(
        jd_path=jd_path,
        candidates_path=cands_path,
        output_path=output_path,
        is_jsonl=is_jsonl,
        limit=args.limit,
    )

    print("\n--------------------------------------------------------------------")
    print("Pipeline execution completed successfully!")
    print(f"Output written to: {output_path.resolve()}")
    print("====================================================================")


if __name__ == "__main__":
    main()
