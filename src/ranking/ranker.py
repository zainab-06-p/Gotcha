"""
Gotcha — Ranker Module
Sorts candidates by final score (descending), breaks ties by candidate_id (ascending),
assigns ranks (1-indexed), and writes the submission CSV.
"""

import logging
import csv
from pathlib import Path
from typing import Any

from src.config import CandidateResult, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    """Sort candidates by score descending and break ties by candidate_id ascending.

    Args:
        results: Unsorted list of CandidateResult objects.

    Returns:
        Sorted list of CandidateResult objects.
    """
    # Round final_score to 4 decimal places first so the sort key matches the
    # displayed CSV value, enabling correct tie-breaking on candidate_id.
    for res in results:
        res.final_score = round(res.final_score, 4)

    # Sort by: final_score descending (primary), then candidate_id ascending (secondary)
    sorted_results = sorted(results, key=lambda x: (-x.final_score, x.candidate_id))
    return sorted_results


def write_submission_csv(
    ranked_results: list[CandidateResult],
    output_path: Path = None,
    top_n: int = 100,
) -> Path:
    """Write the ranked candidate results to the submission CSV file.

    Format required:
        candidate_id, rank, score, reasoning

    Args:
        ranked_results: Sorted list of CandidateResult.
        output_path: Path to output CSV file. Defaults to outputs/submission.csv.
        top_n: Number of candidates to write. Defaults to 100.

    Returns:
        Path to the written CSV file.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "submission.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only top_n
    final_list = ranked_results[:top_n]

    # Write CSV
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            for idx, res in enumerate(final_list, 1):
                # Clean reasoning for single-line CSV cell (no newlines)
                clean_reason = str(res.reasoning).replace("\n", " ").replace("\r", " ").strip()
                # Cap reasoning size if too long (optional, let's keep it concise but descriptive)
                if not clean_reason:
                    clean_reason = "Structured profile match score based on skills, trust and career relevance."
                writer.writerow([
                    res.candidate_id,
                    idx,
                    round(res.final_score, 4),
                    clean_reason
                ])
                
        logger.info("Successfully wrote %d ranked candidates to %s", len(final_list), output_path)
    except Exception as e:
        logger.error("Error writing submission CSV: %s", e)
        raise

    return output_path
