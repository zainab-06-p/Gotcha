"""
Unit tests for Candidate Normalization Module.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.normalizer import (
    _is_sentinel,
    _normalize_sentinels,
    _normalize_skills,
    normalize_candidate,
    compute_data_confidence,
)


class TestNormalizer(unittest.TestCase):
    def test_is_sentinel(self):
        self.assertTrue(_is_sentinel(-1))
        self.assertTrue(_is_sentinel("-1"))
        self.assertFalse(_is_sentinel(None))
        self.assertFalse(_is_sentinel(5))
        self.assertFalse(_is_sentinel("python"))

    def test_normalize_sentinels(self):
        signals = {
            "github_activity_score": -1,
            "profile_completeness_score": 85,
            "recruiter_response_rate": -1,
        }
        normalized = _normalize_sentinels(signals)
        self.assertIsNone(normalized["github_activity_score"])
        self.assertIsNone(normalized["recruiter_response_rate"])
        self.assertEqual(normalized["profile_completeness_score"], 85)

    def test_normalize_skills(self):
        synonyms = {"py": "python", "js": "javascript"}
        skills = [
            {"name": "Py", "proficiency": "expert", "endorsements": 5},
            {"name": "JS", "proficiency": "beginner", "endorsements": 0},
            {"name": "Docker", "proficiency": "intermediate", "endorsements": 2},
        ]
        normalized = _normalize_skills(skills, synonyms)
        self.assertEqual(normalized[0]["canonical_name"], "python")
        self.assertEqual(normalized[1]["canonical_name"], "javascript")
        self.assertEqual(normalized[2]["canonical_name"], "docker")

    def test_data_confidence(self):
        # Empty signals should yield 0.0
        self.assertEqual(compute_data_confidence({}), 0.0)

        candidate = {
            "redrob_signals": {
                "profile_completeness_score": 100,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
                "skill_assessment_scores": {"Python": 90},
            }
        }
        score = compute_data_confidence(candidate)
        # 0.40 * 1.0 (PCS) + 0.15 (email) + 0.15 (phone) + 0.10 (linkedin) + 0.20 (assessments) = 1.0
        self.assertAlmostEqual(score, 1.0)


    def test_robustness_invalid_types(self):
        # normalize_candidate should handle None, non-dict candidate inputs
        self.assertEqual(normalize_candidate(None), {})
        self.assertEqual(normalize_candidate([]), {})
        
        # compute_data_confidence should handle None, non-dict candidate inputs
        self.assertEqual(compute_data_confidence(None), 0.0)
        self.assertEqual(compute_data_confidence([]), 0.0)
        self.assertEqual(compute_data_confidence({"redrob_signals": "not_a_dict"}), 0.0)
        
        # _normalize_sentinels should handle non-dict input
        self.assertEqual(_normalize_sentinels("not_a_dict"), {})
        
        # _normalize_skills should handle non-list input
        self.assertEqual(_normalize_skills(None, {}), [])
        self.assertEqual(_normalize_skills("not_a_list", {}), [])
        
        # profile_completeness_score with invalid float value
        candidate_bad_pcs = {
            "redrob_signals": {
                "profile_completeness_score": "extremely_complete",
                "verified_email": True,
            }
        }
        # Should catch TypeError/ValueError and compute 0.15 (just email verified)
        self.assertAlmostEqual(compute_data_confidence(candidate_bad_pcs), 0.15)


if __name__ == "__main__":
    unittest.main()
