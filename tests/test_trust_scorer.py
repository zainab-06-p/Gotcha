"""
Unit tests for Skill Trust Scoring Module.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.trust_scorer import score_skill_trust, score_candidate_trust
from src.config import PROFICIENCY_SCORES


class TestTrustScorer(unittest.TestCase):
    def test_score_skill_trust_with_assessment(self):
        # Skill with verified assessment score
        skill = {
            "name": "Python",
            "proficiency": "expert",
            "duration_months": 24,
            "endorsements": 10,
        }
        assessments = {"Python": 90.0}
        
        st = score_skill_trust(skill, assessments)
        self.assertEqual(st.skill_name, "Python")
        self.assertEqual(st.assessment_score, 90.0)
        self.assertTrue(0.0 <= st.trust_score <= 1.0)

    def test_score_skill_trust_no_assessment(self):
        # Skill without assessment score
        skill = {
            "name": "Docker",
            "proficiency": "intermediate",
            "duration_months": 12,
            "endorsements": 2,
        }
        st = score_skill_trust(skill, None)
        self.assertIsNone(st.assessment_score)
        self.assertTrue(0.0 <= st.trust_score <= 1.0)

    def test_score_candidate_trust(self):
        candidate = {
            "skills": [
                {"name": "python", "proficiency": "advanced", "duration_months": 36, "endorsements": 5},
                {"name": "java", "proficiency": "beginner", "duration_months": 6, "endorsements": 0},
            ],
            "redrob_signals": {
                "skill_assessment_scores": {"python": 85.0}
            }
        }
        
        # When JD needs python and java
        trust_score = score_candidate_trust(candidate, ["python", "java"])
        self.assertTrue(0.0 <= trust_score <= 1.0)

        # When JD needs only java
        trust_only_java = score_candidate_trust(candidate, ["java"])
        # When JD needs only python (which has assessment, should be higher trust)
        trust_only_py = score_candidate_trust(candidate, ["python"])
        
        self.assertGreater(trust_only_py, trust_only_java)


    def test_robustness_invalid_values(self):
        # Non-dict or None skill input should not crash
        self.assertEqual(score_skill_trust(None, None).trust_score, 0.0)
        self.assertEqual(score_skill_trust("not_a_dict", None).trust_score, 0.0)

        # Invalid proficiency should fallback to beginner score
        skill_bad_prof = {
            "name": "Python",
            "proficiency": "super-ninja",
            "duration_months": "not_an_int",
            "endorsements": "not_an_int",
        }
        st = score_skill_trust(skill_bad_prof, None)
        self.assertEqual(st.proficiency, "super-ninja")
        self.assertEqual(st.duration_months, 0)
        self.assertEqual(st.endorsements, 0)
        self.assertTrue(0.0 <= st.trust_score <= 1.0)

        # score_candidate_trust invalid input types
        self.assertEqual(score_candidate_trust(None, ["python"]), 0.0)
        self.assertEqual(score_candidate_trust("not_a_dict", ["python"]), 0.0)
        self.assertEqual(score_candidate_trust({}, ["python"]), 0.0)
        self.assertEqual(score_candidate_trust({"skills": "not_a_list"}, ["python"]), 0.0)


if __name__ == "__main__":
    unittest.main()
