"""
Unit tests for Honeypot Detection Module.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.honeypot import _extract_keywords, detect_honeypot


class TestHoneypot(unittest.TestCase):
    def test_extract_keywords(self):
        text = "Lately I have been coding in Python and building Django applications."
        keywords = _extract_keywords(text)
        
        # Stop words like "have", "been", "in", "and" should be removed
        self.assertIn("coding", keywords)
        self.assertIn("python", keywords)
        self.assertIn("django", keywords)
        self.assertNotIn("have", keywords)
        self.assertNotIn("and", keywords)

    def test_detect_honeypot_title_mismatch(self):
        # Mismatched title and description
        candidate = {
            "profile": {
                "current_title": "Lead Blockchain Developer",
                "summary": "Experienced engineer looking for backend roles.",
            },
            "career_history": [
                {
                    "title": "Graphic Designer",
                    "description": "Designed logo assets, marketing brochures, and visual brand guidelines using Figma and Illustrator."
                }
            ]
        }
        
        is_hp, reason = detect_honeypot(candidate)
        self.assertTrue(is_hp)
        self.assertIn("title-desc mismatch", reason)

    def test_detect_honeypot_boilerplate_summary(self):
        # Summary contains boilerplate template
        candidate = {
            "profile": {
                "current_title": "Backend Engineer",
                "summary": "Lately I've been curious about how AI tools could augment my work...",
            },
            "career_history": [
                {
                    "title": "Backend Developer",
                    "description": "Coded python APIs and queries in Postgres database."
                }
            ]
        }
        
        is_hp, reason = detect_honeypot(candidate)
        self.assertTrue(is_hp)
        self.assertIn("boilerplate summary", reason)

    def test_legit_candidate_no_honeypot(self):
        # Legit candidate
        candidate = {
            "profile": {
                "current_title": "Python Developer",
                "summary": "Backend developer specializing in building APIs with Python and FastAPI.",
            },
            "career_history": [
                {
                    "title": "Software Engineer",
                    "description": "Developed server-side code using Python, FastAPI, and Postgres database queries."
                }
            ]
        }
        
        is_hp, reason = detect_honeypot(candidate)
        self.assertFalse(is_hp)
        self.assertEqual(reason, "")


    def test_robustness_invalid_structures(self):
        # Non-dict candidate inputs
        is_hp, reason = detect_honeypot(None)
        self.assertFalse(is_hp)
        self.assertEqual(reason, "Invalid candidate structure")

        is_hp, reason = detect_honeypot([])
        self.assertFalse(is_hp)
        self.assertEqual(reason, "Invalid candidate structure")

        # Candidate with profile as a non-dict
        candidate_bad_profile = {
            "profile": "not_a_dict",
            "career_history": []
        }
        is_hp, reason = detect_honeypot(candidate_bad_profile)
        # Should flag "no career descriptions to verify title" since profile has no current_title (empty)
        # and career_history is empty. Wait, if title_keywords is empty and desc_keywords is empty,
        # it won't check mismatch, but it has no title_keywords so overlap check is skipped.
        # Let's see:
        self.assertFalse(is_hp)

        # Candidate with career_history containing non-dict items
        candidate_bad_career = {
            "profile": {
                "current_title": "Python Developer",
            },
            "career_history": ["not_a_dict"]
        }
        is_hp, reason = detect_honeypot(candidate_bad_career)
        self.assertTrue(is_hp)
        self.assertIn("no career descriptions to verify title", reason)


if __name__ == "__main__":
    unittest.main()
