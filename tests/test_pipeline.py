"""
Unit and Integration Tests for Multilingual Healthcare Misinformation RAG
Runs completely with Python's built-in unittest module.
"""

import os
import sys
import json
import csv
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_text
from src.language_detector import detect_language
from src.health_classifier import is_health_related
from src.verdict_engine import assess_verdict_and_risk, generate_explanation


class TestPreprocessing(unittest.TestCase):
    def test_lowercase_and_whitespace(self):
        cleaned = preprocess_text("  TURMERIC  Cures Cancer  ")
        self.assertEqual(cleaned, "turmeric cures cancer")

    def test_url_removal(self):
        cleaned = preprocess_text("Check https://example.com/cure for cancer info")
        self.assertEqual(cleaned, "check for cancer info")

    def test_devanagari_hindi_intact(self):
        hindi_sample = "हल्दी दूध कैंसर ठीक करता है"
        cleaned_hindi = preprocess_text(hindi_sample)
        self.assertIn("हल्दी", cleaned_hindi)
        self.assertIn("कैंसर", cleaned_hindi)

    def test_hinglish_normalization(self):
        hinglish_sample = "Haldi doodh cancer ko thik karta h"
        cleaned = preprocess_text(hinglish_sample)
        self.assertIn("theek", cleaned)
        self.assertIn("hai", cleaned)


class TestLanguageDetection(unittest.TestCase):
    def test_hindi_detection(self):
        self.assertEqual(detect_language("हल्दी दूध कैंसर ठीक करता है"), "Hindi")

    def test_hinglish_detection(self):
        self.assertEqual(detect_language("Haldi doodh cancer ko theek karta hai"), "Hinglish")
        self.assertEqual(detect_language("Vaccines autism cause karte hain"), "Hinglish")
        self.assertEqual(detect_language("Antibiotics cold ko cure karte hain"), "Hinglish")

    def test_english_detection(self):
        self.assertEqual(detect_language("Smoking causes lung cancer"), "English")
        self.assertEqual(detect_language("India won the cricket match"), "English")


class TestHealthClassification(unittest.TestCase):
    def test_health_claims(self):
        self.assertTrue(is_health_related("Turmeric cures cancer"))
        self.assertTrue(is_health_related("Haldi doodh cancer ko theek karta hai"))
        self.assertTrue(is_health_related("Vaccines cause autism"))
        self.assertTrue(is_health_related("Smoking causes lung cancer"))
        self.assertTrue(is_health_related("Antibiotics cold ko cure karte hain"))

    def test_non_health_claims(self):
        self.assertFalse(is_health_related("India won the cricket match"))
        self.assertFalse(is_health_related("What is the weather in Mumbai today?"))
        self.assertFalse(is_health_related("Stock market rose 500 points"))


class TestDatasetIntegrity(unittest.TestCase):
    def test_csv_file(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "evidence_dataset.csv")
        self.assertTrue(os.path.exists(csv_path), "Dataset CSV file must exist.")

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames)
            expected_fields = {"id", "claim_topic", "evidence_text", "stance", "source", "url"}
            self.assertTrue(expected_fields.issubset(fieldnames), f"Missing fields: {expected_fields - fieldnames}")

            rows = list(reader)
            self.assertGreaterEqual(len(rows), 20, "Should have at least 20 records.")

            for r in rows:
                self.assertIn(r["stance"].strip().lower(), {"supports", "refutes", "uncertain"})
                self.assertTrue(r["url"].startswith("http"), f"Invalid URL: {r['url']}")
                self.assertTrue(len(r["evidence_text"].strip()) > 10)


class TestVerdictAndRiskLogic(unittest.TestCase):
    def test_refuted_and_high_risk(self):
        mock_evidence = [{
            "rank": 1,
            "evidence_text": "There is no reliable evidence that turmeric cures cancer.",
            "stance": "refutes",
            "source": "Cancer Research UK",
            "url": "https://example.com",
            "similarity_score": 0.75
        }]
        res = assess_verdict_and_risk("Haldi doodh cancer ko theek karta hai", mock_evidence)
        self.assertEqual(res["verdict"], "Refuted")
        self.assertEqual(res["risk"], "High")

    def test_dangerous_medical_advice_elevation(self):
        mock_evidence = [{
            "rank": 1,
            "evidence_text": "Discontinuing cancer medication is dangerous.",
            "stance": "refutes",
            "source": "WHO",
            "url": "https://example.com",
            "similarity_score": 0.70
        }]
        res = assess_verdict_and_risk("stop chemo and use herbal tea", mock_evidence)
        self.assertEqual(res["verdict"], "Refuted")
        self.assertEqual(res["risk"], "Very High")

    def test_supported_and_low_risk(self):
        mock_evidence = [{
            "rank": 1,
            "evidence_text": "Cigarette smoking is the leading cause of lung cancer.",
            "stance": "supports",
            "source": "CDC",
            "url": "https://example.com",
            "similarity_score": 0.82
        }]
        res = assess_verdict_and_risk("Smoking causes lung cancer", mock_evidence)
        self.assertEqual(res["verdict"], "Supported")
        self.assertEqual(res["risk"], "Low")

    def test_low_similarity_no_evidence(self):
        mock_evidence = [{
            "rank": 1,
            "evidence_text": "Irrelevant text.",
            "stance": "supports",
            "source": "WHO",
            "url": "https://example.com",
            "similarity_score": 0.20
        }]
        res = assess_verdict_and_risk("random query", mock_evidence)
        self.assertEqual(res["verdict"], "No relevant evidence")


class TestColabNotebookStructure(unittest.TestCase):
    def test_notebook_json(self):
        nb_path = os.path.join(os.path.dirname(__file__), "..", "Multilingual_Healthcare_Misinformation_RAG.ipynb")
        self.assertTrue(os.path.exists(nb_path), "Notebook file must exist.")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreaterEqual(len(cells), 20)

        # Check that Step 1 through Step 10 exist
        all_sources = "".join("".join(c["source"]) for c in cells)
        for step_num in range(1, 11):
            self.assertIn(f"Step {step_num}", all_sources)


if __name__ == "__main__":
    unittest.main()
