"""
Tests for the LLM text cleaning stage.
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from jupyterproject.llm_text_cleaning import (
    CleanResult,
    LLMTextCleaner,
    chunk_text,
    clean_csv_with_llm,
)


class TestLLMTextCleaning(unittest.TestCase):
    """Tests for llm_text_cleaning module."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_chunk_text_with_overlap(self):
        """Chunking should split long text and preserve content coverage."""
        text = " ".join(["слово"] * 300)
        chunks = chunk_text(text, max_chunk_chars=200, overlap_chars=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))

    @patch.dict(os.environ, {}, clear=True)
    def test_clean_text_no_api_key_fallback(self):
        """When API key is missing, cleaner should keep the original text."""
        cleaner = LLMTextCleaner(
            {
                "llm_cleaning": {
                    "enabled": True,
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                }
            }
        )
        result = cleaner.clean_text("Текст без изменений.")
        self.assertEqual(result.cleaned_text, "Текст без изменений.")
        self.assertEqual(result.status, "skipped_no_api_key")

    def test_clean_text_skips_when_noise_low(self):
        """Cleaner should skip clean text in budget mode."""
        cleaner = LLMTextCleaner(
            {
                "llm_cleaning": {
                    "enabled": True,
                    "only_if_suspect": True,
                    "min_text_length_for_llm": 0,
                    "suspect_score_threshold": 0.2,
                }
            }
        )
        text = "Городская управа извещает граждан о заседании в понедельник."
        result = cleaner.clean_text(text)
        self.assertEqual(result.status, "skipped_clean_text")
        self.assertEqual(result.cleaned_text, text)

    @patch.object(LLMTextCleaner, "clean_text")
    def test_clean_csv_with_llm(self, mock_clean_text):
        """CSV cleaning should enrich output with llm metadata fields."""
        input_csv = Path(self.test_dir) / "input.csv"
        output_csv = Path(self.test_dir) / "output.csv"
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerow({"id": "1", "text": "Г0р0дская yпpaвa"})

        mock_clean_text.return_value = CleanResult(
            cleaned_text="Городская управа",
            status="cleaned",
            uncertain_spans=[],
            notes="ok",
            change_ratio=0.15,
        )
        config = {
            "data": {"text_column": "text"},
            "llm_cleaning": {"enabled": True, "replace_text_column": True},
        }
        clean_csv_with_llm(str(input_csv), str(output_csv), config)

        with open(output_csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "Городская управа")
        self.assertEqual(rows[0]["raw_text"], "Г0р0дская yпpaвa")
        self.assertEqual(rows[0]["llm_clean_status"], "cleaned")


if __name__ == "__main__":
    unittest.main()
