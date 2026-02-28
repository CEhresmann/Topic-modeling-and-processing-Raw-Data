"""
Tests for the topic_modeling script.
"""

import os
import sys
import unittest
from unittest.mock import mock_open, patch

import pandas as pd
from pymorphy2 import MorphAnalyzer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from jupyterproject.topic_modeling import (
    get_dominant_topic,
    load_config,
    load_data,
    preprocess_text,
)


class TestTopicModelingScript(unittest.TestCase):
    """Tests for the topic modeling script."""

    def test_load_config(self):
        """Test loading the YAML configuration."""
        m = mock_open(read_data="key: value")
        with patch("builtins.open", m):
            config = load_config("fake_path.yaml")
        self.assertEqual(config, {"key": "value"})

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test loading data from a CSV file."""
        mock_df = pd.DataFrame({"text": ["some text", "more text"]})
        mock_read_csv.return_value = mock_df
        df = load_data("fake_path.csv", "text")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test handling of a non-existent data file."""
        df = load_data("non_existent_path.csv", "text")
        self.assertIsNone(df)
        mock_read_csv.assert_called_once_with("non_existent_path.csv")

    def test_preprocess_text(self):
        """Test the text preprocessing function."""
        parser = MorphAnalyzer()
        text = "Тематическое моделирование - это интересно."
        filter_words = ["-", "это"]
        processed_text = preprocess_text(text, filter_words, parser)
        self.assertEqual(processed_text, ["тематический", "моделирование", "интересно"])

    def test_get_dominant_topic(self):
        """Test the dominant topic identification."""

        class MockLdaModel:
            """Mock LDA model for testing."""

            def __init__(self):
                """Initialize the mock model."""
                self.id2word = {0: "a", 1: "b"}

            def get_document_topics(self, _):
                """Return dummy document topics."""
                return [(0, 0.9), (1, 0.1)]

            def doc2bow(self, text):
                """Return a dummy bag-of-words representation."""
                return text  # Dummy implementation

        lda_model = MockLdaModel()
        processed_text = ["a", "a", "b"]
        dominant_topic = get_dominant_topic(processed_text, lda_model)
        self.assertEqual(dominant_topic, [0, 0.9])


if __name__ == "__main__":
    unittest.main()
