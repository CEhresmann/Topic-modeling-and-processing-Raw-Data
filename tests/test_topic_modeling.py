import unittest
from unittest.mock import mock_open, patch
import pandas as pd
from pymorphy2 import MorphAnalyzer
import yaml
import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.topic_modeling import (
    load_config,
    load_data,
    preprocess_text,
    get_dominant_topic,
)


class TestTopicModelingScript(unittest.TestCase):
    def test_load_config(self):
        m = mock_open(read_data="key: value")
        with patch("builtins.open", m):
            config = load_config("fake_path.yaml")
        self.assertEqual(config, {"key": "value"})

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_df = pd.DataFrame({"text": ["some text", "more text"]})
        mock_read_csv.return_value = mock_df
        df = load_data("fake_path.csv", "text")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_load_data_file_not_found(self, mock_read_csv):
        df = load_data("non_existent_path.csv", "text")
        self.assertIsNone(df)

    def test_preprocess_text(self):
        parser = MorphAnalyzer()
        text = "Тематическое моделирование - это интересно."
        filter_words = ["-", "это"]
        processed_text = preprocess_text(text, filter_words, parser)
        self.assertEqual(processed_text, ["тематический", "моделирование", "интересно"])

    def test_get_dominant_topic(self):
        class MockLdaModel:
            def __init__(self):
                self.id2word = {0: "a", 1: "b"}

            def get_document_topics(self, bow):
                return [(0, 0.9), (1, 0.1)]

            def doc2bow(self, text):
                return text  # Dummy implementation

        lda_model = MockLdaModel()
        processed_text = ["a", "a", "b"]
        dominant_topic = get_dominant_topic(processed_text, lda_model)
        self.assertEqual(dominant_topic, [0, 0.9])


if __name__ == "__main__":
    unittest.main()
