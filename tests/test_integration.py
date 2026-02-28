"""
Integration tests for the topic modeling pipeline.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import pandas as pd
import yaml


class TestIntegration(unittest.TestCase):
    """Integration tests for the topic modeling pipeline."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "data.csv")
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.output_html_path = os.path.join(self.test_dir, "lda.html")
        self.stop_words_path = os.path.join(self.test_dir, "stops.txt")

        # Create dummy data
        df = pd.DataFrame(
            {
                "text": [
                    "Тематическое моделирование — это хорошо",
                    "Латентное размещение Дирихле - это интересно",
                ]
                * 20,
                "category": ["ML", "NLP"] * 20,
            }
        )
        df.to_csv(self.data_path, index=False)

        # Create dummy stop words
        with open(self.stop_words_path, "w", encoding="utf-8") as f:
            f.write("это\n")

        # Create dummy config
        config = {
            "data": {
                "file_path": self.data_path,
                "text_column": "text",
                "category_column": "category",
            },
            "preprocessing": {
                "stop_words_path": self.stop_words_path,
                "punctuation": "-",
            },
            "model": {"num_topics": 2, "passes": 1, "random_state": 42},
            "evaluation": {
                "coherence_measure": "c_v",
                "max_topics": 4,
                "start_topics": 2,
                "step_topics": 1,
            },
            "visualization": {"output_html_path": self.output_html_path},
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    def tearDown(self):
        """Tear down the test environment."""
        shutil.rmtree(self.test_dir)

    def test_pipeline(self):
        """Test the full topic modeling pipeline."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "src", "topic_modeling.py")

        result = subprocess.run(
            ["python", script_path, "--config", self.config_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

        self.assertTrue(os.path.exists(self.output_html_path))


if __name__ == "__main__":
    unittest.main()
