"""
Tests for the main script.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from jupyterproject.main import main, process_files


class TestMain(unittest.TestCase):
    """Tests for the main script functionality."""

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("jupyterproject.main.process_file")
    def test_process_files(self, mock_process_file):
        """Test the recursive file processing."""
        # Create dummy files
        Path(self.test_dir, "file1.pdf").touch()
        Path(self.test_dir, "file2.djvu").touch()

        process_files(self.test_dir)

        # Check that process_file was called for each file
        calls = [
            call(os.path.join(self.test_dir, "file1.pdf"), config_path="config.yaml"),
            call(os.path.join(self.test_dir, "file2.djvu"), config_path="config.yaml"),
        ]
        mock_process_file.assert_has_calls(calls, any_order=True)

    @patch("jupyterproject.main.process_files")
    def test_main_process_command(self, mock_process_files):
        """Test the 'process' command."""
        with patch.object(
            sys, "argv", ["main.py", "process", "--input-dir", self.test_dir, "--config", "c.yaml"]
        ):
            main()
        mock_process_files.assert_called_once_with(self.test_dir, "c.yaml")

    @patch("jupyterproject.main.aggregate_to_csv")
    def test_main_aggregate_command(self, mock_aggregate_to_csv):
        """Test the 'aggregate' command."""
        with patch.object(sys, "argv", ["main.py", "aggregate", "--input-dir", self.test_dir]):
            main()
        output_csv = os.path.join(self.test_dir, "aggregated_results.csv")
        mock_aggregate_to_csv.assert_called_once_with(self.test_dir, output_csv)

    @patch("jupyterproject.main.run_topic_modeling")
    @patch("jupyterproject.main.load_config")
    def test_main_topic_model_command(self, mock_load_config, mock_run_topic_modeling):
        """Test the 'topic-model' command."""
        mock_load_config.return_value = {}
        with patch.object(sys, "argv", ["main.py", "topic-model", "--config", "fake_config.yaml"]):
            main()
        mock_load_config.assert_called_once_with("fake_config.yaml")
        mock_run_topic_modeling.assert_called_once_with({})

    @patch("jupyterproject.main.clean_csv_with_llm")
    @patch("jupyterproject.main.load_config")
    def test_main_clean_text_command(self, mock_load_config, mock_clean_csv_with_llm):
        """Test the 'clean-text' command."""
        mock_load_config.return_value = {"data": {"text_column": "text"}}
        input_csv = os.path.join(self.test_dir, "aggregated_results.csv")
        with patch.object(
            sys,
            "argv",
            ["main.py", "clean-text", "--input-csv", input_csv, "--config", "fake_config.yaml"],
        ):
            main()
        expected_output = os.path.join(self.test_dir, "aggregated_results_cleaned.csv")
        mock_load_config.assert_called_once_with("fake_config.yaml")
        mock_clean_csv_with_llm.assert_called_once_with(
            input_csv, expected_output, {"data": {"text_column": "text"}}
        )

    @patch("jupyterproject.main.process_files")
    @patch("jupyterproject.main.aggregate_to_csv")
    @patch("jupyterproject.main.clean_csv_with_llm")
    @patch("jupyterproject.main.run_topic_modeling")
    @patch("jupyterproject.main.load_config")
    def test_main_full_pipeline_command(
        self,
        mock_load_config,
        mock_run_topic_modeling,
        mock_clean_csv_with_llm,
        mock_aggregate_to_csv,
        mock_process_files,
    ):
        """Test the 'full-pipeline' command."""
        config = {"data": {"file_path": "old_path.csv"}, "llm_cleaning": {"enabled": True}}
        mock_load_config.return_value = config
        mock_clean_csv_with_llm.return_value = os.path.join(
            self.test_dir, "aggregated_results_cleaned.csv"
        )

        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "full-pipeline",
                "--input-dir",
                self.test_dir,
                "--config",
                "fake_config.yaml",
            ],
        ):
            main()

        mock_process_files.assert_called_once_with(self.test_dir, "fake_config.yaml")
        output_csv = os.path.join(self.test_dir, "aggregated_results.csv")
        mock_aggregate_to_csv.assert_called_once_with(self.test_dir, output_csv)
        mock_load_config.assert_called_once_with("fake_config.yaml")
        mock_clean_csv_with_llm.assert_called_once_with(
            output_csv, os.path.join(self.test_dir, "aggregated_results_cleaned.csv"), config
        )

        # Check that the config was updated with the new path
        updated_config = {
            "data": {"file_path": os.path.join(self.test_dir, "aggregated_results_cleaned.csv")},
            "llm_cleaning": {"enabled": True},
        }
        mock_run_topic_modeling.assert_called_once_with(updated_config)


if __name__ == "__main__":
    unittest.main()
