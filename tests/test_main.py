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

from main import main, process_files

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMain(unittest.TestCase):
    """Tests for the main script functionality."""

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("main.process_file")
    def test_process_files(self, mock_process_file):
        """Test the recursive file processing."""
        # Create dummy files
        Path(self.test_dir, "file1.pdf").touch()
        Path(self.test_dir, "file2.djvu").touch()

        process_files(self.test_dir)

        # Check that process_file was called for each file
        calls = [
            call(os.path.join(self.test_dir, "file1.pdf")),
            call(os.path.join(self.test_dir, "file2.djvu")),
        ]
        mock_process_file.assert_has_calls(calls, any_order=True)

    @patch("main.process_files")
    def test_main_process_command(self, mock_process_files):
        """Test the 'process' command."""
        with patch.object(sys, "argv", ["main.py", "process", "--input-dir", self.test_dir]):
            main()
        mock_process_files.assert_called_once_with(self.test_dir)

    @patch("main.aggregate_to_csv")
    def test_main_aggregate_command(self, mock_aggregate_to_csv):
        """Test the 'aggregate' command."""
        with patch.object(sys, "argv", ["main.py", "aggregate", "--input-dir", self.test_dir]):
            main()
        output_csv = os.path.join(self.test_dir, "aggregated_results.csv")
        mock_aggregate_to_csv.assert_called_once_with(self.test_dir, output_csv)

    @patch("main.run_topic_modeling")
    @patch("main.load_config")
    def test_main_topic_model_command(self, mock_load_config, mock_run_topic_modeling):
        """Test the 'topic-model' command."""
        mock_load_config.return_value = {}
        with patch.object(sys, "argv", ["main.py", "topic-model", "--config", "fake_config.yaml"]):
            main()
        mock_load_config.assert_called_once_with("fake_config.yaml")
        mock_run_topic_modeling.assert_called_once_with({})

    @patch("main.process_files")
    @patch("main.aggregate_to_csv")
    @patch("main.run_topic_modeling")
    @patch("main.load_config")
    def test_main_full_pipeline_command(
        self, mock_load_config, mock_run_topic_modeling, mock_aggregate_to_csv, mock_process_files
    ):
        """Test the 'full-pipeline' command."""
        config = {"data": {"file_path": "old_path.csv"}}
        mock_load_config.return_value = config

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

        mock_process_files.assert_called_once_with(self.test_dir)
        output_csv = os.path.join(self.test_dir, "aggregated_results.csv")
        mock_aggregate_to_csv.assert_called_once_with(self.test_dir, output_csv)
        mock_load_config.assert_called_once_with("fake_config.yaml")

        # Check that the config was updated with the new path
        updated_config = {"data": {"file_path": output_csv}}
        mock_run_topic_modeling.assert_called_once_with(updated_config)


if __name__ == "__main__":
    unittest.main()
