import unittest
import os
import csv
from pathlib import Path
import tempfile
import shutil

# This is a bit of a hack to make sure the script can be imported
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from aggregateResults import aggregate_to_csv


class TestAggregateResults(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_aggregate_to_csv(self):
        # Create dummy files
        Path(self.test_dir, "file1_распознано.txt").write_text(
            "content1", encoding="utf-8"
        )
        Path(self.test_dir, "file2_распознано.txt").write_text(
            "content2", encoding="utf-8"
        )

        output_csv = Path(self.test_dir) / "output.csv"
        aggregate_to_csv(self.test_dir, str(output_csv))

        self.assertTrue(output_csv.exists())

        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(
                header, ["id", "author", "title", "description", "date", "size", "text"]
            )

            rows = list(reader)
            self.assertEqual(len(rows), 2)

            # Check content of rows, ignoring id
            row1 = rows[0]
            self.assertEqual(row1[2], "file1")  # title
            self.assertEqual(row1[6], "content1")  # text

            row2 = rows[1]
            self.assertEqual(row2[2], "file2")  # title
            self.assertEqual(row2[6], "content2")  # text


if __name__ == "__main__":
    unittest.main()
