"""
Tests for the extract_text_from_pdf script.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from extract_text_from_pdf import DocumentProcessor, OldRussianOCR


class TestOldRussianOCR(unittest.TestCase):
    """Tests for the OldRussianOCR class."""

    def setUp(self):
        """Set up the OCR test environment."""
        with patch("extract_text_from_pdf.load_config", return_value={}):
            self.ocr = OldRussianOCR(engine="tesseract")

    def test_postprocess_old_russian_text(self):
        """Test the postprocessing of old Russian text."""
        text = "Съѣдобный, Ѳедоръ, миръ."
        processed_text = self.ocr.postprocess_old_russian_text(text)
        self.assertEqual(processed_text, "Съедобный, Федоръ, миръ.")


class TestDocumentProcessor(unittest.TestCase):
    """Tests for the DocumentProcessor class."""

    def setUp(self):
        """Set up the document processor test environment."""
        self.mock_ocr = MagicMock()
        self.processor = DocumentProcessor(self.mock_ocr)

    @patch("extract_text_from_pdf.fitz.open")
    def test_process_pdf_with_text(self, mock_fitz_open):
        """Test PDF processing with a text layer."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is some text from a PDF."
        mock_doc = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        self.mock_ocr.postprocess_old_russian_text.side_effect = lambda x: x  # No changes

        results = self.processor.process_pdf("dummy.pdf")
        self.assertEqual(results, {"page_1": "This is some text from a PDF."})
        self.mock_ocr.ocr_image.assert_not_called()

    @patch("extract_text_from_pdf.fitz.open")
    @patch("extract_text_from_pdf.np.frombuffer")
    @patch("extract_text_from_pdf.cv2.cvtColor")
    def test_process_pdf_with_ocr(self, mock_cvt_color, mock_frombuffer, mock_fitz_open):
        """Test PDF processing with OCR."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # No text layer
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"someimagedata"
        mock_pixmap.h = 10
        mock_pixmap.w = 10
        mock_pixmap.n = 3
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        mock_frombuffer.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cvt_color.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        self.mock_ocr.ocr_image.return_value = "ocr text"

        results = self.processor.process_pdf("dummy.pdf")
        self.assertEqual(results, {"page_1": "ocr text"})
        self.mock_ocr.ocr_image.assert_called_once()


if __name__ == "__main__":
    unittest.main()
