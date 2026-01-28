import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# This is a bit of a hack to make sure the script can be imported
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from extractTextFromPDF import OldRussianOCR, DocumentProcessor


class TestOldRussianOCR(unittest.TestCase):
    def setUp(self):
        with patch("extractTextFromPDF.load_config", return_value={}):
            self.ocr = OldRussianOCR(engine="tesseract")

    def test_postprocess_old_russian_text(self):
        text = "Съѣдобный, Ѳедоръ, миръ."
        processed_text = self.ocr.postprocess_old_russian_text(text)
        self.assertEqual(processed_text, "Съедобный, Федоръ, миръ.")


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_ocr = MagicMock()
        self.processor = DocumentProcessor(self.mock_ocr)

    @patch("extractTextFromPDF.fitz.open")
    def test_process_pdf_with_text(self, mock_fitz_open):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is some text from a PDF."
        mock_doc = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        self.mock_ocr.postprocess_old_russian_text.side_effect = (
            lambda x: x
        )  # No changes

        results = self.processor.process_pdf("dummy.pdf")
        self.assertEqual(results, {"page_1": "This is some text from a PDF."})
        self.mock_ocr.ocr_image.assert_not_called()

    @patch("extractTextFromPDF.fitz.open")
    @patch("extractTextFromPDF.np.frombuffer")
    @patch("extractTextFromPDF.cv2.cvtColor")
    def test_process_pdf_with_ocr(
        self, mock_cvt_color, mock_frombuffer, mock_fitz_open
    ):
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
