"""
This module provides functionality to extract text from PDF, DJVU, and image files.
"""

# pylint: disable=no-member
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import easyocr
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.error("Config file not found: %s", config_path)
        return {}
    except yaml.YAMLError as e:
        logging.error("Error parsing config file: %s", e)
        return {}


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _rotate_image(image: np.ndarray) -> np.ndarray:
    """Rotates an image to correct for skew."""
    try:
        coords = np.column_stack(np.where(image < 200))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(
                image,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    except cv2.error:
        pass  # Ignore rotation errors
    return image


class OldRussianOCR:
    """A class for recognizing pre-revolutionary Russian texts."""

    def __init__(
        self,
        engine: str = "tesseract",
        tesseract_path: str | None = None,
        config: dict | None = None,
    ):
        """
        Initializes the OCR system.

        Args:
            engine: The OCR engine to use ('tesseract' or 'easyocr').
            tesseract_path: The path to the tesseract.exe (for Windows).
        """
        self.config = config or {}
        self.ocr_config = self.config.get("ocr", {})
        self.easyocr_reader = None
        self.engine = engine
        if self.engine == "tesseract":
            tesseract_path = tesseract_path or self.config.get("tesseract", {}).get("path")
            if tesseract_path and os.name == "nt":
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.languages = self.ocr_config.get("languages", "rus+chu")
            self.tesseract_oem = int(self.ocr_config.get("tesseract_oem", 1))
            self.psm_candidates = self._load_psm_candidates()
            self.preprocess_variants = self._load_preprocess_variants()
            self.normalize_old_letters = bool(self.ocr_config.get("normalize_old_letters", True))
            self.save_debug_images = bool(self.ocr_config.get("save_debug_images", True))
            self.setup_tesseract()
        elif self.engine == "easyocr":
            self.easyocr_reader = easyocr.Reader(["ru", "en"])
            self.setup_easyocr()
        else:
            self.preprocess_variants = ["adaptive_gaussian"]
            self.normalize_old_letters = True
            self.save_debug_images = False

    def _load_psm_candidates(self) -> list[int]:
        raw_psm = self.ocr_config.get("psm_candidates", [4, 6, 11])
        result = []
        for value in raw_psm:
            try:
                psm = int(value)
                if 0 <= psm <= 13:
                    result.append(psm)
            except (TypeError, ValueError):
                continue
        return result or [4, 6, 11]

    def _load_preprocess_variants(self) -> list[str]:
        allowed = {"adaptive_gaussian", "adaptive_mean", "otsu"}
        raw_variants = self.ocr_config.get("preprocess_variants", ["adaptive_gaussian"])
        variants = [variant for variant in raw_variants if variant in allowed]
        return variants or ["adaptive_gaussian"]

    def setup_tesseract(self) -> None:
        """Configures the Tesseract OCR engine."""
        try:
            langs = pytesseract.get_languages(config="")
            logging.info("Available Tesseract languages: %s", langs)
            required_langs = self.languages.split("+")
            for lang in required_langs:
                if lang not in langs:
                    logging.warning("WARNING: Language '%s' not found in Tesseract.", lang)
        except pytesseract.TesseractNotFoundError as e:
            logging.error("Error: Tesseract not found. Make sure it is installed and in your PATH,")
            logging.error("or specify the path in OldRussianOCR(tesseract_path=...)")
            raise RuntimeError("Tesseract not found") from e
        except OSError as e:
            logging.error("Failed to get Tesseract languages: %s", e)

    @staticmethod
    def setup_easyocr() -> None:
        """Configures the EasyOCR engine."""
        try:
            logging.info("EasyOCR initialized.")
        except OSError as e:
            logging.error("Failed to initialize EasyOCR: %s", e)

    def preprocess_for_old_russian(
        self, image: np.ndarray, variant: str = "adaptive_gaussian"
    ) -> np.ndarray:
        """
        Special preprocessing for pre-revolutionary texts.

        Args:
            image: The input image.

        Returns:
            The processed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # type: ignore[call-overload]
        rotated = _rotate_image(norm_img)
        clip_limit = float(self.ocr_config.get("clahe_clip_limit", 2.0))
        tile_grid = int(self.ocr_config.get("clahe_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        enhanced_contrast = clahe.apply(rotated)

        # Optimized denoising: use bilateral filter (faster, preserves edges)
        # For very noisy images, can still use fastNlMeansDenoising with lower h
        denoise_h = int(self.ocr_config.get("denoise_h", 7))
        if denoise_h > 0:
            # bilateralFilter: d=9, sigmaColor=75, sigmaSpace=75 - good for documents
            # This is ~10x faster than fastNlMeansDenoising
            denoised = cv2.bilateralFilter(enhanced_contrast, 9, 75, 75)
        else:
            denoised = enhanced_contrast

        scale = float(self.ocr_config.get("upscale_factor", 1.4))
        new_width = int(denoised.shape[1] * scale)
        new_height = int(denoised.shape[0] * scale)
        resized = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        block_size = int(self.ocr_config.get("adaptive_block_size", 19))
        if block_size % 2 == 0:
            block_size += 1
        c_value = int(self.ocr_config.get("adaptive_c", 3))
        if variant == "adaptive_mean":
            return cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value
            )
        if variant == "otsu":
            _, threshold = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return threshold
        return cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value
        )

    def postprocess_old_russian_text(self, text: str) -> str:
        """
        Post-processes recognized pre-revolutionary text.

        Args:
            text: The recognized text.

        Returns:
            The corrected text.
        """
        if self.normalize_old_letters:
            replacements = {
                "Ѣ": "е",
                "ѣ": "е",
                "Ѳ": "ф",
                "ѳ": "ф",
                "Ѵ": "и",
                "ѵ": "и",
                "І": "и",
                "і": "и",
                "Ѫ": "у",
                "ѫ": "у",
                "Ѧ": "я",
                "ѧ": "я",
                "Ѯ": "кс",
                "ѯ": "кс",
                "Ѱ": "пс",
                "ѱ": "пс",
            }
            for old, new in replacements.items():
                text = text.replace(old, new.upper() if old.isupper() else new)
        text = text.replace(" | ", " ").replace(" |", " ").replace("| ", " ")
        return text

    # Pre-computed character sets for score_ocr_text (class-level for efficiency)
    _CYRILLIC_SET = frozenset(
        "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯяѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱ"
    )
    _WORD_CHARS = frozenset(
        "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯяѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱ-"
    )
    _VALID_PUNCT = frozenset(".,;:!?()\"'«»- ")

    @staticmethod
    def score_ocr_text(text: str) -> float:
        """Scores OCR output quality to select the best attempt. Optimized single-pass version."""
        cleaned = text.strip()
        if not cleaned:
            return float("-inf")

        total_len = len(cleaned)
        letter_count = 0
        word_count = 0
        bad_symbol_count = 0
        in_word = False

        cyrillic = OldRussianOCR._CYRILLIC_SET
        word_chars = OldRussianOCR._WORD_CHARS
        valid_punct = OldRussianOCR._VALID_PUNCT

        for char in cleaned:
            if char in cyrillic:
                letter_count += 1
                if not in_word:
                    word_count += 1
                    in_word = True
            elif char in word_chars:
                if not in_word:
                    word_count += 1
                    in_word = True
            elif char.isspace():
                in_word = False
            elif char not in valid_punct and not char.isdigit():
                bad_symbol_count += 1

        letter_ratio = letter_count / max(1, total_len)
        word_ratio = word_count / max(1, len(cleaned.split()))
        garbage_penalty = bad_symbol_count / max(1, total_len)

        return (2.5 * letter_ratio) + (1.5 * word_ratio) - (2.0 * garbage_penalty)

    def ocr_image_easyocr(self, image_input: str | np.ndarray) -> str:
        """
        Recognizes text from an image using EasyOCR.
        """
        if not self.easyocr_reader:
            logging.error("EasyOCR is not initialized.")
            return ""
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {image_input}")
            else:
                img = image_input
            processed = self.preprocess_for_old_russian(img, variant=self.preprocess_variants[0])
            raw_results = self.easyocr_reader.readtext(  # type: ignore[assignment]
                processed, detail=0, paragraph=True
            )
            text = " ".join(str(r) for r in raw_results)
            return self.postprocess_old_russian_text(text)
        except OSError as e:
            logging.error("EasyOCR error: %s", e)
            return ""

    def ocr_image(self, image_input: str | np.ndarray) -> str:
        """
        Recognizes text from an image.

        Args:
            image_input: The path to the image or a numpy array.

        Returns:
            The recognized text.
        """
        if self.engine == "easyocr":
            return self.ocr_image_easyocr(image_input)

        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {image_input}")
            else:
                img = image_input
            best_raw_text = ""
            best_score = float("-inf")
            for variant in self.preprocess_variants:
                processed = self.preprocess_for_old_russian(img, variant=variant)
                if self.save_debug_images:
                    debug_dir = Path("debug_ocr")
                    debug_dir.mkdir(exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix=f"_{variant}.png", delete=False, dir=debug_dir
                    ) as temp_file:
                        debug_path = temp_file.name
                    cv2.imwrite(debug_path, processed)
                    logging.info("Debug image saved: %s", debug_path)
                for psm in self.psm_candidates:
                    ocr_config = f"--oem {self.tesseract_oem} --psm {psm} -l {self.languages}"
                    raw_text = pytesseract.image_to_string(processed, config=ocr_config)
                    score = self.score_ocr_text(raw_text)
                    if score > best_score:
                        best_score = score
                        best_raw_text = raw_text
            return self.postprocess_old_russian_text(best_raw_text)
        except OSError as e:
            logging.error("OCR error: %s", e)
            return ""


class DocumentProcessor:
    """A processor for documents of various formats."""

    def __init__(self, ocr_engine: OldRussianOCR, config: dict | None = None):
        self.ocr = ocr_engine
        self.config = config or {}

    def process_pdf(self, pdf_path: str) -> dict:
        """Processes a PDF file using a combination of text extraction and OCR."""
        results = {}
        dpi = self.config.get("ocr", {}).get("dpi", 350)
        logging.info("Starting PDF processing...")
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):  # type: ignore[union-attr]
                logging.info("Processing page %d/%d...", page_num + 1, len(doc))
                text = page.get_text("text")
                if text and len(text.strip()) > 100:
                    logging.info("  -> Page %d: Text layer found.", page_num + 1)
                    results[f"page_{page_num + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    logging.info(
                        "  -> Page %d: Text layer empty or small. Running OCR.", page_num + 1
                    )
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    img_cv = cv2.cvtColor(
                        img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
                    )
                    results[f"page_{page_num + 1}"] = self.ocr.ocr_image(img_cv)
            doc.close()
        except OSError as e:
            logging.error("Critical error processing PDF: %s", e)
        return results

    def _process_pdf_sequential(self, pdf_path: str, dpi: int) -> dict:
        """Sequential PDF processing for small files."""
        results = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):  # type: ignore[union-attr]
                logging.info("Processing page %d/%d...", page_num + 1, len(doc))
                text = page.get_text("text")
                if text and len(text.strip()) > 100:
                    logging.info("  -> Page %d: Text layer found.", page_num + 1)
                    results[f"page_{page_num + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    logging.info(
                        "  -> Page %d: Text layer empty or small. Running OCR.", page_num + 1
                    )
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    img_cv = cv2.cvtColor(
                        img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
                    )
                    results[f"page_{page_num + 1}"] = self.ocr.ocr_image(img_cv)
            doc.close()
        except OSError as e:
            logging.error("Critical error processing PDF: %s", e)
        return results

    def process_djvu(self, djvu_path: str) -> dict:
        """
        Processes a DJVU file.
        """
        dpi = self.config.get("ocr", {}).get("dpi", 400)
        try:
            logging.info("Using PyMuPDF to process DJVU file...")
            return self._process_djvu_with_fitz(djvu_path, dpi)
        except OSError as e:
            logging.warning("PyMuPDF could not open DJVU file: %s", e)
            logging.info("Trying alternative conversion method...")
            return self._process_djvu_fallback(djvu_path)

    def _process_djvu_with_fitz(self, djvu_path: str, dpi: int) -> dict:
        results = {}
        try:
            doc = fitz.open(djvu_path)
            for page_num, page in enumerate(doc):  # type: ignore[union-attr]
                logging.info("Processing DJVU page %d/%d...", page_num + 1, len(doc))
                text = page.get_text("text")
                if text and len(text.strip()) > 100:
                    logging.info("  -> Page %d: Text layer found.", page_num + 1)
                    results[f"page_{page_num + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    logging.info(
                        "  -> Page %d: Text layer empty or small. Running OCR.", page_num + 1
                    )
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    img_cv = cv2.cvtColor(
                        img_np,
                        (
                            cv2.COLOR_RGBA2BGR
                            if pix.n == 4
                            else cv2.COLOR_RGB2BGR
                            if pix.n == 3
                            else cv2.COLOR_GRAY2BGR
                        ),
                    )
                    results[f"page_{page_num + 1}"] = self.ocr.ocr_image(img_cv)
            doc.close()
        except OSError as e:
            logging.error("Error processing DJVU with PyMuPDF: %s", e)
            try:
                results = self._process_djvu_fallback(djvu_path)
            except OSError as fallback_error:
                logging.error("All DJVU processing methods failed: %s", fallback_error)
        return results

    def _process_djvu_fallback(self, djvu_path: str) -> dict:
        results = {}
        try:
            logging.info("Trying to convert DJVU to PDF for processing...")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                pdf_path = tmp_pdf.name
            try:
                subprocess.run(["which", "ddjvu"], capture_output=True, check=True)
                cmd = ["ddjvu", "-format=pdf", "-quality=85", djvu_path, pdf_path]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300, check=False
                )
                if result.returncode == 0:
                    logging.info("DJVU -> PDF conversion successful.")
                    return self.process_pdf(pdf_path)
                raise subprocess.CalledProcessError(result.returncode, cmd)
            except (FileNotFoundError, subprocess.CalledProcessError):
                logging.warning("ddjvu utility not found. Trying alternative methods...")
                try:
                    cmd = ["djvutxt", djvu_path]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=180, check=False
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        text = result.stdout.strip()
                        pages = text.split("\f")
                        for i, page_text in enumerate(pages):
                            if page_text.strip():
                                results[f"page_{i + 1}"] = self.ocr.postprocess_old_russian_text(
                                    page_text.strip()
                                )
                        if results:
                            logging.info(
                                "Text successfully extracted from DJVU: %d pages", len(results)
                            )
                            return results
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass
                logging.error("Failed to process DJVU file automatically.")
            finally:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
        except OSError as e:
            logging.error("Error in fallback DJVU processing: %s", e)
        return results

    def process_image(self, image_path: str) -> str:
        """Processes a single image file for OCR."""
        logging.info("Processing image: %s", image_path)
        return self.ocr.ocr_image(image_path)


def process_file(file_path: str, config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """
    Processes a single file (PDF, DJVU, or image) to extract text.
    """
    config = load_config(config_path)
    logging.info("\n%s\nStarting file processing: %s\n%s", "=" * 80, file_path, "=" * 80)
    try:
        ocr_engine = OldRussianOCR(
            engine=config.get("ocr", {}).get("engine", "tesseract"),
            config=config,
        )
    except RuntimeError as e:
        logging.error("OCR initialization failed: %s", e)
        return
    processor = DocumentProcessor(ocr_engine, config=config)
    if not os.path.exists(file_path):
        logging.error("Error: File '%s' not found!", file_path)
        return
    ext = Path(file_path).suffix.lower()
    results = {}
    if ext == ".pdf":
        results = processor.process_pdf(file_path)
    elif ext in [".djvu", ".djv"]:
        results = processor.process_djvu(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        text = processor.process_image(file_path)
        if text:
            results = {"image_1": text}
    else:
        return
    if not results:
        logging.warning("Failed to extract text from: %s", file_path)
        return
    output_dir = Path(file_path).parent
    output_suffix = config.get("output", {}).get("suffix", "_распознано.txt")
    output_file = output_dir / (Path(file_path).stem + output_suffix)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i, (page, text) in enumerate(results.items()):
                if i > 0:
                    f.write("\n\n" + "=" * 80 + "\n\n")
                f.write(f"--- СТРАНИЦА: {page.replace('_', ' ')} ---\n\n")
                f.write(text.strip())
        logging.info("Results successfully saved to file: %s", output_file)
    except OSError as e:
        logging.error("Error saving file %s: %s", output_file, e)
    total_pages = len(results)
    total_chars = sum(len(text) for text in results.values())
    logging.info("--- Statistics for file '%s' ---", Path(file_path).name)
    logging.info("Total pages/images processed: %d", total_pages)
    logging.info("Total characters recognized: %d", total_chars)
    logging.info("-" * 80)
