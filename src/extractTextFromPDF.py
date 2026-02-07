import os
import sys
from pathlib import Path
from typing import Optional, Union
import tempfile
import logging
import yaml


def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Config file not found: config.yaml")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return None


config = load_config()
if config is None:
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Основные зависимости
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF
    import pdfplumber
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
    import easyocr

    '''
    try:
        import djvu.decode as djvu_decoder

        DJVU_SUPPORT = True
    except ImportError:
        DJVU_SUPPORT = False
    '''
    DJVU_SUPPORT = False
except ImportError as import_error:
    logging.error(f"Ошибка импорта: {import_error}")
    logging.error("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)


class OldRussianOCR:
    """Класс для распознавания дореволюционных текстов"""

    def __init__(self, engine: str = "tesseract", tesseract_path: Optional[str] = None):
        """
        Инициализация OCR системы

        Args:
            engine: OCR движок ('tesseract' or 'easyocr')
            tesseract_path: путь к tesseract.exe (для Windows)
        """
        self.easyocr_reader = easyocr.Reader(["ru", "en"])
        self.engine = engine
        if self.engine == "tesseract":
            tesseract_path = tesseract_path or config.get("tesseract", {}).get("path")
            if tesseract_path and os.name == "nt":
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.languages = config.get("ocr", {}).get("languages", "rus+chu")
            self.setup_tesseract()
        elif self.engine == "easyocr":
            self.setup_easyocr()

    def setup_tesseract(self):
        """Настройка OCR движка"""
        # Проверяем доступные языки
        try:
            langs = pytesseract.get_languages(config="")
            logging.info(f"Доступные языки Tesseract: {langs}")
            required_langs = self.languages.split("+")
            for lang in required_langs:
                if lang not in langs:
                    logging.warning(
                        f"ПРЕДУПРЕЖДЕНИЕ: Язык '{lang}' не найден в Tesseract."
                    )
        except pytesseract.TesseractNotFoundError:
            logging.error(
                "Ошибка: Tesseract не найден. Убедитесь, что он установлен и находится в PATH,"
            )
            logging.error("или укажите путь к нему в OldRussianOCR(tesseract_path=...)")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Не удалось получить список языков Tesseract: {e}")

    @staticmethod
    def setup_easyocr():
        """Настройка EasyOCR"""
        try:
            logging.info("EasyOCR инициализирован.")
        except Exception as e:
            logging.error(f"Не удалось инициализировать EasyOCR: {e}")

    @staticmethod
    def preprocess_for_old_russian(image: np.ndarray) -> np.ndarray:
        """
        Специальная предобработка для дореволюционных текстов

        Args:
            image: входное изображение

        Returns:
            обработанное изображение
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Удаление фона и пожелтения с помощью нормализации
        # Это более простой и часто эффективный способ, чем CLAHE для старых документов
        norm_img = cv2.normalize(
            gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        # 2. Дескьюинг (выравнивание)
        try:
            coords = np.column_stack(np.where(norm_img < 200))  # Ищем темные пиксели
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 0.5:  # Применяем поворот только если он значительный
                (h, w) = norm_img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    norm_img,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                rotated = norm_img
        except cv2.error:
            # Если не удалось найти контуры, пропускаем выравнивание
            rotated = norm_img

        # 3. Локальное улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast = clahe.apply(rotated)

        # 4. Аккуратное шумоподавление
        denoised = cv2.fastNlMeansDenoising(enhanced_contrast, h=10)

        # 5. Масштабирование для тонких штрихов (если необходимо)
        # Увеличение DPI при рендеринге - лучший подход, но это может помочь дополнительно
        scale = 1.5
        new_width = int(denoised.shape[1] * scale)
        new_height = int(denoised.shape[0] * scale)
        resized = cv2.resize(
            denoised, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )  # Lanczos лучше для масштабирования

        # 6. Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )

        return binary

    @staticmethod
    def postprocess_old_russian_text(text: str) -> str:
        """
        Постобработка распознанного дореволюционного текста

        Args:
            text: распознанный текст

        Returns:
            исправленный текст
        """
        replacements = {
            "Ѣ": "е",
            "ѣ": "е",
            "Ѳ": "ф",
            "ѳ": "ф",
            "Ѵ": "и",
            "ѵ": "и",
            "І": "и",
            "і": "и",  # Десятеричное i
            # Более редкие
            "Ѫ": "у",
            "ѫ": "у",
            "Ѧ": "я",
            "ѧ": "я",
            "Ѯ": "кс",
            "ѯ": "кс",
            "Ѱ": "пс",
            "ѱ": "пс",
        }

        # Замена для заглавных букв
        for old, new in replacements.items():
            text = text.replace(old, new.upper() if old.isupper() else new)

        # Дополнительная очистка от типичных ошибок OCR
        text = text.replace(" | ", " ")
        text = text.replace(" |", " ")
        text = text.replace("| ", " ")

        return text

    def ocr_image_easyocr(self, image_input: Union[str, np.ndarray]) -> str:
        """
        Распознавание текста с изображения с помощью EasyOCR
        """
        if not self.easyocr_reader:
            logging.error("EasyOCR не инициализирован.")
            return ""
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(
                        f"Не удалось загрузить изображение: {image_input}"
                    )
            else:
                img = image_input

            processed = self.preprocess_for_old_russian(img)
            results = self.easyocr_reader.readtext(processed, detail=0, paragraph=True)
            text = " ".join(results)
            return self.postprocess_old_russian_text(text)

        except Exception as e:
            logging.error(f"Ошибка EasyOCR: {e}")
            return ""

    def ocr_image(self, image_input: Union[str, np.ndarray]) -> str:
        """
        Распознавание текста с изображения

        Args:
            image_input: путь к изображению или numpy-массив

        Returns:
            распознанный текст
        """
        if self.engine == "easyocr":
            return self.ocr_image_easyocr(image_input)

        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(
                        f"Не удалось загрузить изображение: {image_input}"
                    )
            else:
                img = image_input

            processed = self.preprocess_for_old_russian(img)

            debug_dir = Path("debug_ocr")
            debug_dir.mkdir(exist_ok=True)
            # Используем временный файл для уникальности
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, dir=debug_dir
            ) as temp_file:
                debug_path = temp_file.name
            cv2.imwrite(debug_path, processed)
            logging.info(f"Отладочное изображение сохранено: {debug_path}")

            # Используем PSM 3 (полностью автоматическая сегментация) как основной,
            # он часто дает лучший результат для целых страниц.
            ocr_config = f"--oem 3 --psm 3 -l {self.languages}"
            text = pytesseract.image_to_string(processed, config=ocr_config)

            corrected_text = self.postprocess_old_russian_text(text)

            return corrected_text

        except Exception as e:
            logging.error(f"Ошибка OCR: {e}")
            return ""


class DocumentProcessor:
    """Обработчик документов различных форматов"""

    def __init__(self, ocr_engine: OldRussianOCR):
        self.ocr = ocr_engine

    def _process_page_with_ocr(self, pdf_path: str, page_num: int, dpi: int) -> str:
        """Вспомогательная функция для OCR одной страницы PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=dpi,
                output_folder=tmpdir,
                fmt="png",
                thread_count=4,
            )  # Ускоряем конвертацию
            if images:
                # Конвертируем PIL Image в numpy array для OpenCV
                img_np = np.array(images[0])
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                return self.ocr.ocr_image(img_cv)
        return ""

    def process_pdf(self, pdf_path: str) -> dict:
        """
        Обработка PDF файла с комбинацией извлечения текста и OCR.

        Args:
            pdf_path: путь к PDF файлу

        Returns:
            словарь с текстом по страницам
        """
        results = {}
        dpi = config.get("ocr", {}).get("dpi", 400)
        logging.info("Начинаю обработку PDF...")

        try:
            # Используем PyMuPDF для всего, он быстрее и надежнее
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                logging.info(f"Обработка страницы {page_num + 1}/{len(doc)}...")
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Проверяем, есть ли на странице осмысленный текст
                if text and len(text.strip()) > 100:
                    logging.info(
                        f"  -> Страница {page_num + 1}: Текстовый слой найден."
                    )
                    results[f"page_{page_num + 1}"] = (
                        self.ocr.postprocess_old_russian_text(text)
                    )
                else:
                    logging.info(
                        f"  -> Страница {page_num + 1}: Текстовый слой пуст или мал. Запускаю OCR."
                    )
                    # Конвертируем страницу в изображение для OCR
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.h, pix.w, pix.n
                    )

                    if pix.n == 4:  # RGBA -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    else:  # RGB -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    ocr_text = self.ocr.ocr_image(img_cv)
                    results[f"page_{page_num + 1}"] = ocr_text

            doc.close()

        except Exception as e:
            logging.error(f"Критическая ошибка при обработке PDF: {e}")

        return results

    def process_djvu(self, djvu_path: str) -> dict:
        """
        Process a DJVU file by attempting to use PyMuPDF first, and falling back
        to an external converter if that fails.
        """
        dpi = config.get("ocr", {}).get("dpi", 400)

        try:
            import fitz
            fitz_version = fitz.version[0]
            if int(fitz_version.split('.')[0]) >= 1:
                logging.info("Использую PyMuPDF для обработки DJVU файла...")
                return self._process_djvu_with_fitz(djvu_path, dpi)
            else:
                logging.warning(f"Версия PyMuPDF ({fitz_version}) может не поддерживать DJVU. Попробую обработать...")
                return self._process_djvu_with_fitz(djvu_path, dpi)
        except Exception as e:
            logging.warning(f"PyMuPDF не смог открыть DJVU файл: {e}")
            logging.info("Пробую альтернативный метод с конвертацией...")
            return self._process_djvu_fallback(djvu_path)

    def _process_djvu_with_fitz(self, djvu_path: str, dpi: int) -> dict:
        results = {}

        try:
            doc = fitz.open(djvu_path)

            for page_num in range(len(doc)):
                logging.info(f"Обработка DJVU страницы {page_num + 1}/{len(doc)}...")
                page = doc.load_page(page_num)

                # Пытаемся извлечь текстовый слой (если он есть в DJVU)
                text = page.get_text("text")

                if text and len(text.strip()) > 100:
                    logging.info(f"  -> Страница {page_num + 1}: Текстовый слой найден.")
                    results[f"page_{page_num + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    logging.info(f"  -> Страница {page_num + 1}: Текстовый слой пуст или мал. Запускаю OCR.")

                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)

                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.h, pix.w, pix.n
                    )

                    if pix.n == 4:  # RGBA -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 3:  # RGB -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    else:  # Grayscale
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

                    ocr_text = self.ocr.ocr_image(img_cv)
                    results[f"page_{page_num + 1}"] = ocr_text

            doc.close()

        except Exception as e:
            logging.error(f"Ошибка при обработке DJVU с PyMuPDF: {e}")

            try:
                results = self._process_djvu_fallback(djvu_path)
            except Exception as fallback_error:
                logging.error(f"Все методы обработки DJVU не сработали: {fallback_error}")

        return results

    def _process_djvu_fallback(self, djvu_path: str) -> dict:
        import subprocess
        import tempfile
        import os

        results = {}

        try:
            logging.info("Пытаюсь конвертировать DJVU в PDF для обработки...")

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
                pdf_path = tmp_pdf.name

            try:
                subprocess.run(['which', 'ddjvu'], capture_output=True, check=True)

                cmd = ['ddjvu', '-format=pdf', '-quality=85', djvu_path, pdf_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    logging.info("Конвертация DJVU -> PDF успешна.")
                    results = self.process_pdf(pdf_path)
                else:
                    raise subprocess.CalledProcessError(result.returncode, cmd)

            except (FileNotFoundError, subprocess.CalledProcessError):
                logging.warning("Утилита ddjvu не найдена. Пробую альтернативные методы...")

                try:
                    cmd = ['djvutxt', djvu_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

                    if result.returncode == 0 and result.stdout.strip():
                        text = result.stdout.strip()
                        # Разделяем текст по страницам (формат может быть разным)
                        pages = text.split('')  # Форма feed

                        for i, page_text in enumerate(pages):
                            if page_text.strip():
                                results[f'page_{i + 1}'] = self.ocr.postprocess_old_russian_text(
                                    page_text.strip()
                                )

                        if results:
                            logging.info(f"Текст успешно извлечен из DJVU: {len(results)} страниц")
                            return results
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

                # Если ничего не сработало, предлагаем пользователю конвертировать вручную
                logging.error("Не удалось обработать DJVU файл автоматически.")
                logging.error("Рекомендации:")
                logging.error("1. Установите утилиты: sudo apt-get install djvulibre-bin")
                logging.error("2. Конвертируйте вручную: ddjvu -format=pdf -quality=85 input.djvu output.pdf")
                logging.error("3. Обработайте полученный PDF файл")

            finally:
                # Удаляем временный файл если он существует
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)

        except Exception as e:
            logging.error(f"Ошибка при резервной обработке DJVU: {e}")

        return results

    def process_image(self, image_path: str) -> str:
        """Processes a single image file for OCR."""
        logging.info(f"Обработка изображения: {image_path}")
        return self.ocr.ocr_image(image_path)


def process_file(file_path: str):
    """
    Processes a single file (PDF, DJVU, or image) to extract text.
    """
    logging.info(f"\n{'=' * 80}\nНачинаю обработку файла: {file_path}\n{'=' * 80}")

    ocr_engine = OldRussianOCR(engine=config.get("ocr", {}).get("engine", "tesseract"))
    processor = DocumentProcessor(ocr_engine)

    if not os.path.exists(file_path):
        logging.error(f"Ошибка: Файл '{file_path}' не найден!")
        return

    ext = Path(file_path).suffix.lower()
    results = {}
    supported = False

    if ext == ".pdf":
        supported = True
        results = processor.process_pdf(file_path)
    elif ext in [".djvu", ".djv"]:
        supported = True
        results = processor.process_djvu(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        supported = True
        text = processor.process_image(file_path)
        if text:
            results = {"image_1": text}

    if not supported:
        # Это не целевой файл для обработки, пропускаем.
        return

    if not results:
        logging.warning(f"Не удалось извлечь текст из: {file_path}")
        return

    # Сохраняем результаты в той же директории, где и исходный файл
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
        logging.info(f"Результаты успешно сохранены в файл: {output_file}")
    except IOError as e:
        logging.error(f"Ошибка при сохранении файла {output_file}: {e}")

    # Вывод статистики
    total_pages = len(results)
    total_chars = sum(len(text) for text in results.values())
    logging.info(f"--- Статистика для файла '{Path(file_path).name}' ---")
    logging.info(f"Всего обработано страниц/изображений: {total_pages}")
    logging.info(f"Всего распознано символов: {total_chars}")
    logging.info("-" * 80)
