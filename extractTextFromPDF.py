"""
This script is designed to extract text from PDF and DJVU files, with a special focus on pre-revolutionary Russian texts. It uses a combination of direct text extraction and Optical Character Recognition (OCR) for pages without a text layer.

Key Features:
- Support for both PDF and DJVU file formats.
- Advanced image preprocessing tailored for old, yellowish documents, including deskewing, contrast enhancement, and binarization.
- Utilizes Tesseract OCR with specialized language models for Russian, Church Slavonic, and English.
- Post-processes OCR output to replace pre-revolutionary characters (like Ѣ, Ѳ, Ѵ, І) with their modern equivalents.
- Implements a robust PDF processing pipeline with pdfplumber and a fallback to PyMuPDF for better reliability.
- Handles documents with and without existing text layers.
- Saves the extracted text into a single `.txt` file, preserving the page structure.

Dependencies:
- pytesseract
- Pillow (PIL)
- opencv-python
- numpy
- pdf2image
- PyMuPDF (fitz)
- pdfplumber
- python-djvulibre (optional, for DJVU support)

Usage:
1.  Install the required dependencies from `requirements.txt`.
2.  Make sure Tesseract OCR is installed and configured. For Windows, you may need to specify the path to `tesseract.exe`.
3.  Place the document you want to process in the same directory as the script.
4.  Change the `file_path` variable in the `main()` function to the name of your file.
5.  Run the script: `python your_script_name.py`
6.  The output will be saved as `[original_filename]_распознано.txt`.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile

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

    # Для DJVU (опционально)
    try:
        import djvu.decode as djvu_decoder
        DJVU_SUPPORT = True
    except ImportError:
        DJVU_SUPPORT = False

except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)


class OldRussianOCR:
    """Класс для распознавания дореволюционных текстов"""

    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Инициализация OCR системы

        Args:
            tesseract_path: путь к tesseract.exe (для Windows)
        """
        if tesseract_path and os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Языковые модели для дореформенного русского
        self.languages = "rus+chu" 
        self.setup_ocr_engine()

    def setup_ocr_engine(self):
        """Настройка OCR движка"""
        # Проверяем доступные языки
        try:
            langs = pytesseract.get_languages(config='')
            print(f"Доступные языки Tesseract: {langs}")
            required_langs = self.languages.split('+')
            for lang in required_langs:
                if lang not in langs:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Язык '{lang}' не найден в Tesseract.")
        except pytesseract.TesseractNotFoundError:
            print("Ошибка: Tesseract не найден. Убедитесь, что он установлен и находится в PATH,")
            print("или укажите путь к нему в OldRussianOCR(tesseract_path=...)")
            sys.exit(1)
        except Exception as e:
            print(f"Не удалось получить список языков Tesseract: {e}")

    def preprocess_for_old_russian(self, image: np.ndarray) -> np.ndarray:
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
        norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # 2. Дескьюинг (выравнивание)
        try:
            coords = np.column_stack(np.where(norm_img < 200)) # Ищем темные пиксели
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 0.5: # Применяем поворот только если он значительный
                (h, w) = norm_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(norm_img, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = norm_img
        except Exception:
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
        resized = cv2.resize(denoised, (new_width, new_height),
                            interpolation=cv2.INTER_LANCZOS4) # Lanczos лучше для масштабирования

        # 6. Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(resized, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 4)

        return binary

    def postprocess_old_russian_text(self, text: str) -> str:
        """
        Постобработка распознанного дореволюционного текста

        Args:
            text: распознанный текст

        Returns:
            исправленный текст
        """
        replacements = {
            'Ѣ': 'е', 'ѣ': 'е',
            'Ѳ': 'ф', 'ѳ': 'ф',
            'Ѵ': 'и', 'ѵ': 'и',
            'І': 'и', 'і': 'и', # Десятеричное i
            # Более редкие
            'Ѫ': 'у', 'ѫ': 'у',
            'Ѧ': 'я', 'ѧ': 'я',
            'Ѯ': 'кс', 'ѯ': 'кс',
            'Ѱ': 'пс', 'ѱ': 'пс',
        }

        # Замена для заглавных букв
        for old, new in replacements.items():
            text = text.replace(old, new.upper() if old.isupper() else new)

        # Дополнительная очистка от типичных ошибок OCR
        text = text.replace(' | ', ' ')
        text = text.replace(' |', ' ')
        text = text.replace('| ', ' ')

        return text

    def ocr_image(self, image_input: [str, np.ndarray]) -> str:
        """
        Распознавание текста с изображения

        Args:
            image_input: путь к изображению или numpy-массив

        Returns:
            распознанный текст
        """
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(f"Не удалось загрузить изображение: {image_input}")
            else:
                img = image_input

            processed = self.preprocess_for_old_russian(img)

            debug_dir = Path("debug_ocr")
            debug_dir.mkdir(exist_ok=True)
            # Используем временный файл для уникальности
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=debug_dir) as temp_file:
                debug_path = temp_file.name
            cv2.imwrite(debug_path, processed)
            print(f"Отладочное изображение сохранено: {debug_path}")


            # Используем PSM 3 (полностью автоматическая сегментация) как основной,
            # он часто дает лучший результат для целых страниц.
            config = f'--oem 3 --psm 3 -l {self.languages}'
            text = pytesseract.image_to_string(processed, config=config)

            corrected_text = self.postprocess_old_russian_text(text)

            return corrected_text

        except Exception as e:
            print(f"Ошибка OCR: {e}")
            return ""


class DocumentProcessor:
    """Обработчик документов различных форматов"""

    def __init__(self, ocr_engine: OldRussianOCR):
        self.ocr = ocr_engine

    def _process_page_with_ocr(self, pdf_path: str, page_num: int, dpi: int) -> str:
        """Вспомогательная функция для OCR одной страницы PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path,
                                      first_page=page_num + 1,
                                      last_page=page_num + 1,
                                      dpi=dpi,
                                      output_folder=tmpdir,
                                      fmt='png',
                                      thread_count=4) # Ускоряем конвертацию
            if images:
                # Конвертируем PIL Image в numpy array для OpenCV
                img_np = np.array(images[0])
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                return self.ocr.ocr_image(img_cv)
        return ""


    def process_pdf(self, pdf_path: str, dpi: int = 400) -> dict:
        """
        Обработка PDF файла с комбинацией извлечения текста и OCR.

        Args:
            pdf_path: путь к PDF файлу
            dpi: качество для конвертации в изображение

        Returns:
            словарь с текстом по страницам
        """
        results = {}
        print("Начинаю обработку PDF...")

        try:
            # Используем PyMuPDF для всего, он быстрее и надежнее
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                print(f"Обработка страницы {page_num + 1}/{len(doc)}...")
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Проверяем, есть ли на странице осмысленный текст
                if text and len(text.strip()) > 100:
                    print(f"  -> Страница {page_num + 1}: Текстовый слой найден.")
                    results[f"page_{page_num + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    print(f"  -> Страница {page_num + 1}: Текстовый слой пуст или мал. Запускаю OCR.")
                    # Конвертируем страницу в изображение для OCR
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    if pix.n == 4: # RGBA -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    else: # RGB -> BGR
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    ocr_text = self.ocr.ocr_image(img_cv)
                    results[f"page_{page_num + 1}"] = ocr_text

            doc.close()

        except Exception as e:
            print(f"Критическая ошибка при обработке PDF: {e}")

        return results


    def process_djvu(self, djvu_path: str, dpi: int = 400) -> dict:
        """
        Обработка DJVU файла. Требует предварительной конвертации в PDF.
        """
        if DJVU_SUPPORT:
            print("Обнаружена поддержка DJVU. Попробую обработать напрямую (может быть нестабильно).")
            # Эта часть остается экспериментальной, т.к. рендеринг в python-djvulibre сложен
            return self._process_djvu_native(djvu_path, dpi)
        else:
            print("DJVU поддержка не установлена. Рекомендуется конвертация в PDF.")
            print("Пожалуйста, сконвертируйте DJVU в PDF с помощью внешней утилиты, например:")
            print(f"  ddjvu -format=pdf \"{djvu_path}\" \"{Path(djvu_path).with_suffix('.pdf')}\"")
            return {}

    def _process_djvu_native(self, djvu_path: str, dpi: int) -> dict:
        """Нативная обработка DJVU (экспериментально)"""
        results = {}
        try:
            from djvu.decode import Context, FileURI
            from djvu.renderer import Renderer

            ctx = Context()
            doc = ctx.new_document(FileURI.from_filename(djvu_path))
            doc.decoding_job.wait()

            for i, page in enumerate(doc.pages):
                print(f"Обработка DJVU страницы {i+1}/{len(doc.pages)}...")
                page.decoding_job.wait()

                # Попробуем извлечь текст, если он есть
                text_zones = page.text.sexpr.decode('utf-8')
                # Простое извлечение, может быть неточным
                text = ' '.join(filter(lambda x: not x.startswith(('(')), text_zones.split('"')))
                text = text.strip()

                if text and len(text) > 50:
                    print(f"  -> Страница {i+1}: Текстовый слой найден.")
                    results[f"page_{i + 1}"] = self.ocr.postprocess_old_russian_text(text)
                else:
                    print(f"  -> Страница {i+1}: Запускаю OCR.")
                    renderer = Renderer(page)
                    page_rect = page.get_info()['rect']
                    render_mode = (page_rect, (int(page_rect[2]*dpi/72), int(page_rect[3]*dpi/72)), "ppi", 300)
                    
                    bitmap = renderer.render(*render_mode)
                    pil_img = bitmap.to_pil()
                    img_np = np.array(pil_img)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    ocr_text = self.ocr.ocr_image(img_cv)
                    results[f"page_{i + 1}"] = ocr_text

        except Exception as e:
            print(f"Ошибка при нативной обработке DJVU: {e}")
        return results


    def process_image(self, image_path: str) -> str:
        """
        Обработка одиночного изображения
        """
        print(f"Обработка изображения: {image_path}")
        return self.ocr.ocr_image(image_path)


def process_file(file_path: str):
    
    print(f"\n{'='*80}\nНачинаю обработку файла: {file_path}\n{'='*80}")
    
    # Инициализация OCR должна происходить один раз, но для простоты оставим здесь.
    # В идеале, ocr_engine нужно передавать в функцию.
    ocr_engine = OldRussianOCR(
        # Для Windows, если Tesseract не в PATH, раскомментируйте и укажите путь:
        # tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    processor = DocumentProcessor(ocr_engine)

    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден!")
        return

    ext = Path(file_path).suffix.lower()
    results = {}
    supported = False

    if ext == '.pdf':
        supported = True
        results = processor.process_pdf(file_path, dpi=400)
    elif ext in ['.djvu', '.djv']:
        supported = True
        results = processor.process_djvu(file_path, dpi=400)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        supported = True
        text = processor.process_image(file_path)
        if text:
            results = {"image_1": text}
    
    if not supported:
        # Это не целевой файл для обработки, пропускаем.
        return

    if not results:
        print(f"Не удалось извлечь текст из: {file_path}")
        return

    # Сохраняем результаты в той же директории, где и исходный файл
    output_dir = Path(file_path).parent
    output_file = output_dir / (Path(file_path).stem + "_распознано.txt")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (page, text) in enumerate(results.items()):
                if i > 0:
                    f.write("\n\n" + "="*80 + "\n\n")
                f.write(f"--- СТРАНИЦА: {page.replace('_', ' ')} ---\n\n")
                f.write(text.strip())
        print(f"Результаты успешно сохранены в файл: {output_file}")
    except IOError as e:
        print(f"Ошибка при сохранении файла {output_file}: {e}")

    # Вывод статистики
    total_pages = len(results)
    total_chars = sum(len(text) for text in results.values())
    print(f"--- Статистика для файла '{Path(file_path).name}' ---")
    print(f"Всего обработано страниц/изображений: {total_pages}")
    print(f"Всего распознано символов: {total_chars}")
    print("-" * 80)
