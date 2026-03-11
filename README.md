# Распознавание и Тематическое моделирование текстов

Этот проект предоставляет набор инструментов для извлечения текста из PDF, DJVU и изображений с использованием OCR, с особым акцентом на дореволюционные русские тексты. Он также включает функционал для агрегации извлеченных данных в единый CSV-файл для последующего анализа и тематического моделирования.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#)

## ➤ Ключевые возможности

- **Поддержка форматов**: PDF, DJVU, PNG, JPG и др.
- **Гибкий OCR**: Поддержка двух движков OCR: Tesseract и EasyOCR.
- **Предобработка изображений**: Оптимизировано для старых, пожелтевших документов (выравнивание, повышение контрастности, бинаризация).
- **Постобработка текста**: Автоматическая замена дореволюционных символов (Ѣ, Ѳ, І и др.) на современные аналоги.
- **Конфигурируемость**: Все основные параметры (пути, движок OCR, DPI) вынесены в файл `config.yaml`.
- **Логирование**: Встроенное логирование для отслеживания процесса обработки.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-)

## ➤ Структура проекта

```
/
├── src/
│   └── jupyterproject/
│       ├── __init__.py               # Python-пакет проекта
│       ├── main.py                   # Точка входа CLI
│       ├── extract_text_from_pdf.py  # Модуль для извлечения текста и OCR
│       ├── aggregate_results.py      # Модуль для сборки результатов в CSV
│       ├── topic_modeling.py         # Модуль для тематического моделирования (LDA)
│       └── llm_text_cleaning.py     # LLM-очистка текста
├── tests/                           # Папка с тестами
│   └── test_*.py                    # Юнит-тесты и интеграционные тесты
├── config.yaml                      # Файл конфигурации
├── pyproject.toml                   # Файл конфигурации проекта
├── AGENTS.md                        # Правила для AI-агентов
└── README.md                        # Этот файл
```

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#--)

## ➤ Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/CEhresmann/Topic-modeling-and-processing-Raw-Data.git
    cd Topic-modeling-and-processing-Raw-Data
    ```

2.  **Установите Tesseract OCR (если планируете его использовать):**
    - **Windows**: Скачайте и установите с [официального сайта](https://github.com/UB-Mannheim/tesseract/wiki).
    - **Linux (Ubuntu/Debian)**: `sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-chu`
    - **macOS**: `brew install tesseract tesseract-lang`

3.  **Создайте виртуальное окружение и установите зависимости:**
    Мы рекомендуем использовать `uv` для быстрой установки.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install uv
    uv pip install -e .
    ```

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#---)

## ➤ Конфигурация

Перед запуском, отредактируйте файл `config.yaml`:
- **tesseract.path**: Укажите путь к `tesseract.exe` если вы на Windows и он не добавлен в PATH.
- **ocr.engine**: Выберите движок OCR: `tesseract` или `easyocr`.
- **ocr.languages**: Языки для Tesseract (например, `rus+chu`).
- **ocr.dpi**: DPI для рендеринга PDF.
- **ocr.psm_candidates**: Несколько режимов сегментации (`[4, 6, 11]`) для сложных дореволюционных страниц.
- **ocr.preprocess_variants**: Набор вариантов бинаризации (`adaptive_gaussian`, `adaptive_mean`, `otsu`).
- **ocr.tesseract_oem**: Рекомендуется `1` (LSTM).
- **ocr.normalize_old_letters**: Включите, если хотите нормализовать `ѣ/і/ѳ` к современным буквам.

## ➤ Рекомендуемый OCR-пайплайн для дореволюционных сканов

1. Настройте в `config.yaml`:
```yaml
ocr:
  engine: "tesseract"
  languages: "rus+chu"
  dpi: 500
  tesseract_oem: 1
  psm_candidates: [4, 6, 11]
  preprocess_variants: ["adaptive_gaussian", "adaptive_mean", "otsu"]
  normalize_old_letters: true
  save_debug_images: true
  upscale_factor: 1.8
  denoise_h: 10
  adaptive_block_size: 21
  adaptive_c: 4
```
2. Запустите OCR:
```bash
python -m jupyterproject.main process --input-dir /path/to/documents
```
3. Проверьте `debug_ocr/` и подберите параметры:
   - если пропадают тонкие символы: уменьшите `denoise_h`;
   - если много фона: увеличьте `adaptive_block_size`;
   - если «слипаются» буквы: уменьшите `upscale_factor` до `1.5`.
4. Соберите распознанные тексты:
```bash
python -m jupyterproject.main aggregate --input-dir /path/to/documents
```

## ➤ Минимальный план дообучения Tesseract (если качества всё ещё мало)

1. Соберите 200-1000 строк из ваших страниц с ручной разметкой.
2. Подготовьте тренировочный набор через `tesstrain` + `langdata_lstm`.
3. Дообучите `rus` (fine-tuning), а не обучайте модель с нуля.
4. Подключите полученный `.traineddata` как язык в `ocr.languages`.

## ➤ Этап LLM-очистки корпуса перед тематическим моделированием

Этот этап исправляет OCR-артефакты после агрегации и до LDA.

1. Включите в `config.yaml`:
```yaml
llm_cleaning:
  enabled: true
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.0
  max_chunk_chars: 1800
  overlap_chars: 0
  only_if_suspect: true
  suspect_score_threshold: 0.12
  min_text_length_for_llm: 600
  max_requests_per_run: 15
  enable_chunk_cache: true
  strict_mode: true
  max_change_ratio: 0.35
  min_cyrillic_ratio: 0.30
  replace_text_column: true
  context_notes: "Дореволюционные русские печатные документы и объявления начала XX века."
  cleanup_rules:
    - "исправляй OCR-ошибки символов, не меняя смысл"
    - "сохраняй имена, фамилии, даты, названия населенных пунктов и должностей"
    - "не добавляй информацию, которой нет в исходном фрагменте"
```
2. Укажите API-ключ в окружении:
```bash
export OPENAI_API_KEY="ваш_ключ"
```
3. Очистите агрегированный CSV:
```bash
python -m jupyterproject.main clean-text \
  --input-csv /path/to/aggregated_results.csv \
  --config /path/to/config.yaml
```
По умолчанию будет создан `aggregated_results_cleaned.csv`.

Для free-тарифа/OpenRouter:
- держите `max_requests_per_run` ниже дневного лимита;
- используйте `only_if_suspect: true`, чтобы не отправлять «чистые» тексты;
- `overlap_chars: 0` уменьшает число запросов;
- выбирайте free-модель (`...:free`) в поле `model`, если доступна.

## ➤ Использование

Все операции выполняются через CLI точку входа. После установки доступна команда `text-processor`.

###  workflow 1: Распознавание и сборка
Этот воркфлоу подходит, когда вам нужно только распознать тексты из коллекции документов и собрать их в один CSV файл для дальнейшего анализа.

1.  **Распознавание текста:**
    ```bash
    python -m jupyterproject.main process --input-dir /path/to/your/documents
    ```
    Эта команда рекурсивно обойдет указанную директорию, распознает текст во всех поддерживаемых файлах (PDF, DJVU, изображения) и сохранит результат в файлы `*_распознано.txt` рядом с оригиналами.

2.  **Агрегация в CSV:**
    ```bash
    python -m jupyterproject.main aggregate --input-dir /path/to/your/documents
    ```
    Команда найдет все `*_распознано.txt` файлы в директории, и соберет их содержимое в один `aggregated_results.csv` файл в той же директории.

### workflow 2: Только тематическое моделирование
Если у вас уже есть готовый CSV файл с текстами, вы можете сразу перейти к тематическому моделированию.

```bash
python -m jupyterproject.main topic-model --config /path/to/your/config.yaml
```
Убедитесь, что в `config.yaml` в секции `data` указан правильный путь к вашему CSV файлу.

### workflow 3: Полный цикл
Этот воркфлоу объединяет распознавание, агрегацию и тематическое моделирование в одну команду.

```bash
python -m jupyterproject.main full-pipeline \
  --input-dir /path/to/your/documents \
  --config /path/to/your/config.yaml \
  --workers 8 \
  --resume
```

**Оптимизированные флаги:**
- `--workers N` — количество параллельных процессов (по умолчанию: число CPU ядер)
- `--resume` — пропускать уже обработанные файлы

Скрипт последовательно выполнит все шаги:
1. Распознает тексты из файлов в `--input-dir`.
2. Соберет их в `aggregated_results.csv`.
3. Если `llm_cleaning.enabled=true`, создаст `aggregated_results_cleaned.csv`.
4. Запустит тематическое моделирование на итоговом CSV файле.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#----)

## ➤ Тестирование

Проект покрыт юнит-тестами и интеграционными тестами. Для запуска всех тестов выполните команду из корневой директории проекта:
```bash
python -m unittest discover -s tests
```
