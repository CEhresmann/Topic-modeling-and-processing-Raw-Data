# AGENTS.md — Руководство для агентов

Этот файл содержит указания для агентных систем (Sisyphus, Cursor, Copilot и др.), работающих с репозиторием JupyterProject.

---

## Git Workflow

Проект использует **Conventional Commits**. Подробнее в [CONTRIBUTING.md](CONTRIBUTING.md).

### Формат коммитов

```
<type>(<scope>): <description>
```

Примеры:
- `feat(main): add parallel file processing`
- `fix(ocr): resolve denoising bug`
- `perf(topic_modeling): reduce LDA tuning passes`

### Типы коммитов

| Тип | Описание |
|-----|----------|
| `feat` | Новая функциональность |
| `fix` | Исправление бага |
| `perf` | Оптимизация производительности |
| `docs` | Документация |
| `refactor` | Рефакторинг |
| `test` | Тесты |

---

## 1. Команды

### Установка
```bash
uv pip install -e .
pip install -e .
```

### Тестирование
```bash
python -m unittest discover -s tests
python -m unittest tests.test_main
python -m unittest tests.test_main.TestMain.test_process_files
```

### Линтинг
```bash
ruff check src/
ruff check src/jupyterproject/main.py
black src/
black --check src/
isort src/ --check-only
isort src/ --apply
mypy src/
```

### Полная проверка
```bash
ruff check src/ && black --check src/ && mypy src/ && python -m unittest discover -s tests
```

---

## 2. Стиль кода (обязателен)

### Общие правила
- **Python**: 3.12+
- **Строки**: макс. 100 символов
- **Типизация**: строгая (mypy --strict-equivalent)
- **Форматирование**: Black
- **Импорты**: isort (profile="black")

### Импорты (3 секции)
1. Стандартная библиотека
2. Сторонние библиотеки
3. Локальные

```python
import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from jupyterproject.main import main
```

### Типизация
Обязательны аннотации для аргументов, возвращаемых типов.

```python
def process_file(file_path: str, config_path: str = "config.yaml") -> None:
    config: dict[str, Any] = load_config(config_path)

def load_config(config_path: str | None = None) -> dict[str, Any]:
```

### Именование
| Элемент | Стиль | Пример |
|---------|--------|--------|
| Функции | snake_case | `process_file` |
| Классы | PascalCase | `DocumentProcessor` |
| Константы | UPPER_SNAKE_CASE | `MAX_WORKERS` |
| Переменные | snake_case | `ocr_engine` |

### Документация (ОБЯЗАТЕЛЬНО - PEP 727)

**Формат**: Google-style docstrings + PEP 727 аннотации в Args.

```python
def process_file(
    file_path: str,
    config_path: str = "config.yaml",
    max_workers: int | None = None,
) -> None:
    """Processes a single file to extract text.

    Args:
        file_path: Path to the input file.
        config_path: Path to the YAML configuration file.
        max_workers: Maximum number of parallel workers.

    Returns:
        None. Results are saved to *_распознано.txt files.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If OCR initialization fails.
    """
```

### Обработка ошибок
1. Логировать ошибки
2. Конкретные исключения
3. Не подавлять без причины

```python
try:
    doc = fitz.open(pdf_path)
except OSError as e:
    logging.error("Critical error processing PDF: %s", e)
    return {}
```

**НЕЛЬЗЯ:**
- `as any`, `@ts-ignore`, `@ts-expect-error`
- Пустые `except: pass`
- Изменять аргументы функций без согласования
- Удалять тесты
- Добавлять зависимости без обсуждения

---

## 3. Структура проекта

```
src/jupyterproject/
├── __init__.py
├── main.py
├── extract_text_from_pdf.py
├── aggregate_results.py
├── topic_modeling.py
└── llm_text_cleaning.py
```

---

## 4. CLI

```bash
python -m jupyterproject.main process --input-dir ./docs
python -m jupyterproject.main aggregate --input-dir ./docs
python -m jupyterproject.main topic-model --config config.yaml
python -m jupyterproject.main clean-text --input-csv data.csv --config config.yaml
python -m jupyterproject.main full-pipeline --input-dir ./docs --config config.yaml

# Оптимизированные флаги
python -m jupyterproject.main process --input-dir ./docs --workers 8 --resume
```

---

## 5. Ограничения

- Tesseract: бинарник в PATH
- EasyOCR: ~2GB GPU памяти
- LDA тюнинг: 10+ минут
- LLM очистка: платная (API)

---

*Обновлено: 2026-03-10*
