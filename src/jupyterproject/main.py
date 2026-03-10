"""
Main script for text extraction, aggregation, and topic modeling.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from jupyterproject.aggregate_results import aggregate_to_csv
from jupyterproject.extract_text_from_pdf import process_file
from jupyterproject.llm_text_cleaning import clean_csv_with_llm
from jupyterproject.topic_modeling import load_config, run_topic_modeling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Supported file extensions for OCR processing
SUPPORTED_EXTENSIONS = {".pdf", ".djvu", ".djv", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


def _process_single_file(args: tuple[str, str]) -> tuple[str, bool, str]:
    """
    Worker function for parallel processing. Processes a single file.

    Args:
        args: Tuple of (file_path, config_path)

    Returns:
        Tuple of (file_path, success, error_message)
    """
    file_path, config_path = args
    try:
        process_file(file_path, config_path=config_path)
        return (file_path, True, "")
    except Exception as e:  # pylint: disable=broad-except
        return (file_path, False, str(e))


def process_files(  # pylint: disable=too-many-branches
    directory: str,
    config_path: str = "config.yaml",
    max_workers: int | None = None,
    resume: bool = False,
) -> None:
    """
    Recursively processes files in a directory to extract text.

    Args:
        directory: Path to the directory containing files to process.
        config_path: Path to the configuration file.
        max_workers: Maximum number of parallel workers. Defaults to CPU count.
        resume: If True, skip files that already have *_распознано.txt output.
    """
    print(f"Начинаю рекурсивную обработку файлов в каталоге: {directory}")
    if not os.path.isdir(directory):
        print(f"Ошибка: '{directory}' не является каталогом.")
        return

    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Collect all supported files
    all_files: list[str] = []
    output_suffix = "_распознано.txt"

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = Path(file_path).suffix.lower()

            # Skip unsupported files
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Skip if already processed (resume mode)
            if resume:
                output_file = str(Path(file_path).with_suffix("")) + output_suffix
                if os.path.exists(output_file):
                    logging.info("Пропуск (уже обработан): %s", file_path)
                    continue

            all_files.append(file_path)

    total_files = len(all_files)

    if total_files == 0:
        print("Не найдено файлов для обработки.")
        return

    print(
        f"Найдено {total_files} файлов. Запускаю параллельную обработку ({max_workers} workers)..."
    )

    # Prepare arguments for parallel processing
    process_args = [(f, config_path) for f in all_files]

    processed = 0
    failed = 0
    errors: list[str] = []

    # Use ProcessPoolExecutor for CPU-bound OCR tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_single_file, args): args[0] for args in process_args
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                path, success, error_msg = future.result()
                if success:
                    processed += 1
                else:
                    failed += 1
                    errors.append(f"{path}: {error_msg}")
                    logging.error("Ошибка при обработке %s: %s", path, error_msg)
            except Exception as e:  # pylint: disable=broad-except
                failed += 1
                errors.append(f"{file_path}: {str(e)}")
                logging.error("Исключение при обработке %s: %s", file_path, e)

            # Progress update every 10%
            progress = (processed + failed) / total_files * 100
            if (processed + failed) % max(
                1, total_files // 10
            ) == 0 or processed + failed == total_files:
                print(f"Прогресс: {processed + failed}/{total_files} ({progress:.1f}%)")

    print(f"\nОбработка завершена. Успешно: {processed}, Ошибок: {failed}")
    if errors:
        print("Список ошибок сохранён в: process_errors.log")
        with open("process_errors.log", "w", encoding="utf-8") as f:
            for err in errors:
                f.write(f"{err}\n")


def setup_parsers() -> argparse.ArgumentParser:
    """
    Sets up the command-line argument parsers.
    """
    parser = argparse.ArgumentParser(
        description="""
    Скрипт для извлечения текста из PDF, DJVU и изображений с использованием OCR,
    специализированный для дореволюционных русских текстов, и последующей
    агрегации результатов в CSV файл.
    """
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    process_parser = subparsers.add_parser(
        "process", help="Рекурсивно обработать файлы в директории и извлечь текст."
    )
    process_parser.add_argument(
        "--input-dir", required=True, help="Каталог с исходными файлами для обработки."
    )
    process_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Путь к файлу конфигурации для OCR.",
    )
    process_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Количество параллельных процессов (по умолчанию: число CPU ядер).",
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропускать уже обработанные файлы.",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Собрать результаты из *_распознано.txt файлов в один CSV."
    )
    aggregate_parser.add_argument(
        "--input-dir",
        required=True,
        help="Каталог, в котором находятся *_распознано.txt файлы.",
    )

    model_parser = subparsers.add_parser(
        "topic-model", help="Выполнить тематическое моделирование на основе CSV файла."
    )
    model_parser.add_argument(
        "--config",
        required=True,
        help="Путь к файлу конфигурации.",
    )

    clean_parser = subparsers.add_parser(
        "clean-text",
        help="Очистить агрегированный CSV через LLM-этап перед тематическим моделированием.",
    )
    clean_parser.add_argument("--input-csv", required=True, help="Путь к входному CSV.")
    clean_parser.add_argument(
        "--output-csv",
        required=False,
        help="Путь к выходному CSV (по умолчанию: *_cleaned.csv рядом с входным).",
    )
    clean_parser.add_argument(
        "--config",
        required=True,
        help="Путь к файлу конфигурации.",
    )

    full_pipeline_parser = subparsers.add_parser(
        "full-pipeline", help="Выполнить полный цикл обработки: OCR, агрегация, моделирование."
    )
    full_pipeline_parser.add_argument(
        "--input-dir", required=True, help="Каталог с исходными файлами для обработки."
    )
    full_pipeline_parser.add_argument(
        "--config",
        required=True,
        help="Путь к файлу конфигурации.",
    )
    full_pipeline_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Количество параллельных процессов (по умолчанию: число CPU ядер).",
    )
    full_pipeline_parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропускать уже обработанные файлы.",
    )

    return parser


def main() -> None:
    """
    Main function to run the script.
    """
    parser = setup_parsers()
    args = parser.parse_args()

    if args.command == "process":
        process_files(
            args.input_dir,
            args.config,
            max_workers=getattr(args, "workers", None),
            resume=getattr(args, "resume", False),
        )
    elif args.command == "aggregate":
        output_csv = os.path.join(args.input_dir, "aggregated_results.csv")
        aggregate_to_csv(args.input_dir, output_csv)
    elif args.command == "clean-text":
        tm_config = load_config(args.config)
        output_csv = args.output_csv
        if not output_csv:
            input_path = Path(args.input_csv)
            output_csv = str(input_path.with_name(f"{input_path.stem}_cleaned.csv"))
        clean_csv_with_llm(args.input_csv, output_csv, tm_config)
    elif args.command == "topic-model":
        tm_config = load_config(args.config)
        run_topic_modeling(tm_config)
    elif args.command == "full-pipeline":
        process_files(
            args.input_dir,
            args.config,
            max_workers=getattr(args, "workers", None),
            resume=getattr(args, "resume", False),
        )

        output_csv = os.path.join(args.input_dir, "aggregated_results.csv")
        aggregate_to_csv(args.input_dir, output_csv)

        tm_config = load_config(args.config)
        llm_cfg = tm_config.get("llm_cleaning", {})
        if llm_cfg.get("enabled", False):
            cleaned_output = os.path.join(args.input_dir, "aggregated_results_cleaned.csv")
            output_csv = clean_csv_with_llm(output_csv, cleaned_output, tm_config)
        tm_config["data"]["file_path"] = output_csv
        run_topic_modeling(tm_config)


if __name__ == "__main__":
    main()
