"""
Main script for text extraction, aggregation, and topic modeling.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.aggregate_results import aggregate_to_csv
from src.extract_text_from_pdf import process_file
from src.topic_modeling import load_config, run_topic_modeling


def process_files(directory):
    """
    Recursively processes files in a directory to extract text.
    """
    print(f"Начинаю рекурсивную обработку файлов в каталоге: {directory}")
    if not os.path.isdir(directory):
        print(f"Ошибка: '{directory}' не является каталогом.")
        return

    processed_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path)
            processed_files += 1

    print(f"\nОбработка завершена. Всего просмотрено файлов: {processed_files}.")


def setup_parsers():
    """
    Sets up the command-line argument parsers.
    """
    parser = argparse.ArgumentParser(description="""
    Скрипт для извлечения текста из PDF, DJVU и изображений с использованием OCR,
    специализированный для дореволюционных русских текстов, и последующей
    агрегации результатов в CSV файл.
    """)

    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    process_parser = subparsers.add_parser(
        "process", help="Рекурсивно обработать файлы в директории и извлечь текст."
    )
    process_parser.add_argument(
        "--input-dir", required=True, help="Каталог с исходными файлами для обработки."
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

    return parser


def main():
    """
    Main function to run the script.
    """
    parser = setup_parsers()
    args = parser.parse_args()

    if args.command == "process":
        process_files(args.input_dir)
    elif args.command == "aggregate":
        output_csv = os.path.join(args.input_dir, "aggregated_results.csv")
        aggregate_to_csv(args.input_dir, output_csv)
    elif args.command == "topic-model":
        tm_config = load_config(args.config)
        run_topic_modeling(tm_config)
    elif args.command == "full-pipeline":
        process_files(args.input_dir)

        output_csv = os.path.join(args.input_dir, "aggregated_results.csv")
        aggregate_to_csv(args.input_dir, output_csv)

        tm_config = load_config(args.config)
        tm_config["data"]["file_path"] = output_csv
        run_topic_modeling(tm_config)


if __name__ == "__main__":
    main()
