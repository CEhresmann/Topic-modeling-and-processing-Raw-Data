"""
This script aggregates text from multiple files into a single CSV file.
"""

import csv
import logging
import os
import sys
import uuid
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_text_files(directory: str):
    """Finds all '*_распознано.txt' files in a directory."""
    return Path(directory).rglob("*_распознано.txt")


def aggregate_to_csv(directory: str, output_csv_path: str):
    """Aggregates text from multiple files into a single CSV file."""
    logging.info("Поиск файлов в директории: %s", directory)
    text_files = sorted(find_text_files(directory))

    if not text_files:
        logging.warning("Не найдено файлов '_распознано.txt' для обработки.")
        return

    logging.info("Найдено %d файлов. Начинаю сборку CSV...", len(text_files))

    header = ["id", "author", "title", "description", "date", "size", "text"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for txt_path in text_files:
            try:
                file_size = os.path.getsize(txt_path)
                with open(txt_path, encoding="utf-8") as f:
                    content = f.read()

                title = txt_path.name.replace("_распознано.txt", "")

                doc_id = str(uuid.uuid4())

                # Заполняем строку для CSV. author, description, date остаются пустыми.
                row = [
                    doc_id,
                    "",  # author - нет данных
                    title,
                    "",  # description - нет данных
                    "",  # date - нет данных
                    file_size,
                    content,
                ]
                writer.writerow(row)

            except OSError as e:
                logging.error("Ошибка при обработке файла %s: %s", txt_path, e)

    logging.info("Сборка завершена. Результаты сохранены в: %s", output_csv_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python aggregate_results.py <directory>")
        sys.exit(1)

    target_directory = sys.argv[1]
    output_csv = Path(target_directory) / "aggregated_results.csv"

    if not os.path.isdir(target_directory):
        print(f"Ошибка: '{target_directory}' не является директорией.")
        sys.exit(1)

    aggregate_to_csv(target_directory, str(output_csv))
