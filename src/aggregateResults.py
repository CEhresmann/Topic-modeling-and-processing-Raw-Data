import os
import csv
import uuid
# noinspection PyCompatibility
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_text_files(directory: str):
    return Path(directory).rglob("*_распознано.txt")


def aggregate_to_csv(directory: str, output_csv_path: str):
    logging.info("Поиск файлов в директории: {directory}")
    text_files = sorted(list(find_text_files(directory)))

    if not text_files:
        logging.warning("Не найдено файлов '_распознано.txt' для обработки.")
        return

    logging.info("Найдено {len(text_files)} файлов. Начинаю сборку CSV...")

    header = ["id", "author", "title", "description", "date", "size", "text"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for txt_path in text_files:
            try:
                file_size = os.path.getsize(txt_path)
                with open(txt_path, "r", encoding="utf-8") as f:
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

            except Exception as e:
                logging.error("Ошибка при обработке файла {txt_path}: {e}")

    logging.info("Сборка завершена. Результаты сохранены в: {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Использование: python aggregateResults.py <directory>")
        sys.exit(1)

    target_directory = sys.argv[1]
    output_csv = Path(target_directory) / "aggregated_results.csv"

    if not os.path.isdir(target_directory):
        logging.error("Ошибка: '{target_directory}' не является директорией.")
        sys.exit(1)

    aggregate_to_csv(target_directory, str(output_csv))
