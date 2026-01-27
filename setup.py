import os
import subprocess
import sys
import toml

def get_dependencies():
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            pyproject = toml.load(f)
        dependencies = pyproject["project"]["dependencies"]
        return dependencies
    except (FileNotFoundError, KeyError) as e:
        print(f"Ошибка: не удалось прочитать зависимости из pyproject.toml: {e}")
        sys.exit(1)

def setup_environment():    
    try:
        print("Создание виртуального окружения...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)

        pip_executable = os.path.join(".venv", "bin", "pip")
        if sys.platform == "win32":
            pip_executable = os.path.join(".venv", "Scripts", "pip.exe")

        print("Установка uv...")
        subprocess.run([pip_executable, "install", "uv"], check=True)

        uv_executable = os.path.join(".venv", "bin", "uv")
        if sys.platform == "win32":
            uv_executable = os.path.join(".venv", "Scripts", "uv.exe")
        
        dependencies = get_dependencies()
        if not dependencies:
            print("Зависимости не найдены.")
            
        else:
            print("Установка зависимостей с помощью uv...")
            subprocess.run([uv_executable, "pip", "install"] + dependencies, check=True)

        print("Окружение успешно настроено.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при настройке окружения: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Ошибка: 'uv' не найден. Убедитесь, что он установлен и находится в PATH.")
        sys.exit(1)


try:
    from extractTextFromPDF import process_file
    from aggregate_results import aggregate_to_csv
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что файлы 'setup.py', 'extractTextFromPDF.py' и 'aggregate_results.py' находятся в одной директории.")
    sys.exit(1)

def process_files(directory):
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        setup_environment()
    elif len(sys.argv) > 2 and sys.argv[1] == "process":
        process_files(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "aggregate":
        directory = sys.argv[2]
        output_csv = os.path.join(directory, "aggregated_results.csv")
        aggregate_to_csv(directory, output_csv)
    else:
        print("Использование:")
        print("  python setup.py install - для установки зависимостей")
        print("  python setup.py process <directory> - для обработки файлов")
        print("  python setup.py aggregate <directory> - для сборки результатов в CSV")
