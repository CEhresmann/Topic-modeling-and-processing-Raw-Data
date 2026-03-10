#!/usr/bin/env python3

import argparse
import ast
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str, fix: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"Команда: {cmd}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Ошибка в {description}:")
            if result.stderr:
                print(result.stderr)
            else:
                print(result.stdout)
            return False
        else:
            print(f"{description} успешно завершена")
            if result.stdout.strip():
                print(result.stdout[:500])  # Ограничиваем вывод
            return True
    except subprocess.TimeoutExpired:
        print(f"Таймаут при выполнении: {description}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка в {description}: {e}")
        return False


def find_unused_imports(file_path: Path) -> list[str]:
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = set()
    used_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            current = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                used_names.add(current.id)

    unused = [imp for imp in imports if imp not in used_names]
    return unused


def clean_imports(fix: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print("Поиск неиспользуемых импортов")
    print("=" * 60)

    project_root = Path.cwd()
    found = False

    for py_file in project_root.rglob("*.py"):
        if py_file.name.startswith("_") or ".venv" in str(py_file):
            continue

        unused = find_unused_imports(py_file)
        if unused:
            found = True
            print(f"{py_file.relative_to(project_root)}:")
            for imp in unused:
                print(f"   - {imp}")

    if not found:
        print("Неиспользуемые импорты не найдены")
        return False

    if fix:
        print("\nАвтоматическое исправление импортов...")
        return run_command(
            "ruff check --select F401 --fix .", "Автоматическое удаление неиспользуемых импортов"
        )
    else:
        print("\n Рекомендации:")
        print("1. Проверьте каждый импорт вручную")
        print("2. Удалите только те, что действительно не нужны")
        print("3. Запустите: python project_check.py --clean-imports --fix")
        return True


def check_code_quality(fix: bool = False) -> bool:
    success = True

    if fix:
        success &= run_command("ruff check --fix .", "Автоматическое исправление ошибок с Ruff")
    else:
        success &= run_command("ruff check .", "Проверка кода с Ruff")

    if fix:
        success &= run_command("black .", "Форматирование кода с Black")
    else:
        success &= run_command("black --check .", "Проверка форматирования с Black")

    success &= run_command("ruff check --select I --fix .", "Сортировка импортов")

    return success


def check_types() -> bool:
    return run_command("mypy src/", "Проверка типов с MyPy")


def check_security() -> bool:
    success = True
    success &= run_command("safety scan", "Проверка безопасности зависимостей")
    success &= run_command("bandit -r src/ -f screen", "Статический анализ безопасности кода")
    return success


def check_complexity() -> bool:
    return run_command("radon cc src/ -a", "Анализ сложности кода с Radon")


def check_dead_code() -> bool:
    return run_command("vulture src/ --min-confidence 80", "Поиск неиспользуемого кода с Vulture")


def update_dependencies() -> bool:
    success = True
    success &= run_command(
        "uv pip compile pyproject.toml -o requirements.txt", "Обновление requirements.txt"
    )
    success &= run_command(
        "uv pip install -r requirements.txt", "Установка обновленных зависимостей"
    )
    return success


def create_venv() -> bool:
    print(f"\n{'=' * 60}")
    print("Создание чистого виртуального окружения")
    print("=" * 60)

    try:
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            print("Виртуальное окружение уже активировано")
            response = input("Продолжить? (y/n): ")
            if response.lower() != "y":
                return False

        venv_dir = Path(".venv")
        if venv_dir.exists():
            import shutil

            shutil.rmtree(venv_dir)
            print("Старое виртуальное окружение удалено")

        print("Создаю новое виртуальное окружение...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("Виртуальное окружение создано")

        print("Устанавливаю uv...")
        activate_script = (
            ".venv/bin/activate" if sys.platform != "win32" else ".venv\\Scripts\\activate"
        )

        if sys.platform != "win32":
            subprocess.run(f"source {activate_script} && pip install uv", shell=True, check=True)
        else:
            subprocess.run(f"{activate_script} && pip install uv", shell=True, check=True)

        print("uv установлен")
        return True

    except Exception as e:
        print(f"Ошибка при создании виртуального окружения: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Скрипт для очистки и проверки проекта",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python project_check.py --all               # Полная проверка
  python project_check.py --clean-imports     # Поиск неиспользуемых импортов
  python project_check.py --fix               # Автоматическое исправление
  python project_check.py --update-deps       # Обновление зависимостей
  python project_check.py --create-venv       # Создание чистого окружения
        """,
    )

    parser.add_argument("--all", action="store_true", help="Выполнить все проверки")
    parser.add_argument("--code-quality", action="store_true", help="Проверка качества кода")
    parser.add_argument(
        "--clean-imports", action="store_true", help="Поиск неиспользуемых импортов"
    )
    parser.add_argument("--types", action="store_true", help="Проверка типов")
    parser.add_argument("--security", action="store_true", help="Проверка безопасности")
    parser.add_argument("--complexity", action="store_true", help="Анализ сложности кода")
    parser.add_argument("--dead-code", action="store_true", help="Поиск неиспользуемого кода")
    parser.add_argument("--update-deps", action="store_true", help="Обновление зависимостей")
    parser.add_argument(
        "--create-venv", action="store_true", help="Создание чистого виртуального окружения"
    )
    parser.add_argument("--fix", action="store_true", help="Автоматическое исправление ошибок")
    parser.add_argument(
        "--exclude", type=str, default="", help="Файлы для исключения (через запятую)"
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    success = True

    # Only run venv/deps if explicitly requested (not part of --all)
    if args.create_venv:
        success &= create_venv()

    if args.update_deps:
        success &= update_dependencies()

    if args.clean_imports or args.all:
        success &= clean_imports(fix=args.fix)

    if args.code_quality or args.all:
        success &= check_code_quality(fix=args.fix)

    if args.types or args.all:
        success &= check_types()

    if args.security or args.all:
        success &= check_security()

    if args.complexity or args.all:
        success &= check_complexity()

    if args.dead_code or args.all:
        success &= check_dead_code()

    print(f"\n{'=' * 60}")
    if success:
        print("Все проверки пройдены успешно!")
    else:
        print("Некоторые проверки не пройдены")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
