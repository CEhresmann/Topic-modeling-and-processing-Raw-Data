import argparse
from setup import process_files

def main():
    parser = argparse.ArgumentParser(description="Обработка файлов для тематического моделирования.")
    parser.add_argument("--input-dir", required=True, help="Каталог с исходными файлами.")
    args = parser.parse_args()
    
    process_files(args.input_dir)

if __name__ == "__main__":
    main()
