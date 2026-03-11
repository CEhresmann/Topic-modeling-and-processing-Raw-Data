.PHONY: help install install-dev pre-commit check lint test full-check quality security types complexity dead-code clean all-check venv

help:
	@echo "Available commands:"
	@echo "  make install        - Install project dependencies"
	@echo "  make install-dev   - Install development dependencies (with pre-commit)"
	@echo "  make venv          - Create new virtual environment"
	@echo "  make pre-commit    - Run pre-commit hooks"
	@echo "  make check         - Run linting (ruff, black, isort, mypy)"
	@echo "  make lint          - Run linters only (ruff, black)"
	@echo "  make test          - Run tests"
	@echo "  make quality       - Run code quality checks (ruff, black, isort)"
	@echo "  make types         - Run type checking (mypy)"
	@echo "  make security      - Run security checks (bandit)"
	@echo "  make complexity    - Run complexity analysis (radon)"
	@echo "  make dead-code     - Run dead code detection (vulture)"
	@echo "  make clean         - Clean unused imports (ruff)"
	@echo "  make all-check     - Full check (quality + types + security + complexity + dead-code)"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install

venv:
	python3 scripts/check_and_clean.py --create-venv

pre-commit:
	pre-commit run --all-files

check:
	ruff check src/
	black --check src/
	isort --check-only src/
	mypy src/

lint:
	ruff check src/
	black --check src/

test:
	python -m unittest discover -s tests

quality:
	python3 scripts/check_and_clean.py --code-quality

types:
	python3 scripts/check_and_clean.py --types

security:
	python3 scripts/check_and_clean.py --security

complexity:
	python3 scripts/check_and_clean.py --complexity

dead-code:
	python3 scripts/check_and_clean.py --dead-code

clean:
	python3 scripts/check_and_clean.py --clean-imports --fix

all-check:
	python3 scripts/check_and_clean.py --all --fix

full-check: check test
