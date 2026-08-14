.PHONY: install test lint format docs build

install:
	python -m pip install -e ".[cloud,test,docs]"

test:
	pytest --cov=cloud_qdm --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

docs:
	mkdocs build --strict

build:
	python -m build
