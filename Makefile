.PHONY: install test lint format docs clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

docs:
	cd docs && make html

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
