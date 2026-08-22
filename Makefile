.PHONY: install data train evaluate test lint smoke

install:
	pip install -e ".[dev]"

data:
	python -m greenload.data.generate_synthetic_data --out data/raw_series.npz

train:
	python -m greenload.train

evaluate:
	python -m greenload.evaluate

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/

smoke:
	python -m greenload.train --smoke-test
	python -m greenload.evaluate --smoke-test
