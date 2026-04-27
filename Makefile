# --- FILE: Makefile ---
.PHONY: install test lint format pipeline fetch ck km clean docker

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

pipeline:
	$(PY) scripts/run_pipeline.py --config configs/pipeline.yaml

fetch:
	$(PY) scripts/stages/fetch_data.py --config configs/pipeline.yaml

ck:
	$(PY) scripts/stages/run_markov_test.py --config configs/pipeline.yaml

km:
	$(PY) scripts/stages/run_km.py --config configs/pipeline.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf logs/*.log logs/*.jsonl
	rm -rf data/processed/* data/interim/*
	rm -rf reports/figures/* reports/tables/*

docker:
	docker build -t econophys-langevin:latest .
