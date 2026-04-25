# clv-tf — entrypoints. Each target delegates to the Typer CLI so
# the same commands work without make: `uv run clv <cmd>` is equivalent.

PY := uv run
CLI := $(PY) python -m src.cli

.PHONY: help setup data train eval test smoke baselines all clean fmt

help:
	@echo "Targets:"
	@echo "  setup      install deps via uv"
	@echo "  data       generate synthetic dataset (50k customers x 36 months)"
	@echo "  baselines  fit baselines and write metrics"
	@echo "  train      train deep CLV model"
	@echo "  eval       evaluate all models, write report + plots"
	@echo "  test       run pytest"
	@echo "  smoke      tiny end-to-end run (500 customers, 1 epoch)"
	@echo "  all        setup -> data -> baselines -> train -> eval"
	@echo "  clean      remove generated data, models, runs"
	@echo "  fmt        ruff format + lint"

setup:
	uv sync --extra dev

data:
	$(CLI) generate

baselines:
	$(CLI) baselines

train:
	$(CLI) train

eval:
	$(CLI) evaluate

test:
	$(PY) pytest -q

smoke:
	$(CLI) smoke

all: setup data baselines train eval

clean:
	rm -rf data/synthetic/*.parquet models/* runs/* logs/*

fmt:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests
