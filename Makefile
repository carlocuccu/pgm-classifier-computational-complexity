# Every command the repository offers, in one place.
#
#   make setup     resolve and install the development environment
#   make lint      what CI checks: ruff, formatting
#   make test      the test suite
#   make data      download and rebuild the twelve datasets
#   make check     verify the datasets on disk against their digests, offline
#   make repro     reproduce every reported artefact from scratch
#   make figure    redraw Figure 4 from the deposited measurements
#   make numbers   re-derive every figure quoted in the empirical section
#   make lock      refresh uv.lock after changing the dependencies

.DEFAULT_GOAL := help
.PHONY: help setup lock lint format test data check repro figure numbers clean

UV ?= uv

help:
	@grep -E '^#   ' Makefile | sed 's/^#   /  /'

setup:
	$(UV) sync --locked --group dev

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest

data:
	$(UV) run python scripts/fetch_datasets.py

check:
	$(UV) run python scripts/fetch_datasets.py --check

figure:
	$(UV) run python run_benchmarks.py replot

numbers:
	$(UV) run python scripts/paper_numbers.py

# The full reproduction, in the order the paper needs it. Component B sweeps
# the Skin Segmentation dataset up to N = 8000 and takes hours on one core:
# this is the long path, not a smoke test. `make test` is the quick one.
repro: data
	$(UV) run python run_benchmarks.py A --reps 5
	$(UV) run python run_benchmarks.py B --reps 3
	$(UV) run python scripts/paper_numbers.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ results_table7*.csv
