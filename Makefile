# Every command the repository offers, in one place.
#
#   make setup     install the development environment (no PyTorch)
#   make estimators add PyTorch, for the classifiers and the -m torch tests
#   make lint      what CI checks: ruff, formatting
#   make test      the test suite; the estimator tests skip without PyTorch
#   make data      download and rebuild the twelve datasets
#   make check     verify the datasets on disk against their digests, offline
#   make selftest  exercise the whole measurement pipeline, without PyTorch
#   make repro     reproduce every reported artefact from scratch
#   make figure    redraw Figure 4 from the deposited measurements
#   make numbers   re-derive every figure quoted in the empirical section
#   make lock      refresh uv.lock after changing the dependencies

.DEFAULT_GOAL := help
.PHONY: help setup estimators lock lint format test data check selftest repro \
        figure numbers clean

UV ?= uv

help:
	@grep -E '^#   ' Makefile | sed 's/^#   /  /'

# PyTorch is an extra, not a base dependency: the harness imports it lazily, so
# the datasets, the self-test, the figure and most of the tests need none of it.
setup:
	$(UV) sync --locked --group dev

estimators:
	$(UV) sync --locked --group dev --extra estimators

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

selftest:
	$(UV) run python run_benchmarks.py selftest

figure:
	$(UV) run python run_benchmarks.py replot

numbers:
	$(UV) run python scripts/paper_numbers.py

# The full reproduction, in the order the paper needs it. Component B sweeps
# the Skin Segmentation dataset up to N = 8000 and takes hours on one core:
# this is the long path, not a smoke test. `make test` is the quick one.
repro: estimators data
	$(UV) run --extra estimators python run_benchmarks.py A --reps 5
	$(UV) run --extra estimators python run_benchmarks.py B --reps 3
	$(UV) run python scripts/paper_numbers.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ results_table7*.csv
