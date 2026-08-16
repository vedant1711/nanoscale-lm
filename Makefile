.DEFAULT_GOAL := help
PY := .venv/bin/python
UV := uv

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------- setup
.PHONY: install
install:  ## Create the venv and install the project with dev extras
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

.PHONY: install-all
install-all:  ## Install every optional extra (data, compare, docs, demo)
	$(UV) pip install -e ".[dev,data,compare,docs,demo]"

# ------------------------------------------------------------------------- quality
.PHONY: lint
lint:  ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

.PHONY: fmt
fmt:  ## Auto-fix lint and format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

.PHONY: type
type:  ## mypy --strict
	$(PY) -m mypy

.PHONY: test
test:  ## Run the full test suite
	$(PY) -m pytest

.PHONY: test-fast
test-fast:  ## Run the test suite, skipping slow tests
	$(PY) -m pytest -m "not slow"

.PHONY: check
check: lint type test  ## lint + type + test (what CI runs)

# --------------------------------------------------------------------------- build
.PHONY: schemas
schemas:  ## Export config JSON Schemas to configs/schema/
	$(PY) scripts/export_schemas.py

.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis build dist htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
