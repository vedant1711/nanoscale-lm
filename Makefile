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

# ------------------------------------------------------------------------ pipeline
.PHONY: tokenizer
tokenizer:  ## Train the committed nano BPE vocabulary and write its report
	$(PY) -m nanoscale.cli tokenizer train --out artifacts/tokenizer/nano.json --tier nano
	$(PY) scripts/tokenizer_report.py

.PHONY: train-nano
train-nano: tokenizer  ## Pretrain the nano tier on CPU and refresh its committed artifacts
	$(PY) -m nanoscale.cli train pretrain --tier nano --set train.device=cpu -o runs/nano/pretrain
	$(PY) scripts/plot_loss_curve.py runs/nano/pretrain --out results/curves/nano_loss.png \
		--title "nano tier -- 400 steps on CPU"
	$(PY) scripts/sample_generations.py runs/nano/pretrain/final.pt --name nano_base

.PHONY: ablate
ablate:  ## Run the Phase-5 controlled ablations
	$(PY) scripts/ablate.py

.PHONY: align
align:  ## SFT -> {DPO, DPO+NLL, SimPO} plus the length-exploitation diagnostic
	$(PY) scripts/align_pipeline.py runs/nano/pretrain/final.pt

.PHONY: distill
distill:  ## Compare the three distillation objectives
	$(PY) scripts/distill_compare.py runs/nano/sft/final.pt

.PHONY: quantize
quantize:  ## Measure the bits-vs-accuracy frontier
	$(PY) scripts/quantize_frontier.py runs/nano/pretrain/final.pt

.PHONY: specdec
specdec:  ## Benchmark speculative decoding against autoregressive
	$(PY) scripts/specdec_bench.py runs/nano/pretrain/final.pt \
		--draft runs/nano/distill/reverse_kl/final.pt

.PHONY: evaluate
evaluate:  ## Full evaluation report: bits/byte, minimal pairs, calibration, diversity
	$(PY) scripts/evaluate.py runs/micro/tinystories/final.pt --name micro-tinystories

.PHONY: baseline
baseline:  ## Compare against GPT-2 / distilGPT-2 on tokenizer-independent bits-per-byte
	$(PY) scripts/external_baseline.py runs/micro/tinystories/final.pt

.PHONY: ablate-multiseed
ablate-multiseed:  ## Ablations at 5 seeds with Welch's t-test and Cohen's d
	$(PY) scripts/ablate_multiseed.py --seeds 5

.PHONY: train-micro-tinystories
train-micro-tinystories:  ## Fetch TinyStories and train the 40M natural-language model
	$(PY) scripts/fetch_tinystories.py --mb 320 --valid-mb 12
	$(PY) -m nanoscale.cli tokenizer train --tier micro \
		--out artifacts/tokenizer/micro_tinystories.json
	$(PY) -m nanoscale.cli train pretrain --config configs/micro_tinystories.yaml \
		--tokenizer artifacts/tokenizer/micro_tinystories.json -o runs/micro/tinystories

.PHONY: export-models
export-models:  ## Strip checkpoints to committable inference-only weights
	$(PY) scripts/export_models.py

.PHONY: bench
bench:  ## The unified results table across every variant
	$(PY) scripts/bench_all.py

.PHONY: results
results:  ## Regenerate docs/results.md from the committed artifacts
	$(PY) scripts/build_docs_results.py

.PHONY: docs
docs:  ## Serve the documentation locally
	$(PY) -m mkdocs serve

.PHONY: smoke
smoke:  ## End-to-end: tokenizer -> pretrain -> SFT -> DPO -> quantize -> speculate (CPU, <10 min)
	$(PY) -m pytest tests/e2e -m slow -v -s

.PHONY: train-micro
train-micro:  ## Pretrain the micro tier (needs a GPU and the 'data' extra)
	$(PY) -m nanoscale.cli train pretrain --tier micro -o runs/micro/pretrain
	$(PY) scripts/plot_loss_curve.py runs/micro/pretrain --out results/curves/micro_loss.png \
		--title "micro tier -- FineWeb-Edu, 20:1 token budget"

.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis build dist htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: compress
compress:  ## Neural compression + anomaly detection benchmark vs gzip/bzip2/xz
	$(PY) scripts/compression_bench.py runs/micro/tinystories/final.pt

.PHONY: emergence
emergence:  ## Probe the minimal-pair suite through one training run
	$(PY) scripts/emergence.py --steps 4000 --probe-every 250
