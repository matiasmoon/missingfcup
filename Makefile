# Development tasks. Run `make` on its own for the usual check before a commit.

.PHONY: help all clean lint fmt test build examples docs docs-serve

help:
	@echo "make lint        ruff check + format check (no writes)"
	@echo "make fmt         apply ruff formatting and safe fixes"
	@echo "make test        run the test suite"
	@echo "                 (the PNG export test needs a browser; kaleido fetches"
	@echo "                  one on first use, or run 'plotly_get_chrome -y' first)"
	@echo "make examples    run every script in examples/ (opens figures)"
	@echo "make docs        build the documentation site into site/"
	@echo "make docs-serve  serve the docs locally with live reload"
	@echo "make build       clean, then build the sdist and wheel into dist/"
	@echo "make clean       remove build artifacts, caches and stray .DS_Store"
	@echo "make all         lint + test"

all: lint test

# Removing build/ before packaging is the point of this target: a stale build/
# directory is silently reused and can ship files that no longer exist in src/.
clean:
	rm -rf build dist site .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete

lint:
	ruff check src tests examples
	ruff format --check src tests examples

fmt:
	ruff format src tests examples
	ruff check --fix src tests examples

test:
	pytest tests/ -v

examples:
	@for f in examples/*.py; do echo "--- $$f"; python "$$f" || exit 1; done

# --strict turns a broken cross-reference or a missing docstring target into an
# error, so the docs cannot silently rot.
docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

build: clean
	python -m build
	python -m twine check dist/*
