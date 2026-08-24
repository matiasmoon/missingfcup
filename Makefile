# Development tasks. Run `make` on its own for the usual check before a commit.

.PHONY: help all clean lint fmt test build examples notebook-check docs-stage docs docs-serve

help:
	@echo "make lint        ruff check + format check (no writes)"
	@echo "make fmt         apply ruff formatting and safe fixes"
	@echo "make test        run the test suite"
	@echo "                 (the PNG export test needs a browser; kaleido fetches"
	@echo "                  one on first use, or run 'plotly_get_chrome -y' first)"
	@echo "make examples    run every script in examples/ (opens figures)"
	@echo "make notebook-check"
	@echo "                 execute one analysis notebook against the committed CSVs"
	@echo "                 (needs the [notebooks] extra; no network access required)"
	@echo "make docs        build the documentation site into site/"
	@echo "                 (stages notebooks/ into docs/ first; see docs-stage)"
	@echo "make docs-serve  serve the docs locally with live reload"
	@echo "make build       clean, then build the sdist and wheel into dist/"
	@echo "make clean       remove build artifacts, caches and stray .DS_Store"
	@echo "make all         lint + test"

all: lint test

# Removing build/ before packaging is the point of this target: a stale build/
# directory is silently reused and can ship files that no longer exist in src/.
clean:
	rm -rf build dist site docs/notebooks .cache .pytest_cache .ruff_cache
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

# The notebooks are the only end-to-end use of the public API, so a renamed parameter
# breaks them silently: the committed outputs still render and nothing fails until
# someone re-runs one. This executes the smallest analysis notebook against the CSVs
# already in the repository, which needs no network and catches that drift. The output
# goes to a temporary directory so the committed notebook is left untouched.
notebook-check:
	jupyter nbconvert --to notebook --execute --output-dir "$$(mktemp -d)" \
	  notebooks/titanic/analysis_multi.ipynb

# mkdocs can only put files that live under docs/ into the nav, so the notebooks are
# staged in rather than kept in two places. The copy is generated, not committed, and
# only the analysis notebooks go: the generation notebooks need network access and the
# data/ and plots/ folders are not part of the reading experience.
DOCS_NB := docs/notebooks

docs-stage:
	@rm -rf $(DOCS_NB)
	@mkdir -p $(DOCS_NB)
	@cp notebooks/README.md $(DOCS_NB)/README.md
	@for f in notebooks/*/analysis_*.ipynb; do \
	  d=$(DOCS_NB)/$$(basename $$(dirname $$f)); \
	  mkdir -p $$d; cp $$f $$d/; \
	done
	@echo "staged $$(find $(DOCS_NB) -name '*.ipynb' | wc -l | tr -d ' ') notebooks into $(DOCS_NB)"

# --strict turns a broken cross-reference or a missing docstring target into an
# error, so the docs cannot silently rot.
docs: docs-stage
	mkdocs build --strict

docs-serve: docs-stage
	mkdocs serve

build: clean
	python -m build
	python -m twine check dist/*
