"""Run every script in examples/.

The examples are the package's documentation, so they have to keep working. Executing
them here means a change that breaks an example fails the build instead of being found
by whoever reads the docs next.

``show()`` is stubbed out: the scripts render figures on purpose, and a test run must
not open browser tabs.
"""

import runpy
from pathlib import Path

import plotly.graph_objects as go
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.py"))


def test_examples_directory_is_not_empty():
    assert EXAMPLE_FILES, f"no example scripts found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("script", EXAMPLE_FILES, ids=[f.stem for f in EXAMPLE_FILES])
def test_example_runs(script, monkeypatch, tmp_path):
    # Figures must not be rendered during the test run.
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    # Run from a scratch directory, so a script that writes a file is caught below
    # rather than littering the repository.
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(script), run_name="__main__")

    written = list(tmp_path.iterdir())
    assert not written, (
        f"{script.name} wrote files ({[p.name for p in written]}); examples only show"
    )


def test_every_example_has_a_module_docstring():
    """The docstring is what makes an example readable as documentation."""
    for script in EXAMPLE_FILES:
        first = script.read_text().lstrip()
        assert first.startswith('"""'), f"{script.name} has no module docstring"
