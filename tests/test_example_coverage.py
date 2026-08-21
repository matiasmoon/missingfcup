"""Every public parameter has to appear in a runnable example.

The examples are the documentation, and `test_examples.py` already runs them. That
makes them the one place where a parameter is both described *and* proven to work, so
a parameter nobody demonstrates is a parameter nobody has checked end to end.

These tests read the example scripts rather than the docs, so they cannot pass on a
promise: the call has to be there, spelled correctly, in a file the suite executes.
"""

import ast
import inspect
from collections import defaultdict
from pathlib import Path

import pytest

from missingfcup import MissingData

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

STYLE_OPTIONS = {
    "title",
    "width",
    "height",
    "background_color",
    "text_color",
    "missing_color",
    "present_color",
    "show_legend",
    "max_label_length",
}

PLOTS = [
    "matrix",
    "bar",
    "rate",
    "totals",
    "heatmap",
    "dendrogram",
    "venn",
    "upset",
    "scatterplot",
    "density",
    "boxplot",
    "parallel_coordinates",
]


def parameters_of(plot: str) -> set:
    return {p for p in inspect.signature(getattr(MissingData, plot)).parameters if p != "self"}


def keywords_used() -> dict:
    """Map each plot to the keyword names the examples actually pass to it.

    Only executable code counts. The commented flat-function lines mirror the calls
    above them, so counting them would let a parameter be "covered" by a line that
    never runs.
    """
    used = defaultdict(set)
    for script in sorted(EXAMPLES_DIR.glob("*.py")):
        tree = ast.parse(script.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name in PLOTS:
                used[name] |= {kw.arg for kw in node.keywords if kw.arg}
                # Positional arguments count too: density(column, missing_column).
                positional = list(inspect.signature(getattr(MissingData, name)).parameters)[1:]
                used[name] |= set(positional[: len(node.args)])
    return used


@pytest.mark.parametrize("plot", PLOTS)
def test_every_non_style_parameter_appears_in_an_example(plot):
    """A plot's own options are what make it worth having, so each one is shown."""
    expected = parameters_of(plot) - STYLE_OPTIONS
    missing = sorted(expected - keywords_used()[plot])
    assert not missing, f"{plot}() has undemonstrated parameters: {missing}"


def test_every_style_option_appears_somewhere():
    """The style options are identical on every plot, so demonstrating each one once
    across the whole set is enough; repeating all nine per file would bury the
    parameter the example is actually about."""
    everything = set().union(*keywords_used().values())
    missing = sorted(STYLE_OPTIONS - everything)
    assert not missing, f"style options never demonstrated: {missing}"


def test_examples_do_not_invent_parameters():
    """A typo in an example is a typo in the documentation."""
    for plot, used in keywords_used().items():
        unknown = sorted(used - parameters_of(plot))
        assert not unknown, f"{plot}() example passes unknown parameters: {unknown}"
