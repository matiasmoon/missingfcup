"""The palette presets exist to be legible under colour vision deficiency, so that is
what these tests assert: the contrast ratio between each preset's two colours, measured
under normal vision and under simulated deuteranopia, protanopia and tritanopia.

Asserting the hex values instead would only restate the source. A contrast floor is the
property the presets are for, and it is what breaks if someone adjusts a colour to taste.
"""

import numpy as np
import pytest

import missingfcup as mf
from missingfcup.plots import _palette

# WCAG 2.1 minimum contrast for non-text content (1.4.11).
MIN_CONTRAST = 3.0

# Brettel/Vienot dichromacy simulation matrices, applied in linear RGB.
SIMULATIONS = {
    "deuteranopia": np.array([[0.625, 0.375, 0.0], [0.700, 0.300, 0.0], [0.0, 0.300, 0.700]]),
    "protanopia": np.array([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]]),
    "tritanopia": np.array([[0.950, 0.050, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]]),
}


def _to_rgb(hex_color):
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def _linearize(channel):
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def _relative_luminance(hex_color):
    r, g, b = _linearize(_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first, second):
    a, b = _relative_luminance(first), _relative_luminance(second)
    high, low = max(a, b), min(a, b)
    return (high + 0.05) / (low + 0.05)


def _simulate(hex_color, matrix):
    linear = np.clip(matrix @ _linearize(_to_rgb(hex_color)), 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in srgb)


ACCESSIBLE = [name for name in _palette.PALETTE_NAMES if name != "default"]


@pytest.mark.parametrize("palette", ACCESSIBLE)
@pytest.mark.parametrize("vision", ["normal", *SIMULATIONS])
def test_accessible_palettes_clear_the_contrast_floor(palette, vision):
    """Every preset except the legacy default is legible under every dichromacy."""
    missing, present = _palette.resolve(palette, None, None)
    if vision != "normal":
        missing = _simulate(missing, SIMULATIONS[vision])
        present = _simulate(present, SIMULATIONS[vision])
    ratio = _contrast(missing, present)
    assert ratio >= MIN_CONTRAST, (
        f"palette={palette!r} under {vision} gives {ratio:.2f}:1, below {MIN_CONTRAST}:1"
    )


def test_the_default_palette_is_unchanged():
    """The default is kept for continuity with every figure already published, so a
    change to it has to be deliberate rather than a side effect of editing a preset."""
    assert _palette.resolve("default", None, None) == ("#d62728", "#2ca02c")


def test_an_explicit_colour_overrides_the_preset():
    missing, present = _palette.resolve("grayscale", "#08306b", None)
    assert missing == "#08306b"
    assert present == "#d9d9d9"


def test_an_unknown_palette_names_the_valid_ones():
    with pytest.raises(ValueError, match="grayscale"):
        _palette.resolve("high-contrast", None, None)


@pytest.mark.parametrize("palette", _palette.PALETTE_NAMES)
def test_the_palette_reaches_the_figure(palette):
    """The preset has to survive the whole path from the factory method to the trace."""
    md = mf.MissingData(mf.sample_data())
    expected_missing, expected_present = _palette.resolve(palette, None, None)
    plot = md.matrix(palette=palette)
    assert plot.missing_color == expected_missing
    assert plot.present_color == expected_present
    colorscale = plot.fig.data[0].colorscale
    drawn = {str(entry[1]).lower() for entry in colorscale}
    assert expected_missing.lower() in drawn
    assert expected_present.lower() in drawn
