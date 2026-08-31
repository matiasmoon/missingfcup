"""Named colour pairs for the present/missing distinction every plot draws.

The default pair is red against green, which is the worst case for the two most common
forms of colour vision deficiency. Measured as a WCAG contrast ratio, the two default
colours sit at 1.48:1 against each other in normal vision and fall to 1.07:1 under
simulated tritanopia, where they are effectively one colour.

The fix is not a different pair of hues. Colour vision deficiency collapses hue while
leaving lightness intact, so a pair that differs only in hue fails however carefully the
hues are picked: the Okabe-Ito orange and blue, the usual recommendation, reach 5.29:1
under deuteranopia and 1.09:1 under tritanopia, because those two colours carry almost the
same lightness. A pair separated by *lightness* survives every deficiency, and also
survives monochrome printing, which matters because these figures end up in documents.

Both presets below clear 3:1, the WCAG floor for non-text content, under normal vision and
under simulated deuteranopia, protanopia and tritanopia. ``test_palette.py`` asserts that
rather than trusting it.

The default stays as it is on purpose. Changing it would alter every figure already
committed or published, and the point of a preset is that the caller opts in.
"""

from typing import Dict, Optional, Tuple

# missing, present
_PALETTES: Dict[str, Tuple[str, str]] = {
    # Red/green. Kept as the default for continuity, not because it reads well.
    "default": ("#d62728", "#2ca02c"),
    # Dark navy against light amber. Worst case 4.58:1, under tritanopia.
    "safe": ("#08306b", "#fed976"),
    # No hue at all, so no deficiency can affect it, and it photocopies. 7.35:1 flat.
    "grayscale": ("#404040", "#d9d9d9"),
}

PALETTE_NAMES = tuple(_PALETTES)


def resolve(
    palette: str,
    missing_color: Optional[str],
    present_color: Optional[str],
) -> Tuple[str, str]:
    """Return the (missing, present) pair for ``palette``.

    An explicit ``missing_color`` or ``present_color`` overrides the preset for that one
    colour, so a caller can take a preset and adjust half of it without restating both.
    """
    if palette not in _PALETTES:
        raise ValueError(f"palette must be one of {', '.join(PALETTE_NAMES)}, got {palette!r}")
    preset_missing, preset_present = _PALETTES[palette]
    return (
        missing_color if missing_color is not None else preset_missing,
        present_color if present_color is not None else preset_present,
    )
