"""Shared color helpers for the plots.

Kept in one place so the hex-to-rgba conversion is not reimplemented per plot.
"""

from __future__ import annotations


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex (``#rgb`` or ``#rrggbb``) or ``rgb()``/``rgba()`` color to an
    ``rgba()`` string with the given ``alpha``.

    Already-``rgba()`` colors are returned unchanged. Anything that cannot be parsed
    falls back to a transparent black, so a bad color never raises inside a plot.
    """
    if not isinstance(color, str):
        return f"rgba(0,0,0,{alpha})"
    if color.startswith("rgba("):
        return color
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")

    hex_value = color.lstrip("#")
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    if len(hex_value) != 6:
        return f"rgba(0,0,0,{alpha})"
    try:
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
    except ValueError:
        return f"rgba(0,0,0,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"
