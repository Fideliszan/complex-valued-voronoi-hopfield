"""Shared figure style: palette and the portable Times-family typography of the manuscript."""

from matplotlib import font_manager, rcParams

from cvhnn.config import ROOT

FONT_FAMILY = "Termes Figure"

INK = "#263440"
MUTED = "#52606C"
GRID = "#E6EAF0"
BLUE, TEAL, PURPLE, ORANGE = "#2864A0", "#16877A", "#8F6593", "#CB701B"

RULE_COLORS = {"classic": ORANGE, "voronoi": BLUE}
RULE_MARKERS = {"classic": "s", "voronoi": "o"}
METHOD_COLORS = {"Distance": BLUE, "Inertial score": ORANGE, "KD-tree": TEAL, "Structured": PURPLE}


def configure():
    """Register the bundled Termes Figure fonts and select them for Matplotlib output."""
    for path in (ROOT / "assets/fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
