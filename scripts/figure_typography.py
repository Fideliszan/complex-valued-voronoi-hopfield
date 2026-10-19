"""Portable Times-family typography matching the manuscript's tgtermes font."""
from pathlib import Path
from matplotlib import font_manager, rcParams

FONT_FAMILY = 'Termes Figure'

def configure():
    for path in (Path(__file__).resolve().parents[1] / 'assets/fonts').glob('*.ttf'):
        font_manager.fontManager.addfont(str(path))
    rcParams.update({'font.family': FONT_FAMILY, 'font.weight': 'normal',
                     'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
