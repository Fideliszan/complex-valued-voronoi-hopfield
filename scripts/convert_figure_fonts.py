from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

R = Path(__file__).resolve().parents[1]
for style in ["regular", "italic"]:
    source = R / "assets/fonts" / f"texgyretermes-{style}.otf"
    font = TTFont(source)
    gs = font.getGlyphSet()
    order = font.getGlyphOrder()
    fb = FontBuilder(font["head"].unitsPerEm, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(font.getBestCmap())
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err=0.5, reverse_direction=True))
        glyphs[name] = pen.glyph()
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(font["hmtx"].metrics)
    fb.setupHorizontalHeader(ascent=font["hhea"].ascent, descent=font["hhea"].descent)
    fb.setupNameTable(
        {
            "familyName": "Termes Figure",
            "styleName": style.title(),
            "uniqueFontIdentifier": "TermesFigure-" + style,
            "fullName": "Termes Figure " + style.title(),
            "psName": "TermesFigure-" + style.title(),
            "version": "Version 1.000",
        }
    )
    o = font["OS/2"]
    fb.setupOS2(
        sTypoAscender=o.sTypoAscender,
        sTypoDescender=o.sTypoDescender,
        sTypoLineGap=o.sTypoLineGap,
        usWinAscent=o.usWinAscent,
        usWinDescent=o.usWinDescent,
        fsSelection=1 if style == "italic" else 64,
    )
    fb.setupPost(italicAngle=font["post"].italicAngle)
    fb.setupMaxp()
    fb.font["head"].macStyle = 2 if style == "italic" else 0
    fb.save(R / "assets/fonts" / f"TermesFigure-{style}.ttf")
