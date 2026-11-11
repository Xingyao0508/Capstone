"""Builds Walmart_Pipeline_Walkthrough.pdf from Walmart_Pipeline_Walkthrough.md.
Run from the Pipeline folder:  python docs/make_diagrams.py  then  python docs/build_walkthrough_pdf.py
Supports: #/##/### headings, paragraphs, **bold**, *italic*, `code`, nested -/1. lists, | tables |, > callouts,
![caption](image) and <!-- pagebreak -->."""
import re
from pathlib import Path
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether)

DOCS = Path(__file__).resolve().parent
SRC, OUT = DOCS / "Walmart_Pipeline_Walkthrough.md", DOCS / "Walmart_Pipeline_Walkthrough.pdf"
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
for b, i, n in [(0, 0, "DV"), (1, 0, "DV-B"), (0, 1, "DV-I"), (1, 1, "DV-BI")]:
    addMapping("DV", b, i, n)

BLUE, INK, MUTED, LINE, SOFT, CODE = (colors.HexColor(c) for c in ["#2a78d6", "#0b0b0b", "#52514e", "#d9d8d4", "#f2f6fc", "#7a2e0e"])
W = 7.0 * inch
base = ParagraphStyle("base", fontName="DV", fontSize=9, leading=12.6, textColor=INK, spaceAfter=5)
ST = {
    "h1": ParagraphStyle("h1", base, fontName="DV-B", fontSize=19, leading=23, textColor=BLUE, spaceAfter=4),
    "h2": ParagraphStyle("h2", base, fontName="DV-B", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6, textColor=INK, keepWithNext=1),
    "h3": ParagraphStyle("h3", base, fontName="DV-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=BLUE, keepWithNext=1),
    "p": base,
    "quote": ParagraphStyle("q", base, backColor=SOFT, borderColor=BLUE, borderWidth=0.6, borderPadding=6, spaceBefore=5, spaceAfter=11, leftIndent=6, rightIndent=6),
    "cell": ParagraphStyle("cell", base, fontSize=7.8, leading=10, spaceAfter=0),
    "hcell": ParagraphStyle("hcell", base, fontName="DV-B", fontSize=7.8, leading=10, spaceAfter=0),
    "cap": ParagraphStyle("cap", base, fontName="DV-I", fontSize=7.8, leading=10, textColor=MUTED, spaceBefore=3, spaceAfter=8),
}


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes = []
    def keep(m):
        codes.append(m.group(1)); return f"\x00{len(codes) - 1}\x00"
    t = re.sub(r"`([^`]+)`", keep, t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: f'<font face="DVM" color="#7a2e0e">{codes[int(m.group(1))]}</font>', t)


def table(rows):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    rows = [rows[0]] + [r for r in rows[2:]]
    n = len(rows[0])
    plain = [[re.sub(r"[`*]", "", r[i]) if i < len(r) else "" for i in range(n)] for r in rows]
    lens = [max(max(min(len(r[i]), 60) for r in plain), 6) for i in range(n)]
    tok = [max(len(w) for r in plain for w in (r[i].split() or [""])) for i in range(n)]
    mins = [min(t * 4.8 + 10, 0.42 * W) for t in tok]
    if sum(mins) > W:
        mins = [m * W / sum(mins) for m in mins]
    spare = W - sum(mins)
    widths = [m + spare * l / sum(lens) for m, l in zip(mins, lens)]
    data = [[Paragraph(inline(c), ST["hcell" if ri == 0 else "cell"]) for c in r + [""] * (n - len(r))] for ri, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE), ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return [KeepTogether([t]) if len(rows) <= 12 else t, Spacer(1, 7)]


def image(cap, path):
    p = (DOCS / path).resolve()
    iw, ih = ImageReader(str(p)).getSize()
    w = W; h = w * ih / iw
    if h > 8.3 * inch:
        h = 8.3 * inch; w = h * iw / ih
    return [KeepTogether([Image(str(p), width=w, height=h), Paragraph(inline(cap), ST["cap"])])]


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    story, para, i = [], [], 0
    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), ST["p"])); para.clear()
    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if not s:
            flush(); i += 1; continue
        if s == "<!-- pagebreak -->":
            flush(); story.append(PageBreak()); i += 1; continue
        m = re.match(r"^(#{1,3}) (.*)", ln)
        if m:
            flush(); story.append(Paragraph(inline(m.group(2)), ST["h" + str(len(m.group(1)))])); i += 1; continue
        if s.startswith("|"):
            flush(); block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            tbl = table(block)
            if isinstance(tbl[0], KeepTogether) and story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h2", "h3"):
                tbl[0] = KeepTogether([story.pop()] + tbl[0]._content)
            story += tbl; continue
        m = re.match(r"^!\[(.*)\]\((.*?)\)$", s)
        if m:
            flush(); img = image(m.group(1), m.group(2))[0]
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h2", "h3"):
                img = KeepTogether([story.pop()] + img._content)
            story.append(img); i += 1; continue
        if s.startswith(">"):
            flush(); q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip()); i += 1
            story.append(Paragraph(inline(" ".join(q)), ST["quote"])); continue
        lead = len(ln) - len(ln.lstrip())
        lvl = min((lead + 1) // 3 if lead else 0, 2)
        m = re.match(r"^(-|\d+\.) (.*)", s)
        if m:
            flush()
            bt = "•" if m.group(1) == "-" else m.group(1)
            ind = 14 + lvl * 16
            st = ParagraphStyle("li", base, leftIndent=ind, bulletIndent=ind - 11, spaceAfter=2.5)
            story.append(Paragraph(inline(m.group(2)), st, bulletText=bt)); i += 1; continue
        if lead:
            flush(); story.append(Paragraph(inline(s), ParagraphStyle("ind", base, leftIndent=14 + max(lvl - 1, 0) * 16)))
            i += 1; continue
        para.append(s); i += 1
    flush()

    def deco(c, d):
        c.saveState(); c.setFont("DV", 7.2); c.setFillColor(MUTED)
        c.drawString(0.75 * inch, 0.45 * inch, "Walmart Capstone · Team 2 · Pipeline walkthrough (Steps 01–05 implemented)")
        c.drawRightString(7.75 * inch, 0.45 * inch, f"Page {d.page}"); c.restoreState()
    SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.6 * inch,
                      bottomMargin=0.7 * inch, title="Walmart Pipeline Walkthrough", author="Team 2").build(story, onFirstPage=deco, onLaterPages=deco)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
