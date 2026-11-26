"""Genera el informe editorial especulativo a 2100 de Phy-World L1 V2 (HTML + PDF, 8 páginas A4).

Lee:  reports/l1_atlas/study_2100.json
Crea: reports/l1_atlas/atlas_mundo_l1_2100_especulativo.html
      reports/l1_atlas/atlas_mundo_l1_2100_especulativo.pdf
"""
from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

import weasyprint

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import svg_charts as S  # noqa: E402

DATA_PATH = HERE / "study_2100.json"
FONTS = HERE / "fonts"
OUT_HTML = HERE / "atlas_mundo_l1_2100_especulativo.html"
OUT_PDF = HERE / "atlas_mundo_l1_2100_especulativo.pdf"

D = json.loads(DATA_PATH.read_text(encoding="utf-8"))
META = D["meta"]
SCEN = D["scenarios"]
YEARS = META["years"]  # 2019..2100 (82 años; idx 0=2019, 7=2026, 21=2040, 41=2060, 61=2080, 81=2100)

# Paleta editorial coherente para los 6 escenarios
PAL = {
    "S0_legado": "#94A3B8",      # gris pizarra (referencia legada)
    "S5_adapt_verde": "#1F7A6E", # verde petróleo (transición + adaptación)
    "S2_netzero": "#2F66D0",     # azul editorial (Net-Zero puro)
    "S1_endogeno": "#D95D24",    # naranja quemado (defecto endógeno L1 V2)
    "S3_alta_4c": "#B83227",     # carmesí (+4,2 °C)
    "S4_policrisis": "#581C3E",  # burdeos oscuro (policrisis)
}
SHORT = {
    "S0_legado": "S0 Legado",
    "S5_adapt_verde": "S5 Verde+Tech",
    "S2_netzero": "S2 Net-Zero",
    "S1_endogeno": "S1 Endógeno",
    "S3_alta_4c": "S3 +4,2 °C",
    "S4_policrisis": "S4 Policrisis",
}


def f(v, d=1, suf=""):
    return S.fmt_es(v, d, suf)


def sgn(v, d=1, suf=""):
    if v is None or not math.isfinite(v):
        return "—"
    prefix = "+" if v > 0 else ("−" if v < 0 else "")
    return prefix + S.fmt_es(abs(v), d, suf)


def font_uri(name: str) -> str:
    return (FONTS / name).resolve().as_uri()


def multi_line_chart(
    title: str,
    subtitle: str,
    series: list[dict],  # [{label, color, vals, band=(lo,hi)|None, dashed=False, width=1.8, end_text=None}]
    *,
    years: list[int] = YEARS,
    xmin: int = 2020,
    xmax: int = 2100,
    ymin: float | None = None,
    ymax: float | None = None,
    scale: float = 1.0,
    y_fmt=None,
    end_fmt=None,
    hlines: list[tuple[float, str, str]] | None = None,  # (val_scaled, label, color)
    width: float = 252,
    height: float = 132,
    dark: bool = False,
    caption: str = "",
) -> str:
    """Gráfico SVG editorial multi-escenario con etiquetas finales anti-colisión."""
    ml, mr, mt, mb = 36, 54, 14, 22
    pw, ph = width - ml - mr, height - mt - mb

    all_v = []
    for s in series:
        for v in s["vals"]:
            if v is not None and math.isfinite(v):
                all_v.append(v * scale)
        if s.get("band"):
            for arr in s["band"]:
                for v in arr:
                    if v is not None and math.isfinite(v):
                        all_v.append(v * scale)
    if ymin is None:
        ymin = min(all_v) if all_v else 0.0
    if ymax is None:
        ymax = max(all_v) if all_v else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    xfn = lambda yr: ml + (yr - xmin) / max(xmax - xmin, 1) * pw
    yfn = lambda val: mt + (ymax - val) / (ymax - ymin) * ph

    bg_grid = "#24314E" if dark else "#E7ECF2"
    txt_muted = "#94A3B8" if dark else "#7B8798"
    txt_ink = "#FFFFFF" if dark else "#0E162B"

    parts = []
    # Rejilla horizontal
    ticks = S._nice_ticks(ymin, ymax, 4)
    for tv in ticks:
        if tv < ymin - 1e-9 or tv > ymax + 1e-9:
            continue
        yy = yfn(tv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{bg_grid}" stroke-width="0.6"/>')
        lbl = y_fmt(tv) if y_fmt else S.fmt_es(tv, 1)
        parts.append(f'<text x="{ml-4}" y="{yy+2.6:.1f}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="6.4" fill="{txt_muted}">{escape(lbl)}</text>')

    # Líneas verticales de época (2040, 2060, 2080)
    for yr in (2026, 2040, 2060, 2080, 2100):
        if xmin <= yr <= xmax:
            xx = xfn(yr)
            if yr in (2040, 2060, 2080):
                parts.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="{bg_grid}" stroke-width="0.5" stroke-dasharray="2,2"/>')
            parts.append(f'<text x="{xx:.1f}" y="{mt+ph+11:.1f}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="6.4" fill="{txt_muted}">{yr}</text>')

    # Líneas horizontales de referencia (p. ej. +2.5 °C)
    if hlines:
        for hv, hlbl, hcol in hlines:
            if ymin <= hv <= ymax:
                yy = yfn(hv)
                parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{hcol}" stroke-width="0.8" stroke-dasharray="3,2"/>')
                parts.append(f'<text x="{ml+3}" y="{yy-2.5:.1f}" font-family="IBM Plex Mono, monospace" font-size="6.0" fill="{hcol}">{escape(hlbl)}</text>')

    # Bandas p10-p90 primero
    for s in series:
        if s.get("band"):
            lo, hi = s["band"]
            pts_hi, pts_lo = [], []
            for yr, a, b in zip(years, lo, hi):
                if xmin <= yr <= xmax and a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                    pts_hi.append((xfn(yr), yfn(b * scale)))
                    pts_lo.append((xfn(yr), yfn(a * scale)))
            if len(pts_hi) > 1:
                poly = S._poly_path(pts_hi) + " L " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in reversed(pts_lo)) + " Z"
                parts.append(f'<path d="{poly}" fill="{s["color"]}" fill-opacity="{0.18 if dark else 0.14}" stroke="none"/>')

    # Curvas y etiquetas finales
    end_items = []
    for s in series:
        pts = []
        last_v = None
        for yr, v in zip(years, s["vals"]):
            if xmin <= yr <= xmax and v is not None and math.isfinite(v):
                sv = v * scale
                pts.append((xfn(yr), yfn(sv)))
                last_v = sv
        if len(pts) > 1:
            dash = ' stroke-dasharray="3.2,2.2"' if s.get("dashed") else ""
            sw = s.get("width", 1.7)
            parts.append(f'<path d="{S._poly_path(pts)}" fill="none" stroke="{s["color"]}" stroke-width="{sw}" stroke-linecap="round"{dash}/>')
            parts.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.0" fill="{s["color"]}"/>')
            if last_v is not None and s.get("show_end", True):
                et = s.get("end_text") or (end_fmt(last_v) if end_fmt else S.fmt_es(last_v, 1))
                end_items.append((pts[-1][1], et, s["color"]))

    for yy, txt, col in S._deconflict_y(end_items, mt, mt + ph, min_gap=8.8):
        parts.append(
            f'<text x="{ml+pw+4.5:.1f}" y="{yy+2.5:.1f}" font-family="IBM Plex Mono, monospace" '
            f'font-size="6.4" font-weight="600" fill="{col}">{escape(txt)}</text>'
        )

    svg = f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block;overflow:visible;">{"".join(parts)}</svg>'
    if dark:
        return svg
    head = (
        f'<div class="chart-title">{escape(title)}</div>'
        f'<div class="chart-sub">{escape(subtitle)}</div>'
    )
    cap = f'<div class="chart-cap">{caption}</div>' if caption else ""
    return f'<div class="chart-box">{head}{svg}{cap}</div>'


def grouped_bar_chart(
    title: str,
    subtitle: str,
    categories: list[str],
    groups: list[tuple[str, str, list[float]]],  # (label, color, values)
    *,
    ymax: float | None = None,
    y_fmt=None,
    width: float = 252,
    height: float = 132,
    caption: str = "",
) -> str:
    """Gráfico SVG de barras agrupadas horizontales o verticales por región R10."""
    ml, mr, mt, mb = 32, 10, 14, 24
    pw, ph = width - ml - mr, height - mt - mb
    all_v = [v for _, _, vs in groups for v in vs if v is not None and math.isfinite(v)]
    if ymax is None:
        ymax = max(all_v) * 1.12 if all_v else 1.0
    yfn = lambda val: mt + (ymax - val) / max(ymax, 1e-9) * ph

    parts = []
    for tv in S._nice_ticks(0, ymax, 4):
        if tv > ymax + 1e-6:
            continue
        yy = yfn(tv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#E7ECF2" stroke-width="0.6"/>')
        lbl = y_fmt(tv) if y_fmt else S.fmt_es(tv, 0)
        parts.append(f'<text x="{ml-4}" y="{yy+2.5:.1f}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="6.2" fill="#7B8798">{escape(lbl)}</text>')

    nc = len(categories)
    ng = len(groups)
    slot = pw / max(nc, 1)
    bw = (slot * 0.72) / max(ng, 1)
    for ci, cat in enumerate(categories):
        x0 = ml + ci * slot + slot * 0.14
        for gi, (_, col, vals) in enumerate(groups):
            v = vals[ci]
            if v is None or not math.isfinite(v):
                continue
            bx = x0 + gi * bw
            by = yfn(v)
            bh = max(mt + ph - by, 0.5)
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bw-0.8, 1.5):.1f}" height="{bh:.1f}" fill="{col}" rx="0.8"/>')
        parts.append(f'<text x="{ml + (ci+0.5)*slot:.1f}" y="{mt+ph+10:.1f}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="6.2" fill="#374151">{escape(cat)}</text>')

    # Mini-leyenda arriba derecha
    lx = ml + pw - 2
    for glbl, col, _ in reversed(groups):
        parts.append(f'<text x="{lx:.1f}" y="{mt-4:.1f}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="5.9" fill="{col}" font-weight="600">■ {escape(glbl)}</text>')
        lx -= len(glbl) * 4.0 + 14

    svg = f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block;overflow:visible;">{"".join(parts)}</svg>'
    cap = f'<div class="chart-cap">{caption}</div>' if caption else ""
    return f'<div class="chart-box"><div class="chart-title">{escape(title)}</div><div class="chart-sub">{escape(subtitle)}</div>{svg}{cap}</div>'


def scenario_legend(keys=("S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis", "S0_legado")) -> str:
    items = []
    for k in keys:
        dash = "border-top:2px dashed " + PAL[k] + ";background:none;height:0;" if k == "S0_legado" else f"background:{PAL[k]};"
        items.append(f'<span class="leg-item"><span class="leg-line" style="{dash}"></span>{SHORT[k]}</span>')
    return '<div class="legend-bar">' + "".join(items) + "</div>"


def css_block() -> str:
    return f"""
@font-face {{ font-family: 'Newsreader'; src: url('{font_uri("Newsreader-Bold.ttf")}') format('truetype'); font-weight: 400 800; font-style: normal; }}
@font-face {{ font-family: 'Newsreader'; src: url('{font_uri("Newsreader-Italic.ttf")}') format('truetype'); font-weight: 400 800; font-style: italic; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('{font_uri("IBMPlexSans-Regular.ttf")}') format('truetype'); font-weight: 400; font-style: normal; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('{font_uri("IBMPlexSans-Italic.ttf")}') format('truetype'); font-weight: 400; font-style: italic; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('{font_uri("IBMPlexSans-Medium.ttf")}') format('truetype'); font-weight: 500; font-style: normal; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('{font_uri("IBMPlexSans-SemiBold.ttf")}') format('truetype'); font-weight: 600; font-style: normal; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('{font_uri("IBMPlexSans-Bold.ttf")}') format('truetype'); font-weight: 700; font-style: normal; }}
@font-face {{ font-family: 'IBM Plex Mono'; src: url('{font_uri("IBMPlexMono-Regular.ttf")}') format('truetype'); font-weight: 400; font-style: normal; }}
@font-face {{ font-family: 'IBM Plex Mono'; src: url('{font_uri("IBMPlexMono-Medium.ttf")}') format('truetype'); font-weight: 500; font-style: normal; }}

@page {{
  size: A4;
  margin: 15.5mm 15.5mm 14.5mm 15.5mm;
  @top-left {{
    content: "EL SIGLO XXI EN SEIS BIFURCACIONES · ANÁLISIS ESPECULATIVO A 2100";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.2pt; letter-spacing: 0.7px; color: #8B97A8; vertical-align: bottom; padding-bottom: 4.5mm;
  }}
  @top-right {{
    content: string(chap);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.2pt; letter-spacing: 0.7px; color: #8B97A8; vertical-align: bottom; padding-bottom: 4.5mm;
  }}
  @bottom-left {{
    content: "PHY-WORLD · CAPA L1 V2 (M0–M3 + DEMOCRACIA V2 + ÉLITES R10)";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.2pt; letter-spacing: 0.7px; color: #8B97A8; border-top: 0.5pt solid #E2E8F0; padding-top: 2.2mm;
  }}
  @bottom-right {{
    content: counter(page);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.8pt; color: #475569; border-top: 0.5pt solid #E2E8F0; padding-top: 2.2mm;
  }}
}}
@page cover {{
  margin: 0; background: #0E162B;
  @top-left {{ content: none; }} @top-right {{ content: none; }}
  @bottom-left {{ content: none; border: none; }} @bottom-right {{ content: none; border: none; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 8.3pt; line-height: 1.42; color: #242C3D; }}
.page {{ page-break-after: always; break-after: page; }}
.page:last-child {{ page-break-after: auto; break-after: auto; }}
.chap-mark {{ string-set: chap content(); height: 0; overflow: hidden; visibility: hidden; font-size: 0; line-height: 0; }}

/* Cover */
.cover-page {{
  page: cover; width: 210mm; height: 297mm; background: #0E162B; color: #FFFFFF;
  display: flex; flex-direction: column; justify-content: space-between; padding: 15mm 16mm 13mm 16mm;
}}
.cover-top {{ display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 7.2pt; letter-spacing: 1.4px; color: #8B9AB8; }}
.cover-title {{ font-family: 'Newsreader', serif; font-weight: 700; font-size: 38pt; line-height: 1.04; letter-spacing: -0.6px; margin: 7mm 0 5mm 0; color: #FFFFFF; }}
.cover-title em {{ font-style: italic; font-weight: 600; color: #E8742D; display: block; }}
.cover-sub {{ font-size: 10.2pt; line-height: 1.48; color: #B4C0D6; max-width: 155mm; margin: 0; }}
.cover-badge {{
  display: inline-block; margin-top: 4mm; padding: 1.5mm 3.5mm; border: 0.7pt solid #E8742D;
  font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; letter-spacing: 1.0px; color: #F5B58B; text-transform: uppercase;
}}
.cover-chart-box {{ margin: 3mm 0 2mm 0; background: #141F3B; border: 0.6pt solid #263556; padding: 3.5mm 4mm 2mm 4mm; border-radius: 2px; }}
.cover-chart-lbl {{ font-family: 'IBM Plex Mono', monospace; font-size: 6.5pt; letter-spacing: 0.9px; color: #94A3B8; text-transform: uppercase; margin-bottom: 1.5mm; }}
.cover-kpis {{ border-top: 0.7pt solid #2B3959; padding-top: 5mm; display: grid; grid-template-columns: repeat(3, 1fr); gap: 7mm; }}
.cover-kpi-num {{ font-family: 'Newsreader', serif; font-size: 21pt; font-weight: 700; line-height: 1.1; color: #FFFFFF; margin-bottom: 1.5mm; }}
.cover-kpi-num span {{ color: #E8742D; font-size: 13.5pt; }}
.cover-kpi-desc {{ font-size: 7.6pt; line-height: 1.38; color: #9EACC7; }}
.cover-foot {{ display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 6.6pt; color: #64748B; border-top: 0.5pt solid #1E2C4A; padding-top: 3mm; }}

/* Headings & prose */
.eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; letter-spacing: 1.1px; text-transform: uppercase; color: #D95D24; margin-bottom: 1.5mm; }}
h1 {{ font-family: 'Newsreader', serif; font-weight: 700; font-size: 20.5pt; line-height: 1.1; color: #0E162B; margin: 0 0 2.5mm 0; letter-spacing: -0.3px; }}
h2 {{ font-family: 'Newsreader', serif; font-weight: 700; font-size: 12pt; color: #0E162B; margin: 3.5mm 0 1.5mm 0; border-bottom: 0.8pt solid #0E162B; padding-bottom: 0.8mm; }}
p {{ margin: 0 0 2mm 0; }}
.lede {{ font-size: 8.8pt; line-height: 1.46; color: #334155; margin-bottom: 3mm; }}
code {{ font-family: 'IBM Plex Mono', monospace; font-size: 7.4pt; background: #F1F5F9; padding: 0.3px 2.5px; border-radius: 2px; color: #0E162B; }}
.cover-page code {{ background: rgba(255,255,255,0.12); color: #F8FAFC; font-size: 8.6pt; padding: 0.5px 3.5px; }}

/* Grids & boxes */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4.5mm; }}
.grid-2-tight {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3.5mm; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3.5mm; }}
.chart-rule {{ border-top: 1.1pt solid #0E162B; padding-top: 1.8mm; }}
.chart-title {{ font-weight: 700; font-size: 8.8pt; color: #0E162B; line-height: 1.18; }}
.chart-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 5.9pt; letter-spacing: 0.5px; color: #7B8798; text-transform: uppercase; margin: 0.6mm 0 1.2mm 0; }}
.chart-cap {{ font-size: 7.2pt; line-height: 1.32; color: #475569; margin-top: 1mm; }}

.legend-bar {{
  display: flex; flex-wrap: nowrap; gap: 3.2mm; align-items: center; padding: 1.5mm 2.5mm;
  background: #F8FAFC; border: 0.5pt solid #E2E8F0; margin-bottom: 2.5mm;
  font-family: 'IBM Plex Mono', monospace; font-size: 6.2pt; color: #334155;
}}
.leg-item {{ display: inline-flex; align-items: center; gap: 1.4mm; white-space: nowrap; }}
.leg-line {{ display: inline-block; width: 11px; height: 2.2px; border-radius: 1px; }}

.callout {{
  background: #FDF4EE; border-left: 2.5pt solid #D95D24; padding: 2.5mm 3.5mm; margin: 2.2mm 0;
  font-size: 7.9pt; line-height: 1.4;
}}
.slate-box {{
  background: #F4F6FA; border-top: 1.4pt solid #0E162B; padding: 2.5mm 3.5mm; margin: 2.2mm 0;
  font-size: 7.8pt; line-height: 1.4;
}}
.slate-box h3 {{ font-family: 'Newsreader', serif; font-size: 10pt; margin: 0 0 1.5mm 0; color: #0E162B; }}

.conf-card {{ border-top: 2pt solid #0E162B; background: #F8FAFC; padding: 2.5mm 3mm; font-size: 7.5pt; }}
.conf-card.green {{ border-color: #1F7A6E; }}
.conf-card.amber {{ border-color: #D95D24; }}
.conf-card.red {{ border-color: #B83227; }}
.conf-head {{ display: flex; justify-content: space-between; margin-bottom: 1.5mm; font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; text-transform: uppercase; }}
.conf-card ul {{ margin: 0; padding-left: 3.5mm; }}
.conf-card li {{ margin-bottom: 1.2mm; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 7.0pt; }}
th {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.0pt; text-transform: uppercase; letter-spacing: 0.4px;
  color: #475569; text-align: left; border-top: 1.2pt solid #0E162B; border-bottom: 0.8pt solid #0E162B; padding: 1.4mm 1.5mm;
}}
td {{ padding: 1.25mm 1.5mm; border-bottom: 0.45pt solid #E2E8F0; vertical-align: middle; }}
th.num, td.num {{ text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 6.5pt; white-space: nowrap; }}
.compact-tbl td {{ padding: 0.85mm 1.1mm; font-size: 6.4pt; }}
.compact-tbl th {{ padding: 1.0mm 1.1mm; font-size: 5.7pt; }}
.compact-tbl td.num {{ font-size: 6.1pt; }}
.hi-green {{ color: #1F7A6E; font-weight: 700; }}
.hi-amber {{ color: #B56200; font-weight: 700; }}
.hi-red {{ color: #B83227; font-weight: 700; }}
.hm {{ text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: 6.7pt; font-weight: 600; border-radius: 1px; }}
.hm-0 {{ background: #EEF7F5; color: #1F7A6E; }}
.hm-1 {{ background: #FEF6E4; color: #9A5800; }}
.hm-2 {{ background: #FDE7DC; color: #B84214; }}
.hm-3 {{ background: #FAD5D2; color: #991B1B; }}
"""


def hm_cell(prob: float | None) -> str:
    if prob is None or not math.isfinite(prob):
        return '<td class="hm" style="color:#94A3B8;">—</td>'
    pct = prob * 100.0
    cls = "hm-0" if pct < 15 else ("hm-1" if pct < 45 else ("hm-2" if pct < 80 else "hm-3"))
    return f'<td class="hm {cls}">{f(pct, 0)} %</td>'


def page_01_cover() -> str:
    cover_svg = multi_line_chart(
        "", "",
        [
            {"label": "S0", "color": "#64748B", "vals": SCEN["S0_legado"]["world"]["gdp_pc_world"]["p50"], "dashed": True, "width": 1.3, "end_text": "S0 31,6k"},
            {"label": "S5", "color": "#2DD4BF", "vals": SCEN["S5_adapt_verde"]["world"]["gdp_pc_world"]["p50"], "band": (SCEN["S5_adapt_verde"]["world"]["gdp_pc_world"]["p10"], SCEN["S5_adapt_verde"]["world"]["gdp_pc_world"]["p90"]), "width": 2.1, "end_text": "S5 28,2k"},
            {"label": "S2", "color": "#60A5FA", "vals": SCEN["S2_netzero"]["world"]["gdp_pc_world"]["p50"], "width": 1.8, "end_text": "S2 13,7k"},
            {"label": "S1", "color": "#F97316", "vals": SCEN["S1_endogeno"]["world"]["gdp_pc_world"]["p50"], "band": (SCEN["S1_endogeno"]["world"]["gdp_pc_world"]["p10"], SCEN["S1_endogeno"]["world"]["gdp_pc_world"]["p90"]), "width": 2.2, "end_text": "S1 10,3k"},
            {"label": "S3", "color": "#F87171", "vals": SCEN["S3_alta_4c"]["world"]["gdp_pc_world"]["p50"], "width": 1.8, "end_text": "S3 6,6k"},
            {"label": "S4", "color": "#C084FC", "vals": SCEN["S4_policrisis"]["world"]["gdp_pc_world"]["p50"], "width": 1.6, "end_text": "S4 5,0k"},
        ],
        xmin=2020, xmax=2100, ymin=0, ymax=38000,
        y_fmt=lambda v: f"{v/1000:.0f}k",
        width=500, height=192, dark=True,
    )
    return f"""
<div class="cover-page">
  <div>
    <div class="cover-top">
      <span>PHY-WORLD · MONOGRAFÍA ESPECULATIVA L1 V2</span>
      <span>HORIZONTE 2019 — 2100 · R = {META['R']} MUNDOS</span>
    </div>
    <h1 class="cover-title">El siglo XXI en<em>seis bifurcaciones</em></h1>
    <p class="cover-sub">Un análisis especulativo de largo plazo sobre Phy-World L1 V2 —180 países, capa biofísica y de colas pesadas (<code>M0–M3</code>), Democracia V2 (<code>democracy.fit</code>) y Élites Regionales R10 (<code>elites.py</code>)— explorando dónde divergen la adaptación verde, el estancamiento endógeno y el colapso civilizatorio hasta 2100.</p>
    <div class="cover-badge">Ejercicio exploratorio [M] · Fuera del horizonte validado de 15 años</div>
  </div>
  <div class="cover-chart-box">
    <div class="cover-chart-lbl">PIB per cápita mundial ponderado por población (USD PPA 2017) · Seis trayectorias 2020–2100 (medianas y bandas p10–p90 de S5 y S1)</div>
    {cover_svg}
  </div>
  <div>
    <div class="cover-kpis">
      <div>
        <div class="cover-kpi-num">+1,9 <span>a</span> +4,2 °C</div>
        <div class="cover-kpi-desc">Bifurcación térmica en 2100. Cruzar <strong>+2,5 °C</strong> hacia 2055 activa el quiebre no lineal de cosechas, hambre aguda y conflicto civil en el cinturón tropical.</div>
      </div>
      <div>
        <div class="cover-kpi-num">260 M <span>vs</span> 3,2 B</div>
        <div class="cover-kpi-desc">Mortalidad acumulada en exceso (2020–2100) entre la <strong>Transición Verde + Adaptación (S5)</strong> y la <strong>Policrisis a +4,2 °C (S4)</strong>: la diferencia es de casi 3.000 millones de vidas.</div>
      </div>
      <div>
        <div class="cover-kpi-num">2 <span>crisis</span></div>
        <div class="cover-kpi-desc">El modelo regional R10 revela dos crisis distintas: <strong>colapso biofísico-estatal en el trópico</strong> si falla el clima, y <strong>desgaste oligárquico (E/PSI) en Occidente</strong> incluso en &lt;2 °C.</div>
      </div>
    </div>
    <div class="cover-foot" style="margin-top:4mm;">
      <span>FUENTE: SIMULACIONES L1 V2 (STUDY_2100.JSON · DEMOGRAFÍA WPP EXTENDIDA)</span>
      <span>NOVIEMBRE 2026</span>
    </div>
  </div>
</div>
"""


def page_02_executive() -> str:
    rows = []
    order = ("S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis", "S0_legado")
    for k in order:
        w = SCEN[k]["world"]
        g26 = w["gdp_pc_world"]["p50"][7]
        g00 = w["gdp_pc_world"]["p50"][-1]
        dg = (g00 / g26 - 1.0) * 100.0
        cls_g = "hi-green" if dg > 10 else ("hi-amber" if dg > -30 else "hi-red")
        p2k = w["pop_under_2000_m"]["p50"][-1] if "pop_under_2000_m" in w else None
        exd = w["excess_deaths_cum_m"]["p50"][-1] if "excess_deaths_cum_m" in w else None
        pdyn = w["pop_dyn_b"]["p50"][-1] if "pop_dyn_b" in w else META["pop_flat_b"][-1]
        rows.append(f"""
      <tr>
        <td><strong style="color:{PAL[k]};">{SHORT[k]}</strong><br><span style="font-size:6.2pt;color:#64748B;">{escape(SCEN[k]['label'])}</span></td>
        <td class="num"><strong>+{f(w['Tg']['p50'][-1], 2)} °C</strong></td>
        <td class="num">{f(w['co2']['p50'][-1], 1)} Gt</td>
        <td class="num"><strong>{f(g00, 0)}</strong> <span class="{cls_g}">({sgn(dg, 0, '%')})</span></td>
        <td class="num">{f(w['gini']['p50'][-1], 2)}</td>
        <td class="num">{f(w['demo']['p50'][-1]*100, 1)} %</td>
        <td class="num"><strong>{f(w['hazard']['p50'][-1]*100, 1)} %</strong></td>
        <td class="num">{f(p2k, 0) + ' M' if p2k is not None else '—'}</td>
        <td class="num"><strong>{f(exd, 0) + ' M' if exd is not None else '—'}</strong></td>
        <td class="num">{f(pdyn, 2)} B</td>
      </tr>""")

    return f"""
<div class="page">
  <span class="chap-mark">01 · DISEÑO Y SÍNTESIS</span>
  <div class="eyebrow">Sección 01 · Arquitectura experimental y hallazgo central</div>
  <h1>Tres cuellos de botella gobiernan el siglo XXI:<br>energía, umbral de +2,5 °C y cohesión interna</h1>
  <p class="lede">Tras validar la capa L1 V2 a 15 años (2010–2025) con <strong>Democracia V2</strong>, <strong>Élites Regionales R10</strong> y <strong>Stress M0–M3</strong>, proyectamos el sistema acoplado de 180 Estados hasta <strong>2100</strong> bajo una demografía estilo ONU WPP mediana (pico de 9,54 B en 2080). El objetivo no es predecir el año 2100 con precisión decimal, sino <strong>aislar qué variables desencadenan colapsos sistémicos</strong> y qué resiste incluso en escenarios benignos.</p>

  <div class="callout">
    <strong>Descomposición clave del siglo XXI (por qué S5 y S2 divergen tanto al mismo +1,94 °C):</strong> comparar <strong>S5</strong> (28.228 USD en 2100, +50 % sobre su 2026 / +71 % sobre 2019) frente a <strong>S2</strong> (13.734 USD, −21 %) demuestra que <strong>descarbonizar el clima no basta si la economía sigue atada al agotamiento fósil logístico de Hubbert y carece de adaptación térmica/agrícola</strong>. Por el contrario, resolver ambos (S5) elimina por completo la probabilidad de contracción global y hambruna masiva (0 %), pero <strong>no detiene el alza de captura de élites (E) ni el estrés político (PSI) en Occidente</strong>.
  </div>

  <h2>Tabla maestra · Estado del mundo en 2100 por escenario (mediana de R = {META['R']} mundos)</h2>
  <table>
    <thead>
      <tr>
        <th>Escenario</th>
        <th class="num">Tg 2100</th>
        <th class="num">CO₂</th>
        <th class="num">PIB pc USD (vs 26)</th>
        <th class="num">Gini</th>
        <th class="num">Demo</th>
        <th class="num">Confl.</th>
        <th class="num">&lt;2k kcal</th>
        <th class="num">ExD acum.</th>
        <th class="num">Pob. viva</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>

  <div class="grid-3" style="margin-top:3.2mm;">
    <div class="slate-box" style="margin:0;">
      <h3>1. El bucle endógeno de S1 (+2,84 °C)</h3>
      <p>Sin transición verde, L1 V2 no llega a +4 °C por sí solo: el daño climático M1 y los choques M3 contraen el PIB mundial un −40 % hacia 2100, hundiendo las emisiones de 37 a 10 Gt/año. El calentamiento se auto-frena en +2,84 °C, pero por <strong>empobrecimiento</strong>, con 894 M de muertes en exceso y 26 % de Estados en guerra civil.</p>
    </div>
    <div class="slate-box" style="margin:0;">
      <h3>2. El abismo de +4,2 °C (S3 y S4)</h3>
      <p>Si los feedbacks de carbono (permafrost, bosques) o el uso intensivo de carbón empujan el calentamiento a +4,2 °C pese al menor PIB, el sistema cruza el quiebre agrícola de +2,5 °C tras 2055: el PIB mundial cae −62 % a −71 %, el hambre aguda alcanza a 5,6–6,0 B de personas y la población mundial se contrae a 6,1–6,8 B.</p>
    </div>
    <div class="slate-box" style="margin:0;">
      <h3>3. La paradoja de S5 (Verde + Tech)</h3>
      <p>Desacoplar la energía de Hubbert (sustitución renovable), lograr Net-Zero en 2060 y añadir 40 % de adaptación térmica + techo agrícola 1,65× genera prosperidad global (+50 % PIB vs 2026; África Subsahariana +102 % y LATAM +105 %) y salva <strong>2,2–2,9 mil millones de vidas</strong> frente a +4 °C. Aun así, el PSI en Norteamérica sube de 1,35 a 2,16.</p>
    </div>
  </div>

  <div class="grid-2" style="margin-top:2.8mm;font-size:7.6pt;color:#475569;">
    <div><strong>Diseño experimental controlado:</strong> los 5 escenarios V2 (S5, S2, S1, S3, S4) comparten exactamente las mismas semillas aleatorias (<em>Common Random Numbers</em>), el mismo modelo conductual <code>BehaviorModel</code> entrenado hasta 2019, Democracia V2 y Élites Regionales R10. Toda diferencia entre curvas es causalidad pura del escenario.</div>
    <div><strong>Demografía base vs. población viva:</strong> la línea base WPP extendida sube de 8,03 B (2026) a un pico de 9,54 B (2080) y 9,40 B (2100). La columna <em>Pob. viva</em> descuenta la mortalidad acumulada en exceso por hambrunas agudas (&lt;2.000 kcal), desastres climáticos y pandemias/choques globales de M2–M3.</div>
  </div>
</div>
"""


def page_03_climate_macro() -> str:
    keys6 = ("S0_legado", "S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis")

    def _s(var, show_band_for=("S1_endogeno",), scale=1.0, fmt=None, include_s0=True):
        out = []
        for k in keys6:
            if k == "S0_legado" and not include_s0:
                continue
            d = SCEN[k]["world"][var]
            end_v = d["p50"][-1] * scale
            et = f"{SHORT[k].split()[0]} {fmt(end_v) if fmt else f(end_v,1)}"
            out.append({
                "label": k, "color": PAL[k], "vals": d["p50"],
                "band": (d["p10"], d["p90"]) if k in show_band_for else None,
                "dashed": (k == "S0_legado"), "width": 2.0 if k in ("S5_adapt_verde", "S1_endogeno", "S3_alta_4c") else 1.4,
                "end_text": et,
            })
        return out

    c_tg = multi_line_chart(
        "Calentamiento global respecto a 1850–1900 (Tg)",
        "TEMPERATURA MEDIA SUPERFICIAL (°C) · MEDIANA Y BANDA S1",
        _s("Tg", show_band_for=("S1_endogeno", "S3_alta_4c"), fmt=lambda v: f"+{f(v,2)}°"),
        ymin=1.0, ymax=5.2,
        y_fmt=lambda v: f"+{v:.0f}°C",
        hlines=[(2.5, "Umbral no lineal M1/M2 (+2,5 °C)", "#B83227")],
        height=148,
        caption="S1 se auto-frena en +2,84 °C por caída del PIB; S3/S4 cruzan +2,5 °C hacia 2057 y alcanzan +4,22 °C [p90: +5,3 °C].",
    )
    c_co2 = multi_line_chart(
        "Emisiones mundiales de CO₂ fósil (GtCO₂/año)",
        "RESPUESTA ENDÓGENA (S1) VS TRANSICIÓN (S2/S5) VS FÓSIL (S3/S4)",
        _s("co2", show_band_for=("S1_endogeno",), fmt=lambda v: f"{f(v,0)} Gt"),
        ymin=0, ymax=76,
        y_fmt=lambda v: f"{v:.0f}",
        height=148,
        caption="En S0 (Hubbert puro) el CO₂ colapsa falsamente en 2040; en S1 cae gradualmente de 37 a 10 Gt al contraerse la economía.",
    )
    c_gdp = multi_line_chart(
        "PIB per cápita mundial (USD PPA 2017)",
        "PONDERADO POR POBLACIÓN · BRECHA ENERGÍA-ADAPTACIÓN (S5 VS S2)",
        _s("gdp_pc_world", show_band_for=("S5_adapt_verde", "S1_endogeno"), fmt=lambda v: f"{v/1000:.1f}k"),
        ymin=0, ymax=36000,
        y_fmt=lambda v: f"{v/1000:.0f}k",
        height=148,
        caption="La distancia entre S5 (28,2k) y S2 (13,7k) a igual clima (+1,9 °C) mide el coste puro del estrangulamiento Hubbert sin adaptación.",
    )
    c_gini = multi_line_chart(
        "Desigualdad internacional (Gini ponderado por población)",
        "BRECHA ENTRE PAÍSES (0 = IGUALDAD · 1 = MÁXIMA CONCENTRACIÓN)",
        _s("gini", show_band_for=("S1_endogeno",), fmt=lambda v: f(v, 2)),
        ymin=0.42, ymax=0.75,
        y_fmt=lambda v: f(v, 2),
        height=148,
        caption="Bajo +4 °C (S3/S4), el Gini salta de 0,49 a 0,70–0,71: un planeta dividido entre latitudes templadas y trópico devastado.",
    )
    return f"""
<div class="page">
  <span class="chap-mark">02 · CLIMA, ENERGÍA Y MACROECONOMÍA</span>
  <div class="eyebrow">Sección 02 · Trayectorias globales 2020–2100 (I)</div>
  <h1>De la convergencia verde (+50 %) al<br>derrumbe macroeconómico bajo +4 °C (−71 %)</h1>
  {scenario_legend()}
  <div class="grid-2" style="margin-bottom:2mm;font-size:7.9pt;">
    <p><strong>El quiebre ocurre entre 2045 y 2060.</strong> Hasta 2040, la inercia del sistema climático hace que S1, S2 y S3 evolucionen casi en paralelo. A partir de 2050, el cruce de <strong>+2,5 °C</strong> en S3/S4 desencadena pérdidas cuadráticas de PTF y de cosechas que convierten la desaceleración en contracción irreversible.</p>
    <p><strong>S5 demuestra que la sustitución energética es tan decisiva como el carbono.</strong> Incluso con Net-Zero en 2060 (S2), dejar que el agotamiento fósil de Hubbert multiplique el coste de la energía hunde el PIB mundial un −21 %; sustituirlo con renovables + adaptación térmica (S5) permite duplicar el ingreso respecto a S2 y reducir el Gini internacional.</p>
  </div>
  <div class="grid-2-tight">
    <div class="chart-rule">{c_tg}</div>
    <div class="chart-rule">{c_co2}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:2.5mm;">
    <div class="chart-rule">{c_gdp}</div>
    <div class="chart-rule">{c_gini}</div>
  </div>
  <div class="slate-box" style="margin-top:2.5mm;">
    <strong>Mecanismo de auto-frenado en S1 vs. riesgo de feedbacks en S3:</strong> en S1, L1 acopla emisiones directamente al PIB endógeno (<code>M0</code>), lo que genera un «freno malthusiano» que impide superar +3 °C salvo en la cola alta (p90: +3,32 °C). S3 y S4 modelan el riesgo opuesto que M0 no incluye endógenamente: retroalimentaciones biogeoquímicas (permafrost, incendios boreales, saturación de sumideros oceánicos) que mantienen el calentamiento rumbo a +4,2 °C aun con menor actividad económica.
  </div>
</div>
"""


def page_04_food_demo_conflict() -> str:
    keys5 = ("S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis")

    def _s5(var, band_for=("S1_endogeno",), scale=1.0, fmt=None, add_wpp=False):
        out = []
        if add_wpp:
            out.append({
                "label": "WPP", "color": "#94A3B8", "vals": META["pop_wpp_b"],
                "dashed": True, "width": 1.3, "end_text": f"WPP {f(META['pop_wpp_b'][-1],1)}B",
            })
        for k in keys5:
            d = SCEN[k]["world"][var]
            ev = d["p50"][-1] * scale
            out.append({
                "label": k, "color": PAL[k], "vals": d["p50"],
                "band": (d["p10"], d["p90"]) if k in band_for else None,
                "width": 2.0 if k in ("S5_adapt_verde", "S1_endogeno", "S3_alta_4c") else 1.5,
                "end_text": f"{SHORT[k].split()[0]} {fmt(ev) if fmt else f(ev,1)}",
            })
        return out

    c_pop = multi_line_chart(
        "Población mundial viva (miles de millones)",
        "BASE ONU WPP VS POBLACIÓN TRAS EXCESO DE MORTALIDAD M2–M3",
        _s5("pop_dyn_b", band_for=("S3_alta_4c",), fmt=lambda v: f"{f(v,1)}B", add_wpp=True),
        ymin=5.5, ymax=10.0,
        y_fmt=lambda v: f"{v:.0f}B",
        height=148,
        caption="En S3/S4 la humanidad se contrae desde un pico de ~8,9 B en 2055 hasta 6,1–6,8 B en 2100 (−2,6 a −3,3 B frente a la base WPP).",
    )
    c_fam = multi_line_chart(
        "Población en déficit calórico agudo (<2.000 kcal/día)",
        "MILLONES DE PERSONAS EN RIESGO DE HAMBRUNA AGUDA · M2",
        _s5("pop_under_2000_m", band_for=("S1_endogeno",), fmt=lambda v: f"{v/1000:.1f}B" if v >= 1000 else f"{v:.0f}M"),
        ymin=0, ymax=6500,
        y_fmt=lambda v: f"{v/1000:.0f}B" if v > 0 else "0",
        height=148,
        caption="S5 mantiene el déficit agudo cerca de cero (<150 M en 2100); en S3/S4 supera los 5.500–6.000 M por caída de cosechas tropicales.",
    )
    c_haz = multi_line_chart(
        "Incidencia global de conflicto civil (hazard)",
        "PROPORCIÓN DE ESTADOS CON CONFLICTO INTRAESTATAL ≥ 25 MUERTES (%)",
        _s5("hazard", band_for=("S1_endogeno",), scale=100.0, fmt=lambda v: f"{f(v,1)}%"),
        ymin=8, ymax=46, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        height=148,
        caption="El conflicto se estabiliza en 13,9 % en S5, se duplica al 26,2 % en S1 y alcanza el 38,5–40,9 % de los Estados en S3/S4.",
    )
    c_dem = multi_line_chart(
        "Proporción mundial de democracias · Democracia V2",
        "FRACCIÓN DE PAÍSES CON V-DEM ROW ≥ 2 (%)",
        _s5("demo", band_for=("S1_endogeno",), scale=100.0, fmt=lambda v: f"{f(v,1)}%"),
        ymin=32, ymax=60, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        height=148,
        caption="Solo S5 sostiene la democracia global por encima del 50 % (52,5 %); en S3/S4 cae al 38–41 % por retroceso económico y bélico.",
    )
    return f"""
<div class="page">
  <span class="chap-mark">03 · ALIMENTO, DEMOGRAFÍA Y FRAGILIDAD</span>
  <div class="eyebrow">Sección 03 · Trayectorias globales 2020–2100 (II)</div>
  <h1>El abismo alimentario de +3 a +4 °C:<br>contracción demográfica y fractura bélica</h1>
  {scenario_legend(("S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis"))}
  <div class="grid-2" style="margin-bottom:2mm;font-size:7.9pt;">
    <p><strong>La comida es el principal amplificador de colapso en L1 V2.</strong> Al duplicarse el precio mundial de cereales (<code>p_food</code> 1,9–2,2×) entre 2055 y 2080 en S3/S4, los países importadores de bajos ingresos quedan desplazados del mercado: el déficit agudo (&lt;2.000 kcal) dispara a la vez la mortalidad y la probabilidad de guerra civil.</p>
    <p><strong>La diferencia entre S5 y S3 es literalmente civilizatoria.</strong> En S5 (transición limpia + techo agrícola 1,65×), la humanidad atraviesa el siglo sin colapso demográfico (9,14 B vivos en 2100), con el conflicto acotado al 13,9 % y la democracia estable en 52,5 %. En S4, 2 de cada 5 países arden en conflicto simultáneamente.</p>
  </div>
  <div class="grid-2-tight">
    <div class="chart-rule">{c_pop}</div>
    <div class="chart-rule">{c_fam}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:2.5mm;">
    <div class="chart-rule">{c_haz}</div>
    <div class="chart-rule">{c_dem}</div>
  </div>
  <div class="callout" style="margin-top:2.5mm;">
    <strong>Lectura demográfica crítica:</strong> en los escenarios +4,2 °C (S3 y S4), el descenso de la población mundial de ~8,9 B (2055) a 6,1–6,8 B (2100) se concentra en un <strong>84 % en solo dos regiones: África Subsahariana (1,33–1,48 B muertes acumuladas) y Asia del Sur (0,76–1,11 B)</strong>. En Europa, Norteamérica y Japón/Corea, las calorías siguen por encima de 2.800–3.100 kcal incluso a +4,2 °C.
  </div>
</div>
"""


def page_05_regional_divergence() -> str:
    # 1. Captura de élites E por bloque regional en S1_endogeno
    r_pal = {"NAM": "#B83227", "EUR": "#2F66D0", "CHN": "#D95D24", "LAM": "#1F7A6E", "SSA": "#7C3AED", "REF": "#475569"}
    s_e = []
    for rc in ("NAM", "EUR", "CHN", "LAM", "SSA", "REF"):
        d = SCEN["S1_endogeno"]["regions"][rc]["traj"]["E"]
        s_e.append({
            "label": rc, "color": r_pal[rc], "vals": d["p50"],
            "width": 1.9 if rc in ("NAM", "CHN", "SSA") else 1.5,
            "end_text": f"{rc} {f(d['p50'][-1]*100,1)}%",
        })
    c_e_reg = multi_line_chart(
        "Captura de élites (E) por región en S1 (2020–2100)",
        "CUOTA TOP-10 % PRE-IMPUESTOS · MODELO REGIONAL R10 (ELITES.PY)",
        s_e, ymin=30, ymax=55, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%", height=144,
        caption="Divergencia estructural: NAM (43→46,7 %) y CHN (45→49,8 %) concentran renta, mientras SSA comprime desde niveles extremos (49→45,7 %).",
    )

    # 2. PSI en el Norte templado en S5 vs S1 vs S3 (NAM, REF, EUR, CHN)
    s_psi = [
        {"label": "NAM S3", "color": PAL["S3_alta_4c"], "vals": SCEN["S3_alta_4c"]["regions"]["NAM"]["traj"]["psi"]["p50"], "width": 1.9, "end_text": f"NAM S3 {f(SCEN['S3_alta_4c']['regions']['NAM']['traj']['psi']['p50'][-1],2)}"},
        {"label": "NAM S1", "color": PAL["S1_endogeno"], "vals": SCEN["S1_endogeno"]["regions"]["NAM"]["traj"]["psi"]["p50"], "width": 1.9, "end_text": f"NAM S1 {f(SCEN['S1_endogeno']['regions']['NAM']['traj']['psi']['p50'][-1],2)}"},
        {"label": "NAM S5", "color": PAL["S5_adapt_verde"], "vals": SCEN["S5_adapt_verde"]["regions"]["NAM"]["traj"]["psi"]["p50"], "width": 2.0, "end_text": f"NAM S5 {f(SCEN['S5_adapt_verde']['regions']['NAM']['traj']['psi']['p50'][-1],2)}"},
        {"label": "EUR S1", "color": PAL["S2_netzero"], "vals": SCEN["S1_endogeno"]["regions"]["EUR"]["traj"]["psi"]["p50"], "width": 1.5, "end_text": f"EUR S1 {f(SCEN['S1_endogeno']['regions']['EUR']['traj']['psi']['p50'][-1],2)}"},
        {"label": "CHN S1", "color": "#64748B", "vals": SCEN["S1_endogeno"]["regions"]["CHN"]["traj"]["psi"]["p50"], "dashed": True, "width": 1.4, "end_text": f"CHN S1 {f(SCEN['S1_endogeno']['regions']['CHN']['traj']['psi']['p50'][-1],2)}"},
    ]
    c_psi_n = multi_line_chart(
        "Estrés político (PSI) en el Norte templado",
        "ÍNDICE DE TURCHIN-GOLDSTONE · NAM EN S5/S1/S3 VS EUR Y CHN",
        s_psi, ymin=0.2, ymax=3.4,
        y_fmt=lambda v: f(v, 1), height=144,
        caption="Hallazgo central: incluso en el mejor escenario verde-tecnológico (S5), el PSI de Norteamérica sube de 1,35 a 2,16 (2,99 en S3).",
    )

    # 3. Barras agrupadas de exceso de mortalidad acumulada (M) por región R10
    r_ord = ["SSA", "SAS", "MEA", "LAM", "SEA", "CHN", "EUR", "NAM"]
    ex_s5 = [SCEN["S5_adapt_verde"]["regions"][r]["snap"]["2100"]["exd_cum_m"] for r in r_ord]
    ex_s1 = [SCEN["S1_endogeno"]["regions"][r]["snap"]["2100"]["exd_cum_m"] for r in r_ord]
    ex_s3 = [SCEN["S3_alta_4c"]["regions"][r]["snap"]["2100"]["exd_cum_m"] for r in r_ord]
    c_exd = grouped_bar_chart(
        "Mortalidad acumulada en exceso (2020–2100) por región",
        "MILLONES DE PERSONAS · COMPARACIÓN S5 VS S1 VS S3",
        r_ord,
        [("S5 Verde+Tech", PAL["S5_adapt_verde"], ex_s5), ("S1 Endógeno", PAL["S1_endogeno"], ex_s1), ("S3 +4,2 °C", PAL["S3_alta_4c"], ex_s3)],
        ymax=1450, y_fmt=lambda v: f"{v:.0f}", height=144,
        caption="La carga mortal es radicalmente asimétrica: SSA (1.328 M) y SAS (759 M) concentran el 84 % de las muertes bajo +4,2 °C.",
    )

    # 4. Barras agrupadas de conflicto civil (%) en 2100 por región R10
    hz_s5 = [SCEN["S5_adapt_verde"]["regions"][r]["snap"]["2100"]["hazard"] * 100 for r in r_ord]
    hz_s1 = [SCEN["S1_endogeno"]["regions"][r]["snap"]["2100"]["hazard"] * 100 for r in r_ord]
    hz_s3 = [SCEN["S3_alta_4c"]["regions"][r]["snap"]["2100"]["hazard"] * 100 for r in r_ord]
    c_hzr = grouped_bar_chart(
        "Riesgo de conflicto civil en 2100 por región R10",
        "PROPORCIÓN DE ESTADOS EN CONFLICTO ACTIVO (%) · S5 VS S1 VS S3",
        r_ord,
        [("S5 Verde+Tech", PAL["S5_adapt_verde"], hz_s5), ("S1 Endógeno", PAL["S1_endogeno"], hz_s1), ("S3 +4,2 °C", PAL["S3_alta_4c"], hz_s3)],
        ymax=80, y_fmt=lambda v: f"{v:.0f}%", height=144,
        caption="En S3 (+4,2 °C), más del 50 % de los Estados de SSA (74 %), SAS (58 %), MEA (55 %) y SEA (53 %) sufren guerra civil activa.",
    )

    return f"""
<div class="page">
  <span class="chap-mark">04 · DOS CRISIS REGIONALES</span>
  <div class="eyebrow">Sección 04 · La geografía asimétrica del siglo XXI</div>
  <h1>El Norte sufre desgaste oligárquico;<br>el Sur enfrenta un umbral existencial</h1>
  <div class="grid-2" style="margin-bottom:2mm;font-size:7.9pt;">
    <p><strong>Crisis 1 — Desgaste político-institucional en el Norte templado.</strong> En <code>NAM</code>, <code>EUR</code>, <code>OPA</code>, <code>REF</code> y <code>CHN</code>, el suministro calórico resiste todo el siglo y la mortalidad directa es marginal (&lt;10 M en Occidente). Su vulnerabilidad es <em>interna</em>: concentración persistente en el 10 % superior (<code>E</code>), erosión de la <em>asabiya</em> (de 0,43 a 0,27 en NAM) y escalada del índice de estrés político <code>PSI</code>.</p>
    <p><strong>Crisis 2 — Colapso biofísico-estatal en el cinturón tropical.</strong> En <code>SSA</code>, <code>SAS</code>, <code>MEA</code> y <code>SEA</code>, el destino del siglo XXI lo decide casi exclusivamente la temperatura: pasar de +1,9 °C (S5) a +4,2 °C (S3) convierte un siglo de convergencia económica (+102 % en SSA, +41 % en SEA) en un derrumbe de −75 % a −90 % del PIB y colapso estatal generalizado.</p>
  </div>
  <div class="grid-2-tight">
    <div class="chart-rule">{c_e_reg}</div>
    <div class="chart-rule">{c_psi_n}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:2.5mm;">
    <div class="chart-rule">{c_exd}</div>
    <div class="chart-rule">{c_hzr}</div>
  </div>
  <div class="slate-box" style="margin-top:2.5mm;">
    <strong>Efecto ciego en la frontera (migraciones no modeladas):</strong> L1 mantiene a las poblaciones dentro de sus fronteras nacionales. En la realidad, que <code>SSA</code>, <code>SAS</code> y <code>MEA</code> sufran déficit calórico masivo mientras <code>EUR</code> y <code>NAM</code> conservan &gt;3.000 kcal desencadenaría presiones migratorias de cientos de millones de personas, transmitiendo el choque biofísico del Sur directamente hacia el <code>PSI</code> y la política democrática de Occidente.
  </div>
</div>
"""


def page_06_regions_table() -> str:
    r_codes = ("NAM", "EUR", "OPA", "REF", "CHN", "SAS", "SEA", "LAM", "MEA", "SSA")
    rows = []
    for rc in r_codes:
        lbl = SCEN["S1_endogeno"]["regions"][rc]["label"]
        n_c = SCEN["S1_endogeno"]["regions"][rc]["n"]

        def _cell(sk):
            s26 = SCEN[sk]["regions"][rc]["snap"]["2026"]
            s00 = SCEN[sk]["regions"][rc]["snap"]["2100"]
            dg = (s00["gdp_pc"][1] / s26["gdp_pc"][1] - 1.0) * 100.0
            cg = "hi-green" if dg > 5 else ("hi-amber" if dg > -35 else "hi-red")
            return (
                f'<td class="num"><span class="{cg}">{sgn(dg,0,"%")}</span> · {f(s00["E"][1]*100,1)}% · '
                f'{f(s00["psi"],2)} · {f(s00["hazard"]*100,0)}% · {f(s00["exd_cum_m"],0)}M</td>'
            )

        s26_ref = SCEN["S1_endogeno"]["regions"][rc]["snap"]["2026"]
        s5_00 = SCEN["S5_adapt_verde"]["regions"][rc]["snap"]["2100"]
        s1_00 = SCEN["S1_endogeno"]["regions"][rc]["snap"]["2100"]
        s3_00 = SCEN["S3_alta_4c"]["regions"][rc]["snap"]["2100"]
        rows.append(f"""
      <tr>
        <td><strong>{rc}</strong> <span style="color:#64748B;">({n_c})</span><br><span style="font-size:6.1pt;color:#64748B;">{escape(lbl)}</span></td>
        <td class="num">{f(s26_ref['gdp_pc'][1],0)}</td>
        {_cell("S5_adapt_verde")}
        {_cell("S1_endogeno")}
        {_cell("S3_alta_4c")}
        <td class="num">{f(s5_00['kcal'],0)} / {f(s3_00['kcal'],0)}</td>
      </tr>""")

    # Mini-tabla de democracia y asabiya por macrobloque
    macro_rows = []
    for mc, mname in (("WEST", "Occidente (NAM+EUR+OPA · 43)"), ("ASIA_EM", "Asia Emergente (CHN+SAS+SEA · 23)"), ("SOUTH", "Sur Global (LAM+MEA+SSA · 102)"), ("REF", "Ex-URSS (REF · 12)")):
        b26 = SCEN["S1_endogeno"]["regions"][mc]["snap"]["2026"]
        b5 = SCEN["S5_adapt_verde"]["regions"][mc]["snap"]["2100"]
        b1 = SCEN["S1_endogeno"]["regions"][mc]["snap"]["2100"]
        b3 = SCEN["S3_alta_4c"]["regions"][mc]["snap"]["2100"]
        macro_rows.append(f"""
      <tr>
        <td><strong>{mname}</strong></td>
        <td class="num">{f(b26['asab'],2)} → {f(b5['asab'],2)} / {f(b1['asab'],2)} / {f(b3['asab'],2)}</td>
        <td class="num">{f(b26['demo']*100,0)}% → <strong>{f(b5['demo']*100,0)}%</strong> / {f(b1['demo']*100,0)}% / <span class="hi-red">{f(b3['demo']*100,0)}%</span></td>
        <td class="num">{f(b26['E'][1]*100,1)}% → {f(b5['E'][1]*100,1)}% / {f(b1['E'][1]*100,1)}% / {f(b3['E'][1]*100,1)}%</td>
      </tr>""")

    return f"""
<div class="page">
  <span class="chap-mark">05 · ATLAS REGIONAL R10 (2100)</span>
  <div class="eyebrow">Sección 05 · Desglose completo por las 10 regiones IAM</div>
  <h1>Radiografía regional de 2100:<br>ganadores templados, trópico en vilo</h1>
  <p class="lede">Cada celda muestra cinco indicadores en 2100 para esa región: <strong>variación de PIB pc vs 2026 · captura de élites E · PSI mediano · conflicto civil (%) · muertes en exceso acumuladas (M)</strong>. La última columna compara la disponibilidad calórica media en S5 frente a S3.</p>

  <table class="compact-tbl">
    <thead>
      <tr>
        <th>Región R10</th>
        <th class="num">PIB 26</th>
        <th class="num" style="color:{PAL['S5_adapt_verde']};">S5 Verde+Tech (+1,9 °C)<br>ΔPIB · E · PSI · Haz · ExD</th>
        <th class="num" style="color:{PAL['S1_endogeno']};">S1 Endógeno (+2,8 °C)<br>ΔPIB · E · PSI · Haz · ExD</th>
        <th class="num" style="color:{PAL['S3_alta_4c']};">S3 Alta (+4,2 °C)<br>ΔPIB · E · PSI · Haz · ExD</th>
        <th class="num">kcal 2100<br>S5 / S3</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>

  <h2 style="margin-top:2.4mm;">Cohesión (Asabiya), Democracia y Élites por macrobloque (2026 → 2100: S5 / S1 / S3)</h2>
  <table class="compact-tbl">
    <thead>
      <tr>
        <th>Macrobloque regional</th>
        <th class="num">Asabiya (2026 → S5 / S1 / S3)</th>
        <th class="num">Democracias (2026 → S5 / S1 / S3)</th>
        <th class="num">Captura de élites E (2026 → S5 / S1 / S3)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(macro_rows)}
    </tbody>
  </table>

  <div class="grid-2" style="margin-top:2.2mm;font-size:7.4pt;line-height:1.38;">
    <div>
      <p><strong>Europa (EUR) y ex-URSS (REF) como polos de resiliencia climática relativa.</strong> En S5, Europa casi duplica su PIB per cápita (+88 %, 97,4k USD) y es el único bloque occidental que amortigua la concentración de élites (E ≈ 34,5 %) y conserva más del 80 % de democracias incluso bajo +4,2 °C. <code>REF</code> gana terreno relativo por su latitud fría (+136 % en S5, +15 % en S1), pero sufre el mayor <code>PSI</code> estructural del planeta (3,0–3,4) y la <em>asabiya</em> más baja del Norte (0,23).</p>
      <p><strong>América Latina (LAM) como región bisagra.</strong> En S5 es una de las grandes ganadoras del siglo (+105 %, 36k USD, 3.559 kcal), con leve compresión de élites (51→50 %). En S3 (+4,2 °C), sin embargo, pierde más de la mitad del ingreso (−54 %) y el conflicto se quintuplica (5,8→28,5 %).</p>
    </div>
    <div>
      <p><strong>Por qué la tecnología no cura el PSI occidental.</strong> En <code>NAM</code> y <code>OPA</code>, la combinación de mayor concentración de ingresos (<code>E</code> al 46,6 % y 40,0 %) y decadencia de la <em>asabiya</em> en centros ricos (dinámica de Turchin) impulsa el <code>PSI</code> hacia 2,16–2,99 en <code>NAM</code> y 1,32–1,69 en <code>OPA</code> independientemente de las emisiones. La resiliencia climática compra margen económico, pero no sustituye reformas distributivas ni institucionales.</p>
      <p><strong>Nota sobre SAS (Asia del Sur):</strong> la caída de −13 % incluso en S5 obedece a la anomalía conocida de la bandera binaria de conflicto en India (ver pág. 8). Bajo S3 (+4,2 °C), en cambio, el colapso de <code>SAS</code> (1.666 kcal y 759 M de muertes) es un resultado biofísico genuino del calor húmedo extremo.</p>
    </div>
  </div>
</div>
"""


def page_07_tails_and_countries() -> str:
    tail_rows = []
    for k in ("S5_adapt_verde", "S2_netzero", "S1_endogeno", "S3_alta_4c", "S4_policrisis"):
        t = SCEN[k]["tails"]
        tail_rows.append(f"""
      <tr>
        <td><strong style="color:{PAL[k]};">{SHORT[k]}</strong></td>
        {hm_cell(t.get('gdp_drop_30'))}
        {hm_cell(t.get('gdp_drop_50'))}
        {hm_cell(t.get('famine_500m'))}
        {hm_cell(t.get('famine_1000m'))}
        {hm_cell(t.get('haz_gt_25'))}
        {hm_cell(t.get('haz_gt_35'))}
        {hm_cell(t.get('west_demo_lt_70'))}
        {hm_cell(t.get('demo_lt_35'))}
        {hm_cell(t.get('exd_gt_500m'))}
        {hm_cell(t.get('exd_gt_1000m'))}
      </tr>""")

    # 12 países representativos en 2100 (S5 vs S1 vs S3)
    c_names = {
        "USA": "Estados Unidos", "DEU": "Alemania", "KOR": "Corea del Sur", "CHN": "China",
        "IND": "India*", "RUS": "Rusia*", "BRA": "Brasil", "MEX": "México",
        "EGY": "Egipto", "NGA": "Nigeria", "ZAF": "Sudáfrica", "ARG": "Argentina",
    }
    c_rows = []
    for iso, cname in c_names.items():
        c26 = SCEN["S1_endogeno"]["countries"][iso]["2026"]
        c5 = SCEN["S5_adapt_verde"]["countries"][iso]["2100"]
        c1 = SCEN["S1_endogeno"]["countries"][iso]["2100"]
        c3 = SCEN["S3_alta_4c"]["countries"][iso]["2100"]

        def _fmtc(cobj):
            dg = (cobj["gdp_pc"][1] / c26["gdp_pc"][1] - 1.0) * 100.0
            cls = "hi-green" if dg > 5 else ("hi-amber" if dg > -35 else "hi-red")
            return f'<td class="num"><strong>{f(cobj["gdp_pc"][1],0)}</strong> <span class="{cls}">({sgn(dg,0,"%")})</span> · H {f(cobj["hazard"]*100,0)}%</td>'

        c_rows.append(f"""
      <tr>
        <td><strong>{iso}</strong> · {cname}</td>
        <td class="num">{f(c26['gdp_pc'][1],0)}</td>
        {_fmtc(c5)}
        {_fmtc(c1)}
        {_fmtc(c3)}
        <td class="num">{f(c1['E']*100,1)}%</td>
        <td class="num">{f(c5['kcal'],0)} / {f(c3['kcal'],0)}</td>
      </tr>""")

    return f"""
<div class="page">
  <span class="chap-mark">06 · RIESGOS DE COLA Y PAÍSES</span>
  <div class="eyebrow">Sección 06 · Probabilidades de crisis sistémica y perfiles nacionales</div>
  <h1>Mapa de riesgos de cola en 2100<br>y doce países bajo tres futuros</h1>
  <p class="lede">Cada celda del mapa de calor indica la <strong>fracción de los {META['R']} mundos Monte Carlo</strong> que cruza el umbral señalado en 2100 (o acumulado 2020–2100 en mortalidad). Verde (&lt;15 %), ámbar (15–45 %), naranja (45–80 %) y rojo (&gt;80 %).</p>

  <table>
    <thead>
      <tr>
        <th>Escenario</th>
        <th class="num" style="text-align:center;">PIB<br>&lt;−30 %</th>
        <th class="num" style="text-align:center;">PIB<br>&lt;−50 %</th>
        <th class="num" style="text-align:center;">&lt;2k kcal<br>&gt;500 M</th>
        <th class="num" style="text-align:center;">&lt;2k kcal<br>&gt;1 B</th>
        <th class="num" style="text-align:center;">Confl.<br>&gt;25 %</th>
        <th class="num" style="text-align:center;">Confl.<br>&gt;35 %</th>
        <th class="num" style="text-align:center;">Demo Occ.<br>&lt;70 %</th>
        <th class="num" style="text-align:center;">Demo W<br>&lt;35 %</th>
        <th class="num" style="text-align:center;">ExD<br>&gt;500 M</th>
        <th class="num" style="text-align:center;">ExD<br>&gt;1 B</th>
      </tr>
    </thead>
    <tbody>
      {"".join(tail_rows)}
    </tbody>
  </table>

  <h2>Doce países representativos en 2100 (PIB pc USD, variación vs 2026 y riesgo de conflicto H)</h2>
  <table class="compact-tbl">
    <thead>
      <tr>
        <th>País</th>
        <th class="num">PIB 26</th>
        <th class="num" style="color:{PAL['S5_adapt_verde']};">S5 Verde+Tech (+1,9 °C)</th>
        <th class="num" style="color:{PAL['S1_endogeno']};">S1 Endógeno (+2,8 °C)</th>
        <th class="num" style="color:{PAL['S3_alta_4c']};">S3 Alta (+4,2 °C)</th>
        <th class="num">E 2100</th>
        <th class="num">kcal S5/S3</th>
      </tr>
    </thead>
    <tbody>
      {"".join(c_rows)}
    </tbody>
  </table>

  <div class="grid-2" style="margin-top:2.4mm;font-size:7.6pt;">
    <div><strong>Asimetría dentro de Occidente y el Sur:</strong> bajo S5, <strong>Argentina (+123 %)</strong>, <strong>Sudáfrica (+104 %)</strong>, <strong>China (+103 %)</strong>, <strong>México (+80 %)</strong>, <strong>Alemania (+77 %)</strong> y <strong>Corea (+65 %)</strong> logran fuerte expansión; bajo +4,2 °C (S3), en cambio, <strong>Nigeria (−92 %, H 86 %, 1.496 kcal)</strong>, <strong>Egipto (−84 %, H 51 %, 1.917 kcal)</strong>, <strong>Brasil (−81 %, H 48 %, 2.142 kcal)</strong> y <strong>México (−66 %)</strong> retroceden varias décadas.</div>
    <div><strong>El riesgo democrático occidental en las colas:</strong> incluso en S2 (+1,9 °C sin adaptación), en el <strong>20 %</strong> de los mundos la proporción de democracias en Occidente cae por debajo del 70 % (<strong>31 %</strong> en S3/S4). En S5, esa cola de retroceso democrático occidental se reduce al <strong>6 %</strong> —confirmando que la prosperidad energética protege parcialmente a las instituciones.</div>
  </div>
</div>
"""


def page_08_limits_conclusions() -> str:
    return """
<div class="page">
  <span class="chap-mark">07 · LÍMITES Y CONCLUSIONES</span>
  <div class="eyebrow">Sección 07 · Rigor científico, puntos ciegos y hoja de ruta</div>
  <h1>Qué es señal robusta a 2100,<br>qué es supuesto [M] y qué queda fuera</h1>
  <p class="lede">Todo modelo macrohistórico proyectado 75 años fuera de muestra corre el riesgo de confundir artefactos de especificación con leyes de la historia. Separamos explícitamente la evidencia estructural sólida de los límites conocidos de la capa L1.</p>

  <div class="grid-3">
    <div class="conf-card green">
      <div class="conf-head"><strong>Señal robusta</strong><span>física + empírica</span></div>
      <ul>
        <li><strong>El umbral de +2,5 °C:</strong> la divergencia entre S5/S2 (&lt;2 °C) y S3/S4 (+4,2 °C) se repite en el 100 % de las semillas y se apoya en rendimientos agrícolas y estrés térmico no lineales.</li>
        <li><strong>Asimetría Norte-Sur:</strong> la vulnerabilidad calórica y bélica se concentra en SSA, SAS y MEA; Europa y latitudes templadas resisten mucho mejor el estrés térmico directo.</li>
        <li><strong>Desgaste político occidental:</strong> E, PSI y <em>asabiya</em> se deterioran en NAM incluso bajo crecimiento verde (S5), reflejando la inercia distributiva estimada en <code>elites.py</code>.</li>
      </ul>
    </div>
    <div class="conf-card amber">
      <div class="conf-head"><strong>Sensible a [M]</strong><span>depende de parámetros</span></div>
      <ul>
        <li><strong>Magnitud exacta del PIB en 2100:</strong> variar la sustitución energética de Hubbert y la adaptación térmica mueve el PIB de 2100 de 13,7k (S2) a 28,2k USD (S5) al mismo +1,94 °C.</li>
        <li><strong>Mortalidad acumulada exacta:</strong> los 894 M (S1) o 2,5–3,2 B (S3/S4) dependen de la elasticidad de precios de cereales (−0,35) y del umbral agudo de 2.000 kcal en <code>stress.py</code>.</li>
        <li><strong>Estabilidad democrática de largo plazo:</strong> <code>democracy.fit</code> está validado a 15 años (AUC 0,980); a 75 años extrapola reglas de transición estimadas sobre 1950–2019.</li>
      </ul>
    </div>
    <div class="conf-card red">
      <div class="conf-head"><strong>Fuera de modelo</strong><span>puntos ciegos L1</span></div>
      <ul>
        <li><strong>Migraciones climáticas masivas:</strong> no hay flujos migratorios endógenos del trópico colapsado hacia EUR/NAM, que aliviarían mortalidad tropical pero elevarían el PSI receptor.</li>
        <li><strong>Guerras entre grandes potencias y riesgo nuclear/IA:</strong> L1 solo modela conflicto intraestatal (UCDP) y choques genéricos M3, no guerras interestatales ni disrupciones por IA.</li>
        <li><strong>Artefactos locales conocidos:</strong> bandera binaria de conflicto castiga a India (*) incluso en S5; <code>Tbar = 15 °C</code> imputado sesga la trayectoria térmica de Rusia (*).</li>
      </ul>
    </div>
  </div>

  <h2>Cinco conclusiones estratégicas del ejercicio especulativo a 2100</h2>
  <div class="grid-2" style="font-size:8.0pt;line-height:1.44;margin-top:1.5mm;">
    <div>
      <p><strong>1. El colapso civilizatorio global no es inevitable, pero sí condicionalmente probable por encima de +3 °C.</strong> En S5 (Verde + Adaptación), la probabilidad de contracción global o hambruna masiva es <strong>0 %</strong>. En S3/S4 (+4,2 °C), el colapso humanitario-estatal en el trópico y la contracción demográfica mundial ocurren en el <strong>95–100 %</strong> de los mundos simulados.</p>
      <p><strong>2. Descarbonizar sin sustituir la base energética estrangula la prosperidad (S2 vs. S5).</strong> La brecha de +105 % en PIB per cápita entre S5 y S2 demuestra que una transición climática exitosa exige abundancia energética limpia (renovables, almacenamiento, nuclear) y adaptación agrícola/térmica, no solo recortar emisiones fósiles.</p>
      <p><strong>3. El siglo XXI sin transición (S1) no «explota» en 2040, sino que se desgasta en cámara lenta.</strong> El bucle endógeno PIB ↔ CO₂ frena el calentamiento en +2,84 °C, pero produce un siglo de erosión continua: −40 % de PIB pc, Gini a 0,65, conflicto al 26 % y casi 900 M de muertes en exceso acumuladas.</p>
    </div>
    <div>
      <p><strong>4. El gran hallazgo del modelo regional R10: la tecnología resuelve el trópico, pero no cura la fractura interna de Occidente.</strong> Incluso con +29 % a +88 % de PIB y cero hambrunas en S5, la captura de élites en Norteamérica sube al 46,6 %, la <em>asabiya</em> cae a 0,27 y el <code>PSI</code> supera 2,1. Estabilizar el siglo XXI requiere una doble agenda: transición biofísica global y renovación del contrato social doméstico.</p>
      <p><strong>5. Siguiente paso de fidelidad en Phy-World:</strong> sustituir las parametrizaciones exploratorias <code>[M]</code> de L1 por el acoplamiento físico completo con las capas <strong>F1</strong> (demografía por cohortes WPP), <strong>F3</strong> (sistema energético con sustitución renovable explícita), <strong>F4</strong> (clima FaIR con feedbacks de carbono) y un módulo de <strong>migración bilateral endógena</strong> que conecte las dos mitades del planeta.</p>
    </div>
  </div>

  <div class="slate-box" style="margin-top:3mm;font-size:7.3pt;color:#475569;">
    <strong>Reproducibilidad total:</strong> todos los datos, series anuales 2019–2100, probabilidades de cola y desgloses R10 proceden de <code>reports/l1_atlas/run_study_2100.py</code> → <code>reports/l1_atlas/study_2100.json</code> sobre <code>world4/core/l1_atlas/model.py</code> (modo por defecto L1 V2: <code>stress.py</code> + <code>democracy.py</code> + <code>elites.py</code>), con paridad bit a bit preservada en <code>legacy=True</code>.
  </div>
</div>
"""


def main():
    pages = [
        page_01_cover(),
        page_02_executive(),
        page_03_climate_macro(),
        page_04_food_demo_conflict(),
        page_05_regional_divergence(),
        page_06_regions_table(),
        page_07_tails_and_countries(),
        page_08_limits_conclusions(),
    ]
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>El siglo XXI en seis bifurcaciones · Análisis especulativo a 2100 · Phy-World L1 V2</title>
<style>{css_block()}</style>
</head>
<body>
{"".join(pages)}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    target = OUT_PDF
    try:
        weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(target))
    except PermissionError:
        target = HERE / "atlas_mundo_l1_2100_especulativo_v2.pdf"
        weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(target))
    print(f"PDF generado: {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
