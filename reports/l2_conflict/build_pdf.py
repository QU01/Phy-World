"""Construye el informe editorial PDF de 16 páginas de la capa L2 (F12: Eventos,
Conflicto Georreferenciado y Desplazamiento Humano) a partir de ``study_results.json``
mediante HTML/CSS + SVG vectorial + WeasyPrint.

Uso:
    python reports/l2_conflict/build_pdf.py
"""
from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import weasyprint
import svg_charts_l2 as S

HERE = Path(__file__).parent
FONTS = HERE.parent / "l1_atlas" / "fonts"
DATA_PATH = HERE / "study_results.json"
OUT_HTML = HERE / "atlas_eventos_l2.html"
OUT_PDF = HERE / "atlas_eventos_l2.pdf"

D = json.loads(DATA_PATH.read_text(encoding="utf-8"))
REGS = D["meta"]["regions"]  # ['africa', 'americas', 'asia', 'europe', 'mideast']
REG_LBL = {
    "africa": "África",
    "americas": "Américas",
    "asia": "Asia y Oceanía",
    "europe": "Europa",
    "mideast": "Oriente Medio",
}
REG_SHORT = {
    "africa": "África",
    "americas": "Américas",
    "asia": "Asia",
    "europe": "Europa",
    "mideast": "O. Medio",
}
COUNTRY_NAMES = {
    "UKR": "Ucrania", "SDN": "Sudán", "ISR": "Israel / Palestina*", "MMR": "Myanmar",
    "MEX": "México", "NGA": "Nigeria", "BFA": "Burkina Faso", "MLI": "Malí",
    "COD": "R. D. del Congo", "SOM": "Somalia", "ETH": "Etiopía", "SYR": "Siria",
    "AFG": "Afganistán", "YEM": "Yemen", "PAK": "Pakistán", "COL": "Colombia",
    "BRA": "Brasil", "IRQ": "Irak", "PSE": "Palestina", "SSD": "Sudán del Sur",
    "VEN": "Venezuela", "HTI": "Haití", "ECU": "Ecuador", "IND": "India",
    "RUS": "Rusia", "LBN": "Líbano", "TUR": "Turquía", "DEU": "Alemania",
    "POL": "Polonia", "EGY": "Egipto", "UGA": "Uganda", "TCD": "Chad",
    "IRN": "Irán", "JOR": "Jordania", "KEN": "Kenia", "ETH": "Etiopía",
}


def f(v, d=1, suf=""):
    return S.fmt_es(v, d, suf)


def fk(v):
    return S.fmt_k(v)


def sgn(v, d=1, suf=""):
    if v is None or not math.isfinite(v):
        return "—"
    prefix = "+" if v > 0 else ("−" if v < 0 else "")
    return prefix + S.fmt_es(abs(v), d, suf)


def font_uri(name: str) -> str:
    return (FONTS / name).resolve().as_uri()


def ym_to_float(lbl: str) -> float:
    y, m = lbl.split("-")
    return int(y) + (int(m) - 0.5) / 12.0


def legend_l2(include_prov=True, include_last12=True) -> str:
    items = [
        '<span class="leg-item"><span class="leg-line" style="background:#0E162B;"></span>UCDP GED (final)</span>',
    ]
    if include_prov:
        items.append('<span class="leg-item"><span class="leg-dash" style="border-color:#475569;"></span>2026 provisional</span>')
    items.append('<span class="leg-item"><span class="leg-dash" style="border-color:#2F66D0;"></span>Referencia hurdle</span>')
    if include_last12:
        items.append('<span class="leg-item"><span class="leg-dash" style="border-color:#64748B;"></span>last12</span>')
    items.append('<span class="leg-item"><span class="leg-band"></span>Agentes L2 (p50 y p10–p90 · exploratorio)</span>')
    items.append('<span class="leg-item"><span class="leg-shade"></span>2026-10 a 2029-09</span>')
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
  margin: 16mm 15.5mm 15mm 15.5mm;
  @top-left {{
    content: "LA CAPA DE EVENTOS DEL MUNDO, MES A MES Y CELDA A CELDA";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.3pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    vertical-align: bottom;
    padding-bottom: 4.5mm;
  }}
  @top-right {{
    content: string(chap);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.3pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    vertical-align: bottom;
    padding-bottom: 4.5mm;
  }}
  @bottom-left {{
    content: "PHY-WORLD · CAPA L2 (F12 EVENTOS Y DESPLAZAMIENTO) · EXPLORATORIO";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.2pt;
    letter-spacing: 0.65px;
    color: #8B97A8;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.3mm;
  }}
  @bottom-right {{
    content: counter(page);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.8pt;
    color: #475569;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.3mm;
  }}
}}

@page cover {{
  margin: 0;
  background: #0E162B;
  @top-left {{ content: none; }}
  @top-right {{ content: none; }}
  @bottom-left {{ content: none; border: none; }}
  @bottom-right {{ content: none; border: none; }}
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 0;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 8.4pt;
  line-height: 1.42;
  color: #242C3D;
}}
.page {{ page-break-after: always; break-after: page; }}
.page:last-child {{ page-break-after: auto; break-after: auto; }}
.chap-mark {{
  string-set: chap content();
  height: 0; overflow: hidden; visibility: hidden; font-size: 0; line-height: 0; margin: 0; padding: 0;
}}

/* COVER */
.cover-page {{
  page: cover;
  width: 210mm; height: 297mm;
  background: #0E162B; color: #FFFFFF;
  display: flex; flex-direction: column; justify-content: space-between;
  padding: 15mm 16mm 12mm 16mm;
}}
.cover-top {{
  display: flex; justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace; font-size: 7.2pt; letter-spacing: 1.3px; color: #8B9AB8;
}}
.cover-title-block {{ margin-top: 8mm; max-width: 162mm; }}
.cover-title {{
  font-family: 'Newsreader', serif; font-weight: 700; font-size: 38pt; line-height: 1.04;
  letter-spacing: -0.6px; margin: 0 0 5mm 0; color: #FFFFFF;
}}
.cover-title em {{ font-style: italic; font-weight: 600; color: #E8742D; display: block; }}
.cover-sub {{ font-size: 10.6pt; line-height: 1.46; color: #B4C0D6; max-width: 146mm; margin: 0; }}
.cover-chart-wrap {{ margin: 3mm -16mm 2mm -16mm; }}
.cover-kpis {{
  border-top: 0.7pt solid #2B3959; padding-top: 5.5mm;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8mm;
}}
.cover-kpi-num {{
  font-family: 'Newsreader', serif; font-size: 22pt; font-weight: 700; line-height: 1.1; color: #FFFFFF; margin-bottom: 1.8mm;
}}
.cover-kpi-desc {{ font-size: 7.6pt; line-height: 1.36; color: #9EACC7; }}
.cover-foot {{
  display: flex; justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace; font-size: 6.3pt; letter-spacing: 0.9px; color: #6B7C9E; margin-top: 4mm;
}}

/* TYPOGRAPHY */
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.9pt; font-weight: 500;
  letter-spacing: 1px; color: #D95D24; text-transform: uppercase; margin: 0 0 1.6mm 0;
}}
h1 {{
  font-family: 'Newsreader', serif; font-weight: 700; font-size: 22.5pt; line-height: 1.1;
  letter-spacing: -0.3px; color: #0E162B; margin: 0 0 3.2mm 0;
}}
h2 {{
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 9.6pt;
  color: #0E162B; margin: 3.4mm 0 1.6mm 0;
}}
.lede {{
  font-family: 'Newsreader', serif; font-weight: 400; font-size: 10.7pt; line-height: 1.42;
  color: #263147; margin: 0 0 4.5mm 0;
}}
p {{ margin: 0 0 2.5mm 0; }}
.muted {{ color: #64748B; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}
.badge-red {{
  display: inline-block; background: #FEE2E2; color: #991B1B; font-family: 'IBM Plex Mono', monospace;
  font-size: 6.2pt; font-weight: 500; padding: 0.5mm 1.8mm; border-radius: 1mm; letter-spacing: 0.4px;
  vertical-align: middle; margin-left: 1.5mm;
}}

/* GRIDS & CARDS */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5.2mm; }}
.grid-2-tight {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4.0mm; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4.2mm; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.8mm; }}

.kpi-card {{ border-top: 1.6pt solid #0E162B; padding-top: 2.6mm; margin-bottom: 1.6mm; }}
.kpi-num {{
  font-family: 'Newsreader', serif; font-weight: 700; font-size: 19.5pt; line-height: 1.1;
  color: #0E162B; margin-bottom: 1.2mm;
}}
.kpi-num .unit {{ font-family: 'IBM Plex Mono', monospace; font-size: 8.0pt; font-weight: 400; color: #64748B; }}
.kpi-title {{ font-weight: 600; font-size: 8.3pt; color: #0E162B; margin-bottom: 1.2mm; }}
.kpi-body {{ font-size: 7.8pt; line-height: 1.38; color: #374151; }}

.slate-box {{ background: #F4F6F9; border-radius: 2.3mm; padding: 3.4mm 4.2mm; margin: 2.8mm 0; }}
.slate-box h3 {{ font-size: 8.6pt; font-weight: 600; color: #0E162B; margin: 0 0 1.4mm 0; }}
.slate-box p {{ font-size: 7.9pt; line-height: 1.4; margin: 0 0 1.6mm 0; color: #334155; }}
.slate-box p:last-child {{ margin-bottom: 0; }}

.card-col {{ background: #F4F6F9; border-radius: 2.3mm; padding: 3.5mm 4.0mm; }}
.card-col .swatch {{ display: inline-block; width: 13px; height: 3px; vertical-align: middle; margin-right: 5px; border-radius: 1px; }}
.card-col h3 {{ font-size: 8.5pt; font-weight: 600; margin: 0 0 2mm 0; color: #0E162B; }}
.card-col p, .card-col li {{ font-size: 7.7pt; line-height: 1.38; color: #334155; }}
.card-col ul {{ margin: 1.2mm 0 0 0; padding-left: 4.2mm; }}
.card-col li {{ margin-bottom: 1.0mm; }}

.conf-card {{ border-top: 2.2pt solid #0E162B; padding-top: 2.3mm; }}
.conf-card.green {{ border-top-color: #1F7A54; }}
.conf-card.amber {{ border-top-color: #B58100; }}
.conf-card.red {{ border-top-color: #B83227; }}
.conf-head {{ display: flex; align-items: baseline; gap: 2.2mm; margin-bottom: 1.8mm; }}
.conf-head strong {{ font-size: 9.0pt; color: #0E162B; }}
.conf-head span {{ font-family: 'IBM Plex Mono', monospace; font-size: 6.0pt; color: #64748B; text-transform: uppercase; }}
.conf-card ul {{ margin: 0; padding-left: 3.6mm; }}
.conf-card li {{ font-size: 7.5pt; line-height: 1.36; color: #334155; margin-bottom: 1.4mm; }}

/* TOC */
.toc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2mm 7.5mm; margin-top: 2mm; }}
.toc-row {{
  display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 0.5pt solid #E5E9F0; padding: 1.2mm 0; font-size: 8.0pt;
}}
.toc-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; color: #94A3B8; margin-right: 2.2mm; }}
.toc-pg {{ font-family: 'IBM Plex Mono', monospace; font-size: 7.2pt; color: #475569; }}

/* LEGEND BAR */
.legend-bar {{
  display: flex; flex-wrap: nowrap; gap: 2.8mm; align-items: center;
  padding: 1.0mm 0 1.8mm 0; border-bottom: 0.6pt solid #E5E9F0; margin-bottom: 2.0mm;
  font-size: 6.4pt; color: #475569; white-space: nowrap;
}}
.leg-item {{ display: inline-flex; align-items: center; gap: 1.2mm; }}
.leg-line {{ display: inline-block; width: 11px; height: 2px; }}
.leg-dash {{ display: inline-block; width: 11px; height: 0; border-top: 1.6px dashed; }}
.leg-band {{ display: inline-block; width: 11px; height: 6px; background: rgba(217, 93, 36, 0.24); border-top: 1.7px solid #D95D24; }}
.leg-shade {{ display: inline-block; width: 9px; height: 6px; background: #F1F5F9; border: 0.5pt solid #CBD5E1; }}

/* CHARTS */
.chart-box {{ margin-bottom: 1.0mm; }}
.chart-cap {{ font-size: 6.9pt; line-height: 1.28; color: #475569; margin-top: 0.6mm; }}
.chart-rule {{ border-top: 1.0pt solid #0E162B; padding-top: 1.4mm; }}

/* TABLES */
table {{ width: 100%; border-collapse: collapse; font-size: 7.3pt; margin: 1.6mm 0 2.4mm 0; }}
thead th {{
  font-family: 'IBM Plex Mono', monospace; font-size: 5.9pt; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.35px; color: #64748B; text-align: left;
  padding: 1.4mm 1.3mm; border-top: 1.1pt solid #0E162B; border-bottom: 0.75pt solid #94A3B8;
}}
thead th.num, tbody td.num {{ text-align: right; font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
tbody td {{ padding: 1.2mm 1.3mm; border-bottom: 0.45pt solid #E6ECF2; color: #1E293B; vertical-align: top; }}
tbody tr.group-row td {{
  background: #F4F6F9; font-family: 'IBM Plex Mono', monospace; font-size: 6.2pt; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.55px; color: #0E162B; padding: 1.1mm 1.3mm;
  border-top: 0.65pt solid #CBD5E1; border-bottom: 0.55pt solid #CBD5E1;
}}
.tight-table td, .tight-table th {{ padding: 0.95mm 1.15mm; font-size: 6.8pt; }}
.tight-table thead th {{ font-size: 5.7pt; }}
.hi-orange {{ color: #D95D24; font-weight: 500; }}
.hi-blue {{ color: #2F66D0; font-weight: 500; }}
.hi-green {{ color: #1F7A54; font-weight: 500; }}
.hi-red {{ color: #B83227; font-weight: 500; }}

/* PHASE LEDGER */
.phase-row {{
  display: grid; grid-template-columns: 22mm 1fr; gap: 3.2mm;
  border-top: 0.6pt solid #D9E2EC; padding: 1.4mm 0;
}}
.phase-tag {{ font-family: 'Newsreader', serif; font-weight: 700; font-size: 11.5pt; line-height: 1.05; color: #0E162B; }}
.phase-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 5.4pt; letter-spacing: 0.35px; color: #64748B; text-transform: uppercase; margin-top: 0.4mm; }}
.phase-title {{ font-weight: 600; font-size: 8.0pt; color: #0E162B; margin-bottom: 0.4mm; }}
.phase-desc {{ font-size: 7.3pt; line-height: 1.28; color: #334155; margin-bottom: 0.7mm; }}
.phase-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3.2mm; }}
.phase-mini-lbl {{ font-family: 'IBM Plex Mono', monospace; font-size: 5.4pt; color: #64748B; text-transform: uppercase; margin-bottom: 0.25mm; }}
.phase-mini-txt {{ font-size: 7.0pt; line-height: 1.26; color: #475569; }}

.country-card {{ border-top: 0.8pt solid #CBD5E1; padding-top: 1.2mm; }}
pre.code {{
  background: #F4F6F9; border-radius: 2mm; padding: 2.6mm 3.2mm;
  font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; line-height: 1.42; color: #1E293B; margin: 1.8mm 0 2.5mm 0;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS PARA AGREGADOS Y GRÁFICOS EXTRA
# ═══════════════════════════════════════════════════════════════════════════

def sum_reg_monthly(key: str, stat: str = "p50") -> list[float]:
    L = len(D["regions"]["africa"]["live_36m"][key][stat])
    return [sum(D["regions"][r]["live_36m"][key][stat][i] for r in REGS) for i in range(L)]


def drift_bar_svg(width: float = 515, height: float = 128) -> str:
    """Gráfico comparativo de deriva anual Y1..Y5 (60m) por región (escala logarítmica)."""
    ml, mr, mt, mb = 36, 68, 26, 22
    pw, ph = width - ml - mr, height - mt - mb
    yrs_lbl = ["2025 real", "Año 1 (27)", "Año 2 (28)", "Año 3 (29)", "Año 4 (30)", "Año 5 (31)"]
    cols = {"africa": "#D95D24", "europe": "#2F66D0", "mideast": "#B83227", "asia": "#1F7A6E", "americas": "#64748B"}

    series = {}
    for r in REGS:
        d25 = D["regions"][r]["history"]["summary"]["deaths_2025"]
        ad = D["regions"][r]["drift_60m"]["annual_deaths"]
        series[r] = [d25] + [ad[f"Y{i}"]["p50"] for i in range(1, 6)]

    all_v = [v for vs in series.values() for v in vs]
    lmin = 3.0
    lmax = max(5.6, math.log10(max(all_v) + 1.0) + 0.25)
    xfn = lambda i: ml + i / 5.0 * pw
    yfn = lambda v: mt + (1.0 - (math.log10(max(v, 1000.0)) - lmin) / (lmax - lmin)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="11" font-family="IBM Plex Sans" font-size="8.8" font-weight="600" fill="#0E162B">Estrés de horizonte a 60 meses por región (escala logarítmica de muertes anuales p50)</text>')
    parts.append(f'<text x="0" y="20" font-family="IBM Plex Mono" font-size="5.9" fill="#7B8798">COMPARATIVA 2025 REAL FRENTE A AÑOS SIMULADOS 1 A 5 (2026-10 A 2031-09, R=32)</text>')

    for exp_v, lbl in ((3, "1k"), (4, "10k"), (5, "100k"), (6, "1 M")):
        if lmin <= exp_v <= lmax:
            yy = yfn(10 ** exp_v)
            parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#E7ECF2" stroke-width="0.65"/>')
            parts.append(f'<text x="{ml-5}" y="{yy+2.3:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.2" fill="#7B8798">{lbl}</text>')
    for i, lbl in enumerate(yrs_lbl):
        parts.append(f'<text x="{xfn(i):.1f}" y="{mt+ph+12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.1" fill="#7B8798">{lbl}</text>')

    end_items = []
    for r in REGS:
        vs = series[r]
        pts = [(xfn(i), yfn(v)) for i, v in enumerate(vs)]
        sw = "2.1" if r == "africa" else "1.5"
        parts.append(f'<path d="{S._poly(pts)}" fill="none" stroke="{cols[r]}" stroke-width="{sw}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.0" fill="{cols[r]}" stroke="white" stroke-width="0.5"/>')
        end_items.append((pts[-1][1], f"{REG_SHORT[r]} {fk(vs[-1])}", cols[r]))

    for yy, txt, col in S._deconflict(end_items, mt + 2, mt + ph - 2, gap=10.5):
        parts.append(f'<text x="{ml+pw+5:.1f}" y="{yy+2.2:.1f}" text-anchor="start" font-family="IBM Plex Mono" font-size="6.0" font-weight="500" fill="{col}">{escape(txt)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def coverage_disp_svg(width: float = 515, height: float = 112) -> str:
    """Barras de cobertura real p10-p90 en W13 desplazamiento y prueba 2023-25 vs nominal 80%."""
    dw = D["w13_official"]["displacement_world"]
    rows = [
        ("Refugiados · 1 año (W13, n=955)", dw["ref_1y"]["cov80"] * 100),
        ("Refugiados · 2 años (W13, n=955)", dw["ref_2y"]["cov80"] * 100),
        ("Refugiados · 3 años (W13, n=955)", dw["ref_3y"]["cov80"] * 100),
        ("Desplazados internos · 1 año (n=282)", dw["idp_1y"]["cov80"] * 100),
        ("Desplazados internos · 2 años (n=296)", dw["idp_2y"]["cov80"] * 100),
        ("Desplazados internos · 3 años (n=298)", dw["idp_3y"]["cov80"] * 100),
    ]
    ml, mr, mt, mb = 175, 42, 16, 12
    pw = width - ml - mr
    rh = (height - mt - mb) / len(rows)
    xfn = lambda pct: ml + pct / 100.0 * pw

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    x80 = xfn(80.0)
    parts.append(f'<line x1="{x80:.1f}" y1="{mt-4}" x2="{x80:.1f}" y2="{mt+len(rows)*rh}" stroke="#1F7A54" stroke-width="1.1" stroke-dasharray="2.5,2"/>')
    parts.append(f'<text x="{x80:.1f}" y="9" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" font-weight="500" fill="#1F7A54">NOMINAL 80 %</text>')
    for i, (lbl, pct) in enumerate(rows):
        y0 = mt + i * rh
        parts.append(f'<text x="{ml-6}" y="{y0+rh*0.66:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.1" fill="#0E162B">{escape(lbl)}</text>')
        parts.append(f'<rect x="{ml}" y="{y0+2:.1f}" width="{pw:.1f}" height="{rh-4:.1f}" rx="1.2" fill="#F1F5F9"/>')
        bw = max(2.0, pct / 100.0 * pw)
        parts.append(f'<rect x="{ml}" y="{y0+2:.1f}" width="{bw:.1f}" height="{rh-4:.1f}" rx="1.2" fill="#B83227" opacity="0.85"/>')
        parts.append(f'<text x="{ml+bw+4:.1f}" y="{y0+rh*0.68:.1f}" font-family="IBM Plex Mono" font-size="6.5" font-weight="500" fill="#B83227">{f(pct, 0)} %</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 16 PÁGINAS DEL INFORME EDITORIAL L2
# ═══════════════════════════════════════════════════════════════════════════

def page_01_cover() -> str:
    hero = S.cover_l2_hero_svg(D)
    m = D["meta"]
    w13c = D["w13_official"]["world_country"]
    d25 = D["world_annual"]["deaths"][36]
    y1_ag = sum(D["regions"][r]["live_36m"]["totals_12m"]["agents"]["p50"] for r in REGS)
    y3_ag = sum(D["regions"][r]["drift_60m"]["annual_deaths"]["Y3"]["p50"] for r in REGS)
    return f"""
<div class="page cover-page">
  <div>
    <div class="cover-top">
      <span>PHY-WORLD · CAPA L2 (F12) · EVENTOS Y DESPLAZAMIENTO</span>
      <span>INFORME · NOVIEMBRE DE 2026</span>
    </div>
    <div class="cover-title-block">
      <h1 class="cover-title">La capa de eventos del mundo,<em>mes a mes y celda a celda</em></h1>
      <p class="cover-sub">La capa L2 de Phy-World —191 países, 64.859 celdas PRIO-GRID y 2.059 díadas armadas— auditada contra UCDP GED hasta 2025, proyectada a 36 meses (2026–2029) y sometida a estrés de deriva a 60 meses.</p>
    </div>
  </div>

  <div class="cover-chart-wrap">{hero}</div>

  <div>
    <div class="cover-kpis">
      <div>
        <div class="cover-kpi-num">{f(m['C_total'], 0)}</div>
        <div class="cover-kpi-desc">Celdas de 0,5° PRIO-GRID en 191 países, 2.059 díadas y {f(m['events_total'], 0)} eventos reales de violencia organizada (1989–2026, {f(m['deaths_total']/1e6, 2)} M de muertes).</div>
      </div>
      <div>
        <div class="cover-kpi-num">ROJO <span style="color:#E8742D;font-size:13pt;">W13</span></div>
        <div class="cover-kpi-desc">Veredicto de validación mundial (2015–2024): empata a 1 mes ({f(w13c['1']['agents'], 3)} vs {f(w13c['1']['last12'], 3)}), pero a 36 meses su CRPS es un +59 % peor que <em>last12</em>.</div>
      </div>
      <div>
        <div class="cover-kpi-num">{fk(y1_ag)} → {fk(y3_ag)}</div>
        <div class="cover-kpi-desc">Suma ilustrativa de medianas regionales anuales (Año 1 a Año 3, frente a {fk(d25)} muertes reales en 2025): muestra cómo la iteración sin fatiga se desboca en África.</div>
      </div>
    </div>
    <div class="cover-foot">
      <span>5 REGIONES UCDP · PRUEBAS CIEGAS 2023–2025 · 64 MIEMBROS · 4 ABLACIONES</span>
      <span>ESTADO DECLARADO: EXPLORATORIO (JUNTO A REFERENCIA HURDLE)</span>
    </div>
  </div>
</div>
"""


def page_02_executive() -> str:
    w13c = D["w13_official"]["world_country"]
    w13g = D["w13_official"]["world_cell"]
    d25 = D["world_annual"]["deaths"][36]
    d26 = D["world_annual"]["deaths"][37]
    ref25 = D["world_annual"]["ref_stock"][36]
    idp25 = D["world_annual"]["idp_stock"][36]
    br = D["w13_official"]["bridge_mideast"]
    af_y1 = D["regions"]["africa"]["drift_60m"]["annual_deaths"]["Y1"]["p50"]
    af_y5 = D["regions"]["africa"]["drift_60m"]["annual_deaths"]["Y5"]["p50"]
    return f"""
<div class="page">
  <span class="chap-mark">EN UNA PÁGINA</span>
  <div class="eyebrow">En una página</div>
  <h1>Qué mide la capa L2, dónde falla<br>y por qué no se proyecta a 15 años</h1>
  <p class="lede">Mientras L0 mira un siglo (1950–2100) y L1 quince años (2026–2041), la capa <strong>L2 (F12)</strong> baja al terreno operativo: simula mes a mes cómo 2.059 díadas armadas combaten, se desplazan sobre una malla mundial de 64.859 celdas de 0,5° y generan refugiados y desplazados internos. Este informe la somete a pruebas ciegas retrospectivas hasta diciembre de 2025, sigue sus 36 meses operativos (2026-10 a 2029-09) y la lleva al límite a 60 meses para exponer su mecánica.</p>

  <div class="grid-2">
    <div class="kpi-card">
      <div class="kpi-num">{f(w13c['1']['agents'], 3)} <span class="unit">vs {f(w13c['1']['last12'], 3)}</span></div>
      <div class="kpi-title">Empate a 1 mes, divergencia a 24–36 meses (W13 ROJO)</div>
      <div class="kpi-body">En 20 orígenes mundiales (2015–2024), el simulador de agentes empata por país a 1 mes con <code>last12</code> y <code>hurdle</code>, pero acumula error recursivo hasta quedar +59 % peor a 36 meses ({f(w13c['36']['agents'], 3)} vs {f(w13c['36']['last12'], 3)}). Por celda pierde en los cuatro horizontes ({f(w13g['12']['agents'], 4)} vs {f(w13g['12']['hurdle'], 4)} a 12m).</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{fk(d25)} <span class="unit">muertes en 2025</span></div>
      <div class="kpi-title">Un mundo en máximo reciente de violencia y desplazamiento</div>
      <div class="kpi-body">El GED 26.1 cierra 2025 con {f(d25, 0)} muertes violentas (África {fk(D['regions']['africa']['history']['summary']['deaths_2025'])}, Europa {fk(D['regions']['europe']['history']['summary']['deaths_2025'])}, Oriente Medio {fk(D['regions']['mideast']['history']['summary']['deaths_2025'])}), más {fk(d26)} provisionales en ene–sep 2026, con {f(ref25/1e6, 1)} M de refugiados y {f(idp25/1e6, 1)} M de desplazados internos registrados.</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{f(br['l2_jan'], 3)} <span class="unit">Brier anual</span></div>
      <div class="kpi-title">Acierta mejor si habrá violencia que cuánta — y sirve de puente</div>
      <div class="kpi-body">Su gran virtud es la <em>presencia</em>: en Brier de «≥1 muerte» a 1 mes empata o gana en África, Américas y Oriente Medio, y su puente anual <strong>L2 → L1</strong> (≥25 muertes estatales) mejora la persistencia ({f(br['l2_jan'], 3)} frente a {f(br['pers'], 3)}).</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{fk(af_y1)} → {fk(af_y5)}</div>
      <div class="kpi-title">Sin stock de fatiga, África se desboca tras 24 meses</div>
      <div class="kpi-body">La prueba de estrés a 60 meses muestra por qué L2 <strong>no puede</strong> proyectarse a 15 años: iterar 5 años con cientos de díadas vivas sin límite de efectivos ni recursos multiplica la violencia simulada en África y vacía el sentido de las bandas de desplazamiento (cobertura real 22–38 %).</div>
    </div>
  </div>

  <div class="slate-box">
    <h3>Regla de honestidad científica (W11 / F12) <span class="badge-red">W13 ROJO · EXPLORATORIO</span></h3>
    <p>Ninguna salida de L2 tiene grado de pronóstico operativo. Por política explícita de Phy-World, <strong>todo pronóstico de agentes se etiqueta como exploratorio</strong> y se presenta siempre junto a las referencias estadísticas (<code>hurdle</code> y <code>last12</code>). Los datos de enero–septiembre de 2026 son eventos <em>Candidate Clear</em> (provisionales y subcontados) y nunca se usan como verdad de prueba. Las cinco regiones corren de forma independiente: sus bandas no se suman en un total mundial probabilístico.</p>
  </div>

  <h2>Contenido del informe</h2>
  <div class="toc-grid">
    <div class="toc-row"><span><span class="toc-num">01</span>Qué se prueba y escalas de L2</span><span class="toc-pg">03</span></div>
    <div class="toc-row"><span><span class="toc-num">06</span>Mapa subnacional 0,5° y 12 teatros</span><span class="toc-pg">09–10</span></div>
    <div class="toc-row"><span><span class="toc-num">02</span>Arquitectura: díadas, huella y huida</span><span class="toc-pg">04</span></div>
    <div class="toc-row"><span><span class="toc-num">07</span>Estrés a 60 meses y 4 ablaciones</span><span class="toc-pg">11</span></div>
    <div class="toc-row"><span><span class="toc-num">03</span>Lo que ya pasó: W13 y pruebas 2023–25</span><span class="toc-pg">05</span></div>
    <div class="toc-row"><span><span class="toc-num">08</span>Mapa completo de parámetros L2</span><span class="toc-pg">12–13</span></div>
    <div class="toc-row"><span><span class="toc-num">04</span>Geografía de la violencia (2018–2029)</span><span class="toc-pg">06–07</span></div>
    <div class="toc-row"><span><span class="toc-num">09</span>Incertidumbre y anatomía del fallo</span><span class="toc-pg">14</span></div>
    <div class="toc-row"><span><span class="toc-num">05</span>Desplazamiento humano (ACNUR/IDMC)</span><span class="toc-pg">08</span></div>
    <div class="toc-row"><span><span class="toc-num">10</span>Conclusiones, límites y Apéndice</span><span class="toc-pg">15–16</span></div>
  </div>
</div>
"""


def page_03_scope() -> str:
    return """
<div class="page">
  <span class="chap-mark">01 · QUÉ SE PRUEBA Y ESCALAS</span>
  <div class="eyebrow">Capítulo 01 · Diseño del experimento</div>
  <h1>Tres escalas temporales y por qué<br>L2 se prueba a meses, no a décadas</h1>
  <p class="lede">Cada capa de Phy-World responde a una pregunta física distinta. L0 mide techos biofísicos a un siglo; L1, economía política a 15 años; L2, dinámica mensual de actores armados a 1–36 meses. Forzar L2 a 60 meses revela exactamente dónde muere el horizonte predictivo de un modelo de díadas sin stocks de fuerza.</p>

  <div class="grid-3">
    <div class="card-col">
      <h3><span class="swatch" style="background:#0E162B;"></span>1. Datos reales (1989–2026)</h3>
      <p>Cuatro fuentes públicas enlazadas sobre una misma malla de sitios <code>(celda, país)</code>:</p>
      <ul>
        <li><strong>UCDP GED 26.1:</strong> eventos georreferenciados 1989-01 a 2025-12 con tipo de violencia y muertes (<code>best</code>).</li>
        <li><strong>UCDP Candidate 26.0.X:</strong> ene–sep 2026 (solo <code>code_status = Clear</code>; provisional y subcontado).</li>
        <li><strong>PRIO-GRID 2.0:</strong> 64.859 celdas de 0,5°, población, distancias, relieve y fronteras.</li>
        <li><strong>ACNUR + IDMC + WPP:</strong> stocks anuales de refugiados (por par origen–asilo) y desplazados internos hasta 2025.</li>
      </ul>
    </div>
    <div class="card-col">
      <h3><span class="swatch" style="background:#2F66D0;"></span>2. Pruebas ciegas (sin fuga)</h3>
      <p>Cada corte aplica <code>Panel.truncated(t₀)</code> para borrar toda información posterior al origen:</p>
      <ul>
        <li><strong>W13 mundial (2015–2024):</strong> 20 orígenes por región (100 corridas), país 1–36m y celda 1–12m frente a <code>hurdle</code>, <code>last12</code> y <code>zero</code>.</li>
        <li><strong>Prueba A (2023-01 → 2025-12):</strong> 36 meses ciegos (<em>R=48</em>) que ponen a prueba choques exógenos como Sudán y Gaza.</li>
        <li><strong>Prueba B (2025-01 → 2025-12):</strong> 12 meses ciegos sobre el último año cerrado del GED, país por país.</li>
      </ul>
    </div>
    <div class="card-col">
      <h3><span class="swatch" style="background:#D95D24;"></span>3. Proyección y estrés</h3>
      <p>Dos horizontes futuros desde el último mes con datos (<code>2026-09</code>) y cuatro experimentos causales:</p>
      <ul>
        <li><strong>Operativo 36m (2026-10 → 2029-09):</strong> <em>R=64</em> miembros por región, con huella espacial 0,5°, desplazamiento y puente anual L2 → L1.</li>
        <li><strong>Estrés 60m (hasta 2031-09):</strong> <em>R=32</em> para medir la deriva compuesta de díadas activas y muertes.</li>
        <li><strong>4 ablaciones (CRN):</strong> sin nacimientos, sin difusión vecinal, sin suelo espacial y con letalidad reciente (24m).</li>
      </ul>
    </div>
  </div>

  <h2>Cómo leer cada gráfico de este informe</h2>
  <div class="slate-box">
    <div class="grid-2">
      <div>
        <p><strong>Línea negra sólida:</strong> registro definitivo de UCDP GED 26.1 (hasta diciembre de 2025) o stocks oficiales de ACNUR/IDMC. La <strong>línea gris discontinua</strong> en 2026 marca eventos <em>Candidate Clear</em>: siempre subestiman el total final por retraso de revisión.</p>
        <p><strong>Línea azul discontinua (<code>hurdle</code>) y gris punteada (<code>last12</code>):</strong> las dos varas estadísticas. <code>hurdle</code> ajusta un logit de ocurrencia y una regresión de log-muertes directamente a cada horizonte; <code>last12</code> repite el promedio de los 12 meses previos.</p>
      </div>
      <div>
        <p><strong>Línea y banda naranja (Agentes L2):</strong> mediana (<code>p50</code>) y banda interdecil (<code>p10–p90</code>) del simulador multiagente. Toda curva naranja es <em>exploratoria</em>.</p>
        <p><strong>Métrica decisiva (CRPS de ln(1+muertes)):</strong> evalúa a la vez exactitud y dispersión probabilística en escala logarítmica para que una guerra masiva no borre el error en 180 países pacíficos. <em>Menor siempre es mejor.</em></p>
      </div>
    </div>
  </div>

  <h2>Por qué las cinco regiones no se suman probabilísticamente</h2>
  <p style="font-size:8.0pt;color:#334155;">UCDP organiza el mundo en cinco regiones (<code>africa</code>, <code>americas</code>, <code>asia</code>, <code>europe</code>, <code>mideast</code>). En F12 cada región se estima y simula en un panel separado: un actor con frentes en dos regiones (como el Estado Islámico o potencias que intervienen en el exterior) no acopla sus frentes entre regiones, y los sorteos Monte Carlo son independientes entre sí. Cuando este informe muestra sumas ilustrativas de medianas regionales para dar escala global, lo declara explícitamente y nunca construye bandas mundiales sumando percentiles.</p>
</div>
"""


def page_04_architecture() -> str:
    m = D["meta"]
    rows_html = []
    for r in REGS:
        s = D["regions"][r]["history"]["summary"]
        rows_html.append(
            f"<tr><td><strong>{REG_LBL[r]}</strong></td>"
            f"<td class='num'>{f(s['K'],0)}</td><td class='num'>{f(s['C'],0)}</td>"
            f"<td class='num'>{f(s['S'],0)}</td><td class='num'>{f(s['D'],0)}</td>"
            f"<td class='num'>{f(s['actors'],0)}</td><td class='num'>{f(s['events'],0)}</td>"
            f"<td class='num'>{f(s['deaths_total'],0)}</td><td class='num hi-orange'>{f(s['deaths_2025'],0)}</td></tr>"
        )
    d25_tot = sum(D["regions"][r]["history"]["summary"]["deaths_2025"] for r in REGS)
    s_tot = sum(D["regions"][r]["history"]["summary"]["S"] for r in REGS)
    return f"""
<div class="page">
  <span class="chap-mark">02 · ARQUITECTURA DE L2</span>
  <div class="eyebrow">Capítulo 02 · Qué hay dentro de la capa L2</div>
  <h1>Cinco reglas mensuales: actores,<br>díadas, huella 0,5° y desplazamiento</h1>
  <p class="lede">A diferencia de un modelo estadístico por país, L2 simula la microestructura del conflicto: parejas de actores (díadas) que se activan, libran un número estocástico de eventos, eligen celdas vecinas dentro de su país y empujan bloques de población a huir o retornar.</p>

  <div class="phase-row">
    <div><div class="phase-tag">P1</div><div class="phase-sub">ACTIVAR DÍADA</div></div>
    <div>
      <div class="phase-title">Logit agrupado de combate mensual por díada viva (11 rasgos)</div>
      <div class="phase-desc">Cada díada viva (callada menos de 120 meses) decide si combate este mes según su memoria reciente (<code>own1, own3, own12</code>), meses de silencio (<code>since</code>), edad, actividad de ambos actores en <em>sus otros frentes</em> (<code>side_a, side_b</code>), violencia del resto del país (<code>ctx</code>) y tipo (estatal, no estatal, unilateral contra civiles o interestatal).</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Hallazgo empírico</div><div class="phase-mini-txt"><code>side_a</code> y <code>side_b</code> son positivos en las 5 regiones: un actor ocupado en otro frente combate <em>más</em>, no menos (escalada por impulso).</div></div>
        <div><div class="phase-mini-lbl">Límite declarado</div><div class="phase-mini-txt">No existen stocks de efectivos ni munición: nada frena a una díada por agotamiento físico.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">P2</div><div class="phase-sub">INTENSIDAD</div></div>
    <div>
      <div class="phase-title">Regresión log-lineal de eventos + remuestreo de residuos y letalidad</div>
      <div class="phase-desc">Si la díada se activa, el logaritmo del número de eventos sigue una regresión con los mismos rasgos más residuos empíricos remuestreados (con techo <code>n_cap = 2 × máx. histórico</code>). Cada evento toma su letalidad (muertes y fracción civil) del depósito pasado de la propia díada, encogido hacia su tipo de violencia por <code>m / (m + 10)</code> cuando tiene poca historia.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">Preserva las colas pesadas reales de letalidad sin suponer normalidad.</div></div>
        <div><div class="phase-mini-lbl">Punto débil</div><div class="phase-mini-txt">Sin peso reciente, una guerra de hace 20 años contamina la letalidad si la díada sigue viva.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">P3</div><div class="phase-sub">HUELLA 0,5°</div></div>
    <div>
      <div class="phase-title">Memoria espacial con difusión intranacional y suelo de país (4 parámetros MLE)</div>
      <div class="phase-desc">Cada díada mantiene una huella sobre los sitios <code>(celda, país)</code> que decae (<code>decay</code>), se difunde a las 8 celdas vecinas <em>sin cruzar fronteras internacionales</em> (<code>diffuse</code>) y añade un suelo (<code>floor</code>) repartido entre historia violenta del país y población (<code>mix</code>). Los 4 parámetros se estiman por máxima verosimilitud a 1 paso.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Arreglo clave en desarrollo</div><div class="phase-mini-txt">Bloquear la difusión transfronteriza bajó el CRPS de Oriente Medio de 0,461 a 0,342 (evitó derramar Gaza sobre Jordania).</div></div>
        <div><div class="phase-mini-lbl">Eventos sin localizar</div><div class="phase-mini-txt">Cada país tiene un sitio «sin celda» que conserva la suma nacional exacta (coherencia país = suma de sitios).</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">P4</div><div class="phase-sub">NACIMIENTOS</div></div>
    <div>
      <div class="phase-title">Proceso de Poisson por país para nuevas díadas armadas y muerte a 120 meses</div>
      <div class="phase-desc">Las díadas nuevas nacen según <code>ln λ = c₀ + c₁·ln(1+ev₁₂) + c₂·ln(pop)</code>, heredan tipo y papeles de nacimientos previos en ese país y toman como semilla espacial el suelo del país. Si una díada acumula 120 meses sin eventos, sale del conjunto activo.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Riesgo de horizonte</div><div class="phase-mini-txt">Como <code>c₁ &gt; 0</code>, más eventos generan más díadas nuevas, creando un bucle amplificador a partir del mes 24.</div></div>
        <div><div class="phase-mini-lbl">Puente L2 → L1</div><div class="phase-mini-txt">Agrega muertes estatales anuales por país para calcular <code>P(≥25 muertes)</code> compatible con L1.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">P5</div><div class="phase-sub">DESPLAZAMIENTO</div></div>
    <div>
      <div class="phase-title">Bloques de población por celda: exposición vecinal, huida, frontera y retorno</div>
      <div class="phase-desc">La exposición (muertes/100k en la celda y sus 8 vecinas) acumula memoria <code>φ</code>; huye la fracción <code>1 − exp(−β·mult_k·m^η)</code>, cruza la frontera según distancia <code>σ(g₀ + g₁·z)</code> (el resto son desplazados internos), elige país de asilo proporcional al stock previo y retorna en calma a ritmo <code>ρ·exp(−E/E₀)</code>.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Calibración</div><div class="phase-mini-txt">8 parámetros comunes + multiplicador por país ajustados sobre stocks anuales de ACNUR/IDMC en escala <code>asinh(pers/10k)</code>.</div></div>
        <div><div class="phase-mini-lbl">Límites declarados</div><div class="phase-mini-txt">Solo violencia organizada (no clima/economía), sin refugiados UNRWA y sin muertes indirectas por hambre.</div></div>
      </div>
    </div>
  </div>

  <h2 style="margin-top:2.2mm;">Inventario del panel mundial L2 (1989-01 a 2026-09 · 453 meses)</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Región UCDP</th><th class="num">Países</th><th class="num">Celdas 0,5°</th><th class="num">Sitios</th><th class="num">Díadas</th><th class="num">Actores</th><th class="num">Eventos</th><th class="num">Muertes 89–26</th><th class="num">Muertes 2025</th></tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
      <tr class="group-row">
        <td>MUNDO (5 REGIONES)</td>
        <td class="num">{f(m['K_total'],0)}</td><td class="num">{f(m['C_total'],0)}</td>
        <td class="num">{f(s_tot,0)}</td><td class="num">{f(m['D_total'],0)}</td>
        <td class="num">{f(m['actors_total'],0)}</td><td class="num">{f(m['events_total'],0)}</td>
        <td class="num">{f(m['deaths_total'],0)}</td><td class="num hi-orange">{f(d25_tot,0)}</td>
      </tr>
    </tbody>
  </table>
</div>
"""


def page_05_backtest() -> str:
    w13c = D["w13_official"]["world_country"]
    w13g = D["w13_official"]["world_cell"]
    w13r = D["w13_official"]["by_region"]
    steps_c = [1, 3, 6, 12, 24, 36]
    ch_c = S.horizon_crps_chart(
        "W13 mundial · Nivel país (1 a 36 meses)",
        "CRPS DE ln(1+MUERTES) · 20 ORÍGENES 2015–2024 · MENOR ES MEJOR",
        steps_c,
        [w13c[str(s)]["agents"] for s in steps_c],
        [w13c[str(s)]["hurdle"] for s in steps_c],
        [w13c[str(s)]["last12"] for s in steps_c],
        height=136,
    )
    steps_g = [1, 3, 6, 12]
    ch_g = S.horizon_crps_chart(
        "W13 mundial · Nivel celda 0,5° (1 a 12 meses)",
        "1,30 MILLONES DE CELDAS-ORIGEN · AGENTES PIERDE EN TODOS LOS PASOS",
        steps_g,
        [w13g[str(s)]["agents"] for s in steps_g],
        [w13g[str(s)]["hurdle"] for s in steps_g],
        [w13g[str(s)]["last12"] for s in steps_g],
        height=136,
    )
    hbar_rows = [
        (REG_LBL[r], w13r[r]["c36"][1], w13r[r]["c36"][0], f(w13r[r]["c36"][1], 3), f(w13r[r]["c36"][0], 3))
        for r in REGS
    ]
    ch_bar = S.paired_hbar(hbar_rows, label_a="Mejor referencia a 36m (last12 / hurdle)", label_b="Simulador de agentes L2 a 36m", row_h=19.5)

    # Table combining W13 (2015-24), blind 2023->25 (36m), and blind 2025 (12m)
    bt23 = D["world_backtest_2023"]["country"]
    bt25 = D["world_backtest_2025"]["country"]
    rows_t = []
    for r in REGS:
        t25 = D["regions"][r]["backtest_2025_12m"]["total_2025"]
        t23 = D["regions"][r]["backtest_2023_36m"]["cumulative_36m"]
        c25 = D["regions"][r]["backtest_2025_12m"]["country"]
        rows_t.append(
            f"<tr><td><strong>{REG_LBL[r]}</strong></td>"
            f"<td class='num'>{f(w13r[r]['c1'][0],3)} / {f(w13r[r]['c1'][1],3)}</td>"
            f"<td class='num'>{f(w13r[r]['c12'][0],3)} / {f(w13r[r]['c12'][1],3)}</td>"
            f"<td class='num'>{fk(t23['obs'])}</td><td class='num hi-orange'>{fk(t23['agents']['p50'])}</td>"
            f"<td class='num'>{fk(t25['obs'])}</td><td class='num hi-orange'>{fk(t25['agents']['p50'])}</td>"
            f"<td class='num hi-blue'>{fk(t25['hurdle']['p50'])}</td>"
            f"<td class='num'>{f(c25['agents']['12']['brier1'],3)} / {f(c25['hurdle']['12']['brier1'],3)}</td></tr>"
        )
    return f"""
<div class="page">
  <span class="chap-mark">03 · LO QUE YA PASÓ (W13 Y 2023–2025)</span>
  <div class="eyebrow">Capítulo 03 · Confrontación ciega contra datos reales</div>
  <h1>Empate al mes siguiente, brecha<br>creciente a uno y tres años</h1>
  <p class="lede">Someter L2 a pruebas estrictamente ciegas (<code>P.truncated</code>) deja un diagnóstico sin ambigüedades: el simulador de agentes aguanta el pulso a 1 mes por país, pero su error se acelera más deprisa que <code>hurdle</code> y <code>last12</code> al iterar año tras año. Solo Europa mejora la referencia a 12 y 36 meses.</p>

  <div class="grid-2-tight chart-rule">
    <div>{ch_c}</div>
    <div>{ch_g}</div>
  </div>

  <h2 style="margin-top:2mm;">Brecha por región a 36 meses en W13 (CRPS de ln(1+muertes), país)</h2>
  {ch_bar}

  <h2 style="margin-top:1.5mm;">Marcador completo: W13 (2015–2024), prueba ciega 2023–2025 (36m) y último año cerrado 2025</h2>
  <table class="tight-table">
    <thead>
      <tr>
        <th>Región</th>
        <th class="num">W13 1m (Ag/Ref)</th>
        <th class="num">W13 12m (Ag/Ref)</th>
        <th class="num">Real 23–25</th>
        <th class="num">Ag 23–25</th>
        <th class="num">Real 2025</th>
        <th class="num">Ag 2025</th>
        <th class="num">Hurdle 25</th>
        <th class="num">Brier1 25 (Ag/Hu)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_t)}
      <tr class="group-row">
        <td>MUNDO (PONDERADO / SUMAS)</td>
        <td class="num">{f(w13c['1']['agents'],3)} / {f(w13c['1']['last12'],3)}</td>
        <td class="num">{f(w13c['12']['agents'],3)} / {f(w13c['12']['last12'],3)}</td>
        <td class="num">{fk(sum(D['regions'][r]['backtest_2023_36m']['cumulative_36m']['obs'] for r in REGS))}</td>
        <td class="num hi-orange">{fk(sum(D['regions'][r]['backtest_2023_36m']['cumulative_36m']['agents']['p50'] for r in REGS))}</td>
        <td class="num">{fk(sum(D['regions'][r]['backtest_2025_12m']['total_2025']['obs'] for r in REGS))}</td>
        <td class="num hi-orange">{fk(sum(D['regions'][r]['backtest_2025_12m']['total_2025']['agents']['p50'] for r in REGS))}</td>
        <td class="num hi-blue">{fk(sum(D['regions'][r]['backtest_2025_12m']['total_2025']['hurdle']['p50'] for r in REGS))}</td>
        <td class="num">{f(bt25['agents']['12']['brier1'],3)} / {f(bt25['hurdle']['12']['brier1'],3)}</td>
      </tr>
    </tbody>
  </table>

  <div class="grid-2" style="font-size:7.6pt;color:#334155;">
    <p><strong>Prueba 2023 → 2025 (el examen de choques):</strong> arrancando en enero de 2023 —antes de la guerra civil de Sudán (abr-2023) y del 7 de octubre en Gaza— ningún modelo podía anticipar dos estallidos inéditos. En la prueba de 2025 (ya viendo 2024), Oriente Medio queda clavado ({fk(D['regions']['mideast']['backtest_2025_12m']['total_2025']['agents']['p50'])} vs {fk(D['regions']['mideast']['backtest_2025_12m']['total_2025']['obs'])} reales) y Américas muy cerca ({fk(D['regions']['americas']['backtest_2025_12m']['total_2025']['agents']['p50'])} vs {fk(D['regions']['americas']['backtest_2025_12m']['total_2025']['obs'])}).</p>
    <p><strong>Por qué pierde L2 en magnitud:</strong> en la prueba ciega de 2025, Europa proyecta {fk(D['regions']['europe']['backtest_2025_12m']['total_2025']['agents']['p50'])} muertes frente a {fk(D['regions']['europe']['backtest_2025_12m']['total_2025']['obs'])} reales por sobremuestreo de eventos extremos en Ucrania, mientras en W13 36m África dobla el CRPS de <code>last12</code> (0,931 vs 0,414). En cambio, en <strong>Brier1</strong> de 2025 ({f(bt25['agents']['12']['brier1'],3)} vs {f(bt25['hurdle']['12']['brier1'],3)}), L2 discrimina presencia casi tan bien como <code>hurdle</code>.</p>
  </div>
</div>
"""


def _reg_series_pack(reg: str):
    """Build (xs_obs, y_obs), (xs_prov, y_prov), (xs_f, p10, p50, p90), (xs_f, hu), (xs_f, l12) monthly 2018-2029."""
    hm = D["regions"][reg]["history"]["monthly_2018_2026"]
    xs_h = [ym_to_float(l) for l in hm["labels"]]
    # Final GED ends at 2025-12 -> index 95 (8 years * 12 = 96 months: 0..95)
    xs_obs, y_obs = xs_h[:96], hm["deaths"][:96]
    xs_prov, y_prov = xs_h[95:], hm["deaths"][95:]
    lv = D["regions"][reg]["live_36m"]
    xs_f = [ym_to_float(l) for l in lv["labels"]]
    ag = (xs_f, lv["deaths_monthly"]["p10"], lv["deaths_monthly"]["p50"], lv["deaths_monthly"]["p90"])
    hu = (xs_f, lv["hurdle_monthly"]["p50"])
    l12 = (xs_f, [lv["last12_monthly"]] * len(xs_f))
    return (xs_obs, y_obs), (xs_prov, y_prov), ag, hu, l12


def page_06_geography_1() -> str:
    xticks = [(2018.5, "18"), (2020.5, "20"), (2022.5, "22"), (2024.5, "24"), (2026.5, "26"), (2028.5, "28")]
    vlines = [(2026.0, "26*"), (2026.75, "")]

    # 1. World illustrative monthly sum (declared!)
    hm_lbls = D["regions"]["africa"]["history"]["monthly_2018_2026"]["labels"]
    xs_h = [ym_to_float(l) for l in hm_lbls]
    w_m = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["deaths"][i] for r in REGS) for i in range(len(xs_h))]
    lv_lbls = D["regions"]["africa"]["live_36m"]["labels"]
    xs_f = [ym_to_float(l) for l in lv_lbls]
    w_p50 = sum_reg_monthly("deaths_monthly", "p50")
    w_p10 = sum_reg_monthly("deaths_monthly", "p10")
    w_p90 = sum_reg_monthly("deaths_monthly", "p90")
    w_hu = sum_reg_monthly("hurdle_monthly", "p50")
    w_l12 = [sum(D["regions"][r]["live_36m"]["last12_monthly"] for r in REGS)] * len(xs_f)
    c1 = S.ts_l2_chart(
        "Mundo · Suma ilustrativa mensual", "MUERTES/MES (SUMA DE REGIONES INDEPENDIENTES)",
        obs=(xs_h[:96], w_m[:96]), prov=(xs_h[95:], w_m[95:]),
        agents_band=(xs_f, w_p10, w_p50, w_p90), hurdle=(xs_f, w_hu), last12=(xs_f, w_l12),
        xmin=2018.0, xmax=2029.8, ymax=max(max(w_m[:96]) * 1.15, max(w_p50) * 1.35),
        xticks=xticks, vlines=vlines, height=118,
        caption="La mediana sumada pasa de ~20k/mes reales en 2025 a escalar desde 2028 por la deriva africana; hurdle se mantiene estable.",
    )

    # 2. Africa
    o, pr, ag, hu, l12 = _reg_series_pack("africa")
    c2 = S.ts_l2_chart(
        "África (53 países · 1.092 díadas)", "MUERTES/MES · MAYOR MOTOR DE DERIVA DE L2",
        obs=o, prov=pr, agents_band=ag, hurdle=hu, last12=l12,
        xmin=2018.0, xmax=2029.8, ymax=max(max(o[1]) * 1.15, max(ag[2]) * 1.35),
        xticks=xticks, vlines=vlines, height=118,
        caption="Arranca baja por el subconteo 2026-Clear y luego acelera en 2028–2029 por nacimientos y frentes acoplados.",
    )

    # 3. Middle East
    o, pr, ag, hu, l12 = _reg_series_pack("mideast")
    c3 = S.ts_l2_chart(
        "Oriente Medio (15 países · 265 díadas)", "MUERTES/MES · PICO GAZA/LEVANTE 2023–2025",
        obs=o, prov=pr, agents_band=ag, hurdle=hu, last12=l12,
        xmin=2018.0, xmax=2029.8, ymax=max(max(o[1]) * 1.12, max(ag[3]) * 1.05),
        xticks=xticks, vlines=vlines, height=118,
        caption="Agentes L2 y hurdle convergen a una meseta pos-pico (~750–1.500 muertes/mes) tras el descenso 2025–2026.",
    )

    # 4. Violence by type: State-based
    w_sb_h = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["state_based"][i] for r in REGS) for i in range(len(xs_h))]
    c4 = S.ts_l2_chart(
        "Violencia estatal (gobierno vs. grupo)", "TIPO 1 UCDP · MUERTES/MES (MUNDO)",
        obs=(xs_h[:96], w_sb_h[:96]), prov=(xs_h[95:], w_sb_h[95:]),
        agents_band=(xs_f, sum_reg_monthly("state_based_monthly", "p10"), sum_reg_monthly("state_based_monthly", "p50"), sum_reg_monthly("state_based_monthly", "p90")),
        xmin=2018.0, xmax=2029.8, ymax=max(max(w_sb_h[:96]) * 1.15, max(sum_reg_monthly("state_based_monthly", "p50")) * 1.35),
        xticks=xticks, vlines=vlines, height=118,
        caption="Concentra >75 % de las muertes mundiales (Ucrania, Sudán, Gaza, Sahel, Myanmar) y domina la escalada de 2028–29.",
    )

    # 5. Non-state violence
    w_ns_h = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["non_state"][i] for r in REGS) for i in range(len(xs_h))]
    c5 = S.ts_l2_chart(
        "Violencia no estatal (milicias y cárteles)", "TIPO 2 UCDP · MUERTES/MES (MUNDO)",
        obs=(xs_h[:96], w_ns_h[:96]), prov=(xs_h[95:], w_ns_h[95:]),
        agents_band=(xs_f, sum_reg_monthly("non_state_monthly", "p10"), sum_reg_monthly("non_state_monthly", "p50"), sum_reg_monthly("non_state_monthly", "p90")),
        xmin=2018.0, xmax=2029.8, ymax=max(max(w_ns_h[:96]) * 1.2, max(sum_reg_monthly("non_state_monthly", "p50")) * 1.4),
        xticks=xticks, vlines=vlines, height=118,
        caption="Dominada por México, Brasil y milicias africanas; L2 captura su persistencia crónica a 12–24 meses.",
    )

    # 6. One-sided violence against civilians
    w_os_h = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["one_sided"][i] for r in REGS) for i in range(len(xs_h))]
    c6 = S.ts_l2_chart(
        "Violencia unilateral contra civiles", "TIPO 3 UCDP · MUERTES/MES (MUNDO)",
        obs=(xs_h[:96], w_os_h[:96]), prov=(xs_h[95:], w_os_h[95:]),
        agents_band=(xs_f, sum_reg_monthly("one_sided_monthly", "p10"), sum_reg_monthly("one_sided_monthly", "p50"), sum_reg_monthly("one_sided_monthly", "p90")),
        xmin=2018.0, xmax=2029.8, ymax=max(max(w_os_h[:96]) * 1.2, max(sum_reg_monthly("one_sided_monthly", "p50")) * 1.4),
        xticks=xticks, vlines=vlines, height=118,
        caption="RDC, Sudán y el Sahel concentran la mayor parte de los ataques unilaterales contra población civil.",
    )

    return f"""
<div class="page">
  <span class="chap-mark">04 · GEOGRAFÍA DE LA VIOLENCIA (I)</span>
  <div class="eyebrow">Capítulo 04 · Parte I · Mundo, África, Oriente Medio y tipos de violencia</div>
  <h1>El choque de 2022–2025 y cómo<br>diverge L2 entre 2027 y 2029</h1>
  {legend_l2()}
  <div class="grid-2-tight">
    {c1}{c2}{c3}{c4}{c5}{c6}
  </div>
</div>
"""


def page_07_geography_2() -> str:
    xticks = [(2018.5, "18"), (2020.5, "20"), (2022.5, "22"), (2024.5, "24"), (2026.5, "26"), (2028.5, "28")]
    vlines = [(2026.0, "26*"), (2026.75, "")]
    hm_lbls = D["regions"]["africa"]["history"]["monthly_2018_2026"]["labels"]
    xs_h = [ym_to_float(l) for l in hm_lbls]
    lv_lbls = D["regions"]["africa"]["live_36m"]["labels"]
    xs_f = [ym_to_float(l) for l in lv_lbls]

    # 1. Europe
    o, pr, ag, hu, l12 = _reg_series_pack("europe")
    c1 = S.ts_l2_chart(
        "Europa (44 países · 86 díadas)", "MUERTES/MES · DOMINADA POR UCRANIA",
        obs=o, prov=pr, agents_band=ag, hurdle=hu, last12=l12,
        xmin=2018.0, xmax=2029.8, ymax=max(max(o[1]) * 1.12, max(ag[2]) * 1.35),
        xticks=xticks, vlines=vlines, height=118,
        caption="Única región donde L2 superó a las referencias en W13 12–36m; sin agotamiento militar, prolonga el frente ucraniano.",
    )

    # 2. Asia & Oceania
    o, pr, ag, hu, l12 = _reg_series_pack("asia")
    c2 = S.ts_l2_chart(
        "Asia y Oceanía (44 países · 354 díadas)", "MUERTES/MES · MYANMAR, PAKISTÁN Y SUR DE ASIA",
        obs=o, prov=pr, agents_band=ag, hurdle=hu, last12=l12,
        xmin=2018.0, xmax=2029.8, ymax=max(max(o[1]) * 1.15, max(ag[2]) * 1.35),
        xticks=xticks, vlines=vlines, height=118,
        caption="Tras la salida de Afganistán (2021), Myanmar y Pakistán sostienen la violencia; en Año 3 agentes repunta por nacimientos.",
    )

    # 3. Americas
    o, pr, ag, hu, l12 = _reg_series_pack("americas")
    c3 = S.ts_l2_chart(
        "Américas (35 países · 262 díadas)", "MUERTES/MES · VIOLENCIA CRIMINAL ORGANIZADA",
        obs=o, prov=pr, agents_band=ag, hurdle=hu, last12=l12,
        xmin=2018.0, xmax=2029.8, ymax=max(max(o[1]) * 1.15, max(ag[3]) * 1.1),
        xticks=xticks, vlines=vlines, height=118,
        caption="México, Brasil, Colombia y Haití fijan una meseta criminal; el subconteo provisional de 2026 arrastra el arranque de L2.",
    )

    # 4. Active dyads per month (world sum)
    dy_h = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["active_dyads"][i] for r in REGS) for i in range(len(xs_h))]
    dy_p50 = sum_reg_monthly("active_dyads_monthly", "p50")
    dy_p10 = sum_reg_monthly("active_dyads_monthly", "p10")
    dy_p90 = sum_reg_monthly("active_dyads_monthly", "p90")
    c4 = S.ts_l2_chart(
        "Díadas activas cada mes (mundo)", "PAREJAS DE ACTORES EN COMBATE SIMULTÁNEO",
        obs=(xs_h[:96], dy_h[:96]), prov=(xs_h[95:], dy_h[95:]),
        agents_band=(xs_f, dy_p10, dy_p50, dy_p90),
        xmin=2018.0, xmax=2029.8, y_fmt=lambda v: f(v, 0),
        xticks=xticks, vlines=vlines, height=118,
        caption="El motor de la deriva: las díadas activas/mes saltan de ~125 reales a más de " + f(dy_p50[-1], 0) + " en 2029.",
    )

    # 5. Active countries >=1 death per month (world sum)
    ac_h = [sum(D["regions"][r]["history"]["monthly_2018_2026"]["active_countries_1"][i] for r in REGS) for i in range(len(xs_h))]
    ac_p50 = sum_reg_monthly("active_countries_1", "p50")
    ac_p10 = sum_reg_monthly("active_countries_1", "p10")
    ac_p90 = sum_reg_monthly("active_countries_1", "p90")
    c5 = S.ts_l2_chart(
        "Países con ≥1 muerte mensual", "EXTENSIÓN GEOGRÁFICA DEL CONFLICTO (DE 191)",
        obs=(xs_h[:96], ac_h[:96]), prov=(xs_h[95:], ac_h[95:]),
        agents_band=(xs_f, ac_p10, ac_p50, ac_p90),
        xmin=2018.0, xmax=2029.8, y_fmt=lambda v: f(v, 0),
        xticks=xticks, vlines=vlines, height=118,
        caption="Pasa de ~45 países/mes en 2025 hacia " + f(ac_p50[-1], 0) + " en 2029 por nacimientos de díadas en países con historia previa.",
    )

    # 6. Active 0.5° cells per month (projected 2026-2029)
    ce_p50 = sum_reg_monthly("active_cells", "p50")
    ce_p10 = sum_reg_monthly("active_cells", "p10")
    ce_p90 = sum_reg_monthly("active_cells", "p90")
    c6 = S.ts_l2_chart(
        "Celdas 0,5° activas por mes (2026–2029)", "DIFUSIÓN SUBNACIONAL SOBRE 64.859 CELDAS",
        agents_band=(xs_f, ce_p10, ce_p50, ce_p90),
        xmin=2026.7, xmax=2029.8, y_fmt=lambda v: f(v, 0),
        xticks=[(2027.0, "2027"), (2028.0, "2028"), (2029.0, "2029")],
        shade_from=2026.75, height=118,
        caption="La huella difusiva expande el número de celdas golpeadas cada mes de " + f(ce_p50[0], 0) + " a " + f(ce_p50[-1], 0) + ".",
    )

    return f"""
<div class="page">
  <span class="chap-mark">04 · GEOGRAFÍA DE LA VIOLENCIA (II)</span>
  <div class="eyebrow">Capítulo 04 · Parte II · Europa, Asia, Américas y dinámica interna</div>
  <h1>Tres regiones más acotadas y el<br>bucle de proliferación de díadas</h1>
  {legend_l2()}
  <div class="grid-2-tight">
    {c1}{c2}{c3}{c4}{c5}{c6}
  </div>
</div>
"""


def page_08_displacement() -> str:
    wa = D["world_annual"]
    yrs_d = [y for y in wa["years"] if 1995 <= y <= 2025]
    ref_h = [wa["ref_stock"][y - 1989] / 1e6 for y in yrs_d]
    idp_h = [wa["idp_stock"][y - 1989] / 1e6 for y in yrs_d]
    lv_lbls = D["regions"]["africa"]["live_36m"]["labels"]
    xs_f = [ym_to_float(l) for l in lv_lbls]
    ref_p50 = [v / 1e6 for v in sum_reg_monthly("ref_monthly", "p50")]
    ref_p10 = [v / 1e6 for v in sum_reg_monthly("ref_monthly", "p10")]
    ref_p90 = [v / 1e6 for v in sum_reg_monthly("ref_monthly", "p90")]
    idp_p50 = [v / 1e6 for v in sum_reg_monthly("idp_monthly", "p50")]
    idp_p10 = [v / 1e6 for v in sum_reg_monthly("idp_monthly", "p10")]
    idp_p90 = [v / 1e6 for v in sum_reg_monthly("idp_monthly", "p90")]

    xticks = [(1995, "1995"), (2005, "2005"), (2015, "2015"), (2025, "2025"), (2029, "2029")]
    c_ref = S.ts_l2_chart(
        "Refugiados y solicitantes de asilo (mundo)", "MILLONES DE PERSONAS · ACNUR 1995–2025 + L2 2026–2029",
        obs=(yrs_d, ref_h),
        agents_band=([2025.0] + xs_f, [ref_h[-1]] + ref_p10, [ref_h[-1]] + ref_p50, [ref_h[-1]] + ref_p90),
        xmin=1995, xmax=2029.8, y_fmt=lambda v: f(v, 0, " M"), end_fmt=lambda v: f(v, 1, " M"),
        xticks=xticks, vlines=[(2025.0, "2025")], shade_from=2025.9, height=130,
        caption="Suma ilustrativa de medianas regionales (sin refugiados bajo mandato UNRWA).",
    )
    c_idp = S.ts_l2_chart(
        "Desplazados internos · IDPs (mundo)", "MILLONES DE PERSONAS · MÁX(ACNUR, IDMC) 1995–2025 + L2",
        obs=(yrs_d, idp_h),
        agents_band=([2025.0] + xs_f, [idp_h[-1]] + idp_p10, [idp_h[-1]] + idp_p50, [idp_h[-1]] + idp_p90),
        xmin=1995, xmax=2029.8, y_fmt=lambda v: f(v, 0, " M"), end_fmt=lambda v: f(v, 1, " M"),
        xticks=xticks, vlines=[(2025.0, "2025")], shade_from=2025.9, height=130,
        caption="El repunte 2022–2025 (Sudán, Ucrania, RDC, Siria) supera los 70 M; L2 mantiene stocks elevados.",
    )

    dw = D["w13_official"]["displacement_world"]
    # Top 8 origin countries by total displaced (ref + idp in 2025)
    top_disp = sorted(
        [(iso, c) for iso, c in D["countries"].items() if (c["ref_2025"] + c["idp_2025"]) > 0],
        key=lambda kv: -(kv[1]["ref_2025"] + kv[1]["idp_2025"]),
    )[:8]
    rows_d = []
    for iso, c in top_disp:
        nm = COUNTRY_NAMES.get(iso, iso)
        cor = ", ".join(f"{a[0]} ({int(round(a[1]*100))} %)" for a in c["asylum_top"][:3]) or "—"
        rows_d.append(
            f"<tr><td><strong>{escape(nm)}</strong> <span class='mono muted'>({iso})</span></td>"
            f"<td class='num'>{f(c['ref_2025']/1e6, 2)} M</td><td class='num hi-orange'>{f(c['ref_2029']['p50']/1e6, 2)} M</td>"
            f"<td class='num'>{f(c['idp_2025']/1e6, 2)} M</td><td class='num hi-orange'>{f(c['idp_2029']['p50']/1e6, 2)} M</td>"
            f"<td style='font-size:6.5pt;'>{escape(cor)}</td></tr>"
        )

    return f"""
<div class="page">
  <span class="chap-mark">05 · DESPLAZAMIENTO HUMANO</span>
  <div class="eyebrow">Capítulo 05 · Bloques de población, refugio y retorno</div>
  <h1>Refugiados, desplazados internos y<br>por qué pierde contra la persistencia</h1>
  <p class="lede">El módulo de desplazamiento traduce muertes por celda en huida interna o transfronteriza y retorno en calma. Es útil como contabilidad coherente de escenarios, pero en validación fuera de muestra pierde sistemáticamente contra «el stock no cambia».</p>

  <div class="grid-2-tight chart-rule">
    {c_ref}{c_idp}
  </div>

  <h2 style="margin-top:1.5mm;">Validación mundial W13 (2012–2023): MAE en asinh(personas/10k) — menor es mejor</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Stock</th><th class="num">Horizonte</th><th class="num">N</th><th class="num">Persistencia</th><th class="num">Con muertes reales</th><th class="num">Pronóstico L2 (p50)</th><th class="num">Cobertura p10–p90</th></tr>
    </thead>
    <tbody>
      <tr><td>Refugiados y solicitantes</td><td class="num">1 año</td><td class="num">{dw['ref_1y']['n']}</td><td class="num hi-green">{f(dw['ref_1y']['pers'],3)}</td><td class="num">{f(dw['ref_1y']['obs_d'],3)}</td><td class="num hi-orange">{f(dw['ref_1y']['full'],3)}</td><td class="num hi-red">{f(dw['ref_1y']['cov80']*100,0)} %</td></tr>
      <tr><td>Refugiados y solicitantes</td><td class="num">3 años</td><td class="num">{dw['ref_3y']['n']}</td><td class="num hi-green">{f(dw['ref_3y']['pers'],3)}</td><td class="num">{f(dw['ref_3y']['obs_d'],3)}</td><td class="num hi-orange">{f(dw['ref_3y']['full'],3)}</td><td class="num hi-red">{f(dw['ref_3y']['cov80']*100,0)} %</td></tr>
      <tr><td>Desplazados internos (IDPs)</td><td class="num">1 año</td><td class="num">{dw['idp_1y']['n']}</td><td class="num hi-green">{f(dw['idp_1y']['pers'],3)}</td><td class="num">{f(dw['idp_1y']['obs_d'],3)}</td><td class="num hi-orange">{f(dw['idp_1y']['full'],3)}</td><td class="num hi-red">{f(dw['idp_1y']['cov80']*100,0)} %</td></tr>
      <tr><td>Desplazados internos (IDPs)</td><td class="num">3 años</td><td class="num">{dw['idp_3y']['n']}</td><td class="num hi-green">{f(dw['idp_3y']['pers'],3)}</td><td class="num">{f(dw['idp_3y']['obs_d'],3)}</td><td class="num hi-orange">{f(dw['idp_3y']['full'],3)}</td><td class="num hi-red">{f(dw['idp_3y']['cov80']*100,0)} %</td></tr>
    </tbody>
  </table>

  <h2 style="margin-top:1.5mm;">Ocho mayores orígenes de desplazamiento (real 2025 → L2 exploratorio 2029) y corredores de asilo</h2>
  <table class="tight-table">
    <thead>
      <tr><th>País de origen</th><th class="num">Ref. 2025</th><th class="num">Ref. 2029</th><th class="num">IDP 2025</th><th class="num">IDP 2029</th><th>Principales países de asilo (cuota 2025)</th></tr>
    </thead>
    <tbody>{"".join(rows_d)}</tbody>
  </table>

  <p style="font-size:7.5pt;color:#475569;margin-top:1mm;"><strong>Diagnóstico clave:</strong> incluso cuando el módulo recibe las muertes que ocurrieron de verdad (columna <em>Con muertes reales</em>), pierde contra la persistencia (0,138 vs 0,084 a 1 año en refugiados). El cuello de botella no es la violencia simulada, sino la función de respuesta homogénea y el uso de stocks anuales en vez de flujos mensuales (además de crisis económicas como Venezuela que UCDP no atribuye a combates).</p>
</div>
"""


def page_09_map_hotspots() -> str:
    wmap = S.world_hotspot_map_svg(D["land_world"], D["top_hotspots"], width=520, height=224)
    hot_rows = []
    for i, h in enumerate(D["top_hotspots"][:10], 1):
        nm = COUNTRY_NAMES.get(h["iso3"], h["iso3"])
        hot_rows.append(
            f"<tr><td class='num'>{i}</td><td><strong>{escape(nm)}</strong> <span class='mono muted'>({h['iso3']})</span></td>"
            f"<td class='mono'>{h['gid']}</td><td class='num'>{f(h['lat'],2)}°, {f(h['lon'],2)}°</td>"
            f"<td class='num hi-orange'>{f(h['mean_12m'],0)}</td><td class='num'>{f(h['p50_12m'],0)}</td>"
            f"<td class='num'>{f(h['p_any_12m']*100,0)} %</td></tr>"
        )
    foot_rows = []
    for r in REGS:
        ft = D["regions"][r]["params"]["foot"]
        foot_rows.append(
            f"<tr><td><strong>{REG_SHORT[r]}</strong></td>"
            f"<td class='num'>{f(ft['decay'],3)}</td><td class='num'>{f(ft['diffuse'],3)}</td>"
            f"<td class='num'>{f(ft['floor'],3)}</td><td class='num'>{f(ft['mix'],3)}</td></tr>"
        )
    return f"""
<div class="page">
  <span class="chap-mark">06 · MAPA SUBNACIONAL 0,5°</span>
  <div class="eyebrow">Capítulo 06 · Resolución geográfica de 55 km × 55 km</div>
  <h1>Dónde se concentra la violencia:<br>64.859 celdas PRIO-GRID y Top 80 focos</h1>
  <p class="lede">Más allá del total por país, la gran ventaja de L2 es su resolución subnacional: cada evento cae en una celda de 0,5° (≈55 km en el ecuador). El mapa muestra la malla terrestre completa y las 80 celdas con mayor mortalidad esperada en los próximos 12 meses (2026-10 a 2027-09).</p>

  <div style="margin:1.5mm 0 2.5mm 0;">{wmap}</div>

  <div class="grid-2-tight" style="grid-template-columns: 1.55fr 1fr; gap: 4mm;">
    <div>
      <h2 style="margin-top:0.5mm;">Diez celdas 0,5° más letales proyectadas (próximos 12m)</h2>
      <table class="tight-table">
        <thead>
          <tr><th class="num">#</th><th>País</th><th>GID</th><th class="num">Lat, Lon</th><th class="num">Media 12m</th><th class="num">p50</th><th class="num">P(≥1)</th></tr>
        </thead>
        <tbody>{"".join(hot_rows)}</tbody>
      </table>
    </div>
    <div>
      <h2 style="margin-top:0.5mm;">Huella espacial estimada (MLE)</h2>
      <table class="tight-table">
        <thead>
          <tr><th>Región</th><th class="num">decay</th><th class="num">diff</th><th class="num">floor</th><th class="num">mix</th></tr>
        </thead>
        <tbody>{"".join(foot_rows)}</tbody>
      </table>
      <div class="slate-box" style="padding:2.8mm 3.2mm;margin-top:2mm;">
        <h3 style="font-size:8.0pt;">Qué revelan los 4 parámetros</h3>
        <p style="font-size:7.3pt;"><strong>Memoria alta (<code>decay</code> = 0,60–0,90):</strong> los frentes persisten en las mismas celdas durante meses (0,90 en África, Américas y Asia; 0,60 en Europa). <strong>Difusión selectiva:</strong> mínima en Europa (0,03) y máxima en África (0,30) por insurgencias móviles. <strong>Suelo (<code>floor</code> = 0,02):</strong> reparte nuevas erupciones al 50 % (<code>mix</code> = 0,50) entre historia violenta del país y población.</p>
      </div>
    </div>
  </div>
</div>
"""


def page_10_twelve_theaters() -> str:
    showcase = ["UKR", "SDN", "ISR", "MMR", "MEX", "NGA", "BFA", "MLI", "COD", "SOM", "ETH", "SYR"]
    cards = []
    for iso in showcase:
        c = D["countries"][iso]
        nm = COUNTRY_NAMES.get(iso, iso)
        r = c["region"]
        b25 = D["regions"][r]["backtest_2025_12m"]["by_country_2025"].get(iso, {})
        tag_top = f"25:{fk(c['deaths_2025'])} · Y1:{fk(c['agents_12m']['p50'])}"
        pl1 = f"{int(round(c['p_l1_2027']*100))} %" if c["p_l1_2027"] is not None else "—"
        tag_sub = f"hu Y1:{fk(c['hurdle_12m']['p50'])} · L1:{pl1}"
        svg = S.country_l2_mini_svg(
            iso, nm, c["monthly_obs_2023_2026"], c["monthly_agents_36m"], c["monthly_hurdle_36m"],
            tag_top=tag_top, tag_sub=tag_sub, height=75,
        )
        cards.append(f'<div class="country-card">{svg}</div>')

    # Compact bridge & blind 2025 verification table for 6 flagship theaters
    flag = ["UKR", "SDN", "ISR", "MMR", "MEX", "NGA"]
    frows = []
    for iso in flag:
        c = D["countries"][iso]
        r = c["region"]
        b25 = D["regions"][r]["backtest_2025_12m"]["by_country_2025"].get(iso, {})
        pl27 = f"{f(c['p_l1_2027']*100,0)} %" if c["p_l1_2027"] is not None else "—"
        pl28 = f"{f(c['p_l1_2028']*100,0)} %" if c["p_l1_2028"] is not None else "—"
        frows.append(
            f"<tr><td><strong>{COUNTRY_NAMES[iso]}</strong> <span class='mono muted'>({iso})</span></td>"
            f"<td class='num'>{fk(c['deaths_2025'])}</td>"
            f"<td class='num hi-orange'>{fk(b25.get('agents_p50'))}</td>"
            f"<td class='num hi-blue'>{fk(b25.get('hurdle_p50'))}</td>"
            f"<td class='num hi-orange'>{fk(c['agents_36m']['p50'])}</td>"
            f"<td class='num hi-blue'>{fk(c['hurdle_36m']['p50'])}</td>"
            f"<td class='num'>{f(c['civ_share_12m']*100,0)} %</td>"
            f"<td class='num hi-green'>{pl27} / {pl28}</td></tr>"
        )
    return f"""
<div class="page">
  <span class="chap-mark">06 · DOCE TEATROS Y PUENTE L2 → L1</span>
  <div class="eyebrow">Capítulo 06 (cont.) · Microscopio por país (2023–2029)</div>
  <h1>Doce teatros críticos de conflicto y<br>el puente anual hacia la capa L1</h1>
  <p class="lede" style="margin-bottom:2.2mm;">Cada panel muestra muertes mensuales en escala <code>ln(1+x)</code> desde enero de 2023 hasta septiembre de 2029: negro = GED final, gris = 2026 provisional, naranja = Agentes L2 (p10–p90), azul = referencia <code>hurdle</code>. <code>L1:%</code> indica la probabilidad anual 2027 de ≥25 muertes estatales.</p>

  <div class="grid-4" style="gap:2.5mm 2.8mm;">
    {"".join(cards)}
  </div>

  <h2 style="margin-top:2.5mm;">Verificación ciega 2025, horizonte 36 meses (2026–2029) y puente L2 → L1</h2>
  <table class="tight-table">
    <thead>
      <tr>
        <th>Teatro (*ISR incluye Palestina)</th>
        <th class="num">2025 real</th>
        <th class="num">Ciego 25 Ag</th>
        <th class="num">Ciego 25 Hu</th>
        <th class="num">36m Ag (p50)</th>
        <th class="num">36m Hu (p50)</th>
        <th class="num">% Civiles</th>
        <th class="num">Puente L1 27/28</th>
      </tr>
    </thead>
    <tbody>{"".join(frows)}</tbody>
  </table>

  <p style="font-size:7.4pt;color:#475569;"><strong>Lectura crítica:</strong> en Ucrania (<code>UKR</code>), la prueba ciega de 2025 sobreestimó la mortalidad por muestreo de picos previos y luego en 2026–2029 el subconteo provisional de 2026 hace arrancar <code>Ag</code> por debajo de 2025; en Sudán (<code>SDN</code>), Burkina Faso (<code>BFA</code>), Malí (<code>MLI</code>) y Nigeria (<code>NGA</code>), la multiplicación de díadas impulsa el abanico naranja hacia arriba en 2028–2029, mientras en México (<code>MEX</code>) domina violencia no estatal (tipo 2), por lo que su puente estatal L1 es bajo pese a una mortalidad elevada.</p>
</div>
"""


def page_11_stress_ablations() -> str:
    ch_drift = drift_bar_svg()
    # Build 5-year stress table by region
    drows = []
    for r in REGS:
        d25 = D["regions"][r]["history"]["summary"]["deaths_2025"]
        ad = D["regions"][r]["drift_60m"]["annual_deaths"]
        drows.append(
            f"<tr><td><strong>{REG_LBL[r]}</strong></td><td class='num'>{fk(d25)}</td>"
            f"<td class='num'>{fk(ad['Y1']['p50'])}</td><td class='num'>{fk(ad['Y2']['p50'])}</td>"
            f"<td class='num hi-orange'>{fk(ad['Y3']['p50'])}</td><td class='num hi-orange'>{fk(ad['Y4']['p50'])}</td>"
            f"<td class='num hi-red'>{fk(ad['Y5']['p50'])}</td></tr>"
        )
    # Build 4-ablation table (world sum of medians, CRN seed=77)
    ab_meta = [
        ("completo", "Modelo completo (referencia CRN)", "Todas las piezas encendidas (36m, R=32)"),
        ("sin_nacimientos", "Sin nacimientos de díadas", "Apaga el proceso Poisson de nuevas díadas"),
        ("sin_difusion", "Sin difusión espacial (diffuse = 0)", "La huella no salta a celdas vecinas"),
        ("sin_suelo_espacial", "Sin suelo de país (floor = 0)", "Sin erupciones fuera de la huella previa"),
        ("letalidad_reciente_24m", "Letalidad reciente (vida media 24m)", "Pondera más la letalidad de los últimos 2 años"),
    ]
    base_36 = sum(D["regions"][r]["ablations_36m"]["completo"]["deaths_36m_p50"] for r in REGS)
    base_y3 = sum(D["regions"][r]["ablations_36m"]["completo"]["deaths_y3_p50"] for r in REGS)
    base_cl = sum(D["regions"][r]["ablations_36m"]["completo"]["active_cells_y3"] for r in REGS)
    ab_rows = []
    for k, lbl, desc in ab_meta:
        s36 = sum(D["regions"][r]["ablations_36m"][k]["deaths_36m_p50"] for r in REGS)
        sy3 = sum(D["regions"][r]["ablations_36m"][k]["deaths_y3_p50"] for r in REGS)
        scl = sum(D["regions"][r]["ablations_36m"][k]["active_cells_y3"] for r in REGS)
        sdy = sum(D["regions"][r]["ablations_36m"][k]["active_dyads_y3"] for r in REGS)
        d36 = (s36 / max(base_36, 1) - 1) * 100
        dy3 = (sy3 / max(base_y3, 1) - 1) * 100
        dcl = (scl / max(base_cl, 1) - 1) * 100
        cls = "group-row" if k == "completo" else ""
        ab_rows.append(
            f"<tr class='{cls}'><td><strong>{escape(lbl)}</strong><br><span class='muted' style='font-size:6.4pt;'>{escape(desc)}</span></td>"
            f"<td class='num'>{fk(s36)}</td><td class='num'>{('base' if k=='completo' else sgn(d36, 1, ' %'))}</td>"
            f"<td class='num'>{fk(sy3)}</td><td class='num'>{('base' if k=='completo' else sgn(dy3, 1, ' %'))}</td>"
            f"<td class='num'>{f(scl, 0)} ({('base' if k=='completo' else sgn(dcl, 0, '%'))})</td>"
            f"<td class='num'>{f(sdy, 0)}</td></tr>"
        )

    return f"""
<div class="page">
  <span class="chap-mark">07 · ESTRÉS A 60 MESES Y ABLACIONES</span>
  <div class="eyebrow">Capítulo 07 · Qué dispara la deriva y qué pieza controla cada efecto</div>
  <h1>Estrés de horizonte a cinco años y<br>cuatro ablaciones estructurales</h1>
  <p class="lede">Extender L2 de 36 a 60 meses —y apagar de una en una sus piezas internas con los mismos números aleatorios— identifica el mecanismo exacto que vuelve inestable al simulador: la retroalimentación entre activación multi-frente y nacimientos sin techo de fatiga.</p>

  <div class="chart-rule">{ch_drift}</div>

  <h2 style="margin-top:2mm;">Muertes anuales medianas (p50) en la prueba de estrés 2026-10 → 2031-09</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Región</th><th class="num">2025 real</th><th class="num">Año 1 (2027)</th><th class="num">Año 2 (2028)</th><th class="num">Año 3 (2029)</th><th class="num">Año 4 (2030)</th><th class="num">Año 5 (2031)</th></tr>
    </thead>
    <tbody>{"".join(drows)}</tbody>
  </table>

  <h2 style="margin-top:2mm;">Cuatro ablaciones estructurales a 36 meses (números aleatorios comunes · suma de 5 regiones)</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Experimento</th><th class="num">Muertes 36m</th><th class="num">Δ 36m</th><th class="num">Muertes Año 3</th><th class="num">Δ Año 3</th><th class="num">Celdas activas Y3</th><th class="num">Díadas/m Y3</th></tr>
    </thead>
    <tbody>{"".join(ab_rows)}</tbody>
  </table>

  <div class="grid-2" style="font-size:7.6pt;color:#334155;">
    <p><strong>1. Dos regímenes de deriva opuestos:</strong> África y Asia —con cientos de díadas vivas y fuerte acoplamiento de red— aceleran año tras año entre Año 1 y Año 5. En contraste, Europa y Américas (pocas díadas dominantes) permanecen mucho más acotadas. Esto explica por qué un horizonte de 15 años carecería de sentido físico en L2.</p>
    <p><strong>2. Qué revela cada ablación:</strong> apagar <strong>nacimientos</strong> recorta la proliferación de díadas y frena el desborde de Año 3; apagar la <strong>difusión</strong> y el <strong>suelo espacial</strong> apenas altera el total nacional de muertes pero contrae drásticamente las celdas 0,5° golpeadas; y ponderar <strong>letalidad reciente (24m)</strong> cambia sustancialmente la escala de muertes donde la guerra mutó de intensidad (Ucrania, Gaza, Sudán).</p>
  </div>
</div>
"""


def page_12_params_1() -> str:
    hmap = S.coef_heatmap_svg(D["regions"], REGS, [REG_SHORT[r] for r in REGS], height=192)
    # Table of b_int (key intensity features) + b_birth + D0_alive + n_cap + resid_sd
    i_keys = [
        ("own1", "Eventos mes previo (b_int)"),
        ("own12", "Media 12 meses (b_int)"),
        ("since", "Meses callada (b_int)"),
        ("side_a", "Otros frentes actor A (b_int)"),
        ("ctx", "Violencia resto país (b_int)"),
        ("non_state", "Tipo no estatal (b_int)"),
        ("one_sided", "Tipo unilateral civiles (b_int)"),
    ]
    rows_i = []
    for k, lbl in i_keys:
        tds = "".join(f"<td class='num'>{sgn(D['regions'][r]['params']['b_int'][k], 2)}</td>" for r in REGS)
        rows_i.append(f"<tr><td>{escape(lbl)}</td>{tds}</tr>")
    # Births & diagnostics
    b_rows = [
        ("Nacimientos · intercepto (c₀)", [sgn(D["regions"][r]["params"]["b_birth"]["const"], 2) for r in REGS]),
        ("Nacimientos · log(1+ev₁₂) (c₁)", [sgn(D["regions"][r]["params"]["b_birth"]["log_ev12"], 2) for r in REGS]),
        ("Nacimientos · log población (c₂)", [sgn(D["regions"][r]["params"]["b_birth"]["log_pop"], 2) for r in REGS]),
        ("Díadas vivas al origen (D₀)", [f(D["regions"][r]["params"]["D0_alive"], 0) for r in REGS]),
        ("Techo eventos/díada/mes (n_cap)", [f(D["regions"][r]["params"]["n_cap"], 0) for r in REGS]),
        ("Desv. estándar residuos (ln ev)", [f(D["regions"][r]["params"]["resid_sd"], 2) for r in REGS]),
    ]
    rows_b = []
    for lbl, vals in b_rows:
        rows_b.append(f"<tr><td><strong>{escape(lbl)}</strong></td>" + "".join(f"<td class='num'>{v}</td>" for v in vals) + "</tr>")

    return f"""
<div class="page">
  <span class="chap-mark">08 · PARÁMETROS ESTIMADOS (I)</span>
  <div class="eyebrow">Capítulo 08 · Auditoría completa de parámetros · Parte I</div>
  <h1>Mapa de coeficientes de activación,<br>intensidad y nacimientos de díadas</h1>
  <p class="lede">A diferencia de L1 (un único juego global de parámetros), L2 estima un modelo independiente por región sobre todo el historial hasta septiembre de 2026. Comparar las cinco columnas expone las regularidades universales del conflicto armado y las diferencias estructurales entre regiones.</p>

  <h2 style="margin-top:0.5mm;">Matriz de calor: los 11 coeficientes del logit de activación mensual (<code>b_act</code>)</h2>
  <p class="muted" style="font-family:'IBM Plex Mono';font-size:5.9pt;margin-bottom:1.2mm;">NARANJA = AUMENTA PROBABILIDAD DE COMBATE EL MES SIGUIENTE · AZUL = LA REDUCE</p>
  {hmap}

  <h2 style="margin-top:2mm;">Intensidad de eventos (<code>b_int</code>), tasa Poisson de nacimientos (<code>b_birth</code>) y estado inicial</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Parámetro estimado (hasta 2026-09)</th>{"".join(f"<th class='num'>{REG_SHORT[r]}</th>" for r in REGS)}</tr>
    </thead>
    <tbody>
      {"".join(rows_i)}
      <tr class="group-row"><td colspan="6">NACIMIENTOS DE DÍADAS Y DIAGNÓSTICOS DEL SIMULADOR</td></tr>
      {"".join(rows_b)}
    </tbody>
  </table>

  <p style="font-size:7.5pt;color:#475569;"><strong>Tres universales empíricos:</strong> (1) el silencio pasado (<code>since</code>, −1,15 a −1,76) es el freno más fuerte en las cinco regiones; (2) la memoria reciente (<code>own1, own3, own12</code>) y los <strong>otros frentes del actor A</strong> (<code>side_a &gt; 0</code> en cuatro de cinco regiones) empujan al alza la activación; y (3) <code>c₁ &gt; 0</code> en <code>b_birth</code> confirma que la violencia previa engendra nuevas díadas (fragmentación de facciones), sobre todo en África (<code>D₀ = {D['regions']['africa']['params']['D0_alive']}</code> díadas vivas).</p>
</div>
"""


def page_13_params_2() -> str:
    # Displacement 8 params + mult range across 5 regions
    d_keys = [
        ("beta", "β · tasa base de huida por exposición", 4),
        ("eta", "η · elasticidad no lineal de exposición", 2),
        ("phi", "φ · memoria mensual de exposición", 3),
        ("g0", "g₀ · intercepto cruce de frontera", 2),
        ("g1", "g₁ · efecto distancia a frontera (z)", 2),
        ("rho_idp", "ρ_idp · tasa mensual retorno internos", 4),
        ("rho_ref", "ρ_ref · tasa mensual retorno refugiados", 4),
        ("E0", "E₀ · umbral calma (muertes/100k/mes)", 2),
    ]
    rows_dp = []
    for k, lbl, dec in d_keys:
        tds = "".join(f"<td class='num'>{f(D['regions'][r]['params']['displacement'][k], dec)}</td>" for r in REGS)
        rows_dp.append(f"<tr><td>{escape(lbl)}</td>{tds}</tr>")
    mult_tds = "".join(
        f"<td class='num'>{f(D['regions'][r]['params']['disp_mult_range'][0],2)}–{f(D['regions'][r]['params']['disp_mult_range'][2],0)}×</td>"
        for r in REGS
    )
    rows_dp.append(f"<tr><td><strong>Multiplicador por país (mín–máx)</strong></td>{mult_tds}</tr>")

    # Complete output trajectory table 2025 -> 2029 (p10/p50/p90) & Y5 stress vs hurdle
    rows_out = []
    for r in REGS:
        s = D["regions"][r]["history"]["summary"]
        lv = D["regions"][r]["live_36m"]
        ad = D["regions"][r]["drift_60m"]["annual_deaths"]
        hu_y1 = lv["totals_12m"]["hurdle"]["p50"]
        hu_36 = lv["totals_36m"]["hurdle"]["p50"]
        ag_36 = lv["totals_36m"]["agents"]
        ref25 = D["regions"][r]["history"]["annual"]["ref_stock"][36]
        ref29 = lv["ref_monthly"]["p50"][-1]
        idp25 = D["regions"][r]["history"]["annual"]["idp_stock"][36]
        idp29 = lv["idp_monthly"]["p50"][-1]
        rows_out.append(
            f"<tr><td><strong>{REG_LBL[r]}</strong></td>"
            f"<td class='num'>{fk(s['deaths_2025'])}</td>"
            f"<td class='num'>{fk(s['deaths_2026_ytd'])}</td>"
            f"<td class='num hi-orange'>{fk(lv['totals_12m']['agents']['p50'])}</td>"
            f"<td class='num hi-blue'>{fk(hu_y1)}</td>"
            f"<td class='num hi-orange'>{fk(ag_36['p50'])} <span class='muted'>({fk(ag_36['p10'])}–{fk(ag_36['p90'])})</span></td>"
            f"<td class='num hi-blue'>{fk(hu_36)}</td>"
            f"<td class='num'>{f(ref25/1e6,1)}→{f(ref29/1e6,1)} M</td>"
            f"<td class='num'>{f(idp25/1e6,1)}→{f(idp29/1e6,1)} M</td></tr>"
        )
    return f"""
<div class="page">
  <span class="chap-mark">08 · PARÁMETROS Y SALIDAS (II)</span>
  <div class="eyebrow">Capítulo 08 · Auditoría completa de parámetros · Parte II</div>
  <h1>Parámetros de desplazamiento y<br>tablero consolidado 2025–2029</h1>
  <p class="lede">Los ocho parámetros físicos del módulo de desplazamiento revelan diferencias enormes entre continentes en la propensión a cruzar fronteras, la memoria del miedo y la velocidad de retorno. Abajo se condensa el tablero operativo completo 2025–2029 para las cinco regiones.</p>

  <h2 style="margin-top:0.5mm;">Los 8 parámetros calibrados del módulo de desplazamiento (hasta 2026-09)</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Parámetro de desplazamiento</th>{"".join(f"<th class='num'>{REG_SHORT[r]}</th>" for r in REGS)}</tr>
    </thead>
    <tbody>{"".join(rows_dp)}</tbody>
  </table>

  <h2 style="margin-top:2.5mm;">Tablero consolidado de evolución (2025 real, 2026 provisional y proyección 2026-10 → 2029-09)</h2>
  <table class="tight-table">
    <thead>
      <tr>
        <th>Región</th>
        <th class="num">2025 GED</th>
        <th class="num">2026* 9m</th>
        <th class="num">Año 1 Ag</th>
        <th class="num">Año 1 Hu</th>
        <th class="num">36m Ag p50 (p10–p90)</th>
        <th class="num">36m Hu</th>
        <th class="num">Ref. 25→29</th>
        <th class="num">IDP 25→29</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_out)}</tbody>
  </table>

  <div class="slate-box" style="margin-top:2.5mm;">
    <h3>Cómo leer los dos grandes contrastes de esta tabla</h3>
    <p><strong>1. Multiplicadores de 0,125× a 32× en desplazamiento:</strong> la cantidad de personas que huyen por cada muerte difiere en dos órdenes de magnitud entre guerras urbanas de alta visibilidad (Siria, Sudán, Ucrania) y conflictos criminales o rurales de baja densidad. Un único parámetro <code>β</code> regional no puede absorber esa heterogeneidad sin el multiplicador por país, y aun así su ajuste en muestra tiene un factor de error típico de ~2,7×.</p>
    <p><strong>2. Agentes vs. <code>hurdle</code> a 36 meses:</strong> en casi todas las regiones, el primer año (<code>Año 1</code>) mantiene órdenes de magnitud comparables entre Agentes y <code>hurdle</code>, pero en el acumulado de 36 meses los agentes divergen fuertemente en África y Asia mientras <code>hurdle</code> permanece anclado al régimen reciente. Para uso práctico de magnitud, la referencia <code>hurdle</code> o <code>last12</code> es siempre la lectura más segura.</p>
  </div>
</div>
"""


def page_14_uncertainty() -> str:
    ch_cov = coverage_disp_svg()
    bt23 = D["world_backtest_2023"]["country"]
    bt25 = D["world_backtest_2025"]["country"]
    return f"""
<div class="page">
  <span class="chap-mark">09 · INCERTIDUMBRE Y ANATOMÍA DEL FALLO</span>
  <div class="eyebrow">Capítulo 09 · Diagnóstico estructural de W13</div>
  <h1>Por qué L2 acierta la presencia,<br>falla la magnitud y subcubre al 22–38 %</h1>
  <p class="lede">Un fallo bien diagnosticado vale más que una caja negra que aparenta funcionar. Descomponer los errores de W13 y de las pruebas ciegas 2023–2025 señala cuatro mecanismos concretos detrás del semáforo rojo de L2.</p>

  <div class="grid-2">
    <div class="slate-box" style="margin-top:0;">
      <h3>1. Iteración recursiva vs. ajuste directo</h3>
      <p><code>hurdle</code> se ajusta <em>directamente</em> para cada horizonte (1, 3, 6, 12, 24 y 36 meses). El simulador de agentes, en cambio, se estima solo a 1 paso y luego se realimenta 36 veces con sus propios eventos simulados. Un pequeño sesgo positivo en activación o nacimientos se compone mes tras mes hasta duplicar el error en África al año 3.</p>
    </div>
    <div class="slate-box" style="margin-top:0;">
      <h3>2. Presencia (Brier1) sí; magnitud (CRPS) no</h3>
      <p>En la prueba ciega de 2025 a 12 meses, el Brier de «≥1 muerte» de Agentes ({f(bt25['agents']['12']['brier1'], 4)}) casi iguala a <code>hurdle</code> ({f(bt25['hurdle']['12']['brier1'], 4)}) y en W13 a 1 mes gana en África, Américas y Oriente Medio. El problema surge al multiplicar eventos por depósitos de letalidad de cola gruesa sin techo de fatiga.</p>
    </div>
    <div class="slate-box" style="margin-top:0;">
      <h3>3. El efecto del subconteo de 2026-Candidate</h3>
      <p>Los 9 meses de 2026 (<code>Clear</code>) registran solo la fracción ya verificada de eventos. Como <code>own1</code> y <code>own3</code> dominan el logit de activación, cualquier proyección en vivo que arranque desde un mes provisional subestima el trimestre inicial y deforma los primeros pasos respecto a un cierre GED definitivo.</p>
    </div>
    <div class="slate-box" style="margin-top:0;">
      <h3>4. Regiones desacopladas y ausencia de shocks</h3>
      <p>Cada región corre aislada: ni un actor transregional (Estado Islámico, Rusia, Irán) acopla sus frentes entre regiones, ni existen choques geopolíticos comunes. Además, la forma del simulador se eligió solo con Oriente Medio 2005–2014 y se extrapoló sin reespecificar a las otras cuatro regiones.</p>
    </div>
  </div>

  <h2 style="margin-top:1.5mm;">Subcobertura severa de las bandas p10–p90 en desplazamiento (W13 mundial)</h2>
  <p class="muted" style="font-family:'IBM Plex Mono';font-size:5.9pt;margin-bottom:1mm;">UNA BANDA P10–P90 BIEN CALIBRADA DEBE CUBRIR EL 80 % DE LOS CASOS REALES; L2 SOLO CUBRE 22–38 %</p>
  {ch_cov}

  <h2 style="margin-top:2mm;">Comparativa de presencia vs. magnitud en la prueba ciega 2023–2025 (191 países)</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Horizonte (desde 2023-01)</th><th class="num">CRPS Ag</th><th class="num">CRPS Hu</th><th class="num">CRPS L12</th><th class="num">Brier1 Ag</th><th class="num">Brier1 Hu</th><th class="num">Brier25 Ag</th><th class="num">Brier25 Hu</th></tr>
    </thead>
    <tbody>
      {"".join(
          f"<tr><td><strong>{s} {'mes' if s == '1' else 'meses'}</strong></td>"
          f"<td class='num hi-orange'>{f(bt23['agents'][s]['crps_log'],4)}</td>"
          f"<td class='num hi-blue'>{f(bt23['hurdle'][s]['crps_log'],4)}</td>"
          f"<td class='num'>{f(bt23['last12'][s]['crps_log'],4)}</td>"
          f"<td class='num'>{f(bt23['agents'][s]['brier1'],4)}</td>"
          f"<td class='num'>{f(bt23['hurdle'][s]['brier1'],4)}</td>"
          f"<td class='num'>{f(bt23['agents'][s]['brier25'],4)}</td>"
          f"<td class='num'>{f(bt23['hurdle'][s]['brier25'],4)}</td></tr>"
          for s in ("1", "3", "6", "12", "24", "36")
      )}
    </tbody>
  </table>
  <p style="font-size:7.4pt;color:#475569;"><strong>Por qué subcubre el desplazamiento:</strong> sus bandas solo propagan la varianza de las muertes simuladas dentro de un único vector fijo de parámetros <code>(β, η, φ, g₀, g₁, ρ, E₀)</code>, omitiendo la incertidumbre de la tasa de huida por muerte y los flujos no bélicos.</p>
</div>
"""


def page_15_conclusions() -> str:
    return """
<div class="page">
  <span class="chap-mark">10 · CONCLUSIONES Y QUÉ SIGUE</span>
  <div class="eyebrow">Capítulo 10 · Conclusiones</div>
  <h1>Cuánto fiarse de L2, límites<br>estructurales y hoja de ruta</h1>
  <p class="lede">Probar la capa L2 en las cinco regiones del mundo (191 países y 64.859 celdas) confirma el veredicto de F12 con mayor nitidez: es una infraestructura de datos y simulación subnacional notable para escenarios y presencia de corto plazo, pero hoy <strong>no supera</strong> a referencias estadísticas simples en magnitud ni desplazamiento.</p>

  <div class="grid-3">
    <div class="conf-card green">
      <div class="conf-head"><strong>Firme</strong><span>señal útil hoy</span></div>
      <ul>
        <li><strong>Presencia de conflicto a 1–3 meses:</strong> empata o supera a las referencias en Brier de «≥1 muerte» por país en África, Américas y Oriente Medio.</li>
        <li><strong>Puente anual L2 → L1:</strong> reduce el Brier del evento anual ≥25 muertes estatales frente a la persistencia (0,084 vs 0,108 en Oriente Medio).</li>
        <li><strong>Geografía de focos 0,5° a corto plazo:</strong> localiza con coherencia las celdas activas heredadas y respeta fronteras internacionales.</li>
      </ul>
    </div>
    <div class="conf-card amber">
      <div class="conf-head"><strong>Razonable</strong><span>solo escenarios</span></div>
      <ul>
        <li><strong>Horizonte 3–12 meses por país:</strong> útil como laboratorio contrafactual («qué cambia si se frena la difusión o los nacimientos»), siempre leído junto a <code>hurdle</code> y <code>last12</code>.</li>
        <li><strong>Contabilidad de desplazamiento:</strong> señala la dirección correcta en arranques de guerra (Siria, Sudán), pero no sirve como pronóstico de stock.</li>
        <li><strong>Europa a 12–36 meses:</strong> única región donde L2 bate ligeramente a <code>last12</code> por su baja fragmentación de díadas.</li>
      </ul>
    </div>
    <div class="conf-card red">
      <div class="conf-head"><strong>Frágil o rojo</strong><span>no usar para pronóstico</span></div>
      <ul>
        <li><strong>Horizontes ≥24 meses y estrés a 60m:</strong> sin stocks de fatiga, el bucle de nacimientos y activación desboca África y Asia (+59 % de error mundial a 36m).</li>
        <li><strong>Magnitud por celda a 12 meses:</strong> pierde frente a pronosticar cero en África y Europa, y empata con cero a escala mundial.</li>
        <li><strong>Bandas de desplazamiento:</strong> cubren solo 22–38 % en vez de 80 % y pierden contra la persistencia incluso con muertes reales.</li>
      </ul>
    </div>
  </div>

  <h2>Seis conclusiones elaboradas del estudio global</h2>
  <div class="grid-2" style="font-size:7.9pt;line-height:1.4;">
    <div>
      <p><strong>1. La escala temporal correcta de L2 es 1–12 meses, nunca 15 años.</strong> Al igual que L0 choca con límites de recursos y L1 con la trampa de Hubbert tras 10 años, L2 expone su techo estructural al mes 24: sin recursos militares ni desgaste demográfico, la iteración recursiva de díadas amplifica cualquier pequeño sesgo hasta hacerlo explotar.</p>
      <p><strong>2. La persistencia (<code>last12</code>) es la vara mundial a batir, no solo <code>hurdle</code>.</strong> Fuera de Oriente Medio, repetir el promedio de los últimos 12 meses supera incluso al modelo estadístico directo por país. En violencia organizada, la inercia local de los frentes pesa más que casi cualquier covariable estática.</p>
      <p><strong>3. El mayor valor de L2 está en lo que las referencias no pueden hacer.</strong> Ni <code>last12</code> ni <code>hurdle</code> tienen actores, frentes, difusión espacial ni población desplazada. Las ablaciones demuestran que L2 sí permite descomponer cuánto del riesgo viene de nuevas díadas, de la difusión vecinal o de la letalidad reciente.</p>
    </div>
    <div>
      <p><strong>4. Los choques exógenos (Sudán 2023, Gaza 2023) marcan el techo de predictibilidad.</strong> Ningún modelo entrenado hasta diciembre de 2022 —ni agentes ni <code>hurdle</code>— predijo el salto abrupto de 2023–2025. La utilidad realista de un sistema de alerta temprana está en vigilar escaladas en curso y vulnerabilidades, no en predecir chispas políticas inéditas.</p>
      <p><strong>5. El desplazamiento exige datos mensuales y heterogeneidad por tipo de guerra.</strong> El hecho de perder contra la persistencia incluso con las muertes verdaderas prueba que el problema reside en calibrar una curva única de huida sobre stocks anuales que mezclan guerras civiles, violencia criminal y crisis económicas.</p>
      <p><strong>6. L2 cierra la escalera multiescala de Phy-World, pero pide hibridación.</strong> Su puente hacia L1 ya demostró señal positiva: el salto de calidad real vendrá de acoplar el total nacional con modelos directos estables y reservar el simulador de agentes para el reparto subnacional y de actores.</p>
    </div>
  </div>

  <h2>Qué sigue (hoja de ruta priorizada)</h2>
  <div class="grid-3">
    <div style="border-top:1.2pt solid #0E162B;padding-top:1.8mm;">
      <strong style="font-size:8.2pt;">1. Frenar la iteración ciega</strong>
      <p style="font-size:7.4pt;color:#475569;margin-top:0.8mm;">Condicionar el total mensual por país a <code>hurdle</code> (dejando a los agentes el reparto en díadas, celdas y huida) o calibrar activación y nacimientos multi-paso con ABC-SMC (F7) y un término explícito de fatiga/desgaste.</p>
    </div>
    <div style="border-top:1.2pt solid #0E162B;padding-top:1.8mm;">
      <strong style="font-size:8.2pt;">2. Acoplar regiones y ViEWS</strong>
      <p style="font-size:7.4pt;color:#475569;margin-top:0.8mm;">Permitir que actores transregionales compartan estado entre las 5 regiones, incorporar ViEWS (<code>api.viewsforecasting.org</code>) como referencia externa exigida por W13 y cablear el puente L2 → L1 dentro de F11.</p>
    </div>
    <div style="border-top:1.2pt solid #0E162B;padding-top:1.8mm;">
      <strong style="font-size:8.2pt;">3. Desplazamiento mensual</strong>
      <p style="font-size:7.4pt;color:#475569;margin-top:0.8mm;">Sustituir stocks anuales por flujos mensuales de ACNUR/OIM, separar la respuesta según tipo de violencia (estatal, no estatal, civil) y propagar incertidumbre paramétrica para llevar la cobertura de 22–38 % al 80 %.</p>
    </div>
  </div>
</div>
"""


def page_16_appendix() -> str:
    m = D["meta"]
    return f"""
<div class="page">
  <span class="chap-mark">APÉNDICE · FUENTES Y REPRODUCIBILIDAD</span>
  <div class="eyebrow">Apéndice</div>
  <h1>Fuentes, convenciones, nota de<br>manifiesto y cómo reproducirlo</h1>

  <h2>Fuentes de datos integradas en los 5 paneles regionales de L2</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Fuente</th><th>Cobertura y variables</th><th>Licencia y estado</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>UCDP GED 26.1</strong></td><td>428.892 eventos globales (1989-01 a 2025-12): coordenadas, bandos, tipo (1–3), muertes (<code>best, low, high</code>, civiles).</td><td>CC BY 4.0 · SHA-256 verificado</td></tr>
      <tr><td><strong>UCDP Candidate 26.0.X</strong></td><td>Enero–septiembre de 2026; únicamente eventos revisados con <code>code_status == "Clear"</code> (provisional, nunca verdad de prueba).</td><td>CC BY 4.0 · SHA-256 verificado</td></tr>
      <tr><td><strong>PRIO-GRID 2.0</strong></td><td>Malla mundial de 64.818 celdas de 0,5° (64.859 sumando fronteras compartidas entre regiones): país 2014, GPW v3, montañas, distancias, luces.</td><td>Uso académico (no redistribuible) · ver nota abajo</td></tr>
      <tr><td><strong>ACNUR (<code>refugees</code> 2025.12.3)</strong></td><td>Stocks anuales 1951–2025 de refugiados y solicitantes por par origen–asilo y desplazados internos (máximo ACNUR / IDMC).</td><td>CC BY 4.0 · SHA-256 verificado</td></tr>
      <tr><td><strong>UN WPP 2024</strong></td><td>Población nacional observada y proyectada; se usa para escalar el reparto espacial de celdas GPW 2010 año a año.</td><td>CC BY 3.0 IGO · SHA-256 verificado</td></tr>
    </tbody>
  </table>

  <div class="slate-box">
    <h3>Convenciones de codificación y nota de transparencia sobre PRIO-GRID</h3>
    <p><strong>Agrupaciones territoriales UCDP/PRIO-GRID:</strong> <code>ISR</code> incluye Israel, Gaza y Cisjordania; <code>MAR</code> incluye Marruecos y el Sáhara Occidental; <code>CHN</code> incluye Hong Kong y Macao. El 5,9 % de los eventos fronterizos cae en una celda cuyo país mayoritario difiere del país UCDP; el índice de <em>sitios</em> <code>(celda, país)</code> conserva la contabilidad exacta de ambos niveles.</p>
    <p><strong>Nota sobre la verificación SHA-256 de la API de PRIO-GRID:</strong> los ficheros de UCDP GED 26.1, UCDP Candidate, ACNUR, WPP y <code>prio_petroleum_s.json</code> coincidieron byte a byte con <code>world4/data/l2_manifest.json</code>. Los otros 13 JSON devueltos en vivo por <code>grid.prio.org</code> contienen exactamente las 64.818 celdas válidas, pero con distinto orden de filas respecto al volcado original del manifiesto; los cinco paneles convertidos (<code>l2_{{africa,americas,asia,europe,mideast}}.npz</code>) reproducen al dígito los recuentos oficiales de <code>docs/F12-informe.md</code> (191 países, 64.859 celdas, 2.059 díadas, 428.892 eventos y 4.347.272 muertes).</p>
  </div>

  <div class="grid-2" style="margin-top:2.5mm;">
    <div>
      <h2>Definiciones clave</h2>
      <ul style="padding-left:4mm;font-size:7.6pt;line-height:1.4;color:#334155;">
        <li><strong>Díada.</strong> Par específico de actores armados (ej. Gobierno de Sudán vs. RSF). L2 simula díadas, no países abstractos.</li>
        <li><strong>CRPS de ln(1+muertes).</strong> Regla de puntuación propia que evalúa la distribución completa del ensemble frente al dato real. Menor es mejor.</li>
        <li><strong>Brier1 / Brier25.</strong> Error cuadrático medio de la probabilidad predicha de ≥1 muerte (o ≥25 muertes) mensuales.</li>
        <li><strong>asinh(personas/10k).</strong> Transformación usada en desplazamiento: lineal cerca de cero y logarítmica en stocks grandes (0,69 ≈ factor 2×).</li>
        <li><strong>CRN (números aleatorios comunes).</strong> Semilla idéntica entre corridas de ablación para eliminar ruido Monte Carlo al aislar cada pieza.</li>
      </ul>
    </div>
    <div>
      <h2>Reproducir este informe</h2>
      <p style="font-size:7.6pt;color:#334155;">Generado íntegramente desde los módulos reales de <strong>Phy-World L2 (F12)</strong> en CPU (<code>OMP_NUM_THREADS=1</code>). Tiempo del estudio completo de 5 regiones (incluyendo calibración inicial de desplazamiento): <strong>{f(m['seconds']/60, 1)} minutos</strong>.</p>
      <pre class="code">python world4_cli.py --install-l2 --l2-region world
python reports/l2_conflict/run_study.py
python reports/l2_conflict/build_pdf.py</pre>
      <table class="tight-table">
        <thead><tr><th>Pieza</th><th>Módulo</th></tr></thead>
        <tbody>
          <tr><td>Panel y corte sin fuga</td><td class="mono">world4/core/l2/panel.py</td></tr>
          <tr><td>Simulador de agentes y huella</td><td class="mono">world4/core/l2/agents.py</td></tr>
          <tr><td>Referencias hurdle / last12</td><td class="mono">world4/core/l2/baseline.py</td></tr>
          <tr><td>Desplazamiento y puente L1</td><td class="mono">l2/displacement.py, forecast.py</td></tr>
          <tr><td>Estudio global y maqueta PDF</td><td class="mono">reports/l2_conflict/run_study.py, build_pdf.py</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
"""


def main():
    pages = [
        page_01_cover(),
        page_02_executive(),
        page_03_scope(),
        page_04_architecture(),
        page_05_backtest(),
        page_06_geography_1(),
        page_07_geography_2(),
        page_08_displacement(),
        page_09_map_hotspots(),
        page_10_twelve_theaters(),
        page_11_stress_ablations(),
        page_12_params_1(),
        page_13_params_2(),
        page_14_uncertainty(),
        page_15_conclusions(),
        page_16_appendix(),
    ]
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>La capa de eventos del mundo, mes a mes y celda a celda · Phy-World L2</title>
<style>{css_block()}</style>
</head>
<body>
{"".join(pages)}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(OUT_PDF))
    print(f"PDF L2 generado: {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
