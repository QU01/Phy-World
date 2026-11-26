"""Construye el informe editorial PDF de la capa L1 (Atlas del Sistema-Mundo)
a partir de ``study_results.json`` mediante HTML/CSS + SVG vectorial + WeasyPrint.

Uso:
    python reports/l1_atlas/build_pdf.py
"""
from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import weasyprint
import svg_charts as S

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
DATA_PATH = HERE / "study_results.json"
OUT_HTML = HERE / "atlas_mundo_l1_15_anios.html"
OUT_PDF = HERE / "atlas_mundo_l1_15_anios.pdf"

D = json.loads(DATA_PATH.read_text(encoding="utf-8"))
OW = D["observed_world"]
Y_OBS = OW["years"]  # 1950..2025
SH = D["series"]["hist_1950_2019"]
S00 = D["series"]["test_2000_2025"]
S10 = D["series"]["test_2010_2025"]
S19 = D["series"]["proj_2019_2041"]
Y_H = list(range(1950, 2020))
Y_00 = list(range(2000, 2026))
Y_10 = list(range(2010, 2026))
Y_19 = list(range(2019, 2042))
SC = D["scorecards"]
RW = D["recent_world_check"]
REG10 = D.get("regions_r10", {})


def f(v, d=1, suf=""):
    return S.fmt_es(v, d, suf)


def sgn(v, d=1, suf=""):
    if v is None or not math.isfinite(v):
        return "—"
    prefix = "+" if v > 0 else ("−" if v < 0 else "")
    return prefix + S.fmt_es(abs(v), d, suf)


def font_uri(name: str) -> str:
    return (FONTS / name).resolve().as_uri()


def legend_bar(include_tests=True, include_hist=False) -> str:
    items = [
        '<span class="leg-item"><span class="leg-line" style="background:#0E162B;"></span>Datos reales</span>',
    ]
    if include_hist:
        items.append('<span class="leg-item"><span class="leg-line" style="background:#94A3B8;height:1.5px;"></span>Historia L1</span>')
    if include_tests:
        items.append('<span class="leg-item"><span class="leg-dash" style="border-color:#2F66D0;"></span>Desde 2000</span>')
        items.append('<span class="leg-item"><span class="leg-dash" style="border-color:#1F7A6E;"></span>Desde 2010</span>')
    items.append('<span class="leg-item"><span class="leg-band"></span>Proyección 2019–2041 (mediana y p10–p90)</span>')
    items.append('<span class="leg-item"><span class="leg-shade"></span>2026–2041</span>')
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
  margin: 16.5mm 16mm 15.5mm 16mm;
  @top-left {{
    content: "EL ATLAS DEL SISTEMA-MUNDO, A PRUEBA DE QUINCE AÑOS";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.4pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    vertical-align: bottom;
    padding-bottom: 5mm;
  }}
  @top-right {{
    content: string(chap);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.4pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    vertical-align: bottom;
    padding-bottom: 5mm;
  }}
  @bottom-left {{
    content: "PHY-WORLD · CAPA L1 (ATLAS MUNDO)";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.4pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.5mm;
  }}
  @bottom-right {{
    content: counter(page);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.8pt;
    color: #475569;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.5mm;
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
  font-size: 8.6pt;
  line-height: 1.44;
  color: #242C3D;
}}

.page {{
  page-break-after: always;
  break-after: page;
}}
.page:last-child {{
  page-break-after: auto;
  break-after: auto;
}}
.chap-mark {{
  string-set: chap content();
  height: 0;
  overflow: hidden;
  visibility: hidden;
  font-size: 0;
  line-height: 0;
  margin: 0;
  padding: 0;
}}

/* ── COVER ── */
.cover-page {{
  page: cover;
  width: 210mm;
  height: 297mm;
  background: #0E162B;
  color: #FFFFFF;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 15mm 16mm 12mm 16mm;
}}
.cover-top {{
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 7.2pt;
  letter-spacing: 1.4px;
  color: #8B9AB8;
}}
.cover-title-block {{
  margin-top: 10mm;
  max-width: 158mm;
}}
.cover-title {{
  font-family: 'Newsreader', serif;
  font-weight: 700;
  font-size: 40pt;
  line-height: 1.04;
  letter-spacing: -0.6px;
  margin: 0 0 6mm 0;
  color: #FFFFFF;
}}
.cover-title em {{
  font-style: italic;
  font-weight: 600;
  color: #E8742D;
  display: block;
}}
.cover-sub {{
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 11pt;
  line-height: 1.48;
  color: #B4C0D6;
  max-width: 140mm;
  margin: 0;
}}
.cover-chart-wrap {{
  margin: 4mm -16mm 2mm -16mm;
}}
.cover-kpis {{
  border-top: 0.7pt solid #2B3959;
  padding-top: 6mm;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8mm;
}}
.cover-kpi-num {{
  font-family: 'Newsreader', serif;
  font-size: 23pt;
  font-weight: 700;
  line-height: 1.1;
  color: #FFFFFF;
  margin-bottom: 2mm;
}}
.cover-kpi-num span {{ color: #E8742D; font-size: 15pt; }}
.cover-kpi-desc {{
  font-size: 7.8pt;
  line-height: 1.38;
  color: #9EACC7;
}}
.cover-foot {{
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 6.4pt;
  letter-spacing: 0.9px;
  color: #6B7C9E;
  margin-top: 4mm;
}}

/* ── TYPOGRAPHY ── */
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 7pt;
  font-weight: 500;
  letter-spacing: 1px;
  color: #D95D24;
  text-transform: uppercase;
  margin: 0 0 2mm 0;
}}
h1 {{
  font-family: 'Newsreader', serif;
  font-weight: 700;
  font-size: 24pt;
  line-height: 1.1;
  letter-spacing: -0.3px;
  color: #0E162B;
  margin: 0 0 4mm 0;
}}
h2 {{
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  font-size: 10pt;
  color: #0E162B;
  margin: 4mm 0 2mm 0;
}}
.lede {{
  font-family: 'Newsreader', serif;
  font-weight: 400;
  font-size: 11.2pt;
  line-height: 1.44;
  color: #263147;
  margin: 0 0 5.5mm 0;
}}
p {{ margin: 0 0 3mm 0; }}
.muted {{ color: #64748B; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}

/* ── GRIDS & CARDS ── */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }}
.grid-2-tight {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4.5mm; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4.5mm; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 3.2mm; }}

.kpi-card {{
  border-top: 1.6pt solid #0E162B;
  padding-top: 3mm;
  margin-bottom: 2mm;
}}
.kpi-num {{
  font-family: 'Newsreader', serif;
  font-weight: 700;
  font-size: 21pt;
  line-height: 1.12;
  color: #0E162B;
  margin-bottom: 1.5mm;
}}
.kpi-num .unit {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 8.5pt;
  font-weight: 400;
  color: #64748B;
}}
.kpi-title {{
  font-weight: 600;
  font-size: 8.6pt;
  color: #0E162B;
  margin-bottom: 1.5mm;
}}
.kpi-body {{
  font-size: 8.1pt;
  line-height: 1.4;
  color: #374151;
}}

.slate-box {{
  background: #F4F6F9;
  border-radius: 2.5mm;
  padding: 4mm 4.8mm;
  margin: 3.5mm 0;
}}
.slate-box h3 {{
  font-size: 8.8pt;
  font-weight: 600;
  color: #0E162B;
  margin: 0 0 1.8mm 0;
}}
.slate-box p {{ font-size: 8.1pt; line-height: 1.42; margin: 0 0 2mm 0; color: #334155; }}
.slate-box p:last-child {{ margin-bottom: 0; }}

.card-col {{
  background: #F4F6F9;
  border-radius: 2.5mm;
  padding: 4mm 4.5mm;
}}
.card-col .swatch {{
  display: inline-block;
  width: 14px;
  height: 3px;
  vertical-align: middle;
  margin-right: 5px;
  border-radius: 1px;
}}
.card-col h3 {{
  font-size: 8.8pt;
  font-weight: 600;
  margin: 0 0 2.5mm 0;
  color: #0E162B;
}}
.card-col p, .card-col li {{
  font-size: 7.9pt;
  line-height: 1.4;
  color: #334155;
}}
.card-col ol, .card-col ul {{
  margin: 1.5mm 0 0 0;
  padding-left: 4.5mm;
}}
.card-col li {{ margin-bottom: 1.2mm; }}

/* ── CONFIDENCE CARDS (CHAPTER 10) ── */
.conf-card {{
  border-top: 2.2pt solid #0E162B;
  padding-top: 2.5mm;
}}
.conf-card.green {{ border-top-color: #1F7A54; }}
.conf-card.amber {{ border-top-color: #B58100; }}
.conf-card.red {{ border-top-color: #B83227; }}
.conf-head {{
  display: flex;
  align-items: baseline;
  gap: 2.5mm;
  margin-bottom: 2mm;
}}
.conf-head strong {{ font-size: 9.2pt; color: #0E162B; }}
.conf-head span {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 6.2pt;
  color: #64748B;
  text-transform: uppercase;
}}
.conf-card ul {{
  margin: 0;
  padding-left: 3.8mm;
}}
.conf-card li {{
  font-size: 7.8pt;
  line-height: 1.38;
  color: #334155;
  margin-bottom: 1.6mm;
}}

/* ── TOC ── */
.toc-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5mm 8mm;
  margin-top: 2.5mm;
}}
.toc-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  border-bottom: 0.5pt solid #E5E9F0;
  padding: 1.4mm 0;
  font-size: 8.2pt;
}}
.toc-num {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 7pt;
  color: #94A3B8;
  margin-right: 2.5mm;
}}
.toc-pg {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 7.4pt;
  color: #475569;
}}

/* ── LEGEND BAR ── */
.legend-bar {{
  display: flex;
  flex-wrap: nowrap;
  gap: 3.2mm;
  align-items: center;
  padding: 1.2mm 0 2mm 0;
  border-bottom: 0.6pt solid #E5E9F0;
  margin-bottom: 2.2mm;
  font-size: 6.7pt;
  color: #475569;
  white-space: nowrap;
}}
.leg-item {{ display: inline-flex; align-items: center; gap: 1.4mm; }}
.leg-line {{ display: inline-block; width: 12px; height: 2px; }}
.leg-dash {{ display: inline-block; width: 12px; height: 0; border-top: 1.6px dashed; }}
.leg-band {{ display: inline-block; width: 12px; height: 6.5px; background: rgba(217, 93, 36, 0.24); border-top: 1.8px solid #D95D24; }}
.leg-shade {{ display: inline-block; width: 10px; height: 6.5px; background: #F1F5F9; border: 0.5pt solid #CBD5E1; }}

/* ── CHARTS ── */
.chart-box {{ margin-bottom: 1.2mm; }}
.chart-cap {{
  font-size: 7.0pt;
  line-height: 1.3;
  color: #475569;
  margin-top: 0.8mm;
}}
.chart-rule {{
  border-top: 1.1pt solid #0E162B;
  padding-top: 1.5mm;
}}

/* ── TABLES ── */
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 7.5pt;
  margin: 2mm 0 3mm 0;
}}
thead th {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 6.2pt;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748B;
  text-align: left;
  padding: 1.6mm 1.5mm;
  border-top: 1.2pt solid #0E162B;
  border-bottom: 0.8pt solid #94A3B8;
}}
thead th.num, tbody td.num {{
  text-align: right;
  font-family: 'IBM Plex Mono', monospace;
  font-variant-numeric: tabular-nums;
}}
tbody td {{
  padding: 1.35mm 1.5mm;
  border-bottom: 0.45pt solid #E6ECF2;
  color: #1E293B;
  vertical-align: top;
}}
tbody tr.group-row td {{
  background: #F4F6F9;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 6.4pt;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #0E162B;
  padding: 1.3mm 1.5mm;
  border-top: 0.7pt solid #CBD5E1;
  border-bottom: 0.6pt solid #CBD5E1;
}}
.hi-orange {{ color: #D95D24; font-weight: 500; }}
.hi-blue {{ color: #2F66D0; font-weight: 500; }}
.hi-green {{ color: #1F7A54; font-weight: 500; }}
.hi-red {{ color: #B83227; font-weight: 500; }}

/* ── PHASE LEDGER (CHAPTER 02) ── */
.phase-row {{
  display: grid;
  grid-template-columns: 22mm 1fr;
  gap: 3.5mm;
  border-top: 0.6pt solid #D9E2EC;
  padding: 1.55mm 0;
}}
.phase-tag {{
  font-family: 'Newsreader', serif;
  font-weight: 700;
  font-size: 12pt;
  line-height: 1.05;
  color: #0E162B;
}}
.phase-sub {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 5.5pt;
  letter-spacing: 0.4px;
  color: #64748B;
  text-transform: uppercase;
  margin-top: 0.5mm;
}}
.phase-title {{ font-weight: 600; font-size: 8.1pt; color: #0E162B; margin-bottom: 0.5mm; }}
.phase-desc {{ font-size: 7.4pt; line-height: 1.3; color: #334155; margin-bottom: 0.9mm; }}
.phase-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3.5mm; }}
.phase-mini-lbl {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 5.5pt;
  color: #64748B;
  text-transform: uppercase;
  margin-bottom: 0.3mm;
}}
.phase-mini-txt {{ font-size: 7.1pt; line-height: 1.28; color: #475569; }}

/* ── CODE BLOCK ── */
pre.code {{
  background: #F4F6F9;
  border-radius: 2mm;
  padding: 3mm 3.5mm;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 7pt;
  line-height: 1.45;
  color: #1E293B;
  margin: 2mm 0 3mm 0;
}}

.country-card {{
  border-top: 0.8pt solid #CBD5E1;
  padding-top: 1.4mm;
}}
.tight-table td, .tight-table th {{
  padding: 1.05mm 1.3mm;
  font-size: 7.0pt;
}}
.tight-table thead th {{ font-size: 5.9pt; }}
"""


# ═══════════════════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def page_01_cover() -> str:
    hero = S.cover_hero_svg(D)
    gdp25_o = OW["gdp_pc_world"][75]
    gdp25_s = S19["gdp_pc_world"]["p50"][6]
    err_gdp25 = (gdp25_s / gdp25_o - 1.0) * 100
    res41 = S19["reserves"]["p50"][22] * 100
    gdp26 = S19["gdp_pc_world"]["p50"][7]
    gdp41 = S19["gdp_pc_world"]["p50"][22]
    g15 = (gdp41 / gdp26 - 1.0) * 100
    return f"""
<div class="page cover-page">
  <div>
    <div class="cover-top">
      <span>PHY-WORLD · CAPA L1 · ATLAS MUNDO</span>
      <span>INFORME · NOVIEMBRE DE 2026</span>
    </div>
    <div class="cover-title-block">
      <h1 class="cover-title">El Atlas del Sistema-Mundo,<em>a prueba de quince años</em></h1>
      <p class="cover-sub">La capa L1 de Phy-World —180 países, siete teorías, Democracia V2 (<code>democracy.fit</code>), Élites Regionales R10 (<code>elites.py</code>) y capa biofísica y de colas pesadas (M0–M3: CO₂/TCRE sin colapso Hubbert, clima persistente, frenos alimentarios y colas Student-<em>t</em>)— verificada hasta 2025 y proyectada a 2041.</p>
    </div>
  </div>

  <div class="cover-chart-wrap">{hero}</div>

  <div>
    <div class="cover-kpis">
      <div>
        <div class="cover-kpi-num">{sgn(RW['co2']['rel_err_last_pct'], 1, " %")} · {sgn(RW['Tg']['rel_err_last_pct'], 1, " %")}</div>
        <div class="cover-kpi-desc">Error en CO₂ de 2024 ({f(S19['co2']['p50'][5], 1)} vs 38,6 Gt) y GMST de 2025 ({f(S19['Tg']['p50'][6], 2)} vs 1,41 °C) tras corregir con M0 el colapso artificial de Hubbert (cobertura a 15 años: 86 % y 80 %).</div>
      </div>
      <div>
        <div class="cover-kpi-num">{f(S19['demo']['p50'][6]*100, 1, " %")}</div>
        <div class="cover-kpi-desc">Democracias en 2025 con <code>democracy.fit</code> V2 frente al {f(OW['demo_share'][75]*100, 1, " %")} real (error −0,7 %, AUC 0,980): captura la recesión democrática y frena el rebote en 2041 ({f(S19['demo']['p50'][22]*100, 1, " %")}).</div>
      </div>
      <div>
        <div class="cover-kpi-num">{sgn(g15, 1, " %")}</div>
        <div class="cover-kpi-desc">Crecimiento mediano del PIB pc mundial en 2026–2041 (meseta en {f(gdp41, 0)} USD bajo {f(S19['Tg']['p50'][22], 2)} °C): el L1 legado sobreestimaba +{f(D['ablations_2041']['l1_legado_9e96f6b']['gdp_pc_world']['delta_pct'], 1, " %")}.</div>
      </div>
    </div>
    <div class="cover-foot">
      <span>180 PAÍSES · 100 MUNDOS · L1 POR DEFECTO (STRESS M0–M3 + DEMOCRACIA V2)</span>
      <span>PRUEBA 2000–2025 Y PROYECCIÓN 2026–2041</span>
    </div>
  </div>
</div>
"""


def page_02_executive() -> str:
    sc10 = SC["from_2010_to_2025_15y"]
    sc19 = SC["from_2019_to_2025_recent"]
    ab = D["ablations_2041"]
    gdp25_o, gdp25_s = OW["gdp_pc_world"][75], S19["gdp_pc_world"]["p50"][6]
    err_gdp25 = (gdp25_s / gdp25_o - 1.0) * 100.0
    err_med25 = RW["gdp_pc_med"]["rel_err_last_pct"]
    demo25_o, demo25_s = OW["demo_share"][75] * 100.0, S19["demo"]["p50"][6] * 100.0
    return f"""
<div class="page">
  <span class="chap-mark">EN UNA PÁGINA</span>
  <div class="eyebrow">En una página</div>
  <h1>Qué acertó la capa L1,<br>dónde tropieza y qué dice de 2041</h1>
  <p class="lede">La capa L1 de Phy-World —el <em>Atlas del Sistema-Mundo</em> en su nuevo modo por defecto— integra 180 países-agente con <strong>Democracia V2 (<code>democracy.fit</code>)</strong>, <strong>Élites Regionales R10 (<code>elites.py</code>)</strong> y la <strong>capa biofísica y de colas pesadas (<code>stress.py</code>: M0 CO₂/TCRE sin colapso Hubbert, M1 clima persistente y macro &gt;1,5 °C, M2 frenos alimentarios y M3 colas Student-<em>t</em>)</strong>. Este informe la somete a tres pruebas ciegas hasta 2025 y proyecta todos sus parámetros hasta 2041.</p>

  <div class="grid-2">
    <div class="kpi-card">
      <div class="kpi-num">{f(sc10['co2']['mape_pct'], 1, " %")} <span class="unit">MAPE CO₂ 15a</span> · {f(sc10['gmst']['cover_80']*100, 0, " %")} <span class="unit">cob. GMST</span></div>
      <div class="kpi-title">M0 resuelve el colapso de CO₂ y la cobertura nula de Tg</div>
      <div class="kpi-body">Desacoplar CO₂ y <code>Tg</code> de Hubbert (intensidad de carbono + TCRE efectiva OLS ≤ origen + ENSO) reduce el MAPE de CO₂ a 15 años de 31,7 % a <strong>{f(sc10['co2']['mape_pct'], 1, " %")}</strong> (bate a la persistencia 7,9 %), clava 2024–25 (<strong>39,2 Gt</strong> y <strong>1,40 °C</strong>) y eleva la cobertura p10–p90 de <code>Tg</code> de 0 % al <strong>80–84 %</strong>.</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{f(sc10['democracy']['auc'], 3)} <span class="unit">democ. 15a</span> · {f(sc19['democracy']['auc'], 3)} <span class="unit">2019–25</span></div>
      <div class="kpi-title">Democracia V2 clava la recesión democrática de 2025</div>
      <div class="kpi-body"><code>democracy.fit</code> (poliarquía continua + contagio vecinal + momento de 3 años) acierta la caída real en 2025: <strong>{f(demo25_s, 1, " %")}</strong> frente al <strong>{f(demo25_o, 1, " %")}</strong> real (error −0,7 %, Brier 0,058 vs 0,084 de persistencia) y reemplaza el rebote legado (61,7 %) por estancamiento en <strong>{f(S19['demo']['p50'][22]*100, 1, " %")}</strong> en 2041.</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">+{f(ab['sin_clima_ecologia']['gdp_pc_world']['delta_pct'], 1, " %")} <span class="unit">sin clima</span> · +{f(ab['sin_colas_pesadas_M3']['gdp_pc_world']['delta_pct'], 1, " %")} <span class="unit">sin M3</span></div>
      <div class="kpi-title">Meseta económica en 2026–2041 bajo +1,80 °C</div>
      <div class="kpi-body">Con <code>Tg</code> cruzando 1,5 °C hacia 2029 y llegando a <strong>{f(S19['Tg']['p50'][22], 2)} °C</strong> en 2041, se activa el arrastre macro <code>d_glob</code> (−2,1 %): el PIB pc mundial entra en <strong>meseta (−0,1 % en 15 años, 17.650 USD)</strong>. Clima+M0+M1 (+{f(ab['sin_clima_ecologia']['gdp_pc_world']['delta_pct'], 1, " %")} al apagarlo), M3 (+{f(ab['sin_colas_pesadas_M3']['gdp_pc_world']['delta_pct'], 1, " %")}) y M2 (−{f(abs(ab['sin_frenos_alimentos_M2']['hazard']['delta_pct']), 1, " %")} conflicto) dominan 2041.</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{f(sc19['gdp_pc']['cover_80']*100, 1, " %")} <span class="unit">cob. PIB 19–25</span> · +{f(ab['l1_legado_9e96f6b']['gdp_pc_world']['delta_pct'], 1, " %")} <span class="unit">legado</span></div>
      <div class="kpi-title">Bandas calibradas y fin del sesgo optimista legado</div>
      <div class="kpi-body">M3 eleva la cobertura p10–p90 de PIB en 2019–25 al <strong>{f(sc19['gdp_pc']['cover_80']*100, 1, " %")}</strong> (85,6 % a 15 años). Volver al Atlas legado (<code>legacy=True</code>) inflaría el PIB de 2041 en <strong>+{f(ab['l1_legado_9e96f6b']['gdp_pc_world']['delta_pct'], 1, " %")}</strong>, las democracias en +{f(ab['l1_legado_9e96f6b']['demo']['delta_pct'], 1, " %")} y subestimaría el conflicto en {sgn(ab['l1_legado_9e96f6b']['hazard']['delta_pct'], 1, " %")}.</div>
    </div>
  </div>

  <div class="slate-box">
    <h3>Cómo leer este informe</h3>
    <p>El horizonte validado del Atlas es de <strong>10 años</strong>; extenderlo a 15 años (2026–2041) es una <strong>prueba de estrés estructural</strong>. Con el nuevo modo por defecto (<strong>Democracia V2 + Stress M0–M3</strong>, conservando <code>legacy=True</code> para la paridad bit a bit W2 de <code>9e96f6b</code>), L1 resuelve el colapso de CO₂/Tg, el sesgo democrático y la subcobertura de bandas en PIB, CO₂ y temperatura.</p>
  </div>

  <h2>Contenido</h2>
  <div class="toc-grid">
    <div class="toc-row"><span><span class="toc-num">01</span>Qué se prueba y cómo se lee</span><span class="toc-pg">3</span></div>
    <div class="toc-row"><span><span class="toc-num">06</span>Centro, semiperiferia, periferia y 12 países</span><span class="toc-pg">10</span></div>
    <div class="toc-row"><span><span class="toc-num">02</span>Siete teorías y seis canales de conducta</span><span class="toc-pg">4</span></div>
    <div class="toc-row"><span><span class="toc-num">07</span>Qué mueve los próximos 15 años (ablaciones)</span><span class="toc-pg">11</span></div>
    <div class="toc-row"><span><span class="toc-num">03</span>Lo que ya pasó: pruebas 2000, 2010 y 2019–2025</span><span class="toc-pg">5</span></div>
    <div class="toc-row"><span><span class="toc-num">08</span>Mapa completo de los 38 parámetros a 2041</span><span class="toc-pg">12</span></div>
    <div class="toc-row"><span><span class="toc-num">04</span>Indicador por indicador, de 1970 a 2041</span><span class="toc-pg">6</span></div>
    <div class="toc-row"><span><span class="toc-num">09</span>Estabilidad de parámetros y prueba de bandas</span><span class="toc-pg">14</span></div>
    <div class="toc-row"><span><span class="toc-num">05</span>Canales de gasto y el colapso Hubbert</span><span class="toc-pg">9</span></div>
    <div class="toc-row"><span><span class="toc-num">10</span>Conclusiones: cuánto fiarse, límites y qué sigue</span><span class="toc-pg">15</span></div>
  </div>
</div>
"""


def page_03_what_is_compared() -> str:
    sample_chart = S.ts_chart(
        "PIB per cápita mundial",
        "DÓLARES CONSTANTES 2017 · PONDERADO POR POBLACIÓN · 180 PAÍSES",
        obs=(Y_OBS, OW["gdp_pc_world"]),
        test_past=(Y_00, S00["gdp_pc_world"]["p50"]),
        test_past2=(Y_10, S10["gdp_pc_world"]["p50"]),
        proj=(Y_19, S19["gdp_pc_world"]["p10"], S19["gdp_pc_world"]["p50"], S19["gdp_pc_world"]["p90"]),
        xmin=1970, xmax=2041, ymin=4000, ymax=27000,
        y_fmt=lambda v: f"{v/1000:.0f}k",
        end_fmt=lambda v: f"{v/1000:.1f}k".replace(".", ","),
        width=310, height=145,
        vlines=[(2019, "2019"), (2026, "2026")],
        xticks=[1970, 2000, 2019, 2041],
    )
    return f"""
<div class="page">
  <span class="chap-mark">01 · QUÉ SE PRUEBA</span>
  <div class="eyebrow">Capítulo 01</div>
  <h1>Qué se prueba</h1>
  <p class="lede">Cuatro piezas en cada gráfico: la realidad medida hasta 2025, dos pruebas ciegas iniciadas en el pasado y la proyección de los próximos quince años desde el último ancla del panel.</p>

  <div class="grid-3">
    <div class="card-col">
      <h3><span class="swatch" style="background:#0E162B;"></span>Datos reales (1950–2025)</h3>
      <p>Hasta <strong>2019</strong>, el panel verificado del Atlas (180 países): PWT 10.01 (PIB, capital, población), WDI (rentas de recursos), UCDP/PRIO (conflicto civil), V-Dem (democracia), WID (cuota Top 10 %), Global Macro Database (deuda) y HadCRUT5 / GCB (clima y CO₂).</p>
      <p>De <strong>2020 a 2025</strong> se empalman series oficiales recientes jamás vistas por el modelo: WDI PPP (2025), GMD deuda (2024), V-Dem RoW (2025), UCDP ACD v24.1 (2023), GCB CO₂ (2024) y HadCRUT5 (2025).</p>
    </div>
    <div class="card-col">
      <h3><span class="swatch" style="background:#2F66D0;"></span>Pruebas ciegas al pasado</h3>
      <p>Evaluar un modelo sobre datos que ya vio no demuestra capacidad predictiva. Por eso se ejecutan tres cortes estrictos sin fuga de futuro:</p>
      <ol>
        <li><strong>Desde 2000</strong> (25 años, <code>params_2000</code>): calibra y aprende conducta solo hasta 2000; corre libre hasta 2025.</li>
        <li><strong>Desde 2010</strong> (15 años exactos, <code>params_2010</code>): mide la precisión real al mismo horizonte que proyectamos hoy.</li>
        <li><strong>Desde 2019</strong> (2020–2025): examen ciego post-panel sobre pandemia, inflación, guerras y récord térmico.</li>
      </ol>
    </div>
    <div class="card-col">
      <h3><span class="swatch" style="background:#D95D24;"></span>Proyección L1 (2026–2041)</h3>
      <p>El Atlas portado (commit <code>9e96f6b</code>), calibrado con datos hasta 2019 y anclado en 2019, corre <strong>100 mundos Monte Carlo</strong> hasta 2041 con la regla de conducta estimada (<em>behavioral cloning</em>) para los 6 canales de gasto.</p>
      <p>Se auditan sus <strong>38 variables de salida</strong>, las tres zonas del sistema-mundo, 12 países clave, 7 ablaciones teóricas y la sensibilidad al congelamiento demográfico de 2030.</p>
    </div>
  </div>

  <h2>Por qué iniciar en el pasado cambia el veredicto</h2>
  <div class="grid-2">
    <p>El panel del Atlas cierra en <strong>2019</strong>. Eso nos da una oportunidad científica excepcional en noviembre de 2026: disponemos de <strong>seis años de historia real posterior al cierre del modelo</strong> (2020–2025) que ningún parámetro, ninguna regresión de etapa 1 y ninguna regla de conducta pudo haber visto durante el entrenamiento.</p>
    <p>Al combinar ese examen post-panel con la prueba de 15 años (<strong>2010 → 2025</strong>) y la validación oficial de <strong>49 orígenes móviles (1970–2018)</strong>, podemos distinguir qué comportamientos de 2026–2041 son robustos y cuáles son artefactos mecánicos del modelo aislado.</p>
  </div>

  <h2>Cómo leer un gráfico</h2>
  <div class="grid-2" style="align-items:center;">
    <div>{sample_chart}</div>
    <div style="font-size:8pt;line-height:1.48;">
      <p><span style="display:inline-block;width:16px;height:2px;background:#0E162B;vertical-align:middle;margin-right:6px;"></span><strong>Datos reales.</strong> El punto marca el último año observado (2019 a 2025 según la serie).</p>
      <p><span style="display:inline-block;width:16px;border-top:1.8px dashed #2F66D0;vertical-align:middle;margin-right:6px;"></span><strong>Prueba ciega desde 2000</strong> (azul) y <span style="display:inline-block;width:14px;border-top:1.8px dashed #1F7A6E;vertical-align:middle;margin:0 3px;"></span><strong>desde 2010</strong> (verde): calibrados solo hasta ese corte.</p>
      <p><span style="display:inline-block;width:16px;height:8px;background:rgba(217,93,36,0.22);border-top:2px solid #D95D24;vertical-align:middle;margin-right:6px;"></span><strong>Proyección Atlas L1 desde 2019:</strong> mediana y banda del percentil 10 al 90 de 100 mundos.</p>
      <p class="muted" style="margin-top:2mm;">La línea vertical punteada marca <strong>2019</strong> (último dato del panel). La franja gris suave a la derecha señala los <strong>próximos quince años (2026–2041)</strong>. La cifra del margen derecho es la mediana de 2041.</p>
    </div>
  </div>
</div>
"""


def page_04_architecture() -> str:
    bm = D["behavior_2019"]
    ch_desc = {
        "k": ("Capital doméstico", "Formación bruta de capital fijo / PIB"),
        "m": ("Importaciones del centro", "Importaciones manufactureras / PIB"),
        "x": ("Extracción propia", "Re-inversión de rentas de recursos"),
        "r": ("Tecnología e industria", "I+D + educación pública / PIB"),
        "f": ("Inversión exterior", "IED saliente en recursos / PIB"),
        "w": ("Redistribución social", "Gasto social y transferencias / PIB"),
    }
    ch_rows = ""
    for ch in ("k", "m", "x", "r", "f", "w"):
        r = bm[ch]
        nm, prx = ch_desc[ch]
        ch_rows += (
            f"<tr><td><strong>{ch}</strong> · {nm}</td><td class='muted'>{prx}</td>"
            f"<td class='num'>{f(r['rho'], 3)}</td><td class='num'>{f(r['r2'], 3)}</td>"
            f"<td class='num'>{f(r['sd'], 3)}</td><td class='num'>{f(r['n'], 0)}</td></tr>"
        )
    return f"""
<div class="page">
  <span class="chap-mark">02 · ARQUITECTURA DE L1</span>
  <div class="eyebrow">Capítulo 02</div>
  <h1>Siete teorías acopladas y<br>seis canales de conducta</h1>
  <p class="lede" style="font-size:10.3pt;margin-bottom:3mm;">Cada año, 180 agentes-país ejecutan un paso acoplado de producción, comercio bilateral, extracción, finanzas, instituciones, conflicto y clima. Dos etapas calibran sus parámetros: 28 por máxima verosimilitud directa (etapa 1) y 23 por simulación (etapa 2).</p>

  <div class="phase-row">
    <div><div class="phase-tag">M1</div><div class="phase-sub">Producción y técnica</div></div>
    <div>
      <div class="phase-title">Cobb-Douglas con capital, capital humano, recursos y saltos estocásticos de PTF</div>
      <div class="phase-desc">Y = A·K<sup>α</sup>·R<sup>ψ</sup>·(h·L)<sup>1−α−ψ</sup>. La productividad (ln A) converge hacia la frontera vía comercio, IED y capacidad absortiva (h<sup>1,66</sup>), más saltos Poisson impulsados por el canal <code>r</code> y la <em>asabiya</em>.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">Reproduce el ordenamiento mundial de PIB pc a 15 años (Spearman 0,965) y bate a la persistencia desde 5–10 años.</div></div>
        <div><div class="phase-mini-lbl">Punto débil</div><div class="phase-mini-txt"><code>pop</code> y <code>hc</code> son exógenos (D1/D2); en L1 puro la población WPP acaba en 2030 y se congela hasta 2041.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">M2</div><div class="phase-sub">Sistema-Mundo</div></div>
    <div>
      <div class="phase-title">Centralidad, intercambio desigual bilateral y rentas monopólicas (Wallerstein / Amin)</div>
      <div class="phase-desc">Sobre la red comercial bilateral COW (congelada en 2014), la periferia pierde una fracción por brechas de precios (<code>θ<sub>d</sub> = 0,20</code>) y paga rentas monopólicas (<code>μ = 0,25</code>) por manufacturas del centro.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">Estructura los flujos netos de renta (<code>rent_in</code>) y alimenta la frontera de solidaridad en la periferia.</div></div>
        <div><div class="phase-mini-lbl">Punto débil</div><div class="phase-mini-txt">Casi suma cero a escala mundial: apagarlo solo mueve el PIB mundial de 2041 en −0,22 %.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">M3</div><div class="phase-sub">Demografía estructural</div></div>
    <div>
      <div class="phase-title">Captura de élites (E), sobreproducción (n<sub>e</sub>) e índice de estrés político (Goldstone / Turchin)</div>
      <div class="phase-desc">Las élites capturan la cuota <code>E</code> del producto más rentas adicionales en autocracias; cuando su renta relativa supera a las masas crece <code>n<sub>e</sub></code> y sube <code>PSI = MMP·EMP·SFD</code>.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">Error medio razonable en cuota Top 10 % (MAE 2,4–4,2 pp), pero evolución casi determinista (cobertura 6–19 %).</div></div>
        <div><div class="phase-mini-lbl">Punto débil</div><div class="phase-mini-txt">El peso calibrado de PSI sobre conflicto (<code>b<sub>1</sub> = 0,064</code>) es pequeño; pesa mucho más la <em>asabiya</em> y las rentas.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">M4</div><div class="phase-sub">Frontera metaétnica</div></div>
    <div>
      <div class="phase-title">Solidaridad colectiva (<em>asabiya</em>) que protege frente al conflicto y eleva la eficiencia</div>
      <div class="phase-desc">Crece logísticamente en periferias abiertas expuestas al drenaje del centro (<code>r<sub>a</sub> = 0,02</code>) y decae en centros ricos con sobreproducción de élites (<code>d<sub>a</sub> = 0,007</code>) o durante guerras civiles.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">Palanca clave del riesgo bélico (<code>b<sub>3</sub> = −2,56</code>): su erosión por debajo de 0,50 impulsa +13 % el conflicto en 2041.</div></div>
        <div><div class="phase-mini-lbl">Punto débil</div><div class="phase-mini-txt">Variable latente sin serie observada directa tras el ancla inicial de capacidad estatal/cohesión.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">M5</div><div class="phase-sub">Recursos naturales</div></div>
    <div>
      <div class="phase-title">Curvas logísticas de Hubbert por país, concesiones extranjeras y nacionalizaciones</div>
      <div class="phase-desc">Cada país tiene un depósito fijo <code>U</code> cuya capacidad sigue una logística con pendiente <code>r<sub>R</sub> = 0,145</code>. Un precio mundial limpia la demanda dada la cuota global de rentas <code>ψ<sub>t</sub></code>.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">A corto plazo (2021) las rentas nominales/PIB coinciden en −2,1 % con el Banco Mundial.</div></div>
        <div><div class="phase-mini-lbl">Fallo crítico</div><div class="phase-mini-txt"><code>ψ<sub>t</sub></code> congelada (D3) + agotamiento rápido colapsan la extracción física y el CO₂, disparando el precio ×60.</div></div>
      </div>
    </div>
  </div>

  <div class="phase-row">
    <div><div class="phase-tag">M6–7+V2</div><div class="phase-sub">Deuda, V2 y Stress</div></div>
    <div>
      <div class="phase-title">Crisis financieras, Democracia V2 (<code>democracy.py</code>) y capa Stress M1–M3 (<code>stress.py</code>)</div>
      <div class="phase-desc"><strong>Democracia V2</strong> acopla poliarquía continua (V-Dem), momento de 3 años y contagio COW. <strong>Stress M1–M3</strong> añade clima persistente (<code>ρ<sub>D</sub>=0,94</code>, desastres ×), frenos alimentarios con mercado mundial y colas Student-<em>t</em>(3) + choques globales.</div>
      <div class="phase-cols">
        <div><div class="phase-mini-lbl">Fuerza medida</div><div class="phase-mini-txt">AUC democracia 0,957–0,980 (−0,7 % en 2025); cobertura PIB 2019–25 sube al 77,5 %.</div></div>
        <div><div class="phase-mini-lbl">Supuesto declarado</div><div class="phase-mini-txt">M1–M3 son supuestos <code>[M]</code>; <code>legacy=True</code> restaura <code>9e96f6b</code> bit a bit.</div></div>
      </div>
    </div>
  </div>

  <h2 style="margin-top:1.5mm;margin-bottom:1mm;">Los seis canales de conducta estimados con datos hasta 2019 (<code>BehaviorModel</code>)</h2>
  <table class="tight-table" style="font-size:6.8pt;line-height:1.15;">
    <thead>
      <tr><th>Canal</th><th>Proxy observado</th><th class="num">Persistencia ρ</th><th class="num">R²</th><th class="num">Desv. σ</th><th class="num">Obs. (n)</th></tr>
    </thead>
    <tbody>{ch_rows}</tbody>
  </table>
</div>
"""


def page_05_backtest() -> str:
    sc10 = SC["from_2010_to_2025_15y"]
    sc19 = SC["from_2019_to_2025_recent"]
    v49 = D["validation_49_origins"]

    # Paired horizontal bar chart: 15-year out-of-sample test (2010 -> 2025), normalized error metric
    # To make bars directly comparable visually, plot relative error or exact % metrics where comparable:
    # Actually let's plot exact % of Persistence error (Persistence = 100%, L1 = ratio*100%) — crystal clear AND fair across units!
    bar_specs = [
        ("PIB per cápita (CRPS país-año)", sc10["gdp_pc"]["persist_mae_log"], sc10["gdp_pc"]["crps"],
         f"{f(sc10['gdp_pc']['persist_mae_log'], 3)}", f"{f(sc10['gdp_pc']['crps'], 3)}"),
        ("Temperatura global (MAE °C)", sc10["gmst"]["persist_mae"], sc10["gmst"]["mae"],
         f"{f(sc10['gmst']['persist_mae'], 2)} °C", f"{f(sc10['gmst']['mae'], 2)} °C"),
        ("Conflicto civil (Brier)", sc10["conflict"]["persist_brier"], sc10["conflict"]["brier"],
         f"{f(sc10['conflict']['persist_brier'], 3)}", f"{f(sc10['conflict']['brier'], 3)}"),
        ("Democracia (Brier)", sc10["democracy"]["persist_brier"], sc10["democracy"]["brier"],
         f"{f(sc10['democracy']['persist_brier'], 3)}", f"{f(sc10['democracy']['brier'], 3)}"),
        ("Rentas de recursos (MAE pp)", sc10["res_share"]["persist_mae"], sc10["res_share"]["mae"],
         f"{f(sc10['res_share']['persist_mae']*100, 1)} pp", f"{f(sc10['res_share']['mae']*100, 1)} pp"),
        ("Deuda pública / PIB (MAE pp)", sc10["debt_gdp"]["persist_mae"], sc10["debt_gdp"]["mae"],
         f"{f(sc10['debt_gdp']['persist_mae']*100, 1)} pp", f"{f(sc10['debt_gdp']['mae']*100, 1)} pp"),
        ("Emisiones CO₂ mundial (MAPE)", sc10["co2"]["persist_mape_pct"], sc10["co2"]["mape_pct"],
         f"{f(sc10['co2']['persist_mape_pct'], 1)} %", f"{f(sc10['co2']['mape_pct'], 1)} %"),
    ]
    # Normalize each bar to Persistence = 100 % so all 7 heterogeneous metrics compare cleanly on one chart!
    bar_rows = []
    for lbl, vp, vm, tp, tm in bar_specs:
        ratio = (vm / vp) * 100.0 if vp > 0 else 100.0
        bar_rows.append((lbl, 100.0, min(ratio, 220.0), f"100 % ({tp})", f"{ratio:.0f} % ({tm})"))

    hbar = S.paired_hbar_chart(
        bar_rows,
        label_a="Persistencia / frecuencia pasada (= 100 %)",
        label_b="Atlas L1 · prueba ciega 15 años (2010 → 2025)",
        max_val=250.0,
        row_h=23,
    )

    return f"""
<div class="page">
  <span class="chap-mark">03 · LO QUE YA PASÓ</span>
  <div class="eyebrow">Capítulo 03</div>
  <h1>Lo que ya pasó: tres pruebas<br>ciegas hasta 2025</h1>
  <p class="lede">Antes de mirar 2041, medimos cuánto acierta la capa L1 cuando corre sin ver el futuro: en la validación histórica de 49 orígenes, a quince años exactos (2010 → 2025) y en el examen post-panel jamás visto (2019 → 2025).</p>

  <h2 style="margin-top:1mm;">Error relativo frente a la persistencia a 15 años (2010 → 2025)</h2>
  <p class="muted" style="font-family:'IBM Plex Mono';font-size:6.3pt;margin-bottom:2mm;">ÍNDICE: PERSISTENCIA = 100 % · CUANTO MÁS CORTA LA BARRA NARANJA, MEJOR · CON M0, CO₂ Y GMST BATEN AMPLIAMENTE A LA PERSISTENCIA</p>
  {hbar}

  <h2 style="margin-top:3mm;">Marcador comparado: 49 orígenes (10 a), prueba de 15 años y examen reciente 2020–2025</h2>
  <table class="tight-table">
    <thead>
      <tr>
        <th>Indicador (país-año o mundo)</th>
        <th class="num">49 orígenes (10 a)<br>L1 vs persist.</th>
        <th class="num">Prueba 15 años (2010→25)<br>L1 vs persist.</th>
        <th class="num">Post-panel (2019→25)<br>L1 vs persist.</th>
        <th class="num">Cobertura<br>p10–p90</th>
        <th class="num">Último<br>dato</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>PIB per cápita</strong> (CRPS log)</td>
        <td class="num"><span class="hi-green">{f(v49['gdp']['10']['completo']['crps'], 3)}</span> vs {f(v49['gdp']['10']['persistencia']['crps'], 3)}</td>
        <td class="num"><span class="hi-green">{f(sc10['gdp_pc']['crps'], 3)}</span> vs {f(sc10['gdp_pc']['persist_mae_log'], 3)}</td>
        <td class="num"><span class="hi-red">{f(sc19['gdp_pc']['crps'], 3)}</span> vs {f(sc19['gdp_pc']['persist_mae_log'], 3)}</td>
        <td class="num">{f(sc10['gdp_pc']['cover_80']*100, 0)} / {f(sc19['gdp_pc']['cover_80']*100, 0)} %</td>
        <td class="num">2025</td>
      </tr>
      <tr>
        <td><strong>Conflicto civil</strong> (AUC · Brier)</td>
        <td class="num">0,77 · <span class="hi-green">0,095</span> vs 0,102</td>
        <td class="num"><span class="hi-green">{f(sc10['conflict']['auc'], 2)} · {f(sc10['conflict']['brier'], 3)}</span> vs {f(sc10['conflict']['persist_brier'], 3)}</td>
        <td class="num"><span class="hi-green">{f(sc19['conflict']['auc'], 2)}</span> · {f(sc19['conflict']['brier'], 3)} vs {f(sc19['conflict']['persist_brier'], 3)}</td>
        <td class="num">—</td>
        <td class="num">2023</td>
      </tr>
      <tr>
        <td><strong>Democracia V2</strong> (AUC · Brier)</td>
        <td class="num">0,86 · 0,141 vs 0,133</td>
        <td class="num"><span class="hi-green">{f(sc10['democracy']['auc'], 3)} · {f(sc10['democracy']['brier'], 3)}</span> vs {f(sc10['democracy']['persist_brier'], 3)}</td>
        <td class="num"><span class="hi-green">{f(sc19['democracy']['auc'], 3)} · {f(sc19['democracy']['brier'], 3)}</span> vs {f(sc19['democracy']['persist_brier'], 3)}</td>
        <td class="num">—</td>
        <td class="num">2025</td>
      </tr>
      <tr>
        <td><strong>Deuda pública / PIB</strong> (MAE pp)</td>
        <td class="num">31,6 vs 29,1 (RMSE <span class="hi-green">45,1</span>)</td>
        <td class="num"><span class="hi-red">{f(sc10['debt_gdp']['mae']*100, 1)}</span> vs {f(sc10['debt_gdp']['persist_mae']*100, 1)}</td>
        <td class="num"><span class="hi-red">{f(sc19['debt_gdp']['mae']*100, 1)}</span> vs {f(sc19['debt_gdp']['persist_mae']*100, 1)}</td>
        <td class="num">{f(sc10['debt_gdp']['cover_80']*100, 0)} / {f(sc19['debt_gdp']['cover_80']*100, 0)} %</td>
        <td class="num">2024</td>
      </tr>
      <tr>
        <td><strong>Rentas recursos / PIB</strong> (MAE pp)</td>
        <td class="num">3,5 vs 3,5 (RMSE <span class="hi-green">6,8</span>)</td>
        <td class="num"><span class="hi-red">{f(sc10['res_share']['mae']*100, 2)}</span> vs {f(sc10['res_share']['persist_mae']*100, 2)}</td>
        <td class="num"><span class="hi-red">{f(sc19['res_share']['mae']*100, 2)}</span> vs {f(sc19['res_share']['persist_mae']*100, 2)}</td>
        <td class="num">{f(sc10['res_share']['cover_80']*100, 0)} / {f(sc19['res_share']['cover_80']*100, 0)} %</td>
        <td class="num">2021</td>
      </tr>
      <tr>
        <td><strong>Temperatura GMST</strong> (MAE °C)</td>
        <td class="num"><span class="hi-green">0,106</span> vs 0,208</td>
        <td class="num"><span class="hi-green">{f(sc10['gmst']['mae'], 3)}</span> vs {f(sc10['gmst']['persist_mae'], 3)}</td>
        <td class="num"><span class="hi-green">{f(sc19['gmst']['mae'], 3)}</span> vs {f(sc19['gmst']['persist_mae'], 3)}</td>
        <td class="num"><span class="hi-green">{f(sc10['gmst']['cover_80']*100, 0)} / {f(sc19['gmst']['cover_80']*100, 0)} %</span></td>
        <td class="num">2025</td>
      </tr>
      <tr>
        <td><strong>CO₂ mundial</strong> (MAPE %)</td>
        <td class="num">—</td>
        <td class="num"><span class="hi-green">{f(sc10['co2']['mape_pct'], 1)} %</span> vs {f(sc10['co2']['persist_mape_pct'], 1)} %</td>
        <td class="num"><span class="hi-red">{f(sc19['co2']['mape_pct'], 1)} %</span> vs {f(sc19['co2']['persist_mape_pct'], 1)} %</td>
        <td class="num"><span class="hi-green">{f(sc10['co2']['cover_80']*100, 0)} / {f(sc19['co2']['cover_80']*100, 0)} %</span></td>
        <td class="num">2024</td>
      </tr>
      <tr>
        <td><strong>Cuota de élites E · R10</strong> (MAE pp)</td>
        <td class="num"><span class="hi-green">{f(SC['from_2000_to_2019']['elite_share']['mae']*100, 1)} pp</span> vs 2,7</td>
        <td class="num"><span class="hi-green">{f(sc10['elite_share']['mae']*100, 1)} pp</span> vs 1,8</td>
        <td class="num">—</td>
        <td class="num"><span class="hi-green">{f(SC['from_2000_to_2019']['elite_share']['cover_80']*100, 0)} / {f(sc10['elite_share']['cover_80']*100, 0)} %</span></td>
        <td class="num">2019</td>
      </tr>
    </tbody>
  </table>

  <div class="grid-2" style="margin-top:2.5mm;font-size:8.1pt;">
    <div>
      <p><strong>Por qué L1 V2 gana a 10–15 años en PIB, conflicto y democracia.</strong> En horizontes cortos (1–6 años), el ruido idiosincrático más los choques globales de M3 y el reanclaje pesan más que la tendencia; a 15 años (<strong>2010 → 2025</strong>), el CRPS de PIB cae un <strong>{f((1 - sc10['gdp_pc']['crps']/sc10['gdp_pc']['persist_mae_log'])*100, 0)} %</strong> frente a la persistencia ({f(sc10['gdp_pc']['crps'], 3)} vs {f(sc10['gdp_pc']['persist_mae_log'], 3)}), <strong>Democracia V2</strong> baja el Brier un 20 % (0,078 vs 0,098; AUC 0,957) y <strong>conflicto</strong> alcanza AUC 0,909.</p>
    </div>
    <div>
      <p><strong>Mejora directa en 2019–2025 y matices de rigor.</strong> En el examen reciente (2020–2025), <code>democracy.fit</code> alcanza AUC <strong>{f(sc19['democracy']['auc'], 3)}</strong> (Brier {f(sc19['democracy']['brier'], 3)} vs {f(sc19['democracy']['persist_brier'], 3)}) y M3 eleva la cobertura p10–p90 de PIB al <strong>{f(sc19['gdp_pc']['cover_80']*100, 1)} %</strong>. En conflicto, la brecha de tasa (17,2 % en UCDP v24.1 frente a 10,0 % en el panel histórico de 2019) sigue reflejando la revisión de fuente.</p>
    </div>
  </div>
</div>
"""


def page_06_macro() -> str:
    gdp25_o, gdp25_s = OW["gdp_pc_world"][75], S19["gdp_pc_world"]["p50"][6]
    gdp26_s, gdp41_s = S19["gdp_pc_world"]["p50"][7], S19["gdp_pc_world"]["p50"][22]
    gdp41_lo, gdp41_hi = S19["gdp_pc_world"]["p10"][22], S19["gdp_pc_world"]["p90"][22]
    g15_w = (gdp41_s / gdp26_s - 1.0) * 100.0
    cagr15 = ((gdp41_s / gdp26_s) ** (1.0 / 15.0) - 1.0) * 100.0
    cons26, cons41 = S19["cons_pc"]["p50"][7], S19["cons_pc"]["p50"][22]
    c_gdpw = S.ts_chart(
        "PIB per cápita mundial (ponderado por población)",
        "USD CONSTANTES 2017 · 180 PAÍSES · ANCLA 2019 · PROYECCIÓN 2026–2041",
        obs=(Y_OBS, OW["gdp_pc_world"]),
        test_past=(Y_00, S00["gdp_pc_world"]["p50"]),
        test_past2=(Y_10, S10["gdp_pc_world"]["p50"]),
        proj=(Y_19, S19["gdp_pc_world"]["p10"], S19["gdp_pc_world"]["p50"], S19["gdp_pc_world"]["p90"]),
        xmin=1970, xmax=2041, ymin=4000, ymax=27000,
        y_fmt=lambda v: f"{v/1000:.0f}k",
        end_fmt=lambda v: f"{v/1000:.1f}k".replace(".", ","),
        width=520, height=136,
        caption=f"En 2025 la mediana de L1 ({f(gdp25_s, 0)} USD) cubre el dato real ({f(gdp25_o, 0)}, error {sgn((gdp25_s/gdp25_o-1)*100, 1, '%')}). Entre 2026 y 2041 crece al {sgn(cagr15, 2, '%/año')} hasta {f(gdp41_s, 0)} USD [{f(gdp41_lo, 0)}–{f(gdp41_hi, 0)}].",
    )
    c_med = S.ts_chart(
        "PIB per cápita del país mediano",
        "USD CONSTANTES 2017 · SIN PONDERAR POR POBLACIÓN",
        obs=(Y_OBS, OW["gdp_pc_med"]),
        test_past=(Y_00, S00["gdp_pc_med"]["p50"]),
        test_past2=(Y_10, S10["gdp_pc_med"]["p50"]),
        proj=(Y_19, S19["gdp_pc_med"]["p10"], S19["gdp_pc_med"]["p50"], S19["gdp_pc_med"]["p90"]),
        xmin=1970, xmax=2041, ymin=3000, ymax=27000,
        y_fmt=lambda v: f"{v/1000:.0f}k",
        end_fmt=lambda v: f"{v/1000:.1f}k".replace(".", ","),
        width=252, height=120,
        caption=f"La sobreestimación de 2025 baja a {sgn(RW['gdp_pc_med']['rel_err_last_pct'], 1, '%')} gracias a M1+M3; alcanza {f(S19['gdp_pc_med']['p50'][22], 0)} USD en 2041.",
    )
    c_gini = S.ts_chart(
        "Desigualdad internacional (Gini)",
        "PONDERADO POR POBLACIÓN ENTRE 180 PAÍSES (0 A 1)",
        obs=(Y_OBS, OW["gini"]),
        test_past=(Y_00, S00["gini"]["p50"]),
        test_past2=(Y_10, S10["gini"]["p50"]),
        proj=(Y_19, S19["gini"]["p10"], S19["gini"]["p50"], S19["gini"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.40, ymax=0.66,
        y_fmt=lambda v: f(v, 2),
        end_fmt=lambda v: f(v, 3),
        width=252, height=120,
        caption=f"Sube a {f(S19['gini']['p50'][22], 3)} en 2041 (real 2025: 0,454; banda cubre 67 %): el daño climático M1 en el trópico y el lastre de conflicto en India amplían la brecha.",
    )
    c_drain = S.ts_chart(
        "Drenaje por intercambio desigual (drain)",
        "PORCENTAJE DEL PIB MEDIO · RED BILATERAL COW",
        hist=(Y_H, SH["drain"]["p50"]),
        proj=(Y_19, S19["drain"]["p10"], S19["drain"]["p50"], S19["drain"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.0, ymax=0.9, scale=100.0,
        y_fmt=lambda v: f(v, 1, "%"),
        end_fmt=lambda v: f(v, 2, "%"),
        width=252, height=116,
        caption="Variable interna del Sistema-Mundo: baja suavemente de 0,49 % a 0,44 % del PIB al reducirse la brecha relativa de precios en la semiperiferia.",
    )
    c_rent = S.ts_chart(
        "Flujo neto de rentas (rent_in) y centralidad",
        "RENTA NETA MEDIA (% PIB) · EXCEDENTE DE CENTRO VS PERIFERIA",
        hist=(Y_H, SH["rent_in"]["p50"]),
        proj=(Y_19, S19["rent_in"]["p10"], S19["rent_in"]["p50"], S19["rent_in"]["p90"]),
        xmin=1970, xmax=2041, ymin=-3.6, ymax=0.5, scale=100.0, zero_line=True,
        y_fmt=lambda v: f(v, 1, "%"),
        end_fmt=lambda v: f(v, 2, "%"),
        width=252, height=116,
        caption="Negativo en promedio simple porque ~144 países periféricos y semiperiféricos transfieren rentas (intercambio, patentes, intereses) hacia 36 del centro.",
    )
    return f"""
<div class="page">
  <span class="chap-mark">04 · INDICADOR POR INDICADOR</span>
  <div class="eyebrow">Capítulo 04 · Macroeconomía y Sistema-Mundo</div>
  <h1>Indicador por indicador,<br>de 1970 a 2041</h1>
  {legend_bar(include_tests=True, include_hist=True)}
  <div class="grid-2" style="margin-bottom:2.5mm;font-size:8.1pt;">
    <p>Bajo el nuevo modo por defecto (<strong>M1–M3 + Democracia V2</strong>), el crecimiento agregado de 2026–2041 se vuelve mucho más sobrio ({sgn(g15_w, 1, "%")} en quince años, con consumo per cápita pasando de <strong>{f(cons26, 0)}</strong> a <strong>{f(cons41, 0)} USD</strong>) y una banda p10–p90 que por fin contempla contracción en la cola baja (p10: {f(gdp41_lo, 0)} USD).</p>
    <p>Al mismo tiempo, los daños climáticos persistentes (M1) sobre el trópico y la trampa de conflicto binario en India amplían la divergencia internacional: el Gini sube de 0,469 a <strong>{f(S19['gini']['p50'][22], 3)}</strong> mientras las transferencias centro-periferia permanecen estables.</p>
  </div>
  {c_gdpw}
  <div class="grid-2-tight" style="margin-top:2mm;">
    <div class="chart-rule">{c_med}</div>
    <div class="chart-rule">{c_gini}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:2mm;">
    <div class="chart-rule">{c_drain}</div>
    <div class="chart-rule">{c_rent}</div>
  </div>
</div>
"""


def page_07_sociopolitical() -> str:
    obs_E = OW.get("elite_mean_full", OW["elite_mean"])
    r_w = REG10.get("WEST", {}).get("snap", {})
    r_a = REG10.get("ASIA_EM", {}).get("snap", {})
    r_s = REG10.get("SOUTH", {}).get("snap", {})
    r_ref = REG10.get("REF", {}).get("snap", {})
    r_nam = REG10.get("NAM", {}).get("snap", {})
    c_E = S.ts_chart(
        "Captura de élites · Modelo Regional R10 (E)",
        "CUOTA TOP-10 % PRE-IMPUESTOS · COHORTE WID BALANCEADA (%)",
        obs=(Y_OBS, obs_E),
        hist=(Y_H, SH["E"]["p50"]),
        test_past=(Y_00, S00["E"]["p50"]),
        proj=(Y_19, S19["E"]["p10"], S19["E"]["p50"], S19["E"]["p90"]),
        xmin=1970, xmax=2041, ymin=38, ymax=52, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=104,
        caption=f"Con elites.py R10 se estabiliza en {f(S19['E']['p50'][22]*100, 1, '%')} (cob. 80 %: {f(SC['from_2000_to_2019']['elite_share']['cover_80']*100, 0)} %): sube en Occidente y Asia, cede en Sur Global.",
    )
    c_ne = S.ts_chart(
        "Sobreproducción de élites (ne)",
        "ÍNDICE RELATIVO AL EQUILIBRIO DE ENTRADA (1,0 = BASE)",
        hist=(Y_H, SH["ne"]["p50"]),
        proj=(Y_19, S19["ne"]["p10"], S19["ne"]["p50"], S19["ne"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.80, ymax=1.18, zero_line=False,
        y_fmt=lambda v: f(v, 2),
        end_fmt=lambda v: f(v, 2),
        width=252, height=104,
        caption=f"Sube de {f(S19['ne']['p50'][7], 2)} (2026) a {f(S19['ne']['p50'][22], 2)} (2041); alcanza {f(r_nam.get('2041', {}).get('ne', [0, 1.34])[1], 2)} en Norteamérica y {f(r_ref.get('2041', {}).get('ne', [0, 2.18])[1], 2)} en ex-URSS.",
    )
    c_psi = S.ts_chart(
        "Índice de estrés político (PSI)",
        "MEDIANA ENTRE PAÍSES · MMP × EMP × SFD",
        hist=(Y_H, SH["psi"]["p50"]),
        proj=(Y_19, S19["psi"]["p10"], S19["psi"]["p50"], S19["psi"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.5, ymax=1.35,
        y_fmt=lambda v: f(v, 2),
        end_fmt=lambda v: f(v, 2),
        width=252, height=104,
        caption=f"La mediana global cede ({f(S19['psi']['p50'][7], 2)} → {f(S19['psi']['p50'][22], 2)}), pero oculta rotación: sube en Occidente ({f(r_w.get('2019', {}).get('psi', [0, 0.70])[1], 2)} → {f(r_w.get('2041', {}).get('psi', [0, 0.85])[1], 2)}; NAM {f(r_nam.get('2041', {}).get('psi', [0, 1.63])[1], 2)}).",
    )
    c_asab = S.ts_chart(
        "Solidaridad colectiva · Asabiya (asab)",
        "ÍNDICE 0–1 · FRONTERA METAÉTNICA DE TURCHIN",
        hist=(Y_H, SH["asab"]["p50"]),
        proj=(Y_19, S19["asab"]["p10"], S19["asab"]["p50"], S19["asab"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.40, ymax=0.52,
        y_fmt=lambda v: f(v, 2),
        end_fmt=lambda v: f(v, 3),
        width=252, height=104,
        caption=f"Erosión persistente ({f(S19['asab']['p50'][0], 3)} → {f(S19['asab']['p50'][22], 3)}), más acusada en NAM ({f(r_nam.get('2019', {}).get('asab', [0, 0.45])[1], 3)} → {f(r_nam.get('2041', {}).get('asab', [0, 0.39])[1], 3)}).",
    )
    c_demo = S.ts_chart(
        "Proporción de democracias · Democracia V2 (demo)",
        "FRACCIÓN DE PAÍSES CON V-DEM ROW ≥ 2 (%) · DEMOCRACY.FIT",
        obs=(Y_OBS, OW["demo_share"]),
        test_past=(Y_00, S00["demo"]["p50"]),
        test_past2=(Y_10, S10["demo"]["p50"]),
        proj=(Y_19, S19["demo"]["p10"], S19["demo"]["p50"], S19["demo"]["p90"]),
        xmin=1970, xmax=2041, ymin=20, ymax=72, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=104,
        caption=f"Con democracy.fit V2 clava 2025 ({f(S19['demo']['p50'][6]*100, 1, '%')} vs 50,6 % real; AUC 0,980) y queda en {f(S19['demo']['p50'][22]*100, 1, '%')} en 2041 (poliarquía {f(S19['poly']['p50'][22], 3)}).",
    )
    c_haz = S.ts_chart(
        "Riesgo de conflicto civil (hazard)",
        "PROPORCIÓN DE PAÍSES CON CONFLICTO INTRAESTATAL ≥ 25 MUERTES (%)",
        obs=(Y_OBS, OW["conf_rate"]),
        test_past=(Y_00, S00["hazard"]["p50"]),
        test_past2=(Y_10, S10["hazard"]["p50"]),
        proj=(Y_19, S19["hazard"]["p10"], S19["hazard"]["p50"], S19["hazard"]["p90"]),
        xmin=1970, xmax=2041, ymin=4, ymax=22, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=104,
        caption=f"Sube de {f(S19['hazard']['p50'][7]*100, 1, '%')} (2026) a {f(S19['hazard']['p50'][22]*100, 1, '%')} (2041) al acoplar los frenos alimentarios (M2) y el clima severo (M1).",
    )

    def _rrow(code: str, label: str) -> str:
        sn = REG10.get(code, {}).get("snap", {})
        if not sn:
            return ""
        s19, s41 = sn["2019"], sn["2041"]
        dE = (s41["E"][1] - s19["E"][1]) * 100.0
        cls_e = "hi-red" if dE > 0.3 else ("hi-green" if dE < -0.3 else "")
        return (
            f"<tr><td><strong>{label}</strong></td>"
            f"<td class='num'>{f(s19['E'][1]*100, 1)} → <strong>{f(s41['E'][1]*100, 1)} %</strong> "
            f"(<span class='{cls_e}'>{sgn(dE, 1)}</span>)</td>"
            f"<td class='num'>{f(s19['psi'][1], 2)} → <strong>{f(s41['psi'][1], 2)}</strong></td>"
            f"<td class='num'>{f(s19['asab'][1], 3)} → {f(s41['asab'][1], 3)}</td>"
            f"<td class='num'>{f(s19['demo'][1]*100, 0)} → {f(s41['demo'][1]*100, 0)} %</td>"
            f"<td class='num'>{f(s19['hazard'][1]*100, 1)} → <strong>{f(s41['hazard'][1]*100, 1)} %</strong></td></tr>"
        )

    reg_table = f"""
  <table class="tight-table" style="margin-top:2mm;font-size:6.8pt;">
    <thead>
      <tr>
        <th>Bloque regional R10 (2019 → 2041)</th>
        <th class="num">Captura élites E</th>
        <th class="num">PSI mediano</th>
        <th class="num">Asabiya</th>
        <th class="num">Democracias</th>
        <th class="num">Conflicto</th>
      </tr>
    </thead>
    <tbody>
      {_rrow("WEST", "Occidente (NAM+EUR+OPA · 43)")}
      {_rrow("NAM", "  └ Norteamérica (USA+CAN · 3)")}
      {_rrow("ASIA_EM", "Asia Emergente (CHN+SAS+SEA · 23)")}
      {_rrow("SOUTH", "Sur Global (LAM+MEA+SSA · 102)")}
      {_rrow("REF", "Ex-URSS (REF · 12)")}
    </tbody>
  </table>"""

    return f"""
<div class="page">
  <span class="chap-mark">04 · INDICADOR POR INDICADOR</span>
  <div class="eyebrow">Capítulo 04 · Élites regionales R10, Democracia V2 y conflicto</div>
  <h1>Divergencia sociopolítica regional,<br>estancamiento democrático y más conflicto</h1>
  {legend_bar(include_tests=True, include_hist=True)}
  <div class="grid-2" style="margin-bottom:1.5mm;font-size:7.8pt;">
    <p><strong>Élites Regionales R10 (<code>elites.py</code>) corrige el sesgo global de <code>E</code>.</strong> Al sustituir el término espurio <code>−0,02·conf</code> por un panel jerárquico por 10 regiones con <code>r*−g</code> y choques regionales, el RMSE de 19 años cae un <strong>51 %</strong> (7,1 → 3,5 pp) y la cobertura sube del 22 % al <strong>80 %</strong>, capturando el alza occidental (+0,7 pp; NAM +1,6 pp) frente a la compresión en el Sur Global (−1,5 pp).</p>
    <p><strong>Democracia V2 y fragilidad bajo estrés M1–M2.</strong> Con <code>democracy.fit</code>, la proporción de democracias clava 2025 ({f(S19['demo']['p50'][6]*100, 1, '%')} vs 50,6 %) y se estanca en <strong>{f(S19['demo']['p50'][22]*100, 1, '%')}</strong> en 2041, mientras los frenos alimentarios (M2) y el clima (M1) elevan el conflicto al <strong>{f(S19['hazard']['p50'][22]*100, 1, '%')}</strong>.</p>
  </div>
  <div class="grid-2-tight">
    <div class="chart-rule">{c_E}</div>
    <div class="chart-rule">{c_ne}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:1.2mm;">
    <div class="chart-rule">{c_psi}</div>
    <div class="chart-rule">{c_asab}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:1.2mm;">
    <div class="chart-rule">{c_demo}</div>
    <div class="chart-rule">{c_haz}</div>
  </div>
  {reg_table}
</div>
"""


def page_08_finance_resources_climate() -> str:
    c_debt = S.ts_chart(
        "Deuda pública / PIB (país mediano)",
        "PORCENTAJE DEL PIB · MEDIANA DE PAÍSES · REAL HASTA 2024",
        obs=(Y_OBS, OW["debt_med"]),
        test_past=(Y_00, S00["debt"]["p50"]),
        test_past2=(Y_10, S10["debt"]["p50"]),
        proj=(Y_19, S19["debt"]["p10"], S19["debt"]["p50"], S19["debt"]["p90"]),
        xmin=1970, xmax=2041, ymin=20, ymax=72, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=122,
        caption="Choque ausente: la deuda mediana real saltó a 57,7 % en 2024 (y 94,7 % ponderada por PIB), mientras L1 revierte hacia 48–50 % sin ver el COVID ni los tipos.",
    )
    c_cris = S.ts_chart(
        "Crisis financieras anuales (crisis)",
        "PROPORCIÓN DE PAÍSES EN CRISIS BANCARIA, CAMBIARIA O SOBERANA (%)",
        hist=(Y_H, SH["crisis"]["p50"]),
        proj=(Y_19, S19["crisis"]["p10"], S19["crisis"]["p50"], S19["crisis"]["p90"]),
        xmin=1970, xmax=2041, ymin=0, ymax=17, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=122,
        caption="Ritmo estacionario de ~4,4 % anual (unos 8 países/año). Cada evento destruye permanentemente un 4,5 % de PTF y suma 8 pp de deuda.",
    )
    c_rr = S.ts_chart(
        "Rentas de recursos naturales / PIB",
        "MEDIA ENTRE PAÍSES (%) · REAL WDI HASTA 2021",
        obs=(Y_OBS, OW["rr_mean"]),
        test_past=(Y_00, S00["res_share"]["p50"]),
        test_past2=(Y_10, S10["res_share"]["p50"]),
        proj=(Y_19, S19["res_share"]["p10"], S19["res_share"]["p50"], S19["res_share"]["p90"]),
        xmin=1970, xmax=2041, ymin=0, ymax=11, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 1, "%"),
        width=252, height=122,
        caption="Nominalmente estable (~6,0 %) gracias a que el precio mundial sube en proporción exacta a la caída física de extracción.",
    )
    c_res = S.ts_chart(
        "Reservas restantes (reserves) y agotamiento",
        "FRACCIÓN DEL RECURSO ÚLTIMO AÚN BAJO TIERRA · MEDIANA (%)",
        hist=(Y_H, SH["reserves"]["p50"]),
        proj=(Y_19, S19["reserves"]["p10"], S19["reserves"]["p50"], S19["reserves"]["p90"]),
        xmin=1970, xmax=2041, ymin=0, ymax=100, scale=100.0,
        y_fmt=lambda v: f"{v:.0f}%",
        end_fmt=lambda v: f(v, 2, "%"),
        width=252, height=122,
        caption="Artefacto clave: con r_R = 0,145 la mediana ya cae al 10,3 % en 2019, al 3,7 % en 2026 y al 0,36 % en 2041. La extracción física prácticamente cesa.",
    )
    c_co2 = S.ts_chart(
        "Emisiones mundiales de CO₂ fósil (M0)",
        "GIGATONELADAS DE CO₂ POR AÑO · REAL GCB HASTA 2024",
        obs=(Y_OBS, OW["co2_gt"]),
        test_past=(Y_00, S00["co2"]["p50"]),
        test_past2=(Y_10, S10["co2"]["p50"]),
        proj=(Y_19, S19["co2"]["p10"], S19["co2"]["p50"], S19["co2"]["p90"]),
        xmin=1970, xmax=2041, ymin=0, ymax=48,
        y_fmt=lambda v: f"{v:.0f}",
        end_fmt=lambda v: f(v, 1, " Gt"),
        width=252, height=124,
        caption=f"M0 corrige el desplome Hubbert: 2024 sale en {f(S19['co2']['p50'][5], 1)} Gt (real: 38,6 Gt, +1,5 %) y desciende suavemente a {f(S19['co2']['p50'][22], 1)} Gt en 2041 al desacelerarse el PIB.",
    )
    c_tg = S.ts_chart(
        "Temperatura global · GMST (M0)",
        "°C SOBRE 1850–1900 (HADCRUT5) · REAL HASTA 2025",
        obs=(Y_OBS, OW["gmst"]),
        test_past=(Y_00, S00["Tg"]["p50"]),
        test_past2=(Y_10, S10["Tg"]["p50"]),
        proj=(Y_19, S19["Tg"]["p10"], S19["Tg"]["p50"], S19["Tg"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.2, ymax=2.2,
        y_fmt=lambda v: f(v, 1, "°"),
        end_fmt=lambda v: f(v, 2, "°C"),
        width=252, height=124,
        caption=f"Clava 2025 ({f(S19['Tg']['p50'][6], 2)} vs 1,41 °C real; 80 % cobertura a 15 años), cruza 1,5 °C hacia 2029 y llega a {f(S19['Tg']['p50'][22], 2)} °C [{f(S19['Tg']['p10'][22], 2)}–{f(S19['Tg']['p90'][22], 2)}] en 2041.",
    )
    return f"""
<div class="page">
  <span class="chap-mark">04 · INDICADOR POR INDICADOR</span>
  <div class="eyebrow">Capítulo 04 · Finanzas, recursos naturales y clima (M0+M1)</div>
  <h1>Deuda sin choques, Hubbert contenido<br>y un clima que ya cruza +1,80 °C</h1>
  {legend_bar(include_tests=True, include_hist=True)}
  <div class="grid-2" style="margin-bottom:2mm;font-size:8.0pt;">
    <p>En finanzas públicas, L1 mantiene la reversión fiscal de Bohn y congela la tasa real de EE. UU. (+0,25 %); por eso no captura el salto pandémico de deuda (+11 pp en 2020–24), aunque el frenazo económico de M0–M3 eleva la deuda mediana a {f(S19['debt']['p50'][22]*100, 1)} % en 2041.</p>
    <p><strong>M0 desacopla CO₂ y temperatura del colapso Hubbert:</strong> aunque las reservas logísticas siguen agotándose en el submodelo extractivo, las emisiones siguen el PIB real con intensidad de carbono observada (39,2 Gt en 2024 vs 38,6 reales) y llevan <code>Tg</code> a <strong>{f(S19['Tg']['p50'][22], 2)} °C</strong> en 2041, activando el daño global <code>d_glob</code>.</p>
  </div>
  <div class="grid-2-tight">
    <div class="chart-rule">{c_debt}</div>
    <div class="chart-rule">{c_cris}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:1.5mm;">
    <div class="chart-rule">{c_rr}</div>
    <div class="chart-rule">{c_res}</div>
  </div>
  <div class="grid-2-tight" style="margin-top:1.5mm;">
    <div class="chart-rule">{c_co2}</div>
    <div class="chart-rule">{c_tg}</div>
  </div>
</div>
"""


def page_09_channels_and_hubbert() -> str:
    ch_meta = [
        ("k", "k · Capital doméstico", "% DEL PIB · INVERSIÓN INTERNA", 18, 30),
        ("m", "m · Bienes del centro", "% DEL PIB · IMPORTACIONES DE ALTO VALOR", 4, 12),
        ("r", "r · Tecnología e industria", "% DEL PIB · I+D + EDUCACIÓN ESCALADA", 0.8, 1.8),
        ("x", "x · Extracción propia", "% DEL PIB · CAPITAL EXTRACTIVO DOMÉSTICO", 0.5, 3.0),
        ("f", "f · Inversión exterior", "% DEL PIB · CONCESIONES EN EL EXTRANJERO", 0.0, 3.5),
        ("w", "w · Redistribución social", "% DEL PIB · TRANSFERENCIAS ESCALADAS", 0.15, 0.45),
    ]
    panels = []
    for k, title, sub, ylo, yhi in ch_meta:
        panels.append(
            S.ts_chart(
                title, sub,
                hist=(Y_H, SH[k]["p50"]),
                proj=(Y_19, S19[k]["p10"], S19[k]["p50"], S19[k]["p90"]),
                xmin=1970, xmax=2041, ymin=ylo, ymax=yhi, scale=100.0,
                y_fmt=lambda v: f(v, 1 if (yhi - ylo) < 5 else 0, "%"),
                end_fmt=lambda v: f(v, 2 if (yhi - ylo) < 2 else 1, "%"),
                width=165, height=112,
                xticks=[1970, 2000, 2019, 2041],
            )
        )

    c_price = S.ts_chart(
        "Precio mundial de recursos (price)",
        "ÍNDICE 1970 = 1 · REAL JACKS VS MERCADO L1",
        obs=(Y_OBS, OW["price_idx"]),
        hist=(Y_H, SH["price"]["p50"]),
        proj=(Y_19, S19["price"]["p10"], S19["price"]["p50"], S19["price"]["p90"]),
        xmin=1970, xmax=2041, ymin=0, ymax=95,
        y_fmt=lambda v: f"{v:.0f}×",
        end_fmt=lambda v: f(v, 1, "×"),
        width=252, height=122,
        caption="En 2025 el índice real Jacks es 1,5× (1970=1); en L1 ya va por 6,5× y escala a 59,8× en 2041 para equilibrar una oferta que se extingue.",
    )
    c_nat = S.ts_chart(
        "Capital natural (natcap) y daño climático",
        "ÍNDICE DE CAPITAL NATURAL (0–1) · MEDIANA",
        hist=(Y_H, SH["natcap"]["p50"]),
        proj=(Y_19, S19["natcap"]["p10"], S19["natcap"]["p50"], S19["natcap"]["p90"]),
        xmin=1970, xmax=2041, ymin=0.70, ymax=1.02,
        y_fmt=lambda v: f(v, 2),
        end_fmt=lambda v: f(v, 3),
        width=252, height=122,
        caption="Aunque el CO₂ colapse, el sobregiro ecológico congelado en 2019 (1,78 planetas) sigue degradando el capital natural de 0,863 a 0,774 (−10,3 %).",
    )
    return f"""
<div class="page">
  <span class="chap-mark">05 · CANALES Y DIAGNÓSTICO HUBBERT</span>
  <div class="eyebrow">Capítulo 05 · Conducta estatal y diagnóstico biofísico</div>
  <h1>Seis canales de conducta estables<br>y el espejo exacto de World3</h1>
  <p class="lede">Tras 2019, los 180 Estados dejan de leer presupuestos históricos y deciden su gasto con la regla clonada de la historia (<code>BehaviorModel</code>). Sus presupuestos apenas cambian en 15 años; lo que se rompe es la física de recursos bajo ellos.</p>

  <div class="grid-3">
    <div class="chart-rule">{panels[0]}</div>
    <div class="chart-rule">{panels[1]}</div>
    <div class="chart-rule">{panels[2]}</div>
    <div class="chart-rule">{panels[3]}</div>
    <div class="chart-rule">{panels[4]}</div>
    <div class="chart-rule">{panels[5]}</div>
  </div>

  <div class="slate-box" style="margin-top:2.5mm;">
    <h3>Diagnóstico mecánico: Hubbert aislado en precios (×48) mientras M0 salva CO₂ y clima</h3>
    <div class="grid-2">
      <p><strong>1. El sesgo de depósito fijo en el submodelo extractivo.</strong> Con <code>r<sub>R</sub> = 0,145</code> sin crecimiento de reservas, la mediana de reservas cae al <strong>10,3 % en 2019</strong> y al <strong>0,36 % en 2041</strong>, y el precio de recursos sube ×48 para sostener <code>ψ<sub>t</sub></code> congelada.</p>
      <p><strong>2. Por qué M0 rompió el contagio hacia el clima.</strong> En el L1 legado, ese vaciado desplomaba el CO₂ a 2,7 Gt y congelaba <code>Tg</code> en 1,44 °C. Con <strong>M0 activo por defecto</strong>, CO₂ y <code>Tg</code> se rigen por la intensidad de carbono sobre el PIB y la TCRE efectiva OLS, devolviendo el clima a su senda física real (33,6 Gt y 1,80 °C en 2041).</p>
    </div>
  </div>

  <div class="grid-2-tight" style="margin-top:2mm;">
    <div class="chart-rule">{c_price}</div>
    <div class="chart-rule">{c_nat}</div>
  </div>
</div>
"""


def page_10_zones_and_countries() -> str:
    z = D["zones"]
    zrows = ""
    for zkey, zlbl in (("centro", "Centro (36 países)"), ("semiperiferia", "Semiperiferia (54)"), ("periferia", "Periferia (90)")):
        r19, r26, r41 = z["2019"][zkey], z["2026"][zkey], z["2041"][zkey]
        g15 = (r41["gdp_pc_world"][1] / r26["gdp_pc_world"][1] - 1) * 100
        zrows += (
            f"<tr><td><strong>{zlbl}</strong></td>"
            f"<td class='num'>{f(r19['gdp_pc_world'][1], 0)} → {f(r41['gdp_pc_world'][1], 0)} <span class='muted'>({sgn(g15, 0, '%')})</span></td>"
            f"<td class='num'>{f(r19['E'][1]*100, 1)} → {f(r41['E'][1]*100, 1)}</td>"
            f"<td class='num'>{f(r19['psi'][1], 2)} → {f(r41['psi'][1], 2)}</td>"
            f"<td class='num'>{f(r19['debt'][1]*100, 0)} → {f(r41['debt'][1]*100, 0)}</td>"
            f"<td class='num'>{f(r19['demo'][1]*100, 0)} → {f(r41['demo'][1]*100, 0)}</td>"
            f"<td class='num'>{f(r19['hazard'][1]*100, 1)} → {f(r41['hazard'][1]*100, 1)}</td>"
            f"<td class='num'>{f(r19['dclim'][1]*100, 1)} → {f(r41['dclim'][1]*100, 1)}</td></tr>"
        )

    short_names = {
        "USA": "EE. UU.", "CHN": "China", "DEU": "Alemania", "KOR": "Corea Sur",
        "RUS": "Rusia", "BRA": "Brasil", "MEX": "México", "IND": "India",
        "ZAF": "Sudáfrica", "ARG": "Argentina", "EGY": "Egipto", "NGA": "Nigeria",
    }
    cards = []
    for code in ("USA", "CHN", "DEU", "KOR", "RUS", "BRA", "MEX", "IND", "ZAF", "ARG", "EGY", "NGA"):
        c = D["countries"][code]
        g25o = c["obs_recent"]["gdp_pc_2025"]
        g25s = c["sim"]["2025"]["gdp_pc"][1]
        err25 = sgn((g25s / g25o - 1) * 100, 1, "%") if g25o else "—"
        v41 = f"{c['sim']['2041']['gdp_pc'][1]/1000:.1f}k".replace(".", ",")
        svg = S.country_mini_svg(
            code, short_names[code], c["traj_gdp_obs_2000_2025"], c["traj_gdp_sim_2019_2041"],
            err25_txt=err25, val41_txt=f"2041: {v41}",
        )
        cards.append(f'<div class="country-card">{svg}</div>')

    return f"""
<div class="page">
  <span class="chap-mark">06 · ZONAS Y PAÍSES</span>
  <div class="eyebrow">Capítulo 06 · Geografía del Sistema-Mundo</div>
  <h1>Tres velocidades y doce pruebas<br>país por país hasta 2041</h1>
  <p class="lede">Desagregar el mundo en Centro, Semiperiferia y Periferia —y contrastar doce economías mayores contra su PIB real de 2025— revela tanto la precisión macro de L1 como tres fallos concretos de microestructura.</p>

  <table class="tight-table">
    <thead>
      <tr>
        <th>Zona (fija en 2019)</th>
        <th class="num">PIB pc 19→41 (Δ15a)</th>
        <th class="num">Élites (%)</th>
        <th class="num">PSI</th>
        <th class="num">Deuda (%)</th>
        <th class="num">Democ. (%)</th>
        <th class="num">Conflicto (%)</th>
        <th class="num">Daño (%)</th>
      </tr>
    </thead>
    <tbody>{zrows}</tbody>
  </table>

  <h2 style="margin-top:2.5mm;margin-bottom:1.5mm;">Doce economías: dato real 2000–2025 (negro) frente a proyección Atlas L1 2019–2041 (naranja)</h2>
  <div class="grid-4">
    {"".join(cards)}
  </div>

  <div class="grid-3" style="margin-top:3mm;">
    <div class="slate-box" style="margin:0;padding:3mm 3.5mm;">
      <h3 style="font-size:8pt;">Aciertos y mejoras en 2025</h3>
      <p style="font-size:7.4pt;"><strong>EE. UU.</strong> (−1,9 %), <strong>Nigeria</strong> (−3,8 %) y <strong>Corea del Sur</strong> (+3,8 %) clavan 2025 dentro de ±4 %. Con M3, los errores sobre <strong>Alemania</strong> (+9,6 %) y <strong>México</strong> (+6,8 %) caen casi a la mitad respecto al L1 legado (+12,5 % y +15,5 %).</p>
    </div>
    <div class="slate-box" style="margin:0;padding:3mm 3.5mm;">
      <h3 style="font-size:8pt;">Anomalía 1 · La trampa binaria de India</h3>
      <p style="font-size:7.4pt;">India creció a 8.741 USD en 2025, pero L1 la reduce a 6.148 (−29,7 %) y 4.587 en 2041: la bandera binaria <code>conflict=1</code> de 2019 más el daño climático persistente M1 (−13,9 %) penalizan a toda la nación como si sufriera guerra civil total.</p>
    </div>
    <div class="slate-box" style="margin:0;padding:3mm 3.5mm;">
      <h3 style="font-size:8pt;">Anomalía 2 · Rusia (Tbar) y China V2</h3>
      <p style="font-size:7.4pt;"><strong>Rusia</strong> amplifica su daño climático fantasma a −37,7 % bajo M1 (<code>ρ<sub>D</sub>=0,94</code>) al heredar <code>Tbar = 15 °C</code> por falta de datos 1950–80. En cambio, <strong>Democracia V2</strong> corrige a <strong>China</strong>: su probabilidad de democratizarse en 2041 cae del 59 % legado al <strong>39 %</strong> (poliarquía 0,443).</p>
    </div>
  </div>
</div>
"""


def page_11_ablations() -> str:
    ab = D["ablations_2041"]
    names = {
        "l1_legado_9e96f6b": "L1 legado 9e96f6b (sin V2/M0–M3)",
        "sin_clima_ecologia": "Sin clima y ecología (total)",
        "sin_clima_severo_M1": "Sin clima persistente (M1)",
        "sin_colas_pesadas_M3": "Sin colas pesadas / choques (M3)",
        "sin_deuda_crisis": "Sin deuda y crisis financieras",
        "sin_clima_co2_M0": "Sin CO₂/TCRE real (M0 -> Hubbert)",
        "sin_frontera_metaetnica": "Sin frontera (asabiya)",
        "sin_frenos_alimentos_M2": "Sin frenos alimentarios (M2)",
        "sin_democracia_v2": "Sin Democracia V2 (logit viejo)",
        "sin_instituciones": "Sin instituciones (congeladas)",
        "sin_politica_de_recursos": "Sin política de recursos",
        "sin_demografia_estructural": "Sin demografía estructural",
        "sin_sistema_mundo": "Sin sistema-mundo",
    }
    order = [
        "l1_legado_9e96f6b", "sin_clima_ecologia", "sin_clima_severo_M1", "sin_colas_pesadas_M3",
        "sin_deuda_crisis", "sin_clima_co2_M0", "sin_frontera_metaetnica", "sin_frenos_alimentos_M2",
        "sin_democracia_v2", "sin_instituciones", "sin_politica_de_recursos", "sin_demografia_estructural",
        "sin_sistema_mundo",
    ]
    chart_order = [
        "l1_legado_9e96f6b", "sin_clima_ecologia", "sin_clima_severo_M1", "sin_colas_pesadas_M3",
        "sin_deuda_crisis", "sin_clima_co2_M0", "sin_frenos_alimentos_M2",
    ]

    rows_gdp = [(names[k], ab[k]["gdp_pc_world"]["delta_pct"], sgn(ab[k]["gdp_pc_world"]["delta_pct"], 1, "%")) for k in chart_order]
    rows_haz = [(names[k], ab[k]["hazard"]["delta_pct"], sgn(ab[k]["hazard"]["delta_pct"], 1, "%")) for k in chart_order]

    ch_gdp = S.diverging_bar_chart("Efecto en PIB per cápita 2041", "CAMBIO (%) AL APAGAR CADA MÓDULO (CRN)", rows_gdp)
    ch_haz = S.diverging_bar_chart("Efecto en conflicto civil 2041", "CAMBIO (%) AL APAGAR CADA MÓDULO (CRN)", rows_haz)

    trows = ""
    for k in order:
        v = ab[k]
        trows += (
            f"<tr><td style='padding:1.2px 4px;'><strong>{names[k]}</strong></td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['gdp_pc_world']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['gini']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['E']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['psi']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['asab']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['debt']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['demo']['delta_pct'], 1, '%')}</td>"
            f"<td class='num' style='padding:1.2px 4px;'>{sgn(v['hazard']['delta_pct'], 1, '%')}</td></tr>"
        )

    return f"""
<div class="page">
  <span class="chap-mark">07 · LOS MECANISMOS</span>
  <div class="eyebrow">Capítulo 07 · Ablaciones de las siete teorías y los módulos V2 / M1–M3 (2019 → 2041)</div>
  <h1 style="margin-bottom:1.5mm;">Qué mueve realmente los<br>próximos quince años</h1>
  <p class="lede" style="font-size:9.1pt;margin-bottom:1.5mm;">Mismo ensemble (48 mundos, CRN independientes por módulo) apagando cada teoría y cada nueva pieza desde 2019. El signo indica cuánto cambia 2041 <em>sin</em> ese mecanismo.</p>

  <div class="grid-2-tight">
    <div class="chart-rule">{ch_gdp}</div>
    <div class="chart-rule">{ch_haz}</div>
  </div>

  <h2 style="margin-top:1.5mm;margin-bottom:1mm;">Matriz de sensibilidad estructural en 2041 (% frente al L1 por defecto)</h2>
  <table class="tight-table" style="font-size:6.6pt;line-height:1.15;">
    <thead>
      <tr>
        <th style="padding:1.5px 4px;">Módulo / teoría apagada (2019–2041)</th>
        <th class="num" style="padding:1.5px 4px;">PIB pc</th>
        <th class="num" style="padding:1.5px 4px;">Gini</th>
        <th class="num" style="padding:1.5px 4px;">Élites E</th>
        <th class="num" style="padding:1.5px 4px;">PSI</th>
        <th class="num" style="padding:1.5px 4px;">Asabiya</th>
        <th class="num" style="padding:1.5px 4px;">Deuda</th>
        <th class="num" style="padding:1.5px 4px;">Democ.</th>
        <th class="num" style="padding:1.5px 4px;">Conflicto</th>
      </tr>
    </thead>
    <tbody>{trows}</tbody>
  </table>

  <div class="grid-2" style="margin-top:1.5mm;">
    <div class="card-col" style="padding:2mm 2.8mm;">
      <h3 style="font-size:7.6pt;margin-bottom:1mm;"><span class="swatch" style="background:#D95D24;"></span>Motores dominantes en L1 por defecto</h3>
      <ol style="font-size:7.0pt;line-height:1.25;margin:0;padding-left:3.5mm;">
        <li><strong>Clima total M0+M1 (+11,1 % PIB · −20,1 % conflicto):</strong> al activar el calentamiento real (1,80 °C), M1 aporta +8,9 % y M0 +3,8 % adicionales.</li>
        <li><strong>Colas pesadas M3 (+6,7 % PIB):</strong> choques Student-<em>t</em> y crisis globales sincronizadas se acumulan en <code>ln A</code> y amplían bandas.</li>
        <li><strong>Frenos alimentarios M2 (−16,4 % conflicto):</strong> con 1,80 °C, la presión alimentaria pasa a explicar un sexto del riesgo bélico de 2041.</li>
        <li><strong>Crisis financieras (+3,9 % PIB)</strong> y <strong>asabiya (−10,7 % conflicto)</strong> mantienen su peso estructural.</li>
      </ol>
    </div>
    <div class="card-col" style="padding:2mm 2.8mm;">
      <h3 style="font-size:7.6pt;margin-bottom:1mm;"><span class="swatch" style="background:#2F66D0;"></span>Brecha frente al L1 legado (9e96f6b)</h3>
      <ol style="font-size:7.0pt;line-height:1.25;margin:0;padding-left:3.5mm;">
        <li><strong>Optimismo legado corregido:</strong> <code>legacy=True</code> infla el PIB de 2041 un <strong>+17,8 %</strong>, las democracias un <strong>+16,1 %</strong> y subestima el conflicto (−18,6 %), el daño climático (−68,2 %) y la deuda (−9,4 %).</li>
        <li><strong>Democracia V2 (−10,9 % sesgo):</strong> apagar <code>democracy.fit</code> reactiva el rebote de la Tercera Ola y eleva el conflicto (+4,7 %).</li>
        <li><strong>Sinergia M0+M1:</strong> con el clima real de M0, el impacto de M1 sube de +5,8 % a <strong>+8,9 %</strong> al activarse <code>d_glob</code> (&gt; 1,5 °C).</li>
      </ol>
    </div>
  </div>
</div>
"""


def _param_rows(specs: list[tuple]) -> str:
    html = ""
    for item in specs:
        if len(item) == 1:
            html += f'<tr class="group-row"><td colspan="7">{escape(item[0])}</td></tr>'
            continue
        key, label, unit, scale, dec, obs_key, obs_yr = item
        s = S19[key]
        v19 = s["p50"][0] * scale
        v26 = s["p50"][7] * scale
        v31 = s["p50"][12] * scale
        v36 = s["p50"][17] * scale
        p10, p50, p90 = s["p10"][22] * scale, s["p50"][22] * scale, s["p90"][22] * scale
        if obs_key and obs_yr:
            ov = OW[obs_key][obs_yr - 1950]
            obs_str = f"{f(ov * scale, dec)} <span class='muted'>({obs_yr})</span>" if ov is not None else "—"
        else:
            obs_str = "<span class='muted'>interna</span>"
        if abs(v26) > 1e-9:
            chg_pct = (p50 / v26 - 1.0) * 100.0
            chg_str = sgn(chg_pct, 1, "%")
        else:
            chg_str = sgn(p50 - v26, dec)
        html += (
            f"<tr><td><strong>{escape(label)}</strong> <span class='muted'>{escape(unit)}</span></td>"
            f"<td class='num'>{f(v19, dec)}</td>"
            f"<td class='num'>{obs_str}</td>"
            f"<td class='num'>{f(v26, dec)}</td>"
            f"<td class='num'>{f(v31, dec)}</td>"
            f"<td class='num'>{f(p10, dec)} · <strong class='hi-orange'>{f(p50, dec)}</strong> · {f(p90, dec)}</td>"
            f"<td class='num'>{chg_str}</td></tr>"
        )
    return html


def page_12_all_params_1() -> str:
    specs_1 = [
        ("1. Macroeconomía, ingreso y financiamiento",),
        ("gdp_pc_world", "PIB pc mundial (pob.)", "USD 2017", 1.0, 0, "gdp_pc_world", 2025),
        ("gdp_pc_med", "PIB pc país mediano", "USD 2017", 1.0, 0, "gdp_pc_med", 2025),
        ("ly", "Log PIB pc mediano (ly)", "ln USD", 1.0, 2, None, None),
        ("gni_pc", "INB pc mundial (gni_pc)", "USD 2017", 1.0, 0, None, None),
        ("cons_pc", "Consumo pc mundial (cons_pc)", "USD 2017", 1.0, 0, None, None),
        ("gini", "Gini internacional (pob.)", "0–1", 1.0, 3, "gini", 2025),
        ("jump", "Saltos de PTF (jump)", "% anual", 100.0, 2, None, None),
        ("borrow", "Deuda externa extra (borrow)", "% PIB", 100.0, 2, None, None),
        ("squeeze", "Ahorro forzado (squeeze)", "% PIB", 100.0, 2, None, None),
        ("adjcost", "Coste ajuste capital (adjcost)", "% inv.", 100.0, 1, None, None),
        ("2. Sistema-Mundo (Wallerstein / Amin)",),
        ("core", "Centralidad media (core)", "0–1", 1.0, 3, None, None),
        ("drain", "Drenaje intercam. desigual (drain)", "% PIB", 100.0, 2, None, None),
        ("rent_in", "Flujo neto rentas (rent_in)", "% PIB", 100.0, 2, None, None),
        ("eco_ue", "Intercam. ecol. desigual (eco_ue)", "% PIB", 100.0, 1, None, None),
        ("3. Demografía estructural y frontera metaétnica (Turchin)",),
        ("E", "Captura de élites · R10 (E)", "% ingreso", 100.0, 1, "elite_mean_full", 2019),
        ("ne", "Sobreproducción élites (ne)", "índice", 1.0, 2, None, None),
        ("psi", "Estrés político mediano (psi)", "índice", 1.0, 2, None, None),
        ("asab", "Solidaridad · Asabiya (asab)", "0–1", 1.0, 3, None, None),
        ("4. Deuda pública y crisis financieras",),
        ("debt", "Deuda pública mediana (debt)", "% PIB", 100.0, 1, "debt_med", 2024),
        ("debt_gdpw", "Deuda pública mundial (PIB)", "% PIB", 100.0, 1, "debt_w_gdpw", 2024),
        ("crisis", "Incidencia anual crisis (crisis)", "% países", 100.0, 2, None, None),
    ]
    return f"""
<div class="page">
  <span class="chap-mark">08 · TODOS LOS PARÁMETROS (I)</span>
  <div class="eyebrow">Capítulo 08 · Evolución completa 2026–2041 (parte I)</div>
  <h1>Los 38 parámetros de L1<br>en los próximos quince años</h1>
  <p class="lede">Auditoría exhaustiva de todas las variables de estado, flujos y canales de la capa L1 desde su ancla de 2019 hasta 2041 (mediana y banda p10–p90 de 100 mundos). La columna final mide la variación relativa entre hoy (2026) y 2041.</p>

  <table class="tight-table">
    <thead>
      <tr>
        <th>Parámetro / variable de salida</th>
        <th class="num">2019<br>ancla</th>
        <th class="num">Último<br>real</th>
        <th class="num">2026<br>hoy</th>
        <th class="num">2031<br>+5 a</th>
        <th class="num">2041 (+15 años)<br>p10 · mediana · p90</th>
        <th class="num">Δ 26→41</th>
      </tr>
    </thead>
    <tbody>{_param_rows(specs_1)}</tbody>
  </table>

  <div class="grid-2" style="margin-top:2.5mm;font-size:7.9pt;">
    <p><strong>Lectura de las variables internas.</strong> <code>borrow</code> y <code>squeeze</code> permanecen cerca de 0 % del PIB bajo conducta clonada (no hay saltos bruscos de presupuesto respecto al promedio histórico), mientras el coste de ajuste del capital (<code>adjcost</code>) se estabiliza en torno al 15–17 % de la inversión bruta.</p>
    <p><strong>Nota sobre percentiles.</strong> Igual que en el informe de World3, los percentiles p10 y p90 se calculan fila por fila sobre los 100 mundos: no representan un único escenario simultáneo, sino el abanico donde cae el 80 % de las trayectorias del ensemble.</p>
  </div>
</div>
"""


def page_13_all_params_2_and_pop() -> str:
    specs_2 = [
        ("5. Recursos naturales y mercados extractivos",),
        ("res_share", "Rentas de recursos (res_share)", "% PIB", 100.0, 2, "rr_mean", 2021),
        ("res_share_gdpw", "Rentas mundiales (ψ ponderado)", "% PIB", 100.0, 2, "rr_gdpw", 2021),
        ("reserves", "Reservas restantes (reserves)", "% stock", 100.0, 2, None, None),
        ("fshare", "Extracción extranjera (fshare)", "% ext.", 100.0, 1, None, None),
        ("ims", "Dependencia importación (ims)", "% cons.", 100.0, 1, None, None),
        ("res_out", "Renta neta recursos (res_out)", "% PIB", 100.0, 2, None, None),
        ("price", "Precio mundial recursos (price)", "1970=1", 1.0, 1, "price_idx", 2025),
        ("6. Instituciones (Democracia V2), geopolítica y conflicto civil",),
        ("demo", "Democracias · V2 (demo)", "% países", 100.0, 1, "demo_share", 2025),
        ("poly", "Poliarquía continua (poly)", "V-Dem 0–1", 1.0, 3, None, None),
        ("coup", "Golpes exitosos (coup)", "% países", 100.0, 2, None, None),
        ("hazard", "Riesgo de conflicto (hazard)", "% países", 100.0, 1, "conf_rate", 2023),
        ("conflict", "Incidencia conflicto (conflict)", "% países", 100.0, 1, "conf_rate", 2023),
        ("7. Clima (M1), alimentos (M2) y mortalidad en exceso (M2+M3)",),
        ("Tg", "Temperatura global (Tg)", "°C preind.", 1.0, 2, "gmst", 2025),
        ("dclim", "Daño climático M1 (dclim)", "% PTF", 100.0, 2, None, None),
        ("natcap", "Capital natural mediano (natcap)", "0–1", 1.0, 3, None, None),
        ("co2", "Emisiones CO₂ mundial (co2)", "GtCO₂/a", 1.0, 2, "co2_gt", 2024),
        ("kcal", "Oferta calórica mundial (kcal)", "kcal/p/d", 1.0, 0, None, None),
        ("p_food", "Precio mundial cereales (p_food)", "2019=1", 1.0, 2, None, None),
        ("excess_deaths_cum_m", "Mortalidad exceso acum. (M2+M3)", "millones", 1.0, 1, None, None),
        ("8. Los seis canales de conducta estatal",),
        ("k", "Capital doméstico (k)", "% PIB", 100.0, 1, None, None),
        ("r", "Tecnología e industria (r)", "% PIB", 100.0, 2, None, None),
        ("m", "Importaciones centro (m)", "% PIB", 100.0, 1, None, None),
        ("x", "Extracción propia (x)", "% PIB", 100.0, 2, None, None),
        ("f", "Inversión exterior recursos (f)", "% PIB", 100.0, 2, None, None),
        ("w", "Redistribución social (w)", "% PIB", 100.0, 2, None, None),
    ]
    ps = D["pop_sensitivity"]
    pm = ps["metrics_2041"]
    return f"""
<div class="page">
  <span class="chap-mark">08 · TODOS LOS PARÁMETROS (II)</span>
  <div class="eyebrow">Capítulo 08 · Evolución completa 2026–2041 (parte II)</div>
  <h1 style="margin-bottom:1.5mm;">Recursos, clima, canales<br>y el límite demográfico de 2030</h1>

  <table class="tight-table" style="font-size:6.55pt;line-height:1.12;">
    <thead>
      <tr>
        <th style="padding:1.2px 4px;">Parámetro / variable de salida</th>
        <th class="num" style="padding:1.2px 4px;">2019<br>ancla</th>
        <th class="num" style="padding:1.2px 4px;">Último<br>real</th>
        <th class="num" style="padding:1.2px 4px;">2026<br>hoy</th>
        <th class="num" style="padding:1.2px 4px;">2031<br>+5 a</th>
        <th class="num" style="padding:1.2px 4px;">2041 (+15 años)<br>p10 · mediana · p90</th>
        <th class="num" style="padding:1.2px 4px;">Δ 26→41</th>
      </tr>
    </thead>
    <tbody>{_param_rows(specs_2)}</tbody>
  </table>

  <div class="slate-box" style="margin-top:1.5mm;padding:2.2mm 3mm;">
    <h3 style="font-size:7.6pt;margin-bottom:0.8mm;">Prueba de sensibilidad crítica: la población de L1 se congela en 2030</h3>
    <p style="font-size:7.0pt;line-height:1.25;">El fichero <code>demografia.csv</code> solo trae WPP 2024 hasta <strong>2030</strong> (8.274 M). Al extender suavemente las tasas 2025–2030 hasta 2041 (<strong>8.926 M</strong>, +652 M en África y Asia del Sur), el PIB pc mundial de 2041 cae {sgn(pm['gdp_pc_world']['diff_pct'], 1, '%')} y el Gini sube {sgn(pm['gini']['diff_pct'], 1, '%')}:</p>
    <table class="tight-table" style="margin:1.2mm 0 0 0;background:white;font-size:6.6pt;line-height:1.12;">
      <thead>
        <tr>
          <th style="padding:1.2px 4px;">Indicador en 2041 (mediana)</th>
          <th class="num" style="padding:1.2px 4px;">L1 (pob. fija 2030)</th>
          <th class="num" style="padding:1.2px 4px;">Extensión demográfica</th>
          <th class="num" style="padding:1.2px 4px;">Sesgo de congelar</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="padding:1.2px 4px;">PIB per cápita mundial (USD 2017)</td><td class="num" style="padding:1.2px 4px;">{f(pm['gdp_pc_world']['raw'], 0)}</td><td class="num" style="padding:1.2px 4px;">{f(pm['gdp_pc_world']['ext'], 0)}</td><td class="num hi-red" style="padding:1.2px 4px;">{sgn(pm['gdp_pc_world']['diff_pct'], 1, '%')}</td></tr>
        <tr><td style="padding:1.2px 4px;">Gini internacional</td><td class="num" style="padding:1.2px 4px;">{f(pm['gini']['raw'], 3)}</td><td class="num" style="padding:1.2px 4px;">{f(pm['gini']['ext'], 3)}</td><td class="num hi-red" style="padding:1.2px 4px;">{sgn(pm['gini']['diff_pct'], 1, '%')}</td></tr>
        <tr><td style="padding:1.2px 4px;">Deuda pública mediana (% PIB)</td><td class="num" style="padding:1.2px 4px;">{f(pm['debt']['raw']*100, 1)} %</td><td class="num" style="padding:1.2px 4px;">{f(pm['debt']['ext']*100, 1)} %</td><td class="num" style="padding:1.2px 4px;">{sgn(pm['debt']['diff_pct'], 1, '%')}</td></tr>
        <tr><td style="padding:1.2px 4px;">Riesgo de conflicto civil (% países)</td><td class="num" style="padding:1.2px 4px;">{f(pm['hazard']['raw']*100, 2)} %</td><td class="num" style="padding:1.2px 4px;">{f(pm['hazard']['ext']*100, 2)} %</td><td class="num hi-red" style="padding:1.2px 4px;">{sgn(pm['hazard']['diff_pct'], 1, '%')}</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""


def page_14_uncertainty() -> str:
    # Exact Stage-2 search bounds from world4/core/l1_atlas/calibrate.py (FREE)
    FREE = {
        "g0": (0.0, 0.02), "kappa": (0.0, 0.03), "kappa_m": (0.0, 0.3), "kappa_f": (0.0, 0.15),
        "lam0": (0.0, 1.5), "j0": (0.0, 0.15), "sig_a": (0.005, 0.08), "theta_d": (0.0, 0.4),
        "b0": (-7.0, 0.0), "b1": (0.0, 2.0), "b2": (-1.0, 0.2), "bp": (0.0, 4.5), "b3": (-3.0, 0.0),
        "b4": (0.0, 4.0), "r_R": (0.03, 0.15), "x_hist": (0.05, 1.5),
        "m_scale": (0.05, 1.0), "r_scale": (0.02, 1.0), "w_scale": (0.0, 0.6), "eps_w": (0.0, 2.0),
        "aid_eff": (0.0, 0.35), "phi_h": (0.0, 4.0), "chi_k": (0.0, 40.0),
    }
    cp = D["cutoff_params"]
    param_labels = [
        ("g0", "Deriva frontera PTF (g0)", 100.0, "%"),
        ("kappa", "Difusión base PTF (kappa)", 100.0, "%"),
        ("phi_h", "Capacidad absortiva (phi_h)", 1.0, ""),
        ("sig_a", "Ruido idiosincrático PTF (sig_a)", 100.0, "%"),
        ("theta_d", "Escala intercambio desigual (theta_d)", 1.0, ""),
        ("b1", "Sensibilidad conflicto a PSI (b1)", 1.0, ""),
        ("b3", "Protección de asabiya (b3)", 1.0, ""),
        ("b4", "Maldición de recursos en conflicto (b4)", 1.0, ""),
        ("r_R", "Pendiente logística Hubbert (r_R)", 1.0, ""),
        ("chi_k", "Coste ajuste capital (chi_k)", 1.0, ""),
        ("m_scale", "Escala importaciones centro (m_scale)", 1.0, ""),
        ("w_scale", "Escala redistribución social (w_scale)", 1.0, ""),
    ]
    stab_rows = []
    for k, lbl, sc, suf in param_labels:
        lo, hi = FREE[k]
        vals = [cp[str(yr)][k] for yr in (1970, 1980, 1990, 2000, 2010, 2019)]
        v19 = cp["2019"][k]
        cmin, cmax = min(vals), max(vals)
        d = 2 if hi * sc < 10 else 1
        sep = " a " if cmin < 0 or cmax < 0 else "–"
        rtxt = f"{f(cmin*sc, d)}{sep}{f(cmax*sc, d)}{suf} · {f(v19*sc, d)}{suf}"
        stab_rows.append((lbl, lo, hi, cmin, cmax, v19, rtxt))
    ch_stab = S.param_stability_chart(stab_rows, row_h=16.5)

    # Coverage dot chart (nominal 80 %!)
    sc10 = SC["from_2010_to_2025_15y"]
    sc19 = SC["from_2019_to_2025_recent"]
    def cov_world_10(sim_key, obs_key, last_yr):
        n_ok, n_in = 0, 0
        for yr in range(2011, last_yr + 1):
            ov = OW[obs_key][yr - 1950]
            if ov is not None:
                lo_v = S10[sim_key]["p10"][yr - 2010]
                hi_v = S10[sim_key]["p90"][yr - 2010]
                n_ok += 1
                if lo_v <= ov <= hi_v:
                    n_in += 1
        return (n_in / n_ok * 100.0) if n_ok else 0.0

    cov_rows = [
        ("PIB per cápita (país-año)", sc19["gdp_pc"]["cover_80"] * 100, sc10["gdp_pc"]["cover_80"] * 100),
        ("PIB pc mundial (agregado)", RW["gdp_pc_world"]["in_band_share"] * 100, cov_world_10("gdp_pc_world", "gdp_pc_world", 2025)),
        ("Gini internacional (mundo)", RW["gini"]["in_band_share"] * 100, cov_world_10("gini", "gini", 2025)),
        ("Deuda pública / PIB (país-año)", sc19["debt_gdp"]["cover_80"] * 100, sc10["debt_gdp"]["cover_80"] * 100),
        ("Democracia global (% países)", RW["demo"]["in_band_share"] * 100, cov_world_10("demo", "demo_share", 2025)),
        ("Rentas de recursos (país-año)", sc19["res_share"]["cover_80"] * 100, sc10["res_share"]["cover_80"] * 100),
        ("Emisiones CO₂ mundial", sc19["co2"]["cover_80"] * 100, sc10["co2"]["cover_80"] * 100),
        ("Cuota de élites E (país-año)", SC["from_2000_to_2019"]["elite_share"]["cover_80"] * 100, sc10["elite_share"]["cover_80"] * 100),
        ("Temperatura global GMST", sc19["gmst"]["cover_80"] * 100, sc10["gmst"]["cover_80"] * 100),
    ]
    ch_cov = S.coverage_dot_chart(
        cov_rows,
        label_a="Examen post-panel 2020–2025 (ancla 2019)",
        label_b="Prueba 15 años 2011–2025 (ancla 2010)",
        row_h=17.2,
    )

    return f"""
<div class="page">
  <span class="chap-mark">09 · INCERTIDUMBRE Y BANDAS</span>
  <div class="eyebrow">Capítulo 09 · Estabilidad paramétrica y calibración de bandas</div>
  <h1>Parámetros que oscilan y unas<br>bandas demasiado estrechas (D7)</h1>
  <p class="lede" style="font-size:10.3pt;margin-bottom:3mm;">Que un modelo genere 100 mundos no garantiza que sus bandas p10–p90 cubran el 80 % de la realidad. Aquí se auditan dos fuentes de error: la deriva de los parámetros al cambiar el corte histórico y la cobertura real de las bandas.</p>

  <h2 style="margin-top:0.5mm;">Doce parámetros clave en seis calibraciones sucesivas (1970, 1980, 1990, 2000, 2010 y 2019)</h2>
  <p class="muted" style="font-family:'IBM Plex Mono';font-size:6.0pt;margin-bottom:1.2mm;">BARRA CLARA = CAJA DE BÚSQUEDA · OSCURA = RANGO ENTRE CORTES 1970–2019 · MARCA NARANJA = VALOR DE 2019</p>
  {ch_stab}

  <h2 style="margin-top:2mm;">Prueba de cobertura p10–p90: fracción de observaciones reales dentro de la banda</h2>
  <p class="muted" style="font-family:'IBM Plex Mono';font-size:6.0pt;margin-bottom:1.2mm;">LÍNEA VERTICAL EN 80 % = COBERTURA NOMINAL DE UNA BANDA BIEN CALIBRADA</p>
  {ch_cov}

  <div class="grid-2" style="margin-top:1.5mm;font-size:7.7pt;">
    <p><strong>M0 y M3 reparan tres de las mayores brechas de cobertura (D7).</strong> A 15 años (2010–2025), <strong>PIB pc ({f(sc10['gdp_pc']['cover_80']*100, 1)} %)</strong>, <strong>CO₂ ({f(sc10['co2']['cover_80']*100, 0)} %)</strong> y <strong>GMST ({f(sc10['gmst']['cover_80']*100, 0)} %)</strong> alcanzan por fin el entorno del 80 % nominal (antes 0 % en CO₂ y GMST), y en 2020–2025 el PIB sube a <strong>{f(sc19['gdp_pc']['cover_80']*100, 1)} %</strong> y CO₂ al 60 %. Deuda (43–54 %), rentas (18–30 %) y élites (8 %) siguen subcubiertas.</p>
    <p><strong>Parámetros pegados a bordes o inestables.</strong> <code>r_R</code> (Hubbert, 0,06–0,145) y <code>b4</code> (maldición de recursos, 3,22–3,84) chocan contra el techo de su caja en 2019, mientras <code>g0</code> (0,30–1,68 %) y <code>w_scale</code> (0,02–0,37) recorren más de la mitad del rango según el corte histórico.</p>
  </div>
</div>
"""


def page_15_conclusions() -> str:
    sc10 = SC["from_2010_to_2025_15y"]
    sc19 = SC["from_2019_to_2025_recent"]
    ab = D["ablations_2041"]
    return f"""
<div class="page">
  <span class="chap-mark">10 · CONCLUSIONES Y LÍMITES</span>
  <div class="eyebrow">Capítulo 10 · Conclusiones</div>
  <h1>Cuánto fiarse del nuevo L1 a quince años,<br>límites y qué sigue</h1>
  <p class="lede">Activar <strong>Democracia V2 (<code>democracy.fit</code>)</strong>, <strong>Élites Regionales R10 (<code>elites.py</code>)</strong> y la <strong>capa biofísica y de colas pesadas (<code>stress.py</code>, M0–M3)</strong> como modo por defecto corrige los grandes sesgos del Atlas legado —el colapso de CO₂/Tg, el optimismo democrático, la caída espuria de <code>E</code>, la subcobertura de bandas y el optimismo climático/alimentario— y cambia el signo de 2026–2041 de expansión sostenida a meseta global.</p>

  <div class="grid-3">
    <div class="conf-card green">
      <div class="conf-head"><strong>Firme</strong><span>mejorado y validado</span></div>
      <ul>
        <li><strong>CO₂ y GMST (M0) fuera de muestra:</strong> a 15 años el MAPE de CO₂ cae de 31,7 % a <strong>{f(sc10['co2']['mape_pct'], 1)} %</strong> (bate persist. 7,9 %; cob. 86 %) y GMST alcanza <strong>80–84 %</strong> de cobertura (2025: 1,40 vs 1,41 °C).</li>
        <li><strong>Democracia V2 y conflicto:</strong> AUC democracia <strong>0,957 / 0,980</strong> (−0,7 % en 2025); conflicto <strong>0,908 / 0,913</strong>.</li>
        <li><strong>PIB y cobertura p10–p90:</strong> Spearman 0,965 (15a), cobertura <strong>{f(sc19['gdp_pc']['cover_80']*100, 1)} %</strong> (19–25) y <strong>{f(sc10['gdp_pc']['cover_80']*100, 1)} %</strong> (15a).</li>
      </ul>
    </div>
    <div class="conf-card amber">
      <div class="conf-head"><strong>Razonable</strong><span>depende de supuestos [M]</span></div>
      <ul>
        <li><strong>Meseta económica 2026–2041 (−0,1 %):</strong> con <code>Tg</code> en <strong>1,80 °C</strong>, clima total (+11,1 % PIB al apagarlo), M3 (+6,7 %) y crisis (+3,9 %) frenan el crecimiento; M2 explica el 16,4 % del conflicto.</li>
        <li><strong>Largo plazo endógeno (2050–2100):</strong> al caer el PIB bajo M1+M3, las emisiones descienden endógenamente (11 Gt) y estabilizan <code>Tg</code> en ~2,8 °C (p90: 3,4 °C) con contracción −30 % y conflicto al 23 %.</li>
        <li><strong>Mortalidad y comida:</strong> ~66,5 M muertes en exceso acum. (2020–41), sensibles a techos <code>[M]</code>.</li>
      </ul>
    </div>
    <div class="conf-card red">
      <div class="conf-head"><strong>Frágil o roto</strong><span>pendientes en L1 puro</span></div>
      <ul>
        <li><strong>Precios y reservas extractivas Hubbert:</strong> aunque M0 salvó CO₂ y <code>Tg</code>, las reservas logísticas siguen cayendo al 0,36 % y disparan <code>price</code> ×48 en 2041.</li>
        <li><strong>Población congelada post-2030 y deuda:</strong> extender WPP a 2041 convierte la meseta de PIB pc en contracción (−6,5 %); la deuda ignora el choque de tipos 2020–24.</li>
        <li><strong>Anomalías locales:</strong> bandera binaria de conflicto hunde a India (−30 %); <code>Tbar = 15 °C</code> amplifica el daño fantasma de Rusia (−38 %).</li>
      </ul>
    </div>
  </div>

  <h2>Seis conclusiones elaboradas del nuevo modo por defecto</h2>
  <div class="grid-2" style="font-size:8.0pt;line-height:1.42;">
    <div>
      <p><strong>1. M0 resuelve el mayor defecto biofísico de L1 sin fuga temporal.</strong> Al estimar la intensidad de carbono y la TCRE efectiva estrictamente con datos ≤ origen, el error de CO₂ en 2024 baja de −38,3 % a <strong>+1,5 %</strong> (39,2 vs 38,6 Gt), la GMST de 2025 clava <strong>1,40 °C</strong> (real: 1,41) y la cobertura a 15 años salta de 0 % a <strong>80–86 %</strong>.</p>
      <p><strong>2. Democracia V2 cierra el sesgo de la Tercera Ola.</strong> Con poliarquía continua, momento de 3 años y contagio vecinal COW, L1 reproduce la recesión democrática de 2025 (50,3 % vs 50,6 %, AUC 0,980) y sustituye el rebote a 61,7 % por estancamiento en 51,1 % hacia 2041.</p>
      <p><strong>3. El clima real (1,80 °C en 2041) convierte el crecimiento en meseta.</strong> El L1 legado sobreestimaba el PIB de 2041 en <strong>+17,8 %</strong> y subestimaba el conflicto en −18,6 %. Con M0–M3 activos, el PIB pc mundial queda plano en 2026–2041 (17.670 → 17.650 USD) y el conflicto escala al 15,9 %.</p>
    </div>
    <div>
      <p><strong>4. El bucle endógeno PIB ↔ CO₂ cambia el horizonte 2100.</strong> Al acoplar las emisiones al PIB endógeno (y no imponer +4,4 °C exógenos), la propia contracción económica producida por M1+M3 hacia 2050–2100 reduce el CO₂ y modera el calentamiento a ~2,8 °C, al coste de un retroceso secular del ingreso y el doble de conflictos.</p>
      <p><strong>5. Persisten tres microfallos que amplifican su coste bajo M1.</strong> Bajo <code>ρ<sub>D</sub>=0,94</code>, el valor imputado <code>Tbar = 15 °C</code> en ex-URSS agrava el daño artificial de Rusia; igualmente, la bandera binaria de conflicto sigue hundiendo a India y la población congelada de 2030 oculta una caída real de −6,5 % en 2041.</p>
      <p><strong>6. Paridad preservada con <code>legacy=True</code> y élites R10 calibradas.</strong> El modo por defecto adopta Democracia V2 + Élites Regionales R10 (RMSE 3,5 pp, cobertura 80 % en 2000–19) + Stress M0–M3, mientras <code>Simulator(..., legacy=True)</code> conserva los 38/38 hashes SHA-256 bit a bit del Atlas <code>9e96f6b</code>.</p>
    </div>
  </div>

  <h2>Qué sigue</h2>
  <div class="grid-3">
    <div style="border-top:1.2pt solid #0E162B;padding-top:2mm;">
      <strong style="font-size:8.3pt;">1. Acoplar F1–F4 con M1–M3 activos</strong>
      <p style="font-size:7.6pt;color:#475569;margin-top:1mm;">Sustituir Hubbert y TCRE por mercados (F2), energía (F3) y FaIR (F4) para que M1–M3 operen sobre el calentamiento real (+2,2 °C en 2050, +4,4 °C en 2100) y cohortes WPP (F1).</p>
    </div>
    <div style="border-top:1.2pt solid #0E162B;padding-top:2mm;">
      <strong style="font-size:8.3pt;">2. Corregir los tres microfallos locales</strong>
      <p style="font-size:7.6pt;color:#475569;margin-top:1mm;">Imputar <code>Tbar</code> de la primera década observada en ex-URSS (elimina el −37,7 % fantasma en Rusia), escalar conflicto por intensidad per cápita (India) y añadir choques comunes de tasas/deuda.</p>
    </div>
    <div style="border-top:1.2pt solid #0E162B;padding-top:2mm;">
      <strong style="font-size:8.3pt;">3. Actualizar el ancla a 2024–2025</strong>
      <p style="font-size:7.6pt;color:#475569;margin-top:1mm;">Reconstruir <code>panel_full.csv</code> con PWT 11, UCDP v25, V-Dem 2025 y GMD 2024 para arrancar la proyección desde el mundo pos-pandémico.</p>
    </div>
  </div>
</div>
"""


def page_16_appendix() -> str:
    ov = D["overlaps"]
    return f"""
<div class="page">
  <span class="chap-mark">APÉNDICE</span>
  <div class="eyebrow">Apéndice</div>
  <h1>Fuentes, verificación de empalmes<br>y cómo reproducirlo</h1>

  <h2>Fuentes y verificación del empalme 2020–2025</h2>
  <table class="tight-table">
    <thead>
      <tr><th>Serie</th><th>Fuente (panel hasta 2019 + extensión reciente)</th><th>Verificación en solape 2010–2019</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>PIB per cápita real</strong></td><td>PWT 10.01 (1950–2019) + Banco Mundial WDI <code>NY.GDP.PCAP.PP.KD</code> (2020–2025, empalmado por tasa de crecimiento desde 2019)</td><td>173 países · corr. crecimiento 9 años = <strong>{f(ov['gdp']['corr'], 3)}</strong> · MAE = {f(ov['gdp']['mae_9y'], 3)} log</td></tr>
      <tr><td><strong>Deuda pública / PIB</strong></td><td>Global Macro Database / FMI ya integrado en <code>finanzas.csv</code> y <code>world_full.csv</code> del Atlas (1950–2024)</td><td>Serie nativa del Atlas · 142–191 países en 2020–2024</td></tr>
      <tr><td><strong>Rentas de recursos / PIB</strong></td><td>Banco Mundial WDI <code>NY.GDP.TOTL.RT.ZS</code> (1970–2019 en Atlas + 2020–2021 WDI)</td><td>168 países · corr. 2019 = <strong>{f(ov['rr']['corr'], 4)}</strong> · MAE = {f(ov['rr']['mae']*100, 2)} pp</td></tr>
      <tr><td><strong>Democracia (RoW ≥ 2)</strong></td><td>V-Dem Regimes of the World (1950–2019 en Atlas + 2020–2025 vía OWID)</td><td>162 países · coincidencia exacta 2019 = <strong>{f(ov['demo']['agree_2019']*100, 1)} %</strong></td></tr>
      <tr><td><strong>Conflicto civil</strong></td><td>UCDP/PRIO ACD intraestatal tipos 3 y 4 (1950–2019 vía <code>peacesciencer</code> + 2020–2023 UCDP v24.1)</td><td>Acuerdo 2010–19 = <strong>{f(ov['conf']['agree_2010_2019']*100, 1)} %</strong> (revisión de versión: 10 % vs 16,7 % en 2019)</td></tr>
      <tr><td><strong>Temperatura y CO₂</strong></td><td>HadCRUT5 preindustrial (1850–2025), Global Carbon Budget (1850–2024), Jacks precios reales (1900–2025)</td><td>Series nativas de <code>world_full.csv</code> del Atlas (SHA-256 verificado)</td></tr>
      <tr><td><strong>Población 2020–2030</strong></td><td>Naciones Unidas WPP 2024 variante media en <code>demografia.csv</code> (congelada tras 2030 en L1 puro)</td><td>Sensibilidad 2031–2041 medida aparte extendiendo tasas 2025–2030</td></tr>
    </tbody>
  </table>

  <div class="grid-2" style="margin-top:3mm;">
    <div>
      <h2>Definiciones clave</h2>
      <ul style="padding-left:4mm;font-size:7.8pt;line-height:1.42;color:#334155;">
        <li><strong>CRPS.</strong> <em>Continuous Ranked Probability Score</em>: mide a la vez precisión y dispersión de un ensemble probabilístico (equivale al MAE para pronósticos puntuales). Menor es mejor.</li>
        <li><strong>AUC (rangos promedio).</strong> Área bajo la curva ROC (0,5 = azar; 1,0 = separación perfecta), calculada con <code>world4.validate.metrics.auc</code> para eliminar el sesgo de empates.</li>
        <li><strong>Brier.</strong> Error cuadrático medio de probabilidades predichas frente a eventos binarios (0 o 1).</li>
        <li><strong>Cobertura p10–p90.</strong> Fracción de observaciones reales que caen entre el percentil 10 y el 90 del ensemble. El valor nominal bien calibrado es <strong>80 %</strong>.</li>
        <li><strong>Zonas del Sistema-Mundo.</strong> Centro = 20 % superior de PIB per cápita mundial (36 países); Semiperiferia = 30 % siguiente (54); Periferia = 50 % inferior (90).</li>
        <li><strong>Ablación.</strong> Corrida idéntica con los mismos números aleatorios comunes apagando los parámetros de una sola teoría para aislar su efecto causal dentro del simulador.</li>
      </ul>
    </div>
    <div>
      <h2>Reproducir este informe</h2>
      <p style="font-size:7.8pt;color:#334155;">Todo el análisis sale del repositorio <strong>Phy-World</strong> con el Atlas L1 fijado en el commit <code>9e96f6b</code> (34 ficheros verificados por SHA-256). Tiempo total de simulación: <strong>{f(D['meta']['seconds'], 0)} segundos</strong> en CPU.</p>
      <pre class="code">python world4_cli.py --install-atlas
python reports/l1_atlas/fetch_assets.py
python reports/l1_atlas/run_study.py
python reports/l1_atlas/build_pdf.py</pre>
      <table class="tight-table">
        <thead><tr><th>Pieza</th><th>Módulo</th></tr></thead>
        <tbody>
          <tr><td>Simulador L1 (180 países)</td><td class="mono">world4/core/l1_atlas/model.py</td></tr>
          <tr><td>Conducta clonada (6 canales)</td><td class="mono">world4/core/l1_atlas/behavior.py</td></tr>
          <tr><td>Submodelos etapa 1 y calibración</td><td class="mono">l1_atlas/submodels.py, calibrate.py</td></tr>
          <tr><td>Validación 49 orígenes y métricas</td><td class="mono">l1_atlas/validate.py, validate/metrics.py</td></tr>
          <tr><td>Estudio 15 años y maqueta PDF</td><td class="mono">reports/l1_atlas/run_study.py, build_pdf.py</td></tr>
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
        page_03_what_is_compared(),
        page_04_architecture(),
        page_05_backtest(),
        page_06_macro(),
        page_07_sociopolitical(),
        page_08_finance_resources_climate(),
        page_09_channels_and_hubbert(),
        page_10_zones_and_countries(),
        page_11_ablations(),
        page_12_all_params_1(),
        page_13_all_params_2_and_pop(),
        page_14_uncertainty(),
        page_15_conclusions(),
        page_16_appendix(),
    ]
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>El Atlas del Sistema-Mundo, a prueba de quince años · Phy-World L1</title>
<style>{css_block()}</style>
</head>
<body>
{"".join(pages)}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    target_pdf = OUT_PDF
    try:
        weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(target_pdf))
    except PermissionError:
        target_pdf = HERE / "atlas_mundo_l1_15_anios_v2_regional.pdf"
        weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(target_pdf))
    print(f"PDF generado: {target_pdf} ({target_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
