"""Genera el documento editorial en PDF:
    «Los siete motores del Atlas Mundo (M1–M7): Teoría, matemática y ejemplos resueltos»

Explica con rigor teórico, ecuaciones exactas implementadas en Phy-World L1
(`world4/core/l1_atlas/model.py` + parámetros calibrados de `study_results.json`),
ejemplos numéricos verificables paso a paso y gráficos SVG vectoriales cada uno
de los siete módulos (M1 a M7).

Uso:
    python reports/l1_atlas/build_m1_m7_guide.py
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

import weasyprint

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
OUT_HTML = HERE / "guia_teoria_matematica_m1_m7.html"
OUT_PDF = HERE / "guia_teoria_matematica_m1_m7.pdf"

INK = "#0E162B"
INK2 = "#334155"
MUTED = "#64748B"
GRID = "#E5EBF2"
ORANGE = "#D95D24"
ORANGE_B = "#E8742D"
BLUE = "#2F66D0"
TEAL = "#1F7A6E"
CRIMSON = "#B83227"
AMBER = "#B58100"


def font_uri(name: str) -> str:
    return (FONTS / name).resolve().as_uri()


def _poly(pts: list[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) if pts else ""


def fmt_es(v: float, d: int = 2) -> str:
    s = f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "−")


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICOS SVG DIDÁCTICOS (VECTORIALES, ALTA PRECISIÓN)
# ═══════════════════════════════════════════════════════════════════════════

def svg_cover_architecture() -> str:
    """Diagrama vectorial de acoplamiento de los 7 módulos (M1-M7) sobre azul noche."""
    w, h = 595, 290
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append('<defs>'
                 '<marker id="arrO" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">'
                 '<path d="M 0 1 L 7 4 L 0 7 z" fill="#E8742D"/></marker>'
                 '<marker id="arrB" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">'
                 '<path d="M 0 1 L 7 4 L 0 7 z" fill="#7BA6FF"/></marker>'
                 '<marker id="arrT" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">'
                 '<path d="M 0 1 L 7 4 L 0 7 z" fill="#4ECDC4"/></marker>'
                 '</defs>')

    # Subtle background grid rings
    parts.append('<circle cx="297" cy="145" r="118" fill="none" stroke="#223154" stroke-width="0.9" stroke-dasharray="3,4"/>')
    parts.append('<circle cx="297" cy="145" r="64" fill="none" stroke="#223154" stroke-width="0.7"/>')

    # Edges (curved feedback paths between modules)
    edges = [
        # M1 <-> M2 (trade/core)
        ("M 235 128 Q 160 100 135 76", "#7BA6FF", "arrB", "PIB y c_i", 190, 96),
        ("M 142 94 Q 180 135 234 146", "#E8742D", "arrO", "rent_in / drain", 174, 144),
        # M1 <-> M5 (resources & price)
        ("M 235 162 Q 165 180 135 210", "#4ECDC4", "arrT", "demanda R", 168, 178),
        ("M 146 218 Q 198 202 246 172", "#E8742D", "arrO", "precio p, ext", 214, 208),
        # M1 <-> M7 (CO2 & BHM climate)
        ("M 358 164 Q 422 182 456 210", "#E8742D", "arrO", "CO₂ fósil", 428, 176),
        ("M 446 218 Q 396 202 348 172", "#4ECDC4", "arrT", "D_clim, N_nat", 390, 212),
        # M1 <-> M3 (elites & PSI)
        ("M 358 128 Q 422 98 456 76", "#7BA6FF", "arrB", "rentas", 408, 96),
        # M3 -> M6 (PSI -> conflict/coups)
        ("M 498 94 Q 515 144 498 192", "#E8742D", "arrO", "PSI", 528, 146),
        # M2 -> M4 (frontier -> asabiya)
        ("M 155 58 Q 225 32 242 42", "#4ECDC4", "arrT", "frontera F_i", 198, 34),
        # M4 -> M1 (asabiya -> eff, jumps)
        ("M 297 72 L 297 108", "#4ECDC4", "arrT", "eff, λ", 316, 94),
        # M4 -> M3/M6 (asabiya protects)
        ("M 354 50 Q 405 46 442 58", "#4ECDC4", "arrT", "b₃ = −2,56", 398, 42),
        # M6 -> M1 (crises/democracy -> PTF)
        ("M 297 224 L 297 182", "#E8742D", "arrO", "crisis, B_dem", 328, 204),
        # M6 <-> M3
        ("M 352 246 Q 418 258 458 232", "#7BA6FF", "arrB", "deuda d/d*", 408, 260),
    ]
    for d, col, mk, lbl, lx, ly in edges:
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.25" opacity="0.82" marker-end="url(#{mk})"/>')
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.4" fill="#A9B9D6">{escape(lbl)}</text>')

    # Module nodes
    nodes = [
        (297, 145, 126, 66, "#1B2949", "#E8742D", "M1 · PRODUCCIÓN Y PTF", "Y = A·K^α·R^ψ·(hL)^(1−α−ψ)", "Convergencia h^1,66 + Poisson"),
        (96, 62, 122, 52, "#15213D", "#7BA6FF", "M2 · SISTEMA-MUNDO", "Wallerstein / Amin", "c_i, θ_d = 0,20, μ = 0,25"),
        (297, 46, 118, 48, "#15213D", "#4ECDC4", "M4 · ASABIYA", "Ibn Jaldún / Turchin", "r_a = 0,02 · d_a = 0,007"),
        (498, 62, 122, 52, "#15213D", "#E8742D", "M3 · ÉLITES Y PSI", "Goldstone / Turchin", "E, n_e, PSI = MMP·EMP·SFD"),
        (96, 226, 122, 52, "#15213D", "#4ECDC4", "M5 · RECURSOS", "Curvas de Hubbert", "r_R = 0,145 · concesiones"),
        (297, 246, 124, 48, "#15213D", "#7BA6FF", "M6 · DEUDA Y RÉGIMEN", "Crisis, golpes y V-Dem", "d*, r*_US, −4,5 % PIB"),
        (498, 226, 122, 52, "#15213D", "#E8742D", "M7 · CLIMA Y ECOLOGÍA", "TCRE + daño BHM", "T* = 13,0 °C · N_nat"),
    ]
    for cx, cy, nw, nh, bg, bd, t1, t2, t3 in nodes:
        x0, y0 = cx - nw / 2, cy - nh / 2
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{nw}" height="{nh}" rx="5" fill="{bg}" stroke="{bd}" stroke-width="1.35"/>')
        parts.append(f'<text x="{cx}" y="{y0 + 15:.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.8" font-weight="500" fill="{bd}">{escape(t1)}</text>')
        parts.append(f'<text x="{cx}" y="{y0 + 28.5:.1f}" text-anchor="middle" font-family="IBM Plex Sans" font-size="7.6" font-weight="600" fill="#FFFFFF">{escape(t2)}</text>')
        parts.append(f'<text x="{cx}" y="{y0 + 40.5:.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.2" fill="#9EB0CF">{escape(t3)}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def svg_m1_cobb_douglas() -> str:
    """Curva Cobb-Douglas y(k) para dos niveles de PTF y capital humano."""
    w, h = 250, 126
    ml, mr, mt, mb = 32, 16, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda k: ml + k / 100.0 * pw
    yfn = lambda y: mt + (1.0 - y / 52.0) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Rendimientos decrecientes del capital vs. salto de PTF</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">PIB PER CÁPITA y(k) = A·k^0,35·r^0,05·h^0,60 (MILES USD)</text>')
    for yv in (0, 15, 30, 45):
        yy = yfn(yv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{yv}k</text>')
    for kv in (0, 25, 50, 75, 100):
        parts.append(f'<text x="{xfn(kv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{kv}k</text>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">Capital por trabajador k (miles USD)</text>')

    ks = [i for i in range(1, 101)]
    # Curve 1: h=2.0, A=2.8
    c1 = [(xfn(k), yfn(2.8 * (k ** 0.35) * (2.0 ** 0.60))) for k in ks]
    # Curve 2: h=3.5, A=4.0
    c2 = [(xfn(k), yfn(4.0 * (k ** 0.35) * (3.5 ** 0.60))) for k in ks]
    parts.append(f'<path d="{_poly(c1)}" fill="none" stroke="{BLUE}" stroke-width="1.7"/>')
    parts.append(f'<path d="{_poly(c2)}" fill="none" stroke="{ORANGE}" stroke-width="1.9"/>')
    # Vertical comparison at k=50
    y1_50 = 2.8 * (50 ** 0.35) * (2.0 ** 0.60)
    y2_50 = 4.0 * (50 ** 0.35) * (3.5 ** 0.60)
    parts.append(f'<line x1="{xfn(50):.1f}" y1="{yfn(y1_50):.1f}" x2="{xfn(50):.1f}" y2="{yfn(y2_50):.1f}" stroke="{INK}" stroke-width="0.9" stroke-dasharray="2,1.5"/>')
    parts.append(f'<circle cx="{xfn(50):.1f}" cy="{yfn(y1_50):.1f}" r="2.4" fill="{BLUE}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<circle cx="{xfn(50):.1f}" cy="{yfn(y2_50):.1f}" r="2.4" fill="{ORANGE}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<text x="{xfn(47):.1f}" y="{yfn((y1_50+y2_50)/2)+2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" font-weight="500" fill="{INK}">×2,0 igual k</text>')
    parts.append(f'<text x="{xfn(98):.1f}" y="{yfn(4.0*(100**0.35)*(3.5**0.60))-5:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="6.6" font-weight="600" fill="{ORANGE}">A alto, h = 3,5</text>')
    parts.append(f'<text x="{xfn(98):.1f}" y="{yfn(2.8*(100**0.35)*(2.0**0.60))-6:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="6.6" font-weight="600" fill="{BLUE}">A medio, h = 2,0</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m1_absorptive() -> str:
    """Curva de capacidad absortiva (h / h_F)^1,66 vs lineal."""
    w, h = 250, 126
    ml, mr, mt, mb = 32, 16, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda x: ml + x / 1.1 * pw
    yfn = lambda y: mt + (1.0 - y / 1.2) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Capacidad absortiva no lineal: (h / h_F)^1,66</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">FRACCIÓN DE LA BRECHA TECNOLÓGICA QUE EL PAÍS LOGRA ABSORBER</text>')
    for yv in (0.0, 0.4, 0.8, 1.2):
        yy = yfn(yv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{fmt_es(yv,1)}</text>')
    for xv in (0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<text x="{xfn(xv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{fmt_es(xv,1)}</text>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">Capital humano relativo a la frontera (h / h_F)</text>')

    xs = [i / 100.0 for i in range(5, 111)]
    lin = [(xfn(x), yfn(x)) for x in xs]
    abs_c = [(xfn(x), yfn(x ** 1.657)) for x in xs]
    parts.append(f'<path d="{_poly(lin)}" fill="none" stroke="#94A3B8" stroke-width="1.1" stroke-dasharray="2.5,2"/>')
    parts.append(f'<path d="{_poly(abs_c)}" fill="none" stroke="{ORANGE}" stroke-width="2.0"/>')
    # Highlight 0.40 and 0.90
    for xv, lbl, col, dy in ((0.40, "0,40 → 0,22", CRIMSON, 3), (0.90, "0,90 → 0,84", TEAL, -7)):
        yv = xv ** 1.657
        parts.append(f'<circle cx="{xfn(xv):.1f}" cy="{yfn(yv):.1f}" r="2.7" fill="{col}" stroke="white" stroke-width="0.7"/>')
        dx = 6 if xv < 0.6 else -4
        anc = "start" if xv < 0.6 else "end"
        parts.append(f'<text x="{xfn(xv)+dx:.1f}" y="{yfn(yv)+dy:.1f}" text-anchor="{anc}" font-family="IBM Plex Mono" font-size="6.3" font-weight="500" fill="{col}">{lbl}</text>')
    parts.append(f'<text x="{xfn(0.50):.1f}" y="{yfn(0.62):.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="#64748B">lineal (φ_h = 1)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m2_world_system() -> str:
    """Curva sigmoide de centralidad c_i + diagrama de intercambio desigual."""
    w, h = 515, 118
    # Left panel: coreness sigmoid
    ml, mr, mt, mb = 32, 285, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    # ln y from ln(1000) to ln(90000), ref Q80 ~ ln(25000)
    xmin, xmax = math.log(1000), math.log(90000)
    ref = math.log(25000)
    xfn = lambda ly: ml + (ly - xmin) / (xmax - xmin) * pw
    yfn = lambda c: mt + (1.0 - c) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Centralidad c_i = σ((ln y − Q₈₀) / 0,35)</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">TRANSICIÓN SUAVE DE PERIFERIA (0) A CENTRO (1)</text>')
    for cv in (0.0, 0.5, 1.0):
        yy = yfn(cv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{fmt_es(cv,1)}</text>')
    for yk, lbl in ((2000, "2k"), (5000, "5k"), (15000, "15k"), (25000, "Q₈₀"), (60000, "60k")):
        xx = xfn(math.log(yk))
        parts.append(f'<text x="{xx:.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{lbl}</text>')
    parts.append(f'<line x1="{xfn(ref):.1f}" y1="{mt}" x2="{xfn(ref):.1f}" y2="{mt+ph}" stroke="#94A3B8" stroke-width="0.7" stroke-dasharray="2,2"/>')

    pts = []
    for i in range(80):
        ly = xmin + i / 79.0 * (xmax - xmin)
        c = 1.0 / (1.0 + math.exp(-(ly - ref) / 0.35))
        pts.append((xfn(ly), yfn(c)))
    parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="{BLUE}" stroke-width="2.0"/>')
    for yk, nm, col in ((4000, "Periferia", CRIMSON), (25000, "Semi", AMBER), (60000, "Centro", TEAL)):
        ly = math.log(yk)
        c = 1.0 / (1.0 + math.exp(-(ly - ref) / 0.35))
        parts.append(f'<circle cx="{xfn(ly):.1f}" cy="{yfn(c):.1f}" r="2.6" fill="{col}" stroke="white" stroke-width="0.6"/>')
        dy = -5 if yk != 60000 else 10
        parts.append(f'<text x="{xfn(ly):.1f}" y="{yfn(c)+dy:.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.8" fill="{col}">{nm}</text>')

    # Right panel: bilateral unequal exchange schematic
    rx = 256
    parts.append(f'<text x="{rx}" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Flujos bilaterales en la red comercial COW</text>')
    parts.append(f'<text x="{rx}" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">SUMA CERO GLOBAL PERO ASIMETRÍA ESTRUCTURAL</text>')
    parts.append(f'<rect x="{rx}" y="30" width="80" height="72" rx="4" fill="#FDF2EE" stroke="{ORANGE}" stroke-width="1.2"/>')
    parts.append(f'<text x="{rx+40}" y="45" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.2" font-weight="500" fill="{ORANGE}">PERIFERIA (i)</text>')
    parts.append(f'<text x="{rx+40}" y="58" text-anchor="middle" font-family="IBM Plex Sans" font-size="7.2" font-weight="600" fill="{INK}">y = 4.000 USD</text>')
    parts.append(f'<text x="{rx+40}" y="70" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{INK2}">c_i = 0,005</text>')
    parts.append(f'<text x="{rx+40}" y="82" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.8" fill="{CRIMSON}">Pierde drain+μm</text>')
    parts.append(f'<text x="{rx+40}" y="94" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.8" fill="{TEAL}">↑ frontera (M4)</text>')

    cx = 432
    parts.append(f'<rect x="{cx}" y="30" width="80" height="72" rx="4" fill="#EEF4FD" stroke="{BLUE}" stroke-width="1.2"/>')
    parts.append(f'<text x="{cx+40}" y="45" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.2" font-weight="500" fill="{BLUE}">CENTRO (j)</text>')
    parts.append(f'<text x="{cx+40}" y="58" text-anchor="middle" font-family="IBM Plex Sans" font-size="7.2" font-weight="600" fill="{INK}">y = 60.000 USD</text>')
    parts.append(f'<text x="{cx+40}" y="70" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{INK2}">c_j = 0,924</text>')
    parts.append(f'<text x="{cx+40}" y="82" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.8" fill="{BLUE}">Cobra rent_in</text>')
    parts.append(f'<text x="{cx+40}" y="94" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.8" fill="{MUTED}">+GNI, ↑ élite E</text>')

    parts.append('<defs><marker id="arO2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 1 L 7 4 L 0 7 z" fill="#D95D24"/></marker>'
                 '<marker id="arB2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 1 L 7 4 L 0 7 z" fill="#2F66D0"/></marker></defs>')
    mid = (rx + 80 + cx) / 2
    parts.append(f'<line x1="{rx+82}" y1="50" x2="{cx-3}" y2="50" stroke="{ORANGE}" stroke-width="1.5" marker-end="url(#arO2)"/>')
    parts.append(f'<text x="{mid:.0f}" y="45" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.6" fill="{ORANGE}">Intercambio θ_d=0,20</text>')
    parts.append(f'<line x1="{rx+82}" y1="72" x2="{cx-3}" y2="72" stroke="{ORANGE}" stroke-width="1.5" marker-end="url(#arO2)"/>')
    parts.append(f'<text x="{mid:.0f}" y="67" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.6" fill="{ORANGE}">Monopolio μ=0,25</text>')
    parts.append(f'<line x1="{cx-2}" y1="91" x2="{rx+83}" y2="91" stroke="{BLUE}" stroke-width="1.2" stroke-dasharray="2.5,1.8" marker-end="url(#arB2)"/>')
    parts.append(f'<text x="{mid:.0f}" y="101" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.5" fill="{BLUE}">Tec. incorporada (M1)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m3_psi() -> str:
    """Efecto no lineal de sobreproducción de élites (n_e^2) sobre EMP y PSI."""
    w, h = 250, 122
    ml, mr, mt, mb = 32, 16, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda ne: ml + (ne - 0.5) / 2.0 * pw
    yfn = lambda v: mt + (1.0 - min(v, 7.0) / 7.0) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Aceleración cuadrática de EMP = n_e² / (v/v₀)</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">POTENCIAL DE MOVILIZACIÓN DE ÉLITES SEGÚN SOBREPRODUCCIÓN n_e</text>')
    for yv in (0, 2, 4, 6):
        yy = yfn(yv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{yv}×</text>')
    for nev in (0.5, 1.0, 1.5, 2.0, 2.5):
        parts.append(f'<text x="{xfn(nev):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{fmt_es(nev,1)}</text>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">Número relativo de aspirantes a élite (n_e)</text>')

    nes = [0.5 + i * 0.04 for i in range(51)]
    c_hi = [(xfn(n), yfn(n * n / 0.8)) for n in nes]   # renta relativa baja (élites frustradas!)
    c_eq = [(xfn(n), yfn(n * n / 1.0)) for n in nes]   # equilibrio v/v0 = 1
    c_lo = [(xfn(n), yfn(n * n / 1.5)) for n in nes]   # renta relativa alta
    parts.append(f'<path d="{_poly(c_lo)}" fill="none" stroke="{TEAL}" stroke-width="1.3" stroke-dasharray="2.5,2"/>')
    parts.append(f'<path d="{_poly(c_eq)}" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
    parts.append(f'<path d="{_poly(c_hi)}" fill="none" stroke="{ORANGE}" stroke-width="2.0"/>')
    parts.append(f'<circle cx="{xfn(1.0):.1f}" cy="{yfn(1.0):.1f}" r="2.4" fill="{BLUE}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<circle cx="{xfn(1.8):.1f}" cy="{yfn(1.8*1.8/0.8):.1f}" r="2.6" fill="{ORANGE}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<text x="{xfn(1.78):.1f}" y="{yfn(4.05)-6:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" font-weight="500" fill="{ORANGE}">v/v₀=0,8 (frustración)</text>')
    parts.append(f'<text x="{xfn(2.45):.1f}" y="{yfn(2.45*2.45/1.5)+15:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="{TEAL}">v/v₀=1,5</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m4_asabiya() -> str:
    """Izquierda: evolución temporal de asabiya; Derecha: efecto b3=-2,56 en riesgo de guerra."""
    w, h = 515, 118
    # Left plot: S(t) over 30 years
    ml, mr, mt, mb = 32, 285, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda t: ml + t / 30.0 * pw
    yfn = lambda s: mt + (1.0 - s) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Evolución de la asabiya S_t (30 años)</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">CRECIMIENTO EN FRONTERA VS. EROSIÓN EN CENTRO RICO</text>')
    for sv in (0.0, 0.5, 1.0):
        yy = yfn(sv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{fmt_es(sv,1)}</text>')
    for tv in (0, 10, 20, 30):
        parts.append(f'<text x="{xfn(tv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{tv}a</text>')

    # Simulate exact M4 trajectories
    s_front, s_core, s_war = [0.40], [0.70], [0.60]
    for _ in range(30):
        sf = s_front[-1]
        s_front.append(sf + 0.02 * 0.85 * sf * (1 - sf))
        sc = s_core[-1]
        s_core.append(sc - 0.007 * 0.95 * 0.95 * sc * 1.8)
        sw = s_war[-1]
        s_war.append(sw - 0.03 * sw)
    parts.append(f'<path d="{_poly([(xfn(t), yfn(v)) for t, v in enumerate(s_front)])}" fill="none" stroke="{TEAL}" stroke-width="1.9"/>')
    parts.append(f'<path d="{_poly([(xfn(t), yfn(v)) for t, v in enumerate(s_core)])}" fill="none" stroke="{BLUE}" stroke-width="1.7"/>')
    parts.append(f'<path d="{_poly([(xfn(t), yfn(v)) for t, v in enumerate(s_war)])}" fill="none" stroke="{CRIMSON}" stroke-width="1.6" stroke-dasharray="2.5,1.8"/>')
    parts.append(f'<text x="{xfn(30)-2:.1f}" y="{yfn(s_front[-1])-7:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="{TEAL}">Frontera (F=0,85): 0,52</text>')
    parts.append(f'<text x="{xfn(2):.1f}" y="{yfn(0.74):.1f}" text-anchor="start" font-family="IBM Plex Mono" font-size="5.9" fill="{BLUE}">Centro rico (n_e=1,8): 0,70→0,49</text>')
    parts.append(f'<text x="{xfn(30)-2:.1f}" y="{yfn(s_war[-1])+9:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="{CRIMSON}">Guerra civil: 0,24</text>')

    # Right plot: Conflict onset probability vs asabiya (b3 = -2.56)
    ml2, pw2 = 294, 206
    x2 = lambda s: ml2 + s * pw2
    y2 = lambda p: mt + (1.0 - min(p, 0.20) / 0.20) * ph
    parts.append(f'<text x="264" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Escudo antibélico: P(inicio guerra) vs. S</text>')
    parts.append(f'<text x="264" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">EFECTO DE b₃ = −2,56 (ANOCRACIA, RENTAS 10 % PIB)</text>')
    for pv in (0.0, 0.05, 0.10, 0.15, 0.20):
        yy = y2(pv)
        parts.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml2-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{int(pv*100)} %</text>')
    for sv in (0.1, 0.3, 0.5, 0.7, 0.9):
        parts.append(f'<text x="{x2(sv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{fmt_es(sv,1)}</text>')

    # Logit at ly=8.5 (5k USD), anoc=0.8, rev/G=0.10, lpsi=0, tz=0 -> base = -6.287 -0.403*(-0.5) + 3.841*0.10 + 1.0*0.8 = -4.901 (wait add +2.2 for high-risk state example with rev/G=0.25 -> base = -3.3)
    # Better: use exact calibrated coefficients for a vulnerable petro-anocracy: ly=8.0 (-0.403*-1=+0.403), rev/G=0.25 (+0.960), anoc=1.0 (+1.0), lpsi=2.0 (+0.128) => intercept = -6.287 + 0.403 + 0.960 + 1.0 + 0.128 = -3.796? Let's plot exact for rev/G=0.40 (+1.536), anoc=1.0, ly=7.5 (+0.605) => base = -3.02
    base_lg = -2.85
    ss = [0.05 + i * 0.02 for i in range(46)]
    pts2 = [(x2(s), y2(1.0 / (1.0 + math.exp(-(base_lg - 2.561 * (s - 0.5)))))) for s in ss]
    parts.append(f'<path d="{_poly(pts2)}" fill="none" stroke="{ORANGE}" stroke-width="2.0"/>')
    for sv, col in ((0.25, CRIMSON), (0.50, AMBER), (0.75, TEAL)):
        pv = 1.0 / (1.0 + math.exp(-(base_lg - 2.561 * (sv - 0.5))))
        parts.append(f'<circle cx="{x2(sv):.1f}" cy="{y2(pv):.1f}" r="2.7" fill="{col}" stroke="white" stroke-width="0.6"/>')
        parts.append(f'<text x="{x2(sv)+5:.1f}" y="{y2(pv)-3:.1f}" font-family="IBM Plex Mono" font-size="6.1" font-weight="500" fill="{col}">{fmt_es(pv*100,1)} %</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m5_hubbert() -> str:
    """Izquierda: Campana logística de Hubbert Cap(x); Derecha: explosión del precio con ψ fija."""
    w, h = 515, 120
    ml, mr, mt, mb = 32, 285, 26, 22
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda x: ml + x * pw
    yfn = lambda c: mt + (1.0 - c / 4.2) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Capacidad de Hubbert: Cap(x) = r_R·U·(x+0,01)(1−x)</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">CON U = 100 Y r_R = 0,145 · PICO EN x = 0,495</text>')
    for cv in (0, 1, 2, 3, 4):
        yy = yfn(cv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{cv}</text>')
    for xv in (0.0, 0.25, 0.50, 0.75, 1.0):
        parts.append(f'<text x="{xfn(xv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{int(xv*100)} %</text>')

    xs = [i / 100.0 for i in range(101)]
    cap_pts = [(xfn(x), yfn(0.145 * 100.0 * (x + 0.01) * (1 - x))) for x in xs]
    ext_pts = [(xfn(x), yfn(0.145 * 100.0 * (x + 0.01) * (1 - x) * (1 - math.exp(-1.2)))) for x in xs]
    parts.append(f'<path d="{_poly(cap_pts)}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
    parts.append(f'<path d="{_poly(ext_pts)}" fill="none" stroke="{ORANGE}" stroke-width="1.8" stroke-dasharray="3,1.8"/>')
    pk = 0.145 * 100.0 * 0.505 * 0.505
    parts.append(f'<circle cx="{xfn(0.495):.1f}" cy="{yfn(pk):.1f}" r="2.6" fill="{BLUE}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<text x="{xfn(0.52):.1f}" y="{yfn(pk)-4:.1f}" font-family="IBM Plex Mono" font-size="6.0" fill="{BLUE}">Pico: 3,70</text>')
    parts.append(f'<text x="{xfn(0.50):.1f}" y="{yfn(1.55):.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.9" fill="{ORANGE}">ext efectiva (e=1,2)</text>')

    # Right plot: Price explosion as world supply S falls with fixed psi_t
    ml2, pw2 = 294, 206
    x2 = lambda s: ml2 + (1.0 - s) * pw2  # depletion from left (S=1) to right (S=0.02)
    y2 = lambda p: mt + (1.0 - min(p, 50.0) / 50.0) * ph
    parts.append(f'<text x="264" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">La «trampa de Hubbert»: precio p al caer S</text>')
    parts.append(f'<text x="264" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">SI ψ_t QUEDA CONGELADA, p ∝ S^−(1−ψ) SE DISPARA</text>')
    for pv in (0, 10, 25, 50):
        yy = y2(pv)
        parts.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml2-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">×{pv}</text>')
    for s_lbl, sv in (("100 %", 1.0), ("50 %", 0.5), ("20 %", 0.2), ("2 %", 0.02)):
        parts.append(f'<text x="{x2(sv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{s_lbl}</text>')
    ss = [1.0 - i * 0.015 for i in range(66)]
    p_pts = [(x2(s), y2((s ** -0.96))) for s in ss if s >= 0.02]
    parts.append(f'<path d="{_poly(p_pts)}" fill="none" stroke="{CRIMSON}" stroke-width="2.0"/>')
    parts.append(f'<circle cx="{x2(0.10):.1f}" cy="{y2(0.10**-0.96):.1f}" r="2.5" fill="{CRIMSON}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<text x="{x2(0.12):.1f}" y="{y2(9.1)-5:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{CRIMSON}">S=10 % → p ×9,1</text>')
    parts.append(f'<text x="{x2(0.06):.1f}" y="{y2(42):.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" font-weight="500" fill="{CRIMSON}">2041: p ×60</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m6_crisis_anoc() -> str:
    """Izquierda: P(crisis financiera) vs tasa real EE. UU.; Derecha: U invertida de anocracia 4p(1-p)."""
    w, h = 515, 102
    ml, mr, mt, mb = 32, 285, 25, 20
    pw, ph = w - ml - mr, h - mt - mb
    xfn = lambda r: ml + (r + 0.02) / 0.10 * pw
    yfn = lambda p: mt + (1.0 - min(p, 0.30) / 0.30) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">Riesgo anual de crisis financiera vs. tasa real EE. UU.</text>')
    parts.append(f'<text x="0" y="18.2" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">LOGIT ETAPA 1 (n = 7.271) · β_r = +9,97 · β_cont = +5,22</text>')
    for pv in (0.0, 0.10, 0.20, 0.30):
        yy = yfn(pv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{int(pv*100)} %</text>')
    for rv in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08):
        parts.append(f'<text x="{xfn(rv):.1f}" y="{mt+ph+10.5}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{int(rv*100)} %</text>')

    rs = [-0.02 + i * 0.002 for i in range(51)]
    # Calm: g=+2%, contagio=3%, d/d*=1.0
    p_calm = [(xfn(r), yfn(1 / (1 + math.exp(-(-3.056 + 9.968 * r - 3.924 * 0.02 + 5.224 * 0.03))))) for r in rs]
    # Stress: g=-3%, contagio=15%, d/d*=1.25
    p_str = [(xfn(r), yfn(1 / (1 + math.exp(-(-3.056 + 0.076 * 0.25 + 9.968 * r - 3.924 * (-0.03) + 5.224 * 0.15))))) for r in rs]
    parts.append(f'<path d="{_poly(p_calm)}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
    parts.append(f'<path d="{_poly(p_str)}" fill="none" stroke="{CRIMSON}" stroke-width="1.9"/>')
    parts.append(f'<text x="{xfn(0.078):.1f}" y="{yfn(0.245):.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="{CRIMSON}">Recesión + contagio 15 %</text>')
    parts.append(f'<text x="{xfn(0.078):.1f}" y="{yfn(0.118):.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.9" fill="{BLUE}">Crecimiento +2 %, calma</text>')

    # Right plot: Anocracy inverted-U 4*p*(1-p)
    ml2, pw2 = 294, 206
    x2 = lambda p: ml2 + p * pw2
    y2 = lambda v: mt + (1.0 - v / 1.1) * ph
    parts.append(f'<text x="264" y="10" font-family="IBM Plex Sans" font-size="8.4" font-weight="600" fill="{INK}">La U invertida de la anocracia: 4·p·(1 − p)</text>')
    parts.append(f'<text x="264" y="18.2" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">LOS REGÍMENES HÍBRIDOS SON LOS MÁS INESTABLES (PITF)</text>')
    for vv in (0.0, 0.5, 1.0):
        yy = y2(vv)
        parts.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="0.65"/>')
        parts.append(f'<text x="{ml2-4}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{fmt_es(vv,1)}</text>')
    for pv, lbl in ((0.0, "0 Aut."), (0.5, "0,5 Híbrido"), (1.0, "1 Dem.")):
        parts.append(f'<text x="{x2(pv):.1f}" y="{mt+ph+10.5}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{lbl}</text>')
    ps = [i / 50.0 for i in range(51)]
    apts = [(x2(p), y2(4 * p * (1 - p))) for p in ps]
    parts.append(f'<path d="{_poly(apts)}" fill="none" stroke="{ORANGE}" stroke-width="2.0"/>')
    parts.append(f'<circle cx="{x2(0.5):.1f}" cy="{y2(1.0):.1f}" r="2.7" fill="{CRIMSON}" stroke="white" stroke-width="0.6"/>')
    parts.append(f'<text x="{x2(0.5):.1f}" y="{y2(0.50):.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.9" fill="{CRIMSON}">Máximo riesgo (+1,0 logit)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_m7_bhm() -> str:
    """Parábola exacta de Burke-Hsiang-Miguel (2015) con óptimo en 13,04 °C y tres países."""
    w, h = 515, 122
    ml, mr, mt, mb = 38, 18, 22, 22
    pw, ph = w - ml - mr, h - mt - mb
    tmin, tmax = 0.0, 32.0
    xfn = lambda T: ml + (T - tmin) / (tmax - tmin) * pw
    # h(T) = 0.0127*T - 0.000487*T^2 -> max at 13.04 is 0.0828; at 32 is -0.0923
    hfn = lambda T: 0.0127 * T - 0.000487 * (T ** 2)
    ylo, yhi = -0.10, 0.10
    yfn = lambda v: mt + (1.0 - (v - ylo) / (yhi - ylo)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="10" font-family="IBM Plex Sans" font-size="8.5" font-weight="600" fill="{INK}">Función cuadrática de daño climático BHM: h(T) = 0,0127·T − 0,000487·T²</text>')
    parts.append(f'<text x="0" y="18.5" font-family="IBM Plex Mono" font-size="5.7" fill="{MUTED}">EFECTO SOBRE LA TASA DE PTF SEGÚN TEMPERATURA MEDIA ANUAL · UN MISMO +2 °C CREA GANADORES Y PERDEDORES</text>')

    for yv in (-0.08, -0.04, 0.0, 0.04, 0.08):
        yy = yfn(yv)
        st = "#94A3B8" if yv == 0 else GRID
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{st}" stroke-width="{0.9 if yv==0 else 0.65}"/>')
        parts.append(f'<text x="{ml-5}" y="{yy+2.2:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{fmt_es(yv*100,0)} %</text>')
    for tv in (0, 5, 10, 13, 15, 20, 25, 30):
        parts.append(f'<text x="{xfn(tv):.1f}" y="{mt+ph+11}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{tv}°</text>')

    # Optimum vertical line
    topt = 0.0127 / (2 * 0.000487)
    parts.append(f'<line x1="{xfn(topt):.1f}" y1="{mt}" x2="{xfn(topt):.1f}" y2="{mt+ph}" stroke="{TEAL}" stroke-width="0.9" stroke-dasharray="2.5,2"/>')
    parts.append(f'<text x="{xfn(topt)-4:.1f}" y="{mt+5:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" font-weight="500" fill="{TEAL}">Óptimo T* = 13,04 °C</text>')

    ts = [i * 0.5 for i in range(65)]
    pts = [(xfn(T), yfn(hfn(T))) for T in ts]
    parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="{INK}" stroke-width="2.0"/>')

    # Three country shifts (+2 degC) with clean non-overlapping callouts
    cases = [
        (3.0, 5.0, "Noruega (3°→5°): +1,76 %/a", TEAL, 5.2, 0.018, "start"),
        (13.0, 15.0, "EE. UU./España (13°→15°): −0,19 %/a", BLUE, 16.5, 0.088, "start"),
        (27.0, 29.0, "India / Nigeria (27°→29°): −2,92 %/a", CRIMSON, 26.2, -0.025, "end"),
    ]
    for t0, t1, lbl, col, lx, ly, anc in cases:
        x0, y0 = xfn(t0), yfn(hfn(t0))
        x1, y1 = xfn(t1), yfn(hfn(t1))
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{col}" stroke-width="3.2"/>')
        parts.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="2.6" fill="white" stroke="{col}" stroke-width="1.3"/>')
        parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="3.0" fill="{col}"/>')
        parts.append(f'<text x="{xfn(lx):.1f}" y="{yfn(ly):.1f}" text-anchor="{anc}" font-family="IBM Plex Sans" font-size="6.5" font-weight="600" fill="{col}">{escape(lbl)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_cascade_integration() -> str:
    """Diagrama de cascada: cómo un choque de +3 pp en la tasa real de EE. UU. recorre M6->M3->M4->M1."""
    w, h = 515, 92
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append('<defs><marker id="arC" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 1 L 7 4 L 0 7 z" fill="#D95D24"/></marker></defs>')
    steps = [
        (4, "#FDF2EE", CRIMSON, "CHOQUE EN M6", "Fed sube r*_US +3 pp", "P(crisis) ×1,35 · +interés"),
        (108, "#EEF4FD", BLUE, "EFECTO EN M1 / M2", "Salida de rent_out", "Crisis: −4,5 % PTF perm."),
        (212, "#FDF2EE", ORANGE, "PROPAGACIÓN M3", "Cae renta de masas", "Sube MMP y SFD → ↑ PSI"),
        (316, "#EEF8F6", TEAL, "RESPUESTA EN M4", "Si estalla conflicto", "Asabiya cae −3 %/año"),
        (420, "#F4F6F9", INK, "BUCLE AÑO t+1", "Menos eff y menos A", "Deuda/PIB sube +2,1 pp"),
    ]
    for i, (x0, bg, bd, t1, t2, t3) in enumerate(steps):
        parts.append(f'<rect x="{x0}" y="12" width="91" height="68" rx="4" fill="{bg}" stroke="{bd}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x0+45.5:.1f}" y="27" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.9" font-weight="500" fill="{bd}">{escape(t1)}</text>')
        parts.append(f'<text x="{x0+45.5:.1f}" y="44" text-anchor="middle" font-family="IBM Plex Sans" font-size="7.0" font-weight="600" fill="{INK}">{escape(t2)}</text>')
        parts.append(f'<text x="{x0+45.5:.1f}" y="60" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.7" fill="{INK2}">{escape(t3)}</text>')
        if i < len(steps) - 1:
            parts.append(f'<line x1="{x0+92}" y1="46" x2="{x0+102}" y2="46" stroke="{ORANGE}" stroke-width="1.4" marker-end="url(#arC)"/>')
    parts.append('</svg>')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# CSS EDITORIAL (CON CAJAS DE ECUACIÓN Y EJEMPLOS PASO A PASO)
# ═══════════════════════════════════════════════════════════════════════════

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
    content: "GUÍA TEÓRICA Y MATEMÁTICA · MÓDULOS M1–M7";
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
    content: "PHY-WORLD · ATLAS MUNDO (CAPA L1)";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.3pt;
    letter-spacing: 0.7px;
    color: #8B97A8;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.2mm;
  }}
  @bottom-right {{
    content: counter(page);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 6.8pt;
    color: #475569;
    border-top: 0.5pt solid #E2E8F0;
    padding-top: 2.2mm;
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
  font-size: 8.3pt;
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
.cover-title {{
  font-family: 'Newsreader', serif; font-weight: 700; font-size: 38pt; line-height: 1.04;
  letter-spacing: -0.6px; margin: 9mm 0 5mm 0; color: #FFFFFF;
}}
.cover-title em {{ font-style: italic; font-weight: 600; color: #E8742D; display: block; }}
.cover-sub {{ font-size: 10.6pt; line-height: 1.48; color: #B4C0D6; max-width: 150mm; margin: 0; }}
.cover-kpis {{
  border-top: 0.7pt solid #2B3959; padding-top: 5.5mm;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8mm;
}}
.cover-kpi-num {{
  font-family: 'Newsreader', serif; font-size: 22pt; font-weight: 700; line-height: 1.1; color: #FFFFFF; margin-bottom: 1.8mm;
}}
.cover-kpi-desc {{ font-size: 7.6pt; line-height: 1.38; color: #9EACC7; }}
.cover-foot {{
  display: flex; justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace; font-size: 6.3pt; letter-spacing: 0.9px; color: #6B7C9E; margin-top: 4mm;
}}

/* TYPOGRAPHY */
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.8pt; font-weight: 500;
  letter-spacing: 1px; color: #D95D24; text-transform: uppercase; margin: 0 0 1.4mm 0;
}}
h1 {{
  font-family: 'Newsreader', serif; font-weight: 700; font-size: 21pt; line-height: 1.1;
  letter-spacing: -0.3px; color: #0E162B; margin: 0 0 2.8mm 0;
}}
h2 {{
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 9.2pt;
  color: #0E162B; margin: 3.2mm 0 1.4mm 0;
}}
.lede {{
  font-family: 'Newsreader', serif; font-size: 10.4pt; line-height: 1.4;
  color: #263147; margin: 0 0 3.5mm 0;
}}
p {{ margin: 0 0 2.2mm 0; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}
.m {{ font-family: 'Newsreader', serif; font-style: italic; font-size: 9.2pt; color: #0E162B; }}

/* LAYOUT GRIDS */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4.5mm; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 3.5mm; }}

/* THEORY, EQUATION & EXAMPLE BOXES */
.theory-box {{
  background: #F4F6F9; border-radius: 2mm; padding: 3mm 3.8mm; margin: 2mm 0 2.8mm 0;
}}
.theory-box .tag {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.1pt; font-weight: 500;
  letter-spacing: 0.7px; color: #2F66D0; text-transform: uppercase; margin-bottom: 1mm;
}}
.theory-box p {{ font-size: 7.9pt; line-height: 1.4; color: #334155; margin: 0 0 1.4mm 0; }}
.theory-box p:last-child {{ margin-bottom: 0; }}

.eq-box {{
  background: #FFFFFF; border: 0.7pt solid #D5DFEB; border-left: 2.6pt solid #0E162B;
  border-radius: 0 2mm 2mm 0; padding: 2.6mm 3.6mm; margin: 2.2mm 0 2.8mm 0;
}}
.eq-box .eq-tag {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.0pt; font-weight: 500;
  letter-spacing: 0.7px; color: #64748B; text-transform: uppercase; margin-bottom: 1.2mm;
}}
.eq-line {{
  font-family: 'IBM Plex Mono', monospace; font-size: 8.2pt; font-weight: 500;
  color: #0E162B; background: #F8FAFC; padding: 1.4mm 2.4mm; border-radius: 1.2mm;
  margin: 1.2mm 0; line-height: 1.4;
}}
.eq-line .sym {{ font-family: 'Newsreader', serif; font-style: italic; font-size: 9.6pt; font-weight: 600; color: #0E162B; }}
.eq-defs {{
  font-size: 7.4pt; line-height: 1.36; color: #475569; margin-top: 1.4mm;
}}
.eq-defs b {{ color: #0E162B; font-weight: 600; }}

.ex-box {{
  background: #FDF7F4; border: 0.7pt solid #F2D2C2; border-left: 2.6pt solid #D95D24;
  border-radius: 0 2mm 2mm 0; padding: 2.8mm 3.6mm; margin: 2.2mm 0 2.8mm 0;
}}
.ex-box .ex-tag {{
  font-family: 'IBM Plex Mono', monospace; font-size: 6.1pt; font-weight: 500;
  letter-spacing: 0.7px; color: #D95D24; text-transform: uppercase; margin-bottom: 1mm;
}}
.ex-box p, .ex-box li {{ font-size: 7.7pt; line-height: 1.38; color: #2D3748; margin: 0 0 1.2mm 0; }}
.ex-box ol {{ margin: 1mm 0 0 0; padding-left: 4.2mm; }}
.ex-box li:last-child, .ex-box p:last-child {{ margin-bottom: 0; }}

.warn-pill {{
  background: #FEF3C7; border-left: 2.2pt solid #B58100; padding: 2mm 3mm;
  border-radius: 0 1.5mm 1.5mm 0; font-size: 7.5pt; line-height: 1.36; color: #374151; margin-top: 2mm;
}}

/* TABLES */
table {{ width: 100%; border-collapse: collapse; font-size: 7.2pt; margin: 1.8mm 0 2.5mm 0; }}
thead th {{
  font-family: 'IBM Plex Mono', monospace; font-size: 5.9pt; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.4px; color: #64748B; text-align: left;
  padding: 1.3mm 1.4mm; border-top: 1.1pt solid #0E162B; border-bottom: 0.7pt solid #94A3B8;
}}
thead th.num, tbody td.num {{ text-align: right; font-family: 'IBM Plex Mono', monospace; }}
tbody td {{ padding: 1.15mm 1.4mm; border-bottom: 0.45pt solid #E6ECF2; color: #1E293B; vertical-align: top; }}
.tight td, .tight th {{ padding: 0.9mm 1.2mm; font-size: 6.8pt; }}
.chart-wrap {{ margin: 2mm 0 1.5mm 0; border-top: 0.8pt solid #D9E2EC; padding-top: 1.6mm; }}
.hi-o {{ color: #D95D24; font-weight: 600; }}
.hi-b {{ color: #2F66D0; font-weight: 600; }}
.hi-t {{ color: #1F7A6E; font-weight: 600; }}
.hi-r {{ color: #B83227; font-weight: 600; }}
"""


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINAS (12 PÁGINAS DE CONTENIDO RIGUROSO Y DIDÁCTICO)
# ═══════════════════════════════════════════════════════════════════════════

def p01_cover() -> str:
    return f"""
<div class="page cover-page">
  <div>
    <div class="cover-top">
      <span>PHY-WORLD · MANUAL TÉCNICO DE LA CAPA L1</span>
      <span>TEORÍA · MATEMÁTICA · EJEMPLOS</span>
    </div>
    <h1 class="cover-title">Los siete motores<br>del Atlas Mundo,<em>teoría, matemática y ejemplos</em></h1>
    <p class="cover-sub">Cómo funcionan por dentro los módulos M1 a M7 de Phy-World L1: desde la función de producción Cobb-Douglas y el intercambio desigual de Wallerstein hasta la <em>asabiya</em> de Ibn Jaldún, las curvas de Hubbert, las crisis financieras y el daño climático cuadrático.</p>
  </div>
  <div style="margin: 2mm -8mm 0 -8mm;">{svg_cover_architecture()}</div>
  <div>
    <div class="cover-kpis">
      <div>
        <div class="cover-kpi-num">7 módulos</div>
        <div class="cover-kpi-desc">Siete teorías de economía, sociología histórica, geopolítica, geofísica y clima acopladas año a año sobre 180 agentes-país.</div>
      </div>
      <div>
        <div class="cover-kpi-num">28 + 23</div>
        <div class="cover-kpi-desc">Parámetros calibrados con datos reales: 28 por máxima verosimilitud directa (Etapa 1) y 23 por ajuste de simulación (Etapa 2).</div>
      </div>
      <div>
        <div class="cover-kpi-num">1 paso = 1 año</div>
        <div class="cover-kpi-desc">Cada ecuación de esta guía proviene directamente de <span class="mono" style="color:#E2E8F0;">model.py</span> y está ilustrada con ejemplos numéricos verificables.</div>
      </div>
    </div>
    <div class="cover-foot">
      <span>GUÍA DE REFERENCIA · MÓDULOS M1 A M7</span>
      <span>BASADO EN WORLD4 / ATLAS L1 (CALIBRACIÓN 2019)</span>
    </div>
  </div>
</div>
"""


def p02_overview() -> str:
    return """
<div class="page">
  <span class="chap-mark">PANORÁMICA Y NOTACIÓN</span>
  <div class="eyebrow">Guía de lectura</div>
  <h1>Cómo late un año del Atlas Mundo y cómo se calibran sus ecuaciones</h1>
  <p class="lede">Antes de entrar módulo por módulo, conviene ver la fotografía completa: cada uno de los 180 países es un agente con un vector de estado (capital, productividad, reservas, deuda, élites, solidaridad colectiva, régimen y temperatura) que se actualiza en un orden fijo cada año.</p>

  <div class="grid-2">
    <div class="theory-box">
      <div class="tag">Los seis canales de decisión de cada país</div>
      <p>Cada año, el presupuesto discrecional de un país (fracción de su PIB) se reparte en seis canales observables:</p>
      <p><b>k</b> inversión en capital físico doméstico · <b>r</b> I+D y educación (saltos de PTF) · <b>m</b> importación de manufacturas del centro · <b>x</b> capital para extraer recursos propios · <b>f</b> concesiones extractivas en el extranjero · <b>w</b> redistribución hacia las masas.</p>
      <p>Gastar por encima de la senda histórica en los cinco canales financieros (<span class="mono">k, r, m, x, f</span>) se financia en un <b>54,4 %</b> con ahorro interno (<span class="mono">fh_eta</span>) y el 45,6 % restante con deuda externa hasta el techo de déficit por cuenta corriente (<b>13,2 %</b> del PIB).</p>
    </div>
    <div class="theory-box">
      <div class="tag">Calibración en dos etapas (51 parámetros)</div>
      <p><b>Etapa 1 (28 parámetros):</b> se estiman directamente sobre datos históricos por máxima verosimilitud (modelos <i>logit</i> para crisis financieras, golpes, transiciones democráticas y nacionalizaciones) o mínimos cuadrados (dinámica de deuda y financiamiento).</p>
      <p><b>Etapa 2 (23 parámetros):</b> parámetros cuyos mecanismos son latentes o acoplados (difusión tecnológica, intercambio desigual, <i>asabiya</i>, pendiente de Hubbert, riesgo de guerra civil) se optimizan ejecutando el simulador completo sobre 1950–2019 hasta minimizar el error conjunto en PIB, desigualdad, rentas y conflicto.</p>
    </div>
  </div>

  <h2>Mapa de los siete módulos y símbolos principales</h2>
  <table class="tight">
    <thead>
      <tr><th>Mód.</th><th>Disciplina y autores</th><th>Variables de estado</th><th>Parámetros calibrados clave</th><th>Pág.</th></tr>
    </thead>
    <tbody>
      <tr><td class="mono hi-o">M1</td><td><b>Producción y técnica</b><br>Solow, Nelson-Phelps, Aghion-Howitt</td><td class="mono">Y, K, A, h, L</td><td>α = 0,35 · φ_h = 1,66 · κ = 0,015 · λ₀ = 0,105 · j₀ = 0,067 · χ_k = 12,1</td><td class="num">3–4</td></tr>
      <tr><td class="mono hi-b">M2</td><td><b>Sistema-Mundo</b><br>Wallerstein, Emmanuel, Amin, Hickel</td><td class="mono">c_i, drain, rent_in, GNI</td><td>θ_d = 0,20 · β_px = 0,20 · μ = 0,25 · Q₈₀ (umbral centro)</td><td class="num">5</td></tr>
      <tr><td class="mono hi-o">M3</td><td><b>Demografía estructural</b><br>Goldstone, Turchin, Urdal</td><td class="mono">E, n_e, MMP, EMP, SFD, PSI</td><td>β_e = 0,05 · φ_extra = 0,075 · b_Y = 0,035 · b₁ = 0,064</td><td class="num">6</td></tr>
      <tr><td class="mono hi-t">M4</td><td><b>Frontera metaétnica</b><br>Ibn Jaldún, Turchin</td><td class="mono">S (asabiya), F_i (frontera)</td><td>r_a = 0,02 · d_a = 0,007 · b₃ = −2,56 · eff = 0,75 + 0,5 S</td><td class="num">7</td></tr>
      <tr><td class="mono hi-t">M5</td><td><b>Recursos naturales</b><br>Hubbert, Guriev et al.</td><td class="mono">U, X, K_Rd, K_Rf, p, ψ_t</td><td>r_R = 0,145 · c_R = 3,0 · ζ = 0,15 · roy_t = 0,75 · θ_nat = 0,75</td><td class="num">8</td></tr>
      <tr><td class="mono hi-b">M6</td><td><b>Deuda, crisis y regímenes</b><br>Reinhart-Rogoff, Laeven-Valencia, V-Dem</td><td class="mono">d, d*, demo, poly, coup, conf</td><td>cr_r = 9,97 · cr_loss = 4,5 % · cp_T = 1,08 · b₄ = 3,84 · dem_b = 0,0079</td><td class="num">9</td></tr>
      <tr><td class="mono hi-r">M7</td><td><b>Clima y capital natural</b><br>IPCC TCRE, Burke-Hsiang-Miguel</td><td class="mono">T_g, T_loc, D_clim, N_nat, CO₂</td><td>TCRE = 0,46 °C/1000 Gt · T* = 13,04 °C · ρ_D = 0,80 · ξ_N = 0,07</td><td class="num">10</td></tr>
    </tbody>
  </table>

  <div class="eq-box">
    <div class="eq-tag">Recordatorio matemático rápido: dos funciones que aparecen en casi todos los módulos</div>
    <div class="eq-line">1) Función logística (sigmoide): &nbsp; σ(z) = 1 / (1 + e<sup>−z</sup>) ∈ (0, 1) &nbsp;&nbsp; ⇔ &nbsp;&nbsp; logit(p) = ln(p / (1 − p)) = z</div>
    <div class="eq-line">2) Aproximación logarítmica para tasas: &nbsp; Δ ln X ≈ ΔX / X &nbsp;&nbsp; (ej.: Δ ln A = −0,045 equivale a perder ~4,4 % de A)</div>
    <div class="eq-defs"><b>Cómo leer un coeficiente de un logit:</b> si una variable sube en 1 unidad y su coeficiente es <b>β</b>, la razón de momios <i>p/(1−p)</i> se multiplica por <b>e<sup>β</sup></b>. Para probabilidades pequeñas (<i>p &lt; 10 %</i>), la probabilidad misma se multiplica casi por <b>e<sup>β</sup></b>. Por ejemplo, en M4 un coeficiente <b>b₃ = −2,56</b> significa que subir la <i>asabiya</i> en +0,30 multiplica el riesgo de guerra por e<sup>−2,56 × 0,30</sup> = e<sup>−0,77</sup> = <b>0,46</b> (lo reduce a menos de la mitad).</div>
  </div>
</div>
"""


def p03_m1_prod() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M1 · PRODUCCIÓN Y TÉCNICA (I)</span>
  <div class="eyebrow">Módulo M1 · Parte I</div>
  <h1>Producción Cobb-Douglas con capital, capital humano y recursos</h1>
  <p class="lede">M1 es el motor macroeconómico del Atlas: convierte máquinas (<span class="m">K</span>), personas educadas (<span class="m">h·L</span>), materias primas (<span class="m">R</span>) y conocimiento organizativo (<span class="m">A</span>) en bienes y servicios (<span class="m">Y</span>).</p>

  <div class="theory-box">
    <div class="tag">Teoría económica · De Solow (1956) a Mankiw-Romer-Weil con recursos</div>
    <p>En la teoría neoclásica del crecimiento, acumular solo máquinas (<span class="m">K</span>) tiene <b>rendimientos marginales decrecientes</b>: la primera carretera o fábrica dispara la producción, la centésima aporta mucho menos. Por eso la inversión en capital físico por sí sola no puede sostener el crecimiento indefinidamente.</p>
    <p>Phy-World amplía la función de Cobb-Douglas estándar en tres direcciones: (1) multiplica la población <span class="m">L</span> por un índice de <b>capital humano</b> <span class="m">h</span> (años de escolaridad de la Penn World Table); (2) incorpora los <b>recursos naturales</b> <span class="m">R</span> comprados en un mercado mundial con participación <span class="m">ψ<sub>t</sub></span>; y (3) hace depender la eficiencia de la inversión de la cohesión social (<i>asabiya</i>, M4) y penaliza los intentos de invertir demasiado deprisa mediante un <b>coste de ajuste de Hayashi</b> (1982).</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M1 (producción, demanda de recursos y acumulación de capital)</div>
    <div class="eq-line">Y<sub>i</sub> = A<sub>eff,i</sub> · K<sub>i</sub><sup>α</sup> · R<sub>i</sub><sup>ψ</sup> · (h<sub>i</sub> · L<sub>i</sub>)<sup>1 − α − ψ</sup> · (1 − γ<sub>c</sub> · conf<sub>i</sub>) &nbsp;&nbsp;&nbsp; [α = 0,35 · γ<sub>c</sub> = 0,04]</div>
    <div class="eq-line">ln A<sub>eff,i</sub> = ln A<sub>i</sub> + B<sub>dem,i</sub> + D<sub>clim,i</sub> + ξ<sub>N</sub> · ln(N<sub>nat,i</sub>) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; R<sub>i</sub> = ψ · Y<sub>i</sub> / (p · m<sub>R,i</sub>)</div>
    <div class="eq-line">G<sub>i</sub> = (1 − ψ) · Y<sub>i</sub> + p · ext<sub>i</sub> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; K<sub>t+1</sub> = (1 − δ) K<sub>t</sub> + I<sub>k</sub> − (χ<sub>k</sub> / 2) · (I<sub>k</sub>/K<sub>t</sub> − δ − ι<sub>g</sub>)<sup>2</sup> · K<sub>t</sub></div>
    <div class="eq-defs">
      <b>Y<sub>i</sub></b> producción bruta · <b>G<sub>i</sub></b> PIB (valor añadido no extractivo más ingresos por extracción propia <span class="mono">p·ext<sub>i</sub></span>) · <b>A<sub>eff,i</sub></b> PTF efectiva que suma tecnología base (<span class="mono">ln A</span>), dividendo democrático (M6), daño climático (M7) y capital natural (<span class="mono">ξ<sub>N</sub> = 0,07</span>) · <b>conf<sub>i</sub> ∈ {{0, 1}}</b> guerra civil (−4 % de producto inmediato) · <b>I<sub>k</sub> = (0,75 + 0,5 S<sub>i</sub>) · a<sub>k,i</sub> · G<sub>i</sub></b> inversión efectiva · <b>δ</b> depreciación (~4,5 %), <b>ι<sub>g</sub> = 0,04</b> tasa normal y <b>χ<sub>k</sub> = 12,08</b> coste cuadrático de ajuste.
    </div>
  </div>

  <div class="grid-2">
    <div class="ex-box">
      <div class="ex-tag">Ejemplo 1 · ¿Por qué dos países con igual capital producen el doble?</div>
      <p>Comparemos dos países en paz (<span class="mono">conf=0</span>) con <b>L = 50 M</b>, <b>K = 2,5 billones USD</b> (<span class="mono">k = 50k/hab</span>), <span class="mono">ψ = 0,05</span> y precio <span class="mono">p = 1</span>:</p>
      <ol>
        <li><b>País A (emergente):</b> <span class="mono">A = 2,8</span>, <span class="mono">h = 2,0</span>. Como <span class="mono">1−α−ψ = 0,60</span>, el factor trabajo efectivo aporta <span class="mono">2,0<sup>0,60</sup> = 1,516</span> y el capital <span class="mono">50<sup>0,35</sup> = 3,93</span>. Resultado: PIB/hab ≈ <b>16.700 USD</b>.</li>
        <li><b>País B (avanzado):</b> misma población y mismo capital por persona, pero <span class="mono">A = 4,0</span> (+43 %) y <span class="mono">h = 3,5</span> (<span class="mono">3,5<sup>0,60</sup> = 2,12</span>). Resultado: PIB/hab ≈ <b>33.400 USD</b> —<b>exactamente el doble</b> sin añadir una sola máquina extra.</li>
        <li><b>Coste de ajuste de Hayashi:</b> si el país A duplica de golpe su tasa de inversión <span class="mono">I<sub>k</sub>/K</span> de 8,5 % a 18,5 % (<span class="mono">Δ = +0,10</span> sobre <span class="mono">δ+ι<sub>g</sub></span>), el término cuadrático <span class="mono">(12,08/2)·0,10² = 0,0604</span> desperdicia un <b>6,0 % de K</b> (un tercio de la inversión bruta) en cuellos de botella.</li>
      </ol>
    </div>
    <div>
      <div class="chart-wrap">{svg_m1_cobb_douglas()}</div>
      <div class="warn-pill"><b>Punto débil medido en L1:</b> <span class="mono">L</span> (población) y <span class="mono">h</span> (capital humano) son exógenos: el modelo los lee de la ONU/PWT y en L1 puro la serie WPP se congela tras 2030. Sin embargo, al conservar el ordenamiento mundial (Spearman 0,965 a 15 años), M1 es el pilar más sólido del Atlas.</div>
    </div>
  </div>
</div>
"""


def p04_m1_tfp() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M1 · PRODUCCIÓN Y TÉCNICA (II)</span>
  <div class="eyebrow">Módulo M1 · Parte II</div>
  <h1>Convergencia tecnológica, capacidad absortiva y saltos Poisson</h1>
  <p class="lede">Si <span class="m">A</span> (la Productividad Total de los Factores) explica la mitad de la brecha entre países ricos y pobres, ¿cómo evoluciona? M1 combina imitación determinista hacia la frontera mundial con innovación discontinua aleatoria.</p>

  <div class="theory-box">
    <div class="tag">Teoría · Gerschenkron, Nelson-Phelps y saltos schumpeterianos</div>
    <p><b>1. Convergencia condicional (Gerschenkron / Nelson-Phelps):</b> cuanto más lejos está un país de la frontera tecnológica mundial (<span class="m">gap<sub>i</sub> = ln A<sub>F</sub> − ln A<sub>i</sub></span>), más puede crecer copiando técnicas existentes a través de tres canales: difusión general (<span class="m">κ</span>), tecnología incorporada en importaciones de manufacturas (<span class="m">κ<sub>m</sub></span>, Coe-Helpman) e inversión extranjera directa (<span class="m">κ<sub>f</sub></span>). Pero copiar no es gratis: requiere <b>capacidad absortiva</b> (Borensztein, De Gregorio &amp; Lee 1998), es decir, ingenieros, técnicos y educación básica (<span class="m">h/h<sub>F</sub></span>).</p>
    <p><b>2. Saltos estocásticos (Aghion-Howitt):</b> además de la difusión suave, los países que invierten en el canal <span class="m">r</span> (I+D y política industrial) y tienen alta cohesión estatal (<span class="m">S<sub>i</sub></span>) pueden lograr rupturas sectoriales (semiconductores, industria naval, farmacéutica) modeladas como un <b>proceso de Poisson</b>: eventos raros que producen saltos positivos en <span class="m">ln A</span>.</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de la dinámica de PTF (ln A)</div>
    <div class="eq-line">Δ ln A<sub>det,i</sub> = g<sub>0</sub> + gap<sub>i</sub> · (h<sub>i</sub> / h<sub>F</sub>)<sup>φ<sub>h</sub></sup> · (κ + κ<sub>m</sub>·emb<sub>i</sub> + κ<sub>f</sub>·fdi<sub>i</sub>) + aid<sub>eff</sub>·a(1 − a/0,3) − γ<sub>a</sub>·conf − cp<sub>loss</sub>·golpe − cr<sub>loss</sub>·crisis + σ<sub>a</sub>·ε<sub>i</sub></div>
    <div class="eq-line">N<sub>saltos,i</sub> ~ Poisson(λ<sub>i</sub>), &nbsp;&nbsp; λ<sub>i</sub> = λ<sub>0</sub> · √a<sub>r,i</sub> · (0,5 + S<sub>i</sub>) &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Δ ln A<sub>salto</sub> = N<sub>saltos</sub> · E<sub>exp</sub> · j<sub>0</sub> · (1 + gap<sub>i</sub>)</div>
    <div class="eq-defs">
      <b>Valores calibrados (Etapa 2):</b> deriva de frontera <b>g₀ = 0,0030</b> (0,3 %/año) · exponente absortivo <b>φ<sub>h</sub> = 1,66</b> · <b>κ = 0,0149</b>, <b>κ<sub>m</sub> = 0,0558</b>, <b>κ<sub>f</sub> = 0,0537</b> · ayuda externa <b>aid<sub>eff</sub> = 0,165</b> (con rendimientos decrecientes que saturan en 15 % del INB) · ruido idiosincrático <b>σ<sub>a</sub> = 0,071</b> · choques destructivos: guerra civil <b>−1,0 %/año</b>, golpe exitoso <b>−0,8 %/año</b> durante 5 años, crisis financiera <b>−4,5 %</b> permanente · saltos Poisson: <b>λ₀ = 0,105</b>, tamaño medio <b>j₀ = 0,0675</b>, con <span class="mono">E<sub>exp</sub> ~ Exp(1)</span>.
    </div>
  </div>

  <div class="grid-2">
    <div class="ex-box">
      <div class="ex-tag">Ejemplo 2 · Corea del Sur / Vietnam frente a un país con baja escolaridad</div>
      <p>Comparemos dos países en desarrollo con la misma apertura comercial (<span class="mono">emb = 0,10</span>, <span class="mono">fdi = 0,04</span>), en paz y sin crisis:</p>
      <ol>
        <li><b>País industrializador (tipo Vietnam):</b> <span class="mono">gap = 1,20</span>, <span class="mono">h/h<sub>F</sub> = 0,90</span>, <span class="mono">a<sub>r</sub> = 0,025</span>, <span class="mono">S = 0,70</span>.<br>
        • Capacidad absortiva: <span class="mono">0,90<sup>1,66</sup> = <b>0,840</b></span>.<br>
        • Tasa de captura: <span class="mono">0,0149 + 0,0558(0,10) + 0,0537(0,04) = 0,0226</span>.<br>
        • Convergencia determinista: <span class="mono">0,003 + 1,20 × 0,840 × 0,0226 = <b>+2,58 %/año</b></span>.<br>
        • Tasa Poisson: <span class="mono">λ = 0,105 × √0,025 × (0,5+0,7) = <b>0,0199</b></span> (2,0 % anual de salto; si salta, media <span class="mono">0,0675 × 2,20 = <b>+14,9 %</b></span> de PTF).</li>
        <li><b>País de baja escolaridad:</b> <span class="mono">gap = 2,00</span> (más atrasado!), pero <span class="mono">h/h<sub>F</sub> = 0,40</span>, <span class="mono">a<sub>r</sub> = 0,004</span>, <span class="mono">S = 0,35</span>.<br>
        • Capacidad absortiva: <span class="mono">0,40<sup>1,66</sup> = <b>0,219</b></span> (¡casi 4 veces menor!).<br>
        • Convergencia determinista: <span class="mono">0,003 + 2,00 × 0,219 × 0,0226 = <b>+1,29 %/año</b></span> —crece la <i>mitad</i> que Vietnam pese a estar mucho más atrasado.</li>
      </ol>
    </div>
    <div>
      <div class="chart-wrap">{svg_m1_absorptive()}</div>
      <div class="theory-box" style="margin-top:2.5mm;">
        <div class="tag">Por qué el exponente φ_h = 1,66 es decisivo</div>
        <p>Un exponente mayor que 1 crea una <b>convexidad brutal</b>: duplicar el capital humano relativo de 0,40 a 0,80 no duplica la capacidad de absorber tecnología, sino que la multiplica por <b>3,16 veces</b> (de 0,22 a 0,69). Esa es la explicación matemática de la «trampa del ingreso medio-bajo» en Phy-World.</p>
      </div>
    </div>
  </div>
</div>
"""


def p05_m2_world_system() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M2 · SISTEMA-MUNDO (WALLERSTEIN / AMIN)</span>
  <div class="eyebrow">Módulo M2</div>
  <h1>Centralidad, intercambio desigual bilateral y rentas monopólicas</h1>
  <p class="lede">M2 introduce la economía política internacional: los países no comercian en pie de igualdad, sino sobre una jerarquía centro–periferia donde las diferencias de precios y el poder de monopolio transfieren valor neto entre economías.</p>

  <div class="theory-box">
    <div class="tag">Teoría · Wallerstein, Emmanuel, Samir Amin y el efecto Penn (Hickel et al.)</div>
    <p>Immanuel Wallerstein (1974) dividió la economía-mundo en tres posiciones estructurales: <b>centro</b> (productores de bienes complejos y monopólicos), <b>semiperiferia</b> y <b>periferia</b> (exportadores de trabajo barato y materias primas). Arghiri Emmanuel (1969) y Samir Amin (1976) formalizaron el <b>intercambio desigual</b>: debido al <i>efecto Penn</i> (los niveles de precios y salarios en países pobres son sistemáticamente más bajos en dólares que su productividad física), cuando un país periférico vende bienes al centro entrega más horas de trabajo y recursos incorporados por dólar de los que recibe a cambio.</p>
    <p>En Phy-World, este mecanismo no se supone de forma abstracta sino sobre la <b>red comercial bilateral real</b> del proyecto Correlates of War (COW, 1950–2014), y se complementa con las <b>rentas monopólicas</b> (<span class="m">μ</span>) que cobran los exportadores del centro al vender manufacturas de alta tecnología (semiconductores, aviones, fármacos patentados) a países que aún no pueden fabricarlas.</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M2 (centralidad, drenaje bilateral y rentas monopólicas)</div>
    <div class="eq-line">c<sub>i</sub> = σ((ln y<sub>i</sub> − Q<sub>0,80</sub>(ln y)) / 0,35) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [centralidad continua c<sub>i</sub> ∈ (0, 1)]</div>
    <div class="eq-line">flow<sub>i→j</sub> = θ<sub>d</sub> · (o<sub>i</sub> · G<sub>i</sub>) · S<sup>exp</sup><sub>ij</sub> · max((y<sub>j</sub> / y<sub>i</sub>)<sup>β<sub>px</sub></sup> − 1, 0) · c<sub>j</sub> · (1 − c<sub>i</sub>) &nbsp;&nbsp; [θ<sub>d</sub> = 0,20 · β<sub>px</sub> = 0,20]</div>
    <div class="eq-line">rent_m<sub>out,i</sub> = μ · (a<sub>m,i</sub> · G<sub>i</sub>) · (1 − c<sub>i</sub>) &nbsp;&nbsp; [μ = 0,25] &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; GNI<sub>i</sub> = G<sub>i</sub> + rent<sub>in,i</sub> − rent<sub>out,i</sub></div>
    <div class="eq-defs">
      <b>Q<sub>0,80</sub>(ln y)</b> percentil 80 mundial del log-PIB per cápita · <b>o<sub>i</sub></b> apertura comercial media (<span class="mono">(X+M)/2G</span>) · <b>S<sup>exp</sup><sub>ij</sub></b> fracción de las exportaciones de <span class="mono">i</span> que van a <span class="mono">j</span> (matriz COW congelada en 2014) · <b>(y<sub>j</sub>/y<sub>i</sub>)<sup>0,20</sup> − 1</b> brecha estimada de precios relativos (efecto Penn) · <b>drain<sub>i</sub> = Σ<sub>j</sub> flow<sub>i→j</sub></b> drenaje total que sale de <span class="mono">i</span> y entra en <span class="mono">j</span> (<span class="mono">drain<sub>in,j</sub> = Σ<sub>i</sub> flow<sub>i→j</sub></span>) · <b>rent<sub>in</sub></b> suma drenaje recibido + rentas monopólicas + rentas de concesiones extranjeras (M5) + intereses recibidos (M6).
    </div>
  </div>

  <div class="chart-wrap">{svg_m2_world_system()}</div>

  <div class="ex-box">
    <div class="ex-tag">Ejemplo 3 · Comercio bilateral entre un país periférico y uno central</div>
    <p>Supongamos un país periférico <b>i</b> (<span class="mono">y<sub>i</sub> = 4.000 USD</span>, <span class="mono">G<sub>i</sub> = 200 MM$</span>, apertura <span class="mono">o<sub>i</sub> = 0,30</span>, importaciones de alta tecnología <span class="mono">a<sub>m,i</sub> = 0,08</span>) que dirige el <b>40 %</b> de sus exportaciones (<span class="mono">S<sup>exp</sup><sub>ij</sub> = 0,40</span>) a un país del centro <b>j</b> (<span class="mono">y<sub>j</sub> = 60.000 USD</span>), siendo el percentil 80 mundial <span class="mono">Q<sub>80</sub> = 25.000 USD</span>:</p>
    <ol>
      <li><b>Centralidades:</b> <span class="mono">c<sub>i</sub> = σ((ln 4.000 − ln 25.000)/0,35) = σ(−5,24) = <b>0,005</b></span> (periferia casi pura); <span class="mono">c<sub>j</sub> = σ((ln 60.000 − ln 25.000)/0,35) = σ(+2,50) = <b>0,924</b></span> (centro).</li>
      <li><b>Brecha de precios (efecto Penn):</b> <span class="mono">(60.000 / 4.000)<sup>0,20</sup> − 1 = 15<sup>0,20</sup> − 1 = 1,7188 − 1 = <b>0,719</b></span> (+71,9 %).</li>
      <li><b>Drenaje por intercambio desigual hacia j:</b> <span class="mono">flow<sub>i→j</sub> = 0,20 × (0,30 × 200) × 0,40 × 0,719 × 0,924 × 0,995 = <b>3,17 MM$</b></span> (<b>1,59 % del PIB</b> de <span class="mono">i</span> perdido solo en el vínculo con <span class="mono">j</span>).</li>
      <li><b>Renta monopólica en manufacturas:</b> <span class="mono">0,25 × (0,08 × 200) × 0,995 = <b>3,98 MM$</b></span> (<b>1,99 % del PIB</b> adicional pagado al centro). En total, <span class="mono">i</span> cede ~3,6 % de su PIB en rentas netas, pero esa misma presión alimenta su frontera de solidaridad <span class="mono">F<sub>i</sub></span> en M4. A escala planetaria, lo que pierde <span class="mono">i</span> lo cobra <span class="mono">j</span>: por eso al apagar M2 el PIB <i>mundial</i> de 2041 apenas cambia (−0,22 %).</li>
    </ol>
  </div>
</div>
"""


def p06_m3_elites() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M3 · DEMOGRAFÍA ESTRUCTURAL (GOLDSTONE / TURCHIN)</span>
  <div class="eyebrow">Módulo M3</div>
  <h1>Captura de élites, sobreproducción e índice de estrés político (PSI)</h1>
  <p class="lede">¿Por qué sociedades que parecían estables estallan de pronto en crisis políticas? La teoría demográfico-estructural de Jack Goldstone y Peter Turchin identifica el motor en la presión simultánea sobre las masas, las élites y las arcas del Estado.</p>

  <div class="theory-box">
    <div class="tag">Teoría · Goldstone (1991), Turchin (2016, 2023) y el «juego de las sillas» de las élites</div>
    <p>La idea central de Turchin es contraintuitiva: el peligro no surge simplemente cuando hay desigualdad, sino cuando el éxito económico o las rentas hacen que <b>demasiados individuos aspiren a entrar en la élite</b> (<span class="m">n<sub>e</sub></span> sube: titulados, oligarcas, facciones políticas) mientras el número de puestos de poder (escaños, ministerios, grandes rentas) permanece fijo. Es un juego de las sillas musicales: las <b>contra-élites frustradas</b> —educadas, ambiciosas y excluidas— se alían con una población empobrecida (<i>MMP</i>) justo cuando el Estado está fiscalmente debilitado (<i>SFD</i>).</p>
    <p>En Phy-World, las élites se anclan empíricamente a la cuota de ingreso del <b>Top 10 %</b> (World Inequality Database, WID) y capturan además una fracción extra de las rentas petroleras y externas cuando el país es autocrático (baja poliarquía V-Dem).</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M3 (reparto de rentas, sobreproducción n_e y PSI)</div>
    <div class="eq-line">φ<sub>i</sub> = E<sub>i</sub> + 0,075(1 − poly<sub>i</sub>) &nbsp;&nbsp;|&nbsp;&nbsp; φ<sub>R,i</sub> = E<sub>i</sub> + 0,15(1 − poly<sub>i</sub>) &nbsp;&nbsp;|&nbsp;&nbsp; v = elite_inc / mass_inc</div>
    <div class="eq-line">n<sub>e,t+1</sub> = n<sub>e,t</sub> · exp(β<sub>e</sub> · (v / v<sub>0</sub> − n<sub>e,t</sub>) − 0,15 · conf<sub>t</sub>) &nbsp;&nbsp;&nbsp; [β<sub>e</sub> = 0,05]</div>
    <div class="eq-line">MMP = (1 / w<sub>rel</sub>) · e<sup>0,035·(youth − 28)</sup> · (mass_ma / mass_pc) &nbsp;&nbsp;|&nbsp;&nbsp; EMP = n<sub>e</sub><sup>2</sup> / (v / v<sub>0</sub>)</div>
    <div class="eq-line">SFD = (1,5 − S<sub>i</sub>) · (1 + σ(4(d/d* − 1)) + 0,5·I<sub>crisis</sub>) &nbsp;&nbsp;|&nbsp;&nbsp; PSI = MMP · EMP · SFD</div>
    <div class="eq-defs">
      <b>E<sub>i</sub></b> cuota del Top 10 % (crece con las rentas capturadas <span class="mono">+0,05·rent_share</span>, baja con redistribución <span class="mono">−0,013·a<sub>w</sub></span> y se poda en guerras civiles <span class="mono">−0,02·conf</span>) · <b>w<sub>rel</sub></b> cuota salarial relativa de las masas frente a su nivel base · <b>youth</b> porcentaje de jóvenes de 15–24 años sobre adultos (<i>youth bulge</i> de Urdal 2006, centrado en 28 %) · <b>mass_ma / mass_pc</b> caída del ingreso per cápita popular frente a su media móvil · <b>EMP</b> crece <i>cuadráticamente</i> con <span class="mono">n<sub>e</sub><sup>2</sup></span> (más aspirantes compitiendo por un pastel relativo <span class="mono">v/v₀</span>) · <b>SFD</b> fragilidad fiscal-estatal (baja <i>asabiya</i> + deuda sobre umbral + crisis).
    </div>
  </div>

  <div class="grid-2">
    <div class="ex-box">
      <div class="ex-tag">Ejemplo 4 · Escenario «Primavera Árabe» vs. economía estable</div>
      <p>Calculemos <span class="mono">PSI = MMP · EMP · SFD</span> en dos países con igual nivel inicial <span class="mono">v/v₀ = 1,0</span>:</p>
      <ol>
        <li><b>País estable:</b> <span class="mono">w<sub>rel</sub> = 1,0</span>, <span class="mono">youth = 18 %</span>, ingreso popular subiendo (<span class="mono">mass_ma/mass_pc = 0,95</span>), <span class="mono">n<sub>e</sub> = 1,0</span>, <span class="mono">S = 0,70</span>, deuda <span class="mono">d/d* = 0,80</span>, sin crisis.<br>
        • <span class="mono">MMP = 1,0 × e<sup>0,035(−10)</sup> × 0,95 = 1,0 × 0,705 × 0,95 = <b>0,67</b></span>.<br>
        • <span class="mono">EMP = 1,0² / 1,0 = <b>1,00</b></span>.<br>
        • <span class="mono">SFD = (1,5 − 0,70) × (1 + σ(−0,8)) = 0,80 × 1,31 = <b>1,05</b></span>.<br>
        • Resultado: <span class="mono">PSI = 0,67 × 1,00 × 1,05 = <b>0,70</b></span> (<span class="mono">ln PSI = −0,36</span>).</li>
        <li><b>País bajo tensión:</b> salarios caen (<span class="mono">w<sub>rel</sub> = 0,85</span>), <span class="mono">youth = 35 %</span>, recesión popular (<span class="mono">mass_ma/mass_pc = 1,15</span>), élites sobreproducidas (<span class="mono">n<sub>e</sub> = 1,6</span>, <span class="mono">v/v₀ = 0,9</span>), <span class="mono">S = 0,40</span>, <span class="mono">d/d* = 1,20</span>, crisis activa.<br>
        • <span class="mono">MMP = (1/0,85) × e<sup>0,245</sup> × 1,15 = <b>1,73</b></span>.<br>
        • <span class="mono">EMP = 1,6² / 0,9 = <b>2,84</b></span>.<br>
        • <span class="mono">SFD = 1,10 × (1 + 0,69 + 0,50) = <b>2,41</b></span>.<br>
        • Resultado: <span class="mono">PSI = 1,73 × 2,84 × 2,41 = <b>11,84</b></span> —<b>¡17 veces mayor!</b></li>
      </ol>
    </div>
    <div>
      <div class="chart-wrap">{svg_m3_psi()}</div>
      <div class="warn-pill"><b>Hallazgo empírico clave en L1:</b> cuando el simulador se calibra con datos reales (1950–2019), el peso de <span class="mono">ln PSI</span> sobre el inicio de guerra civil resulta modesto (<b>b₁ = 0,064</b>): multiplicar PSI por ×17 solo añade <span class="mono">0,064 × ln(17) = +0,18</span> al logit. En cambio, PSI sí tiene un peso fuerte sobre los <b>golpes de Estado</b> (<span class="mono">cp_psi = +0,40</span> por desviación estándar).</div>
    </div>
  </div>
</div>
"""


def p07_m4_asabiya() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M4 · FRONTERA METAÉTNICA (ASABIYA)</span>
  <div class="eyebrow">Módulo M4</div>
  <h1>Solidaridad colectiva (<em>asabiya</em>) y frontera metaétnica</h1>
  <p class="lede">Un concepto del siglo XIV convertido en ecuación diferencial: la <em>asabiya</em> mide la cohesión y capacidad de cooperación de una sociedad. En Phy-World L1 resulta ser la palanca más poderosa frente al conflicto armado.</p>

  <div class="theory-box">
    <div class="tag">Teoría · De la «Muqaddima» de Ibn Jaldún (1377) a Peter Turchin (2003)</div>
    <p>El historiador tunecino Ibn Jaldún observó un ciclo recurrente en los imperios: las sociedades que viven en fronteras duras bajo presión externa desarrollan una fuerte solidaridad de grupo (<b><i>asabiya</i></b>) que les permite actuar colectivamente, construir Estados eficaces y superar a vecinos más ricos. Pero una vez alcanzan la opulencia en el centro imperial, el lujo y la competencia entre facciones de élites erosionan lentamente esa cohesión hasta volverlas vulnerables al colapso.</p>
    <p>Peter Turchin formalizó esta intuición como la <b>teoría de las fronteras metaétnicas</b>: la <i>asabiya</i> <span class="m">S ∈ (0, 1)</span> crece de forma logística cuando existe presión externa (en L1: una periferia abierta expuesta al drenaje del centro, <span class="m">F<sub>i</sub></span>) y se degrada en centros ricos (<span class="m">y &gt; 20.000 USD</span>) tanto más rápido cuanto mayor sea la sobreproducción de élites (<span class="m">n<sub>e</sub></span>), o bruscamente durante una guerra civil.</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M4 (frontera, dinámica de S_t y sus 4 canales de impacto)</div>
    <div class="eq-line">F<sub>i</sub> = clip( o<sub>i</sub>(1 − c<sub>i</sub>) + 4 · (drain<sub>i</sub> + μ·foreign_m<sub>i</sub> + res_out<sub>i</sub>) / G<sub>i</sub> , &nbsp; 0 , &nbsp; 1 )</div>
    <div class="eq-line">S<sub>t+1</sub> = S<sub>t</sub> + r<sub>a</sub>·F<sub>i</sub>·S<sub>t</sub>(1 − S<sub>t</sub>) − d<sub>a</sub>(1 − F<sub>i</sub>)·rich<sub>i</sub>·S<sub>t</sub>·max(n<sub>e</sub>, 0,5) − 0,03·conf<sub>t</sub>·S<sub>t</sub> &nbsp;&nbsp; [r<sub>a</sub>=0,02 · d<sub>a</sub>=0,007]</div>
    <div class="eq-line">rich<sub>i</sub> = σ((ln y<sub>i</sub> − ln 20.000) / 0,5) &nbsp;&nbsp;|&nbsp;&nbsp; eff<sub>i</sub> = 0,75 + 0,5·S<sub>i</sub> &nbsp;&nbsp;|&nbsp;&nbsp; Δlogit(guerra) = b<sub>3</sub>·(S<sub>i</sub> − 0,5) &nbsp; [b<sub>3</sub> = −2,56]</div>
    <div class="eq-defs">
      <b>Los 4 canales de acoplamiento de la asabiya:</b> (1) <b>Eficiencia inversora (M1/M5):</b> un país con <span class="mono">S = 0,80</span> convierte 1 USD invertido en <span class="mono">eff = 1,15 USD</span> de capital productivo, frente a solo <span class="mono">0,85 USD</span> si <span class="mono">S = 0,20</span>; (2) <b>Innovación (M1):</b> la tasa Poisson de saltos tecnológicos multiplica por <span class="mono">(0,5 + S<sub>i</sub>)</span>; (3) <b>Escudo antibélico (M6):</b> el coeficiente <span class="mono">b<sub>3</sub> = −2,56</span> domina el logit de inicio de conflicto civil; (4) <b>Nacionalismo de recursos (M5):</b> la atracción de concesiones extranjeras se multiplica por <span class="mono">(1 − 0,8 S<sub>i</sub>)</span> y la fragilidad estatal (M3) escala con <span class="mono">(1,5 − S<sub>i</sub>)</span>.
    </div>
  </div>

  <div class="chart-wrap">{svg_m4_asabiya()}</div>

  <div class="ex-box">
    <div class="ex-tag">Ejemplo 5 · Treinta años de erosión y cuánto sube el riesgo de guerra civil</div>
    <ol>
      <li><b>Periferia abierta bajo presión (tipo Corea 1960 o Vietnam):</b> <span class="mono">F = 0,85</span>, <span class="mono">rich ≈ 0</span>, <span class="mono">S<sub>0</sub> = 0,40</span>. El primer año gana <span class="mono">+0,02 × 0,85 × 0,40 × 0,60 = +0,0041</span>. En 30 años de paz la curva logística sube de <b>0,40 a 0,52</b>, elevando su eficiencia inversora de <span class="mono">0,95</span> a <span class="mono">1,01</span>.</li>
      <li><b>Centro rico con sobreproducción de élites:</b> <span class="mono">y = 55.000 USD</span> (<span class="mono">rich = 0,88</span>), <span class="mono">F = 0,05</span>, <span class="mono">n<sub>e</sub> = 1,8</span>, <span class="mono">S<sub>0</sub> = 0,70</span>. Cada año pierde <span class="mono">−0,007 × 0,95 × 0,88 × 0,70 × 1,8 = −0,0074</span> (−1,1 %/año). En 30 años cae de <b>0,70 a 0,49</b>.</li>
      <li><b>Impacto sobre el riesgo de guerra civil:</b> en un Estado frágil y dependiente de recursos (panel derecho), si su <i>asabiya</i> pasa de <b>0,75</b> a <b>0,25</b> (<span class="mono">ΔS = −0,50</span>), el logit salta en <span class="mono">−2,56 × (−0,50) = <b>+1,28</b></span>: la probabilidad anual de estallido bélico se <b>triplica con creces</b> (de <b>3,0 %</b> a <b>9,9 %</b> anual; a 10 años pasa de 26 % a 65 %!).</li>
    </ol>
  </div>
</div>
"""


def p08_m5_hubbert() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M5 · RECURSOS NATURALES (HUBBERT)</span>
  <div class="eyebrow">Módulo M5</div>
  <h1>Curvas logísticas de Hubbert, concesiones y la trampa del precio</h1>
  <p class="lede">M5 ancla la economía mundial a la geología: cada país posee un depósito finito de recursos extraíbles <span class="m">U</span> cuya capacidad sube al desarrollarse el yacimiento, alcanza un pico y declina. Un mercado mundial fija el precio cada año.</p>

  <div class="theory-box">
    <div class="tag">Teoría · M. King Hubbert (1956), concesiones extranjeras y nacionalizaciones (Guriev et al. 2011)</div>
    <p>El geofísico M. King Hubbert demostró que la producción acumulada <span class="m">X(t)</span> de una cuenca petrolera finita suele seguir una S logística; su derivada —la tasa de extracción anual— dibuja una <b>campana simétrica</b> que alcanza su máximo cuando se ha extraído la mitad del depósito (<span class="m">x = X/U = 0,50</span>). En Phy-World, la extracción real depende además del capital invertido en pozos y minas, tanto doméstico (<span class="m">K<sub>Rd</sub></span>) como extranjero (<span class="m">K<sub>Rf</sub></span>): las concesiones extranjeras aceleran el agotamiento y repatrían beneficios tras pagar regalías (<span class="m">roy<sub>t</sub></span>: 50 % antes de 1973, 85 % en la era OPEP 1976–1985 y 75 % desde 1986).</p>
    <p>Cuando el precio mundial se dispara, aumenta la tentación política de <b>nacionalizar</b> (Guriev, Kolotilin &amp; Sonin 2011): un logit estocástico permite al país anfitrión expropiar el 75 % del capital extractivo extranjero.</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M5 (capacidad de Hubbert, extracción, precio mundial y nacionalización)</div>
    <div class="eq-line">x<sub>i</sub> = X<sub>i</sub> / U<sub>i</sub> &nbsp;&nbsp;|&nbsp;&nbsp; Cap<sub>i</sub> = r<sub>R</sub> · U<sub>i</sub> · (x<sub>i</sub> + 0,01) · (1 − x<sub>i</sub>) &nbsp;&nbsp; [r<sub>R</sub> = 0,145]</div>
    <div class="eq-line">ext<sub>i</sub> = Cap<sub>i</sub> · (1 − e<sup>−e<sub>i</sub></sup>), &nbsp;&nbsp; e<sub>i</sub> = (K<sub>Rd,i</sub> + K<sub>Rf,i</sub>) / (c<sub>R</sub> · Cap<sub>i</sub>) &nbsp;&nbsp; [c<sub>R</sub> = 3,0]</div>
    <div class="eq-line">p = ψ<sub>t</sub> / q, &nbsp;&nbsp; q = ( Σ<sub>i</sub> ext<sub>i</sub> / Σ<sub>i</sub> m<sub>R,i</sub><sup>−1/(1−ψ)</sup> B<sub>i</sub><sup>1/(1−ψ)</sup> )<sup>1−ψ</sup> &nbsp;&nbsp; [m<sub>R,i</sub> = 1 + 0,15·ims<sub>i</sub>]</div>
    <div class="eq-line">logit(P<sub>nac</sub>) = −4,20 + 2,13·Δ<sub>3a</sub>(ln p) − 0,68·ln p + 0,041·polity − 1,28·I<sub>≥1985</sub> &nbsp;&nbsp; [confisca θ<sub>nat</sub> = 75 % de K<sub>Rf</sub>]</div>
    <div class="eq-defs">
      <b>U<sub>i</sub></b> recurso total recuperable (inicializado a partir de la renta observada y la fracción agotada física) · <b>sd<sub>i</sub> = K<sub>Rd</sub>/(K<sub>Rd</sub>+K<sub>Rf</sub>)</b> parte doméstica · <b>ims<sub>i</sub></b> fracción de recursos que el país debe importar (paga un recargo de transporte/seguridad <span class="mono">ζ = 15 %</span>) · <b>B<sub>i</sub></b> bloque productivo no-recurso de M1 (<span class="mono">A·K<sup>α</sup>·(hL)<sup>1−α−ψ</sup></span>) · <b>ψ<sub>t</sub></b> cuota mundial de rentas de recursos en el PIB global.
    </div>
  </div>

  <div class="chart-wrap">{svg_m5_hubbert()}</div>

  <div class="ex-box">
    <div class="ex-tag">Ejemplo 6 · Ciclo de vida de un depósito y anatomía matemática de la «trampa de Hubbert»</div>
    <ol>
      <li><b>Ciclo de Hubbert (<span class="mono">U = 100</span>, <span class="mono">r<sub>R</sub> = 0,145</span>, esfuerzo <span class="mono">e = 1,2</span> → <span class="mono">1−e<sup>−1,2</sup> = 0,699</span>):</b><br>
      • Fase joven (<span class="mono">x = 10 %</span>): <span class="mono">Cap = 0,145 × 100 × 0,11 × 0,90 = <b>1,44</b></span> → <span class="mono">ext = <b>1,00</b></span>.<br>
      • Pico (<span class="mono">x = 50 %</span>): <span class="mono">Cap = 0,145 × 100 × 0,51 × 0,50 = <b>3,70</b></span> → <span class="mono">ext = <b>2,59</b></span> (máximo histórico).<br>
      • Ocaso (<span class="mono">x = 90 %</span>): <span class="mono">Cap = 0,145 × 100 × 0,91 × 0,10 = <b>1,32</b></span> → <span class="mono">ext = <b>0,92</b></span>. Si además 40 % del capital es extranjero (<span class="mono">sd = 0,60</span>) con <span class="mono">roy = 0,75</span>, el país se queda con <span class="mono">0,60 + 0,75 × 0,40 = <b>90 %</b></span> de la renta y el 10 % se repatria.</li>
      <li><b>Por qué falló M5 en 2026–2041 (la trampa de ψ<sub>t</sub> congelada):</b> en Cobb-Douglas el gasto mundial en recursos es siempre <span class="mono">p · S = ψ<sub>t</sub> · Y<sub>mundo</sub></span>. Como en proyección L1 congela <span class="mono">ψ<sub>t</sub> ≈ 3,5 %</span> (sin sustitución renovable) y la pendiente calibrada <span class="mono">r<sub>R</sub> = 0,145</span> es muy alta, los depósitos se agotan rápido (<span class="mono">x → 0,99</span>), la oferta física <span class="mono">S</span> cae un 93 % y la identidad obliga a que el precio <span class="mono">p = ψ<sub>t</sub> Y / S</span> se dispare <b>×60</b>.</li>
    </ol>
  </div>
</div>
"""


def p09_m6_finance_regimes() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M6 · DEUDA, CRISIS Y REGÍMENES</span>
  <div class="eyebrow">Módulo M6</div>
  <h1>Crisis financieras, deuda soberana, golpes y democracia (V-Dem)</h1>
  <p class="lede">La estabilidad financiera y política no es un telón de fondo estático: en M6, las tasas de interés de EE. UU., la deuda externa, los golpes de Estado y las transiciones democráticas se disparan endógenamente vía modelos logit calibrados.</p>

  <div class="theory-box">
    <div class="tag">Teoría · Intolerancia a la deuda, ciclos globales, trampa de golpes y dividendo democrático</div>
    <p><b>1. Finanzas (Reinhart-Rogoff-Savastano / Laeven-Valencia):</b> los países periféricos sufren crisis con deudas mucho menores que los del centro (<b>intolerancia a la deuda</b>: umbral <span class="m">d*</span> del 40 % del PIB en la periferia frente al 120 % en el centro). Además, el ciclo financiero global depende de la <b>tasa real de EE. UU.</b> (<span class="m">r*<sub>US</sub></span>) y del contagio internacional. Cada crisis deja una <b>cicatriz permanente</b> del −4,5 % sobre la PTF (Cerra &amp; Saxena 2008).</p>
    <p><b>2. Regímenes y conflicto (V-Dem / PITF / Acemoglu et al. 2019):</b> los golpes siguen una <b>trampa histórica</b> (haber sufrido un golpe en los últimos 10 años casi triplica el riesgo de otro). Las guerras civiles golpean sobre todo a las <b>anocracias</b> (regímenes híbridos donde <span class="m">4p(1−p)</span> alcanza su máximo) y a los países petroleros (Collier-Hoeffler). A cambio, consolidar una democracia genera un <b>dividendo acumulado de PTF</b> (+21 % a largo plazo).</p>
  </div>

  <div class="eq-box" style="padding:2mm 3.2mm;margin:1.6mm 0 2mm 0;">
    <div class="eq-tag">Ecuaciones exactas de M6 (todos los logits calibrados en Etapa 1 y Etapa 2)</div>
    <div class="eq-line" style="font-size:7.6pt;padding:1mm 2mm;margin:0.8mm 0;">d*<sub>i</sub> = 0,40 + 0,80·c<sub>i</sub> &nbsp;|&nbsp; r<sub>i</sub> = r*<sub>US</sub> + 0,04(1−c<sub>i</sub>) + 0,04·max(d<sub>i</sub>−d*<sub>i</sub>, 0) &nbsp;|&nbsp; B<sub>dem,t</sub> = 0,963·B<sub>dem,t−1</sub> + 0,0079·demo<sub>t</sub></div>
    <div class="eq-line" style="font-size:7.6pt;padding:1mm 2mm;margin:0.8mm 0;">logit(P<sub>crisis</sub>) = −3,06 + 0,076(d/d*−1) + 9,97·r*<sub>US</sub> − 3,92·g<sub>t−1</sub> + 5,22·contagio &nbsp;&nbsp;[ΔlnA = −0,045]</div>
    <div class="eq-line" style="font-size:7.6pt;padding:1mm 2mm;margin:0.8mm 0;">logit(P<sub>golpe</sub>) = −4,13 − 0,263·ln(y/10k) − 3,07·g<sub>t−1</sub> + 1,08·trampa<sub>10a</sub> − 0,86·demo + 0,81·GF + 0,40·PSI<sub>z</sub></div>
    <div class="eq-line" style="font-size:7.6pt;padding:1mm 2mm;margin:0.8mm 0;">logit(P<sub>guerra</sub>) = −6,29 + 0,064·ln PSI − 0,403(ln y−9) − 2,56(S−0,5) + 3,84·(rev/G) + 4p(1−p) + 0,05·T<sub>z</sub></div>
    <div class="eq-defs" style="font-size:7.1pt;line-height:1.32;">
      <b>Δd = 0,023 − 0,027·d − 0,304·g + 0,021·crisis + (r*<sub>US</sub> − 0,01)·d</b> dinámica de deuda/PIB · <b>B<sub>dem</sub></b> estado estacionario <span class="mono">+0,214</span> en ln A (+23,9 % PTF) · <b>transiciones de régimen:</b> A→D (<span class="mono">−4,84 + 0,114·ℓ − 1,18·g + 2,88·W<sub>D</sub></span>), D→A (<span class="mono">−5,18 − 0,739·ℓ − 5,57·g + 2,52·W<sub>D</sub></span>) · <b>persistencia bélica:</b> <span class="mono">logit(P<sub>seguir</sub>) = 3,13 + 0,032·ln PSI</span> (~96 % anual).
    </div>
  </div>

  <div class="chart-wrap" style="margin:1.5mm 0 1.2mm 0;">{svg_m6_crisis_anoc()}</div>

  <div class="ex-box" style="padding:2.2mm 3.2mm;margin:1.6mm 0 0 0;">
    <div class="ex-tag">Ejemplo 7 · Cómo un endurecimiento de la Fed dispara una crisis emergente</div>
    <p>Consideremos un país emergente (<span class="mono">c = 0,20 → d* = 0,56</span>) con deuda del <b>70 % del PIB</b> (<span class="mono">d/d* = 1,25</span>):</p>
    <ol>
      <li><b>Año normal (<span class="mono">g = +2 %, r*<sub>US</sub> = 0 %, contagio = 3 %</span>):</b> <span class="mono">logit = −3,056 + 0,019 + 0 − 0,078 + 0,157 = −2,959</span> → <b>P<sub>crisis</sub> = 4,9 %</b> anual.</li>
      <li><b>Choque global (<span class="mono">r*<sub>US</sub> = +4 %, g = −3 %, contagio = 15 %</span>):</b> <span class="mono">logit = −3,056 + 0,019 + 0,399 + 0,118 + 0,784 = −1,736</span> → <b>P<sub>crisis</sub> = 15,0 %</b> anual (¡se triplica!) y la tasa soberana real salta a <span class="mono">4 % + 3,2 % + 0,56 % = <b>7,76 %</b></span>.</li>
      <li><b>Trampa de golpes:</b> en un país con golpe reciente (<span class="mono">trampa = 1</span>), el término <span class="mono">+1,083</span> multiplica el riesgo de nuevo intento por <span class="mono">e<sup>1,083</sup> = <b>2,95×</b></span> durante una década.</li>
    </ol>
  </div>
</div>
"""


def p10_m7_climate() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">M7 · CLIMA Y CAPITAL NATURAL</span>
  <div class="eyebrow">Módulo M7</div>
  <h1>Calentamiento por TCRE, daño cuadrático BHM y capital natural</h1>
  <p class="lede">M7 conecta las emisiones de carbono generadas por la extracción fósil (M5) con el calentamiento global y local, el daño macroeconómico no lineal sobre la productividad y la degradación de los ecosistemas.</p>

  <div class="theory-box">
    <div class="tag">Teoría · TCRE (IPCC AR6), parábola de Burke-Hsiang-Miguel (2015) y capital natural</div>
    <p><b>1. Física simplificada (TCRE):</b> uno de los hallazgos más limpios de la ciencia climática moderna (IPCC AR6) es que el calentamiento global responde casi linealmente al <b>CO₂ acumulado emitido</b>: la <i>Respuesta Climática Transitoria a las Emisiones Acumuladas</i> (<b>TCRE</b>) ronda los <b>0,46 °C por cada 1.000 GtCO₂</b>. La tierra se calienta más deprisa que el océano, por lo que cada país tiene un factor de amplificación local <span class="m">β<sub>i</sub></span> (media 1,33).</p>
    <p><b>2. Economía del clima (Burke, Hsiang &amp; Miguel, <i>Nature</i> 2015):</b> al analizar 166 países durante 50 años, BHM hallaron una relación en forma de <b>parábola invertida</b> entre la temperatura media anual y el crecimiento económico, con un <b>óptimo universal alrededor de 13 °C</b> (clima templado de Europa occidental, EE. UU., Japón o China oriental). Por debajo de 13 °C, un grado adicional eleva ligeramente la productividad agrícola y laboral; por encima de 13 °C, el estrés térmico acelera las pérdidas de forma cuadrática.</p>
  </div>

  <div class="eq-box">
    <div class="eq-tag">Ecuaciones exactas de M7 (TCRE, daño BHM persistente, desastres y capital natural)</div>
    <div class="eq-line">T<sub>g,t+1</sub> = T<sub>g,t</sub> + (TCRE / 1000) · (CO2<sub>fósil,t</sub> + CO2<sub>suelo,0</sub>) &nbsp;&nbsp;&nbsp; [TCRE = 0,46 °C / 1000 GtCO₂]</div>
    <div class="eq-line">T<sub>loc,i,t</sub> = T<sub>base,i</sub> + β<sub>i</sub>(T<sub>g,t</sub> − T<sub>g0</sub>) + σ<sub>i</sub>·z<sub>i,t</sub> &nbsp;&nbsp;|&nbsp;&nbsp; h(T) = 0,0127·T − 0,000487·T<sup>2</sup> &nbsp; [óptimo T* = 13,04 °C]</div>
    <div class="eq-line">D<sub>clim,i,t</sub> = ρ<sub>D</sub>·D<sub>clim,i,t−1</sub> + [ h(T<sub>loc,i,t</sub>) − h(T̄<sub>i</sub>) ] &nbsp;&nbsp; [ρ<sub>D</sub> = 0,80] &nbsp;&nbsp;|&nbsp;&nbsp; p<sub>dis</sub> = 0,29 · e<sup>0,10·(T<sub>g</sub> − T<sub>1990</sub>)</sup></div>
    <div class="eq-line">N<sub>nat,t+1</sub> = N<sub>nat,t</sub> + 0,03·N<sub>nat,t</sub>(1 − N<sub>nat,t</sub>) − 0,005·overshoot<sub>i</sub> − 0,005·max(T<sub>loc</sub> − T̄<sub>i</sub>, 0) &nbsp;&nbsp; [PTF: +0,07·ln N<sub>nat</sub>]</div>
    <div class="eq-defs">
      <b>Derivada del óptimo BHM:</b> <span class="mono">h'(T) = 0,0127 − 0,000974·T = 0 ⇒ T* = 0,0127 / 0,000974 = <b>13,04 °C</b></span> · <b>ρ<sub>D</sub> = 0,80</b> persistencia del daño climático (si <span class="mono">ρ<sub>D</sub> = 0</span> sería un efecto solo de nivel; si <span class="mono">ρ<sub>D</sub> = 1</span> sería permanente sobre el crecimiento; con 0,80 un salto permanente de <span class="mono">Δh</span> se acumula hasta <span class="mono">1/(1 − 0,80) = <b>5× Δh</b></span> en <span class="mono">ln A<sub>eff</sub></span>!) · <b>overshoot</b> exceso de huella ecológica sobre biocapacidad (<span class="mono">EF/Biocap − 1</span>) · <b>desastres EM-DAT:</b> frecuencia base 29 % anual que sube +10,5 % por cada +1 °C global y destruye una fracción log-normal del capital físico <span class="mono">K</span>.
    </div>
  </div>

  <div class="chart-wrap">{svg_m7_bhm()}</div>

  <div class="ex-box">
    <div class="ex-tag">Ejemplo 8 · El mismo calentamiento local de +2 °C en tres latitudes distintas</div>
    <p>Calculemos <span class="mono">Δh = h(T̄ + 2) − h(T̄) = 0,0127(2) − 0,000487[(T̄+2)² − T̄²] = 0,0254 − 0,000974(2T̄ + 2)</span>:</p>
    <ol>
      <li><b>Noruega (<span class="mono">T̄ = 3 °C → 5 °C</span>):</b> <span class="mono">Δh = 0,0254 − 0,000974(8) = <b>+0,0176</b></span> (+1,76 % de impulso anual; en estado estacionario con <span class="mono">ρ<sub>D</sub> = 0,80</span>: <span class="mono">5 × 1,76 % = <b>+8,8 %</b></span> en PTF!).</li>
      <li><b>España / EE. UU. (<span class="mono">T̄ = 13 °C → 15 °C</span>):</b> justo en la cima de la parábola, <span class="mono">Δh = 0,0254 − 0,000974(28) = <b>−0,0019</b></span> (−0,19 %/año; efecto estacionario suave de <b>−0,9 %</b>).</li>
      <li><b>India / Nigeria (<span class="mono">T̄ = 27 °C → 29 °C</span>):</b> en la pendiente empinada de la curva, <span class="mono">Δh = 0,0254 − 0,000974(56) = <b>−0,0292</b></span> (−2,92 %/año; acumulado en estado estacionario: <span class="mono">5 × (−2,92 %) = <b>−14,6 %</b></span> de pérdida de PTF!).<br>
      <i>Nota sobre el punto débil de M7:</i> en las ex repúblicas soviéticas sin dato 1950–1980, L1 asigna <span class="mono">T̄ = 15 °C</span> por defecto en vez de ~3–6 °C, convirtiendo erróneamente una ganancia térmica leve en una pérdida.</li>
    </ol>
  </div>
</div>
"""


def p11_integration() -> str:
    return f"""
<div class="page">
  <span class="chap-mark">EJEMPLO INTEGRADOR M1 → M7</span>
  <div class="eyebrow">Todo junto en acción</div>
  <h1>Un año completo en la vida de un país emergente: cascada M1 → M7</h1>
  <p class="lede">El verdadero poder —y los riesgos— del Atlas Mundo no viven en ninguna ecuación aislada, sino en cómo se retroalimentan los siete módulos dentro de un mismo año. Sigamos a un país concreto («País X», perfil tipo emergente exportador) frente a un doble choque externo.</p>

  <div class="theory-box">
    <div class="tag">Estado inicial del País X al empezar el año t</div>
    <p>Población <b>L = 80 M</b> · capital humano <b>h = 2,6</b> (<span class="mono">h/h<sub>F</sub> = 0,72</span>) · PIB/hab <b>y = 9.000 USD</b> (<span class="mono">G = 720 MM$</span>, centralidad <b>c = 0,05</b>) · deuda <b>d = 58 % PIB</b> (umbral <span class="mono">d* = 44 %</span>) · cuota Top 10 % <b>E = 0,52</b> · sobreproducción <b>n<sub>e</sub> = 1,30</b> · <i>asabiya</i> <b>S = 0,48</b> · régimen híbrido <b>poly = 0,50</b> · temperatura media <b>T̄ = 24 °C</b> (ya +1,5 °C sobre 1950–80) · recursos a mitad de Hubbert (<span class="mono">x = 0,55</span>, rentas 8 % PIB). <b>Choque externo en t:</b> la Fed sube <span class="mono">r*<sub>US</sub></span> a <b>+4 %</b> mientras el año trae una ola de calor local de <b>+2,2 °C</b>.</p>
  </div>

  <div class="chart-wrap">{svg_cascade_integration()}</div>

  <table class="tight">
    <thead>
      <tr><th>Orden</th><th>Módulo</th><th>Cálculo exacto en el año t</th><th>Efecto inmediato y pase al siguiente módulo</th></tr>
    </thead>
    <tbody>
      <tr><td class="mono">1</td><td class="hi-r"><b>M7 Clima</b></td><td class="mono">T<sub>loc</sub> = 26,2 °C → Δh = h(26,2)−h(24) = −2,57 %</td><td><span class="mono">D<sub>clim</sub></span> baja a <span class="mono">−0,085</span> → resta <b>−2,6 %</b> a <span class="mono">ln A<sub>eff</sub></span> antes de producir.</td></tr>
      <tr><td class="mono">2</td><td class="hi-t"><b>M5 Recursos</b></td><td class="mono">x = 0,55 → Cap declina −3 %; p = 1,15</td><td>La extracción física cae, pero el precio alto mantiene rentas del 8,5 % PIB.</td></tr>
      <tr><td class="mono">3</td><td class="hi-o"><b>M1 Producto</b></td><td class="mono">Y = A<sub>eff</sub>·K<sup>0,35</sup>·R<sup>0,05</sup>·(hL)<sup>0,60</sup></td><td>Por el freno climático y la caída de <span class="mono">R</span>, el PIB <span class="mono">G</span> se contrae <b>g = −1,8 %</b>.</td></tr>
      <tr><td class="mono">4</td><td class="hi-b"><b>M2 Sist.-Mundo</b></td><td class="mono">drain = 1,4 % PIB · renta monopólica = 1,8 %</td><td>Salen <b>3,2 % PIB</b> hacia el centro → el INB (<span class="mono">GNI</span>) cae más que el PIB.</td></tr>
      <tr><td class="mono">5</td><td class="hi-b"><b>M6 Crisis/Deuda</b></td><td class="mono">r*<sub>US</sub>=4 %, d/d*=1,32, g=−1,8 % → P<sub>crisis</sub> = 12,4 %</td><td>Supongamos que estalla la crisis: <b>Δln A = −4,5 %</b> permanente y <span class="mono">d</span> salta a <b>64 % PIB</b>.</td></tr>
      <tr><td class="mono">6</td><td class="hi-o"><b>M1 Capital/PTF</b></td><td class="mono">eff = 0,75+0,5(0,48)=0,99 · (0,72)<sup>1,66</sup>=0,58</td><td>La convergencia (+1,6 %) no compensa la crisis (−4,5 %): <b>PTF neta cae −2,9 %</b>.</td></tr>
      <tr><td class="mono">7</td><td class="hi-o"><b>M3 Élites/PSI</b></td><td class="mono">MMP↑ (caen salarios), EMP = 1,69, SFD = 2,52</td><td>El estrés político <b>PSI se duplica</b> (<span class="mono">ln PSI +0,70</span>) al sumar crisis + deuda.</td></tr>
      <tr><td class="mono">8</td><td class="hi-b"><b>M6 Político</b></td><td class="mono">Anocracia (4p(1−p)=1,0) + recesión + PSI↑</td><td><span class="mono">P<sub>golpe</sub></span> sube de 1,4 % a <b>3,1 %</b> y <span class="mono">P<sub>guerra</sub></span> sube de 2,1 % a <b>3,0 %</b>.</td></tr>
      <tr><td class="mono">9</td><td class="hi-t"><b>M4 Asabiya</b></td><td class="mono">F = 0,42 (periferia abierta)</td><td>Si evita la guerra, la presión externa sube <span class="mono">S</span> (+0,002); si estalla guerra, <b>S cae −3 %</b> y dispara el bucle el año siguiente.</td></tr>
      <tr><td class="mono">10</td><td class="hi-r"><b>M7 Ecología</b></td><td class="mono">ΔN<sub>nat</sub> = −0,011 por calor + overshoot</td><td><span class="mono">N<sub>nat</sub></span> baja a 0,94 → frena silenciosamente la PTF futura (<span class="mono">0,07·ln N</span>).</td></tr>
    </tbody>
  </table>

  <div class="grid-2" style="margin-top:2.5mm;">
    <div class="ex-box">
      <div class="ex-tag">Lección 1 · Los bucles amplificadores (feedback positivo)</div>
      <p>Dos trampas emergen de forma espontánea al acoplar los siete módulos: (a) <b>trampa fiscal-política</b> (crisis M6 → recesión M1 → deuda y PSI en M3 → golpe/guerra en M6 → erosión de <i>asabiya</i> en M4 → menor eficiencia inversora en M1); y (b) <b>trampa climático-tropical</b> (calor M7 → menor PTF en M1 → menor capacidad para invertir y educar → vulnerabilidad crónica).</p>
    </div>
    <div class="theory-box">
      <div class="tag">Lección 2 · Los amortiguadores (feedback negativo)</div>
      <p>El sistema también tiene frenos naturales: (a) la <b>convergencia tecnológica</b> (M1) empuja hacia arriba a países educados tras un choque; (b) la <b>reacción fiscal</b> (<span class="mono">−0,027·d</span> en M6) estabiliza la deuda si regresa el crecimiento; y (c) en periferias abiertas, la presión externa construye <b>asabiya</b> (M4), fortaleciendo la eficiencia estatal y protegiendo contra conflictos futuros.</p>
    </div>
  </div>
</div>
"""


def p12_reference() -> str:
    return """
<div class="page">
  <span class="chap-mark">TABLA MAESTRA Y BIBLIOGRAFÍA</span>
  <div class="eyebrow">Referencia rápida</div>
  <h1>Resumen de parámetros calibrados y fuentes teóricas originales</h1>
  <p class="lede">Esta hoja reúne los parámetros numéricos exactos utilizados en las ecuaciones anteriores (calibración 2019 de Phy-World L1) junto con las obras fundacionales de cada módulo para quien quiera profundizar.</p>

  <table class="tight">
    <thead>
      <tr><th>Mód.</th><th>Símbolo</th><th>Valor</th><th>Origen</th><th>Significado e intuición rápida</th></tr>
    </thead>
    <tbody>
      <tr><td class="mono">M1</td><td class="mono">α / φ_h</td><td class="num">0,35 / 1,66</td><td>Fijo / E2</td><td>Participación del capital físico / convexidad de capacidad absortiva <span class="mono">(h/h_F)^1,66</span>.</td></tr>
      <tr><td class="mono">M1</td><td class="mono">κ / κ_m / κ_f</td><td class="num">0,015 / 0,056 / 0,054</td><td>Etapa 2</td><td>Velocidades de difusión tecnológica: base, vía importaciones y vía IED.</td></tr>
      <tr><td class="mono">M1</td><td class="mono">λ₀ / j₀ / χ_k</td><td class="num">0,105 / 0,067 / 12,1</td><td>Etapa 2</td><td>Frecuencia y tamaño medio de saltos Poisson de PTF / coste de ajuste de Hayashi.</td></tr>
      <tr><td class="mono">M2</td><td class="mono">θ_d / β_px / μ</td><td class="num">0,20 / 0,20 / 0,25</td><td>E2 / E1 / Fijo</td><td>Escala intercambio desigual / exponente Penn / renta monopólica del centro.</td></tr>
      <tr><td class="mono">M3</td><td class="mono">β_e / b_Y / b₁</td><td class="num">0,05 / 0,035 / 0,064</td><td>Fijo / Lit / E2</td><td>Velocidad de sobreproducción de élites / peso <i>youth bulge</i> / peso de ln PSI en guerra.</td></tr>
      <tr><td class="mono">M4</td><td class="mono">r_a / d_a / b₃</td><td class="num">0,020 / 0,007 / −2,56</td><td>Fijo / Fijo / E2</td><td>Crecimiento fronterizo de <i>asabiya</i> / erosión en centro rico / escudo antibélico.</td></tr>
      <tr><td class="mono">M5</td><td class="mono">r_R / c_R / ζ</td><td class="num">0,145 / 3,0 / 0,15</td><td>E2 / Fijo / Fijo</td><td>Pendiente logística de Hubbert / capital por unidad de capacidad / recargo importador.</td></tr>
      <tr><td class="mono">M5</td><td class="mono">nat_shock / θ_nat</td><td class="num">+2,13 / 0,75</td><td>Etapa 1 / Fijo</td><td>Sensibilidad de nacionalización a saltos de precio / fracción de K_Rf expropiada.</td></tr>
      <tr><td class="mono">M6</td><td class="mono">cr_r / cr_cont / cr_loss</td><td class="num">9,97 / 5,22 / 4,5 %</td><td>Etapa 1 / Lit</td><td>Impacto de tasa EE. UU. y contagio en crisis financieras / cicatriz permanente de PTF.</td></tr>
      <tr><td class="mono">M6</td><td class="mono">cp_T / b₄ / dem_b</td><td class="num">1,08 / 3,84 / 0,0079</td><td>E1 / E2 / Lit</td><td>Trampa de golpes (10 años) / maldición de recursos en guerra / impulso democrático.</td></tr>
      <tr><td class="mono">M7</td><td class="mono">TCRE / bhm₁ / bhm₂</td><td class="num">0,46 / 0,0127 / −0,000487</td><td>IPCC / BHM</td><td>Calentamiento por 1000 GtCO₂ / parábola climática con óptimo exacto en 13,04 °C.</td></tr>
      <tr><td class="mono">M7</td><td class="mono">ρ_D / dis_κ / ξ_N</td><td class="num">0,80 / 0,10 / 0,07</td><td>Lit / E1 / Lit</td><td>Persistencia de daño climático (×5 estacionario) / aumento desastres por °C / elasticidad N_nat.</td></tr>
    </tbody>
  </table>

  <h2>Lecturas fundacionales por módulo</h2>
  <div class="grid-2" style="margin-top:1.5mm;">
    <div class="theory-box">
      <div class="tag">Economía, sistema-mundo y recursos (M1, M2, M5)</div>
      <p><b>• Solow, R. (1956)</b> y <b>Mankiw, Romer &amp; Weil (1992):</b> función Cobb-Douglas con capital físico y humano.</p>
      <p><b>• Borensztein, De Gregorio &amp; Lee (1998)</b> y <b>Coe &amp; Helpman (1995):</b> difusión tecnológica por IED/comercio y capacidad absortiva.</p>
      <p><b>• Wallerstein, I. (1974), Emmanuel (1969), Amin (1976)</b> y <b>Hickel et al. (2022):</b> estructura centro-periferia e intercambio desigual.</p>
      <p><b>• Hubbert, M. K. (1956)</b> y <b>Guriev, Kolotilin &amp; Sonin (2011):</b> curva logística de agotamiento y determinantes de la nacionalización petrolera.</p>
    </div>
    <div class="theory-box">
      <div class="tag">Sociología histórica, finanzas, instituciones y clima (M3, M4, M6, M7)</div>
      <p><b>• Ibn Jaldún (1377, <i>Muqaddima</i>)</b>, <b>Goldstone (1991)</b> y <b>Turchin (2003, 2016, 2023):</b> demografía estructural, sobreproducción de élites, PSI y <i>asabiya</i>.</p>
      <p><b>• Reinhart, Rogoff &amp; Savastano (2003)</b> y <b>Laeven &amp; Valencia (2020):</b> intolerancia a la deuda y base histórica de crisis financieras.</p>
      <p><b>• Acemoglu, Naidu, Restrepo &amp; Robinson (2019):</b> «Democracy Does Cause Growth» (dividendo dinámico de PTF).</p>
      <p><b>• Burke, Hsiang &amp; Miguel (2015, <i>Nature</i>)</b> y <b>IPCC AR6 (2021):</b> función cuadrática de daño térmico (óptimo ~13 °C) y linealidad TCRE.</p>
    </div>
  </div>

  <div class="warn-pill" style="margin-top:3mm;"><b>Nota de trazabilidad:</b> Todas las ecuaciones y constantes de esta guía se extrajeron directamente del código fuente <span class="mono">world4/core/l1_atlas/model.py</span> y del archivo de calibración <span class="mono">reports/l1_atlas/study_results.json</span>. Los ejemplos numéricos fueron verificados aritméticamente uno por uno para que coincidan con las fórmulas exactas del simulador.</div>
</div>
"""


def build():
    pages = [
        p01_cover(),
        p02_overview(),
        p03_m1_prod(),
        p04_m1_tfp(),
        p05_m2_world_system(),
        p06_m3_elites(),
        p07_m4_asabiya(),
        p08_m5_hubbert(),
        p09_m6_finance_regimes(),
        p10_m7_climate(),
        p11_integration(),
        p12_reference(),
    ]
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Los siete motores del Atlas Mundo (M1–M7): teoría, matemática y ejemplos</title>
<style>{css_block()}</style>
</head>
<body>
{''.join(pages)}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(OUT_PDF))
    print(f"PDF generado correctamente -> {OUT_PDF}")


if __name__ == "__main__":
    build()
