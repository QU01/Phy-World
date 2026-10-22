"""Construye ``escalas_del_futuro.pdf`` a partir de ``estudio.json`` (figuras con matplotlib, maqueta con reportlab).

    python reports/escenarios/informe_pdf.py

Las fuentes (Instrument Serif, IBM Plex; licencia OFL) se descargan a ``fonts/`` la primera vez.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FONT_URLS = {
    "InstrumentSerif-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/instrumentserif/InstrumentSerif-Regular.ttf",
    "InstrumentSerif-Italic.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/instrumentserif/InstrumentSerif-Italic.ttf",
    **{f"IBMPlexSans-{w}.ttf": f"https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-{w}.ttf"
       for w in ("Regular", "Italic", "Medium", "SemiBold", "Bold")},
    "IBMPlexMono-Regular.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-mono/fonts/complete/ttf/IBMPlexMono-Regular.ttf",
}
(HERE / "fonts").mkdir(exist_ok=True)
for name, url in FONT_URLS.items():
    if not (HERE / "fonts" / name).is_file():
        urllib.request.urlretrieve(url, HERE / "fonts" / name)

import graficos as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,  # noqa: E402
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib.fonts import addMapping  # noqa: E402

D = json.loads((HERE / "estudio.json").read_text(encoding="utf-8"))
FIG = HERE / "figuras"
FIG.mkdir(exist_ok=True)
OUT = HERE / "escalas_del_futuro.pdf"
DEST = D["destinies"]

# ── tipografía ──
F = HERE / "fonts"
pdfmetrics.registerFont(TTFont("Serif", F / "InstrumentSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Serif-It", F / "InstrumentSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans", F / "IBMPlexSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans-It", F / "IBMPlexSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Md", F / "IBMPlexSans-Medium.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Sb", F / "IBMPlexSans-SemiBold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bd", F / "IBMPlexSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F / "IBMPlexMono-Regular.ttf"))
addMapping("Sans", 0, 0, "Sans")
addMapping("Sans", 1, 0, "Sans-Sb")
addMapping("Sans", 0, 1, "Sans-It")
addMapping("Sans", 1, 1, "Sans-Sb")
addMapping("Serif", 0, 0, "Serif")
addMapping("Serif", 0, 1, "Serif-It")
addMapping("Serif", 1, 0, "Serif")
addMapping("Serif", 1, 1, "Serif-It")

INK, INK2, INK3 = colors.HexColor(G.INK), colors.HexColor(G.INK2), colors.HexColor(G.INK3)
RULE, RULE2 = colors.HexColor(G.RULE), colors.HexColor(G.RULE2)
ACC, ACC_SOFT = colors.HexColor(G.ACCENT), colors.HexColor(G.ACCENT_SOFT)
PAPER = colors.HexColor("#f3f5f4")

S = {
    "h1": ParagraphStyle("h1", fontName="Serif", fontSize=30, leading=32, textColor=INK, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Serif", fontSize=19, leading=22, textColor=INK, spaceBefore=4, spaceAfter=6),
    "h2b": ParagraphStyle("h2b", fontName="Serif", fontSize=16, leading=19, textColor=INK, spaceBefore=2, spaceAfter=4,
                          keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="Sans-Sb", fontSize=10.2, leading=13.5, textColor=INK, spaceBefore=8, spaceAfter=3,
                         keepWithNext=1),
    "eyebrow": ParagraphStyle("eb", fontName="Mono", fontSize=7.4, leading=10, textColor=ACC, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9, leading=13.6, textColor=INK2, spaceAfter=5,
                           alignment=TA_LEFT),
    "lede": ParagraphStyle("lede", fontName="Sans", fontSize=10.6, leading=15.6, textColor=INK2, spaceAfter=8),
    "bullet": ParagraphStyle("bul", fontName="Sans", fontSize=9, leading=13.4, textColor=INK2, leftIndent=11,
                             bulletIndent=0, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=7.4, leading=10, textColor=INK3, spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.6, leading=9.8, textColor=INK2),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Sb", fontSize=7.6, leading=9.8, textColor=INK),
    "call": ParagraphStyle("call", fontName="Sans", fontSize=9, leading=13.4, textColor=INK),
    "num": ParagraphStyle("num", fontName="Serif", fontSize=30, leading=30, textColor=INK),
    "numl": ParagraphStyle("numl", fontName="Sans", fontSize=7.6, leading=10, textColor=INK2),
}

W_PAGE, H_PAGE = A4
MARGIN = 18 * mm
TEXT_W = W_PAGE - 2 * MARGIN


# ── utilidades de datos ──
def pct(v):
    if v <= 0:
        return "0 %"
    if v < 0.005:
        return "<1 %"
    return f"{round(v * 100)} %"


def num(v, d=1):
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def med(r, k):
    return r[k]["q"][1]


def P(r, j):
    return r["p_destiny"][j]


def lvl_label(lv, v):
    if lv in ("poll", "res", "agro"):
        return "0 (nada)" if v == 0 else f"{v * 100:g} %/año"
    if lv == "soil":
        return "Sí" if v else "No"
    if lv == "fam":
        return "Sin política" if not v else f"{num(v, 2).rstrip('0').rstrip(',')} hijos"
    if lv == "suff":
        return "Sin política" if not v else f"{v:g}"
    return str(v)


def dose(lv, ctx, geo):
    return D["dose"][lv][ctx][geo]


# ── figuras ──
def fig_dose(lv):
    rows_s, rows_p = dose(lv, "solo", "abund"), dose(lv, "paquete", "abund")
    labels = [lvl_label(lv, r["level"]) for r in rows_s]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.28 * len(labels) + 1.1), sharey=True)
    G.destiny_bars(axes[0], labels, [r["p_destiny"] for r in rows_s], "Sola (el resto apagado)")
    G.destiny_bars(axes[1], labels, [r["p_destiny"] for r in rows_p], "Dentro del paquete de referencia")
    G.destiny_legend(fig, y=-0.12 if len(labels) > 3 else -0.3)
    fig.tight_layout(w_pad=1.5)
    return G.save(fig, FIG / f"dose_{lv}.png")


METRICS = [("hwi_mean", "Bienestar medio 2027–2100 (HWI)", 2), ("le_min", "Esperanza de vida mínima (años)", 0),
           ("le_2100", "Esperanza de vida en 2100 (años)", 0), ("pop_2100", "Población 2100 (miles de millones)", 1),
           ("hef_2100", "Huella ecológica 2100 (planetas)", 2),
           ("io_2027_2060", "Producción industrial 2027–2060 vs inercia", 2)]


def fig_metrics(lv):
    levels = [r["level"] for r in dose(lv, "solo", "abund")]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1))
    for ax, (k, title, _) in zip(axes.flat, METRICS):
        for ctx, geo, color, lab, band in (("paquete", "abund", G.ACCENT, "En paquete · recursos abundantes", True),
                                           ("solo", "abund", G.INK, "Sola · recursos abundantes", False),
                                           ("paquete", "base", G.NEUTRAL, "En paquete · recursos base", False)):
            rows = dose(lv, ctx, geo)
            if k not in rows[0]:
                continue
            q = np.array([r[k]["q"] for r in rows])
            if band:
                G.band_line(ax, x, q, color, lab)
            else:
                ax.plot(x, q[:, 1], color=color, lw=1.5, marker="o", ms=2.8, mec="white", mew=0.5, label=lab)
        ax.set_title(title, fontsize=7.6)
        ax.set_xticks(x, [lvl_label(lv, v).replace(" %/año", "%").replace(" (nada)", "").replace("Sin política", "—")
                          .replace(" hijos", "") for v in levels], fontsize=6.6)
        if k == "io_2027_2060":
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
            ax.axhline(0, color=G.INK3, lw=0.8)
        if k == "hef_2100":
            ax.axhline(1, color=G.INK3, lw=0.8, ls=(0, (3, 2)))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0.04, 1, 1))
    return G.save(fig, FIG / f"metrics_{lv}.png")


TRAJ_VARS = {"poll": ["ppolx", "le", "hef", "pop"], "res": ["nrfr", "iopc", "le", "pop"],
             "agro": ["fpc", "pop", "le", "hef"], "fam": ["pop", "le", "iopc", "fpc"],
             "suff": ["iopc", "ppolx", "le", "hef"], "soil": ["fpc", "le", "pop", "hef"]}


def fig_traj(lv, ctx="paquete", geo="abund"):
    rows = dose(lv, ctx, geo)
    n = len(rows)
    ramp = [G.NEUTRAL] + G.RAMP[-(n - 1):] if n > 2 else [G.NEUTRAL, G.ACCENT]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95))
    for ax, v in zip(axes, TRAJ_VARS[lv]):
        for r, c in zip(rows, ramp):
            t = r["traj"]
            ax.plot(t["years"], t[v][1], color=c, lw=1.3 if c != G.NEUTRAL else 1.6,
                    label=lvl_label(lv, r["level"]))
        ax.axvline(2026, color=G.INK3, lw=0.7)
        ax.set_xlim(1950, 2100)
        ax.set_xticks([1950, 2000, 2050, 2100])
        ax.set_title(D["vars"][v]["label"], fontsize=7.4)
        ax.set_ylim(bottom=0)
        if v == "hef":
            ax.axhline(1, color=G.INK3, lw=0.7, ls=(0, (3, 2)))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(7, n), bbox_to_anchor=(0.5, -0.1), fontsize=6.8,
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
    return G.save(fig, FIG / f"traj_{lv}_{ctx}_{geo}.png")


def fig_timing():
    T = D["timing"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    yl = [f"{s:g}×" for s in T["scales"]]
    xl = [str(y) for y in T["years"]]
    grid = T["grid"]["abund"]
    G.heat(axes[0], [[r["p_destiny"][4] for r in row] for row in grid], xl, yl, G.BLUES, 0, 0.7,
           title="Transición sostenible", ylabel="Intensidad del paquete")
    G.heat(axes[1], [[r["p_destiny"][0] for r in row] for row in grid], xl, yl, G.REDS, 0, 1, title="Colapso")
    G.heat(axes[2], [[r["p_destiny"][1] for r in row] for row in grid], xl, yl, G.REDS, 0, 1,
           title="Crisis humanitaria")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=6.4)
        ax.set_xlabel("Año de arranque", fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(w_pad=0.6)
    return G.save(fig, FIG / "timing.png")


def fig_timing_hwi():
    T = D["timing"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for ax, geo, title in ((axes[0], "abund", "Recursos abundantes"), (axes[1], "base", "Recursos base")):
        for s, row, c in zip(T["scales"], T["grid"][geo], [G.NEUTRAL] + G.RAMP[1:]):
            ax.plot(T["years"], [r["hwi_mean"]["mean"] for r in row], color=c, lw=1.5, marker="o", ms=2.6,
                    mec="white", mew=0.5, label=f"{s:g}×")
        ax.set_title(f"Bienestar medio 2027–2100 · {title}", fontsize=7.6)
        ax.set_xlabel("Año de arranque", fontsize=7)
    axes[0].legend(ncol=4, fontsize=6.4, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    return G.save(fig, FIG / "timing_hwi.png")


def fig_geology():
    Gd = D["geology"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    xs = Gd["mults"]
    for s, row, c in zip(Gd["scales"], Gd["grid"], [G.NEUTRAL, G.RAMP[2], G.RAMP[4], G.RAMP[6]]):
        axes[0].plot(xs, [r["p_destiny"][0] for r in row], color=c, marker="o", ms=2.6, mec="white", mew=0.5,
                     label=f"paquete {s:g}×")
        axes[1].plot(xs, [r["p_destiny"][4] for r in row], color=c, marker="o", ms=2.6, mec="white", mew=0.5)
        axes[2].plot(xs, [r["le_2100"]["q"][1] for r in row], color=c, marker="o", ms=2.6, mec="white", mew=0.5)
    for ax, t in zip(axes, ("Probabilidad de colapso", "Transición sostenible", "Esperanza de vida 2100 (mediana)")):
        ax.set_title(t, fontsize=7.6)
        ax.set_xlabel("Recurso inicial (× el de World3)", fontsize=7)
        ax.set_xticks(xs, [f"{m:g}" for m in xs], fontsize=6.4)
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0].legend(fontsize=6.4)
    fig.tight_layout(w_pad=1.0)
    return G.save(fig, FIG / "geology.png")


def fig_interactions():
    I = D["interactions"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    rl = [f"{r * 100:g}%" for r in I["rates"]]
    G.heat(axes[0], [[r["p_destiny"][4] for r in row] for row in I["tech"]], rl, rl, G.BLUES, 0, 0.9,
           title="Transición sostenible: agro × contaminación", xlabel="Tecnología de contaminación (%/año)",
           ylabel="Agro-tecnología (%/año)")
    fl = ["—" if not f else f"{f:g}" for f in I["fams"]]
    sl = ["—" if not s else f"{s:g}" for s in I["suffs"]]
    G.heat(axes[1], [[r["hwi_mean"]["mean"] for r in row] for row in I["social"]], sl, fl, G.GREENS, 0.5, 0.73,
           fmt="{:.2f}", title="Bienestar medio: familia × suficiencia", xlabel="Producción industrial p. c. deseada",
           ylabel="Hijos deseados por familia")
    fig.tight_layout(w_pad=2.0)
    return G.save(fig, FIG / "interactions.png")


def fig_interactions_soc():
    I = D["interactions"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fl = ["—" if not f else f"{f:g}" for f in I["fams"]]
    sl = ["—" if not s else f"{s:g}" for s in I["suffs"]]
    G.heat(axes[0], [[r["p_destiny"][4] for r in row] for row in I["social"]], sl, fl, G.BLUES, 0, 0.7,
           title="Transición sostenible: familia × suficiencia", xlabel="Producción industrial p. c. deseada",
           ylabel="Hijos deseados por familia")
    G.heat(axes[1], [[r["iopc_2100"]["q"][1] for r in row] for row in I["social"]], sl, fl, G.GREENS, 100, 360,
           fmt="{:.0f}", title="Producción industrial p. c. en 2100 (mediana)", xlabel="Producción industrial p. c. deseada")
    axes[1].set_yticklabels([])
    fig.tight_layout(w_pad=2.0)
    return G.save(fig, FIG / "interactions_soc.png")


SC_COLORS = {"inercia": G.NEUTRAL, "timido": G.RAMP[2], "moderado": G.RAMP[4], "ltg": G.ACCENT, "ambicioso": "#0d366b",
             "tecno": "#e87ba4", "sobrio": "#eda100", "tardio": "#6250d6", "verde_sin_comida": "#eb6834"}


def fig_scen_overview():
    sc = D["scenarios"]
    names = [s["name"] for s in sc]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    G.destiny_bars(axes[0], names, [s["geo"]["abund"]["p_destiny"] for s in sc], "Recursos abundantes")
    G.destiny_bars(axes[1], names, [s["geo"]["base"]["p_destiny"] for s in sc], "Recursos base")
    G.destiny_legend(fig, y=-0.08)
    fig.tight_layout(w_pad=1.5)
    return G.save(fig, FIG / "scen_overview.png")


def fig_scen_profile(s, geo="abund"):
    vars_ = ["pop", "le", "hwi", "hef", "iopc", "ppolx"]
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.45))
    bau = D["bau"][geo]["traj"]
    t = s["geo"][geo]["traj"]
    for ax, v in zip(axes, vars_):
        ax.plot(bau["years"], bau[v][1], color=G.NEUTRAL, lw=1.1)
        q = np.array(t[v]).T
        ax.fill_between(t["years"], q[:, 0], q[:, 2], color=G.ACCENT, alpha=0.18, lw=0)
        ax.plot(t["years"], q[:, 1], color=G.ACCENT, lw=1.5)
        ax.axvline(s["year"], color=G.ACCENT, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(D["vars"][v]["label"].replace("Producción industrial per cápita", "Industria p. c.")
                     .replace("Contaminación persistente", "Contaminación").replace("Índice de bienestar humano", "Bienestar (HWI)"),
                     fontsize=6.6)
        ax.set_xlim(1950, 2100)
        ax.set_xticks([1950, 2025, 2100], ["1950", "2025", "2100"], fontsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.set_ylim(bottom=0)
        if v == "hef":
            ax.axhline(1, color=G.INK3, lw=0.6, ls=(0, (3, 2)))
    fig.tight_layout(w_pad=0.4)
    return G.save(fig, FIG / f"scen_{s['key']}_{geo}.png")


def fig_scorecard():
    sc = D["scenarios"]
    keys = [("hwi_mean", "Bienestar medio"), ("le_min", "Esperanza de vida mínima"), ("pop_2100", "Población 2100 (mm)"),
            ("hef_2100", "Huella 2100"), ("ppolx_peak", "Pico de contaminación (×1970)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(sc))[::-1]
    for ax, (k, t) in zip(axes, keys):
        for yi, s in zip(y, sc):
            q = s["geo"]["abund"][k]["q"]
            c = SC_COLORS[s["key"]]
            ax.plot([q[0], q[2]], [yi, yi], color=c, lw=2.2, alpha=0.35, solid_capstyle="round")
            ax.plot(q[1], yi, "o", color=c, ms=4.5, mec="white", mew=0.7)
        ax.set_title(t, fontsize=7.2)
        ax.grid(axis="y", visible=False)
        if k == "hef_2100":
            ax.axvline(1, color=G.INK3, lw=0.7, ls=(0, (3, 2)))
    axes[0].set_yticks(y, [s["name"] for s in sc], fontsize=7)
    fig.tight_layout(w_pad=0.6)
    return G.save(fig, FIG / "scorecard.png")


# ── piezas de maqueta ──
def img(path, width=TEXT_W):
    from PIL import Image as PImage
    w, h = PImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def para(t, st="body"):
    return Paragraph(t, S[st])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=False, align_right_from=1, font=7.6):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) if not isinstance(c, Paragraph) else c
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK3),
          ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE2), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        st.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]))
    t.setStyle(TableStyle(st))
    return t


def callout(text, title=None):
    inner = []
    if title:
        inner.append(Paragraph(title, S["h3"]))
    inner.append(Paragraph(text, S["call"]))
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACC_SOFT), ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def kpis(items):
    cells = [[Paragraph(v, S["num"]), Spacer(1, 4), Paragraph(lab, S["numl"])] for v, lab in items]
    t = Table([cells], colWidths=[TEXT_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.6, RULE),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9)]))
    return t


def dest_row(p):
    return " · ".join(f"{DEST[j].split(' ')[0].lower()} {pct(v)}" for j, v in enumerate(p) if v >= 0.005)


# ── página ──
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=20 * mm,
                         bottomMargin=17 * mm, title="Escalas del futuro", author="QUASAR Phy-World",
                         subject="Estudio de escenarios por escala de implementación (World3-03)")
        body = Frame(MARGIN, 17 * mm, TEXT_W, H_PAGE - 37 * mm, id="body", leftPadding=0, rightPadding=0,
                     topPadding=0, bottomPadding=0)
        cover = Frame(MARGIN, 20 * mm, TEXT_W, H_PAGE - 40 * mm, id="cover", leftPadding=0, rightPadding=0)
        self.addPageTemplates([PageTemplate("cover", [cover], onPage=self.draw_cover),
                               PageTemplate("body", [body], onPageEnd=self.draw_body)])
        self.section = ""

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h2":
            self.section = f.getPlainText()
            key = f"s{self.seq.nextf('sec')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(self.section, key, level=0)

    def draw_body(self, c, doc):
        c.saveState()
        c.setFont("Mono", 6.8)
        c.setFillColor(INK3)
        c.drawString(MARGIN, H_PAGE - 12 * mm, "ESCALAS DEL FUTURO · PHY-WORLD L0 (WORLD3-03)")
        c.drawRightString(W_PAGE - MARGIN, H_PAGE - 12 * mm, self.section.upper()[:60])
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(MARGIN, H_PAGE - 13.6 * mm, W_PAGE - MARGIN, H_PAGE - 13.6 * mm)
        c.drawString(MARGIN, 10 * mm, "Modos de comportamiento del modelo, no pronósticos")
        c.drawRightString(W_PAGE - MARGIN, 10 * mm, f"{doc.page}")
        c.restoreState()

    def draw_cover(self, c, doc):
        c.saveState()
        c.setFillColor(colors.HexColor("#0f1c19"))
        c.rect(0, 0, W_PAGE, H_PAGE, stroke=0, fill=1)
        # abanico de trayectorias: población mediana de los nueve escenarios (recursos abundantes)
        x0, x1, y0, y1 = MARGIN, W_PAGE - MARGIN, 70 * mm, 150 * mm
        c.setStrokeColor(colors.HexColor("#2a3532"))
        c.setLineWidth(0.4)
        for k in range(5):
            yy = y0 + (y1 - y0) * k / 4
            c.line(x0, yy, x1, yy)
        for s in D["scenarios"]:
            t = s["geo"]["abund"]["traj"]
            ys = np.array(t["pop"][1])
            xs = np.array(t["years"])
            col = colors.HexColor(SC_COLORS[s["key"]] if s["key"] != "ltg" else "#46c49c")
            c.setStrokeColor(col)
            c.setLineWidth(1.6 if s["key"] in ("ltg", "inercia") else 0.9)
            p = c.beginPath()
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                X = x0 + (xx - 1950) / 150 * (x1 - x0)
                Y = y0 + yy / 10 * (y1 - y0)
                p.moveTo(X, Y) if i == 0 else p.lineTo(X, Y)
            c.drawPath(p, stroke=1, fill=0)
        c.setFillColor(colors.HexColor("#8a9692"))
        c.setFont("Mono", 6.6)
        for yr in (1950, 2000, 2026, 2050, 2100):
            c.drawString(x0 + (yr - 1950) / 150 * (x1 - x0) - 6, y0 - 9, str(yr))
        c.drawString(x0, y1 + 6, "POBLACIÓN MUNDIAL (MILES DE MILLONES), MEDIANA DE 500 MUNDOS · NUEVE ESCENARIOS")
        c.setFillColor(colors.HexColor("#46c49c"))
        c.setFont("Mono", 8)
        c.drawString(MARGIN, H_PAGE - 30 * mm, "QUASAR PHY-WORLD · ESTUDIO DE ESCENARIOS · OCTUBRE 2026")
        c.setFillColor(colors.HexColor("#e5ebe8"))
        c.setFont("Serif", 58)
        c.drawString(MARGIN, H_PAGE - 62 * mm, "Escalas del")
        c.setFont("Serif-It", 58)
        c.setFillColor(colors.HexColor("#46c49c"))
        c.drawString(MARGIN, H_PAGE - 82 * mm, "futuro")
        c.setFillColor(colors.HexColor("#b7c2be"))
        c.setFont("Sans", 11.5)
        lines = ["Cada palanca de política, modelada a siete intensidades distintas,",
                 "sola y en combinación, en dos geologías y a lo largo de 2027–2060;",
                 "por qué aparece el valle de mitad de siglo, y cómo implementar y medir cada medida.",
                 "Núcleo L0 de Phy-World (World3-03) · 500 mundos calibrados con la historia."]
        for i, l in enumerate(lines):
            c.drawString(MARGIN, H_PAGE - 98 * mm - i * 16, l)
        c.setFont("Sans", 7.6)
        c.setFillColor(colors.HexColor("#8a9692"))
        c.drawString(MARGIN, 22 * mm, "Modos de comportamiento del modelo World3-03 (Limits to Growth, 2004), no pronósticos.")
        c.drawString(MARGIN, 17 * mm, f"Generado {D['meta']['generated']} · semilla {D['meta']['seed']} · "
                                      f"{D['meta']['members']} mundos · reports/escenarios/estudio.py")
        c.restoreState()


# ── contenido ──
def build():
    print("figuras...")
    figs = {lv: (fig_dose(lv), fig_metrics(lv), fig_traj(lv)) for lv in ("poll", "res", "agro", "fam", "suff")}
    figs["soil"] = (fig_dose("soil"), None, fig_traj("soil"))
    f_timing, f_timing_hwi, f_geo = fig_timing(), fig_timing_hwi(), fig_geology()
    f_inter, f_inter_soc = fig_interactions(), fig_interactions_soc()
    f_over, f_score = fig_scen_overview(), fig_scorecard()
    prof = {s["key"]: fig_scen_profile(s) for s in D["scenarios"]}

    sc = {s["key"]: s for s in D["scenarios"]}
    A = lambda k: sc[k]["geo"]["abund"]  # noqa: E731
    B = lambda k: sc[k]["geo"]["base"]  # noqa: E731
    bauA, bauB = D["bau"]["abund"], D["bau"]["base"]
    story = [NextPageTemplate("body"), PageBreak()]

    # ── resumen ──
    story += [para("Resumen", "eyebrow"), para("Lo esencial en una página", "h2"),
              para("El informe anterior trataba cada política como un interruptor. Este estudio las gradúa: cada palanca "
                   "se modela a siete intensidades, sola y dentro de un paquete, con recursos base y abundantes, y con "
                   "arranques entre 2027 y 2060. Son 380 configuraciones, cada una sobre los mismos 500 mundos cuya "
                   "historia reproduce la población real.", "lede"),
              kpis([(pct(P(A("ltg"), 4)), "transición sostenible con el paquete de referencia (escala 1×) y recursos abundantes"),
                    (pct(D["timing"]["grid"]["abund"][5][0]["p_destiny"][4]), "con el paquete al 125 %: la escala óptima del estudio"),
                    (num(med(A("ltg"), "le_min"), 0), "años: esperanza de vida mínima mediana con el paquete 1×, antes de recuperarse"),
                    ("95–100 %", "colapso con recursos base en cualquier escala y año de arranque")]),
              Spacer(1, 10)]
    story += bullets([
        "<b>La escala importa más que el sí o el no.</b> Todas las palancas al 25 % (mejoras de 1 %/año) no cambian casi "
        f"nada: colapso {pct(P(A('timido'), 0))}. Al 50 %, el colapso baja a {pct(P(A('moderado'), 0))}. Al 100 % desaparece y "
        f"la transición sostenible llega a {pct(P(A('ltg'), 4))}. El umbral está entre el 50 % y el 75 % de la intensidad de "
        "<i>Limits to Growth</i>.",
        "<b>Hay un óptimo, no un «cuanto más, mejor».</b> El paquete al 125 % da la mayor probabilidad de transición sostenible "
        f"({pct(D['timing']['grid']['abund'][5][0]['p_destiny'][4])}). Al 150 % baja a {pct(D['timing']['grid']['abund'][6][0]['p_destiny'][4])} "
        "porque una fecundidad de 1,5 hijos produce una contracción demográfica de más del 20 %.",
        "<b>La agro-tecnología sola tiene una dosis peligrosa.</b> Sin el resto del paquete, mejorar el rendimiento un 3 %/año "
        f"deja el colapso en {pct(P(dose('agro', 'solo', 'abund')[3], 0))}, pero al 6 %/año sube a "
        f"{pct(P(dose('agro', 'solo', 'abund')[5], 0))}: más comida sostiene más población y más industria, y la contaminación "
        "llega después. Dentro del paquete, en cambio, la palanca es la más rentable de todas.",
        "<b>La tecnología limpia tiene rendimientos decrecientes.</b> En el paquete, pasar de 0 a 3 %/año sube la transición "
        f"sostenible de {pct(P(dose('poll', 'paquete', 'abund')[0], 4))} a {pct(P(dose('poll', 'paquete', 'abund')[3], 4))}; "
        f"pasar de 4 a 8 %/año sólo añade {round((P(dose('poll', 'paquete', 'abund')[6], 4) - P(dose('poll', 'paquete', 'abund')[4], 4)) * 100)} puntos.",
        "<b>Una crisis de transición es casi inevitable desde 2027.</b> Incluso con el paquete completo, en dos de cada tres "
        "mundos la esperanza de vida baja de 55 años antes de recuperarse. Sólo el paquete ambicioso lleva la mediana del valle "
        f"por encima de 55 ({num(med(A('ambicioso'), 'le_min'), 0)} años).",
        "<b>Esperar cuesta más que quedarse corto.</b> El paquete 1× arrancado en 2045 (sostenible "
        f"{pct(P(A('tardio'), 4))}) rinde menos que el paquete al 75 % arrancado en 2027 "
        f"({pct(D['timing']['grid']['abund'][3][0]['p_destiny'][4])}).",
        f"<b>El valle de 2040–2060 tiene causa.</b> Con recursos abundantes, el {round(vcase('ltg_abund')['share']['comida'] * 100)} % de la "
        "caída de la esperanza de vida es comida: la contaminación de hoy llega a los suelos dos décadas después. Los retrasos de "
        "≈ 20 años del modelo y una tecnología que reacciona en vez de anticipar fijan la fecha (capítulo «Anatomía del valle»).",
        "<b>La geología fija el techo.</b> Con los recursos de World3, el colapso alcanza al menos al 95 % de los mundos en cualquier escala y fecha; con 1,5 veces "
        f"esos recursos y el paquete 1×, la transición sostenible ya es posible ({pct(D['geology']['grid'][2][2]['p_destiny'][4])}).",
    ])
    story.append(PageBreak())

    # ── cómo leer ──
    story += [para("Método", "eyebrow"), para("Cómo leer este estudio", "h2"),
              para("Todas las cifras salen del núcleo L0 de Phy-World: World3-03, el modelo de <i>Limits to Growth: The 30-Year "
                   "Update</i> (2004), reimplementado en NumPy y verificado contra Vensim con un error máximo de 6,9·10<super>-6</super>. "
                   "Partimos de 6.000 combinaciones de seis parámetros inciertos y conservamos las 500 primeras que reproducen la "
                   "población mundial de 2000 y 2020 (UN WPP 2024) con menos de 6 % de error. Cada configuración se corre sobre esos "
                   "mismos 500 mundos, así que las diferencias entre escenarios vienen de las políticas y no del azar."),
              para("Cinco destinos", "h3"),
              para("Cada mundo termina en una clase. Respecto al informe del árbol de decisión, la clase «crisis y recuperación» se "
                   "separa en dos, porque mezclaba sufrimiento con demografía: una población que se reduce por baja fecundidad, con "
                   "esperanza de vida alta, no es una crisis."),
              table([["Destino", "Regla", "Qué significa"],
                     ["Colapso", "Esperanza de vida en 2100 menor de 55 años", "La sociedad pierde su base material y no se recupera."],
                     ["Crisis humanitaria y recuperación", "La esperanza de vida baja de 55 en algún año, pero en 2100 supera 55",
                      "Hambre, contaminación o escasez temporal: un valle profundo y una salida."],
                     ["Contracción ordenada", "Sin crisis; la población cae más de 20 % desde su pico",
                      "La población se reduce por decisión (baja fecundidad), sin perder salud."],
                     ["Meseta en sobregiro", "Sin crisis ni contracción; huella ecológica 2100 mayor de 1,1",
                      "Se vive bien, pero por encima de la capacidad de la Tierra."],
                     ["Transición sostenible", "El resto", "Sin crisis y dentro de los límites en 2100."]],
                    [38 * mm, 62 * mm, TEXT_W - 100 * mm], zebra=True),
              Spacer(1, 6),
              para("Las palancas y sus escalas", "h3"),
              para("Cada palanca actúa sobre un parámetro de World3-03 desde el año de política (2027 salvo que se diga otra cosa). "
                   "La escala de referencia, 1×, es el escenario 9 de <i>Limits to Growth</i> adelantado a hoy."),
              table([["Palanca", "Parámetro", "Niveles", "Referencia 1×", "Lectura en el mundo real"],
                     ["Control de la contaminación", "pptcm", "0–8 %/año", "4 %/año",
                      "Reducción anual de la contaminación persistente por unidad producida. 4 %/año la divide a la mitad cada 17 años."],
                     ["Eficiencia de recursos", "rtcm", "0–8 %/año", "4 %/año",
                      "Menos materiales no renovables por unidad de producto. La COP28 se comprometió a duplicar la mejora de eficiencia energética hasta ≈ 4 %/año."],
                     ["Agro-tecnología", "lytcrm", "0–8 %/año", "4 %/año",
                      "Crecimiento del rendimiento de la tierra cuando falta comida. La Revolución Verde rondó 2 %/año."],
                     ["Protección de suelos", "llmytm", "sí / no", "sí", "Frena la pérdida de fertilidad por erosión y uso intensivo."],
                     ["Transición demográfica", "zpgt, fcest, objetivo", "3 a 1,5 hijos", "2 hijos",
                      "Familia deseada y anticoncepción plenamente eficaz. La fecundidad mundial está hoy cerca de 2,2."],
                     ["Suficiencia material", "iet, iopcd", "500 a 250", "350",
                      "Producción industrial per cápita deseada (USD de 1968). Hoy el modelo está en ≈ 350: 350 congela el nivel actual, 250 lo recorta un 30 %, 500 lo empuja al alza."]],
                    [30 * mm, 22 * mm, 20 * mm, 20 * mm, TEXT_W - 92 * mm], zebra=True),
              Spacer(1, 6),
              para("Las palancas tecnológicas no son gratis: en World3-03 encarecen el capital (suben la relación capital-producto), "
                   "y la suficiencia desvía producción hacia consumo y servicios. Por eso cada capítulo incluye el coste en producción "
                   "industrial acumulada 2027–2060 frente a la inercia. Recursos base = el recurso no renovable de World3; recursos "
                   "abundantes = el doble y mejor tecnología de extracción (escenario 2 del libro)."),
              PageBreak()]
    story += valley_chapter()

    # ── capítulos por palanca ──
    chapters = [
        ("poll", "Tecnología de control de la contaminación",
         "Reduce la contaminación persistente que genera cada unidad de producción industrial y agrícola. En el modelo, esa "
         "contaminación (plásticos, metales pesados, compuestos persistentes, gases de larga vida) se acumula con retraso y "
         "baja la esperanza de vida y el rendimiento de la tierra.",
         "La tecnología crece a la tasa elegida mientras la contaminación esté por encima del nivel deseado y deja de crecer "
         "cuando baja. Mientras se despliega, encarece el capital: cada unidad de producción necesita más inversión.",
         ),
        ("res", "Eficiencia en el uso de recursos",
         "Reduce los materiales no renovables (minerales, combustibles fósiles) que hacen falta por persona y por unidad de "
         "producción industrial.",
         "Mientras el uso de recursos supera el deseado, la tecnología mejora a la tasa elegida, con retraso y con coste en "
         "capital. Su efecto es estirar el stock de recursos: retrasa el momento en que extraer cuesta tanto que la industria se "
         "descapitaliza.",
         ),
        ("agro", "Agro-tecnología y protección de suelos",
         "Sube el rendimiento de la tierra cultivada (semillas, riego, fertilización, manejo) y, como sub-palanca, protege los "
         "suelos de la erosión.",
         "El rendimiento crece a la tasa elegida mientras la comida per cápita esté por debajo de la deseada. Más comida baja la "
         "mortalidad y libera capital agrícola, pero también sostiene más población y más actividad industrial.",
         ),
        ("fam", "Transición demográfica",
         "Fija el tamaño de familia deseado y hace que la anticoncepción sea plenamente eficaz desde el año de política: "
         "educación, acceso a planificación familiar, autonomía de las mujeres.",
         "En World3-03 la natalidad se ajusta al tamaño de familia deseado con el retraso de una generación. Menos nacimientos "
         "significa menos bocas y más capital por persona, pero también una población que envejece y, por debajo de 2 hijos, "
         "se contrae.",
         ),
        ("suff", "Suficiencia material",
         "Fija un nivel deseado de producción industrial per cápita. Cuando la economía lo alcanza, una parte mayor de la "
         "producción va a consumo y servicios en lugar de a reinvertir en más industria.",
         "Es la palanca de los límites deliberados: frena el crecimiento del capital industrial, con menos extracción y menos "
         "contaminación, a cambio de menos producción material.",
         ),
    ]
    for lv, title, what, how in chapters:
        f_dose, f_met, f_traj = figs[lv]
        rows_s, rows_p, rows_b = dose(lv, "solo", "abund"), dose(lv, "paquete", "abund"), dose(lv, "paquete", "base")
        tab = [["Nivel", "Col.", "Crisis", "Contr.", "Sobreg.", "Sost.", "HWI", "EV mín.", "EV 2100", "Pob.",
                "Huella", "Ind. 27–60"]]
        for r in rows_p:
            tab.append([lvl_label(lv, r["level"])] + [pct(v) for v in r["p_destiny"]] +
                       [num(r["hwi_mean"]["mean"], 3), num(med(r, "le_min"), 0), num(med(r, "le_2100"), 0),
                        num(med(r, "pop_2100"), 1), num(med(r, "hef_2100"), 2),
                        f"{r['io_2027_2060']['mean'] * 100:+.0f} %"])
        cw = [20 * mm] + [12.4 * mm] * 5 + [13 * mm] * 6
        cw[-1] = TEXT_W - sum(cw[:-1])
        lect, impl = LEVER_TEXT[lv](rows_s, rows_p, rows_b)
        story += [para(f"Palanca · {LEVERS_LABEL[lv]}", "eyebrow"), para(title, "h2"),
                  para("Qué es", "h3"), para(what), para("Cómo actúa en el modelo", "h3"), para(how),
                  para("Escalas: probabilidad de cada destino", "h3"), img(f_dose),
                  para("Recursos abundantes. Izquierda: la palanca sola. Derecha: con las demás palancas en su referencia 1×. "
                       "Con recursos base todas las escalas colapsan (100 %); lo que cambia es la profundidad de la caída.", "cap")]
        story += lect + impl
        story += [PageBreak(),
                  para("Indicadores por escala", "h3"), img(f_met),
                  para("Mediana y banda p10–p90 (verde: en paquete con recursos abundantes); negro: la palanca sola; gris: en "
                       "paquete con recursos base. La última gráfica es la producción industrial acumulada 2027–2060 frente a la inercia.", "cap"),
                  CondPageBreak(70 * mm),
                  para("Trayectorias por escala (en paquete, recursos abundantes)", "h3"), img(f_traj),
                  para("Mediana de 500 mundos. Gris: la palanca apagada; del azul claro al oscuro, más intensidad. La línea vertical marca 2026.", "cap"),
                  CondPageBreak(60 * mm),
                  para("Tabla de escalas (en paquete, recursos abundantes)", "h3"), table(tab, cw, zebra=True),
                  para("Col. = colapso, Crisis = crisis humanitaria, Contr. = contracción ordenada, Sobreg. = meseta en sobregiro, "
                       "Sost. = transición sostenible (probabilidades sobre 500 mundos). HWI = bienestar medio 2027–2100. EV = esperanza "
                       "de vida (años), Pob. = población 2100 (miles de millones) y Huella = planetas en 2100, en mediana. Ind. 27–60 = "
                       "producción industrial acumulada 2027–2060 frente a la inercia.", "cap")]
        if lv == "agro":
            sd = dose("soil", "paquete", "abund")
            story += [CondPageBreak(80 * mm), para("Sub-palanca: protección de suelos", "h3"), img(figs["soil"][0], TEXT_W * 0.85),
                      para(f"Dentro del paquete y con recursos abundantes, quitar la protección de suelos sube el colapso de "
                           f"{pct(P(sd[1], 0))} a {pct(P(sd[0], 0))} y baja la transición sostenible de {pct(P(sd[1], 4))} a "
                           f"{pct(P(sd[0], 4))}. Sola, la protección no cambia nada: hace falta tener tierra productiva que proteger "
                           "y presión sobre ella. Es la palanca más barata del estudio (coste en producción industrial casi nulo) y "
                           "la que más pierde quien la omite. La huella de 2100 sin ella (≈ "
                           f"{num(med(sd[0], 'hef_2100'), 2)} planetas) parece buena, pero es engañosa: es tierra perdida, no tierra liberada.")]
        story += [CondPageBreak(150 * mm)] + impl_flowables(lv) + [PageBreak()]

    # ── momento × intensidad ──
    T = D["timing"]["grid"]["abund"]
    story += [para("Cruce", "eyebrow"), para("Cuándo y cuánto: el paquete en el tiempo", "h2"),
              para("Aquí todas las palancas se mueven juntas con una sola intensidad (de 0 a 1,5 veces la referencia) y un año de "
                   "arranque entre 2027 y 2060. Cada celda es la probabilidad sobre 500 mundos con recursos abundantes."),
              img(f_timing),
              para("Cada fila es una intensidad del paquete; cada columna, un año de arranque.", "cap"),
              img(f_timing_hwi),
              para("Bienestar medio 2027–2100 por intensidad y año de arranque.", "cap")]
    story += bullets([
        f"<b>La ventana de la transición sostenible se cierra hacia 2045.</b> Con el paquete 1×: {pct(T[4][0]['p_destiny'][4])} en 2027, "
        f"{pct(T[4][1]['p_destiny'][4])} en 2030, {pct(T[4][2]['p_destiny'][4])} en 2035, {pct(T[4][3]['p_destiny'][4])} en 2040 y "
        f"{pct(T[4][4]['p_destiny'][4])} en 2045. Al 125 %: {pct(T[5][0]['p_destiny'][4])}, {pct(T[5][1]['p_destiny'][4])}, "
        f"{pct(T[5][2]['p_destiny'][4])}, {pct(T[5][3]['p_destiny'][4])} y {pct(T[5][4]['p_destiny'][4])}.",
        f"<b>La intensidad compensa el retraso, pero sólo un tiempo.</b> Arrancar en 2035 al 125 % ({pct(T[5][2]['p_destiny'][4])}) "
        f"rinde más que en 2035 al 100 % ({pct(T[4][2]['p_destiny'][4])}), pero menos que el 100 % en 2030 "
        f"({pct(T[4][1]['p_destiny'][4])}). Desde 2045, ninguna intensidad devuelve la transición sostenible.",
        f"<b>Las escalas bajas no protegen ni a tiempo.</b> Al 25 %, el colapso queda entre {pct(T[1][0]['p_destiny'][0])} y "
        f"{pct(T[1][6]['p_destiny'][0])}; al 50 % arrancado en 2027 baja a {pct(T[2][0]['p_destiny'][0])}, pero sube a "
        f"{pct(T[2][6]['p_destiny'][0])} si se arranca en 2060.",
        "<b>Con el paquete fuerte, esperar no lleva al colapso sino a la crisis.</b> Al 100–150 %, el colapso casi no aparece "
        "hasta 2050; lo que crece con el retraso es la crisis humanitaria: el paquete tardío rescata, pero después del valle.",
    ])
    story.append(PageBreak())

    # ── geología ──
    Gg = D["geology"]["grid"]
    story += [para("Incertidumbre", "eyebrow"), para("La geología como techo", "h2"),
              para("El recurso no renovable inicial es la incertidumbre que más pesa en World3-03. Aquí lo variamos de 1 a 3 veces "
                   "el valor del modelo, con la tecnología de extracción del escenario 2, y lo cruzamos con cuatro intensidades del paquete."),
              img(f_geo), para("Cada línea es una intensidad del paquete; el eje horizontal, el recurso inicial.", "cap")]
    story += bullets([
        f"<b>Sin políticas, más recursos no evitan el colapso: lo cambian de causa.</b> Con 3 veces el recurso, el colapso sigue en "
        f"{pct(Gg[0][6]['p_destiny'][0])}; ya no es por agotamiento, sino por la contaminación que esa abundancia permite producir.",
        f"<b>El umbral de recursos depende de la intensidad.</b> Al 50 %, el colapso baja de {pct(Gg[1][0]['p_destiny'][0])} a "
        f"{pct(Gg[1][4]['p_destiny'][0])} entre 1× y 2× recursos. Al 100 %, casi desaparece desde 1,75× y la transición sostenible "
        f"se estabiliza en {pct(Gg[2][4]['p_destiny'][4])}. Al 150 %, basta con 1,5× ({pct(Gg[3][2]['p_destiny'][0])} de colapso).",
        f"<b>La tecnología de extracción también cuenta.</b> Con el recurso de World3 (1×) pero con la extracción mejorada del "
        f"escenario 2, el paquete al 150 % baja el colapso a {pct(Gg[3][0]['p_destiny'][0])}; sin esa mejora es ≥ 95 % en todas las escalas.",
        f"<b>Más allá de 2× recursos no se gana nada.</b> Las curvas se aplanan: el límite pasa a ser la contaminación y la "
        f"tierra, no la geología.",
        "<b>Implicación:</b> como no sabemos cuántos recursos son económicamente extraíbles, la escala del paquete funciona "
        "como un seguro. Una intensidad mayor amplía el rango de geologías en las que el mundo no colapsa.",
    ])
    story.append(PageBreak())

    # ── interacciones ──
    I = D["interactions"]
    story += [para("Cruce", "eyebrow"), para("Interacciones: palancas que se necesitan", "h2"),
              para("Las palancas no suman: se multiplican o se estorban. Aquí cruzamos dos pares, con el resto del paquete en su "
                   "referencia 1× y recursos abundantes."),
              img(f_inter),
              para("Izquierda: transición sostenible por tasa de agro-tecnología (filas) y de control de la contaminación (columnas). "
                   "Derecha: bienestar medio por tamaño de familia (filas) y producción industrial deseada (columnas).", "cap")]
    story += bullets([
        f"<b>Agro y contaminación son complementarias.</b> Sin control de la contaminación, ningún nivel de agro-tecnología pasa de "
        f"{pct(max(r[0]['p_destiny'][4] for r in I['tech']))} de transición sostenible. Sin agro-tecnología, el máximo es "
        f"{pct(max(I['tech'][0][j]['p_destiny'][4] for j in range(7)))}. Juntas al 8 %/año llegan a {pct(I['tech'][6][6]['p_destiny'][4])}.",
        f"<b>El control de la contaminación tiene un umbral de 3 %/año.</b> Por debajo, más agro-tecnología evita el colapso "
        f"pero no lleva a la sostenibilidad: con 2 %/año de control, la transición sostenible se queda en ≈ {pct(I['tech'][4][2]['p_destiny'][4])} "
        "por mucho que suba la agro. Desde 3 %/año, cada punto de agro suma.",
        f"<b>Familia pequeña y suficiencia se sustituyen en parte.</b> El bienestar más alto ({num(max(max(r['hwi_mean']['mean'] for r in row) for row in I['social']), 2)}) "
        "sale de familias de 1,5 hijos con poca restricción al consumo: menos personas permiten más producción por persona.",
    ])
    story += [CondPageBreak(95 * mm), img(f_inter_soc),
              para("Izquierda: transición sostenible por tamaño de familia y suficiencia. Derecha: producción industrial per cápita en 2100.", "cap")]
    story += bullets([
        f"<b>Para la sostenibilidad, la suficiencia pesa más que la familia.</b> Con 2 hijos, bajar la producción deseada de 500 a 300 "
        f"lleva la transición sostenible de {pct(I['social'][4][1]['p_destiny'][4])} a {pct(I['social'][4][5]['p_destiny'][4])}.",
        f"<b>Por debajo de 1,75 hijos, la población se contrae.</b> Con 1,5 hijos, entre {pct(min(r['p_destiny'][2] for r in I['social'][6]))} y "
        f"{pct(max(r['p_destiny'][2] for r in I['social'][6]))} de los mundos terminan en contracción ordenada: con más bienestar, "
        "pero con una población que cae más de 20 %.",
    ])
    story.append(PageBreak())

    # ── escenarios compuestos ──
    story += [para("Escenarios", "eyebrow"), para("Nueve futuros compuestos", "h2"),
              para("Las palancas combinadas en nueve historias coherentes: cinco escalas del mismo paquete (de la inercia al "
                   "ambicioso), dos énfasis opuestos (tecnología frente a sobriedad), el paquete tardío y una transición «verde» que "
                   "olvida la agricultura."),
              img(f_over), para("Probabilidad de cada destino por escenario.", "cap"),
              CondPageBreak(80 * mm),
              para("Tarjeta de resultados (recursos abundantes)", "h3"), img(f_score),
              para("Mediana (punto) y p10–p90 (barra) de 500 mundos.", "cap")]
    rows = [["Escenario", "Palancas", "Col.", "Crisis", "Contr.", "Sost.", "HWI", "EV mín.",
             "Pob.", "Huella", "Col. base"]]
    for s in D["scenarios"]:
        a = s["geo"]["abund"]
        pol = s["policy"]
        lev = (f"{pol['poll'] * 100:g}/{pol['res'] * 100:g}/{pol['agro'] * 100:g} %, "
               f"{'suelos, ' if pol['soil'] else ''}{('%g hijos' % pol['fam']) if pol['fam'] else 'sin dem.'}, "
               f"{('cons. %g' % pol['suff']) if pol['suff'] else 'sin suf.'}{', 2045' if s['year'] > 2030 else ''}")
        rows.append([Paragraph(f"<b>{s['name']}</b>", S["cell"]), lev, pct(P(a, 0)), pct(P(a, 1)), pct(P(a, 2)), pct(P(a, 4)),
                     num(a["hwi_mean"]["mean"], 3), num(med(a, "le_min"), 0), num(med(a, "pop_2100"), 1),
                     num(med(a, "hef_2100"), 2), pct(P(s["geo"]["base"], 0))])
    cw = [23 * mm, 40 * mm, 12 * mm, 11 * mm, 12 * mm, 13 * mm, 12 * mm, 12 * mm, 11 * mm, 11 * mm]
    cw.append(TEXT_W - sum(cw))
    story += [CondPageBreak(70 * mm), para("Tabla comparativa", "h3"), table(rows, cw, zebra=True),
              para("Palancas: contaminación/recursos/agro en %/año. Col. = colapso, Crisis = crisis humanitaria, Contr. = contracción ordenada, Sost. = transición sostenible, HWI = bienestar medio, EV mín. = esperanza de vida mínima, Pob. = población 2100 (mm). Recursos abundantes salvo «Col. base».", "cap"),
              PageBreak()]

    for s in D["scenarios"]:
        a, b = s["geo"]["abund"], s["geo"]["base"]
        block = [para(s["name"], "h3"), para(s["what"], "cap"), img(prof[s["key"]]),
                 para(f"<b>Recursos abundantes:</b> {dest_row(a['p_destiny'])}. Pico de población "
                      f"{num(med(a, 'pop_peak'), 1)} mm hacia {round(med(a, 'pop_peak_year'))}; en 2100, {num(med(a, 'pop_2100'), 1)} mm "
                      f"con {num(med(a, 'le_2100'), 0)} años de esperanza de vida (mínimo en el siglo: {num(med(a, 'le_min'), 0)}). "
                      f"Huella 2100: {num(med(a, 'hef_2100'), 2)}; pico de contaminación {num(med(a, 'ppolx_peak'), 0)} veces 1970. "
                      f"<b>Recursos base:</b> colapso {pct(P(b, 0))}, esperanza de vida 2100 {num(med(b, 'le_2100'), 0)} años, "
                      f"población {num(med(b, 'pop_2100'), 1)} mm."),
                 para(SCEN_TEXT[s["key"]]), Spacer(1, 6)]
        story.append(KeepTogether(block))
    story.append(para("Verde: el escenario (mediana y p10–p90); gris: la inercia. Recursos abundantes; la línea discontinua "
                      "marca el año de arranque.", "cap"))
    story.append(PageBreak())

    # ── conclusiones ──
    story += [para("Síntesis", "eyebrow"), para("Qué escala elegir", "h2"),
              para("Ordenando lo que muestran los capítulos, la escala mínima eficaz de cada palanca, la que se acerca al máximo "
                   "beneficio sin entrar en zonas de riesgo, sería esta:")]
    story.append(table([
        ["Palanca", "Escala mínima eficaz", "Escala sugerida", "Riesgo si se excede", "Riesgo si se queda corta"],
        ["Control de la contaminación", "3 %/año", "4 %/año", "Coste de capital creciente; rendimiento marginal casi nulo desde 6 %",
         "Por debajo de 3 %/año la huella no baja de 1,1 y no hay transición sostenible"],
        ["Eficiencia de recursos", "2 %/año", "2–4 %/año", "Menos producción industrial sin ganancia de destino",
         "Colapso residual de 4–11 % aun con el resto del paquete"],
        ["Agro-tecnología", "3 %/año", "4–6 %/año con control de contaminación ≥ 3 %", "Sola por encima de 4 %/año: más población y colapso por contaminación",
         "Crisis alimentarias: la palanca con mayor efecto sobre el colapso"],
        ["Protección de suelos", "Sí", "Sí", "Ninguno detectado", "Colapso de ≈ 40 % aun con el paquete completo"],
        ["Transición demográfica", "2,25 hijos", "1,75–2 hijos", "Con 1,5 hijos, contracción de más del 20 % en uno de cada cuatro mundos",
         "Con 3 hijos, 12 mil millones en 2100 y más crisis"],
        ["Suficiencia material", "400", "300–350", "Con 250, menos bienestar material sin más sostenibilidad",
         "Con 500, más contaminación y menos transición sostenible"],
    ], [30 * mm, 24 * mm, 30 * mm, 45 * mm, TEXT_W - 129 * mm], zebra=True))
    story += [Spacer(1, 8), para("Cinco lecciones", "h3")]
    story += bullets([
        "<b>Hay un umbral de escala.</b> Por debajo de la mitad de la intensidad de <i>Limits to Growth</i>, el paquete entero "
        "apenas cambia el destino. La política tímida no es una versión pequeña de la buena: es otra cosa.",
        "<b>El orden importa menos que la combinación.</b> Ninguna palanca sola produce una transición sostenible en ninguna "
        "escala. Las parejas críticas son agro + contaminación y agro + suelos.",
        "<b>Hay dosis que dañan.</b> La agro-tecnología sin control de la contaminación y la fecundidad muy baja tienen óptimos "
        "intermedios. Más no siempre es mejor; conviene escalar por etapas y vigilar.",
        "<b>El tiempo convierte la escala en valle.</b> Desde 2027 casi todos los caminos atraviesan una crisis humanitaria "
        "antes de recuperarse; cada año de espera la hace más profunda. Sólo la intensidad alta la evita en la mayoría de mundos.",
        "<b>La incertidumbre geológica pide margen.</b> Como no sabemos cuántos recursos quedan, una escala del 100–125 % es "
        "un seguro: amplía el rango de mundos en los que no hay colapso.",
    ])
    story += [Spacer(1, 6), callout(
        "World3-03 es un mundo de una sola región, con la tecnología como tasas fijas y sin precios ni clima explícito. Su inercia "
        "toca techo de población hacia 2029, antes que las proyecciones de la ONU. Use estas cifras para comparar escalas y "
        "combinaciones entre sí, no para fechar el futuro.", "Cómo usar estas cifras"), PageBreak()]

    # ── metodología ──
    story += [para("Anexo", "eyebrow"), para("Metodología, límites y reproducibilidad", "h2"),
              para("Experimentos", "h3")]
    story += bullets([
        "<b>A. Dosis-respuesta:</b> 6 palancas × 7 niveles (2 para suelos) × 2 contextos (sola, en paquete) × 2 geologías.",
        "<b>B. Momento × intensidad:</b> paquete a 7 intensidades (0–1,5×) × 7 años de arranque (2027–2060) × 2 geologías.",
        "<b>C. Geología continua:</b> recurso inicial de 1 a 3 veces × 4 intensidades del paquete.",
        "<b>D. Interacciones:</b> agro × contaminación (7 × 7) y familia × suficiencia (7 × 7), con recursos abundantes.",
        "<b>E. Escenarios compuestos:</b> 9 escenarios × 2 geologías, con trayectorias completas.",
        "<b>F. Anatomía del valle:</b> 6 casos con trayectorias de 14 variables; la caída de ln(esperanza de vida) entre 2026 y "
        "el valle se separa en los multiplicadores de World3-03 (comida, contaminación y hacinamiento calculados de sus tablas; "
        "servicios de salud por residuo) y se resume con la mediana sobre los mundos con caída mayor del 5 %.",
    ])
    story += [para("Paquete a escala s", "h3"),
              para("Contaminación, recursos y agro-tecnología = 4 % × s por año; suelos protegidos si s > 0; tamaño de familia "
                   "objetivo = 3 − s (mínimo 1,4); producción industrial deseada = 500 − 150 s (mínimo 200). Con s = 1 es el escenario 9 "
                   "de <i>Limits to Growth</i> desde 2027."),
              para("Incertidumbre", "h3"),
              para("Seis parámetros con previas uniformes [M]: recurso no renovable inicial (0,75–1,3 × 10<super>12</super>), vida "
                   "media de absorción de la contaminación (1–2,5 años en 1970), retardo de transmisión de la contaminación (15–30 años), "
                   "retardo tecnológico (15–30 años), tamaño de familia normal (3,4–4,2) y factor de emisión industrial (0,07–0,13). "
                   "Filtro de historia: población 2000 y 2020 dentro de ±6 % de UN WPP 2024."),
              para("Cambios en el código", "h3"),
              para("Para graduar la palanca demográfica, <font name='Mono'>world4/core/l0.py</font> gana la constante "
                   "<font name='Mono'>dcfszpg</font> (tamaño de familia tras <font name='Mono'>zpgt</font>), con valor por defecto 2,0: "
                   "el comportamiento previo no cambia y las pruebas de World3-03 siguen pasando. La tecnología de recursos usa la tabla "
                   "<font name='Mono'>rtcm</font> con signo + porque en el port el flujo de <font name='Mono'>rct</font> es "
                   "<font name='Mono'>-rctcr</font> (ver el informe del árbol de decisión)."),
              para("Límites", "h3")]
    story += bullets([
        "Modos de comportamiento, no pronósticos. Las cajas de incertidumbre, los umbrales de destino y la traducción de "
        "palancas a parámetros son supuestos declarados [M].",
        "Las tasas tecnológicas de World3 son condicionales (sólo actúan mientras el problema existe) y no tienen precio: un "
        "8 %/año es una cota, no una propuesta realista.",
        "La capa regional de Phy-World (F6–F8: 10 regiones, clima FaIR, alimentos FAOSTAT) daría la distribución entre regiones; "
        "sus datos no se pudieron descargar en esta sesión (red restringida).",
    ])
    story += [para("Reproducir", "h3"),
              para("<font name='Mono' size='8'>PYTHONPATH=. python reports/escenarios/estudio.py</font> (≈ 2 min) y "
                   "<font name='Mono' size='8'>python reports/escenarios/informe_pdf.py</font>. "
                   f"Semilla {D['meta']['seed']}; {D['meta']['members']} mundos; cálculo de {D['meta']['runtime_s']} s.")]

    doc = Doc(OUT)
    doc.build(story)
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


LEVERS_LABEL = {"poll": "contaminación", "res": "recursos", "agro": "alimentos y suelos", "fam": "demografía",
                "suff": "consumo"}


def _poll_text(rs, rp, rb):
    return [para("Lectura", "h3")] + bullets([
        f"<b>Sola, es la palanca más eficaz contra el colapso.</b> De 0 a 8 %/año el colapso baja de {pct(P(rs[0], 0))} a "
        f"{pct(P(rs[6], 0))}; el resto pasa a crisis humanitaria. Retrasa y aplana el pico de contaminación "
        f"({num(med(rs[0], 'ppolx_peak'), 0)} → {num(med(rs[6], 'ppolx_peak'), 0)} veces 1970).",
        f"<b>En paquete, es la llave de la sostenibilidad.</b> Sin ella, la huella 2100 queda en {num(med(rp[0], 'hef_2100'), 2)} "
        f"planetas y la transición sostenible en {pct(P(rp[0], 4))}. Desde 3 %/año la huella baja de 1,1 y la transición "
        f"sostenible sube a {pct(P(rp[3], 4))}.",
        f"<b>Rendimientos decrecientes.</b> De 4 a 8 %/año sólo se ganan {round((P(rp[6], 4) - P(rp[4], 4)) * 100)} puntos, y la "
        f"producción industrial per cápita de 2100 baja de {num(med(rp[4], 'iopc_2100'), 0)} a {num(med(rp[6], 'iopc_2100'), 0)}: "
        "la tecnología se paga con capital.",
    ]), [para("Implicaciones", "h3"),
          para("En el mundo real, 3–4 %/año equivale a reducir a la mitad la contaminación persistente por unidad producida cada "
               "17–23 años: eliminar los contaminantes persistentes de los procesos, economía circular de químicos y plásticos, "
               "control de metales pesados. Es más del doble del ritmo histórico de descarbonización de la economía mundial. "
               "<b>Escala sugerida: 4 %/año, siempre junto a la agro-tecnología.</b>")]


def _res_text(rs, rp, rb):
    return [para("Lectura", "h3")] + bullets([
        f"<b>Sola, casi no cambia el destino.</b> El colapso pasa de {pct(P(rs[0], 0))} a {pct(P(rs[6], 0))}: estirar los "
        "recursos sólo retrasa el problema y deja que la contaminación llegue primero.",
        f"<b>En paquete, elimina el colapso residual.</b> Sin ella, el paquete deja {pct(P(rp[0], 0))} de colapso; con 2 %/año, "
        f"{pct(P(rp[2], 0))}. Más allá de 3 %/año no se gana nada en destinos.",
        f"<b>Tiene coste en bienestar a escalas altas.</b> El bienestar medio cae de {num(rp[2]['hwi_mean']['mean'], 3)} (2 %/año) a "
        f"{num(rp[6]['hwi_mean']['mean'], 3)} (8 %/año), porque el capital se va en tecnología.",
        f"<b>Deja recursos para después de 2100:</b> la fracción restante en 2100 pasa de {num(med(rp[0], 'nrfr_2100'), 2)} a "
        f"{num(med(rp[6], 'nrfr_2100'), 2)}.",
    ]), [para("Implicaciones", "h3"),
          para("Equivale a duplicar el ritmo histórico de mejora de la eficiencia energética y material, el compromiso de la COP28 "
               "para 2030: reciclaje de metales, diseño para durar, electrificación eficiente. Su función es de seguro. "
               "<b>Escala sugerida: 2–4 %/año; más allá, el coste supera al beneficio.</b>")]


def _agro_text(rs, rp, rb):
    best = max(range(len(rs)), key=lambda i: -P(rs[i], 0))
    return [para("Lectura", "h3")] + bullets([
        f"<b>Sola, tiene una dosis óptima y luego daña.</b> El colapso baja de {pct(P(rs[0], 0))} a {pct(P(rs[best], 0))} con "
        f"{lvl_label('agro', rs[best]['level'])}, y vuelve a subir a {pct(P(rs[5], 0))} con 6 %/año. Más comida sostiene más "
        f"población (mediana 2100: {num(med(rs[3], 'pop_2100'), 1)} → {num(med(rs[4], 'pop_2100'), 1)} mm entre 3 y 4 %/año) y la "
        "industria que la alimenta acaba chocando con la contaminación.",
        f"<b>En paquete, es la palanca más rentable.</b> La transición sostenible sube sin pausa, de {pct(P(rp[0], 4))} a "
        f"{pct(P(rp[6], 4))}, y la crisis humanitaria baja de {pct(P(rp[0], 1))} a {pct(P(rp[6], 1))}.",
        f"<b>Es la que más mueve el valle.</b> La esperanza de vida mínima pasa de {num(med(rp[0], 'le_min'), 0)} a "
        f"{num(med(rp[6], 'le_min'), 0)} años: la crisis de transición es, sobre todo, alimentaria.",
        f"<b>Con recursos base</b> sube la esperanza de vida de 2100 ({num(med(rb[0], 'le_2100'), 0)} → {num(med(rb[6], 'le_2100'), 0)} "
        "años) aunque el colapso no se evite.",
    ]), [para("Implicaciones", "h3"),
          para("2 %/año es el ritmo de la Revolución Verde; 4–6 %/año pide una segunda revolución agrícola sostenida durante décadas "
               "(mejoramiento, agua, suelos, agricultura de precisión). La lección es de secuencia: escalar la agro-tecnología "
               "sin el control de la contaminación y la transición demográfica es la combinación más peligrosa del estudio. "
               "<b>Escala sugerida: 4–6 %/año dentro del paquete, nunca sola por encima de 3 %/año.</b>")]


def _fam_text(rs, rp, rb):
    return [para("Lectura", "h3")] + bullets([
        f"<b>Sola, evita poco el colapso.</b> Con 1,5 hijos, el colapso baja a {pct(P(rs[6], 0))} (desde {pct(P(rs[0], 0))}); con 3 hijos "
        f"(por encima del objetivo implícito de hoy) empeora a {pct(P(rs[1], 0))}.",
        f"<b>En paquete, decide cuánta gente hay en 2100.</b> De 3 a 1,5 hijos la población 2100 va de {num(med(rp[1], 'pop_2100'), 1)} a "
        f"{num(med(rp[6], 'pop_2100'), 1)} mm y el bienestar medio de {num(rp[1]['hwi_mean']['mean'], 3)} a {num(rp[6]['hwi_mean']['mean'], 3)}.",
        f"<b>El óptimo para la sostenibilidad está en 1,75 hijos</b> ({pct(P(rp[5], 4))} de transición sostenible). Con 1,5 hijos la "
        f"sostenibilidad cae a {pct(P(rp[6], 4))} porque {pct(P(rp[6], 2))} de los mundos se contraen más de 20 %: no es una crisis, "
        "pero sí un cambio social profundo (envejecimiento, menos población activa).",
        f"<b>Es la palanca más barata en capital</b> (industria 2027–2060: {rp[4]['io_2027_2060']['mean'] * 100:+.0f} % con 2 hijos, igual que el paquete).",
    ]), [para("Implicaciones", "h3"),
          para("La fecundidad mundial ronda hoy 2,2 hijos por mujer y baja; la palanca no es imponer un número, sino acelerar lo que "
               "ya funciona: educación de las niñas, salud reproductiva, acceso a anticoncepción. El modelo dice que el objetivo "
               "útil es de 1,75 a 2 hijos y que ir más allá cambia la sostenibilidad por contracción. "
               "<b>Escala sugerida: 1,75–2 hijos.</b>")]


def _suff_text(rs, rp, rb):
    return [para("Lectura", "h3")] + bullets([
        f"<b>Sola, sólo ayuda cuando es fuerte.</b> Con 250 (un 30 % por debajo del nivel actual) el colapso baja a {pct(P(rs[6], 0))}; "
        f"con 500 (más inversión que hoy) la esperanza de vida de 2100 empeora a {num(med(rs[1], 'le_2100'), 0)} años.",
        f"<b>En paquete, cambia bienestar material por sostenibilidad.</b> De 500 a 300 la transición sostenible sube de "
        f"{pct(P(rp[1], 4))} a {pct(P(rp[5], 4))} y el pico de contaminación baja de {num(med(rp[1], 'ppolx_peak'), 0)} a "
        f"{num(med(rp[5], 'ppolx_peak'), 0)} veces 1970.",
        f"<b>Pero el bienestar baja</b> de {num(rp[1]['hwi_mean']['mean'], 3)} a {num(rp[6]['hwi_mean']['mean'], 3)}, y la producción "
        f"industrial 2027–2060 cae {abs(rp[6]['io_2027_2060']['mean']) * 100:.0f} % frente a la inercia con 250. Es la palanca con el "
        "coste más visible.",
        f"<b>Por debajo de 300 no se gana nada:</b> la transición sostenible se estanca ({pct(P(rp[5], 4))} → {pct(P(rp[6], 4))}).",
    ]), [para("Implicaciones", "h3"),
          para("350 significa congelar la producción industrial per cápita en el nivel actual del modelo y orientar el crecimiento "
               "hacia servicios, salud y educación. Es la palanca políticamente más difícil y la que más complementa a la tecnología: "
               "con menos industria, cada punto de control de la contaminación rinde más. "
               "<b>Escala sugerida: 300–350.</b>")]


LEVER_TEXT = {"poll": _poll_text, "res": _res_text, "agro": _agro_text, "fam": _fam_text, "suff": _suff_text}

SCEN_TEXT = {
    "inercia": "Sin políticas nuevas, la población toca techo entre 2029 y 2040 según la geología y cae a menos de la mitad. Con más recursos "
               "el colapso no se evita: llega por contaminación, con picos de 70 veces el nivel de 1970.",
    "timido": "Mejoras de 1 %/año y políticas sociales suaves cambian casi nada: el colapso apenas baja. Es la lección central "
              "del estudio: por debajo de un umbral, la política existe en el papel pero no en el resultado.",
    "moderado": "A mitad de escala el colapso cae mucho, pero casi todo se convierte en crisis humanitaria: el mundo se salva "
                "después de un valle profundo. Es el primer escalón donde el paquete cambia el destino.",
    "ltg": "El escenario 9 de Limits to Growth, desde hoy. Elimina el colapso con recursos abundantes, y uno de cada tres mundos "
           "llega a una transición sostenible sin crisis. Los demás pasan por un valle antes de recuperarse.",
    "ambicioso": "Al 150 % el valle se vuelve poco profundo (esperanza de vida mínima mediana por encima de 55) y la huella baja "
                 "de 1. El coste: una población que se contrae en uno de cada cuatro mundos y menos producción industrial a medio plazo.",
    "tecno": "Tecnología al 150 % sin políticas sociales: la sostenibilidad es parecida a la del paquete 1×, pero con 9 mil "
             "millones en 2100, un tercio menos de producción industrial por persona y nueve años menos de esperanza de vida en "
             "2100. La tecnología sola compra tiempo, no holgura.",
    "sobrio": "Políticas sociales fuertes con tecnología tímida: se evita el colapso, pero la huella sigue por encima de 1 y "
              "nadie llega a una transición sostenible. La sobriedad sin tecnología no basta.",
    "tardio": "El mismo paquete que funciona en 2027, aplicado en 2045, ya no lleva a la sostenibilidad: casi todos los mundos "
              "pasan por una crisis humanitaria. Dieciocho años de espera convierten un plan de transición en un plan de rescate.",
    "verde_sin_comida": "Mucha tecnología limpia y de recursos, pero sin agro-tecnología ni suelos: el colapso casi desaparece, pero la "
                        "crisis humanitaria domina. Sin comida, no hay transición, por limpia que sea.",
}


# ── anatomía del valle ──
VCOL = {"bau_base": "#5f6b67", "bau_abund": "#a7b0ad", "ltg_base": "#eb6834", "ltg_abund": G.ACCENT,
        "amb_abund": "#0d366b", "late_abund": "#6250d6"}


def vcase(key):
    return next(c for c in D["valley"]["cases"] if c["key"] == key)


def fig_valley_chain(key, panels, fname):
    c = vcase(key)
    yrs = np.array(c["years"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9))
    vy = c["valley_year"]
    for ax, (v, title) in zip(axes.flat, panels):
        ax.axvspan(vy[0], vy[2], color=G.INK3, alpha=0.10, lw=0)
        ax.axvline(vy[1], color=G.INK3, lw=0.8, ls=(0, (2, 2)))
        ax.axvline(2026, color=G.INK3, lw=0.7)
        if isinstance(v, list):
            for vv, col, lab in zip(v, (G.ACCENT, "#6250d6"), ("Agro-tecnología", "Contaminación por unidad")):
                ax.plot(yrs, c["median"][vv], color=col, lw=1.6, label=lab)
            ax.legend(fontsize=6.2, loc="upper left")
        else:
            if v == "le":
                ax.fill_between(yrs, c["le_band"][0], c["le_band"][1], color=G.ACCENT, alpha=0.16, lw=0)
            ax.plot(yrs, c["median"][v], color=G.ACCENT if v == "le" else G.INK, lw=1.7)
        ax.set_title(title, fontsize=7.4)
        ax.set_xlim(1990, 2100)
        ax.set_xticks([2000, 2026, 2050, 2075, 2100])
        ax.tick_params(axis="x", labelsize=6.2)
        ax.set_ylim(bottom=0)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    return G.save(fig, FIG / fname)


def fig_valley_le():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ends = []
    for c in D["valley"]["cases"]:
        col = VCOL[c["key"]]
        ax.plot(c["years"], c["median"]["le"], color=col, lw=1.8 if "abund" in c["key"] and "bau" not in c["key"] else 1.4,
                label=c["name"])
        vy = c["valley_year"][1]
        i = c["years"].index(int(vy)) if int(vy) in c["years"] else int(np.argmin(np.abs(np.array(c["years"]) - vy)))
        ax.plot(c["years"][i], c["median"]["le"][i], "o", color=col, ms=4.5, mec="white", mew=0.8)
        ends.append([c["median"]["le"][-1], c["name"], col])
    ax.axvline(2026, color=G.INK3, lw=0.7)
    ax.text(2027, 4, "hoy", fontsize=6.4, color=G.INK3)
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 4.2)
    for y, name, col in ends:
        ax.text(2102, y, name.replace(" · recursos ", " · "), fontsize=6.2, color=G.INK2, va="center")
        ax.plot([2100, 2101.5], [y, y], color=col, lw=1.2)
    ax.set_xlim(1990, 2100)
    ax.set_ylim(0, 90)
    ax.set_ylabel("años", fontsize=7)
    ax.set_title("Esperanza de vida (mediana de 500 mundos); el punto marca el año mediano del valle", fontsize=7.6)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    return G.save(fig, FIG / "valley_le.png")


SHARE_COL = {"comida": "#2a78d6", "salud": "#eb6834", "contaminacion": "#1baf7a", "hacinamiento": "#eda100"}
SHARE_LAB = {"comida": "Comida", "salud": "Servicios de salud", "contaminacion": "Contaminación (directa)",
             "hacinamiento": "Hacinamiento sin industria"}


def fig_valley_share():
    cases = D["valley"]["cases"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    y = np.arange(len(cases))[::-1]
    for yi, c in zip(y, cases):
        pos, neg = 0.0, 0.0
        for k in ("comida", "salud", "contaminacion", "hacinamiento"):
            v = c["share"][k]
            if v >= 0:
                ax.barh(yi, max(v - 0.006, 0), left=pos + 0.003, color=SHARE_COL[k], height=0.6, lw=0)
                if v >= 0.1:
                    ax.text(pos + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=6.4,
                            color="white" if k in ("comida",) else G.INK)
                pos += v
            else:
                ax.barh(yi, v, left=neg, color=SHARE_COL[k], height=0.6, lw=0, alpha=0.55)
                neg += v
    ax.axvline(0, color=G.INK3, lw=0.8)
    ax.set_yticks(y, [c["name"] for c in cases], fontsize=7)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(-0.15, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Reparto de la caída de la esperanza de vida entre 2026 y el valle (mediana por mundo)", fontsize=7.6)
    handles = [plt.matplotlib.patches.Patch(color=SHARE_COL[k], label=SHARE_LAB[k]) for k in SHARE_COL]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.06), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return G.save(fig, FIG / "valley_share.png")


def valley_chapter():
    f_le, f_share = fig_valley_le(), fig_valley_share()
    f_a = fig_valley_chain("ltg_abund", [("ppolx", "Contaminación persistente (×1970)"), ("lfert", "Fertilidad del suelo"),
                                         ("ly", "Rendimiento de la tierra"), ("fpc", "Alimento per cápita"),
                                         (["lyt", "ppt"], "Tecnologías (índice, 2026 = 1)"), ("le", "Esperanza de vida (años)")],
                           "valley_chain_a.png")
    f_b = fig_valley_chain("ltg_base", [("nrfr", "Recursos no renovables restantes"),
                                        ("fcaor", "Fracción del capital usada en obtener recursos"),
                                        ("iopc", "Producción industrial p. c."), ("sopc", "Servicios p. c."),
                                        ("fpc", "Alimento per cápita"), ("le", "Esperanza de vida (años)")],
                           "valley_chain_b.png")
    V = D["valley"]
    a, b, amb, late = vcase("ltg_abund"), vcase("ltg_base"), vcase("amb_abund"), vcase("late_abund")
    dl = V["delays"]
    rg = V["ranges"]
    at = lambda c, y, k: c["at"][str(y)][k]  # noqa: E731
    out = [para("Mecanismo", "eyebrow"), para("Anatomía del valle: por qué 2040–2060", "h2"),
           para("En casi todas las trayectorias del estudio, las variables se hunden entre 2040 y 2060 y luego, si las "
                "políticas alcanzan, se recuperan. No es un artefacto ni azar: es la forma típica de World3-03, un "
                "<b>sobregiro seguido de corrección</b>. Este capítulo la desarma: qué factor hunde la esperanza de vida en cada "
                "caso, por qué en esas fechas y qué produce la recuperación.", "lede"),
           img(f_le),
           para(f"Con el paquete 1× y recursos abundantes, el valle cae en {round(a['valley_year'][1])} "
                f"(p10–p90: {round(a['valley_year'][0])}–{round(a['valley_year'][2])}). Con el paquete al 150 % llega antes "
                f"({round(amb['valley_year'][1])}) y es menos profundo; con el paquete tardío, la esperanza de vida mediana baja a "
                f"{num(late['le_valley'], 0)} años.", "cap"),
           para("Tres ingredientes", "h3")]
    out += bullets([
        f"<b>En 2026 el modelo ya está en sobregiro.</b> La contaminación persistente está en ≈ {num(at(a, 2026, 'ppolx'), 0)} veces "
        "el nivel de 1970 y subiendo, y la huella ecológica pasa de 1. Parte del daño ya está «en tránsito»: llegará aunque se actúe hoy.",
        f"<b>Retrasos de unos 20 años.</b> La contaminación tarda ≈ {dl['pptd']:g} años en transmitirse al ambiente "
        f"(entre {rg['pptd'][0]:g} y {rg['pptd'][1]:g} en el ensemble), una tecnología ≈ {dl['tdd']:g} años en desarrollarse y "
        f"aplicarse ({rg['tdd'][0]:g}–{rg['tdd'][1]:g}), el capital industrial dura {dl['alic1']:g} años y la población sigue "
        "creciendo por inercia hasta su pico. 2026 más dos décadas de retraso más la inercia demográfica dan un valle en 2045–2060.",
        "<b>La tecnología es reactiva.</b> En World3-03 las tecnologías agrícola y de contaminación sólo avanzan <i>mientras</i> "
        "el problema existe (comida por debajo de la deseada, contaminación por encima de la deseada) y luego tardan en hacer "
        "efecto. Primero llega el daño y después la solución.",
    ])
    out += [para("Retrasos del modelo que fijan la fecha", "h3"),
            table([["Retraso", "Parámetro", "Valor", "Qué retrasa"],
                   ["Transmisión de la contaminación", "pptd", f"{dl['pptd']:g} años ({rg['pptd'][0]:g}–{rg['pptd'][1]:g})",
                    "Entre que se emite un contaminante persistente y que llega a suelos, agua y personas."],
                   ["Desarrollo tecnológico", "tdd", f"{dl['tdd']:g} años ({rg['tdd'][0]:g}–{rg['tdd'][1]:g})",
                    "Entre que una tecnología mejora en el laboratorio y que se aplica en la producción."],
                   ["Vida media de absorción", "ahl70", f"{dl['ahl70']:g} años en 1970 ({rg['ahl70'][0]:g}–{rg['ahl70'][1]:g})",
                    "Cuánto tarda el ambiente en limpiar la contaminación; crece cuando la contaminación es alta."],
                   ["Vida del capital industrial", "alic", f"{dl['alic1']:g} años", "Cuánto tarda en reponerse o desaparecer la base industrial."],
                   ["Vida del capital de servicios", "alsc", f"{dl['alsc1']:g} años", "Hospitales, escuelas, infraestructura social."],
                   ["Percepción y ajuste social", "lpd, hsid, sad", f"{dl['lpd']:g} años",
                    "Cuánto tardan la natalidad y los servicios en reaccionar a cambios de esperanza de vida e ingreso."]],
                  [42 * mm, 22 * mm, 34 * mm, TEXT_W - 98 * mm], zebra=True),
            PageBreak(),
            para("Mecanismo A: contaminación → suelos → comida", "h3"),
            para("Es el que domina con recursos abundantes, y el único con recuperación. Paquete 1× desde 2027:"),
            img(f_a),
            para("Mediana de 500 mundos. Banda gris: años del valle de esperanza de vida (p10–p90); línea discontinua: año mediano "
                 "del valle; línea fina: 2026.", "cap")]
    rows = [["Año", "Contaminación (×1970)", "Fertilidad del suelo", "Rendimiento de la tierra", "Alimento p. c.", "Esperanza de vida"]]
    for y in (2026, 2045, 2055, 2070, 2100):
        rows.append([str(y), num(at(a, y, "ppolx"), 0), num(at(a, y, "lfert"), 0), num(at(a, y, "ly"), 0),
                     num(at(a, y, "fpc"), 0), num(at(a, y, "le"), 0)])
    out += [table(rows, [18 * mm] + [(TEXT_W - 18 * mm) / 5] * 5, zebra=True), Spacer(1, 4)]
    out += bullets([
        f"<b>El {round(a['share']['comida'] * 100)} % de la caída es comida.</b> La contaminación emitida en las décadas de 2020 y "
        f"2030 llega a la tierra en 2040–2060. Con 20–30 veces el nivel de 1970, la fertilidad se degrada mucho más rápido de "
        f"lo que se regenera (de {num(at(a, 2026, 'lfert'), 0)} a {num(at(a, 2055, 'lfert'), 0)}), el rendimiento se reduce "
        f"a la mitad y el alimento por persona cae de {num(at(a, 2026, 'fpc'), 0)} a {num(at(a, 2055, 'fpc'), 0)}. La "
        "contaminación casi no mata directamente: mata a través de los suelos.",
        f"<b>La política ya está actuando, pero va por detrás.</b> En 2055 la contaminación por unidad producida ya bajó a "
        f"{num(at(a, 2055, 'ppt'), 2)} de su valor inicial, pero la emitida antes sigue llegando.",
        f"<b>La recuperación.</b> La agro-tecnología madura (rendimiento tecnológico ×{num(at(a, 2055, 'lyt'), 1)} en 2055 y "
        f"×{num(at(a, 2070, 'lyt'), 1)} en 2070), la contaminación se absorbe y el suelo se regenera "
        f"({num(at(a, 2100, 'lfert'), 0)} en 2100). Además, hay menos población presionando ({num(at(a, 2070, 'pop'), 1)} mil "
        "millones en 2070). Los mismos retrasos que causan el valle sostienen la salida.",
    ])
    out += [CondPageBreak(110 * mm), para("Mecanismo B: agotamiento → capital → todo", "h3"),
            para("Domina con los recursos de World3 y casi no tiene recuperación. Paquete 1× desde 2027:"),
            img(f_b),
            para("Mismas convenciones que la figura anterior.", "cap")]
    out += bullets([
        f"<b>Extraer cuesta cada vez más capital.</b> La fracción del capital que se va en obtener recursos sube de "
        f"{pct(at(b, 2026, 'fcaor'))} en 2026 a {pct(at(b, 2035, 'fcaor'))} en 2035 y {pct(at(b, 2055, 'fcaor'))} en 2055.",
        f"<b>La industria se descapitaliza.</b> La producción industrial por persona cae de {num(at(b, 2026, 'iopc'), 0)} a "
        f"{num(at(b, 2055, 'iopc'), 0)}, y con ella los insumos agrícolas, los servicios de salud y la infraestructura. La caída "
        f"se reparte entre hacinamiento sin industria ({pct(b['share']['hacinamiento'])}), salud ({pct(b['share']['salud'])}) "
        f"y comida ({pct(b['share']['comida'])}).",
        f"<b>Sin recuperación.</b> Los recursos no renovables no se regeneran y el capital se deprecia más rápido de lo que se "
        f"repone: la esperanza de vida se queda en ≈ {num(b['le_2100'], 0)} años.",
    ])
    out += [CondPageBreak(90 * mm), para("Qué cambia el valle", "h3"), img(f_share),
            para("Caída de ln(esperanza de vida) entre 2026 y el año del valle, separada en los cuatro multiplicadores de "
                 "World3-03. Los valores negativos (a la izquierda) son factores que mejoraron mientras los otros empeoraban.", "cap")]
    out += bullets([
        f"<b>Más intensidad, valle más corto y menos hondo.</b> Con el paquete al 150 % la tecnología avanza más rápido mientras "
        f"dura el problema y la contaminación llega a {num(at(amb, 2055, 'ppolx'), 0)} en lugar de {num(at(a, 2055, 'ppolx'), 0)} "
        f"veces 1970 en 2055. El valle llega en {round(amb['valley_year'][1])} y la esperanza de vida no baja de "
        f"{num(amb['le_valley'], 0)} años: lo que se ataca son los retrasos.",
        f"<b>Esperar a 2045 no mueve el valle, lo ahonda.</b> El valle llega casi en la misma fecha "
        f"({round(late['valley_year'][1])}) porque el daño ya estaba en tránsito, pero la esperanza de vida mediana baja a "
        f"{num(late['le_valley'], 0)} años antes de recuperarse.",
    ])
    out += [para("Cómo detectarlo a tiempo", "h3"),
            para("Si el mecanismo A es el relevante, estas señales deberían moverse antes que la esperanza de vida:"),
            table([["Señal temprana", "Variable del modelo", "Cómo se mide hoy"],
                   ["Contaminantes persistentes en el ambiente", "ppolx",
                    "Monitoreo de COP y mercurio (Plan de Vigilancia Mundial del Convenio de Estocolmo, Evaluación Mundial del Mercurio)."],
                   ["Degradación de suelos", "lfert",
                    "Proporción de tierra degradada (ODS 15.3.1), carbono orgánico del suelo, tasas de erosión."],
                   ["Rendimiento agrícola que se estanca", "ly",
                    "Rendimiento de cereales por hectárea (FAOSTAT) y productividad total de los factores agrícola (USDA ERS)."],
                   ["Capital absorbido por la extracción", "fcaor",
                    "Parte de la inversión que va a minería y energía; retorno energético de la energía (EROI); costes de extracción."],
                   ["Alimento por persona", "fpc", "Suministro de calorías (FAOSTAT) y prevalencia de subalimentación (ODS 2.1.1)."]],
                  [48 * mm, 26 * mm, TEXT_W - 74 * mm], zebra=True),
            para("Cuánto confiar en la fecha", "h3")]
    out += bullets([
        "<b>La forma es estructural; la fecha, lo menos sólido.</b> En el modelo, la producción industrial por persona ya toca "
        "techo hacia 2026–2027; en el mundo real sigue creciendo. Esa calibración viene de los datos de los años setenta de "
        "<i>Limits to Growth</i>.",
        f"<b>El ensemble sólo se ajustó a la población.</b> No a la producción, la contaminación ni la comida. La esperanza de "
        f"vida mediana en 2026 sale en {num(b['le_2026'], 0)}–{num(a['le_2026'], 0)} años, por debajo de la real (≈ 73).",
        "<b>Comparaciones con datos.</b> Herrington (2021) encontró que las tendencias hasta 2020 se parecen a los escenarios "
        "de <i>Limits to Growth</i> que frenan el crecimiento hacia 2030–2040, pero el declive todavía no aparece en los datos.",
        "<b>Lectura recomendada:</b> «si los retrasos son de unos 20 años y no anticipamos el daño, hay un valle a mitad de "
        "siglo», no «en 2050 pasará X». Calibrar también con producción, contaminación y comida (la capa F7 de Phy-World) "
        "acotaría la fecha.",
    ])
    out.append(PageBreak())
    return out


# ── implementación y medición de cada medida ──
IMPL = {
    "poll": {
        "scales": [
            ("1–2 %/año", "pptcm −1 a −2 %", "El ritmo histórico: normas de emisión vigentes y mejoras graduales. Es el orden de "
             "magnitud con el que ha bajado la intensidad de carbono de la economía mundial."),
            ("3–4 %/año (ref.)", "pptcm −4 %", "La contaminación persistente por unidad producida se reduce a la mitad cada 17–23 "
             "años: salida acelerada de contaminantes persistentes, reducción fuerte de plásticos de un solo uso, industria "
             "descarbonizada y química más limpia."),
            ("6–8 %/año", "pptcm −6 a −8 %", "A la mitad cada 9–12 años: prohibición de clases enteras de sustancias, sustitución "
             "química completa en sectores clave, captura en origen. Sin precedente histórico sostenido."),
        ],
        "instr": [
            "<b>Regulación de sustancias</b> con calendarios de salida: Convenio de Estocolmo (contaminantes orgánicos persistentes), "
            "Convenio de Minamata (mercurio) y reglamentos de registro y restricción de químicos.",
            "<b>Precio a la contaminación:</b> impuestos o mercados de emisiones para gases de larga vida y contaminantes "
            "persistentes; la recaudación financia la sustitución.",
            "<b>Responsabilidad extendida del productor</b> para plásticos, electrónicos y químicos: quien pone el producto en el "
            "mercado paga su fin de vida.",
            "<b>Normas de emisión y mejores técnicas disponibles</b> en industria y agricultura (uso de fertilizantes y plaguicidas).",
            "<b>Financiamiento de I+D y compras públicas</b> de alternativas limpias para crear mercado.",
        ],
        "measure": [
            ("Emisiones de CO<sub>2</sub> por unidad de valor agregado (ODS 9.4.1)", "Global Carbon Budget, AIE",
             "Generación por unidad producida (ppgf)", "Baja ≈ 4 %/año de forma sostenida"),
            ("Emisiones de mercurio y de contaminantes orgánicos persistentes", "Evaluación Mundial del Mercurio (PNUMA); "
             "Plan de Vigilancia del Convenio de Estocolmo", "Generación de contaminación (ppgr)", "Bajan en términos absolutos, no sólo por unidad"),
            ("Residuos peligrosos per cápita y proporción tratada (ODS 12.4.2)", "Estadísticas nacionales, División de Estadística de la ONU",
             "Generación de contaminación (ppgr)", "Generación baja; tratamiento sube"),
            ("Fugas de plásticos al ambiente", "Global Plastics Outlook (OCDE)", "Generación de contaminación (ppgr)",
             "Fugas a la mitad en ≈ 17 años"),
            ("Concentraciones ambientales: COP en leche materna, mercurio en peces", "Programas de biomonitoreo",
             "Stock de contaminación (ppolx)", "Se estabilizan y bajan, con ≈ 20 años de retraso"),
        ],
        "alert": "La intensidad baja pero el volumen total sube porque la producción crece más rápido que la mejora. El modelo "
                 "sólo cuenta la reducción por unidad producida: lo que importa es la carga total.",
    },
    "res": {
        "scales": [
            ("1–2 %/año", "rtcm 1–2 %", "Ritmo histórico de mejora de la eficiencia energética mundial, del orden de 1–2 %/año."),
            ("2–4 %/año (ref.)", "rtcm 4 %", "Duplicar la tasa histórica: el compromiso de la COP28 para la eficiencia energética "
             "hasta 2030, extendido a materiales (reciclaje, durabilidad, diseño)."),
            ("6–8 %/año", "rtcm 6–8 %", "Economía circular profunda: reutilización y reciclaje como norma, cambio de modelos de "
             "negocio a servicios. Tiene coste de capital alto en el modelo."),
        ],
        "instr": [
            "<b>Ecodiseño y derecho a reparar:</b> durabilidad, reparabilidad y reciclabilidad mínimas por ley.",
            "<b>Contenido reciclado obligatorio</b> en envases, construcción y metales; metas de recuperación de metales críticos.",
            "<b>Impuestos a materiales vírgenes, vertederos e incineración</b>, que abaratan relativamente lo reciclado.",
            "<b>Normas mínimas de eficiencia energética</b> y etiquetado en equipos, edificios y vehículos.",
            "<b>Eliminación de subsidios a combustibles fósiles</b> y compras públicas circulares.",
        ],
        "measure": [
            ("Huella material per cápita y por PIB (ODS 8.4.1 / 12.2.1)", "Base de datos de flujos de materiales del IRP (PNUMA)",
             "Recursos usados por persona (nruf)", "Baja ≈ 2–4 %/año sin caída del bienestar"),
            ("Consumo material doméstico por PIB (ODS 12.2.2)", "IRP (PNUMA), oficinas de estadística", "Uso de recursos (nrur)",
             "Baja de forma sostenida"),
            ("Intensidad energética primaria (ODS 7.3.1)", "AIE, Banco Mundial (seguimiento del ODS 7)", "Recursos por unidad producida",
             "Mejora ≥ 4 %/año, como pide la COP28"),
            ("Tasa de uso circular de materiales", "Eurostat y equivalentes nacionales", "Recursos usados por persona (nruf)",
             "Sube año a año"),
            ("Reservas / producción de minerales clave", "USGS Mineral Commodity Summaries", "Recursos restantes (nrfr)",
             "No cae más rápido que la mejora de eficiencia"),
        ],
        "alert": "Efecto rebote: la eficiencia abarata el uso y el consumo total sube. Medir siempre el uso absoluto, no sólo la intensidad.",
    },
    "agro": {
        "scales": [
            ("1–2 %/año", "lytcrm 1–2 %", "El crecimiento reciente del rendimiento de cereales: mejoras graduales con la inversión actual."),
            ("2–3 %/año", "lytcrm 2–3 %", "El ritmo de la Revolución Verde. Sola, es la mejor dosis (más allá aparece la trampa de la abundancia)."),
            ("4–6 %/año (ref.)", "lytcrm 4–6 % + suelos", "Una segunda revolución agrícola sostenida durante décadas, siempre "
             "con control de la contaminación ≥ 3 %/año y transición demográfica."),
            ("8 %/año", "lytcrm 8 %", "Cota del modelo, sin precedente histórico."),
        ],
        "instr": [
            "<b>Inversión pública en I+D agrícola y extensión</b> (sistemas nacionales y CGIAR): mejoramiento genético, variedades "
            "resistentes al calor y la sequía.",
            "<b>Agua:</b> riego eficiente, cosecha de agua, manejo de cuencas.",
            "<b>Uso eficiente de fertilizantes:</b> agricultura de precisión y dosis según el suelo, para subir rendimiento sin "
            "subir la contaminación por nitrógeno.",
            "<b>Menos pérdidas y desperdicio de alimentos</b> (ODS 12.3): almacenamiento, cadena de frío, mercados.",
            "<b>Pequeños productores:</b> crédito, seguros y acceso a mercados.",
            "<b>Suelos:</b> agricultura de conservación (labranza mínima, cultivos de cobertura, rotaciones), agroforestería, "
            "control de erosión, restauración de tierras degradadas, pagos por servicios ecosistémicos y metas de neutralidad "
            "en la degradación de tierras (CNULD).",
        ],
        "measure": [
            ("Rendimiento de cereales (t/ha)", "FAOSTAT", "Rendimiento de la tierra (ly)", "Crece al ritmo de la escala elegida"),
            ("Productividad total de los factores agrícola", "USDA ERS (International Agricultural Productivity)",
             "Tecnología de rendimiento (lyt)", "Crece sin más insumos por hectárea"),
            ("Gasto en I+D agrícola como % del PIB agrícola", "ASTI (IFPRI)", "Esfuerzo tecnológico (lytcrm)", "Se duplica o triplica"),
            ("Prevalencia de subalimentación (ODS 2.1.1) y suministro de calorías", "FAO (SOFI), FAOSTAT",
             "Alimento per cápita (fpc)", "Nunca baja, ni durante el valle"),
            ("Proporción de tierra degradada (ODS 15.3.1) y carbono orgánico del suelo", "CNULD, FAO",
             "Fertilidad del suelo (lfert)", "La degradación se detiene antes de 2040"),
            ("Índice de pérdida de alimentos (ODS 12.3.1)", "FAO", "Pérdidas entre cosecha y consumo", "Baja"),
        ],
        "alert": "El rendimiento sube pero la población, la contaminación por nitrógeno y plaguicidas, o la expansión agrícola "
                 "suben más rápido: es la «dosis peligrosa» de este capítulo. Agro-tecnología sin control de la contaminación es "
                 "la combinación más arriesgada del estudio.",
    },
    "fam": {
        "scales": [
            ("3–2,5 hijos", "dcfszpg 3–2,5", "Tendencias actuales en países de fecundidad alta, con mejoras graduales de acceso."),
            ("2,25–2 hijos (ref.)", "dcfszpg 2", "Fecundidad de reemplazo en una generación: acelerar lo que ya funciona en educación y salud reproductiva."),
            ("1,75 hijos", "dcfszpg 1,75", "Niveles actuales de buena parte de Europa y América. El óptimo de sostenibilidad del estudio."),
            ("1,5 hijos", "dcfszpg 1,5", "Niveles de Asia oriental y el sur de Europa: contracción y envejecimiento rápidos."),
        ],
        "instr": [
            "Medidas voluntarias y basadas en derechos; la coerción queda fuera. En el modelo, «control natal plenamente eficaz» "
            "significa acceso pleno, no obligación.",
            "<b>Educación secundaria completa para las niñas</b> y fin del matrimonio infantil.",
            "<b>Acceso universal a anticoncepción moderna</b> y salud reproductiva (en la línea de FP2030).",
            "<b>Menos mortalidad infantil:</b> cuando los hijos sobreviven, las familias desean menos.",
            "<b>Empleo y autonomía económica de las mujeres.</b>",
            "<b>Pensiones y protección social</b>, para que la vejez no dependa del número de hijos.",
        ],
        "measure": [
            ("Tasa global de fecundidad", "UN World Population Prospects", "Fecundidad total (tf)", "Converge a ≈ 2 en una generación"),
            ("Tamaño ideal de familia declarado", "Encuestas DHS y MICS", "Tamaño de familia deseado (dcfs): es literalmente la variable del modelo",
             "Baja hacia 2"),
            ("Demanda de planificación familiar satisfecha con métodos modernos (ODS 3.7.1)", "UNFPA, DHS",
             "Eficacia del control natal (fce)", "Cerca del 100 %"),
            ("Tasa de natalidad adolescente (ODS 3.7.2)", "UN WPP, DHS", "Natalidad", "Baja"),
            ("Finalización de secundaria de las niñas (ODS 4.1.2)", "UNESCO UIS", "Motor del tamaño deseado", "Sube"),
            ("Mortalidad de menores de 5 años (ODS 3.2.1)", "UN IGME", "Esperanza de vida (le)", "Baja"),
        ],
        "alert": "Fecundidad sostenida por debajo de 1,5: la población se contrae más de 20 % y envejece rápido. El objetivo útil "
                 "es 1,75–2 hijos.",
    },
    "suff": {
        "scales": [
            ("500", "iopcd 500", "Crecimiento industrial intensivo: más inversión que hoy. Empeora el destino."),
            ("450–400", "iopcd 450–400", "Estabilización suave de la producción material per cápita."),
            ("350 (ref.)", "iopcd 350", "Congelar la producción industrial per cápita en el nivel actual del modelo y crecer en "
             "servicios, salud, educación y cuidados."),
            ("300–250", "iopcd 300–250", "Reducir el consumo material un 15–30 % en las economías de alta huella. Más allá de 300 "
             "no mejora el destino y baja el bienestar."),
        ],
        "instr": [
            "<b>Reorientar gasto público e inversión</b> hacia salud, educación, cuidados y cultura.",
            "<b>Reducción de la jornada laboral:</b> repartir las ganancias de productividad como tiempo libre y no como más producción.",
            "<b>Impuestos progresivos al consumo de alta huella</b> (vuelos frecuentes, vehículos pesados, bienes de lujo).",
            "<b>Vida útil mínima de productos</b>, garantías extendidas, reparación y modelos de uso compartido o alquiler.",
            "<b>Límites a la publicidad</b> de bienes de alta huella.",
            "<b>Presupuestos de bienestar</b> e indicadores más allá del PIB para orientar la política económica.",
        ],
        "measure": [
            ("Valor agregado manufacturero per cápita (ODS 9.2.1)", "ONUDI, Banco Mundial", "Producción industrial per cápita (iopc)",
             "Estable al nivel elegido"),
            ("Huella material per cápita (ODS 12.2.1)", "IRP (PNUMA)", "Recursos por persona", "Baja en los países de alta huella"),
            ("Peso de los servicios en el consumo de los hogares", "Cuentas nacionales", "Servicios per cápita (sopc)", "Sube"),
            ("Esperanza de vida saludable, IDH, indicadores de bienestar", "OMS, PNUD", "Bienestar (hwi, le)",
             "Siguen mejorando: si bajan, la suficiencia se está volviendo recesión"),
            ("Horas trabajadas por persona", "OIT, OCDE", "Uso del tiempo", "Bajan sin perder ingreso real"),
        ],
        "alert": "Estancamiento con caída de la salud, el empleo o los servicios: la suficiencia se convierte en recesión. La "
                 "palanca sólo funciona si el bienestar se mide y sigue subiendo.",
    },
}


def impl_flowables(lv):
    d = IMPL[lv]
    out = [para("Implementación y medición", "eyebrow"), para("Cómo se implementa y cómo se mide", "h2b"),
           para("Formas de implementarla según la escala", "h3"),
           table([["Escala", "En el modelo", "Cómo se vería en la práctica"]] + [list(r) for r in d["scales"]],
                 [26 * mm, 30 * mm, TEXT_W - 56 * mm], zebra=True),
           para("Instrumentos", "h3")]
    out += bullets(d["instr"])
    out += [para("Cómo se mediría", "h3"),
            table([["Indicador", "Fuente", "Variable del modelo", "Señal de que va a la escala de referencia"]] +
                  [[Paragraph(r[0], S["cell"]), r[1], r[2], r[3]] for r in d["measure"]],
                  [52 * mm, 40 * mm, 38 * mm, TEXT_W - 130 * mm], zebra=True),
            Spacer(1, 6), callout(d["alert"], "Señal de alerta")]
    return out


if __name__ == "__main__":
    build()
