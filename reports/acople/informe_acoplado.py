"""Construye ``modelo_acoplado.pdf``: la implementación del acople World3 ↔ sociedad, su calibración, su validación
histórica y lo que produce en el ensemble de 500 mundos.

    PYTHONPATH=. python reports/acople/calibrar.py      # parámetros + validación (≈ 1 min, requiere el Atlas)
    PYTHONPATH=. python reports/acople/simular.py       # experimentos (≈ 1 min)
    PYTHONPATH=. python reports/acople/informe_acoplado.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "escenarios"))
warnings.simplefilter("ignore")

import informe_pdf as K  # noqa: E402
import graficos as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import CondPageBreak, NextPageTemplate, PageBreak, Spacer  # noqa: E402

C = json.loads((HERE / "calibracion.json").read_text(encoding="utf-8"))
A = json.loads((HERE / "acoplado.json").read_text(encoding="utf-8"))
FIG = HERE / "figuras"
FIG.mkdir(exist_ok=True)
OUT = HERE / "modelo_acoplado.pdf"
para, bullets, table, img, callout, kpis, pct, num = K.para, K.bullets, K.table, K.img, K.callout, K.kpis, K.pct, K.num
TW = K.TEXT_W
EX = A["experiments"]
YS = np.array(A["years_soc"])
YW = np.array(A["years_w3"])
EV = {"conflict_onset": ("Inicios de conflicto civil", "#b3302b"), "coup": ("Intentos de golpe", "#eb6834"),
      "dem_breakdown": ("Quiebres democráticos", "#6250d6"), "sovdebt_onset": ("Crisis de deuda", "#2a78d6"),
      "irregular_exit": ("Salidas irregulares", "#eda100")}
HZ = C["hazards"]
VARN = {"lny": "ingreso (ln)", "growth": "crecimiento", "depth": "profundidad de la caída", "youth": "jóvenes 15-24",
        "conf10": "conflicto en 10 años", "ln_dur": "duración (ln)", "coup5": "golpe en 5 años",
        "conflict": "en conflicto", "dem": "democracia", "ln_debt": "deuda (ln 1+d)", "crisis10": "crisis en 10 años"}


def sgn(v, d=1):
    return ("+" if v >= 0 else "−") + num(abs(v), d)


def med(x):
    return float(np.asarray(x)[1]) if np.ndim(x) == 1 else np.asarray(x)[1]


# ── figuras ──
def fig_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.0))
        ax.text(x + 0.15, y + h - 0.28, title, fontsize=8.2, fontweight="bold", color=G.INK, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.68 - 0.3 * i, ln, fontsize=6.6, color=G.INK2, va="top")

    box(0.05, 0.5, 2.55, 3.7, "World3-03 (L0)", ["población · capital industrial", "y de servicios · comida",
                                                 "recursos · contaminación", "", "500 mundos filtrados", "por la población",
                                                 "de 2000 y 2020"], "#eef5f2", G.ACCENT)
    box(7.35, 0.5, 2.6, 3.7, "Sociedad: 180 países", ["ingreso propio (AR(1))", "deuda / PIB (estimada)",
                                                      "población y jóvenes: WPP", "", "9 riesgos logit anuales:",
                                                      "conflicto, golpes, democracia,", "deuda, salidas irregulares"],
        "#f4f1fb", "#6250d6")
    arr = dict(arrowstyle="-|>", mutation_scale=12, lw=1.4)
    xa, xb, xm = 2.75, 7.2, 4.975
    ax.add_patch(FancyArrowPatch((xa, 3.4), (xb, 3.4), color=G.ACCENT, **arr))
    ax.text(xm, 3.62, "IDA · producto mundial per cápita (io + so) / pop", ha="center", fontsize=6.6, color=G.ACCENT)
    ax.text(xm, 3.1, "cada país lo sigue con su ruido; Σ PIB = PIB de World3", ha="center", fontsize=6, color=G.INK3)
    ax.add_patch(FancyArrowPatch((xb, 2.1), (xa, 2.1), color="#b3302b", **arr))
    ax.text(xm, 2.32, "VUELTA 1 · capital destruido", ha="center", fontsize=6.6, color="#b3302b")
    ax.text(xm, 1.8, "huella de cada conflicto, golpe o crisis de deuda", ha="center", fontsize=6, color=G.INK3)
    ax.text(xm, 1.56, "(proyecciones locales, h = 0..10) → fracción del PIB mundial", ha="center", fontsize=6, color=G.INK3)
    ax.add_patch(FancyArrowPatch((xb, 0.95), (xa, 0.95), color="#2a78d6", **arr))
    ax.text(xm, 1.17, "VUELTA 2 · capacidad de política", ha="center", fontsize=6.6, color="#2a78d6")
    ax.text(xm, 0.65, "flujos de tecnología × (1 − caída del gasto tras eventos)", ha="center", fontsize=6, color=G.INK3)
    ax.text(5.0, 4.45, "Un paso por año desde 2020 (separación de operadores dentro del integrador de World3)",
            ha="center", fontsize=7, color=G.INK2)
    return G.save(fig, FIG / "acople_diagrama.png")


def fig_auc():
    keys = [k for k in HZ if "auc_oos" in HZ[k]]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    y = np.arange(len(keys))[::-1]
    ax.barh(y + 0.18, [HZ[k]["auc_oos_lny"] for k in keys], height=0.34, color="#cbc5b8", label="sólo el ingreso")
    ax.barh(y - 0.18, [HZ[k]["auc_oos"] for k in keys], height=0.34, color=G.ACCENT, label="modelo del acople")
    for yi, k in zip(y, keys):
        ax.text(HZ[k]["auc_oos"] + 0.005, yi - 0.18, num(HZ[k]["auc_oos"], 2), va="center", fontsize=6.4, color=G.INK)
    ax.axvline(0.5, color=G.INK3, lw=0.8, ls=":")
    ax.set_xlim(0.4, 0.86)
    ax.set_yticks(y, [HZ[k]["name"] for k in keys], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC fuera de muestra (entrena 1960-1999, predice 2000-2019); 0,5 = azar", fontsize=6.8)
    ax.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout()
    return G.save(fig, FIG / "acople_auc.png")


def fig_impulses():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    names = {"conflict_onset": ("Inicio de conflicto", "#b3302b"), "coup_success": ("Golpe exitoso", "#eb6834"),
             "sovdebt_onset": ("Crisis de deuda", "#2a78d6")}
    for ax, (k, (nm, col)) in zip(axes, names.items()):
        r = C["lp"][k]
        h = np.array([x["h"] for x in r])
        b = np.array([x["b"] for x in r])
        se = np.array([x["se"] for x in r])
        ax.fill_between(h, b - 1.96 * se, b + 1.96 * se, color=col, alpha=0.16, lw=0)
        ax.plot(h, b, "-o", color=col, ms=3, mec="white", mew=0.5)
        ax.axhline(0, color=G.INK3, lw=0.7)
        ax.set_title(nm, fontsize=7.6)
        ax.set_xlabel("años después", fontsize=6.6)
        ax.text(10, b[-1] - 1.2, f"{sgn(b[-1])} %", ha="right", fontsize=6.6, color=col)
    axes[0].set_ylabel("PIB per cápita, %\nfrente a la trayectoria previa", fontsize=6.6)
    fig.tight_layout()
    return G.save(fig, FIG / "acople_huellas.png")


def fig_history():
    H = C["history"]
    yrs = np.array(H["years"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    keys = [("conflict_onset", "Inicios de conflicto"), ("coup", "Intentos de golpe"), ("dem_breakdown", "Quiebres democráticos"),
            ("sovdebt_onset", "Crisis de deuda"), ("in_conflict", "Países en conflicto"), ("demo_countries", "Democracias (fracción)")]
    for ax, (k, nm) in zip(axes.flat, keys):
        s = H["sim"][k]
        ax.fill_between(yrs, s["p10"], s["p90"], color=G.ACCENT, alpha=0.16, lw=0)
        ax.plot(yrs, s["mean"], color=G.ACCENT, lw=1.6, label="simulado (media, 10-90 %)")
        o = np.array(H["obs"][k], float)
        ax.plot(yrs, o, "o", color=G.INK, ms=2.8, label="observado")
        ax.set_title(nm, fontsize=7.6)
        ax.tick_params(labelsize=6.2)
    axes[0, 0].legend(fontsize=5.8, loc="upper left")
    fig.tight_layout()
    return G.save(fig, FIG / "acople_historia.png")


def fig_valley_ratios():
    vr, emp = C["history"]["valley_ratios"], C["history"]["valley_ratios_empirical"]
    keys = ["sovdebt_onset", "dem_breakdown", "coup", "irregular_exit", "conflict_onset"]
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        ax.plot([vr[k]["ratio"], emp[k]], [yi, yi], color=G.RULE2 if hasattr(G, "RULE2") else "#d9d6cf", lw=3, zorder=1)
        ax.plot(emp[k], yi, "o", color=G.INK, ms=6, zorder=2, label="historia 1950-2019" if yi == y[0] else None)
        ax.plot(vr[k]["ratio"], yi, "o", color=G.ACCENT, ms=6, zorder=3, label="modelo simulado 2000-2019" if yi == y[0] else None)
    ax.axvline(1, color=G.INK3, lw=0.8, ls=":")
    ax.set_yticks(y, [EV[k][0] if k in EV else k for k in keys], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("razón de la tasa en los 5 años tras una caída del PIB ≥ 10 % (sin guerra) frente a la tasa sin caída", fontsize=6.6)
    ax.legend(fontsize=6.4, loc="lower right")
    fig.tight_layout()
    return G.save(fig, FIG / "acople_valles.png")


def fig_decades():
    """Eventos por década en el colapso de inercia, frente a un mundo sin valle y a la historia reciente."""
    decs = list(EX["inercia"]["acople"]["decades"])
    hist = A["historia"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharex=True)
    for ax, k in zip(axes, ["conflict_onset", "coup", "dem_breakdown", "sovdebt_onset"]):
        nm, col = EV[k]
        v1 = [EX["inercia"]["acople"]["decades"][d][k] for d in decs]
        v2 = [A["sin_valle"]["decades"][d][k] for d in decs]
        v3 = [EX["ltg_abund"]["acople"]["decades"][d][k] for d in decs]
        x = np.arange(len(decs))
        ax.plot(x, v1, "-o", color=col, ms=3, mec="white", mew=0.5, label="colapso (inercia, recursos base)")
        ax.plot(x, v3, "-s", color=col, ms=2.6, alpha=0.55, lw=1.1, label="LtG, recursos abundantes")
        ax.plot(x, v2, "--", color=G.INK3, lw=1.1, label="sin valle (+1,5 %/año)")
        ob = np.array(hist["obs"][k], float)
        ax.axhline(np.nanmean(ob) * 10, color=G.INK, lw=0.7, ls=":", label="2000-2019 observado")
        ax.set_title(nm, fontsize=7.2)
        ax.set_xticks(x[::2], [d[:4] for d in decs][::2], fontsize=6)
        ax.tick_params(axis="y", labelsize=6.2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("eventos por década", fontsize=6.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return G.save(fig, FIG / "acople_decadas.png")


def fig_states():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    cases = [("inercia", "#b3302b", "Colapso (inercia)"), ("inercia_abund", "#eda100", "Inercia, abundantes"),
             ("ltg_abund", G.ACCENT, "LtG, abundantes")]
    for ax, (k, tl) in zip(axes, [("growth_world", "Crecimiento del producto mundial p. c."),
                                  ("demo_pop", "Población en democracia"), ("valley10", "Población en un valle ≥ 10 %")]):
        for key, col, nm in cases:
            q = np.array(EX[key]["acople"]["soc"][k]).T
            if k == "growth_world":
                q = q * 100
            G.band_line(ax, YS, q, col, label=nm, marker=None, lw=1.4, alpha=0.12)
        if k == "growth_world":
            q = np.array(A["sin_valle"]["soc"][k]).T * 100
            ax.plot(YS, q[:, 1], "--", color=G.INK3, lw=1, label="sin valle")
            ax.axhline(0, color=G.INK3, lw=0.6)
            ax.set_ylabel("% al año", fontsize=6.4)
        else:
            q = np.array(A["sin_valle"]["soc"][k]).T
            ax.plot(YS, q[:, 1], "--", color=G.INK3, lw=1, label="sin valle")
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_title(tl, fontsize=7.2)
        ax.tick_params(labelsize=6.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return G.save(fig, FIG / "acople_estados.png")


def group_sizes():
    P = json.loads((HERE.parents[1] / "world4" / "data" / "social_params.json").read_text(encoding="utf-8"))
    lny0 = np.array([r[-1] for r in P["init"]["lny"]])
    gi = np.digitize(lny0, [np.log(4000), np.log(20000)])
    return {name: int((gi == i).sum()) for i, name in enumerate(("bajo", "medio", "alto"))}


GS = group_sizes()


def fig_groups():
    g = EX["inercia"]["acople"]["groups"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=False)
    cols = {"bajo": "#b3302b", "medio": "#eda100", "alto": "#1c5cab"}
    names = {"bajo": f"ingreso bajo en 2019 (< 4 mil USD; {GS['bajo']} países)",
             "medio": f"medio (4-20 mil; {GS['medio']})", "alto": f"alto (> 20 mil; {GS['alto']})"}
    for ax, (k, tl) in zip(axes, [("conflict_onset", "Inicios de conflicto"), ("dem_breakdown", "Quiebres democráticos"),
                                  ("sovdebt_onset", "Crisis de deuda"), ("demo_pop", "Población en democracia")]):
        for grp, col in cols.items():
            x = np.array(g[grp][k]) * (1 if k == "demo_pop" else 100 / GS[grp])
            v = np.convolve(x, np.ones(10) / 10, mode="valid")
            ax.plot(YS[9:], v, color=col, lw=1.5, label=names[grp])
        ax.set_title(tl, fontsize=7.2)
        ax.tick_params(labelsize=6.2)
        ax.set_ylim(bottom=0)
        if k == "demo_pop":
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("eventos por 100 países y año\n(media móvil de 10 años)", fontsize=6.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return G.save(fig, FIG / "acople_grupos.png")


def fig_feedback():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    cases = [("inercia_abund", "Inercia, abundantes"), ("ltg_abund", "LtG, abundantes"), ("estres_doble", "LtG, estrés doble")]
    for ax, (k, nm) in zip(axes, cases):
        a = np.array(EX[k]["ida"]["w3"]["le"]).T
        b = np.array(EX[k]["acople"]["w3"]["le"]).T
        G.band_line(ax, YW, a, G.INK3, label="sólo ida", marker=None, lw=1.3, alpha=0.1)
        G.band_line(ax, YW, b, "#b3302b", label="con vuelta", marker=None, lw=1.3, alpha=0.1)
        ax.set_xlim(2000, 2100)
        ax.set_title(nm, fontsize=7.2)
        ax.tick_params(labelsize=6.2)
    axes[0].set_ylabel("esperanza de vida (años)", fontsize=6.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return G.save(fig, FIG / "acople_vuelta.png")


# ── documento ──
class Doc(K.Doc):
    def __init__(self, path):
        super().__init__(path)
        self.title = "El modelo acoplado"
        self.subject = "World3-03 acoplado en las dos direcciones a un módulo social de 180 países calibrado con datos"

    def draw_body(self, c, doc):
        c.saveState()
        c.setFont("Mono", 6.8)
        c.setFillColor(K.INK3)
        c.drawString(K.MARGIN, K.H_PAGE - 12 * mm, "EL MODELO ACOPLADO · QUASAR PHY-WORLD")
        c.drawRightString(K.W_PAGE - K.MARGIN, K.H_PAGE - 12 * mm, self.section.upper()[:60])
        c.setStrokeColor(K.RULE)
        c.setLineWidth(0.5)
        c.line(K.MARGIN, K.H_PAGE - 13.6 * mm, K.W_PAGE - K.MARGIN, K.H_PAGE - 13.6 * mm)
        c.drawString(K.MARGIN, 10 * mm, "Modos de comportamiento de un modelo calibrado con datos, no pronósticos")
        c.drawRightString(K.W_PAGE - K.MARGIN, 10 * mm, f"{doc.page}")
        c.restoreState()

    def draw_cover(self, c, doc):
        c.saveState()
        c.setFillColor(colors.HexColor("#0f1c19"))
        c.rect(0, 0, K.W_PAGE, K.H_PAGE, stroke=0, fill=1)
        # quiebres democráticos por década: colapso frente a un mundo sin valle
        decs = list(EX["inercia"]["acople"]["decades"])
        v1 = [EX["inercia"]["acople"]["decades"][d]["dem_breakdown"] for d in decs]
        v2 = [A["sin_valle"]["decades"][d]["dem_breakdown"] for d in decs]
        x0, x1, y0, y1 = K.MARGIN, K.W_PAGE - K.MARGIN, 62 * mm, 128 * mm
        top = max(v1)
        w = (x1 - x0) / len(decs)
        for i, (a, b) in enumerate(zip(v1, v2)):
            c.setFillColor(colors.HexColor("#3a4744"))
            c.rect(x0 + i * w + 6, y0, w / 2 - 8, (y1 - y0) * b / top, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#46c49c"))
            c.rect(x0 + i * w + w / 2, y0, w / 2 - 8, (y1 - y0) * a / top, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#8a9692"))
            c.setFont("Mono", 6.6)
            c.drawCentredString(x0 + i * w + w / 2, y0 - 10, decs[i])
        c.drawString(x0, y1 + 26, "QUIEBRES DEMOCRÁTICOS POR DÉCADA · VERDE: EL VALLE DE WORLD3 · GRIS: UN MUNDO SIN VALLE")
        c.setFillColor(colors.HexColor("#46c49c"))
        c.setFont("Mono", 8)
        c.drawString(K.MARGIN, K.H_PAGE - 30 * mm, "QUASAR PHY-WORLD · IMPLEMENTACIÓN DEL ACOPLE · OCTUBRE 2026")
        c.setFillColor(colors.HexColor("#e5ebe8"))
        c.setFont("Serif", 50)
        c.drawString(K.MARGIN, K.H_PAGE - 60 * mm, "El modelo")
        c.setFont("Serif-It", 50)
        c.setFillColor(colors.HexColor("#46c49c"))
        c.drawString(K.MARGIN, K.H_PAGE - 78 * mm, "acoplado")
        c.setFillColor(colors.HexColor("#b7c2be"))
        c.setFont("Sans", 11.5)
        for i, ln in enumerate(["World3 y 180 países que se retroalimentan: el valle biofísico mueve",
                                "conflictos, golpes, democracias y deudas, y estos le devuelven",
                                "capital destruido y capacidad de política perdida. Calibrado y validado con datos."]):
            c.drawString(K.MARGIN, K.H_PAGE - 94 * mm - i * 16, ln)
        c.setFont("Sans", 7.6)
        c.setFillColor(colors.HexColor("#8a9692"))
        c.drawString(K.MARGIN, 22 * mm, "world4/core/social.py · reports/acople/calibrar.py · simular.py · panel del Atlas 1950-2019 · UN WPP 2024")
        c.drawString(K.MARGIN, 17 * mm, f"Generado {A['meta']['generated']} · semilla {A['meta']['seed']} · {A['meta']['members']} mundos · "
                                        f"{len(A['iso3'])} países")
        c.restoreState()


def build():
    print("figuras...")
    f_dia, f_auc, f_imp, f_his, f_val = fig_diagram(), fig_auc(), fig_impulses(), fig_history(), fig_valley_ratios()
    f_dec, f_sta, f_grp, f_fb = fig_decades(), fig_states(), fig_groups(), fig_feedback()
    H = C["history"]
    T = H["totals"]
    vr, vre = H["valley_ratios"], H["valley_ratios_empirical"]
    inc, sv = EX["inercia"]["acople"]["decades"], A["sin_valle"]["decades"]
    ratio = lambda k: inc["2090s"][k] / sv["2090s"][k]  # noqa: E731
    ltg = EX["ltg_abund"]
    imp = {k: C["lp"][k][-1]["b"] for k in ("conflict_onset", "coup_success", "sovdebt_onset")}
    cap = C["capacity"]
    gl = lambda k: 1 - EX[k]["acople"]["soc"]["gdp_index"][1][-1]  # noqa: E731

    story = [NextPageTemplate("body"), PageBreak()]
    # ── resumen ──
    story += [para("Resumen", "eyebrow"), para("Lo que se implementó y lo que muestra", "h2"),
              para("El informe anterior encontró en 70 años de datos cómo se conectan la economía y la estabilidad política. Aquí esas "
                   "dinámicas se convierten en un modelo: <b>World3 y 180 países que se retroalimentan cada año</b>. El valle "
                   "biofísico de World3 llega a los países como ingreso; los países responden con conflictos, golpes, quiebres "
                   "democráticos y crisis de deuda; y esos eventos le devuelven a World3 capital destruido y menos capacidad para "
                   "aplicar políticas. Todo número sale de los datos; cada supuesto está declarado.", "lede"),
              kpis([(f"×{num(ratio('dem_breakdown'), 1)}", "quiebres democráticos en la década de 2090 en el colapso, frente a un mundo sin valle"),
                    (f"×{num(ratio('conflict_onset'), 1)}", "inicios de conflicto civil en la misma comparación"),
                    (f"{num(T['coup']['sim'], 0)} / {num(T['coup']['obs'], 0)}", "intentos de golpe simulados / observados, 2000-2019 (validación)"),
                    (f"{sgn(ltg['paired']['le_2100'][1])} años", "esperanza de vida 2100 por la vuelta, paquete LtG con recursos abundantes")]),
              Spacer(1, 8)]
    story += bullets([
        f"<b>El modelo reproduce la historia reciente.</b> Arrancando del estado de 1999 y siguiendo el PIB mundial observado, simula "
        f"{num(T['coup']['sim'], 0)} intentos de golpe en 2000-2019 (observados: {num(T['coup']['obs'], 0)}), "
        f"{num(T['coup_success']['sim'], 0)} golpes exitosos ({num(T['coup_success']['obs'], 0)}), "
        f"{num(T['conflict_onset']['sim'], 0)} inicios de conflicto ({num(T['conflict_onset']['obs'], 0)}) y "
        f"{num(T['dem_breakdown']['sim'], 0)} quiebres democráticos ({num(T['dem_breakdown']['obs'], 0)}).",
        f"<b>El valle de World3 es, en la sociedad, una ola de inestabilidad.</b> En el colapso de inercia, hacia la década de 2090 hay "
        f"{num(inc['2090s']['dem_breakdown'], 0)} quiebres democráticos por década (sin valle: {num(sv['2090s']['dem_breakdown'], 0)}), "
        f"{num(inc['2090s']['sovdebt_onset'], 0)} crisis de deuda ({num(sv['2090s']['sovdebt_onset'], 0)}), "
        f"{num(inc['2090s']['conflict_onset'], 0)} inicios de conflicto ({num(sv['2090s']['conflict_onset'], 0)}) y "
        f"{num(inc['2090s']['coup'], 0)} intentos de golpe ({num(sv['2090s']['coup'], 0)}). La población que vive en democracia baja "
        f"de {pct(inc['2020s']['demo_pop'])} a {pct(inc['2090s']['demo_pop'])}; sin valle, sube a {pct(sv['2090s']['demo_pop'])}.",
        "<b>Lo que más sube son la deuda y la democracia, no la guerra.</b> Como en los datos: las caídas del ingreso rompen primero las "
        "finanzas públicas y los regímenes; la guerra civil crece por el empobrecimiento acumulado, más lento.",
        f"<b>La vuelta hacia World3 es real pero de segundo orden.</b> Los eventos destruyen entre {pct(gl('inercia_abund'))} y "
        f"{pct(gl('inercia'))} del PIB mundial acumulado a 2100, pero eso mueve la producción industrial acumulada de World3 sólo "
        f"{sgn(ltg['paired']['io_2020_2100'][1] * 100)} % y la esperanza de vida de 2100 {sgn(ltg['paired']['le_2100'][1])} años (paquete "
        f"LtG); en la prueba de estrés más dura, {sgn(EX['estres_doble']['paired']['le_2100'][1])} años. Cambia el destino de menos de "
        f"{pct(max(e['paired']['destiny_changed'] for e in EX.values()) + 0.005)} de los mundos.",
        "<b>Por qué tan poco:</b> los eventos se concentran en países pobres, que pesan poco en el PIB mundial; y la huella medida de "
        "cada evento es de unos pocos puntos. Además, en World3 menos capital también es menos contaminación y menos extracción: con "
        "recursos abundantes y sin política, la vuelta incluso <i>mejora</i> un poco la esperanza de vida de 2100. La sociedad sufre el "
        "valle mucho más de lo que lo agrava.",
    ])
    story.append(PageBreak())

    # ── arquitectura ──
    story += [para("Arquitectura", "eyebrow"), para("Cómo está construido el acople", "h2"),
              para("El L0 regional, que sería la base ideal, necesita datos de FAOSTAT y Zenodo que no se pudieron descargar. Esta "
                   "implementación es la «ruta 3» del informe de factibilidad, pero en lugar de unir World3 con el Atlas completo, lo une "
                   "con un módulo social nuevo, más pequeño, cuyas ecuaciones salen directamente de los datos y se validan contra ellos."),
              img(f_dia), para("Figura 1. Los dos sentidos del acople. El módulo corre dentro del integrador de World3: en cada año calendario "
                               "lee el producto mundial, avanza la sociedad un año y devuelve los ajustes.", "cap")]
    story += [para("La ida: World3 → sociedad", "h3")]
    story += bullets([
        "<b>Una sola vía: el ingreso.</b> World3 da el producto mundial per cápita, (io + so) / pop. Cada país lo sigue con un ruido "
        f"propio, un AR(1) en el crecimiento (ρ = {num(C['noise']['rho'], 2)}) cuyas innovaciones se remuestrean de los residuos "
        f"históricos 1990-2019 por grupo de ingreso (DE {num(C['noise']['sd'][0] * 100, 1)}, {num(C['noise']['sd'][1] * 100, 1)} y "
        f"{num(C['noise']['sd'][2] * 100, 1)} %), sin los años de conflicto, golpe o crisis. Cada año se normaliza para que la suma de "
        "los PIB nacionales sea el PIB de World3.",
        "<b>Demografía de datos:</b> la población de cada país y el peso de los jóvenes de 15-24 años en la población adulta vienen de "
        "UN WPP 2024 (estimaciones hasta 2023, variante media después).",
        "<b>Deuda pública:</b> Δ ln(1 + deuda/PIB) = "
        f"{num(C['debt']['const'], 3)} {sgn(C['debt']['growth'], 3)}·crecimiento {sgn(C['debt']['ln_debt'], 3)}·ln(1 + deuda/PIB) "
        f"(n = {num(C['debt']['n'], 0)}): una recesión sube la deuda y la deuda revierte lentamente.",
        "<b>Nueve riesgos anuales</b> (logit agrupado 1960-2019, efectos de década, se proyecta con el nivel de la década de 2010): "
        "inicio y fin de conflicto civil, intento de golpe y su éxito, salida irregular del líder, quiebre democrático, democratización, "
        "inicio y fin de crisis de deuda. Los eventos se sortean con números aleatorios comunes, así que dos escenarios difieren sólo por "
        "su efecto.",
    ])
    story += [para("La vuelta: sociedad → World3", "h3")]
    story += bullets([
        f"<b>Capital destruido.</b> Cada inicio de conflicto, golpe exitoso o crisis de deuda deja en el PIB del país su huella medida "
        f"(proyecciones locales de Jordà, año 0 a 10): {sgn(imp['conflict_onset'])} %, {sgn(imp['coup_success'])} % y "
        f"{sgn(imp['sovdebt_onset'])} % a 10 años, y se mantiene después. La fracción del PIB mundial perdida cada año destruye la misma "
        "fracción del capital industrial y de servicios de World3.",
        f"<b>Capacidad de política.</b> Tras una crisis de deuda el gasto público cae ≈ {pct(cap['crisis'])} (−1,5 puntos del PIB sobre "
        f"un gasto medio de {num(cap['spending_mean']['sovdebt_onset'] * 100, 0)} % del PIB), tras un golpe ≈ {pct(cap['coup'])} y durante un "
        f"conflicto ≈ {pct(cap['conflict'])}. La parte del PIB mundial afectada reduce en esa proporción los flujos de las tecnologías "
        "de World3 (rendimiento agrícola, eficiencia de recursos, control de la contaminación) [supuesto: gasto ≈ capacidad].",
        "<b>Lo que no se programó, a propósito:</b> canales «hambre → guerra» o «contaminación → conflicto» (los datos no los sostienen "
        "una vez controlado el ingreso), efectos del conflicto sobre la mortalidad de World3 (−0,7 años en el país, despreciable en el "
        "agregado) y guerras entre Estados.",
    ])
    story.append(PageBreak())

    # ── calibración ──
    story += [para("Calibración", "eyebrow"), para("Los nueve riesgos y su validación fuera de muestra", "h2"),
              para("Cada riesgo se estimó con los datos de 1960-2019. Para saber si predicen, se reestimaron sólo con 1960-1999 y se "
                   "probaron en 2000-2019, años que el modelo no vio. La referencia es un modelo con el ingreso como única variable.")]
    rows = [["Evento (muestra)", "Coeficientes (logit, por unidad)", "n / eventos", "AUC fuera", "Sólo ingreso"]]
    for k, h in HZ.items():
        co = ", ".join(f"{VARN[v]} {sgn(b, 2)}" for v, b in zip(h["vars"], h["coef"])) or f"tasa constante {pct(h.get('rate_1990', h['rate']))}"
        rows.append([f"{h['name']} <font size='6.4' color='#6b7773'>({h['sample']})</font>", co, f"{h['n']:,} / {h['events']}".replace(",", " "),
                     num(h["auc_oos"], 3) if "auc_oos" in h else "—", num(h["auc_oos_lny"], 3) if "auc_oos_lny" in h else "—"])
    story += [table(rows, [44 * mm, 76 * mm, 20 * mm, 16 * mm, TW - 156 * mm], zebra=True), Spacer(1, 4), img(f_auc, TW * 0.92),
              para("Figura 2. Capacidad de predecir 2000-2019 con un modelo entrenado hasta 1999. El intercepto de los años 2000 no se "
                   "conoce en el entrenamiento: se usa el de los noventa.", "cap")]
    story += bullets([
        "<b>Bien:</b> conflicto civil (inicio y fin), golpes, salidas irregulares y crisis de deuda predicen claramente mejor que el azar "
        "y, salvo las salidas irregulares, mejor que el ingreso solo; la deuda es la que más gana (la deuda acumulada y el crecimiento "
        "explican lo que el ingreso no).",
        "<b>Regular:</b> el quiebre democrático predice bien (0,77), pero algo peor que el ingreso solo (0,79): con 60 eventos, los "
        "términos de crecimiento y profundidad añaden ruido fuera de muestra. Se conservan porque son el canal del valle que la "
        "investigación documentó; se declaran.",
        "<b>Mal:</b> la democratización casi no se puede predecir (0,52), y el éxito de un golpe dado el intento no depende de nada "
        f"observable (se usa su tasa de 1990-2019, {pct(HZ['coup_success']['rate_1990'])}). Las crisis de deuda en los datos duran "
        "casi siempre un año.",
        "Selección de variables: se compararon pocas especificaciones anidadas por su AUC fuera de muestra en el quiebre democrático "
        "(se quitaron jóvenes y golpes previos) y en el éxito del golpe; es un uso leve del periodo de prueba, que se declara.",
    ])
    story += [CondPageBreak(70 * mm), para("Las huellas de los eventos en el PIB", "h3"), img(f_imp),
              para("Figura 3. Proyecciones locales del ln PIB per cápita tras cada evento (efectos fijos de país y año, controles de "
                   "crecimiento previo, ingreso y conflicto; IC 95 %). Son las huellas que el modelo aplica, año por año.", "cap"),
              PageBreak()]

    # ── validación ──
    story += [para("Validación", "eyebrow"), para("¿Reproduce el modelo la historia 2000-2019?", "h2"),
              para(f"La prueba: se arranca el módulo social con el estado observado de los {H['countries']} países en 1999 (ingresos de "
                   "1992-1999, conflictos, democracia, deuda e historia de golpes y crisis), se le da el PIB mundial observado año por "
                   f"año y la demografía observada, y se simulan {H['members']} historias. Ningún dato posterior a 1999 entra salvo el PIB "
                   "mundial y la demografía."),
              img(f_his), para("Figura 4. Eventos por año, simulados (media y banda 10-90 % de 200 historias) frente a los observados en los "
                               "mismos países.", "cap")]
    rows = [["Indicador, 2000-2019", "Simulado", "Observado", "Lectura"]]
    read = {"conflict_onset": "10 % alto", "coup": "exacto", "coup_success": "≈", "irregular_exit": "≈ (datos a 2015)",
            "dem_breakdown": "10 % alto", "democratization": "15 % alto", "sovdebt_onset": "alto: la fuente acaba en 2017",
            "in_conflict": "16 % bajo: los conflictos acaban algo rápido", "demo_countries": "≈", "valley10": "×1,5: ruido de 1990-2019"}
    lab = {"conflict_onset": "Inicios de conflicto civil", "coup": "Intentos de golpe", "coup_success": "Golpes exitosos",
           "irregular_exit": "Salidas irregulares", "dem_breakdown": "Quiebres democráticos", "democratization": "Democratizaciones",
           "sovdebt_onset": "Crisis de deuda", "in_conflict": "Países-año en conflicto", "demo_countries": "Fracción de democracias (media)",
           "valley10": "Países-año en un valle ≥ 10 %"}
    for k, v in T.items():
        d = 2 if k == "demo_countries" else 0
        rows.append([lab[k], num(v["sim"], d), num(v["obs"], d), read[k]])
    story += [table(rows, [56 * mm, 22 * mm, 22 * mm, TW - 100 * mm], zebra=True), Spacer(1, 6),
              para("El resultado central del informe de dinámicas", "h3"), img(f_val, TW * 0.95),
              para("Figura 5. Lo que pasa en los 5 años tras una caída del PIB per cápita ≥ 10 % sin guerra previa: en la historia y en "
                   "el modelo.", "cap")]
    story += bullets([
        f"<b>El orden se reproduce:</b> deuda y quiebre democrático suben más (×{num(vr['dem_breakdown']['ratio'], 1)} y "
        f"×{num(vr['sovdebt_onset']['ratio'], 1)}), los golpes algo menos (×{num(vr['coup']['ratio'], 1)}) y el conflicto civil no "
        f"cambia (×{num(vr['conflict_onset']['ratio'], 1)}), como en la historia.",
        f"<b>Las magnitudes se quedan cortas,</b> sobre todo en la deuda (×{num(vr['sovdebt_onset']['ratio'], 1)} frente a "
        f"×{num(vre['sovdebt_onset'], 1)}). Dos razones: el ruido del ingreso, calibrado en 1990-2019, crea más valles pequeños que los "
        "de 2000-2019, que diluyen la razón; y en la historia las caídas reales suelen seguir cayendo, mientras que en el modelo el "
        "ruido revierte. Por eso los experimentos incluyen pruebas de estrés con riesgos ×1,5.",
    ])
    story.append(PageBreak())

    # ── resultados: sociedad ──
    story += [para("Resultados I", "eyebrow"), para("Lo que el valle de World3 le hace a la sociedad", "h2"),
              para("El mismo ensemble de 500 mundos del estudio de escalas. Con la geología base, todos colapsan, con o sin política; "
                   "con recursos abundantes, el paquete Limits to Growth lleva a un tercio de los mundos a una transición sostenible. "
                   "Para separar el efecto del valle del de la demografía (un mundo que envejece tiene menos golpes y menos guerras), "
                   "la misma sociedad se corre también con un mundo que crece 1,5 % al año sin valle."),
              img(f_dec), para("Figura 6. Eventos por década (media de 500 mundos). La línea punteada es la tasa observada en 2000-2019, "
                               "llevada a década.", "cap"),
              img(f_sta), para("Figura 7. El crecimiento mundial que reciben los países y dos estados de la sociedad (mediana y banda 10-90 %).", "cap")]
    rows = [["Década de 2090, eventos por década", "Colapso (inercia)", "Inercia, abundantes", "LtG, abundantes", "Sin valle"]]
    for k in ("conflict_onset", "coup", "dem_breakdown", "sovdebt_onset", "irregular_exit"):
        rows.append([EV[k][0], num(inc["2090s"][k], 0), num(EX["inercia_abund"]["acople"]["decades"]["2090s"][k], 0),
                     num(EX["ltg_abund"]["acople"]["decades"]["2090s"][k], 0), num(sv["2090s"][k], 0)])
    for k, nm in (("demo_pop", "Población en democracia"), ("pop_conflict", "Población en países en conflicto")):
        rows.append([nm, pct(inc["2090s"][k]), pct(EX["inercia_abund"]["acople"]["decades"]["2090s"][k]),
                     pct(EX["ltg_abund"]["acople"]["decades"]["2090s"][k]), pct(sv["2090s"][k])])
    story += [CondPageBreak(60 * mm), table(rows, [58 * mm, 28 * mm, 28 * mm, 28 * mm, TW - 142 * mm], zebra=True), Spacer(1, 6)]
    story += bullets([
        "<b>En un mundo que crece, la inestabilidad baja sola.</b> Sin valle, los golpes y los conflictos nuevos caen a la mitad hacia "
        "2090: los países se enriquecen y envejecen. Es la tendencia de las últimas décadas.",
        "<b>En el colapso, esa tendencia se invierte.</b> Con el producto mundial cayendo ≈ 2,5-3 % al año desde 2030, los quiebres "
        "democráticos se multiplican por cuatro frente al mundo sin valle, las crisis de deuda por dos y los conflictos y golpes por "
        "tres. La democracia retrocede a menos de un cuarto de la población.",
        "<b>Con el paquete LtG y recursos abundantes</b> el valle es más corto (2050-2070): los quiebres democráticos y las crisis de "
        "deuda suben en esas décadas y se quedan en ese nivel después; los conflictos y los golpes siguen bajando, pero menos que sin "
        "valle. La transición tiene un costo político visible y duradero.",
        "<b>La población en países en conflicto baja en todos los casos</b> porque hoy incluye a países muy poblados con conflictos de "
        "baja intensidad (India, Nigeria, Pakistán…) que el modelo deja terminar; los conflictos nuevos ocurren sobre todo en países "
        "pobres y menos poblados.",
    ])
    story += [CondPageBreak(70 * mm), para("Dónde ocurre: por grupo de ingreso", "h3"), img(f_grp),
              para("Figura 8. Colapso de inercia, por el ingreso de cada país en 2019.", "cap")]
    gi = EX["inercia"]["acople"]["groups"]
    r10 = lambda grp, k, i: 100 / GS[grp] * float(np.mean(gi[grp][k][i:i + 10]))  # noqa: E731
    story += bullets([
        f"<b>Los conflictos son de los países pobres:</b> ≈ {num(r10('bajo', 'conflict_onset', 0), 1)} inicios por 100 países y año en "
        f"el grupo de ingreso bajo, frente a {num(r10('alto', 'conflict_onset', 0), 1)} en el alto, en la década de 2020.",
        f"<b>El colapso alcanza a los países ricos por la democracia.</b> En World3 la producción por persona cae a una octava parte "
        f"hacia 2100: el grupo que en 2019 tenía {num(gi['alto']['gdp_pc'][0] / 1000, 0)} mil USD por persona termina con "
        f"≈ {num(gi['alto']['gdp_pc'][-1] / 1000, 1)} mil, el ingreso de un país de renta media-baja de hoy. Sus quiebres democráticos "
        f"pasan de {num(r10('alto', 'dem_breakdown', 0), 1)} a {num(r10('alto', 'dem_breakdown', 70), 1)} por 100 países y año, y la parte "
        f"de su población en democracia baja de {pct(np.mean(gi['alto']['demo_pop'][:10]))} a {pct(np.mean(gi['alto']['demo_pop'][-10:]))}.",
        f"<b>Los países de ingreso medio concentran las crisis de deuda y los quiebres</b> ({num(r10('medio', 'dem_breakdown', 70), 1)} "
        f"quiebres por 100 países y año en la década de 2090): son los que cruzan los umbrales de riesgo de 3-8 mil dólares que "
        "mostraban los datos.",
    ])
    story.append(PageBreak())

    # ── resultados: vuelta ──
    story += [para("Resultados II", "eyebrow"), para("Lo que la sociedad le hace a World3", "h2"),
              para("Cada experimento se corrió dos veces con los mismos números aleatorios: «sólo ida» (la sociedad sigue a World3 sin "
                   "tocarlo; World3 queda idéntico al de los informes anteriores) y «con vuelta». Las diferencias son pareadas, mundo por "
                   "mundo."),
              img(f_fb), para("Figura 9. Esperanza de vida de World3 con y sin la vuelta (mediana y banda 10-90 %).", "cap")]
    rows = [["Experimento", "PIB mundial destruido a 2100", "Producción industrial 2020-2100", "Esperanza de vida 2100", "Población 2100",
             "Destino cambia"]]
    for k, e in EX.items():
        p = e["paired"]
        rows.append([e["name"], pct(1 - e["acople"]["soc"]["gdp_index"][1][-1]), f"{sgn(p['io_2020_2100'][1] * 100)} %",
                     f"{sgn(p['le_2100'][1])} años [{sgn(p['le_2100'][0])}, {sgn(p['le_2100'][2])}]",
                     f"{sgn(p['pop_2100'][1] * 1000, 0)} M", pct(p["destiny_changed"])])
    story += [table(rows, [50 * mm, 24 * mm, 24 * mm, 34 * mm, 18 * mm, TW - 150 * mm], zebra=True),
              para("Medianas de las diferencias pareadas (con vuelta − sólo ida) y, en la esperanza de vida, percentiles 10 y 90. El PIB "
                   "destruido es la pérdida acumulada por los eventos frente a la misma trayectoria sin ellos.", "cap")]
    rows = [["Destinos (con vuelta)"] + [d.split(" ")[0] for d in A["destinies"]]]
    for k in ("inercia_abund", "moderado_abund", "ltg_abund", "estres_doble"):
        rows.append([EX[k]["name"]] + [f"{pct(a)} → {pct(b)}" for a, b in zip(EX[k]["ida"]["destiny"], EX[k]["acople"]["destiny"])])
    story += [Spacer(1, 4), table(rows, [56 * mm] + [(TW - 56 * mm) / 5] * 5, zebra=True),
              para("Probabilidad de cada destino sin → con vuelta. Con recursos base todos los mundos colapsan en los dos casos.", "cap")]
    story += bullets([
        "<b>Los eventos destruyen 7-11 % del PIB mundial a 2100</b> (12-22 % en las pruebas de estrés). Es mucho para las personas de "
        "los países afectados, pero se traduce en sólo 1-4 % menos producción industrial acumulada en World3.",
        "<b>Dos efectos de signo contrario.</b> Menos capital significa menos producción, menos servicios y menos inversión en "
        "tecnología (daño); pero en World3 también significa menos extracción y menos contaminación (freno). Con el paquete LtG domina "
        "el daño: la esperanza de vida de 2100 baja ≈ 0,6 años (2 años en el estrés doble). Sin política y con recursos abundantes "
        "domina el freno: sube ≈ 0,9 años.",
        "<b>La capacidad de política casi no cambia</b> (mínimo ≈ 99,3 %): los países en crisis pesan poco en el PIB mundial, y la caída "
        "de su gasto tras los eventos es de 2-5 %.",
        "<b>Los destinos no cambian.</b> El acople no convierte transiciones en colapsos ni al revés: lo decisivo sigue siendo la "
        "geología y la política. Lo que sí cambia es lo que el destino significa para la gente: el mismo valle, vivido con el doble o "
        "el triple de crisis políticas.",
    ])
    story += [callout("Con lo que dicen los datos, la sociedad sufre el valle de World3 mucho más de lo que lo agrava. El efecto "
                      "dominante va de la biofísica a la política, no al revés. Dos advertencias: los datos sólo tienen crisis locales, "
                      "en las que el resto del mundo seguía comerciando y prestando, y un colapso simultáneo podría tener huellas mayores "
                      "(las pruebas de estrés acotan ese caso). Y el modelo no tiene guerras entre Estados.",
                      "La conclusión, con sus límites"), PageBreak()]

    # ── uso y límites ──
    story += [para("Uso y límites", "eyebrow"), para("Cómo correrlo y qué no hace", "h2"),
              para("Desde Python", "h3"),
              para("<font name='Mono' size='7.8'>from world4.core import social<br/>"
                   "w3, soc = social.run_coupled(n=200, seed=1)&nbsp;&nbsp;&nbsp;&nbsp;# World3 + sociedad, con vuelta<br/>"
                   "w3, soc = social.run_coupled(n=200, feedback=False)&nbsp;&nbsp;# sólo ida<br/>"
                   "soc['dem_breakdown']&nbsp;&nbsp;# (años, mundos): quiebres por año; también coup, sovdebt_onset, demo_pop, loss…<br/>"
                   "soc.groups['medio']['conflict_onset']; soc.country['p_conflict']&nbsp;&nbsp;# por grupo y por país<br/>"
                   "social.run_coupled(..., impulse_scale=2, hazard_scale=1.5)&nbsp;&nbsp;# estrés</font>"),
              para("Con cualquier política o ensemble del estudio de escalas, el gancho se pasa a "
                   "<font name='Mono' size='8'>estudio.run(U, pol, hook=social.Coupler(n))</font>; "
                   "<font name='Mono' size='8'>l0.run(..., hook=...)</font> acepta cualquier módulo externo con la misma firma."),
              para("Reproducir", "h3"),
              para("<font name='Mono' size='8'>PYTHONPATH=. python reports/acople/calibrar.py</font> (estima y valida; requiere el Atlas y "
                   "descarga una vez las proyecciones de WPP de GitHub) · "
                   "<font name='Mono' size='8'>reports/acople/simular.py</font> · "
                   "<font name='Mono' size='8'>reports/acople/informe_acoplado.py</font>. Los parámetros quedan en "
                   "<font name='Mono' size='8'>world4/data/social_params.json</font>, así que el modelo corre sin el Atlas. Pruebas: "
                   "<font name='Mono' size='8'>tests/test_social.py</font> (sin gancho World3 queda bit a bit; con sólo ida también; "
                   "contabilidad del PIB mundial; números comunes)."),
              para("Supuestos y límites", "h3")]
    story += bullets([
        "<b>El reparto del ingreso entre países no tiene tendencia propia:</b> nadie converge ni diverge salvo por el ruido y los eventos. "
        "En un mundo real, Asia y África crecerían más que el promedio.",
        "<b>Todos los países reciben el mismo valle.</b> Sin el L0 regional no hay forma de que la escasez de comida golpee más a unos "
        "que a otros. Es la limitación principal; la ruta 2 la resolvería.",
        "<b>La demografía no responde al valle:</b> la población y los jóvenes son los de WPP aunque World3 colapse. Como los jóvenes "
        "elevan el riesgo de golpes y conflictos, esto probablemente subestima la inestabilidad en los colapsos con mucha mortalidad.",
        "<b>Las huellas a 10 años son permanentes</b> y se suman sin interacción: dos crisis seguidas cuestan el doble.",
        "<b>La capacidad de política es el gasto público,</b> un supuesto: un Estado puede gastar igual y aplicar peor, o al revés.",
        "<b>Ningún país vivió un valle de World3.</b> Todo lo estimado viene de crisis locales con un mundo que seguía creciendo.",
        "<b>Es un modo de comportamiento,</b> no un pronóstico: World3 es el modelo de 2004 y la sociedad está calibrada para 1960-2019.",
    ])
    doc = Doc(OUT)
    doc.build(story)
    print(f"{OUT} ({num(OUT.stat().st_size / 1e6, 1)} MB)")


if __name__ == "__main__":
    build()
