"""Construye ``dinamicas_sociedad_biofisica.pdf``: el informe de la investigación empírica de ``investigacion.py``.

    PYTHONPATH=. python reports/acople/informe_dinamicas.py

Reutiliza la maqueta y la paleta de ``reports/escenarios`` (fuentes OFL descargadas allí la primera vez).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "escenarios"))
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import informe_pdf as K  # noqa: E402  (maqueta: estilos, tablas, página)
import graficos as G  # noqa: E402
import investigacion as I  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import CondPageBreak, NextPageTemplate, PageBreak, Paragraph, Spacer  # noqa: E402

R = json.loads((HERE / "resultados.json").read_text(encoding="utf-8"))
FIG = HERE / "figuras"
FIG.mkdir(exist_ok=True)
OUT = HERE / "dinamicas_sociedad_biofisica.pdf"
para, bullets, table, img, callout, kpis, pct, num = K.para, K.bullets, K.table, K.img, K.callout, K.kpis, K.pct, K.num


def sgn(v, d):
    return ("+" if v >= 0 else "−") + num(abs(v), d)
TW = K.TEXT_W
S = K.S

OUTS = ["y_conflict_onset", "y_coup", "y_irregular_exit", "y_dem_breakdown", "y_sovdebt_onset", "y_interstate_mid"]
ONAME = {"y_conflict_onset": "Conflicto civil", "y_coup": "Intento de golpe", "y_irregular_exit": "Salida irregular",
         "y_dem_breakdown": "Quiebre democrático", "y_sovdebt_onset": "Crisis de deuda", "y_interstate_mid": "Disputa entre Estados",
         "y_protest_removal": "Caída por protestas"}
OCOL = {"y_conflict_onset": "#b3302b", "y_coup": "#eb6834", "y_irregular_exit": "#eda100", "y_dem_breakdown": "#6250d6",
        "y_sovdebt_onset": "#2a78d6", "y_interstate_mid": "#1baf7a"}
DRIVERS = {"growth": "Crecimiento del PIB p. c.", "dle3": "Cambio de esperanza de vida (3 años)",
           "yield_shock": "Rendimiento de cereales vs tendencia", "cereal_price_x_poor": "Precio mundial de cereales × país pobre",
           "res_rents": "Rentas de recursos / PIB", "depletion": "Agotamiento de recursos / RNB",
           "oil_price_x_exporter": "Precio del petróleo × exportador", "ln_co2pc": "CO2 per cápita (ln)",
           "ln_pm25": "PM2.5 (ln)", "eco_deficit": "Huella − biocapacidad", "youth": "Jóvenes 15-24",
           "pop_growth": "Crecimiento de la población", "temp_anom": "Anomalía anual de temperatura",
           "ln_dis": "Desastres climáticos", "ln_debt": "Deuda pública (ln)"}


# ── datos extra calculados del panel ──
P = I.load()
INC_BINS = [0, 7, 8, 9, 10, 11, 13]
GR_BINS = [-1, -0.05, -0.02, 0, 0.02, 0.05, 1]
INC = {o: I.binned(P, "lny", o, INC_BINS) for o in OUTS[:5]}
GRO = {o: I.binned(P, "growth", o, GR_BINS) for o in OUTS[:5]}


def world_cereal():
    w = I.pd.read_csv(I.atlas_dir() / "world_full.csv").set_index("year")
    cereal = np.log(w[["recursos__jacks_wheat", "recursos__jacks_corn", "recursos__jacks_rice"]]).mean(1)
    on = P.groupby("year")["y_conflict_onset"].mean()
    df = I.pd.DataFrame({"shock": cereal.diff().shift(1), "onset": on}).loc[1962:2015].dropna()
    return df


WC = world_cereal()


# ── figuras ──
def fig_matrix():
    drivers = list(DRIVERS)
    outs = OUTS
    A2, A3 = R["A2_single"], R["A3_pooled"]
    Z = np.full((len(drivers), len(outs)), np.nan)
    T = [["" for _ in outs] for _ in drivers]
    for i, x in enumerate(drivers):
        for j, o in enumerate(outs):
            r = A2.get(f"{o}|{x}")
            if not r or r.get("rel") is None:
                continue
            Z[i, j] = np.clip(r["rel"], -0.6, 0.6)
            mark = "**" if r["q"] < 0.1 else ("*" if r["p"] < 0.05 else "")
            T[i][j] = f"{r['rel'] * 100:+.0f}%{mark}"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#1c5cab", "#78aee8", "#f0efec", "#eb8a7f", "#b3302b"])
    ax.imshow(Z, cmap=cm, vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(len(drivers)):
        for j in range(len(outs)):
            if T[i][j]:
                bold = T[i][j].endswith("**")
                ax.text(j, i, T[i][j].replace("**", " *"), ha="center", va="center", fontsize=6.6, color=G.INK,
                        fontweight="bold" if bold else "normal")
    ax.set_xticks(range(len(outs)), [ONAME[o] for o in outs], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(drivers)), [DRIVERS[d] for d in drivers], fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Dentro de cada país: cambio de la probabilidad anual por 1 DE del factor (año previo)", fontsize=7.8)
    fig.tight_layout()
    return G.save(fig, FIG / "matriz_intra.png")


def fig_between():
    drivers = ["lny", "growth", "youth", "res_rents", "ln_pm25", "eco_deficit", "dle3", "temp_anom"]
    names = {"lny": "Nivel de ingreso (ln)", **DRIVERS}
    outs = OUTS
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(drivers))[::-1]
    off = np.linspace(-0.3, 0.3, len(outs))
    for j, o in enumerate(outs):
        for yi, x in zip(y, drivers):
            r = R["A3_pooled"].get(f"{o}|{x}", {})
            if "or_sd" not in r:
                continue
            filled = r["p"] < 0.05
            ax.plot(r["or_sd"], yi + off[j], "o", ms=4.6, color=OCOL[o], mfc=OCOL[o] if filled else "white", mew=1.1,
                    label=ONAME[o] if yi == y[0] else None)
    ax.axvline(1, color=G.INK3, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.35, 0.5, 0.7, 1, 1.4, 2, 2.8], ["0,35", "0,5", "0,7", "1", "1,4", "2", "2,8"])
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, [names[d] for d in drivers], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("razón de probabilidades por 1 DE (entre países y en el tiempo); relleno = p < 0,05", fontsize=6.8)
    ax.legend(fontsize=6.2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    return G.save(fig, FIG / "entre_paises.png")


def _bin_plot(data, xt, title, xlabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    x = np.arange(len(xt))
    for o, rows in data.items():
        ax.plot(x, [r["rate"] for r in rows], "-o", color=OCOL[o], ms=4, mec="white", mew=0.6, lw=1.7, label=ONAME[o])
    ax.set_xticks(x, xt, fontsize=6.8)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, fontsize=7.8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=6.4, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return G.save(fig, FIG / fname)


def fig_income():
    xt = ["< 1,1 mil", "1,1–3 mil", "3–8 mil", "8–22 mil", "22–60 mil", "> 60 mil"]
    return _bin_plot(INC, xt, "Probabilidad anual de cada evento según el ingreso por persona del año previo (USD 2017)",
                     "PIB per cápita (USD de 2017, PWT)", "gradiente_ingreso.png")


def fig_growth():
    xt = ["< −5 %", "−5 a −2 %", "−2 a 0 %", "0 a 2 %", "2 a 5 %", "> 5 %"]
    return _bin_plot(GRO, xt, "Probabilidad anual de cada evento según el crecimiento del PIB per cápita del año previo",
                     "crecimiento del PIB per cápita", "gradiente_crecimiento.png")


def fig_depth():
    xt = ["sin caída", "0–5 %", "5–10 %", "10–20 %", "20–30 %", "> 30 %"]
    data = {o: R["D3_depth"][o] for o in OUTS[:5]}
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    x = np.arange(len(xt))
    for o, rows in data.items():
        ax.plot(x, [r["rate"] for r in rows], "-o", color=OCOL[o], ms=4, mec="white", mew=0.6, lw=1.7, label=ONAME[o])
        for xi, r in zip(x, rows):
            if r["n"] < 60:
                ax.plot(xi, r["rate"], "o", ms=7, mfc="none", mec=OCOL[o], mew=0.8)
    ax.set_xticks(x, xt, fontsize=6.8)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("caída del PIB per cápita respecto de su máximo de los 8 años previos (países sin guerra en curso)", fontsize=7)
    ax.set_title("Probabilidad de que el evento ocurra en los 5 años siguientes", fontsize=7.8)
    ax.legend(fontsize=6.4, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return G.save(fig, FIG / "dosis_caida.png")


def fig_valleys():
    D2 = R["D2_clean"]
    defs = [("gdp10", "PIB ≥ 10 % bajo su máximo"), ("gdp", "PIB ≥ 20 % bajo su máximo"), ("le", "Esperanza de vida −3 años")]
    outs = ["y_coup", "y_irregular_exit", "y_dem_breakdown", "y_sovdebt_onset", "y_conflict_onset", "y_interstate_mid"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(outs))[::-1]
    for ax, (k, title) in zip(axes, defs):
        for yi, o in zip(y, outs):
            r = D2[k]["rates"][o]
            lo, hi, m = r["ratio_lo"] or 0, r["ratio_hi"] or 0, r["ratio"] or 0
            sig = lo > 1
            ax.plot([max(lo, 0.05), hi], [yi, yi], color=OCOL[o], lw=2.2, alpha=0.45, solid_capstyle="round")
            ax.plot(max(m, 0.05), yi, "o", color=OCOL[o], ms=5, mfc=OCOL[o] if sig else "white", mew=1.1)
            ax.text(10.5, yi, f"{r['events']:.0f}/{r['exposure']}", fontsize=5.6, color=G.INK3, va="center")
        ax.axvline(1, color=G.INK3, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 14)
        ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0,1", "0,3", "1", "3", "10"], fontsize=6.4)
        ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.set_title(f"{title}\n{D2[k]['episodes_no_war']} episodios sin guerra", fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [ONAME[o] for o in outs], fontsize=7)
    fig.supxlabel("razón de tasas en los 5 años siguientes (IC 95 %, Poisson exacto); relleno = IC sobre 1", fontsize=6.8)
    fig.tight_layout()
    return G.save(fig, FIG / "valles.png")


def fig_auc():
    outs = [o for o in R["B"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    x = np.arange(len(outs))
    b = [R["B"][o]["base_same"]["logit_auc"] for o in outs]
    f = [R["B"][o]["full_same"]["logit_auc"] for o in outs]
    ax.bar(x - 0.18, b, 0.34, color=G.NEUTRAL, label="Base al estilo del Atlas")
    ax.bar(x + 0.18, f, 0.34, color=G.ACCENT, label="Base + variables de World3")
    for xi, bb, ff in zip(x, b, f):
        ax.text(xi - 0.18, bb + 0.01, f"{bb:.2f}", ha="center", fontsize=6.2)
        ax.text(xi + 0.18, ff + 0.01, f"{ff:.2f}", ha="center", fontsize=6.2)
    ax.axhline(0.5, color=G.INK3, lw=0.8, ls=(0, (3, 2)))
    ax.set_ylim(0.4, 0.9)
    ax.set_xticks(x, [ONAME[o] for o in outs], fontsize=6.8)
    ax.set_title("Capacidad de predicción fuera de muestra (AUC; entrenado 1961-1999, evaluado 2000-2019)", fontsize=7.8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return G.save(fig, FIG / "auc.png")


def fig_lp():
    shocks = [("shock_conflict", "Inicio de conflicto civil"), ("shock_coup", "Golpe exitoso"),
              ("shock_sovdebt", "Crisis de deuda soberana"), ("shock_breakdown", "Quiebre democrático")]
    vars_ = [("gdppc", "PIB per cápita (%)"), ("life_expectancy", "Esperanza de vida (años)"), ("csh_i", "Inversión (pp del PIB)")]
    fig, axes = plt.subplots(len(shocks), len(vars_), figsize=(7.2, 7.4), sharex=True)
    for i, (s, sname) in enumerate(shocks):
        for j, (v, vname) in enumerate(vars_):
            ax = axes[i, j]
            rows = R["C"][s][v]
            h = [r["h"] for r in rows]
            b = np.array([r["b"] for r in rows])
            se = np.array([r["se"] for r in rows])
            ax.fill_between(h, b - 1.96 * se, b + 1.96 * se, color=G.ACCENT, alpha=0.16, lw=0)
            ax.plot(h, b, color=G.ACCENT, lw=1.8, marker="o", ms=2.6, mec="white", mew=0.5)
            ax.axhline(0, color=G.INK3, lw=0.8)
            if i == 0:
                ax.set_title(vname, fontsize=7.4)
            if j == 0:
                ax.set_ylabel(f"{sname}\n(n = {rows[0]['events']})", fontsize=6.8)
            ax.tick_params(labelsize=6.2)
    for ax in axes[-1]:
        ax.set_xlabel("años después del evento", fontsize=6.8)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    return G.save(fig, FIG / "proyecciones_locales.png")


def fig_world():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.scatter(WC["shock"], WC["onset"], s=16, color=G.ACCENT, edgecolor="white", lw=0.5)
    for yr, r in WC.iterrows():
        if abs(r["shock"]) > 0.15 or r["onset"] > 0.045:
            ax.annotate(str(yr), (r["shock"], r["onset"]), fontsize=6, color=G.INK2, xytext=(3, 2), textcoords="offset points")
    b = np.polyfit(WC["shock"], WC["onset"], 1)
    xs = np.linspace(WC["shock"].min(), WC["shock"].max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=G.INK3, lw=1)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("cambio del precio real mundial de cereales el año anterior (trigo, maíz, arroz; Δ log)", fontsize=6.8)
    ax.set_ylabel("países con inicio de\nconflicto civil", fontsize=6.8)
    ax.set_title("Escala mundial, 1962-2015: precios de los cereales y nuevos conflictos civiles", fontsize=7.8)
    fig.tight_layout()
    return G.save(fig, FIG / "mundo_cereales.png")


# ── documento ──
class Doc(K.Doc):
    def __init__(self, path):
        super().__init__(path)
        self.title = "Sociedad y límites biofísicos"
        self.subject = "Relaciones históricas entre las variables de World3 y las del Atlas del Sistema-Mundo"

    def draw_body(self, c, doc):
        c.saveState()
        c.setFont("Mono", 6.8)
        c.setFillColor(K.INK3)
        c.drawString(K.MARGIN, K.H_PAGE - 12 * mm, "SOCIEDAD Y LÍMITES BIOFÍSICOS · QUASAR PHY-WORLD")
        c.drawRightString(K.W_PAGE - K.MARGIN, K.H_PAGE - 12 * mm, self.section.upper()[:60])
        c.setStrokeColor(K.RULE)
        c.setLineWidth(0.5)
        c.line(K.MARGIN, K.H_PAGE - 13.6 * mm, K.W_PAGE - K.MARGIN, K.H_PAGE - 13.6 * mm)
        c.drawString(K.MARGIN, 10 * mm, "Asociaciones condicionadas en datos históricos, no efectos causales demostrados")
        c.drawRightString(K.W_PAGE - K.MARGIN, 10 * mm, f"{doc.page}")
        c.restoreState()

    def draw_cover(self, c, doc):
        c.saveState()
        c.setFillColor(colors.HexColor("#0f1c19"))
        c.rect(0, 0, K.W_PAGE, K.H_PAGE, stroke=0, fill=1)
        # el gradiente de ingreso como imagen de portada: golpes por tramo de ingreso
        x0, x1, y0, y1 = K.MARGIN, K.W_PAGE - K.MARGIN, 62 * mm, 128 * mm
        rows = INC["y_coup"]
        n = len(rows)
        w = (x1 - x0) / n
        top = max(r["rate"] for r in rows)
        for i, r in enumerate(rows):
            h = (y1 - y0) * r["rate"] / top
            c.setFillColor(colors.HexColor("#46c49c"))
            c.rect(x0 + i * w + 4, y0, w - 8, h, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#b7c2be"))
            c.setFont("Mono", 7)
            c.drawCentredString(x0 + i * w + w / 2, y0 + h + 4, f"{r['rate'] * 100:.1f} %".replace(".", ","))
        c.setFillColor(colors.HexColor("#8a9692"))
        c.setFont("Mono", 6.6)
        for i, lab in enumerate(["< 1,1 mil", "1,1–3 mil", "3–8 mil", "8–22 mil", "22–60 mil", "> 60 mil"]):
            c.drawCentredString(x0 + i * w + w / 2, y0 - 10, lab)
        c.drawString(x0, y1 + 26, "INTENTOS DE GOLPE POR AÑO SEGÚN EL INGRESO POR PERSONA (USD 2017) · 180 PAÍSES, 1950-2019")
        c.setFillColor(colors.HexColor("#46c49c"))
        c.setFont("Mono", 8)
        c.drawString(K.MARGIN, K.H_PAGE - 30 * mm, "QUASAR PHY-WORLD · INVESTIGACIÓN EMPÍRICA · OCTUBRE 2026")
        c.setFillColor(colors.HexColor("#e5ebe8"))
        c.setFont("Serif", 50)
        c.drawString(K.MARGIN, K.H_PAGE - 60 * mm, "Sociedad y límites")
        c.setFont("Serif-It", 50)
        c.setFillColor(colors.HexColor("#46c49c"))
        c.drawString(K.MARGIN, K.H_PAGE - 78 * mm, "biofísicos")
        c.setFillColor(colors.HexColor("#b7c2be"))
        c.setFont("Sans", 11.5)
        for i, l in enumerate(["Qué dicen 70 años de datos de 180 países sobre cómo se conectan",
                               "el ingreso, la comida, los recursos, la contaminación y el clima",
                               "con el conflicto, los golpes, la democracia y las crisis de deuda."]):
            c.drawString(K.MARGIN, K.H_PAGE - 94 * mm - i * 16, l)
        c.setFont("Sans", 7.6)
        c.setFillColor(colors.HexColor("#8a9692"))
        c.drawString(K.MARGIN, 22 * mm, "Base empírica para acoplar el núcleo World3 (L0) con el Atlas del Sistema-Mundo (L1) de Phy-World.")
        c.drawString(K.MARGIN, 17 * mm, f"Generado {R['meta']['generated']} · reports/acople/investigacion.py · panel del Atlas, 180 países, 1950-2019")
        c.restoreState()


def build():
    print("figuras...")
    f_mat, f_bet, f_inc, f_gro = fig_matrix(), fig_between(), fig_income(), fig_growth()
    f_dep, f_val, f_auc, f_lp, f_w = fig_depth(), fig_valleys(), fig_auc(), fig_lp(), fig_world()
    A2, A3, D2, C, B, Fw = R["A2_single"], R["A3_pooled"], R["D2_clean"], R["C"], R["B"], R["F_world"]
    g10 = D2["gdp10"]["rates"]
    le = D2["le"]["rates"]
    inc = {o: [r["rate"] for r in INC[o]] for o in INC}
    gro = {o: [r["rate"] for r in GRO[o]] for o in GRO}
    lp = lambda s, v, h: C[s][v][h]  # noqa: E731
    ci = lambda r: f"{sgn(r['b'], 1)} (IC ±{num(1.96 * r['se'], 1)})"  # noqa: E731
    rr = lambda r: f"×{num(r['ratio'], 1)} (IC {num(r['ratio_lo'], 1)}–{num(r['ratio_hi'], 1)})"  # noqa: E731

    story = [NextPageTemplate("body"), PageBreak()]
    # ── resumen ──
    story += [para("Resumen", "eyebrow"), para("Lo que encontramos", "h2"),
              para("World3 ve el mundo como un sistema biofísico; el Atlas lo ve como un sistema de Estados, élites y "
                   "conflictos. Para unirlos hace falta saber cómo se conectan en la realidad. Revisamos 70 años de datos de "
                   "180 países con cinco métodos distintos. La respuesta es más estrecha de lo que sugiere la intuición: el "
                   "puente casi único es el <b>ingreso</b>, y lo que dispara es sobre todo <b>inestabilidad política</b>, no "
                   "guerra.", "lede"),
              kpis([(f"×{num(inc['y_coup'][2] / inc['y_coup'][4], 0)}".replace(".", ","),
                     f"más intentos de golpe al año en países de 3–8 mil USD por persona ({pct(inc['y_coup'][2])}) que en los de "
                     f"22–60 mil ({pct(inc['y_coup'][4])})"),
                    (f"×{num(g10['y_sovdebt_onset']['ratio'], 1)}", "crisis de deuda en los 5 años tras una caída del PIB ≥ 10 %"),
                    (f"−{num(abs(lp('shock_sovdebt', 'gdppc', 10)['b']), 1)} %", "PIB per cápita 10 años después de una crisis de deuda"),
                    (f"{sgn(B['y_conflict_onset']['full_same']['logit_auc'] - B['y_conflict_onset']['base_same']['logit_auc'], 2)}".replace(".", ","),
                     "mejora de la predicción del conflicto civil (AUC) al añadir las variables de World3")]),
              Spacer(1, 8)]
    story += bullets([
        f"<b>El ingreso es la variable que conecta.</b> Entre países, la probabilidad anual de un intento de golpe baja de "
        f"{pct(inc['y_coup'][0])} (menos de 1,1 mil USD por persona) a {pct(inc['y_coup'][4])} (22–60 mil); la de un quiebre "
        f"democrático, de ≈ {pct(inc['y_dem_breakdown'][1])} a {pct(inc['y_dem_breakdown'][4])}; la de un conflicto civil nuevo, de "
        f"{pct(inc['y_conflict_onset'][0])} a {pct(inc['y_conflict_onset'][5])}.",
        f"<b>Cuando la economía se contrae, la política se rompe.</b> Con el PIB per cápita cayendo, los intentos de golpe "
        f"ocurren en {pct(gro['y_coup'][0])} de los años, frente a {pct(gro['y_coup'][4])} con crecimiento de 2–5 %; las crisis "
        f"de deuda, {pct(gro['y_sovdebt_onset'][0])} frente a {pct(gro['y_sovdebt_onset'][4])}; los quiebres democráticos, "
        f"{pct(gro['y_dem_breakdown'][0])} frente a {pct(gro['y_dem_breakdown'][3])}.",
        f"<b>Las recesiones profundas traen inestabilidad, no guerra civil.</b> En los 5 años tras una caída del PIB per cápita de "
        f"10 % o más (sin guerra previa), las crisis de deuda se multiplican {rr(g10['y_sovdebt_onset'])}, los quiebres "
        f"democráticos {rr(g10['y_dem_breakdown'])} y los golpes {rr(g10['y_coup'])}; los conflictos civiles, "
        f"{rr(g10['y_conflict_onset'])}: sin cambio.",
        "<b>Comida, contaminación y recursos no tienen canal propio en estos datos.</b> El rendimiento agrícola no se relaciona "
        "con nada; el PM2.5 y la huella sólo marcan a los países pobres; las rentas de recursos se asocian a más conflicto entre "
        "países pero a menos golpes dentro de cada país. La excepción es mundial: tras las subidas del precio de los cereales "
        "hay más conflictos nuevos en el mundo.",
        "<b>La sociedad devuelve el golpe a la economía.</b> Un conflicto civil resta ≈ 4–5 % al PIB per cápita a 5 años; una "
        "crisis de deuda, ≈ 6–7 % a 10 años; un golpe, ≈ 2–3 %. Las pérdidas no se recuperan solas.",
        "<b>Para el acople:</b> el Atlas ya contiene el canal que los datos respaldan. Lo que falta es aplicarlo al valle de World3, "
        "donde la producción por persona cae 50–70 %: la predicción empírica es una ola de crisis de deuda, golpes y quiebres "
        "democráticos, con un efecto menor sobre las guerras.",
    ])
    story.append(PageBreak())

    # ── pregunta y datos ──
    story += [para("Planteamiento", "eyebrow"), para("La pregunta, los datos y los métodos", "h2"),
              para("Phy-World tiene dos capas que hoy no se hablan. El núcleo L0 (World3-03) mueve población, capital, comida, "
                   "recursos y contaminación a escala mundial; en él aparece el valle de 2040-2060. La capa L1 (el Atlas) mueve 180 "
                   "países con élites, presión política, conflicto, golpes, democracia y deuda. Antes de unirlas hay que responder con "
                   "datos: <b>¿qué variables de World3 se relacionan, históricamente, con qué fenómenos sociales, y en qué "
                   "dirección?</b>"),
              para("Cómo traducimos World3 a datos observables", "h3"),
              table([["Variable de World3", "Análogo en el panel", "Fuente", "Años"],
                     ["Producción industrial per cápita", "PIB per cápita (nivel y crecimiento)", "Penn World Table 10.01", "1950-2019"],
                     ["Alimento per cápita", "Rendimiento de cereales frente a su tendencia; precio real mundial de trigo, maíz y arroz",
                      "Banco Mundial (WDI); precios reales de materias primas", "1961-2019"],
                     ["Recursos no renovables", "Rentas de recursos / PIB; agotamiento / RNB; precio del petróleo × exportador",
                      "WDI; Atlas", "1950/1970-2019"],
                     ["Contaminación persistente", "PM2.5; CO2 per cápita", "WDI; Global Carbon Project (OWID)", "1990/1950-2019"],
                     ["Huella ecológica", "Huella − biocapacidad per cápita", "Global Footprint Network", "1961-2019"],
                     ["Esperanza de vida", "Esperanza de vida y su cambio en 3 años", "UN WPP 2024", "1950-2019"],
                     ["Demografía", "Jóvenes 15-24 sobre adultos; crecimiento de la población", "UN WPP 2024", "1950-2019"],
                     ["(Clima)", "Anomalía anual de temperatura; desastres climáticos", "FAOSTAT/NASA GISTEMP; EM-DAT", "1961/1950-2019"]],
                    [36 * mm, 58 * mm, 54 * mm, TW - 148 * mm], zebra=True),
              Spacer(1, 6), para("Los fenómenos sociales", "h3"),
              table([["Evento", "Definición", "Fuente", "Frecuencia"],
                     ["Conflicto civil", "Inicio de conflicto armado intraestatal (≥ 25 muertes) tras ≥ 2 años de paz", "UCDP/PRIO", f"{pct(R['A2_single']['y_conflict_onset|growth']['base'])} de los años"],
                     ["Intento de golpe", "Golpe exitoso o fallido", "Powell & Thyne", f"{pct(R['A2_single']['y_coup|growth']['base'])}"],
                     ["Salida irregular", "El líder sale por golpe, revuelta, asesinato…", "Archigos 4.1", f"{pct(R['A2_single']['y_irregular_exit|growth']['base'])}"],
                     ["Caída por protestas", "El líder es removido por protesta popular", "Archigos 4.1", f"{pct(R['A2_single']['y_protest_removal|growth']['base'])}"],
                     ["Quiebre democrático", "Paso de democracia a autocracia (Regímenes del Mundo)", "V-Dem v15", f"{pct(R['A2_single']['y_dem_breakdown|growth']['base'])} de las democracias"],
                     ["Crisis de deuda", "Inicio de impago o reestructuración soberana", "Global Macro Database", f"{pct(R['A2_single']['y_sovdebt_onset|growth']['base'])}"],
                     ["Disputa entre Estados", "Inicio de una disputa militarizada interestatal", "GML MID 2.2.1 (hasta 2010)", f"{pct(R['A2_single']['y_interstate_mid|growth']['base'])}"]],
                    [30 * mm, 76 * mm, 40 * mm, TW - 146 * mm], zebra=True),
              Spacer(1, 6), para("Cinco miradas complementarias", "h3")]
    story += bullets([
        "<b>Dentro de cada país</b> (efectos fijos de país y de año, factores rezagados un año): ¿cuando un país cambia, cambia "
        "su riesgo? Controla todo lo que es fijo en el país y todo lo que es común al mundo en un año. Cada factor se estima por "
        "separado, controlando el nivel de ingreso, y el efecto se expresa por una desviación estándar <i>dentro</i> del país. "
        "Como probamos 105 relaciones, corregimos por pruebas múltiples (FDR de Benjamini-Hochberg).",
        "<b>Entre países</b> (logit agrupado con efectos de década): ¿los países con más de algo tienen más eventos? Es la mirada "
        "que importa para niveles, como los que mueve World3, pero mezcla causa con contexto.",
        "<b>Predicción fuera de muestra:</b> entrenamos con 1961-1999 y evaluamos con 2000-2019. Si una variable de World3 lleva "
        "información propia, debe mejorar la predicción sobre una base al estilo del Atlas.",
        "<b>Episodios:</b> caídas fuertes del PIB o de la esperanza de vida <i>sin guerra previa</i>, y qué pasa en los 5 años "
        "siguientes, con intervalos de confianza exactos.",
        "<b>Proyecciones locales</b> (Jordà 2005): cuánto cambian el PIB, la esperanza de vida y la inversión en los 10 años "
        "siguientes a un conflicto, un golpe, una crisis de deuda o un quiebre democrático.",
    ])
    story += [callout("Todo lo que sigue son asociaciones condicionadas: comparan a un país consigo mismo, controlan los ciclos "
                      "mundiales y miran hacia adelante en el tiempo, pero no son experimentos. Donde los datos no alcanzan para "
                      "distinguir causa de efecto, lo decimos.", "Cómo leer las cifras"), CondPageBreak(170 * mm)]

    # ── dinámica 1: ingreso ──
    story += [para("Dinámica 1", "eyebrow"), para("El ingreso: el puente principal", "h2"),
              para("El hallazgo más robusto de toda la investigación es también el más antiguo de la literatura: la pobreza y las "
                   "caídas del ingreso acompañan a casi todas las formas de inestabilidad. Lo que añadimos es la forma: tiene "
                   "umbrales marcados."),
              img(f_inc), para("Tasa anual de cada evento por tramo de PIB per cápita del año previo. 180 países, 1950-2019.", "cap")]
    story += bullets([
        f"<b>Un umbral entre 8 y 22 mil dólares.</b> Los intentos de golpe pasan de {pct(inc['y_coup'][2])} al año (3–8 mil) a "
        f"{pct(inc['y_coup'][3])} (8–22 mil) y {pct(inc['y_coup'][4])} (22–60 mil). Los quiebres democráticos siguen la misma "
        f"escalera ({pct(inc['y_dem_breakdown'][2])} → {pct(inc['y_dem_breakdown'][3])} → {pct(inc['y_dem_breakdown'][4])}). "
        "Es la regularidad que Przeworski y coautores describieron para la supervivencia de las democracias: por encima de cierto "
        "ingreso, casi ninguna cae.",
        f"<b>El conflicto civil baja de forma más gradual:</b> {pct(inc['y_conflict_onset'][0])}, {pct(inc['y_conflict_onset'][1])}, "
        f"{pct(inc['y_conflict_onset'][2])}, {pct(inc['y_conflict_onset'][3])} y {pct(inc['y_conflict_onset'][4])} por tramo. Es la "
        "relación de Fearon y Laitin (2003): en países pobres el Estado controla peor el territorio y rebelarse cuesta menos.",
        f"<b>Las crisis de deuda son un fenómeno de ingresos medios:</b> ≈ 2 % anual hasta los 22 mil dólares y "
        f"{pct(inc['y_sovdebt_onset'][4])} por encima.",
        f"<b>Entre países, una desviación estándar más de ingreso</b> (≈ ×3,5) multiplica las probabilidades de conflicto por "
        f"{num(A3['y_conflict_onset|lny']['or_sd'], 2)}, de golpe por {num(A3['y_coup|lny']['or_sd'], 2)} y de quiebre democrático por "
        f"{num(A3['y_dem_breakdown|lny']['or_sd'], 2)}.",
    ])
    story += [CondPageBreak(90 * mm), para("El crecimiento: lo que se mueve dentro de cada país", "h3"), img(f_gro),
              para("Tasa anual de cada evento según el crecimiento del PIB per cápita del año previo.", "cap")]
    g = A2
    story += bullets([
        f"<b>Las contracciones duplican los golpes y las salidas irregulares.</b> Con el PIB per cápita bajando, los intentos de "
        f"golpe ocurren en {pct(gro['y_coup'][0])}–{pct(gro['y_coup'][2])} de los años; con crecimiento de 2–5 %, en "
        f"{pct(gro['y_coup'][4])}. Dentro de cada país, una desviación estándar más de crecimiento baja los golpes "
        f"{num(abs(g['y_coup|growth']['rel']) * 100, 0)} % y las salidas irregulares {num(abs(g['y_irregular_exit|growth']['rel']) * 100, 0)} %: "
        "dos de las pocas relaciones que sobreviven la corrección por pruebas múltiples.",
        f"<b>Las crisis de deuda y los quiebres democráticos son los más sensibles a la recesión.</b> Con caídas de más de 5 %, las "
        f"crisis de deuda llegan a {pct(gro['y_sovdebt_onset'][0])} y los quiebres a {pct(gro['y_dem_breakdown'][0])} de los años; "
        "con crecimiento moderado, ≈ 1 %.",
        f"<b>El conflicto civil responde menos:</b> {pct(gro['y_conflict_onset'][0])} con contracciones fuertes frente a "
        f"{pct(gro['y_conflict_onset'][5])} con crecimiento alto; dentro de cada país el efecto es de "
        f"{sgn(g['y_conflict_onset|growth']['rel'] * 100, 0)} % por desviación estándar (p = {num(g['y_conflict_onset|growth']['p'], 3)}), "
        "pero no supera la corrección por pruebas múltiples. Es coherente con Miguel, Satyanath y Sergenti (2004), que identificaron "
        "el efecto con lluvias en África, pero en el panel mundial es más débil.",
    ])
    story.append(CondPageBreak(170 * mm))

    # ── dinámica 2: valles económicos ──
    story += [para("Dinámica 2", "eyebrow"), para("Valles económicos: inestabilidad, no guerra", "h2"),
              para("El valle de World3 es, ante todo, una caída de la producción por persona. ¿Qué pasó, históricamente, en los países "
                   "que vivieron una caída así sin estar ya en guerra? Medimos la probabilidad de cada evento en los 5 años "
                   "siguientes según la profundidad de la caída respecto del máximo de los 8 años previos."),
              img(f_dep), para("Círculos vacíos: tramos con menos de 60 observaciones. Excluye los años en guerra civil.", "cap")]
    dd = R["D3_depth"]
    story += bullets([
        f"<b>Crisis de deuda:</b> de {pct(dd['y_sovdebt_onset'][0]['rate'])} sin caída a {pct(dd['y_sovdebt_onset'][3]['rate'])} con caídas "
        "de 10–20 %. La relación va en las dos direcciones: la recesión dificulta pagar y el impago profundiza la recesión (ver "
        "Dinámica 6).",
        f"<b>Golpes y salidas irregulares:</b> suben con la caída hasta el 10–20 % ({pct(dd['y_coup'][3]['rate'])} y "
        f"{pct(dd['y_irregular_exit'][3]['rate'])} en 5 años, frente a {pct(dd['y_coup'][0]['rate'])} y "
        f"{pct(dd['y_irregular_exit'][0]['rate'])}) y luego bajan: en las caídas más extremas suele haber ya un régimen nuevo o "
        "un desorden que los datos registran de otra forma.",
        f"<b>Quiebres democráticos:</b> de {pct(dd['y_dem_breakdown'][0]['rate'])} a ≈ {pct(dd['y_dem_breakdown'][2]['rate'])} con caídas "
        f"de 5–20 %, y {pct(dd['y_dem_breakdown'][5]['rate'])} en las caídas de más del 30 % (pocos casos).",
        f"<b>Conflicto civil:</b> sube sólo con caídas extremas ({pct(dd['y_conflict_onset'][0]['rate'])} → "
        f"{pct(dd['y_conflict_onset'][5]['rate'])} con más de 30 %).",
    ])
    story += [CondPageBreak(95 * mm), para("Los episodios, uno por uno", "h3"),
              para("Contamos los eventos en los 5 años siguientes al inicio de cada episodio sin guerra previa y los comparamos con la "
                   "tasa del resto de los países-año sin episodio ni guerra. Los números a la derecha son eventos / años observados."),
              img(f_val)]
    story += bullets([
        f"<b>Con caídas del PIB de 10 % o más</b> ({D2['gdp10']['episodes_no_war']} episodios): crisis de deuda {rr(g10['y_sovdebt_onset'])}, "
        f"quiebres democráticos {rr(g10['y_dem_breakdown'])}, golpes {rr(g10['y_coup'])}, salidas irregulares "
        f"{rr(g10['y_irregular_exit'])}. Conflicto civil {rr(g10['y_conflict_onset'])} y disputas entre Estados "
        f"{rr(g10['y_interstate_mid'])}: nada.",
        f"<b>Con caídas de 20 % o más</b> ({D2['gdp']['episodes_no_war']} episodios) el patrón es el mismo pero con menos precisión.",
        "Entre los ejemplos están la transición postsoviética (Armenia, Bielorrusia, Bulgaria, Kazajistán, los bálticos), "
        "Argentina 2001-2002, Grecia 2012 y los colapsos petroleros de los Estados del Golfo en los años ochenta.",
    ])
    story.append(CondPageBreak(150 * mm))

    # ── dinámica 3: mortalidad ──
    story += [para("Dinámica 3", "eyebrow"), para("Valles de mortalidad: la analogía más cercana", "h2"),
              para("En el valle de World3 lo que cae es la esperanza de vida. Históricamente, la mayoría de las caídas grandes de "
                   "esperanza de vida las causan las guerras, así que para no confundir causa y efecto separamos las que ocurrieron "
                   f"<b>sin guerra previa</b>: {D2['le']['episodes_no_war']} de {D2['le']['episodes_all']} episodios de caída de 3 años "
                   "o más en 5 años."),
              para("Los episodios", "h3"),
              para(", ".join(f"{c} {y}" for c, y in D2["le"]["examples_no_war"]) + ". Predominan la epidemia de VIH en África austral "
                   "(Botsuana, Esuatini, Lesoto, Namibia, Sudáfrica, Zambia, Zimbabue, Kenia), la transición postsoviética "
                   "(Estonia, Letonia) y crisis puntuales (el terremoto de Haití en 2010). Algunos casos, como Burundi 1972 y Ruanda 1990, "
                   "incluyen violencia masiva que la base de conflictos armados no registra como guerra en el año de inicio: el "
                   "filtro no es perfecto.", "cap"),
              table([["Evento en los 5 años siguientes", "Tasa tras el episodio", "Referencia", "Razón (IC 95 %)", "Eventos / años"]] +
                    [[r["name"], pct(r["after"]), pct(r["reference"]), rr(r) if r["ratio"] else "—",
                      f"{num(r['events'], 0)} / {r['exposure']}"] for r in le.values()],
                    [52 * mm, 28 * mm, 24 * mm, 36 * mm, TW - 140 * mm], zebra=True),
              Spacer(1, 6)]
    story += bullets([
        f"<b>Golpes: {rr(le['y_coup'])}.</b> Es el único efecto con intervalo claramente por encima de 1.",
        f"<b>Conflicto civil {rr(le['y_conflict_onset'])} y salidas irregulares {rr(le['y_irregular_exit'])}:</b> el punto central "
        "sugiere aumento, pero con 5 eventos el intervalo incluye 1.",
        "<b>Crisis de deuda y quiebres democráticos: ninguno.</b> Muchas de estas crisis de mortalidad ocurrieron en democracias "
        "estables de ingreso medio (Botsuana, Sudáfrica, los bálticos) que no se quebraron.",
        "<b>Lectura:</b> la mortalidad masiva por enfermedad o crisis no se tradujo automáticamente en guerra. El riesgo que sí "
        "aparece es la ruptura del poder (golpes). La muestra es pequeña: es una pista, no una ley.",
    ])
    story.append(PageBreak())

    # ── dinámica 4: comida, contaminación, recursos, clima ──
    story += [para("Dinámica 4", "eyebrow"), para("Comida, contaminación, recursos y clima", "h2"),
              para("World3 tiene canales específicos: comida, contaminación persistente, recursos. Si alguno tuviera un efecto social "
                   "propio, debería verse dentro de cada país, controlando el ingreso. La matriz muestra el efecto de cada factor, uno a "
                   "uno, sobre cada evento."),
              img(f_mat),
              para("Celdas: cambio relativo de la probabilidad anual por 1 DE del factor dentro del país (año previo), con efectos fijos "
                   "de país y año y control del ingreso. En negrita con *: sobrevive la corrección por pruebas múltiples (q < 0,10); "
                   "con *: p < 0,05 sin corregir. Rojo = más eventos; azul = menos.", "cap")]
    sig = sorted([(k, v) for k, v in A2.items() if v["q"] < 0.10], key=lambda kv: kv[1]["q"])
    story += bullets([
        f"<b>De 105 relaciones, {len(sig)} sobreviven la corrección:</b> " + "; ".join(
            f"{DRIVERS.get(k.split('|')[1], k.split('|')[1]).lower()} → {ONAME[k.split('|')[0]].lower()} "
            f"({sgn(v['rel'] * 100, 0)} %)" for k, v in sig) + ".",
        "<b>Comida:</b> el rendimiento de cereales frente a su tendencia no se relaciona con ningún evento. Los años de mala cosecha "
        "en un país no anticipan conflicto ni golpes una vez que se considera el ingreso. Los países que importan comida se protegen "
        "con el comercio; los que no, sufren la caída como una caída de ingreso.",
        "<b>Contaminación y huella:</b> dentro de cada país, nada robusto. Entre países, el PM2.5 aparece fuertemente asociado "
        f"(razón de probabilidades {num(A3['y_conflict_onset|ln_pm25']['or_sd'], 1)} para conflicto, {num(A3['y_coup|ln_pm25']['or_sd'], 1)} "
        "para golpes), pero desaparece al comparar a cada país consigo mismo: marca un tipo de país (pobre, en industrialización "
        "temprana), no un mecanismo.",
        f"<b>Recursos:</b> entre países, más rentas se asocian a más conflicto civil ({num(A3['y_conflict_onset|res_rents']['or_sd'], 2)}), "
        f"como en Collier y Hoeffler; dentro de cada país, a menos golpes ({sgn(A2['y_coup|res_rents']['rel'] * 100, 0)} %): los "
        "Estados rentistas compran lealtades. El agotamiento no tiene efecto robusto.",
        "<b>Clima:</b> la anomalía anual de temperatura se asocia débilmente a salidas irregulares del líder "
        f"({sgn(A2['y_irregular_exit|temp_anom']['rel'] * 100, 0)} %, p = {num(A2['y_irregular_exit|temp_anom']['p'], 3)}) y no al conflicto. "
        "Estudios con medidas locales y mensuales (Hsiang, Burke y Miguel 2013) encuentran efectos que una anomalía nacional anual no "
        "alcanza a ver.",
    ])
    story += [CondPageBreak(110 * mm), para("Entre países: lo que distingue a los países inestables", "h3"), img(f_bet),
              para("Razón de probabilidades por 1 DE (logit agrupado con efectos de década, errores por país).", "cap")]
    story += bullets([
        f"<b>Los jóvenes son el segundo marcador más fuerte</b> después del ingreso: un país con una desviación estándar más de "
        f"jóvenes de 15-24 tiene {num(A3['y_coup|youth']['or_sd'], 1)} veces las probabilidades de golpe y "
        f"{num(A3['y_conflict_onset|youth']['or_sd'], 1)} las de conflicto civil (Urdal 2006). Dentro de cada país el efecto no es robusto: "
        "el bulto juvenil cambia despacio y su efecto se confunde con el desarrollo.",
        "<b>La lección metodológica:</b> casi todo se correlaciona con la inestabilidad entre países porque casi todo se correlaciona con "
        "el ingreso. Por eso un modelo acoplado no debe sumar estos factores como causas independientes.",
    ])
    story += [CondPageBreak(80 * mm), para("La excepción mundial: el precio de los cereales", "h3"), img(f_w)]
    wf = Fw["onset~cereal_shock"]
    story += bullets([
        f"<b>A escala mundial sí aparece una señal:</b> los años que siguen a subidas del precio real de los cereales tienen más "
        f"inicios de conflicto civil (correlación {sgn(wf['corr'], 2)}, p = {num(wf['p'], 3)}, {wf['n']} años). Coincide con la literatura "
        "sobre precios de alimentos y disturbios (Bellemare 2015).",
        "<b>Pero no se ve país por país:</b> el choque de precios en países pobres no predice conflicto dentro de cada país. Con sólo "
        "53 años y años extremos como 1973-1974, la señal mundial es sugerente pero frágil. El canal más probable son las protestas "
        "urbanas, que este panel no registra.",
    ])
    story.append(PageBreak())

    # ── dinámica 5: predicción ──
    story += [para("Dinámica 5", "eyebrow"), para("¿World3 añade información sobre la inestabilidad?", "h2"),
              para("La prueba más exigente: entrenar con los datos de 1961-1999 y predecir 2000-2019. La base usa lo que ya tiene el "
                   "Atlas (ingreso, crecimiento, democracia, jóvenes, crecimiento de la población, deuda y la historia de eventos del "
                   "país). Luego añadimos todas las variables de World3: cambio de esperanza de vida, comida, recursos, contaminación, "
                   "huella y clima."),
              img(f_auc),
              para("AUC: 0,5 es azar, 1 es predicción perfecta. Mismo conjunto de prueba para ambas versiones (filas con todas las variables).", "cap")]
    story += bullets([
        "<b>Las variables de World3 no mejoran la predicción de ningún evento.</b> Las diferencias son de ±0,01-0,02, dentro del "
        "ruido. El modelo de árboles (gradient boosting), que capta no linealidades, da la misma conclusión.",
        "<b>Por qué importa:</b> la información que la biofísica tiene sobre la inestabilidad llega a través del ingreso, que el "
        "Atlas ya modela. Acoplar World3 al Atlas no requiere ecuaciones sociales nuevas: requiere que el ingreso del Atlas sienta "
        "la escasez, la contaminación y la comida de World3.",
        "<b>Lo difícil de predecir sigue siendo difícil:</b> las crisis de deuda (AUC ≈ 0,58) y las caídas por protestas (≈ 0,6) son "
        "casi impredecibles con estas variables; el conflicto, los golpes y los quiebres (0,74-0,79) lo son moderadamente.",
    ])
    story.append(PageBreak())

    # ── dinámica 6: de la sociedad a la economía ──
    story += [para("Dinámica 6", "eyebrow"), para("De vuelta: lo que la inestabilidad le hace a la economía", "h2"),
              para("Para cerrar el lazo hace falta la dirección contraria: cuando ocurre un conflicto, un golpe o una crisis de deuda, "
                   "¿cuánto cambian las variables que World3 sí modela? Estimamos proyecciones locales: la diferencia, h años después "
                   "del evento, respecto del año previo, con efectos fijos de país y año y controles del crecimiento pasado, el "
                   "ingreso y el conflicto en curso."),
              img(f_lp),
              para("Efecto medio y banda de confianza del 95 % (errores agrupados por país). PIB en % (100 × Δ log); esperanza de "
                   "vida en años; inversión en puntos del PIB.", "cap")]
    story += bullets([
        f"<b>Conflicto civil:</b> PIB per cápita {ci(lp('shock_conflict', 'gdppc', 2))} % a 2 años y "
        f"{ci(lp('shock_conflict', 'gdppc', 5))} % a 5 años y {ci(lp('shock_conflict', 'gdppc', 10))} % a 10: sin recuperación "
        "clara, aunque a 10 años el intervalo ya incluye cero. Esperanza de vida "
        f"{ci(lp('shock_conflict', 'life_expectancy', 0))} años el año del inicio, recuperada en ≈ 2 años. Inversión "
        f"{ci(lp('shock_conflict', 'csh_i', 2))} puntos a 2 años.",
        f"<b>Crisis de deuda soberana:</b> la pérdida más grande y más persistente: {ci(lp('shock_sovdebt', 'gdppc', 5))} % a 5 años "
        f"y {ci(lp('shock_sovdebt', 'gdppc', 10))} % a 10. Es la «cicatriz» que Cerra y Saxena (2008) documentaron: el PIB no vuelve "
        "a su tendencia previa.",
        f"<b>Golpe exitoso:</b> {ci(lp('shock_coup', 'gdppc', 0))} % el mismo año y {ci(lp('shock_coup', 'gdppc', 2))} % a 2 años.",
        "<b>Quiebre democrático:</b> sin efecto económico claro en promedio (62 casos).",
        "<b>Rendimiento agrícola y población:</b> ningún efecto medible de ninguno de los cuatro eventos.",
    ])
    story += [callout("Juntando las dos direcciones aparece un círculo vicioso con base empírica: una caída del ingreso multiplica "
                      "las crisis de deuda (×3,4 en 5 años) y cada crisis de deuda resta ≈ 6 % más de PIB per cápita. Los golpes y "
                      "los conflictos añaden pérdidas de 2-5 %. En el valle de World3 este lazo profundizaría y alargaría la caída.",
                      "El lazo que sí existe"), PageBreak()]

    # ── síntesis ──
    story += [para("Síntesis", "eyebrow"), para("El mapa de las dinámicas y cómo acoplar", "h2"),
              para("Ordenando los seis hallazgos por la solidez de la evidencia:")]
    story.append(table([
        ["Relación", "Evidencia", "Magnitud", "Uso en el modelo acoplado"],
        ["Ingreso (nivel) → golpes, quiebres, conflicto", "Fuerte entre países; umbrales claros",
         "Golpes 5,7 % → 1,4 % → 0,1 % al año entre 3–8, 8–22 y 22–60 mil USD", "Ya en el Atlas: alimentarlo con el ingreso afectado por el valle"],
        ["Crecimiento → golpes, salidas irregulares, crisis de deuda", "Fuerte dentro de cada país (sobrevive FDR)",
         f"−{num(abs(A2['y_coup|growth']['rel']) * 100, 0)} % de golpes por 1 DE de crecimiento", "Ya en el Atlas"],
        ["Caída del PIB ≥ 10 % → deuda, quiebres, golpes", "Moderada-fuerte (IC sobre 1)",
         f"×{num(g10['y_sovdebt_onset']['ratio'], 1)}, ×{num(g10['y_dem_breakdown']['ratio'], 1)}, ×{num(g10['y_coup']['ratio'], 1)} en 5 años",
         "Contrastar contra el Atlas en el valle; recalibrar si se queda corto"],
        ["Caída del PIB → conflicto civil", "Débil (sólo caídas > 30 %)", "×0,9 con 10 %; ×1,7 con > 30 %", "No reforzar"],
        ["Crisis de mortalidad sin guerra → golpes", "Moderada, muestra pequeña",
         f"{rr(le['y_coup'])}", "Supuesto declarado, con rango"],
        ["Comida, contaminación, huella → inestabilidad", "Ninguna dentro de cada país", "—", "No programar como canal propio"],
        ["Precio mundial de cereales → conflictos nuevos", "Débil, sólo a escala mundial", f"r = {sgn(wf['corr'], 2)}", "Prueba de sensibilidad"],
        ["Conflicto → PIB, inversión", "Moderada", "≈ −4–5 % a 5 años; inversión −1 pp", "Choque persistente a la productividad"],
        ["Crisis de deuda → PIB", "Fuerte", "≈ −6–7 % a 10 años", "Choque persistente; ya parcial en el Atlas"],
        ["Golpe → PIB", "Moderada", "≈ −2–3 % a 2 años", "Choque transitorio"],
    ], [44 * mm, 34 * mm, 48 * mm, TW - 126 * mm], zebra=True))
    story += [Spacer(1, 8), para("Qué significa para el valle de World3", "h3")]
    story += bullets([
        "<b>El valle sería, socialmente, una ola de crisis políticas y financieras.</b> Con la producción por persona cayendo "
        "50-70 %, la experiencia histórica predice muchas más crisis de deuda, golpes y quiebres democráticos, en especial en los "
        "países de ingreso medio que hoy están cerca de los umbrales de 8-22 mil dólares.",
        "<b>Más guerras civiles sólo en los países más pobres y en las caídas más profundas.</b> La guerra civil depende sobre todo "
        "del nivel de pobreza; un mundo que se empobrece mueve países hacia los tramos de mayor riesgo, pero lentamente.",
        "<b>El lazo de vuelta alarga el valle.</b> Cada crisis de deuda o conflicto resta varios puntos de PIB de forma persistente: la "
        "recuperación que World3 muestra después de 2060 sería más lenta y más desigual entre países.",
    ])
    story += [para("Cómo construir el acople (camino 2)", "h3")]
    story += bullets([
        "<b>Una sola vía de entrada:</b> que el ingreso regional del L0 regional (o el PIB de los países en el L1) incorpore los "
        "efectos de la escasez de recursos, la contaminación y la comida. No añadir ecuaciones «hambre → guerra».",
        "<b>Las ecuaciones sociales del Atlas</b> (presión política, conflicto, golpes, democracia, deuda) ya responden al ingreso y al "
        "crecimiento; hay que comprobar que reproducen las razones de esta investigación en caídas grandes, y recalibrar si no.",
        "<b>La vuelta:</b> choques persistentes a la productividad y a la inversión por cada conflicto (≈ −4–5 %), crisis de deuda "
        "(≈ −6–7 %) y golpe (≈ −2–3 %), con sus intervalos.",
        "<b>La capacidad de aplicar políticas</b> debería caer con los conflictos y las crisis de deuda. Es el siguiente análisis "
        "pendiente: medir cómo cambian la recaudación y el gasto público tras esos eventos.",
    ])
    story.append(PageBreak())

    # ── límites ──
    story += [para("Límites", "eyebrow"), para("Lo que estos datos no pueden decir", "h2")]
    story += bullets([
        "<b>Correlación condicionada, no causalidad.</b> Los efectos fijos y los rezagos quitan mucho sesgo, pero no todo: un tercer "
        "factor que cambie a la vez el crecimiento y la estabilidad (una mala política económica, por ejemplo) produciría los mismos "
        "patrones.",
        "<b>Ningún país vivió un valle de World3.</b> Las caídas históricas son locales: el resto del mundo seguía creciendo, "
        "comerciando y prestando. Un valle global quita esos amortiguadores; los efectos podrían ser mayores que los estimados.",
        "<b>Faltan datos clave:</b> calorías por persona (FAOSTAT no se pudo descargar en esta sesión), protestas y disturbios, "
        "homicidios, suicidios y consumo de alcohol, y migraciones. Las disputas entre Estados sólo llegan a 2010 y las guerras "
        "interestatales del COW a 2003.",
        "<b>Medidas gruesas:</b> el rendimiento nacional y la anomalía anual de temperatura promedian regiones enteras; estudios con "
        "datos locales encuentran efectos que aquí no se ven.",
        "<b>Pocos eventos raros:</b> 62 quiebres democráticos, 27 crisis de mortalidad sin guerra. Los intervalos son anchos.",
    ])
    story += [para("Reproducir", "h3"),
              para("<font name='Mono' size='8'>PYTHONPATH=. python reports/acople/investigacion.py</font> (≈ 1,5 min) y "
                   "<font name='Mono' size='8'>PYTHONPATH=. python reports/acople/informe_dinamicas.py</font>. Requiere el Atlas instalado "
                   "(<font name='Mono' size='8'>world4_cli.py --install-atlas</font>); el notebook "
                   "<font name='Mono' size='8'>notebooks/reproducir_reporte.ipynb</font> lo hace."),
              PageBreak()]

    # ── anexo ──
    story += [para("Anexo", "eyebrow"), para("Las 105 relaciones dentro de cada país", "h2"),
              para("Cada fila: un factor, un evento, efectos fijos de país y año, control del ingreso, factor rezagado un año. Efecto en "
                   "puntos porcentuales de probabilidad anual por 1 DE del factor dentro del país, y relativo a la tasa base. q: valor "
                   "corregido por pruebas múltiples (Benjamini-Hochberg).", "cap")]
    rows = [["Evento", "Factor", "Efecto (pp)", "Relativo", "p", "q", "n"]]
    for k, v in sorted(A2.items(), key=lambda kv: kv[1]["q"]):
        o, x = k.split("|")
        rows.append([ONAME[o], DRIVERS.get(x, x), f"{sgn(v['b_sd'] * 100, 2)}", f"{sgn((v['rel'] or 0) * 100, 0)} %",
                     f"{num(v['p'], 3)}" if v["p"] == v["p"] else "—", f"{num(v['q'], 2)}", str(v["n"])])
    story.append(table(rows, [34 * mm, 54 * mm, 18 * mm, 18 * mm, 14 * mm, 12 * mm, TW - 150 * mm], zebra=True))
    doc = Doc(OUT)
    doc.build(story)
    print(f"{OUT} ({num(OUT.stat().st_size / 1e6, 1)} MB)")


if __name__ == "__main__":
    build()
