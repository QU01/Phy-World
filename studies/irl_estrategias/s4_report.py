"""Paso 4b: el informe PDF (reportlab) con las figuras de ``s4_figures`` y las cifras de ``out/*.json``."""
import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, BaseDocTemplate, Frame)

import common as c
import s4_figures as F

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "docs" / "Informe_IRL_estrategias_AtlasMundo_World3.pdf"
FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVS", FD + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("DVSB", FD + "DejaVuSerif-Bold.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

NAVY = colors.HexColor("#14213d")
BLUE = colors.HexColor(F.BLUE)
ORANGE = colors.HexColor(F.ORANGE)
AQUA = colors.HexColor(F.AQUA)
INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor(F.INK2)
SOFT = colors.HexColor("#f3f6fb")
LINE = colors.HexColor("#d9dce3")
W, H = A4
M = 18 * mm

ST = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.4, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.6, leading=10.4, textColor=INK2, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DVI", fontSize=7.6, leading=10.2, textColor=INK2, spaceBefore=2, spaceAfter=9),
    "h1": ParagraphStyle("h1", fontName="DVSB", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=4),
    "kick": ParagraphStyle("kick", fontName="DVB", fontSize=7.8, leading=10, textColor=ORANGE, spaceAfter=1),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=4),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, leftIndent=11, bulletIndent=1,
                          spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.4, leading=9.4, textColor=INK),
    "kpi_v": ParagraphStyle("kv", fontName="DVB", fontSize=17, leading=20, textColor=NAVY, alignment=TA_LEFT),
    "kpi_l": ParagraphStyle("kl", fontName="DV", fontSize=7.4, leading=9.6, textColor=INK2),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.8, leading=12.6, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, ST[s])


def bullets(items):
    return [Paragraph(t, ST["bul"], bulletText="•") for t in items]


def fig(path, width=W - 2 * M, caption=None):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    h = width * ih / iw
    out = [Image(str(path), width=width, height=h)]
    if caption:
        out.append(P(caption, "cap"))
    return KeepTogether(out)


def section(num, title, kicker):
    return [CondPageBreak(60 * mm), P(f"{num} · {kicker.upper()}", "kick"), P(title, "h1"),
            Table([[""]], colWidths=[W - 2 * M], rowHeights=[1.2],
                  style=[("LINEABOVE", (0, 0), (-1, -1), 1.2, ORANGE)]), Spacer(1, 6)]


def table(rows, widths, header=True, zebra=True, align_right_from=1):
    data = [[P(str(x), "cellb" if (header and i == 0) else "cell") for x in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
           ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE)]
    if header:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf5")))
    if zebra:
        for i in range(1 + header, len(rows), 2):
            sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f9fb")))
    t.setStyle(TableStyle(sty))
    return t


def callout(text, color=BLUE):
    t = Table([[P(text, "box")]], colWidths=[W - 2 * M])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return KeepTogether([t, Spacer(1, 8)])


def kpis(items):
    cells = [[Table([[P(v, "kpi_v")], [P(l, "kpi_l")]], colWidths=[(W - 2 * M) / len(items) - 6])
              for v, l in items]]
    t = Table(cells, colWidths=[(W - 2 * M) / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEABOVE", (0, 0), (-1, 0), 2, NAVY),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEAFTER", (0, 0), (-2, -1), 0.5, LINE)]))
    return KeepTogether([t, Spacer(1, 10)])


def pct(x, d=1):
    return f"{x * 100:+.{d}f} %".replace(".", ",")


def fp(x):
    return f"{x * 100:.1f} %".replace(".", ",")


def num(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", ",")


# ---------------------------------------------------------------- páginas
def cover(canv, doc):
    canv.saveState()
    canv.setFillColor(NAVY)
    canv.rect(0, 0, W, H, fill=1, stroke=0)
    # retícula sutil de meridianos
    canv.setStrokeColor(colors.HexColor("#22325a"))
    canv.setLineWidth(0.6)
    cx, cy, r = W * 0.78, H * 0.70, 95 * mm
    for k in range(-5, 6):
        canv.ellipse(cx - r * abs(k) / 5, cy - r, cx + r * abs(k) / 5, cy + r)
    for k in range(-4, 5):
        y = cy + r * k / 5
        dx = r * np.sqrt(max(0, 1 - (k / 5) ** 2))
        canv.line(cx - dx, y, cx + dx, y)
    canv.circle(cx, cy, r)
    canv.setFillColor(colors.HexColor(F.ORANGE))
    canv.rect(M, H * 0.47, 28 * mm, 2.2 * mm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("DVB", 9)
    canv.drawString(M, H * 0.47 + 7 * mm, "QUASAR PHY-WORLD · WORLD4 · ESTUDIO DE ESTRATEGIAS")
    canv.setFont("DVSB", 30)
    y = H * 0.40
    for line in ("¿Qué persiguen", "realmente los países,", "y qué les conviene?"):
        canv.drawString(M, y, line)
        y -= 12.5 * mm
    canv.setFont("DV", 11.5)
    canv.setFillColor(colors.HexColor("#c9d3ea"))
    y -= 3 * mm
    for line in ("Política revelada por aprendizaje por refuerzo inverso en el Atlas del Sistema-Mundo (L1)",
                 "y estrategias de desarrollo evaluadas contra World3-03 (L0): rendimiento, desarrollo,",
                 "daño ambiental y probabilidad de colapso."):
        canv.drawString(M, y, line)
        y -= 6 * mm
    canv.setFont("DV", 8.5)
    canv.setFillColor(colors.HexColor("#9fb0d6"))
    canv.drawString(M, 20 * mm, f"Informe técnico · {dt.date.today():%d/%m/%Y} · resultados exploratorios, no pronósticos")
    canv.drawRightString(W - M, 20 * mm, "studies/irl_estrategias")
    canv.restoreState()


def later(canv, doc):
    canv.saveState()
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.5)
    canv.line(M, H - 12 * mm, W - M, H - 12 * mm)
    canv.setFont("DV", 7)
    canv.setFillColor(INK2)
    canv.drawString(M, H - 10 * mm, "Atlas Mundo (L1) × World3 (L0) · política revelada y estrategias")
    canv.drawRightString(W - M, 10 * mm, f"{doc.page}")
    canv.setFillColor(colors.HexColor(F.ORANGE))
    canv.rect(M, 9.4 * mm, 6 * mm, 1.2 * mm, fill=1, stroke=0)
    canv.restoreState()


# ---------------------------------------------------------------- contenido
def build():
    irl = c.load("irl.json")
    d = c.load("strategies.json")
    sens = c.load("sensitivity.json") if (c.OUT / "sensitivity.json").exists() else None
    L = c.load("learning.json") if (c.OUT / "learning.json").exists() else None
    figs = F.make_all()
    S = d["strategies"]
    base = S["base"]
    ks = F.strat_order(d)
    g = irl["global"]
    alt = g["racionalidad_alternativas"]
    cont = irl["contemporaneo"]["theta_std"]
    rev_prof = g["desvios_rentables_revelada"]
    wel_prof = g["desvios_rentables_bienestar"]
    best_col = min(ks, key=lambda k: S[k]["w3"]["p_collapse_2070"])
    best_pol = min(ks, key=lambda k: S[k]["w3"]["ppolx_peak"])
    best_dev = max(ks, key=lambda k: S[k]["world"]["dev_index"])
    best_hwi = max(ks, key=lambda k: S[k]["w3"]["hwi_2100"])
    paises = irl["paises"]
    prat = sorted(paises.items(), key=lambda kv: -kv[1]["racionalidad"])
    story = [NextPageTemplate("later"), PageBreak()]

    # ------------------------------------------------ resumen ejecutivo
    story += section("00", "Resumen ejecutivo", "en una página")
    story.append(kpis([(f"{irl['n_total']:,}".replace(",", "."), "desvíos contrafactuales simulados para el IRL (6 orígenes, 1960-2005)"),
                       (f"{g['racionalidad']:.1%}".replace(".", ","), f"desigualdades explicadas por la recompensa revelada (azar: {alt['aleatoria_media']:.1%})".replace(".", ",")),
                       (f"{len(S)}", "estrategias probadas en 180 países, 2019-2030, con números aleatorios comunes"),
                       (f"{d['setup']['w3_members']}", "estados del mundo de World3-03 por estrategia, 2020-2100")]))
    fx = L["fixed_scores"] if L else {}
    learned = None
    if L:
        reps = L["world"]["aprendida_mejor"]["validacion"]
        learned = {"score": np.mean([r["score"] for r in reps]), "metrics": {k: np.mean([r["metrics"][k] for r in reps]) for k in reps[0]["metrics"]},
                   "p2070": np.mean([r["w3"]["p_collapse_2070"] - r["base_w3"]["p_collapse_2070"] for r in reps])}
        top_fixed = max((v["score"], k) for k, v in fx.items() if k != "base")
    items = [
        f"<b>Lo que los países persiguen de verdad no es el bienestar de su población.</b> La recompensa que hace óptima la "
        f"conducta observada pesa sobre todo la <b>inercia</b> (no cambiar el presupuesto: {num(g['theta_std']['inercia'])}) y el "
        f"<b>poder relativo</b> (cuota del ingreso mundial: {num(g['theta_std']['poder'])}); el peso del consumo per cápita es "
        f"<b>negativo</b> ({num(g['theta_std']['bienestar'], sign=True)}). Explica {fp(g['racionalidad'])} de las condiciones de "
        f"optimalidad, frente a {fp(alt['bienestar'])} de una recompensa de bienestar y {fp(alt['aleatoria_media'])} del azar.",
        f"<b>La política está cambiando.</b> En los orígenes 2000-2005 el bienestar pasa a pesar en positivo ({num(cont['bienestar'], sign=True)}), "
        f"el poder sube a {num(cont['poder'])} y aparece la autonomía en recursos ({num(cont['autonomia'], sign=True)}). El centro del sistema-mundo "
        f"se explica por la paz; la periferia y las autocracias, por poder e inercia con bienestar negativo.",
        f"<b>Países:</b> la conducta más «racional» (mejor explicada por una recompensa) es la de {prat[0][0]} ({prat[0][1]['racionalidad']:.0%}), "
        f"{prat[1][0]} y {prat[2][0]}; la menos, {prat[-1][0]} ({prat[-1][1]['racionalidad']:.0%}). En {sum(v['racionalidad'] > v['racionalidad_bienestar'] for v in paises.values())} "
        f"de {len(paises)} países foco la recompensa revelada explica más que la de bienestar.",
        f"<b>Desarrollo en el L1 (2030):</b> el <b>{S[best_dev]['label']}</b> es la mejor estrategia para el desarrollo "
        f"({num((S[best_dev]['world']['dev_index'] - base['world']['dev_index']) * 100, sign=True)} puntos de índice, "
        f"{pct(S[best_dev]['world']['gdp_pc'] / base['world']['gdp_pc'] - 1)} de PIB per cápita mundial) y la mejor para 173 de 180 "
        f"países por separado; pero sube la energía fósil {pct(S[best_dev]['world']['fossil_cum_twh'] / base['world']['fossil_cum_twh'] - 1)} "
        f"y, en World3, la probabilidad de colapso antes de 2070 (de {base['w3']['p_collapse_2070']:.0%} a {S[best_dev]['w3']['p_collapse_2070']:.0%}).",
        f"<b>Colapso (World3):</b> bajo la línea base el colapso antes de 2070 ocurre en el {base['w3']['p_collapse_2070']:.0%} de los mundos. "
        f"Lo que más lo reduce es <b>{" y ".join(S[k]['label'] for k in ks if S[k]['w3']['p_collapse_2070'] <= S[best_col]['w3']['p_collapse_2070'] + 0.005)}</b> ({S[best_col]['w3']['p_collapse_2070']:.0%}): más servicios sociales aceleran la "
        f"transición demográfica y desvían capital de la industria contaminante. Lo que más reduce el daño ambiental es <b>{S[best_pol]['label']}</b> "
        f"(pico de contaminación {num(S[best_pol]['w3']['ppolx_peak'], 1)} frente a {num(base['w3']['ppolx_peak'], 1)}), pero en World3 la "
        f"tecnología consume capital y no evita el colapso económico.",
    ]
    if L:
        items.append(f"<b>Aprendizaje:</b> el método de entropía cruzada encontró una estrategia diferenciada por zona con puntuación "
                     f"compuesta {num(learned['score'], 2, True)} fuera de muestra, frente a {num(top_fixed[0], 2, True)} de la mejor estrategia "
                     f"fija ({S[top_fixed[1]]['label']}): más redistribución en todas partes, menos acumulación de capital en el centro y "
                     f"más en la periferia. Cambia P(colapso ≤ 2070) en {num(learned['p2070'] * 100, 1, True)} puntos y el pico de "
                     f"contaminación en {pct(learned['metrics']['ppolx'], 0)}.")
    RLx = F.rl_results()
    if RLx:
        scx = {a: np.mean([e["score"] for e in RLx[a]["eval"]]) for a in RLx}
        bx = max(scx, key=scx.get)
        evx = RLx[bx]["eval"]
        mx = lambda k: np.mean([e["metrics"][k] for e in evx])  # noqa: E731
        items.append(f"<b>Aprendizaje por refuerzo:</b> cinco agentes (on-policy PPO; off-policy SAC y TD3; basados en modelo "
                     f"PETS-MPC y MBPO-Dyna) aprendieron una política que reacciona al estado del mundo cada dos años. El mejor, "
                     f"<b>{F.RL_NAME[bx]}</b>, obtiene {num(scx[bx], 2, True)} fuera de muestra (CEM {num(learned['score'], 2, True) if learned else '—'}): "
                     f"{num(mx('p2070') * 100, 1, True)} pp de colapso antes de 2070, {num(mx('psevere') * 100, 1, True)} pp de colapso "
                     f"severo y {pct(mx('ppolx'), 0)} de pico de contaminación, sin perder desarrollo. Su receta: el centro redistribuye "
                     f"más y acumula cada vez menos capital; la periferia invierte fuerte al principio y extrae menos.")
    items.append("<b>Ninguna estrategia dentro de la región de confianza del IRL (±0,5 logit) evita el colapso de World3 en la mayoría de "
                 "los mundos.</b> Mueven su fecha y su severidad; el resultado clásico de <i>Los límites del crecimiento</i> sobrevive a la "
                 "conducta realista de 180 países.")
    story += bullets(items)
    story.append(Spacer(1, 4))
    story.append(callout("<b>Cómo leer este informe.</b> Todo resultado es un <b>modo de comportamiento exploratorio</b>, no un pronóstico. "
                         "El L1 está validado fuera de muestra hasta 10 años; el L0 (World3-03) es un modelo agregado de 1972/2004 sin "
                         "calibración a datos recientes. El acoplamiento L1→L0 de este estudio es un supuesto declarado [M] (sección 4.3). "
                         "Las limitaciones están en la sección 8.", ORANGE))

    # ------------------------------------------------ 1. qué se hizo
    story += section("01", "Qué se hizo", "método")
    story.append(fig(figs["pipeline"], caption="Figura 1. Flujo del estudio. El IRL convierte la conducta observada en una recompensa "
                                                "por país; esa recompensa y la de bienestar valoran cada estrategia; el L0 juzga sus "
                                                "consecuencias biofísicas a 2100."))
    story += [P("<b>Los dos niveles del modelo.</b>", "h2"),
              P("<b>L1 — Atlas del Sistema-Mundo</b> (portado sin cambios, paridad bit a bit con su commit 9e96f6b): 180 países con "
                "producción, tecnología y difusión, deuda, élites y presión estructural-demográfica, conflicto civil, democracia, recursos y "
                "centro-periferia. Cada país asigna su ahorro y divisas a seis <b>canales</b>: capital doméstico (k), tecnología/I+D y "
                "educación (r), importación de bienes del centro (m), extracción propia de recursos (x), inversión exterior (f) y "
                "redistribución (w). World4 le acopla población por cohortes y capital humano endógenos (F1), mercados mundiales de "
                "recursos (F2) y la mezcla energética con EROI (F3)."),
              P("<b>L0 — World3-03</b> (reimplementado; error máximo frente a Vensim 6,9·10⁻⁶): población, capital industrial y de "
                "servicios, agricultura, recursos no renovables y contaminación persistente del mundo agregado, de 1900 a 2100."),
              P("<b>Paso 1 · Aprendizaje por refuerzo inverso (IRL).</b>", "h2"),
              P("Estimador de Bajari, Benkard y Levin (2007), el mismo del Atlas, ampliado. (1) Se clona la regla de decisión que los países "
                "siguieron (ajuste parcial en logit por canal, estimado sólo con datos hasta cada origen; R² 0,79-0,97). (2) Desde seis "
                "orígenes (1960, 1970, 1980, 1990, 2000, 2005) se simula 15 años con todos los países siguiendo la regla y se registran sumas "
                "descontadas (γ = 0,95) de seis rasgos de recompensa: bienestar (crecimiento del consumo per cápita), poder (cuota del "
                "ingreso mundial), élite (renta por miembro de la élite), paz (−riesgo de conflicto), autonomía (−dependencia de recursos "
                "ajenos) e inercia (−coste de ajustar el presupuesto). (3) Subconjuntos de países se desvían ±0,5 logit en un canal con "
                "todo lo demás idéntico, incluidos los números aleatorios. Si la conducta es óptima para r = θ·φ, cada desvío debe empeorar "
                "θ·ΔW. (4) θ es la dirección que mejor satisface esas desigualdades; la fracción satisfecha es el <b>índice de racionalidad</b>."),
              P(f"Se simularon <b>{irl['n_total']:,} desvíos</b>: 200 sorteos aleatorios por origen (15 % de los países) como en el Atlas, más "
                f"150 sorteos por origen en los que los <b>20 países foco</b> se desvían a la vez, lo que permite estimar una recompensa "
                f"revelada <b>por país</b> (≈1.070 desigualdades cada uno) con bootstrap.".replace(",", ".")),
              P("<b>Paso 2 · Estrategias en el L1 y su consecuencia en el L0.</b>", "h2"),
              P(f"Desde el origen vivo 2019 hasta 2030, {d['setup']['R']} miembros por corrida con números aleatorios comunes. Cada "
                "estrategia desplaza el logit de la regla clonada (la misma herramienta y el mismo tamaño que el IRL: una región de "
                "confianza alrededor de lo observado, no un óptimo de esquina) y, en tres casos, las perillas de energía y fecundidad del "
                "L1, con un coste declarado en capital. Se midió la adopción <b>global</b> y la <b>cuasi-unilateral</b> (6 grupos de países "
                "al azar; cada país se mide cuando su grupo adopta). Seis señales mundiales del L1 se traducen en constantes de política "
                "de World3 desde 2020 (sección 4.3) y se corre World3 sobre 400 estados del mundo."),
              P("<b>Paso 3 · Aprendizaje.</b>", "h2"),
              P("Un método de entropía cruzada (CEM) busca (a) una estrategia mundial diferenciada por zona del sistema-mundo con un "
                "objetivo de tres pilares (desarrollo, ambiente, colapso), validada con semillas nuevas, y (b) la mejor respuesta de cada "
                "uno de los 20 países foco aprendiendo a la vez.")]
    story.append(table([["Estrategia", "Instrumento en el L1", "Qué representa"]] +
                       [[S[k]["label"], ", ".join(f"{ch} {v:+.2f}" for ch, v in S[k]["offsets"].items()) +
                         ("; " + ", ".join(f"{a} {b:+g}/año" for a, b in S[k]["calib"].items()) if S[k]["calib"] else ""),
                         S[k]["what"]] for k in ks], [44 * mm, 52 * mm, W - 2 * M - 96 * mm]))
    story.append(P("Tabla 1. Las estrategias. Canales en logit (±0,5 = la región de confianza del IRL); «intensity», «nonfossil» y «tfr» "
                   "son las perillas del L1 (eficiencia energética, difusión no fósil, fecundidad) por año; su coste en k ya está incluido.",
                   "cap"))

    # ------------------------------------------------ 2. IRL
    story += section("02", "La política real: lo que revela la conducta", "aprendizaje por refuerzo inverso")
    story.append(fig(figs["irl_rationality"], caption="Figura 2. Índice de racionalidad. La recompensa revelada explica más desigualdades "
                                                       "de optimalidad que cualquier recompensa «normativa» y que el azar."))
    story.append(fig(figs["irl_weights"], caption="Figura 3. Pesos de la recompensa revelada (estandarizados). Barras sólidas: todos los "
                                                   "orígenes, con IC 90 % por bootstrap de países; rayadas: sólo orígenes 2000-2005."))
    story += [P(f"La lectura es incómoda pero robusta: los gobiernos actúan <b>como si</b> maximizaran su peso en el mundo y la "
                f"estabilidad de sus presupuestos, y aceptaran a cambio menos consumo para su población. El intervalo del peso del "
                f"bienestar ({num(g['theta_std_ic90']['bienestar'][0])} a {num(g['theta_std_ic90']['bienestar'][1])}) excluye el cero. "
                f"Coincide con el IRL original del Atlas (bienestar {num(irl['atlas_original']['theta_std']['bienestar'], sign=True)}, poder "
                f"{num(irl['atlas_original']['theta_std']['poder'], sign=True)}, inercia {num(irl['atlas_original']['theta_std']['inercia'], sign=True)}), "
                f"ahora con un origen más y el doble de desvíos. Una recompensa con sólo inercia ya explica "
                f"{fp(g['parsimonioso']['racionalidad'])}: la persistencia presupuestaria es el rasgo más fuerte de la política real."),
              ]
    story.append(fig(figs["irl_groups"], caption="Figura 4. Recompensa revelada por zona del sistema-mundo (cuantiles de ingreso), régimen "
                                                  "(V-Dem) y origen. Escala divergente: azul = peso positivo, naranja = negativo."))
    story += bullets([
        "<b>Centro:</b> su conducta se explica casi sólo por <b>paz</b> (+0,97): protegen la estabilidad. La periferia y las autocracias "
        "comparten la firma global: poder e inercia, bienestar negativo.",
        "<b>Democracias:</b> peso positivo en poder (+0,18) y peso negativo en paz (−0,94): en el modelo su conducta tolera más riesgo de "
        "conflicto del que habría con desvíos alternativos (probablemente porque el riesgo de conflicto en democracias ricas es casi nulo y "
        "poco sensible).",
        "<b>Tendencia histórica:</b> la racionalidad sube de 53 % (1960) a 62 % (2005) y el bienestar pasa de −0,54 (1970) a +0,11/+0,17 "
        "(2000-2005): la política de los países se ha vuelto más coherente y algo más orientada al bienestar."])
    story.append(fig(figs["irl_countries"], caption="Figura 5. Recompensa revelada de los 20 países foco (pesos estandarizados) e índice "
                                                     "de racionalidad bajo su recompensa propia (azul) y la de bienestar (gris), en puntos "
                                                     "sobre 50 %. Ordenados por racionalidad."))
    # tabla de países
    rows = [["País", "Racion.", "Rasgo dominante", "Segundo rasgo", "¿Qué cambiaría? (desvío más rentable bajo su θ)"]]
    for iso, v in prat:
        th = v["theta_std"]
        o = sorted(c.FEATS, key=lambda f: -abs(th[f]))
        dv = v["desvios_rentables_revelada"]
        best = max(dv, key=dv.get)
        rows.append([iso, f"{v['racionalidad']:.0%}", f"{F.FEAT_1[o[0]]} ({th[o[0]]:+.2f})", f"{F.FEAT_1[o[1]]} ({th[o[1]]:+.2f})",
                     f"{c.CH_ES[best[0]]} {'+' if best[1] == '+' else '−'}0,5 ({dv[best]:.0%} de los casos)"])
    story.append(table(rows, [13 * mm, 15 * mm, 40 * mm, 40 * mm, W - 2 * M - 108 * mm]))
    story.append(P("Tabla 2. Política revelada por país. Signo + = el país lo persigue; − = lo sacrifica. Última columna: el desvío que, "
                   "bajo la propia recompensa del país, habría sido rentable más a menudo (lo que «quisiera» hacer más).", "cap"))
    story.append(fig(figs["irl_deviations"], caption="Figura 6. Desvíos rentables. Bajo la recompensa revelada, a los países les habría "
                                                      "convenido más I+D (r+) y más extracción (x+); bajo una recompensa de bienestar, les "
                                                      "habría convenido gastar MENOS en todo y consumir más (el bienestar del IRL es miope: "
                                                      "15 años con γ = 0,95)."))
    story.append(callout(f"<b>Conclusión del paso 1.</b> La «política real» de los países es una mezcla de <b>inercia presupuestaria</b> y "
                         f"<b>búsqueda de poder relativo</b>, con el bienestar de la población como variable de ajuste. Bajo su propia "
                         f"recompensa, los países querrían invertir más en tecnología ({rev_prof['r+']:.0%} de los desvíos r+ rentables) y "
                         f"extraer más recursos ({rev_prof['x+']:.0%}); importar menos del centro ({rev_prof['m-']:.0%})."))

    # ------------------------------------------------ 3. L1
    story += section("03", "Estrategias en el L1: rendimiento y desarrollo de cada nación", "2019 → 2030")
    story.append(fig(figs["l1_scatter"], caption="Figura 7. Efecto mundial de cada estrategia en 2030 frente a la línea base (mediana "
                                                  "de diferencias pareadas, 32 miembros). Índice de desarrollo: media geométrica tipo IDH "
                                                  "de ingreso, capital humano y paz (1 − riesgo de conflicto)."))
    rows = [["Estrategia", "PIB pc mundial", "Consumo pc", "Índice desarrollo", "ID quintil pobre", "Fósil 2020-30", "Riesgo conflicto",
             "GMST 2030"]]
    for k in ["base"] + ks:
        w = S[k]["world"]
        b = base["world"]
        if k == "base":
            rows.append([S[k]["label"], f"{w['gdp_pc']:,.0f} USD".replace(",", "."), f"{w['cons_pc']:,.0f} USD".replace(",", "."),
                         num(w["dev_index"] * 100, 2), num(w["dev_index_poorest_quintile"] * 100, 2),
                         f"{w['fossil_cum_twh'] / 1e6:.2f} M TWh".replace(".", ","), f"{w['hazard']:.2%}".replace(".", ","),
                         f"{w['Tg_2030']:.2f} °C".replace(".", ",")])
        else:
            rows.append([S[k]["label"], pct(w["gdp_pc"] / b["gdp_pc"] - 1), pct(w["cons_pc"] / b["cons_pc"] - 1),
                         num((w["dev_index"] - b["dev_index"]) * 100, 2, True),
                         num((w["dev_index_poorest_quintile"] - b["dev_index_poorest_quintile"]) * 100, 2, True),
                         pct(w["fossil_cum_twh"] / b["fossil_cum_twh"] - 1), f"{(w['hazard'] - b['hazard']) * 100:+.2f} pp".replace(".", ","),
                         f"{w['Tg_2030'] - b['Tg_2030']:+.3f} °C".replace(".", ",")])
    story.append(table(rows, [42 * mm] + [(W - 2 * M - 42 * mm) / 7] * 7))
    story.append(P("Tabla 3. Resultados mundiales 2030. Primera fila: niveles de la línea base; resto: diferencias frente a ella "
                   "(ponderadas por población). GMST según la TCRE del Atlas (los datos de FaIR no estaban disponibles).", "cap"))
    story += [P("<b>Lecturas.</b>", "h2")] + bullets([
        "El <b>Estado desarrollista</b> (más capital y más I+D) es la única estrategia que mueve el desarrollo de forma apreciable en 11 "
        "años; su coste es consumo presente y más deuda.",
        "La <b>transición verde</b> pagada con capital reduce la energía fósil un 17 % pero cuesta ~7 % de PIB per cápita en 2030; el "
        "<b>desarrollo verde</b> (con más I+D y menos extracción) consigue −14,5 % de fósiles <b>sin perder desarrollo</b>: es la "
        "combinación más eficiente del plano desarrollo–ambiente.",
        "El <b>Estado de bienestar</b> no cambia el PIB en el L1 (las transferencias son internas), pero cambia lo que World3 recibe.",
        "«Lo que conviene» según la recompensa de bienestar del IRL (gastar menos en todo) da más consumo inmediato y el peor desarrollo."])
    story.append(fig(figs["l1_countries"], caption="Figura 8. Δ índice de desarrollo 2030 (puntos) para cada país foco cuando su grupo "
                                                    "adopta la estrategia (cuasi-unilateral). *Perillas mundiales: adopción global."))
    story.append(fig(figs["best_counts"], caption="Figura 9. ¿Qué estrategia es la mejor para cada nación? Depende de qué se maximice: el "
                                                   "desarrollo lo da el desarrollismo; el bienestar de consumo, gastar menos; la recompensa "
                                                   "que los propios países revelan, «más de lo mismo» o la transición verde."))
    story.append(callout("<b>Rendimiento por nación.</b> Hay una brecha sistemática entre lo que maximiza el <b>desarrollo</b> de casi todos "
                         "los países (Estado desarrollista: 173 de 180), lo que maximiza su <b>bienestar inmediato</b> (consumir: 167 de 180) "
                         "y lo que maximiza la <b>recompensa que revelan</b> (intensificar su conducta actual o la transición verde). "
                         "Egipto y Venezuela son las excepciones visibles: para ellos el desarrollismo resta desarrollo en el modelo "
                         "(deuda y conflicto) y les conviene la apertura."))

    # ------------------------------------------------ 4. L0
    story += section("04", "Consecuencias en el L0: World3, daño ambiental y colapso", "2020 → 2100")
    story.append(fig(figs["w3_bars"], caption="Figura 10. World3-03 bajo cada estrategia (400 estados del mundo). Colapso: caída ≥ 20 % "
                                               "de la población o ≥ 50 % del producto industrial per cápita desde su máximo; severo: ≥ 40 % "
                                               "o ≥ 75 %. Líneas punteadas: línea base."))
    rows = [["Estrategia", "Colapso ≤ 2050", "Colapso ≤ 2070", "Colapso severo", "Año mediano", "Pico contam.", "Sobre-carga (años)",
             "HWI 2100", "Vida 2100 (años)"]]
    for k in ["base"] + ks:
        w3 = S[k]["w3"]
        rows.append([S[k]["label"], f"{w3['p_collapse_2050']:.0%}", f"{w3['p_collapse_2070']:.0%}", f"{w3['p_severe']:.0%}",
                     f"{w3['collapse_year_median']:.0f}" if w3["collapse_year_median"] else "—", num(w3["ppolx_peak"], 1),
                     num(w3["overshoot_hef_years"], 0), num(w3["hwi_2100"], 3), num(w3["le_2100"], 1)])
    story.append(table(rows, [44 * mm] + [(W - 2 * M - 44 * mm) / 8] * 8))
    story.append(P("Tabla 4. Indicadores de World3 (medianas o probabilidades sobre 400 mundos). Pico de contaminación: índice 1970 = 1. "
                   "Años en sobrecarga: suma de (huella ecológica − 1) 2020-2100. HWI: índice de bienestar humano de World3-03.", "cap"))
    story.append(fig(figs["w3_bands"], caption="Figura 11. Trayectorias de World3: la línea base, la estrategia que menos colapsa, la "
                                                "que menos contamina y el desarrollismo."))
    story += [P("<b>Lecturas.</b>", "h2")] + bullets([
        f"El colapso es la norma: {base['w3']['p_collapse_2070']:.0%} de los mundos antes de 2070 bajo la conducta revelada. Las "
        "estrategias desplazan esa probabilidad entre 77 % y 97 %.",
        f"<b>{S[best_col]['label']}</b> y el <b>Estado de bienestar</b> son las que más reducen el colapso y el pico de contaminación a la vez, "
        "por la vía de los servicios: más salud y educación aceleran la caída de la fecundidad y desvían inversión de la industria.",
        f"La <b>Transición demográfica</b> da el mayor bienestar humano en 2100 (HWI {num(S['demografica']['w3']['hwi_2100'], 3)}, "
        f"esperanza de vida {num(S['demografica']['w3']['le_2100'], 1)} años).",
        "El <b>desarrollismo</b> y la <b>apertura</b> aumentan el colapso: más capital industrial es más contaminación y agotamiento.",
        "La <b>transición verde</b> reduce la contaminación un tercio, pero en World3 la tecnología de recursos y contaminación cuesta "
        "capital (tablas icor de World3-03): el producto industrial cae antes y el colapso es más severo. Es el resultado conocido del "
        "escenario 6 de <i>Limits to Growth</i>: la tecnología sola no basta."])
    story.append(P("<b>4.3 · El acoplamiento L1 → L0 y su sensibilidad</b>", "h2"))
    story.append(P("Cada estrategia produce en el L1 seis señales relativas a la línea base que se trasladan a World3 desde 2020 y se "
                   "mantienen hasta 2100 [M]: (1) la razón consumo/producto escala la fracción consumida del producto industrial (fioac2); "
                   "(2) la mejora anual de la intensidad fósil del PIB es la tasa de tecnología de recursos; (3) esa mejora más la ganancia "
                   "de cuota no fósil es la tasa de tecnología de contaminación; (4) el gasto social (canal w) escala los servicios per "
                   "cápita indicados; (5) el crecimiento no explicado por capital escala la relación capital-producto; (6) la población "
                   "2030 escala el tamaño de familia deseado. Además, cada estado del mundo tiene un progreso tecnológico autónomo (0-2 %, "
                   "0-3 % y 0-2 % anual en recursos, contaminación y rendimientos), igual para todas las estrategias."))
    story.append(table([["Estrategia", "fioac2 ×", "Tec. recursos %/año", "Tec. contam. %/año", "Servicios ×", "Capital/producto ×",
                         "Familia ×"]] +
                       [[S[k]["label"], num(S[k]["w3_mapping"]["fioac2_factor"], 3), num(S[k]["w3_mapping"]["resource_tech_rate"] * 100, 2),
                         num(S[k]["w3_mapping"]["pollution_tech_rate"] * 100, 2), num(S[k]["w3_mapping"]["services_factor"], 3),
                         num(S[k]["w3_mapping"]["icor_factor"], 3), num(S[k]["w3_mapping"]["family_factor"], 3)] for k in ks],
                       [42 * mm] + [(W - 2 * M - 42 * mm) / 6] * 6))
    story.append(P("Tabla 5. Lo que cada estrategia del L1 le dice a World3.", "cap"))
    if sens:
        story.append(fig(figs["sensitivity"], caption="Figura 12. Atribución: cuántos puntos de P(colapso ≤ 2070) aporta cada canal del "
                                                       "acoplamiento (el canal se apaga y se mide la diferencia). Naranja = el canal AUMENTA "
                                                       "el colapso; azul = lo reduce."))
        same = all(np.sign(sens[k]["intensity"]["0.5"]["p_collapse_2070"] - sens["_base"]["p_collapse_2070"]) ==
                   np.sign(sens[k]["intensity"]["2.0"]["p_collapse_2070"] - sens["_base"]["p_collapse_2070"]) for k in ks
                   if abs(S[k]["w3"]["p_collapse_2070"] - base["w3"]["p_collapse_2070"]) > 0.01)
        story.append(P(f"<b>Robustez:</b> con todas las señales a la mitad o al doble, el signo del efecto sobre el colapso "
                       f"{'se mantiene en todas las estrategias con efecto apreciable' if same else 'cambia en alguna estrategia'}; "
                       f"la magnitud escala aproximadamente con la intensidad (p. ej., Estado de bienestar: "
                       f"{sens['bienestar_social']['intensity']['0.5']['p_collapse_2070']:.0%} a ×0,5 y "
                       f"{sens['bienestar_social']['intensity']['2.0']['p_collapse_2070']:.0%} a ×2)."))

    # ------------------------------------------------ 5. aprendizaje
    if L:
        story += section("05", "Aprendizaje de estrategias", "entropía cruzada")
        story.append(P("El objetivo compuesto pondera por igual tres pilares, cada métrica dividida por su dispersión entre las estrategias "
                       "fijas: <b>desarrollo</b> (Δ índice mundial, Δ índice del quintil más pobre, Δ valor de bienestar), <b>ambiente</b> "
                       "(−Δ energía fósil, −Δ pico de contaminación de World3) y <b>colapso</b> (−ΔP(colapso ≤ 2070), −ΔP(severo), "
                       "+ΔHWI 2100). La búsqueda tiene 21 dimensiones: seis canales por zona (caja ±0,5) y tres perillas mundiales con coste."))
        story.append(fig(figs["learning_curve"], caption="Figura 13. Convergencia del CEM (12 miembros por evaluación, semilla de búsqueda)."))
        story.append(fig(figs["learned_policy"], width=(W - 2 * M) * 0.9,
                         caption="Figura 14. La estrategia aprendida: desvío de cada canal (logit) por zona del sistema-mundo."))
        lw = L["world"]["aprendida_mejor"]
        kn = lw["perillas"]
        story.append(P(f"Perillas mundiales aprendidas: eficiencia energética {kn['intensity'] * 100:+.2f} %/año, difusión no fósil "
                       f"{kn['nonfossil']:+.3f} logit/año, fecundidad {kn['tfr'] * 100:+.2f} %/año".replace(".", ",") + "."))
        story.append(fig(figs["pillars"], caption="Figura 15. Clasificación final por pilares (desviaciones típicas frente a la línea base). "
                                                   "La estrategia aprendida se evalúa fuera de muestra (semillas 101 y 202, R = 32)."))
        reps = lw["validacion"]
        mv = lambda k: np.mean([r["metrics"][k] for r in reps])  # noqa: E731
        story.append(table([["Métrica (aprendida − base, fuera de muestra)", "Valor"],
                            ["Δ índice de desarrollo mundial (puntos)", num(mv("dev") * 100, 3, True)],
                            ["Δ índice de desarrollo del quintil más pobre (puntos)", num(mv("dev_q1") * 100, 3, True)],
                            ["Δ valor de bienestar (ponderado por población)", num(mv("welfare"), 4, True)],
                            ["Δ energía fósil acumulada 2020-2030", pct(mv("fossil"))],
                            ["Δ pico de contaminación de World3", pct(mv("ppolx"))],
                            ["Δ P(colapso ≤ 2070)", f"{mv('p2070') * 100:+.1f} pp".replace(".", ",")],
                            ["Δ P(colapso severo)", f"{mv('psevere') * 100:+.1f} pp".replace(".", ",")],
                            ["Δ HWI 2100", num(mv("hwi"), 3, True)]], [110 * mm, 40 * mm]))
        story.append(P("Tabla 6. La estrategia aprendida frente a la línea base, media de dos semillas no usadas en la búsqueda.", "cap"))
        story.append(callout(f"<b>Qué aprendió el planificador.</b> Redistribuir más en todas las zonas (w {lw['por_zona']['periferia']['w']:+.2f} / "
                             f"{lw['por_zona']['semiperiferia']['w']:+.2f} / {lw['por_zona']['centro']['w']:+.2f}), que el <b>centro</b> invierta menos "
                             f"en capital ({lw['por_zona']['centro']['k']:+.2f}) y la <b>periferia</b> más ({lw['por_zona']['periferia']['k']:+.2f}) con más "
                             f"importación de bienes con tecnología ({lw['por_zona']['periferia']['m']:+.2f}) y menos extracción "
                             f"({lw['por_zona']['periferia']['x']:+.2f}), más eficiencia moderada y planificación familiar. Compra ambiente "
                             f"({pct(mv('ppolx'))} de pico de contaminación, {pct(mv('fossil'))} de energía fósil) y "
                             f"{num(mv('p2070') * 100, 1, True)} pp de colapso antes de 2070 con desarrollo mundial casi igual y mejor para el quintil más "
                             f"pobre; el precio es un HWI 2100 algo menor ({num(mv('hwi'), 3, True)}). Es convergencia: menos acumulación en el "
                             f"centro, más en la periferia.".replace("pp de", "pp de").replace(".1f", "")))
        story.append(fig(figs["learned_countries"], caption="Figura 16. Aprendizaje multiagente: la mejor respuesta que aprende cada país "
                                                             "foco (todos aprendiendo a la vez, cada uno con su propio objetivo de desarrollo "
                                                             "y bienestar) y su Δ índice de desarrollo nacional."))
        C = L["countries"]
        agg = np.array([[C[x]["offsets"][ch] for ch in c.CHANNELS] for x in C])
        story.append(callout("<b>Qué aprendieron los países.</b> En promedio, la mejor respuesta aprendida es: " +
                             ", ".join(f"{c.CH_ES[ch].lower()} {v:+.2f}" for ch, v in zip(c.CHANNELS, agg.mean(0))) +
                             f". Los canales con signo común en la mayoría de los países son: " +
                             ", ".join(f"{c.CH_ES[ch].lower()} ({'+' if np.mean(agg[:, j] > 0) > 0.5 else '−'} en "
                                       f"{max(np.mean(agg[:, j] > 0), np.mean(agg[:, j] < 0)):.0%})"
                                       for j, ch in enumerate(c.CHANNELS) if max(np.mean(agg[:, j] > 0), np.mean(agg[:, j] < 0)) >= 0.7)
                             + "."))

    # ------------------------------------------------ 5b. aprendizaje por refuerzo
    RL = F.rl_results()
    if RL:
        story += section("06", "Aprendizaje por refuerzo: la estrategia óptima en lazo cerrado", "RL on-policy, off-policy y basado en modelo")
        story.append(P("El CEM del paso 3 elige un vector fijo para toda la década. Aquí el planificador es un <b>agente que observa y "
                       "reacciona</b>: en 2020, 2022, 2024, 2026 y 2028 observa el estado del L1 por zona (PIB per cápita, crecimiento "
                       "del consumo, riesgo de conflicto, deuda, intensidad fósil; tiempo, renta de recursos, deuda mundial y la media "
                       "de sus acciones pasadas: 36 rasgos) y fija los desvíos de los 6 canales en cada zona (18 acciones continuas en "
                       "±0,5 logit). La recompensa es la misma puntuación compuesta de tres pilares del paso 3 (más una recompensa "
                       "intermedia de potencial que no cambia el total), frente al mismo miembro de la línea base con números "
                       "aleatorios comunes. Cada corrida del L1 da 48 episodios en paralelo. Los términos de World3 se calculan durante "
                       "el entrenamiento con un <b>emulador</b> (red neuronal entrenada con 600 corridas de World3 sobre 400 mundos; R² "
                       "fuera de muestra 0,99-1,00); la evaluación final usa World3 de verdad."))
        fam = [["Algoritmo", "Familia", "Cómo aprende", "Episodios reales"]]
        how = {"PPO": "política gaussiana y crítico; gradiente de política con recorte y GAE, sólo con datos de la política actual",
               "SAC": "actor-crítico con entropía máxima y doble Q; reutiliza toda la experiencia (memoria de repetición)",
               "TD3": "actor determinista, doble Q con retardo y suavizado del objetivo; memoria de repetición",
               "PETS": "ensemble de 5 redes aprende la dinámica y la recompensa; en cada decisión planifica con CEM sobre el modelo (MPC)",
               "MBPO": "el mismo modelo genera ramas cortas desde estados reales; SAC aprende sobre 90 % de datos imaginados"}
        for a, r in RL.items():
            fam.append([F.RL_NAME[a].split(" (")[0], r["family"], how[a], f"{r['curve'][-1]['episodes']:,}".replace(",", ".")])
        story.append(table(fam, [24 * mm, 30 * mm, W - 2 * M - 80 * mm, 26 * mm]))
        story.append(P("Tabla 7. Los cinco agentes. Off-policy y basados en modelo arrancan con 20 corridas aleatorias compartidas "
                       "(960 episodios); PPO, on-policy, no puede usarlas.", "cap"))
        story.append(fig(figs["rl_curves"], caption="Figura 17. Curvas de aprendizaje frente a la experiencia real consumida."))
        story.append(fig(figs["rl_compare"], caption="Figura 18. Comparación final con la misma evaluación que el CEM: semillas 101 y 202 "
                                                      "(nunca vistas), R = 32, World3 de verdad sobre 400 mundos."))
        best = max(RL, key=lambda a: np.mean([e["score"] for e in RL[a]["eval"]]))
        ev = RL[best]["eval"]
        mvb = lambda k: np.mean([e["metrics"][k] for e in ev])  # noqa: E731
        rows = [["Agente", "Puntuación", "Δ ID mundial", "Δ ID quintil pobre", "Δ fósil", "Δ pico contam.", "Δ P(col. ≤2070)",
                 "Δ P(severo)", "Δ HWI 2100"]]
        allrows = [(F.RL_NAME[a], RL[a]["eval"]) for a in RL]
        if L:
            allrows.append(("CEM (paso 3)", L["world"]["aprendida_mejor"]["validacion"]))
        for name, evs in sorted(allrows, key=lambda r: -np.mean([e["score"] for e in r[1]])):
            mmet = lambda k: np.mean([e["metrics"][k] for e in evs])  # noqa: E731
            rows.append([name, num(np.mean([e["score"] for e in evs]), 3, True), num(mmet("dev") * 100, 3, True),
                         num(mmet("dev_q1") * 100, 3, True), pct(mmet("fossil")), pct(mmet("ppolx")),
                         f"{num(mmet('p2070') * 100, 1, True)} pp", f"{num(mmet('psevere') * 100, 1, True)} pp", num(mmet("hwi"), 3, True)])
        story.append(table(rows, [34 * mm] + [(W - 2 * M - 34 * mm) / 8] * 8))
        story.append(P("Tabla 8. Resultados fuera de muestra frente a la línea base (media de las dos semillas).", "cap"))
        story.append(fig(figs["rl_policy"], caption=f"Figura 19. La política aprendida por el mejor agente ({F.RL_NAME[best]}): "
                                                     "acción media por zona, canal y año de decisión (logit; azul = más, naranja = menos)."))
        A = np.mean([np.array(e["actions_mean"]) for e in ev], 0)
        Sd = np.mean([np.array(e["actions_sd"]) for e in ev], 0)
        zn = ("periferia", "semiperiferia", "centro")
        desc = []
        for z in range(3):
            m = A[:, z * 6:(z + 1) * 6].mean(0)
            o = np.argsort(-np.abs(m))[:3]
            desc.append(f"<b>{zn[z]}</b>: " + ", ".join(f"{c.CH_ES[c.CHANNELS[j]].lower()} {m[j]:+.2f}" for j in o))
        trend = A[-1] - A[0]
        j = int(np.argmax(np.abs(trend)))
        story.append(callout(f"<b>La estrategia óptima encontrada por RL ({F.RL_NAME[best]}).</b> Puntuación fuera de muestra "
                             f"{num(np.mean([e['score'] for e in ev]), 2, True)}. Rasgos dominantes (media 2020-2028): " + "; ".join(desc) +
                             f". La política <b>cambia en el tiempo</b>: el mayor ajuste entre 2020 y 2028 es en "
                             f"{c.CH_ES[c.CHANNELS[j % 6]].lower()} ({zn[j // 6]}, {trend[j]:+.2f}). Reacciona al estado: la "
                             f"desviación típica de sus acciones entre los 32 mundos es {num(Sd.mean(), 3)} logit en promedio. Resultado: "
                             f"{pct(mvb('ppolx'))} de pico de contaminación, {num(mvb('p2070') * 100, 1, True)} pp de colapso antes de 2070, "
                             f"{num(mvb('dev_q1') * 100, 2, True)} puntos de desarrollo en el quintil más pobre.", AQUA))
        sc = {a: np.mean([e["score"] for e in RL[a]["eval"]]) for a in RL}
        tr_end = {a: np.mean([p["train_score"] for p in RL[a]["curve"][-5:]]) for a in RL}
        eps = {a: RL[a]["curve"][-1]["episodes"] for a in RL}
        fixed_top = max(v["score"] for k, v in fx.items()) if fx else 0.0

        def first_cross(a):
            for p in RL[a]["curve"]:
                if p.get("greedy_score", -9) > fixed_top:
                    return p["episodes"]
            return None
        lines = []
        if "PPO" in RL:
            lines.append(f"<b>On-policy (PPO)</b> sólo aprende de su propia experiencia: necesitó {eps['PPO']:,} episodios ".replace(",", ".") +
                         f"y fue el más lento al principio, pero el más estable y el mejor al final ({num(sc['PPO'], 2, True)} fuera de muestra).")
        offp = [a for a in ("SAC", "TD3") if a in RL]
        if offp:
            lines.append("<b>Off-policy</b> (" + ", ".join(f"{a} {num(sc[a], 2, True)}" for a in offp) + ") reutiliza cada episodio muchas "
                         "veces y aprovecha los datos aleatorios compartidos: llega antes a buenas políticas, pero se estanca algo por "
                         "debajo de PPO (posiblemente porque la recompensa terminal es ruidosa y eso dificulta aprender el crítico).")
        if "MBPO" in RL:
            fc = first_cross("MBPO")
            lines.append(f"<b>Basado en modelo, Dyna (MBPO)</b> es el más eficiente en datos: " +
                         (f"supera a la mejor estrategia fija con sólo {fc:,} episodios reales. ".replace(",", ".") if fc else "") +
                         f"Pero explota los errores de su modelo: puntuación de entrenamiento {num(tr_end['MBPO'], 2, True)} frente a "
                         f"{num(sc['MBPO'], 2, True)} fuera de muestra.")
        if "PETS" in RL:
            lines.append(f"<b>Basado en modelo, MPC (PETS)</b> planifica en el modelo en cada decisión. La dinámica por zona se aprende "
                         f"bien, pero la recompensa terminal (que agrega 180 países y World3) se predice mal (R² ≈ 0,5): sin política "
                         f"que corrija los errores, queda en {num(sc['PETS'], 2, True)}.")
        dirs = []
        for a in RL:
            Aa = np.mean([np.array(e["actions_mean"]) for e in RL[a]["eval"]], 0).mean(0)
            dirs.append(Aa)
        dirs = np.array(dirs)
        agree = [(z, j) for z in range(3) for j in range(6)
                 if abs(dirs[:, z * 6 + j].mean()) > 0.1 and np.all(np.sign(dirs[:, z * 6 + j]) == np.sign(dirs[:, z * 6 + j].mean()))]
        if agree:
            lines.append("<b>En qué coinciden todos los agentes</b> (mismo signo en los cinco y media > 0,1 logit): " +
                         "; ".join(f"{c.CH_ES[c.CHANNELS[j]].lower()} en {zn[z]} {dirs[:, z * 6 + j].mean():+.2f}" for z, j in agree) +
                         ". Es la misma dirección que el CEM (redistribuir más, acumular menos capital en el centro); el RL añade el "
                         "calendario y la reacción al estado.")
        story += [P("<b>Qué enseña la comparación de familias.</b>", "h2")] + bullets(lines)
        story.append(callout("<b>Advertencia.</b> Los agentes optimizan el modelo, incluido el acoplamiento L1 → World3 declarado en la "
                             "sección 4.3: buena parte de su ventaja viene de los canales que ese acoplamiento hace más influyentes "
                             "(gasto social → servicios de World3; acumulación de capital → contaminación). La evaluación con semillas "
                             "nuevas y World3 de verdad descarta el sobreajuste al azar y al emulador, no al supuesto de acoplamiento.", ORANGE))

    # ------------------------------------------------ 6. conclusiones
    story += section("07", "Conclusiones y recomendaciones", "síntesis")
    rk = sorted(((v["score"], k) for k, v in fx.items() if k != "base"), reverse=True) if fx else []
    story += bullets([
        "<b>La política real es inercial y de poder relativo.</b> Cualquier estrategia que dependa de que los gobiernos maximicen el "
        "bienestar va contra su recompensa revelada; las que mejor encajan son las que también aumentan su peso relativo (tecnología, I+D).",
        "<b>Para el desarrollo nacional,</b> invertir en capital y en tecnología/educación domina en casi todos los países a 10 años, pero "
        "es la estrategia que más acerca el colapso biofísico en World3.",
        "<b>Para el planeta,</b> la combinación que mejor equilibra los tres pilares no es la de un solo instrumento: gasto social (servicios "
        "de salud y educación), transición demográfica y eficiencia/no fósiles financiados sin recortar la I+D." +
        (f" En la clasificación compuesta las mejores estrategias fijas son: {', '.join(S[k]['label'] for _, k in rk[:3])}." if rk else ""),
        "<b>La tecnología verde sola no evita el colapso de World3</b> porque compite por capital; sí reduce el daño ambiental.",
        "<b>Dentro de la región de confianza de la conducta observada (±0,5 logit) ninguna estrategia evita el colapso</b> en la mayoría de "
        "los 400 mundos. Evitarlo exigiría desvíos mucho mayores de lo que los países han hecho nunca: fuera de lo que el modelo puede "
        "validar."])

    # ------------------------------------------------ 7. limitaciones
    story += section("08", "Limitaciones y honestidad", "lo que este estudio no dice")
    story += bullets([
        "<b>Datos no disponibles en este entorno:</b> el calibrado de FaIR (Zenodo) y FAOSTAT estaban bloqueados por la red. El clima del "
        "L1 es la TCRE del Atlas (no FaIR, F4 apagado), no hay sector de alimentos (F5) y el L0 regional de 10 regiones (F6), que necesita "
        "FAOSTAT, no se pudo usar: el L0 es World3-03 agregado.",
        "<b>El acoplamiento L1→L0 es un supuesto del estudio [M]</b> (sección 4.3): seis señales medidas en 2019-2030 que se mantienen hasta "
        "2100. La atribución y la prueba ×0,5/×2 muestran qué canal pesa en cada resultado.",
        "<b>World3-03 no está calibrado a datos posteriores a 2000</b>; el ensemble sólo se filtra por la población de 2020. Sus "
        "probabilidades de colapso describen el modelo, no el mundo.",
        "<b>El IRL identifica una recompensa lineal en seis rasgos</b>; el índice de racionalidad (≈ 58-62 %) indica que la conducta está "
        "lejos de ser óptima para cualquier recompensa de esa familia. Las recompensas por país tienen intervalos amplios (Figura 5).",
        "<b>Bandas del L1 demasiado estrechas</b> (defecto D7 del Atlas: cubren 60-70 % en vez de 80 %). Las diferencias pareadas con "
        "números aleatorios comunes son más fiables que los niveles.",
        "<b>Aprendizaje por refuerzo:</b> la recompensa de World3 durante el entrenamiento sale de un emulador (R² 0,99-1,00) y las "
        "puntuaciones de entrenamiento se calculan por miembro, mientras que la evaluación final (la que se compara) usa la mediana de "
        "32 miembros y World3 de verdad; por eso las dos escalas no coinciden exactamente. Un entrenamiento por algoritmo y semilla: "
        "las diferencias pequeñas entre agentes no son concluyentes. TD3 y SAC se relanzaron tras un fallo de memoria.",
        "<b>El aprendizaje multiagente</b> es una mejor respuesta aprendida con todos los países foco moviéndose a la vez, no un equilibrio "
        "demostrado; 12 miembros por evaluación.",
        "Las fuentes de los datos del Atlas se descargaron del commit fijado y se verificaron por SHA-256 (con fin de línea normalizado)."])
    story += [P("<b>Reproducir</b>", "h2"),
              P("<font name='DV' size='7.6'>python world4_cli.py --install-atlas --install-wpp --install-energy<br/>"
                "cd studies/irl_estrategias &amp;&amp; python s1_irl.py &amp;&amp; python s2_strategies.py &amp;&amp; "
                "python s2b_sensitivity.py &amp;&amp; python s3_learning.py &amp;&amp; python w3_surrogate.py &amp;&amp; python rl_train.py init "
                "&amp;&amp; python rl_train.py PPO (SAC, TD3, PETS, MBPO) &amp;&amp; python s4_report.py</font>", "small"),
              P(f"Tiempo de cómputo: IRL {irl['setup']['segundos'] // 60} min; estrategias {d['setup']['segundos'] // 60} min" +
                (f"; aprendizaje {L['setup']['segundos'] // 60} min." if L else "."), "small")]
    return story


def main():
    PDF.parent.mkdir(exist_ok=True)
    doc = BaseDocTemplate(str(PDF), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=17 * mm, bottomMargin=16 * mm,
                          title="Atlas Mundo × World3: política revelada y estrategias", author="Quasar Phy-World (World4)",
                          subject="IRL, estrategias L1 y colapso en World3")
    frame = Frame(M, 16 * mm, W - 2 * M, H - 33 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=cover),
                          PageTemplate(id="later", frames=[frame], onPage=later)])
    doc.build(build())
    print(PDF)


if __name__ == "__main__":
    main()
