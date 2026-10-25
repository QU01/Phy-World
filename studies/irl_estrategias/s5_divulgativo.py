"""Paso 5: el informe divulgativo. Mismos resultados que el informe técnico, en lenguaje llano, con las
implicaciones y las decisiones que exige usar un sistema así. Reutiliza la maqueta de ``s4_report``."""
import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, KeepTogether, NextPageTemplate, PageBreak,  # noqa: E402
                                PageTemplate, Spacer, Table, TableStyle)

import common as c  # noqa: E402
import s4_figures as F  # noqa: E402
import s4_report as R  # noqa: E402
from s4_report import M, NAVY, W, H, P, bullets, callout, fig, kpis, section, table  # noqa: E402

PDF = R.ROOT / "docs" / "Informe_divulgativo_estrategias_mundo.pdf"
R.ST["lead"] = ParagraphStyle("lead", fontName="DV", fontSize=10.6, leading=15.4, textColor=R.INK, spaceAfter=7)
R.ST["q"] = ParagraphStyle("q", fontName="DVSB", fontSize=12, leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=3)


def num(x, d=2):
    return f"{x:+.{d}f}".replace(".", ",").replace("-", "−")


def pc(x):
    return f"{x * 100:.0f} %"


# ---------------------------------------------------------------- figuras sencillas
def fig_motives(irl):
    g = irl["global"]["theta_std"]
    k = irl["contemporaneo"]["theta_std"]
    names = {"inercia": "No cambiar el presupuesto", "poder": "Ganar peso en el mundo", "autonomia": "Depender menos de recursos ajenos",
             "paz": "Evitar conflictos internos", "elite": "Enriquecer a la élite", "bienestar": "Que la gente consuma más"}
    order = sorted(names, key=lambda f: g[f])
    fig_, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(order))
    ax.barh(y + 0.19, [g[f] for f in order], 0.36, color=[F.BLUE if g[f] >= 0 else F.ORANGE for f in order],
            label="1960-2005 (toda la historia)")
    ax.barh(y - 0.19, [k[f] for f in order], 0.36, color="none", edgecolor=F.INK2, hatch="////", lw=0.8, label="2000-2005 (lo más reciente)")
    ax.set_yticks(y)
    ax.set_yticklabels([names[f] for f in order], fontsize=8.5)
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["lo sacrifican", "0", "", "lo persiguen", ""], fontsize=7.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    return F.save(fig_, "div_motives")


def fig_collapse(d, ppo):
    b = d["strategies"]["base"]["w3"]
    p = ppo["eval"][0]["w3"]
    fig_, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, var, title, sc in ((axes[0], "pop", "Población mundial (miles de millones)", 1e-9),
                               (axes[1], "ppolx", "Contaminación persistente (1970 = 1)", 1)):
        for w3, col, lab in ((b, F.INK2, "Si los países siguen como hasta ahora"), (p, F.BLUE, "Con la estrategia aprendida por la IA")):
            yrs = np.array(w3["years"])
            B = np.array(w3["bands"][var]) * sc
            m = yrs >= 1990
            ax.fill_between(yrs[m], B[m, 0], B[m, 2], color=col, alpha=0.12, lw=0)
            ax.plot(yrs[m], B[m, 1], color=col, lw=2, label=lab)
        ax.axvline(2020, color=F.MUTED, lw=0.6, ls=":")
        ax.set_title(title, fontsize=8.5)
    axes[0].legend(fontsize=6.8, loc="lower left")
    fig_.text(0.5, -0.04, "Línea: el caso típico; sombra: 8 de cada 10 mundos posibles (World3, 400 mundos)", ha="center",
              fontsize=7, color=F.INK2)
    return F.save(fig_, "div_collapse")


def fig_calendar(ppo):
    A = np.mean([np.array(e["actions_mean"]) for e in ppo["eval"]], 0)
    rows = [("Centro: invertir en capital", A[:, 12]), ("Centro: gasto social", A[:, 17]),
            ("Periferia: invertir en capital", A[:, 0]), ("Periferia: extraer recursos", A[:, 3]),
            ("Semiperiferia: gasto social", A[:, 11]), ("Semiperiferia: invertir fuera", A[:, 10])]
    years = [2020, 2022, 2024, 2026, 2028]
    fig_, ax = plt.subplots(figsize=(6.8, 2.8))
    cols = [F.ORANGE, F.BLUE, F.AQUA, F.YELLOW, F.VIOLET, F.MAGENTA]
    for (lab, v), col in zip(rows, cols):
        ax.plot(years, v, color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xlim(2019.5, 2028.5)
    ax.set_xticks(years)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels(["mucho menos", "menos", "como ahora", "más", "mucho más"], fontsize=7.5)
    ax.spines["right"].set_visible(False)
    return F.save(fig_, "div_calendar")


# ---------------------------------------------------------------- semáforo
def semaforo(d, L, RL):
    S = d["strategies"]
    b = S["base"]

    def cell(v, good_up, small):
        if abs(v) < small:
            return "= igual", colors.HexColor("#eeeeec")
        better = (v > 0) == good_up
        if better:
            return ("▲ mejora" if good_up else "▼ mejora"), colors.HexColor("#d6ead9")
        return ("▼ empeora" if good_up else "▲ empeora"), colors.HexColor("#f6d9cc")

    rows = []
    for k in F.strat_order(d):
        w, w3 = S[k]["world"], S[k]["w3"]
        rows.append((S[k]["label"], (w["dev_index"] - b["world"]["dev_index"]) * 100,
                     w3["ppolx_peak"] / b["w3"]["ppolx_peak"] - 1, w3["p_collapse_2070"] - b["w3"]["p_collapse_2070"]))
    if L:
        mv = L["world"]["aprendida_mejor"]["validacion"]
        rows.append(("Búsqueda automática (CEM)", np.mean([r["metrics"]["dev"] for r in mv]) * 100,
                     np.mean([r["metrics"]["ppolx"] for r in mv]), np.mean([r["metrics"]["p2070"] for r in mv])))
    if "PPO" in RL:
        ev = RL["PPO"]["eval"]
        rows.append(("IA que aprende (PPO)", np.mean([r["metrics"]["dev"] for r in ev]) * 100,
                     np.mean([r["metrics"]["ppolx"] for r in ev]), np.mean([r["metrics"]["p2070"] for r in ev])))
    data = [["Estrategia", "Desarrollo de los países\n(a 2030)", "Daño ambiental\n(contaminación)", "Riesgo de colapso\n(antes de 2070)"]]
    sty = []
    for i, (lab, dv, pp, cl) in enumerate(rows, start=1):
        cs = [cell(dv, True, 0.02), cell(pp, False, 0.03), cell(cl, False, 0.015)]
        data.append([lab] + [f"{t}  ({v})" for (t, _), v in zip(cs, (f"{dv:+.2f} pts".replace(".", ","), f"{pp * 100:+.0f} %",
                                                                         f"{cl * 100:+.0f} pts"))])
        for j, (_, col) in enumerate(cs, start=1):
            sty.append(("BACKGROUND", (j, i), (j, i), col))
        if "IA" in lab or "CEM" in lab:
            sty.append(("FONTNAME", (0, i), (0, i), "DVB"))
    t = Table([[R.P(str(x).replace("\n", "<br/>"), "cellb" if i == 0 else "cell") for x in r] for i, r in enumerate(data)],
              colWidths=[50 * mm] + [(W - 2 * M - 50 * mm) / 3] * 3, repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf5")),
                           ("GRID", (0, 0), (-1, -1), 0.6, colors.white)] + sty))
    return t


# ---------------------------------------------------------------- portada
def cover(canv, doc):
    canv.saveState()
    canv.setFillColor(NAVY)
    canv.rect(0, 0, W, H, fill=1, stroke=0)
    canv.setStrokeColor(colors.HexColor("#22325a"))
    cx, cy, r = W * 0.8, H * 0.72, 90 * mm
    for k in range(-5, 6):
        canv.ellipse(cx - r * abs(k) / 5, cy - r, cx + r * abs(k) / 5, cy + r)
    for k in range(-4, 5):
        y = cy + r * k / 5
        dx = r * np.sqrt(max(0, 1 - (k / 5) ** 2))
        canv.line(cx - dx, y, cx + dx, y)
    canv.setFillColor(colors.HexColor(F.AQUA))
    canv.rect(M, H * 0.47, 28 * mm, 2.2 * mm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("DVB", 9)
    canv.drawString(M, H * 0.47 + 7 * mm, "INFORME PARA NO ESPECIALISTAS")
    canv.setFont("DVSB", 30)
    y = H * 0.40
    for line in ("Lo que hacen los países,", "lo que les conviene", "y lo que aguanta el planeta"):
        canv.drawString(M, y, line)
        y -= 12.5 * mm
    canv.setFont("DV", 11.5)
    canv.setFillColor(colors.HexColor("#c9d3ea"))
    y -= 3 * mm
    for line in ("Qué aprendimos simulando 180 países y el planeta con inteligencia artificial,",
                 "y qué decisiones exige usar una herramienta así."):
        canv.drawString(M, y, line)
        y -= 6 * mm
    canv.setFont("DV", 8.5)
    canv.setFillColor(colors.HexColor("#9fb0d6"))
    canv.drawString(M, 20 * mm, f"{dt.date.today():%d/%m/%Y} · versión divulgativa del informe técnico "
                                "«Informe_IRL_estrategias_AtlasMundo_World3.pdf»")
    canv.restoreState()


def later(canv, doc):
    canv.saveState()
    canv.setStrokeColor(R.LINE)
    canv.setLineWidth(0.5)
    canv.line(M, H - 12 * mm, W - M, H - 12 * mm)
    canv.setFont("DV", 7)
    canv.setFillColor(R.INK2)
    canv.drawString(M, H - 10 * mm, "Lo que hacen los países, lo que les conviene y lo que aguanta el planeta · versión divulgativa")
    canv.drawRightString(W - M, 10 * mm, f"{doc.page}")
    canv.setFillColor(colors.HexColor(F.AQUA))
    canv.rect(M, 9.4 * mm, 6 * mm, 1.2 * mm, fill=1, stroke=0)
    canv.restoreState()


# ---------------------------------------------------------------- contenido
def build():
    irl = c.load("irl.json")
    d = c.load("strategies.json")
    L = c.load("learning.json")
    RL = F.rl_results()
    S = d["strategies"]
    b = S["base"]
    g = irl["global"]
    ppo = RL["PPO"]
    ev = ppo["eval"]
    mp = lambda k: np.mean([e["metrics"][k] for e in ev])  # noqa: E731
    f_mot, f_col, f_cal = fig_motives(irl), fig_collapse(d, ppo), fig_calendar(ppo)
    story = [NextPageTemplate("later"), PageBreak()]

    # ------------------------------------------------ en una página
    story += section("0", "Lo esencial en una página", "resumen")
    story.append(kpis([(pc(g["racionalidad"]), "de las decisiones reales de los países se explican por lo que «persiguen» de verdad"),
                       (pc(b["w3"]["p_collapse_2070"]), "de los futuros simulados terminan en colapso antes de 2070 si nada cambia"),
                       (f"−{abs(mp('p2070')) * 100:.0f} pts",
                        "menos riesgo de colapso con la mejor estrategia que encontró la IA"),
                       (f"{abs(mp('ppolx')) * 100:.0f} %", "menos contaminación máxima con esa misma estrategia")]))
    story += bullets([
        "<b>Los gobiernos no actúan pensando sobre todo en el bienestar de su gente.</b> Mirando sus decisiones de 1960 a 2019, lo "
        "que mejor las explica es <b>no cambiar sus presupuestos</b> y <b>ganar peso frente a otros países</b>. El consumo de la "
        "población funciona como lo que se ajusta para lograr lo demás. En los años 2000 esto empieza a cambiar.",
        "<b>Lo que más hace crecer a cada país a corto plazo es también lo que más acerca al planeta a sus límites.</b> Invertir mucho "
        "en industria y tecnología es lo mejor para el desarrollo de casi todos los países (173 de 180), pero eleva el riesgo de colapso.",
        "<b>Si el mundo sigue como hasta ahora, el modelo del planeta (World3) prevé un colapso en 9 de cada 10 futuros posibles</b> "
        "antes de 2070: una caída fuerte de la población o de la producción industrial.",
        "<b>Ninguna estrategia «sola» lo evita.</b> La tecnología verde reduce la contaminación pero cuesta capital; el gasto social "
        "reduce el riesgo, pero no lo elimina.",
        f"<b>Una inteligencia artificial que aprende por ensayo y error encontró una estrategia mejor que todas las probadas a mano:</b> "
        f"que los países ricos redistribuyan más y acumulen menos capital industrial, y que los países pobres inviertan fuerte al "
        f"principio y extraigan menos recursos. Reduce el riesgo de colapso en {abs(mp('p2070')) * 100:.0f} puntos y la contaminación "
        f"máxima un {abs(mp('ppolx')) * 100:.0f} %, con un coste mínimo en desarrollo ({num(mp('dev') * 100)} puntos sobre 100).",
        "<b>La herramienta no decide por nadie.</b> Lo que llama «mejor» depende de cómo se ponderan desarrollo, ambiente y riesgo, "
        "y esa ponderación es una decisión política, no técnica (sección 6)."])

    # ------------------------------------------------ 1. qué es
    story += section("1", "¿Qué es este sistema?", "la herramienta")
    story.append(P("Piense en un <b>simulador de vuelo, pero del mundo</b>. No predice el futuro: permite probar «qué pasaría si…» "
                   "de forma ordenada y comparar decisiones en las mismas condiciones. Tiene dos pisos:", "lead"))
    story.append(table([["Piso", "Qué simula", "Con qué datos", "Hasta cuándo"],
                        ["Los países (Atlas del Sistema-Mundo)", "180 países que deciden cada año cuánto invierten en industria, en "
                         "tecnología y educación, cuánto importan, cuántos recursos extraen, cuánto invierten fuera y cuánto redistribuyen. "
                         "También su deuda, sus élites, su riesgo de conflicto y su democracia.", "Datos reales 1950-2019 (Penn World "
                         "Table, Banco Mundial, V-Dem, UCDP…). Probado contra la historia hasta 10 años vista.", "2030"],
                        ["El planeta (World3)", "El modelo de <i>Los límites del crecimiento</i>: población, industria, agricultura, "
                         "recursos no renovables y contaminación del mundo entero.", "Versión 2004 del modelo, sin recalibrar a datos "
                         "recientes. Se corre en 400 «mundos posibles» con supuestos distintos.", "2100"]],
                       [36 * mm, 70 * mm, 50 * mm, W - 2 * M - 156 * mm]))
    story.append(Spacer(1, 6))
    story.append(P("Lo que deciden los países en el primer piso (2019-2030) se traduce en señales para el segundo: cuánto se consume y "
                   "cuánto se invierte, cuánta energía fósil se usa, cuánto se gasta en servicios sociales, cuánta población hay. Así "
                   "se ve qué consecuencias de largo plazo tiene una decisión de corto plazo. Esa traducción entre pisos es un "
                   "<b>supuesto del estudio</b>, no un dato, y es una de las principales fuentes de incertidumbre."))
    story.append(callout("<b>Tres reglas para leer este informe.</b> (1) Ninguna cifra es una predicción: son comparaciones entre "
                         "escenarios dentro del modelo. (2) Las diferencias entre estrategias son más fiables que los niveles. (3) "
                         "«Colapso» significa aquí que, dentro del modelo, la población cae al menos un 20 % o la producción "
                         "industrial por persona cae a la mitad desde su máximo.", R.ORANGE))

    # ------------------------------------------------ 2. qué persiguen
    story += section("2", "¿Qué persiguen realmente los países?", "leer intenciones a partir de actos")
    story.append(P("Los gobiernos dicen muchas cosas. Para saber qué persiguen <b>de verdad</b> se usó una técnica llamada "
                   "<b>aprendizaje por refuerzo inverso</b>: en vez de preguntar qué objetivo tienen, se mira lo que hicieron y se busca "
                   "el objetivo que haría que esas decisiones fueran las más sensatas. Es como deducir que alguien quiere ahorrar porque "
                   "siempre elige la opción barata.", "lead"))
    story.append(P(f"Para ello se simularon <b>{irl['n_total']:,} variantes de la historia</b> en las que algunos países hacían algo "
                   "distinto de lo que hicieron (invertir un poco más, extraer un poco menos…). Si lo que hicieron fue mejor que la "
                   "variante según cierto objetivo, ese objetivo gana puntos.".replace(",", ".")))
    story.append(fig(f_mot, caption="Qué pesa en las decisiones reales de los países. Barras a la derecha: lo persiguen; a la "
                                    "izquierda: lo sacrifican."))
    story += bullets([
        "<b>Inercia:</b> lo que más explica las decisiones es no mover los presupuestos. Cambiar cuesta (política, burocracia, "
        "intereses), y los países lo evitan.",
        "<b>Poder relativo:</b> ganar peso en la economía mundial frente a los demás.",
        "<b>El consumo de la población pesa en negativo:</b> se sacrifica cuando hace falta para lo anterior.",
        f"El objetivo así encontrado explica el {pc(g['racionalidad'])} de las decisiones, frente al "
        f"{pc(g['racionalidad_alternativas']['bienestar'])} de un objetivo de «bienestar de la gente» y el 50 % del puro azar. No es "
        "una explicación perfecta: los países no son actores perfectamente racionales.",
        "<b>Hay diferencias:</b> los países ricos se explican sobre todo por evitar conflictos internos; los pobres y los regímenes "
        "autoritarios, por poder e inercia. En los años 2000 el bienestar de la población empieza a pesar en positivo."])
    story.append(callout("<b>Implicación.</b> Las propuestas que exigen a los gobiernos actuar «por el bien de la gente» chocan con lo "
                         "que de hecho los mueve. Las que tienen más opciones de aplicarse son las que, además, les dan peso o no les "
                         "obligan a cambios bruscos de presupuesto."))

    # ------------------------------------------------ 3. dilema
    story += section("3", "El dilema: crecer hoy frente a aguantar mañana", "estrategias probadas")
    story.append(P("Se probaron diez estrategias, más dos que salen del análisis anterior («hacer más de lo que ya hacen» y «hacer lo "
                   "que conviene a la gente»), en los 180 países a la vez. Cada una se juzgó en tres frentes: el desarrollo de los "
                   "países hasta 2030, el daño ambiental y el riesgo de colapso del planeta hasta 2100.", "lead"))
    story.append(semaforo(d, L, RL))
    story.append(P("Semáforo frente a «seguir como hasta ahora». Desarrollo: índice tipo IDH (ingreso, educación, paz). Daño ambiental: "
                   "pico de contaminación persistente en World3. Riesgo: puntos porcentuales de probabilidad de colapso antes de 2070.",
                   "cap"))
    story += bullets([
        "<b>Ninguna estrategia simple mejora los tres frentes a la vez.</b> El <b>Estado desarrollista</b> (invertir mucho en industria "
        "y tecnología) es la mejor para crecer y la peor para el planeta.",
        "El <b>Estado de bienestar</b> (más gasto social) no acelera el crecimiento a 2030, pero es de lo que más reduce el riesgo de "
        "colapso: más salud y educación reducen la natalidad y desvían recursos de la industria más contaminante.",
        "La <b>transición verde</b> sola corta la contaminación, pero se paga con capital; en el modelo del planeta eso adelanta la caída "
        "industrial. La tecnología ayuda, pero no basta.",
        f"La estrategia de la IA (última fila) es la que más reduce el riesgo ({num(mp('p2070') * 100, 0)} puntos) y de las que más "
        f"reducen la contaminación, con un coste mínimo en desarrollo. El Estado de bienestar la sigue de cerca "
        f"({num((S['bienestar_social']['w3']['p_collapse_2070'] - b['w3']['p_collapse_2070']) * 100, 0)} puntos) sin tocar el desarrollo. "
        "Ninguna mejora los tres frentes a la vez de forma clara."])
    story.append(callout("<b>Implicación.</b> Hay un conflicto real entre el interés de cada país a corto plazo y la estabilidad del "
                         "planeta a largo plazo. Cada país, por su cuenta, tiene incentivos para elegir la estrategia que más lo hace "
                         "crecer, y si todos lo hacen el riesgo común aumenta. Es un problema de coordinación internacional, no solo de "
                         "buena política nacional."))

    # ------------------------------------------------ 4. planeta
    story += section("4", "¿Qué le pasa al planeta?", "World3, de 2020 a 2100")
    story.append(fig(f_col, caption="Población y contaminación en el modelo del planeta: «seguir igual» frente a la estrategia "
                                     "aprendida por la IA."))
    story += bullets([
        f"Con las decisiones actuales, el modelo muestra el patrón clásico de <i>Los límites del crecimiento</i>: la población llega a "
        f"un máximo hacia 2040 y después cae. En {pc(b['w3']['p_collapse_2070'])} de los 400 mundos simulados hay colapso antes de 2070.",
        "Ninguna estrategia dentro de lo que los países han hecho históricamente evita el colapso en la mayoría de los mundos. Lo que "
        "sí cambia es <b>cuándo llega y cuánto se cae</b>.",
        "Esto es un resultado del modelo World3, que tiene 50 años, no se ha recalibrado con datos recientes y se considera pesimista "
        "por muchos economistas. Debe leerse como una señal de alerta sobre la dirección, no como un pronóstico."])

    # ------------------------------------------------ 5. IA
    story += section("5", "Lo que aprendió la inteligencia artificial", "aprendizaje por refuerzo")
    story.append(P("Probar estrategias a mano tiene un límite: hay demasiadas combinaciones. Por eso se entrenaron <b>cinco "
                   "programas que aprenden por ensayo y error</b>, como aprende un jugador: prueban, ven el resultado, ajustan. "
                   "Cada dos años (2020, 2022, …, 2028) miran cómo está el mundo y deciden cuánto deben cambiar los países sus "
                   "presupuestos. Jugaron miles de partidas simuladas.", "lead"))
    story.append(table([["Programa", "Cómo aprende, en una frase", "Resultado"]] +
                       [[F.RL_NAME[a].split(" (")[0], {
                           "PPO": "Sólo aprende de lo que acaba de hacer; lento pero firme.",
                           "SAC": "Recuerda todas sus partidas y las repasa muchas veces.",
                           "TD3": "Como el anterior, con otra forma de valorar las jugadas.",
                           "PETS": "Se construye un «mapa» del mundo y planifica sobre él en cada jugada.",
                           "MBPO": "Usa su mapa para imaginar partidas y entrenarse con ellas."}[a],
                         f"{np.mean([e['score'] for e in RL[a]['eval']]):+.2f}".replace(".", ",")] for a in RL] +
                       [["(referencia) mejor estrategia hecha a mano", "Estado de bienestar", "+0,67"]],
                       [32 * mm, W - 2 * M - 62 * mm, 30 * mm]))
    story.append(P("Puntuación combinada de desarrollo, ambiente y riesgo (más es mejor; 0 = seguir igual), medida en simulaciones "
                   "que los programas no habían visto.", "cap"))
    story.append(KeepTogether([P("<b>La receta del mejor programa (PPO):</b>", "q"),
                               fig(f_cal, caption="El calendario aprendido: cuánto más o menos que ahora deberían dedicar los "
                                                  "países a cada partida.")]))
    story += bullets([
        "<b>Países ricos (centro):</b> dedicar más a gasto social y redistribución, y acumular cada vez menos capital industrial.",
        "<b>Países pobres (periferia):</b> invertir fuerte en los primeros años, para recortar después, y extraer menos recursos "
        "naturales durante toda la década.",
        "<b>Países de renta media (semiperiferia):</b> más gasto social e invertir menos fuera conforme pasa el tiempo.",
        "<b>El orden importa:</b> la misma mezcla aplicada de forma fija desde el principio rinde menos. Lo que funciona es una "
        "secuencia que reacciona a cómo evoluciona el mundo.",
        f"<b>Resultado:</b> {abs(mp('p2070')) * 100:.0f} puntos menos de riesgo de colapso antes de 2070, "
        f"{abs(mp('psevere')) * 100:.0f} puntos menos de colapso grave y {abs(mp('ppolx')) * 100:.0f} % menos contaminación máxima, "
        f"a cambio de una pérdida de desarrollo mínima ({num(mp('dev') * 100)} puntos sobre 100 en 2030)."])
    story.append(callout("<b>Ojo: la IA optimiza el simulador, no el mundo.</b> Una parte de su ventaja viene precisamente de cómo "
                         "se tradujo en este estudio el gasto social de los países al modelo del planeta. Si esa traducción es "
                         "incorrecta, la receta también lo será. Por eso sus propuestas son hipótesis para discutir, no instrucciones.",
                         R.ORANGE))

    # ------------------------------------------------ 6. implicaciones y decisiones
    story += section("6", "Implicaciones y decisiones que exige este sistema", "lo importante")
    story.append(P("<b>A. Lo que implican los resultados</b>", "q"))
    story += bullets([
        "<b>Para cada gobierno:</b> la estrategia que maximiza el crecimiento propio es la que más riesgo crea para todos. Elegir un "
        "camino más equilibrado supone renunciar a algo de crecimiento relativo, justo lo que los países más valoran.",
        "<b>Para la cooperación internacional:</b> la receta aprendida pide más a los países ricos (frenar su acumulación de capital) y "
        "más inversión temprana a los pobres. Eso solo es viable con <b>transferencias y acuerdos</b>: financiación, tecnología, "
        "comercio. Sin coordinación, cada país tiene incentivos para desviarse.",
        "<b>Para la política social:</b> el gasto en salud, educación y protección social aparece no solo como una cuestión de "
        "justicia, sino como una herramienta de estabilidad a largo plazo, por su efecto en la natalidad y en el tipo de crecimiento.",
        "<b>Para la política ambiental:</b> la tecnología verde reduce daños, pero si se financia quitando inversión productiva puede "
        "debilitar la economía. Hay que decidir de dónde sale el dinero.",
        "<b>Para la diplomacia:</b> como los gobiernos se mueven por inercia y poder, las propuestas tienen más opciones si son "
        "graduales y si no restan peso relativo a quien las adopta."])
    story.append(CondPageBreak(70 * mm))
    story.append(P("<b>B. Las decisiones que exige usar una herramienta así</b>", "q"))
    dec = [["Decisión", "Por qué importa", "Quién debería tomarla"],
           ["¿Qué cuenta como «mejor»?", "La IA maximiza una puntuación que pondera por igual desarrollo, ambiente y riesgo. Con otros "
            "pesos (más peso al crecimiento, o al riesgo) la receta cambia.", "Responsables políticos y ciudadanía, no los técnicos."],
           ["¿Cuánto riesgo es aceptable?", f"Pasar de {b['w3']['p_collapse_2070'] * 10:.1f} a "
            f"{(b['w3']['p_collapse_2070'] + mp('p2070')) * 10:.1f} de cada 10 futuros con colapso es una mejora, pero sigue siendo un "
            "riesgo altísimo. ¿Basta con reducirlo o hay que eliminarlo?".replace(".", ",", 2), "Decisión política y ética."],
           ["¿Quién paga el ajuste?", "La receta pide más esfuerzo a los países ricos al principio y cambios en los pobres después.",
            "Negociación internacional."],
           ["¿Cuánto confiar en el modelo?", "El modelo del planeta es antiguo y pesimista; la conexión entre los dos pisos es un "
            "supuesto. Decisiones grandes exigen contrastar con otros modelos.", "Comités científicos independientes."],
           ["¿Qué papel tiene la IA?", "La IA explora miles de opciones que nadie podría probar a mano, pero no entiende el mundo: "
            "aprovecha lo que el modelo permite. Debe proponer, no decidir.", "Supervisión humana con reglas claras."],
           ["¿Hasta dónde llegar?", "Todo se probó dentro de lo que los países han hecho históricamente. Cambios mayores podrían "
            "evitar el colapso en el modelo, pero quedan fuera de lo que puede validarse.", "Decisión política informada por expertos."],
           ["¿Transparencia y datos?", "Faltaron datos de clima y alimentos por limitaciones técnicas; cualquier uso real exige datos "
            "completos, código abierto y revisión externa.", "Instituciones que encarguen y auditen los estudios."]]
    story.append(table(dec, [36 * mm, W - 2 * M - 86 * mm, 50 * mm]))
    story.append(Spacer(1, 6))
    story.append(callout("<b>En una frase:</b> el sistema sirve para ordenar la discusión. Muestra qué se gana y qué se pierde con cada "
                         "camino y qué decisiones no pueden delegarse en un algoritmo: cuánto valoramos el crecimiento frente a la "
                         "seguridad del planeta, cuánto riesgo aceptamos y quién paga el ajuste.", R.AQUA))

    # ------------------------------------------------ 7. límites
    story += section("7", "Lo que este estudio no dice", "límites")
    story += bullets([
        "<b>No es un pronóstico.</b> Ningún número dice qué pasará; dice qué pasa en el modelo bajo ciertos supuestos.",
        "<b>El modelo del planeta (World3) es de 1972/2004</b> y no se ha ajustado a los datos de los últimos 20 años.",
        "<b>Faltan piezas:</b> no se pudieron descargar los datos del modelo climático detallado ni los de agricultura, así que el clima "
        "se aproxima de forma simple y no hay sector de alimentos.",
        "<b>La conexión entre los dos pisos es un supuesto</b> del estudio. Se comprobó que el orden de las estrategias se mantiene si "
        "esa conexión es la mitad o el doble de fuerte, pero no se puede descartar que sea distinta.",
        "<b>Los objetivos de los países</b> explican bien sus decisiones en promedio, pero los de cada país concreto son inciertos.",
        "<b>La IA se entrenó una vez por método:</b> las diferencias pequeñas entre programas no son concluyentes."])
    story.append(P("<b>Pequeño glosario</b>", "q"))
    story.append(table([["Término", "Qué significa aquí"],
                        ["Centro / semiperiferia / periferia", "Países de renta alta (el 20 % más rico), media y baja en 2019."],
                        ["Colapso", "Caída de la población de al menos el 20 % o de la producción industrial por persona de al menos "
                         "el 50 % desde su máximo, dentro del modelo."],
                        ["Mundos posibles", "Versiones del modelo del planeta con supuestos distintos sobre recursos, contaminación y "
                         "tecnología; sirven para medir la incertidumbre."],
                        ["Aprendizaje por refuerzo", "Un programa que aprende a decidir probando y viendo el resultado."],
                        ["Aprendizaje por refuerzo inverso", "Deducir qué objetivo persigue alguien a partir de sus decisiones."]],
                       [50 * mm, W - 2 * M - 50 * mm]))
    story.append(Spacer(1, 6))
    story.append(P("El detalle técnico, los métodos y todas las cifras están en el informe técnico "
                   "(docs/Informe_IRL_estrategias_AtlasMundo_World3.pdf) y el código en studies/irl_estrategias.", "small"))
    return story


def main():
    doc = BaseDocTemplate(str(PDF), pagesize=R.A4, leftMargin=M, rightMargin=M, topMargin=17 * mm, bottomMargin=16 * mm,
                          title="Lo que hacen los países, lo que les conviene y lo que aguanta el planeta",
                          author="Quasar Phy-World (World4)", subject="Informe divulgativo")
    frame = Frame(M, 16 * mm, W - 2 * M, H - 33 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=cover),
                          PageTemplate(id="later", frames=[frame], onPage=later)])
    doc.build(build())
    print(PDF)


if __name__ == "__main__":
    main()
