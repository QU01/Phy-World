"""Paso 4a: figuras del informe (PNG, 200 ppp) a partir de ``out/*.json``."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import common as c  # noqa: E402

FIG = c.OUT / "fig"
FIG.mkdir(exist_ok=True)
# paleta de referencia (dataviz, modo claro): categóricos en orden fijo, divergente azul-gris-naranja
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                         "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
NAVY = "#14213d"
DIV = LinearSegmentedColormap.from_list("div", ["#c4461a", "#f3a47f", "#eeeeec", "#7fb0e8", "#1d5fae"])
DIV_R = DIV.reversed()
SHORT = {"desarrollista": "Desarro-\nllista", "conocimiento": "Conoci-\nmiento", "bienestar_social": "Bienestar\nsocial",
         "extractivista": "Extracti-\nvismo", "austeridad": "Auste-\nridad", "apertura": "Apertura", "verde": "Verde*",
         "demografica": "Demográ-\nfica*", "verde_desarrollista": "Desarrollo\nverde*", "irl_revelada": "IRL\nrevelada",
         "irl_bienestar": "IRL\nbienestar"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc3ef", "#2a78d6", "#123f78"])
FEAT_ES = {"bienestar": "Bienestar\n(consumo)", "poder": "Poder\n(cuota PIB)", "elite": "Élite\n(renta)",
           "paz": "Paz", "autonomia": "Autonomía\n(recursos)", "inercia": "Inercia\n(ajuste)"}
FEAT_1 = {k: v.replace("\n", " ") for k, v in FEAT_ES.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "axes.titleweight": "bold",
                     "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False})


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def label_of(d, k):
    return d["strategies"][k]["label"]


# ---------------------------------------------------------------- metodología
def pipeline():
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    boxes = [(1, "Datos 1950-2019\n180 países", "Atlas: panel PWT,\nV-Dem, UCDP, WDI", "#eef4fc"),
             (21, "Regla clonada", "lo que los países\nhicieron (ajuste\nparcial por canal)", "#eef4fc"),
             (41, "IRL (BBL)", "48 500 desvíos:\nqué recompensa hace\nóptima la conducta", "#fdeee6"),
             (61, "Estrategias L1\n2019 → 2030", "12 estrategias +\naprendizaje CEM", "#e8f6f0"),
             (81, "L0 World3-03\n2020 → 2100", "400 mundos: colapso,\ncontaminación, HWI", "#fdf3dc")]
    for x, t, s, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 9), 17, 22, boxstyle="round,pad=0.4,rounding_size=2", fc=col, ec=MUTED, lw=0.8))
        ax.text(x + 8.5, 25, t, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        ax.text(x + 8.5, 15, s, ha="center", va="center", fontsize=6.6, color=INK2)
    for x in (18.6, 38.6, 58.6, 78.6):
        ax.annotate("", xy=(x + 2.2, 20), xytext=(x, 20), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.annotate("", xy=(69.5, 8.2), xytext=(49.5, 8.2), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1,
                                                                        connectionstyle="arc3,rad=0.35"))
    ax.text(59.5, 1.2, "θ revelado por país: valor de cada estrategia", ha="center", fontsize=7, color=ORANGE)
    return save(fig, "pipeline")


# ---------------------------------------------------------------- IRL
def irl_weights(irl):
    g = irl["global"]
    th = np.array([g["theta_std"][f] for f in c.FEATS])
    ci = np.array([g["theta_std_ic90"][f] for f in c.FEATS])
    cont = np.array([irl["contemporaneo"]["theta_std"][f] for f in c.FEATS])
    order = np.argsort(th)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(th))
    ax.barh(y + 0.18, th[order], height=0.34, color=[BLUE if v >= 0 else ORANGE for v in th[order]],
            label="1960-2005 (todos los orígenes)")
    ax.errorbar(th[order], y + 0.18, xerr=[th[order] - ci[order, 0], ci[order, 1] - th[order]], fmt="none",
                ecolor=INK2, elinewidth=0.9, capsize=2.5)
    ax.barh(y - 0.18, cont[order], height=0.34, color="none", edgecolor=INK2, hatch="////", lw=0.8,
            label="2000-2005 (contemporáneo)")
    ax.set_yticks(y)
    ax.set_yticklabels([FEAT_1[c.FEATS[i]] for i in order])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("peso estandarizado en la recompensa revelada (θ, norma 1); barras: IC 90 % bootstrap por país")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    return save(fig, "irl_weights")


def irl_rationality(irl):
    g = irl["global"]
    alt = g["racionalidad_alternativas"]
    items = [("Recompensa revelada (IRL)", g["racionalidad"], BLUE),
             ("Sólo inercia (parsimoniosa)", g["parsimonioso"]["racionalidad"], AQUA),
             ("Bienestar (consumo + paz)", alt["bienestar"], MUTED), ("Élite", alt["elite"], MUTED),
             ("Poder", alt["poder"], MUTED), ("Recompensa al azar (media)", alt["aleatoria_media"], "#c9c8c2")]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, col) in zip(y, items):
        ax.barh(yi, (v - 0.5) * 100, color=col, height=0.6)
        ax.text((v - 0.5) * 100 + 0.15, yi, f"{v:.1%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("índice de racionalidad: % de desigualdades de optimalidad satisfechas, menos 50 % (azar)")
    ax.set_xlim(-0.6, 11)
    ax.grid(axis="y", visible=False)
    return save(fig, "irl_rationality")


def heat(ax, M, rows, cols, fmt="{:+.2f}", vmax=None, cmap=DIV, fs=7):
    vmax = vmax or np.nanmax(np.abs(M))
    ax.imshow(M, cmap=cmap, vmin=-vmax if cmap in (DIV, DIV_R) else 0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="white" if abs(v) > 0.62 * vmax else INK)


def irl_groups(irl):
    rows, M, rat = [], [], []
    names = {"periferia": "Periferia", "semiperiferia": "Semiperiferia", "centro": "Centro", "democracias": "Democracias",
             "autocracias": "Autocracias"}
    for k in ("por_zona", "por_regimen"):
        for g, v in irl[k].items():
            rows.append(names.get(g, g))
            M.append([v["theta_std"][f] for f in c.FEATS])
            rat.append(v["racionalidad"])
    for g, v in irl["por_decada"].items():
        rows.append(f"Origen {g}")
        M.append([v["theta_std"][f] for f in c.FEATS])
        rat.append(v["racionalidad"])
    M = np.array(M)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [6, 1]})
    heat(ax, M, rows, [FEAT_ES[f] for f in c.FEATS], vmax=1)
    ax.xaxis.tick_top()
    heat(ax2, np.array(rat)[:, None] - 0.5, [""] * len(rows), ["Racio-\nnalidad"], fmt="", vmax=0.2, cmap=SEQ)
    for i, r in enumerate(rat):
        ax2.text(0, i, f"{r:.0%}", ha="center", va="center", fontsize=7.5, color=INK)
    ax2.xaxis.tick_top()
    ax2.set_yticks([])
    for y in (2.5, 4.5):
        ax.axhline(y, color="white", lw=2.5)
        ax2.axhline(y, color="white", lw=2.5)
    return save(fig, "irl_groups")


def irl_countries(irl):
    P = irl["paises"]
    isos = list(P)
    M = np.array([[P[i]["theta_std"][f] for f in c.FEATS] for i in isos])
    rat = np.array([P[i]["racionalidad"] for i in isos])
    ratw = np.array([P[i]["racionalidad_bienestar"] for i in isos])
    o = np.argsort(-rat)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 6.4), gridspec_kw={"width_ratios": [6, 1.6]})
    heat(ax, M[o], [isos[i] for i in o], [FEAT_ES[f] for f in c.FEATS], vmax=1)
    ax.xaxis.tick_top()
    yy = np.arange(len(o))
    ax2.barh(yy, (rat[o] - 0.5) * 100, color=BLUE, height=0.38, label="revelada")
    ax2.barh(yy + 0.4, (ratw[o] - 0.5) * 100, color=MUTED, height=0.38, label="bienestar")
    ax2.set_ylim(len(o) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.set_xlabel("racionalidad − 50 %\n(puntos)", fontsize=7)
    ax2.legend(fontsize=6.5, loc="lower right")
    ax2.grid(axis="y", visible=False)
    return save(fig, "irl_countries")


def irl_deviations(irl):
    g = irl["global"]
    keys = [f"{ch}{s}" for ch in c.CHANNELS for s in ("+", "-")]
    rv = np.array([g["desvios_rentables_revelada"][k] for k in keys])
    wv = np.array([g["desvios_rentables_bienestar"][k] for k in keys])
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, rv * 100, 0.38, color=BLUE, label="bajo la recompensa REVELADA")
    ax.bar(x + 0.2, wv * 100, 0.38, color=ORANGE, label="bajo la recompensa de BIENESTAR")
    ax.axhline(50, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["+" if k[1] == "+" else "−" for k in keys], fontsize=8)
    short = {"k": "Capital", "r": "I+D", "m": "Importar", "x": "Extraer", "f": "Inv. exterior", "w": "Redistribuir"}
    for j, ch in enumerate(c.CHANNELS):
        ax.text(2 * j + 0.5, -17, short[ch], ha="center", va="top", fontsize=7.2, color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de casos en que el desvío\nhabría sido rentable")
    ax.legend(fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    return save(fig, "irl_deviations")


# ---------------------------------------------------------------- estrategias
ORDER = None


def strat_order(d):
    return [k for k in d["strategies"] if k != "base"]


def l1_scatter(d):
    """Dos paneles con el mismo eje de estrategias: desarrollo y presión fósil (sin doble eje)."""
    S = d["strategies"]
    b = S["base"]["world"]
    ks = strat_order(d)
    dy = np.array([(S[k]["world"]["dev_index"] - b["dev_index"]) * 100 for k in ks])
    gd = np.array([(S[k]["world"]["gdp_pc"] / b["gdp_pc"] - 1) * 100 for k in ks])
    dx = np.array([(S[k]["world"]["fossil_cum_twh"] / b["fossil_cum_twh"] - 1) * 100 for k in ks])
    o = np.argsort(dy)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True)
    y = np.arange(len(ks))
    for ax, v, title, good in ((axes[0], dy, "Δ índice de desarrollo 2030\n(puntos sobre 100)", 1),
                               (axes[1], gd, "Δ PIB per cápita mundial 2030\n(%)", 1),
                               (axes[2], dx, "Δ energía fósil acumulada\n2020-2030 (%)", -1)):
        vv = v[o]
        ax.barh(y, vv, color=[BLUE if x * good >= 0 else ORANGE for x in vv], height=0.62)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8.2)
        ax.grid(axis="y", visible=False)
        lim = np.abs(vv).max() * 1.45
        ax.set_xlim(-lim, lim)
        for yi, x in zip(y, vv):
            ax.text(x + (lim * 0.03 if x >= 0 else -lim * 0.03), yi, f"{x:+.2f}" if abs(x) < 1 else f"{x:+.1f}",
                    va="center", ha="left" if x >= 0 else "right", fontsize=6.3, color=INK2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([S[ks[i]]["label"] for i in o], fontsize=7)
    fig.text(0.5, -0.03, "azul = mejor que la línea base · naranja = peor (para la energía fósil, menos es mejor)",
             ha="center", fontsize=7, color=INK2)
    return save(fig, "l1_scatter")


def country_matrix(d, key="dev", mode="global"):
    S = d["strategies"]
    iso = d["iso3"]
    ks = strat_order(d)
    M = np.array([[np.asarray(S[k].get(mode, S[k]["global"])[key], float)[iso.index(x)] for k in ks] for x in c.FOCUS])
    return M, ks


def l1_countries(d):
    M, ks = country_matrix(d, "dev", "unilateral")
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    S = d["strategies"]
    heat(ax, M, c.FOCUS, [SHORT[k] for k in ks], fmt="{:+.2f}", fs=6.2)
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=6.2)
    return save(fig, "l1_countries")


def best_counts(d):
    S = d["strategies"]
    act = np.array(d["active"])
    ks = strat_order(d)
    pop = np.array(d["pop_2030"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
    for ax, (key, title) in zip(axes, (("dev", "Máximo desarrollo (ΔID)"), ("welfare_value", "Máximo bienestar"),
                                       ("revealed_value", "Máxima recompensa REVELADA"))):
        M = np.array([np.asarray(S[k].get("unilateral", S[k]["global"])[key], float) for k in ks])
        M = np.where(np.isfinite(M), M, -np.inf)
        best = np.argmax(M, 0)
        cnt = np.array([np.sum(act & (best == j)) for j in range(len(ks))])
        pp = np.array([np.nansum(np.where(act & (best == j), pop, 0)) for j in range(len(ks))]) / np.nansum(pop[act])
        y = np.arange(len(ks))
        ax.barh(y, cnt, color=BLUE, height=0.6)
        for yi, cv, pv in zip(y, cnt, pp):
            if cv:
                ax.text(cv + 0.8, yi, f"{cv} ({pv:.0%} pob.)", va="center", fontsize=6.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0, max(cnt) * 1.6 + 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(ks)))
    axes[0].set_yticklabels([S[k]["label"] for k in ks], fontsize=7)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("número de países para los que la estrategia es la mejor (adopción cuasi-unilateral)")
    return save(fig, "best_counts")


# ---------------------------------------------------------------- L0
def w3_bars(d):
    S = d["strategies"]
    ks = ["base"] + strat_order(d)
    labs = [S[k]["label"] for k in ks]
    p70 = np.array([S[k]["w3"]["p_collapse_2070"] for k in ks])
    sev = np.array([S[k]["w3"]["p_severe"] for k in ks])
    pp = np.array([S[k]["w3"]["ppolx_peak"] for k in ks])
    hwi = np.array([S[k]["w3"]["hwi_2100"] for k in ks])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8), sharey=True)
    y = np.arange(len(ks))
    ax = axes[0]
    ax.scatter(p70 * 100, y, color=RED, s=30, zorder=3, label="colapso ≤ 2070")
    ax.scatter(sev * 100, y, color=INK2, marker="D", s=18, zorder=3, label="colapso severo ≤ 2100")
    ax.hlines(y, np.minimum(p70, sev) * 100, np.maximum(p70, sev) * 100, color=GRID, lw=2)
    ax.axvline(p70[0] * 100, color=RED, lw=0.6, ls=":")
    ax.set_title("Probabilidad de colapso (%)", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=1)
    ax.set_xlim(60, 101)
    ax = axes[1]
    ax.barh(y, pp, color=[ORANGE if v > pp[0] else AQUA for v in pp], height=0.6)
    ax.axvline(pp[0], color=INK2, lw=0.6, ls=":")
    ax.set_title("Pico de contaminación\npersistente (×1970)", fontsize=8.5)
    ax = axes[2]
    ax.barh(y, hwi, color=[BLUE if v >= hwi[0] else MUTED for v in hwi], height=0.6)
    ax.axvline(hwi[0], color=INK2, lw=0.6, ls=":")
    ax.set_title("Índice de bienestar\nhumano 2100 (HWI)", fontsize=8.5)
    ax.set_xlim(0.15, max(hwi) * 1.08)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labs, fontsize=7)
    axes[0].invert_yaxis()
    for a in axes:
        a.grid(axis="y", visible=False)
    return save(fig, "w3_bars")


def w3_bands(d, picks):
    S = d["strategies"]
    cols = [INK2, BLUE, AQUA, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (var, title, sc) in zip(axes, (("pop", "Población (miles de millones)", 1e-9),
                                           ("iopc", "Producto industrial per cápita", 1), ("ppolx", "Contaminación persistente", 1))):
        for k, col in zip(picks, cols):
            yrs = np.array(S[k]["w3"]["years"])
            B = np.array(S[k]["w3"]["bands"][var]) * sc
            m = yrs >= 1990
            ax.fill_between(yrs[m], B[m, 0], B[m, 2], color=col, alpha=0.12, lw=0)
            ax.plot(yrs[m], B[m, 1], color=col, lw=1.8, label=S[k]["label"])
        ax.axvline(2020, color=MUTED, lw=0.6, ls=":")
        ax.set_title(title, fontsize=8)
    axes[0].legend(fontsize=6.3, loc="lower left")
    fig.text(0.5, -0.04, "World3-03, 400 estados del mundo: mediana y banda p10-p90; política desde 2020", ha="center",
             fontsize=7, color=INK2)
    return save(fig, "w3_bands")


def sensitivity(sens, d):
    S = d["strategies"]
    ks = strat_order(d)
    chans = ["fioac2_factor", "resource_tech_rate", "pollution_tech_rate", "services_factor", "icor_factor", "family_factor"]
    names = ["Consumo/\ninversión", "Tecnología\nrecursos", "Tecnología\ncontaminación", "Servicios\nsociales",
             "Productividad\ndel capital", "Fecundidad"]
    M = np.array([[sens[k]["attribution"][ch]["p_collapse_2070"] * 100 for ch in chans] for k in ks])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    heat(ax, M, [S[k]["label"] for k in ks], names, fmt="{:+.1f}", vmax=max(5, np.nanmax(np.abs(M))), cmap=DIV_R)
    ax.xaxis.tick_top()
    return save(fig, "sensitivity")


# ---------------------------------------------------------------- aprendizaje
def learning_curve(L):
    h = L["history"]
    it = np.array([r["iter"] for r in h])
    s = np.array([r["score"] for r in h])
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.scatter(it + np.random.default_rng(0).uniform(-0.18, 0.18, len(it)), s, s=9, color=BLUE, alpha=0.45, lw=0)
    best = [s[it <= i].max() for i in np.unique(it)]
    ax.plot(np.unique(it), best, color=ORANGE, lw=2, label="mejor hasta la iteración")
    fx = L["fixed_scores"]
    top = max((v["score"], k) for k, v in fx.items())
    ax.axhline(top[0], color=INK2, lw=0.8, ls="--")
    ax.text(0, top[0], f"  mejor estrategia fija del paso 2", fontsize=7, color=INK2, va="bottom")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("iteración del método de entropía cruzada (24 candidatos por iteración)")
    ax.set_ylabel("puntuación compuesta\n(pilares estandarizados)")
    ax.legend(fontsize=7, loc="lower right")
    return save(fig, "learning_curve")


def learned_policy(L):
    v = L["world"]["aprendida_mejor"]["por_zona"]
    zn = ["periferia", "semiperiferia", "centro"]
    M = np.array([[v[z][ch] for ch in c.CHANNELS] for z in zn])
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    heat(ax, M, ["Periferia", "Semiperiferia", "Centro"], ["Capital", "I+D", "Importar", "Extraer", "Inv.\nexterior", "Redis-\ntribuir"],
         vmax=0.5, fs=7.5)
    ax.xaxis.tick_top()
    return save(fig, "learned_policy")


def pillars(L, d):
    S = d["strategies"]
    fx = L["fixed_scores"]
    rows = [(k, S[k]["label"], fx[k]) for k in fx if k != "base"]
    for name, lab in (("aprendida_mejor", "APRENDIDA (CEM)"),):
        reps = L["world"][name]["validacion"]
        rows.append((name, lab, {"score": np.mean([r["score"] for r in reps]),
                                 "pilares": {p: np.mean([r["pilares"][p] for r in reps]) for p in reps[0]["pilares"]}}))
    rows.sort(key=lambda r: r[2]["score"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(rows))
    w = 0.26
    for j, (p, col) in enumerate((("desarrollo", BLUE), ("ambiente", AQUA), ("colapso", ORANGE))):
        ax.barh(y + (j - 1) * w, [r[2]["pilares"][p] for r in rows], height=w, color=col, label=f"pilar {p}")
    ax.scatter([r[2]["score"] for r in rows], y, color=INK, marker="D", s=22, zorder=4, label="puntuación total")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.2)
    for t in ax.get_yticklabels():
        if "APRENDIDA" in t.get_text():
            t.set_weight("bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("desviaciones típicas frente a la línea base (positivo = mejor)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    return save(fig, "pillars")


def learned_countries(L):
    C = L["countries"]
    isos = [x for x in c.FOCUS if x in C]
    M = np.array([[C[x]["offsets"][ch] for ch in c.CHANNELS] for x in isos])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 6.0), gridspec_kw={"width_ratios": [6, 2]})
    heat(ax, M, isos, ["Capital", "I+D", "Importar", "Extraer", "Inv.\nexterior", "Redis-\ntribuir"], vmax=0.5, fs=6.8)
    ax.xaxis.tick_top()
    dv = np.array([C[x]["efectos"]["dev"] for x in isos])
    wv = np.array([C[x]["efectos"]["welfare_value"] for x in isos])
    yy = np.arange(len(isos))
    ax2.barh(yy, dv, color=BLUE, height=0.6)
    ax2.set_ylim(len(isos) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.set_xlabel("ΔID nacional 2030\n(puntos)", fontsize=7)
    ax2.grid(axis="y", visible=False)
    return save(fig, "learned_countries")


def make_all():
    irl = c.load("irl.json")
    d = c.load("strategies.json")
    out = {"pipeline": pipeline(), "irl_weights": irl_weights(irl), "irl_rationality": irl_rationality(irl),
           "irl_groups": irl_groups(irl), "irl_countries": irl_countries(irl), "irl_deviations": irl_deviations(irl),
           "l1_scatter": l1_scatter(d), "l1_countries": l1_countries(d), "best_counts": best_counts(d),
           "w3_bars": w3_bars(d)}
    S = d["strategies"]
    ks = strat_order(d)
    by_col = sorted(ks, key=lambda k: S[k]["w3"]["p_collapse_2070"])
    by_pol = sorted(ks, key=lambda k: S[k]["w3"]["ppolx_peak"])
    picks = ["base", by_col[0], by_pol[0] if by_pol[0] != by_col[0] else by_pol[1], "desarrollista"]
    out["w3_bands"] = w3_bands(d, picks)
    if (c.OUT / "sensitivity.json").exists():
        out["sensitivity"] = sensitivity(c.load("sensitivity.json"), d)
    if (c.OUT / "learning.json").exists():
        L = c.load("learning.json")
        out.update(learning_curve=learning_curve(L), learned_policy=learned_policy(L), pillars=pillars(L, d),
                   learned_countries=learned_countries(L))
    return out


if __name__ == "__main__":
    print(make_all())
