"""Proyección por árbol de decisión sobre el núcleo L0 de Phy-World (World3-03, ``world4.core.l0``).

Qué hace, en orden:

1. **Ensemble de estados del mundo.** Perillas inciertas de World3-03 (recurso no renovable, vida media
   de absorción de la contaminación, retardos de transmisión y de tecnología, tamaño de familia deseado,
   factor de emisión industrial) muestreadas de cajas uniformes [M] y filtradas por *history matching*
   contra la población mundial observada (UN WPP 2024) en 2000 y 2020: sólo sobreviven los mundos cuya
   historia se parece a la nuestra.
2. **Árbol de decisión.** Cinco decisiones (cuándo actuar, tecnología limpia, agro-tecnología y suelos,
   transición demográfica, suficiencia material) y un nodo de azar (geología: recursos base o
   abundantes). 32 políticas × 2 estados = 64 hojas, cada una con todo el ensemble (números aleatorios
   comunes). Retro-inducción: el azar promedia, la decisión elige el mayor bienestar esperado.
3. **Árbol aprendido (CART).** Monte Carlo con decisiones, año de acción continuo e incertidumbres a la
   vez; un ``DecisionTreeClassifier`` aprende qué separa los destinos.
4. **Barridos**: el coste de esperar (año de acción 2027-2070) y el efecto marginal de cada palanca.

Todo es un MODO DE COMPORTAMIENTO del modelo de 2004, no un pronóstico (la regla de honestidad de
Phy-World). Salida: ``data.json`` junto a este fichero.

    python reports/arbol_decision/proyeccion.py
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np

from world4.core import l0

warnings.simplefilter("ignore")
OUT = Path(__file__).with_name("data.json")
SEED = 20261021
T1 = 2100.0
DT = 0.5
START = 2026                      # el presente del informe
NOW, LATE = 2027.0, 2045.0        # actuar ya o esperar
NEVER = 4000.0                    # la convención de World3-03 para "nunca"

# Población mundial observada (miles de millones), para el history matching y la figura histórica
WPP_OBS = Path(__file__).resolve().parents[2] / "data_cache" / "wpp2024" / "wpp_age.npz"

# ── Incertidumbre: cajas [M] alrededor del valor de World3-03 ──
UNCERTAIN = {
    "nri_mult": (0.75, 1.30, "recurso no renovable inicial (× 1e12)"),
    "ahl70": (1.0, 2.5, "vida media de absorción de la contaminación en 1970 (años)"),
    "pptd": (15.0, 30.0, "retardo de transmisión de la contaminación (años)"),
    "tdd": (15.0, 30.0, "retardo de desarrollo tecnológico (años)"),
    "dcfsn": (3.4, 4.2, "tamaño de familia deseado normal (hijos)"),
    "imef": (0.07, 0.13, "factor de emisión de materiales industriales"),
}

# ── Palancas: lo que cambia cada decisión en World3-03 (escenarios 2-9 de Limits to Growth 2004) ──
# rtcm con signo +: en l0.py el flujo de rct es -rctcr, así que la tabla [-0.04, 0] del escenario 6 de
# Limits to Growth AUMENTARÍA el uso de recursos; con +0.04 la tecnología ahorra recursos, como en el libro.
TECH_CLEAN = {"pptcm": (-1, 0, 1, [-0.04, 0]), "rtcm": (-1, 0, 1, [0.04, 0])}
TECH_AGRO = {"lytcrm": (0, 1, 1, [0, 0.04])}
OFF_TABLES = {"pptcm": (-1, 0, 1, [0, 0]), "rtcm": (-1, 0, 1, [0, 0]), "lytcrm": (0, 1, 1, [0, 0])}
RICH_TABLE = {"fcaor2": (0, 1, 0.1, [1.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])}
P_ABUNDANT = 0.5                  # probabilidad subjetiva declarada del nodo de azar

DECISIONS = [
    {"key": "year", "label": "Momento de actuar", "options": ["2027", "2045"]},
    {"key": "clean", "label": "Tecnología limpia y eficiencia de recursos", "options": ["No", "Sí"]},
    {"key": "agro", "label": "Agro-tecnología y protección de suelos", "options": ["No", "Sí"]},
    {"key": "demo", "label": "Transición demográfica (2 hijos)", "options": ["No", "Sí"]},
    {"key": "suff", "label": "Suficiencia material (consumo estable)", "options": ["No", "Sí"]},
]
CHANCE = {"key": "res", "label": "Geología: recursos no renovables", "options": ["Base", "Abundantes"],
          "p": [1 - P_ABUNDANT, P_ABUNDANT]}

DESTINIES = ["Colapso", "Crisis y recuperación", "Meseta en sobregiro", "Transición sostenible"]
VARS = {
    "pop": ("Población", "miles de millones", 1e-9),
    "iopc": ("Producción industrial per cápita", "USD 1968 / persona·año", 1.0),
    "fpc": ("Alimento per cápita", "kg equiv. vegetal / persona·año", 1.0),
    "le": ("Esperanza de vida", "años", 1.0),
    "ppolx": ("Contaminación persistente", "índice, 1970 = 1", 1.0),
    "nrfr": ("Recursos no renovables restantes", "fracción del inicial", 1.0),
    "hwi": ("Índice de bienestar humano", "0-1", 1.0),
    "hef": ("Huella ecológica", "planetas (1 = capacidad de la Tierra)", 1.0),
}


def observed_population():
    d = np.load(WPP_OBS)
    years = d["years"]
    # 236 códigos de WPP: países y territorios (sin agregados); popM/popF en miles por edad simple
    tot = (d["popM"].sum(2) + d["popF"].sum(2)).sum(0) * 1e3 / 1e9
    return {int(y): float(v) for y, v in zip(years, tot)}


def sample_uncertain(n, rng):
    return {k: rng.uniform(lo, hi, n) for k, (lo, hi, _) in UNCERTAIN.items()}


def constants_for(u, *, res_abundant, year, clean, agro, demo, suff):
    """Constantes de World3-03 por miembro (arrays (n,)) para una política y un estado de la geología."""
    n = len(u["nri_mult"])
    full = lambda v: np.broadcast_to(np.asarray(v, float), (n,))  # noqa: E731
    on = lambda flag: np.where(np.asarray(flag, bool), year, NEVER)  # noqa: E731
    nri = 1e12 * u["nri_mult"] * np.where(np.asarray(res_abundant, bool), 2.0, 1.0)
    c = {
        "nri": nri, "ahl70": u["ahl70"], "pptd": u["pptd"], "tdd": u["tdd"], "dcfsn": u["dcfsn"],
        "imef": u["imef"],
        "fcaortm": np.where(np.asarray(res_abundant, bool), 2002.0, NEVER) * np.ones(n),
        "pyear": full(year),                     # activa las tecnologías elegidas (tablas)
        "llmytm": on(agro) * np.ones(n),         # protección de suelos
        "zpgt": on(demo) * np.ones(n), "fcest": on(demo) * np.ones(n),
        "iet": on(suff) * np.ones(n), "iopcd": np.where(np.asarray(suff, bool), 350.0, 400.0) * np.ones(n),
    }
    return c


def tables_for(clean, agro):
    t = dict(OFF_TABLES)
    t.update(RICH_TABLE)                         # sólo actúa después de fcaortm
    if clean:
        t.update(TECH_CLEAN)
    if agro:
        t.update(TECH_AGRO)
    return t


def simulate(u, tables, consts):
    n = len(next(iter(consts.values())))
    r = l0.run(constants=consts, tables=tables, n=n, t1=T1, dt=DT, record=tuple(VARS))
    yearly = np.isclose(r.time % 1.0, 0.0)
    return r.time[yearly].astype(int), {k: r.series[k][yearly] for k in VARS}


def outcomes(years, s):
    """Métricas por miembro y destino (clase 0-3)."""
    i0 = int(np.searchsorted(years, START))
    i27 = int(np.searchsorted(years, 2027))
    pop = s["pop"][i0:]
    peak = pop.max(0)
    peak_year = years[i0:][pop.argmax(0)]
    pop2100 = pop[-1]
    hwi = s["hwi"]
    hwi_now = hwi[i0]
    hwi_mean = hwi[i27:].mean(0)
    hwi2100 = hwi[-1]
    hef2100 = s["hef"][-1]
    hef_mean = s["hef"][i27:].mean(0)
    le_min = s["le"][i0:].min(0)
    le2100 = s["le"][-1]
    drop = 1 - pop2100 / peak
    # destinos [M]: colapso = la esperanza de vida no vuelve a 55 años en 2100; crisis y recuperación =
    # cae más de 20 % la población o la esperanza de vida baja de 45, pero se recupera; meseta en
    # sobregiro = sin crisis, con la huella > 1,1 planetas en 2100; sostenible = el resto
    collapse = le2100 < 55
    crisis = ~collapse & ((drop > 0.20) | (le_min < 45))
    overshoot = ~collapse & ~crisis & (hef2100 > 1.1)
    destiny = np.select([collapse, crisis, overshoot], [0, 1, 2], 3)
    return {
        "pop_peak": peak * 1e-9, "pop_peak_year": peak_year.astype(float), "pop_2100": pop2100 * 1e-9,
        "hwi_mean": hwi_mean, "hwi_2100": hwi2100, "hef_2100": hef2100, "hef_mean": hef_mean,
        "le_2100": le2100, "le_min": le_min, "iopc_2100": s["iopc"][-1], "fpc_2100": s["fpc"][-1],
        "ppolx_peak": s["ppolx"][i0:].max(0), "nrfr_2100": s["nrfr"][-1], "pop_drop": drop,
        "destiny": destiny,
    }


def q(x, ps=(10, 50, 90)):
    return [float(np.percentile(x, p)) for p in ps]


def summary(o):
    d = o["destiny"]
    out = {"p_destiny": [float(np.mean(d == k)) for k in range(4)], "n": int(d.size)}
    for k, v in o.items():
        if k == "destiny":
            continue
        out[k] = {"mean": float(np.mean(v)), "q": q(v)}
    return out


def fan(years, s, every=1, y0=1950):
    keep = (years >= y0) & ((years - y0) % every == 0)
    res = {"years": years[keep].tolist()}
    for k, (_, _, scale) in VARS.items():
        arr = s[k][keep] * scale
        res[k] = [[float(f"{v:.4g}") for v in np.percentile(arr, p, axis=1)] for p in (10, 50, 90)]
    return res


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    obs = observed_population()

    # 1) ensemble filtrado por la historia (common random numbers en todo el informe)
    cand = sample_uncertain(6000, rng)
    c = constants_for(cand, res_abundant=np.zeros(6000), year=NEVER, clean=False, agro=False, demo=False,
                      suff=False)
    yrs, s = simulate(cand, tables_for(False, False), c)
    p2000 = s["pop"][np.searchsorted(yrs, 2000)] * 1e-9
    p2020 = s["pop"][np.searchsorted(yrs, 2020)] * 1e-9
    ok = (np.abs(p2000 / obs[2000] - 1) < 0.06) & (np.abs(p2020 / obs[2020] - 1) < 0.06)
    idx = np.flatnonzero(ok)[:500]
    U = {k: v[idx] for k, v in cand.items()}
    N = len(idx)
    print(f"history matching: {ok.sum()} / {ok.size} aceptados; se usan {N}")
    hm = {"candidates": int(ok.size), "accepted": int(ok.sum()), "used": N,
          "window": "±6 % de la población WPP 2024 en 2000 y 2020",
          "pop2000_q": q(p2000[idx]), "pop2020_q": q(p2020[idx]),
          "obs2000": obs[2000], "obs2020": obs[2020],
          "prior": {k: [lo, hi, what] for k, (lo, hi, what) in UNCERTAIN.items()}}
    # historia: modelo (ensemble aceptado) vs observado, 1950-2023
    hist_keep = (yrs >= 1950) & (yrs <= 2023)
    history = {"years": yrs[hist_keep].tolist(),
               "model": [np.round(np.percentile(s["pop"][hist_keep][:, idx] * 1e-9, p, axis=1), 4).tolist()
                         for p in (10, 50, 90)],
               "observed": [round(obs[int(y)], 4) for y in yrs[hist_keep]]}

    # 2) las 64 hojas del árbol
    leaves = {}
    fans = {}
    for year_i, clean, agro, demo, suff in itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)):
        year = (NOW, LATE)[year_i]
        for res in (0, 1):
            cc = constants_for(U, res_abundant=np.full(N, res), year=year, clean=clean, agro=agro, demo=demo,
                               suff=suff)
            y, ss = simulate(U, tables_for(clean, agro), cc)
            key = f"{year_i}{clean}{agro}{demo}{suff}{res}"
            leaves[key] = summary(outcomes(y, ss))
            fans[key] = fan(y, ss, every=2)
    years = y

    # retro-inducción del árbol: orden DECISIONS... luego CHANCE
    def node(prefix, depth):
        if depth == len(DECISIONS):
            kids = []
            for r_i, label in enumerate(CHANCE["options"]):
                leaf = leaves[prefix + str(r_i)]
                kids.append({"type": "leaf", "label": label, "p": CHANCE["p"][r_i], "key": prefix + str(r_i),
                             "value": leaf["hwi_mean"]["mean"], "p_destiny": leaf["p_destiny"],
                             "stats": {k: leaf[k] for k in ("pop_2100", "pop_peak", "hwi_2100", "hef_2100",
                                                            "le_2100", "ppolx_peak", "iopc_2100", "fpc_2100")}})
            val = sum(k["p"] * k["value"] for k in kids)
            pd = [sum(k["p"] * k["p_destiny"][j] for k in kids) for j in range(4)]
            return {"type": "chance", "label": CHANCE["label"], "value": val, "p_destiny": pd, "children": kids,
                    "key": prefix}
        dec = DECISIONS[depth]
        kids = []
        for o_i, label in enumerate(dec["options"]):
            ch = node(prefix + str(o_i), depth + 1)
            ch["edge"] = label
            kids.append(ch)
        best = int(np.argmax([k["value"] for k in kids]))
        for i, k in enumerate(kids):
            k["best"] = i == best
        return {"type": "decision", "label": dec["label"], "value": kids[best]["value"],
                "p_destiny": kids[best]["p_destiny"], "children": kids, "best_choice": best, "key": prefix}

    tree = node("", 0)

    # ranking de las 32 políticas (promedio sobre la geología)
    policies = []
    for year_i, clean, agro, demo, suff in itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)):
        p = f"{year_i}{clean}{agro}{demo}{suff}"
        a, b = leaves[p + "0"], leaves[p + "1"]
        w = CHANCE["p"]
        mix = lambda f: w[0] * f(a) + w[1] * f(b)  # noqa: E731
        policies.append({
            "key": p, "year": DECISIONS[0]["options"][year_i], "clean": clean, "agro": agro, "demo": demo,
            "suff": suff,
            "value": mix(lambda L: L["hwi_mean"]["mean"]),
            "p_destiny": [mix(lambda L, j=j: L["p_destiny"][j]) for j in range(4)],
            "pop_2100": mix(lambda L: L["pop_2100"]["mean"]),
            "hef_2100": mix(lambda L: L["hef_2100"]["mean"]),
            "le_2100": mix(lambda L: L["le_2100"]["mean"]),
            "hwi_2100": mix(lambda L: L["hwi_2100"]["mean"]),
            "ppolx_peak": mix(lambda L: L["ppolx_peak"]["mean"]),
            "by_res": [{"value": L["hwi_mean"]["mean"], "p_destiny": L["p_destiny"], "hwi_q": L["hwi_mean"]["q"]}
                       for L in (a, b)],
        })
    policies.sort(key=lambda r: -r["value"])

    # efecto marginal de cada palanca (promedio sobre todas las demás y sobre la geología)
    marginal = []
    for j, dec in enumerate(DECISIONS):
        on = [p for p in policies if p["key"][j] == "1"]
        off = [p for p in policies if p["key"][j] == "0"]
        marginal.append({
            "key": dec["key"], "label": dec["label"], "on": dec["options"][1], "off": dec["options"][0],
            "d_value": float(np.mean([p["value"] for p in on]) - np.mean([p["value"] for p in off])),
            "d_collapse": float(np.mean([p["p_destiny"][0] for p in on]) - np.mean([p["p_destiny"][0] for p in off])),
            "d_sust": float(np.mean([p["p_destiny"][3] for p in on]) - np.mean([p["p_destiny"][3] for p in off])),
            "d_hef": float(np.mean([p["hef_2100"] for p in on]) - np.mean([p["hef_2100"] for p in off])),
            "d_pop": float(np.mean([p["pop_2100"] for p in on]) - np.mean([p["pop_2100"] for p in off])),
        })

    # 3) el coste de esperar: todas las palancas, año de acción 2027..2070 (y BAU como referencia)
    wait = []
    for yr in range(2027, 2071, 1):
        row = {"year": yr}
        for name, flags in (("all", (1, 1, 1, 1)), ("tech", (1, 1, 0, 0)), ("social", (0, 0, 1, 1))):
            acc = []
            for res in (0, 1):
                cc = constants_for(U, res_abundant=np.full(N, res), year=float(yr), clean=flags[0], agro=flags[1],
                                   demo=flags[2], suff=flags[3])
                y_, ss = simulate(U, tables_for(flags[0], flags[1]), cc)
                acc.append(outcomes(y_, ss))
            mixo = {k: np.concatenate([acc[0][k], acc[1][k]]) for k in acc[0]}
            row[name] = {"value": float(mixo["hwi_mean"].mean()),
                         "p_collapse": float(np.mean(mixo["destiny"] == 0)),
                         "p_sust": float(np.mean(mixo["destiny"] == 3)),
                         "hef_2100": float(mixo["hef_2100"].mean()),
                         "pop_2100": float(mixo["pop_2100"].mean())}
        wait.append(row)

    # 4) CART sobre un Monte Carlo conjunto (decisiones + año continuo + incertidumbre)
    from sklearn.tree import DecisionTreeClassifier
    M = 4000                                     # por combinación de tablas
    X_rows, y_rows, hwi_rows, hef_rows = [], [], [], []
    feat = ["clean", "agro", "demo", "suff", "year", "res"] + list(UNCERTAIN)
    for clean, agro in itertools.product((0, 1), (0, 1)):
        u = sample_uncertain(M, rng)
        demo = rng.integers(0, 2, M)
        suff = rng.integers(0, 2, M)
        res = rng.integers(0, 2, M)
        year = rng.uniform(2027, 2070, M)
        cc = constants_for(u, res_abundant=res, year=year, clean=clean, agro=agro, demo=demo, suff=suff)
        y_, ss = simulate(u, tables_for(clean, agro), cc)
        # mismo filtro de historia que el ensemble principal
        p00 = ss["pop"][np.searchsorted(y_, 2000)] * 1e-9
        p20 = ss["pop"][np.searchsorted(y_, 2020)] * 1e-9
        keep = (np.abs(p00 / obs[2000] - 1) < 0.06) & (np.abs(p20 / obs[2020] - 1) < 0.06)
        o = outcomes(y_, ss)
        X = np.column_stack([np.full(M, clean), np.full(M, agro), demo, suff, year, res] + [u[k] for k in UNCERTAIN])
        X_rows.append(X[keep])
        y_rows.append(o["destiny"][keep])
        hwi_rows.append(o["hwi_mean"][keep])
        hef_rows.append(o["hef_2100"][keep])
    X = np.vstack(X_rows)
    Y = np.concatenate(y_rows)
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=150, random_state=0).fit(X, Y)
    tr = clf.tree_
    feat_labels = {"clean": "Tecnología limpia", "agro": "Agro-tecnología", "demo": "Transición demográfica",
                   "suff": "Suficiencia material", "year": "Año de acción", "res": "Recursos abundantes",
                   **{k: v[2] for k, v in UNCERTAIN.items()}}

    def cart(i):
        counts = tr.value[i][0]
        counts = counts / counts.sum()
        nd = {"n": int(tr.n_node_samples[i]), "dist": [float(v) for v in counts],
              "cls": int(np.argmax(counts))}
        if tr.children_left[i] != -1:
            f = feat[tr.feature[i]]
            nd.update(feature=f, feature_label=feat_labels[f], threshold=float(tr.threshold[i]),
                      binary=f in ("clean", "agro", "demo", "suff", "res"),
                      left=cart(tr.children_left[i]), right=cart(tr.children_right[i]))
        return nd

    acc = float(clf.score(X, Y))
    from sklearn.model_selection import cross_val_score
    cv = float(np.mean(cross_val_score(DecisionTreeClassifier(max_depth=4, min_samples_leaf=150, random_state=0),
                                       X, Y, cv=5)))
    importance = sorted(({"feature": f, "label": feat_labels[f], "importance": float(v)}
                         for f, v in zip(feat, clf.feature_importances_)), key=lambda r: -r["importance"])
    # nube de futuros: muestra del Monte Carlo para el diagrama de dispersión
    pick = rng.choice(len(Y), size=min(2500, len(Y)), replace=False)
    cloud = {"hef": np.round(np.concatenate(hef_rows)[pick], 3).tolist(),
             "hwi": np.round(np.concatenate(hwi_rows)[pick], 3).tolist(),
             "destiny": Y[pick].astype(int).tolist(),
             "year": np.round(X[pick, 4], 1).tolist(),
             "levers": X[pick, :4].sum(1).astype(int).tolist()}

    # trayectorias de arquetipos (geología mezclada 50/50 -> se publican las dos)
    best_key = policies[0]["key"]
    archetypes = {
        "bau": {"label": "Inercia (ninguna palanca)", "key": "00000"},
        "tech": {"label": "Sólo tecnología, 2027", "key": "01100"},
        "social": {"label": "Sólo demografía y suficiencia, 2027", "key": "00011"},
        "late": {"label": "Todo, pero en 2045", "key": "11111"},
        "best": {"label": "Ruta óptima del árbol", "key": best_key},
    }

    data = {
        "meta": {"generated": time.strftime("%Y-%m-%d"), "model": "Phy-World L0 = World3-03 (NumPy)",
                 "start": START, "end": int(T1), "seed": SEED, "members": N, "runtime_s": None,
                 "decision_years": [NOW, LATE]},
        "decisions": DECISIONS, "chance": CHANCE, "destinies": DESTINIES,
        "vars": {k: {"label": v[0], "unit": v[1]} for k, v in VARS.items()},
        "history_matching": hm, "history": history,
        "tree": tree, "policies": policies, "marginal": marginal, "wait": wait,
        "cart": {"tree": cart(0), "accuracy": acc, "cv_accuracy": cv, "samples": int(len(Y)),
                 "importance": importance, "max_depth": 4, "min_samples_leaf": 150},
        "cloud": cloud, "archetypes": archetypes,
        "leaves": leaves, "fans": fans, "years": years.tolist(),
    }
    data["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"ok en {data['meta']['runtime_s']} s -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print("mejor ruta:", policies[0]["key"], round(policies[0]["value"], 4), [round(x, 3) for x in policies[0]["p_destiny"]])
    bau = next(p for p in policies if p["key"] == "00000")
    print("inercia:", round(bau["value"], 4), [round(x, 3) for x in bau["p_destiny"]])
    print("CART acc", round(acc, 3), "cv", round(cv, 3), [(r["feature"], round(r["importance"], 3)) for r in importance[:6]])


if __name__ == "__main__":
    main()
