"""Estudio de escenarios por ESCALA de implementación sobre el núcleo L0 de Phy-World (World3-03).

Continúa ``reports/arbol_decision``: allí cada palanca era sí/no; aquí cada una tiene niveles de
intensidad, con el mismo ensemble de 500 mundos filtrados por la historia (UN WPP 2024, 2000 y 2020)
y las mismas reglas de destino. Experimentos:

A. Dosis-respuesta de cada palanca (7 niveles), sola y dentro del paquete de referencia, en las dos
   geologías (recursos base y abundantes).
B. Intensidad del paquete × año de arranque.
C. Geología continua: recurso inicial × intensidad del paquete.
D. Interacciones: agro-tecnología × tecnología de contaminación; familia objetivo × suficiencia.
E. Nueve escenarios compuestos con trayectorias completas.

Todo es un modo de comportamiento del modelo de 2004, no un pronóstico. Salida: ``estudio.json``.

    PYTHONPATH=. python reports/escenarios/estudio.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arbol_decision"))
import proyeccion as base  # noqa: E402  (ensemble, historia y destinos compartidos)

from world4.core import l0  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).with_name("estudio.json")
NEVER = base.NEVER
YEAR = 2027.0
REC = tuple(base.VARS) + ("io",)

# ── palancas y sus niveles ──
LEVERS = {
    "poll": {"label": "Tecnología de control de la contaminación", "unit": "%/año de mejora",
             "levels": [0, .01, .02, .03, .04, .06, .08], "ref": .04},
    "res": {"label": "Eficiencia en el uso de recursos", "unit": "%/año de mejora",
            "levels": [0, .01, .02, .03, .04, .06, .08], "ref": .04},
    "agro": {"label": "Agro-tecnología (rendimiento)", "unit": "%/año de mejora",
             "levels": [0, .01, .02, .03, .04, .06, .08], "ref": .04},
    "soil": {"label": "Protección de suelos", "unit": "sí/no", "levels": [0, 1], "ref": 1},
    "fam": {"label": "Transición demográfica", "unit": "hijos deseados por familia",
            "levels": [0, 3.0, 2.5, 2.25, 2.0, 1.75, 1.5], "ref": 2.0},
    "suff": {"label": "Suficiencia material", "unit": "producción industrial per cápita deseada (USD 1968)",
             "levels": [0, 500, 450, 400, 350, 300, 250], "ref": 350},
}
OFF = {"poll": 0, "res": 0, "agro": 0, "soil": 0, "fam": 0, "suff": 0}
REF = {k: v["ref"] for k, v in LEVERS.items()}


def package(s: float) -> dict:
    """Paquete a escala ``s`` (1 = escenario 9 de Limits to Growth adelantado a hoy)."""
    if s <= 0:
        return dict(OFF)
    return {"poll": .04 * s, "res": .04 * s, "agro": .04 * s, "soil": 1, "fam": max(1.4, 3.0 - 1.0 * s),
            "suff": max(200.0, 500 - 150 * s)}


def run(U, pol: dict, *, geo: float = 1.0, year: float = YEAR, rec=REC):
    """Una política (dict de niveles) sobre el ensemble ``U``; ``geo`` multiplica el recurso inicial
    (> 1 incluye la mejor tecnología de extracción del escenario 2)."""
    n = len(U["nri_mult"])
    full = lambda v: np.full(n, float(v))  # noqa: E731
    c = {"nri": 1e12 * U["nri_mult"] * geo, "ahl70": U["ahl70"], "pptd": U["pptd"], "tdd": U["tdd"],
         "dcfsn": U["dcfsn"], "imef": U["imef"],
         "fcaortm": full(2002.0 if geo > 1 else NEVER), "pyear": full(year),
         "llmytm": full(year if pol["soil"] else NEVER),
         "zpgt": full(year if pol["fam"] else NEVER), "fcest": full(year if pol["fam"] else NEVER),
         "dcfszpg": full(pol["fam"] or 2.0),
         "iet": full(year if pol["suff"] else NEVER), "iopcd": full(pol["suff"] or 400.0)}
    t = {"pptcm": (-1, 0, 1, [-pol["poll"], 0]), "rtcm": (-1, 0, 1, [pol["res"], 0]),
         "lytcrm": (0, 1, 1, [0, pol["agro"]]), **base.RICH_TABLE}
    r = l0.run(constants=c, tables=t, n=n, t1=base.T1, dt=base.DT, record=rec)
    yearly = np.isclose(r.time % 1.0, 0.0)
    return r.time[yearly].astype(int), {k: r.series[k][yearly] for k in rec}


# ── F. anatomía del valle: qué hunde la esperanza de vida y qué la recupera ──
VALLEY_REC = REC + ("fcaor", "ly", "al", "lfert", "sopc", "lyt", "ppt", "rct")
_TAB = l0.TABLES
_C = l0.CONSTANTS


def _tab(name, x):
    lo, _, step, ys = _TAB[name]
    return np.interp(x, lo + step * np.arange(len(ys)), ys)


def valley(U, pol, geo, year=YEAR):
    """Trayectorias medianas, año del valle y reparto de la caída de ln(esperanza de vida) entre sus cuatro
    multiplicadores de World3-03: comida (lmf), servicios de salud (lmhs, por residuo), contaminación (lmp) y
    hacinamiento (lmc)."""
    yrs, r = run(U, pol, geo=geo, year=year, rec=VALLEY_REC)
    i0 = int(np.searchsorted(yrs, base.START))
    le = r["le"]
    m = np.arange(le.shape[1])
    imin = i0 + le[i0:].argmin(0)
    lmf = _tab("lmf", r["fpc"] / _C["sfpc"])
    lmp = _tab("lmp", r["ppolx"])
    lmc = 1 - _tab("cmi", r["iopc"]) * _tab("fpu", r["pop"])
    lmhs = le / (_C["len"] * lmf * lmp * lmc)
    tot = np.log(le[imin, m]) - np.log(le[i0])
    share = {k: np.log(v[imin, m]) - np.log(v[i0]) for k, v in
             (("comida", lmf), ("salud", lmhs), ("contaminacion", lmp), ("hacinamiento", lmc))}
    deep = tot < -0.05                       # sólo los mundos con un valle de verdad (> 5 % de caída)
    keep = (yrs >= 1990) & ((yrs - 1990) % 2 == 0)
    scale = {"pop": 1e-9, "al": 1e-9}
    out = {"years": yrs[keep].tolist(),
           "median": {k: [float(f"{v:.4g}") for v in np.median(r[k][keep] * scale.get(k, 1), 1)] for k in VALLEY_REC},
           "le_band": [[float(f"{v:.4g}") for v in np.percentile(le[keep], p, 1)] for p in (10, 90)],
           "valley_year": base.q(yrs[imin].astype(float)), "le_2026": float(np.median(le[i0])),
           "le_valley": float(np.median(le[imin, m])), "le_2100": float(np.median(le[-1])),
           "share": {k: float(np.median(v[deep] / tot[deep])) if deep.any() else 0.0 for k, v in share.items()},
           "deep_frac": float(deep.mean()),
           "at": {str(yy): {k: float(np.median(r[k][int(np.searchsorted(yrs, yy))]) * scale.get(k, 1)) for k in VALLEY_REC}
                  for yy in (2026, 2035, 2045, 2055, 2070, 2100)}}
    return out


BAU_IO = {}

# Cinco destinos: el informe del árbol tenía cuatro; aquí «crisis y recuperación» se separa en una crisis
# humanitaria (la esperanza de vida baja de 55 años en algún momento) y una contracción ordenada (la
# población cae más de 20 % desde su pico sin que la esperanza de vida baje de 55), que es lo que producen
# las políticas de fecundidad muy baja [M].
DESTINIES = ["Colapso", "Crisis humanitaria y recuperación", "Contracción ordenada", "Meseta en sobregiro",
             "Transición sostenible"]
LE_FLOOR = 55.0


def outcomes(years, s):
    o = base.outcomes(years, s)
    collapse = o["le_2100"] < LE_FLOOR
    crisis = ~collapse & (o["le_min"] < LE_FLOOR)
    contraction = ~collapse & ~crisis & (o["pop_drop"] > 0.20)
    overshoot = ~collapse & ~crisis & ~contraction & (o["hef_2100"] > 1.1)
    o["destiny"] = np.select([collapse, crisis, contraction, overshoot], [0, 1, 2, 3], 4)
    return o


def summary(o):
    d = o["destiny"]
    out = {"p_destiny": [float(np.mean(d == k)) for k in range(len(DESTINIES))], "n": int(d.size)}
    for k, v in o.items():
        if k != "destiny":
            out[k] = {"mean": float(np.mean(v)), "q": base.q(v)}
    return out


def summarise(years, s, geo_key, traj=False):
    o = outcomes(years, s)
    i27, i60 = int(np.searchsorted(years, 2027)), int(np.searchsorted(years, 2060))
    io_sum = s["io"][i27:i60].sum(0)
    out = summary(o)
    if geo_key in BAU_IO:
        rel = io_sum / BAU_IO[geo_key] - 1
        out["io_2027_2060"] = {"mean": float(rel.mean()), "q": base.q(rel)}
    fpc = s["fpc"][int(np.searchsorted(years, base.START)):]
    out["fpc_min"] = {"mean": float(fpc.min(0).mean()), "q": base.q(fpc.min(0))}
    ip = s["ppolx"][int(np.searchsorted(years, base.START)):]
    out["ppolx_peak_year"] = {"mean": float(years[base.START - years[0] + ip.argmax(0)].mean()),
                              "q": base.q(years[base.START - years[0] + ip.argmax(0)].astype(float))}
    if traj:
        keep = (years >= 1950) & ((years - 1950) % 2 == 0)
        out["traj"] = {"years": years[keep].tolist(),
                       **{k: [[float(f"{v:.4g}") for v in np.percentile(s[k][keep] * base.VARS[k][2], p, axis=1)]
                              for p in (10, 50, 90)] for k in base.VARS}}
    return out, o


def main():
    t0 = time.time()
    rng = np.random.default_rng(base.SEED)
    obs = base.observed_population()
    # el mismo ensemble que el árbol de decisión (misma semilla, mismo filtro)
    cand = base.sample_uncertain(6000, rng)
    c = base.constants_for(cand, res_abundant=np.zeros(6000), year=NEVER, clean=False, agro=False, demo=False,
                           suff=False)
    yrs, s = base.simulate(cand, base.tables_for(False, False), c)
    p2000 = s["pop"][np.searchsorted(yrs, 2000)] * 1e-9
    p2020 = s["pop"][np.searchsorted(yrs, 2020)] * 1e-9
    ok = (np.abs(p2000 / obs[2000] - 1) < 0.06) & (np.abs(p2020 / obs[2020] - 1) < 0.06)
    idx = np.flatnonzero(ok)[:500]
    U = {k: v[idx] for k, v in cand.items()}
    print(f"ensemble: {len(idx)} mundos")
    GEOS = {"base": 1.0, "abund": 2.0}
    res = {"meta": {}, "levers": LEVERS, "geos": GEOS, "destinies": DESTINIES,
           "vars": {k: {"label": v[0], "unit": v[1]} for k, v in base.VARS.items()}}

    # inercia (referencia del coste)
    res["bau"] = {}
    for g, gv in GEOS.items():
        y, ss = run(U, OFF, geo=gv)
        i27, i60 = int(np.searchsorted(y, 2027)), int(np.searchsorted(y, 2060))
        BAU_IO[g] = ss["io"][i27:i60].sum(0)
        res["bau"][g], _ = summarise(y, ss, g, traj=True)

    # A. dosis-respuesta
    dose = {}
    for lv, spec in LEVERS.items():
        dose[lv] = {}
        for ctx, basepol in (("solo", OFF), ("paquete", REF)):
            dose[lv][ctx] = {}
            for g, gv in GEOS.items():
                rows = []
                for level in spec["levels"]:
                    pol = dict(basepol)
                    pol[lv] = level
                    y, ss = run(U, pol, geo=gv)
                    summ, _ = summarise(y, ss, g, traj=True)
                    summ["level"] = level
                    rows.append(summ)
                dose[lv][ctx][g] = rows
        print(f"A {lv} ok ({time.time() - t0:.0f} s)")
    res["dose"] = dose

    # B. intensidad × año
    scales = [0, .25, .5, .75, 1.0, 1.25, 1.5]
    years_ = [2027, 2030, 2035, 2040, 2045, 2050, 2060]
    timing = {g: [] for g in GEOS}
    for g, gv in GEOS.items():
        for sc in scales:
            row = []
            for yr in years_:
                y, ss = run(U, package(sc), geo=gv, year=float(yr))
                summ, _ = summarise(y, ss, g)
                row.append({k: summ[k] for k in ("p_destiny", "hwi_mean", "pop_2100", "le_2100", "hef_2100")})
            timing[g].append(row)
    res["timing"] = {"scales": scales, "years": years_, "grid": timing}
    print(f"B ok ({time.time() - t0:.0f} s)")

    # C. geología continua (con tecnología de extracción) × intensidad
    geo_mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    gscales = [0, .5, 1.0, 1.5]
    geo = []
    for sc in gscales:
        row = []
        for gm in geo_mults:
            y, ss = run(U, package(sc), geo=gm if gm > 1 else 1.0001)
            summ, _ = summarise(y, ss, "none")
            row.append({k: summ[k] for k in ("p_destiny", "hwi_mean", "pop_2100", "le_2100", "hef_2100",
                                             "nrfr_2100", "ppolx_peak")})
        geo.append(row)
    res["geology"] = {"mults": geo_mults, "scales": gscales, "grid": geo}
    print(f"C ok ({time.time() - t0:.0f} s)")

    # D. interacciones (recursos abundantes; el resto del paquete en su referencia)
    rates = [0, .01, .02, .03, .04, .06, .08]
    inter_tech = []
    for ra in rates:
        row = []
        for rp in rates:
            pol = dict(REF, agro=ra, poll=rp)
            y, ss = run(U, pol, geo=2.0)
            summ, _ = summarise(y, ss, "abund")
            row.append({k: summ[k] for k in ("p_destiny", "hwi_mean", "hef_2100", "le_2100")})
        inter_tech.append(row)
    fams = [0, 3.0, 2.5, 2.25, 2.0, 1.75, 1.5]
    suffs = [0, 500, 450, 400, 350, 300, 250]
    inter_soc = []
    for fm in fams:
        row = []
        for sf in suffs:
            pol = dict(REF, fam=fm, suff=sf)
            y, ss = run(U, pol, geo=2.0)
            summ, _ = summarise(y, ss, "abund")
            row.append({k: summ[k] for k in ("p_destiny", "hwi_mean", "hef_2100", "le_2100", "pop_2100",
                                             "iopc_2100")})
        inter_soc.append(row)
    res["interactions"] = {"rates": rates, "tech": inter_tech, "fams": fams, "suffs": suffs, "social": inter_soc}
    print(f"D ok ({time.time() - t0:.0f} s)")

    # E. escenarios compuestos
    SCEN = [
        ("inercia", "Inercia", "Ninguna política nueva: el mundo sigue sus tendencias.", OFF, YEAR),
        ("timido", "Tímido", "Todas las palancas al 25 %: mejoras de 1 %/año, 2,75 hijos, consumo 462.", package(.25), YEAR),
        ("moderado", "Moderado", "Todas al 50 %: mejoras de 2 %/año, 2,5 hijos, consumo 425.", package(.5), YEAR),
        ("ltg", "Limits to Growth", "El escenario 9 del libro, desde 2027: 4 %/año, 2 hijos, consumo 350.", package(1.0), YEAR),
        ("ambicioso", "Ambicioso", "Todas al 150 %: 6 %/año, 1,5 hijos, consumo 275.", package(1.5), YEAR),
        ("tecno", "Tecno-optimista", "Tecnología al 150 % (6 %/año y suelos), sin políticas sociales.",
         dict(OFF, poll=.06, res=.06, agro=.06, soil=1), YEAR),
        ("sobrio", "Sobriedad", "Políticas sociales fuertes (1,75 hijos, consumo 300) y tecnología tímida (1 %/año).",
         dict(OFF, poll=.01, res=.01, agro=.01, soil=1, fam=1.75, suff=300), YEAR),
        ("tardio", "Paquete tardío", "El paquete Limits to Growth completo, pero desde 2045.", package(1.0), 2045.0),
        ("verde_sin_comida", "Verde sin agro", "Contaminación y recursos al 150 %, políticas sociales de referencia, sin agro-tecnología.",
         dict(REF, poll=.06, res=.06, agro=0, soil=0), YEAR),
    ]
    scen = []
    for key, name, what, pol, yr in SCEN:
        entry = {"key": key, "name": name, "what": what, "policy": pol, "year": yr, "geo": {}}
        for g, gv in GEOS.items():
            y, ss = run(U, pol, geo=gv, year=yr)
            entry["geo"][g], _ = summarise(y, ss, g, traj=True)
        scen.append(entry)
    res["scenarios"] = scen
    print(f"E ok ({time.time() - t0:.0f} s)")

    # F. anatomía del valle
    res["valley"] = {
        "cases": [
            {"key": "bau_base", "name": "Inercia · recursos base", **valley(U, OFF, 1.0)},
            {"key": "bau_abund", "name": "Inercia · recursos abundantes", **valley(U, OFF, 2.0)},
            {"key": "ltg_base", "name": "Paquete 1× · recursos base", **valley(U, REF, 1.0)},
            {"key": "ltg_abund", "name": "Paquete 1× · recursos abundantes", **valley(U, REF, 2.0)},
            {"key": "amb_abund", "name": "Paquete 1,5× · recursos abundantes", **valley(U, package(1.5), 2.0)},
            {"key": "late_abund", "name": "Paquete 1× desde 2045 · recursos abundantes", **valley(U, REF, 2.0, 2045.0)},
        ],
        "delays": {k: _C[k] for k in ("pptd", "tdd", "alic1", "alsc1", "ahl70", "lpd", "hsid", "sad")},
        "ranges": {k: list(base.UNCERTAIN[k][:2]) for k in ("pptd", "tdd", "ahl70")},
    }
    print(f"F ok ({time.time() - t0:.0f} s)")

    res["meta"] = {"generated": time.strftime("%Y-%m-%d"), "members": len(idx), "seed": base.SEED,
                   "runtime_s": round(time.time() - t0, 1), "year": YEAR}
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"listo en {res['meta']['runtime_s']} s -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
