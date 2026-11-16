"""Un mundo sintético pequeño con las mismas claves que ``data.conflict.load()``.

Sirve para que las pruebas del L2 corran sin descargar nada (y en CI): dos países sobre una malla de
``side x side`` celdas, un gobierno por país, dos grupos armados y civiles; cinco díadas con una dinámica
conocida (persistente, con un frente que se mueve hacia el este) y un desplazamiento proporcional a las
muertes. No pretende parecerse a ninguna región: pretende que los errores de contabilidad se vean.
"""
from __future__ import annotations

import numpy as np

from world4.data import conflict

ISO3 = ("AAA", "BBB")


def world(side: int = 6, months: int = 132, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    C = side * side
    row, col = np.divmod(np.arange(C), side)
    row, col = row + 240, col + 430                      # en algún lugar de la malla de PRIO-GRID
    gid = row * 720 + col + 1
    cell_country = (np.arange(C) % side >= side // 2).astype(np.int16)        # oeste AAA, este BBB
    cov = np.abs(rng.normal(size=(C, len(conflict.COVARIATES)))).astype(np.float32)
    pop_cell = rng.uniform(2e4, 4e5, C)
    years = np.arange(conflict.YEAR0, conflict.YEAR0 + months // 12)
    pop_years = np.arange(conflict.YEAR0, conflict.YEAR0 + months // 12 + 6)
    pop = np.stack([pop_cell[cell_country == k].sum() * 1.01 ** np.arange(len(pop_years)) for k in range(2)])

    actors = ["Government of AAA", "Government of BBB", "Front One", "Front Two", "Civilians"]
    kinds, homes = [0, 0, 1, 1, 2], [0, 1, -1, -1, -1]
    # díada: (lado a, lado b, tipo, país, celda inicial, eventos/mes cuando está activa, muertes por evento)
    dy = [(0, 2, 1, 0, 0, 6.0, 4.0), (1, 3, 1, 1, side - 1, 3.0, 2.0), (2, 3, 2, 0, side, 2.0, 3.0),
          (2, 4, 3, 0, 2 * side, 1.5, 6.0), (0, 1, 1, 1, C - 1, 0.6, 20.0)]
    ev = {k: [] for k in ("t0", "cell", "country", "type", "dyad", "best", "civ")}
    for d, (a, b, ty, k, c0, rate, leth) in enumerate(dy):
        active, level, centre = d < 3, rate, float(c0)
        for t in range(months):
            active = rng.random() < (0.9 if active else 0.08)
            if not active:
                continue
            level = float(np.clip(level * np.exp(rng.normal(0, 0.25)), 0.3 * rate, 4 * rate))
            centre = float(np.clip(centre + (0.04 if d == 0 else 0.0), 0, C - 1))          # el frente 0 avanza
            for _ in range(rng.poisson(level) + 1):
                c = int(np.clip(round(centre + rng.normal(0, 0.8)), 0, C - 1))
                best = float(rng.geometric(1.0 / leth))
                unloc = rng.random() < 0.05
                ev["t0"].append(t); ev["cell"].append(-1 if unloc else c)
                ev["country"].append(k if unloc or rng.random() < 0.9 else int(cell_country[c]))
                ev["type"].append(ty); ev["dyad"].append(d); ev["best"].append(best)
                ev["civ"].append(best if ty == 3 else np.floor(0.2 * best))
    E = len(ev["t0"])
    t0 = np.array(ev["t0"], np.int32)
    span = (rng.random(E) < 0.01) & (t0 < months - 2)                         # algunos eventos cruzan de mes
    best = np.array(ev["best"], np.float32)
    # desplazamiento: stocks que siguen a las muertes acumuladas del país, con un retorno lento
    deaths_ky = np.zeros((2, len(years)))
    np.add.at(deaths_ky, (np.array(ev["country"]), np.minimum(t0 // 12, len(years) - 1)), best)
    asylum = np.array(["AAA", "BBB", "CCC", "DDD"])
    ref_stock = np.zeros((2, len(asylum), len(years)), np.float32)
    idp = np.zeros((2, len(years)), np.float32)
    r, i = np.zeros(2), np.zeros(2)
    for y in range(len(years)):
        r, i = 0.9 * r + 30.0 * deaths_ky[:, y], 0.7 * i + 80.0 * deaths_ky[:, y]
        ref_stock[:, 2, y], ref_stock[:, 3, y], idp[:, y] = 0.7 * r, 0.3 * r, i
    return {
        "region": np.array("synthetic"), "iso3": np.array(ISO3), "year0": np.array(conflict.YEAR0),
        "T": np.array(months), "last_final_month": np.array(months - 1),
        "gid": gid.astype(np.int64), "row": row.astype(np.int32), "col": col.astype(np.int32),
        "lon": -180.0 + (col + 0.5) * 0.5, "lat": -90.0 + (row + 0.5) * 0.5, "cell_country": cell_country,
        "cov": cov, "cov_names": np.array(conflict.COVARIATES),
        "pop_gpw": np.repeat(pop_cell[:, None], 5, 1).astype(np.float32),
        "pop_gpw_years": np.array([1990, 1995, 2000, 2005, 2010]),
        "ev_t0": t0, "ev_t1": (t0 + span).astype(np.int32), "ev_cell": np.array(ev["cell"], np.int32),
        "ev_country": np.array(ev["country"], np.int16), "ev_type": np.array(ev["type"], np.int8),
        "ev_dyad": np.array(ev["dyad"], np.int32), "ev_best": best, "ev_low": best, "ev_high": best,
        "ev_deaths_a": np.zeros(E, np.float32), "ev_deaths_b": np.zeros(E, np.float32),
        "ev_civ": np.array(ev["civ"], np.float32), "ev_candidate": np.zeros(E, np.int8),
        "dyad_id": np.arange(len(dy), dtype=np.int64),
        "dyad_name": np.array([f"{actors[a]} - {actors[b]}" for a, b, *_ in dy]),
        "dyad_type": np.array([d[2] for d in dy], np.int8), "dyad_a": np.array([d[0] for d in dy], np.int32),
        "dyad_b": np.array([d[1] for d in dy], np.int32),
        "actor_id": np.arange(len(actors), dtype=np.int64), "actor_name": np.array(actors),
        "actor_kind": np.array(kinds, np.int8), "actor_home": np.array(homes, np.int16),
        "years": years, "asylum_iso3": asylum, "ref_stock": ref_stock, "ref_flow": np.zeros_like(ref_stock),
        "idp_stock": idp, "idmc_stock": np.full(idp.shape, np.nan, np.float32), "pop_years": pop_years, "pop": pop,
        "pop_first_projected": np.array(int(years[-1]) + 1),
    }


def panel(**kw):
    from world4.core.l2 import panel as pn
    return pn.from_arrays(world(**kw))
