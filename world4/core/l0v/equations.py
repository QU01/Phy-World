"""Ecuaciones de esperanza de vida y fecundidad por grupo de países, estimadas con datos (fase V2).

Las tablas de World3 no sirven por grupos: evaluadas con el ingreso de un grupo pobre dan 30 años de esperanza de
vida en África subsahariana (real: 62), y con la economía real se saturan (medido, ``docs/V2-informe.md``). Aquí
se sustituyen por dos ecuaciones logísticas, acotadas, estimadas con el panel de PAÍSES de
``world4.data.world`` (efectos fijos de país, ponderado por población, datos <= ``cutoff``):

    logit[(e0 - 20) / (92 - 20)]      = u + b_y · ln(ingreso por persona) + b_k · ln(alimento por persona) + b_t · t
    logit[(TFR - 1,0) / (8,5 - TFR)]  = v + c_e · e0 + c_t · t

* La esperanza de vida sube con el ingreso, con el alimento y con el tiempo (la técnica sanitaria que se difunde).
* La fecundidad baja con la esperanza de vida (sobreviven más hijos) y con el tiempo; el ingreso NO entra: las
  formas con ingreso predicen de más la caída posterior a 2000 (``selection``).
* En el modelo las ecuaciones van ANCLADAS: cada grupo parte de su valor observado en el año de arranque y la
  ecuación sólo da el cambio desde entonces. Así no importan las unidades de World3 (sólo entran cocientes).
**Reglas para lo que los datos no cubren** (``Rules``; son decisiones declaradas, no estimaciones). En la muestra
la esperanza de vida casi siempre sube y los servicios casi nunca caen, así que las ecuaciones no dicen qué pasa en
una escasez. Sin reglas, con la industria un 84 % menor la esperanza de vida seguía subiendo hasta 85 años, y
cuando el alimento faltaba la fecundidad volvía a 5 hijos (medido en V2 y V3). Las reglas:

1. **Alimento.** Por debajo de la subsistencia manda la tabla de alimento de World3 (``lmf``).
2. **Contaminación.** Multiplica con la tabla de World3 (``lmp``).
3. **Servicios de salud.** Si los servicios por persona de un grupo caen por debajo de los que tenía en el origen,
   su esperanza de vida se multiplica por la tabla de salud de World3 (``lmhs2``, con su retardo de 20 años)
   relativa a su valor en el origen. Por encima no actúa: ahí mandan los datos.
4. **El progreso necesita servicios.** El término del tiempo de las dos ecuaciones (la técnica sanitaria y la
   difusión de la baja fecundidad) avanza a un ritmo proporcional a ``min(1, servicios / servicios en el origen)``:
   con los servicios a la mitad, el progreso va a la mitad.
5. **La fecundidad no rebota por una crisis.** En la ecuación de la fecundidad entra la esperanza de vida MÁXIMA
   alcanzada desde el origen, no la del momento.
6. **Suelo de la fecundidad: 1,0 hijos por mujer**, el valor regional más bajo observado (1,02 en Asia oriental
   en 2023). Techo de la esperanza de vida: 92 años.

Las reglas 3 a 5 no cambian nada mientras los servicios y la esperanza de vida suban, que es lo que pasa en todas
las pruebas con datos; sólo deciden las proyecciones con escasez.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

E_LO, E_HI = 20.0, 92.0
F_LO, F_HI = 1.0, 8.5


@dataclass(frozen=True)
class Rules:
    health_table: bool = True             # regla 3
    progress_needs_services: bool = True  # regla 4
    fertility_ratchet: bool = True        # regla 5


NO_RULES = Rules(False, False, False)
E0_TERMS = ("income", "food", "time")
TFR_TERMS = ("e0", "time")
ZERO = {"e0": {k: 0.0 for k in E0_TERMS}, "tfr": {k: 0.0 for k in TFR_TERMS}}        # congelar las tasas


def e0_z(e0):
    e0 = np.clip(e0, E_LO + 0.5, E_HI - 0.5)
    return np.log((e0 - E_LO) / (E_HI - e0))


def e0_of(z):
    return E_LO + (E_HI - E_LO) / (1 + np.exp(-z))


def tfr_z(tfr, lo: float = F_LO):
    tfr = np.clip(tfr, lo + 0.03, F_HI - 0.05)
    return np.log((tfr - lo) / (F_HI - tfr))


def tfr_of(z, lo: float = F_LO):
    return lo + (F_HI - lo) / (1 + np.exp(-z))


def _country_panel():
    """Regresores por país y año: ``{"income", "food", "time", "e0", "tfr", "weight"}``, todos ``(C, Y)``."""
    from world4.data import world, wpp
    d, c = wpp.load(), world.load_countries()
    ptot = (np.nansum(d["popM"], 2, dtype=float) + np.nansum(d["popF"], 2, dtype=float)) * 1e3
    pmid = ptot.copy()
    pmid[:, 1:] = 0.5 * (ptot[:, 1:] + ptot[:, :-1])
    gdp = c["gdp"].copy()
    gdp[:, -1] = np.where(np.isfinite(gdp[:, -1]), gdp[:, -1], gdp[:, -2])          # 2023: el PIB de 2022
    with np.errstate(invalid="ignore", divide="ignore"):
        income, food = np.log(gdp / pmid), np.log(c["kcal"])
    time = np.broadcast_to((c["years"] - c["years"][0]).astype(float), income.shape)
    return {"years": c["years"], "income": income, "food": food, "time": time, "e0": c["e0"], "tfr": c["tfr"],
            "weight": pmid}


def _within(X, y, w):
    """Mínimos cuadrados con efectos fijos de país, ponderados: ``X`` (C, T, k), ``y`` y ``w`` (C, T)."""
    ok = np.isfinite(X).all(2) & np.isfinite(y)
    xs, ys = [], []
    for i in range(X.shape[0]):
        m = ok[i]
        if m.sum() < 10:
            continue
        s = np.sqrt(w[i][m])
        xs.append((X[i][m] - X[i][m].mean(0)) * s[:, None])
        ys.append((y[i][m] - y[i][m].mean()) * s)
    return np.linalg.lstsq(np.concatenate(xs), np.concatenate(ys), rcond=None)[0]


def estimate(cutoff: int | None = None, e0_terms=E0_TERMS, tfr_terms=TFR_TERMS, floor: float = F_LO) -> dict:
    """Coeficientes de las dos ecuaciones con datos <= ``cutoff`` (None: todos)."""
    p = _country_panel()
    n = len(p["years"]) if cutoff is None else int(cutoff - p["years"][0] + 1)
    cut = lambda a: a[:, :n]  # noqa: E731
    be = _within(np.stack([cut(p[k]) for k in e0_terms], 2), cut(e0_z(p["e0"])), cut(p["weight"]))
    bf = _within(np.stack([cut(p[k]) for k in tfr_terms], 2), cut(tfr_z(p["tfr"], floor)), cut(p["weight"]))
    return {"cutoff": int(p["years"][n - 1]), "floor": float(floor),
            "e0": {**{k: 0.0 for k in E0_TERMS}, **dict(zip(e0_terms, map(float, be)))},
            "tfr": {**{k: 0.0 for k in TFR_TERMS + ("income",)}, **dict(zip(tfr_terms, map(float, bf)))}}


@lru_cache(maxsize=8)
def params(cutoff: int | None = None) -> dict:
    """Las ecuaciones del modelo, estimadas con datos <= ``cutoff`` (en caché)."""
    return estimate(cutoff)


class EstimatedSector:
    """Las dos ecuaciones para ``G`` grupos, ancladas en lo observado en el arranque, con las reglas de ``Rules``."""

    def __init__(self, G: int, par: dict, constants: dict, tables: dict, dt: float = 0.5, rules: Rules = Rules()):
        from world4.core import l0
        self.G, self.par, self.dt, self.rules = G, par, dt, rules
        self.floor = float(par.get("floor", F_LO))
        const = lambda k: float(np.asarray(constants[k], float).reshape(-1)[0])  # noqa: E731
        self.sfpc, self.hsid = const("sfpc"), const("hsid")
        self.tab = {k: l0._grid(k, tables[k]) for k in ("lmf", "lmp", "hsapc", "lmhs2")}
        self.ref = None

    def _health(self, effective):
        return np.interp(effective, *self.tab["lmhs2"])

    def anchor(self, t, income, food, services, ppolx, e0, tfr):
        """Fija el punto de partida: los regresores en ``t`` y la e0 y la fecundidad observadas de cada grupo."""
        e0 = np.asarray(e0, float)
        services = np.asarray(services, float)
        self.ehspc = np.interp(services, *self.tab["hsapc"])               # servicios de salud efectivos (retardados)
        self.ref = {"t": float(t), "income": np.log(income), "food": np.log(food), "services": services,
                    "lmp": float(np.interp(ppolx, *self.tab["lmp"])), "health": self._health(self.ehspc),
                    "ze": e0_z(e0), "zf": tfr_z(np.asarray(tfr, float), self.floor), "e0": e0}
        self.clock = np.zeros(self.G)                                      # años de progreso desde el origen
        self.last_t = float(t)
        self.e0_max = e0.copy()

    def step(self, t, income, food, services, ppolx):
        """``(e0, TFR)`` de cada grupo en ``t``. ``income``, ``food`` y ``services``: por persona, en las unidades
        del modelo (sólo entran cocientes con el origen)."""
        r, pe, pf, rules = self.ref, self.par["e0"], self.par["tfr"], self.rules
        services = np.asarray(services, float)
        step = t - self.last_t
        self.last_t = float(t)
        pace = np.minimum(1.0, services / r["services"]) if rules.progress_needs_services else 1.0
        self.clock = self.clock + pace * step
        dy, dk = np.log(income) - r["income"], np.log(food) - r["food"]
        base = e0_of(r["ze"] + pe["income"] * dy + pe["food"] * dk + pe["time"] * self.clock)
        famine = np.interp(np.minimum(np.asarray(food, float) / self.sfpc, 1.0), *self.tab["lmf"])   # 1 si hay subsistencia
        health = 1.0
        if rules.health_table:
            self.ehspc = self.ehspc + step * (np.interp(services, *self.tab["hsapc"]) - self.ehspc) / self.hsid
            health = np.minimum(1.0, self._health(self.ehspc) / r["health"])
        e0 = base * famine * health * float(np.interp(ppolx, *self.tab["lmp"])) / r["lmp"]
        self.e0_max = np.maximum(self.e0_max, e0)
        seen = self.e0_max if rules.fertility_ratchet else e0
        tfr = tfr_of(r["zf"] + pf["e0"] * (seen - r["e0"]) + pf["time"] * self.clock + pf.get("income", 0.0) * dy,
                     self.floor)
        return e0, tfr


# ------------------------------------------------------------------ evidencia de la elección de forma
E0_SPECS = {"income": ("income",), "income+time": ("income", "time"), "income+food+time": E0_TERMS}
TFR_SPECS = {"e0": ("e0",), "income": ("income",), "income+e0": ("income", "e0"), "e0+time": TFR_TERMS,
             "income+e0+time": ("income", "e0", "time")}


def selection(kind: str = "regions", cutoffs=(1990, 2000), floors=(0.8, 1.0, 1.3, 1.5)) -> dict:
    """Error fuera de muestra (``cutoff + 1`` a 2023) de cada forma candidata, aplicada a los grupos ``kind`` con
    sus regresores OBSERVADOS y anclada en ``cutoff``, frente a congelar el último valor y a prolongar la tendencia
    de los diez años anteriores. Esperanza de vida: error de la media mundial (%). Fecundidad: error de los
    nacimientos mundiales (%), con las mujeres de 15 a 49 observadas.

    Es la tabla con la que se ELIGIÓ la forma de ``estimate``: mirar el periodo de prueba para elegir hace
    optimista el error de la forma elegida [declarado]."""
    from world4.core.l0v import groups
    gi, dr = groups.inputs(kind), groups.world_drivers()
    Y = len(gi.years)
    pop = gi.popM.sum(2) + gi.popF.sum(2)
    x = {"income": np.log(gi.rho) + np.log(dr["gdp_pc"])[:, None], "food": np.log(gi.phi) + np.log(dr["kcal"])[:, None],
         "time": np.repeat((gi.years - gi.years[0]).astype(float)[:, None], gi.G, 1), "e0": gi.e0}
    women = gi.popF[:, :, 15:50].sum(2)
    out = {"groups": kind, "e0": {}, "tfr": {}}
    for cut in cutoffs:
        i = int(cut - gi.years[0])
        h = np.arange(Y - i)[:, None]
        mean = lambda v: (v * pop[i:]).sum(1) / pop[i:].sum(1)  # noqa: E731
        births = lambda v: (v * women[i:]).sum(1)  # noqa: E731
        err = lambda f, v, real: float(100 * np.abs(f(v) / f(real) - 1)[1:].mean())  # noqa: E731
        res = {"frozen": err(mean, np.repeat(gi.e0[i][None], Y - i, 0), gi.e0[i:]),
               "trend": err(mean, gi.e0[i] + (gi.e0[i] - gi.e0[i - 10]) / 10 * h, gi.e0[i:])}
        for name, terms in E0_SPECS.items():
            b = estimate(cut, e0_terms=terms)["e0"]
            res[name] = err(mean, e0_of(e0_z(gi.e0[i]) + sum(b[k] * (x[k][i:] - x[k][i]) for k in terms)), gi.e0[i:])
        out["e0"][cut] = res
        for lo in floors:
            z0, zm = tfr_z(gi.tfr[i], lo), tfr_z(gi.tfr[i - 10], lo)
            res = {"frozen": err(births, np.repeat(gi.tfr[i][None], Y - i, 0), gi.tfr[i:]),
                   "trend": err(births, tfr_of(z0 + (z0 - zm) / 10 * h, lo), gi.tfr[i:])}
            for name, terms in TFR_SPECS.items():
                b = estimate(cut, tfr_terms=terms, floor=lo)["tfr"]
                res[name] = err(births, tfr_of(z0 + sum(b[k] * (x[k][i:] - x[k][i]) for k in terms), lo), gi.tfr[i:])
            out["tfr"][(cut, lo)] = res
    return out
