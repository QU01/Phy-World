"""World3-03 con la población por cohortes (fase 1 del L0 vigente).

El acople usa el gancho de ``world4.core.l0.run`` y no toca sus ecuaciones: World3 sigue calculando la esperanza
de vida ``le`` y la fecundidad total ``tf`` con sus tablas, y las cohortes las convierten en muertes y
nacimientos por edad; en cada paso las cuatro cajas de World3 (``p1``..``p4``) se SUSTITUYEN por la suma de las
cohortes (``"set"`` del gancho). Lo demás de World3 (fuerza laboral, alimento por persona, ...) lee esas cajas.

Arranque: hasta ``start`` corre el World3 original; al final del año ``start`` (``t = start + 0,5``) la
población pasa a ser la observada por edad simple y sexo (UN WPP 2024, que da la población a 31 de diciembre).
Con ``start = 1950`` (el primer año de WPP) la población mundial baja un 7 % en ese salto: World3 llegaba a 1950
con más gente de la que había.

Diagnóstico: ``fertility="observed"`` o ``mortality="observed"`` imponen a las cohortes la fecundidad total o
la esperanza de vida REALES (hasta 2023; después, las del modelo). Con las dos se mide el motor de cohortes
solo; con una, de qué ecuación de World3 viene el error. Las series ``tf`` y ``le`` de la corrida siguen siendo
las de World3; las que usaron las cohortes están en ``CohortRun``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world4.core.l0v import lifetable
from world4.core.l0v.cohorts import A, Cohorts
from world4.sectors.population import asfr_single

EPS = 1e-9


@dataclass
class WorldInputs:
    """Lo que las cohortes leen de los datos: todo del mundo, con su año."""
    years: np.ndarray            # años de WPP (población a 31 de diciembre)
    popM: np.ndarray             # (años, 101) personas
    popF: np.ndarray
    asfr_t: np.ndarray           # tiempos de los patrones de fecundidad
    asfr: np.ndarray             # (periodos, 101) fracción por edad simple, suma 1
    srb_years: np.ndarray
    srb: np.ndarray
    obs_years: np.ndarray        # años de las series observadas
    tfr: np.ndarray
    le: np.ndarray
    family: lifetable.Family


def world_inputs() -> WorldInputs:
    from world4.data import world, wpp
    d, w = wpp.load(), world.load()
    ok = np.isfinite(w["tfr"]) & np.isfinite(w["le"])
    return WorldInputs(
        years=d["years"], popM=np.nansum(d["popM"], 0, dtype=float) * 1e3,
        popF=np.nansum(d["popF"], 0, dtype=float) * 1e3,
        asfr_t=w["asfr_periods"] + 0.5, asfr=asfr_single(w["asfr7"]),
        srb_years=w["srb_years"], srb=w["srb"],
        obs_years=w["years"][ok], tfr=w["tfr"][ok], le=w["le"][ok], family=lifetable.wpp_world_family())


@dataclass
class CohortRun:
    time: np.ndarray                                   # tiempos en los que mandan las cohortes
    series: dict = field(default_factory=dict)         # births, deaths (personas/año), tfr, le usadas: (T, n)
    years: np.ndarray = None                           # años con foto por edad (a 31 de diciembre)
    popM: np.ndarray = None                            # (años, n, 101)
    popF: np.ndarray = None

    def __getitem__(self, k):
        return self.series[k]


class CohortCoupler:
    """El gancho de ``l0.run``: lleva las cohortes y devuelve las cuatro cajas de World3."""

    def __init__(self, n: int = 1, *, start: int = 1950, dt: float = 0.5, fertility: str = "model",
                 mortality: str = "model", inputs: WorldInputs | None = None):
        for name, v in (("fertility", fertility), ("mortality", mortality)):
            if v not in ("model", "observed"):
                raise ValueError(f"{name} must be 'model' or 'observed', got {v!r}")
        self.inp = inputs or world_inputs()
        if start not in self.inp.years:
            raise ValueError(f"start must be a WPP year ({self.inp.years[0]}-{self.inp.years[-1]}), got {start}")
        self.n, self.dt, self.start, self.t0 = n, dt, int(start), start + 0.5
        self.fertility, self.mortality = fertility, mortality
        self.c: Cohorts | None = None
        self.rec = {k: [] for k in ("time", "births", "deaths", "tfr", "le")}
        self.snap = {"years": [], "popM": [], "popF": []}

    def _observed(self, t, name, model):
        y = self.inp.obs_years
        if t > y[-1] or t < y[0]:
            return model
        return np.full(self.n, np.interp(t, y, getattr(self.inp, name)))

    def _snapshot(self, t):
        if abs((t - 0.5) - round(t - 0.5)) < EPS:                 # t = año + 0,5: fin de ese año
            b = self.c.by_age()
            self.snap["years"].append(int(round(t - 0.5)))
            self.snap["popM"].append(b[:, 0].copy())
            self.snap["popF"].append(b[:, 1].copy())

    def __call__(self, t, st, a):
        if t < self.t0 - self.dt - EPS:
            return None
        if self.c is None:                                        # el paso que termina en t0: entra el dato
            i = int(np.where(self.inp.years == self.start)[0][0])
            self.c = Cohorts(self.inp.popM[i], self.inp.popF[i], dt=self.dt, n=self.n)
            return {"set": self.c.groups()}
        self._snapshot(t)
        tfr = self._observed(t, "tfr", a["tf"]) if self.fertility == "observed" else np.asarray(a["tf"], float)
        le = self._observed(t, "le", a["le"]) if self.mortality == "observed" else np.asarray(a["le"], float)
        inp = self.inp
        pattern = np.array([np.interp(t, inp.asfr_t, inp.asfr[:, x]) for x in range(A)])
        pattern = pattern / pattern.sum()
        srb = float(np.interp(t, inp.srb_years, inp.srb))
        births, deaths = self.c.step(inp.family.mx(le), tfr, pattern, srb)
        for k, v in (("time", t), ("births", births / self.dt), ("deaths", deaths / self.dt), ("tfr", tfr), ("le", le)):
            self.rec[k].append(np.array(v, float, copy=True))
        return {"set": self.c.groups()}

    def result(self) -> CohortRun:
        s = {k: np.stack(self.rec[k]) for k in ("births", "deaths", "tfr", "le")}
        return CohortRun(time=np.array(self.rec["time"], float), series=s, years=np.array(self.snap["years"], int),
                         popM=np.stack(self.snap["popM"]), popF=np.stack(self.snap["popF"]))

    def patch(self, run) -> None:
        """Las tasas brutas de la corrida, que World3 calculó con sus cuatro cajas, pasan a ser las de las cohortes."""
        res = self.result()
        idx = np.searchsorted(run.time, res.time - EPS)
        for name, key in (("cbr", "births"), ("cdr", "deaths")):
            if name in run.series and "pop" in run.series:
                run.series[name][idx] = 1000 * res[key] / run.series["pop"][idx]


def run(*, scenario: int = 1, constants: dict | None = None, tables: dict | None = None, t1: float = 2100.0,
        dt: float = 0.5, n: int = 1, start: int = 1950, fertility: str = "model", mortality: str = "model",
        record=None, inputs: WorldInputs | None = None):
    """World3-03 con cohortes desde ``start``. Devuelve ``(World3Run, CohortRun)``."""
    from world4.core import l0
    hook = CohortCoupler(n, start=start, dt=dt, fertility=fertility, mortality=mortality, inputs=inputs)
    r = l0.run(scenario=scenario, constants=constants, tables=tables, n=n, t1=t1, dt=dt,
               record=record or l0.RECORD, hook=hook)
    hook.patch(r)
    return r, hook.result()
