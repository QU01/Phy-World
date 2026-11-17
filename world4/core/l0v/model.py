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


@dataclass
class GroupRun:
    names: tuple
    time: np.ndarray                                   # tiempos en los que mandan las cohortes
    series: dict = field(default_factory=dict)         # pop, births, deaths, tfr, le, income: (T, G); world_tfr, world_le: (T,)
    years: np.ndarray = None                           # años con foto por edad (a 31 de diciembre)
    popM: np.ndarray = None                            # (años, G, 101)
    popF: np.ndarray = None

    def __getitem__(self, k):
        return self.series[k]


class GroupCoupler:
    """El gancho de ``l0.run`` con el mundo partido en grupos de países (``world4.core.l0v.groups``).

    World3 da el producto (industria + servicios) por persona, el alimento por persona y la contaminación del
    MUNDO. Cada grupo recibe su parte: su ingreso por persona es el del mundo por su ingreso relativo ``rho`` (y
    el alimento por ``phi``), renormalizados con las poblaciones del modelo para que la suma de los grupos sea el
    total de World3. Con eso, las ecuaciones dan la esperanza de vida y la fecundidad de cada grupo, y sus
    cohortes dan nacimientos y muertes. World3 recibe la suma.

    ``origin`` separa la historia del modelo: hasta ``origin`` las cohortes de cada grupo llevan su fecundidad y
    su esperanza de vida OBSERVADAS (la historia es dato; el motor la reproduce con un 1 % de error), al final
    del año ``origin`` la población por edad vuelve a ser la observada, y desde ahí mandan las ecuaciones. Con
    ``origin = start`` las ecuaciones mandan desde el principio.

    ``equations="estimated"`` (por defecto): las ecuaciones estimadas con datos de ``equations.py``, ancladas en
    lo observado en ``origin``; ``par`` son sus coeficientes. ``equations="world3"``: las tablas de World3
    evaluadas por grupo (``popsector``), que corren desde 1900 con el reparto de ``start`` para que sus retardos
    tengan historia; se conservan porque su fallo es un resultado (``docs/V2-informe.md``).

    ``rho`` y ``phi`` son los observados (1950-2022 y 1961-2013); fuera de esos años se quedan en el último
    valor [S: el reparto relativo no cambia]. La migración neta entre grupos es la observada hasta ``origin`` y
    después se queda en la media de los diez años anteriores [S].
    """

    def __init__(self, gi, constants: dict, tables: dict, *, start: int = 1950, dt: float = 0.5,
                 origin: int | None = None, fertility: str = "model", mortality: str = "model",
                 migration: bool = True, equations: str = "estimated", par: dict | None = None,
                 drivers: dict | None = None, rules=None):
        from world4.core.l0v.popsector import PopSector
        from world4.sectors.population import migration_profile
        for name, v in (("fertility", fertility), ("mortality", mortality)):
            if v not in ("model", "observed"):
                raise ValueError(f"{name} must be 'model' or 'observed', got {v!r}")
        if start not in gi.years:
            raise ValueError(f"start must be a WPP year ({gi.years[0]}-{gi.years[-1]}), got {start}")
        self.gi, self.G, self.dt, self.start, self.t0 = gi, gi.G, dt, int(start), start + 0.5
        self.origin = int(start if origin is None else origin)
        if not start <= self.origin <= gi.years[-1]:
            raise ValueError(f"origin must be between start ({start}) and {gi.years[-1]}, got {origin}")
        self.fertility, self.mortality, self.migration = fertility, mortality, migration
        j = int(self.origin - gi.years[0])
        recent = gi.migration[max(1, j - 9): j + 1]                        # los diez años hasta el origen
        self.mig_after = recent.mean(0) if len(recent) else np.zeros(gi.G)
        if equations not in ("estimated", "world3"):
            raise ValueError(f"equations must be 'estimated' or 'world3', got {equations!r}")
        self.equations, self.drivers = equations, drivers
        if equations == "world3":
            self.sector = PopSector(gi.G, constants, tables, dt)
        else:
            from world4.core.l0v import equations as _eq
            self.sector = _eq.EstimatedSector(gi.G, par or _eq.params(), constants, tables, dt,
                                              rules if rules is not None else _eq.Rules())
        self.profile = migration_profile()
        i = int(start - gi.years[0])
        tot = gi.popM[i].sum(1) + gi.popF[i].sum(1)
        self.share0 = tot / tot.sum()
        self.c: Cohorts | None = None
        self.rec = {k: [] for k in ("time", "pop", "births", "deaths", "tfr", "le", "income", "world_tfr", "world_le")}
        self.snap = {"years": [], "popM": [], "popF": []}

    def _at(self, t, arr):
        """Serie anual ``(Y, G)`` en ``t``, sostenida fuera de sus años."""
        y = self.gi.years
        return np.array([np.interp(t, y, arr[:, g]) for g in range(self.G)])

    def _world_rates(self, mx, tfr, pattern):
        c, inp = self.c, self.gi
        W = c.N[:, 1]
        f = tfr[:, None] * c._along(pattern)
        wtfr = float(((W * f).sum(0) / np.maximum(W.sum(0), 1e-9))[:-1].sum() / c.k)
        by = c.by_age()                                                   # (G, 2, 101)
        pooled = (by * mx).sum(0) / np.maximum(by.sum(0), 1e-9)
        return wtfr, float(lifetable.e0_both(pooled[0], pooled[1]) + np.interp(self._t, inp.offset_t, inp.offset))

    def __call__(self, t, st, a):
        gi = self.gi
        self._t = t
        pops = self.c.total() if self.c is not None else self.share0
        share = pops / pops.sum()
        rho, phi = self._at(t, gi.rho), self._at(t, gi.phi)
        rho, phi = rho / (share * rho).sum(), phi / (share * phi).sum()   # reparto de media 1
        scalar = lambda k: float(np.asarray(a[k], float).reshape(-1)[0])  # noqa: E731
        iopc, sopc, fpc, ppolx = scalar("iopc"), scalar("sopc"), scalar("fpc"), scalar("ppolx")
        ypc = (scalar("io") + scalar("so")) / scalar("pop")
        if self.drivers is not None:                                      # diagnóstico: la economía observada
            iopc, sopc, fpc, ypc = (float(np.interp(t, self.drivers["time"], self.drivers[k]))
                                    for k in ("iopc", "sopc", "fpc", "ypc"))
        world3 = self.equations == "world3"
        if world3:
            le, tf = self.sector.step(t, iopc * rho, sopc * rho, fpc * phi, ppolx, scalar("pop"))
        if t < self.t0 - self.dt - EPS:
            return None
        i0 = int(self.start - gi.years[0])
        anchor_now = not world3 and abs(t - self.origin) < EPS            # las ecuaciones parten de lo observado aquí
        if anchor_now:
            jo = int(self.origin - gi.years[0])
            self.sector.anchor(t, ypc * rho, fpc * phi, sopc * rho, ppolx, gi.e0[jo], gi.tfr[jo])
        if self.c is None:                                                # el paso que termina en t0: entra el dato
            self.c = Cohorts(gi.popM[i0], gi.popF[i0], dt=self.dt, n=self.G)
            return {"set": {k: v.sum() for k, v in self.c.groups().items()}}
        if abs((t - 0.5) - round(t - 0.5)) < EPS:                         # t = año + 0,5: fin de ese año
            b = self.c.by_age()
            self.snap["years"].append(int(round(t - 0.5)))
            self.snap["popM"].append(b[:, 0].copy())
            self.snap["popF"].append(b[:, 1].copy())
        if self.origin > self.start and abs(t - (self.origin + 0.5)) < EPS:   # fin del año origen: el estado es dato
            jo = int(self.origin - gi.years[0])
            self.c = Cohorts(gi.popM[jo], gi.popF[jo], dt=self.dt, n=self.G)
        history = t < self.origin - EPS                                   # la historia es dato
        if history or anchor_now:
            le, tf = self._at(t, gi.e0), self._at(t, gi.tfr)
        elif not world3:
            le, tf = self.sector.step(t, ypc * rho, fpc * phi, sopc * rho, ppolx)
        observed = gi.years[0] <= t <= gi.years[-1]
        if self.fertility == "observed" and observed:
            tf = self._at(t, gi.tfr)
        if self.mortality == "observed" and observed:
            le = self._at(t, gi.e0)
        j = int(np.clip(round(t) - gi.years[0], 0, len(gi.years) - 1))
        pattern, srb = gi.pattern[j], gi.srb[j]
        mx = np.stack([gi.families[g].mx(le[g]) for g in range(self.G)])
        wtfr, wle = self._world_rates(mx, tf, pattern)
        pop_before = self.c.total()
        births, deaths = self.c.step(mx, tf, pattern, srb)
        if self.migration:
            year = int(np.ceil(t + self.dt - 0.5 - EPS))                  # año natural en curso
            net = gi.migration[year - gi.years[0]] if year <= self.origin else self.mig_after
            self.c.migrate(net * self.dt, self.profile)
        for k, v in (("time", t), ("pop", pop_before), ("births", births / self.dt), ("deaths", deaths / self.dt),
                     ("tfr", tf), ("le", le), ("income", rho), ("world_tfr", wtfr), ("world_le", wle)):
            self.rec[k].append(np.array(v, float, copy=True))
        return {"set": {k: v.sum() for k, v in self.c.groups().items()}}

    def result(self) -> GroupRun:
        s = {k: np.stack(self.rec[k]) for k in self.rec if k != "time"}
        return GroupRun(names=self.gi.names, time=np.array(self.rec["time"], float), series=s,
                        years=np.array(self.snap["years"], int), popM=np.stack(self.snap["popM"]),
                        popF=np.stack(self.snap["popF"]))

    def patch(self, run) -> None:
        """Tasas brutas, esperanza de vida y fecundidad total del mundo pasan a ser las de la suma de los grupos."""
        res = self.result()
        idx = np.searchsorted(run.time, res.time - EPS)
        for name, key in (("cbr", "births"), ("cdr", "deaths")):
            if name in run.series and "pop" in run.series:
                run.series[name][idx, 0] = 1000 * res[key].sum(1) / run.series["pop"][idx, 0]
        for name, key in (("le", "world_le"), ("tf", "world_tfr")):
            if name in run.series:
                run.series[name][idx, 0] = res[key]


def observed_economy(base, anchor: int = 1970) -> dict:
    """Las entradas económicas de la demografía si la economía de World3 hubiera seguido a la real: el valor de
    ``base`` (una corrida de World3) en ``anchor`` por el índice observado (PIB por persona para el producto
    industrial, de servicios y total por persona; kilocalorías para el alimento). Antes del primer dato y después
    del último, la corrida ``base`` reescalada para empalmar."""
    from world4.core.l0v import groups as _groups
    dr = _groups.world_drivers()
    t = np.asarray(base.time, float)
    yrs = dr["years"].astype(float)
    ia = int(np.argmin(np.abs(t - anchor)))
    i0 = int(np.argmin(np.abs(t - yrs[0])))
    out = {"time": t}
    i1 = int(np.argmin(np.abs(t - yrs[-1])))
    series = {k: np.asarray(base[k], float)[:, 0] for k in ("iopc", "sopc", "fpc")}
    series["ypc"] = (np.asarray(base["io"], float)[:, 0] + np.asarray(base["so"], float)[:, 0]) \
        / np.asarray(base["pop"], float)[:, 0]
    for key, idx in (("iopc", dr["gdp_pc"]), ("sopc", dr["gdp_pc"]), ("ypc", dr["gdp_pc"]), ("fpc", dr["kcal"])):
        m = series[key]
        obs = m[ia] * np.interp(t, yrs, idx)
        out[key] = np.where(t < yrs[0], m * obs[i0] / m[i0], np.where(t > yrs[-1], m * obs[i1] / m[i1], obs))
    return out


def _eq_params(cutoff):
    from world4.core.l0v import equations
    return equations.params(cutoff)


def run(*, scenario: int = 1, constants: dict | None = None, tables: dict | None = None, t1: float = 2100.0,
        dt: float = 0.5, n: int = 1, start: int = 1950, fertility: str = "model", mortality: str = "model",
        groups=None, origin: int | None = None, migration: bool = True, equations: str = "estimated",
        cutoff: int | None = None, par: dict | None = None, economy: str = "model", technology=None,
        climate=None, rules=None, record=None, inputs: WorldInputs | None = None):
    """World3-03 con cohortes desde ``start``. Devuelve ``(World3Run, CohortRun)``, o ``(World3Run, GroupRun)``
    si ``groups`` nombra una partición (``"regions"``, ``"income"``, ``"world"``) o trae un ``GroupInputs``.

    Con grupos: hasta ``origin`` (por defecto ``start``) las cohortes llevan la fecundidad y la esperanza de vida
    observadas de cada grupo, y desde ``origin`` mandan las ecuaciones; ``equations`` elige las ecuaciones de esperanza de vida y fecundidad (``"estimated"`` o
    ``"world3"``); ``cutoff`` es el último año de datos con el que se estiman (None: todos) y ``par`` las da
    hechas; ``economy="observed"`` mueve la demografía con la economía real en lugar de con la de World3
    (diagnóstico, hasta 2022; después sigue la de World3 reescalada).

    ``technology``: None deja la tecnología de World3 (apagada en el escenario 1); ``"estimated"`` usa las curvas
    de ``world4.core.l0v.technology`` estimadas con datos <= ``cutoff``; también admite un ``Tech``.

    ``climate``: None, ``"estimated"`` o un ``Carbon`` (``world4.core.l0v.carbon``): el calentamiento desde el
    origen que causan las emisiones del modelo resta rendimiento agrícola. Necesita ``technology``.

    ``rules``: las reglas de las ecuaciones estimadas para la escasez (``equations.Rules``; por defecto, todas).

    La corrida devuelta lleva ``origin`` y ``cutoff`` como atributos, para que el marcador sepa qué es historia."""
    from world4.core import l0
    from world4.core.l0v import technology as _tech
    tech = None
    if technology is not None:
        tech = _tech.params(cutoff) if isinstance(technology, str) else technology
        sc_t0 = l0.SCENARIOS[scenario][1]
        tables = {**tech.tables({**l0.TABLES, **sc_t0, **(tables or {})}), **(tables or {})}
        constants = {**tech.constants({**l0.CONSTANTS, **l0.SCENARIOS[scenario][0]}), **(constants or {})}
    carbon_par = None
    if climate is not None:
        from world4.core.l0v import carbon as _carbon
        if tech is None:
            raise ValueError("climate needs technology: the damage acts on the yield technology curve")
        carbon_par = _carbon.params(cutoff) if isinstance(climate, str) else climate
    if groups is None:
        hook = CohortCoupler(n, start=start, dt=dt, fertility=fertility, mortality=mortality, inputs=inputs)
    else:
        if n != 1:
            raise ValueError("groups of countries need n = 1 (one world)")
        from world4.core.l0v import groups as _groups
        gi = _groups.inputs(groups) if isinstance(groups, str) else groups
        sc_c, sc_t = l0.SCENARIOS[scenario]
        if economy not in ("model", "observed"):
            raise ValueError(f"economy must be 'model' or 'observed', got {economy!r}")
        drivers = None
        if economy == "observed":
            drivers = observed_economy(l0.run(scenario=scenario, constants=constants, tables=tables, dt=dt, t1=t1,
                                              record=("iopc", "sopc", "fpc", "io", "so", "pop")))
        hook = GroupCoupler(gi, {**l0.CONSTANTS, **sc_c, **(constants or {})}, {**l0.TABLES, **sc_t, **(tables or {})},
                            start=start, origin=origin, dt=dt, fertility=fertility, mortality=mortality,
                            migration=migration, equations=equations, drivers=drivers, rules=rules,
                            par=par if par is not None or equations == "world3" else _eq_params(cutoff))
    run_origin = None if groups is None else int(start if origin is None else origin)
    tech_hook = None if tech is None else _tech.TechHook(tech, dt, climate=carbon_par, origin=run_origin)
    full = hook if tech is None else _tech.compose(tech_hook, hook)
    r = l0.run(scenario=scenario, constants=constants, tables=tables, n=n, t1=t1, dt=dt,
               record=record or l0.RECORD, hook=full)
    hook.patch(r)
    if tech_hook is not None:
        tech_hook.patch(r)
    r.origin, r.cutoff = run_origin, cutoff
    return r, hook.result()
