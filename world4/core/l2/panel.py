"""El panel del L2: sitios, meses, díadas y actores, y los rasgos que comparten la vara y el simulador.

Un **sitio** es un par (celda, país). Hace falta porque a 0,5° una celda de frontera recibe eventos de dos
países (el 6 % de los eventos localizados de Oriente Medio ocurren en un país que no es el de su celda), y
porque los eventos que UCDP sólo sitúa en el país necesitan un lugar: cada país tiene un sitio «sin celda».
Las celdas se obtienen sumando sitios por celda y los países sumando por país, así que los tres niveles son
coherentes por construcción.

Todo lo que aquí depende del tiempo se calcula con datos ESTRICTAMENTE anteriores al mes que describe:
``lagged(x)[:, t]`` sólo usa ``x[:, :t]``. ``Panel.truncated(t0)`` borra lo ocurrido desde ``t0``; un
pronóstico hecho sobre el panel truncado debe ser idéntico al hecho sobre el completo (prueba de fuga).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TYPES = ("state_based", "non_state", "one_sided")          # type_of_violence 1, 2, 3 de UCDP
WINDOWS = (1, 3, 12, 36)                                    # medias móviles de los rasgos (meses)
COV_USED = ("capdist", "bdist1", "mountains_mean", "ttime_mean", "urban_gc", "nlights_calib_mean",
            "excluded", "barren_gc")
TSL_CAP = 120.0                                             # «meses desde el último evento» se satura aquí


@dataclass
class Panel:
    region: str
    iso3: tuple
    year0: int
    T: int
    last_final_month: int                                    # último mes del GED final; después, candidato
    # celdas
    gid: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    cell_country: np.ndarray                                 # (C,) país de la celda o -1
    nbr: np.ndarray                                          # (C, 8) celdas vecinas o -1
    cov: np.ndarray                                          # (C, len(COV_USED)) estandarizadas, sin NaN
    pop_cell: np.ndarray                                     # (C, Yp) personas
    # sitios
    site_cell: np.ndarray                                    # (S,) celda o -1 (sin localizar)
    site_country: np.ndarray                                 # (S,)
    site_nbr: np.ndarray                                     # (S, 8) sitio de cada celda vecina en el mismo país o -1
    # eventos (E,)
    ev: dict = field(default_factory=dict)
    # díadas y actores
    dyad_name: np.ndarray = None
    dyad_type: np.ndarray = None                             # 1, 2, 3
    dyad_a: np.ndarray = None
    dyad_b: np.ndarray = None
    actor_name: np.ndarray = None
    actor_kind: np.ndarray = None                            # 0 gobierno, 1 no estatal, 2 civiles
    actor_home: np.ndarray = None
    # anuales
    years: np.ndarray = None
    pop_years: np.ndarray = None
    pop: np.ndarray = None                                   # (K, Yp)
    asylum_iso3: np.ndarray = None
    ref_stock: np.ndarray = None                             # (K, J, Y)
    ref_flow: np.ndarray = None
    idp_stock: np.ndarray = None                             # (K, Y) máx(ACNUR, IDMC)
    raw: dict = field(default=None, repr=False)              # los arrays de origen (para ``truncated``)
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ tamaños y fechas
    @property
    def K(self) -> int:
        return len(self.iso3)

    @property
    def C(self) -> int:
        return len(self.gid)

    @property
    def S(self) -> int:
        return len(self.site_cell)

    @property
    def D(self) -> int:
        return len(self.dyad_type)

    def month(self, year: int, month: int = 1) -> int:
        return (int(year) - self.year0) * 12 + int(month) - 1

    def ym(self, t: int) -> tuple[int, int]:
        return self.year0 + int(t) // 12, int(t) % 12 + 1

    def label(self, t: int) -> str:
        y, m = self.ym(t)
        return f"{y}-{m:02d}"

    # ------------------------------------------------------------------ tensores densos
    def _spread(self, index: np.ndarray, n_index: int, value: np.ndarray, mask=None) -> np.ndarray:
        """(n_index, T): cada evento reparte ``value`` a partes iguales entre los meses que abarca."""
        e = self.ev
        out = np.zeros((n_index, self.T), np.float32)          # float32: a escala mundial son muchas series
        ok = (index >= 0) if mask is None else (index >= 0) & mask
        t0, t1 = e["t0"][ok], e["t1"][ok]
        idx, v = index[ok], value[ok].astype(float)
        span = t1 - t0 + 1
        one = span == 1
        np.add.at(out, (idx[one], t0[one]), v[one])
        for i in np.where(~one)[0]:
            out[idx[i], t0[i]:t1[i] + 1] += v[i] / span[i]
        return out

    def site_deaths(self, kind: str = "all") -> np.ndarray:
        """(S, T) muertes (``best``) por sitio y mes. ``kind``: all, state_based, non_state, one_sided, civ."""
        key = ("site_deaths", kind)
        if key not in self._cache:
            e = self.ev
            if kind == "civ":
                self._cache[key] = self._spread(e["site"], self.S, e["civ"])
            else:
                mask = None if kind == "all" else e["type"] == TYPES.index(kind) + 1
                self._cache[key] = self._spread(e["site"], self.S, e["best"], mask)
        return self._cache[key]

    def site_events(self) -> np.ndarray:
        if "site_events" not in self._cache:
            self._cache["site_events"] = self._spread(self.ev["site"], self.S, np.ones(len(self.ev["site"])))
        return self._cache["site_events"]

    def to_cells(self, x: np.ndarray) -> np.ndarray:
        """Suma (..., S, T) sobre los sitios de cada celda -> (..., C, T); los sitios sin celda no cuentan."""
        ok = self.site_cell >= 0
        return _group_sum(x[..., ok, :], self.site_cell[ok], self.C)

    def to_countries(self, x: np.ndarray) -> np.ndarray:
        return _group_sum(x, self.site_country, self.K)

    def deaths(self, level: str, kind: str = "all") -> np.ndarray:
        """(U, T) con ``level`` cell o country."""
        key = ("deaths", level, kind)
        if key not in self._cache:
            s = self.site_deaths(kind)
            self._cache[key] = self.to_cells(s) if level == "cell" else self.to_countries(s)
        return self._cache[key]

    def dyad_series(self, what: str = "events") -> np.ndarray:
        """(D, T): eventos, muertes o muertes de civiles de cada díada por mes."""
        key = ("dyad", what)
        if key not in self._cache:
            e = self.ev
            v = {"events": np.ones(len(e["dyad"])), "deaths": e["best"], "civ": e["civ"]}[what]
            self._cache[key] = self._spread(e["dyad"], self.D, v)
        return self._cache[key]

    def pop_at(self, t: int, level: str = "country") -> np.ndarray:
        j = int(np.clip(self.ym(t)[0] - int(self.pop_years[0]), 0, len(self.pop_years) - 1))
        return self.pop[:, j] if level == "country" else self.pop_cell[:, j]

    # ------------------------------------------------------------------ corte «a fecha de»
    def truncated(self, t0: int) -> "Panel":
        """Panel reconstruido sin nada de lo ocurrido desde el mes ``t0``: ni eventos, ni desplazamiento de ese
        año en adelante, ni los sitios que sólo existen por un evento posterior. Un evento que empezó antes
        pero termina en ``t0`` o después tampoco está: UCDP lo registra cuando acaba, y sus muertes incluyen
        meses que aún no han pasado. La población proyectada se conserva (WPP la publica por adelantado)."""
        d = dict(self.raw)
        keep = d["ev_t1"] < t0
        for k in [k for k in d if k.startswith("ev_")]:
            d[k] = d[k][keep]
        cut = d["years"] >= self.ym(t0)[0]
        for k in ("ref_stock", "ref_flow", "idp_stock", "idmc_stock"):
            a = d[k].astype(float).copy()
            a[..., cut] = np.nan
            d[k] = a
        d["T"] = np.array(int(t0))
        d["last_final_month"] = np.array(min(self.last_final_month, int(t0) - 1))
        return from_arrays(d)


def _group_sum(x: np.ndarray, group: np.ndarray, n: int) -> np.ndarray:
    """Suma el penúltimo eje de ``x`` (..., S, T) por grupo -> (..., n, T)."""
    order = np.argsort(group, kind="stable")
    g = group[order]
    present, start = np.unique(g, return_index=True)
    out = np.zeros(x.shape[:-2] + (n,) + x.shape[-1:], x.dtype if x.dtype.kind == "f" else float)
    if len(g):
        out[..., present, :] = np.add.reduceat(np.take(x, order, axis=-2), start, axis=-2)
    return out


# ---------------------------------------------------------------------- construcción
def _neighbours(row, col) -> np.ndarray:
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    out = np.full((len(row), 8), -1, np.int32)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i, (r, c) in enumerate(zip(row, col)):
        for j, (dr, dc) in enumerate(offs):
            out[i, j] = pos.get((int(r) + dr, int(c) + dc), -1)
    return out


def _standardise(cov: np.ndarray, names, used=COV_USED) -> np.ndarray:
    """Covariables de celda: log1p en las de escala (distancias, tiempos, luces), NaN -> mediana, z-score."""
    names = list(names)
    out = []
    for n in used:
        x = cov[:, names.index(n)].astype(float)
        if n in ("capdist", "bdist1", "ttime_mean", "nlights_calib_mean"):
            x = np.log1p(np.maximum(x, 0))
        if n == "excluded":
            x = np.where(np.isfinite(x), x, 0.0)             # sin grupo excluido registrado
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        sd = x.std()
        out.append((x - x.mean()) / (sd if sd > 0 else 1.0))
    return np.stack(out, 1)


def from_arrays(d: dict) -> Panel:
    """El panel a partir de los arrays de ``data.conflict`` (o de un mundo sintético con las mismas claves)."""
    K = len(d["iso3"])
    C = len(d["gid"])
    cc = d["cell_country"].astype(int)
    ev_cell, ev_country = d["ev_cell"].astype(int), d["ev_country"].astype(int)
    # sitios: el de cada celda con su país, los pares (celda, país) vistos en eventos y uno sin celda por país
    pairs = {(c, int(cc[c])) for c in range(C) if cc[c] >= 0}
    pairs |= {(int(c), int(k)) for c, k in zip(ev_cell, ev_country) if c >= 0}
    pairs = sorted(pairs) + [(-1, k) for k in range(K)]
    sidx = {p: i for i, p in enumerate(pairs)}
    site_cell = np.array([p[0] for p in pairs], np.int32)
    site_country = np.array([p[1] for p in pairs], np.int32)
    nbr = _neighbours(d["row"], d["col"])
    # vecinos de un sitio: los sitios de las celdas vecinas EN EL MISMO PAÍS. La huella de una díada no se
    # difunde a través de una frontera; a otro país sólo llega por eventos observados o por una díada nueva.
    site_nbr = np.full((len(pairs), 8), -1, np.int32)
    for i, (c, k) in enumerate(pairs):
        if c >= 0:
            site_nbr[i] = [sidx.get((int(c2), k), -1) if c2 >= 0 else -1 for c2 in nbr[c]]
    ev_site = np.array([sidx[(int(c), int(k))] for c, k in zip(ev_cell, ev_country)], np.int32)

    # población por celda: reparto de GPW (el último año disponible) dentro del país, escalado al total de WPP
    gp = d["pop_gpw"].astype(float)
    last = np.where(np.isfinite(gp[:, -1]), gp[:, -1], np.nanmax(np.nan_to_num(gp), 1))
    last = np.nan_to_num(last)
    pop_cell = np.zeros((C, d["pop"].shape[1]))
    for k in range(K):
        m = cc == k
        tot = last[m].sum()
        share = last[m] / tot if tot > 0 else np.full(m.sum(), 1.0 / max(m.sum(), 1))
        pop_cell[m] = share[:, None] * d["pop"][k][None]
    pop_cell[cc < 0] = last[cc < 0][:, None]

    idp = np.fmax(d["idp_stock"].astype(float), d["idmc_stock"].astype(float))
    ev = {"t0": d["ev_t0"].astype(int), "t1": d["ev_t1"].astype(int), "site": ev_site, "cell": ev_cell,
          "country": ev_country, "type": d["ev_type"].astype(int), "dyad": d["ev_dyad"].astype(int),
          "best": d["ev_best"].astype(float), "low": d["ev_low"].astype(float), "high": d["ev_high"].astype(float),
          "civ": d["ev_civ"].astype(float), "candidate": d["ev_candidate"].astype(int)}
    return Panel(
        region=str(d["region"]), iso3=tuple(str(x) for x in d["iso3"]), year0=int(d["year0"]), T=int(d["T"]),
        last_final_month=int(d["last_final_month"]), gid=d["gid"], lon=d["lon"], lat=d["lat"], cell_country=cc,
        nbr=nbr, cov=_standardise(d["cov"], d["cov_names"]), pop_cell=pop_cell, site_cell=site_cell,
        site_country=site_country, site_nbr=site_nbr, ev=ev, dyad_name=d["dyad_name"],
        dyad_type=d["dyad_type"].astype(int), dyad_a=d["dyad_a"].astype(int), dyad_b=d["dyad_b"].astype(int),
        actor_name=d["actor_name"], actor_kind=d["actor_kind"].astype(int), actor_home=d["actor_home"].astype(int),
        years=d["years"].astype(int), pop_years=d["pop_years"].astype(int), pop=d["pop"].astype(float),
        asylum_iso3=d["asylum_iso3"], ref_stock=d["ref_stock"].astype(float), ref_flow=d["ref_flow"].astype(float),
        idp_stock=idp, raw=d)


_PANELS: dict = {}


def load(region: str = "mideast") -> Panel:
    if region not in _PANELS:
        from world4.data import conflict
        _PANELS[region] = from_arrays(conflict.load(region))
    return _PANELS[region]


# ---------------------------------------------------------------------- rasgos con retardo
def lagged_mean(x: np.ndarray, k: int, cols=None) -> np.ndarray:
    """(U, T+1): la columna ``t`` es la media de ``x[:, t-k:t]`` (con menos meses al principio, la de los que hay).
    La columna ``T`` describe el mes siguiente al último dato: es la que usa un pronóstico vivo. Con ``cols``
    sólo se devuelven esas columnas (U, len(cols)): a escala mundial no caben todas."""
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1, dtype=np.float64)], -1)
    t = np.arange(x.shape[-1] + 1) if cols is None else np.asarray(cols)
    lo = np.maximum(t - k, 0)
    return (cs[..., t] - cs[..., lo]) / np.maximum(t - lo, 1)


def months_since(x: np.ndarray, cap: float = TSL_CAP, cols=None) -> np.ndarray:
    """(U, T+1): meses desde el último mes con ``x > 0`` ANTES de ``t`` (``cap`` si nunca o hace más)."""
    U, T = x.shape
    out = np.full((U, T + 1), cap, np.float32)
    last = np.full(U, -np.inf)
    for t in range(1, T + 1):
        last = np.where(x[:, t - 1] > 0, t - 1, last)
        out[:, t] = np.minimum(t - last, cap)
    return out if cols is None else out[:, np.asarray(cols)]


def neighbour_sum(x: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Suma de ``x`` (U, ...) sobre los vecinos de cada unidad (los -1 no cuentan)."""
    ok = nbr >= 0
    g = x[np.maximum(nbr, 0)]                                 # (U, 8, ...)
    return (g * ok.reshape(ok.shape + (1,) * (x.ndim - 1))).sum(1)
