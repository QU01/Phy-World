"""Geometría de la malla PRIO-GRID y los eventos de UCDP como matrices dispersas mes x celda."""
from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy import sparse

from world4.data import conflict_grid as cdata

TYPES = cdata.TYPES
STATIC = ("pop", "capdist", "bdist", "ttime", "mountains", "urban", "imr", "excluded", "resources")


def _fill(a, value):
    return np.where(np.isfinite(a), a, value)


class Grid:
    """Las 64 818 celdas terrestres de PRIO-GRID 2.0: vecindades (anillos de Chebyshev 1 y 2, con la
    longitud periódica), país (Gleditsch-Ward, regla de mayoría de PRIO-GRID, 2014) y covariables."""

    def __init__(self, d: dict | None = None):
        d = cdata.load() if d is None else d
        self.gid, self.row, self.col = d["gid"], d["row"], d["col"]
        self.lon, self.lat = d["lon"], d["lat"]
        self.n = len(self.gid)
        R = np.full((360, 720), -1, np.int64)
        R[self.row - 1, self.col - 1] = np.arange(self.n)
        self.raster = R
        gw = _fill(d["pg_gwno"], -1).astype(np.int64)
        self.country_codes, self.country = np.unique(gw, return_inverse=True)
        self.n_countries = len(self.country_codes)
        self.cells_per_country = np.bincount(self.country, minlength=self.n_countries).astype(float)
        res = np.zeros(self.n, bool)
        for k in ("petroleum_s", "diamsec_s", "goldplacer_s", "gem_s"):
            res |= np.isfinite(d[f"pg_{k}"]) & (d[f"pg_{k}"] > 0)
        med = lambda k: np.nanmedian(d[f"pg_{k}"])  # noqa: E731
        self.static = np.column_stack([
            np.log1p(_fill(d["pg_pop_gpw_sum"], 0.0)), np.log1p(d["pg_capdist"]),
            np.log1p(_fill(d["pg_bdist1"], med("bdist1"))), np.log1p(_fill(d["pg_ttime_mean"], med("ttime_mean"))),
            _fill(d["pg_mountains_mean"], 0.0), _fill(d["pg_urban_gc"], 0.0),
            np.log1p(_fill(d["pg_imr_mean"], med("imr_mean"))), _fill(d["pg_excluded"], 0.0), res.astype(float)])

    def _ring(self, r: int) -> sparse.csr_matrix:
        offs = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1) if max(abs(dr), abs(dc)) == r]
        rows, cols = [], []
        for dr, dc in offs:
            rr, cc = self.row - 1 + dr, (self.col - 1 + dc) % 720
            ok = (rr >= 0) & (rr < 360)
            j = np.full(self.n, -1)
            j[ok] = self.raster[rr[ok], cc[ok]]
            ok = j >= 0
            rows.append(np.nonzero(ok)[0])
            cols.append(j[ok])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(self.n, self.n))

    @cached_property
    def ring1(self) -> sparse.csr_matrix:
        """A[i, j] = 1 si j es una de las 8 celdas que rodean a i."""
        return self._ring(1)

    @cached_property
    def ring2(self) -> sparse.csr_matrix:
        """A[i, j] = 1 si j está en el segundo anillo (16 celdas) alrededor de i."""
        return self._ring(2)

    @cached_property
    def country_matrix(self) -> sparse.csr_matrix:
        """(países x celdas): suma por país."""
        return sparse.csr_matrix((np.ones(self.n, np.float32), (self.country, np.arange(self.n))),
                                 shape=(self.n_countries, self.n))

    def distance_to(self, active: np.ndarray, cap: float = 50.0) -> np.ndarray:
        """Distancia (en celdas, euclídea en la malla) a la celda activa más cercana; ``cap`` si no hay
        ninguna. Aproximación declarada: no corrige por la latitud ni cruza el antimeridiano."""
        from scipy import ndimage
        if not active.any():
            return np.full(self.n, cap)
        img = np.ones((360, 720), bool)
        img[self.row[active] - 1, self.col[active] - 1] = False
        dist = ndimage.distance_transform_edt(img)
        return np.minimum(dist[self.row - 1, self.col - 1], cap)

    def africa_middle_east(self) -> np.ndarray:
        """Celdas de los países Gleditsch-Ward 400-699: el dominio de los pronósticos ``pgm`` de ViEWS."""
        gw = self.country_codes[self.country]
        return (gw >= 400) & (gw < 700)


class Events:
    """Eventos de UCDP agregados a (mes, celda) por tipo: conteos y muertes (``best``), como CSR (meses x celdas);
    y conteos por (mes, país) con TODOS los eventos, también los que no tienen celda fiable."""

    def __init__(self, grid: Grid, d: dict | None = None):
        d = cdata.load() if d is None else d
        self.grid = grid
        self.n_months = int(d["n_months"])
        self.last_final_month = int(d["last_final_month"])
        self.versions = tuple(str(v) for v in d["versions"])
        m, c, t, deaths = d["ev_month"], d["ev_cell"], d["ev_type"], d["ev_deaths"].astype(float)
        shape = (self.n_months, grid.n)
        placed = c >= 0
        self.count, self.deaths = {}, {}
        for k, v in enumerate(TYPES):
            s = placed & (t == k)
            self.count[v] = sparse.csr_matrix((np.ones(s.sum()), (m[s], c[s])), shape=shape)
            self.deaths[v] = sparse.csr_matrix((deaths[s], (m[s], c[s])), shape=shape)
            self.count[v].sum_duplicates()
            self.deaths[v].sum_duplicates()
        # país de cada evento (Gleditsch-Ward de UCDP) en el índice de países de la malla
        pos = {int(g): i for i, g in enumerate(grid.country_codes)}
        ci = np.array([pos.get(int(g), -1) for g in d["ev_country"]])
        ok = ci >= 0
        self.country_count = {}
        self.country_deaths = np.zeros((self.n_months, grid.n_countries))
        for k, v in enumerate(TYPES):
            s = ok & (t == k)
            a = np.zeros((self.n_months, grid.n_countries))
            np.add.at(a, (m[s], ci[s]), 1.0)
            self.country_count[v] = a
        np.add.at(self.country_deaths, (m[ok], ci[ok]), deaths[ok])

    def month(self, m: int) -> tuple[dict, dict]:
        """Conteos y muertes densos (n,) por tipo en el mes ``m``."""
        return ({v: self.count[v][m].toarray().ravel() for v in TYPES},
                {v: self.deaths[v][m].toarray().ravel() for v in TYPES})

    def _keys(self, kind: str, v: str):
        cache = self.__dict__.setdefault("_key_cache", {})
        if (kind, v) not in cache:
            A = ((self.count if kind == "count" else self.deaths)[v] if v != "any"
                 else sum((self.count if kind == "count" else self.deaths)[u] for u in TYPES)).tocoo()
            keys = A.row.astype(np.int64) * self.grid.n + A.col
            o = np.argsort(keys)
            cache[(kind, v)] = (keys[o], A.data[o])
        return cache[(kind, v)]

    def lookup(self, kind: str, v: str, months: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Valores en pares (mes, celda) (``v`` = tipo o 'any'); 0 fuera del rango de meses."""
        keys, vals = self._keys(kind, v)
        q = np.asarray(months, np.int64) * self.grid.n + np.asarray(cells, np.int64)
        k = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[k] == q, vals[k], 0.0)

    def window(self, kind: str, v: str, m0: int, m1: int) -> np.ndarray:
        """(m1 - m0, n): meses m0..m1-1 densos (``v`` = tipo o 'any')."""
        src = self.count if kind == "count" else self.deaths
        m0, m1 = max(m0, 0), min(m1, self.n_months)
        if v == "any":
            return sum(src[u][m0:m1].toarray() for u in TYPES)
        return src[v][m0:m1].toarray()
