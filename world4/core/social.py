"""Módulo social de 170+ países acoplado a World3-03 (L0) en las dos direcciones, calibrado con datos.

Es la «ruta 3» del informe ``reports/acople/dinamicas_sociedad_biofisica.pdf``: el L0 regional (ruta 2)
necesita FAOSTAT y Zenodo, que no estaban disponibles. Todo lo que aquí es un número sale de
``social_params.json``, que escribe ``reports/acople/calibrar.py`` con el panel del Atlas 1950-2019 y las
proyecciones de UN WPP 2024; lo que es un supuesto se marca [S].

Ida (World3 → sociedad). El único puente es el ingreso, como pide la evidencia: World3 da el producto
mundial per cápita ``(io + so) / pop`` y cada país lo sigue con su propio ruido (un AR(1) cuyas innovaciones
se remuestrean de los residuos históricos por grupo de ingreso, sin años de evento), de modo que la suma
de los PIB nacionales es, cada año, el PIB de World3 [S: el reparto relativo entre países no tiene tendencia
propia]. La población por país y el peso de los jóvenes son los de WPP (variante media) [S: no responden al
valle de World3]. Con el ingreso, el crecimiento, la profundidad de la caída desde el máximo de 8 años, los
jóvenes, la deuda y la historia propia, nueve modelos logit agrupados dan cada año la probabilidad de:
inicio y fin de conflicto civil, intento de golpe y su éxito, salida irregular del líder, quiebre democrático
y democratización, inicio y fin de crisis de deuda soberana. La deuda / PIB sigue una ecuación estimada.

Vuelta (sociedad → World3). Cada inicio de conflicto, golpe exitoso o crisis de deuda deja en el PIB del
país la huella de su proyección local (Jordà, h = 0..10, se mantiene después) [S: el nivel a 10 años es
permanente]. La pérdida agregada del año (fracción del PIB mundial) destruye la misma fracción del capital
industrial y de servicios de World3 [S: la pérdida de producto se traslada a capital, que en World3 es
proporcional al producto]. Además, la capacidad de implementar política (los flujos de las tecnologías
``lyt``, ``rct`` y ``ppt``) cae con la parte del PIB mundial en crisis de deuda, conflicto o tras un golpe,
en la proporción en que cae el gasto público tras esos eventos [S: gasto ≈ capacidad].

Lo que NO hace, a propósito: no hay canal hambre → guerra ni contaminación → conflicto propios (los datos no
los sostienen una vez controlado el ingreso); el conflicto no cambia la mortalidad de World3 (−0,7 años de
esperanza de vida en el país, despreciable en el agregado); no hay guerras entre Estados.

Uso::

    from world4.core import social
    w3, soc = social.run_coupled(n=100, seed=1)            # World3 + sociedad, con retroalimentación
    w3, soc = social.run_coupled(n=100, feedback=False)    # sólo ida: la sociedad no toca a World3
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).resolve().parents[1] / "data" / "social_params.json"
EVENTS = ("conflict_onset", "conflict_end", "coup", "coup_success", "irregular_exit", "dem_breakdown",
          "democratization", "sovdebt_onset", "sovdebt_end")
SHOCKS = ("conflict_onset", "coup_success", "sovdebt_onset")   # los que dejan huella en el PIB
GROUPS = ("bajo", "medio", "alto")                            # por ingreso de 2019 (< 4 000, < 20 000, más)
GROUP_CUTS = (np.log(4000.0), np.log(20000.0))
FAR = 99.0                                                    # «nunca» en los contadores de años desde


@lru_cache(maxsize=4)
def load_params(path: str | None = None) -> dict:
    p = Path(path) if path else PARAMS
    if not p.exists():
        raise FileNotFoundError(f"{p} no existe: córrase reports/acople/calibrar.py (necesita el Atlas)")
    return json.loads(p.read_text(encoding="utf-8"))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class SocialRun:
    """Resultados por año (T, n) del módulo social; ``country`` guarda medias entre miembros (T, N)."""
    years: np.ndarray
    series: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    country: dict = field(default_factory=dict)
    iso3: list = field(default_factory=list)
    region: np.ndarray | None = None

    def __getitem__(self, k):
        return self.series[k]


class Society:
    """Estado (n miembros × N países) y su paso anual. ``step(year, ypc)`` recibe el PIB mundial per cápita
    del año (en unidades de World3 o del panel: sólo cuentan sus cambios) y devuelve la fracción del PIB
    mundial perdida ese año por los eventos y la capacidad de implementación."""

    def __init__(self, params: dict, n: int, *, seed: int = 0, init: dict | None = None,
                 demog: dict | None = None, hazard_scale: float = 1.0, impulse_scale: float = 1.0):
        self.P = params
        self.n = n
        ini = init or params["init"]
        self.dem_data = demog or params["demog"]
        self.dyears = np.asarray(self.dem_data["years"], int)
        N = len(ini["lny"])
        self.N = N
        rep = lambda x: np.repeat(np.asarray(x, float)[None], n, 0)  # noqa: E731
        hist = np.asarray(ini["lny"], float)                         # (N, 8): de Y-7 a Y
        self.hist = np.repeat(hist.T[:, None, :], n, 1)              # (8, n, N)
        self.lny = self.hist[-1].copy()
        self.e = rep(ini["e"])
        self.ln_debt = rep(ini["ln_debt"])
        self.conflict = rep(ini["conflict"]) > 0.5
        self.cdur = rep(ini["cdur"])
        self.dem = rep(ini["dem"]) > 0.5
        self.crisis = rep(ini["crisis"]) > 0.5
        self.since = {k: rep(ini[f"since_{k}"]) for k in ("conflict", "coup", "crisis", "coup_success")}
        self.year = int(ini["year"])
        self.rng = np.random.default_rng(seed)
        self.hscale = hazard_scale
        H = len(params["impulses"]["conflict_onset"])
        self.H = H
        self.pending = np.zeros((H, n, N))                           # incrementos de ln PIB por venir
        self.kern = {k: impulse_scale * np.diff(np.r_[0.0, np.asarray(v, float)])
                     for k, v in params["impulses"].items()}
        self.noise = params["noise"]
        self.pools = [np.asarray(x, float) for x in self.noise["pools"]]
        g0 = self.hist[-1, 0]
        self.group0 = np.digitize(g0, GROUP_CUTS)                    # (N,) grupo de ingreso de 2019
        self.ypc_prev = None
        self.feedback = True                                          # la serie mundial ya incluye las pérdidas
        self.loss_index = np.ones(n)                                  # PIB mundial con / sin eventos (acumulado)

    # ── demografía de WPP ──
    def _demog(self, year):
        j = int(np.clip(np.searchsorted(self.dyears, year), 0, len(self.dyears) - 1))
        return np.asarray(self.dem_data["pop"], float)[:, j], np.asarray(self.dem_data["youth"], float)[:, j]

    def _features(self, youth_prev):
        lny = self.hist[-1]
        f = {"lny": lny, "growth": self.hist[-1] - self.hist[-2],
             "depth": 1.0 - np.exp(lny - self.hist.max(0)), "youth": np.broadcast_to(youth_prev, lny.shape),
             "ln_debt": self.ln_debt, "conflict": self.conflict.astype(float), "dem": self.dem.astype(float),
             "conf10": (self.since["conflict"] <= 9).astype(float), "coup5": (self.since["coup"] <= 4).astype(float),
             "crisis10": (self.since["crisis"] <= 9).astype(float), "ln_dur": np.log(np.maximum(self.cdur, 1.0))}
        return f

    def hazard(self, name, f):
        h = self.P["hazards"][name]
        x = np.full(f["lny"].shape, h["const"], float)
        for v, b in zip(h["vars"], h["coef"]):
            x = x + b * f[v]
        p = _sigmoid(x)
        return np.clip(p * self.hscale, 0, 1) if name in ("conflict_onset", "coup", "irregular_exit",
                                                          "dem_breakdown", "sovdebt_onset") else p

    def step(self, year: int, ypc_world: np.ndarray):
        """Un año. ``ypc_world`` (n,) es el PIB mundial per cápita del modelo biofísico ese año."""
        n, N = self.n, self.N
        pop, youth = self._demog(year)
        _, youth_prev = self._demog(year - 1)
        f = self._features(youth_prev)
        U = self.rng.random((len(EVENTS), n, N))                     # números comunes: siempre los mismos
        u_noise = self.rng.random((n, N))
        p = {k: self.hazard(k, f) for k in EVENTS}
        ev = {k: U[i] < p[k] for i, k in enumerate(EVENTS)}
        peace = ~self.conflict
        onset = peace & ev["conflict_onset"]
        end = self.conflict & ev["conflict_end"]
        coup = ev["coup"]
        succ = coup & ev["coup_success"]
        bd = self.dem & ev["dem_breakdown"]
        dz = ~self.dem & ev["democratization"]
        cr_on = ~self.crisis & ev["sovdebt_onset"]
        cr_end = self.crisis & ev["sovdebt_end"]
        out = {"conflict_onset": onset, "conflict_end": end, "coup": coup, "coup_success": succ,
               "irregular_exit": ev["irregular_exit"], "dem_breakdown": bd, "democratization": dz,
               "sovdebt_onset": cr_on}
        self.last = out
        expected = {"conflict_onset": np.where(peace, p["conflict_onset"], 0), "coup": p["coup"],
                    "sovdebt_onset": np.where(~self.crisis, p["sovdebt_onset"], 0),
                    "dem_breakdown": np.where(self.dem, p["dem_breakdown"], 0)}

        # huellas en el PIB de los eventos de este año (h = 0 entra ya)
        for k, flag in (("conflict_onset", onset), ("coup_success", succ), ("sovdebt_onset", cr_on)):
            if flag.any():
                self.pending += self.kern[k][:, None, None] * flag[None]
        imp = self.pending[0].copy()
        self.pending = np.roll(self.pending, -1, 0)
        self.pending[-1] = 0.0

        # ingreso: ruido relativo AR(1) con innovaciones remuestreadas por grupo de ingreso
        grp = np.digitize(self.lny, GROUP_CUTS)
        u = np.empty((n, N))
        for g, pool in enumerate(self.pools):
            m = grp == g
            if m.any():
                u[m] = pool[(u_noise[m] * len(pool)).astype(int)]
        self.e = self.noise["rho"] * self.e + u
        pre = self.lny + self.e
        if self.ypc_prev is None:
            g_world = np.zeros(n)
        else:
            g_world = np.log(ypc_world) - np.log(self.ypc_prev)
        # PIB mundial antes de los eventos de este año = PIB mundial del modelo biofísico
        w = pop[None] / pop.sum()
        target = np.log(np.sum(w * np.exp(self.lny), 1)) + g_world      # (n,)
        c = target - np.log(np.sum(w * np.exp(pre), 1))
        pre = pre + c[:, None]
        new = pre + imp
        Ypre = np.sum(w * np.exp(pre), 1)
        Ypost = np.sum(w * np.exp(new), 1)
        loss = 1.0 - Ypost / Ypre
        self.loss_index *= 1.0 - loss
        growth = new - self.lny
        self.lny = new
        self.hist = np.roll(self.hist, -1, 0)
        self.hist[-1] = new
        # con vuelta, World3 pierde la misma fracción y el año siguiente parte de ahí; sin vuelta (sólo ida) World3
        # no se entera y la suma de los PIB nacionales queda por debajo del PIB de World3 en la pérdida acumulada
        self.ypc_prev = np.asarray(ypc_world, float) * ((1.0 - loss) if self.feedback else 1.0)

        # deuda / PIB
        d = self.P["debt"]
        self.ln_debt = np.maximum(self.ln_debt + d["const"] + d["growth"] * growth + d["ln_debt"] * self.ln_debt
                                  + d["onset"] * cr_on, 0.0)

        # estados y contadores
        self.conflict = (self.conflict & ~end) | onset
        self.cdur = np.where(self.conflict, np.where(onset, 1.0, self.cdur + 1.0), 0.0)
        self.dem = (self.dem & ~bd) | dz
        self.crisis = (self.crisis & ~cr_end) | cr_on
        for k, flag in (("conflict", self.conflict), ("coup", coup), ("crisis", self.crisis), ("coup_success", succ)):
            self.since[k] = np.where(flag, 0.0, np.minimum(self.since[k] + 1.0, FAR))
        self.year = year

        # capacidad de implementar política: parte del PIB mundial en conflicto o en los 5 años tras una crisis o un golpe
        kap = self.P["capacity"]
        Yc = w * np.exp(new)
        share = lambda flag: np.sum(Yc * flag, 1) / Ypost  # noqa: E731
        win = kap["window"] - 1                                      # la ventana h = 0..4 de la proyección del gasto
        capacity = 1.0 - (kap["crisis"] * share(self.since["crisis"] <= win) + kap["conflict"] * share(self.conflict)
                          + kap["coup"] * share(self.since["coup_success"] <= win))
        rec = {"loss": loss, "capacity": capacity, "gdp_index": self.loss_index.copy(),
               "in_conflict": self.conflict.sum(1).astype(float), "pop_conflict": np.sum(w * self.conflict, 1),
               "demo_pop": np.sum(w * self.dem, 1), "demo_countries": self.dem.mean(1),
               "in_crisis": self.crisis.sum(1).astype(float), "gdp_crisis": share(self.crisis),
               "valley10": np.sum(w * (f["depth"] >= 0.1), 1), "countries_valley10": (f["depth"] >= 0.1).sum(1) * 1.0,
               "growth_world": g_world,
               **{k: v.sum(1).astype(float) for k, v in out.items()},
               **{"p_" + k: v.sum(1) for k, v in expected.items()}}
        grp0 = self.group0
        groups = {}
        for g, name in enumerate(GROUPS):
            m = grp0 == g
            wg = w[:, m] / w[:, m].sum()
            groups[name] = {"conflict_onset": onset[:, m].sum(1) * 1.0, "coup": coup[:, m].sum(1) * 1.0,
                            "sovdebt_onset": cr_on[:, m].sum(1) * 1.0, "dem_breakdown": bd[:, m].sum(1) * 1.0,
                            "pop_conflict": np.sum(wg * self.conflict[:, m], 1), "demo_pop": np.sum(wg * self.dem[:, m], 1),
                            "gdp_pc": np.sum(wg * np.exp(new[:, m]), 1)}
        cty = {"p_conflict": np.where(peace, p["conflict_onset"], 1.0).mean(0), "conflict": self.conflict.mean(0),
               "p_coup": p["coup"].mean(0), "dem": self.dem.mean(0), "crisis": self.crisis.mean(0),
               "lny": new.mean(0)}
        return loss, capacity, rec, groups, cty


class Coupler:
    """El gancho de ``l0.run``: corre ``Society`` un paso por año desde ``start`` con el PIB per cápita de
    World3 y devuelve a World3 la destrucción de capital (``feedback``) y la capacidad (``capacity``)."""

    TECH = ("lyt", "rct", "ppt")

    def __init__(self, n, *, params=None, seed=0, start=2020, feedback=True, capacity=True, **kw):
        self.P = params or load_params()
        self.soc = Society(self.P, n, seed=seed, **kw)
        self.soc.feedback = feedback
        self.start = start
        self.feedback = feedback
        self.use_capacity = capacity
        self.cap = np.ones(n)
        self.rows, self.grows, self.crows, self.years = [], [], [], []
        self.scale0 = None

    def __call__(self, t, st, a):
        flow = {k: self.cap for k in self.TECH} if self.use_capacity else None
        if not np.isclose(t % 1.0, 0.0):
            return {"flow": flow}
        y = int(round(t))
        ypc = (a["io"] + a["so"]) / a["pop"]
        if y < self.start:
            if y == self.start - 1:
                self.soc.ypc_prev = ypc
            return {"flow": flow}
        loss, cap, rec, grp, cty = self.soc.step(y, ypc)
        self.years.append(y)
        self.rows.append(rec)
        self.grows.append(grp)
        self.crows.append(cty)
        if self.use_capacity:
            self.cap = cap
            flow = {k: self.cap for k in self.TECH}
        scale = {"ic": 1.0 - loss, "sc": 1.0 - loss} if self.feedback else None
        return {"flow": flow, "scale": scale}

    def result(self) -> SocialRun:
        keys = self.rows[0].keys()
        series = {k: np.stack([r[k] for r in self.rows]) for k in keys}
        groups = {g: {k: np.stack([r[g][k] for r in self.grows]) for k in self.grows[0][g]} for g in GROUPS}
        country = {k: np.stack([r[k] for r in self.crows]) for k in self.crows[0]}
        return SocialRun(years=np.asarray(self.years), series=series, groups=groups, country=country,
                         iso3=list(self.P["countries"]["iso3"]), region=np.asarray(self.P["countries"]["region"]))


def run_coupled(*, constants=None, tables=None, n=1, t1=2100.0, dt=0.5, record=None, seed=0, start=2020,
                feedback=True, capacity=True, params=None, **kw):
    """World3-03 con el módulo social. Devuelve ``(World3Run, SocialRun)``. ``kw`` pasa a ``Society``
    (``hazard_scale``, ``impulse_scale`` para pruebas de sensibilidad)."""
    from world4.core import l0
    hook = Coupler(n, params=params, seed=seed, start=start, feedback=feedback, capacity=capacity, **kw)
    r = l0.run(constants=constants, tables=tables, n=n, t1=t1, dt=dt, record=record or l0.RECORD, hook=hook)
    return r, hook.result()
