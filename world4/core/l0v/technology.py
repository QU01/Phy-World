"""Tecnología estimada con datos para el L0 vigente (fase 2).

En el escenario 1 de World3 las tres tecnologías están apagadas (rendimiento agrícola, uso de recursos por unidad
de producción, contaminación generada por unidad de actividad se quedan en 1) y, cuando se encienden en otros
escenarios, encarecen el capital. Los datos dicen otra cosa: los rendimientos de cereales se multiplicaron por 2,1
entre 1970 y 2014 y el CO2 fósil creció mucho menos que la producción. Aquí las tres tecnologías son CURVAS en el
tiempo, estimadas con las series mundiales de ``world4.data.world``:

    rendimiento   lyt(t) = cap - (cap - 1) · exp(-r_y · (t - t0))        sube hacia un techo
    recursos      rct(t) = suelo + (1 - suelo) · exp(-r_r · (t - t0))    baja hacia un suelo
    contaminación ppt(t) = suelo + (1 - suelo) · exp(-r_p · (t - t0))

con ``t0`` el año de políticas de World3 (1975). Son los stocks ``lyt``, ``rct`` y ``ppt`` de World3, que el gancho
SUSTITUYE por la curva. World3 retarda 20 años (``tdd``) el paso de la tecnología desarrollada a la tecnología
en uso; aquí la curva se estima con lo que el mundo HIZO, así que ya es la tecnología en uso y el retardo baja a
1 año (con 20, el ajuste llevaba la curva de recursos a su suelo en diez años y el suelo, que es un supuesto,
decidía el resultado: medido).

**Tierra.** Las tierras de cultivo del mundo llevan casi estables desde 1990 (1.517 millones de hectáreas en
1990, 1.593 en 2016, HYDE) con el rendimiento duplicado. World3 no lo reproduce por dos lados:

* acorta la vida de la tierra cuando sube el rendimiento (tabla ``llmy1``): con los rendimientos REALES deja la
  tierra cultivable en 1.162 millones de hectáreas en 2016, 654 en 2023 y 88 en 2050 (medido). Aquí rige desde ``t0`` el
  control de la erosión del escenario 5 del propio libro (tabla ``llmy2``, ``llmytm``), que limita la pérdida a
  unos 8 millones de hectáreas al año [S: es una elección; el ajuste a la tierra no distingue más erosión de
  menos roturación, y ésta es la que deja los flujos brutos en un orden creíble];
* sin esa erosión, rotura demasiada tierra (1.995 millones de hectáreas en 2016). Desde ``t0`` el cambio neto de
  tierra cultivable de World3 (roturación menos erosión y urbanización) se multiplica por ``land_restraint``, una
  perilla que se estima con las tierras de cultivo (cambiar la tierra potencial total, ``palt``, movía toda la
  historia y dejaba 1970 un 15 % bajo: medido).

**Coste de capital.** World3 multiplica la relación capital-producto por tres tablas, una por tecnología. Aquí
cada multiplicador ``m`` pasa a ``1 + kappa · (m - 1)``: ``kappa = 1`` es World3, ``kappa = 0`` es tecnología sin
coste, y se estima con el PIB.

**Estimación** (``calibrate``): las cinco perillas ``(r_y, r_r, r_p, kappa, land_restraint)`` minimizan el error
cuadrático en logaritmos, con datos de 1971 a ``cutoff``, de cuatro índices (1970 = 1): rendimiento de la tierra
frente al de cereales, uso de recursos frente al CO2 fósil, generación de contaminación frente al CO2 fósil y
producción (industria + servicios) frente al PIB; y de la tierra cultivable frente a las tierras de cultivo, ésta
en hectáreas. La población de esas corridas es la observada (cohortes con la fecundidad y la esperanza
de vida reales). Los valores estimados viajan en ``world4/data/l0v_technology.json``.

Techo del rendimiento (2) y suelos (0,2): los datos hasta hoy no los alcanzan, así que no los identifican [S].
Pesan en 2100 y serán ejes de incertidumbre, igual que los recursos no renovables iniciales de World3 (``nri``),
que aquí no se estiman.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PARAMS_FILE = Path(__file__).resolve().parents[2] / "data" / "l0v_technology.json"
KNOBS = ("r_yield", "r_resource", "r_pollution", "kappa", "land_restraint")
LEVELS = ("arable_land",)                 # se comparan en sus unidades, no como índice
FIT = (("land_yield", "ly", "cereal_yield"), ("resource_use", None, "fossil_co2"),
       ("pollution_generation", "ppgr", "fossil_co2"), ("output", None, "gdp"), ("arable_land", "al", "cropland"))


@dataclass(frozen=True)
class Tech:
    r_yield: float = 0.0
    r_resource: float = 0.0
    r_pollution: float = 0.0
    kappa: float = 1.0
    land_restraint: float = 1.0           # fracción de la roturación neta de World3 que ocurre desde ``t0``
    t0: float = 1975.0
    erosion_control: bool = False         # True: desde ``t0``, la tabla de erosión del escenario 5 (``llmy2``)
    yield_cap: float = 2.0
    resource_floor: float = 0.2
    pollution_floor: float = 0.2
    freeze_at: float | None = None        # desde este año la tecnología deja de avanzar (escenario)
    delay: float = 1.0                    # ``tdd`` de World3: años entre desarrollar una tecnología y usarla

    def paths(self, t):
        """``(lyt, rct, ppt)`` en ``t``."""
        tt = t if self.freeze_at is None else min(t, self.freeze_at)
        h = max(tt - self.t0, 0.0)
        return (self.yield_cap - (self.yield_cap - 1) * np.exp(-self.r_yield * h),
                self.resource_floor + (1 - self.resource_floor) * np.exp(-self.r_resource * h),
                self.pollution_floor + (1 - self.pollution_floor) * np.exp(-self.r_pollution * h))

    def constants(self, base: dict) -> dict:
        """Las constantes de World3 que cambian (``base``: las del escenario)."""
        out = {"tdd": self.delay}
        if self.erosion_control:
            out["llmytm"] = self.t0
        return out

    def tables(self, tables: dict) -> dict:
        """Las tres tablas de coste de capital de ``tables`` con el multiplicador llevado a ``1 + kappa·(m - 1)``."""
        out = {}
        for name, base in (("icormrrct", None), ("icormlyt", 1.0), ("icormpt", 1.0)):
            lo, hi, step, vals = tables[name]
            vals = np.asarray(vals, float)
            ref = float(vals[-1]) if base is None else base           # icormrrct vale 3 sin tecnología (nruf = 1)
            out[name] = (lo, hi, step, list(ref * (1 + self.kappa * (vals / ref - 1))))
        return out


class TechHook:
    """Gancho de ``l0.run`` que pone los tres stocks de tecnología en su curva y frena el cambio de tierra.

    Con ``climate`` (un ``carbon.Carbon``) lleva además la cuenta de las emisiones del modelo desde ``origin`` (o
    desde el corte de ``climate``), del calentamiento que causan, y resta rendimiento con la regla de
    ``Carbon.damage``."""

    def __init__(self, tech: Tech, dt: float = 0.5, climate=None, origin: int | None = None, obs: dict | None = None):
        self.tech, self.dt, self.climate = tech, dt, climate
        self.rec = {"time": [], "warming": []}
        if climate is not None:
            from world4.core.l0v import carbon
            o = carbon.observed(obs)
            self.ref = int(climate.cutoff if origin is None else origin)
            self.fi_anchor = float(o["fi"][o["years"] == carbon.ANCHOR][0])
            self.anchor_year = float(carbon.ANCHOR)
            self.luc = carbon.luc_after(o, self.ref)
            self.prev = self.use_anchor = None
            self.cum = 0.0                                             # GtCO2 emitidas desde ``ref``

    def _warming(self, t, st):
        nr = np.array(st["nr"], float, copy=True)
        warming = 0.0 * nr
        if self.prev is not None:
            use = (self.prev - nr) / self.dt                           # uso de recursos del paso anterior, por año
            if abs(t - self.anchor_year) < 1e-9:
                self.use_anchor = use
            if t > self.ref and self.use_anchor is not None:
                self.cum = self.cum + (self.fi_anchor * use / self.use_anchor + self.luc) * self.dt
                warming = self.climate.tcre * self.cum / 1000.0
        self.prev = nr
        self.rec["time"].append(t)
        self.rec["warming"].append(np.array(warming, float, copy=True))
        return warming

    def __call__(self, t, st, a):
        damage = 1.0 if self.climate is None else self.climate.damage(self._warming(t, st))
        if t + self.dt < self.tech.t0:
            return None
        lyt, rct, ppt = self.tech.paths(t + self.dt)                  # "set" fija el stock ya integrado
        out = {"set": {"lyt": lyt * damage, "rct": rct, "ppt": ppt}}
        if t >= self.tech.t0 and np.any(np.asarray(self.tech.land_restraint) != 1.0):
            out["flow"] = {"al": self.tech.land_restraint, "pal": self.tech.land_restraint}
        return out

    def patch(self, run) -> None:
        """Añade a la corrida el calentamiento desde el origen que el gancho usó (``warming``, °C)."""
        if self.climate is not None:
            run.series["warming"] = np.stack(self.rec["warming"])


def compose(*hooks):
    """Un gancho que llama a varios y junta lo que devuelven (los factores se multiplican; ``set`` se une)."""
    hooks = [h for h in hooks if h is not None]
    if len(hooks) == 1:
        return hooks[0]

    def hook(t, st, a):
        out = {"flow": {}, "scale": {}, "set": {}}
        for h in hooks:
            r = h(t, st, a) or {}
            for kind in ("flow", "scale"):
                for k, v in (r.get(kind) or {}).items():
                    out[kind][k] = out[kind].get(k, 1.0) * v
            out["set"].update(r.get("set") or {})
        return out
    return hook


# ------------------------------------------------------------------ parámetros estimados
def stored() -> dict:
    return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))


BLOCKS = {False: "by_cutoff", True: "by_cutoff_curve"}     # con el stock de World3 / con la curva de ``resources``


def params(cutoff: int | None = None, resources: bool = False) -> Tech:
    """La tecnología estimada con datos <= ``cutoff`` (None: la última guardada). ``resources``: la estimada con la
    curva de coste de los recursos (``resources``) en lugar de con el stock de World3."""
    st = stored()
    d = st[BLOCKS[bool(resources)]]
    key = str(max(int(k) for k in d) if cutoff is None else int(cutoff))
    if key not in d:
        raise ValueError(f"no stored technology for cutoff {cutoff}; available {sorted(d)} (technology.calibrate)")
    return Tech(**{**st["fixed"], **{k: d[key][k] for k in KNOBS}})


# ------------------------------------------------------------------ estimación
def _indices(run):
    """Los cuatro índices del modelo (1970 = 1) en años enteros: ``(años, {nombre: valores})``."""
    from world4.validate import world3
    m = world3.model_series(run)
    years = m["years"]
    t = np.asarray(run.time, float)
    ok = np.isclose(t, np.round(t))
    gen = np.asarray(run["ppgr"], float)[ok, 0]
    i70 = int(np.where(years == world3.BASE_YEAR)[0][0])
    raw = {"land_yield": m["land_yield"], "resource_use": m["resource_use"], "pollution_generation": gen,
           "output": m["output"], "arable_land": m["arable_land"]}
    return years, {k: v if k in LEVELS else v / v[i70] for k, v in raw.items()}


def _observed():
    from world4.data import world
    from world4.validate import world3
    w = world.load()
    i70 = int(world3.BASE_YEAR - w["years"][0])
    return w["years"], {name: w[obs] if name in LEVELS else w[obs] / w[obs][i70] for name, _, obs in FIT}


def loss(tech: Tech, cutoff: int, first: int = 1971, detail: bool = False, inputs=None, resources=None):
    """Error cuadrático medio en logaritmos de los cuatro índices, de ``first`` a ``cutoff`` (suma de los cuatro)."""
    from world4.core.l0v import model
    r, _ = model.run(technology=tech, fertility="observed", mortality="observed", t1=float(cutoff) + 1.0, inputs=inputs,
                     resources=resources, cutoff=cutoff if resources is not None else None)
    years, mod = _indices(r)
    oy, obs = _observed()
    parts = {}
    for name in mod:
        sel = (oy >= first) & (oy <= cutoff) & np.isfinite(obs[name])
        m = np.interp(oy[sel], years, mod[name])
        parts[name] = float(np.mean((np.log(m) - np.log(obs[name][sel])) ** 2))
    return parts if detail else float(sum(parts.values()))


BOUNDS = {"r_yield": (0.0, 0.3), "r_resource": (0.0, 0.3), "r_pollution": (0.0, 0.3), "kappa": (-1.0, 2.0),
          "land_restraint": (0.0, 1.5)}
STARTS = ({"r_yield": 0.015, "r_resource": 0.05, "r_pollution": 0.035, "kappa": 0.0, "land_restraint": 0.4},
          {"r_yield": 0.02, "r_resource": 0.045, "r_pollution": 0.025, "kappa": -0.3, "land_restraint": 0.1})
ESTIMATED = {"erosion_control": True}     # lo que la variante estimada fija además de las perillas


def calibrate(cutoff: int = 2023, fixed: dict | None = None, log=None, resources=None) -> dict:
    """Estima las perillas con datos <= ``cutoff``; las de ``fixed`` se quedan en el valor dado.
    Devuelve ``{perilla: valor, "loss", "rms_pct", "fixed"}``."""
    from scipy.optimize import minimize
    from world4.core.l0v import model
    inputs = model.world_inputs()
    fixed = {**ESTIMATED, **(fixed or {})}
    free = [k for k in KNOBS if k not in fixed]

    def tech_of(x):
        return Tech(**{**fixed, **dict(zip(free, map(float, x)))})

    def f(x):
        if any(not BOUNDS[k][0] <= v <= BOUNDS[k][1] for k, v in zip(free, x)):
            return 1e3
        return loss(tech_of(x), cutoff, inputs=inputs, resources=resources)

    runs = [minimize(f, [x0[k] for k in free], method="Nelder-Mead",
                     options={"xatol": 2e-4, "fatol": 1e-7, "maxiter": 900}) for x0 in STARTS]
    res = min(runs, key=lambda r: r.fun)                             # dos arranques: el simplex se queda en mínimos locales
    tech = tech_of(res.x)
    parts = loss(tech, cutoff, detail=True, inputs=inputs, resources=resources)
    if log:
        log(f"cutoff {cutoff}: {sum(r.nfev for r in runs)} runs, loss {res.fun:.5f} (starts: "
            + ", ".join(f"{r.fun:.5f}" for r in runs) + ")")
    return {**{k: float(getattr(tech, k)) for k in KNOBS}, "loss": float(res.fun),
            "rms_pct": {k: float(100 * np.sqrt(v)) for k, v in parts.items()}}


def write(cutoffs=(1990, 2000, 2010, 2023), log=print, blocks=(False, True)) -> dict:
    """Recalcula y guarda ``l0v_technology.json`` (unos minutos por bloque). ``blocks``: False, con el stock de
    recursos de World3; True, con la curva de coste de ``resources``. Los bloques que no se piden se conservan."""
    out = stored() if PARAMS_FILE.exists() else {}
    out["_doc"] = ["Technology curves of the current World3 (world4.core.l0v.technology), estimated with world series "
                   "up to each cutoff year: by_cutoff with the fixed resource stock of World3, by_cutoff_curve with "
                   "the fossil cost curve of world4.core.l0v.resources. Model parameters, no third-party data. "
                   "Rebuild: python -m world4.core.l0v.technology"]
    out["fixed"] = {k: v for k, v in asdict(Tech(**ESTIMATED)).items() if k not in KNOBS and k != "freeze_at"}
    for b in blocks:
        out[BLOCKS[b]] = {str(c): calibrate(c, log=log, resources="estimated" if b else None) for c in cutoffs}
    PARAMS_FILE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    write()
