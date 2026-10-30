"""Sistema de regresiones de la densidad de eventos de violencia organizada por país, con el Atlas.

Tres usos:

1. **Pronóstico directo desde cualquier año** (``DensitySystem``): un modelo por horizonte h = 1..5 años, binomial
   negativa con ln(superficie) como exposición, sobre los factores que el estudio (``factors``) seleccionó para ese
   horizonte (``reports/factores_eventos/resultados.json``). Desde un origen t (1990-2025) da, para cada país, los
   eventos esperados en t + h, la densidad (eventos por 10 000 km²), P(al menos un evento), p10/p50/p90 y el
   aporte de cada factor. Hasta 2019 usa el Atlas observado; después, el Atlas congelado en su último dato y la
   historia de UCDP GED al día (``country_panel``).
2. **Densidad por celda** (``cell_density``): reparte los eventos esperados del país entre sus celdas PRIO-GRID en
   proporción a su historia reciente (vida media de 2 años) más un fondo uniforme, para pintar el mapa.
3. **Escenario con el L1** (``scenario``): el modelo de 1 año aplicado año a año a las trayectorias que simula el
   Atlas (ingreso, crecimiento, deuda, democracia, golpes, crisis y conflicto civil, por miembro del ensemble;
   ``structural.engine.ensemble``) con la historia de eventos simulada por la propia binomial negativa: densidad
   con bandas hasta 2030 desde 2019 (modo vivo del L1) o desde cualquier origen 1990-2018.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from world4.conflict import country_panel as cp
from world4.conflict.factors import NB

HORIZONS = (1, 2, 3, 4, 5)
STUDY = Path(__file__).resolve().parents[2] / "reports" / "factores_eventos" / "resultados.json"
# si el estudio no está: historia + los factores que más aportan en la primera corrida del estudio
DEFAULT_FEATURES = list(cp.FACTORS["historia"]) + ["ln_area", "forest", "corruption", "unga_us", "polyarchy", "excluded"]
L1_DRIVEN = ("lny", "growth", "debt", "democ", "coups10", "crisis", "h_acd")


class Prep:
    """Imputación (mediana del entrenamiento, con indicador de faltante) y estandarización, reutilizable."""

    def fit(self, X: pd.DataFrame, flag_min: float = 0.03):
        self.cols = list(X.columns)
        self.med = X.median().fillna(0.0)
        self.flags = [c for c in self.cols if X[c].isna().mean() >= flag_min]
        Z = self._raw(X)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1.0).fillna(1.0)
        self.keep = list(Z.columns[Z.std() > 1e-9])
        return self

    def _raw(self, X):
        Z = X[self.cols].copy()
        for c in self.flags:
            Z[f"{c}__na"] = X[c].isna().astype(float)
        return Z.fillna(self.med)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._raw(X)
        return (Z[self.keep] - self.mu[self.keep]) / self.sd[self.keep]

    def to_dict(self):
        return {"cols": self.cols, "med": self.med.to_dict(), "flags": self.flags, "mu": self.mu.to_dict(),
                "sd": self.sd.to_dict(), "keep": self.keep}

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.cols, p.flags, p.keep = d["cols"], d["flags"], d["keep"]
        p.med, p.mu, p.sd = pd.Series(d["med"]), pd.Series(d["mu"]), pd.Series(d["sd"])
        return p


def study_features(path: Path | None = None) -> dict[int, list[str]]:
    """Factores por horizonte según el estudio; los horizontes no estudiados toman el del más cercano."""
    path = path or STUDY
    if not path.is_file():
        return {h: list(DEFAULT_FEATURES) for h in HORIZONS}
    res = json.loads(path.read_text(encoding="utf-8"))
    time_only = set(cp.FACTORS["World3 mundial"])
    # las variables que sólo cambian con el año (World3 mundial, precios mundiales, tendencia) quedan fuera: el
    # estudio no muestra que aporten fuera de muestra y extrapolan como tendencias (sin w3_iopc, que la selección
    # tomó a 1 año, el RPS de 2020-2025 queda igual: 0,1777 frente a 0,1780)
    sel = {int(k[1:]): [c for c in v["forward"]["selected"] if c not in time_only]
           for k, v in res.items() if k.startswith("h") and "forward" in v}
    if not sel:
        return {h: list(DEFAULT_FEATURES) for h in HORIZONS}
    return {h: sel[min(sel, key=lambda s: (abs(s - h), s))] for h in HORIZONS}


def _offset(df):
    return np.log(df.area_km2.to_numpy() / cp.AREA_UNIT)


class DensitySystem:
    def __init__(self, features: dict[int, list[str]] | None = None, horizons=HORIZONS, mean: str = "ppml"):
        self.features = features or study_features()
        self.mean = mean
        self.horizons = tuple(horizons)
        self.models: dict[int, NB] = {}
        self.preps: dict[int, Prep] = {}

    def fit(self, train_until: int = cp.LAST_GED_YEAR, first_year: int = 1990):
        """Entrena cada horizonte con los objetivos <= ``train_until`` (todos los países, también los que el Atlas
        no tiene: sus variables del Atlas van imputadas y marcadas)."""
        self.train_until = int(train_until)
        panel = cp.build()
        for h in self.horizons:
            d = cp.with_targets(panel, h)
            d = d[(d.year >= first_year) & d.y.notna() & (d.year + h <= train_until)]
            cols = self.features[h]
            prep = Prep().fit(d[cols])
            self.preps[h] = prep
            self.models[h] = NB(mean=self.mean).fit(prep.transform(d[cols]).to_numpy(), d.y.to_numpy(), _offset(d))
        return self

    def predict(self, origin: int, h: int, *, top: int = 3) -> pd.DataFrame:
        """Predicción para ``origin + h`` con la información de ``origin`` (todos los países)."""
        panel = cp.build()
        d = panel[panel.year == origin].reset_index(drop=True)
        if d.empty:
            raise ValueError(f"origin {origin} outside {cp.FIRST_YEAR}-{cp.LAST_GED_YEAR}")
        m, prep = self.models[h], self.preps[h]
        Z = prep.transform(d[self.features[h]])
        off = _offset(d)
        mu = m.mu(Z.to_numpy(), off)
        r = 1.0 / m.alpha_of(mu)
        p = r / (r + mu)
        q = np.stack([stats.nbinom.ppf(x, r, p) for x in (0.1, 0.5, 0.9)], 1)
        contrib = Z.to_numpy() * m.b[1:]
        names = list(Z.columns)
        drivers = []
        for row in contrib:
            o = np.argsort(-np.abs(row))[:top]
            drivers.append([{"factor": names[j], "log_effect": float(row[j])} for j in o])
        out = pd.DataFrame({"iso3": d.iso3, "in_atlas": d.in_atlas, "origin": origin, "target": origin + h,
                            "atlas_frozen_years": d.atlas_frozen, "events_expected": mu,
                            "density": mu / np.exp(off), "p_any": 1.0 - p ** r, "p10": q[:, 0], "p50": q[:, 1],
                            "p90": q[:, 2], "area_km2": d.area_km2, "drivers": drivers})
        obs = cp.with_targets(panel, h)
        obs = obs[obs.year == origin].set_index("iso3").y
        out["observed"] = out.iso3.map(obs)
        return out.sort_values("events_expected", ascending=False).reset_index(drop=True)

    def coefficients(self) -> dict:
        return {h: {name: float(np.exp(b)) for name, b in zip(self.preps[h].keep, self.models[h].b[1:])}
                for h in self.horizons}

    def to_dict(self):
        return {"train_until": self.train_until, "features": {str(h): v for h, v in self.features.items()},
                "models": {str(h): {"b": m.b.tolist(), "alpha": m.alpha,
                                    "disp": None if m.disp is None else list(map(float, m.disp))}
                           for h, m in self.models.items()},
                "preps": {str(h): p.to_dict() for h, p in self.preps.items()}}

    @classmethod
    def from_dict(cls, d):
        s = cls({int(h): v for h, v in d["features"].items()}, horizons=tuple(int(h) for h in d["models"]))
        s.train_until = d["train_until"]
        for h, m in d["models"].items():
            nb = NB()
            nb.b, nb.alpha = np.asarray(m["b"]), m["alpha"]
            nb.disp = None if m.get("disp") is None else np.asarray(m["disp"])
            s.models[int(h)] = nb
            s.preps[int(h)] = Prep.from_dict(d["preps"][h])
        return s


# ---------------------------------------------------------------- validación: origen rodante
def backtest(origins=range(2005, 2025), horizons=HORIZONS, log=print, mean: str = "ppml", features=None) -> dict:
    """Para cada origen T se entrena con objetivos <= T y se predicen T + h <= 2025. Compara el sistema con la
    historia sola y con la densidad media (null). Origen >= 2019: el Atlas ya está congelado (fuera del Atlas)."""
    from world4.conflict import factors as F
    from world4.conflict import metrics as M
    feats = features or study_features()
    hist = list(cp.FACTORS["historia"])
    rows = []
    for T in origins:
        hs = [h for h in horizons if T + h <= cp.LAST_GED_YEAR]
        if not hs:
            continue
        systems = {"sistema": DensitySystem(feats, hs, mean=mean).fit(T),
                   "historia": DensitySystem({h: hist for h in hs}, hs, mean=mean).fit(T)}
        for h in hs:
            for name, s in systems.items():
                pr = s.predict(T, h)
                pr = pr[pr.observed.notna()]
                m = s.models[h]
                y = pr.observed.to_numpy()
                mu = pr.events_expected.to_numpy()
                r = 1.0 / m.alpha_of(mu)
                F_ = np.stack([stats.nbinom.cdf(t, r, r / (r + mu)) for t in F.CLASSES], 1)
                O = np.stack([(y <= t).astype(float) for t in F.CLASSES], 1)
                rows.append({"origin": T, "h": h, "model": name, "rps": float(((F_ - O) ** 2).sum(1).mean()),
                             "ls": float(stats.nbinom.logpmf(y, r, r / (r + mu)).mean()),
                             "aupr_any": M.average_precision(pr.p_any.to_numpy(), y > 0),
                             "expected": float(mu.sum()), "observed": float(y.sum()),
                             "spearman": float(stats.spearmanr(np.log1p(mu), np.log1p(y)).statistic),
                             "atlas_frozen": T >= cp.LAST_ATLAS_YEAR})
        log(f"origin {T}: done")
    df = pd.DataFrame(rows)
    summ = {}
    for (h, name), g in df.groupby(["h", "model"]):
        for era, gg in (("todos", g), ("atlas_observado", g[~g.atlas_frozen]), ("atlas_congelado", g[g.atlas_frozen])):
            if len(gg):
                summ.setdefault(str(h), {}).setdefault(era, {})[name] = {
                    "rps": float(gg.rps.mean()), "aupr_any": float(gg.aupr_any.mean()),
                    "spearman": float(gg.spearman.mean()), "ls": float(gg.ls.mean()),
                    "expected_over_observed": float(gg.expected.sum() / gg.observed.sum()), "origins": int(len(gg))}
    for h, eras in summ.items():
        for era, ms in eras.items():
            if "sistema" in ms and "historia" in ms:
                ms["skill_vs_historia"] = 1.0 - ms["sistema"]["rps"] / ms["historia"]["rps"]
    return {"summary": summ, "rows": rows}


# ---------------------------------------------------------------- de país a celdas
def cell_density(pred: pd.DataFrame, *, halflife_years: float = 2.0, background: float = 0.05) -> pd.DataFrame:
    """Reparte ``events_expected`` de cada país entre sus celdas: peso = eventos de la celda con decaimiento
    (vida media ``halflife_years``) hasta el origen + ``background`` × la media del país (para que una celda sin
    historia no quede en cero). Devuelve gid, lon, lat, iso3, eventos esperados y densidad (por 10 000 km²)."""
    from world4.data import conflict_grid as cdata
    d, iso, _ = cp._cells()
    origin = int(pred.origin.iloc[0])
    m_end = (origin - cdata.FIRST_YEAR + 1) * 12
    age_years = (m_end - 1 - d["ev_month"]) / 12.0
    ok = (d["ev_cell"] >= 0) & (d["ev_month"] < m_end)
    w_hist = np.bincount(d["ev_cell"][ok], weights=0.5 ** (age_years[ok] / halflife_years), minlength=len(iso))
    area = np.nan_to_num(d["pg_landarea"])
    num_to_iso3 = cp.build()[["isonum", "iso3"]].drop_duplicates().set_index("isonum").iso3.to_dict()
    expected = dict(zip(pred.iso3, pred.events_expected))
    out_e = np.zeros(len(iso))
    for num in np.unique(iso[iso > 0]):
        c = num_to_iso3.get(int(num))
        if c is None or c not in expected:
            continue
        idx = np.nonzero(iso == num)[0]
        w = w_hist[idx] + background * max(w_hist[idx].mean(), 1e-6) * (area[idx] / max(area[idx].mean(), 1e-9))
        if w.sum() <= 0:
            w = area[idx]
        out_e[idx] = expected[c] * w / w.sum()
    keep = out_e > 0
    return pd.DataFrame({"gid": d["gid"][keep], "lon": d["lon"][keep], "lat": d["lat"][keep],
                         "iso3": [num_to_iso3.get(int(n)) for n in iso[keep]], "events_expected": out_e[keep],
                         "density": out_e[keep] / np.maximum(area[keep], 1.0) * cp.AREA_UNIT})


# ---------------------------------------------------------------- escenario con el L1
def scenario(origin: int = 2019, members: int = 32, years: int = 11, seed: int = 0, system=None) -> dict:
    """Densidad por país año a año sobre las trayectorias del L1 (``L1_DRIVEN`` salen del ensemble; los demás
    factores quedan en su valor del origen; la historia de eventos se simula con la propia binomial negativa).
    Devuelve, por país, p10/p50/p90 de eventos y de densidad para origin+1 .. origin+years."""
    from world4.core.l1_atlas import calibrate
    from world4.structural import engine as E
    if system is None:
        f1 = study_features()[1]
        f1 = f1 + [c for c in L1_DRIVEN if c not in f1]          # que las trayectorias del L1 cuenten
        system = DensitySystem({1: f1}, horizons=(1,)).fit(train_until=min(origin, cp.LAST_GED_YEAR))
    sysm = system
    feats = sysm.features[1]
    m, prep = sysm.models[1], sysm.preps[1]
    years = min(years, 11)
    ens = E.ensemble(origin, R=members, seed=seed, live=(origin == cp.LAST_ATLAS_YEAR))
    W = calibrate.W
    l1_idx = {c: i for i, c in enumerate(W.iso3)}
    panel = cp.build()
    base = panel[panel.year == origin].reset_index(drop=True)
    hist5 = (panel[(panel.year > origin - 5) & (panel.year <= origin)].pivot(index="iso3", columns="year", values="ev_all")
             .reindex(base.iso3).fillna(0.0).to_numpy())                                  # (N, 5)
    N, R = len(base), members
    rng = np.random.default_rng(seed)
    # vecinos entre países del panel
    nb = cp.neighbours()
    pos = {int(n): k for k, n in enumerate(base.isonum)}
    A = np.zeros((N, N))
    for k, n in enumerate(base.isonum):
        for o in nb.get(int(n), ()):
            if o in pos:
                A[k, pos[o]] = 1.0
    ev_hist = np.repeat(hist5[None], R, 0)                                              # (R, N, 5)
    last_ev = np.where(hist5[:, ::-1] > 0, np.arange(5)[None], 99).min(1)
    since = np.repeat(last_ev[None].astype(float), R, 0)                                 # años desde el último
    area10 = base.area_km2.to_numpy() / cp.AREA_UNIT
    coup_hist = np.repeat(base.coups10.fillna(0).to_numpy()[None], R, 0)
    out_ev = np.zeros((years, R, N))
    X0 = base[feats].copy()
    idx = np.array([l1_idx.get(c, -1) for c in base.iso3])
    has = idx >= 0
    for k in range(1, years + 1):
        Xr = []
        for r in range(R):
            X = X0.copy()
            ev_t = ev_hist[r, :, -1]
            upd = {"h_ev": np.log1p(ev_t), "h_ev_l1": np.log1p(ev_hist[r, :, -2]),
                   "h_ev_5y": np.log1p(ev_hist[r].mean(1)), "h_dens": np.log1p(ev_t / area10),
                   "h_since": np.where(since[r] < 99, 0.5 ** (since[r] / 2.0), 0.0), "h_nb": np.log1p(A @ ev_t)}
            # el año de las variables es origin + k - 1 (información del año anterior al objetivo)
            j = k - 1
            ly = np.where(has, ens["ly"][r, np.maximum(idx, 0), j], np.nan)
            ly0 = np.where(has, ens["ly"][r, np.maximum(idx, 0), max(j - 1, 0)], np.nan)
            l1v = {"lny": ly, "growth": ly - ly0 if j > 0 else X0.get("growth", pd.Series(np.nan, index=X0.index)).to_numpy(),
                   "debt": np.where(has, ens["debt"][r, np.maximum(idx, 0), j], np.nan),
                   "democ": np.where(has, ens["demo"][r, np.maximum(idx, 0), j] == 1, np.nan).astype(float),
                   "crisis": np.where(has, ens["crisis"][r, np.maximum(idx, 0), j] == 1, np.nan).astype(float),
                   "h_acd": np.where(has, ens["conflict_real"][r, np.maximum(idx, 0), j] == 1, np.nan).astype(float)}
            if j > 0:
                coup_hist[r] = coup_hist[r] + np.where(has, ens["coup"][r, np.maximum(idx, 0), j] == 1, 0)
            l1v["coups10"] = coup_hist[r]
            for c, v in {**upd, **l1v}.items():
                if c in X.columns:
                    val = np.asarray(v, float)
                    X[c] = np.where(np.isfinite(val), val, X[c].to_numpy())
            Xr.append(prep.transform(X).to_numpy())
        mu = np.stack([m.mu(Z, np.log(area10)) for Z in Xr])                             # (R, N)
        rr = 1.0 / m.alpha_of(mu)
        draw = rng.negative_binomial(rr, rr / (rr + mu)).astype(float)
        out_ev[k - 1] = draw
        ev_hist = np.concatenate([ev_hist[:, :, 1:], draw[:, :, None]], 2)
        since = np.where(draw > 0, 0, since + 1)
    q = np.percentile(out_ev, [10, 50, 90], axis=1)                                      # (3, years, N)
    mean = out_ev.mean(1)
    ys = list(range(origin + 1, origin + years + 1))
    obs = panel.pivot(index="iso3", columns="year", values="ev_all").reindex(base.iso3)
    observed = [[float(obs.at[c, y]) if y in obs.columns and y <= cp.LAST_GED_YEAR else None for y in ys]
                for c in base.iso3]
    return {"origin": origin, "years": ys, "members": R, "live_l1": origin == cp.LAST_ATLAS_YEAR,
            "iso3": list(base.iso3), "area_km2": base.area_km2.tolist(),
            "events_mean": mean.T.tolist(), "events_p10": q[0].T.tolist(), "events_p50": q[1].T.tolist(),
            "events_p90": q[2].T.tolist(), "observed": observed,
            "world_events": {"mean": out_ev.sum(2).mean(1).tolist(),
                             "p10": np.percentile(out_ev.sum(2), 10, axis=1).tolist(),
                             "p90": np.percentile(out_ev.sum(2), 90, axis=1).tolist()},
            "l1_driven": [c for c in L1_DRIVEN if c in feats], "frozen": [c for c in feats if c not in L1_DRIVEN
                                                                          and not c.startswith("h_")]}


# ---------------------------------------------------------------- servicio (API, CLI, servidor)
CAVEATS = (
    "country-year counts of UCDP GED 26.1 events (organized violence, all types) placed in the country where they "
    "happened (Natural Earth geometry); density = events per 10 000 km² of land",
    "mean by Poisson pseudo-maximum likelihood (calibrated totals), negative binomial predictive distribution with "
    "a dispersion that depends on the mean; factors chosen out of sample by reports/factores_eventos (targets <= 2019)",
    "after 2019 the Atlas has no data: its variables stay at their 2019 value and only the event history moves",
    "Afghanistan, Somalia, South Sudan, Libya and six smaller countries are not in the Atlas: their Atlas variables "
    "are imputed and their forecast rests on their event history",
    "predictive association, not causal effects; 2020-2025 was never used to choose the factors (the honest test)",
)


def _model_path(train_until: int) -> Path:
    from world4.data import conflict_grid as cdata
    return cdata.root() / "models" / f"density_{train_until}.json"


def get_system(train_until: int = cp.LAST_GED_YEAR, refit: bool = False) -> DensitySystem:
    p = _model_path(train_until)
    study_mtime = STUDY.stat().st_mtime if STUDY.is_file() else 0
    if p.is_file() and not refit and p.stat().st_mtime >= study_mtime:
        return DensitySystem.from_dict(json.loads(p.read_text(encoding="utf-8")))
    s = DensitySystem().fit(train_until)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s.to_dict()), encoding="utf-8")
    return s


def payload(origin: int | None = None, horizon: int = 1, *, cells: bool = False) -> dict:
    origin = cp.LAST_GED_YEAR if origin is None else int(origin)
    if not (cp.FIRST_YEAR + 1 <= origin <= cp.LAST_GED_YEAR):
        raise ValueError(f"origin must be a year in {cp.FIRST_YEAR + 1}-{cp.LAST_GED_YEAR}")
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    # entrenado sólo con lo que se sabía en el origen (objetivos <= origen)
    s = get_system(train_until=origin)
    pr = s.predict(origin, horizon)
    bt = None
    btp = STUDY.with_name("backtest.json")
    if btp.is_file():
        b = json.loads(btp.read_text(encoding="utf-8"))["summary"].get(str(horizon), {})
        bt = {era: {"rps_system": v["sistema"]["rps"], "rps_history_only": v["historia"]["rps"],
                    "skill_vs_history": v.get("skill_vs_historia"), "aupr_any": v["sistema"]["aupr_any"],
                    "expected_over_observed": v["sistema"]["expected_over_observed"]} for era, v in b.items()}
    cols = ["iso3", "in_atlas", "events_expected", "density", "p_any", "p10", "p50", "p90", "area_km2",
            "atlas_frozen_years", "observed", "drivers"]
    rows = pr[cols].replace({np.nan: None}).to_dict(orient="records")
    out = {"origin": origin, "target": origin + horizon, "horizon": horizon, "train_until": s.train_until,
           "countries": rows, "world_events_expected": float(pr.events_expected.sum()),
           "factors": s.features[horizon],
           "units": {"events_expected": "events in the target year", "density": "events per 10 000 km² of land",
                     "p_any": "probability of at least one event", "p10": "events", "p50": "events", "p90": "events",
                     "area_km2": "km²", "observed": "events (UCDP GED) when the target year has data",
                     "world_events_expected": "events", "log_effect": "contribution to ln(expected events)"},
           "band": "p10-p90 of the negative binomial predictive distribution per country",
           "validation": bt, "exploratory": False, "caveats": list(CAVEATS)}
    if cells:
        cd = cell_density(pr)
        out["cells"] = {k: (np.round(cd[k].to_numpy(), 5).tolist() if cd[k].dtype.kind == "f" else cd[k].tolist())
                        for k in cd.columns}
    return out
