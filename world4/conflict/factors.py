"""¿Qué factores predicen mejor los eventos de violencia organizada de un país? Estudio fuera de muestra con el
Atlas y World3 (``country_panel``).

Modelo: número de eventos de UCDP GED del país en el año t + h, con ln(superficie) como exposición (exp(X b) es
la densidad, eventos por 10 000 km²): media por PPML y binomial negativa (NB2) como distribución (``NB``). Validación: tres bloques de años objetivo
(``BLOCKS``); cada uno se predice con un modelo entrenado sólo con objetivos anteriores. Métrica principal: log
verosimilitud predictiva media por país-año (``ls``, en nats; mayor es mejor), que juzga la distribución entera
(ceros, colas); además AUPR de "al menos un evento" y RMSE de ln(1 + densidad).

Métrica principal: el RPS (ranked probability score) sobre órdenes de magnitud del número de eventos (``CLASSES``:
0, 1-9, 10-99, 100-999, >= 1000), propio y acotado por fila (0-4; menor es mejor). La log verosimilitud media se
reporta, pero no se usa para ordenar: la deciden unos pocos estallidos sorpresa (Ucrania 2014: 0,8 eventos
esperados, 1201 observados, pesa más que cientos de país-años), así que también se dan la mediana de su cambio por
fila y la parte de los país-años que mejoran.

Comparaciones (a 1 y 5 años):
  null              constante (la densidad media)
  historia          la historia del conflicto (``FACTORS['historia']``): la línea base
  historia + f      cada factor por separado, con su indicador de dato faltante
  historia + grupo  cada grupo entero
  todo              historia + todos los grupos del Atlas; y + World3 mundial
  selección         hacia delante desde la historia, añadiendo el factor que más mejora ``ls`` mientras mejore
El Atlas termina en 2019: el estudio usa objetivos <= 2019 y países del Atlas. Correlación predictiva, no causal.
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from world4.conflict import country_panel as cp

BLOCKS = ((2005, 2009), (2010, 2014), (2015, 2019))
CLASSES = (0, 9, 99, 999)                    # umbrales superiores de las clases 0, 1-9, 10-99, 100-999 (y >= 1000)
FIRST_FEATURE_YEAR = 1990
MISSING_FLAG_MIN = 0.03


# ---------------------------------------------------------------- preparación por pliegue
def prepare(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Imputa (último valor del país, que ya viene en el panel; luego la mediana del entrenamiento), añade
    indicadores de faltante donde el entrenamiento tiene >= 3 % y estandariza con el entrenamiento."""
    Xtr, Xte = train[cols].copy(), test[cols].copy()
    extra_tr, extra_te = {}, {}
    for c in cols:
        miss = Xtr[c].isna().mean()
        if miss >= MISSING_FLAG_MIN:
            extra_tr[f"{c}__na"] = Xtr[c].isna().astype(float)
            extra_te[f"{c}__na"] = Xte[c].isna().astype(float)
        med = Xtr[c].median()
        med = 0.0 if not np.isfinite(med) else med
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)
    Xtr = pd.concat([Xtr, pd.DataFrame(extra_tr, index=Xtr.index)], axis=1)
    Xte = pd.concat([Xte, pd.DataFrame(extra_te, index=Xte.index)], axis=1)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0).fillna(1.0)
    keep = Xtr.std() > 1e-9
    Xtr, Xte = (Xtr.loc[:, keep] - mu[keep]) / sd[keep], (Xte.loc[:, keep] - mu[keep]) / sd[keep]
    return Xtr, Xte, {"mu": mu[keep], "sd": sd[keep]}


class NB:
    """Conteos con exposición y una binomial negativa (NB2) como distribución predictiva.

    ``mean="ppml"`` (por omisión): los coeficientes de la media salen de la pseudo-máxima verosimilitud de
    Poisson (PPML, Santos Silva y Tenreyro 2006), cuyas ecuaciones cuadran el total de eventos en el
    entrenamiento por construcción; después la dispersión por máxima verosimilitud NB2 con esa media fija, que
    depende de la media (ln alpha = d0 + d1 ln mu): un país casi sin eventos y uno con miles no tienen la misma forma.
    ``mean="nb"``: NB2 por máxima verosimilitud conjunta, que pesa poco los conteos grandes (1 / (1 + alpha mu)) y
    descalibra el total cuando hay países con miles de eventos. Penalización L2 pequeña (``l2`` por fila) sobre
    predictores estandarizados en ambos casos."""

    def __init__(self, l2: float = 1e-3, mean: str = "ppml"):
        self.l2, self.mean_method = l2, mean

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray, max_iter: int = 200):
        if self.mean_method == "ppml":
            return self._fit_ppml(X, y, offset, max_iter)
        return self._fit_nb(X, y, offset, max_iter)

    def _fit_ppml(self, X, y, offset, max_iter):
        from scipy import optimize, special
        X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        n, k = X.shape
        lam = self.l2 * n
        y = np.asarray(y, float)
        b = np.zeros(k)
        b[0] = np.log(max(y.sum(), 1e-3) / np.exp(offset).sum())
        for _ in range(max_iter):                                  # Newton con retroceso
            eta = np.clip(X @ b + offset, -30, 30)
            mu = np.exp(eta)
            pen = np.r_[0.0, b[1:]]
            g = X.T @ (y - mu) - lam * pen
            H = X.T @ (X * mu[:, None]) + lam * np.diag(np.r_[1e-9, np.ones(k - 1)])
            step = np.linalg.solve(H, g)
            f0 = (y * eta - mu).sum() - 0.5 * lam * (b[1:] @ b[1:])
            t = 1.0
            while True:
                bn = b + t * step
                en = np.clip(X @ bn + offset, -30, 30)
                fn = (y * en - np.exp(en)).sum() - 0.5 * lam * (bn[1:] @ bn[1:])
                if fn >= f0 - 1e-10 * abs(f0) or t < 1e-6:
                    break
                t *= 0.5
            b = bn
            if np.max(np.abs(t * step)) < 1e-8:
                break
        self.b = b
        mu = np.exp(np.clip(X @ b + offset, -30, 30))

        lmu = np.log(mu)

        def nll(a):
            al = np.exp(np.clip(a[0] + a[1] * lmu, np.log(1e-4), np.log(1e3)))
            r = 1.0 / al
            return -(special.gammaln(y + r) - special.gammaln(r) + r * np.log(r / (r + mu))
                     + y * np.log(mu / (r + mu))).sum()

        res = optimize.minimize(nll, np.array([0.0, 0.0]), method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
        self.disp = res.x                                          # ln alpha = d0 + d1 ln mu
        self.alpha = float(np.exp(self.disp[0]))
        self.converged = True
        return self

    def alpha_of(self, mu):
        """Dispersión NB2 por fila: constante (``mean="nb"``) o ln alpha = d0 + d1 ln mu (``mean="ppml"``)."""
        if getattr(self, "disp", None) is None:
            return np.full(np.shape(mu), self.alpha)
        return np.exp(np.clip(self.disp[0] + self.disp[1] * np.log(mu), np.log(1e-4), np.log(1e3)))

    def _fit_nb(self, X, y, offset, max_iter):
        from scipy import optimize, special
        X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        n, k = X.shape
        lam = self.l2 * n
        ybar = max(y.mean(), 1e-3)
        x0 = np.zeros(k + 1)
        x0[0] = np.log(ybar) - offset.mean()
        x0[-1] = np.log(1.0)

        def f(th):
            b, la = th[:-1], th[-1]
            a = np.exp(la)
            eta = np.clip(X @ b + offset, -30, 30)
            mu = np.exp(eta)
            r = 1.0 / a
            ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu))
                  + y * np.log(mu / (r + mu)))
            pen = 0.5 * lam * (b[1:] @ b[1:])
            dmu = (y - mu) / (1 + a * mu)                       # d ll / d eta
            gb = X.T @ dmu - lam * np.r_[0.0, b[1:]]
            dr = special.digamma(y + r) - special.digamma(r) + np.log(r / (r + mu)) + (mu - y) / (r + mu)
            ga = (dr * (-r)).sum()                              # d ll / d ln a  (dr/d ln a = -r)
            return -(ll.sum() - pen), -np.r_[gb, ga]

        res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B",
                                bounds=[(None, None)] * k + [(np.log(1e-4), np.log(1e3))],
                                options={"maxiter": max_iter * 10, "gtol": 1e-6})
        self.b, self.alpha = res.x[:-1], float(np.exp(res.x[-1]))
        self.disp = None
        self.converged = bool(res.success)
        return self

    def mu(self, X, offset):
        X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        return np.exp(np.clip(X @ self.b + offset, -30, 30))

    def logscore(self, X, y, offset):
        mu = self.mu(X, offset)
        r = 1.0 / self.alpha_of(mu)
        return stats.nbinom.logpmf(y, r, r / (r + mu))

    def rps(self, X, y, offset):
        """RPS sobre las clases de ``CLASSES`` (por fila)."""
        mu = self.mu(X, offset)
        r = 1.0 / self.alpha_of(mu)
        F = np.stack([stats.nbinom.cdf(t, r, r / (r + mu)) for t in CLASSES], 1)
        O = np.stack([(y <= t).astype(float) for t in CLASSES], 1)
        return ((F - O) ** 2).sum(1)

    def p_any(self, X, offset):
        mu = self.mu(X, offset)
        r = 1.0 / self.alpha_of(mu)
        return 1.0 - (r / (r + mu)) ** r


# ---------------------------------------------------------------- evaluación
def _data(h: int, atlas_only: bool = True) -> pd.DataFrame:
    df = cp.with_targets(cp.build(), h)
    df = df[(df.year >= FIRST_FEATURE_YEAR) & df.y.notna() & (df.year + h <= cp.LAST_ATLAS_YEAR)]
    if atlas_only:
        df = df[df.in_atlas == 1]
    return df.reset_index(drop=True)


def evaluate(df: pd.DataFrame, cols: list[str], h: int, blocks=BLOCKS, mean: str = "ppml") -> dict:
    """Puntuaciones fuera de muestra de un conjunto de predictores (concatenando los bloques)."""
    from world4.conflict import metrics as M
    ls, rp, sq, py, yy = [], [], [], [], []
    for b0, b1 in blocks:
        tr = df[df.year + h < b0]
        te = df[(df.year + h >= b0) & (df.year + h <= b1)]
        if cols:
            Xtr, Xte, _ = prepare(tr, te, cols)
        else:
            Xtr, Xte = np.zeros((len(tr), 0)), np.zeros((len(te), 0))
        otr, ote = np.log(tr.area_km2.to_numpy() / cp.AREA_UNIT), np.log(te.area_km2.to_numpy() / cp.AREA_UNIT)
        m = NB(mean=mean).fit(np.asarray(Xtr), tr.y.to_numpy(), otr)
        ls.append(m.logscore(np.asarray(Xte), te.y.to_numpy(), ote))
        rp.append(m.rps(np.asarray(Xte), te.y.to_numpy(), ote))
        dens_hat = m.mu(np.asarray(Xte), ote) / np.exp(ote)
        sq.append((np.log1p(dens_hat) - np.log1p(te.y.to_numpy() / np.exp(ote))) ** 2)
        py.append(m.p_any(np.asarray(Xte), ote))
        yy.append(te.y.to_numpy() > 0)
    ls, rp, sq, py, yy = (np.concatenate(v) for v in (ls, rp, sq, py, yy))
    return {"rps": float(rp.mean()), "ls": float(ls.mean()), "_ls_rows": ls, "rmse_log_density": float(np.sqrt(sq.mean())),
            "aupr_any": M.average_precision(py, yy), "auroc_any": M.auroc(py, yy), "n": int(len(ls))}


def _vs(r: dict, base: dict) -> dict:
    """Compara con la línea base: ``skill`` = 1 - RPS / RPS base (positivo = mejor), y la log verosimilitud por
    fila (media, mediana del cambio y parte de los país-años que mejoran)."""
    d = r["_ls_rows"] - base["_ls_rows"]
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out.update({"skill": 1.0 - r["rps"] / base["rps"], "dls": float(d.mean()), "dls_median": float(np.median(d)),
                "ls_improved_share": float((d > 0).mean())})
    return out


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def screen(h: int, log=print, mean: str = "ppml") -> dict:
    df = _data(h)
    hist = list(cp.FACTORS["historia"])
    out = {"h": h, "rows": int(len(df)), "countries": int(df.iso3.nunique())}
    base = evaluate(df, hist, h, mean=mean)
    out["null"] = _vs(evaluate(df, [], h, mean=mean), base)
    out["historia"] = _clean(base)
    out["null_vs_historia"] = 1.0 - base["rps"] / evaluate(df, [], h, mean=mean)["rps"]
    log(f"h={h}: RPS null {out['null']['rps']:.4f}, historia {base['rps']:.4f}")
    single = {}
    for g, fs in cp.FACTORS.items():
        if g == "historia":
            continue
        for f in fs:
            single[f] = {"group": g, **_vs(evaluate(df, hist + [f], h, mean=mean), base)}
    out["single"] = single
    groups = {}
    for g, fs in cp.FACTORS.items():
        if g != "historia":
            groups[g] = _vs(evaluate(df, hist + list(fs), h, mean=mean), base)
    groups["análogos de World3 (por país)"] = _vs(evaluate(df, hist + sorted(set(cp.WORLD3_ANALOGS)), h, mean=mean), base)
    out["groups"] = groups
    # cada grupo quitado del modelo completo del Atlas: lo que sólo él aporta
    atlas_groups = [g for g in cp.FACTORS if g not in ("historia", "World3 mundial")]
    atlas_all = [f for g in atlas_groups for f in cp.FACTORS[g]]
    full = evaluate(df, hist + atlas_all, h, mean=mean)
    out["todo_atlas"] = _vs(full, base)
    out["todo_atlas_mas_world3"] = _vs(evaluate(df, hist + atlas_all + list(cp.FACTORS["World3 mundial"]), h, mean=mean), base)
    out["todo_atlas_mas_world3"]["vs_todo_atlas"] = 1.0 - out["todo_atlas_mas_world3"]["rps"] / full["rps"]
    drop = {}
    for g in atlas_groups:
        r = evaluate(df, hist + [f for f in atlas_all if f not in cp.FACTORS[g]], h, mean=mean)
        drop[g] = 1.0 - full["rps"] / r["rps"]          # cuánto empeora (en skill) sin el grupo
    out["unique_group_contribution"] = drop
    log(f"h={h}: skill todo el Atlas {out['todo_atlas']['skill']:+.3%}, + World3 mundial "
        f"{out['todo_atlas_mas_world3']['skill']:+.3%}")
    return out


def forward_select(h: int, max_steps: int = 14, min_rel_gain: float = 0.002, start=None, log=print,
                   mean: str = "ppml") -> dict:
    """Selección hacia delante por RPS fuera de muestra desde la historia del conflicto; para cuando el mejor
    candidato mejora el RPS en menos de ``min_rel_gain`` (relativo)."""
    df = _data(h)
    cols = list(start or cp.FACTORS["historia"])
    pool = [f for g, fs in cp.FACTORS.items() if g != "historia" for f in fs if f not in cols]
    base = evaluate(df, cols, h, mean=mean)
    cur = base
    path = [{"add": None, **_vs(cur, base)}]
    for step in range(max_steps):
        best, best_r = None, None
        for f in pool:
            r = evaluate(df, cols + [f], h, mean=mean)
            if best_r is None or r["rps"] < best_r["rps"]:
                best, best_r = f, r
        if best is None or 1.0 - best_r["rps"] / cur["rps"] < min_rel_gain:
            break
        gain = 1.0 - best_r["rps"] / cur["rps"]
        cols.append(best)
        pool.remove(best)
        cur = best_r
        path.append({"add": best, "group": cp.GROUP_OF[best], "gain": gain, **_vs(cur, base)})
        log(f"h={h} step {step + 1}: + {best} ({cp.GROUP_OF[best]}) RPS {cur['rps']:.4f} (-{gain:.2%})")
    return {"h": h, "selected": cols, "path": path}


def coefficients(cols: list[str], h: int, mean: str = "ppml") -> list[dict]:
    """El modelo final con todos los objetivos <= 2019: razón de tasas por desviación estándar (IRR) e intervalo
    de 90 % por bootstrap de países (el error que ignora la dependencia dentro del país sería demasiado estrecho)."""
    df = _data(h)
    X, _, meta = prepare(df, df.iloc[:0], cols)
    off = np.log(df.area_km2.to_numpy() / cp.AREA_UNIT)
    m = NB(mean=mean).fit(np.asarray(X), df.y.to_numpy(), off)
    rng = np.random.default_rng(0)
    countries = df.iso3.unique()
    idx_by = {c: np.nonzero(df.iso3.to_numpy() == c)[0] for c in countries}
    boots = []
    for _ in range(60):
        pick = np.concatenate([idx_by[c] for c in rng.choice(countries, len(countries))])
        mb = NB(mean=mean).fit(np.asarray(X)[pick], df.y.to_numpy()[pick], off[pick])
        boots.append(mb.b[1:])
    boots = np.array(boots)
    out = []
    for j, name in enumerate(X.columns):
        lo, hi = np.percentile(boots[:, j], [5, 95])
        base = name.split("__")[0]
        out.append({"factor": name, "group": cp.GROUP_OF.get(base, ""), "irr_per_sd": float(np.exp(m.b[j + 1])),
                    "irr_lo90": float(np.exp(lo)), "irr_hi90": float(np.exp(hi))})
    return out


def run(horizons=(1, 5), log=print, mean: str = "ppml") -> dict:
    warnings.simplefilter("ignore")
    res = {"meta": {"run_date": date.today().isoformat(), "blocks": [list(b) for b in BLOCKS],
                    "model": ("PPML mean, NB2 predictive distribution with ln alpha = d0 + d1 ln mu" if mean == "ppml" else
                              "NB2 by joint maximum likelihood") + "; ln(land area / 10 000 km²) offset; L2 1e-3 per row",
                    "estimator": mean,
                    "outcome": "UCDP GED 26.1 events of organized violence (all types) in the country in year t + h",
                    "metric": "ls = mean out-of-sample log predictive likelihood per country-year (nats; higher is better)",
                    "factors": cp.FACTORS, "world3_analogs": cp.WORLD3_ANALOGS}}
    for h in horizons:
        res[f"h{h}"] = scr = screen(h, log=log, mean=mean)
        sel = forward_select(h, log=log, mean=mean)
        scr["forward"] = sel
        scr["coefficients"] = coefficients(sel["selected"], h, mean=mean)
    return res


def save(res: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
