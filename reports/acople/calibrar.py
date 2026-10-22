"""Calibra el módulo social acoplado (``world4/core/social.py``) con datos históricos y lo valida.

Estima con el panel del Atlas (180 países, 1950-2019):

1. nueve modelos logit agrupados (efectos de década, errores agrupados por país) para los eventos sociales,
   con validación fuera de muestra (entrena < 2000, prueba 2000-2019) frente a un modelo sólo con el ingreso;
2. la ecuación de la deuda / PIB;
3. el ruido del ingreso relativo de cada país (AR(1) e innovaciones históricas por grupo de ingreso, sin
   años de evento);
4. las huellas en el PIB (proyecciones locales h = 0..10) de los inicios de conflicto, golpes exitosos y
   crisis de deuda, y la caída del gasto público tras ellos (la «capacidad»);
5. el estado inicial de 2019 y la demografía 2019-2100 de UN WPP 2024 (variante media).

Valida la simulación en la historia: arranca en 1999 con el estado observado, sigue el PIB mundial observado
2000-2019 y compara los eventos simulados por año con los observados, la frecuencia de valles y las
razones de eventos tras caídas del PIB ≥ 10 % (el resultado central del informe de dinámicas).

Salidas: ``world4/data/social_params.json`` (lo que lee el modelo) y ``calibracion.json`` (diagnósticos).

    PYTHONPATH=. python reports/acople/calibrar.py [--wpp RUTA/popprojAge1dt.rda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import investigacion as I  # noqa: E402

from world4.core import social  # noqa: E402

warnings.simplefilter("ignore")
ROOT = HERE.parents[1]
PARAMS = ROOT / "world4" / "data" / "social_params.json"
OUT = HERE / "calibracion.json"
WPP_PROJ_URL = "https://raw.githubusercontent.com/PPgp/wpp2024/main/data/popprojAge1dt.rda"
SEED = 20261022
NOISE_FROM = 1990          # las innovaciones del ingreso, de la era posterior a la Guerra Fría
FAR = social.FAR

HAZARDS = {   # predictores (todos en t-1) y muestra de riesgo
    "conflict_onset": (["lny", "growth", "depth", "youth", "conf10"], "paz en t-1"),
    "conflict_end": (["lny", "growth", "ln_dur"], "conflicto en t-1"),
    "coup": (["lny", "growth", "depth", "youth", "coup5", "conflict", "dem"], "todos"),
    "coup_success": ([], "años con intento"),
    "irregular_exit": (["lny", "growth", "depth", "youth", "coup5", "conflict", "dem"], "todos"),
    "dem_breakdown": (["lny", "growth", "depth"], "democracia en t-1"),
    "democratization": (["lny", "growth", "depth", "youth"], "autocracia en t-1"),
    "sovdebt_onset": (["lny", "growth", "depth", "ln_debt", "crisis10"], "sin crisis en t-1"),
    "sovdebt_end": ([], "crisis en t-1"),
}
NAMES = {"conflict_onset": "Inicio de conflicto civil", "conflict_end": "Fin del conflicto civil",
         "coup": "Intento de golpe", "coup_success": "Éxito del golpe (dado un intento)",
         "irregular_exit": "Salida irregular del líder", "dem_breakdown": "Quiebre democrático",
         "democratization": "Democratización", "sovdebt_onset": "Inicio de crisis de deuda soberana",
         "sovdebt_end": "Fin de la crisis de deuda"}


# ── panel con los mismos rasgos que calcula el simulador ──
def _since(flag: pd.Series, iso: pd.Series) -> pd.Series:
    out = np.empty(len(flag))
    last, prev = None, FAR
    for i, (f, c) in enumerate(zip(flag.values, iso.values)):
        if c != last:
            prev, last = FAR, c
        prev = 0.0 if f else min(prev + 1.0, FAR)
        out[i] = prev
    return pd.Series(out, index=flag.index)


def _duration(flag: pd.Series, iso: pd.Series) -> pd.Series:
    out = np.empty(len(flag))
    last, d = None, 0.0
    for i, (f, c) in enumerate(zip(flag.values, iso.values)):
        if c != last:
            d, last = 0.0, c
        d = d + 1.0 if f else 0.0
        out[i] = d
    return pd.Series(out, index=flag.index)


def panel() -> pd.DataFrame:
    p = I.load()
    g = p.groupby("iso3")
    p["depth"] = 1 - np.exp(p["lny"] - g["lny"].transform(lambda s: s.rolling(8, min_periods=4).max()))
    p["ln_debt"] = np.log1p(p["debt_gdp"].clip(lower=0))
    p["dem"] = p["democracy"]
    p["coup_any"] = (p["coup_attempts"] > 0).astype(float).where(p["coup_attempts"].notna())
    p["succ_any"] = (p["coup_success"] > 0).astype(float).where(p["coup_success"].notna())
    p["crisis"] = p["crisis_sovdebt"]
    iso = p["iso3"]
    p["since_conflict"] = _since(p["conflict"].fillna(0) > 0, iso)
    p["since_coup"] = _since(p["coup_any"].fillna(0) > 0, iso)
    p["since_coup_success"] = _since(p["succ_any"].fillna(0) > 0, iso)
    p["since_crisis"] = _since(p["crisis"].fillna(0) > 0, iso)
    p["cdur"] = _duration(p["conflict"].fillna(0) > 0, iso)
    p["conf10"] = (p["since_conflict"] <= 9).astype(float)
    p["coup5"] = (p["since_coup"] <= 4).astype(float)
    p["crisis10"] = (p["since_crisis"] <= 9).astype(float)
    p["ln_dur"] = np.log(np.maximum(p["cdur"], 1.0))
    lag = g[["lny", "growth", "depth", "youth", "ln_debt", "conflict", "dem", "conf10", "coup5", "crisis10",
             "ln_dur", "crisis"]].shift(1)
    for c in lag:
        p[c + "_l"] = lag[c]
    dem_prev = p["dem_l"]
    conf_prev = p["conflict_l"]
    cr_prev = p["crisis_l"]
    # resultados en t, cada uno en su muestra de riesgo
    p["o_conflict_onset"] = p["y_conflict_onset"].where(conf_prev == 0)
    p["o_conflict_end"] = (p["conflict"] == 0).astype(float).where(conf_prev == 1)
    p["o_coup"] = p["coup_any"]
    p["o_coup_success"] = p["succ_any"].where(p["coup_any"] == 1)
    p["o_irregular_exit"] = p["y_irregular_exit"]
    p["o_dem_breakdown"] = (p["dem"] == 0).astype(float).where((dem_prev == 1) & p["dem"].notna())
    p["o_democratization"] = (p["dem"] == 1).astype(float).where((dem_prev == 0) & p["dem"].notna())
    p["o_sovdebt_onset"] = p["y_sovdebt_onset"].where(cr_prev != 1)
    p["o_sovdebt_end"] = (p["crisis"] == 0).astype(float).where((cr_prev == 1) & p["crisis"].notna())
    return p


# ── 1. riesgos ──
def _design(df, vars_, decades):
    X = df[[v + "_l" for v in vars_]].copy()
    X.columns = vars_
    for d in decades[1:]:
        X[f"d{d}"] = (df["year"] // 10 * 10 == d).astype(float)
    return sm.add_constant(X, has_constant="add")


def fit_hazard(p, name, vars_, train=None, ref_decade=2010):
    o = "o_" + name
    df = p[["iso3", "year", o] + [v + "_l" for v in vars_]].dropna()
    df = df[df["year"] >= 1960]
    if train is not None:
        df = df[train(df)]
    if not vars_:                     # sólo una tasa: la de 1990-2019
        r = float(np.clip(df.loc[df["year"] >= 1990, o].mean(), 0.005, 0.995))
        return {"vars": [], "coef": [], "se": [], "const": float(np.log(r / (1 - r))), "ref_decade": None,
                "n": int(len(df)), "events": int(df[o].sum()), "rate": float(df[o].mean()), "rate_1990": r,
                "decades": []}
    decades = sorted((df["year"] // 10 * 10).unique())
    M = _design(df, vars_, decades)
    r = sm.Logit(df[o].astype(float), M).fit(disp=0, maxiter=200, cov_type="cluster",
                                             cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
    ref = min(ref_decade, decades[-1])
    const = float(r.params["const"] + (r.params.get(f"d{ref}", 0.0) if ref != decades[0] else 0.0))
    return {"vars": vars_, "coef": [float(r.params[v]) for v in vars_], "se": [float(r.bse[v]) for v in vars_],
            "const": const, "ref_decade": int(ref), "n": int(len(df)), "events": int(df[o].sum()),
            "rate": float(df[o].mean()), "decades": [int(d) for d in decades]}


def score(h, df):
    x = np.full(len(df), h["const"])
    for v, b in zip(h["vars"], h["coef"]):
        x = x + b * df[v + "_l"].values
    return 1 / (1 + np.exp(-x))


def hazards(p):
    out = {}
    for name, (vars_, sample) in HAZARDS.items():
        h = fit_hazard(p, name, vars_)
        h["sample"] = sample
        h["name"] = NAMES[name]
        # fuera de muestra: entrena < 2000 (intercepto de los 90), prueba 2000-2019
        tr = fit_hazard(p, name, vars_, train=lambda d: d["year"] < 2000, ref_decade=1990)
        tr0 = fit_hazard(p, name, ["lny"], train=lambda d: d["year"] < 2000, ref_decade=1990)
        o = "o_" + name
        te = p[(p["year"] >= 2000)][["iso3", "year", o] + [v + "_l" for v in vars_]].dropna()
        if vars_ and te[o].nunique() == 2:
            h["auc_oos"] = float(I.auc(te[o].values, score(tr, te)))
            h["auc_oos_lny"] = float(I.auc(te[o].values, score(tr0, te)))
            h["oos_pred_rate"] = float(score(tr, te).mean())
            h["oos_obs_rate"] = float(te[o].mean())
            h["oos_n"] = int(len(te))
        out[name] = h
        print(f"  {name:16s} n={h['n']:5d} ev={h['events']:4d} AUC fuera={h.get('auc_oos', float('nan')):.3f} "
              f"(sólo ingreso {h.get('auc_oos_lny', float('nan')):.3f})")
    return out


# ── 2. deuda ──
def debt_equation(p):
    g = p.groupby("iso3")
    df = pd.DataFrame({"iso3": p["iso3"], "d": p["ln_debt"] - g["ln_debt"].shift(1), "growth": p["growth"],
                       "ln_debt": g["ln_debt"].shift(1), "onset": p["y_sovdebt_onset"].fillna(0),
                       "year": p["year"]}).dropna()
    df = df[(df["year"] >= 1970) & (df["d"].abs() < 1.0)]
    M = sm.add_constant(df[["growth", "ln_debt", "onset"]])
    r = sm.OLS(df["d"], M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
    return {"const": float(r.params["const"]), "growth": float(r.params["growth"]),
            "ln_debt": float(r.params["ln_debt"]), "onset": float(r.params["onset"]),
            "se": {k: float(v) for k, v in r.bse.items()}, "n": int(len(df)), "r2": float(r.rsquared)}


# ── 3. ruido del ingreso relativo ──
def noise(p, rng):
    e = p["growth"] - p.groupby("year")["growth"].transform("mean")
    g = p.groupby("iso3")
    df = pd.DataFrame({"iso3": p["iso3"], "year": p["year"], "e": e, "e_l": e.groupby(p["iso3"]).shift(1),
                       "lny_l": p["lny_l"], "since_conflict": p["since_conflict"],
                       "since_cs": p["since_coup_success"], "since_crisis_onset": _since(p["y_sovdebt_onset"].fillna(0) > 0,
                                                                                         p["iso3"]),
                       "crisis": p["crisis"].fillna(0)}).dropna(subset=["e", "e_l", "lny_l"])
    clean = (df["since_conflict"] >= 6) & (df["since_cs"] >= 6) & (df["since_crisis_onset"] >= 6) & (df["crisis"] == 0)
    df = df[clean & (df["year"] >= NOISE_FROM)]
    rho = float(np.sum(df["e"] * df["e_l"]) / np.sum(df["e_l"] ** 2))
    u = df["e"] - rho * df["e_l"]
    keep = u.abs() < 0.4
    grp = np.digitize(df["lny_l"], social.GROUP_CUTS)
    pools, sd = [], []
    for k in range(3):
        x = u[keep & (grp == k)].values
        x = x - x.mean()
        sd.append(float(x.std()))
        pools.append([round(float(v), 5) for v in rng.choice(x, min(3000, len(x)), replace=False)])
    _ = g
    return {"rho": rho, "pools": pools, "sd": sd, "n": int(keep.sum()), "trimmed": int((~keep).sum())}


# ── 4. huellas en el PIB y capacidad ──
def impulses(p):
    shocks = {"conflict_onset": p["y_conflict_onset"].fillna(0),
              "coup_success": (p["coup_success"] > 0).astype(float),
              "sovdebt_onset": p["y_sovdebt_onset"].fillna(0)}
    imp, lp, cap = {}, {}, {}
    exp_mean = {}
    for k, s in shocks.items():
        p["_shock"] = s
        r = I.local_projection(p, "_shock", "gdppc", H=10, log=True, scale=100)
        lp[k] = [{"h": x["h"], "b": x["b"], "se": x["se"]} for x in r]
        imp[k] = [round(x["b"] / 100, 5) for x in r]
        f = I.local_projection(p, "_shock", "gov_exp_gdp", H=4)
        lp[k + "_gasto"] = [{"h": x["h"], "b": x["b"], "se": x["se"]} for x in f]
        prev = p.groupby("iso3")["gov_exp_gdp"].shift(1)[s > 0].mean()
        exp_mean[k] = float(prev)
        cap[k] = float(max(0.0, -np.mean([x["b"] for x in f]) / prev))
    p.drop(columns="_shock", inplace=True)
    return imp, lp, {"crisis": cap["sovdebt_onset"], "conflict": cap["conflict_onset"], "coup": cap["coup_success"],
                     "window": 5, "spending_mean": exp_mean}


# ── 5. estado inicial y demografía ──
def _wpp_age_estimates():
    d = np.load(ROOT / "data_cache" / "wpp2024" / "wpp_age.npz")
    return d["codes"], d["years"], d["popM"] + d["popF"]


def _wpp_projections(path):
    import rdata
    df = rdata.read_rda(str(path))["popprojAge1dt"]
    df = df[["country_code", "year", "age", "popM", "popF"]]
    df["pop"] = df["popM"] + df["popF"]
    df["year"] = df["year"].astype(int)
    df["country_code"] = df["country_code"].astype(int)
    return df


def demography(countries, years, wpp_path=None):
    """Población (miles) y peso de los jóvenes 15-24 en la población de 15+ por país y año: estimaciones de
    WPP 2024 hasta 2023 y su proyección media después."""
    codes, eyears, P = _wpp_age_estimates()
    idx = {int(c): i for i, c in enumerate(codes)}
    pop = np.full((len(countries), len(years)), np.nan)
    youth = np.full_like(pop, np.nan)
    proj = None
    if max(years) > eyears.max():
        if wpp_path is None:
            dl = ROOT / "data_cache" / "wpp2024" / "raw"
            dl.mkdir(parents=True, exist_ok=True)
            wpp_path = dl / "popprojAge1dt.rda"
            if not wpp_path.exists():
                print("  descargando proyecciones de WPP 2024 ...")
                urllib.request.urlretrieve(WPP_PROJ_URL, wpp_path)
        proj = _wpp_projections(wpp_path)
        proj = proj.pivot_table(index=["country_code", "year"], columns="age", values="pop", aggfunc="sum")
    for i, iso in enumerate(countries):
        for j, y in enumerate(years):
            if y <= eyears.max() and iso in idx:
                a = P[idx[iso], int(y - eyears[0])]
            elif proj is not None and (iso, y) in proj.index:
                a = proj.loc[(iso, y)].values
            else:
                continue
            pop[i, j] = a.sum()
            youth[i, j] = a[15:25].sum() / a[15:].sum()
    return pop, youth


def initial_state(p, year):
    """Estado de cada país en ``year`` con lo que el panel conoce hasta entonces."""
    q = p[p["year"] <= year]
    rows = {k: [] for k in ("iso3", "isonum", "name", "lny", "e", "ln_debt", "conflict", "cdur", "dem", "crisis",
                            "since_conflict", "since_coup", "since_crisis", "since_coup_success")}
    e_all = p["growth"] - p.groupby("year")["growth"].transform("mean")
    med_debt = float(q.loc[q["year"] == year, "ln_debt"].median())
    for iso, d in q.groupby("iso3"):
        d = d.sort_values("year")
        if d["year"].max() != year or not np.isfinite(d["lny"].iloc[-1]):
            continue
        ly = d.set_index("year")["lny"].reindex(range(year - 7, year + 1)).bfill().ffill()
        if ly.isna().any():
            continue
        last = lambda c: d[c].dropna().iloc[-1] if d[c].notna().any() else np.nan  # noqa: E731
        rows["iso3"].append(iso)
        rows["isonum"].append(int(d["isonum"].iloc[-1]))
        rows["name"].append(str(d["country"].iloc[-1]))
        rows["lny"].append([round(float(v), 5) for v in ly.values])
        ev = e_all.loc[d.index].iloc[-1]
        rows["e"].append(float(np.clip(ev if np.isfinite(ev) else 0.0, -0.1, 0.1)))
        ld = last("ln_debt")
        rows["ln_debt"].append(float(ld) if np.isfinite(ld) else med_debt)
        rows["conflict"].append(float(d["conflict"].iloc[-1] > 0))
        rows["cdur"].append(float(d["cdur"].iloc[-1]))
        dm = last("dem")
        rows["dem"].append(float(dm) if np.isfinite(dm) else 0.0)
        cr = last("crisis")
        rows["crisis"].append(float(cr) if np.isfinite(cr) else 0.0)
        for k in ("since_conflict", "since_coup", "since_crisis", "since_coup_success"):
            rows[k].append(float(d[k].iloc[-1]))               # los años sin dato cuentan como años sin evento
    return rows


# ── validación en la historia ──
def history_check(p, params, R=200, y0=1999, y1=2019, seed=SEED):
    ini = initial_state(p, y0)
    keep = [params["countries"]["iso3"].index(c) if c in params["countries"]["iso3"] else -1 for c in ini["iso3"]]
    years = list(range(y0, y1 + 1))
    q = p[p["iso3"].isin(ini["iso3"])]
    pv = q.pivot(index="iso3", columns="year", values="pop").reindex(ini["iso3"])[years].ffill(axis=1).bfill(axis=1)
    yv = q.pivot(index="iso3", columns="year", values="youth").reindex(ini["iso3"])[years].ffill(axis=1).bfill(axis=1)
    demog = {"years": years, "pop": pv.values.tolist(), "youth": yv.values.tolist()}
    gdp = q.pivot(index="iso3", columns="year", values="gdppc").reindex(ini["iso3"])[years]
    world = (gdp * pv).sum() / pv.where(gdp.notna()).sum()                     # PIB mundial per cápita observado
    soc = social.Society(params, R, seed=seed, init={**ini, "year": y0}, demog=demog)
    soc.ypc_prev = np.full(R, world[y0])
    sim = {k: [] for k in ("conflict_onset", "coup", "coup_success", "irregular_exit", "dem_breakdown",
                           "democratization", "sovdebt_onset", "in_conflict", "demo_countries", "valley10")}
    LNY, EV = [], {k: [] for k in ("conflict_onset", "coup", "dem_breakdown", "sovdebt_onset", "irregular_exit")}
    CONF = []
    for y in years[1:]:
        _, _, rec, _, _ = soc.step(y, np.full(R, world[y]))
        for k in sim:
            sim[k].append(rec[k] if k != "valley10" else rec["countries_valley10"])
        LNY.append(soc.lny.copy())
        CONF.append(soc.conflict.copy())
        for k in EV:
            EV[k].append(soc.last[k].copy())
    # observado, mismos países
    obs = {}
    o = q[(q["year"] > y0) & (q["year"] <= y1)]
    for k, col in (("conflict_onset", "o_conflict_onset"), ("coup", "o_coup"), ("coup_success", "succ_any"),
                   ("irregular_exit", "o_irregular_exit"), ("dem_breakdown", "o_dem_breakdown"),
                   ("democratization", "o_democratization"), ("sovdebt_onset", "o_sovdebt_onset")):
        s = o.groupby("year")[col].agg(["sum", "count"])
        s.loc[s["count"] < 0.5 * s["count"].max(), "sum"] = np.nan        # años con la fuente casi vacía
        obs[k] = [float(s["sum"].get(y, np.nan)) for y in years[1:]]
    obs["in_conflict"] = [float(o.loc[o["year"] == y, "conflict"].sum()) for y in years[1:]]
    obs["demo_countries"] = [float(o.loc[o["year"] == y, "dem"].fillna(0).sum() / len(ini["iso3"])) for y in years[1:]]
    obs["valley10"] = [float((o.loc[o["year"] == y, "depth"] >= 0.1).sum()) for y in years[1:]]
    out = {"years": years[1:], "countries": len(ini["iso3"]), "members": R,
           "sim": {k: {"mean": np.mean(np.stack(v), 1).tolist(), "p10": np.percentile(np.stack(v), 10, 1).tolist(),
                       "p90": np.percentile(np.stack(v), 90, 1).tolist()} for k, v in sim.items()},
           "obs": obs, "totals": {}}
    for k in sim:
        sv = np.stack(sim[k]).mean(1)
        ov = np.asarray(obs[k], float)
        m = np.isfinite(ov)
        out["totals"][k] = {"sim": float(sv[m].sum() if k not in ("demo_countries",) else sv[m].mean()),
                            "obs": float(ov[m].sum() if k not in ("demo_countries",) else ov[m].mean()),
                            "years": int(m.sum())}
    # razones de eventos tras caídas ≥ 10 % en el panel simulado (sin guerra en los 3 años previos), H = 5
    LNY, CONF = np.stack(LNY), np.stack(CONF)                     # (T, R, N)
    T = LNY.shape[0]
    hist0 = np.repeat(np.asarray(ini["lny"], float).T[:, None, :], R, 1)
    full = np.concatenate([hist0, LNY], 0)                       # (8 + T, R, N)
    depth = np.stack([1 - np.exp(full[t + 8] - full[t + 1:t + 9].max(0)) for t in range(T)])
    flag = depth >= 0.1
    start = flag & ~np.concatenate([np.zeros_like(flag[:1]), flag[:-1]], 0)
    c0 = np.asarray(ini["conflict"], bool)[None, None].repeat(R, 1)
    confs = np.concatenate([c0, CONF], 0)
    war_prev = np.stack([confs[max(0, t - 2):t + 1].any(0) for t in range(T)])
    clean = start & ~war_prev & ~CONF
    ratios = {}
    for k in EV:
        E = np.stack(EV[k]).astype(float)
        ev, expo = 0.0, 0
        for t in range(T - 1):
            m = clean[t]
            if m.any():
                ev += E[t:t + 5][:, m].sum()
                expo += m.sum() * min(5, T - t)
        ref = E[~flag & ~war_prev].mean()
        ratios[k] = {"after": ev / max(expo, 1), "reference": float(ref), "ratio": float(ev / max(expo, 1) / ref),
                     "episodes": int(clean[:T - 1].sum())}
    out["valley_ratios"] = ratios
    emp = json.loads((HERE / "resultados.json").read_text())["D2_clean"]["gdp10"]["rates"]
    out["valley_ratios_empirical"] = {k: emp["y_" + k]["ratio"] for k in ratios if "y_" + k in emp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpp", default=None, help="popprojAge1dt.rda ya descargado")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    p = panel()
    print("riesgos:")
    hz = hazards(p)
    debt = debt_equation(p)
    print(f"deuda: Δln(1+d) = {debt['const']:.3f} {debt['growth']:+.3f}·g {debt['ln_debt']:+.3f}·ln(1+d) "
          f"{debt['onset']:+.3f}·crisis")
    nz = noise(p, rng)
    print(f"ruido: rho={nz['rho']:.3f} sd por grupo={[round(s, 4) for s in nz['sd']]}")
    imp, lp, cap = impulses(p)
    print("huellas a 10 años:", {k: v[-1] for k, v in imp.items()}, "capacidad:", {k: round(v, 3) for k, v in cap.items()
                                                                                  if k != "spending_mean"})
    ini = initial_state(p, 2019)
    years = list(range(2019, 2101))
    pop, youth = demography(ini["isonum"], years, args.wpp)
    ok = np.isfinite(pop).all(1) & np.isfinite(youth).all(1)
    print(f"países: {len(ini['iso3'])} con estado 2019, {int(ok.sum())} con WPP completo")
    sel = np.flatnonzero(ok)
    ini = {k: [v[i] for i in sel] if isinstance(v, list) else v for k, v in ini.items()}
    from world4.core.l0r import regions
    reg = regions.region_index(ini["iso3"], ini["isonum"]).tolist()
    params = {
        "meta": {"generated": time.strftime("%Y-%m-%d"), "source": "Atlas del Sistema-Mundo (panel 1950-2019); "
                 "UN WPP 2024 (estimaciones 1950-2023 y variante media 2024-2100)", "seed": SEED,
                 "script": "reports/acople/calibrar.py"},
        "hazards": hz, "debt": debt, "noise": nz, "impulses": imp, "capacity": cap,
        "countries": {"iso3": ini["iso3"], "isonum": ini["isonum"], "name": ini["name"], "region": reg,
                      "region_codes": list(regions.CODES)},
        "init": {"year": 2019, **{k: ini[k] for k in ("lny", "e", "ln_debt", "conflict", "cdur", "dem", "crisis",
                                                       "since_conflict", "since_coup", "since_crisis",
                                                       "since_coup_success")}},
        "demog": {"years": years, "pop": np.round(pop[sel], 1).tolist(), "youth": np.round(youth[sel], 5).tolist()},
    }
    PARAMS.write_text(json.dumps(params, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    social.load_params.cache_clear()
    print(f"-> {PARAMS} ({PARAMS.stat().st_size / 1e6:.2f} MB)")
    print("validación 2000-2019 ...")
    hc = history_check(p, params)
    for k, v in hc["totals"].items():
        print(f"  {k:16s} simulado {v['sim']:8.2f}  observado {v['obs']:8.2f}  ({v['years']} años)")
    for k, v in hc["valley_ratios"].items():
        print(f"  tras caída ≥10 %: {k:16s} simulado ×{v['ratio']:.2f}  empírico ×{hc['valley_ratios_empirical'].get(k, float('nan')):.2f}")
    diag = {"meta": params["meta"], "hazards": hz, "debt": debt, "noise": {k: v for k, v in nz.items() if k != "pools"},
            "lp": lp, "capacity": cap, "history": hc, "runtime_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listo en {diag['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
