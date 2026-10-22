"""¿Qué relaciones históricas hay entre las variables de World3 y las del Atlas? Base empírica para acoplar L0 y L1.

Datos: el panel del Atlas (``panel_full.csv``: 180 países, 1950-2019) y sus series mundiales (precios reales de
materias primas de Jacks). Cada variable de World3 tiene un análogo observable por país:

    World3                     análogo en el panel
    producción industrial p.c. ln PIB per cápita (PWT) y su crecimiento
    alimento per cápita        rendimiento de cereales frente a su tendencia; precio real mundial de cereales
    recursos no renovables     rentas de recursos / PIB; agotamiento de recursos / RNB
    contaminación persistente  PM2.5 (desde 1990); CO2 per cápita
    esperanza de vida          esperanza de vida (WPP) y su cambio en 3 años
    huella ecológica           déficit ecológico per cápita (huella − biocapacidad)
    demografía                 crecimiento de la población, peso de los jóvenes 15-24

Partes:
  A. World3 → sociedad: probabilidades lineales con efectos fijos de país y año (errores agrupados por país) para
     inicio de conflicto civil, golpes, quiebre democrático, crisis de deuda soberana, salida irregular del líder,
     caída del líder por protestas e inicio de disputa militarizada entre Estados.
  B. Predicción fuera de muestra: logit entrenado en 1961-1999 y evaluado en 2000-2019 (AUC), con y sin las
     variables de World3, sobre una línea base al estilo del Atlas.
  C. Sociedad → World3: proyecciones locales (Jordà 2005) a 0-10 años de PIB per cápita, esperanza de vida,
     inversión, rendimiento de cereales y población tras un inicio de conflicto, un golpe exitoso, una crisis de
     deuda soberana y un quiebre democrático.
  D. «Valles» históricos: episodios de caída fuerte de la esperanza de vida o del PIB per cápita y la tasa de
     eventos sociales a su alrededor frente al resto de años.
  E. Formas no lineales: probabilidad de inicio de conflicto por tramos de crecimiento y de cambio de esperanza de vida.

Todo es correlación condicionada (con efectos fijos y rezagos), no causalidad identificada. Salida: ``resultados.json``.

    python reports/acople/investigacion.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.simplefilter("ignore")
HERE = Path(__file__).parent
OUT = HERE / "resultados.json"


def atlas_dir() -> Path:
    from world4.data import atlas
    return atlas.require() / "data" / "processed"


# ── preparación ──
def load() -> pd.DataFrame:
    d = atlas_dir()
    p = pd.read_csv(d / "panel_full.csv", low_memory=False).sort_values(["iso3", "year"]).reset_index(drop=True)
    w = pd.read_csv(d / "world_full.csv")
    cereal = w[["year", "recursos__jacks_wheat", "recursos__jacks_corn", "recursos__jacks_rice"]].set_index("year")
    cereal = np.log(cereal).mean(1)                                     # precio real medio de cereales (log)
    oil = np.log(w.set_index("year")["recursos__petroleo_precio_real_usd2019_bbl"])
    g = p.groupby("iso3")
    p["lny"] = np.log(p["gdppc"])
    p["growth"] = g["lny"].diff()
    p["growth3"] = p["lny"] - g["lny"].shift(3)
    p["dle3"] = p["life_expectancy"] - g["life_expectancy"].shift(3)
    ly = np.log(p["cereal_yield"].where(p["cereal_yield"] > 0))
    trend = ly.groupby(p["iso3"]).transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    p["yield_shock"] = ly - trend                                        # caída del rendimiento frente a su tendencia
    p["cereal_price_shock"] = p["year"].map(cereal.diff())               # Δ ln precio real mundial de cereales
    poor = (p["lny"] < p.groupby("year")["lny"].transform("median")).astype(float)
    p["cereal_price_x_poor"] = p["cereal_price_shock"] * poor
    oil_x = (p["renta_petroleo_pct_pib"].fillna(0) / 100) if "renta_petroleo_pct_pib" in p else 0
    p["oil_price_x_exporter"] = p["year"].map(oil.diff()) * oil_x
    p["depletion"] = p["depl_natres_pct_gni"] / 100
    p["ln_pm25"] = np.log(p["pm25"])
    p["ln_co2pc"] = np.log(p["co2_pc"].clip(lower=0.01))
    p["eco_deficit"] = -p["eco_deficit_pc"]                              # + = huella por encima de la biocapacidad
    p["youth"] = p["youth_bulge_15_24_adult"]
    p["ln_debt"] = np.log1p(p["debt_gdp"].clip(lower=0))
    p["temp_anom"] = p["tmp_anom_fao"]                                  # anomalía anual (°C); la _sd es un atributo fijo del país
    p["ln_dis"] = np.log1p(p["dis_n_climate"])
    p["democracy"] = (p["regime_row"] >= 2).astype(float).where(p["regime_row"].notna())
    # resultados (eventos en t)
    p["y_conflict_onset"] = p["onset"].where(g["conflict"].shift(1) == 0)
    p["y_coup"] = (p["coup_attempts"] > 0).astype(float).where(p["coup_attempts"].notna())
    p["y_dem_breakdown"] = p["dem_breakdown"].where(p.groupby("iso3")["democracy"].shift(1) == 1)
    first_sov = (p["crisis_sovdebt"] == 1) & (g["crisis_sovdebt"].shift(1) != 1)
    p["y_sovdebt_onset"] = first_sov.astype(float).where(p["crisis_sovdebt"].notna())
    p["y_irregular_exit"] = p["leader_irregular_exit"]
    p["y_protest_removal"] = p["leader_removed_protest"]
    p["y_interstate_mid"] = p["mid_onset"].clip(upper=1)
    p["y_interstate_fatal"] = p["mid_fatal"].clip(upper=1)
    for c in [c for c in p.columns if c.startswith("y_")]:      # algunas fuentes cuentan eventos: se binarizan
        p[c] = (p[c] > 0).astype(float).where(p[c].notna())
    return p


W3 = {   # análogos de World3 (rezagados un año en los modelos)
    "growth": "Crecimiento del PIB per cápita (año previo)",
    "lny": "Nivel de ingreso (ln PIB per cápita)",
    "dle3": "Cambio de la esperanza de vida en 3 años",
    "yield_shock": "Rendimiento de cereales frente a su tendencia",
    "cereal_price_x_poor": "Choque del precio mundial de cereales × país pobre",
    "res_rents": "Rentas de recursos / PIB",
    "oil_price_x_exporter": "Choque del precio del petróleo × exportador",
    "ln_co2pc": "ln CO2 per cápita",
    "eco_deficit": "Huella ecológica − biocapacidad (per cápita)",
    "youth": "Peso de los jóvenes 15-24",
    "pop_growth": "Crecimiento de la población",
    "temp_anom": "Anomalía anual de temperatura (°C)",
    "ln_dis": "Desastres climáticos (ln 1+n)",
}
OUTCOMES = {
    "y_conflict_onset": "Inicio de conflicto civil",
    "y_coup": "Intento de golpe",
    "y_dem_breakdown": "Quiebre democrático",
    "y_sovdebt_onset": "Crisis de deuda soberana",
    "y_irregular_exit": "Salida irregular del líder",
    "y_protest_removal": "Líder derrocado por protestas",
    "y_interstate_mid": "Disputa militarizada entre Estados",
}


def lagged(p: pd.DataFrame, cols) -> pd.DataFrame:
    return p.groupby("iso3")[list(cols)].shift(1)


# ── A. efectos fijos ──
def fe_lpm(p, outcome, preds):
    X = lagged(p, preds)
    df = pd.concat([p[["iso3", "year", outcome]], X], axis=1).dropna()
    df = df[df.groupby("iso3")[outcome].transform("count") >= 5]
    sd = df[list(preds)].std()
    Z = (df[list(preds)] - df[list(preds)].mean()) / sd                 # efectos por 1 DE
    D = pd.get_dummies(df["iso3"], drop_first=True, dtype=float)
    T = pd.get_dummies(df["year"], prefix="y", drop_first=True, dtype=float)
    M = sm.add_constant(pd.concat([Z, D, T], axis=1))
    r = sm.OLS(df[outcome].astype(float), M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
    base = df[outcome].mean()
    return {"n": int(len(df)), "countries": int(df["iso3"].nunique()), "base_rate": float(base),
            "years": [int(df["year"].min()), int(df["year"].max())],
            "coef": {k: {"b": float(r.params[k]), "se": float(r.bse[k]), "p": float(r.pvalues[k]),
                         "rel": float(r.params[k] / base) if base > 0 else None, "sd": float(sd[k])} for k in preds}}


# ── B. predicción fuera de muestra ──
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else None


def oos(p, outcome, base_preds, extra_preds, split=2000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    allp = list(dict.fromkeys(base_preds + extra_preds))
    X = lagged(p, allp)
    df = pd.concat([p[["iso3", "year", outcome]], X], axis=1)
    # historia del propio país: eventos en los 10 años previos (lo que usa el Atlas como persistencia)
    df["hist10"] = p.groupby("iso3")[outcome].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    df = df.dropna(subset=[outcome])
    tr, te = df[df["year"] < split], df[df["year"] >= split]
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "events_test": int(te[outcome].sum())}
    for name, cols in (("base", base_preds + ["hist10"]), ("base+World3", allp + ["hist10"])):
        a, b = tr.dropna(subset=cols), te.dropna(subset=cols)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(a[cols], a[outcome])
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                            random_state=0).fit(a[cols], a[outcome])
        res[name] = {"logit_auc": auc(b[outcome], lr.predict_proba(b[cols])[:, 1]),
                     "gbm_auc": auc(b[outcome], gb.predict_proba(b[cols])[:, 1]), "n_test": int(len(b))}
    # mismo conjunto de prueba para ambos (filas con todas las variables)
    a, b = tr.dropna(subset=allp + ["hist10"]), te.dropna(subset=allp + ["hist10"])
    for name, cols in (("base_same", base_preds + ["hist10"]), ("full_same", allp + ["hist10"])):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(a[cols], a[outcome])
        res[name] = {"logit_auc": auc(b[outcome], lr.predict_proba(b[cols])[:, 1]), "n_test": int(len(b))}
    return res


# ── C. proyecciones locales ──
def local_projection(p, shock, outcome, H=10, scale=1.0, log=False):
    g = p.groupby("iso3")
    y = np.log(p[outcome]) if log else p[outcome]
    y = y * scale
    out = []
    for h in range(0, H + 1):
        lead = y.groupby(p["iso3"]).shift(-h)
        base = y.groupby(p["iso3"]).shift(1)
        df = pd.DataFrame({"iso3": p["iso3"], "year": p["year"], "dy": lead - base, "shock": p[shock],
                           "dy_l1": base - y.groupby(p["iso3"]).shift(2), "dy_l2": y.groupby(p["iso3"]).shift(2) - y.groupby(p["iso3"]).shift(3),
                           "lny_l1": g["lny"].shift(1), "conf_l1": g["conflict"].shift(1)}).dropna()
        D = pd.get_dummies(df["iso3"], drop_first=True, dtype=float)
        T = pd.get_dummies(df["year"], prefix="y", drop_first=True, dtype=float)
        M = sm.add_constant(pd.concat([df[["shock", "dy_l1", "dy_l2", "lny_l1", "conf_l1"]], D, T], axis=1))
        r = sm.OLS(df["dy"], M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
        out.append({"h": h, "b": float(r.params["shock"]), "se": float(r.bse["shock"]), "n": int(len(df)),
                    "events": int(df["shock"].sum())})
    return out


# ── D. valles históricos ──
def valleys(p):
    g = p.groupby("iso3")
    le_drop = (g["life_expectancy"].transform(lambda s: s.rolling(5, min_periods=5).max()) - p["life_expectancy"]) >= 3
    y_peak = g["lny"].transform(lambda s: s.rolling(8, min_periods=4).max())
    gdp_drop = (y_peak - p["lny"]) >= np.log(1 / 0.8)                  # ≥ 20 % por debajo del máximo de 8 años
    p = p.assign(v_le=le_drop.astype(float), v_gdp=gdp_drop.astype(float))
    # el primer año de cada episodio
    for v in ("v_le", "v_gdp"):
        p[v + "_start"] = ((p[v] == 1) & (p.groupby("iso3")[v].shift(1) != 1)).astype(float)
    events = {"y_conflict_onset": "Inicio de conflicto civil", "y_coup": "Intento de golpe",
              "y_irregular_exit": "Salida irregular del líder", "y_dem_breakdown": "Quiebre democrático",
              "y_sovdebt_onset": "Crisis de deuda soberana", "y_protest_removal": "Líder derrocado por protestas"}
    res = {}
    for v, name in (("v_le", "Caída de ≥ 3 años de esperanza de vida (en 5 años)"),
                    ("v_gdp", "PIB per cápita ≥ 20 % bajo su máximo de 8 años")):
        inside = p[v] == 1
        ep = p[p[v + "_start"] == 1][["iso3", "year", "country"]]
        r = {"name": name, "country_years": int(inside.sum()), "episodes": int(len(ep)),
             "examples": ep.groupby("iso3").first().reset_index().assign(
                 dec=lambda d: (d.year // 10) * 10).sort_values("year")[["country", "year"]].values.tolist()[:60],
             "rates": {}}
        for e, en in events.items():
            a = p.loc[inside, e].mean()
            b = p.loc[~inside & p[e].notna(), e].mean()
            r["rates"][e] = {"name": en, "in_valley": float(a), "outside": float(b), "ratio": float(a / b) if b > 0 else None,
                             "n_in": int(p.loc[inside, e].notna().sum())}
        res[v] = r
    return res


# ── E. formas no lineales ──
def binned(p, x, y, bins):
    X = p.groupby("iso3")[x].shift(1)
    df = pd.DataFrame({"x": X, "y": p[y]}).dropna()
    cut = pd.cut(df["x"], bins)
    t = df.groupby(cut, observed=True)["y"].agg(["mean", "count"])
    return [{"bin": str(i), "rate": float(r["mean"]), "n": int(r["count"])} for i, r in t.iterrows()]


def main():
    t0 = time.time()
    p = load()
    out = {"meta": {"generated": time.strftime("%Y-%m-%d"), "countries": int(p["iso3"].nunique()),
                    "years": [int(p["year"].min()), int(p["year"].max())]},
           "predictors": W3, "outcomes": OUTCOMES}
    # A: dos especificaciones (sin PM2.5, 1961+; con PM2.5, 1991+)
    core = ["growth", "lny", "dle3", "yield_shock", "cereal_price_x_poor", "res_rents", "oil_price_x_exporter",
            "ln_co2pc", "youth", "pop_growth", "temp_anom", "ln_dis"]
    out["A"] = {o: fe_lpm(p, o, core) for o in OUTCOMES}
    out["A_pm25"] = {o: fe_lpm(p, o, ["growth", "lny", "dle3", "ln_pm25", "eco_deficit", "res_rents"])
                     for o in ("y_conflict_onset", "y_coup", "y_irregular_exit", "y_dem_breakdown")}
    print(f"A ok ({time.time() - t0:.0f} s)")
    base = ["lny", "growth", "democracy", "youth", "pop_growth", "ln_debt"]
    extra = ["dle3", "yield_shock", "cereal_price_x_poor", "res_rents", "oil_price_x_exporter", "ln_co2pc",
             "temp_anom", "ln_dis", "eco_deficit"]
    out["B"] = {o: oos(p, o, base, extra) for o in OUTCOMES}
    print(f"B ok ({time.time() - t0:.0f} s)")
    p["shock_conflict"] = p["y_conflict_onset"].fillna(0)
    p["shock_coup"] = p["coup_success"].clip(upper=1).fillna(0)
    p["shock_sovdebt"] = p["y_sovdebt_onset"].fillna(0)
    p["shock_breakdown"] = p["y_dem_breakdown"].fillna(0)
    lp = {}
    for s in ("shock_conflict", "shock_coup", "shock_sovdebt", "shock_breakdown"):
        lp[s] = {"gdppc": local_projection(p, s, "gdppc", log=True, scale=100),
                 "life_expectancy": local_projection(p, s, "life_expectancy"),
                 "csh_i": local_projection(p, s, "csh_i", scale=100),
                 "cereal_yield": local_projection(p, s, "cereal_yield", log=True, scale=100),
                 "pop": local_projection(p, s, "pop", log=True, scale=100)}
        print(f"C {s} ok ({time.time() - t0:.0f} s)")
    out["C"] = lp
    out["D"] = valleys(p)
    out["E"] = {
        "onset_by_growth": binned(p, "growth", "y_conflict_onset", [-1, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 1]),
        "onset_by_dle3": binned(p, "dle3", "y_conflict_onset", [-40, -3, -1, 0, 1, 2, 3, 40]),
        "coup_by_growth": binned(p, "growth", "y_coup", [-1, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 1]),
        "irregular_by_growth": binned(p, "growth", "y_irregular_exit", [-1, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 1]),
        "onset_by_income": binned(p, "lny", "y_conflict_onset", [0, 7, 8, 9, 10, 11, 13]),
        "onset_by_yield": binned(p, "yield_shock", "y_conflict_onset", [-5, -0.3, -0.15, -0.05, 0.05, 0.15, 5]),
    }
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0, default=float), encoding="utf-8")
    print(f"listo en {out['meta']['runtime_s']} s -> {OUT}")




# ── refinamientos: escalas dentro del país, un factor a la vez, FDR, valles sin guerra, dosis-respuesta ──
def within_sd(p, x):
    v = p[x] - p.groupby("iso3")[x].transform("mean")
    return float(v.std())


def bh(pvals):
    """q-valores de Benjamini-Hochberg."""
    p = np.nan_to_num(np.asarray(pvals, float), nan=1.0)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def single_driver(p, outcome, x, controls=("lny",)):
    """y_it = a_i + d_t + b x_{i,t-1} + controles_{t-1}; b en pp por 1 DE dentro del país."""
    cols = [x] + [c for c in controls if c != x]
    X = lagged(p, cols)
    df = pd.concat([p[["iso3", "year", outcome]], X], axis=1).dropna()
    D = pd.get_dummies(df["iso3"], drop_first=True, dtype=float)
    T = pd.get_dummies(df["year"], prefix="y", drop_first=True, dtype=float)
    M = sm.add_constant(pd.concat([df[cols], D, T], axis=1))
    r = sm.OLS(df[outcome].astype(float), M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
    wsd = within_sd(p, x)
    base = df[outcome].mean()
    return {"b_sd": float(r.params[x] * wsd), "se_sd": float(r.bse[x] * wsd), "p": float(r.pvalues[x]),
            "base": float(base), "rel": float(r.params[x] * wsd / base) if base > 0 else None, "n": int(len(df)),
            "within_sd": wsd}


def pooled_logit(p, outcome, x, controls=()):
    """Entre países y en el tiempo (sin efectos fijos de país, con efectos de década): odds ratio por 1 DE total."""
    cols = [x] + list(controls)
    X = lagged(p, cols)
    df = pd.concat([p[["iso3", "year", outcome]], X], axis=1).dropna()
    Z = (df[cols] - df[cols].mean()) / df[cols].std()
    dec = pd.get_dummies((df["year"] // 10).astype(int), prefix="d", drop_first=True, dtype=float)
    M = sm.add_constant(pd.concat([Z, dec], axis=1))
    try:
        r = sm.Logit(df[outcome].astype(float), M).fit(disp=0, cov_type="cluster",
                                                      cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
        return {"or_sd": float(np.exp(r.params[x])), "p": float(r.pvalues[x]), "n": int(len(df))}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def valleys_clean(p, H=5):
    """Episodios de caída (sin conflicto armado en los 3 años previos ni al inicio) y eventos en los H años
    SIGUIENTES al inicio, frente a la tasa de los países-año sin episodio."""
    g = p.groupby("iso3")
    out = {}
    defs = {
        "le": ("Caída de ≥ 3 años de esperanza de vida en 5 años",
               (g["life_expectancy"].transform(lambda s: s.rolling(5, min_periods=5).max()) - p["life_expectancy"]) >= 3),
        "gdp": ("PIB per cápita ≥ 20 % bajo su máximo de 8 años",
                (g["lny"].transform(lambda s: s.rolling(8, min_periods=4).max()) - p["lny"]) >= np.log(1 / 0.8)),
        "gdp10": ("PIB per cápita ≥ 10 % bajo su máximo de 8 años",
                  (g["lny"].transform(lambda s: s.rolling(8, min_periods=4).max()) - p["lny"]) >= np.log(1 / 0.9)),
    }
    war_prev = g["conflict"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max()).fillna(0)
    war_now = p["conflict"].fillna(0)
    interstate = p["ucdp_interstate"].fillna(0)
    events = {"y_conflict_onset": "Inicio de conflicto civil", "y_coup": "Intento de golpe",
              "y_irregular_exit": "Salida irregular del líder", "y_dem_breakdown": "Quiebre democrático",
              "y_sovdebt_onset": "Crisis de deuda soberana", "y_interstate_mid": "Disputa militarizada entre Estados"}
    for key, (name, flag) in defs.items():
        start = flag & ~flag.groupby(p["iso3"]).shift(1).fillna(False).astype(bool)
        clean = start & (war_prev == 0) & (war_now == 0) & (interstate == 0)
        res = {"name": name, "episodes_all": int(start.sum()), "episodes_no_war": int(clean.sum()),
               "examples_no_war": p.loc[clean, ["country", "year"]].values.tolist()[:80], "rates": {}}
        idx = np.flatnonzero(clean.values)
        for e, en in events.items():
            ys = p[e].values
            iso = p["iso3"].values
            after, ev, expo = [], 0.0, 0
            for i in idx:
                j = [k for k in range(i, min(i + H, len(p))) if iso[k] == iso[i]]
                vals = ys[j]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    after.append(vals.mean())
                    ev += vals.sum()
                    expo += len(vals)
            # tasa de referencia: países-año sin episodio y sin conflicto previo, mismos años (± 0)
            ref = p.loc[(~flag) & (war_prev == 0), e].mean()
            a = float(np.mean(after)) if after else float("nan")
            from scipy.stats import chi2
            lo = chi2.ppf(0.025, 2 * ev) / 2 / expo if ev > 0 else 0.0           # IC 95 % exacto de Poisson
            hi = chi2.ppf(0.975, 2 * ev + 2) / 2 / expo if expo else float("nan")
            a = ev / expo if expo else float("nan")                               # tasa agrupada eventos / años
            res["rates"][e] = {"name": en, "after": a, "reference": float(ref), "ratio": a / ref if ref > 0 else None,
                               "episodes": len(after), "events": float(ev), "exposure": int(expo),
                               "ratio_lo": lo / ref if ref > 0 else None, "ratio_hi": hi / ref if ref > 0 else None}
        out[key] = res
    return out


def depth_response(p, outcome, H=5):
    """Probabilidad del evento en los H años siguientes según la profundidad de la caída del PIB per cápita
    desde su máximo de 8 años (excluye países en conflicto en el año de medición)."""
    g = p.groupby("iso3")
    depth = 1 - np.exp(p["lny"] - g["lny"].transform(lambda s: s.rolling(8, min_periods=4).max()))
    fut = g[outcome].transform(lambda s: s.shift(-1).rolling(H, min_periods=1).max().shift(-(H - 1)))
    df = pd.DataFrame({"depth": depth, "fut": fut, "war": p["conflict"].fillna(0)}).dropna()
    df = df[df["war"] == 0]
    cut = pd.cut(df["depth"], [-0.01, 0.0001, 0.05, 0.1, 0.2, 0.3, 1.0])
    t = df.groupby(cut, observed=True)["fut"].agg(["mean", "count"])
    return [{"bin": str(i), "rate": float(r["mean"]), "n": int(r["count"])} for i, r in t.iterrows()]


def world_series(p):
    """Serie mundial: crecimiento medio, precio real de cereales y tasa de eventos por año (sin efectos fijos)."""
    d = atlas_dir()
    w = pd.read_csv(d / "world_full.csv").set_index("year")
    cereal = np.log(w[["recursos__jacks_wheat", "recursos__jacks_corn", "recursos__jacks_rice"]]).mean(1)
    agg = p.groupby("year").agg(growth=("growth", "median"), onset=("y_conflict_onset", "mean"),
                                coup=("y_coup", "mean"), irregular=("y_irregular_exit", "mean"),
                                protest=("y_protest_removal", "mean"))
    agg["cereal_shock"] = cereal.diff().reindex(agg.index)
    agg = agg.loc[1961:2015]
    res = {}
    for e in ("onset", "coup", "irregular", "protest"):
        for x in ("growth", "cereal_shock"):
            df = agg[[e, x]].copy()
            df["x1"] = df[x].shift(1)
            df = df.dropna()
            r = sm.OLS(df[e], sm.add_constant(df[["x1"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            res[f"{e}~{x}"] = {"b_sd": float(r.params["x1"] * df["x1"].std()), "p": float(r.pvalues["x1"]),
                               "corr": float(df[e].corr(df["x1"])), "n": int(len(df)), "mean": float(df[e].mean())}
    return res


def refine():
    t0 = time.time()
    p = load()
    out = json.loads(OUT.read_text(encoding="utf-8"))
    drivers = ["growth", "dle3", "yield_shock", "cereal_price_x_poor", "res_rents", "depletion", "oil_price_x_exporter",
               "ln_co2pc", "ln_pm25", "eco_deficit", "youth", "pop_growth", "temp_anom", "ln_dis", "ln_debt"]
    sd = {}
    for o in OUTCOMES:
        for x in drivers:
            sd[f"{o}|{x}"] = single_driver(p, o, x, ("lny",))
    keys = list(sd)
    q = bh([sd[k]["p"] for k in keys])
    for k, qq in zip(keys, q):
        sd[k]["q"] = float(qq)
    out["A2_single"] = sd
    out["A3_pooled"] = {f"{o}|{x}": pooled_logit(p, o, x) for o in OUTCOMES
                        for x in ("lny", "growth", "dle3", "res_rents", "youth", "eco_deficit", "ln_pm25", "temp_anom")}
    out["D2_clean"] = valleys_clean(p)
    out["D3_depth"] = {o: depth_response(p, o) for o in ("y_conflict_onset", "y_coup", "y_irregular_exit",
                                                         "y_dem_breakdown", "y_sovdebt_onset")}
    out["F_world"] = world_series(p)
    out["meta"]["refine_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0, default=float), encoding="utf-8")
    print(f"refinamiento listo en {out['meta']['refine_s']} s")


if __name__ == "__main__":
    import sys
    if "--solo-refinar" not in sys.argv:
        main()
    refine()
