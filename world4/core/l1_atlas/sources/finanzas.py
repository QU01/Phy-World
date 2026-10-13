"""Deuda, finanzas y crisis (modulo "finanzas").

Construye un panel pais-anio (ISO3) con deuda publica, deuda externa, servicio de la deuda,
ingresos/gastos fiscales, cuenta corriente, reservas, tipos de interes, credito privado y
crisis (bancarias, cambiarias, de deuda soberana), mas series mundiales (tipos de EE.UU.,
indice del dolar, proporcion de paises en crisis).

Fuentes (todas descargadas via raw.githubusercontent.com / media.githubusercontent.com, porque
imf.org, worldbank.org y macrohistory.net estan bloqueados en este entorno):

* Global Macro Database (GMD, Mueller, Xu, Lehbib & Chen 2025), release 2025_12, ficheros
  por variable en github.com/KMueller-Lab/Global-Macro-Database-Stata/data/final.
  Deuda (combina IMF HPDD/Mauro et al., WEO, JST, Reinhart-Rogoff...), ingresos/gastos,
  tipos, cuenta corriente, REER y crisis (Laeven-Valencia con relleno Reinhart-Rogoff;
  las dummies de crisis de GMD marcan el ANIO DE INICIO).
* World Development Indicators (Banco Mundial) via el paquete DDF de open-numbers
  (github.com/open-numbers/ddf--open_numbers--world_development_indicators): deuda externa,
  servicio de la deuda, reservas, IED, credito privado, intereses/ingresos.
* IMF Historical Public Debt Database (Abbas et al. 2010; Mauro et al. 2015), copia en
  github.com/unbalancedparentheses/forex-centuries (1800-2015, ISO2).
* Laeven & Valencia (2018) "Systemic Banking Crises Revisited" (episodios con perdidas de
  producto y coste fiscal), copia en github.com/muzammilafroz/EWS; y la actualizacion
  Laeven & Valencia (2026) 1970-2025 (tabla A1 extraida) en github.com/MMJGGR/banking-stability-copilot.
* Jorda-Schularick-Taylor Macrohistory Database R6 (18 economias avanzadas, credito/PIB,
  tipos, crisisJST), copia LFS en github.com/axfreeman/MacroEconomic-History-Server-Builder.
* FRED DTWEXM / DTWEXBGS (indices del dolar) via forex-centuries.

Salidas:
  data/processed/sources/finanzas.csv        (iso3, year, variables; ratios como fracciones)
  data/processed/sources/finanzas_world.csv  (year, variables mundiales)
  data/processed/sources/finanzas_dict.json  (diccionario de variables)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "finanzas"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"
Y0, Y1 = 1950, 2024

GH = "https://raw.githubusercontent.com"
GMD = f"{GH}/KMueller-Lab/Global-Macro-Database-Stata/HEAD/data/final/{{v}}_2025_12.csv"
WDI = f"{GH}/open-numbers/ddf--open_numbers--world_development_indicators/HEAD/datapoints/ddf--datapoints--{{v}}--by--geo--time.csv"
FC = f"{GH}/unbalancedparentheses/forex-centuries/HEAD/data/sources"

# GMD variable -> (output name, scale, description)
GMD_VARS = {
    "govdebt_GDP": ("debt_gdp", 0.01, "Deuda publica bruta / PIB (serie combinada GMD: gobierno general, si no central)"),
    "gen_govdebt_GDP": ("debt_gdp_gen", 0.01, "Deuda bruta del gobierno general / PIB"),
    "cgovdebt_GDP": ("debt_gdp_cgov", 0.01, "Deuda del gobierno central / PIB"),
    "govrev_GDP": ("gov_rev_gdp", 0.01, "Ingresos publicos / PIB (combinado)"),
    "govexp_GDP": ("gov_exp_gdp", 0.01, "Gasto publico / PIB (combinado)"),
    "gen_govtax_GDP": ("gov_tax_gdp", 0.01, "Ingresos tributarios del gobierno general / PIB"),
    "govdef_GDP": ("gov_balance_gdp", 0.01, "Saldo fiscal / PIB (negativo = deficit)"),
    "CA_GDP": ("ca_gdp", 0.01, "Saldo de cuenta corriente / PIB"),
    "ltrate": ("ltrate", 0.01, "Tipo de interes a largo plazo (bono ~10 anios), nominal"),
    "strate": ("strate", 0.01, "Tipo de interes a corto plazo, nominal"),
    "cbrate": ("cbrate", 0.01, "Tipo oficial del banco central, nominal"),
    "infl": ("infl", 0.01, "Inflacion IPC anual"),
    "REER": ("reer", 1.0, "Tipo de cambio efectivo real (indice 2015=100; sube = apreciacion)"),
    "USDfx": ("usdfx", 1.0, "Unidades de moneda local por USD"),
    "nGDP_USD": ("ngdp_usd", 1.0, "PIB nominal en millones de USD (ponderaciones)"),
    "BankingCrisis": ("crisis_bank", 1.0, "Inicio de crisis bancaria (GMD: Laeven-Valencia, relleno Reinhart-Rogoff)"),
    "CurrencyCrisis": ("crisis_currency", 1.0, "Inicio de crisis cambiaria (GMD: Laeven-Valencia, relleno Reinhart-Rogoff)"),
    "SovDebtCrisis": ("crisis_sovdebt", 1.0, "Inicio de crisis de deuda soberana (impago/reestructuracion; GMD: LV + RR)"),
}
CRISIS = {"BankingCrisis", "CurrencyCrisis", "SovDebtCrisis"}  # ~12 MB each (long notes) -> large/

# WDI indicator (DDF id) -> (output name, scale, description)
WDI_VARS = {
    "dt_dod_dect_gn_zs": ("ext_debt_gni", 0.01, "Deuda externa total / INB (WDI DT.DOD.DECT.GN.ZS)"),
    "dt_tds_dect_gn_zs": ("debt_service_gni", 0.01, "Servicio total de la deuda externa / INB (DT.TDS.DECT.GN.ZS)"),
    "dt_tds_dect_ex_zs": ("debt_service_exports", 0.01, "Servicio de la deuda externa / exportaciones+renta primaria (DT.TDS.DECT.EX.ZS)"),
    "dt_dod_dstc_ir_zs": ("st_debt_reserves", 0.01, "Deuda externa a corto plazo / reservas (DT.DOD.DSTC.IR.ZS)"),
    "fi_res_totl_mo": ("reserves_months", 1.0, "Reservas totales en meses de importaciones (FI.RES.TOTL.MO)"),
    "fi_res_totl_cd": ("reserves_usd", 1.0, "Reservas totales incl. oro, USD corrientes (FI.RES.TOTL.CD)"),
    "bx_klt_dinv_wd_gd_zs": ("fdi_in_gdp", 0.01, "Entradas netas de IED / PIB (BX.KLT.DINV.WD.GD.ZS)"),
    "fs_ast_prvt_gd_zs": ("credit_priv_gdp", 0.01, "Credito interno al sector privado / PIB (FS.AST.PRVT.GD.ZS)"),
    "gc_xpn_intp_rv_zs": ("interest_rev", 0.01, "Pagos de intereses / ingresos del gobierno (GC.XPN.INTP.RV.ZS)"),
    "gc_dod_totl_gd_zs": ("debt_gdp_cgov_wdi", 0.01, "Deuda del gobierno central / PIB (GC.DOD.TOTL.GD.ZS)"),
    "fr_inr_rinr": ("real_rate_wdi", 0.01, "Tipo de interes real (prestamos, deflactor PIB) (FR.INR.RINR)"),
    "bn_cab_xoka_gd_zs": ("ca_gdp_wdi", 0.01, "Cuenta corriente / PIB (BN.CAB.XOKA.GD.ZS)"),
    "ny_gdp_mktp_kd_zg": ("g_real_wdi", 0.01, "Crecimiento del PIB real (NY.GDP.MKTP.KD.ZG)"),
}

OTHER = {
    "hpdd_debt_gdp.csv": f"{FC}/imf_hpdd/imf_hpdd_debt_gdp.csv",
    "fred_usd_major_index.csv": f"{FC}/fred/daily/fred_usd_major_index.csv",
    "fred_usd_broad_index.csv": f"{FC}/fred/daily/fred_usd_broad_index.csv",
    "laeven_valencia_2018_episodes.csv": f"{GH}/muzammilafroz/EWS/HEAD/LaevenValencia2018_sysBankingCrisesDB.csv",
    "laeven_valencia_2026_banking.csv": f"{GH}/MMJGGR/banking-stability-copilot/HEAD/data/reference/systemic_banking_crises_1970_2025.csv",
}
JST_URL = ("https://media.githubusercontent.com/media/axfreeman/MacroEconomic-History-Server-Builder/"
           "HEAD/DATA/SOURCE/ORIGINALS/JST/JSTdatasetR6.xlsx")


# ----------------------------------------------------------------------------- download
def _get(url: str, dst: Path, tries: int = 6) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(tries):
        try:
            done = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if done else "wb") as f:
                if done and r.status != 206:  # server ignored Range: restart
                    f.seek(0), f.truncate()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            head = tmp.read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                raise RuntimeError(f"git-lfs pointer: {url}")
            tmp.rename(dst)
            return dst
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001  (conexiones cortadas por el proxy)
            print(f"  retry {k + 1}/{tries} {dst.name}: {e}")
            time.sleep(2 + 2 * k)
    raise RuntimeError(f"no se pudo descargar {url}")


def download() -> None:
    for v in GMD_VARS:
        d = (LARGE if v in CRISIS else RAW / "gmd") / f"{v}_2025_12.csv"
        _get(GMD.format(v=v), d)
    for v in WDI_VARS:
        _get(WDI.format(v=v), RAW / "wdi" / f"{v}.csv")
    for name, url in OTHER.items():
        _get(url, RAW / name)
    try:
        _get(JST_URL, RAW / "JSTdatasetR6.xlsx")
    except RuntimeError as e:
        print("  JST no disponible:", e)


# ----------------------------------------------------------------------------- readers
def _gmd() -> pd.DataFrame:
    out = None
    for v, (name, scale, _) in GMD_VARS.items():
        p = (LARGE if v in CRISIS else RAW / "gmd") / f"{v}_2025_12.csv"
        d = pd.read_csv(p, usecols=["ISO3", "year", v]).rename(columns={"ISO3": "iso3", v: name})
        d[name] = d[name] * scale
        out = d if out is None else out.merge(d, on=["iso3", "year"], how="outer")
    return out


def _wdi() -> pd.DataFrame:
    out = None
    for v, (name, scale, _) in WDI_VARS.items():
        d = pd.read_csv(RAW / "wdi" / f"{v}.csv")
        d = d.rename(columns={"geo": "iso3", "time": "year", v: name})
        d["iso3"] = d["iso3"].str.upper()
        d[name] = d[name] * scale
        out = d if out is None else out.merge(d, on=["iso3", "year"], how="outer")
    return out


def _hpdd() -> pd.DataFrame:
    import country_converter as coco
    d = pd.read_csv(RAW / "hpdd_debt_gdp.csv", keep_default_na=False, na_values=[""])
    codes = d["country"].unique()
    iso3 = dict(zip(codes, coco.convert(list(codes), src="ISO2", to="ISO3", not_found=None)))
    d["iso3"] = d["country"].map(iso3)
    d = d[d["iso3"].str.len() == 3]
    return d.assign(debt_gdp_hpdd=d["value"] / 100)[["iso3", "year", "debt_gdp_hpdd"]]


def _lv_episodes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Laeven-Valencia: episodios 2018 (con perdidas) y actualizacion 2026 (1970-2025)."""
    import country_converter as coco
    e = pd.read_csv(RAW / "laeven_valencia_2018_episodes.csv")
    e = e[e["fsysbank"] > 0].dropna(subset=["startyear"])
    names = e["isocodes"].unique()
    iso = dict(zip(names, coco.convert(list(names), to="ISO3", not_found=None)))
    e["iso3"] = e["isocodes"].map(iso)
    e["year"] = e["startyear"].astype(int)
    ep = pd.DataFrame({
        "iso3": e["iso3"], "year": e["year"],
        "lv_output_loss": e["yloss"] / 100,          # perdida acumulada de PIB vs tendencia (% PIB tendencial)
        "lv_fiscal_cost": e["fcost_v1"] / 100,       # coste fiscal bruto directo (% PIB)
        "lv_peak_npl": e["maxnpl"] / 100,            # NPL maximos (% prestamos)
        "lv_debt_increase": e["gsdebt"] / 100,       # aumento deuda publica (% PIB)
    }).groupby(["iso3", "year"], as_index=False).first()

    n = pd.read_csv(RAW / "laeven_valencia_2026_banking.csv")
    n = n[n["classification"].str.contains("systemic", case=False, na=False)]
    rows = []
    for r in n.itertuples():
        end = int(r.label_end_year) if pd.notna(r.label_end_year) else int(r.start_year)
        for y in range(int(r.start_year), end + 1):
            rows.append((r.country_code, y, int(y == r.start_year)))
    act = (pd.DataFrame(rows, columns=["iso3", "year", "lv_bank_onset"])
           .groupby(["iso3", "year"], as_index=False)["lv_bank_onset"].max())
    act["lv_bank_active"] = 1
    return ep, act


def _jst() -> pd.DataFrame | None:
    p = RAW / "JSTdatasetR6.xlsx"
    if not p.exists():
        return None
    j = pd.read_excel(p, sheet_name="JRT6 Data")
    out = pd.DataFrame({
        "iso3": j["iso"], "year": j["year"],
        "credit_gdp_jst": j["tloans"] / j["gdp"],     # prestamos bancarios al sector no financiero / PIB
        "debt_gdp_jst": j["debtgdp"],
        "ltrate_jst": j["ltrate"] / 100, "stir_jst": j["stir"] / 100,
        "crisis_jst": j["crisisJST"],
    })
    return out


def _usd_index() -> pd.DataFrame:
    out = []
    for f, c, name in [("fred_usd_major_index.csv", "DTWEXM", "usd_index_major"),
                       ("fred_usd_broad_index.csv", "DTWEXBGS", "usd_index_broad")]:
        d = pd.read_csv(RAW / f, parse_dates=["observation_date"])
        d[c] = pd.to_numeric(d[c], errors="coerce")
        out.append(d.groupby(d["observation_date"].dt.year)[c].mean().rename(name))
    w = pd.concat(out, axis=1)
    w.index.name = "year"
    return w.reset_index()


# ----------------------------------------------------------------------------- build
def _onset(s: pd.Series, g: pd.Series) -> pd.Series:
    prev = s.groupby(g).shift(1).fillna(0)
    return ((s == 1) & (prev == 0)).astype(float).where(s.notna())


def build() -> pd.DataFrame:
    download()
    OUT.mkdir(parents=True, exist_ok=True)

    df = _gmd()
    df = df.merge(_wdi(), on=["iso3", "year"], how="outer")
    df = df.merge(_hpdd(), on=["iso3", "year"], how="outer")
    ep, act = _lv_episodes()
    df = df.merge(ep, on=["iso3", "year"], how="left").merge(act, on=["iso3", "year"], how="left")
    jst = _jst()
    if jst is not None:
        df = df.merge(jst, on=["iso3", "year"], how="left")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    # LV 2026 cubre 1970-2025 para todos los paises: fuera de episodios = 0
    in_lv = df["year"].between(1970, 2025)
    for c in ("lv_bank_active", "lv_bank_onset"):
        df[c] = df[c].where(df[c].notna(), np.where(in_lv, 0.0, np.nan))

    # Derivadas: crecimiento real, tipo real, r - g, depreciacion, cualquier crisis
    g = df.groupby("iso3")
    # g: PWT 10.01 rgdpna (columna vertebral del modelo, 1950-2019); WDI para huecos/2020+.
    # (El rGDP de GMD-Stata final no es fiable: crece como el PIB nominal.)
    pwt = pd.read_csv(ROOT / "data" / "raw" / "pwt1001.csv", usecols=["isocode", "year", "rgdpna"])
    pwt = pwt.rename(columns={"isocode": "iso3"}).sort_values(["iso3", "year"])
    pwt["g_pwt"] = pwt.groupby("iso3")["rgdpna"].pct_change(fill_method=None)
    df = df.merge(pwt[["iso3", "year", "g_pwt"]], on=["iso3", "year"], how="left")
    df["g_real"] = df["g_pwt"].fillna(df["g_real_wdi"])
    df = df.drop(columns=["g_pwt"])
    g = df.groupby("iso3")
    df["r_real_lt"] = (1 + df["ltrate"]) / (1 + df["infl"]) - 1
    df["r_minus_g"] = df["r_real_lt"] - df["g_real"]
    # tipo implicito efectivo sobre la deuda (intereses/ingresos * ingresos/PIB / deuda_{t-1})
    df["implicit_rate"] = (df["interest_rev"] * df["gov_rev_gdp"]) / g["debt_gdp"].shift(1)
    df.loc[df["implicit_rate"].abs() > 1, "implicit_rate"] = np.nan
    df["r_minus_g_eff"] = (1 + df["implicit_rate"]) / (1 + df["infl"]) - 1 - df["g_real"]
    df["fx_depr"] = np.log(df["usdfx"]).groupby(df["iso3"]).diff()
    df["debt_gdp_change"] = g["debt_gdp"].diff()
    cr = df[["crisis_bank", "crisis_currency", "crisis_sovdebt"]]
    df["crisis_any"] = cr.max(axis=1).where(cr.notna().any(axis=1))
    df["crisis_twin"] = (cr.sum(axis=1) >= 2).astype(float).where(cr.notna().any(axis=1))
    for c in ("r_real_lt", "g_real", "r_minus_g", "implicit_rate", "r_minus_g_eff", "fx_depr"):
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------- mundo
    w = df[df["year"].between(Y0 - 5, Y1)]
    us = w[w["iso3"] == "USA"].set_index("year")
    world = pd.DataFrame(index=pd.Index(range(Y0, Y1 + 1), name="year"))
    world["us_ltrate"] = us["ltrate"]
    world["us_strate"] = us["strate"]
    world["us_policy_rate"] = us["cbrate"]
    world["us_infl"] = us["infl"]
    world["us_real_ltrate"] = (1 + us["ltrate"]) / (1 + us["infl"]) - 1
    world["us_real_strate"] = (1 + us["strate"]) / (1 + us["infl"]) - 1
    world["us_reer"] = us["reer"]
    world["us_debt_gdp"] = us["debt_gdp"]
    # pesos: PIB en USD de GMD, descartando valores imposibles (> PIB de EE.UU., p.ej. ZWE redenominado)
    usgdp = w.loc[w["iso3"] == "USA"].set_index("year")["ngdp_usd"]
    wgt = w.dropna(subset=["ngdp_usd"])
    wgt = wgt[(wgt["ngdp_usd"] <= wgt["year"].map(usgdp) * 1.0001) | (wgt["iso3"] == "USA")]
    for c, name in [("crisis_bank", "share_bank_onset"), ("crisis_currency", "share_currency_onset"),
                    ("crisis_sovdebt", "share_sovdebt_onset"), ("lv_bank_active", "share_bank_active_lv")]:
        world[name] = w.groupby("year")[c].mean()
        x = wgt.dropna(subset=[c])
        world[name + "_gdpw"] = (x[c] * x["ngdp_usd"]).groupby(x["year"]).sum() / x.groupby("year")["ngdp_usd"].sum()
    x = wgt.dropna(subset=["debt_gdp"])
    world["world_debt_gdp_gdpw"] = (x["debt_gdp"] * x["ngdp_usd"]).groupby(x["year"]).sum() / x.groupby("year")["ngdp_usd"].sum()
    world["world_debt_gdp_median"] = w.groupby("year")["debt_gdp"].median()
    world["world_r_minus_g_median"] = w.groupby("year")["r_minus_g"].median()
    world["world_r_minus_g_eff_median"] = w.groupby("year")["r_minus_g_eff"].median()
    world["n_countries_debt"] = w.groupby("year")["debt_gdp"].count()
    world = world.join(_usd_index().set_index("year"))
    world = world.reset_index()
    world.to_csv(OUT / "finanzas_world.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- panel
    df = df[df["year"].between(Y0, Y1) & (df["iso3"].str.len() == 3)]
    keep = [c for c in df.columns if c not in ("iso3", "year")]
    df = df.dropna(subset=keep, how="all")
    df["year"] = df["year"].astype(int)
    df.to_csv(OUT / "finanzas.csv", index=False, float_format="%.6g")

    desc = {n: d for (n, _, d) in GMD_VARS.values()} | {n: d for (n, _, d) in WDI_VARS.values()}
    desc |= {
        "debt_gdp_hpdd": "Deuda bruta gobierno / PIB, IMF Historical Public Debt Database (1800-2015)",
        "lv_output_loss": "Laeven-Valencia 2018: perdida de producto acumulada del episodio bancario (fraccion PIB tendencial), en el anio de inicio",
        "lv_fiscal_cost": "Laeven-Valencia 2018: coste fiscal bruto directo (fraccion PIB), anio de inicio",
        "lv_peak_npl": "Laeven-Valencia 2018: morosidad maxima (fraccion de prestamos), anio de inicio",
        "lv_debt_increase": "Laeven-Valencia 2018: aumento de deuda publica asociado (fraccion PIB), anio de inicio",
        "lv_bank_active": "Laeven-Valencia 2026 (1970-2025): 1 si crisis bancaria sistemica en curso",
        "lv_bank_onset": "Laeven-Valencia 2026 (1970-2025): 1 en el anio de inicio de crisis bancaria sistemica",
        "credit_gdp_jst": "JST R6: prestamos bancarios totales al sector no financiero / PIB (18 economias avanzadas)",
        "debt_gdp_jst": "JST R6: deuda publica / PIB",
        "ltrate_jst": "JST R6: tipo a largo plazo nominal", "stir_jst": "JST R6: tipo a corto plazo nominal",
        "crisis_jst": "JST R6: inicio de crisis financiera sistemica",
        "g_real": "Crecimiento del PIB real: PWT 10.01 rgdpna (1951-2019), WDI en huecos y 2020+",
        "r_real_lt": "Tipo real a largo ex post = (1+ltrate)/(1+infl)-1",
        "r_minus_g": "Diferencial r-g = r_real_lt - g_real",
        "implicit_rate": "Tipo implicito sobre la deuda = intereses/PIB / deuda_{t-1} (WDI x GMD)",
        "r_minus_g_eff": "r-g con tipo implicito efectivo: (1+implicit_rate)/(1+infl)-1-g_real (cubre emergentes)",
        "fx_depr": "Depreciacion log frente al USD (d ln usdfx; >0 = depreciacion)",
        "debt_gdp_change": "Variacion anual de debt_gdp",
        "crisis_any": "Inicio de alguna crisis (bancaria, cambiaria o soberana) segun GMD",
        "crisis_twin": "Inicio simultaneo de >=2 tipos de crisis (GMD)",
    }
    wdesc = {
        "us_ltrate": "EE.UU. tipo a largo nominal (bono 10 anios, GMD)",
        "us_strate": "EE.UU. tipo a corto nominal (letras 3 meses, GMD)",
        "us_policy_rate": "EE.UU. tipo oficial de la Fed (descuento/objetivo fondos federales, GMD cbrate)",
        "us_infl": "EE.UU. inflacion IPC",
        "us_real_ltrate": "EE.UU. tipo real largo ex post (tipo mundial libre de riesgo, proxy)",
        "us_real_strate": "EE.UU. tipo real corto ex post (ciclo financiero global / politica monetaria)",
        "us_reer": "EE.UU. tipo de cambio efectivo real (GMD, 2015=100)",
        "us_debt_gdp": "EE.UU. deuda publica / PIB",
        "share_bank_onset": "Fraccion de paises con inicio de crisis bancaria (GMD)",
        "share_currency_onset": "Fraccion de paises con inicio de crisis cambiaria (GMD)",
        "share_sovdebt_onset": "Fraccion de paises con inicio de crisis soberana (GMD)",
        "share_bank_active_lv": "Fraccion de paises con crisis bancaria sistemica activa (LV 2026)",
        "world_debt_gdp_gdpw": "Deuda publica / PIB mundial ponderada por PIB en USD",
        "world_debt_gdp_median": "Mediana de deuda publica / PIB",
        "world_r_minus_g_median": "Mediana de r-g entre paises",
        "world_r_minus_g_eff_median": "Mediana de r-g efectivo (tipo implicito) entre paises",
        "n_countries_debt": "Numero de paises con dato de deuda",
        "usd_index_major": "Indice nominal del dolar vs. monedas principales (FRED DTWEXM, media anual, 1973-2019)",
        "usd_index_broad": "Indice nominal amplio del dolar (FRED DTWEXBGS, media anual, 2006-)",
    }
    for k in list(wdesc):
        if k.startswith("share_"):
            wdesc[k + "_gdpw"] = wdesc[k] + ", ponderada por PIB en USD"
    cov = {c: {"n": int(df[c].notna().sum()), "countries": int(df.loc[df[c].notna(), "iso3"].nunique()),
               "years": [int(df.loc[df[c].notna(), "year"].min()), int(df.loc[df[c].notna(), "year"].max())]}
           for c in keep if df[c].notna().any()}
    meta = {"panel": {k: {"desc": desc.get(k, ""), **cov.get(k, {})} for k in keep},
            "world": {k: wdesc.get(k, "") for k in world.columns if k != "year"},
            "notes": "Ratios en fracciones (0.6 = 60%). Dummies de crisis GMD = anio de inicio. "
                     "Ver docs/research/finanzas.md."}
    (OUT / "finanzas_dict.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"finanzas.csv {df.shape}, finanzas_world.csv {world.shape}")
    return df


if __name__ == "__main__":
    build()
