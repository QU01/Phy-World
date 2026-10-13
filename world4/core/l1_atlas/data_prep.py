"""Build the country-year panel used to calibrate and validate the model.

Sources (downloaded to data/raw, see README):
  * Penn World Table 10.01 (Feenstra, Inklaar & Timmer 2015), via the CRAN `pwt10` package:
      real GDP (rgdpna), population, capital stock (rnna), human capital (hc),
      labour share (labsh), investment / export / import shares (csh_i, csh_x, csh_m).
  * UCDP/PRIO Armed Conflict Dataset, via CRAN `peacesciencer`: intrastate conflict years.
  * V-Dem polyarchy + Polity2, via CRAN `peacesciencer` (gwcode_democracy).
  * World Bank WDI NY.GDP.TOTL.RT.ZS: total natural resources rents (% of GDP), 1970-2019.
Output: data/processed/panel.csv (iso3, year, ...), 1950-2019.
"""
from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def gw_to_iso3(codes):
    cc = coco.CountryConverter()
    table = cc.data[["ISO3", "GWcode"]].dropna()
    table["GWcode"] = pd.to_numeric(table["GWcode"], errors="coerce")
    m = dict(zip(table["GWcode"].astype(int), table["ISO3"]))
    # Gleditsch-Ward special codes not in the converter table
    m.update({260: "DEU", 255: "DEU", 678: "YEM", 679: "YEM", 816: "VNM", 817: "VNM",
              365: "RUS", 345: "SRB", 347: "XKX", 315: "CZE", 316: "CZE"})
    return [m.get(int(c)) if pd.notna(c) else None for c in codes]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pwt = pd.read_csv(RAW / "pwt1001.csv")
    pwt = pwt.rename(columns={"isocode": "iso3"})
    keep = ["iso3", "country", "year", "rgdpna", "pop", "rnna", "hc", "labsh",
            "csh_i", "csh_x", "csh_m", "delta"]
    p = pwt[keep].dropna(subset=["rgdpna", "pop", "rnna"]).copy()
    p = p[(p.year >= 1950) & (p.year <= 2019)]
    p["gdppc"] = p.rgdpna / p["pop"]  # 2017 USD (millions / millions)

    # --- UCDP intrastate conflict (any active conflict with the country as side A)
    acd = pd.read_csv(RAW / "ucdp_acd.csv")
    intra = acd[acd.type_of_conflict.isin(["intrastate", "internationalized intrastate"])]
    intra = intra.assign(iso3=gw_to_iso3(intra.gwno_a))
    conf = (intra.groupby(["iso3", "year"]).intensity_level.max().rename("conflict").reset_index())
    conf["year"] = conf.year.astype(int)
    onset = pd.read_csv(RAW / "ucdp_onsets.csv")
    onset = onset.assign(iso3=gw_to_iso3(onset.gwcode))
    onset = onset.groupby(["iso3", "year"]).sumonset2.max().rename("onset").reset_index()
    onset["year"] = onset.year.astype(int)

    dem = pd.read_csv(RAW / "gwcode_democracy.csv")
    dem = dem.assign(iso3=gw_to_iso3(dem.gwcode))
    dem = dem.groupby(["iso3", "year"])[["v2x_polyarchy", "polity2"]].mean().reset_index()
    dem["year"] = dem.year.astype(int)

    wdi = pd.read_csv(RAW / "wdi_resource_rents.csv", skiprows=4)
    wdi = wdi.melt(id_vars=["Country Code"], value_vars=[c for c in wdi.columns if c.isdigit()],
                   var_name="year", value_name="res_rents")
    wdi = wdi.rename(columns={"Country Code": "iso3"}).dropna()
    wdi["year"] = wdi.year.astype(int)
    wdi["res_rents"] = wdi.res_rents / 100.0

    p = p.merge(wdi, on=["iso3", "year"], how="left")
    p = p.merge(conf, on=["iso3", "year"], how="left").merge(onset, on=["iso3", "year"], how="left")
    p = p.merge(dem, on=["iso3", "year"], how="left")
    p["conflict"] = (p.conflict.fillna(0) > 0).astype(int)
    p["onset"] = (p.onset.fillna(0) > 0).astype(int)

    # Fill slow-moving covariates within country (hc, labsh, shares) so every
    # observed country-year has a value; remaining gaps get cross-country medians by year.
    p = p.sort_values(["iso3", "year"])
    p["res_rents_obs"] = p.res_rents  # unfilled copy for validation
    for c in ["hc", "labsh", "csh_i", "csh_x", "csh_m", "delta", "v2x_polyarchy", "res_rents"]:
        p[c] = p.groupby("iso3")[c].transform(lambda s: s.interpolate(limit_direction="both"))
        p[c] = p[c].fillna(p.groupby("year")[c].transform("median"))
    p["csh_x"] = p.csh_x.clip(0, 1.5)
    p["csh_m"] = (-p.csh_m).clip(0, 1.5)  # PWT reports imports as negative share
    p["csh_i"] = p.csh_i.clip(0.01, 0.7)
    p["res_rents"] = p.res_rents.clip(0.001, 0.6)

    p["isonum"] = coco.convert(p.iso3.tolist(), src="ISO3", to="ISOnumeric", not_found=None)
    p.to_csv(OUT / "panel.csv", index=False)
    print(p.shape, p.iso3.nunique(), "countries;",
          "conflict-years:", int(p.conflict.sum()), "onsets:", int(p.onset.sum()),
          "resource-rent obs:", int(p.res_rents_obs.notna().sum()))


if __name__ == "__main__":
    main()
