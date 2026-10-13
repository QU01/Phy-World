"""Las 10 regiones del L0 (decisión abierta 1 de la hoja de ruta: regiones de estilo IAM, para poder
comparar con la base SSP; la zona del sistema-mundo queda como atributo del L1).

Se construyen con las subregiones M49 de la ONU (``UNlocations.txt`` de WPP 2024, ya descargado por
F3) más cuatro excepciones de la OCDE. Son parecidas, NO idénticas, a las R10 de los IAM [declarado]:

    NAM  Norteamérica (EE. UU., Canadá)
    LAM  América Latina y el Caribe
    EUR  Europa (sin la antigua URSS)
    REF  Antigua URSS (economías reformadas: Rusia, Ucrania, Bielorrusia, Moldavia, Cáucaso y Asia Central)
    MEA  Oriente Medio y Norte de África (Asia occidental sin el Cáucaso, más Irán)
    SSA  África subsahariana
    SAS  Asia meridional (sin Irán)
    CHN  China y el resto de Asia oriental (sin Japón ni Corea del Sur)
    SEA  Sudeste asiático y Pacífico (sin Australia ni Nueva Zelanda)
    OPA  OCDE del Pacífico (Japón, Corea del Sur, Australia, Nueva Zelanda)
"""
from __future__ import annotations

import numpy as np

CODES = ("NAM", "LAM", "EUR", "REF", "MEA", "SSA", "SAS", "CHN", "SEA", "OPA")
NAMES = {"NAM": "North America", "LAM": "Latin America and Caribbean", "EUR": "Europe",
         "REF": "Former Soviet Union", "MEA": "Middle East and North Africa", "SSA": "Sub-Saharan Africa",
         "SAS": "South Asia", "CHN": "China and East Asia", "SEA": "South-East Asia and Pacific",
         "OPA": "OECD Pacific"}
OVERRIDES = {"USA": "NAM", "CAN": "NAM", "BMU": "NAM", "JPN": "OPA", "KOR": "OPA", "AUS": "OPA", "NZL": "OPA", "IRN": "MEA",
             "RUS": "REF", "UKR": "REF", "BLR": "REF", "MDA": "REF", "ARM": "REF", "AZE": "REF", "GEO": "REF",
             "TWN": "CHN", "HKG": "CHN", "MAC": "CHN"}
BY_SUBREGION = {
    "Northern America": "NAM", "Caribbean": "LAM", "Central America": "LAM", "South America": "LAM",
    "Northern Europe": "EUR", "Western Europe": "EUR", "Southern Europe": "EUR", "Eastern Europe": "EUR",
    "Central Asia": "REF", "Northern Africa": "MEA", "Western Asia": "MEA",
    "Eastern Africa": "SSA", "Middle Africa": "SSA", "Southern Africa": "SSA", "Western Africa": "SSA",
    "Southern Asia": "SAS", "Eastern Asia": "CHN", "South-Eastern Asia": "SEA", "Melanesia": "SEA",
    "Micronesia": "SEA", "Polynesia": "SEA", "Australia/New Zealand": "OPA",
}


def region_index(iso3, isonum) -> np.ndarray:
    """Índice de región (0-9) de cada país del L1, o -1 si no se puede asignar."""
    import pandas as pd
    from world4.data import energy
    u = pd.read_csv(energy.root() / "raw" / "UNlocations.txt", sep="\t")
    u = u[u.location_type == 4]
    sub = {int(c): str(r) for c, r in zip(u.country_code, u.reg_name)}
    out = np.full(len(iso3), -1)
    for i, (c, m) in enumerate(zip(iso3, isonum)):
        code = OVERRIDES.get(c) or BY_SUBREGION.get(sub.get(int(m), ""))
        if code is not None:
            out[i] = CODES.index(code)
    return out
