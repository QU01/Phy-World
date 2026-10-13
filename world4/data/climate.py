"""Datos del clima de F4: el conjunto calibrado y restringido de FaIR y la temperatura observada.

* **FaIR calibration data 1.6.0** (Smith, Zenodo 10.5281/zenodo.18828694, CC BY 4.0, versión del
  concepto 10.5281/zenodo.7112539): 841 configuraciones posteriores para FaIR v2.2.4 (calibración
  «fastmip v1»), las propiedades de las especies, las emisiones históricas CMIP7 1750-2023 y los
  forzamientos solar, volcánico, de uso del suelo y de riego. Se elige la 1.6.0 (marzo de 2026) porque
  es la que corresponde al paquete instalado (``fair==2.2.4``) y trae todos los ficheros de una corrida
  histórica; la versión revisada por pares es la 1.4.1 (Smith et al. 2024, GMD 17, 8569), para FaIR
  2.1.3 y sin emisiones ni forzamientos sueltos. El método de calibración es el mismo.
* **HadCRUT5** (Met Office Hadley Centre / CRU, versión 5.1.0.0, Open Government Licence v3): GMST
  anual. El fichero se actualiza cada mes, así que el manifiesto fija el SHA-256 de las filas
  1850-2023 (las que usa la validación) y no el del fichero completo.

Aviso de fuga declarado: el conjunto posterior de FaIR se restringió con la GMST observada hasta 2023
y con el calentamiento de 2004-2023 (``warming_baselines.csv``). En un *hindcast* desde 1990 los
parámetros climáticos NO son fuera de muestra; las emisiones sí (las genera el L1).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

ZENODO = "https://zenodo.org/records/18828694/files/{}?download=1"
HADCRUT = ("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/analysis/diagnostics/"
           "HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv")
FAIR_FILES = ("calibrated_constrained_parameters.csv", "species_configs_properties.csv",
              "historical_emissions_1750-2023_cmip7.csv", "solar_forcing_timebounds_cmip7.csv",
              "volcanic_forcing_timebounds_cmip7.csv", "land_use_forcing_timebounds_cmip7.csv",
              "irrigation_forcing_timebounds_cmip7.csv", "warming_baselines.csv")
HADCRUT_FILE = "hadcrut5_global_annual.csv"
MANIFEST = Path(__file__).with_name("climate_manifest.json")
EMIS_END = 2023                      # último año de emisiones históricas
GMST_END = 2023                      # último año de GMST usado (filas fijadas por SHA-256)


class ClimateDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "climate"


def raw(name: str) -> Path:
    return root() / "raw" / name


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha(p: Path) -> str:
    return _sha_bytes(p.read_bytes())


def _hadcrut_rows(p: Path) -> bytes:
    """Filas de HadCRUT5 desde la cabecera hasta ``GMST_END`` (lo que fija el manifiesto)."""
    lines = p.read_text(encoding="utf-8").splitlines()
    keep = [lines[0]] + [ln for ln in lines[1:] if ln.split(",")[0].strip().isdigit()
                         and int(ln.split(",")[0]) <= GMST_END]
    return ("\n".join(keep) + "\n").encode("utf-8")


def verify_raw() -> dict:
    missing, bad = [], []
    files = manifest()["files"]
    for name, meta in files.items():
        p = raw(name)
        if not p.is_file():
            missing.append(name)
            continue
        sha = _sha_bytes(_hadcrut_rows(p)) if name == HADCRUT_FILE else _sha(p)
        if sha != meta["sha256"]:
            bad.append(name)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(root() / "raw")}


def install(log=print) -> dict:
    d = root() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    for name in manifest()["files"]:
        if not raw(name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(HADCRUT if name == HADCRUT_FILE else ZENODO.format(name), raw(name))
    v = verify_raw()
    if not v["ok"]:
        raise ClimateDataError(f"climate files missing {v['missing']} or with another SHA-256 {v['mismatched']}")
    load.cache_clear()
    return v


def available() -> bool:
    try:
        import fair  # noqa: F401
    except ImportError:
        return False
    return all(raw(n).is_file() for n in manifest()["files"])


def require():
    try:
        import fair  # noqa: F401
    except ImportError as e:
        raise ClimateDataError("F4 needs the FaIR package: pip install fair==2.2.4 (extra 'climate')") from e
    if not available():
        raise ClimateDataError(f"no climate data at {root() / 'raw'}: python world4_cli.py --install-climate")


@lru_cache(maxsize=1)
def load() -> dict:
    """Tablas en memoria: parámetros (DataFrame), emisiones históricas por especie (dict año->valor
    como array 1750..2023 en las unidades del fichero), forzamientos por límite temporal y la GMST."""
    import pandas as pd
    require()
    params = pd.read_csv(raw("calibrated_constrained_parameters.csv"), index_col=0)
    em = pd.read_csv(raw("historical_emissions_1750-2023_cmip7.csv"))
    years = np.array([int(c) for c in em.columns if c.isdigit()])
    vals = em[[str(y) for y in years]].to_numpy(float)
    emis = {v: (u, vals[k]) for k, (v, u) in enumerate(zip(em["variable"], em["unit"]))}
    forc = {}
    for key, fname in (("Solar", "solar_forcing_timebounds_cmip7.csv"), ("Volcanic", "volcanic_forcing_timebounds_cmip7.csv")):
        d = pd.read_csv(raw(fname))
        forc[key] = (d.iloc[:, 0].to_numpy(int), d.iloc[:, 1].to_numpy(float))
    for key, fname in (("Land use", "land_use_forcing_timebounds_cmip7.csv"),
                       ("Irrigation", "irrigation_forcing_timebounds_cmip7.csv")):
        d = pd.read_csv(raw(fname))
        forc[key] = (d.iloc[:, 0].to_numpy(int), d["M"].to_numpy(float))   # escenarios idénticos hasta 2023
    h = pd.read_csv(raw(HADCRUT_FILE))
    h = h[h.iloc[:, 0] <= GMST_END]
    hy = h.iloc[:, 0].to_numpy(int)
    ha = h.iloc[:, 1].to_numpy(float)
    base = ha[(hy >= 1850) & (hy <= 1900)].mean()
    return {"params": params, "emis_years": years, "emis": emis, "forcing": forc,
            "gmst_years": hy, "gmst": ha - base}


def gmst_observed(years) -> np.ndarray:
    """GMST de HadCRUT5 respecto de 1850-1900 (°C) para ``years``; NaN fuera de 1850-2023."""
    d = load()
    pos = {int(y): k for k, y in enumerate(d["gmst_years"])}
    return np.array([d["gmst"][pos[int(y)]] if int(y) in pos else np.nan for y in years])


def co2_ffi_observed(years) -> np.ndarray:
    """Emisiones de CO2 fósiles e industriales (GtCO2/año, CMIP7) para ``years``."""
    d = load()
    unit, v = d["emis"]["CO2 FFI"]
    y0 = int(d["emis_years"][0])
    return np.array([v[int(y) - y0] if y0 <= int(y) <= EMIS_END else np.nan for y in years])
