"""Energía primaria por país y fuente para el sector de energía de F3.

Fuente: OWID energy-data (Energy Institute Statistical Review + EIA + Ember), fijada al commit
``COMMIT`` y comprobada contra ``energy_manifest.json`` (SHA-256). Licencia de OWID CC BY 4.0;
la del Energy Institute no es verificable ([?] en ``sources.json``): no viaja en el repo.
Regiones: las seis áreas de la ONU (``UNlocations.txt`` de WPP 2024, CC BY 3.0 IGO).

Energía primaria por el método de sustitución (como el EI), en TWh, 1965-2024. Fuentes:
``fossil`` (carbón + petróleo + gas), ``nuclear``, ``hydro``, ``vre`` (eólica + solar), ``other``
(biocombustibles + otras renovables). La biomasa tradicional NO está (el EI no la cuenta).

Desglose imputado [declarado por país y año en ``imputed``]: sólo 79 de los 180 países del L1
tienen el desglose del EI (96,5 % de la energía mundial en 2019). Para el resto, cuando hay energía
primaria total y generación eléctrica por fuente (casi todos desde 2000), el consumo primario de
nuclear, hidro, eólica y solar = generación / eficiencia de sustitución del año, MEDIDA como la
mediana del cociente en los países que tienen ambas series; ``other`` = 0 y ``fossil`` = el resto.
Sin generación eléctrica (antes de 2000 en muchos países pequeños): todo ``fossil``.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

COMMIT = "f8de88a4f6f9cab44f1cf606047cbd6c4414042c"
URLS = {"owid-energy-data.csv": f"https://raw.githubusercontent.com/owid/energy-data/{COMMIT}/owid-energy-data.csv",
        "UNlocations.txt": "https://raw.githubusercontent.com/PPgp/wpp2024/main/data/UNlocations.txt"}
MANIFEST = Path(__file__).with_name("energy_manifest.json")
SOURCES = ("fossil", "nuclear", "hydro", "vre", "other")
REGIONS = ("Africa", "Asia", "Europe", "Latin America and the Caribbean", "Northern America", "Oceania")
YEAR0, YEAR1 = 1965, 2024
ELEC = {"nuclear": ("nuclear_consumption", "nuclear_electricity"), "hydro": ("hydro_consumption", "hydro_electricity"),
        "solar": ("solar_consumption", "solar_electricity"), "wind": ("wind_consumption", "wind_electricity")}


class EnergyDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "energy"


def npz_path() -> Path:
    return root() / "energy.npz"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    raw = root() / "raw"
    missing, bad = [], []
    for name, meta in manifest()["files"].items():
        p = raw / name
        if not p.is_file():
            missing.append(name)
        elif _sha(p) != meta["sha256"]:
            bad.append(name)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(raw)}


def install(log=print) -> dict:
    raw = root() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in manifest()["files"]:
        if not (raw / name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(URLS[name], raw / name)
    v = verify_raw()
    if not v["ok"]:
        raise EnergyDataError(f"energy files missing {v['missing']} or with another SHA-256 {v['mismatched']}")
    convert()
    load.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path())}


def convert() -> Path:
    import pandas as pd
    raw = root() / "raw"
    d = pd.read_csv(raw / "owid-energy-data.csv", low_memory=False)
    d = d[d.iso_code.notna() & (d.iso_code.str.len() == 3) & d.year.between(YEAR0, YEAR1)]
    # eficiencia de sustitución por fuente y año: mediana de consumo / generación donde hay ambas
    eff = {}
    for s, (cc, ec) in ELEC.items():
        both = d[(d[cc] > 0) & (d[ec] > 0)]
        r = (both[ec] / both[cc]).groupby(both.year).median()
        eff[s] = r.reindex(range(YEAR0, YEAR1 + 1)).ffill().bfill()
    isos = np.array(sorted(d.iso_code.unique()))
    ii = {c: k for k, c in enumerate(isos)}
    C, Y = len(isos), YEAR1 - YEAR0 + 1
    arr = {s: np.full((C, Y), np.nan) for s in SOURCES}
    pe = np.full((C, Y), np.nan)
    imputed = np.zeros((C, Y), np.uint8)              # 0 dato EI, 1 desde electricidad, 2 todo fósil
    for row in d.itertuples(index=False):
        i, t = ii[row.iso_code], int(row.year) - YEAR0
        P = row.primary_energy_consumption
        if not (P == P):
            continue
        pe[i, t] = P
        g = lambda name: getattr(row, name)  # noqa: E731
        full = all(g(c) == g(c) for c in ("coal_consumption", "oil_consumption", "gas_consumption",
                                            "hydro_consumption", "solar_consumption", "wind_consumption"))
        if full:
            nuc = g("nuclear_consumption")
            arr["nuclear"][i, t] = nuc if nuc == nuc else 0.0
            arr["hydro"][i, t] = g("hydro_consumption")
            arr["vre"][i, t] = g("solar_consumption") + g("wind_consumption")
            bio, oth = g("biofuel_consumption"), g("other_renewable_consumption")
            arr["other"][i, t] = (bio if bio == bio else 0.0) + (oth if oth == oth else 0.0)
            arr["fossil"][i, t] = g("coal_consumption") + g("oil_consumption") + g("gas_consumption")
            continue
        y = int(row.year)
        e = {s: g(ec) for s, (cc, ec) in ELEC.items()}
        if any(v == v for v in e.values()):
            conv = {s: (e[s] / eff[s][y] if e[s] == e[s] else 0.0) for s in e}
            nf = {"nuclear": conv["nuclear"], "hydro": conv["hydro"], "vre": conv["solar"] + conv["wind"], "other": 0.0}
            tot = sum(nf.values())
            if tot > P:                               # la conversión no puede superar la energía total
                nf = {k: v * P / tot for k, v in nf.items()}
                tot = P
            for k, v in nf.items():
                arr[k][i, t] = v
            arr["fossil"][i, t] = P - tot
            imputed[i, t] = 1
        else:
            for k in ("nuclear", "hydro", "vre", "other"):
                arr[k][i, t] = 0.0
            arr["fossil"][i, t] = P
            imputed[i, t] = 2
    # regiones de la ONU por código numérico
    u = pd.read_csv(raw / "UNlocations.txt", sep="\t")
    u = u[u.location_type == 4]
    region_of_code = dict(zip(u.country_code.astype(int), u.area_name))
    out = npz_path()
    np.savez_compressed(out, isos=isos, years=np.arange(YEAR0, YEAR1 + 1), pe=pe, imputed=imputed,
                        **{f"src_{s}": arr[s] for s in SOURCES},
                        eff_years=np.arange(YEAR0, YEAR1 + 1), **{f"eff_{s}": eff[s].to_numpy() for s in eff},
                        region_codes=np.array(list(region_of_code)), region_names=np.array(list(region_of_code.values())))
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise EnergyDataError(f"no energy data at {p}: python world4_cli.py --install-energy")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file()


def aligned(iso3, isonum, years) -> dict:
    """Arrays (N, T) en el orden y los años del L1 (NaN fuera de 1965-2024), y la región de cada país."""
    d = load()
    pos = {c: k for k, c in enumerate(d["isos"])}
    k = np.array([pos.get(c, -1) for c in iso3])
    years = np.asarray(years)
    ty = years - int(d["years"][0])
    okt = (ty >= 0) & (ty < len(d["years"]))
    out = {}
    for name in ["pe", "imputed"] + [f"src_{s}" for s in SOURCES]:
        a = d[name].astype(float)
        b = np.full((len(iso3), len(years)), np.nan)
        rows = k >= 0
        b[np.ix_(rows, okt)] = a[np.ix_(k[rows], ty[okt])]
        out[name] = b
    rmap = dict(zip(d["region_codes"].astype(int), d["region_names"].astype(str)))
    out["region"] = np.array([REGIONS.index(rmap[c]) if rmap.get(c) in REGIONS else -1 for c in isonum])
    return out
