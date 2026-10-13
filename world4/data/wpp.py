"""Estructura por edad y sexo de UN WPP 2024 para el sector de población de F1.

El panel del Atlas sólo trae agregados de WPP (TFR, e0, migración neta, cuotas por edad). El
modelo de cohortes necesita además, por país:

* población por edad simple (0-100+) y sexo, 1950-2023 (``popAge1dt``);
* tasas de mortalidad por quinquenio de edad y sexo, por periodo quinquenal (``mx5dt``);
* patrón de fecundidad por edad (``percentASFR5dt``) y razón de sexos al nacer (``sexRatio1dt``).

Se descargan del paquete R ``PPgp/wpp2024`` (CC BY 3.0 IGO, redistribuible con atribución), se
comprueban contra ``wpp_manifest.json`` (SHA-256) y se convierten una vez a ``wpp_age.npz`` en
``data_cache/wpp2024/``. La conversión necesita ``rdata`` (extra ``world4[rebuild]``); leer el
``.npz`` no.

``known_time``: toda la edición se publicó el 2024-07-11. Las estimaciones de 1950-2023 son la
historia REVISADA de 2024; usarlas en un origen de 1990 es una fuga de revisión que aquí no se
puede medir (no hay ediciones antiguas cargadas). Se declara en cada resultado que las usa.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/PPgp/wpp2024/main/data/"
MANIFEST = Path(__file__).with_name("wpp_manifest.json")
YEAR0, YEAR1 = 1950, 2023                 # estimaciones (no proyección) de la edición 2024
AGES = 101                                # 0..99 y 100+
MX_GROUPS = ("0", "1", *[str(a) for a in range(5, 100, 5)], "100+")      # 22 grupos
ASFR_GROUPS = tuple(f"{a}-{a + 4}" for a in range(15, 50, 5))            # 7 grupos


class WPPDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "wpp2024"


def npz_path() -> Path:
    return root() / "wpp_age.npz"


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
    """Descarga lo que falte, verifica SHA-256 y escribe ``wpp_age.npz``."""
    raw = root() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in manifest()["files"]:
        if not (raw / name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(URL + name, raw / name)
    v = verify_raw()
    if not v["ok"]:
        raise WPPDataError(f"WPP 2024 files missing {v['missing']} or with another SHA-256 {v['mismatched']}")
    convert()
    load.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path())}


def _read(name: str):
    import pandas as pd
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(root() / "raw" / name)
    df = pd.DataFrame(obj[list(obj)[0]]).reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    df = df[pd.to_numeric(df.country_code) < 900]               # >= 900: agregados regionales
    df["country_code"] = df.country_code.astype(int)
    df["year"] = df.year.astype(int)
    return df


def convert() -> Path:
    pop = _read("popAge1dt.rda")
    pop = pop[pop.year.between(YEAR0, YEAR1)]
    codes = np.array(sorted(pop.country_code.unique()))
    ci = {c: i for i, c in enumerate(codes)}
    C, Y = len(codes), YEAR1 - YEAR0 + 1
    popM = np.full((C, Y, AGES), np.nan, np.float32)
    popF = np.full((C, Y, AGES), np.nan, np.float32)
    i, t, a = pop.country_code.map(ci).to_numpy(), pop.year.to_numpy() - YEAR0, pop.age.astype(int).to_numpy()
    popM[i, t, a], popF[i, t, a] = pop.popM.to_numpy(), pop.popF.to_numpy()

    mx = _read("mx5dt.rda")
    mx = mx[mx.country_code.isin(ci)]
    periods = np.array(sorted(mx.year.unique()))                 # año central: 1953, 1958, ...
    pi = {p: k for k, p in enumerate(periods)}
    gi = {g: k for k, g in enumerate(MX_GROUPS)}
    mxM = np.full((C, len(periods), len(MX_GROUPS)), np.nan, np.float32)
    mxF = mxM.copy()
    idx = (mx.country_code.map(ci).to_numpy(), mx.year.map(pi).to_numpy(), mx.age.astype(str).map(gi).to_numpy())
    mxM[idx], mxF[idx] = mx.mxM.to_numpy(), mx.mxF.to_numpy()

    fa = _read("percentASFR5dt.rda")
    fa = fa[fa.country_code.isin(ci) & fa.year.isin(pi)]
    ai = {g: k for k, g in enumerate(ASFR_GROUPS)}
    pasfr = np.full((C, len(periods), len(ASFR_GROUPS)), np.nan, np.float32)
    pasfr[fa.country_code.map(ci).to_numpy(), fa.year.map(pi).to_numpy(), fa.age.astype(str).map(ai).to_numpy()] = \
        fa.pasfr.to_numpy()

    sr = _read("sexRatio1dt.rda")
    sr = sr[sr.country_code.isin(ci) & sr.year.between(YEAR0, YEAR1)]
    srb = np.full((C, Y), np.nan, np.float32)
    srb[sr.country_code.map(ci).to_numpy(), sr.year.to_numpy() - YEAR0] = sr.srb.to_numpy()

    out = npz_path()
    np.savez_compressed(out, codes=codes, years=np.arange(YEAR0, YEAR1 + 1), periods=periods,
                        popM=popM, popF=popF, mxM=mxM, mxF=mxF, pasfr=pasfr, srb=srb)
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise WPPDataError(f"no WPP age data at {p}: python world4_cli.py --install-wpp")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file()


def aligned(isonum) -> dict:
    """Arrays de ``load()`` reordenados a la lista de países del L1 (código numérico UN M49 =
    ISO 3166 numérico). Países sin fila en WPP: NaN y ``has[i] = False``."""
    d = load()
    pos = {int(c): k for k, c in enumerate(d["codes"])}
    k = np.array([pos.get(int(c), -1) for c in isonum])
    has = k >= 0
    out = {"has": has, "years": d["years"], "periods": d["periods"]}
    for name in ("popM", "popF", "mxM", "mxF", "pasfr", "srb"):
        a = d[name]
        b = np.full((len(k),) + a.shape[1:], np.nan, np.float32)
        b[has] = a[k[has]]
        out[name] = b
    return out


def period_for(year: int, periods) -> int:
    """Índice del último periodo quinquenal CERRADO en ``year`` (centro + 2 <= year)."""
    ok = np.where(np.asarray(periods) + 2 <= year)[0]
    return int(ok[-1]) if len(ok) else 0
