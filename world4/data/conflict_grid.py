"""Eventos de violencia organizada geolocalizados (UCDP GED) y la malla PRIO-GRID para el módulo de
predicción de conflicto por celda (``world4.conflict``).

Fuentes (``conflict_grid_manifest.json``; ninguna viaja en el repo, se instalan con
``--install-conflict-grid`` y se comprueban por SHA-256):

* **UCDP GED 26.1** (Sundberg y Melander 2013; codebook Högbladh 2026): todos los eventos de violencia
  organizada 1989-2025 con fecha, coordenadas, celda PRIO-GRID, tipo (estatal, no estatal, unilateral) y
  muertes (``best``). CC BY 4.0 [V] (página de descargas de UCDP, leída en esta sesión).
* **UCDP Candidate 26.01.26.09** (Hegre, Croicu, Eck y Högbladh 2020): los eventos candidatos de
  enero a septiembre de 2026, publicados cada mes con un retraso de a lo sumo un mes. Son PRELIMINARES
  (``code_status`` distinto de "Clear" en ~35 %): se usan como el último dato y como la verdad de la
  validación de 2026, y se declara así. CC BY 4.0 [V].
* **PRIO-GRID 2.0** (Tollefsen, Strand y Buhaug 2012), por su API (``grid.prio.org/api``): las 64 818
  celdas terrestres de 0,5° x 0,5° y covariables estructurales (país, distancias, población,
  accesibilidad, terreno, recursos). Libre con cita [S]; licencia sin verificar, se trata como no
  redistribuible.

Reglas de agregación (declaradas, ``RULES``):

* Mes del evento = mes de ``date_start`` (sólo el 1,75 % de los eventos cruza de mes).
* Un evento cuenta en su celda si ``where_prec <= 4`` (punto exacto, radio de 25 km, centroide de
  división administrativa de 1.er o 2.º orden). Con ``where_prec`` 5-7 (rasgo lineal, sólo país, aguas
  internacionales) no tiene celda fiable: cuenta sólo para el país (``country_id``, Gleditsch-Ward).
* La celda es la ``priogrid_gid`` de UCDP, aunque esté asignada a otro país por la regla de mayoría de
  PRIO-GRID (así lo exige el codebook de GED).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

GED_VERSION, CANDIDATE_VERSION = "26.1", "26.01.26.09"
URLS = {
    "ged261-csv.zip": "https://ucdp.uu.se/downloads/ged/ged261-csv.zip",
    "GEDEvent_v26_01_26_09.csv": "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_01_26_09.csv",
}
PRIOGRID_API = "https://grid.prio.org/api/data"
# nombre -> (id en la API de PRIO-GRID, año que se toma; 0 = estática). El último año publicado de cada una.
PRIOGRID_VARS = {
    "gwno": (50, 2014), "capdist": (9, 2014), "bdist1": (6, 2014), "pop_gpw_sum": (75, 2010),
    "ttime_mean": (88, 0), "mountains_mean": (63, 0), "landarea": (61, 0), "urban_gc": (91, 0),
    "imr_mean": (54, 0), "excluded": (31, 2013), "petroleum_s": (70, 0), "diamsec_s": (14, 0),
    "goldplacer_s": (40, 0), "gem_s": (37, 0),
}
MANIFEST = Path(__file__).with_name("conflict_grid_manifest.json")
TYPES = ("sb", "ns", "os")                     # type_of_violence 1, 2, 3 de UCDP
FIRST_YEAR = 1989
WHERE_PREC_MAX = 4
RULES = {
    "month": "month of date_start (1.75 % of GED 25.1 events end in another month)",
    "cell": f"events with where_prec <= {WHERE_PREC_MAX} count in their UCDP priogrid_gid; where_prec 5-7 only "
            "in their country (country_id, Gleditsch-Ward)",
    "deaths": "UCDP 'best' estimate",
    "candidate": f"months after GED {GED_VERSION} come from UCDP Candidate {CANDIDATE_VERSION}: preliminary events, "
                 "about a third flagged 'Check ...' by UCDP",
}


class ConflictGridDataError(RuntimeError):
    pass


def root() -> Path:
    from world4 import home
    return home.data_cache() / "conflict_grid"


def npz_path() -> Path:
    return root() / "conflict_grid.npz"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pg_path(name: str) -> Path:
    return root() / "raw" / "priogrid" / f"{name}.json"


def _pg_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(gid, valor) ordenados por gid de una variable de PRIO-GRID descargada."""
    d = json.loads(_pg_path(name).read_text(encoding="utf-8"))
    cells = d if isinstance(d, list) else d["cells"]
    gid = np.array([c["gid"] for c in cells], np.int64)
    val = np.array([np.nan if c.get("value") is None else float(c["value"]) for c in cells]) \
        if name != "basegrid" else np.zeros(len(cells))
    o = np.argsort(gid, kind="stable")
    return gid[o], val[o]


def _pg_canonical_sha(name: str) -> str:
    """La API devuelve JSON sin orden garantizado: se comprueba el hash de la tabla (gid, valor) ordenada."""
    gid, val = _pg_table(name)
    return hashlib.sha256(gid.tobytes() + np.round(val, 9).tobytes()).hexdigest()


def verify_raw() -> dict:
    raw = root() / "raw"
    man = manifest()
    missing, bad = [], []
    for name, meta in man["files"].items():
        p = raw / name
        if not p.is_file():
            missing.append(name)
        elif _sha(p) != meta["sha256"]:
            bad.append(name)
    for name, meta in man["priogrid"].items():
        if not _pg_path(name).is_file():
            missing.append(f"priogrid/{name}")
        elif _pg_canonical_sha(name) != meta["canonical_sha256"]:
            bad.append(f"priogrid/{name}")
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(raw)}


def download(log=print) -> None:
    raw = root() / "raw"
    (raw / "priogrid").mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (raw / name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(url, raw / name)
    if not _pg_path("basegrid").is_file():
        log("downloading PRIO-GRID basegrid")
        urllib.request.urlretrieve(f"{PRIOGRID_API}/basegrid", _pg_path("basegrid"))
    for name, (vid, year) in PRIOGRID_VARS.items():
        if not _pg_path(name).is_file():
            log(f"downloading PRIO-GRID {name}")
            q = f"?startYear={year}&endYear={year}" if year else ""
            urllib.request.urlretrieve(f"{PRIOGRID_API}/{vid}{q}", _pg_path(name))


def make_manifest() -> dict:
    """Lo que se escribió en ``conflict_grid_manifest.json`` tras la descarga del 2026-10-29."""
    raw = root() / "raw"
    return {"source": f"UCDP GED {GED_VERSION} + UCDP Candidate {CANDIDATE_VERSION} (ucdp.uu.se/downloads, CC BY 4.0); "
                      "PRIO-GRID 2.0 through grid.prio.org/api",
            "files": {n: {"sha256": _sha(raw / n), "url": u} for n, u in URLS.items()},
            "priogrid": {n: {"canonical_sha256": _pg_canonical_sha(n),
                             "api": f"{PRIOGRID_API}/" + ("basegrid" if n == "basegrid" else str(PRIOGRID_VARS[n][0])),
                             "year": 0 if n == "basegrid" else PRIOGRID_VARS[n][1]}
                         for n in ("basegrid",) + tuple(PRIOGRID_VARS)}}


def install(log=print) -> dict:
    download(log)
    v = verify_raw()
    if not v["ok"]:
        raise ConflictGridDataError(f"conflict-grid files missing {v['missing']} or with another hash {v['mismatched']}")
    convert(log)
    load.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path())}


def gid_to_rowcol(gid):
    """PRIO-GRID: gid = (fila - 1) * 720 + columna; fila 1 en -90°, columna 1 en -180° (codebook 2.0)."""
    gid = np.asarray(gid, np.int64)
    return (gid - 1) // 720 + 1, (gid - 1) % 720 + 1


def rowcol_to_lonlat(row, col):
    """Centro de la celda en grados decimales."""
    return -180.0 + (np.asarray(col) - 0.5) * 0.5, -90.0 + (np.asarray(row) - 0.5) * 0.5


def _read_events(log) -> "pd.DataFrame":
    import pandas as pd
    raw = root() / "raw"
    cols = ["id", "type_of_violence", "where_prec", "priogrid_gid", "country_id", "date_start", "best"]
    ged = pd.read_csv(raw / "ged261-csv.zip", usecols=cols + ["code_status"], low_memory=False)
    cand = pd.read_csv(raw / "GEDEvent_v26_01_26_09.csv", usecols=cols + ["code_status"], low_memory=False)
    ged["candidate"], cand["candidate"] = False, True
    last_final = pd.to_datetime(ged.date_start).max()
    cand = cand[pd.to_datetime(cand.date_start) > last_final]          # sólo los meses que GED aún no tiene
    log(f"GED {GED_VERSION}: {len(ged)} events to {last_final:%Y-%m}; candidate {CANDIDATE_VERSION}: {len(cand)} after it")
    d = pd.concat([ged, cand], ignore_index=True)
    ds = pd.to_datetime(d.date_start)
    d["month"] = (ds.dt.year - FIRST_YEAR) * 12 + ds.dt.month - 1
    return d


def convert(log=print) -> Path:
    gid, _ = _pg_table("basegrid")
    row, col = gid_to_rowcol(gid)
    lon, lat = rowcol_to_lonlat(row, col)
    static = {}
    for name in PRIOGRID_VARS:
        g, v = _pg_table(name)
        a = np.full(len(gid), np.nan)
        k = np.searchsorted(gid, g)
        ok = (k < len(gid)) & (gid[np.minimum(k, len(gid) - 1)] == g)
        a[k[ok]] = v[ok]
        static[name] = a
    d = _read_events(log)
    cell = np.searchsorted(gid, d.priogrid_gid.fillna(-1).to_numpy(np.int64))
    cell = np.where((cell < len(gid)) & (gid[np.minimum(cell, len(gid) - 1)] == d.priogrid_gid.fillna(-1).to_numpy())
                    & (d.where_prec.to_numpy() <= WHERE_PREC_MAX), cell, -1)
    off_grid = int(((d.where_prec <= WHERE_PREC_MAX).to_numpy() & (cell < 0)).sum())
    log(f"{(cell >= 0).sum()} events placed in a cell; {(cell < 0).sum()} only in their country "
        f"({off_grid} with a precise location outside the PRIO-GRID land cells)")
    n_months = int(d.month.max()) + 1
    out = npz_path()
    np.savez_compressed(
        out, gid=gid, row=row, col=col, lon=lon, lat=lat, **{f"pg_{k}": v for k, v in static.items()},
        ev_month=d.month.to_numpy(np.int32), ev_cell=cell.astype(np.int32),
        ev_type=(d.type_of_violence.to_numpy() - 1).astype(np.int8), ev_deaths=d.best.to_numpy(np.float32),
        ev_country=d.country_id.to_numpy(np.int32), ev_candidate=d.candidate.to_numpy(bool),
        ev_where_prec=d.where_prec.to_numpy(np.int8), first_year=np.int32(FIRST_YEAR), n_months=np.int32(n_months),
        last_final_month=np.int32(d.month[~d.candidate].max()),
        versions=np.array([GED_VERSION, CANDIDATE_VERSION]))
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise ConflictGridDataError(f"no conflict-grid data at {p}: python world4_cli.py --install-conflict-grid")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file()


def month_label(m: int) -> str:
    """Índice de mes (0 = enero de 1989) -> 'AAAA-MM'."""
    return f"{FIRST_YEAR + m // 12}-{m % 12 + 1:02d}"


def month_index(label: str) -> int:
    y, mo = (int(x) for x in str(label)[:7].split("-"))
    return (y - FIRST_YEAR) * 12 + mo - 1
