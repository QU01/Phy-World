"""Datos del L2 (F12): eventos de violencia organizada, malla, desplazamiento y población de una región.

Fuentes, todas comprobadas contra ``l2_manifest.json`` (SHA-256) y ninguna incluida en el repo:

* **UCDP GED 26.1** (1989-2025, global; CC BY 4.0 [V], leída en ucdp.uu.se/downloads el 2026-11-16) y
  **UCDP Candidate 26.0.X** (enero-septiembre de 2026, mensual). Del candidato sólo entran los eventos con
  ``code_status == "Clear"``: el resto está pendiente de revisión por UCDP y puede desaparecer [declarado].
  El fichero candidato cambia de nombre cada mes: si ya no está en el servidor la instalación sigue sin él
  y lo anuncia (el panel acaba entonces en diciembre de 2025).
* **PRIO-GRID 2.0** (API de grid.prio.org): país de cada celda, población GPW v3 (1990-2010), distancias,
  montaña, tiempo de viaje, luces nocturnas, grupos excluidos. «Freely available», sin licencia explícita y
  con variables de terceros: se trata como NO redistribuible.
* **ACNUR, paquete R ``refugees``** (CC BY 4.0 [V], LICENSE.md del repo): refugiados y solicitantes por
  origen y asilo, desplazados internos, y la tabla de IDMC.
* **UN WPP 2024** (``PPgp/wpp2024``, CC BY 3.0 IGO): población total por país, estimada hasta 2023 y
  proyectada después.

Convenciones del panel que escribe ``convert``:

* mes ``t = (año - 1989) * 12 + mes - 1``; un evento que cruza de mes reparte sus muertes a partes iguales
  entre los meses que abarca;
* la celda de un evento es su ``priogrid_gid`` cuando UCDP conoce el lugar al menos como línea o provincia
  (``where_prec`` <= 5; 4 y 5 son centroides, no el punto). Con ``where_prec`` 6-7 (sólo el país, o aguas
  y espacio aéreo internacionales) el evento cuenta en el país y NO en ninguna celda;
* el país de un evento es el que dice UCDP (``country_id``), no el de la celda;
* muertes = ``best``; ``low`` y ``high`` se guardan.

``known_time``: UCDP publica el GED del año ``y`` a mediados de ``y + 1`` y el candidato con un mes de
retraso. Los pronósticos retrospectivos de W13 usan el GED final hasta el mes anterior al origen, que en su
día sólo existía como candidato (o no existía): es una fuga de revisión declarada (``D-L2-1``).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from world4.core.l2 import regions

MANIFEST = Path(__file__).with_name("l2_manifest.json")
YEAR0 = 1989                                  # primer año del GED
POP_YEAR1 = 2035                              # último año de población (proyección de WPP)
UNHCR_COMMIT = "c7bd34a3de4fdfb8f46bf6345d7401247315906e"      # refugees 2025.12.3
WPP_COMMIT = "2da7768ae64fc74105d3f9e98f9a74d37b62f99a"
GED_ZIP, GED_CSV = "ged261-csv.zip", "GEDEvent_v26_1.csv"
CANDIDATE = "GEDEvent_v26_01_26_09.csv"
PRIO_VARS = {                                  # variable: (primer año, último año) o None si es estática
    "gwno": (2014, 2014), "pop_gpw_sum": (1990, 2010), "capdist": (2014, 2014), "bdist1": (2014, 2014),
    "excluded": (2013, 2013), "nlights_calib_mean": (2012, 2012), "mountains_mean": None, "ttime_mean": None,
    "petroleum_s": None, "urban_gc": None, "landarea": None, "imr_mean": None, "agri_gc": None, "barren_gc": None,
}
COVARIATES = ("capdist", "bdist1", "excluded", "nlights_calib_mean", "mountains_mean", "ttime_mean",
              "petroleum_s", "urban_gc", "landarea", "imr_mean", "agri_gc", "barren_gc")
OPTIONAL = (CANDIDATE,)
_UN = "https://raw.githubusercontent.com/PopulationStatistics/refugees/" + UNHCR_COMMIT + "/data/"
_WPP = "https://raw.githubusercontent.com/PPgp/wpp2024/" + WPP_COMMIT + "/data/"


def _prio_url(var: str) -> str:
    span = PRIO_VARS[var]
    q = "" if span is None else f"?startYear={span[0]}&endYear={span[1]}"
    return f"https://grid.prio.org/api/data/{var}{q}"


URLS = {
    GED_ZIP: "https://ucdp.uu.se/downloads/ged/" + GED_ZIP,
    CANDIDATE: "https://ucdp.uu.se/downloads/candidateged/" + CANDIDATE,
    "unhcr_population.rda": _UN + "population.rda", "unhcr_flows.rda": _UN + "flows.rda",
    "unhcr_idmc.rda": _UN + "idmc.rda",
    "popAge1dt.rda": _WPP + "popAge1dt.rda", "popproj1dt.rda": _WPP + "popproj1dt.rda",
    **{f"prio_{v}.json": _prio_url(v) for v in PRIO_VARS},
}
GED_COLS = ["id", "year", "code_status", "type_of_violence", "dyad_new_id", "dyad_name", "side_a_new_id", "side_a",
            "side_b_new_id", "side_b", "where_prec", "priogrid_gid", "country", "country_id", "date_start",
            "date_end", "deaths_a", "deaths_b", "deaths_civilians", "best", "high", "low"]


class ConflictDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "l2"


def npz_path(region: str = "mideast") -> Path:
    return root() / f"l2_{region}.npz"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    """Cada fichero del manifiesto. Los opcionales (el candidato del mes) pueden faltar; no pueden ser otros."""
    raw = root() / "raw"
    missing, bad, absent_optional = [], [], []
    for name, meta in manifest()["files"].items():
        p = raw / name
        if not p.is_file():
            (absent_optional if name in OPTIONAL else missing).append(name)
        elif _sha(p) != meta["sha256"]:
            bad.append(name)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad,
            "absent_optional": absent_optional, "root": str(raw)}


def install(src: str | Path | None = None, region: str = "mideast", log=print) -> dict:
    """Descarga (o copia desde ``src``, un directorio con los ficheros ya bajados), comprueba y convierte."""
    import shutil
    raw = root() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in manifest()["files"]:
        dst = raw / name
        if dst.is_file():
            continue
        if src is not None and (Path(src) / name).is_file():
            shutil.copy2(Path(src) / name, dst)
            continue
        log(f"downloading {name}")
        try:
            urllib.request.urlretrieve(URLS[name], dst)
        except Exception as e:  # noqa: BLE001
            dst.unlink(missing_ok=True)
            if name in OPTIONAL:
                log(f"[degraded] {name} is not on the server any more ({type(e).__name__}): the panel will end "
                    "with the final GED")
                continue
            raise ConflictDataError(f"cannot download {name} from {URLS[name]}: {e}") from e
    v = verify_raw()
    if not v["ok"]:
        raise ConflictDataError(f"L2 files missing {v['missing']} or with another SHA-256 {v['mismatched']}")
    log(f"converting {', '.join(regions.names(region))}")
    out = convert(region)
    load.cache_clear()
    return {**v, "npz": str(out), "npz_sha256": _sha(out),
            "regions": {r: str(npz_path(r)) for r in regions.names(region)}}


# ------------------------------------------------------------------ conversión
def _month(ts) -> np.ndarray:
    return (ts.dt.year.to_numpy() - YEAR0) * 12 + ts.dt.month.to_numpy() - 1


def _read_ged(raw: Path):
    """El GED mundial más lo que el candidato añade después de su último día; ordenado por fecha."""
    import pandas as pd
    with zipfile.ZipFile(raw / GED_ZIP) as z, z.open(GED_CSV) as fh:
        g = pd.read_csv(fh, usecols=GED_COLS + ["region"], low_memory=False)
    g["candidate"] = 0
    cand = raw / CANDIDATE
    if cand.is_file():
        c = pd.read_csv(cand, usecols=GED_COLS + ["region"], low_memory=False)
        c = c[c.code_status == "Clear"].copy()
        c["candidate"] = 1
        last = pd.to_datetime(g.date_start).max()
        c = c[pd.to_datetime(c.date_start) > last]          # el candidato sólo añade lo posterior al GED final
        g = pd.concat([g, c], ignore_index=True)
    g["d0"] = pd.to_datetime(g.date_start)
    g["d1"] = pd.to_datetime(g.date_end)
    return g.sort_values(["d0", "id"]).reset_index(drop=True)


_PRIO: dict = {}


def _prio(raw: Path, var: str) -> list:
    if var not in _PRIO:
        _PRIO[var] = json.loads((raw / f"prio_{var}.json").read_text(encoding="utf-8"))["cells"]
    return _PRIO[var]


def _gwno(raw: Path) -> dict:
    """País (código GW, con los alias de ``regions.PRIO_ALIAS``) de cada celda de PRIO-GRID."""
    return {c["gid"]: regions.PRIO_ALIAS.get(int(c["value"]), int(c["value"])) for c in _prio(raw, "gwno")}


def country_table() -> dict:
    """GW -> (ISO3, M49, continente), del paquete ``country_converter`` con las excepciones de
    ``regions.GW_ISO3``. Sólo hace falta al convertir (extra ``world4[l2]``)."""
    import logging

    import country_converter as coco
    lvl = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        d = coco.CountryConverter().data
    finally:
        logging.getLogger().setLevel(lvl)
    by_iso = {str(r.ISO3): (int(r.ISOnumeric), str(r.continent)) for r in d.itertuples() if r.ISOnumeric == r.ISOnumeric}
    out = {int(r.GWcode): (str(r.ISO3), int(r.ISOnumeric), str(r.continent))
           for r in d.itertuples() if r.GWcode == r.GWcode and r.ISOnumeric == r.ISOnumeric}
    for g, iso in regions.GW_ISO3.items():
        out[g] = (iso, *by_iso[iso])
    return out


def resolve_regions(g, gwno: dict) -> dict:
    """Región -> [(iso3, gw, m49)] para todos los países con celdas en PRIO-GRID o eventos en el GED."""
    table = country_table()
    codes = sorted(set(gwno.values()) | {int(x) for x in g.country_id.unique()})
    ucdp = {int(k): str(v) for k, v in g.drop_duplicates("country_id")[["country_id", "region"]].values}
    where = regions.assign(codes, ucdp, table)
    out = {r: [] for r in regions.WORLD}
    for gw, r in where.items():
        if gw in table:
            out[r].append((table[gw][0], gw, table[gw][1]))
    return out


def _rda(path: Path):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = rdata.read_rda(str(path))
    return next(iter(d.values()))


def _actor_kind(name: str) -> int:
    """0 gobierno(s), 1 grupo no estatal, 2 civiles."""
    if name == "Civilians":
        return 2
    return 0 if name.startswith("Government of") else 1


def convert(region: str = "mideast") -> Path:
    """Escribe ``l2_<region>.npz`` (uno por región si ``region`` es ``world``; devuelve el último). Necesita
    ``rdata`` y ``country_converter`` (extra ``world4[l2]``); leer los ``.npz`` no."""
    raw = root() / "raw"
    g = _read_ged(raw)
    gwno = _gwno(raw)
    members = resolve_regions(g, gwno)
    tables = {"popu": _rda(raw / "unhcr_population.rda"), "flows": _rda(raw / "unhcr_flows.rda"),
              "idmc": _rda(raw / "unhcr_idmc.rda"), "est": _rda(raw / "popAge1dt.rda"),
              "proj": _rda(raw / "popproj1dt.rda")}
    tables["est"] = tables["est"][tables["est"].year >= YEAR0].groupby(["country_code", "year"]).pop.sum().reset_index()
    tables["last_final_month"] = int(np.minimum(_month(g.d1), _month(g.d0).max())[g.candidate.to_numpy() == 0].max())
    out = None
    for name in regions.names(region):
        reg = regions.build(name, members[name])
        out = _convert_one(reg, g[g.country_id.isin(reg.gw)].reset_index(drop=True), gwno, tables, int(_month(g.d0).max()) + 1)
    _PRIO.clear()
    return out


def _convert_one(reg: regions.Region, g, gwno: dict, tables: dict, T: int) -> Path:
    import pandas as pd
    raw = root() / "raw"
    gw_idx = {g_: k for k, g_ in enumerate(reg.gw)}

    # ---- eventos (``T`` es el del GED mundial: todas las regiones acaban en el mismo mes)
    t0, t1 = _month(g.d0), _month(g.d1)
    t1 = np.minimum(t1, T - 1)                  # el mes siguiente al último con eventos que empiezan es parcial
    located = (g.where_prec.to_numpy() <= 5) & g.priogrid_gid.notna().to_numpy()
    ev_gid = np.where(located, g.priogrid_gid.fillna(0).to_numpy(), 0).astype(np.int64)

    # ---- celdas: las del país según PRIO-GRID más cualquiera con un evento de la región
    gids = sorted({gid for gid, v in gwno.items() if v in gw_idx} | set(ev_gid[located].tolist()))
    gids = np.array(gids, np.int64)
    cidx = {int(x): i for i, x in enumerate(gids)}
    row, col = (gids - 1) // 720, (gids - 1) % 720
    lon, lat = -180.0 + (col + 0.5) * 0.5, -90.0 + (row + 0.5) * 0.5
    cell_country = np.array([gw_idx.get(gwno.get(int(x), 0), -1) for x in gids], np.int16)
    cov = np.full((len(gids), len(COVARIATES)), np.nan, np.float32)
    for j, var in enumerate(COVARIATES):
        for c in _prio(raw, var):
            i = cidx.get(c["gid"])
            if i is not None and c["value"] is not None:
                cov[i, j] = c["value"]
    pop_years = np.array([1990, 1995, 2000, 2005, 2010])
    pop_gpw = np.full((len(gids), len(pop_years)), np.nan, np.float32)
    for c in _prio(raw, "pop_gpw_sum"):
        i = cidx.get(c["gid"])
        if i is not None and c["value"] is not None:
            pop_gpw[i, int(np.searchsorted(pop_years, c["year"]))] = c["value"]

    # ---- actores y díadas
    names = pd.concat([g[["side_a_new_id", "side_a"]].set_axis(["id", "name"], axis=1),
                       g[["side_b_new_id", "side_b"]].set_axis(["id", "name"], axis=1)]).drop_duplicates("id")
    names = names.sort_values("id").reset_index(drop=True)
    aidx = {int(a): i for i, a in enumerate(names.id)}
    ucdp_name = {n: k for k, n in zip(g.country_id.map(gw_idx), g.country)}
    home = []
    for n in names.name:
        k = -1
        if n.startswith("Government of") and "," not in n:
            k = ucdp_name.get(n[len("Government of "):], -1)
        home.append(k)
    dy = g.drop_duplicates("dyad_new_id").sort_values("dyad_new_id")
    didx = {int(d): i for i, d in enumerate(dy.dyad_new_id)}

    # ---- desplazamiento (ACNUR) y población (WPP)
    popu, flows, idmc = tables["popu"], tables["flows"], tables["idmc"]
    years = np.arange(YEAR0, int(popu.year.max()) + 1)
    unit = {iso: k for k, iso in enumerate(reg.iso3)}
    for iso, extra in reg.extra_iso3.items():
        unit.update({e: unit[iso] for e in extra})
    asylum = sorted({str(x) for x in pd.concat([popu.coa_iso, flows.coa_iso]).dropna().unique() if len(str(x)) == 3})
    jidx = {a: j for j, a in enumerate(asylum)}
    K, J, Y = reg.K, len(asylum), len(years)
    ref_stock, ref_flow = np.zeros((K, J, Y), np.float32), np.zeros((K, J, Y), np.float32)
    idp_stock, idmc_stock = np.zeros((K, Y), np.float32), np.full((K, Y), np.nan, np.float32)
    for tab, arr, cols in ((popu, ref_stock, ["refugees", "asylum_seekers"]),
                           (flows, ref_flow, ["refugees", "asylum_seekers", "refugee_like"])):
        d = tab[tab.coo_iso.isin(unit) & tab.coa_iso.isin(jidx) & tab.year.between(years[0], years[-1])]
        v = d[cols].fillna(0).sum(axis=1).to_numpy()
        np.add.at(arr, (d.coo_iso.map(unit).to_numpy(), d.coa_iso.map(jidx).to_numpy(),
                        d.year.astype(int).to_numpy() - YEAR0), v)
    d = popu[popu.coo_iso.isin(unit) & popu.year.between(years[0], years[-1])]
    np.add.at(idp_stock, (d.coo_iso.map(unit).to_numpy(), d.year.astype(int).to_numpy() - YEAR0),
              d.idps.fillna(0).to_numpy())
    d = idmc[idmc.coo_iso.isin(unit) & idmc.year.between(years[0], years[-1])]
    tmp = np.zeros((K, Y), np.float32)
    seen = np.zeros((K, Y), bool)
    kk, yy = d.coo_iso.map(unit).to_numpy(), d.year.astype(int).to_numpy() - YEAR0
    np.add.at(tmp, (kk, yy), d.total.fillna(0).to_numpy())
    seen[kk, yy] = True
    idmc_stock[seen] = tmp[seen]

    m49 = {c: k for k, c in enumerate(reg.m49)}
    for iso, extra in reg.extra_m49.items():
        m49.update({e: reg.index(iso) for e in extra})
    pop_y = np.arange(YEAR0, POP_YEAR1 + 1)
    pop = np.zeros((K, len(pop_y)))
    est = tables["est"]
    est = est[est.country_code.isin(m49)]
    proj = tables["proj"]
    proj = proj[proj.country_code.isin(m49) & (proj.year <= POP_YEAR1)].copy()
    proj["pop"] = proj.popM + proj.popF                                  # la mediana de la proyección
    for tab in (est, proj):
        np.add.at(pop, (tab.country_code.map(m49).to_numpy(), tab.year.astype(int).to_numpy() - YEAR0),
                  tab["pop"].to_numpy() * 1e3)
    first_projected = int(proj.year.min())

    out = npz_path(reg.code)
    final = g.candidate.to_numpy() == 0
    np.savez_compressed(
        out, region=np.array(reg.code), iso3=np.array(reg.iso3), gw=np.array(reg.gw), m49=np.array(reg.m49),
        year0=np.array(YEAR0), T=np.array(T),
        last_final_month=np.array(int(tables["last_final_month"]) if "last_final_month" in tables
                                  else int(t1[final].max()) if final.any() else T - 1),
        # celdas
        gid=gids, row=row.astype(np.int32), col=col.astype(np.int32), lon=lon, lat=lat, cell_country=cell_country,
        cov=cov, cov_names=np.array(COVARIATES), pop_gpw=pop_gpw, pop_gpw_years=pop_years,
        # eventos
        ev_t0=t0.astype(np.int32), ev_t1=t1.astype(np.int32),
        ev_cell=np.array([cidx[int(x)] if ok else -1 for x, ok in zip(ev_gid, located)], np.int32),
        ev_country=g.country_id.map(gw_idx).to_numpy().astype(np.int16),
        ev_type=g.type_of_violence.to_numpy().astype(np.int8),
        ev_dyad=g.dyad_new_id.map(didx).to_numpy().astype(np.int32),
        ev_best=g.best.to_numpy().astype(np.float32), ev_low=g.low.to_numpy().astype(np.float32),
        ev_high=g.high.to_numpy().astype(np.float32),
        ev_deaths_a=g.deaths_a.fillna(0).to_numpy().astype(np.float32),
        ev_deaths_b=g.deaths_b.fillna(0).to_numpy().astype(np.float32),
        ev_civ=g.deaths_civilians.fillna(0).to_numpy().astype(np.float32),
        ev_candidate=g.candidate.to_numpy().astype(np.int8),
        # díadas y actores
        dyad_id=dy.dyad_new_id.to_numpy().astype(np.int64), dyad_name=dy.dyad_name.to_numpy().astype(str),
        dyad_type=dy.type_of_violence.to_numpy().astype(np.int8),
        dyad_a=dy.side_a_new_id.map(aidx).to_numpy().astype(np.int32),
        dyad_b=dy.side_b_new_id.map(aidx).to_numpy().astype(np.int32),
        actor_id=names.id.to_numpy().astype(np.int64), actor_name=names.name.to_numpy().astype(str),
        actor_kind=np.array([_actor_kind(n) for n in names.name], np.int8), actor_home=np.array(home, np.int16),
        # desplazamiento y población
        years=years, asylum_iso3=np.array(asylum), ref_stock=ref_stock, ref_flow=ref_flow, idp_stock=idp_stock,
        idmc_stock=idmc_stock, pop_years=pop_y, pop=pop, pop_first_projected=np.array(first_projected),
    )
    return out


@lru_cache(maxsize=8)
def load(region: str = "mideast") -> dict:
    p = npz_path(region)
    if not p.is_file():
        raise ConflictDataError(f"no L2 data at {p}: python world4_cli.py --install-l2")
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "gw" in d:                                # la lista de países de la región viaja con sus datos
        regions.build(region, list(zip((str(x) for x in d["iso3"]), d["gw"].tolist(), d["m49"].tolist())))
    return d


def available(region: str = "mideast") -> bool:
    return all(npz_path(r).is_file() for r in regions.names(region))
