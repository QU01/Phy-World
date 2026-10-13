"""Open-source intelligence layers for the 3D globe (static snapshots, published next to the page).

The artifact viewer cannot reach external hosts at run time, so live feeds (AIS vessel positions,
ADS-B aircraft) are replaced by open snapshots of the networks they move on:

  plantas     WRI Global Power Plant Database v1.3 (2021): plants >= 50 MW, fuel and capacity
  vuelos      OpenFlights routes (2014 snapshot): busiest airport pairs by number of airlines
  maritimo    Major / middle / minor shipping lanes (Benden 2021, from AIS density; GitHub mirror)
  comercio    Bilateral trade flows from the model's COW dyadic trade matrices, every 5 years
  conflictos  Countries with active armed conflict each year (UCDP/PRIO, the model's panel) and
              geocoded events of UCDP GED v24.1 (1989-2023) on a 0.5 degree grid per year
  cables      Submarine telecom cables (TeleGeography, crawled copy; CC BY-NC-SA 3.0)
  puertos     NGA World Port Index (cleaned copy): large and medium harbours
  energia     GEM LNG terminals, GEM GOGET oil and gas fields, GeoNuclearData reactors in
              construction or planned
  ductos      GEM oil and gas pipelines in operation (GOIT + GGIT, CC BY 4.0)
  trafico     Real traffic samples: ADS-B aircraft over Paris (7 Oct 2021, traffic package,
              OpenSky-derived) and AIS vessels near Gothenburg (5 Jul 2017, Danish Maritime Authority)
Raw downloads go to data/raw/osint/ (not committed); outputs to dashboard/osint/*.json.
"""
import csv
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "osint"
OUT = ROOT / "dashboard" / "osint"
SRC = {
    "wri.csv": "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv",
    "airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "routes.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
    "lanes.geojson": "https://raw.githubusercontent.com/newzealandpaul/Shipping-Lanes/main/data/Shipping_Lanes_v1.geojson",
    "GEDEvent_v24_1_full.csv": "https://media.githubusercontent.com/media/jmanali1996/Civilian-Conflicts/main/GEDEvent_v24_1.csv",
    "submarine_cable-geo.json": "https://raw.githubusercontent.com/lintaojlu/submarine_cable_information/master/web/public/api/v3/cable/cable-geo.json",
    "wpi_ports_casa.csv": "https://raw.githubusercontent.com/johnx25bd/casa-digital-visualisation/master/data/Final%20Data/ports.csv",
    "geonuclear_nuclear_power_plants.csv": "https://raw.githubusercontent.com/cristianst85/GeoNuclearData/master/data/csv/denormalized/nuclear_power_plants.csv",
    "gem_lng_terminals.geojson": "https://raw.githubusercontent.com/GlobalEnergyMonitor/goit-ggit-interim-maps/main/trackers/ggit-goget-gogpt/lng_map_latest.geojson",
    "gem_goget_oil_gas_fields.geojson": "https://raw.githubusercontent.com/GlobalEnergyMonitor/goit-ggit-interim-maps/main/trackers/ggit-goget/goget_map_latest.geojson",
    "gem_goit_map_latest.geojson": "https://raw.githubusercontent.com/GlobalEnergyMonitor/goit-ggit-data-ops/map-data/goit_map_latest.geojson",
    "gem_ggit_map_latest.geojson": "https://raw.githubusercontent.com/GlobalEnergyMonitor/goit-ggit-data-ops/map-data/ggit_map_latest.geojson",
}
FUEL = {"Coal": "carbon", "Gas": "gas", "Oil": "petroleo", "Petcoke": "petroleo", "Nuclear": "nuclear",
        "Hydro": "hidro", "Wind": "eolica", "Solar": "solar", "Geothermal": "otra", "Biomass": "otra",
        "Waste": "otra", "Cogeneration": "otra", "Storage": "otra", "Other": "otra", "Wave and Tidal": "otra"}


def fetch(name):
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / name
    if not f.exists():
        urllib.request.urlretrieve(SRC[name], f)
    return f


def plants(min_mw=50):
    rows = []
    with open(fetch("wri.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mw = float(r["capacity_mw"] or 0)
            if mw < min_mw:
                continue
            rows.append([round(float(r["longitude"]), 2), round(float(r["latitude"]), 2), round(mw),
                         FUEL.get(r["primary_fuel"], "otra"), r["name"][:48], r["country"],
                         int(float(r["commissioning_year"])) if r["commissioning_year"] else None])
    rows.sort(key=lambda x: -x[2])
    tot = Counter()
    for r in rows:
        tot[r[3]] += r[2]
    return {"fuente": "WRI Global Power Plant Database v1.3.0 (2021), centrales de 50 MW o más",
            "campos": ["lon", "lat", "mw", "combustible", "nombre", "pais", "inicio"],
            "capacidad_por_combustible_gw": {k: round(v / 1000, 1) for k, v in tot.most_common()},
            "filas": rows}


def flights(top=2500):
    ap = {}
    with open(fetch("airports.dat"), newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            try:
                ap[r[0]] = (round(float(r[7]), 2), round(float(r[6]), 2), r[4] if r[4] != "\\N" else r[5], r[3])
            except (ValueError, IndexError):
                continue
    pairs = Counter()
    with open(fetch("routes.dat"), newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            a, b = r[3], r[5]
            if a in ap and b in ap and a != b:
                pairs[tuple(sorted((a, b)))] += 1
    rows = []
    for (a, b), n in pairs.most_common(top):
        (x1, y1, c1, k1), (x2, y2, c2, k2) = ap[a], ap[b]
        rows.append([x1, y1, x2, y2, n, c1, c2, int(k1 != k2)])
    deg = Counter()
    for (a, b), n in pairs.items():
        deg[a] += n
        deg[b] += n
    hubs = [[ap[a][0], ap[a][1], n, ap[a][2], ap[a][3]] for a, n in deg.most_common(150)]
    return {"fuente": "OpenFlights (instantánea de rutas de 2014); aerolíneas por par de aeropuertos",
            "campos": ["lon1", "lat1", "lon2", "lat2", "aerolineas", "iata1", "iata2", "internacional"],
            "n_rutas_total": int(sum(pairs.values())), "filas": rows, "hubs": hubs}


def shipping(step=1):
    g = json.load(open(fetch("lanes.geojson")))
    paths = []
    for f in g["features"]:
        kind = {"Major": "principal", "Middle": "media", "Minor": "menor"}.get(f["properties"].get("Type"), "menor")
        geom = f["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for ln in lines:
            pts = [[round(x, 2), round(y, 2)] for x, y in ln[::step]]
            if len(pts) >= 2:
                paths.append([kind, pts])
    return {"fuente": "Shipping Lanes v1 (Benden 2021, derivado de densidad AIS), github.com/newzealandpaul/Shipping-Lanes",
            "campos": ["tipo", "puntos[lon,lat]"], "filas": paths}


def conflict_events(res=0.5):
    """UCDP GED v24.1 events aggregated per year on a res-degree grid:
    [lon, lat, events, deaths (best), dominant type 1 state / 2 non-state / 3 one-sided]."""
    import pandas as pd
    df = pd.read_csv(fetch("GEDEvent_v24_1_full.csv"), usecols=["year", "type_of_violence", "latitude", "longitude", "best"])
    df["gx"] = (np.floor(df.longitude / res) * res + res / 2).round(2)
    df["gy"] = (np.floor(df.latitude / res) * res + res / 2).round(2)
    g = df.groupby(["year", "gx", "gy"])
    agg = g.agg(n=("best", "size"), d=("best", "sum")).reset_index()
    dom = df.groupby(["year", "gx", "gy", "type_of_violence"])["best"].sum().reset_index()
    dom = dom.sort_values("best").drop_duplicates(["year", "gx", "gy"], keep="last")
    agg = agg.merge(dom[["year", "gx", "gy", "type_of_violence"]], on=["year", "gx", "gy"])
    out = {int(y): sub[["gx", "gy", "n", "d", "type_of_violence"]].values.tolist() for y, sub in agg.groupby("year")}
    out = {y: [[float(a), float(b), int(c), int(d), int(e)] for a, b, c, d, e in rows] for y, rows in out.items()}
    tot = df.groupby("year").best.sum()
    return {"fuente": "UCDP Georeferenced Event Dataset v24.1 (Sundberg y Melander 2013; Davies et al. 2024), CC BY 4.0; "
                      "copia en GitHub del archivo oficial", "campos": ["lon", "lat", "eventos", "muertes", "tipo"],
            "resolucion_grados": res, "muertes_por_anio": {int(k): int(v) for k, v in tot.items()}, "por_anio": out}


def cables():
    g = json.load(open(fetch("submarine_cable-geo.json")))
    rows = []
    for f in g["features"]:
        geom = f["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for ln in lines:
            pts = [[round(x, 2), round(y, 2)] for x, y in ln]
            if len(pts) >= 2:
                rows.append([f["properties"]["name"][:60], f["properties"].get("color", "#8ad"), pts])
    return {"fuente": "TeleGeography Submarine Cable Map (copia rastreada en GitHub), CC BY-NC-SA 3.0",
            "campos": ["nombre", "color", "puntos"], "n_cables": len(g["features"]), "filas": rows}


def ports():
    rows = []
    with open(fetch("wpi_ports_casa.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["harborsize"] in ("L", "M"):
                rows.append([round(float(r["longitude"]), 2), round(float(r["latitude"]), 2), r["port_name"].title()[:32],
                             r["harborsize"], r["iso3"], int(float(r["busiest_ports_ranking"] or 0))])
    return {"fuente": "NGA World Port Index (Pub. 150), copia depurada en GitHub; puertos grandes (L) y medianos (M)",
            "campos": ["lon", "lat", "nombre", "tamano", "iso3", "ranking_mas_transitados"], "filas": rows}


def energy():
    rows = []
    g = json.load(open(fetch("gem_lng_terminals.geojson")))
    for f in g["features"]:
        p = f["properties"]
        if p.get("Status") != "operating" or not f.get("geometry"):
            continue
        x, y = f["geometry"]["coordinates"][:2]
        cap = float(p.get("Capacity") or 0)
        rows.append([round(x, 2), round(y, 2), cap, "gnl", p["PipelineName"][:50], p.get("FacilityType", "")])
    g = json.load(open(fetch("gem_goget_oil_gas_fields.geojson")))
    for f in g["features"]:
        p = f["properties"]
        boe = float(p.get("CapacityBOEd") or 0)
        if p.get("Status") != "operating" or boe < 20000 or not f.get("geometry"):
            continue
        x, y = f["geometry"]["coordinates"][:2]
        rows.append([round(x, 2), round(y, 2), round(boe), "yacimiento", p["PipelineName"].split(" Oil and Gas")[0][:50], ""])
    with open(fetch("geonuclear_nuclear_power_plants.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["Status"] in ("Under Construction", "Planned") and r["Latitude"]:
                rows.append([round(float(r["Longitude"]), 2), round(float(r["Latitude"]), 2), float(r["Capacity"] or 1000),
                             "nuclear_nuevo", r["Name"][:50], r["Status"]])
    return {"fuente": "Global Energy Monitor (terminales de GNL en operación; yacimientos de petróleo y gas con ≥ 20 mil bep/d), CC BY 4.0; "
                      "GeoNuclearData (reactores en construcción o planeados), ODbL",
            "campos": ["lon", "lat", "capacidad (mtpa | bep/d | MW)", "tipo", "nombre", "detalle"], "filas": rows}


def pipelines(step=0.1):
    g = json.load(open(fetch("gem_pipelines_oil_gas_compact.geojson")))
    rows = []
    for f in g["features"]:
        p = f["properties"]
        if p.get("Status") != "operating" or not f.get("geometry"):
            continue
        geom = f["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for ln in lines:
            pts = []
            for x, y in ln:
                q = [round(round(x / step) * step, 1), round(round(y / step) * step, 1)]
                if not pts or q != pts[-1]:
                    pts.append(q)
            if len(pts) >= 2:
                rows.append(["petroleo" if "Oil" in (p.get("Fuel") or "") or "NGL" in (p.get("Fuel") or "") else "gas",
                             (p.get("PipelineName") or "")[:50], pts])
    return {"fuente": "Global Energy Monitor, Global Oil Infrastructure Tracker y Global Gas Infrastructure Tracker "
                      "(ductos en operación), CC BY 4.0", "campos": ["fluido", "nombre", "puntos"], "filas": rows}


def traffic():
    import pandas as pd
    tr = []
    a = pd.read_csv(fetch("adsb_traffic_quickstart_20211007_paris.csv"), usecols=["timestamp", "icao24", "callsign", "latitude", "longitude", "altitude"])
    a = a.dropna(subset=["latitude", "longitude", "altitude"]).sort_values("timestamp")
    for k, s in a.groupby("icao24"):
        s = s.iloc[::15]
        if len(s) > 3:
            tr.append(["avion", str(s.callsign.dropna().iloc[0]) if s.callsign.notna().any() else k,
                       [[round(x, 3), round(y, 3), round(z)] for x, y, z in zip(s.longitude, s.latitude, s.altitude)]])
    v = pd.read_csv(fetch("ais_denmark_20170705_gothenburg.csv"), usecols=["timestamp_utc", "mmsi", "lat", "lon", "sog", "ship_type"])
    v = v.sort_values("timestamp_utc")
    for k, s in v.groupby("mmsi"):
        s = s[s.sog > 0.5].iloc[::10]
        if len(s) > 3:
            tr.append(["barco", f"{s.ship_type.iloc[0]} · MMSI {k}", [[round(x, 4), round(y, 4), 0] for x, y in zip(s.lon, s.lat)]])
    return {"fuente": "ADS-B: muestra del paquete traffic (OpenSky Network), París, 7 oct 2021 12-15 UTC; "
                      "AIS: Agencia Marítima Danesa vía movingpandas, Gotemburgo, 5 jul 2017",
            "campos": ["tipo", "id", "puntos[lon,lat,altitud_pies]"], "filas": tr}


def centroids(geo, iso_num):
    """Label point per country: mean of the vertices of its largest polygon."""
    out = {}
    for f in geo["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        ring = max((p[0] for p in polys), key=len)
        a = np.array(ring)
        out[f["id"]] = [round(float(a[:, 0].mean()), 2), round(float(a[:, 1].mean()), 2)]
    return out


def model_layers():
    from .calibrate import W
    from .geo import load_countries
    geo = load_countries()
    cen = centroids(geo, W.isonum)
    num = {i: n for i, n in enumerate(W.isonum)}
    pos = [cen.get(num[i]) for i in range(W.N)]
    # active conflicts per year (UCDP/PRIO in the panel)
    conf = {}
    for t in range(W.T):
        conf[1950 + t] = [i for i in range(W.N) if W.conflict[i, t] == 1 and pos[i] is not None]
    # bilateral trade flows: exports X_i = openness * GDP split by COW export shares
    trade = {}
    if W.S_exp is not None:
        for t in range(0, min(W.T, 65), 5):
            X = np.nan_to_num(W.open_[:, t] * W.Y[:, t] / 2)             # PWT output in millions of USD
            F = X[:, None] * np.nan_to_num(W.S_exp[t])
            idx = np.dstack(np.unravel_index(np.argsort(-F, axis=None)[:300], F.shape))[0]
            trade[1950 + t] = [[int(i), int(j), round(float(F[i, j]) / 1e3, 1)] for i, j in idx
                               if F[i, j] > 0 and pos[i] is not None and pos[j] is not None]
    return {"posiciones": pos, "conflictos_por_anio": conf, "comercio_por_anio": trade,
            "fuente_conflictos": "UCDP/PRIO Armed Conflict Dataset (país con conflicto activo en el año)",
            "fuente_comercio": "Correlates of War, Dyadic Trade 1950-2014 (cuotas) × apertura × PIB (PWT); miles de millones de USD PPA de 2017"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("plantas", plants), ("vuelos", flights), ("maritimo", shipping), ("modelo", model_layers),
                     ("conflictos_ged", conflict_events), ("cables", cables), ("puertos", ports), ("energia", energy),
                     ("ductos", pipelines), ("trafico", traffic)):
        d = fn()
        txt = json.dumps(d, separators=(",", ":"), ensure_ascii=False).replace("\ufffd", "")   # broken bytes in some names
        (OUT / f"{name}.json").write_text(txt, encoding="utf-8")
        print(name, round((OUT / f"{name}.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
