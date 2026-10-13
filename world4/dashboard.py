"""F9: el dashboard con el globo del Atlas.

Reutiliza el globo del dashboard del Atlas: la misma ``globe.gl`` 2.46.2, fijada como en el Atlas, y la misma
geometría (Natural Earth 1:110m del paquete world-atlas, ``data/raw/countries-110m.json`` del repo del Atlas en
el commit fijado, verificada por SHA-256 en ``geo_manifest.json``). Las capas son de World4:

* **L1 F1-F5 desde el origen 2019 a 10 años (2029)**, por país: crecimiento del PIB per cápita, cambio de la
  población, riesgo de conflicto y deuda/PIB; p10/p50/p90 del ensemble CRUDO (las bandas recalibradas de W8
  sólo existen en ``api.query`` y en el estimador estructural). Una sola corrida del L1 da todos los países.
* **L0 regional a 2100** (exploratorio): PIB per cápita de cada una de las 10 regiones con el posterior de F7.

Un solo fichero HTML con los datos y la geometría dentro (se abre sin servidor). Lo único externo es la
librería del globo, desde el CDN y con versión fijada; sin red, la página lo dice y muestra la tabla.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

GEO_FILE = "data/raw/countries-110m.json"
GLOBE_GL = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js"
MANIFEST = Path(__file__).resolve().parent / "data" / "geo_manifest.json"
HORIZON = 10


def geo_path() -> Path:
    from world4.data import atlas
    return atlas.root() / GEO_FILE


def install_geo() -> Path:
    """Descarga la geometría del repo del Atlas en el commit fijado y comprueba su SHA-256."""
    from world4.data import atlas
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p = geo_path()
    if not (p.is_file() and atlas.sha256(p) == man["sha256"]):
        url = f"https://raw.githubusercontent.com/QU01/Reproducible-Research-Project-2/{atlas.ATLAS_COMMIT}/{GEO_FILE}"
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:          # noqa: S310 (URL fija)
            blob = r.read()
        if hashlib.sha256(blob).hexdigest() != man["sha256"]:
            raise RuntimeError(f"{GEO_FILE} from the Atlas repo does not match its SHA-256 in {MANIFEST.name}")
        p.write_bytes(blob)
    return p


def _q(a, axis=0):
    q = np.nanpercentile(a, [10, 50, 90], axis=axis)
    return [None if not np.isfinite(v) else round(float(v), 5) for v in q]


def layers(R: int = 32, seed: int = 0, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l0r import regions
    from world4.structural import engine
    W = engine.live_world()
    t0 = engine.L1_LAST_DATA_YEAR - 1950
    log(f"dashboard: L1 F1-F5 from {1950 + t0}, {R} members, {HORIZON} years")
    f = coupling.forecast(t0, R=R, seed=seed, world=W, horizon=HORIZON, f5=coupling.F5Config())
    ly, pop, haz, debt = f["ly"], f["pop"], f["hazard"], f["debt"]
    g = (ly[:, :, -1] - ly[:, :, 0]) / HORIZON * 100
    dp = (pop[:, :, -1] / pop[:, :, 0] - 1) * 100
    rows = {}
    ridx = regions.region_index(W.iso3, W.isonum)
    for i, iso in enumerate(W.iso3):
        if not np.isfinite(ly[:, i, 0]).any():
            continue
        rows[str(int(W.isonum[i]))] = {
            "iso3": iso, "region": regions.CODES[ridx[i]] if ridx[i] >= 0 else None,
            "gdp_pc_growth": _q(g[:, i]), "population_change": _q(dp[:, i]),
            "conflict_hazard": _q(haz[:, i, -1]), "debt_gdp": _q(debt[:, i, -1])}
    log("dashboard: L0 regional to 2100 with the F7 posterior")
    from world4 import api
    r0 = api.run_l0r(posterior=True, members=200)
    # PIB per cápita por región (la API da el mundo): una corrida del ensemble para las regiones
    from world4.core.l0r import model
    from world4.calib import posterior
    from world4.scenarios import robust
    post = posterior.load()["l0r"]
    mult = posterior.draw_l0r(post, 200, np.random.default_rng(seed))
    p, ini = robust._base()
    rr = model.L0R(p, ini, M=200, mult=mult).run(81, record=("Y", "P"))
    gpc = rr["Y"][-1] / rr["P"][-1]
    reg = {c: _q(gpc[:, k]) for k, c in enumerate(regions.CODES)}
    return {"origin": 1950 + t0, "target": 1950 + t0 + HORIZON, "members": R, "countries": rows,
            "units": {"gdp_pc_growth": "%/year, mean 2019-2029", "population_change": "% 2019-2029",
                      "conflict_hazard": "probability of civil conflict in 2029", "debt_gdp": "public debt / GDP, 2029",
                      "l0_gdp_pc_2100": "2017 USD PPP per person, 2100 (exploratory)"},
            "l0_regions_gdp_pc_2100": reg, "l0_world": {"gmst_2100": [r0["world"]["gmst"][q][-1] for q in ("p10", "p50", "p90")]},
            "notes": ["raw L1 ensemble p10/p50/p90 (not recalibrated; W8 recalibration lives in api.query)",
                      "origin 2019: no data after 2019 is assimilated (7 years behind today)",
                      "the L0 layer to 2100 is a behaviour mode, not a forecast"],
            "exploratory": True}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Phy-World globe</title>
<style>
:root{--bg:#0b1522;--ink:#dfe8f1;--dim:#8fa3b8;--panel:rgba(10,16,22,.72)}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
#globe{position:fixed;inset:0}#ui{position:fixed;left:12px;top:12px;max-width:340px;background:var(--panel);
padding:10px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.12)}h1{font-size:15px;margin:0 0 6px}
select{width:100%;margin:6px 0;background:#13202e;color:var(--ink);border:1px solid #2a3a4c;padding:4px;border-radius:4px}
.note{color:var(--dim);font-size:11.5px}#tip{position:fixed;right:12px;top:12px;max-width:300px;background:var(--panel);
padding:10px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.12);display:none}
#fallback{display:none;position:fixed;inset:60px 20px 20px;overflow:auto;background:var(--panel);padding:12px;border-radius:8px}
table{border-collapse:collapse;font-size:12px}td,th{padding:2px 6px;border-bottom:1px solid #22303e;text-align:left}
.legend{display:flex;height:10px;border-radius:3px;overflow:hidden;margin-top:4px}.legend span{flex:1}
</style></head><body><div id="globe"></div>
<div id="ui"><h1>Phy-World (World4)</h1>
<div class="note" id="sub"></div>
<select id="layer"></select><div class="legend" id="legend"></div>
<div class="note" id="range"></div><p class="note" id="notes"></p></div>
<div id="tip"></div><div id="fallback"></div>
<script>const D=__DATA__;const GEO=__GEO__;</script>
<script src="__GLOBE__"></script>
<script>
const LAYERS={gdp_pc_growth:"GDP per person growth",population_change:"Population change",
 conflict_hazard:"Conflict hazard",debt_gdp:"Debt / GDP",l0_gdp_pc_2100:"L0 region GDP per person, 2100"};
const sel=document.getElementById("layer");for(const k in LAYERS){const o=document.createElement("option");o.value=k;o.textContent=LAYERS[k];sel.appendChild(o);}
document.getElementById("sub").textContent=`L1 F1-F5, origin ${D.origin} -> ${D.target}, ${D.members} members. Exploratory.`;
document.getElementById("notes").innerHTML=D.notes.map(n=>"&middot; "+n).join("<br>");
const PAL=["#2c7bb6","#7fbbd9","#e7f0c4","#fdae61","#d7191c"];
document.getElementById("legend").innerHTML=PAL.map(c=>`<span style="background:${c}"></span>`).join("");
function val(id,k){const r=D.countries[id];if(!r)return null;if(k==="l0_gdp_pc_2100"){const q=D.l0_regions_gdp_pc_2100[r.region];return q?q[1]:null}return r[k]?r[k][1]:null}
function band(id,k){const r=D.countries[id];if(!r)return null;return k==="l0_gdp_pc_2100"?D.l0_regions_gdp_pc_2100[r.region]:r[k]}
let lo=0,hi=1;function scale(k){const v=GEO.features.map(f=>val(String(f.id),k)).filter(x=>x!==null).sort((a,b)=>a-b);
 lo=v[Math.floor(v.length*.05)];hi=v[Math.floor(v.length*.95)];document.getElementById("range").textContent=
 `${lo.toPrecision(3)} ... ${hi.toPrecision(3)} ${D.units[k]} (5th-95th percentile of countries, p50)`}
function color(v){if(v===null)return "rgba(120,130,140,.35)";const t=Math.max(0,Math.min(1,(v-lo)/(hi-lo||1)));return PAL[Math.min(4,Math.floor(t*5))]}
function tip(f,k){const id=String(f.id),r=D.countries[id],b=band(id,k);const t=document.getElementById("tip");
 if(!r||!b){t.style.display="block";t.innerHTML=`<b>${f.properties.name}</b><br><span class=note>not simulated</span>`;return}
 t.style.display="block";t.innerHTML=`<b>${f.properties.name}</b> (${r.iso3}, L0 region ${r.region})<br>${LAYERS[k]}:<br>`+
 `p50 <b>${b[1]}</b> &nbsp; p10-p90 [${b[0]}, ${b[2]}]<br><span class=note>${D.units[k]}</span>`}
function table(k){const rows=Object.entries(D.countries).map(([id,r])=>{const b=band(id,k)||[];return `<tr><td>${r.iso3}</td><td>${b[0]}</td><td>${b[1]}</td><td>${b[2]}</td></tr>`}).join("");
 const fb=document.getElementById("fallback");fb.style.display="block";fb.innerHTML=`<p>The globe library could not load (offline?). ${LAYERS[k]} [${D.units[k]}]</p><table><tr><th>country</th><th>p10</th><th>p50</th><th>p90</th></tr>${rows}</table>`}
let G=null;function draw(){const k=sel.value;scale(k);if(!G){table(k);return}
 G.polygonCapColor(f=>color(val(String(f.id),k))).onPolygonHover(f=>f&&tip(f,k))}
if(typeof Globe==="function"){G=Globe()(document.getElementById("globe")).backgroundColor("#0b1522")
 .showAtmosphere(true).atmosphereColor("#3a6ea5").polygonsData(GEO.features).polygonSideColor(()=>"rgba(0,0,0,.15)")
 .polygonStrokeColor(()=>"#0b1522").polygonAltitude(0.008);G.globeMaterial().color.set("#13202e");G.controls().autoRotate=true;G.controls().autoRotateSpeed=0.4}
sel.onchange=draw;draw();
</script></body></html>"""


def build(out_dir, R: int = 32, log=print) -> dict:
    from world4.core.l1_atlas.geo import load_countries
    p = install_geo()
    geo = load_countries(p)
    data = layers(R=R, log=log)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page = (PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__GEO__", json.dumps(geo, separators=(",", ":"))).replace("__GLOBE__", GLOBE_GL))
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return {"html": str(out / "index.html"), "countries": len(data["countries"]), "bytes": (out / "index.html").stat().st_size}
