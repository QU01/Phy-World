"""F9: el informe HTML autocontenido de World4.

Un solo fichero: sin CDN, sin fuentes web, sin JavaScript, sin librería de gráficos. Cada figura es SVG
construido aquí (el patrón de informe de la familia, de Phy-HX y Phy-Prop): una figura no puede fallar
porque falte una dependencia en la máquina que la abre.

Lo que lleva y por qué:

* **El semáforo de la campaña** (W1-W12) primero: lo que se puede creer de todo lo demás depende de él.
* **Trayectorias con bandas** del L0 a 2100 (población, PIB per cápita, energía, GMST, cereal per cápita,
  tierras de cultivo): sin política y, si se pide, con la política θ. Banda p10-p90 sobre los estados del
  mundo (posterior de F7 × incertidumbre profunda × eficacia de las palancas) más la discrepancia declarada
  del L0 y el error del dato.
* **Los lazos dominantes** (puntuación de *Loops that Matter*, F6): el porqué del modo.
* **Los límites cruzados**: las restricciones de F8 y los años en que la banda cruza umbrales de GMST,
  con WALLS / TIGHT / THE LAST BLOCKER.
* **Las vulnerabilidades** de la política (PRIM), si hay política.
* **La procedencia de los datos**: cada fuente con su licencia y su marca, cada manifiesto con sus SHA-256,
  el posterior con su corte y su revisión.

Todo número lleva unidad y banda o dice por qué no la tiene. Nada de esto es un pronóstico: más allá de
10 años son modos de comportamiento (``exploratory``).
"""
from __future__ import annotations

import html
import json
import math
from datetime import date

import numpy as np

W, H = 520, 250
PAD_L, PAD_R, PAD_T, PAD_B = 58, 14, 22, 34
COLORS = {"base": "#5b6b7d", "policy": "#1f7a5a", "warn": "#b5651d"}
VERDICT_COLORS = {"GREEN": "#1f7a5a", "AMBER": "#b88a00", "RED": "#b23b3b", "BLOCKED": "#5b6b7d", "INFO": "#3a6ea5"}
THRESHOLDS_GMST = (1.5, 2.0, 3.0)


def _e(x) -> str:
    return html.escape(str(x))


def _nice(lo, hi, n=5):
    span = hi - lo if hi > lo else abs(hi) + 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(v)
        v += step
    return ticks


def _fmt(v) -> str:
    a = abs(v)
    if a >= 1e4 or (a < 1e-2 and a > 0):
        return f"{v:.2g}"
    return f"{v:.3g}"


def fig_bands(title: str, unit: str, years, series: list, hlines=()) -> str:
    """``series``: [{label, color, p10, p50, p90}] ; ``hlines``: [(valor, etiqueta)]."""
    ys = np.asarray(years, float)
    vals = np.concatenate([np.r_[s["p10"], s["p90"]] for s in series] + [np.array([h for h, _ in hlines] or [np.nan])])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    x0, x1 = float(ys.min()), float(ys.max())
    X = lambda x: PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)  # noqa: E731
    Y = lambda y: H - PAD_B - (y - lo) / (hi - lo) * (H - PAD_T - PAD_B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{_e(title)}">',
           f'<text x="{PAD_L}" y="14" class="t">{_e(title)} <tspan class="u">[{_e(unit)}]</tspan></text>']
    for t in _nice(lo, hi):
        if lo <= t <= hi:
            out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="g"/>'
                       f'<text x="{PAD_L - 6}" y="{Y(t) + 4:.1f}" class="a" text-anchor="end">{_fmt(t)}</text>')
    for t in _nice(x0, x1, 6):
        if x0 <= t <= x1:
            out.append(f'<text x="{X(t):.1f}" y="{H - PAD_B + 16}" class="a" text-anchor="middle">{int(round(t))}</text>')
    for v, lab in hlines:
        if lo <= v <= hi:
            out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="h"/>'
                       f'<text x="{W - PAD_R - 2}" y="{Y(v) - 3:.1f}" class="a" text-anchor="end">{_e(lab)}</text>')
    for s in series:
        top = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(ys, s["p90"]))
        bot = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(ys[::-1], np.asarray(s["p10"])[::-1]))
        out.append(f'<polygon points="{top} {bot}" fill="{s["color"]}" fill-opacity="0.18" stroke="none"/>')
        mid = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(ys, s["p50"]))
        out.append(f'<polyline points="{mid}" fill="none" stroke="{s["color"]}" stroke-width="1.8"/>')
    for k, s in enumerate(series):
        out.append(f'<rect x="{PAD_L + 8 + 150 * k}" y="{PAD_T + 2}" width="10" height="10" fill="{s["color"]}" '
                   f'fill-opacity="0.6"/><text x="{PAD_L + 22 + 150 * k}" y="{PAD_T + 11}" class="a">{_e(s["label"])}</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ datos
VARS = (  # clave, título, unidad, función (serie del L0 -> nivel mundial (n, años)), variable de discrepancia
    ("pop", "World population", "billion persons", lambda s: np.sum(s["pop"], 1) / 1e3, "pop"),
    ("gdp_pc", "World GDP per person", "2017 USD PPP", lambda s: np.sum(s["gdp"], 1) / np.sum(s["pop"], 1), "gdp"),
    ("energy", "Primary energy", "thousand TWh", lambda s: np.sum(s["energy"], 1) / 1e3, "energy"),
    ("gmst", "Global mean surface temperature", "degC above 1850-1900", lambda s: s["gmst"], "gmst"),
    ("cereal_pc", "Cereal per person", "kg/person/year", lambda s: np.sum(s["cereal"], 1) * 1e3 / (np.sum(s["pop"], 1) * 1e6), "cereal"),
    ("cropland", "Cropland", "million ha", lambda s: np.sum(s["cropland"], 1) / 1e3, "cropland"),
)


def trajectories(theta=None, *, narrative: str = "ssp2", members: int = 200, seed: int = 0) -> dict:
    from world4.calib import l0r
    from world4.scenarios import levers, narratives, robust
    mult = narratives.sow(members, narrative, seed)
    rng = np.random.default_rng(seed + 3)
    runs = {"no policy": np.zeros(len(levers.NAMES))}
    if theta is not None and np.any(np.asarray(theta) != 0):
        runs["policy"] = np.asarray(theta, float)
    out = {"years": list(range(robust.START, robust.END + 1)), "narrative": narrative, "members": members, "vars": {}}
    h = np.arange(robust.END - robust.START + 1)
    for label, th in runs.items():
        r = robust.run(th, mult, record=l0r.RECORD)
        s = l0r.series(r, r["lnC"])
        for key, title, unit, fn, dv in VARS:
            g = fn(s)
            sd = np.sqrt(l0r.disc_sd(dv, h) ** 2 + l0r.OBS_SD[dv] ** 2)
            noise = sd[None, None] * rng.standard_normal((4,) + g.shape)
            e = (g[None] + noise) if key == "gmst" else g[None] * np.exp(noise)
            q = np.percentile(e.reshape(-1, g.shape[1]), [10, 50, 90], 0)
            out["vars"].setdefault(key, {"title": title, "unit": unit, "series": {}})["series"][label] = {
                "p10": q[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist()}
    g = out["vars"]["gmst"]["series"]
    out["gmst_crossing"] = {"unit": "calendar year: first year the p90 of GMST is above the threshold (degC keys); "
                                    "null = not before 2100",
                            "by_policy": {lab: {str(t): next((y for y, v in zip(out["years"], s["p90"]) if v > t), None)
                                                for t in THRESHOLDS_GMST} for lab, s in g.items()}}
    return out


def provenance() -> dict:
    from world4.data import atlas
    from pathlib import Path
    src = json.loads((Path(atlas.__file__).with_name("sources.json")).read_text(encoding="utf-8"))["sources"]
    man = {}
    for name in ("atlas", "wpp", "energy", "climate", "food"):
        p = Path(atlas.__file__).with_name(f"{name}_manifest.json")
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8"))
            files = m.get("files", {})
            man[name] = {"files": len(files), "with_sha256": sum(1 for v in files.values()
                                                               if (v.get("sha256") if isinstance(v, dict) else v)),
                         "source": m.get("source", m.get("repo", ""))}
    out = {"sources": src, "manifests": man}
    try:
        from world4.calib import posterior
        p = posterior.load()
        out["posterior"] = {"created": p["created"], "model_revision": p["model_revision"],
                            "atlas_commit": p["atlas_commit"], "l0r_cutoff": p["l0r"].get("cutoff"),
                            "l1_origin": p.get("l1", {}).get("origin"), "path": str(posterior.PATH.name)}
    except Exception as e:                       # noqa: BLE001
        out["posterior"] = {"missing": str(e)}
    return out


def collect(theta=None, *, narrative: str = "ssp2", members: int = 200, campaign: bool = True, log=print) -> dict:
    """Todo lo que el informe necesita, como datos (también va al JSON de al lado)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4 import MODEL_REVISION, api
    from world4.scenarios import levers, robust
    d = {"model_revision": MODEL_REVISION, "date": date.today().isoformat(), "narrative": narrative,
         "theta": None if theta is None else dict(zip(levers.NAMES, np.asarray(theta, float).tolist())),
         "exploratory": True}
    log("report: trajectories with bands (L0 ensemble)")
    d["trajectories"] = trajectories(theta, narrative=narrative, members=members)
    log("report: dominant loops")
    d["loops"] = api.loop_dominance(top=3, step=10)
    log("report: limits and policy")
    th = np.zeros(len(levers.NAMES)) if theta is None else np.asarray(theta, float)
    d["policy"] = robust.evaluate_policy(th, members=members, narrative=narrative, vulnerability=theta is not None)
    d["diagnosis"] = robust.diagnose(narrative=narrative)
    d["provenance"] = provenance()
    if campaign:
        log("report: acceptance campaign")
        from world4 import campaign as camp
        c = camp.build()
        scores = c.score_all()
        d["campaign"] = {"gate": c.gate(scores), "scores": [
            {"cid": s.criterion.cid, "name": s.criterion.what, "verdict": s.verdict, "measured": s.measured,
             "cause": s.cause, "threshold": s.criterion.threshold, "waits_on": list(s.waits_on)} for s in scores]}
    return d


# ------------------------------------------------------------------ HTML
CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1d2733;background:#fbfbf9;margin:0}
main{max-width:1120px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 8px;border-bottom:1px solid #d9dde2;padding-bottom:4px}
.sub{color:#5b6b7d}.note{color:#5b6b7d;font-size:12.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:10px 12px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:4px 6px;border-bottom:1px solid #eceff2;vertical-align:top}
th{background:#f3f4f6;font-weight:600}.v{font-weight:700;color:#fff;border-radius:4px;padding:1px 6px;font-size:11.5px}
.t{font-size:12.5px;font-weight:600;fill:#1d2733}.u{font-weight:400;fill:#5b6b7d}.a{font-size:10.5px;fill:#5b6b7d}
.g{stroke:#eceff2}.h{stroke:#b23b3b;stroke-dasharray:4 3}
.banner{background:#fff4e5;border:1px solid #f0c48a;border-radius:8px;padding:10px 12px;margin:14px 0}
.wall{color:#b23b3b;font-weight:700}.tight{color:#b88a00;font-weight:700}
code{font-size:12px;background:#f3f4f6;padding:0 3px;border-radius:3px}
@media (prefers-color-scheme: dark){body{background:#12171d;color:#dfe5ec}.card,th{background:#1a2129;border-color:#2a333d}
td{border-color:#2a333d}.t{fill:#dfe5ec}.g{stroke:#2a333d}.banner{background:#2b2214;border-color:#6b5025}code{background:#1a2129}}
"""


def build(d: dict) -> str:
    T = d["trajectories"]
    yrs = T["years"]
    lab_color = {"no policy": COLORS["base"], "policy": COLORS["policy"]}
    figs = []
    for key, v in T["vars"].items():
        series = [{"label": lab, "color": lab_color[lab], **s} for lab, s in v["series"].items()]
        hl = [(t, f"{t:g} degC") for t in THRESHOLDS_GMST] if key == "gmst" else ()
        figs.append(f'<div class="card">{fig_bands(v["title"], v["unit"], yrs, series, hl)}</div>')
    pol = d["policy"]
    cons_rows = "".join(
        f"<tr><td>{_e(k)}</td><td>{pol['constraints'][k]:+.3f}</td><td>{pol['baseline_constraints'][k]:+.3f}</td>"
        f"<td>{'<span class=wall>crossed</span>' if pol['constraints'][k] > 0 else 'inside'}</td></tr>"
        for k in pol["constraints"])
    obj_rows = "".join(f"<tr><td>{_e(k)}</td><td>{v:+.3f}</td><td>{pol['baseline_objectives'][k]:+.3f}</td></tr>"
                       for k, v in pol["objectives"].items())
    cross = "".join(f"<tr><td>{_e(lab)}</td>" + "".join(f"<td>{c or 'not before 2100'}</td>" for c in row.values()) + "</tr>"
                    for lab, row in T["gmst_crossing"]["by_policy"].items())
    dg = d["diagnosis"]
    diag = "".join(f'<li><span class="wall">WALL</span> {_e(w["name"])}: violated by {w["violated_share"]:.0%} of '
                   f'{dg["population"]} policies, best g {w["g_best"]:+.3f} (needs &le; 0). {_e(w["advice"])}</li>'
                   for w in dg["walls"])
    diag += "".join(f'<li><span class="tight">TIGHT</span> {_e(w["name"])}: violated by {w["violated_share"]:.0%}, '
                    f'best g {w["g_best"]:+.3f}</li>' for w in dg["tight"])
    diag += "".join(f'<li><b>THE LAST BLOCKER</b> {_e(b["name"])}: {b["n"]} policies blocked by it alone (closest '
                    f'g {b["g_closest"]:+.3f})</li>' for b in dg["last_blocker"][:2])
    loops = "".join(f"<tr><td>{_e(r.get('year', ''))}</td><td>{_e(r.get('loop', r.get('name', '')))}</td>"
                    f"<td>{r.get('score', float('nan')):+.2f}</td></tr>" for r in _loop_rows(d["loops"]))
    vul = ""
    for label, v in (pol.get("vulnerability") or {}).items():
        if isinstance(v, dict) and v.get("chosen"):
            ch = v["chosen"]
            dims = "".join(f"<li><code>{_e(b['name'])}</code> in [{b['lo']:.3g}, {b['hi']:.3g}] of "
                           f"[{b['range'][0]:.3g}, {b['range'][1]:.3g}]</li>" for b in ch["restricted"][:5])
            vul += (f'<div class="card"><b>{_e(label)}</b>: fails in {v["base_rate"]:.0%} of '
                    f'{pol["vulnerability"]["members"]} worlds. PRIM box: {ch["coverage"]:.0%} of failures at density '
                    f'{ch["density"]:.0%}<ul>{dims}</ul></div>')
    camp = ""
    if "campaign" in d:
        camp = "".join(
            f'<tr><td>{_e(s["cid"])}</td><td><span class="v" style="background:{VERDICT_COLORS.get(s["verdict"], "#555")}">'
            f'{_e(s["verdict"])}</span></td><td>{_e(s["name"])}</td><td>{_e((s["measured"] or "")[:420])}</td>'
            f'<td>{_e(s["cause"] or ", ".join(s["waits_on"]))}</td></tr>' for s in d["campaign"]["scores"])
        g = d["campaign"]["gate"]
        camp_head = (f"<p>Release gate: <b>{'PASSES' if g.get('passed') else 'BLOCKED'}</b> "
                     f"({len(g.get('blocking', []))} criteria red or blocked).</p>")
    else:
        camp_head = "<p class=note>campaign not measured in this report (--no-campaign)</p>"
    pv = d["provenance"]
    srcs = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v.get('name', ''))}</td><td>{_e(v.get('license', ''))}</td>"
                   f"<td>{_e(v.get('mark', ''))}</td><td>{'yes' if v.get('redistributable') else 'no'}</td>"
                   f"<td>{_e(v.get('vintage', ''))}</td></tr>" for k, v in pv["sources"].items())
    mans = "".join(f"<tr><td>{_e(k)}</td><td>{v['files']}</td><td>{v['with_sha256']}</td><td>{_e(v['source'])}</td></tr>"
                   for k, v in pv["manifests"].items())
    post = pv.get("posterior", {})
    theta = d.get("theta")
    pol_txt = (", ".join(f"{k} = {v:g}" for k, v in theta.items() if v) if theta else "no policy (all levers at 0)")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Phy-World report</title>
<style>{CSS}</style></head><body><main>
<h1>QUASAR Phy-World (World4) report</h1>
<div class="sub">model revision {d['model_revision']} &middot; {d['date']} &middot; narrative {_e(d['narrative'])} &middot;
policy: {_e(pol_txt)}</div>
<div class="banner"><b>Exploratory.</b> Everything beyond 10 years is a behaviour mode under declared uncertainty,
never a forecast. Bands are p10-p90 over {T['members']} states of the world (F7 posterior x deep uncertainty x
policy effectiveness) plus the L0's declared discrepancy and data error. Marks: [V] verified, [S] secondary source,
[M] modelled assumption, [?] not verified.</div>
<h2>1. Acceptance campaign</h2>{camp_head}
<table><tr><th>id</th><th>verdict</th><th>criterion</th><th>measured</th><th>cause / waits on</th></tr>{camp}</table>
<h2>2. Trajectories with bands (L0, {yrs[0]}-{yrs[-1]})</h2><div class="grid">{''.join(figs)}</div>
<h2>3. Dominant loops (Loops that Matter score, world view of the L0, no policy)</h2>
<table><tr><th>year</th><th>loop</th><th>relative score</th></tr>{loops}</table>
<h2>4. Limits</h2>
<p class="note">Robust objectives (minimised; percentiles over the states of the world) and constraints g &le; 0
(declared [M]: regional cereal per person &ge; 90 % of 2019 at p10; cropland 2100 &le; 1.10 x 2019 at p90; world
debt 2100 &le; 1.5 x GDP at p90).</p>
<div class="grid"><div class="card"><table><tr><th>objective</th><th>policy</th><th>no policy</th></tr>{obj_rows}</table></div>
<div class="card"><table><tr><th>constraint</th><th>g policy</th><th>g no policy</th><th></th></tr>{cons_rows}</table></div>
<div class="card"><table><tr><th>p90 GMST first above</th>{''.join(f'<th>{t:g} degC</th>' for t in THRESHOLDS_GMST)}</tr>{cross}</table></div></div>
<p>Why no policy is feasible, over a Latin hypercube of {dg['population']} policies ({dg['feasible']} feasible):</p><ul>{diag or '<li>no wall, nothing tight</li>'}</ul>
<h2>5. Vulnerabilities of the policy (PRIM)</h2>{('<div class="grid">' + vul + '</div>') if vul else '<p class=note>no policy given</p>'}
<h2>6. Data provenance</h2>
<p>Posterior: {_e(post)}</p>
<table><tr><th>manifest</th><th>files</th><th>with SHA-256</th><th>source</th></tr>{mans}</table><br>
<table><tr><th>source</th><th>name</th><th>licence</th><th>mark</th><th>redistributable</th><th>vintage</th></tr>{srcs}</table>
<p class="note">Third-party data never travel inside the package; they are installed with <code>--install-*</code> and
checked by SHA-256 (W12).</p>
</main></body></html>"""


def _loop_rows(loops) -> list:
    """``api.loop_dominance`` -> filas {year, loop, score} (tolera las dos formas que devuelve)."""
    rows = []
    for item in loops:
        if isinstance(item, dict) and "loops" in item:
            for lp in item["loops"]:
                name, score = (lp[0], lp[1]) if isinstance(lp, (list, tuple)) else (lp.get("name", ""), lp.get("score"))
                rows.append({"year": item.get("year"), "loop": name, "score": float(score)})
        elif isinstance(item, dict):
            rows.append({"year": item.get("year"), "loop": item.get("loop", item.get("name", "")),
                         "score": float(item.get("score", item.get("relative", float("nan"))))})
    return rows


def write(path, theta=None, *, narrative: str = "ssp2", members: int = 200, campaign: bool = True, log=print) -> dict:
    from pathlib import Path
    d = collect(theta, narrative=narrative, members=members, campaign=campaign, log=log)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(d), encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                                      encoding="utf-8")
    return {"html": str(p), "json": str(p.with_suffix(".json")), "bytes": p.stat().st_size}
