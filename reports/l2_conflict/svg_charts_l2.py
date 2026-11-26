"""Generador de gráficos SVG vectoriales para el informe editorial de la capa L2 (Eventos y Conflicto).
Paleta y lenguaje visual coherente con los informes Phy-World L0 y L1.
"""
from __future__ import annotations

import math
from html import escape

INK = "#0E162B"
INK2 = "#374151"
MUTED = "#7B8798"
GRID = "#E7ECF2"
ORANGE = "#D95D24"
ORANGE_BRIGHT = "#E8742D"
ORANGE_BAND = "rgba(217, 93, 36, 0.20)"
BLUE = "#2F66D0"
BLUE_BAND = "rgba(47, 102, 208, 0.14)"
TEAL = "#1F7A6E"
CRIMSON = "#B83227"


def fmt_es(v: float | None, d: int = 1, suffix: str = "") -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) >= 1000 and d == 0:
        s = f"{v:,.0f}".replace(",", ".")
    elif abs(v) >= 1000:
        s = f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        s = f"{v:.{d}f}".replace(".", ",")
    s = s.replace("-", "−")
    return f"{s}{suffix}"


def fmt_k(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) >= 1e6:
        return fmt_es(v / 1e6, 2, " M")
    if abs(v) >= 1e4:
        return fmt_es(v / 1e3, 0, "k")
    if abs(v) >= 1e3:
        return fmt_es(v / 1e3, 1, "k")
    return fmt_es(v, 0)


def _nice_ticks(vmin: float, vmax: float, n: int = 4) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm <= 1.2 else 2 if norm <= 2.2 else 2.5 if norm <= 3.0 else 5 if norm <= 6.0 else 10) * mag
    start = math.floor(vmin / step) * step
    ticks, val = [], start
    while val <= vmax + step * 1e-6:
        if val >= vmin - step * 1e-6:
            ticks.append(round(val, 10))
        val += step
    return ticks


def _poly(pts: list[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) if pts else ""


def _deconflict(items: list[tuple[float, str, str]], ymin: float, ymax: float, gap: float = 10.0):
    if not items:
        return []
    ordered = sorted(items, key=lambda t: t[0])
    ys = [max(ymin + 4, min(ymax - 2, t[0])) for t in ordered]
    for _ in range(12):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                mid = 0.5 * (ys[i] + ys[i - 1])
                ys[i - 1] = max(ymin + 4, mid - gap / 2)
                ys[i] = min(ymax - 2, ys[i - 1] + gap)
    return [(y, t[1], t[2]) for y, t in zip(ys, ordered)]


def ts_l2_chart(
    title: str,
    subtitle: str,
    *,
    obs: tuple[list[float], list[float | None]] | None = None,
    prov: tuple[list[float], list[float | None]] | None = None,
    hurdle: tuple[list[float], list[float | None]] | None = None,
    last12: tuple[list[float], list[float | None]] | None = None,
    agents_band: tuple[list[float], list[float | None], list[float | None], list[float | None]] | None = None,
    xmin: float = 2018.0,
    xmax: float = 2030.0,
    ymin: float | None = 0.0,
    ymax: float | None = None,
    y_fmt=None,
    end_fmt=None,
    xticks: list[tuple[float, str]] | None = None,
    vlines: list[tuple[float, str]] | None = None,
    shade_from: float | None = 2026.75,
    width: float = 252,
    height: float = 128,
    caption: str = "",
) -> str:
    """Gráfico de serie temporal mensual/anual para L2 (agentes vs hurdle vs last12 vs real)."""
    ml, mr, mt, mb = 36, 42, 33, 21
    pw, ph = width - ml - mr, height - mt - mb

    all_y = []
    for ser in (obs, prov, hurdle, last12):
        if ser:
            all_y += [v for x, v in zip(ser[0], ser[1]) if xmin <= x <= xmax and v is not None and math.isfinite(v)]
    if agents_band:
        xs, p10, p50, p90 = agents_band
        # Cap extreme upper-tail spikes visually only when ymax is explicit; otherwise use p90
        all_y += [v for x, v in zip(xs, p90) if xmin <= x <= xmax and v is not None and math.isfinite(v)]
        all_y += [v for x, v in zip(xs, p10) if xmin <= x <= xmax and v is not None and math.isfinite(v)]
    if not all_y:
        all_y = [0.0, 1.0]
    lo, hi = min(all_y), max(all_y)
    pad = (hi - lo) * 0.12 if hi > lo else 1.0
    ylo = ymin if ymin is not None else max(0.0, lo - pad * 0.4)
    yhi = ymax if ymax is not None else hi + pad
    ticks = _nice_ticks(ylo, yhi, n=4)
    if ymin is None:
        ylo = min(ylo, ticks[0])
    if ymax is None:
        yhi = max(yhi, ticks[-1])

    xfn = lambda x: ml + (x - xmin) / max(xmax - xmin, 1e-9) * pw
    yfn = lambda v: mt + (1.0 - (v - ylo) / max(yhi - ylo, 1e-12)) * ph

    if y_fmt is None:
        y_fmt = fmt_k
    if end_fmt is None:
        end_fmt = y_fmt

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">']
    parts.append(f'<text x="0" y="12" font-family="IBM Plex Sans" font-size="9.2" font-weight="600" fill="{INK}">{escape(title)}</text>')
    parts.append(f'<text x="0" y="22.5" font-family="IBM Plex Mono" font-size="6.1" letter-spacing="0.3" fill="{MUTED}">{escape(subtitle.upper())}</text>')

    if shade_from is not None and xmin < shade_from < xmax:
        xs0 = xfn(shade_from)
        parts.append(f'<rect x="{xs0:.2f}" y="{mt}" width="{ml + pw - xs0:.2f}" height="{ph}" fill="#F6F8FB"/>')

    for tv in ticks:
        if ylo - 1e-9 <= tv <= yhi + 1e-9:
            yy = yfn(tv)
            parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{GRID}" stroke-width="0.65"/>')
            parts.append(f'<text x="{ml - 5}" y="{yy + 2.4:.2f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.5" fill="{MUTED}">{escape(y_fmt(tv))}</text>')

    for xv, lbl in (vlines or []):
        if xmin < xv < xmax:
            xx = xfn(xv)
            parts.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt + ph}" stroke="#94A3B8" stroke-width="0.75" stroke-dasharray="2.2,2.2"/>')
            parts.append(f'<text x="{xx:.2f}" y="{mt - 3}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{escape(lbl)}</text>')

    for xv, lbl in (xticks or []):
        if xmin - 1e-6 <= xv <= xmax + 1e-6:
            parts.append(f'<text x="{xfn(xv):.2f}" y="{mt + ph + 12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.5" fill="{MUTED}">{escape(lbl)}</text>')

    cid = f"l2c_{abs(hash((title, subtitle, width, height))) % 10**8}"
    parts.append(f'<defs><clipPath id="{cid}"><rect x="{ml - 1}" y="{mt - 2}" width="{pw + 2}" height="{ph + 4}"/></clipPath></defs>')
    parts.append(f'<g clip-path="url(#{cid})">')

    end_labels = []
    if agents_band:
        xs, p10, p50, p90 = agents_band
        hi_pts = [(xfn(x), yfn(min(v, yhi * 1.15))) for x, v in zip(xs, p90) if xmin <= x <= xmax and v is not None]
        lo_pts = [(xfn(x), yfn(max(v, ylo))) for x, v in zip(xs, p10) if xmin <= x <= xmax and v is not None]
        if len(hi_pts) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in (hi_pts + lo_pts[::-1]))
            parts.append(f'<polygon points="{poly}" fill="{ORANGE_BAND}"/>')
        m_pts = [(xfn(x), yfn(v)) for x, v in zip(xs, p50) if xmin <= x <= xmax and v is not None]
        if m_pts:
            parts.append(f'<path d="{_poly(m_pts)}" fill="none" stroke="{ORANGE}" stroke-width="1.8"/>')
            last_v = [v for x, v in zip(xs, p50) if xmin <= x <= xmax and v is not None][-1]
            end_labels.append((yfn(min(last_v, yhi)), end_fmt(last_v), ORANGE))

    if last12:
        pts = [(xfn(x), yfn(v)) for x, v in zip(last12[0], last12[1]) if xmin <= x <= xmax and v is not None]
        if pts:
            parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="#64748B" stroke-width="1.15" stroke-dasharray="1.8,2.2"/>')

    if hurdle:
        pts = [(xfn(x), yfn(v)) for x, v in zip(hurdle[0], hurdle[1]) if xmin <= x <= xmax and v is not None]
        if pts:
            parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="{BLUE}" stroke-width="1.45" stroke-dasharray="3.2,2.0"/>')
            last_h = [v for x, v in zip(hurdle[0], hurdle[1]) if xmin <= x <= xmax and v is not None][-1]
            end_labels.append((yfn(min(last_h, yhi)), end_fmt(last_h), BLUE))

    if prov:
        pts = [(xfn(x), yfn(v)) for x, v in zip(prov[0], prov[1]) if xmin <= x <= xmax and v is not None]
        if pts:
            parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="#475569" stroke-width="1.35" stroke-dasharray="2.0,1.6"/>')

    if obs:
        pts = [(xfn(x), yfn(v)) for x, v in zip(obs[0], obs[1]) if xmin <= x <= xmax and v is not None]
        if pts:
            parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="{INK}" stroke-width="1.7"/>')
            parts.append(f'<circle cx="{pts[-1][0]:.2f}" cy="{pts[-1][1]:.2f}" r="2.2" fill="{INK}" stroke="white" stroke-width="0.7"/>')

    parts.append("</g>")

    for yy, txt, col in _deconflict(end_labels, mt, mt + ph):
        parts.append(f'<text x="{ml + pw + 5:.2f}" y="{yy + 2.4:.2f}" font-family="IBM Plex Mono" font-size="6.7" font-weight="500" fill="{col}">{escape(txt)}</text>')

    parts.append("</svg>")
    svg = "\n".join(parts)
    if caption:
        return f'<div class="chart-box">{svg}<div class="chart-cap">{caption}</div></div>'
    return f'<div class="chart-box">{svg}</div>'


def horizon_crps_chart(
    title: str,
    subtitle: str,
    steps: list[int],
    ag: list[float],
    hu: list[float],
    l12: list[float],
    *,
    width: float = 252,
    height: float = 142,
) -> str:
    """Curva de error CRPS de ln(1+muertes) según el horizonte en meses (1..36)."""
    ml, mr, mt, mb = 36, 46, 32, 22
    pw, ph = width - ml - mr, height - mt - mb
    ylo = 0.0
    yhi = max(max(ag), max(hu), max(l12)) * 1.15
    ticks = _nice_ticks(ylo, yhi, n=4)
    yhi = max(yhi, ticks[-1])

    xfn = lambda s: ml + (s - steps[0]) / max(steps[-1] - steps[0], 1) * pw
    yfn = lambda v: mt + (1.0 - (v - ylo) / max(yhi - ylo, 1e-12)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">']
    parts.append(f'<text x="0" y="12" font-family="IBM Plex Sans" font-size="9.0" font-weight="600" fill="{INK}">{escape(title)}</text>')
    parts.append(f'<text x="0" y="22" font-family="IBM Plex Mono" font-size="6.0" fill="{MUTED}">{escape(subtitle.upper())}</text>')

    for tv in ticks:
        yy = yfn(tv)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{GRID}" stroke-width="0.65"/>')
        d = 3 if yhi < 0.05 else 2
        parts.append(f'<text x="{ml - 5}" y="{yy + 2.4:.2f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.4" fill="{MUTED}">{fmt_es(tv, d)}</text>')

    for s in steps:
        parts.append(f'<text x="{xfn(s):.2f}" y="{mt + ph + 12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.4" fill="{MUTED}">{s}m</text>')

    series = [
        (l12, "#64748B", "1.4", "2.2,2.0", "last12"),
        (hu, BLUE, "1.6", "", "hurdle"),
        (ag, ORANGE, "2.0", "", "Agentes"),
    ]
    end_labels = []
    for vals, col, sw, dash, lbl in series:
        pts = [(xfn(s), yfn(v)) for s, v in zip(steps, vals)]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="{col}" stroke-width="{sw}"{da}/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="{col}" stroke="white" stroke-width="0.6"/>')
        d = 3 if yhi < 0.05 else 2
        end_labels.append((pts[-1][1], f"{lbl} {fmt_es(vals[-1], d)}", col))

    for yy, txt, col in _deconflict(end_labels, mt, mt + ph, gap=10.0):
        parts.append(f'<text x="{ml + pw + 5:.2f}" y="{yy + 2.3:.2f}" font-family="IBM Plex Mono" font-size="6.1" font-weight="500" fill="{col}">{escape(txt)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def paired_hbar(
    rows: list[tuple[str, float, float, str, str]],
    *,
    label_a: str = "Mejor referencia (last12 / hurdle)",
    label_b: str = "Simulador de agentes L2",
    width: float = 520,
    row_h: float = 22,
) -> str:
    ml, mr, mt, mb = 148, 62, 24, 6
    pw = width - ml - mr
    height = mt + len(rows) * row_h + mb
    vmax = max(max(r[1], r[2]) for r in rows) * 1.15

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<rect x="{ml}" y="5" width="12" height="6" rx="1" fill="{BLUE}"/>')
    parts.append(f'<text x="{ml + 17}" y="10.5" font-family="IBM Plex Sans" font-size="7.2" fill="{INK2}">{escape(label_a)}</text>')
    parts.append(f'<rect x="{ml + 195}" y="5" width="12" height="6" rx="1" fill="{ORANGE}"/>')
    parts.append(f'<text x="{ml + 212}" y="10.5" font-family="IBM Plex Sans" font-size="7.2" fill="{INK2}">{escape(label_b)}</text>')

    for i, (lbl, va, vb, ta, tb) in enumerate(rows):
        y0 = mt + i * row_h
        parts.append(f'<text x="{ml - 8}" y="{y0 + 11}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.6" fill="{INK}">{escape(lbl)}</text>')
        wa, wb = max(1.5, va / vmax * pw), max(1.5, vb / vmax * pw)
        parts.append(f'<rect x="{ml}" y="{y0 + 1.5}" width="{wa:.2f}" height="6.5" rx="1" fill="{BLUE}"/>')
        parts.append(f'<text x="{ml + wa + 4:.2f}" y="{y0 + 7.2}" font-family="IBM Plex Mono" font-size="6.6" fill="{MUTED}">{escape(ta)}</text>')
        parts.append(f'<rect x="{ml}" y="{y0 + 9.8}" width="{wb:.2f}" height="6.5" rx="1" fill="{ORANGE}"/>')
        parts.append(f'<text x="{ml + wb + 4:.2f}" y="{y0 + 15.5}" font-family="IBM Plex Mono" font-size="6.6" font-weight="500" fill="{INK}">{escape(tb)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def world_hotspot_map_svg(
    land_pts: list[list[float]],
    hotspots: list[dict],
    *,
    width: float = 520,
    height: float = 225,
) -> str:
    """Mapa vectorial mundial de focos 0,5° PRIO-GRID (próximos 12 meses, Agentes L2)."""
    ml, mr, mt, mb = 8, 8, 22, 16
    pw, ph = width - ml - mr, height - mt - mb
    # Crop latitudes -56 to +74 for crisp modern cartographic framing
    lon_min, lon_max = -132.0, 162.0
    lat_min, lat_max = -52.0, 72.0
    xfn = lambda lon: ml + (lon - lon_min) / (lon_max - lon_min) * pw
    yfn = lambda lat: mt + (1.0 - (lat - lat_min) / (lat_max - lat_min)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" fill="#0E162B"/>')
    parts.append(f'<text x="12" y="14" font-family="IBM Plex Mono" font-size="6.6" letter-spacing="0.7" fill="#95A5C6">FOCOS SUBNACIONALES DE VIOLENCIA (0,5° PRIO-GRID) · PRÓXIMOS 12 MESES (2026-10 A 2027-09)</text>')

    # Subtle graticule
    for lat in (-30, 0, 30, 60):
        yy = yfn(lat)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" stroke="#1E2B4A" stroke-width="0.5"/>')
    for lon in (-90, -30, 30, 90, 150):
        xx = xfn(lon)
        parts.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#1E2B4A" stroke-width="0.5"/>')

    # Land dots from PRIO-GRID itself!
    dots = []
    for lon, lat in land_pts:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            dots.append(f'<circle cx="{xfn(lon):.1f}" cy="{yfn(lat):.1f}" r="0.82" fill="#2A3A5E"/>')
    parts.append('<g>' + "".join(dots) + '</g>')

    # Hotspots (sorted smallest first so biggest glow on top)
    for h in sorted(hotspots, key=lambda z: z["mean_12m"]):
        lon, lat, m = h["lon"], h["lat"], h["mean_12m"]
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        x, y = xfn(lon), yfn(lat)
        r = max(1.7, min(9.0, 0.95 * math.log1p(m) - 0.5))
        col = "#FF4D3D" if m >= 500 else ("#E8742D" if m >= 100 else "#F5A623")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 1.65:.1f}" fill="{col}" opacity="0.22"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}" opacity="0.88" stroke="#0E162B" stroke-width="0.4"/>')

    # Curated crisp regional flashpoint callouts (anchored cleanly inside frame)
    callouts = [
        (-102, 23, -84, 34, "México (cárteles)"),
        (37, 48.5, 42, 60, "Ucrania"),
        (34.4, 31.4, 52, 37, "Gaza / Levante"),
        (32.5, 15.6, 48, 17, "Sudán"),
        (-1.5, 13.5, -24, 15, "Sahel"),
        (9.5, 11.5, -6, 3, "Nigeria"),
        (29.0, -1.5, 18, -16, "RDC"),
        (45.3, 2.0, 58, -5, "Somalia"),
        (96.0, 21.0, 108, 28, "Myanmar"),
    ]
    for lon0, lat0, lon1, lat1, txt in callouts:
        x0, y0, x1, y1 = xfn(lon0), yfn(lat0), xfn(lon1), yfn(lat1)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#8DA0C4" stroke-width="0.6" opacity="0.75"/>')
        anc = "end" if lon1 < lon0 else "start"
        dx = -2.5 if lon1 < lon0 else 2.5
        parts.append(f'<text x="{x1 + dx:.1f}" y="{y1 + 2.2:.1f}" text-anchor="{anc}" font-family="IBM Plex Sans" font-size="6.4" font-weight="500" fill="#E2E8F0">{escape(txt)}</text>')

    # Legend strip at bottom
    parts.append(f'<circle cx="16" cy="{height - 7}" r="2.2" fill="#F5A623"/><text x="22" y="{height - 4.8}" font-family="IBM Plex Mono" font-size="5.9" fill="#95A5C6">&lt;100 muertes/12m</text>')
    parts.append(f'<circle cx="115" cy="{height - 7}" r="3.4" fill="#E8742D"/><text x="122" y="{height - 4.8}" font-family="IBM Plex Mono" font-size="5.9" fill="#95A5C6">100–500</text>')
    parts.append(f'<circle cx="176" cy="{height - 7}" r="4.8" fill="#FF4D3D"/><text x="184" y="{height - 4.8}" font-family="IBM Plex Mono" font-size="5.9" fill="#95A5C6">&gt;500 muertes/12m (media celda 0,5°)</text>')
    parts.append(f'<text x="{width - 10}" y="{height - 4.8}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.8" fill="#6B7C9E">BASE: 64.859 CELDAS PRIO-GRID 2.0</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def coef_heatmap_svg(
    regions_data: dict,
    reg_order: list[str],
    reg_labels: list[str],
    *,
    width: float = 520,
    height: float = 195,
) -> str:
    """Matriz de calor de los 11 coeficientes de activación de díadas (b_act) en las 5 regiones."""
    feats = [
        ("own1", "Eventos mes previo (own1)"),
        ("own3", "Media 3 meses (own3)"),
        ("own12", "Media 12 meses (own12)"),
        ("since", "Meses callada (since)"),
        ("age", "Edad de la díada (age)"),
        ("side_a", "Otros frentes actor A"),
        ("side_b", "Otros frentes actor B"),
        ("ctx", "Violencia resto país (ctx)"),
        ("non_state", "Tipo: no estatal"),
        ("one_sided", "Tipo: unilateral civiles"),
        ("interstate", "Tipo: entre gobiernos"),
    ]
    ml, mr, mt, mb = 152, 16, 26, 8
    pw, ph = width - ml - mr, height - mt - mb
    cw = pw / len(reg_order)
    rh = ph / len(feats)

    def cell_color(v: float) -> tuple[str, str]:
        # Diverging: negative -> cobalt blue, positive -> burnt orange
        a = min(abs(v) / 1.4, 1.0)
        if v >= 0:
            r = int(248 - a * (248 - 217))
            g = int(242 - a * (242 - 93))
            b = int(238 - a * (238 - 36))
        else:
            r = int(240 - a * (240 - 47))
            g = int(244 - a * (244 - 102))
            b = int(252 - a * (252 - 208))
        txt = "#FFFFFF" if a > 0.52 else INK
        return f"rgb({r},{g},{b})", txt

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    for j, rlbl in enumerate(reg_labels):
        xc = ml + (j + 0.5) * cw
        parts.append(f'<text x="{xc:.1f}" y="16" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.8" font-weight="500" fill="{INK}">{escape(rlbl.upper())}</text>')

    for i, (fk, flbl) in enumerate(feats):
        y0 = mt + i * rh
        parts.append(f'<text x="{ml - 7}" y="{y0 + rh * 0.65:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.3" fill="{INK}">{escape(flbl)}</text>')
        for j, rk in enumerate(reg_order):
            v = regions_data[rk]["params"]["b_act"][fk]
            bg, fg = cell_color(v)
            x0 = ml + j * cw + 1.2
            parts.append(f'<rect x="{x0:.1f}" y="{y0 + 1:.1f}" width="{cw - 2.4:.1f}" height="{rh - 2:.1f}" rx="1.5" fill="{bg}"/>')
            s = ("+" if v > 0 else ("−" if v < 0 else "")) + fmt_es(abs(v), 2)
            parts.append(f'<text x="{x0 + (cw - 2.4)/2:.1f}" y="{y0 + rh * 0.66:.1f}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.7" font-weight="500" fill="{fg}">{s}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def country_l2_mini_svg(
    iso3: str,
    name: str,
    obs_23_26: list[float],
    ag_36: dict,
    hu_36: dict,
    *,
    tag_top: str,
    tag_sub: str,
    width: float = 122,
    height: float = 76,
) -> str:
    """Mini-gráfico mensual de muertes (2023-01 a 2029-09) para 12 focos globales."""
    ml, mr, mt, mb = 6, 6, 22, 13
    pw, ph = width - ml - mr, height - mt - mb
    n_obs = len(obs_23_26)     # 45 meses (2023-01 .. 2026-09)
    n_tot = n_obs + 36         # 81 meses (hasta 2029-09)
    xfn = lambda m: ml + m / (n_tot - 1) * pw

    # Plot on log1p scale so both bursts and chronic wars read crisply!
    lo = 0.0
    hi_vals = [math.log1p(v) for v in obs_23_26] + [math.log1p(v) for v in ag_36["p90"]] + [math.log1p(v) for v in hu_36["p50"]]
    hi = max(max(hi_vals) * 1.12, 2.5)
    yfn = lambda v: mt + (1.0 - (math.log1p(max(v, 0.0)) - lo) / hi) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    fs_nm = "6.8" if len(name) > 14 else "7.5"
    parts.append(f'<text x="0" y="9" font-family="IBM Plex Sans" font-size="{fs_nm}" font-weight="600" fill="{INK}">{escape(name)}</text>')
    parts.append(f'<text x="{width}" y="9" text-anchor="end" font-family="IBM Plex Mono" font-size="6.0" fill="{ORANGE}">{escape(tag_top)}</text>')
    parts.append(f'<text x="0" y="17.3" font-family="IBM Plex Mono" font-size="5.6" fill="{MUTED}">{iso3} · {escape(tag_sub)}</text>')

    x_cut = xfn(n_obs - 1)
    parts.append(f'<rect x="{x_cut:.2f}" y="{mt}" width="{ml + pw - x_cut:.2f}" height="{ph}" fill="#F6F8FB"/>')
    parts.append(f'<line x1="{ml}" y1="{mt + ph:.2f}" x2="{ml + pw}" y2="{mt + ph:.2f}" stroke="{GRID}" stroke-width="0.6"/>')
    parts.append(f'<line x1="{x_cut:.2f}" y1="{mt}" x2="{x_cut:.2f}" y2="{mt + ph}" stroke="#B8C4D4" stroke-width="0.6" stroke-dasharray="1.6,1.6"/>')

    # Agents p10-p90 ribbon
    hi_pts = [(xfn(n_obs + h), yfn(v)) for h, v in enumerate(ag_36["p90"])]
    lo_pts = [(xfn(n_obs + h), yfn(v)) for h, v in enumerate(ag_36["p10"])]
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in (hi_pts + lo_pts[::-1]))
    parts.append(f'<polygon points="{poly}" fill="{ORANGE_BAND}"/>')

    # Hurdle median (blue dashed)
    hu_pts = [(xfn(n_obs + h), yfn(v)) for h, v in enumerate(hu_36["p50"])]
    parts.append(f'<path d="{_poly(hu_pts)}" fill="none" stroke="{BLUE}" stroke-width="1.0" stroke-dasharray="2.2,1.6"/>')

    # Agents median (orange solid)
    ag_pts = [(xfn(n_obs + h), yfn(v)) for h, v in enumerate(ag_36["p50"])]
    parts.append(f'<path d="{_poly(ag_pts)}" fill="none" stroke="{ORANGE}" stroke-width="1.3"/>')

    # Observed 2023-2025 (solid ink) + 2026 candidate (gray)
    obs_final = [(xfn(m), yfn(obs_23_26[m])) for m in range(min(36, n_obs))]
    parts.append(f'<path d="{_poly(obs_final)}" fill="none" stroke="{INK}" stroke-width="1.25"/>')
    if n_obs > 36:
        obs_cand = [(xfn(m), yfn(obs_23_26[m])) for m in range(35, n_obs)]
        parts.append(f'<path d="{_poly(obs_cand)}" fill="none" stroke="#64748B" stroke-width="1.0" stroke-dasharray="1.5,1.2"/>')
        parts.append(f'<circle cx="{obs_final[-1][0]:.2f}" cy="{obs_final[-1][1]:.2f}" r="1.6" fill="{INK}" stroke="white" stroke-width="0.5"/>')

    parts.append(f'<text x="{ml}" y="{height - 2}" font-family="IBM Plex Mono" font-size="5.5" fill="{MUTED}">2023</text>')
    parts.append(f'<text x="{x_cut:.2f}" y="{height - 2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.5" fill="{MUTED}">26</text>')
    parts.append(f'<text x="{ml + pw}" y="{height - 2}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.5" fill="{MUTED}">2029</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def cover_l2_hero_svg(D: dict, width: float = 595, height: float = 315) -> str:
    """Portada heroica a sangre sobre azul noche: muertes anuales mundiales (1995-2031) y deriva de horizonte."""
    ml, mr, mt, mb = 28, 28, 34, 34
    pw, ph = width - ml - mr, height - mt - mb
    xmin, xmax = 1995, 2031
    xfn = lambda yr: ml + (yr - xmin) / (xmax - xmin) * pw

    wa = D["world_annual"]
    yrs_h = wa["years"]  # 1989..2026
    obs_d = wa["deaths"] # index 0=1989, 36=2025, 37=2026(9m)

    # Sum regional drift_60m annual p50/p10/p90 across the 5 regions (illustrative sum, declared!)
    regs = D["meta"]["regions"]
    f_yrs = [2025, 2027, 2028, 2029, 2030, 2031]
    d25 = obs_d[36]
    ag_p50 = [d25] + [sum(D["regions"][r]["drift_60m"]["annual_deaths"][f"Y{i}"]["p50"] for r in regs) for i in range(1, 6)]
    ag_p10 = [d25] + [sum(D["regions"][r]["drift_60m"]["annual_deaths"][f"Y{i}"]["p10"] for r in regs) for i in range(1, 6)]
    ag_p90 = [d25] + [sum(D["regions"][r]["drift_60m"]["annual_deaths"][f"Y{i}"]["p90"] for r in regs) for i in range(1, 6)]
    # Hurdle annual sum for Y1..Y3 (2027..2029)
    hu_yrs = [2025, 2027, 2028, 2029]
    hu_p50 = [d25] + [
        sum(sum(D["regions"][r]["live_36m"]["hurdle_monthly"]["p50"][(i - 1) * 12:i * 12]) for r in regs)
        for i in range(1, 4)
    ]

    # Use sqrt/compressed scale or linear capped cleanly so both 50k–250k history and 2027-31 blowup look dramatic!
    ymax_v = max(max(obs_d[6:37]), max(ag_p90[:4])) * 1.15
    ymax_v = min(ymax_v, 900_000.0)
    yfn = lambda v: mt + (1.0 - min(v, ymax_v) / ymax_v) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    for yr in range(1995, 2032, 5):
        xx = xfn(yr)
        parts.append(f'<line x1="{xx:.1f}" y1="{mt - 14}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#4E6591" stroke-width="0.65" opacity="0.22"/>')
    for yr in (2025, 2029, 2031):
        xx = xfn(yr)
        parts.append(f'<line x1="{xx:.1f}" y1="{mt - 14}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#6B82B4" stroke-width="0.7" opacity="0.32" stroke-dasharray="2.5,2.5"/>')

    # Regional ghost curves (1995-2025)
    ghost_meta = [
        ("africa", "África", 2001, -6),
        ("mideast", "Oriente Medio", 2014, -7),
        ("europe", "Europa", 2023, -7),
        ("asia", "Asia", 2008, -6),
        ("americas", "Américas", 2019, 10),
    ]
    for rk, rlbl, lyr, dy in ghost_meta:
        rd = D["regions"][rk]["history"]["annual"]["deaths"]
        pts = [(xfn(y), yfn(v)) for y, v in zip(yrs_h, rd) if 1995 <= y <= 2025]
        parts.append(f'<path d="{_poly(pts)}" fill="none" stroke="#6B82B4" stroke-width="0.95" opacity="0.45"/>')
        v_l = rd[lyr - 1989]
        parts.append(f'<text x="{xfn(lyr):.1f}" y="{yfn(v_l) + dy:.1f}" font-family="IBM Plex Mono" font-size="6.6" fill="#7E95C4" opacity="0.88">{escape(rlbl)}</text>')

    # Agents ribbon 2025 -> 2031
    hi_pts = [(xfn(y), yfn(v)) for y, v in zip(f_yrs, ag_p90)]
    lo_pts = [(xfn(y), yfn(v)) for y, v in zip(f_yrs, ag_p10)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (hi_pts + lo_pts[::-1]))
    parts.append(f'<polygon points="{poly}" fill="rgba(232, 116, 45, 0.26)"/>')

    # Hurdle line (cobalt blue)
    hu_pts = [(xfn(y), yfn(v)) for y, v in zip(hu_yrs, hu_p50)]
    parts.append(f'<path d="{_poly(hu_pts)}" fill="none" stroke="#6E9AFF" stroke-width="2.0" stroke-dasharray="4,2.5"/>')

    # Agents median (burnt orange)
    ag_pts = [(xfn(y), yfn(v)) for y, v in zip(f_yrs, ag_p50)]
    parts.append(f'<path d="{_poly(ag_pts)}" fill="none" stroke="{ORANGE_BRIGHT}" stroke-width="2.5"/>')

    # Observed world deaths 1995..2025 (pure white + glow at 2025)
    obs_pts = [(xfn(y), yfn(v)) for y, v in zip(yrs_h, obs_d) if 1995 <= y <= 2025]
    parts.append(f'<path d="{_poly(obs_pts)}" fill="none" stroke="#FFFFFF" stroke-width="2.6"/>')
    x25, y25 = obs_pts[-1]
    parts.append(f'<circle cx="{x25:.1f}" cy="{y25:.1f}" r="6.5" fill="rgba(255,255,255,0.22)"/>')
    parts.append(f'<circle cx="{x25:.1f}" cy="{y25:.1f}" r="3.5" fill="#FFFFFF"/>')

    # Labels
    parts.append(f'<text x="{xfn(2002):.1f}" y="{yfn(obs_d[14]) - 42:.1f}" font-family="IBM Plex Sans" font-size="8.6" font-weight="500" fill="#FFFFFF">Muertes anuales reales (UCDP GED, hasta 2025)</text>')
    parts.append(f'<text x="{width - 14:.1f}" y="{yfn(ag_p50[-1]) - 10:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="8.5" font-weight="500" fill="{ORANGE_BRIGHT}">Agentes L2 · deriva 2027–2031</text>')
    parts.append(f'<text x="{width - 14:.1f}" y="{yfn(hu_p50[-1]) + 15:.1f}" text-anchor="end" font-family="IBM Plex Sans" font-size="8.0" fill="#8BB0FF">Referencia hurdle</text>')

    for yr, lbl in ((1995, "1995"), (2005, "2005"), (2015, "2015"), (2025, "2025 · GED"), (2029, "2029"), (2031, "2031")):
        parts.append(f'<text x="{xfn(yr):.1f}" y="{mt + ph + 16}" text-anchor="middle" font-family="IBM Plex Mono" font-size="7.1" fill="#95A5C6">{lbl}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
