"""Generador de gráficos SVG vectoriales de alta precisión editorial para el informe L1 (Atlas Mundo).
Estilo inspirado en la maqueta Phy-World · World4: limpio, sin ruido visual, etiquetas directas
sin colisiones y paleta coherente (tinta #0E162B, azul #2F66D0, naranja quemado #D95D24).
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
AMBER = "#B58100"


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


def _nice_ticks(vmin: float, vmax: float, n: int = 4) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1.2:
        step = 1 * mag
    elif norm <= 2.2:
        step = 2 * mag
    elif norm <= 3.0:
        step = 2.5 * mag
    elif norm <= 6.0:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    ticks = []
    val = start
    while val <= vmax + step * 1e-6:
        if val >= vmin - step * 1e-6:
            ticks.append(round(val, 10))
        val += step
    return ticks


def _poly_path(pts: list[tuple[float, float]]) -> str:
    if not pts:
        return ""
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _segments(years: list[int], vals: list[float | None], xfn, yfn) -> list[list[tuple[float, float]]]:
    segs, cur = [], []
    for yr, v in zip(years, vals):
        if v is not None and math.isfinite(v):
            cur.append((xfn(yr), yfn(v)))
        else:
            if cur:
                segs.append(cur)
                cur = []
    if cur:
        segs.append(cur)
    return segs


def _deconflict_y(items: list[tuple[float, str, str]], ymin: float, ymax: float, min_gap: float = 10.5):
    """Separa etiquetas verticales en el margen derecho para evitar cualquier solapamiento."""
    if not items:
        return []
    ordered = sorted(items, key=lambda t: t[0])
    ys = [max(ymin + 4, min(ymax - 2, t[0])) for t in ordered]
    for _ in range(12):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                mid = 0.5 * (ys[i] + ys[i - 1])
                ys[i - 1] = max(ymin + 4, mid - min_gap / 2)
                ys[i] = min(ymax - 2, ys[i - 1] + min_gap)
    return [(y, t[1], t[2]) for y, t in zip(ys, ordered)]


def ts_chart(
    title: str,
    subtitle: str,
    *,
    obs: tuple[list[int], list[float | None]] | None = None,
    hist: tuple[list[int], list[float | None]] | None = None,
    test_past: tuple[list[int], list[float | None]] | None = None,
    test_past2: tuple[list[int], list[float | None]] | None = None,
    proj: tuple[list[int], list[float | None], list[float | None], list[float | None]] | None = None,
    xmin: int = 1970,
    xmax: int = 2041,
    ymin: float | None = None,
    ymax: float | None = None,
    y_fmt=None,
    end_fmt=None,
    width: float = 340,
    height: float = 165,
    vlines: list[tuple[int, str]] | None = None,
    xticks: list[int] | None = None,
    scale: float = 1.0,
    zero_line: bool = False,
    caption: str = "",
) -> str:
    """Gráfico de serie temporal limpio con banda p10-p90 y etiquetas directas."""
    ml, mr, mt, mb = 36, 42, 34, 22
    pw, ph = width - ml - mr, height - mt - mb

    def sc(arr):
        return [None if (v is None or not math.isfinite(v)) else v * scale for v in arr]

    all_y = []
    if obs:
        obs_y = sc(obs[1])
        all_y += [v for yr, v in zip(obs[0], obs_y) if xmin <= yr <= xmax and v is not None]
    else:
        obs_y = None
    if hist:
        hist_y = sc(hist[1])
        all_y += [v for yr, v in zip(hist[0], hist_y) if xmin <= yr <= xmax and v is not None]
    else:
        hist_y = None
    if test_past:
        tp_y = sc(test_past[1])
        all_y += [v for yr, v in zip(test_past[0], tp_y) if xmin <= yr <= xmax and v is not None]
    else:
        tp_y = None
    if test_past2:
        tp2_y = sc(test_past2[1])
        all_y += [v for yr, v in zip(test_past2[0], tp2_y) if xmin <= yr <= xmax and v is not None]
    else:
        tp2_y = None
    if proj:
        p_yrs = proj[0]
        p10, p50, p90 = sc(proj[1]), sc(proj[2]), sc(proj[3])
        all_y += [v for yr, v in zip(p_yrs, p10) if xmin <= yr <= xmax and v is not None]
        all_y += [v for yr, v in zip(p_yrs, p90) if xmin <= yr <= xmax and v is not None]
    else:
        p_yrs = p10 = p50 = p90 = None

    if not all_y:
        all_y = [0.0, 1.0]
    lo, hi = min(all_y), max(all_y)
    pad = (hi - lo) * 0.12 if hi > lo else 1.0
    ylo = ymin if ymin is not None else lo - pad * 0.6
    yhi = ymax if ymax is not None else hi + pad
    ticks = _nice_ticks(ylo, yhi, n=4)
    if ymin is None:
        ylo = min(ylo, ticks[0])
    if ymax is None:
        yhi = max(yhi, ticks[-1])

    def xfn(yr: float) -> float:
        return ml + (yr - xmin) / (xmax - xmin) * pw

    def yfn(v: float) -> float:
        return mt + (1.0 - (v - ylo) / max(yhi - ylo, 1e-12)) * ph

    if y_fmt is None:
        d_auto = 0 if (yhi - ylo) >= 20 else (1 if (yhi - ylo) >= 2 else 2)
        y_fmt = lambda v: fmt_es(v, d_auto)
    if end_fmt is None:
        end_fmt = y_fmt

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;overflow:visible;">'
    ]
    # Header
    parts.append(f'<text x="0" y="12" font-family="IBM Plex Sans" font-size="9.5" font-weight="600" fill="{INK}">{escape(title)}</text>')
    parts.append(f'<text x="0" y="23" font-family="IBM Plex Mono" font-size="6.4" letter-spacing="0.4" fill="{MUTED}">{escape(subtitle.upper())}</text>')

    # Shaded 15-year forecast zone (2026-2041)
    if xmax >= 2041 and xmin <= 2026:
        x26, x41 = xfn(2026), xfn(2041)
        parts.append(f'<rect x="{x26:.2f}" y="{mt}" width="{x41 - x26:.2f}" height="{ph}" fill="#F6F8FB"/>')

    # Horizontal grid + y ticks
    for tv in ticks:
        if ylo - 1e-9 <= tv <= yhi + 1e-9:
            yy = yfn(tv)
            stroke = "#CBD5E1" if (zero_line and abs(tv) < 1e-9) else GRID
            sw = "0.9" if (zero_line and abs(tv) < 1e-9) else "0.65"
            parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{stroke}" stroke-width="{sw}"/>')
            parts.append(f'<text x="{ml - 5}" y="{yy + 2.5:.2f}" text-anchor="end" font-family="IBM Plex Mono" font-size="6.8" fill="{MUTED}">{escape(y_fmt(tv))}</text>')

    # Vertical reference lines (e.g. 2019 panel cutoff, 2026 today)
    default_vlines = [(2019, "2019")] if width < 210 else [(2019, "2019"), (2026, "2026")]
    for yr_v, lbl in (vlines if vlines is not None else default_vlines):
        if xmin < yr_v < xmax:
            xx = xfn(yr_v)
            dash = "2.2,2.2" if yr_v == 2019 else "1.2,2.4"
            col = "#94A3B8" if yr_v == 2019 else "#B8C4D4"
            parts.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt + ph}" stroke="{col}" stroke-width="0.75" stroke-dasharray="{dash}"/>')
            parts.append(f'<text x="{xx:.2f}" y="{mt - 3}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.3" fill="{MUTED}">{escape(lbl)}</text>')

    # X-axis ticks
    for xt in (xticks or [1970, 2000, 2019, 2041]):
        if xmin <= xt <= xmax:
            xx = xfn(xt)
            parts.append(f'<text x="{xx:.2f}" y="{mt + ph + 12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.8" fill="{MUTED}">{xt}</text>')

    # Clip data paths inside plot box
    clip_id = f"c_{abs(hash((title, subtitle, width, height))) % 10**8}"
    parts.append(f'<defs><clipPath id="{clip_id}"><rect x="{ml - 1}" y="{mt - 2}" width="{pw + 2}" height="{ph + 4}"/></clipPath></defs>')
    parts.append(f'<g clip-path="url(#{clip_id})">')

    # Historical replay (subtle gray-blue line if requested)
    if hist and hist_y is not None:
        for seg in _segments(hist[0], hist_y, xfn, yfn):
            parts.append(f'<path d="{_poly_path(seg)}" fill="none" stroke="#94A3B8" stroke-width="1.15" opacity="0.85"/>')

    # Projection p10-p90 ribbon
    end_labels = []
    if proj and p_yrs is not None:
        pts_hi, pts_lo = [], []
        for yr, lo_v, hi_v in zip(p_yrs, p10, p90):
            if xmin <= yr <= xmax and lo_v is not None and hi_v is not None:
                pts_hi.append((xfn(yr), yfn(hi_v)))
                pts_lo.append((xfn(yr), yfn(lo_v)))
        if len(pts_hi) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in (pts_hi + pts_lo[::-1]))
            parts.append(f'<polygon points="{poly}" fill="{ORANGE_BAND}" stroke="none"/>')

    # Past out-of-sample tests
    if test_past and tp_y is not None:
        for seg in _segments(test_past[0], tp_y, xfn, yfn):
            parts.append(f'<path d="{_poly_path(seg)}" fill="none" stroke="{BLUE}" stroke-width="1.35" stroke-dasharray="3.2,2.2"/>')
    if test_past2 and tp2_y is not None:
        for seg in _segments(test_past2[0], tp2_y, xfn, yfn):
            parts.append(f'<path d="{_poly_path(seg)}" fill="none" stroke="{TEAL}" stroke-width="1.25" stroke-dasharray="1.8,1.8"/>')

    # Projection median
    if proj and p_yrs is not None:
        for seg in _segments(p_yrs, p50, xfn, yfn):
            parts.append(f'<path d="{_poly_path(seg)}" fill="none" stroke="{ORANGE}" stroke-width="1.85"/>')
        # End bracket & label at xmax
        valid_p = [(yr, v, lo_v, hi_v) for yr, v, lo_v, hi_v in zip(p_yrs, p50, p10, p90) if v is not None and yr <= xmax]
        if valid_p:
            _, v_end, lo_end, hi_end = valid_p[-1]
            end_labels.append((yfn(v_end), end_fmt(v_end), ORANGE))

    # Observed data line + terminal dot
    if obs and obs_y is not None:
        for seg in _segments(obs[0], obs_y, xfn, yfn):
            parts.append(f'<path d="{_poly_path(seg)}" fill="none" stroke="{INK}" stroke-width="1.75"/>')
        valid_o = [(yr, v) for yr, v in zip(obs[0], obs_y) if xmin <= yr <= xmax and v is not None]
        if valid_o:
            yr_last, v_last = valid_o[-1]
            parts.append(f'<circle cx="{xfn(yr_last):.2f}" cy="{yfn(v_last):.2f}" r="2.3" fill="{INK}" stroke="white" stroke-width="0.8"/>')

    parts.append('</g>')

    # Right-margin end labels (outside clip)
    if proj and p_yrs is not None:
        valid_p = [(yr, v, lo_v, hi_v) for yr, v, lo_v, hi_v in zip(p_yrs, p50, p10, p90) if v is not None and yr <= xmax]
        if valid_p:
            yr_e, v_e, lo_e, hi_e = valid_p[-1]
            xe = xfn(yr_e) + 2.5
            if lo_e is not None and hi_e is not None and abs(yfn(hi_e) - yfn(lo_e)) > 4:
                parts.append(f'<line x1="{xe:.2f}" y1="{yfn(hi_e):.2f}" x2="{xe:.2f}" y2="{yfn(lo_e):.2f}" stroke="{ORANGE}" stroke-width="1.3" opacity="0.65"/>')
    for yy, txt, col in _deconflict_y(end_labels, mt, mt + ph):
        parts.append(f'<text x="{ml + pw + 6:.2f}" y="{yy + 2.5:.2f}" font-family="IBM Plex Mono" font-size="7.0" font-weight="500" fill="{col}">{escape(txt)}</text>')

    parts.append("</svg>")
    svg = "\n".join(parts)
    if caption:
        return f'<div class="chart-box">{svg}<div class="chart-cap">{caption}</div></div>'
    return f'<div class="chart-box">{svg}</div>'


def paired_hbar_chart(
    rows: list[tuple[str, float, float, str, str]],
    *,
    width: float = 520,
    row_h: float = 24,
    label_a: str = "Persistencia / referencia",
    label_b: str = "Atlas L1 (fuera de muestra)",
    max_val: float | None = None,
) -> str:
    """Barras horizontales pareadas (estilo Capítulo 03 del informe de referencia)."""
    ml, mr, mt, mb = 162, 54, 26, 8
    pw = width - ml - mr
    height = mt + len(rows) * row_h + mb
    vmax = max_val or max(max(r[1], r[2]) for r in rows) * 1.12

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    # Legend at top
    parts.append(f'<rect x="{ml}" y="6" width="12" height="6" rx="1" fill="{BLUE}"/>')
    parts.append(f'<text x="{ml + 17}" y="11.5" font-family="IBM Plex Sans" font-size="7.3" fill="{INK2}">{escape(label_a)}</text>')
    parts.append(f'<rect x="{ml + 188}" y="6" width="12" height="6" rx="1" fill="{ORANGE}"/>')
    parts.append(f'<text x="{ml + 205}" y="11.5" font-family="IBM Plex Sans" font-size="7.3" fill="{INK2}">{escape(label_b)}</text>')

    for i, (lbl, va, vb, ta, tb) in enumerate(rows):
        y0 = mt + i * row_h
        parts.append(f'<text x="{ml - 8}" y="{y0 + 11.5}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.8" fill="{INK}">{escape(lbl)}</text>')
        wa = max(1.5, va / vmax * pw)
        wb = max(1.5, vb / vmax * pw)
        parts.append(f'<rect x="{ml}" y="{y0 + 2}" width="{wa:.2f}" height="6.8" rx="1" fill="{BLUE}"/>')
        parts.append(f'<text x="{ml + wa + 4:.2f}" y="{y0 + 8.0}" font-family="IBM Plex Mono" font-size="6.8" fill="{MUTED}">{escape(ta)}</text>')
        parts.append(f'<rect x="{ml}" y="{y0 + 10.5}" width="{wb:.2f}" height="6.8" rx="1" fill="{ORANGE}"/>')
        parts.append(f'<text x="{ml + wb + 4:.2f}" y="{y0 + 16.5}" font-family="IBM Plex Mono" font-size="6.8" font-weight="500" fill="{INK}">{escape(tb)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def coverage_dot_chart(
    rows: list[tuple[str, float, float]],
    *,
    label_a: str = "Cobertura país-año (2020–2025)",
    label_b: str = "Cobertura 15 años (desde 2010)",
    width: float = 520,
    row_h: float = 20,
) -> str:
    """Gráfico de puntos de cobertura p10-p90 con línea nominal del 80 % (estilo pág. 14)."""
    ml, mr, mt, mb = 158, 26, 36, 10
    pw = width - ml - mr
    height = mt + len(rows) * row_h + mb
    xfn = lambda pct: ml + pct / 100.0 * pw

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    # Legend
    parts.append(f'<circle cx="{ml + 4}" cy="8" r="3.4" fill="{INK}"/>')
    parts.append(f'<text x="{ml + 12}" y="10.8" font-family="IBM Plex Sans" font-size="7.3" fill="{INK2}">{escape(label_a)}</text>')
    parts.append(f'<circle cx="{ml + 195}" cy="8" r="3.4" fill="white" stroke="{ORANGE}" stroke-width="1.4"/>')
    parts.append(f'<text x="{ml + 203}" y="10.8" font-family="IBM Plex Sans" font-size="7.3" fill="{INK2}">{escape(label_b)}</text>')

    # Axis ticks
    for p in (0, 20, 40, 60, 80, 100):
        xx = xfn(p)
        col = "#475569" if p == 80 else GRID
        sw = "1.0" if p == 80 else "0.6"
        parts.append(f'<line x1="{xx:.2f}" y1="{mt - 6}" x2="{xx:.2f}" y2="{height - mb}" stroke="{col}" stroke-width="{sw}"/>')
        parts.append(f'<text x="{xx:.2f}" y="{mt - 10}" text-anchor="middle" font-family="IBM Plex Mono" font-size="6.7" fill="{INK if p == 80 else MUTED}">{p} %</text>')

    for i, (lbl, va, vb) in enumerate(rows):
        yc = mt + i * row_h + row_h * 0.5
        parts.append(f'<text x="{ml - 8}" y="{yc + 2.6:.2f}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.6" fill="{INK}">{escape(lbl)}</text>')
        xa, xb = xfn(va), xfn(vb)
        parts.append(f'<line x1="{min(xa, xb):.2f}" y1="{yc:.2f}" x2="{max(xa, xb):.2f}" y2="{yc:.2f}" stroke="#CBD5E1" stroke-width="1.2"/>')
        parts.append(f'<circle cx="{xb:.2f}" cy="{yc:.2f}" r="3.4" fill="white" stroke="{ORANGE}" stroke-width="1.4"/>')
        parts.append(f'<circle cx="{xa:.2f}" cy="{yc:.2f}" r="3.4" fill="{INK}"/>')
        # Value label for main dot: place outside the connecting segment so it never crosses the line
        if va >= vb:
            if va > 92:
                tx, anc = xa - 6.5, "end"
            else:
                tx, anc = xa + 6.5, "start"
        else:
            if va < 12:
                tx, anc = max(xa, xb) + 6.5, "start"
            else:
                tx, anc = xa - 6.5, "end"
        parts.append(f'<text x="{tx:.2f}" y="{yc + 2.4:.2f}" text-anchor="{anc}" font-family="IBM Plex Mono" font-size="6.4" fill="{INK}">{va:.0f} %</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def diverging_bar_chart(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    *,
    width: float = 252,
    row_h: float = 21,
) -> str:
    """Barras divergentes centradas en 0 para ablaciones de mecanismos."""
    ml, mr, mt, mb = 104, 34, 32, 8
    pw = width - ml - mr
    height = mt + len(rows) * row_h + mb
    vmax = max(max(abs(r[1]) for r in rows), 0.5) * 1.48
    x0 = ml + pw * 0.54

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="0" y="11" font-family="IBM Plex Sans" font-size="8.6" font-weight="600" fill="{INK}">{escape(title)}</text>')
    parts.append(f'<text x="0" y="21" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">{escape(subtitle.upper())}</text>')
    parts.append(f'<line x1="{x0:.2f}" y1="{mt - 2}" x2="{x0:.2f}" y2="{height - mb}" stroke="#94A3B8" stroke-width="0.8"/>')

    for i, (lbl, val, txt) in enumerate(rows):
        yc = mt + i * row_h + row_h * 0.5
        parts.append(f'<text x="{ml - 6}" y="{yc + 2.5:.2f}" text-anchor="end" font-family="IBM Plex Sans" font-size="6.9" fill="{INK2}">{escape(lbl)}</text>')
        bw = abs(val) / vmax * (pw * 0.44)
        col = ORANGE if val >= 0 else BLUE
        bx = x0 if val >= 0 else x0 - bw
        parts.append(f'<rect x="{bx:.2f}" y="{yc - 4.5:.2f}" width="{max(bw, 1.0):.2f}" height="9" rx="1" fill="{col}"/>')
        tx = x0 + bw + 3.5 if val >= 0 else x0 - bw - 3.5
        anc = "start" if val >= 0 else "end"
        parts.append(f'<text x="{tx:.2f}" y="{yc + 2.4:.2f}" text-anchor="{anc}" font-family="IBM Plex Mono" font-size="6.2" fill="{INK}">{escape(txt)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def param_stability_chart(
    rows: list[tuple[str, float, float, float, float, float, str]],
    *,
    width: float = 520,
    row_h: float = 19,
) -> str:
    """Barras de estabilidad de parámetros calibrados entre cortes 1970-2019 (estilo pág. 13).
    Cada tupla: (nombre, lo_box, hi_box, min_cut, max_cut, val_2019, texto_derecha).
    """
    ml, mr, mt, mb = 185, 118, 22, 8
    pw = width - ml - mr
    height = mt + len(rows) * row_h + mb

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    parts.append(f'<text x="{ml}" y="11" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">MÍNIMO BUSCADO</text>')
    parts.append(f'<text x="{ml + pw}" y="11" text-anchor="end" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">MÁXIMO BUSCADO</text>')
    parts.append(f'<text x="{ml + pw + 10}" y="11" font-family="IBM Plex Mono" font-size="6.1" fill="{MUTED}">1970–2019 · CORTE 2019</text>')

    for i, (lbl, lo, hi, cmin, cmax, v19, rtxt) in enumerate(rows):
        yc = mt + i * row_h + row_h * 0.5
        span = max(hi - lo, 1e-9)
        pos = lambda v: ml + max(0.0, min(1.0, (v - lo) / span)) * pw
        parts.append(f'<text x="{ml - 8}" y="{yc + 2.5:.2f}" text-anchor="end" font-family="IBM Plex Sans" font-size="7.3" fill="{INK}">{escape(lbl)}</text>')
        # Full search box
        parts.append(f'<rect x="{ml}" y="{yc - 3.6:.2f}" width="{pw}" height="7.2" rx="1.5" fill="#E6ECF3"/>')
        # Observed cutoff range 1970-2019
        x1, x2 = pos(cmin), pos(cmax)
        parts.append(f'<rect x="{x1:.2f}" y="{yc - 3.6:.2f}" width="{max(x2 - x1, 3.0):.2f}" height="7.2" rx="1.5" fill="#3B4D6B"/>')
        # 2019 value tick
        xv = pos(v19)
        parts.append(f'<line x1="{xv:.2f}" y1="{yc - 4.8:.2f}" x2="{xv:.2f}" y2="{yc + 4.8:.2f}" stroke="{ORANGE_BRIGHT}" stroke-width="2.0"/>')
        parts.append(f'<text x="{ml + pw + 10}" y="{yc + 2.5:.2f}" font-family="IBM Plex Mono" font-size="6.5" fill="{INK2}">{escape(rtxt)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def cover_hero_svg(data: dict, width: float = 595, height: float = 320) -> str:
    """Gráfico heroico a sangre en fondo azul noche (#0E162B), estilo portada del PDF anexado."""
    ml, mr, mt, mb = 28, 28, 34, 34
    pw, ph = width - ml - mr, height - mt - mb
    xmin, xmax = 1970, 2041

    xfn = lambda yr: ml + (yr - xmin) / (xmax - xmin) * pw

    obs_w = data["observed_world"]
    yrs_obs = obs_w["years"]  # 1950..2025
    s19 = data["series"]["proj_2019_2041"]
    yrs_19 = list(range(2019, 2042))
    s00 = data["series"]["test_2000_2025"]
    yrs_00 = list(range(2000, 2026))
    sh = data["series"]["hist_1950_2019"]
    yrs_h = list(range(1950, 2020))

    # Main curve: World GDP per capita (USD 2017) normalized to 0..1 visually
    y_min, y_max = 4000.0, 28000.0
    yfn = lambda v: mt + (1.0 - (v - y_min) / (y_max - y_min)) * ph

    # Ghost curves (rescaled to 0.12..0.82 of plot height)
    def ghost_y(arr, lo, hi):
        return [mt + (1.0 - (0.08 + 0.72 * max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9))))) * ph
                if (v is not None and math.isfinite(v)) else None for v in arr]

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    # Subtle vertical decade gridlines
    for yr in range(1970, 2045, 5):
        xx = xfn(yr)
        op = "0.28" if yr in (1970, 2000, 2019, 2026, 2041) else "0.12"
        parts.append(f'<line x1="{xx:.2f}" y1="{mt - 14}" x2="{xx:.2f}" y2="{mt + ph}" stroke="#4E6591" stroke-width="0.7" opacity="{op}"/>')

    # Horizontal subtle gridlines
    for frac in (0.2, 0.4, 0.6, 0.8):
        yy = mt + frac * ph
        parts.append(f'<line x1="0" y1="{yy:.2f}" x2="{width}" y2="{yy:.2f}" stroke="#4E6591" stroke-width="0.55" opacity="0.14"/>')

    # Ghost curves: GMST, Debt, Democracy, Conflict, Natural Capital (1970-2041 stitched hist+proj)
    ghosts = [
        ("temperatura global", 0.2, 2.0, "Tg", 2033, -7),
        ("deuda pública", 0.2, 0.75, "debt", 1986, -6),
        ("democracia", 0.15, 0.75, "demo", 2008, -7),
        ("conflicto civil", 0.04, 0.22, "hazard", 1983, 11),
        ("capital natural", 0.65, 1.05, "natcap", 2032, 11),
    ]
    for gname, glo, ghi, gkey, lbl_yr, dy in ghosts:
        full_yrs = list(range(1970, 2042))
        full_vals = [sh[gkey]["p50"][yr - 1950] if yr <= 2019 else s19[gkey]["p50"][yr - 2019] for yr in full_yrs]
        gys = ghost_y(full_vals, glo, ghi)
        pts = [(xfn(yr), yy) for yr, yy in zip(full_yrs, gys) if yy is not None]
        parts.append(f'<path d="{_poly_path(pts)}" fill="none" stroke="#6B82B4" stroke-width="0.9" opacity="0.42"/>')
        lbl_idx = lbl_yr - 1970
        if 0 <= lbl_idx < len(gys) and gys[lbl_idx] is not None:
            parts.append(f'<text x="{xfn(lbl_yr):.2f}" y="{gys[lbl_idx] + dy:.2f}" font-family="IBM Plex Mono" font-size="6.8" fill="#7D93C2" opacity="0.85">{escape(gname)}</text>')

    # Projection p10-p90 ribbon (2019-2041)
    pts_hi = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_19, s19["gdp_pc_world"]["p90"])]
    pts_lo = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_19, s19["gdp_pc_world"]["p10"])]
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in (pts_hi + pts_lo[::-1]))
    parts.append(f'<polygon points="{poly}" fill="rgba(232, 116, 45, 0.28)" stroke="none"/>')

    # Past out-of-sample test from 2000 (2000-2025) in luminous cobalt blue
    pts_00 = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_00, s00["gdp_pc_world"]["p50"])]
    parts.append(f'<path d="{_poly_path(pts_00)}" fill="none" stroke="#6E9AFF" stroke-width="1.9" stroke-dasharray="4.2,2.8"/>')

    # Projection median (2019-2041) in bright burnt orange
    pts_19 = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_19, s19["gdp_pc_world"]["p50"])]
    parts.append(f'<path d="{_poly_path(pts_19)}" fill="none" stroke="{ORANGE_BRIGHT}" stroke-width="2.5"/>')

    # Observed world GDP pc (1970-2025) in pure crisp white + glowing endpoint at 2025
    pts_obs = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_obs, obs_w["gdp_pc_world"]) if yr >= 1970 and v is not None]
    parts.append(f'<path d="{_poly_path(pts_obs)}" fill="none" stroke="#FFFFFF" stroke-width="2.6"/>')
    # Marker at 2019 (panel anchor) and glowing dot at 2025 (latest real data)
    x19, y19 = xfn(2019), yfn(obs_w["gdp_pc_world"][69])
    x25, y25 = xfn(2025), yfn(obs_w["gdp_pc_world"][75])
    parts.append(f'<circle cx="{x19:.2f}" cy="{y19:.2f}" r="3.0" fill="#0E162B" stroke="#FFFFFF" stroke-width="1.4"/>')
    parts.append(f'<circle cx="{x25:.2f}" cy="{y25:.2f}" r="6.5" fill="rgba(255,255,255,0.22)"/>')
    parts.append(f'<circle cx="{x25:.2f}" cy="{y25:.2f}" r="3.6" fill="#FFFFFF"/>')

    # Direct curve labels
    parts.append(f'<text x="{xfn(1972):.2f}" y="{yfn(obs_w["gdp_pc_world"][22]) - 26:.2f}" font-family="IBM Plex Sans" font-size="8.6" font-weight="500" fill="#FFFFFF">PIB per cápita real (hasta 2025)</text>')
    parts.append(f'<text x="{xfn(2005):.2f}" y="{yfn(s00["gdp_pc_world"]["p50"][5]) + 22:.2f}" font-family="IBM Plex Sans" font-size="8.2" fill="#8BB0FF">Prueba ciega desde 2000</text>')
    parts.append(f'<text x="{width - 16:.2f}" y="{yfn(s19["gdp_pc_world"]["p90"][-1]) - 8:.2f}" text-anchor="end" font-family="IBM Plex Sans" font-size="8.6" font-weight="500" fill="{ORANGE_BRIGHT}">Atlas L1 · proyección 2026–2041</text>')

    # X axis ticks
    for yr, lbl in ((1970, "1970"), (2000, "2000"), (2019, "2019 · ancla"), (2026, "2026 · hoy"), (2041, "2041")):
        xx = xfn(yr)
        parts.append(f'<text x="{xx:.2f}" y="{mt + ph + 16}" text-anchor="middle" font-family="IBM Plex Mono" font-size="7.2" fill="#95A5C6">{lbl}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def country_mini_svg(
    iso3: str,
    name: str,
    obs_2000_2025: list[float | None],
    sim_2019_2041: dict,
    *,
    err25_txt: str,
    val41_txt: str,
    width: float = 122,
    height: float = 76,
) -> str:
    """Mini-gráfico de PIB per cápita (2000-2041) para grilla de 12 países."""
    ml, mr, mt, mb = 6, 6, 22, 13
    pw, ph = width - ml - mr, height - mt - mb
    xmin, xmax = 2000, 2041
    xfn = lambda yr: ml + (yr - xmin) / (xmax - xmin) * pw

    vals = [v for v in obs_2000_2025 if v is not None] + sim_2019_2041["p10"] + sim_2019_2041["p90"]
    lo, hi = min(vals) * 0.85, max(vals) * 1.08
    yfn = lambda v: mt + (1.0 - (v - lo) / max(hi - lo, 1e-9)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    # Card frame & title
    parts.append(f'<text x="0" y="9" font-family="IBM Plex Sans" font-size="7.6" font-weight="600" fill="{INK}">{escape(name)}</text>')
    parts.append(f'<text x="{width}" y="9" text-anchor="end" font-family="IBM Plex Mono" font-size="6.2" fill="{ORANGE}">{escape(val41_txt)}</text>')
    parts.append(f'<text x="0" y="17.5" font-family="IBM Plex Mono" font-size="5.8" fill="{MUTED}">{iso3} · 2025: {escape(err25_txt)}</text>')

    # Forecast shading 2026-2041
    parts.append(f'<rect x="{xfn(2026):.2f}" y="{mt}" width="{xfn(2041) - xfn(2026):.2f}" height="{ph}" fill="#F6F8FB"/>')
    # Gridline bottom & mid
    parts.append(f'<line x1="{ml}" y1="{mt + ph:.2f}" x2="{ml + pw}" y2="{mt + ph:.2f}" stroke="{GRID}" stroke-width="0.6"/>')
    parts.append(f'<line x1="{xfn(2019):.2f}" y1="{mt}" x2="{xfn(2019):.2f}" y2="{mt + ph}" stroke="#B8C4D4" stroke-width="0.6" stroke-dasharray="1.8,1.8"/>')

    yrs_s = list(range(2019, 2042))
    pts_hi = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_s, sim_2019_2041["p90"])]
    pts_lo = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_s, sim_2019_2041["p10"])]
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in (pts_hi + pts_lo[::-1]))
    parts.append(f'<polygon points="{poly}" fill="{ORANGE_BAND}"/>')
    pts_m = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_s, sim_2019_2041["p50"])]
    parts.append(f'<path d="{_poly_path(pts_m)}" fill="none" stroke="{ORANGE}" stroke-width="1.35"/>')

    yrs_o = list(range(2000, 2026))
    pts_o = [(xfn(yr), yfn(v)) for yr, v in zip(yrs_o, obs_2000_2025) if v is not None]
    if pts_o:
        parts.append(f'<path d="{_poly_path(pts_o)}" fill="none" stroke="{INK}" stroke-width="1.35"/>')
        parts.append(f'<circle cx="{pts_o[-1][0]:.2f}" cy="{pts_o[-1][1]:.2f}" r="1.9" fill="{INK}" stroke="white" stroke-width="0.6"/>')

    parts.append(f'<text x="{ml}" y="{height - 2}" font-family="IBM Plex Mono" font-size="5.6" fill="{MUTED}">2000</text>')
    parts.append(f'<text x="{xfn(2019):.2f}" y="{height - 2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="5.6" fill="{MUTED}">19</text>')
    parts.append(f'<text x="{ml + pw}" y="{height - 2}" text-anchor="end" font-family="IBM Plex Mono" font-size="5.6" fill="{MUTED}">2041</text>')
    parts.append("</svg>")
    return "\n".join(parts)

