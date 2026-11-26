"""Genera los gráficos de alta resolución y el resumen PDF del nuevo predictor
de democracia V2 (``world4.core.l1_atlas.democracy``):
  1. democracia_v2_curvas.png:
       (A) Serie temporal 1950-2041 de % democracias (real 1950-2025 vs L1 viejo vs Nuevo V2)
       (B) Serie temporal 1950-2041 de calidad democrática continua (vdem_polyarchy)
       (C) Recta de predicción ciega 2004-2019: Δ poly predicho vs Δ poly real (aumentos y decrementos)
  2. democracia_v2_mapa_2041.png:
       Mapamundi coroplético (64.818 celdas PRIO-GRID 0,5°) con la tendencia proyectada
       de cada país a 2041 (Δ vdem_polyarchy = poly_2041 − poly_2019).
  3. democracia_v2_2041.pdf:
       Dossier visual de 2 páginas con ambos gráficos y tabla de mayores avances/retrocesos.

Uso:
    python reports/l1_atlas/plot_democracy_2041.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import weasyprint

from world4.core.l1_atlas import democracy, model, submodels
from world4.core.l2 import panel as l2_panel, regions as l2_regions

HERE = Path(__file__).parent
OUT_CURVES = HERE / "democracia_v2_curvas.png"
OUT_MAP = HERE / "democracia_v2_mapa_2041.png"
OUT_PDF = HERE / "democracia_v2_2041.pdf"
OUT_HTML = HERE / "democracia_v2_2041.html"

# Palette matching Phy-World editorial reports
INK = "#0E162B"
INK2 = "#334155"
MUTED = "#64748B"
ORANGE = "#D95D24"
BLUE = "#2F66D0"
TEAL = "#0F766E"
CRIMSON = "#B83227"
GRID = "#E2E8F0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def run_simulations():
    W = model.build_world(horizon=2041)
    LY = np.log(W.Y / W.pop)
    LY[~W.observed] = np.nan
    cjson = json.loads((model.ROOT / "results" / "calibration.json").read_text(encoding="utf-8"))
    p_2019 = model.Params(**cjson["params"])

    # Observed recent 2020-2025 democracy share from L1 study_results.json (V-Dem / OWID)
    l1_res = json.loads((HERE / "study_results.json").read_text(encoding="utf-8"))
    obs_world_demo = l1_res["observed_world"]["demo_share"]  # 1950..2025 (len 76)

    # 1. Blind 2004 -> 2019 backtest (R=64) for the prediction scatter/line chart
    cut04 = 54
    ov04, _ = submodels.estimate(W, LY, cut04)
    p_04 = p_2019.with_vector(list(ov04), list(ov04.values()))
    dm04 = democracy.fit(W, LY, cut04, rho_u=0.25)
    o_v2_04 = model.Simulator(W, p_04, R=64, seed=42, democracy_model=dm04).run(
        0, 70, freeze_from=cut04, assimilate_at=cut04
    )

    # 2. Live 2019 -> 2041 integrated L1 projection (R=80)
    dm19 = democracy.fit(W, LY, 69, rho_u=0.25)
    o_old = model.Simulator(W, p_2019, R=80, seed=42).run(0, W.T, freeze_from=69, assimilate_at=69)
    o_v2 = model.Simulator(W, p_2019, R=80, seed=42, democracy_model=dm19).run(
        0, W.T, freeze_from=69, assimilate_at=69
    )
    return W, o_old, o_v2, o_v2_04, obs_world_demo


def plot_curves(W, o_old, o_v2, o_v2_04, obs_world_demo):
    row_raw = W.extra.get("regime_row", np.full((W.N, W.T), np.nan))
    demo_raw = np.where(np.isfinite(row_raw), (row_raw >= 2).astype(float), np.nan)
    poly_raw = W.extra.get("vdem_polyarchy", np.full((W.N, W.T), np.nan))
    ok19 = np.isfinite(demo_raw[:, 69]) & W.observed[:, 69]

    yrs_all = np.arange(1950, 2042)
    yrs_obs = np.arange(1950, 2020)
    yrs_rec = np.arange(2019, 2026)
    yrs_f = np.arange(2019, 2042)

    # Historical V-Dem share & polyarchy (1950-2019)
    hist_demo = np.array([np.nanmean(demo_raw[:, t]) * 100.0 for t in range(70)])
    hist_poly = np.array([np.nanmean(poly_raw[:, t]) for t in range(70)])

    # Recent observed 2019-2025 from study_results.json (rescaled cleanly to match 2019 anchor)
    rec_raw = np.array([obs_world_demo[y - 1950] * 100.0 for y in yrs_rec])
    rec_shift = hist_demo[-1] - rec_raw[0]
    rec_demo = rec_raw + rec_shift

    # Forecast series (2019-2041) on ok19 countries
    old_demo = np.array([np.nanmean(o_old["demo"][:, ok19, t]) * 100.0 for t in range(69, 92)])
    v2_demo_r = np.array([np.nanmean(o_v2["demo"][:, ok19, t], axis=1) * 100.0 for t in range(69, 92)])  # (23, R)
    v2_d_p50 = np.median(v2_demo_r, axis=1)
    v2_d_p10 = np.percentile(v2_demo_r, 10, axis=1)
    v2_d_p90 = np.percentile(v2_demo_r, 90, axis=1)

    v2_poly_r = np.array([np.nanmean(o_v2["poly"][:, ok19, t], axis=1) for t in range(69, 92)])
    v2_p_p50 = np.median(v2_poly_r, axis=1)
    v2_p_p10 = np.percentile(v2_poly_r, 10, axis=1)
    v2_p_p90 = np.percentile(v2_poly_r, 90, axis=1)

    # Blind 2004 -> 2019 predicted vs actual delta poly
    cut04 = 54
    ok04 = np.isfinite(poly_raw[:, cut04]) & np.isfinite(poly_raw[:, 69]) & W.observed[:, cut04]
    dp_obs = poly_raw[ok04, 69] - poly_raw[ok04, cut04]
    dp_pred = np.nanmean(o_v2_04["poly"][:, ok04, 69], axis=0) - poly_raw[ok04, cut04]
    slope, intercept = np.polyfit(dp_pred, dp_obs, 1)
    corr = float(np.corrcoef(dp_pred, dp_obs)[0, 1])

    fig = plt.figure(figsize=(13.2, 4.7), dpi=220, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.0], wspace=0.25, left=0.055, right=0.985, top=0.82, bottom=0.14)

    # Panel A: % Democracies 1950-2041
    ax1 = fig.add_subplot(gs[0])
    ax1.axvspan(2019, 2041, color="#F4F6FA", zorder=0)
    ax1.axvline(2019, color="#94A3B8", ls="--", lw=0.9)
    ax1.fill_between(yrs_f, v2_d_p10, v2_d_p90, color=ORANGE, alpha=0.20, label="Nuevo V2 (p10–p90)")
    ax1.plot(yrs_f, old_demo, color=BLUE, ls="--", lw=1.8, label=f"L1 viejo ({old_demo[-1]:.1f} %)")
    ax1.plot(yrs_f, v2_d_p50, color=ORANGE, lw=2.3, label=f"Nuevo democracy.fit ({v2_d_p50[-1]:.1f} %)")
    ax1.plot(yrs_obs, hist_demo, color=INK, lw=2.0, label="Real V-Dem (1950–2019)")
    ax1.plot(yrs_rec, rec_demo, color=CRIMSON, ls=":", lw=2.2, marker="o", ms=3.0, label="Real 2020–2025 (ciego)")
    ax1.set_xlim(1955, 2042)
    ax1.set_ylim(20, 72)
    ax1.grid(True, axis="y", color=GRID, lw=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("A · Fracción mundial de democracias (%)", fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax1.set_ylabel("% de países (RoW ≥ 2)", fontsize=8.8, color=INK2)
    ax1.set_xlabel("Año", fontsize=8.5, color=MUTED)
    ax1.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax1.annotate(f"2025 V2: {v2_d_p50[6]:.1f}%", xy=(2025, v2_d_p50[6]), xytext=(2021.5, 42.5),
                 fontsize=7.3, fontweight="bold", color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))

    # Panel B: Continuous Polyarchy 1950-2041
    ax2 = fig.add_subplot(gs[1])
    ax2.axvspan(2019, 2041, color="#F4F6FA", zorder=0)
    ax2.axvline(2019, color="#94A3B8", ls="--", lw=0.9)
    ax2.fill_between(yrs_f, v2_p_p10, v2_p_p90, color=ORANGE, alpha=0.20, label="Nuevo V2 (p10–p90)")
    ax2.axhline(hist_poly[-1], xmin=(2019-1955)/(2042-1955), xmax=1.0, color=BLUE, ls="--", lw=1.4, label="L1 viejo (poly congelada)")
    ax2.plot(yrs_f, v2_p_p50, color=ORANGE, lw=2.3, label=f"Nuevo V2 ({v2_p_p50[-1]:.3f} en 2041)")
    ax2.plot(yrs_obs, hist_poly, color=INK, lw=2.0, label="Real V-Dem (1950–2019)")
    ax2.set_xlim(1955, 2042)
    ax2.set_ylim(0.22, 0.62)
    ax2.grid(True, axis="y", color=GRID, lw=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("B · Calidad democrática media (V-Dem)", fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax2.set_ylabel("Índice vdem_polyarchy (0–1)", fontsize=8.8, color=INK2)
    ax2.set_xlabel("Año", fontsize=8.5, color=MUTED)
    ax2.legend(loc="lower right", fontsize=7.0, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax2.annotate(f"Erosión gradual:\n0,532 → {v2_p_p50[-1]:.3f}", xy=(2035, v2_p_p50[16]), xytext=(2019.5, 0.43),
                 fontsize=7.3, fontweight="bold", color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))

    # Panel C: Calibration line of predicted vs observed delta poly (blind 2004->2019!)
    ax3 = fig.add_subplot(gs[2])
    ax3.axhline(0, color="#94A3B8", lw=0.8)
    ax3.axvline(0, color="#94A3B8", lw=0.8)
    cols_sc = np.where(dp_obs >= 0, TEAL, CRIMSON)
    ax3.scatter(dp_pred, dp_obs, c=cols_sc, s=22, alpha=0.68, edgecolors="white", linewidths=0.4, zorder=3)
    xs_line = np.linspace(-0.22, 0.22, 50)
    ax3.plot(xs_line, xs_line, color="#94A3B8", ls="--", lw=1.1, label="Recta ideal (y = x)")
    ax3.plot(xs_line, intercept + slope * xs_line, color=ORANGE, lw=2.3, label=f"Ajuste real (pend.={slope:.2f}, r={corr:.2f})")
    ax3.set_xlim(-0.22, 0.22)
    ax3.set_ylim(-0.35, 0.35)
    ax3.grid(True, color=GRID, lw=0.6)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("C · Recta de predicción (ciega 2004→2019)", fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax3.set_xlabel("Δ poly predicho por democracy.fit (15a)", fontsize=8.3, color=INK2)
    ax3.set_ylabel("Δ poly observado real (2004→2019)", fontsize=8.3, color=INK2)
    ax3.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax3.text(0.20, 0.29, "Acierta\navance", ha="right", fontsize=7.0, color=TEAL, fontweight="bold")
    ax3.text(-0.20, -0.32, "Acierta\nretroceso", ha="left", fontsize=7.0, color=CRIMSON, fontweight="bold")

    fig.suptitle(
        "Nuevo predictor democrático V2 (democracy.fit) dentro de Phy-World L1 · Curvas 1950–2041 y recta de calibración ciega",
        fontsize=12.0, fontweight="bold", color=INK, y=0.96,
    )
    fig.savefig(OUT_CURVES, dpi=220)
    plt.close(fig)
    print(f"Gráfico de curvas guardado -> {OUT_CURVES}")


def plot_world_map_2041(W, o_v2):
    poly_raw = W.extra.get("vdem_polyarchy", np.full((W.N, W.T), np.nan))
    row_raw = W.extra.get("regime_row", np.full((W.N, W.T), np.nan))
    demo_raw = np.where(np.isfinite(row_raw), (row_raw >= 2).astype(float), np.nan)

    # Compute projected delta poly (2019 -> 2041) for every country in L1
    delta_by_iso = {}
    stats_list = []
    for i, iso in enumerate(W.iso3):
        p19 = poly_raw[i, 69]
        p41 = float(np.nanmean(o_v2["poly"][:, i, 91]))
        d19 = demo_raw[i, 69]
        d41 = float(np.nanmean(o_v2["demo"][:, i, 91]))
        if np.isfinite(p19) and np.isfinite(p41) and W.observed[i, 69]:
            dp = p41 - p19
            delta_by_iso[iso] = dp
            stats_list.append({
                "iso": iso, "name": W.names[i], "p19": float(p19), "p41": p41,
                "dp": float(dp), "d19": float(d19) if np.isfinite(d19) else np.nan, "d41": d41,
            })

    # Build a crisp 0.5° global grid image (rows = lat, cols = lon) directly from all 5 L2 PRIO-GRID panels!
    # Lat: -56 to +76 (step 0.5 -> 264 rows); Lon: -180 to +180 (step 0.5 -> 720 cols)
    lat_min, lat_max = -56.0, 76.0
    lon_min, lon_max = -180.0, 180.0
    n_lat = int((lat_max - lat_min) * 2)
    n_lon = int((lon_max - lon_min) * 2)
    grid_val = np.full((n_lat, n_lon), np.nan, dtype=float)
    grid_land = np.zeros((n_lat, n_lon), dtype=bool)
    # Also compute country centroids for crisp callout annotations
    iso_coords = {}

    for reg in l2_regions.WORLD:
        P = l2_panel.load(reg)
        for ci in range(P.C):
            lon, lat = float(P.lon[ci]), float(P.lat[ci])
            if not (lat_min <= lat < lat_max and lon_min <= lon < lon_max):
                continue
            r_i = int(round((lat_max - lat - 0.25) * 2))
            c_i = int(round((lon - lon_min - 0.25) * 2))
            if 0 <= r_i < n_lat and 0 <= c_i < n_lon:
                grid_land[r_i, c_i] = True
                ck = int(P.cell_country[ci])
                if ck >= 0:
                    iso = P.iso3[ck]
                    iso_coords.setdefault(iso, []).append((lon, lat))
                    if iso in delta_by_iso:
                        grid_val[r_i, c_i] = delta_by_iso[iso]

    # Build RGB image: ocean = deep editorial navy #0E162B, land without V-Dem = #2B3954,
    # land with V-Dem = diverging Crimson (-0.22) -> Warm Gray (0) -> Emerald Teal (+0.22)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dem_div",
        ["#8C1D18", "#D9483B", "#F29E74", "#E6EAF0", "#76C7B7", "#148F77", "#0B5D51"],
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.22)

    img = np.zeros((n_lat, n_lon, 3), dtype=float)
    img[:, :] = mcolors.to_rgb("#0E162B")
    land_no_data = grid_land & ~np.isfinite(grid_val)
    img[land_no_data] = mcolors.to_rgb("#2C3B59")
    has_val = np.isfinite(grid_val)
    rgba = cmap(norm(np.clip(grid_val[has_val], -0.22, 0.22)))
    img[has_val] = rgba[:, :3]

    fig, ax = plt.subplots(figsize=(13.6, 6.9), dpi=220, facecolor="#0E162B")
    ax.set_facecolor("#0E162B")
    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", interpolation="nearest", aspect="auto")

    # Subtle graticule
    for lat in (-30, 0, 30, 60):
        ax.axhline(lat, color="#253456", lw=0.45, alpha=0.6)
    for lon in (-120, -60, 0, 60, 120):
        ax.axvline(lon, color="#253456", lw=0.45, alpha=0.6)

    # Curated callouts for flagship countries (cleanly spaced so zero overlap!)
    callouts = [
        ("USA", "EE. UU.", -100, 39, -135, 34),
        ("MEX", "México", -102, 23, -132, 16),
        ("HND", "Honduras", -86.5, 14.8, -112, 5),
        ("BRA", "Brasil", -51, -10, -28, -22),
        ("ARG", "Argentina", -65, -36, -92, -43),
        ("HUN", "Hungría", 19.5, 47.2, 8, 63),
        ("UKR", "Ucrania", 31.5, 49.0, 34, 65),
        ("GEO", "Georgia", 43.5, 42.2, 55, 56),
        ("TUR", "Turquía", 35.0, 39.0, 56, 36),
        ("NER", "Níger", 9.0, 17.5, -22, 19),
        ("NGA", "Nigeria", 8.0, 9.5, -18, 2),
        ("ZAF", "Sudáfrica", 24.5, -29.0, 48, -36),
        ("RUS", "Rusia", 95.0, 62.0, 118, 70),
        ("CHN", "China", 104.0, 35.0, 142, 39),
        ("IND", "India", 79.0, 22.0, 62, 7),
        ("NPL", "Nepal", 84.0, 28.3, 98, 12),
        ("MNG", "Mongolia", 103.0, 46.8, 134, 52),
        ("IDN", "Indonesia", 114.0, -2.0, 136, -7),
    ]
    for iso, lbl, x0, y0, x1, y1 in callouts:
        if iso not in delta_by_iso:
            continue
        dp = delta_by_iso[iso]
        sgn = "+" if dp >= 0 else "−"
        txt = f"{lbl} ({sgn}{abs(dp):.2f})"
        col_txt = "#7CE8D2" if dp >= 0.04 else ("#FF958A" if dp <= -0.04 else "#E2E8F0")
        ax.annotate(
            txt, xy=(x0, y0), xytext=(x1, y1),
            fontsize=7.3, fontweight="bold", color=col_txt,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0E162B", edgecolor=col_txt, lw=0.65, alpha=0.88),
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.75, alpha=0.85),
        )

    ax.set_xlim(-145, 165)
    ax.set_ylim(-54, 75)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Colorbar at bottom
    cax = fig.add_axes([0.22, 0.075, 0.56, 0.028])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
    cb.set_ticklabels(["−0,20\n(retroceso fuerte)", "−0,10", "−0,05", "0,00\n(estable)", "+0,05", "+0,10", "+0,20\n(avance fuerte)"])
    cb.ax.tick_params(labelsize=7.6, colors="#CBD5E1", length=3)
    cb.outline.set_edgecolor("#334155")

    # Summary counts badge
    n_down = sum(1 for s in stats_list if s["dp"] <= -0.04)
    n_flat = sum(1 for s in stats_list if -0.04 < s["dp"] < 0.04)
    n_up = sum(1 for s in stats_list if s["dp"] >= 0.04)
    fig.text(
        0.03, 0.945,
        "Tendencia democrática proyectada por país hacia 2041 (nuevo predictor democracy.fit en Phy-World L1)",
        fontsize=12.6, fontweight="bold", color="#FFFFFF",
    )
    fig.text(
        0.03, 0.908,
        f"Cambio esperado en calidad democrática Δ vdem_polyarchy = poly(2041) − poly(2019) · 80 corridas Monte Carlo · "
        f"Retroceso (≤−0,04): {n_down} países · Estable: {n_flat} · Avance (≥+0,04): {n_up} países · Gris oscuro = sin dato V-Dem",
        fontsize=8.2, color="#9EB0CF",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(OUT_MAP, dpi=220, facecolor="#0E162B")
    plt.close(fig)
    print(f"Mapamundi 2041 guardado -> {OUT_MAP}")
    return stats_list


def build_pdf(stats_list):
    top_down = sorted(stats_list, key=lambda s: s["dp"])[:10]
    top_up = sorted(stats_list, key=lambda s: -s["dp"])[:10]

    def tbl(rows, cls):
        trs = []
        for s in rows:
            sgn = "+" if s["dp"] >= 0 else "−"
            trs.append(
                f"<tr><td><strong>{s['name']}</strong> <span class='mono muted'>({s['iso']})</span></td>"
                f"<td class='num'>{s['p19']:.2f}</td><td class='num'>{s['p41']:.2f}</td>"
                f"<td class='num {cls}'>{sgn}{abs(s['dp']):.2f}</td>"
                f"<td class='num'>{s['d41']*100:.0f} %</td></tr>"
            )
        return "".join(trs)

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<style>
@page {{ size: A4 landscape; margin: 11mm 12mm; }}
body {{ font-family: 'IBM Plex Sans', 'DejaVu Sans', sans-serif; color: #0E162B; margin: 0; font-size: 8.2pt; }}
.page {{ page-break-after: always; break-after: page; }}
.page:last-child {{ page-break-after: auto; break-after: auto; }}
h1 {{ font-size: 16pt; margin: 0 0 2mm 0; color: #0E162B; }}
.sub {{ font-size: 8.5pt; color: #475569; margin: 0 0 3.5mm 0; }}
img {{ width: 100%; border-radius: 2.5mm; display: block; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; margin-top: 3.5mm; }}
table {{ width: 100%; border-collapse: collapse; font-size: 7.4pt; }}
th {{ text-align: left; font-size: 6.4pt; text-transform: uppercase; color: #64748B; border-top: 1pt solid #0E162B; border-bottom: 0.7pt solid #94A3B8; padding: 1.2mm 1.5mm; }}
th.num, td.num {{ text-align: right; font-family: monospace; }}
td {{ padding: 1.1mm 1.5mm; border-bottom: 0.45pt solid #E2E8F0; }}
.red {{ color: #B83227; font-weight: 600; }}
.teal {{ color: #0F766E; font-weight: 600; }}
.mono {{ font-family: monospace; }}
.muted {{ color: #64748B; }}
</style></head><body>
<div class="page">
  <h1>Nuevo predictor democrático V2 (<code>democracy.fit</code>): curvas 1950–2041 y recta de predicción ciega</h1>
  <p class="sub">Comparación entre el modelo heredado de L1 (azul), el nuevo predictor continuo+transición integrado en Phy-World L1 (naranja, banda p10–p90) y los datos reales de V-Dem (1950–2019 y seguimiento 2020–2025 fuera de muestra).</p>
  <img src="{OUT_CURVES.resolve().as_uri()}"/>
  <div class="grid-2">
    <div>
      <strong style="color:#B83227;">Top 10 países con mayor tendencia de retroceso (2019 → 2041)</strong>
      <table><thead><tr><th>País</th><th class="num">poly 19</th><th class="num">poly 41</th><th class="num">Δ poly</th><th class="num">P(demo 41)</th></tr></thead>
      <tbody>{tbl(top_down, 'red')}</tbody></table>
    </div>
    <div>
      <strong style="color:#0F766E;">Top 10 países con mayor tendencia de avance (2019 → 2041)</strong>
      <table><thead><tr><th>País</th><th class="num">poly 19</th><th class="num">poly 41</th><th class="num">Δ poly</th><th class="num">P(demo 41)</th></tr></thead>
      <tbody>{tbl(top_up, 'teal')}</tbody></table>
    </div>
  </div>
</div>
<div class="page">
  <h1>Mapamundi de tendencia democrática hacia 2041 (163 países · malla PRIO-GRID 0,5°)</h1>
  <p class="sub">Color por país = cambio medio proyectado en <code>vdem_polyarchy</code> entre 2019 y 2041 (80 corridas Monte Carlo acopladas a los 7 módulos de L1). Rojo = erosión/retroceso; verde azulado = avance democrático.</p>
  <img src="{OUT_MAP.resolve().as_uri()}"/>
</div>
</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    weasyprint.HTML(filename=str(OUT_HTML)).write_pdf(str(OUT_PDF))
    print(f"PDF democracia V2 generado -> {OUT_PDF}")


def main():
    W, o_old, o_v2, o_v2_04, obs_world_demo = run_simulations()
    plot_curves(W, o_old, o_v2, o_v2_04, obs_world_demo)
    stats = plot_world_map_2041(W, o_v2)
    build_pdf(stats)
    print("\nTop 5 retroceso 2041:", [(s["iso"], round(s["dp"], 3)) for s in sorted(stats, key=lambda x: x["dp"])[:5]])
    print("Top 5 avance 2041:   ", [(s["iso"], round(s["dp"], 3)) for s in sorted(stats, key=lambda x: -x["dp"])[:5]])


if __name__ == "__main__":
    main()
