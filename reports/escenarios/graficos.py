"""Figuras del estudio de escenarios (matplotlib -> PNG a 200 ppp), con la paleta y la tipografía del informe."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FONTS = Path(__file__).with_name("fonts")
for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

INK, INK2, INK3 = "#0f1c19", "#3f4c48", "#6b7773"
RULE, RULE2 = "#d9e0dd", "#e8edeb"
ACCENT, ACCENT_SOFT = "#0b6e55", "#d6ebe3"
NEUTRAL = "#8a9692"
DEST = ["#b3302b", "#eb8a7f", "#cbc5b8", "#78aee8", "#1c5cab"]  # divergente: gris neutro = contracción ordenada
DEST_SHORT = ["Colapso", "Crisis humanitaria", "Contracción ordenada", "Meseta en sobregiro", "Transición sostenible"]
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # secuencial azul

plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 8.2, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "axes.titlesize": 8.6, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 6, "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "axes.grid": True, "grid.color": RULE2, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "legend.frameon": False, "legend.fontsize": 7.2,
    "figure.dpi": 200, "savefig.dpi": 200, "lines.linewidth": 1.6, "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return path


def destiny_bars(ax, labels, probs, title=None):
    """Barras horizontales apiladas de la probabilidad de cada destino por nivel."""
    probs = np.asarray(probs)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for j in range(probs.shape[1]):
        w = probs[:, j]
        ax.barh(y, np.maximum(w - 0.006, 0), left=left + 0.003, color=DEST[j], height=0.62, linewidth=0)
        for yi, l0, wi in zip(y, left, w):
            if wi >= 0.12:
                ax.text(l0 + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=6.6,
                        color="white" if j in (0, 4) else INK)
        left += w
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, labelsize=7.4)
    if title:
        ax.set_title(title)


def destiny_legend(fig, y=-0.02):
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(DEST, DEST_SHORT)]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, y), handlelength=1, handleheight=1,
               columnspacing=1.6)


def band_line(ax, x, q, color, label=None, alpha=0.18, lw=1.8, marker="o"):
    q = np.asarray(q)
    ax.fill_between(x, q[:, 0], q[:, 2], color=color, alpha=alpha, linewidth=0)
    ax.plot(x, q[:, 1], color=color, lw=lw, label=label, marker=marker, ms=3.2, mec="white", mew=0.6)


def heat(ax, Z, xt, yt, cmap, vmin=0, vmax=1, fmt="{:.0%}", title=None, xlabel=None, ylabel=None, text_dark=0.55):
    Z = np.asarray(Z)
    im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.4,
                    color="white" if frac > text_dark else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return im


BLUES = LinearSegmentedColormap.from_list("w4blue", ["#f3f6fa"] + RAMP[1:])
REDS = LinearSegmentedColormap.from_list("w4red", ["#fbf3f2", "#f2b8b1", "#e07a6f", "#c4473d", "#8f2420"])
GREENS = LinearSegmentedColormap.from_list("w4green", ["#f2f7f5", "#bfe0d3", "#6fb89b", "#2b8a68", "#0b5a45"])
