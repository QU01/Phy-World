# -*- coding: utf-8 -*-
"""
QUASAR Phy-World / cli_style.py
============================
Terminal presentation, no dependencies. This is the QUASAR family style
module, transplanted UNCHANGED from Phy-CT: the same procedural quasar
logo, the same 256-colour palette, the same wordmark and the same panels
as Phy-CC, Phy-CB and Phy-Wing. Only the tool name and the tagline
change, which is exactly what the module was written to allow - a family
that looks like four different programs is four programs.

Two rules that are NOT cosmetic:

  1. **Colour is optional and turns itself off.** If the output is not a
     TTY, or `NO_COLOR` is set, or the user passed `--no-color`, not a
     single ANSI code is emitted. A CI log full of `\\x1b[36m` cannot be
     read.
  2. **With colour off, everything printed is ASCII.** The Windows
     console is cp1252 and a half-block glyph in a `print` breaks CI with
     a `UnicodeEncodeError`, not with physics. The block glyphs and the
     arrows only appear on the colour path, which is a TTY by
     construction. Accents live in the HTML report, which is UTF-8.
"""

from __future__ import annotations

import math
import os
import shutil
import sys

_COLOR = True

#: Shown in the banner, next to the tool name.
VERSION = "v0.1"


def init(no_color: bool = False) -> None:
    """`no_color=True` forces plain text. Otherwise colour is used only
    if the stream is a TTY and `NO_COLOR` is not set."""
    global _COLOR
    _COLOR = not (no_color or os.environ.get("NO_COLOR") or
                  not sys.stdout.isatty())
    if _COLOR and os.name == "nt":
        _enable_windows_vt()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                               # noqa: BLE001
            pass


def _enable_windows_vt() -> None:
    """Turns on ANSI processing in the Windows console. Without this the
    escape codes are printed literally, which is worse than no colour."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:                                   # noqa: BLE001
        pass


def enabled() -> bool:
    return _COLOR


def width(default: int = 74) -> int:
    try:
        return max(40, min(shutil.get_terminal_size().columns, 100))
    except Exception:                                   # noqa: BLE001
        return default


# ==========================================================================
# Colour
# ==========================================================================
_RESET = "\x1b[0m"

#: "quasar" palette (256 colours): from the cyan of the outer ring to the
#: violet of the inner edge, white core.
PALETTE = {
    "brand":  "38;5;51",    # bright cyan (ring)
    "brand2": "38;5;45",    # cyan-blue
    "brand3": "38;5;39",    # sky blue
    "brand4": "38;5;99",    # blue-violet
    "accent": "38;5;141",   # violet
    "core":   "38;5;231",   # white (quasar core)
    "ok":     "38;5;42",    # green
    "warn":   "38;5;214",   # amber
    "err":    "38;5;203",   # coral red
    "dim":    "38;5;245",   # grey
    "key":    "38;5;80",    # muted cyan
    "val":    "38;5;231",   # bright white
}

#: Angular gradient of the ring (direction of rotation).
_RING_SHADES = ["brand", "brand2", "brand3", "brand4", "accent"]


def c(text, style: str = "", *, bold: bool = False) -> str:
    """Wraps `text` in a palette style (or a raw ANSI code)."""
    if not _COLOR:
        return str(text)
    code = PALETTE.get(style, style)
    parts = []
    if bold:
        parts.append("1")
    if code:
        parts.append(code)
    if not parts:
        return str(text)
    return f"\x1b[{';'.join(parts)}m{text}{_RESET}"


#: Kept because the rest of Phy-CT and its test suite use them by name.
def bold(t):    return c(t, "val", bold=True)
def dim(t):     return c(t, "dim")
def cyan(t):    return c(t, "brand")
def blue(t):    return c(t, "brand3")
def green(t):   return c(t, "ok")
def yellow(t):  return c(t, "warn")
def red(t):     return c(t, "err")
def magenta(t): return c(t, "accent")


# ==========================================================================
# Logo
# ==========================================================================
def _quasar_raster(n: int = 22) -> list:
    """n x n raster of the logo: 0 empty, 1..len(_RING_SHADES) ring
    (angular gradient index), 9 core.

    Ring between 64 % and 100 % of the radius, core at 34 %, and the cut
    of the logo: a straight band through the centre (direction ~62 deg)
    erases the ring in two opposite segments -- the core is untouched.
    3x3 supersampling per pixel to smooth the circle on small rasters.
    """
    cx = cy = (n - 1) / 2.0
    R = n / 2.0 - 0.3
    ring_in, core_r = 0.64 * R, 0.34 * R
    th = math.radians(62.0)
    ux, uy = math.cos(th), math.sin(th)
    band_hw = 0.085 * n

    sub = (-1.0 / 3, 0.0, 1.0 / 3)
    px = [[0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            hits_core = hits_ring = 0
            for oy in sub:
                for ox in sub:
                    dx, dy = x + ox - cx, cy - (y + oy)   # y upwards
                    r = math.hypot(dx, dy)
                    if r <= core_r:
                        hits_core += 1
                    elif ring_in <= r <= R and \
                            abs(dx * uy - dy * ux) >= band_hw:
                        hits_ring += 1
            if hits_core >= 5:
                px[y][x] = 9
            elif hits_ring >= 5:
                dx, dy = x - cx, cy - y
                ang = math.atan2(dy, dx) % (2 * math.pi)
                shade = int(ang / (2 * math.pi) * len(_RING_SHADES))
                px[y][x] = 1 + min(shade, len(_RING_SHADES) - 1)
    return px


def _quasar_logo(n: int = 22) -> list:
    """Logo as terminal lines. With colour: half-blocks (two vertical
    pixels per cell, so the circle comes out round) with the gradient;
    without colour: flat ASCII ('#' ring, '@' core)."""
    px = _quasar_raster(n)
    if not _COLOR:
        return ["".join("@" if v == 9 else "#" if v else " " for v in px[y])
                for y in range(0, n, 2)]

    def paint(v: int, ch: str) -> str:
        style = "core" if v == 9 else _RING_SHADES[v - 1]
        return c(ch, style, bold=(v == 9))

    out = []
    for j in range(0, n, 2):
        top = px[j]
        bot = px[j + 1] if j + 1 < n else [0] * n
        line = []
        for x in range(n):
            t, b = top[x], bot[x]
            if t and b:
                line.append(paint(t, "█"))
            elif t:
                line.append(paint(t, "▀"))
            elif b:
                line.append(paint(b, "▄"))
            else:
                line.append(" ")
        out.append("".join(line))
    return out


# ==========================================================================
# Banner
# ==========================================================================
#: QUASAR wordmark in half-blocks (3 rows, 4-column glyphs).
_WORDMARK = [
    "▄▀▀▄ █  █ ▄▀▀▄ "
    "▄▀▀▀ ▄▀▀▄ █▀▀▄",
    "█  █ █  █ █▀▀█ "
    "▀▀▀▄ █▀▀█ █▄▄▀",
    "▀▄▄▚ ▀▄▄▀ █  █ "
    "▄▄▄▀ █  █ █ ▀▄",
]


def banner(n: int = 26) -> str:
    """Banner: procedural logo on the left, wordmark and tagline on the
    right, vertically aligned."""
    logo = _quasar_logo(n)
    if _COLOR:
        right = [
            "",
            "",
            c(_WORDMARK[0], "val", bold=True),
            c(_WORDMARK[1], "val", bold=True),
            c(_WORDMARK[2], "val", bold=True),
            "",
            c("P h y - W o r l d", "brand", bold=True)
            + c(f"   {VERSION}", "dim"),
            "",
            c("integrated world-system model: biophysical limits x social structure", "dim"),
            c("World3-03 core", "key") + c(" + ", "dim")
            + c("180 country agents", "key") + c(" → ", "dim")
            + c("projections with declared limits", "key"),
        ]
    else:
        right = [
            "",
            "QUASAR  Phy-World  " + VERSION,
            "integrated world-system model: biophysical limits x social structure",
            "World3-03 core + 180 country agents -> projections with declared limits",
        ]
    lines = []
    n_rows = max(len(logo), len(right))
    for i in range(n_rows):
        left = logo[i] if i < len(logo) else " " * n
        pad = n - _vlen(left)
        r = right[i] if i < len(right) else ""
        lines.append("  " + left + " " * max(0, pad) + "   " + r)
    return "\n".join(lines)


def logo_mark() -> str:
    """Minimal logo glyph for inline use (ring + core)."""
    return c("◉", "brand", bold=True) if _COLOR else "(o)"


# ==========================================================================
# Layout
# ==========================================================================
def _vlen(s: str) -> int:
    """Visible length: discounts ANSI sequences."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
        else:
            out += 1
            i += 1
    return out


W = 74


def rule(ch: str = "=") -> None:
    """PRINTS a rule. Phy-CT calls it with a fill character, not with a
    title -- the family signature `rule(title, style)` is `head()` here,
    because changing this one would silently reformat every section."""
    print(c(ch * width(W), "dim"))


def head(title: str = "", style: str = "brand") -> str:
    """Family rule with a title: returns a string, does not print."""
    w = width(W)
    if not title:
        return c("─" * w if _COLOR else "-" * w, "dim")
    lead = "─◉─" if _COLOR else "---"
    label = f" {title} "
    fill = max(0, w - len(lead) - len(label))
    return (c(lead, style, bold=True) + c(label, style, bold=True)
            + c(("─" if _COLOR else "-") * fill, "dim"))


def title(t: str) -> None:
    print()
    print(head(t))


def section(t: str) -> None:
    print()
    print(head(t, "brand2"))


def kv(k: str, v, unit: str = "", note: str = "", width: int = 26) -> None:
    line = "  " + c(f"{k:<{width}}", "key") + " " + str(v)
    if unit:
        line += " " + c(unit, "dim")
    if note:
        line += c(f"   {note}", "dim")
    print(line)


def panel(title_: str, rows: list, style: str = "brand") -> str:
    """Rounded box with key/value pairs and the logo glyph in the title."""
    w = min(width(W), 74)
    inner = w - 2
    klen = max((len(k) for k, _ in rows), default=0)
    tl, tr, bl, br = ("╭", "╮", "╰", "╯") if _COLOR \
        else ("+", "+", "+", "+")
    hz, vt = ("─", "│") if _COLOR else ("-", "|")
    mk = "◉ " if _COLOR else "(o) "
    out = [c(tl + hz * inner + tr, style)]
    pad = inner - _vlen(f" {mk}{title_} ")
    out.append(c(vt, style) + c(f" {mk}", style, bold=True)
               + c(f"{title_} ", "val", bold=True)
               + " " * max(0, pad) + c(vt, style))
    out.append(c(("├" if _COLOR else "+") + hz * inner
                 + ("┤" if _COLOR else "+"), style))
    for k, v in rows:
        visible = len(f" {k.ljust(klen)}  ") + _vlen(str(v))
        pad = max(0, inner - visible)
        out.append(c(vt, style) + c(f" {k.ljust(klen)}  ", "key")
                   + str(v) + " " * pad + c(vt, style))
    out.append(c(bl + hz * inner + br, style))
    return "\n".join(out)


# ==========================================================================
# States
# ==========================================================================
def status(ok_: bool, yes: str = "OK", no: str = "NO") -> str:
    return c(yes, "ok", bold=True) if ok_ else c(no, "err", bold=True)


def mark(m: str) -> str:
    """Traceability mark, coloured. `[?]` in red on purpose: it is the
    one that forbids using the number as an anchor."""
    return {"V": c("[V]", "ok"), "S": c("[S]", "brand3"),
            "M": c("[M]", "warn")}.get(m, c("[?]", "err"))


def bar(frac: float, width: int = 28, ch: str = "#") -> str:
    """Loss bar. With colour it uses the ring gradient, like the family
    progress bar; without colour, plain ASCII."""
    n = max(0, min(width, int(round(frac * width))))
    if not _COLOR:
        return ch * n + "." * (width - n)
    segs = [c("█", _RING_SHADES[min(
        int(i / max(width - 1, 1) * len(_RING_SHADES)),
        len(_RING_SHADES) - 1)]) for i in range(n)]
    return "".join(segs) + c("░" * (width - n), "dim")


def progress_bar(frac: float, label: str = "", barw: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * barw))
    if not _COLOR:
        return ("[" + "#" * fill + "-" * (barw - fill)
                + f"] {int(frac * 100):3d}%  " + label)
    segs = [c("█", _RING_SHADES[min(
        int(i / max(barw - 1, 1) * len(_RING_SHADES)),
        len(_RING_SHADES) - 1)]) for i in range(fill)]
    headg = c("◉", "core", bold=True) if 0 < fill < barw else ""
    rest = max(0, barw - fill - (1 if headg else 0))
    return (c("▕", "dim") + "".join(segs) + headg
            + c("░" * rest, "dim") + c("▏", "dim")
            + c(f" {int(frac * 100):3d}%  ", "val") + label)


def warn(t: str) -> None:
    print("  " + (c("⚠ ", "warn", bold=True) if _COLOR else "/!\\ ")
          + c(t, "warn"))


def warn_text(t: str) -> str:
    """`warn` PRINTS; this one returns, for use inside a composed line."""
    return c(t, "warn")


def err(t: str) -> None:
    print("  " + (c("✗ ", "err", bold=True) if _COLOR else "[X] ")
          + c(t, "err"))


def ok(t: str) -> None:
    print("  " + (c("✓ ", "ok", bold=True) if _COLOR else "[OK] ")
          + t)


def step(t: str) -> None:
    print("  " + (c("▸ ", "brand") if _COLOR else "> ") + t)


#: Everything degraded in this run, in order. `--strict` turns it into a
#: failure at the end; without `--strict` it is only announced.
DEGRADATIONS: list = []


def degraded(what: str, why: str) -> None:
    """ANNOUNCED degradation (CONVENTIONS 7): what cannot be done is
    said, with the reason. Never silently.

    It is also RECORDED. Announcing each degradation separately is fine
    for a human watching the screen, but a CI needs one single question
    at the end -- "did this run in full?" -- and `--strict` asks it.
    """
    DEGRADATIONS.append((what, why))
    print("  " + c("[degraded] ", "warn", bold=True)
          + c(f"{what}: {why}", "warn"))


def strict_verdict() -> list:
    """What was degraded, for `--strict` to decide. Empty = all ran."""
    return list(DEGRADATIONS)


def reset_degradations() -> None:
    DEGRADATIONS.clear()
