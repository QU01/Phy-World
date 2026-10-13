"""Qué se sabía y cuándo: ``known_time`` por fuente, para que ningún backtest use el futuro.

Regla (supuesto [S] por fuente, en ``sources.json``): el dato del año ``y`` de una fuente con
retraso de publicación ``lag`` se considera conocido el 1 de julio de ``y + lag``. Las fuentes
con ``vintage`` fechado y años de PROYECCIÓN (WPP 2024+) se conocen todas en la fecha de la
edición, sea cual sea el año al que se refieren.

En F0 esto sólo se MIDE: el Atlas portado no lo aplica (W2 exige no cambiar su
comportamiento). ``leakage_report`` cuantifica cuánto adelanta el Atlas a cada fuente en su
validación de origen móvil; F0.5 lo aplicará en el puente con Prime Radiant y F1 en el L1.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

SOURCES_FILE = Path(__file__).with_name("sources.json")
PROJECTION_EDITIONS = {"wpp": (2024, date(2024, 7, 11))}   # fuente: (primer año proyectado, fecha)


@lru_cache(maxsize=1)
def registry() -> dict:
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def sources() -> dict:
    return registry()["sources"]


def known_time(source: str, year: int) -> date:
    s = sources()[source]
    if source in PROJECTION_EDITIONS and year >= PROJECTION_EDITIONS[source][0]:
        return PROJECTION_EDITIONS[source][1]
    return date(int(year) + int(s["lag_years"]), 7, 1)


def latest_year(source: str, as_of: date) -> int:
    """Último año de dato observado (no proyección) de ``source`` publicado en ``as_of``."""
    lag = int(sources()[source]["lag_years"])
    y = as_of.year - lag
    return y if date(y + lag, 7, 1) <= as_of else y - 1


def usable(source: str, year: int, as_of: date) -> bool:
    return known_time(source, year) <= as_of


def leakage_report(origin_years=range(1970, 2019)) -> dict:
    """Para cada fuente: cuántos años de dato usa el Atlas en un origen ``t0`` que aún no estaban
    publicados el 31 de diciembre de ``t0`` (el Atlas asume dato del año t0 disponible en t0)."""
    out = {}
    for sid, s in sources().items():
        ahead = [max(0, t0 - latest_year(sid, date(t0, 12, 31))) for t0 in origin_years]
        out[sid] = {"lag_years": s["lag_years"], "max_years_ahead": max(ahead),
                    "mark": s["mark"], "used_in": s.get("used_in", [])}
    return out


def license_table() -> list[dict]:
    rows = []
    for sid, s in sources().items():
        red = s.get("redistributable")
        rows.append({"id": sid, "name": s["name"], "license": s.get("license") or "?",
                     "mark": s["mark"], "redistributable": "yes" if red else ("NO" if red is False else "?"),
                     "lag_years": s["lag_years"], "used_in": ",".join(s.get("used_in", [])),
                     "note": s.get("note", "")})
    return rows
