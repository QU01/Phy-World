"""Regiones del L2. Una región es un conjunto de países con sus tres códigos (ISO3, Gleditsch-Ward, M49).

``mideast`` es la región «Middle East» de UCDP: los 15 Estados que el GED clasifica así. UCDP codifica en
«Israel» los eventos de Gaza y Cisjordania y PRIO-GRID (cshapes) asigna esas celdas al código 666: la unidad
``ISR`` del L2 es Israel MÁS los territorios palestinos [declarado]; su población y sus desplazados suman
``PSE``. No es la región ``MEA`` del L0 regional (Oriente Medio y Norte de África, otra lista de países).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    code: str
    name: str
    ucdp_region: str
    iso3: tuple
    gw: tuple                      # Gleditsch-Ward, el ``country_id`` del GED y el ``gwno`` de PRIO-GRID
    m49: tuple                     # código numérico de la ONU (WPP)
    extra_m49: dict                # iso3 -> códigos M49 que se suman a esa unidad
    extra_iso3: dict               # iso3 -> ISO3 que se suman a esa unidad (ACNUR)

    @property
    def K(self) -> int:
        return len(self.iso3)

    def index(self, iso3: str) -> int:
        return self.iso3.index(iso3.upper())


MIDEAST = Region(
    code="mideast", name="Middle East", ucdp_region="Middle East",
    iso3=("ARE", "BHR", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "QAT", "SAU", "SYR", "TUR", "YEM"),
    gw=(696, 692, 651, 630, 645, 666, 663, 690, 660, 698, 694, 670, 652, 640, 678),
    m49=(784, 48, 818, 364, 368, 376, 400, 414, 422, 512, 634, 682, 760, 792, 887),
    extra_m49={"ISR": (275,)}, extra_iso3={"ISR": ("PSE",)},
)

REGIONS = {MIDEAST.code: MIDEAST}


def get(code: str) -> Region:
    try:
        return REGIONS[code.lower()]
    except KeyError:
        raise KeyError(f"unknown L2 region {code!r}; available: {sorted(REGIONS)}") from None
