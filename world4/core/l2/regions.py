"""Regiones del L2. Una región es un conjunto de países con sus tres códigos (ISO3, Gleditsch-Ward, M49).

Son las cinco regiones de UCDP y entre todas cubren el mundo: ``africa``, ``americas``, ``asia`` (con
Oceanía), ``europe`` (con Rusia y el Cáucaso) y ``mideast``. ``world`` no es una región: es la lista de las
cinco, que se estiman y se simulan por separado (una díada vive en su región; lo que se pierde es el
acople de un actor con sus frentes de OTRA región, que afecta a unos pocos actores globales [declarado]).

**Qué país va en qué región.** Un país con algún evento en el GED va donde lo pone UCDP. Uno sin eventos
va por su continente (Oceanía a ``asia``), salvo los quince de ``mideast``, que son una lista fija: fue la
primera región, la que eligió la forma del modelo, y no debe cambiar si un día cambia el GED.

**Unidades que suman más de un territorio** (``EXTRA``): UCDP y PRIO-GRID (cshapes) codifican Gaza y
Cisjordania en Israel, el Sáhara Occidental en Marruecos y Hong Kong y Macao en China; la población de la
ONU y los desplazados de ACNUR de esos territorios se suman a la unidad. No es una posición sobre su
estatus: es la única forma de que eventos, celdas, población y desplazados hablen de lo mismo.

No es la región ``MEA`` del L0 regional (Oriente Medio y Norte de África, otra lista de países).
"""
from __future__ import annotations

from dataclasses import dataclass, field

WORLD = ("africa", "americas", "asia", "europe", "mideast")
UCDP_NAME = {"africa": "Africa", "americas": "Americas", "asia": "Asia", "europe": "Europe",
             "mideast": "Middle East"}
CONTINENT = {"Africa": "africa", "America": "americas", "Asia": "asia", "Oceania": "asia", "Europe": "europe"}
# códigos Gleditsch-Ward que el conversor no conoce o que PRIO-GRID y UCDP escriben distinto
GW_ISO3 = {345: "SRB", 678: "YEM", 816: "VNM", 316: "CZE"}
PRIO_ALIAS = {340: 345}                  # Serbia: cshapes usa 340 desde 2006, UCDP mantiene 345
EXTRA = {"ISR": ("PSE",), "MAR": ("ESH",), "CHN": ("HKG", "MAC")}
EXTRA_M49 = {"PSE": 275, "ESH": 732, "HKG": 344, "MAC": 446}


@dataclass(frozen=True)
class Region:
    code: str
    name: str
    ucdp_region: str
    iso3: tuple = ()
    gw: tuple = ()                 # Gleditsch-Ward, el ``country_id`` del GED y el ``gwno`` de PRIO-GRID
    m49: tuple = ()                # código numérico de la ONU (WPP)
    extra_m49: dict = field(default_factory=dict)      # iso3 -> códigos M49 que se suman a esa unidad
    extra_iso3: dict = field(default_factory=dict)     # iso3 -> ISO3 que se suman a esa unidad (ACNUR)

    @property
    def K(self) -> int:
        return len(self.iso3)

    def index(self, iso3: str) -> int:
        return self.iso3.index(iso3.upper())


_ME_ISO3 = ("ARE", "BHR", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "QAT", "SAU", "SYR", "TUR", "YEM")
MIDEAST = Region(
    code="mideast", name="Middle East", ucdp_region="Middle East", iso3=_ME_ISO3,
    gw=(696, 692, 651, 630, 645, 666, 663, 690, 660, 698, 694, 670, 652, 640, 678),
    m49=(784, 48, 818, 364, 368, 376, 400, 414, 422, 512, 634, 682, 760, 792, 887),
    extra_m49={"ISR": (275,)}, extra_iso3={"ISR": ("PSE",)},
)

REGIONS = {MIDEAST.code: MIDEAST}


def names(code: str) -> tuple:
    """Las regiones que hay detrás de un nombre: las cinco para ``world``, una para lo demás."""
    code = code.lower()
    if code == "world":
        return WORLD
    if code not in WORLD:
        raise KeyError(f"unknown L2 region {code!r}; available: {', '.join(WORLD)}, world")
    return (code,)


def get(code: str) -> Region:
    """La región con sus países. ``mideast`` es fija; las demás existen cuando ``resolve`` las ha construido
    con los datos (lo hace ``data.conflict`` al convertir y al cargar)."""
    code = code.lower()
    if code not in REGIONS:
        if code in WORLD:
            raise KeyError(f"L2 region {code!r} has no country list yet: python world4_cli.py --install-l2 "
                           f"--l2-region {code}")
        raise KeyError(f"unknown L2 region {code!r}; available: {', '.join(WORLD)}")
    return REGIONS[code]


def assign(gw_codes, ucdp_region_of: dict, table: dict) -> dict:
    """Región de cada código GW. ``ucdp_region_of``: GW -> nombre de región de UCDP para los países con
    eventos. ``table``: GW -> (iso3, m49, continente) del conversor de códigos. Los países de ``mideast`` son
    los de su lista fija; el resto, UCDP si tiene eventos y el continente si no."""
    by_ucdp = {v: k for k, v in UCDP_NAME.items()}
    out = {}
    for g in gw_codes:
        if g in MIDEAST.gw:
            out[g] = "mideast"
        elif g in ucdp_region_of and by_ucdp.get(ucdp_region_of[g], "mideast") != "mideast":
            out[g] = by_ucdp[ucdp_region_of[g]]
        elif g in table and table[g][2] in CONTINENT:
            out[g] = CONTINENT[table[g][2]]
    return out


def build(code: str, members: list) -> Region:
    """Registra la región ``code`` con ``members`` = [(iso3, gw, m49), ...] y la devuelve."""
    code = code.lower()
    if code == "mideast":
        return MIDEAST
    members = sorted(members)
    iso = tuple(m[0] for m in members)
    r = Region(code=code, name=UCDP_NAME[code], ucdp_region=UCDP_NAME[code], iso3=iso,
               gw=tuple(int(m[1]) for m in members), m49=tuple(int(m[2]) for m in members),
               extra_iso3={k: v for k, v in EXTRA.items() if k in iso},
               extra_m49={k: tuple(EXTRA_M49[e] for e in v) for k, v in EXTRA.items() if k in iso})
    REGIONS[code] = r
    return r
