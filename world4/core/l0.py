"""World4 L0 — núcleo agregado. En F0 es World3-03 reimplementado desde sus ecuaciones.

Fuente de las ecuaciones: la transcripción de World3-03 en sintaxis DYNAMO de pydynamo
(``src/pydynamo/world3/code_pydynamo_w3.py``, dominio público / Unlicense), contrastada con
la salida del modelo Vensim World3-03 que publica WorldDynamics.jl (``test/world3_03.jl``,
MIT). Nada se copia de pyworld3 (CeCILL 2.1, y además es la versión de 1974).

Semántica (la de Vensim, no la de DYNAMO): en cada instante t los auxiliares se calculan con
los stocks de t; después se integra por Euler, stock(t+dt) = stock(t) + dt·flujo(t). Los
retardos son stocks también: SMOOTH (1.er orden), SMOOTH3 = DLINF3 (3 etapas de información)
y DELAY3 (3 etapas de material). Las tablas son interpolación lineal con saturación en los
extremos (TABHL), que es exactamente ``np.interp``.

Todo está vectorizado sobre un eje de regiones de tamaño ``n``: con ``n=1`` es el World3-03
global; F6 lo usará con 10 regiones. Ninguna ecuación mezcla regiones todavía.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Constantes (scenario 1, "business as usual" de Limits to Growth: The 30-Year Update) ──
CONSTANTS: dict[str, float] = {
    # población
    "p1i": 65e7, "p2i": 70e7, "p3i": 19e7, "p4i": 6e7,
    "len": 28.0, "sfpc": 230.0, "hsid": 20.0, "rlt": 30.0, "pet": 4000.0,
    "mtfn": 12.0, "lpd": 20.0, "zpgt": 4000.0, "dcfszpg": 2.0, "dcfsn": 3.8, "sad": 20.0, "ieat": 3.0,
    "fcest": 4000.0, "lmhs_year": 1940.0,
    # capital
    "ici": 2.1e11, "icor1": 3.0, "alic1": 14.0, "alic2": 14.0, "iet": 4000.0,
    "fioac1": 0.43, "fioac2": 0.43, "iopcd": 400.0,
    "sci": 1.44e11, "alsc1": 20.0, "alsc2": 20.0, "scor1": 1.0, "scor2": 1.0,
    "lfpf": 0.75, "lufdt": 2.0,
    # agricultura
    "palt": 3.2e9, "ali": 0.9e9, "pali": 2.3e9, "lfh": 0.7, "pl": 0.1,
    "aii": 5e9, "alai1": 2.0, "alai2": 2.0, "lyf1": 1.0, "lyti": 1.0, "drf": 2.0,
    "appyear": 4000.0, "io70": 7.9e11, "sd": 0.07, "alln": 1000.0, "llmytm": 4000.0,
    "uildt": 10.0, "uili": 8.2e6, "lferti": 600.0, "ilf": 600.0, "fspd": 2.0, "pfri": 1.0,
    # recursos no renovables
    "nri": 1e12, "nruf1": 1.0, "rcti": 1.0, "drur": 4.8e9, "fcaortm": 4000.0,
    # contaminación persistente
    "ppgf1": 1.0, "tdd": 20.0, "ppti": 1.0, "dppolx": 1.2, "frpm": 0.02, "imef": 0.1,
    "imti": 10.0, "fipm": 0.001, "amti": 1.0, "pptd": 20.0, "ppoli": 2.5e7,
    "ppol70": 1.36e8, "ahl70": 1.5,
    # indicadores de World3-03
    "pof": 0.22, "rhgdp": 9508.0, "rlgdp": 24.0, "tl": 1.91, "hpgha": 1e9, "hpup": 4.0,
    # control
    "pyear": 1975.0,
}

# ── Tablas: (x_bajo, x_alto, paso, valores) ──
TABLES: dict[str, tuple[float, float, float, list[float]]] = {
    "m1": (20, 80, 10, [0.0567, 0.0366, 0.0243, 0.0155, 0.0082, 0.0023, 0.001]),
    "m2": (20, 80, 10, [0.0266, 0.0171, 0.0110, 0.0065, 0.0040, 0.0016, 0.0008]),
    "m3": (20, 80, 10, [0.0562, 0.0373, 0.0252, 0.0171, 0.0118, 0.0083, 0.006]),
    "m4": (20, 80, 10, [0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04]),
    "lmf": (0, 5, 1, [0, 1, 1.43, 1.5, 1.5, 1.5]),
    "hsapc": (0, 2000, 250, [0, 20, 50, 95, 140, 175, 200, 220, 230]),
    "lmhs1": (0, 100, 20, [1, 1.1, 1.4, 1.6, 1.7, 1.8]),
    "lmhs2": (0, 100, 20, [1, 1.5, 1.9, 2, 2, 2]),
    "fpu": (0, 16e9, 2e9, [0, 0.2, 0.4, 0.5, 0.58, 0.65, 0.72, 0.78, 0.80]),
    "cmi": (0, 1600, 200, [0.5, 0.05, -0.1, -0.08, -0.02, 0.05, 0.1, 0.15, 0.2]),
    "lmp": (0, 100, 10, [1, 0.99, 0.97, 0.95, 0.90, 0.85, 0.75, 0.65, 0.55, 0.40, 0.20]),
    "fm": (0, 80, 10, [0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.79, 0.84, 0.87]),
    "cmple": (0, 80, 10, [3, 2.1, 1.6, 1.4, 1.3, 1.2, 1.1, 1.05, 1]),
    "sfsn": (0, 800, 200, [1.25, 0.94, 0.715, 0.59, 0.5]),
    "frsn": (-0.2, 0.2, 0.1, [0.5, 0.6, 0.7, 0.85, 1]),
    "fce": (0, 3, 0.5, [0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 1]),
    "fsafc": (0, 10, 2, [0, 0.005, 0.015, 0.025, 0.03, 0.035]),
    "icormrrct": (0, 1, 0.1, [3.75, 3.6, 3.47, 3.36, 3.25, 3.16, 3.1, 3.06, 3.02, 3.01, 3]),
    "icormlyt": (1, 2, 0.2, [1, 1.05, 1.12, 1.25, 1.35, 1.5]),
    "icormpt": (0, 1, 0.1, [1.25, 1.2, 1.15, 1.11, 1.08, 1.05, 1.03, 1.02, 1.01, 1, 1]),
    "fioacv": (0, 2, 0.2, [0.3, 0.32, 0.34, 0.36, 0.38, 0.43, 0.73, 0.77, 0.81, 0.82, 0.83]),
    "isopc1": (0, 1600, 200, [40, 300, 640, 1000, 1220, 1450, 1650, 1800, 2000]),
    "isopc2": (0, 1600, 200, [40, 300, 640, 1000, 1220, 1450, 1650, 1800, 2000]),
    "fioas1": (0, 2, 0.5, [0.3, 0.2, 0.1, 0.05, 0]),
    "fioas2": (0, 2, 0.5, [0.3, 0.2, 0.1, 0.05, 0]),
    "jpicu": (50, 800, 150, [0.37, 0.18, 0.12, 0.09, 0.07, 0.06]),
    "jpscu": (50, 800, 150, [1.1, 0.6, 0.35, 0.2, 0.15, 0.15]),
    "jph": (2, 30, 4, [2, 0.5, 0.4, 0.3, 0.27, 0.24, 0.2, 0.2]),
    "cuf": (1, 11, 2, [1, 0.9, 0.7, 0.3, 0.1, 0.1]),
    "ifpc1": (0, 1600, 200, [230, 480, 690, 850, 970, 1070, 1150, 1210, 1250]),
    "ifpc2": (0, 1600, 200, [230, 480, 690, 850, 970, 1070, 1150, 1210, 1250]),
    "fioaa1": (0, 2.5, 0.5, [0.4, 0.2, 0.1, 0.025, 0, 0]),
    "fioaa2": (0, 2.5, 0.5, [0.4, 0.2, 0.1, 0.025, 0, 0]),
    "dcph": (0, 1, 0.1, [1e5, 7400, 5200, 3500, 2400, 1500, 750, 300, 150, 75, 50]),
    "lymc": (0, 1000, 40, [1, 3, 4.5, 5.0, 5.3, 5.6, 5.9, 6.1, 6.35, 6.6, 6.9, 7.2, 7.4, 7.6, 7.8,
                           8, 8.2, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6, 9.8, 10]),
    "lytcrm": (0, 1, 1, [0, 0]),
    "lymap1": (0, 30, 10, [1, 1, 0.7, 0.4]),
    "lymap2": (0, 30, 10, [1, 1, 0.98, 0.95]),
    "fiald": (0, 2, 0.25, [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1]),
    "mlymc": (0, 600, 40, [0.075, 0.03, 0.015, 0.011, 0.009, 0.008, 0.007, 0.006, 0.005, 0.005,
                           0.005, 0.005, 0.005, 0.005, 0.005, 0.005]),
    "llmy1": (0, 9, 1, [1.2, 1, 0.63, 0.36, 0.16, 0.055, 0.04, 0.025, 0.015, 0.01]),
    "llmy2": (0, 9, 1, [1.2, 1, 0.63, 0.36, 0.29, 0.26, 0.24, 0.22, 0.21, 0.2]),
    "uilpc": (0, 1600, 200, [0.005, 0.008, 0.015, 0.025, 0.04, 0.055, 0.07, 0.08, 0.09]),
    "lfdr": (0, 30, 10, [0, 0.1, 0.3, 0.5]),
    "lfrt": (0, 0.10, 0.02, [20.0, 13.0, 8.0, 4.0, 2.0, 2.0]),
    "falm": (0, 4, 1, [0.0, 0.04, 0.07, 0.09, 0.1]),
    "rtcm": (-1, 0, 1, [0, 0]),
    "pcrum": (0, 1600, 200, [0, 0.85, 2.6, 3.4, 3.8, 4.1, 4.4, 4.7, 5]),
    "fcaor1": (0, 1, 0.1, [1.0, 0.9, 0.7, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "fcaor2": (0, 1, 0.1, [1.0, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "pptcm": (-1, 0, 1, [0, 0]),
    "ahlm": (1, 1001, 250, [1.0, 11.0, 21.0, 31.0, 41.0]),
    "ei": (0, 7000, 1000, [0, 0.81, 0.88, 0.92, 0.95, 0.98, 0.99, 1]),
    "gdppc": (0, 1000, 200, [120, 600, 1200, 1800, 2500, 3200]),
    "lei": (25, 85, 10, [0, 0.16, 0.33, 0.5, 0.67, 0.84, 1]),
}

# Escenarios 2-10 del libro de 2004 (mismos cambios que ``scenarios_limits_to_growth.py`` de
# pydynamo). En F0 sólo el escenario 1 tiene ancla Vensim; el resto se publica sin candado.
_TECH = {"fcaortm": 2002.0, "nri": 2e12}
_TECH_T = {"fcaor2": (0, 1, 0.1, [1.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])}
SCENARIOS: dict[int, tuple[dict, dict]] = {
    1: ({}, {}),
    2: (dict(_TECH), dict(_TECH_T)),
    3: ({**_TECH, "pyear": 2002.0}, {**_TECH_T, "pptcm": (-1, 0, 1, [-0.04, 0])}),
    4: ({**_TECH, "pyear": 2002.0}, {**_TECH_T, "pptcm": (-1, 0, 1, [-0.04, 0]),
                                      "lytcrm": (0, 1, 1, [0, 0.04])}),
    5: ({**_TECH, "pyear": 2002.0, "llmytm": 2002.0},
        {**_TECH_T, "pptcm": (-1, 0, 1, [-0.04, 0]), "lytcrm": (0, 1, 1, [0, 0.04])}),
}

STOCKS = ("p1", "p2", "p3", "p4", "ic", "sc", "al", "pal", "uil", "lfert", "nr", "ppol",
          "lyt", "rct", "ppt")


def _grid(name: str, spec) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, step, ys = spec
    xs = np.arange(len(ys)) * step + lo
    if not np.isclose(xs[-1], hi):
        raise ValueError(f"tabla {name}: {len(ys)} valores no cubren [{lo}, {hi}] con paso {step}")
    return xs, np.asarray(ys, float)


@dataclass
class _Smooth:
    """SMOOTH / SMOOTHI / SMOOTH3 / DELAY3 como stocks explícitos (orden 1 o 3)."""

    order: int
    material: bool = False
    s: np.ndarray | None = None

    def init(self, x, delay):
        x = np.asarray(x, float)
        # información: cada etapa vale la entrada; material: cada etapa guarda x·D/orden
        self.s = np.repeat((x * delay / self.order if self.material else x)[None], self.order, 0)

    def out(self, delay):
        return self.s[-1] * self.order / delay if self.material else self.s[-1]

    def deriv(self, x, delay):
        k = self.order / delay
        if self.material:
            flows = self.s * k                       # salida de cada etapa
            inflow = np.concatenate([np.asarray(x, float)[None], flows[:-1]])
            return inflow - flows
        prev = np.concatenate([np.asarray(x, float)[None], self.s[:-1]])
        return (prev - self.s) * k


@dataclass
class World3Run:
    time: np.ndarray
    series: dict[str, np.ndarray] = field(default_factory=dict)   # nombre -> (T, n)

    def __getitem__(self, k):
        return self.series[k]


# Variables que se registran por defecto (stocks + las cinco del ancla Vensim + indicadores)
RECORD = STOCKS + ("pop", "f", "fpc", "io", "iopc", "so", "sopc", "ppolx", "le", "cbr", "cdr",
                   "nrfr", "fcaor", "ly", "aiph", "hwi", "hef", "luf", "cuf", "tf", "ppgr")


def run(*, scenario: int = 1, constants: dict | None = None, tables: dict | None = None,
        t0: float = 1900.0, t1: float = 2100.0, dt: float = 0.5, n: int = 1,
        record: tuple[str, ...] = RECORD) -> World3Run:
    """Corre World3-03 de ``t0`` a ``t1`` (ambos incluidos) con paso ``dt``.

    ``constants`` y ``tables`` sustituyen a las del escenario; cada constante puede ser un
    escalar o un array de forma (n,) para regiones o miembros de un ensemble.
    """
    sc_c, sc_t = SCENARIOS[scenario]
    C = {**CONSTANTS, **sc_c, **(constants or {})}
    T = {**TABLES, **sc_t, **(tables or {})}
    c = {k: np.broadcast_to(np.asarray(v, float), (n,)) for k, v in C.items()}
    grids = {name: _grid(name, spec) for name, spec in T.items()}
    tab = lambda name, x: np.interp(x, *grids[name])  # noqa: E731

    steps = int(round((t1 - t0) / dt))
    times = t0 + dt * np.arange(steps + 1)
    out = {k: np.empty((steps + 1, n)) for k in record}

    st = {
        "p1": c["p1i"].copy(), "p2": c["p2i"].copy(), "p3": c["p3i"].copy(), "p4": c["p4i"].copy(),
        "ic": c["ici"].copy(), "sc": c["sci"].copy(), "al": c["ali"].copy(), "pal": c["pali"].copy(),
        "uil": c["uili"].copy(), "lfert": c["lferti"].copy(), "nr": c["nri"].copy(),
        "ppol": c["ppoli"].copy(), "lyt": c["lyti"].copy(), "rct": c["rcti"].copy(),
        "ppt": c["ppti"].copy(),
    }
    sm = {
        "ehspc": _Smooth(1), "ple": _Smooth(3), "diopc": _Smooth(3), "aiopc": _Smooth(1),
        "fcfpc": _Smooth(3), "lufd": _Smooth(1), "ai": _Smooth(1), "pfr": _Smooth(1),
        "lyf2": _Smooth(1), "nruf2": _Smooth(1), "ppgf2": _Smooth(1),
        "ppapr": _Smooth(3, material=True),
    }
    # Inicializaciones que no dependen de auxiliares (SMOOTHI con valor explícito)
    sm["lufd"].init(np.ones(n), c["lufdt"])          # cuf(t0) = 1
    sm["ai"].init(c["aii"], c["alai1"])
    sm["pfr"].init(c["pfri"], c["fspd"])
    sm["lyf2"].init(st["lyt"], c["tdd"])
    sm["nruf2"].init(st["rct"], c["tdd"])
    sm["ppgf2"].init(st["ppt"], c["tdd"])

    def after(year_key, t):
        """IF THEN ELSE(Time >= año, valor_2, valor_1) de Vensim."""
        return t >= c[year_key]

    for k, t in enumerate(times):
        first = k == 0
        a = {}                                       # auxiliares de este instante
        pop = st["p1"] + st["p2"] + st["p3"] + st["p4"]
        a["pop"] = pop
        ppolx = st["ppol"] / c["ppol70"]
        a["ppolx"] = ppolx
        nrfr = st["nr"] / c["nri"]
        a["nrfr"] = nrfr
        fcaor = np.where(after("fcaortm", t), tab("fcaor2", nrfr), tab("fcaor1", nrfr))
        a["fcaor"] = fcaor

        # tecnología (stocks lyt, rct, ppt y sus suavizados)
        lymt = np.where(after("pyear", t), sm["lyf2"].out(c["tdd"]), c["lyf1"])
        nruf = np.where(after("pyear", t), sm["nruf2"].out(c["tdd"]), c["nruf1"])
        ppgf = np.where(after("pyear", t), sm["ppgf2"].out(c["tdd"]), c["ppgf1"])
        icor2 = (tab("icormrrct", nruf) * tab("icormlyt", lymt) * tab("icormpt", ppgf))
        icor = np.where(after("pyear", t), icor2, c["icor1"])
        alic = np.where(after("pyear", t), c["alic2"], c["alic1"])
        alsc = np.where(after("pyear", t), c["alsc2"], c["alsc1"])
        scor = np.where(after("pyear", t), c["scor2"], c["scor1"])
        alai = np.where(after("pyear", t), c["alai2"], c["alai1"])

        # capital y producción
        cuf = tab("cuf", sm["lufd"].out(c["lufdt"]))
        io = st["ic"] * (1 - fcaor) * cuf / icor
        iopc = io / pop
        so = st["sc"] * cuf / scor
        sopc = so / pop
        a.update(cuf=cuf, io=io, iopc=iopc, so=so, sopc=sopc)

        # agricultura
        ai = sm["ai"].out(alai)
        falm = tab("falm", sm["pfr"].out(c["fspd"]))
        aiph = ai * (1 - falm) / st["al"]
        lymc = tab("lymc", aiph)
        lymap = np.where(after("appyear", t), tab("lymap2", io / c["io70"]),
                         tab("lymap1", io / c["io70"]))
        ly = lymt * st["lfert"] * lymc * lymap
        f = ly * st["al"] * c["lfh"] * (1 - c["pl"])
        fpc = f / pop
        fr = fpc / c["sfpc"]
        ifpc = np.where(after("pyear", t), tab("ifpc2", iopc), tab("ifpc1", iopc))
        fioaa = np.where(after("pyear", t), tab("fioaa2", fpc / ifpc), tab("fioaa1", fpc / ifpc))
        tai = io * fioaa
        dcph = tab("dcph", st["pal"] / c["palt"])
        mpld = ly / (dcph * c["sd"])
        mlymc = tab("mlymc", aiph)
        mpai = alai * ly * mlymc / lymc
        fiald = tab("fiald", mpld / mpai)
        ldr = tai * fiald / dcph
        cai = tai * (1 - fiald)
        llmy1 = tab("llmy1", ly / c["ilf"])
        w = 0.95 ** ((t - c["llmytm"]) / 1.0)
        llmy = np.where(after("llmytm", t), w * llmy1 + (1 - w) * tab("llmy2", ly / c["ilf"]), llmy1)
        ler = st["al"] / (c["alln"] * llmy)
        uilr = tab("uilpc", iopc) * pop
        lrui = np.maximum(0.0, (uilr - st["uil"]) / c["uildt"])
        lfd = st["lfert"] * tab("lfdr", ppolx)
        lfr = (c["ilf"] - st["lfert"]) / tab("lfrt", falm)
        lytcr = np.where(after("pyear", t), st["lyt"] * tab("lytcrm", c["drf"] - fr), 0.0)
        a.update(f=f, fpc=fpc, ly=ly, aiph=aiph)

        # servicios, empleo
        isopc = np.where(after("pyear", t), tab("isopc2", iopc), tab("isopc1", iopc))
        fioas = np.where(after("pyear", t), tab("fioas2", sopc / isopc), tab("fioas1", sopc / isopc))
        fioacc = np.where(after("pyear", t), c["fioac2"], c["fioac1"])
        fioac = np.where(after("iet", t), tab("fioacv", iopc / c["iopcd"]), fioacc)
        fioai = 1 - fioaa - fioas - fioac
        jobs = (st["ic"] * tab("jpicu", iopc) * 1e-3 + st["sc"] * tab("jpscu", sopc) * 1e-3
                + tab("jph", aiph) * st["al"])
        lf = (st["p2"] + st["p3"]) * c["lfpf"]
        luf = jobs / lf
        a["luf"] = luf

        # recursos y contaminación
        pcrum = tab("pcrum", iopc)
        nrur = pop * pcrum * nruf
        rtcm = tab("rtcm", 1 - nrur / c["drur"])
        rctcr = np.where(after("pyear", t), st["rct"] * rtcm, 0.0)
        ppgio = pcrum * pop * c["frpm"] * c["imef"] * c["imti"]
        ppgao = aiph * st["al"] * c["fipm"] * c["amti"]
        ppgr = (ppgio + ppgao) * ppgf
        pptcm = tab("pptcm", 1 - ppolx / c["dppolx"])
        pptcr = np.where(after("pyear", t), st["ppt"] * pptcm, 0.0)
        ahl = c["ahl70"] * tab("ahlm", ppolx)
        ppasr = st["ppol"] / (ahl * 1.4)
        a["ppgr"] = ppgr

        # población
        hsapc = tab("hsapc", sopc)
        if first:
            sm["ehspc"].init(hsapc, c["hsid"])
        # ÚNICA comparación estricta del modelo: Vensim conmuta lmhs con Time > 1940. Con >= el
        # error contra la salida de Vensim sube de 1e-5 a 0,4 % (medido en la prueba de W1).
        lmhs = np.where(t > c["lmhs_year"], tab("lmhs2", sm["ehspc"].out(c["hsid"])),
                        tab("lmhs1", sm["ehspc"].out(c["hsid"])))
        lmc = 1 - tab("cmi", iopc) * tab("fpu", pop)
        le = c["len"] * tab("lmf", fpc / c["sfpc"]) * lmhs * tab("lmp", ppolx) * lmc
        if first:
            sm["ple"].init(le, c["lpd"])
            sm["diopc"].init(iopc, c["sad"])
            sm["aiopc"].init(iopc, c["ieat"])
        m = [tab(f"m{i}", le) for i in (1, 2, 3, 4)]
        d = [st[f"p{i}"] * m[i - 1] for i in (1, 2, 3, 4)]
        mat1 = st["p1"] * (1 - m[0]) / 15
        mat2 = st["p2"] * (1 - m[1]) / 30
        mat3 = st["p3"] * (1 - m[2]) / 20
        mtf = c["mtfn"] * tab("fm", le)
        aiopc = sm["aiopc"].out(c["ieat"])
        fie = (iopc - aiopc) / aiopc
        dcfs = np.where(after("zpgt", t), c["dcfszpg"],
                        c["dcfsn"] * tab("frsn", fie) * tab("sfsn", sm["diopc"].out(c["sad"])))
        dtf = dcfs * tab("cmple", sm["ple"].out(c["lpd"]))
        nfc = mtf / dtf - 1
        fcapc = tab("fsafc", nfc) * sopc
        if first:
            sm["fcfpc"].init(fcapc, c["hsid"])
        fce = np.where(after("fcest", t), 1.0, tab("fce", sm["fcfpc"].out(c["hsid"])))
        tf = np.minimum(mtf, mtf * (1 - fce) + dtf * fce)
        deaths = d[0] + d[1] + d[2] + d[3]
        births = np.where(after("pet", t), deaths, tf * st["p2"] * 0.5 / c["rlt"])
        a.update(le=le, tf=tf, cbr=1000 * births / pop, cdr=1000 * deaths / pop)
        if first:
            sm["ppapr"].init(ppgr, c["pptd"])

        # indicadores de World3-03
        gdppc = tab("gdppc", iopc)
        a["hwi"] = (tab("lei", le) + tab("ei", gdppc)
                    + np.log10(gdppc / c["rlgdp"]) / np.log10(c["rhgdp"] / c["rlgdp"])) / 3
        a["hef"] = (st["al"] / c["hpgha"] + st["uil"] / c["hpgha"]
                    + ppgr * c["hpup"] / c["hpgha"]) / c["tl"]

        for name in record:
            out[name][k] = st[name] if name in st else a[name]
        if k == steps:
            break

        # ── flujos de los stocks (Euler) ──
        icir = io * fioai
        flows = {
            "p1": births - d[0] - mat1, "p2": mat1 - d[1] - mat2, "p3": mat2 - d[2] - mat3,
            "p4": mat3 - d[3],
            "ic": icir - st["ic"] / alic, "sc": io * fioas - st["sc"] / alsc,
            "al": ldr - ler - lrui, "pal": -ldr, "uil": lrui, "lfert": lfr - lfd,
            "nr": -nrur, "ppol": sm["ppapr"].out(c["pptd"]) - ppasr,
            "lyt": lytcr, "rct": -rctcr, "ppt": pptcr,
        }
        smooth_in = {
            "ehspc": (hsapc, c["hsid"]), "ple": (le, c["lpd"]), "diopc": (iopc, c["sad"]),
            "aiopc": (iopc, c["ieat"]), "fcfpc": (fcapc, c["hsid"]), "lufd": (luf, c["lufdt"]),
            "ai": (cai, alai), "pfr": (fr, c["fspd"]), "lyf2": (st["lyt"], c["tdd"]),
            "nruf2": (st["rct"], c["tdd"]), "ppgf2": (st["ppt"], c["tdd"]),
            "ppapr": (ppgr, c["pptd"]),
        }
        derivs = {k2: sm[k2].deriv(x, dl) for k2, (x, dl) in smooth_in.items()}
        for k2, fl in flows.items():
            st[k2] = st[k2] + dt * fl
        for k2, dv in derivs.items():
            sm[k2].s = sm[k2].s + dt * dv

    return World3Run(time=times, series=out)
