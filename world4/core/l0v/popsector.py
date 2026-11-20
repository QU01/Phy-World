"""Las ecuaciones de esperanza de vida y fecundidad de World3-03, aplicadas por grupo de países.

Son las mismas de ``world4.core.l0`` (mismas tablas, constantes y retardos), sacadas de su bucle para poder
evaluarlas con las entradas de cada grupo: su producto industrial y de servicios por persona, su alimento por
persona, y la contaminación y la población del mundo. Con un solo grupo y las entradas del mundo devuelven
exactamente la ``le`` y la ``tf`` de ``l0.run`` (prueba): lo único que cambia con varios grupos es DÓNDE se
evalúan las tablas.

    esperanza de vida = len · lmf(alimento) · lmhs(servicios de salud, retardados) · lmp(contaminación) · lmc(hacinamiento)
    fecundidad total  = min(mtf, mtf·(1 - fce) + dtf·fce), con mtf la máxima biológica (crece con la esperanza
                        de vida), dtf la deseada (baja con el producto industrial retardado y con la esperanza de
                        vida percibida) y fce la eficacia del control de la natalidad (crece con los servicios)
"""
from __future__ import annotations

import numpy as np

from world4.core import l0


class PopSector:
    def __init__(self, G: int, constants: dict, tables: dict, dt: float):
        self.G, self.dt = G, dt
        self.c = {k: float(np.asarray(v, float).reshape(-1)[0]) for k, v in constants.items()}
        self.grids = {name: l0._grid(name, spec) for name, spec in tables.items()}
        self.sm = {"ehspc": l0._Smooth(1), "ple": l0._Smooth(3), "diopc": l0._Smooth(3), "aiopc": l0._Smooth(1),
                   "fcfpc": l0._Smooth(3)}
        self.first = True

    def _tab(self, name, x):
        return np.interp(x, *self.grids[name])

    def step(self, t: float, iopc, sopc, fpc, ppolx, pop):
        """``(le, tf)`` de cada grupo en ``t`` y avance de los retardos a ``t + dt``. ``iopc``, ``sopc``, ``fpc``:
        ``(G,)``; ``ppolx`` y ``pop``: del mundo."""
        c, sm, tab = self.c, self.sm, self._tab
        iopc, sopc, fpc = (np.broadcast_to(np.asarray(v, float), (self.G,)) for v in (iopc, sopc, fpc))
        hsapc = tab("hsapc", sopc)
        if self.first:
            sm["ehspc"].init(hsapc, c["hsid"])
        ehspc = sm["ehspc"].out(c["hsid"])
        lmhs = tab("lmhs2", ehspc) if t > c["lmhs_year"] else tab("lmhs1", ehspc)
        lmc = 1 - tab("cmi", iopc) * tab("fpu", pop)
        le = c["len"] * tab("lmf", fpc / c["sfpc"]) * lmhs * tab("lmp", ppolx) * lmc
        if self.first:
            sm["ple"].init(le, c["lpd"])
            sm["diopc"].init(iopc, c["sad"])
            sm["aiopc"].init(iopc, c["ieat"])
        mtf = c["mtfn"] * tab("fm", le)
        aiopc = sm["aiopc"].out(c["ieat"])
        fie = (iopc - aiopc) / aiopc
        dcfs = (np.full(self.G, c["dcfszpg"]) if t >= c["zpgt"]
                else c["dcfsn"] * tab("frsn", fie) * tab("sfsn", sm["diopc"].out(c["sad"])))
        dtf = dcfs * tab("cmple", sm["ple"].out(c["lpd"]))
        fcapc = tab("fsafc", mtf / dtf - 1) * sopc
        if self.first:
            sm["fcfpc"].init(fcapc, c["hsid"])
        fce = np.ones(self.G) if t >= c["fcest"] else tab("fce", sm["fcfpc"].out(c["hsid"]))
        tf = np.minimum(mtf, mtf * (1 - fce) + dtf * fce)
        self.first = False
        for name, x, delay in (("ehspc", hsapc, c["hsid"]), ("ple", le, c["lpd"]), ("diopc", iopc, c["sad"]),
                               ("aiopc", iopc, c["ieat"]), ("fcfpc", fcapc, c["hsid"])):
            sm[name].s = sm[name].s + self.dt * sm[name].deriv(x, delay)
        return le, tf
