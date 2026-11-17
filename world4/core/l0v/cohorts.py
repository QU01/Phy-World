"""Población por cohortes de edad y sexo, con el mismo paso que el L0.

World3 reparte la población en cuatro cajas (0-14, 15-44, 45-64, 65+) de las que se sale a ritmo constante: una
generación numerosa no avanza como una ola, se difumina. Aquí cada sexo se guarda en casillas de edad de ancho
``dt`` (con ``dt = 0,5``: 200 casillas hasta los 100 años y una abierta de 100+), y en cada paso toda la
población avanza UNA casilla: el envejecimiento es exacto, sin difusión numérica.

En un paso de ``t`` a ``t + dt`` (Euler explícito, como el resto del L0):

* muertes: cada casilla sobrevive con ``exp(-m · dt)``, con ``m`` la tasa de las edades que RECORRE en el
  paso: la última casilla de cada año de edad cumple años a mitad del paso, así que lleva la media de su tasa y
  la de la edad siguiente (sin esto la edad 0 se expondría 1,25 años a la mortalidad infantil con ``dt = 0,5``,
  y las muertes mundiales de 1951 saldrían un 6 % altas: medido);
* nacimientos: ``dt · TFR · Σ mujeres(a) · patrón(a)``, con ``patrón`` la fracción de la fecundidad que cae en
  cada edad simple (suma 1), así que ``TFR`` es la fecundidad total de periodo por definición;
* los nacidos se reparten por la razón de sexos al nacer y sobreviven medio paso con la tasa de la edad 0.

Todo va vectorizado sobre un eje de ``n`` miembros o regiones, como ``world4.core.l0``.
"""
from __future__ import annotations

import numpy as np

A = 101                                   # edades simples 0..99 y 100+
GROUPS = {"p1": (0, 15), "p2": (15, 45), "p3": (45, 65), "p4": (65, A)}      # las cuatro cajas de World3


class Cohorts:
    def __init__(self, popM, popF, dt: float = 0.5, n: int = 1):
        """``popM``, ``popF``: personas por edad simple, forma ``(101,)`` o ``(n, 101)``."""
        k = round(1 / dt)
        if abs(k * dt - 1) > 1e-9:
            raise ValueError(f"dt must divide one year, got {dt}")
        self.dt, self.k, self.n = dt, k, n
        self.age = np.r_[np.repeat(np.arange(A - 1), k), A - 1]            # edad simple de cada casilla
        self.nxt = np.minimum(self.age + 1, A - 1)
        # parte del paso que la casilla pasa en la edad siguiente: 1/2 para la última casilla de cada año
        self.w = np.r_[np.tile(np.r_[np.zeros(k - 1), 0.5], A - 1), 0.0]
        self.N = np.stack([self._bins(popM), self._bins(popF)], 1)          # (n, 2, casillas)

    def _bins(self, pop):
        pop = np.broadcast_to(np.asarray(pop, float), (self.n, A))
        return np.concatenate([np.repeat(pop[:, :-1] / self.k, self.k, axis=1), pop[:, -1:]], 1)

    # ---------------------------------------------------------------- lectura
    def total(self):
        return self.N.sum((1, 2))

    def by_age(self):
        """Personas por sexo y edad simple: ``(n, 2, 101)``."""
        k = self.k
        body = self.N[:, :, :-1].reshape(self.n, 2, A - 1, k).sum(-1)
        return np.concatenate([body, self.N[:, :, -1:]], 2)

    def groups(self) -> dict:
        s = self.by_age().sum(1)
        return {g: s[:, a:b].sum(1) for g, (a, b) in GROUPS.items()}

    # ---------------------------------------------------------------- paso
    def _along(self, by_age):
        """Una magnitud por edad simple (..., 101), promediada sobre las edades que cada casilla recorre en el paso."""
        return (1 - self.w) * by_age[..., self.age] + self.w * by_age[..., self.nxt]

    def step(self, mx, tfr, pattern, srb=1.05):
        """Avanza ``dt``. ``mx``: tasas ``(n, 2, 101)``; ``tfr``: ``(n,)``; ``pattern``: ``(101,)`` o ``(n, 101)``
        con suma 1; ``srb``: razón de sexos al nacer, escalar o ``(n,)``.
        Devuelve ``(nacimientos, muertes)`` del paso, en personas, forma ``(n,)``."""
        dt = self.dt
        mx = np.broadcast_to(np.asarray(mx, float), (self.n, 2, A))
        tfr = np.broadcast_to(np.asarray(tfr, float), (self.n,))
        pattern = np.asarray(pattern, float)
        surv = np.exp(-self._along(mx) * dt)
        births = dt * tfr * (self.N[:, 1] * self._along(pattern)).sum(1)
        srb = np.broadcast_to(np.asarray(srb, float), (self.n,))
        born = births[:, None] * np.stack([srb, np.ones(self.n)], 1) / (1 + srb)[:, None]
        infant = born * np.exp(-mx[:, :, 0] * dt / 2)
        alive = self.N * surv
        deaths = (self.N - alive).sum((1, 2)) + (born - infant).sum(1)
        new = np.empty_like(self.N)
        new[:, :, 0] = infant
        new[:, :, 1:] = alive[:, :, :-1]
        new[:, :, -1] += alive[:, :, -1]                                     # la casilla abierta retiene a los suyos
        self.N = new
        return births, deaths

    def migrate(self, net, profile):
        """Suma ``net`` migrantes netos ``(n,)`` (negativo: salen), mitad de cada sexo, repartidos por edad con
        ``profile`` ``(101,)`` de suma 1. Una casilla no baja de cero: devuelve lo realmente sumado ``(n,)``."""
        net = np.broadcast_to(np.asarray(net, float), (self.n,))
        per_bin = np.asarray(profile, float)[self.age] / np.r_[np.full((A - 1) * self.k, self.k), 1.0]
        before = self.total()
        self.N = np.maximum(self.N + 0.5 * net[:, None, None] * per_bin, 0.0)
        return self.total() - before
