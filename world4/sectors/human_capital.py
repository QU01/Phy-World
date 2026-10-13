"""Capital humano endógeno (F1): escolaridad por cohorte con retardo, ``hc`` como stock. Cierra D2.

* Se invierte la fórmula de la PWT (ln hc = phi(s), rendimientos de Psacharopoulos) para obtener
  los años medios de escolaridad ``s`` que hay detrás de cada ``hc`` observado.
* En el origen, la escolaridad por edad sigue un perfil lineal por cohorte, s(a) = s0 + b0 (m - a),
  con ``b0`` la ganancia anual de los 5 años previos y ``m`` la edad media de 25-64: la media
  ponderada de 25-64 reproduce exactamente ``s0`` [S: perfil supuesto, no observado].
* Cada año envejecen las cohortes con su escolaridad; la que cumple 25 entra con
  ``s_entrada(t) = s_entrada(t-1) + b_t``, donde la ganancia por cohorte
      b_t = f0 + f1 s_adultos_{t-1} + f2 ly_{t-1} + f3 r_{t-1}       (f3 >= 0)
  se estima con el cambio anualizado a 5 años de la escolaridad adulta observada (el ritmo de la
  expansión educativa) y ``r`` = esfuerzo del canal r del L1 en unidades del dato (I+D + educación
  pública / PIB): la política educativa de hoy llega al stock con el retardo de una generación.
* hc_t = hc_origen * exp(phi(s_adultos_t) - phi(s_adultos_origen)): continuo en el origen.

``s_adultos`` es la media de 25-64 ponderada con las cohortes del sector de población, así que la
mortalidad y la migración también mueven el stock.
"""
from __future__ import annotations

import numpy as np

from world4.sectors import _panel, population

A = population.A
ADULT = slice(25, 65)
ENTRY_AGE = 25


def fit(W, LY, cutoff_t: int, lag: int = 5) -> _panel.Equation:
    p = population.panel(W, LY)
    s = p["s"]
    y = (s - population._lag(s, lag)) / lag
    regs = [population._lag(s, lag), population._lag(p["ly"], lag), population._lag(p["r"], lag)]
    cols, country, year = population._stack([y] + regs, lag + 1, cutoff_t)
    X = np.column_stack([np.ones_like(cols[0])] + cols[1:])
    return _panel.fit("schooling", cols[0], X, country, ("const", "schooling", "ly", "r"), 1950 + cutoff_t,
                      signs={"r": +1}, n_countries=W.N, recent=year > cutoff_t - 10, year=year)


class HumanCapital:
    def __init__(self, W, LY, t0: int, eq: _panel.Equation, pop: population.Population, R: int, seed: int,
                 noise: bool = True):
        self.W, self.t0, self.eq, self.pop, self.R = W, t0, eq, pop, R
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        N = W.N
        self.hc0 = W.hc[:, t0].astype(float)
        p = population.panel(W, LY)
        s0 = p["s"][:, t0]
        lo = max(0, t0 - 5)
        b0 = np.clip(np.nan_to_num((s0 - p["s"][:, lo]) / max(t0 - lo, 1)), 0.0, 0.25)
        self.ok = np.isfinite(s0) & (self.hc0 > 0)
        Pb = pop.P.sum(2)                                       # (R, N, A)
        ages = np.arange(A, dtype=float)
        wa = Pb[..., ADULT]
        m = np.sum(wa * ages[ADULT], -1) / np.maximum(wa.sum(-1), 1e-12)          # (R, N)
        s0r = np.broadcast_to(np.nan_to_num(s0), (R, N))
        prof = s0r[..., None] + b0[None, :, None] * (m[..., None] - ages)
        prof[..., :ENTRY_AGE] = (s0r + b0[None] * (m - ENTRY_AGE))[..., None]
        self.sa = np.clip(prof, 0.0, 16.0)                      # (R, N, A) escolaridad por edad
        self.s_entry = self.sa[..., ENTRY_AGE].copy()
        self.s_adult0 = self.adult()
        self.B = eq.draw(R, self.rng) if noise else np.broadcast_to(eq.beta, (R, len(eq.beta))).copy()
        self.common, self.own = _panel.shocks(eq, R, N, self.rng, on=noise)
        self.hist = {t0: np.broadcast_to(self.hc0, (R, N)).copy()}

    def adult(self):
        w = self.pop.P.sum(2)[..., ADULT]
        return np.sum(w * self.sa[..., ADULT], -1) / np.maximum(w.sum(-1), 1e-12)

    def advance(self, t: int, ly, r_data):
        """Escolaridad y hc del año ``t`` (llamar DESPUÉS de avanzar la población a ``t``)."""
        R, N = self.R, self.W.N
        sad = self.adult()
        X = np.stack([np.ones((R, N)), sad, ly, r_data], -1)
        b = np.einsum("rnk,rk->rn", np.nan_to_num(X), self.B) + self.eq.offset
        if self.noise:
            b = b + self.common() + self.own()
        self.s_entry = np.clip(self.s_entry + np.clip(b, 0.0, 0.3), 0.0, 16.0)
        sa = np.empty_like(self.sa)
        sa[..., 1:] = self.sa[..., :-1]
        sa[..., A - 1] = 0.5 * (self.sa[..., A - 1] + self.sa[..., A - 2])     # grupo abierto 100+
        sa[..., :ENTRY_AGE + 1] = self.s_entry[..., None]
        self.sa = sa
        hc = self.hc0[None] * np.exp(population.phi_hc(self.adult()) - population.phi_hc(self.s_adult0))
        hc = np.where(self.ok[None], hc, self.hc0[None])
        self.hist[t] = hc
        return hc
