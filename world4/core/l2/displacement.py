"""Los bloques de población del L2: quién huye de la violencia, adónde y cuándo vuelve.

Cada celda tiene residentes. Cada mes:

1. *Exposición.* ``e`` = muertes por 100 000 habitantes en la celda y sus ocho vecinas (la gente huye de la
   violencia cercana, no sólo de la de su celda), con memoria ``phi``: ``m_t = phi m_{t-1} + e_t``. Las muertes
   que UCDP sólo sitúa en el país se reparten por población.
2. *Huida.* Sale la fracción ``1 - exp(-beta m^eta)`` de los residentes que quedan.
3. *Destino.* Cruza la frontera la parte ``sigmoide(g0 + g1 z)`` de los que huyen, con ``z`` la distancia
   (estandarizada) de la celda a la frontera; el resto son desplazados internos. Los refugiados se reparten
   entre países de asilo como el stock ya existente de su país de origen (la gente va donde hay compatriotas).
4. *Retorno.* Cada stock vuelve a ritmo ``rho`` por mes cuando el país está en calma:
   ``rho exp(-E / E0)``, con ``E`` las muertes por 100 000 y mes del país en los últimos 12 meses.

Los ocho parámetros comunes se ajustan a los stocks ANUALES por país de origen (refugiados y solicitantes de
ACNUR; desplazados internos: el mayor de ACNUR e IDMC) con los años anteriores al origen, moviendo el modelo
con las muertes observadas. Después cada país recibe un multiplicador de ``beta`` (los desplazados por muerte
difieren en un orden de magnitud entre guerras), elegido en una rejilla y encogido hacia 1 con una penalización
``SHRINK * ln(m)^2`` [M]; un país sin historia de desplazamiento se queda en 1. Los stocks de desplazados
internos sólo cuentan desde el primer año en que el país tiene dato (antes no hay cero: no hay registro).
El reparto por celda no tiene dato contra el que medirse: es ``exploratory`` [M].

Límites declarados: sólo hay desplazamiento por violencia organizada (no económico ni climático); UCDP no
cuenta las muertes indirectas (hambre, enfermedad), que en Yemen son la mayoría; los desplazados internos de
ACNUR empiezan cuando ACNUR empieza a operar en el país, no cuando empieza el desplazamiento; los refugiados
palestinos bajo mandato de UNRWA no están.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world4.core.l2 import panel as pn

NAMES = ("beta", "eta", "phi", "g0", "g1", "rho_idp", "rho_ref", "E0")
START = np.array([np.log(2e-3), np.log(0.7), 1.0, -1.0, -0.5, np.log(0.01), np.log(0.005), np.log(0.5)])
SCALE = 1e4                              # los stocks se comparan en asinh(personas / SCALE)
TOP_ASYLUM = 12
MULT_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SHRINK = 0.05                            # penalización del multiplicador por país, en unidades de la pérdida


def unpack(x: np.ndarray) -> dict:
    """Vector sin restricciones -> parámetros en su escala natural."""
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    return {"beta": float(np.exp(x[0])), "eta": float(np.clip(np.exp(x[1]), 0.2, 2.0)), "phi": float(0.98 * sig(x[2])),
            "g0": float(x[3]), "g1": float(x[4]), "rho_idp": float(min(np.exp(x[5]), 0.5)),
            "rho_ref": float(min(np.exp(x[6]), 0.5)), "E0": float(np.exp(x[7]))}


@dataclass
class Geometry:
    """Lo que no cambia de un mes a otro."""
    cc: np.ndarray                       # (C,) país de la celda (las de fuera de la región, -1)
    nbr: np.ndarray                      # (C, 8)
    zb: np.ndarray                       # (C,) distancia a la frontera, estandarizada
    K: int


def geometry(P: pn.Panel) -> Geometry:
    return Geometry(cc=P.cell_country, nbr=P.nbr, zb=P.cov[:, pn.COV_USED.index("bdist1")], K=P.K)


def cell_deaths(P: pn.Panel, site: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """(..., S) muertes por sitio -> (..., C): las de sitios sin celda se reparten por población en su país."""
    loc = P.site_cell >= 0
    out = np.zeros(site.shape[:-1] + (P.C,))
    np.add.at(out, (Ellipsis, P.site_cell[loc]), site[..., loc])
    un = site[..., ~loc]                                               # (..., K) en el orden de los países
    k_of = P.site_country[~loc]
    cc = P.cell_country
    tot = np.zeros(P.K)
    np.add.at(tot, cc[cc >= 0], pop[cc >= 0])
    share = np.where(cc >= 0, pop / np.maximum(tot[np.maximum(cc, 0)], 1.0), 0.0)
    by_k = np.zeros(site.shape[:-1] + (P.K,))
    by_k[..., k_of] = un
    return out + by_k[..., np.maximum(cc, 0)] * share


class State:
    """Estado (R, ...) de los bloques de población."""

    def __init__(self, G: Geometry, home: np.ndarray, idp: np.ndarray, ref: np.ndarray, memory: np.ndarray,
                 recent: np.ndarray):
        self.G, self.home, self.idp, self.ref, self.memory, self.recent = G, home, idp, ref, memory, recent

    def step(self, th: dict, deaths: np.ndarray, pop: np.ndarray, pop_k: np.ndarray) -> dict:
        """Un mes. ``deaths`` (R, C); ``pop`` (C,) y ``pop_k`` (K,) del año. Devuelve los flujos por país."""
        G = self.G
        ok = G.nbr >= 0
        d_nb = deaths + (deaths[:, np.maximum(G.nbr, 0)] * ok[None]).sum(2)
        p_nb = pop + (pop[np.maximum(G.nbr, 0)] * ok).sum(1)
        e = 1e5 * d_nb / np.maximum(p_nb, 1.0)[None]
        self.memory = th["phi"] * self.memory + e
        mult = th.get("mult")
        beta = th["beta"] if mult is None else th["beta"] * np.asarray(mult)[np.maximum(G.cc, 0)][None]
        out_frac = 1.0 - np.exp(-beta * self.memory ** th["eta"])
        leave = self.home * out_frac
        self.home = self.home - leave
        cross = 1.0 / (1.0 + np.exp(-(th["g0"] + th["g1"] * G.zb)))
        inside = G.cc >= 0
        new_ref, new_idp = np.zeros((len(leave), G.K)), np.zeros((len(leave), G.K))
        np.add.at(new_ref, (slice(None), G.cc[inside]), (leave * cross[None])[:, inside])
        np.add.at(new_idp, (slice(None), G.cc[inside]), (leave * (1 - cross)[None])[:, inside])
        dk = np.zeros((len(leave), G.K))
        np.add.at(dk, (slice(None), G.cc[inside]), deaths[:, inside])
        self.recent = np.concatenate([self.recent[:, :, 1:], (1e5 * dk / np.maximum(pop_k, 1.0)[None])[:, :, None]], 2)
        calm = np.exp(-self.recent.mean(2) / th["E0"])
        back_idp, back_ref = th["rho_idp"] * calm * self.idp, th["rho_ref"] * calm * self.ref
        self.idp = self.idp + new_idp - back_idp
        self.ref = self.ref + new_ref - back_ref
        # los que vuelven se reparten entre las celdas del país en proporción a los que faltan
        missing = np.maximum(pop[None] - self.home, 0.0) * inside[None]
        tot = np.zeros((len(leave), G.K))
        np.add.at(tot, (slice(None), G.cc[inside]), missing[:, inside])
        back = (back_idp + back_ref)[:, np.maximum(G.cc, 0)]
        self.home = self.home + np.where(inside[None], back * missing / np.maximum(tot[:, np.maximum(G.cc, 0)], 1.0), 0.0)
        return {"new_ref": new_ref, "new_idp": new_idp}


def simulate_history(P: pn.Panel, th: dict, t1: int, t_start: int = 0) -> dict:
    """Modelo movido por las muertes observadas de ``[t_start, t1)``. Stocks (K, años) a fin de cada año."""
    G = geometry(P)
    site = P.site_deaths()
    y0 = P.ym(t_start)[0]
    j0 = int(np.clip(y0 - 1 - int(P.years[0]), 0, len(P.years) - 1))
    ref0 = np.nan_to_num(P.ref_stock[:, :, j0].sum(1)) if y0 - 1 >= int(P.years[0]) else np.zeros(P.K)
    st = State(G, P.pop_at(t_start, "cell")[None].copy(), np.zeros((1, P.K)), ref0[None].copy(),
               np.zeros((1, P.C)), np.zeros((1, P.K, 12)))
    years = np.arange(y0, P.ym(t1 - 1)[0] + 1)
    out = {"years": years, "ref": np.full((P.K, len(years)), np.nan), "idp": np.full((P.K, len(years)), np.nan)}
    for t in range(t_start, t1):
        pop = P.pop_at(t, "cell")
        st.step(th, cell_deaths(P, site[:, t], pop)[None], pop, P.pop_at(t))
        y, m = P.ym(t)
        if m == 12 or t == t1 - 1:
            out["ref"][:, y - y0], out["idp"][:, y - y0] = st.ref[0], st.idp[0]
    return out


def _targets(P: pn.Panel, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = years - int(P.years[0])
    ok = (j >= 0) & (j < len(P.years))
    ref, idp = np.full((P.K, len(years)), np.nan), np.full((P.K, len(years)), np.nan)
    ref[:, ok] = P.ref_stock[:, :, j[ok]].sum(1)
    idp[:, ok] = P.idp_stock[:, j[ok]]
    seen = np.cumsum(np.nan_to_num(idp) > 0, 1) > 0          # antes del primer dato no hay registro, no un cero
    return ref, np.where(seen, idp, np.nan)


def country_loss(th: dict, P: pn.Panel, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """Por país: suma de errores cuadráticos de los stocks de fin de año en asinh(personas / SCALE) y número
    de datos, con los años cerrados antes de ``t1``."""
    sim = simulate_history(P, th, t1)
    ref, idp = _targets(P, sim["years"])
    full = sim["years"] < P.ym(t1)[0]                       # el último año puede estar a medias
    f = lambda a: np.arcsinh(a / SCALE)  # noqa: E731
    err = np.concatenate([(f(sim["ref"]) - f(ref))[:, full], (f(sim["idp"]) - f(idp))[:, full]], 1)
    return np.nansum(err ** 2, 1), np.isfinite(err).sum(1)


def loss(x: np.ndarray, P: pn.Panel, t1: int) -> float:
    sse, n = country_loss(unpack(x), P, t1)
    return float(sse.sum() / n.sum()) if n.sum() else float("nan")


def fit(P: pn.Panel, t0: int, maxiter: int = 400, start=None) -> dict:
    """Parámetros con datos anteriores al mes ``t0`` (stocks hasta el año anterior)."""
    from scipy.optimize import minimize
    t1 = P.month(P.ym(t0)[0], 1)                              # hasta el fin del año anterior al origen
    x0 = START if start is None else np.asarray(start, float)
    if t1 < 36:
        return {"x": x0, **unpack(x0), "mult": np.ones(P.K), "loss": float("nan"), "fitted": False}
    r = minimize(loss, x0, args=(P, t1), method="Nelder-Mead",
                 options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
    th = unpack(r.x)
    # multiplicador por país: los países no se influyen entre sí, así que una pasada por valor de la rejilla basta
    per = []
    for m in MULT_GRID:
        sse, n = country_loss({**th, "mult": np.full(P.K, m)}, P, t1)
        per.append(sse / np.maximum(n, 1) + SHRINK * np.log(m) ** 2)
    mult = np.array(MULT_GRID)[np.argmin(np.stack(per), 0)]
    sse, n = country_loss({**th, "mult": mult}, P, t1)
    return {"x": r.x, **th, "mult": mult, "loss_common": float(r.fun), "loss": float(sse.sum() / max(n.sum(), 1)),
            "loss_start": loss(x0, P, t1), "fitted": True, "evaluations": int(r.nfev) + len(MULT_GRID) + 1}


def asylum_shares(P: pn.Panel, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """(K, TOP_ASYLUM + 1) reparto de los refugiados de cada origen por país de asilo (el último es «otros»),
    del último stock anual anterior al origen; un origen sin stock usa el reparto del conjunto de la región."""
    j = int(np.clip(P.ym(t0)[0] - 1 - int(P.years[0]), 0, len(P.years) - 1))
    stock = np.nan_to_num(P.ref_stock[:, :, j])
    pooled = stock.sum(0)
    top = np.argsort(-pooled)[:TOP_ASYLUM]
    sh = np.concatenate([stock[:, top], (stock.sum(1) - stock[:, top].sum(1))[:, None]], 1)
    ps = np.concatenate([pooled[top], [pooled.sum() - pooled[top].sum()]])
    ps = ps / max(ps.sum(), 1.0) if ps.sum() > 0 else np.full(len(ps), 1.0 / len(ps))
    tot = sh.sum(1, keepdims=True)
    return np.where(tot > 0, sh / np.maximum(tot, 1.0), ps[None]), top


def initial_state(P: pn.Panel, th: dict, t0: int, members: int) -> State:
    """Estado en el origen: stocks del último año publicado, memoria de exposición de los 60 meses previos."""
    G = geometry(P)
    j = int(np.clip(P.ym(t0)[0] - 1 - int(P.years[0]), 0, len(P.years) - 1))
    have = P.ym(t0)[0] - 1 >= int(P.years[0])
    ref = np.nan_to_num(P.ref_stock[:, :, j].sum(1)) if have else np.zeros(P.K)
    idp = np.nan_to_num(P.idp_stock[:, j]) if have else np.zeros(P.K)
    site = P.site_deaths()
    mem, recent = np.zeros(P.C), np.zeros((P.K, 12))
    ok = G.nbr >= 0
    for t in range(max(t0 - 60, 0), t0):
        pop = P.pop_at(t, "cell")
        d = cell_deaths(P, site[:, t], pop)
        d_nb = d + (d[np.maximum(G.nbr, 0)] * ok).sum(1)
        p_nb = pop + (pop[np.maximum(G.nbr, 0)] * ok).sum(1)
        mem = th["phi"] * mem + 1e5 * d_nb / np.maximum(p_nb, 1.0)
        dk = np.zeros(P.K)
        np.add.at(dk, G.cc[G.cc >= 0], d[G.cc >= 0])
        recent = np.concatenate([recent[:, 1:], (1e5 * dk / np.maximum(P.pop_at(t), 1.0))[:, None]], 1)
    pop = P.pop_at(t0, "cell")
    away = (ref + idp)[np.maximum(G.cc, 0)] * np.where(G.cc >= 0, pop / np.maximum(P.pop_at(t0)[np.maximum(G.cc, 0)], 1.0), 0)
    home = np.maximum(pop - away, 0.0)
    rep = lambda a: np.repeat(a[None], members, 0)  # noqa: E731
    return State(G, rep(home), rep(idp), rep(ref), rep(mem), rep(recent))


def simulate(P: pn.Panel, th: dict, t0: int, site_deaths: np.ndarray | None = None,
             by_cell: np.ndarray | None = None) -> dict:
    """Pronóstico con las muertes simuladas por sitio ``site_deaths`` (R, S, H) o ya por celda ``by_cell``
    (R, C, H). Stocks (R, K, H) por mes."""
    R, _, H = (site_deaths if by_cell is None else by_cell).shape
    st = initial_state(P, th, t0, R)
    out = {k: np.zeros((R, P.K, H), np.float32) for k in ("ref", "idp", "new_ref", "new_idp")}
    for h in range(H):
        pop = P.pop_at(t0 + h, "cell")
        d = cell_deaths(P, site_deaths[:, :, h].astype(float), pop) if by_cell is None else by_cell[:, :, h]
        fl = st.step(th, d, pop, P.pop_at(t0 + h))
        out["ref"][:, :, h], out["idp"][:, :, h] = st.ref, st.idp
        out["new_ref"][:, :, h], out["new_idp"][:, :, h] = fl["new_ref"], fl["new_idp"]
    return out
