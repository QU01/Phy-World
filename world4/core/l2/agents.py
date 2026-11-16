"""El simulador de agentes del L2: actores armados, díadas, huella espacial y nacimiento de díadas.

**Quién decide.** Un *actor* es una organización que UCDP nombra: un gobierno, un grupo armado no estatal o
«civiles» (el blanco de la violencia unilateral). Una *díada* es un par de actores que se enfrentan (o un
actor y los civiles). El estado de un actor en este v0 es su ACTIVIDAD en todos sus frentes; UCDP no observa
efectivos ni recursos, así que no hay un stock de fuerza que no se pueda medir [declarado].

**Qué pasa cada mes**, en cada miembro del ensemble:

1. *Combatir o no.* Cada díada viva se activa con una probabilidad que depende de su propia historia
   reciente, del tiempo que lleva callada, de su edad, de la actividad de cada uno de sus dos actores EN SUS
   OTROS FRENTES (un actor ocupado en otro frente puede retirarse de éste o escalar en todos) y de la
   violencia del resto del país. Es un logit agrupado estimado con los meses anteriores al origen.
2. *Cuánto.* Si se activa, el número de eventos sigue una regresión log-lineal con los mismos rasgos y
   residuos remuestreados.
3. *Con qué letalidad.* Cada evento toma sus muertes de los eventos pasados de la propia díada, mezclados
   con los de su tipo de violencia cuando la díada tiene poca historia (encogimiento m / (m + 10)).
4. *Dónde.* Cada díada tiene una huella sobre los sitios (celda, país): la memoria con decaimiento de dónde
   ha combatido, que se difunde a las celdas vecinas DEL MISMO PAÍS (una frontera no se cruza por difusión:
   a otro país sólo se llega por eventos observados o por una díada nueva). Una parte pequeña de los eventos
   (el suelo) cae fuera de la huella: donde ha habido violencia de cualquier díada del país, mezclado con
   donde vive la gente. Los eventos se reparten así y actualizan la huella: el frente se mueve con lo
   simulado, no con lo observado. Decaimiento, difusión, suelo y mezcla se eligen por verosimilitud a un
   paso de los lugares de los eventos anteriores al origen.
5. *Díadas nuevas.* Nacen por país con una tasa de Poisson que crece con la violencia que ya hay (la guerra
   fragmenta). La nueva copia el tipo y los papeles de una díada nacida antes en ese país: gobierno del país,
   otro gobierno, civiles, un grupo que ya existía o uno nuevo. Una díada que lleva ``ALIVE_MONTHS`` meses
   callada muere.

Las guerras entre Estados no tienen maquinaria aparte: en UCDP son díadas de dos gobiernos y aquí también
(rasgo ``interstate``). Hay tan pocas que su parte del pronóstico es exploratoria aunque el resto no lo sea.

**Lo que es supuesto y no estimación [M]:** la semilla espacial de una díada nueva, el reparto de las
muertes entre sitios en proporción a los eventos (un evento masivo no concentra sus muertes en un sitio) y
el techo de eventos por díada y mes (``N_CAP`` veces el máximo histórico de la región).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world4.core.l2 import panel as pn
from world4.core.l2 import stats

ALIVE_MONTHS = 120          # meses callada tras los que una díada deja de existir
HIST = 36                   # meses de historia que cada díada recuerda
POOL = 512                  # tamaño del depósito de letalidad por díada
SHRINK = 10.0               # eventos de la propia díada que pesan como el depósito de su tipo
MARK_HALFLIFE = None        # meses; None = los últimos 500 eventos de la díada pesan lo mismo
BIRTH_SLOTS = 40            # díadas nuevas como máximo por miembro
BIRTH_WINDOW = 180          # meses de nacimientos que sirven de plantilla
CENSOR = 24                 # los «nacimientos» de los primeros meses del GED son díadas que ya existían
N_CAP = 2.0
FOOT_GRID = {"decay": (0.6, 0.8, 0.9, 0.97), "diffuse": (0.0, 0.03, 0.1, 0.3), "floor": (0.002, 0.02),
             "mix": (0.0, 0.5, 0.9)}
COUNTRY_DECAY = 0.97        # memoria mensual de la huella de TODA la violencia de un país [M]
FOOT_WINDOW = 120
FEATURES = ("own1", "own3", "own12", "since", "age", "side_a", "side_b", "ctx", "non_state", "one_sided",
            "interstate", "l1")
ROLES = ("gov_home", "gov_other", "civilians", "existing", "new")


# ---------------------------------------------------------------------- rasgos de las díadas en la historia
def dyad_country(P: pn.Panel, t0: int) -> np.ndarray:
    """País principal de cada díada con los eventos anteriores a ``t0`` (-1 si aún no tiene)."""
    e = P.ev
    m = e["t0"] < t0
    cnt = np.zeros((P.D, P.K))
    np.add.at(cnt, (e["dyad"][m], e["country"][m]), 1.0)
    return np.where(cnt.sum(1) > 0, cnt.argmax(1), -1)


def _first_event(n: np.ndarray) -> np.ndarray:
    has = n > 0
    return np.where(has.any(1), has.argmax(1), n.shape[1] + 10 ** 6)


def _other_fronts(P: pn.Panel, n3: np.ndarray, side: np.ndarray) -> np.ndarray:
    """(D, T+1): eventos (media de 3 meses) del actor ``side`` de cada díada en sus OTRAS díadas."""
    tot = np.zeros((len(P.actor_kind), n3.shape[1]))
    for s in (P.dyad_a, P.dyad_b):
        np.add.at(tot, s, n3)
    out = tot[side] - n3
    return np.where((P.actor_kind[side] == 2)[:, None], 0.0, out)        # «civiles» no es un actor que decida


def history_features(P: pn.Panel, t0: int, l1=None) -> dict:
    """Rasgos (D, T+1, F) y lo que hace falta para entrenar. ``l1`` (K, T+1) es la covariable del L1 o None."""
    n = P.dyad_series("events")
    first = _first_event(n)
    n1, n3, n12 = (pn.lagged_mean(n, k) for k in (1, 3, 12))
    since = pn.months_since(n, cap=float(ALIVE_MONTHS))
    t = np.arange(n.shape[1] + 1)
    age = np.maximum(t[None] - first[:, None], 0)
    home = dyad_country(P, t0)
    by_country = np.zeros((P.K, n.shape[1] + 1))
    np.add.at(by_country, np.maximum(home, 0), np.where((home >= 0)[:, None], n3, 0.0))
    ctx = by_country[np.maximum(home, 0)] - n3
    kind_a, kind_b = P.actor_kind[P.dyad_a], P.actor_kind[P.dyad_b]
    one = np.ones_like(n1)
    cov = np.zeros_like(n1) if l1 is None else l1[np.maximum(home, 0)]
    F = np.stack([np.log1p(n1), np.log1p(n3), np.log1p(n12), np.log1p(since), np.log1p(age),
                  np.log1p(_other_fronts(P, n3, P.dyad_a)), np.log1p(_other_fronts(P, n3, P.dyad_b)),
                  np.log1p(np.maximum(ctx, 0)), one * (P.dyad_type == 2)[:, None], one * (P.dyad_type == 3)[:, None],
                  one * ((kind_a == 0) & (kind_b == 0))[:, None], cov], -1)
    return {"F": F, "n": n, "first": first, "since": since, "home": home}


# ---------------------------------------------------------------------- huella espacial
def _events_by_month(P: pn.Panel, t_lo: int, t_hi: int, dyads: np.ndarray) -> list:
    """Por cada mes de ``[t_lo, t_hi)``: (fila de la díada, sitio, número de eventos que empiezan ese mes)."""
    e = P.ev
    pos = np.full(P.D, -1)
    pos[dyads] = np.arange(len(dyads))
    m = (e["t0"] >= t_lo) & (e["t0"] < t_hi) & (pos[e["dyad"]] >= 0)
    key = (e["t0"][m] - t_lo) * (len(dyads) * P.S) + pos[e["dyad"][m]] * P.S + e["site"][m]
    uk, cnt = np.unique(key, return_counts=True)
    month, rest = np.divmod(uk, len(dyads) * P.S)
    di, si = np.divmod(rest, P.S)
    cut = np.searchsorted(month, np.arange(t_hi - t_lo + 1))
    return [(di[cut[j]:cut[j + 1]], si[cut[j]:cut[j + 1]], cnt[cut[j]:cut[j + 1]].astype(np.float32))
            for j in range(t_hi - t_lo)]


def footprint_at(P: pn.Panel, t0: int, dyads: np.ndarray, decay: float, warm: int = 240) -> np.ndarray:
    """(len(dyads), S): huella en el mes ``t0``, la suma de los eventos anteriores con peso ``decay^antigüedad``."""
    W = np.zeros((len(dyads), P.S), np.float32)
    lo = max(t0 - warm, 0)
    for j, (di, si, cnt) in enumerate(_events_by_month(P, lo, t0, dyads)):
        W[di, si] += cnt * np.float32(decay ** (t0 - 1 - (lo + j)))
    return W


_OPS: dict = {}


def _diffusion(site_nbr: np.ndarray, diffuse: float):
    """Matriz dispersa (S, S) tal que ``W @ A`` es la huella difundida. Se guarda por (vecindad, difusión)."""
    from scipy import sparse
    key = (site_nbr.shape, hash(site_nbr.tobytes()), float(diffuse))
    if key not in _OPS:
        S = len(site_nbr)
        ok = site_nbr >= 0
        deg = np.maximum(ok.sum(1), 1)
        i = np.repeat(np.arange(S), site_nbr.shape[1])[ok.ravel()]
        j = site_nbr.ravel()[ok.ravel()]
        N = sparse.csr_matrix((diffuse / deg[i], (i, j)), shape=(S, S))
        keep = np.where(ok.any(1), 1.0 - diffuse, 1.0)
        _OPS[key] = (sparse.diags(keep) + N).T.tocsr()                       # se guarda traspuesta
    return _OPS[key]


def spread(W: np.ndarray, site_nbr: np.ndarray, diffuse: float) -> np.ndarray:
    """Huella difundida: cada sitio conserva ``1 - diffuse`` y reparte el resto entre sus vecinos.
    ``W`` es (..., S); la masa total se conserva."""
    if diffuse <= 0:
        return W
    At = _diffusion(site_nbr, diffuse)
    flat = np.asarray(W, np.float64).reshape(-1, W.shape[-1])
    return np.asarray(At @ flat.T).T.reshape(W.shape)


def site_prior(P: pn.Panel, t0: int) -> np.ndarray:
    """(K, S): reparto de la población de cada país entre sus sitios con celda (suelo de la huella)."""
    pop = P.pop_at(t0, "cell")
    out = np.zeros((P.K, P.S))
    loc = P.site_cell >= 0
    out[P.site_country[loc], np.where(loc)[0]] = pop[P.site_cell[loc]] + 1.0
    tot = out.sum(1, keepdims=True)
    return np.where(tot > 0, out / np.maximum(tot, 1e-12), 1.0 / P.S)


def fit_footprint(P: pn.Panel, t0: int, home: np.ndarray) -> dict:
    """Decaimiento, difusión y suelo por verosimilitud a un paso de los sitios de los eventos de
    ``[t0 - FOOT_WINDOW, t0)``; la huella de partida de cada combinación se calienta con los 60 meses previos."""
    lo = max(t0 - FOOT_WINDOW, 0)
    warm = max(lo - 60, 0)
    n = P.dyad_series("events")
    dy = np.where((n[:, warm:t0].sum(1) > 0) & (home >= 0))[0]
    if len(dy) == 0 or t0 - lo < 6:
        return {"decay": 0.9, "diffuse": 0.1, "floor": 0.02, "mix": 0.5, "loglik": float("nan"), "fitted": False}
    months = _events_by_month(P, warm, t0, dy)
    pop_prior = site_prior(P, t0)
    hd = home[dy]
    combos = [(df, fl, mx) for df in FOOT_GRID["diffuse"] for fl in FOOT_GRID["floor"] for mx in FOOT_GRID["mix"]]
    best = None
    for decay in FOOT_GRID["decay"]:
        W = np.zeros((len(dy), P.S), np.float32)
        Wc = np.zeros((P.K, P.S))                             # huella de toda la violencia de cada país
        ll = dict.fromkeys(combos, 0.0)
        for j, (di, si, cnt) in enumerate(months):
            if warm + j >= lo and len(di):                    # sólo cuentan los sitios donde hubo eventos
                ctot = Wc.sum(1)[hd[di]]
                cshare = np.where(ctot > 0, Wc[hd[di], si] / np.maximum(ctot, 1e-12), pop_prior[hd[di], si])
                priors = {mx: mx * cshare + (1 - mx) * pop_prior[hd[di], si] for mx in FOOT_GRID["mix"]}
                for diffuse in FOOT_GRID["diffuse"]:
                    sp = spread(W, P.site_nbr, diffuse)
                    tot = sp.sum(1)[di]
                    share = np.where(tot > 0, sp[di, si] / np.maximum(tot, 1e-12), 0.0)
                    for floor in FOOT_GRID["floor"]:
                        for mx, pr in priors.items():
                            p = np.where(tot > 0, (1 - floor) * share + floor * pr, pr)
                            ll[(diffuse, floor, mx)] += float(np.sum(cnt * np.log(np.maximum(p, 1e-12))))
            W *= np.float32(decay)
            W[di, si] += cnt
            Wc *= COUNTRY_DECAY
            np.add.at(Wc, (hd[di], si), cnt)
        for (diffuse, floor, mx), v in ll.items():
            if best is None or v > best["loglik"]:
                best = {"decay": decay, "diffuse": diffuse, "floor": floor, "mix": mx, "loglik": v, "fitted": True}
    return best


def country_footprint(P: pn.Panel, t0: int, home: np.ndarray, warm: int = 240) -> np.ndarray:
    """(K, S): huella de toda la violencia de cada país en el mes ``t0`` (díadas con ese país principal)."""
    dy = np.where(home >= 0)[0]
    out = np.zeros((P.K, P.S))
    lo = max(t0 - warm, 0)
    for j, (di, si, cnt) in enumerate(_events_by_month(P, lo, t0, dy)):
        np.add.at(out, (home[dy][di], si), cnt * COUNTRY_DECAY ** (t0 - 1 - (lo + j)))
    return out


# ---------------------------------------------------------------------- el modelo estimado
@dataclass
class AgentModel:
    t0: int
    b_act: np.ndarray                    # logit de activación
    b_int: np.ndarray                    # log-eventos si activa
    resid: np.ndarray
    n_cap: float
    foot: dict
    # díadas vivas en el origen
    dyads: np.ndarray                    # índices en el panel
    home: np.ndarray                     # (D0,) país principal
    dtype: np.ndarray                    # (D0,) 1, 2, 3
    side_a: np.ndarray                   # (D0,) actor
    side_b: np.ndarray
    hist: np.ndarray                     # (D0, HIST) eventos de los últimos HIST meses (el último, el más reciente)
    since: np.ndarray                    # (D0,)
    age: np.ndarray                      # (D0,)
    W0: np.ndarray                       # (D0, S) huella en el origen
    C0: np.ndarray                       # (K, S) huella de toda la violencia de cada país en el origen
    civ_share: np.ndarray                # (D0 + 3,) parte de las muertes que son civiles
    pools: np.ndarray                    # (D0 + 3, POOL) muertes por evento; las 3 últimas filas, por tipo
    # nacimientos
    b_birth: np.ndarray
    templates: dict = field(default_factory=dict)
    gov_actor: np.ndarray = None         # (K,) actor gobierno de cada país o -1
    other_gov: np.ndarray = None         # actores gobierno de fuera de la región
    civ_actor: int = -1
    n_actors: int = 0
    prior: np.ndarray = None             # (K, S)
    actor_kind: np.ndarray = None
    use_l1: bool = False
    names: tuple = FEATURES

    @property
    def D0(self) -> int:
        return len(self.dyads)


def _roles(P: pn.Panel, d: int, home: int, born: int, first_seen: np.ndarray) -> tuple[int, int]:
    """Papel de cada lado de una díada al nacer (índices en ``ROLES``)."""
    out = []
    for a in (P.dyad_a[d], P.dyad_b[d]):
        k = P.actor_kind[a]
        if k == 2:
            out.append(2)
        elif k == 0:
            out.append(0 if P.actor_home[a] == home else 1)
        else:
            out.append(3 if first_seen[a] < born else 4)
    return out[0], out[1]


def fit(P: pn.Panel, t0: int, l1=None, seed: int = 0, foot: dict | None = None,
        mark_halflife: float | None = MARK_HALFLIFE) -> AgentModel:
    """Estima todo con eventos anteriores al mes ``t0``. ``l1`` (K, >= t0 + 1) es opcional. ``foot`` reutiliza
    los parámetros de huella de un origen ANTERIOR (ahorra la búsqueda; nunca los de uno posterior)."""
    h = history_features(P, t0, l1)
    F, n, first, since, home = h["F"], h["n"], h["first"], h["since"], h["home"]
    rng = np.random.default_rng(seed)

    # ---- activación e intensidad: filas (díada, mes) con la díada ya nacida y viva
    tt = np.arange(n.shape[1])
    rows = (tt[None] > first[:, None]) & (tt[None] < t0) & (since[:, :-1] < ALIVE_MONTHS) & (home >= 0)[:, None]
    X, y = F[:, :-1][rows], n[rows]
    use = np.ones(len(FEATURES), bool)
    use[FEATURES.index("l1")] = l1 is not None
    act = (y > 0).astype(float)
    b_act = np.zeros(1 + len(FEATURES))
    b_int = np.zeros(1 + len(FEATURES))
    resid = np.zeros(1)
    sel = np.r_[True, use]
    if 0 < act.sum() < len(act):
        b_act[sel] = stats.logit_fit(X[:, use], act)
        pos = y > 0
        b_int[sel], resid = stats.ols_fit(X[pos][:, use], np.log(np.maximum(y[pos], 0.25)))
    elif len(act):
        b_act[0] = 30.0 if act.all() else -30.0
    n_hist = n[:, :t0]
    n_cap = max(N_CAP * float(n_hist.max(initial=1.0)), 10.0)

    # ---- díadas vivas en el origen y su estado
    alive = np.where((first < t0) & (since[:, t0] < ALIVE_MONTHS) & (home >= 0))[0]
    hist = np.zeros((len(alive), HIST))
    lo = max(t0 - HIST, 0)
    hist[:, HIST - (t0 - lo):] = n[alive, lo:t0]
    foot = fit_footprint(P, t0, home) if foot is None else dict(foot)
    foot.setdefault("mix", 0.0)
    W0 = footprint_at(P, t0, alive, foot["decay"])
    C0 = country_footprint(P, t0, home)

    # ---- letalidad: depósito por díada, encogido hacia el de su tipo
    e = P.ev
    past = e["t0"] < t0
    recent = past & (e["t0"] >= t0 - 120)
    type_pool, type_civ = [], []
    for ty in (1, 2, 3):
        m = recent & (e["type"] == ty)
        if not m.any():
            m = past & (e["type"] == ty)
        b = e["best"][m] if m.any() else np.array([1.0])
        type_pool.append(b)
        type_civ.append(float(e["civ"][m].sum() / max(e["best"][m].sum(), 1.0)) if m.any() else 0.0)
    pools = np.zeros((len(alive) + 3, POOL), np.float32)
    civ_share = np.zeros(len(alive) + 3)
    order = np.argsort(e["dyad"][past], kind="stable")
    dy_sorted = e["dyad"][past][order]
    best_sorted, civ_sorted = e["best"][past][order], e["civ"][past][order]
    age_sorted = (t0 - 1 - e["t0"][past][order]).astype(float)
    lo_i, hi_i = np.searchsorted(dy_sorted, alive, "left"), np.searchsorted(dy_sorted, alive, "right")
    for i, d in enumerate(alive):
        own = best_sorted[lo_i[i]:hi_i[i]][-500:]
        ty = int(P.dyad_type[d]) - 1
        w = len(own) / (len(own) + SHRINK)
        from_own = rng.random(POOL) < w
        pick = rng.integers(0, max(len(own), 1), POOL)
        if mark_halflife is not None and len(own):            # la letalidad reciente pesa más que la antigua
            pw = 0.5 ** (age_sorted[lo_i[i]:hi_i[i]][-500:] / mark_halflife)
            pick = np.minimum(np.searchsorted(np.cumsum(pw) / pw.sum(), rng.random(POOL)), len(own) - 1)
        pools[i] = np.where(from_own, own[pick] if len(own) else 0.0,
                            type_pool[ty][rng.integers(0, len(type_pool[ty]), POOL)])
        tot = best_sorted[lo_i[i]:hi_i[i]].sum()
        own_civ = civ_sorted[lo_i[i]:hi_i[i]].sum() / tot if tot > 0 else type_civ[ty]
        civ_share[i] = w * own_civ + (1 - w) * type_civ[ty]
    for ty in range(3):
        pools[len(alive) + ty] = type_pool[ty][rng.integers(0, len(type_pool[ty]), POOL)]
        civ_share[len(alive) + ty] = type_civ[ty]

    # ---- nacimientos: tasa por país y mes, y plantillas
    born = np.where((first >= CENSOR) & (first < t0) & (home >= 0))[0]
    births = np.zeros((P.K, t0))
    np.add.at(births, (home[born], first[born]), 1.0)
    ev_k = P.to_countries(P.site_events())                                   # (K, T) eventos del país
    yr = np.clip(P.year0 + np.arange(t0) // 12 - int(P.pop_years[0]), 0, len(P.pop_years) - 1)
    bx = np.stack([np.log1p(pn.lagged_mean(ev_k, 12)[:, :t0]), np.log(np.maximum(P.pop[:, yr], 1.0))], -1)
    tb = np.arange(CENSOR, t0)
    if len(tb) >= 12 and births[:, tb].sum() > 0:
        b_birth = stats.poisson_fit(bx[:, tb].reshape(-1, 2), births[:, tb].reshape(-1))
    else:
        b_birth = np.array([-20.0, 0.0, 0.0])
    first_seen = np.full(len(P.actor_kind), 10 ** 9)
    for side in (P.dyad_a, P.dyad_b):
        np.minimum.at(first_seen, side, first)
    tpl = [d for d in born if first[d] >= t0 - BIRTH_WINDOW] or list(born)
    roles = np.array([_roles(P, d, home[d], first[d], first_seen) for d in tpl], int).reshape(-1, 2)
    templates = {"home": home[tpl].astype(int), "type": P.dyad_type[tpl].astype(int), "role_a": roles[:, 0],
                 "role_b": roles[:, 1],
                 "n0": np.maximum(np.round(n[tpl, np.minimum(first[tpl], n.shape[1] - 1)]), 1.0) if tpl else np.zeros(0)}

    gov = np.full(P.K, -1)
    for a in np.where(P.actor_kind == 0)[0]:
        if P.actor_home[a] >= 0:
            gov[P.actor_home[a]] = a
    civ = np.where(P.actor_kind == 2)[0]
    return AgentModel(
        t0=int(t0), b_act=b_act, b_int=b_int, resid=resid, n_cap=n_cap, foot=foot, dyads=alive, home=home[alive],
        dtype=P.dyad_type[alive], side_a=P.dyad_a[alive], side_b=P.dyad_b[alive], hist=hist,
        since=since[alive, t0], age=np.maximum(t0 - first[alive], 0).astype(float), W0=W0, C0=C0, civ_share=civ_share,
        pools=pools, b_birth=b_birth, templates=templates, gov_actor=gov,
        other_gov=np.where((P.actor_kind == 0) & (P.actor_home < 0))[0], civ_actor=int(civ[0]) if len(civ) else -1,
        n_actors=len(P.actor_kind), prior=site_prior(P, t0), actor_kind=P.actor_kind.copy(), use_l1=l1 is not None)


# ---------------------------------------------------------------------- simulación
@dataclass
class AgentRun:
    """Salida de ``simulate``. Todo es (R, ..., H) con H meses desde el origen."""
    t0: int
    site_deaths: np.ndarray              # (R, S, H) float32
    site_civ: np.ndarray                 # (R, S, H) muertes de civiles (esperadas)
    country_type: np.ndarray             # (R, K, H, 3) muertes por país y tipo de violencia
    dyad_events: np.ndarray              # (R, D0 + BIRTH_SLOTS, H)
    births: np.ndarray                   # (R, K, H)
    active_dyads: np.ndarray             # (R, K, H) díadas con algún evento


def _other(tot_actor: np.ndarray, side: np.ndarray, own: np.ndarray, kind: np.ndarray) -> np.ndarray:
    r = np.arange(side.shape[0])[:, None]
    out = tot_actor[r, side] - own
    return np.where(kind[side] == 2, 0.0, np.maximum(out, 0.0))


def simulate(P: pn.Panel, M: AgentModel, horizon: int, members: int = 64, seed: int = 0, l1=None,
             births: bool = True) -> AgentRun:
    """``members`` mundos desde ``M.t0``. ``l1`` (K, horizon) es la covariable del L1 por mes de pronóstico."""
    R, D0, S, K = members, M.D0, P.S, P.K
    Dn = D0 + BIRTH_SLOTS
    rng = np.random.default_rng(seed)
    rep = lambda a, dt=float: np.concatenate([np.repeat(np.asarray(a, dt)[None], R, 0),  # noqa: E731
                                              np.zeros((R, BIRTH_SLOTS), dt)], 1)
    exists = rep(np.ones(D0), bool)
    home, dtype = rep(M.home, int), rep(M.dtype, int)
    side_a, side_b = rep(M.side_a, int), rep(M.side_b, int)
    since, age = rep(M.since), rep(M.age)
    pool_row = rep(np.arange(D0), int)
    hist = np.concatenate([np.repeat(M.hist[None], R, 0), np.zeros((R, BIRTH_SLOTS, HIST))], 1)
    W = np.zeros((R, Dn, S), np.float32)
    W[:, :D0] = M.W0[None]
    CF = np.repeat(M.C0[None].astype(np.float32), R, 0)                       # (R, K, S)
    # los actores nuevos reciben identificadores propios por miembro y hueco (dos por hueco)
    n_act = M.n_actors + 2 * BIRTH_SLOTS
    kind = np.concatenate([M.actor_kind, np.ones(2 * BIRTH_SLOTS, int)])
    next_slot = np.full(R, D0)
    foot = M.foot
    out = AgentRun(t0=M.t0, site_deaths=np.zeros((R, S, horizon), np.float32),
                   site_civ=np.zeros((R, S, horizon), np.float32), country_type=np.zeros((R, K, horizon, 3), np.float32),
                   dyad_events=np.zeros((R, Dn, horizon), np.float32), births=np.zeros((R, K, horizon), np.float32),
                   active_dyads=np.zeros((R, K, horizon), np.float32))
    rr = np.arange(R)[:, None]
    yr = lambda t: int(np.clip(P.ym(t)[0] - int(P.pop_years[0]), 0, len(P.pop_years) - 1))  # noqa: E731

    for h in range(horizon):
        t = M.t0 + h
        # todos los sorteos del mes tienen forma fija: números comunes entre variantes del modelo
        u_act, i_res = rng.random((R, Dn)), rng.integers(0, len(M.resid), (R, Dn))
        u_birth = rng.random((R, K, 4))
        # ---- rasgos del estado simulado
        n3 = hist[:, :, -3:].mean(2)
        tot_actor = np.zeros((R, n_act))
        for side in (side_a, side_b):
            np.add.at(tot_actor, (rr, side), np.where(exists, n3, 0.0))
        by_k = np.zeros((R, K))
        np.add.at(by_k, (rr, home), np.where(exists, n3, 0.0))
        cov = np.zeros((R, Dn)) if l1 is None or not M.use_l1 else np.asarray(l1)[:, h][home]
        X = np.stack([np.log1p(hist[:, :, -1]), np.log1p(n3), np.log1p(hist[:, :, -12:].mean(2)),
                      np.log1p(since), np.log1p(age), np.log1p(_other(tot_actor, side_a, n3, kind)),
                      np.log1p(_other(tot_actor, side_b, n3, kind)), np.log1p(np.maximum(by_k[rr, home] - n3, 0.0)),
                      (dtype == 2) * 1.0, (dtype == 3) * 1.0, ((kind[side_a] == 0) & (kind[side_b] == 0)) * 1.0, cov], -1)
        # ---- combatir o no, y cuánto
        active = exists & (u_act < stats.sigmoid(M.b_act[0] + X @ M.b_act[1:]))
        ln = M.b_int[0] + X @ M.b_int[1:] + M.resid[i_res]
        n = np.where(active, np.clip(np.round(np.exp(np.clip(ln, -5, 12))), 1.0, M.n_cap), 0.0)

        # ---- nacimientos (entran activos este mes con los eventos de su plantilla)
        if births and len(M.templates.get("home", ())):
            ev12 = np.zeros((R, K))
            np.add.at(ev12, (rr, home), np.where(exists, hist[:, :, -12:].mean(2), 0.0))
            lam = np.exp(np.clip(M.b_birth[0] + M.b_birth[1] * np.log1p(ev12)
                                 + M.b_birth[2] * np.log(np.maximum(P.pop[:, yr(t)], 1.0))[None], -30, 3))
            nb = (u_birth[:, :, 0] < 1 - np.exp(-lam)).astype(int)            # a lo sumo una por país y mes
            for r, k in zip(*np.where(nb > 0)):
                j = next_slot[r]
                if j >= Dn:
                    continue
                next_slot[r] += 1
                tk = np.where(M.templates["home"] == k)[0]
                tk = tk if len(tk) >= 5 else np.arange(len(M.templates["home"]))
                q = tk[int(u_birth[r, k, 1] * len(tk))]
                ty = int(M.templates["type"][q])
                sides = []
                for which, role in enumerate((M.templates["role_a"][q], M.templates["role_b"][q])):
                    new_id = M.n_actors + 2 * (j - D0) + which
                    if role == 0:
                        a = M.gov_actor[k] if M.gov_actor[k] >= 0 else new_id
                    elif role == 1:
                        a = M.other_gov[int(u_birth[r, k, 2] * len(M.other_gov))] if len(M.other_gov) else new_id
                    elif role == 2:
                        a = M.civ_actor if M.civ_actor >= 0 else new_id
                    elif role == 3:
                        cand = np.unique(np.concatenate([side_a[r][exists[r] & (home[r] == k)],
                                                         side_b[r][exists[r] & (home[r] == k)]]))
                        cand = cand[kind[cand] == 1]
                        a = cand[int(u_birth[r, k, 3] * len(cand))] if len(cand) else new_id
                    else:
                        a = new_id
                    sides.append(int(a))
                exists[r, j], home[r, j], dtype[r, j] = True, k, ty
                side_a[r, j], side_b[r, j] = sides
                since[r, j], age[r, j], pool_row[r, j] = 0.0, 0.0, D0 + ty - 1
                # semilla espacial [M]: la huella del grupo que ya existía, o la del país mezclada con la población
                mine = exists[r] & ((side_a[r] == sides[0]) | (side_b[r] == sides[0])) & (np.arange(Dn) != j)
                seed_w = W[r, mine].sum(0) if (M.templates["role_a"][q] == 3 and mine.any()) else \
                    W[r, exists[r] & (home[r] == k)].sum(0)
                tot = float(seed_w.sum())
                W[r, j] = (0.5 * seed_w / tot + 0.5 * M.prior[k]) if tot > 0 else M.prior[k]
                n[r, j] = min(float(M.templates["n0"][q]), M.n_cap)
                active[r, j] = True
                out.births[r, k, h] += 1.0

        # ---- letalidad, lugar y actualización de la huella (sólo las díadas activas)
        ar, aj = np.where(active)
        if len(ar):
            na = n[ar, aj].astype(np.int64)
            ev_row = np.repeat(np.arange(len(ar)), na)
            marks = M.pools[pool_row[ar, aj][ev_row], rng.integers(0, POOL, len(ev_row))]
            deaths = np.bincount(ev_row, weights=marks, minlength=len(ar))
            w = spread(W[ar, aj].astype(np.float64), P.site_nbr, foot["diffuse"])
            tot = w.sum(1, keepdims=True)
            hk = home[ar, aj]
            prior = M.prior[hk]
            if foot["mix"] > 0:
                cf = CF[ar, hk].astype(np.float64)
                ctot = cf.sum(1, keepdims=True)
                prior = np.where(ctot > 0, foot["mix"] * cf / np.maximum(ctot, 1e-300) + (1 - foot["mix"]) * prior, prior)
            p = np.where(tot > 0, (1 - foot["floor"]) * w / np.maximum(tot, 1e-300) + foot["floor"] * prior, prior)
            p = p / p.sum(1, keepdims=True)
            counts = rng.multinomial(na, p)                                   # (A, S)
            d_int = np.round(deaths).astype(np.int64)
            d_site = rng.multinomial(d_int, counts / np.maximum(na[:, None], 1)).astype(np.float32)
            np.add.at(out.site_deaths[:, :, h], (ar,), d_site)
            np.add.at(out.site_civ[:, :, h], (ar,), d_site * M.civ_share[pool_row[ar, aj]][:, None].astype(np.float32))
            np.add.at(out.country_type[:, :, h, :], (ar[:, None], P.site_country[None, :],
                                                      (dtype[ar, aj] - 1)[:, None]), d_site)
            np.add.at(out.active_dyads[:, :, h], (ar, home[ar, aj]), 1.0)
            W *= foot["decay"]
            W[ar, aj] += counts.astype(np.float32)
            CF *= COUNTRY_DECAY
            np.add.at(CF, (ar, hk), counts.astype(np.float32))
        else:
            W *= foot["decay"]
            CF *= COUNTRY_DECAY
        out.dyad_events[:, :, h] = n

        # ---- memoria, edad y muerte de las díadas calladas
        hist = np.concatenate([hist[:, :, 1:], n[:, :, None]], 2)
        since = np.where(active, 1.0, since + 1.0)
        age = age + exists
        exists &= since < ALIVE_MONTHS
    return out
