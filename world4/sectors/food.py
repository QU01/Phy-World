"""F5: alimentos, suelo, nitrógeno y agua por país (180 países del L1), por miembro del ensemble.

Ecuaciones estimadas con datos <= origen (``_panel.fit``: MCO agrupado ponderado por el tamaño del
país, covarianza por país, choques AR(1) comunes y propios), en diferencias:

    nfert     Δln n = b1 gapN₋₁ + b2 Δly₋₁ + b3 Δln n₋₁, SIN constante     (n: N mineral por ha de cultivo)
              gapN = ln n del percentil 90 entre países EN EL ORIGEN - ln n: la intensidad converge a la
              frontera del origen y no la rebasa [S]. Con constante libre, los datos de 1961-1990 (la
              Revolución Verde) daban +4 % anual también en la frontera y el hindcast desde 1990 llegaba a
              300 Tg de N intencional en 2023 (real ~165): la forma se cambió DESPUÉS de ver ese hindcast
              [declarado en el informe de F5].
    yield     Δln Y = c + b1 gap₋₁ + b2 Δln n + b3 ΔT + b4 ΔT·(T̄-15)/10 + b5 Δln Y₋₁
              gap = ln Y del percentil 90 entre países - ln Y (convergencia); ΔT = cambio de la anomalía
              de temperatura local del año (el clima del L1, que con F4 es FaIR) [signos: gap +, n +]
    area      Δln A = c + b1 Δln pob₋₁ + b2 Δln Y₋₁ + b3 Δln A₋₁            (superficie de cereales)
    cropland  Δln C = c + b1 Δln A + b2 Δln C₋₁
    pasture   Δln P = c + b1 Δln P₋₁ + b2 Δln pob₋₁
    forest    ΔF/L = c + b1 (ΔC + ΔP)/L + b2 ΔF₋₁/L₋₁       (sólo con >= 8 años de bosque FRA, desde 1990)
    kcal      Δln k = c + b1 ln k₋₁ + b2 Δly₋₁ + b3 Δln(cereal propio per cápita) + b4 Δln k₋₁
              (la cosecha mundial per cápita no se identifica, b ≈ 0; la propia del mismo año sí: el
              comercio no amortigua del todo una mala cosecha) [signo b3 +]
    irrig     Δln I = c + b1 Δln C + b2 Δln I₋₁
    ef_nc     Δln EFnc = c + b1 Δln k + b2 Δln(C/pob)        (huella sin carbono, GFN)
    biocap    Δln BC = c + b1 (Δln Y + Δln C) + b2 Δln P      (biocapacidad total, GFN)

Cuentas (W7):
    * suelo: cultivo + pastos + bosque + resto = superficie de tierra (el resto absorbe; si quedara
      negativo, se recortan los pastos) — cerrado por construcción;
    * N en tierras de cultivo (balance de FAOSTAT): entradas = mineral + estiércol + fijación +
      deposición; salidas = cosecha + residuos + semilla; balance = entradas - salidas. El mineral sale
      de la ecuación; estiércol, fijación y deposición por ha quedan en su valor del origen [S]; las
      salidas siguen a la cosecha (cultivo × rendimiento relativo) [S].

Huella y biocapacidad (cierra D4 del Atlas, sin MRIO): huella de consumo = parte de carbono (GFN
``ef_carbon_pc``, proporcional al CO2, que aquí sigue al fósil per cápita de F3: la huella de carbono
de GFN es lineal en el CO2) + parte sin carbono (ecuación ``ef_nc``); biocapacidad con ``biocap``. Ni
EXIOBASE (no comercial) ni GLORIA (licencia sin verificar): la huella sale de los flujos físicos del
propio modelo, que es una de las opciones de la hoja de ruta.

Límites planetarios (indicadores, [S]): N intencional = N mineral + fijación biológica en cultivos frente
a 62 Tg N/año (Steffen et al. 2015); extracción agrícola de agua (WDI/AQUASTAT; riego × agua por ha
de riego del origen) frente a 4000 km³/año de uso consuntivo (Rockström et al. 2009): la extracción NO
es uso consuntivo; se informa como cota superior.
"""
from __future__ import annotations

import numpy as np

from world4.sectors import _panel

KCAL_MIN = 2300.0            # umbral de déficit calórico para la mortalidad de F1 [S]
N_BOUNDARY_TG = 62.0         # Steffen et al. (2015) [S]
WATER_BOUNDARY_KM3 = 4000.0  # Rockström et al. (2009), uso consuntivo [S]
FOREST_MIN_YEARS = 8
SIGNS = {"yield": {"gap": 1, "dln_n": 1}, "biocap": {"dlnYC": 1, "dlnP": 1},
         "ef_nc": {"dln_kcal": 1}, "kcal": {"dly": 1, "dln_own_cereal_pc": 1}, "forest": {"d_agri": -1}}


def _lag(a, k=1):
    out = np.full_like(a, np.nan, dtype=float)
    out[..., k:] = a[..., :-k]
    return out


def _ln(a):
    a = np.asarray(a, float)
    return np.log(np.where(a > 0, a, np.nan))


_PANEL = {}


def panel(W, LY=None) -> dict:
    """Series (N, T) en los años del L1."""
    key = id(W)
    if key in _PANEL:
        return _PANEL[key]
    from world4.data import food as fdata
    from world4.sectors import energy, population
    a = fdata.aligned(W.iso3, W.isonum, W.years)
    LY = np.log(W.Y / W.pop) if LY is None else LY
    C = a["cropland"]
    n = a["n_fert"] / np.where(C > 0, C * 1000.0, np.nan)            # t N por ha
    lnN = np.log(n * 1000.0 + 1.0)                                     # ln(kg N/ha + 1)
    Y = a["cereal_prod"] / np.where(a["cereal_area"] > 0, a["cereal_area"], np.nan)
    lY = _ln(Y)
    q90 = np.array([np.nanpercentile(lY[:, t], 90) if np.isfinite(lY[:, t]).sum() > 20 else np.nan
                    for t in range(W.T)])
    pp = population.panel(W, LY)
    tmp = W.locf("tmp_level")
    tbar = pp["tbar"]
    tdev = tmp - tbar[:, None]
    pop = np.where(W.observed, W.pop, np.nan)
    world_cpc = np.nansum(np.where(np.isfinite(a["cereal_prod"]) & np.isfinite(pop), a["cereal_prod"], 0), 0) / \
        np.nansum(np.where(np.isfinite(a["cereal_prod"]) & np.isfinite(pop), pop, 0), 0)
    world_cpc = np.where(np.isfinite(a["cereal_prod"]).sum(0) > 50, world_cpc, np.nan)
    ep = energy.panel(W)
    fossil_pc = ep["src"]["src_fossil"] / pop
    efc = W.locf("ef_carbon_pc") if "ef_carbon_pc" in W.extra else np.full((W.N, W.T), np.nan)
    ef = W.extra.get("ef_cons_pc", np.full((W.N, W.T), np.nan))
    bc = W.extra.get("biocap_pc", np.full((W.N, W.T), np.nan))
    ef_nc = ef - np.where(np.isfinite(W.extra.get("ef_carbon_pc", np.full((W.N, W.T), np.nan))),
                          W.extra.get("ef_carbon_pc", np.full((W.N, W.T), np.nan)), np.nan)
    qN90 = np.array([np.nanpercentile(lnN[:, t], 90) if np.isfinite(lnN[:, t]).sum() > 20 else np.nan
                     for t in range(W.T)])
    out = {"src": a, "lnN": lnN, "qN90": qN90, "lY": lY, "gap": q90[None] - lY, "q90": q90, "lA": _ln(a["cereal_area"]),
           "lC": _ln(C), "lP": _ln(a["pasture"]), "F": a["forest"], "L": a["land"], "lk": _ln(a["kcal"]),
           "lI": _ln(a["irrigated"]), "lpop": _ln(pop), "ly": LY, "tdev": tdev, "tbar": tbar,
           "lwcpc": _ln(world_cpc), "lfossil_pc": _ln(fossil_pc), "ef": ef, "ef_carbon": efc,
           "lef_nc": _ln(ef_nc), "lbc_tot": _ln(bc * pop), "pop": pop}
    _PANEL[key] = out
    return out


def _stack(cols, t_lo, t_hi):
    sl = slice(t_lo, t_hi + 1)
    N = cols[0].shape[0]
    n_t = t_hi - t_lo + 1
    return [c[:, sl].ravel() for c in cols], np.repeat(np.arange(N), n_t), np.tile(np.arange(t_lo, t_hi + 1), N)


def fit(W, LY, cutoff_t: int) -> dict:
    """Ecuaciones de F5 con datos <= ``cutoff_t`` (FAOSTAT empieza en 1961)."""
    p = panel(W, LY)
    N, T = W.N, W.T
    t_lo = 1963 - 1950
    cy = 1950 + cutoff_t
    rep = lambda v: np.broadcast_to(v, (N, T))  # noqa: E731
    d = lambda a: a - _lag(a)  # noqa: E731
    eqs = {}

    def go(name, y, regs, terms, weight, t_min=t_lo):
        cols, country, year = _stack([y] + regs + [weight], t_min, cutoff_t)
        X = np.column_stack([np.ones_like(cols[0])] + cols[1:-1])
        eqs[name] = _panel.fit(name, cols[0], X, country, ("const",) + terms, cy, signs=SIGNS.get(name),
                               n_countries=N, recent=year > cutoff_t - 10, year=year, weights=cols[-1])

    s = p["src"]
    w_prod, w_c, w_pop = _lag(s["cereal_prod"]), _lag(s["cropland"]), _lag(p["pop"])
    dN, dY, dA, dC, dP, dk = d(p["lnN"]), d(p["lY"]), d(p["lA"]), d(p["lC"]), d(p["lP"]), d(p["lk"])
    dly = d(p["ly"])
    dT = d(p["tdev"])
    qN = p["qN90"][cutoff_t]
    cols, country, year = _stack([dN, qN - _lag(p["lnN"]), _lag(dly), _lag(dN), w_c], t_lo, cutoff_t)
    eqs["nfert"] = _panel.fit("nfert", cols[0], np.column_stack(cols[1:-1]), country, ("gapN", "dly", "dlnN"), cy,
                              signs={"gapN": 1}, n_countries=N, recent=year > cutoff_t - 10, year=year,
                              weights=cols[-1])
    go("yield", dY, [_lag(p["gap"]), dN, dT, dT * (p["tbar"][:, None] - 15.0) / 10.0, _lag(dY)],
       ("gap", "dln_n", "dT", "dT_hot", "dlnY"), w_prod)
    go("area", dA, [_lag(d(p["lpop"])), _lag(dY), _lag(dA)], ("dlnpop", "dlnY", "dlnA"), _lag(s["cereal_area"]))
    go("cropland", dC, [dA, _lag(dC)], ("dlnA", "dlnC"), w_c)
    go("pasture", dP, [_lag(dP), _lag(d(p["lpop"]))], ("dlnP", "dlnpop"), _lag(s["pasture"]))
    own = d(_ln(s["cereal_prod"] / p["pop"]))
    go("kcal", dk, [_lag(p["lk"]), _lag(dly), own, _lag(dk)], ("lnk", "dly", "dln_own_cereal_pc", "dlnk"), w_pop)
    go("irrig", d(p["lI"]), [dC, _lag(d(p["lI"]))], ("dlnC", "dlnI"), _lag(s["irrigated"]))
    go("ef_nc", d(p["lef_nc"]), [dk, d(p["lC"] - p["lpop"])], ("dln_kcal", "dln_crop_pc"), w_pop)
    go("biocap", d(p["lbc_tot"]), [dY + dC, dP], ("dlnYC", "dlnP"), w_pop)
    # bosque: sólo hay FRA desde 1990
    L = p["L"]
    dF = d(p["F"]) / L
    agri = (d(s["cropland"]) + d(s["pasture"])) / L
    n_forest_years = max(0, cutoff_t - (1991 - 1950) + 1)
    if n_forest_years >= FOREST_MIN_YEARS:
        go("forest", dF, [agri, _lag(dF)], ("d_agri", "dF_lag"), _lag(L), t_min=1992 - 1950)
    eqs["meta"] = {"forest": "forest" in eqs, "cutoff": cy}
    return eqs


class Food:
    """Estado (R, N) por país y miembro. ``step(t, ...)`` avanza el año ``t`` con el L1 de ese año."""

    def __init__(self, W, LY, t0: int, eqs: dict, R: int, seed: int, noise: bool = True):
        self.W, self.t0, self.R, self.eqs, self.noise = W, t0, R, eqs, noise
        self.rng = np.random.default_rng(seed)
        p = panel(W, LY)
        self.p = p
        s = p["src"]
        N = W.N
        rep = lambda v: np.broadcast_to(v, (R, N)).copy()  # noqa: E731
        at = lambda a: a[:, t0]  # noqa: E731
        d0 = lambda a: np.clip(np.nan_to_num(a[:, t0] - a[:, t0 - 1]), -0.3, 0.3)  # noqa: E731  (saltos de dato acotados)
        self.ok = np.isfinite(at(p["lY"])) & np.isfinite(at(p["lC"])) & np.isfinite(at(p["lk"]))
        self.lY, self.lA, self.lC = rep(at(p["lY"])), rep(at(p["lA"])), rep(at(p["lC"]))
        lnN0 = at(p["lnN"])
        prev = np.nanmax(p["lnN"][:, max(0, t0 - 3):t0], 1) if t0 > 0 else lnN0
        # un 0 aislado en el origen (Mongolia 2019) es un fallo de registro: se toma el último valor > 0 de 3 años
        self.lnN = rep(np.nan_to_num(np.where((lnN0 > 0) | ~np.isfinite(prev), lnN0, prev)))
        self.lP = rep(np.where(np.isfinite(at(p["lP"])), at(p["lP"]), -np.inf))
        self.lk = rep(at(p["lk"]))
        self.lI = rep(np.where(np.isfinite(at(p["lI"])), at(p["lI"]), -np.inf))
        self.L = np.where(np.isfinite(at(p["L"])), at(p["L"]), np.nan)
        F0 = at(p["F"])
        self.forest_on = bool(eqs["meta"]["forest"])
        self.F = rep(np.where(np.isfinite(F0), F0, np.nan))
        self.dY, self.dA, self.dC, self.dP = rep(d0(p["lY"])), rep(d0(p["lA"])), rep(d0(p["lC"])), rep(d0(p["lP"]))
        self.dN, self.dk, self.dI = rep(d0(p["lnN"])), rep(d0(p["lk"])), rep(d0(p["lI"]))
        self.dF = rep(np.nan_to_num((F0 - p["F"][:, t0 - 1]) / self.L))
        self.tdev_prev = rep(np.nan_to_num(at(p["tdev"])))
        self.tbar = p["tbar"]
        self.lpop0 = at(p["lpop"])
        self.B = {k: (e.draw(R, self.rng) if noise else np.broadcast_to(e.beta, (R, len(e.beta))).copy())
                  for k, e in eqs.items() if k != "meta"}
        self.shock = {k: _panel.shocks(e, R, N, self.rng, on=noise) for k, e in eqs.items() if k != "meta"}
        # frontera de rendimiento: la del origen, que crece con la tendencia de los 20 años previos
        q = p["q90"][max(0, t0 - 19):t0 + 1]
        okq = np.isfinite(q)
        self.q90_0 = float(p["q90"][t0])
        self.qN0 = float(p["qN90"][t0])
        self.q90_g = float(np.polyfit(np.arange(okq.sum()), q[okq], 1)[0]) if okq.sum() > 5 else 0.0
        # N por ha en el origen (t N/ha) para estiércol, fijación, deposición y semilla; salidas por cosecha
        C0 = np.where(self.ok, np.exp(at(p["lC"])) * 1000.0, np.nan)        # ha
        self.nb0 = {k: np.nan_to_num(s[f"nb_{k}"][:, t0] / C0) for k in
                    ("manure", "fixation", "deposition")}
        self.out0 = {k: np.nan_to_num(s[f"nb_{k}"][:, t0]) for k in ("removal", "residue")}
        self.harvest0 = np.exp(at(p["lC"]) + at(p["lY"]))
        # agua por ha equipada de riego en el origen (la última extracción agrícola <= origen)
        wag = s.get("water_ag_km3")
        last = np.full(N, np.nan)
        if wag is not None:
            for t in range(t0 + 1):
                last = np.where(np.isfinite(wag[:, t]), wag[:, t], last)
        I0 = np.exp(at(p["lI"]))
        self.water_per_irr = np.where(np.isfinite(last) & (I0 > 0), last / I0, np.nan)    # km3 por 1000 ha
        # huella (parte de carbono y sin carbono) y biocapacidad en el origen
        locf = lambda a: np.array([a[i, :t0 + 1][np.isfinite(a[i, :t0 + 1])][-1]  # noqa: E731
                                   if np.isfinite(a[i, :t0 + 1]).any() else np.nan for i in range(N)])
        ef0, efc0 = locf(p["ef"]), locf(np.where(np.isfinite(p["ef_carbon"]), p["ef_carbon"], np.nan))
        efc0 = np.where(np.isfinite(efc0), np.minimum(efc0, ef0), 0.0)
        self.ef0_c, self.ef0_nc = rep(efc0), rep(ef0 - efc0)
        self.bc0 = rep(locf(np.where(np.isfinite(p["lbc_tot"]), np.exp(p["lbc_tot"]), np.nan)))
        self.lef_nc, self.lbc = rep(np.zeros(N)), rep(np.zeros(N))
        self.fossil_pc0 = np.exp(at(p["lfossil_pc"]))
        self.ef_on = np.isfinite(ef0)
        self.hist = {t0: self._snapshot(None)}

    # -------------------------------------------------------------- un año
    def _noise(self, k):
        if k not in self.shock:
            return 0.0
        common, own = self.shock[k]
        return common() + own()

    def _lin(self, k, cols):
        X = np.stack(cols, -1)
        return np.einsum("rnk,rk->rn", np.nan_to_num(X), self.B[k]) + self.eqs[k].offset + self._noise(k)

    def step(self, t, ly, dly, tloc, pop, fossil_pc=None):
        """Año ``t`` con el L1 de ese año: ``ly``, ``dly`` (R, N), temperatura local ``tloc`` (R, N),
        población ``pop`` (R, N) y, si hay F3, el fósil per cápita (R, N) para la huella de carbono."""
        R, N = self.R, self.W.N
        one = np.ones((R, N))
        ly = np.where(np.isfinite(ly), ly, np.nan)
        dly = np.nan_to_num(dly)
        lpop = np.log(np.maximum(pop, 1e-9))
        dlpop = np.nan_to_num(lpop - getattr(self, "_lpop_prev", np.broadcast_to(self.lpop0, (R, N))))
        self._lpop_prev = lpop
        tdev = np.where(np.isfinite(tloc), tloc - self.tbar[None], self.tdev_prev)
        dT = tdev - self.tdev_prev
        self.tdev_prev = tdev
        # N mineral
        dN = np.clip(self._lin("nfert", [self.qN0 - self.lnN, dly, self.dN]), -0.5, 0.5)
        self.lnN, self.dN = np.maximum(self.lnN + dN, 0.0), dN
        # rendimiento
        q90 = self.q90_0 + self.q90_g * (t - self.t0)
        gap = q90 - self.lY
        dY = np.clip(self._lin("yield", [one, gap, dN, dT, dT * (self.tbar[None] - 15.0) / 10.0, self.dY]), -0.5, 0.5)
        dA = np.clip(self._lin("area", [one, self._dlpop_prev if hasattr(self, "_dlpop_prev") else dlpop, self.dY,
                                        self.dA]), -0.3, 0.3)
        self._dlpop_prev = dlpop
        self.lY, self.dY = self.lY + dY, dY
        self.lA, self.dA = self.lA + dA, dA
        # tierras
        C_old = np.exp(self.lC)
        P_old = np.exp(self.lP)
        dC = np.clip(self._lin("cropland", [one, dA, self.dC]), -0.2, 0.2)
        dP = np.clip(self._lin("pasture", [one, self.dP, dlpop]), -0.2, 0.2)
        self.lC, self.dC = self.lC + dC, dC
        self.lP, self.dP = self.lP + dP, dP
        L = self.L[None]
        C, P = np.exp(self.lC), np.exp(self.lP)
        P = np.where(np.isfinite(L), np.minimum(P, np.maximum(L - C - np.nan_to_num(self.F), 0.0)), P)
        self.lP = np.log(np.maximum(P, 1e-12))
        if self.forest_on:
            agri = ((C - C_old) + (P - np.nan_to_num(P_old))) / L
            dF = np.clip(self._lin("forest", [one, agri, self.dF]), -0.05, 0.05)
            F = np.clip(self.F + dF * L, 0.0, np.maximum(L - C - P, 0.0))
            self.dF = np.where(np.isfinite(self.F), (F - self.F) / L, 0.0)
            self.F = np.where(np.isfinite(self.F), F, self.F)
        # riego
        dI = np.clip(self._lin("irrig", [one, dC, self.dI]), -0.2, 0.2)
        self.lI, self.dI = self.lI + dI, dI
        # calorías: con la cosecha propia per cápita del año
        down = dA + dY - dlpop
        dk = np.clip(self._lin("kcal", [one, self.lk, np.broadcast_to(dly, (R, N)), down, self.dk]), -0.2, 0.2)
        self.lk, self.dk = self.lk + dk, dk
        # huella sin carbono y biocapacidad
        dcrop_pc = dC - dlpop
        self.lef_nc = self.lef_nc + np.clip(self._lin("ef_nc", [one, dk, dcrop_pc]), -0.2, 0.2)
        self.lbc = self.lbc + np.clip(self._lin("biocap", [one, dY + dC, dP]), -0.2, 0.2)
        self.fossil_pc = fossil_pc
        self.pop = pop
        self.hist[t] = self._snapshot(pop)
        return self.hist[t]

    # -------------------------------------------------------------- salidas
    def production(self):
        return np.where(self.ok[None], np.exp(self.lA + self.lY), np.nan)        # t de cereales

    def n_budget(self):
        """Balance de N de las tierras de cultivo (R, N), t N; cerrado por construcción."""
        C = np.exp(self.lC) * 1000.0
        mineral = (np.exp(self.lnN) - 1.0) / 1000.0 * C
        inputs = {"mineral": mineral, **{k: v[None] * C for k, v in self.nb0.items()}}
        harvest = np.exp(self.lC + self.lY)
        rel = harvest / np.where(self.harvest0 > 0, self.harvest0, np.nan)[None]
        outputs = {k: v[None] * np.nan_to_num(rel) for k, v in self.out0.items()}
        tin, tout = sum(inputs.values()), sum(outputs.values())
        ok = self.ok[None]
        return {"inputs": {k: np.where(ok, v, np.nan) for k, v in inputs.items()},
                "outputs": {k: np.where(ok, v, np.nan) for k, v in outputs.items()},
                "input": np.where(ok, tin, np.nan), "output": np.where(ok, tout, np.nan),
                "balance": np.where(ok, tin - tout, np.nan)}

    def land(self):
        """Cultivo, pastos, bosque y resto (R, N), miles de ha; suman la superficie de tierra."""
        L = np.broadcast_to(self.L[None], (self.R, self.W.N))
        C, P = np.exp(self.lC), np.exp(self.lP)
        F = np.nan_to_num(self.F)
        return {"cropland": C, "pasture": P, "forest": F, "other": L - C - P - F, "land": L}

    def footprint(self):
        """Huella de consumo y biocapacidad per cápita (R, N), hectáreas globales: el D4 del Atlas."""
        if self.fossil_pc is not None:
            rc = self.fossil_pc / np.where(self.fossil_pc0 > 0, self.fossil_pc0, np.nan)[None]
            rc = np.where(np.isfinite(rc), rc, 1.0)
        else:
            rc = 1.0
        ef = self.ef0_c * rc + self.ef0_nc * np.exp(self.lef_nc)
        pop0 = np.exp(self.lpop0)[None]
        bc = self.bc0 * np.exp(self.lbc) / np.maximum(self.pop if self.pop is not None else pop0, 1e-9)
        return {"ef_cons_pc": np.where(self.ef_on[None], ef, np.nan), "biocap_pc": bc}

    def water(self):
        return np.exp(self.lI) * self.water_per_irr[None]                       # km3 de extracción agrícola

    def world(self):
        """Indicadores mundiales (R,) de los países simulados."""
        nb = self.n_budget()
        ok = self.ok[None]
        k = np.exp(self.lk)
        pop = self.pop if self.pop is not None else np.exp(self.lpop0)[None]
        wpop = np.where(ok & np.isfinite(k), pop, 0.0)
        return {"cereal_prod": np.nansum(self.production(), 1), "cropland": np.nansum(np.where(ok, np.exp(self.lC), 0), 1),
                "kcal": np.nansum(np.where(wpop > 0, k * wpop, 0), 1) / np.maximum(wpop.sum(1), 1e-9),
                "food_kcal_total": np.nansum(np.where(wpop > 0, k * wpop, 0), 1),
                "n_input": np.nansum(nb["input"], 1), "n_balance": np.nansum(nb["balance"], 1),
                "n_intentional_tg": (np.nansum(nb["inputs"]["mineral"], 1) + np.nansum(nb["inputs"]["fixation"], 1)) / 1e6,
                "water_ag_km3": np.nansum(np.where(ok, self.water(), 0), 1),
                "forest": np.nansum(np.where(ok, np.nan_to_num(self.F), 0), 1)}

    def _snapshot(self, pop):
        self.pop = pop if pop is not None else getattr(self, "pop", None)
        if not hasattr(self, "fossil_pc"):
            self.fossil_pc = None
        return {"prod": self.production(), "cropland": np.where(self.ok[None], np.exp(self.lC), np.nan),
                "kcal": np.where(self.ok[None], np.exp(self.lk), np.nan), "world": self.world(),
                "forest": np.where(self.ok[None], self.F, np.nan)}

    def shortfall(self):
        """Déficit calórico (R, N) para la mortalidad de F1: max(0, ln KCAL_MIN - ln k)."""
        return np.where(self.ok[None], np.maximum(np.log(KCAL_MIN) - self.lk, 0.0), 0.0)
