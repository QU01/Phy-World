"""Hybrid agent-based model: World-Systems + Structural-Demographic + Metaethnic Frontier.

Every country is an agent. Each year it produces output with a Cobb-Douglas technology
    Y = A * K^alpha * (hc * L)^(1-alpha)
plus a natural-resource input R (share psi_t), bought on a world market, and splits a
discretionary budget (share of GDP) across six channels:

    k  domestic capital accumulation
    r  technology / industrial upgrading (R&D, industrial policy) -> stochastic tech jumps
    m  buy high-value goods from the core (quality consumption + embodied tech, but pays
       a monopoly rent to core exporters; the part produced by one's own industry stays home)
    x  extraction capital for the country's own natural resources
    f  invest abroad in other countries' natural resources (concessions; profits repatriate)
    w  redistribution to the masses (lowers elite capture and mass mobilisation potential)

Natural resources
    * Each country has ultimate recoverable resources U. Its gross extraction capacity follows a
      logistic (Hubbert) curve in cumulative extraction X:  Cap = r_R * U * (x + eps) * (1 - x),
      x = X / U: it rises while the deposit is being opened up and falls as it is depleted.
    * Actual extraction = Cap * (1 - exp(-e)), e = extraction capital / (c_R * Cap). Domestic
      and foreign extraction capital compete for the same capacity: foreign concessions take
      their share of the output, speed up depletion (x rises faster) and so shrink the host's
      own gross capacity. The host keeps a royalty; the rest of the rent is repatriated.
    * Resources enter production (Y = A K^alpha R^psi (hL)^(1-alpha-psi)); a world price clears
      the market. psi_t is the observed world rent share (Cobb-Douglas pins it), so the model
      explains the cross-country distribution of rents, not the global oil cycles. Countries that must import resources pay a markup zeta on the imported
      share, so running out of one's own resources slows growth unless one secures supply
      abroad (investing in foreign extraction counts as owned supply).
    * Elites capture a larger share phi_R of resource income (resource curse).

World-systems layer (Wallerstein / Emmanuel / Amin)
    * "coreness" c_i in (0,1) is a smooth function of relative productivity.
    * Unequal exchange: the periphery loses a share tau of its traded output, which accrues
      to core exporters in proportion to their technological weight c_j^2 * Y_j.
    * Monopoly rents mu on high-value goods bought from the core.
    * FDI goes to hosts with high marginal product of capital, and repatriated profits
      flow back to the owners.
Structural-demographic layer (Goldstone / Turchin)
    * Elites capture a share E of output plus a share phi of all rents flowing into the country.
    * Elite numbers n_e grow when income per elite is high relative to the masses (elite
      overproduction). Stagnating mass income raises mass mobilisation potential (MMP).
    * PSI = MMP * EMP * SFD drives the hazard of internal instability (civil conflict).
      Conflict destroys output and prunes the elite.
Metaethnic-frontier layer (Turchin)
    * Collective solidarity (asabiya) S grows logistically where there is an external
      frontier: exposure to a core that drains the country (open periphery, rents extracted).
      It decays in rich cores, and faster under elite overproduction.
    * Asabiya raises state capacity: investment efficiency, tech-jump success and
      resistance to instability.
Technology
    * Deterministic diffusion toward the frontier through trade and FDI (Gerschenkron).
    * Stochastic jumps: Poisson hazard rising with upgrading effort and asabiya;
      jump size rises with the distance to the frontier.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
ALPHA = 0.35
CHANNELS = ["k", "r", "m", "x", "f", "w"]
FIN_CH = ("k", "r", "m", "x", "f")   # uses of saving / foreign exchange (w is a domestic transfer)
CHANNEL_NAMES = {"k": "Capital doméstico", "r": "Tecnología/industria propia",
                 "m": "Importar bienes del centro", "x": "Extracción propia de recursos",
                 "f": "Invertir fuera en recursos", "w": "Redistribución"}
EPS_X = 0.01


@dataclass
class Params:
    # technology
    g0: float = 0.008          # frontier drift of log TFP per year
    kappa: float = 0.015       # diffusion speed toward frontier (per unit gap)
    kappa_m: float = 0.10      # diffusion from embodied tech in manufactured imports (Coe-Helpman)
    kappa_f: float = 0.064     # diffusion from FDI inflows (estimated in docs/research/comercio.md)
    phi_h: float = 1.0         # absorptive capacity: diffusion scales with (hc / hc_frontier)^phi_h
                               # (Borensztein, De Gregorio & Lee 1998; Nelson-Phelps)
    lam0: float = 0.60         # tech-jump hazard scale
    j0: float = 0.04           # mean jump size (log TFP), scaled by (1+gap)
    sig_a: float = 0.020       # idiosyncratic TFP noise
    aid_eff: float = 0.35      # growth effect of ODA, 0.35*a*(1-a/0.3) (Galiani et al. 2017)
    # world-system flows (bilateral trade network, COW 1950-2014)
    theta_d: float = 0.15      # unequal-exchange drain scale on price-level gaps (Hickel et al.)
    beta_px: float = 0.20      # price level ~ (relative income)^beta (Penn effect, estimated)
    tau: float = 0.03          # kept for ablations: 0 switches the drain off
    mu: float = 0.25           # monopoly rent share on high-value core goods
    core_q: float = 0.80       # coreness threshold quantile of world productivity
    core_s: float = 0.35       # smoothness of coreness (log points)
    # structural-demographic (docs/research/elites.md)
    phi_extra: float = 0.075   # extra capture of incoming rents in autocracies: phi = E + x(1-p)
    phiR_extra: float = 0.15   # extra capture of resource rents: phi_R = E + x(1-p)
    beta_e: float = 0.05       # elite-overproduction speed
    eps_r: float = 0.50        # rents -> elite share
    eps_w: float = 0.60        # redistribution -> lower elite share
    eps_0: float = 0.05        # reversion of elite share toward its entry level
    kappa_mmp: float = 1.0     # MMP exponent on falling mass income
    bY: float = 0.035          # youth bulge (15-24 / 15+, per pp above 28%) in MMP (Urdal 2006)
    b0: float = -3.35          # conflict onset intercept
    b1: float = 0.6            # onset sensitivity to log PSI
    b2: float = -0.43          # onset sensitivity to log income
    b3: float = -1.0           # asabiya protects against instability
    b4: float = 2.5            # resource rents / GDP raise onset risk (Collier-Hoeffler)
    b5: float = 1.0            # partial democracies (anocracy 4p(1-p)) are most unstable (PITF)
    bp: float = 2.0            # persistence logit of an ongoing conflict
    gam_c: float = 0.04        # output loss from conflict
    gam_a: float = 0.010       # TFP loss per conflict year
    gam_cl: float = 0.05       # temperature anomaly (per SD) in the conflict logit (BHM 2015b)
    # metaethnic frontier / asabiya (docs/research/demografia.md)
    r_a: float = 0.02          # annual asabiya growth at the frontier
    d_a: float = 0.007         # annual asabiya decay in rich cores
    s0: float = 0.50           # initial asabiya when no proxy is observed
    # natural resources (docs/research/recursos.md)
    r_R: float = 0.075         # Hubbert steepness of the logistic extraction capacity
    x0: float = 0.02           # depleted fraction at entry when no physical data exist
    e0: float = 1.2            # initial extraction effort (capital / capacity)
    c_R: float = 3.0           # extraction capital per unit of capacity
    d_R: float = 0.06          # depreciation of extraction capital
    zeta: float = 0.15         # markup paid on imported resources
    roy: float = 0.30          # kept for ablations (1 = host keeps all rents); era path below
    phi_R: float = 0.70        # kept for ablations
    nu_a: float = 0.80         # asabiya resists foreign concessions (resource nationalism)
    nat0: float = -4.0         # nationalisation hazard intercept (Guriev et al. 2011 style)
    nat_shock: float = 1.54    # 3-year change in log real resource price
    nat_level: float = -1.08   # log real price level
    nat_dem: float = -0.061    # polity scale (-10..10)
    nat_post85: float = -1.55  # post-1985 regime
    nat_theta: float = 0.75    # share of foreign extraction capital taken over
    # debt and crises (docs/research/finanzas.md)
    fd0: float = 0.01          # debt dynamics estimated on data: dd = fd0 + fd_d d + fd_g g + fd_c crisis
    fd_d: float = -0.02        # mean reversion of debt / GDP (fiscal reaction, Bohn)
    fd_g: float = -0.5         # growth lowers debt / GDP
    fd_c: float = 0.05         # debt jump in a crisis year
    r_bar: float = 0.01        # mean world real rate in the estimation sample
    prem_p: float = 0.04       # periphery risk premium
    prem_d: float = 0.04       # premium per unit of debt above the intolerance threshold
    dstar0: float = 0.4        # debt intolerance threshold in the periphery ...
    dstar1: float = 0.8        # ... plus this much in the core
    ext_int: float = 0.5       # share of peripheral interest paid to foreign (core) creditors
    cr0: float = -4.0          # crisis onset intercept
    cr_d: float = 0.5          # debt above the threshold
    cr_r: float = 15.0         # world (US) real interest rate
    cr_g: float = -5.5         # lagged growth
    cr_cont: float = 3.0       # contagion: share of countries in crisis last year
    cr_loss: float = 0.045     # permanent output loss of a crisis (Laeven-Valencia, Cerra-Saxena)
    cr_debt: float = 0.08      # debt jump after a crisis
    # financing and consumption vs saving (docs/research/financiamiento.md)
    fh_eta: float = 0.54       # share of extra investment financed by domestic saving (dCA/dI = eta - 1)
    cm0: float = 0.14          # ceiling on the current-account deficit / GDP (q90 in the data)
    chi_k: float = 0.0         # capital adjustment cost (chi/2)(I/K - iota)^2 K (Hayashi 1982)
    iota_g: float = 0.04       # normal net investment rate in the adjustment cost
    x_amort: float = 0.10      # amortisation of extra foreign debt per year (~10-year maturity)
    # institutions and geopolitics (docs/research/geopolitica.md)
    cp0: float = -4.35         # coup attempt intercept
    cp_l: float = -0.28        # log(income/10k)
    cp_g: float = -2.7         # lagged growth
    cp_T: float = 1.37         # coup trap (coup in the previous 10 years)
    cp_D: float = -0.71        # democracy
    cp_CW: float = 0.80        # Cold War
    cp_psi: float = 0.40       # political stress (standardised log PSI)
    cp_loss: float = 0.008     # growth loss per year for 5 years after a successful coup
    ad0: float = -5.32         # autocracy -> democracy
    ad_l: float = 0.25
    ad_g: float = -1.5
    ad_W: float = 0.7          # share of democracies in the world
    da0: float = -3.34         # democracy -> autocracy
    da_l: float = -0.73
    da_g: float = -5.2
    da_W: float = -1.1
    dem_b: float = 0.0079      # democracy TFP impulse (Acemoglu et al. 2019)
    dem_rho: float = 0.963
    # climate and ecology (docs/research/clima.md)
    bhm1: float = 0.0127       # h(T) = bhm1 T + bhm2 T^2 (Burke-Hsiang-Miguel 2015)
    bhm2: float = -0.000487
    rho_D: float = 0.8         # persistence of climate damage (0 level, 1 growth effect)
    tcre: float = 0.46         # degC per 1000 GtCO2 (IPCC AR6; 0.46 in our data)
    g_eps: float = -0.0025     # annual change of CO2 per unit of fossil extraction
    dis_p: float = 0.29        # annual probability of a damaging climate disaster (EM-DAT)
    dis_mu: float = -7.5       # log damage / GDP given a disaster
    dis_sd: float = 2.5
    dis_kappa: float = 0.10    # disaster frequency increase per degC of global warming
    g_N: float = 0.03          # regeneration of natural capital
    eta_N: float = 0.005       # degradation per unit of ecological overshoot (footprint/biocapacity - 1)
    zeta_N: float = 0.005      # degradation per degC of local warming
    xi_N: float = 0.07         # TFP elasticity to natural capital (Johnson et al. 2021)
    # historical policy mapping (observed actions -> model channels)
    x_hist: float = 0.30       # share of resource rents reinvested in extraction
    r_hist: float = 0.015      # fallback upgrading effort when no R&D/education data
    f_hist: float = 0.02       # fallback outward resource FDI of core countries
    w_hist: float = 0.03       # fallback redistribution
    m_hist: float = 0.35       # fallback share of imports that are high-value core goods
    m_scale: float = 0.5       # model m = m_scale * manufactured imports / GDP
    r_scale: float = 0.3       # model r = r_scale * (R&D + public education) / GDP
    f_scale: float = 1.0       # model f = f_scale * outward FDI / GDP
    w_scale: float = 0.2       # model w = w_scale * social spending / GDP

    def vector(self, names):
        return np.array([getattr(self, n) for n in names])

    def with_vector(self, names, v):
        d = asdict(self)
        d.update({n: float(x) for n, x in zip(names, v)})
        return Params(**d)


def load_panel():
    """Model panel: PWT core + every merged source (src/merge_sources.py) when available."""
    f = ROOT / "data" / "processed" / "panel_full.csv"
    p = pd.read_csv(f if f.exists() else ROOT / "data" / "processed" / "panel.csv", low_memory=False)
    p = p[p.year.between(1950, 2019)]
    return p


def load_world_series(horizon=2019):
    """World-level exogenous series (T,) aligned to 1950-horizon, NaN where missing."""
    f = ROOT / "data" / "processed" / "world_full.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f).set_index("year").reindex(range(1950, horizon + 1))
    return {c: pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float) for c in d.columns}


@dataclass
class World:
    """Exogenous inputs aligned to arrays of shape (N, T)."""
    iso3: list
    names: list
    isonum: list
    years: np.ndarray
    pop: np.ndarray
    hc: np.ndarray
    open_: np.ndarray
    delta: np.ndarray
    csh_i: np.ndarray
    csh_m: np.ndarray
    labsh: np.ndarray
    rr: np.ndarray          # natural-resource rents / GDP (filled)
    rr_obs: np.ndarray      # same, NaN where WDI has no observation
    psi_t: np.ndarray       # world resource rents / world GDP by year (T,)
    K: np.ndarray
    Y: np.ndarray
    conflict: np.ndarray
    onset: np.ndarray
    entry: np.ndarray       # index of first observed year
    observed: np.ndarray    # bool (N, T): data available
    extra: dict = field(default_factory=dict)  # optional country-year series (N, T), unfilled
    world: dict = field(default_factory=dict)  # optional world series (T,)
    S_exp: np.ndarray = None   # (T, N, N) export shares i -> j (COW dyadic trade), None if absent
    M_imp: np.ndarray = None   # (T, N, N) import shares of i from j
    _locf: dict = field(default_factory=dict)

    def locf(self, name, default=np.nan):
        """Country-year series carried forward from the last observation (never backward),
        so the value at t only uses information up to t. Missing -> default."""
        if name not in self._locf:
            a = self.extra.get(name)
            if a is None:
                self._locf[name] = np.full((self.N, self.T), np.nan)
            else:
                self._locf[name] = pd.DataFrame(a.T).ffill().values.T
        key = (name, None if default is None or (isinstance(default, float) and np.isnan(default)) else float(default))
        if key not in self._locf:
            out = self._locf[name]
            self._locf[key] = np.where(np.isfinite(out), out, default)
        return self._locf[key]

    def wlocf(self, name, t, default=np.nan):
        """World series at year index t, carried forward."""
        a = self.world.get(name)
        if a is None:
            return default
        past = a[:t + 1]
        ok = np.where(np.isfinite(past))[0]
        return float(past[ok[-1]]) if len(ok) else default

    @property
    def N(self):
        return len(self.iso3)

    @property
    def T(self):
        return len(self.years)


def build_world(panel=None, horizon=2019):
    """Exogenous inputs 1950-horizon. Beyond 2019 (projections) there are no observations:
    population follows UN WPP 2024 (medium variant, rescaled to PWT in 2019) and every other
    series is carried forward from 2019."""
    p = load_panel() if panel is None else panel
    years = np.arange(1950, horizon + 1)
    iso = sorted(p.iso3.unique())
    idx = {c: i for i, c in enumerate(iso)}
    N, T = len(iso), len(years)

    ii = p.iso3.map(idx).to_numpy()
    tt = (p.year - 1950).to_numpy()

    def grid(col, fill="ffill"):
        g = np.full((N, T), np.nan)
        g[ii, tt] = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
        df = pd.DataFrame(g.T)
        df = df.ffill().bfill() if fill == "ffill" else df
        return df.values.T

    observed = ~np.isnan(grid("rgdpna", fill=None))
    entry = observed.argmax(axis=1)
    rr = np.clip(grid("res_rents"), 0.001, 0.6)
    Yg = grid("rgdpna", fill=None)
    psi_t = np.nansum(np.where(observed, rr * Yg, 0), 0) / np.nansum(np.where(observed, Yg, 0), 0)
    meta = p.groupby("iso3").agg(country=("country", "first"), isonum=("isonum", "first"))
    w = World(
        iso3=iso, names=[meta.loc[c, "country"] for c in iso],
        isonum=[int(meta.loc[c, "isonum"]) if pd.notna(meta.loc[c, "isonum"]) else -1 for c in iso],
        years=years,
        pop=grid("pop"), hc=grid("hc"),
        open_=np.clip((grid("csh_x") + grid("csh_m")) / 2, 0.02, 1.2),
        delta=grid("delta"), csh_i=grid("csh_i"), csh_m=grid("csh_m"), labsh=grid("labsh"),
        rr=rr, rr_obs=grid("res_rents_obs", fill=None), psi_t=psi_t,
        K=grid("rnna", fill=None), Y=grid("rgdpna", fill=None),
        conflict=np.nan_to_num(grid("conflict", fill=None)),
        onset=np.nan_to_num(grid("onset", fill=None)),
        entry=entry, observed=observed,
    )
    w.world = load_world_series(horizon)
    if horizon > 2019:
        f = ROOT / "data" / "processed" / "sources" / "demografia.csv"
        if f.exists():
            d = pd.read_csv(f, usecols=["iso3", "year", "pop_wpp"])
            d = d[d.iso3.isin(idx) & d.year.between(2019, horizon)]
            wpp = np.full((N, T), np.nan)
            wpp[d.iso3.map(idx).to_numpy(), d.year.to_numpy() - 1950] = d.pop_wpp.to_numpy()
            ratio = w.pop[:, 69] / wpp[:, 69]
            w.pop = w.pop.copy()
            for t in range(70, T):
                w.pop[:, t] = np.where(np.isfinite(wpp[:, t] * ratio), wpp[:, t] * ratio, w.pop[:, t - 1])
    dy = ROOT / "data" / "processed" / "sources" / "comercio_dyadic.csv.gz"
    if dy.exists():
        d = pd.read_csv(dy)
        d = d[d.iso3_o.isin(idx) & d.iso3_d.isin(idx) & (d.iso3_o != d.iso3_d) & (d.flow > 0)]
        F = np.zeros((T, N, N), np.float32)
        yi = d.year.to_numpy() - 1950
        ok = (yi >= 0) & (yi < T)
        F[yi[ok], d.iso3_o.map(idx).to_numpy()[ok], d.iso3_d.map(idx).to_numpy()[ok]] = d.flow.to_numpy()[ok]
        last = int(d.year.max()) - 1950
        F[last + 1:] = F[last]                       # hold the last observed network (2014)
        ex = F.sum(2, keepdims=True)
        w.S_exp = np.divide(F, ex, out=np.zeros_like(F), where=ex > 0)
        im = F.transpose(0, 2, 1)                    # im[t, i, j] = flow j -> i
        tot = im.sum(2, keepdims=True)
        w.M_imp = np.divide(im, tot, out=np.zeros_like(im), where=tot > 0)
    core_cols = {"iso3", "country", "year", "isonum"}
    for c in p.columns:
        if c not in core_cols and pd.api.types.is_numeric_dtype(p[c]):
            w.extra[c] = grid(c, fill=None)
    return w


def _nanq(a, q):
    """Fast row-wise nan-quantile (R, N) -> (R, 1), linear interpolation like numpy."""
    srt = np.sort(np.where(np.isfinite(a), a, np.inf), axis=1)
    n = np.isfinite(a).sum(1)
    pos = np.maximum(n - 1, 0) * q
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    r = np.arange(a.shape[0])
    v = srt[r, lo] + (pos - lo) * (srt[r, hi] - srt[r, lo])
    return np.where(n > 0, v, np.nan)[:, None]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))




class Simulator:
    """Vectorised over R Monte-Carlo worlds x N countries.

    Two regimes of exogenous inputs:
      t <= obs_until  "observed": institutions (democracy, coups), crises, temperature and the
                      world interest rate are taken from the data (history is replayed);
      t >  obs_until  "forecast": coups, regime changes, financial crises, nationalisations,
                      warming, local temperature and disasters are endogenous / stochastic, and
                      every exogenous series is frozen at its last value up to obs_until.
    """

    REC = ["ly", "core", "E", "ne", "psi", "asab", "conflict", "gni_pc", "rent_in", "drain",
           "jump", "cons_pc", "res_share", "reserves", "fshare", "ims", "res_out",
           "debt", "crisis", "demo", "coup", "temp", "dclim", "natcap", "co2", "eco_ue",
           "borrow", "squeeze", "adjcost"] + CHANNELS

    def __init__(self, world: World, params: Params, R=8, seed=0):
        self.w, self.p, self.R = world, params, R
        self.rng = np.random.default_rng(seed)
        self.obs_until = world.T - 1
        self.psi_freeze = None

    # ------------------------------------------------------------------ exogenous inputs
    def te(self, t):
        """Year index whose exogenous information is available at t."""
        return min(t, self.obs_until)

    def ex(self, name, t, default):
        return self.w.locf(name, default)[:, self.te(t)]

    def wx(self, name, t, default):
        return self.w.wlocf(name, self.te(t), default)

    def observed(self, t):
        return t <= self.obs_until

    def ca_ref(self, t):
        """Reference current account / GDP: observed (carried forward) up to obs_until, then
        each country's 10-year mean before the forecast origin."""
        ca = self.w.locf("ca_gdp")
        if self.observed(t):
            v = ca[:, t]
        else:
            lo = max(0, self.obs_until - 9)
            blk = ca[:, lo:self.obs_until + 1]
            n = np.isfinite(blk).sum(1)
            v = np.where(n > 0, np.nansum(blk, 1) / np.maximum(n, 1), np.nan)
        return np.where(np.isfinite(v), np.clip(v, -0.3, 0.3), -0.03)

    def roy_t(self, t):
        """Host-government take on foreign extraction by era (docs/research/recursos.md)."""
        if self.p.roy >= 1.0:
            return 1.0
        y = 1950 + self.te(t)
        if y <= 1972:
            return 0.5
        if y <= 1976:
            return 0.5 + (y - 1972) * 0.0875
        return 0.85 if y <= 1985 else 0.75

    # ------------------------------------------------------------------ state
    def reset(self, t0=0):
        w, R, N = self.w, self.R, self.w.N
        self.t = t0
        self.active = np.zeros((R, N), bool)
        z = lambda v=0.0: np.full((R, N), float(v))
        self.K, self.lnA, self.E, self.ne = z(1.0), z(), z(0.45), z(1.0)
        self.E0 = z(0.45)
        self.asab, self.conf = z(self.p.s0), z(0.0)
        self.mass_ma = z(1.0)
        self.v0, self.w0 = z(1.0), z(1.0)
        # natural resources
        self.U, self.X = z(1e-6), z(0.0)
        self.KRd, self.KRf, self.claims = z(0.0), z(0.0), z(0.0)
        self.mR = z(1.0)
        self.price = np.ones((R, 1))
        self.price_hist = []
        # debt, institutions, climate, ecology
        self.debt, self.crisis_t = z(0.4), z(0.0)
        self.xdebt = z(0.0)          # foreign debt from spending above the reference path
        self.demo, self.poly = z(0.0), z(0.4)
        self.Bdem = z(0.0)
        self.coup_t, self.coup_trap = z(0.0), z(0.0)
        self.Dclim, self.Nnat = z(0.0), z(1.0)
        self.G_prev, self.g_prev = z(np.nan), z(0.0)
        self.Tg = np.full((R, 1), self.w.wlocf("clima__gmst_preind", t0, 0.3))
        self.E_ext0 = None
        self.crisis_share = 0.0
        self.last = {}
        self.psi_freeze = None
        new = w.observed[:, t0] & (w.entry <= t0)
        self._init_country(new, t0, price=1.0)
        self.price = self._clearing_price_from_gdp(new, w.Y[:, t0])
        self._anchor(new, t0, w.Y[new, t0])
        self.lnA_F = self._q_lnA()

    def _init_country(self, mask, t, price):
        """Initialise countries in mask (N,) from data at year index t (TFP anchored separately)."""
        w, p = self.w, self.p
        if not mask.any():
            return
        L = w.pop[mask, t]
        G = w.Y[mask, t]
        self.K[:, mask] = w.K[mask, t]
        # elite share: WID top-10% pre-tax income share, else a mapping from the labour share
        top10 = w.locf("top10_share_ptinc")[mask, t]
        fb = np.clip(0.45 + 0.5 * ((1 - w.labsh[mask, t]) - 0.47), 0.25, 0.70)
        E = np.clip(np.where(np.isfinite(top10), top10, fb), 0.15, 0.85)
        self.E[:, mask] = E
        self.E0[:, mask] = E
        self.ne[:, mask] = 1.0
        # asabiya: observed cohesion / state-capacity proxy
        self.asab[:, mask] = np.clip(w.locf("asabiya_proxy", p.s0)[mask, t], 0.05, 0.95)
        self.conf[:, mask] = w.conflict[mask, t]
        self.mass_ma[:, mask] = (1 - E) * (G / L)
        self.v0[:, mask] = E / (1 - E)
        self.w0[:, mask] = 1 - E
        # resources: observed rent share pins current extraction; depleted fraction from
        # physical fossil data (value-weighted cumulative / recoverable) where available
        x0 = np.clip(w.locf("fosil_x_valor", p.x0)[mask, t], 0.002, 0.9)
        E0 = w.rr[mask, t] * G / np.maximum(price, 1e-6)
        cap_per_U = p.r_R * (x0 + EPS_X) * (1 - x0)
        cap0 = E0 / (1 - np.exp(-p.e0))
        self.U[:, mask] = cap0 / cap_per_U
        self.X[:, mask] = x0 * cap0 / cap_per_U
        self.KRd[:, mask] = p.e0 * p.c_R * cap0
        self.KRf[:, mask] = 0.0
        self.claims[:, mask] = 0.0
        self.mR[:, mask] = 1.0
        # debt and institutions
        self.debt[:, mask] = np.clip(w.locf("debt_gdp", 0.25)[mask, t], 0.0, 3.0)
        self.crisis_t[:, mask] = 0.0
        poly = w.locf("vdem_polyarchy", 0.35)[mask, t]
        demo = (w.locf("regime_row", 1.0)[mask, t] >= 2).astype(float)
        self.poly[:, mask], self.demo[:, mask] = poly, demo
        self.Bdem[:, mask] = demo * p.dem_b / (1 - p.dem_rho)   # democracies start at steady state
        self.coup_trap[:, mask] = (w.locf("coups_past10", 0.0)[mask, t] > 0).astype(float) * 10
        self.coup_t[:, mask] = 0.0
        self.Dclim[:, mask], self.Nnat[:, mask] = 0.0, 1.0
        self.G_prev[:, mask] = np.nan
        self.active[:, mask] = True

    # ---------------------------------------------------------------- resource helpers
    def _extraction(self):
        p = self.p
        x = np.clip(self.X / np.maximum(self.U, 1e-12), 0.0, 1.0)
        cap = p.r_R * self.U * (x + EPS_X) * (1 - x)
        KR = self.KRd + self.KRf
        e = KR / np.maximum(p.c_R * cap, 1e-12)
        ext = np.where(self.active, cap * (1 - np.exp(-e)), 0.0)
        sd = np.where(KR > 0, self.KRd / np.maximum(KR, 1e-12), 1.0)
        return cap, e, ext, sd, x

    def _psi(self, t):
        w = self.w
        if self.psi_freeze is not None and t > self.psi_freeze:
            return float(np.mean(w.psi_t[max(0, self.psi_freeze - 9):self.psi_freeze + 1]))
        return float(w.psi_t[min(t, w.T - 1)])

    def _clearing_price_from_gdp(self, mask, G):
        psi = self._psi(self.t)
        _, _, ext, _, _ = self._extraction()
        a = psi / (1 - psi)
        m = self.mR
        Gm = np.where(self.active, np.broadcast_to(G, m.shape), 0.0)
        num = a * np.nansum(Gm / m, 1, keepdims=True)
        den = ext.sum(1, keepdims=True) + a * np.sum(ext / m, 1, keepdims=True)
        return num / np.maximum(den, 1e-12)

    def _effective_lnA(self):
        return self.lnA + self.Bdem + self.Dclim + self.p.xi_N * np.log(np.maximum(self.Nnat, 0.05))

    def _anchor(self, mask, t, G):
        """Set TFP so the model reproduces observed GDP G of countries in mask at the current price."""
        w, p = self.w, self.p
        if not np.any(mask):
            return
        _, _, ext, _, _ = self._extraction()
        pr = self.price
        psi = self._psi(t)
        Yt = np.maximum((G[None] - pr * ext[:, mask]) / (1 - psi), 0.3 * G[None])
        pj = pr * self.mR[:, mask]
        B = Yt ** (1 - psi) / (psi / pj) ** psi
        L, hc = w.pop[mask, t], w.hc[mask, t]
        conf = 1 - p.gam_c * self.conf[:, mask]
        lnA_eff = (np.log(B / conf) - ALPHA * np.log(self.K[:, mask]) - (1 - ALPHA - psi) * np.log(hc * L))
        self.lnA[:, mask] = lnA_eff - (self._effective_lnA()[:, mask] - self.lnA[:, mask])

    def _activate(self, t):
        new = (self.w.entry == t) & ~self.active[0]
        if new.any():
            self._init_country(new, t, price=self.price)
            self._anchor(new, t, self.w.Y[new, t])

    def _q_lnA(self, q=0.9):
        return _nanq(np.where(self.active, self.lnA, np.nan), q)

    def assimilate(self, t):
        """Re-anchor observable state to data at year t (forecast origin)."""
        w = self.w
        m = w.observed[:, t]
        self.K[:, m] = w.K[m, t]
        top10 = w.locf("top10_share_ptinc")[m, t]
        self.E[:, m] = np.where(np.isfinite(top10), np.clip(top10, 0.15, 0.85), self.E[:, m])
        self.conf[:, m] = w.conflict[m, t]
        d = w.locf("debt_gdp")[m, t]
        self.debt[:, m] = np.where(np.isfinite(d), np.clip(d, 0, 3), self.debt[:, m])
        self.xdebt[:, m] = 0.0
        _, _, ext, _, _ = self._extraction()
        target = w.rr[m, t] * w.Y[m, t] / np.maximum(self.price, 1e-6)
        sc = np.clip(target / np.maximum(ext[:, m], 1e-12), 0.05, 20)
        for arr in (self.U, self.X, self.KRd, self.KRf):
            arr[:, m] *= sc
        self._anchor(m, t, w.Y[m, t])
        self.G_prev[:, m] = np.nan            # the re-anchoring jump is not growth
        self.lnA_F = np.maximum(self.lnA_F, self._q_lnA())

    # ------------------------------------------------------------------ policy helpers
    def hist_actions(self, t, freeze_from=None):
        """Observed actions (shares of GDP) mapped to model channels; with freeze_from, each
        country keeps its 10-year average up to that year (no future information)."""
        w, p = self.w, self.p
        R, N = self.R, w.N

        def obs(name, fallback):
            a = w.locf(name)
            if freeze_from is not None and t > freeze_from:
                lo = max(0, freeze_from - 9)
                v = np.nanmean(a[:, lo:freeze_from + 1], axis=1)
            else:
                v = a[:, t]
            return np.where(np.isfinite(v), v, fallback)

        c = self.last.get("core", np.zeros((R, N)))
        k = obs("acc_k", w.csh_i[:, t])
        rr = obs("res_rents", w.rr[:, t])
        acts = dict(
            k=k,
            r=p.r_scale * obs("acc_r", np.nan),
            m=p.m_scale * np.clip(obs("acc_m", np.nan), 0, 0.8),
            x=p.x_hist * rr,
            f=p.f_scale * np.clip(obs("acc_f", np.nan), 0, 0.3),
            w=p.w_scale * obs("acc_w", np.nan),
        )
        fb = dict(r=p.r_hist * k / 0.22, m=p.m_hist * w.csh_m[:, t], f=None, w=np.full(N, p.w_hist))
        out = {}
        for ch in CHANNELS:
            v = np.broadcast_to(acts[ch], (R, N))
            if ch == "f":
                v = np.where(np.isfinite(v), v, p.f_hist * c)
            elif ch in fb:
                v = np.where(np.isfinite(v), v, fb[ch])
            out[ch] = v
        return out

    # ------------------------------------------------------------------ dynamics
    def step(self, actions):
        """Advance one year. actions: dict channel -> (R, N) shares of GDP."""
        w, p, R, t = self.w, self.p, self.R, self.t
        N = w.N
        act = self.active
        L, hc = w.pop[:, t], w.hc[:, t]
        te = self.te(t)
        o = w.open_[:, te]
        dlt = np.nan_to_num(w.delta[:, te], nan=0.045)
        rng = self.rng
        psi = self._psi(t)
        obs_mode = self.observed(t)
        # common random numbers: every draw has a fixed shape each year
        u_jump, e_jump, n_tfp = rng.random((R, N)), rng.exponential(1.0, (R, N)), rng.standard_normal((R, N))
        u_conf, u_coup, u_succ = rng.random((R, N)), rng.random((R, N)), rng.random((R, N))
        u_reg, u_cris, u_nat = rng.random((R, N)), rng.random((R, N)), rng.random((R, N))
        u_dis, z_dis, z_tmp = rng.random((R, N)), rng.standard_normal((R, N)), rng.standard_normal((R, N))

        # ---------------------------------------------------------------- climate
        tclim = w.locf("tmp_level")                 # observed local temperature (carried forward)
        Tbar = np.nanmean(tclim[:, :min(31, self.obs_until + 1)], axis=1)     # 1950-1980 climatology
        Tbar = np.where(np.isfinite(Tbar), Tbar, 15.0)
        sig_i = np.clip(self.ex("tmp_anom_fao_sd", t, 0.31), 0.15, 1.0)
        if obs_mode:
            Tloc = np.broadcast_to(np.where(np.isfinite(tclim[:, t]), tclim[:, t], Tbar), (R, N))
            self.Tg = np.full((R, 1), w.wlocf("clima__gmst_preind", t, float(self.Tg.mean())))
        else:
            beta_i = np.clip(self.ex("tmp_pattern_beta", t, 1.33), 0.5, 2.5)
            base = np.nanmean(tclim[:, max(0, self.obs_until - 9):self.obs_until + 1], axis=1)
            base = np.where(np.isfinite(base), base, Tbar)
            Tg0 = w.wlocf("clima__gmst_preind", self.obs_until, 1.0)
            Tloc = base[None] + beta_i[None] * (self.Tg - Tg0) + sig_i[None] * z_tmp
        h = lambda T: p.bhm1 * T + p.bhm2 * T ** 2
        self.Dclim = np.where(act, np.maximum(p.rho_D * self.Dclim + h(Tloc) - h(Tbar)[None], -1.5), 0.0)
        # disasters destroy capital (probability rises with global warming)
        Tg1990 = w.wlocf("clima__gmst_preind", 40, 0.7)
        p_dis = p.dis_p * np.exp(p.dis_kappa * (self.Tg - Tg1990))
        loss = np.minimum(np.exp(p.dis_mu + p.dis_sd * z_dis), 0.5)
        hit = (u_dis < p_dis) & act

        # ---------------------------------------------------------------- resources & production
        cap, eff_e, ext, sd, xdep = self._extraction()
        Kp = np.maximum(self.K, 1e-6)
        B = (np.exp(self._effective_lnA()) * Kp ** ALPHA * (hc * L) ** (1 - ALPHA - psi)
             * (1 - p.gam_c * self.conf))
        B = np.where(act, B, 0.0)
        S = ext.sum(1, keepdims=True)
        inv = 1.0 / (1 - psi)
        denom = np.sum(np.where(act, self.mR ** (-inv) * B ** inv, 0.0), 1, keepdims=True)
        q = (np.maximum(S, 1e-12) / np.maximum(denom, 1e-12)) ** (1 - psi)
        price = psi / q
        pj = price * self.mR
        Y = np.where(act, (B * (psi / pj) ** psi) ** inv, 0.0)
        Rres = np.where(act, psi * Y / pj, 0.0)
        rev = price * ext
        G = np.where(act, (1 - psi) * Y + rev, 0.0)
        Gs = np.maximum(G, 1e-9)
        ly = np.where(act, np.log(Gs / L), np.nan)
        g = np.where(np.isfinite(self.G_prev), np.log(Gs / np.maximum(self.G_prev, 1e-9)), 0.0)
        g = np.clip(g, -0.5, 0.5)

        foreign_ext = (1 - sd) * ext
        cl_n = self.claims / np.maximum(self.claims.sum(1, keepdims=True), 1e-12)
        owned = sd * ext + cl_n * foreign_ext.sum(1, keepdims=True)
        ims = np.where(act, np.clip(1 - owned / np.maximum(Rres, 1e-12), 0.0, 1.0), 0.0)
        roy = self.roy_t(t)
        res_out = (1 - roy) * price * foreign_ext
        res_in = cl_n * res_out.sum(1, keepdims=True)
        res_inc = price * sd * ext + roy * price * foreign_ext

        ref = _nanq(ly, p.core_q)
        core = np.where(act, _sigmoid((ly - ref) / p.core_s), 0.0)
        gap = np.where(act, np.clip(self.lnA_F - self.lnA, 0.0, 4.0), 0.0)
        a = {c: np.where(act, np.nan_to_num(actions[c]), 0.0) for c in CHANNELS}

        # ---------------------------------------------------------------- financing: saving vs borrowing
        # Spending on the saving-using channels above the reference path (observed actions, or the
        # 10-year mean before a forecast origin) is financed a share fh_eta by domestic saving
        # (less consumption) and the rest by foreign borrowing, up to a ceiling on the current-account
        # deficit; any excess is forced saving (a consumption squeeze). Cutting spending below the
        # reference raises consumption only by fh_eta; the rest leaves as capital exports.
        ref = self.hist_actions(t, None if obs_mode else self.obs_until)
        extra_b = sum(a[c] - np.where(act, np.nan_to_num(ref[c]), 0.0) for c in FIN_CH)
        want = (1 - p.fh_eta) * extra_b
        room = np.maximum(p.cm0 + self.ca_ref(t)[None], 0.0)
        borrow = np.where(act, np.where(want > 0, np.minimum(want, room), want), 0.0)
        squeeze = np.where(act, want - borrow, 0.0)
        dom_burden = np.where(act, extra_b - borrow, 0.0)     # borne by residents (share of GDP)

        # ---------------------------------------------------------------- world-system flows
        wexp = core ** 2 * G
        wexp_n = wexp / np.maximum(wexp.sum(1, keepdims=True), 1e-9)
        ypc = np.where(act, Gs / L, np.nan)
        if w.S_exp is not None and p.tau > 0:
            Sx = w.S_exp[te][None]                                            # (1, N, N)
            ratio = np.nan_to_num((ypc[:, None, :] / ypc[:, :, None]) ** p.beta_px - 1)
            gapm = np.maximum(ratio, 0.0) * core[:, None, :] * (1 - core)[:, :, None]
            X = (o * G)[:, :, None]
            flows = p.theta_d * X * Sx * gapm * act[:, None, :]
            drain = flows.sum(2)
            drain_in = flows.sum(1)
        else:
            drain = p.tau * o * G * (1 - core)
            drain_in = drain.sum(1, keepdims=True) * wexp_n
        mspend = a["m"] * G
        foreign_m = mspend * (1 - core)
        if w.M_imp is not None:
            Mi = w.M_imp[te][None] * (core[:, None, :] * act[:, None, :])
            tot = Mi.sum(2, keepdims=True)
            Mi = np.where(tot > 0, Mi / np.maximum(tot, 1e-12), wexp_n[:, None, :])
            rent_m_in = p.mu * np.einsum("rij,ri->rj", Mi, foreign_m)
        else:
            rent_m_in = p.mu * foreign_m.sum(1, keepdims=True) * wexp_n

        # resource concessions abroad (compete with the host's own extraction capital)
        fout = a["f"] * G
        host_w = np.where(act, cap * np.exp(-eff_e) * o * (1 - self.conf)
                          * np.clip(1 - p.nu_a * self.asab, 0.02, 1), 0.0)
        host_w = host_w / np.maximum(host_w.sum(1, keepdims=True), 1e-12)
        fin = fout.sum(1, keepdims=True) * host_w
        self.claims = self.claims * (1 - p.d_R) + fout
        self.KRf = self.KRf * (1 - p.d_R) + fin
        eff = 0.75 + 0.5 * self.asab
        self.KRd = np.where(act, self.KRd * (1 - p.d_R) + eff * a["x"] * G, self.KRd)
        self.X = np.where(act, self.X + ext, self.X)

        # nationalisations: foreign extraction capital taken over by the host
        if obs_mode:
            nat = (np.nan_to_num(w.extra.get("nac_evento_petroleo", np.zeros((N, w.T)))[:, t]) +
                   np.nan_to_num(w.extra.get("nac_evento_mineria", np.zeros((N, w.T)))[:, t]) > 0)
            nat = np.broadcast_to(nat, (R, N))
        else:
            self.price_hist.append(price[:, 0])
            lp = np.log(np.maximum(price[:, 0], 1e-6))
            sh = lp - np.log(np.maximum(self.price_hist[-4] if len(self.price_hist) > 3 else price[:, 0], 1e-6))
            polity = 20 * self.poly - 10
            lg = (p.nat0 + p.nat_shock * sh[:, None] + p.nat_level * (lp[:, None] - np.log(1.0))
                  + p.nat_dem * polity + p.nat_post85 * (1950 + t >= 1985))
            nat = (u_nat < _sigmoid(lg)) & (self.KRf > 0) & act
        take = np.where(nat, p.nat_theta * self.KRf, 0.0)
        self.KRf -= take
        self.KRd += take

        # ---------------------------------------------------------------- debt and financial crises
        rstar = float(np.clip(self.wx("finanzas__us_real_strate", t, 0.01), -0.05, 0.10))
        dstar = p.dstar0 + p.dstar1 * core
        rate = rstar + p.prem_p * (1 - core) + p.prem_d * np.maximum(self.debt - dstar, 0.0)
        # the extra foreign debt pays the country's rate and is amortised out of national income
        xint = np.where(self.xdebt > 0, rate, rstar) * self.xdebt * G     # creditors earn the world rate
        xrep = p.x_amort * self.xdebt
        interest_abroad = p.ext_int * (1 - core) * np.maximum(rate, 0) * (self.debt - self.xdebt) * G + xint
        int_in = interest_abroad.sum(1, keepdims=True) * wexp_n
        if obs_mode:
            crisis = np.broadcast_to(np.nan_to_num(w.extra.get("crisis_any", np.zeros((N, w.T)))[:, t]) > 0,
                                     (R, N)) & act
        else:
            lg = (p.cr0 + p.cr_d * (self.debt / dstar - 1) + p.cr_r * rstar + p.cr_g * self.g_prev
                  + p.cr_cont * self.crisis_share)
            crisis = (u_cris < _sigmoid(lg)) & act & (self.crisis_t <= 0)
        # reduced-form debt dynamics estimated on data, plus the burden of world-rate shocks
        dref = self.debt - self.xdebt
        dd = p.fd0 + p.fd_d * dref + p.fd_g * g + p.fd_c * crisis + (rstar - p.r_bar) * dref
        self.xdebt = np.where(act, self.xdebt + borrow - xrep, self.xdebt)
        self.debt = np.where(act, np.clip(dref + dd, 0, 4) + self.xdebt, self.debt)
        self.crisis_t = np.where(crisis, 3.0, np.maximum(self.crisis_t - 1, 0))
        self.crisis_share = float(np.mean(crisis[act]))

        rent_in = drain_in + rent_m_in + res_in + int_in
        rent_out = drain + p.mu * foreign_m + res_out + interest_abroad
        gni = np.maximum(G + rent_in - rent_out, 1e-6 * Gs)

        # ---------------------------------------------------------------- capital & technology
        Ik = eff * a["k"] * G
        Kc = np.maximum(self.K, 1e-9)
        adj = 0.5 * p.chi_k * (Ik / Kc - dlt - p.iota_g) ** 2 * Kc
        adj = np.minimum(adj, 0.9 * Ik)
        self.K = np.where(act, np.maximum((1 - dlt) * self.K + Ik - adj, 0.05 * Kc), self.K)
        self.K = np.where(hit, self.K * (1 - np.minimum(loss * Gs / np.maximum(self.K, 1e-9), 0.5)), self.K)
        fdi_in = np.clip(self.ex("fdi_in_gdp", t, 0.02), 0, 0.2)
        aid = np.clip(self.ex("oda_gni", t, 0.0) / 100, 0, 0.3)
        embodied = np.minimum(foreign_m / Gs, 0.3)
        hq = np.nanquantile(np.where(act[0], hc, np.nan), 0.9)
        absorb = np.clip(hc / hq, 0.05, 1.5) ** p.phi_h
        dln = (p.g0 + gap * absorb * (p.kappa + p.kappa_m * embodied + p.kappa_f * fdi_in)
               + p.aid_eff * aid * (1 - aid / 0.3)
               + p.sig_a * n_tfp - p.gam_a * self.conf - p.cp_loss * (self.coup_t > 0)
               - p.cr_loss * crisis)
        lam = p.lam0 * np.sqrt(np.maximum(a["r"], 0)) * (0.5 + self.asab)
        lamc = np.clip(np.nan_to_num(np.where(act, lam, 0.0)), 0, 5)
        e0_ = np.exp(-lamc)
        c1 = e0_ * (1 + lamc)
        c2 = c1 + e0_ * lamc ** 2 / 2
        njump = (u_jump > e0_).astype(int) + (u_jump > c1) + (u_jump > c2)
        jsize = np.where(njump > 0, e_jump * p.j0 * (1 + gap) * njump, 0.0)
        self.lnA = np.where(act, self.lnA + np.clip(dln + jsize, -0.3, 0.3), self.lnA)
        self.lnA_F = np.maximum(self.lnA_F + p.g0, self._q_lnA())

        # ---------------------------------------------------------------- structural-demographic
        phi = np.clip(self.E + p.phi_extra * (1 - self.poly), 0, 0.95)
        phiR = np.clip(self.E + p.phiR_extra * (1 - self.poly), 0, 0.95) if p.phi_R != 0.5 else np.full_like(phi, 0.5)
        base = G - res_inc
        elite_inc = self.E * base + phiR * res_inc + phi * rent_in
        mass_inc = ((1 - self.E) * base + (1 - phiR) * res_inc + (1 - phi) * rent_in
                    - rent_out + a["w"] * G - (1 - self.E) * (dom_burden + xrep) * G)
        mass_inc = np.maximum(mass_inc, 0.05 * Gs)
        mass_pc = mass_inc / L
        v = elite_inc / mass_inc
        grow = np.clip(p.beta_e * (v / self.v0 - self.ne) - 0.15 * self.conf, -0.5, 0.5)
        self.ne = np.where(act, np.clip(self.ne * np.exp(grow), 0.1, 10.0), self.ne)
        wrel = np.clip((mass_inc / Gs) / np.maximum(self.w0, 1e-3), 0.2, 5)
        youth = self.ex("youth_bulge_15_24_adult", t, 0.28) * 100
        mmp = (1 / wrel) * np.exp(p.bY * (youth - 28)) * (self.mass_ma / np.maximum(mass_pc, 1e-9)) ** p.kappa_mmp
        emp = self.ne ** 2 / np.maximum(v / self.v0, 1e-3)
        sfd = (1.5 - self.asab) * (1 + _sigmoid(4 * (self.debt / dstar - 1)) + 0.5 * (self.crisis_t > 0))
        psi_idx = np.where(act, mmp * emp * sfd, np.nan)
        self.mass_ma = 0.9 * self.mass_ma + 0.1 * mass_pc
        rent_share = (phi * rent_in + np.maximum(phiR - self.E, 0) * res_inc) / Gs
        self.E = np.clip(self.E + p.eps_r * rent_share * 0.1 - p.eps_w * a["w"] * 0.1
                         + p.eps_0 * (self.E0 - self.E) * 0.1 - 0.02 * self.conf, 0.05, 0.95)

        # ---------------------------------------------------------------- institutions (coups, regimes)
        lpsi = np.log(np.maximum(psi_idx, 1e-6))
        mu_l, sd_l = np.nanmean(np.where(act, lpsi, np.nan)), np.nanstd(np.where(act, lpsi, np.nan)) + 1e-9
        psi_z = np.nan_to_num((lpsi - mu_l) / sd_l)
        ell = np.nan_to_num(ly) - np.log(10000.0)
        cw = self.wx("geopolitica__cold_war", t, 0.0)
        if obs_mode:
            succ = np.broadcast_to(np.nan_to_num(w.extra.get("coup_success_any", np.zeros((N, w.T)))[:, t]) > 0, (R, N))
            att = np.broadcast_to(np.nan_to_num(w.extra.get("coup_any", np.zeros((N, w.T)))[:, t]) > 0, (R, N))
            self.poly = np.broadcast_to(w.locf("vdem_polyarchy", 0.35)[:, t], (R, N)).copy()
            self.demo = np.broadcast_to((w.locf("regime_row", 1.0)[:, t] >= 2).astype(float), (R, N)).copy()
        else:
            lg = (p.cp0 + p.cp_l * ell + p.cp_g * self.g_prev + p.cp_T * (self.coup_trap > 0)
                  + p.cp_D * self.demo + p.cp_CW * cw + p.cp_psi * psi_z)
            att = (u_coup < _sigmoid(lg)) & act
            succ = att & (u_succ < 0.5)
            WD = float(np.mean(self.demo[act]))
            lad = p.ad0 + p.ad_l * ell + p.ad_g * self.g_prev + p.ad_W * WD
            lda = p.da0 + p.da_l * ell + p.da_g * self.g_prev + p.da_W * WD
            to_d = (self.demo == 0) & (u_reg < _sigmoid(lad)) & ~succ & act
            to_a = ((self.demo == 1) & (u_reg < _sigmoid(lda)) | (succ & (self.demo == 1))) & act
            self.demo = np.where(to_d, 1.0, np.where(to_a, 0.0, self.demo))
            self.poly = np.where(to_d, np.maximum(self.poly, 0.6), np.where(to_a, np.minimum(self.poly, 0.3), self.poly))
        self.coup_t = np.where(succ, 5.0, np.maximum(self.coup_t - 1, 0))
        self.coup_trap = np.where(att, 10.0, np.maximum(self.coup_trap - 1, 0))
        self.Bdem = np.where(act, p.dem_rho * self.Bdem + p.dem_b * self.demo, self.Bdem)

        # ---------------------------------------------------------------- civil conflict
        anoc = 4 * self.poly * (1 - self.poly)
        tz = (Tloc - Tbar[None]) / sig_i[None]
        p_on = _sigmoid(p.b0 + p.b1 * lpsi + p.b2 * (np.nan_to_num(ly) - 9.0) + p.b3 * (self.asab - 0.5)
                        + p.b4 * np.clip(rev / Gs, 0, 0.8) + p.b5 * anoc + p.gam_cl * tz)
        p_stay = _sigmoid(p.bp + p.b1 * lpsi * 0.5)
        hazard = np.where(self.conf > 0, p_stay, p_on)
        newconf = (u_conf < hazard).astype(float)
        self.hazard_onset = np.where(act, p_on, np.nan)
        self.hazard = np.where(act, hazard, np.nan)

        # ---------------------------------------------------------------- asabiya (metaethnic frontier)
        frontier = np.clip(o * (1 - core) + 4.0 * (drain + p.mu * foreign_m + res_out) / Gs, 0, 1)
        rich = _sigmoid((np.nan_to_num(ly) - np.log(20000.0)) / 0.5)
        self.asab = np.clip(self.asab + p.r_a * frontier * self.asab * (1 - self.asab)
                            - p.d_a * (1 - frontier) * rich * self.asab * np.maximum(self.ne, 0.5)
                            - 0.03 * self.conf * self.asab, 0.02, 0.98)
        self.asab = np.where(act, self.asab, p.s0)

        # ---------------------------------------------------------------- ecology & emissions
        over = np.clip(self.ex("ef_cons_pc", t, 2.5) / np.maximum(self.ex("biocap_pc", t, 2.5), 0.05) - 1, 0, 2)
        self.Nnat = np.where(act, np.clip(self.Nnat + p.g_N * self.Nnat * (1 - self.Nnat)
                                          - p.eta_N * over - p.zeta_N * np.maximum(Tloc - Tbar[None], 0), 0.05, 1.0),
                             self.Nnat)
        fossil_share = np.clip(self.ex("renta_no_renovable_pct_pib", t, np.nan) / np.maximum(self.ex("renta_total_pct_pib", t, np.nan), 1e-6), 0, 1)
        fossil_share = np.where(np.isfinite(fossil_share), fossil_share, 0.7)
        fext = ext * fossil_share
        if self.E_ext0 is None or obs_mode:
            self.E_ext0 = np.maximum(fext.sum(1, keepdims=True), 1e-9)
            self.co2_0 = self.wx("clima__co2_mt", t, 30000.0) / 1000.0
            self.luc_0 = self.wx("clima__co2_luc_mt", t, 5000.0) / 1000.0
            self.t_em0 = t
        co2_world = (self.co2_0 * fext.sum(1, keepdims=True) / self.E_ext0 * (1 + p.g_eps) ** (t - self.t_em0))
        co2_i = co2_world * fext / np.maximum(fext.sum(1, keepdims=True), 1e-12)
        if not obs_mode:
            self.Tg = self.Tg + p.tcre / 1000.0 * (co2_world + self.luc_0)
        eco_ue = (o * G * (1 - core) * (w.S_exp[te] @ np.ones(N) if w.S_exp is not None else 1.0))  # periphery exports

        # ---------------------------------------------------------------- bookkeeping
        self.mR = np.where(act, 1 + p.zeta * ims, 1.0)
        self.price = price
        self.G_prev = np.where(act, Gs, self.G_prev)
        self.g_prev = np.where(act, g, 0.0)
        cons = (1 - sum(a[c] for c in CHANNELS)) * gni + mspend * 0.5 + a["w"] * G * 0.5 + (borrow - xrep) * G
        nz = lambda v: np.where(act, v, np.nan)
        rec = dict(ly=ly, core=core, E=self.E.copy(), ne=self.ne.copy(), psi=psi_idx, asab=self.asab.copy(),
                   conflict=self.conf.copy(), gni_pc=nz(gni / L),
                   rent_in=nz((rent_in - rent_out) / Gs), drain=nz(drain / Gs), jump=nz(jsize),
                   cons_pc=nz(np.maximum(cons, 1e-6) / L),
                   res_share=nz(rev / Gs), reserves=nz(1 - np.clip(self.X / np.maximum(self.U, 1e-12), 0, 1)),
                   fshare=nz(1 - sd), ims=nz(ims), res_out=nz((res_in - res_out) / Gs),
                   debt=nz(self.debt.copy()), crisis=nz(crisis.astype(float)), demo=nz(self.demo.copy()),
                   coup=nz(np.broadcast_to(succ, (R, N)).astype(float)), temp=nz(np.broadcast_to(Tloc, (R, N))),
                   dclim=nz(self.Dclim.copy()), natcap=nz(self.Nnat.copy()), co2=nz(co2_i * 1000),
                   eco_ue=nz(eco_ue / Gs), borrow=nz(borrow), squeeze=nz(squeeze),
                   adjcost=nz(np.where(Ik > 0, adj / np.maximum(Ik, 1e-12), 0.0)),
                   **{c: nz(a[c]) for c in CHANNELS})
        self.last = rec
        self.last_price = price[:, 0].copy()
        self.last_Tg = self.Tg[:, 0].copy()
        self.conf = np.where(act, newconf, 0.0)
        self.t += 1
        if self.t < w.T:
            self._activate(self.t)
        return rec

    def run(self, t0=0, t1=None, policy="hist", freeze_from=None, assimilate_at=None, record=True):
        """Run from year index t0 to t1 (exclusive). With freeze_from, exogenous inputs are
        frozen and institutions/crises/climate become endogenous after that year.
        Returns dict of (R, N, T) arrays plus 'price' and 'Tg' (R, T)."""
        w = self.w
        t1 = w.T if t1 is None else t1
        self.obs_until = w.T - 1 if freeze_from is None else freeze_from
        self.reset(t0)
        self.psi_freeze = freeze_from
        out = {k: np.full((self.R, w.N, t1 - t0), np.nan) for k in self.REC}
        out["hazard"] = np.full((self.R, w.N, t1 - t0), np.nan)
        out["price"] = np.full((self.R, t1 - t0), np.nan)
        out["Tg"] = np.full((self.R, t1 - t0), np.nan)
        for t in range(t0, t1):
            if assimilate_at is not None and t == assimilate_at:
                self.assimilate(t)
            if policy == "hist":
                acts = self.hist_actions(t, freeze_from)
            else:
                acts = policy(self, t)
            rec = self.step(acts)
            if record:
                for k in self.REC:
                    out[k][:, :, t - t0] = rec[k]
                out["hazard"][:, :, t - t0] = self.hazard
                out["price"][:, t - t0] = self.last_price
                out["Tg"][:, t - t0] = self.last_Tg
        return out
