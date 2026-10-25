"""Paso 2: estrategias en el L1 (2019-2030) y su consecuencia en el L0 (World3-03, 2020-2100).

* Estrategias de ``common.STRATEGIES`` más dos derivadas del IRL del paso 1:
  ``irl_revelada`` (los desvíos que la recompensa REVELADA hace rentables: «más de lo mismo») e
  ``irl_bienestar`` (los desvíos rentables en bienestar: lo que les convendría).
* Adopción GLOBAL (todos los países a la vez) con números aleatorios comunes frente a la línea base.
* Adopción CUASI-UNILATERAL (estrategias de canales): los países se reparten en 6 grupos al azar y cada
  corrida aplica la estrategia a un grupo; cada país se mide cuando su grupo la adopta (el resto, 5/6 del
  mundo, sigue la regla). Es la respuesta que el IRL interpreta como «mejor respuesta».
* Cada adopción global pasa al L0: señales mundiales -> constantes de World3 (``common.world3_constants``)
  sobre 400 estados del mundo.
"""
import time

import numpy as np

import common as c

R, SEED, GROUPS = 32, 0, 6


def irl_strategies():
    try:
        g = c.load("irl.json")["global"]
    except FileNotFoundError:
        return {}
    out = {}
    for key, name, label in (("desvios_rentables_revelada", "irl_revelada", "IRL: más de lo revelado"),
                             ("desvios_rentables_bienestar", "irl_bienestar", "IRL: lo que conviene (bienestar)")):
        d = g[key]
        offs = {}
        for ch in c.CHANNELS:
            up, dn = d.get(f"{ch}+", 0.0), d.get(f"{ch}-", 0.0)
            if up > 0.5 and up >= dn:
                offs[ch] = 0.5
            elif dn > 0.5:
                offs[ch] = -0.5
        out[name] = dict(label=label, offsets=offs, calib={},
                         what=f"desvíos de ±0,5 logit rentables en > 50 % de los casos ({key.split('_')[-1]})")
    return out


def offsets_with_cost(s):
    offs = dict(s["offsets"])
    cost = c.green_cost(s["calib"])
    if cost:
        offs["k"] = offs.get("k", 0.0) + cost
    return offs


def dev_index(gdp_pc, hc, hazard):
    """Índice de desarrollo tipo IDH (0-1): media geométrica de ingreso, educación y paz."""
    inc = np.clip((np.log(gdp_pc) - np.log(100)) / (np.log(75000) - np.log(100)), 0.01, 1)
    edu = np.clip((hc - 1) / (3.8 - 1), 0.01, 1)
    paz = np.clip(1 - hazard, 0.01, 1)
    return (inc * edu * paz) ** (1 / 3)


def per_country(m, mb, th):
    """Diferencias pareadas (mediana sobre miembros) por país: (N,) cada una."""
    from world4.core.l1_atlas import irl as airl
    wel = np.array(airl.HAND["bienestar"], float)
    rel = lambda a, b: np.nanmedian(a / b - 1, 0)  # noqa: E731
    dif = lambda a, b: np.nanmedian(a - b, 0)  # noqa: E731
    ID, IDb = dev_index(m["gdp_pc"], m["hc"], m["hazard"]), dev_index(mb["gdp_pc"], mb["hc"], mb["hazard"])
    dv = m["val"] - mb["val"]                          # (R, N, F)
    rev = np.einsum("rnf,nf->rn", dv, th) if th.ndim == 2 else dv @ th
    return {"gdp_pc": rel(m["gdp_pc"], mb["gdp_pc"]), "cons_pc": rel(m["cons_pc"], mb["cons_pc"]),
            "hc": dif(m["hc"], mb["hc"]), "hazard": dif(m["hazard"], mb["hazard"]), "debt": dif(m["debt"], mb["debt"]),
            "fossil": rel(m["fossil_twh"], mb["fossil_twh"]), "dev": dif(ID, IDb) * 100,
            "welfare_value": np.nanmedian(dv @ wel, 0), "revealed_value": np.nanmedian(rev, 0),
            "dev_level": np.nanmedian(ID, 0)}


def world_summary(f, m, sig):
    W, _ = c.live()
    pop = m["pop"]
    wsum = lambda x: np.nansum(x * pop, 1) / np.nansum(np.where(np.isfinite(x), pop, 0), 1)  # noqa: E731
    ID = dev_index(m["gdp_pc"], m["hc"], m["hazard"])
    return {"gdp_pc": float(np.median(wsum(m["gdp_pc"]))), "cons_pc": float(np.median(wsum(m["cons_pc"]))),
            "hc": float(np.median(wsum(m["hc"]))), "hazard": float(np.median(wsum(m["hazard"]))),
            "dev_index": float(np.median(wsum(ID))), "dev_index_poorest_quintile": float(np.median(
                [np.nanmean(np.where(m["gdp_pc0"][r] <= np.nanquantile(m["gdp_pc0"][r], 0.2), ID[r], np.nan)) for r in range(len(ID))])),
            "fossil_cum_twh": float(np.median(sig["fossil_cum"])), "nonfossil_share_2030": float(np.median(sig["nonfossil_share"])),
            "Tg_2030": float(np.median(sig["Tg"])), "gY": float(np.median(sig["gY"])), "pop_2030": float(np.median(sig["Pw"])),
            "debt": float(np.median(np.nansum(m["debt"] * m["gdp_pc"] * pop, 1) / np.nansum(m["gdp_pc"] * pop, 1)))}


def theta_matrix():
    """θ revelado por país (N, F): el propio en los países foco, el global en el resto."""
    W, _ = c.live()
    th = np.tile(c.irl_theta(), (W.N, 1))
    try:
        d = c.load("irl.json")
        sd = np.array(d["global"]["sd"])
        for iso, v in d["paises"].items():
            t = np.array([v["theta_std"][k] for k in c.FEATS]) / sd
            th[list(W.iso3).index(iso)] = t / np.abs(t).max()
    except FileNotFoundError:
        pass
    return th


def main():
    t_all = time.time()
    W, _ = c.live()
    strategies = {**c.STRATEGIES, **irl_strategies()}
    th = theta_matrix()
    ens = c.world3_ensemble(400)
    print(f"World3: {ens['n']} estados del mundo (de {ens['drawn']} sorteados)", flush=True)
    fb = c.l1_run(R=R, seed=SEED)
    mb, sb = c.country_metrics(fb), c.world_signals(fb)
    res = {"setup": {"R": R, "seed": SEED, "origin": c.ORIGIN, "end": c.END, "w3_members": ens["n"],
                     "w3_tech_strata": {k: v.tolist() for k, v in ens["tech"].items()}},
           "iso3": list(W.iso3), "strategies": {}}
    active = np.isfinite(mb["gdp_pc"]).all(0)
    res["active"] = active.tolist()
    res["pop_2030"] = np.nanmedian(mb["pop"], 0).tolist()
    res["gdp_pc_2019"] = np.nanmedian(mb["gdp_pc0"], 0).tolist()
    rng = np.random.default_rng(123)
    group = rng.integers(0, GROUPS, W.N)
    for name, s in strategies.items():
        t = time.time()
        offs = offsets_with_cost(s)
        f = fb if name == "base" else c.l1_run(offsets=offs, calib=s["calib"], R=R, seed=SEED)
        m, sig = c.country_metrics(f), c.world_signals(f)
        mp = c.world3_constants(sig, sb)
        w3 = c.world3_metrics(mp, ens)
        collapse = w3.pop("_collapse")
        entry = {"label": s["label"], "what": s["what"], "offsets": offs, "calib": s["calib"],
                 "world": world_summary(f, m, sig), "w3_mapping": mp, "w3": w3,
                 "global": {k: v.tolist() for k, v in per_country(m, mb, th).items()}}
        if name == "base":
            entry["w3_collapse_members"] = collapse.tolist()
        if s["offsets"] and not s["calib"] and name != "base":
            uni = {k: np.full(W.N, np.nan) for k in entry["global"]}
            for g in range(GROUPS):
                mask = group == g
                fu = c.l1_run(offsets=offs, R=R, seed=SEED, mask=mask)
                pc = per_country(c.country_metrics(fu), mb, th)
                for k in uni:
                    uni[k][mask] = pc[k][mask]
            entry["unilateral"] = {k: v.tolist() for k, v in uni.items()}
        res["strategies"][name] = entry
        wd = entry["world"]
        print(f"{name:20s} PIBpc {wd['gdp_pc']:8.0f} ID {wd['dev_index']:.4f} fósil {wd['fossil_cum_twh']:.3g} "
              f"P(colapso<=2070) {w3['p_collapse_2070']:.2f} severo {w3['p_severe']:.2f} ppolx {w3['ppolx_peak']:.1f} "
              f"HWI2100 {w3['hwi_2100']:.3f} ({time.time() - t:.0f}s)", flush=True)
    res["setup"]["segundos"] = round(time.time() - t_all)
    c.dump("strategies.json", res)


if __name__ == "__main__":
    main()
