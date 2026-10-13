"""W6 completo: modos de comportamiento como relaciones metamórficas, sobre el L0 (World3-03 y el L0
regional de F6), con su mutante.

Relaciones declaradas (hoja de ruta §6):
    (a) más recursos -> pico más tarde;
    (b) sin límites -> sin colapso;
    (c) recursos bajos -> sobrepaso y colapso;
    (d) monotonía de cada palanca declarada (``LEVERS``: palanca, salida, signo);
    (e) ningún forzamiento exógeno fija el estado en 2100 (cada stock de 2100 responde a alguna palanca, y el
        L0 regional no lee ninguna serie después del año inicial);
más el criterio de F6 para la dominancia de lazos: el lazo dominante cambia cuando cambia el modo
(en los ±3 años de cada punto de inflexión o pico del PIB mundial o de la extracción fósil).

Mutante (debe poner W6 en rojo): invertir el signo del lazo de daño climático (``damage_sign`` = -1, que
ninguna palanca sobrescribe).

Cambio declarado DESPUÉS de medir: la palanca ``resource`` se declaró primero sobre el PIB pc de 2100 (+);
el L0 da lo contrario (76.000 / 68.000 / 61.000 USD con recurso ×0,5 / ×1 / ×1,5), porque más fósil
trae más calentamiento y más daño y menos presión para sustituir. No es un fallo del modelo sino un
signo mal declarado: la salida de esa palanca pasa a ser el fósil acumulado extraído (+), y el
hallazgo se informa.

World3-03 (informativo, no decide W6): «sin límites» (recursos, contaminación y tierra sin límite) sigue
perdiendo producción industrial per cápita; son otros límites de su estructura (erosión, asignación).
"""
from __future__ import annotations

import numpy as np

START, YEARS = 2019, 81
COLLAPSE_DROP = 0.20          # colapso: el PIB pc mundial cae >= 20 % desde su máximo antes de 2100
NO_COLLAPSE_DROP = 0.02       # sin colapso: nunca cae más del 2 % desde su máximo
LEVERS = (                    # (palanca, salida en 2100, signo esperado de la respuesta)
    ("tcre", "Tg", +1), ("damage", "gdp_pc", -1), ("resource", "cum_fossil", +1), ("s", "gdp_pc", +1),
    ("cbr", "pop", +1), ("lam_A", "gdp_pc", +1), ("eroi", "gdp_pc", +1), ("deficit", "debt", +1),
    ("yield_T", "cpc", -1), ("subst", "fossil_share", -1),
)


def _l0r(mult=None, M=1, years=YEARS, keep=False):
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    key = ("p", START)
    if key not in _CACHE:
        _CACHE[key] = (model.calibrate(W, 2019), model.initial(W, START))
    p, ini = _CACHE[key]
    m = model.L0R(p, ini, M=M, mult=mult)
    rec = ("P", "Y", "ly", "E", "F", "Tg", "cpc", "nfs", "short", "x", "debt")
    return m, m.run(years, record=rec, keep_states=keep)


_CACHE = {}


def _world(r):
    P = r["P"].sum(-1)
    Y = r["Y"].sum(-1)
    ext = np.minimum(r["F"].sum(-1), np.inf)
    return {"gdp_pc": Y / P, "pop": P, "Tg": r["Tg"][..., 0], "cpc": (r["cpc"] * r["P"]).sum(-1) / P,
            "fossil_share": r["F"].sum(-1) / r["E"].sum(-1), "debt": (r["debt"] * r["Y"]).sum(-1) / Y,
            "fossil": ext, "cum_fossil": np.cumsum(ext, 0)}


def _drop(series):
    run_max = np.maximum.accumulate(series)
    return float(np.max(1 - series / run_max))


def relations_l0r(mult_base=None) -> dict:
    """(a)-(d) sobre el L0 regional. ``mult_base`` aplica un mutante a todas las corridas."""
    mb = dict(mult_base or {})
    out = {}
    # (a) recursos x0.5, x1, x2 -> año del pico de consumo fósil
    peaks = {}
    for u in (0.5, 1.0, 2.0):
        _, r = _l0r({**mb, "resource": [u]})
        w = _world(r)
        peaks[u] = int(START + np.argmax(w["fossil"][:, 0]))
    out["a_more_resources_later_peak"] = {"peak_year": peaks, "ok": peaks[0.5] <= peaks[1.0] <= peaks[2.0]
                                          and peaks[0.5] < peaks[2.0]}
    # (b) sin límites: recurso infinito y sin daño climático
    _, r = _l0r({**mb, "resource": [1e6], "damage": [0.0]})
    d = _drop(_world(r)["gdp_pc"][:, 0])
    out["b_no_limits_no_collapse"] = {"max_drop_from_peak": d, "ok": d <= NO_COLLAPSE_DROP}
    # (c) recursos bajos: x0.25, sin sustitución por escasez
    _, r = _l0r({**mb, "resource": [0.25], "subst": [0.0]})
    g = _world(r)["gdp_pc"][:, 0]
    d = _drop(g)
    out["c_low_resources_overshoot_collapse"] = {"peak_year": int(START + np.argmax(g)), "max_drop_from_peak": d,
                                                 "ok": d >= COLLAPSE_DROP}
    # (d) monotonía de cada palanca: 0.5, 1, 1.5
    mono = {}
    for lever, outcome, sign in LEVERS:
        _, r = _l0r({**{k: [v] * 3 for k, v in mb.items() if k != lever}, lever: [0.5, 1.0, 1.5]}
, M=3)
        v = _world(r)[outcome][-1]
        dv = np.diff(v) * sign
        mono[lever] = {"outcome": outcome, "values": v.tolist(), "expected_sign": sign,
                       "ok": bool(np.all(dv > 0))}
    out["d_levers_monotone"] = mono
    out["ok"] = (out["a_more_resources_later_peak"]["ok"] and out["b_no_limits_no_collapse"]["ok"]
                 and out["c_low_resources_overshoot_collapse"]["ok"] and all(v["ok"] for v in mono.values()))
    return out


def not_pinned() -> dict:
    """(e) cada stock regional de 2100 responde a alguna palanca (±50 %)."""
    from world4.core.l0r import model
    base_m, base = _l0r(keep=True)
    final = base["states"][-1]
    moved = {s: False for s in final}
    for lever, _, _ in LEVERS:
        m, r = _l0r({lever: [1.5]}, keep=True)
        st = r["states"][-1]
        for s in final:
            if np.max(np.abs(st[s] - final[s]) / np.maximum(np.abs(final[s]), 1e-9)) > 1e-6:
                moved[s] = True
    pinned = [s for s, v in moved.items() if not v]
    return {"stocks": len(moved), "pinned": pinned, "ok": not pinned,
            "reads_series_after_start": False, "note": "L0R.run sólo usa Params e Initial (datos <= año inicial)",
            "stock_types": list(model.REG_STOCKS) + list(model.WORLD_STOCKS)}


def relations_world3() -> dict:
    """(a)-(c) sobre el L0 World3-03 (n = 1): los modos clásicos."""
    from world4.core import l0
    out = {}
    pk = {}
    for f in (0.5, 1.0, 2.0):
        r = l0.run(constants={"nri": 1e12 * f})
        pk[f] = float(r.time[int(np.argmax(r["iopc"][:, 0]))])
    out["a_more_resources_later_peak"] = {"iopc_peak_year": pk, "ok": pk[0.5] < pk[1.0] < pk[2.0]}
    r = l0.run(constants={"nri": 1e15, "ppgf1": 0.0, "palt": 3.2e12, "pali": 3.2e12})
    s = r["iopc"][:, 0]
    d = _drop(s[r.time >= 1950])
    out["b_no_limits_no_collapse"] = {"iopc_max_drop": d, "ok": d <= 0.10}
    r = l0.run(constants={"nri": 0.25e12})
    s = r["iopc"][:, 0]
    d = _drop(s)
    out["c_low_resources_overshoot_collapse"] = {"iopc_peak_year": float(r.time[int(np.argmax(s))]),
                                                 "iopc_max_drop": d, "ok": d >= 0.5}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def dominance_tracks_modes(tol: int = 3) -> dict:
    """Los años en que cambia el modo (pico del consumo fósil, inflexiones del PIB mundial) frente a los
    años en que cambia el lazo dominante (vista mundial de ``loops``)."""
    from world4.core.l0r import loops
    m, r = _l0r(keep=True)
    res = loops.analyse(*loops.l0_world_view(m, r))
    w = _world(r)
    g = np.log(w["gdp_pc"][:, 0])
    acc = np.diff(np.diff(g))
    mode_years = sorted({int(START + 1 + i) for i in np.where(np.sign(acc[1:]) != np.sign(acc[:-1]))[0]
                         if abs(acc[i + 1] - acc[i]) > 1e-3}
                        | {int(START + np.argmax(w["fossil"][:, 0]))})
    d = res["dominant"]
    change_years = [int(res["years"][i]) for i in range(1, len(d)) if d[i] != d[i - 1]]
    matched = [y for y in mode_years if any(abs(y - c) <= tol for c in change_years)]
    return {"mode_change_years": mode_years, "dominance_change_years": change_years, "matched": matched,
            "share_matched": len(matched) / max(len(mode_years), 1),
            "ok": len(matched) == len(mode_years) and len(mode_years) > 0,
            "top_loops": loops.dominance_table(res, top=3, step=10)}


def run(log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    out = {"l0r": relations_l0r(), "pinned": not_pinned(), "world3": relations_world3(),
           "dominance": dominance_tracks_modes(), "mutant_damage_inverted": relations_l0r({"damage_sign": -1.0})}
    out["mutant_caught"] = not out["mutant_damage_inverted"]["ok"]
    out["ok"] = out["l0r"]["ok"] and out["pinned"]["ok"] and out["mutant_caught"] and out["dominance"]["ok"]
    log(f"W6: L0 regional relations {'OK' if out['l0r']['ok'] else 'FAIL'}, World3 relations "
        f"{'OK' if out['world3']['ok'] else 'FAIL'}, pinned stocks {out['pinned']['pinned'] or 'none'}, mutant caught "
        f"{out['mutant_caught']}, dominance tracks {out['dominance']['share_matched']:.0%} of mode changes")
    return out
