"""F8: el problema inverso del Atlas como modelo ALTERNATIVO de conducta en el L1.

«Lo que harían los países» frente a «lo que les convendría»:

* **harían** = la regla de conducta clonada de los datos (``behavior.BehaviorPolicy``), la del L1 de siempre;
* **convendría** = la misma regla desplazada ±0,5 en el logit de cada canal en la dirección que el
  problema inverso del Atlas (BBL, ``irl.py``) midió como RENTABLE EN BIENESTAR en más de la mitad de los
  casos (``irl.json``: ``desvios_rentables``). Son exactamente los desvíos y el tamaño (0,5) que el Atlas
  usó para estimar la recompensa revelada: una región de confianza alrededor de lo observado, no un
  óptimo de esquina (la misma idea que ``recommend.py``).

Se comparan con números aleatorios comunes desde el origen vivo 2019 hasta 2030 con el L1 F1-F5:
consumo y PIB per cápita, riesgo de conflicto, deuda, CO2 fósil, y el valor de cada trayectoria con la
recompensa REVELADA (``theta_unidades`` de ``irl.json``: poder relativo e inercia pesan; el bienestar,
en negativo) y con la de bienestar (``irl.HAND["bienestar"]``).

El entrenamiento por refuerzo del Atlas (``recommend.py``, PPO anclado) necesita ``autograd`` y horas de
cómputo: no corre aquí (extra ``world4[rl]``); lo que se conserva es su problema inverso y su región de
confianza.
"""
from __future__ import annotations

import json

import numpy as np

DELTA = 0.5                   # el desvío de ``irl.collect_deviations`` (logit)
GAMMA = 0.95                  # el descuento del IRL
HORIZON = 11                  # 2019 -> 2030 (el modo vivo de F0.5)


def irl_results() -> dict:
    from world4.core.l1_atlas._paths import ATLAS_ROOT
    return json.loads((ATLAS_ROOT / "results" / "irl.json").read_text(encoding="utf-8"))


def advised_offsets(irl: dict | None = None) -> dict:
    """{canal: ±DELTA} para los canales cuyo desvío en una dirección fue rentable en bienestar > 50 %."""
    d = (irl or irl_results())["desvios_rentables"]
    out = {}
    for ch in ("k", "r", "m", "x", "f", "w"):
        up, down = d.get(f"{ch}+", 0.0), d.get(f"{ch}-", 0.0)
        if up > 0.5 and up >= down:
            out[ch] = DELTA
        elif down > 0.5:
            out[ch] = -DELTA
    return out


def compare(R: int = 32, seed: int = 0, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import irl as airl
    from world4.structural import engine
    W = engine.live_world()
    t0 = engine.L1_LAST_DATA_YEAR - 1950
    irl = irl_results()
    offs = advised_offsets(irl)
    runs = {}
    for name, o in (("would", None), ("advised", offs)):
        runs[name] = coupling.forecast(t0, R=R, seed=seed, world=W, horizon=HORIZON, f5=coupling.F5Config(),
                                       offsets=o, raw=True)
    th_rev = np.array([irl["theta_unidades"][f] for f in airl.FEATS])
    th_wel = np.array(airl.HAND["bienestar"], float)
    out = {"origin": 2019, "end": 2019 + HORIZON, "R": R, "advised_offsets_logit": offs,
           "profitable_deviation_shares": irl["desvios_rentables"], "rationality_index": irl["racionalidad"],
           "revealed_weights_units": irl["theta_unidades"], "conducts": {}, "exploratory": True, "fidelity": "L1"}
    for name, f in runs.items():
        raw = f["raw"]
        pop = np.nansum(f["pop"], 1)                                   # (R, H+1)
        cons = np.nansum(raw["cons_pc"] * f["pop"], 1) / pop
        gdp = np.nansum(np.exp(f["ly"]) * f["pop"], 1) / pop
        haz = np.nanmean(f["hazard"], 1)
        Y = np.exp(f["ly"]) * f["pop"]
        debt = np.nansum(f["debt"] * Y, 1) / np.nansum(Y, 1)
        phi = airl.reward_features(raw, W, t0)                          # (R, N, H, F)
        val = airl.discounted(phi, GAMMA)                               # (R, N, F)
        out["conducts"][name] = {
            "consumption_pc_2030": float(np.median(cons[:, -1])), "gdp_pc_2030": float(np.median(gdp[:, -1])),
            "conflict_hazard_mean_2030": float(np.median(haz[:, -1])), "debt_world_2030": float(np.median(debt[:, -1])),
            "co2_ffi_2030": float(np.median(f["climate_co2_ffi"][:, -1])),
            "_val": val, "_cons": cons[:, -1], "_gdp": gdp[:, -1], "_haz": haz[:, -1]}
    a, b = out["conducts"]["advised"], out["conducts"]["would"]
    dv = a.pop("_val") - b.pop("_val")                                  # (R, N, F)
    rev = np.nanmean(dv @ th_rev, 1)                                    # media por país, por miembro
    wel = np.nanmean(dv @ th_wel, 1)
    out["advised_minus_would"] = {
        "consumption_pc_2030_rel": float(np.median(a["_cons"] / b["_cons"] - 1)),
        "gdp_pc_2030_rel": float(np.median(a["_gdp"] / b["_gdp"] - 1)),
        "conflict_hazard_2030": float(np.median(a["_haz"] - b["_haz"])),
        "revealed_value": float(np.median(rev)), "revealed_value_share_members_better": float(np.mean(rev > 0)),
        "welfare_value": float(np.median(wel)), "welfare_value_share_members_better": float(np.mean(wel > 0)),
        "countries_with_higher_welfare_value": float(np.nanmean(np.nanmean(dv @ th_wel, 0) > 0))}
    for c in out["conducts"].values():
        for k in [k for k in c if k.startswith("_")]:
            c.pop(k)
    d = out["advised_minus_would"]
    log(f"conduct 2019->2030: advised (IRL-profitable deviations {offs}) vs would: consumption pc "
        f"{d['consumption_pc_2030_rel']:+.1%}, GDP pc {d['gdp_pc_2030_rel']:+.1%}, conflict hazard "
        f"{d['conflict_hazard_2030']:+.4f}; welfare value {d['welfare_value']:+.4f} (better in "
        f"{d['welfare_value_share_members_better']:.0%} of members), revealed value {d['revealed_value']:+.4f} "
        f"(better in {d['revealed_value_share_members_better']:.0%})")
    return out
