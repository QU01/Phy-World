"""Paso 2b: ¿cuánto del resultado en World3 depende de cada canal del acoplamiento L1 -> L0?

Para cada estrategia del paso 2: (a) atribución: se apaga un canal del acoplamiento a la vez (se deja en
su valor neutro) y se mide el cambio en P(colapso <= 2070), P(colapso severo), pico de contaminación y
HWI 2100; (b) intensidad: todas las señales a la mitad y al doble (en escala logarítmica para los
factores).
"""
import numpy as np

import common as c

NEUTRAL = {"fioac2_factor": 1.0, "resource_tech_rate": 0.0, "pollution_tech_rate": 0.0, "services_factor": 1.0,
           "icor_factor": 1.0, "family_factor": 1.0}
KEYS = ("p_collapse_2070", "p_severe", "ppolx_peak", "hwi_2100", "le_2100")


def scaled(mp, a):
    return {k: (v * a if k.endswith("rate") else float(np.exp(np.log(v) * a))) for k, v in mp.items()}


def main():
    ens = c.world3_ensemble(400)
    d = c.load("strategies.json")["strategies"]
    out = {}
    base = c.world3_metrics(NEUTRAL, ens)
    for name, v in d.items():
        mp = v["w3_mapping"]
        full = {k: v["w3"][k] for k in KEYS}
        att = {}
        for ch in NEUTRAL:
            r = c.world3_metrics({**mp, ch: NEUTRAL[ch]}, ens)
            att[ch] = {k: full[k] - r[k] for k in KEYS}       # contribución del canal
        inten = {}
        for a in (0.5, 2.0):
            r = c.world3_metrics(scaled(mp, a), ens)
            inten[str(a)] = {k: r[k] for k in KEYS}
        out[name] = {"full": full, "attribution": att, "intensity": inten}
        print(name, {k: round(full[k], 3) for k in KEYS}, "x0.5", {k: round(inten["0.5"][k], 3) for k in KEYS[:2]},
              "x2", {k: round(inten["2.0"][k], 3) for k in KEYS[:2]}, flush=True)
    out["_base"] = {k: base[k] for k in KEYS}
    c.dump("sensitivity.json", out)


if __name__ == "__main__":
    main()
