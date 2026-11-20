"""Recalibración de la ecuación de fecundidad de World3 con datos, y su prueba fuera de muestra.

World3 hace depender el tamaño de familia deseado del producto industrial por persona de 20 años antes (tabla
``sfsn``). Aquí se reestiman sus tres primeros puntos (0, 200 y 400 $ por persona; el resto conserva su proporción)
para que la fecundidad total del modelo con cohortes siga a la observada hasta ``cutoff``, y se mide el error
DESPUÉS de ``cutoff``, que el ajuste no vio.

Resultado medido con ``cutoff = 2000`` (``python -m world4.core.l0v.fit``): el error dentro de muestra baja de
8,4 % a 4,1 %, pero fuera de muestra (2001-2023) SUBE de 5,1 % a 16,3 %. El ajuste prolonga la caída rápida de
1970-2000 y no ve que después se frena. Una sola curva mundial de fecundidad frente al ingreso medio no
reproduce las dos fases: por eso ``world4.core.l0v.model`` NO usa esta tabla y la fecundidad queda como en
World3 hasta que se trate por grupos de países.
"""
from __future__ import annotations

import numpy as np

SFSN_TAIL = (0.59 / 0.715, 0.5 / 0.715)          # los puntos de 600 y 800 $, en proporción al de 400 $
X0 = (1.25, 0.248, 0.239)                        # la tabla de World3-03 en esta parametrización


def sfsn_table(theta):
    """``theta = (v0, a, b)``: valor en 0 $, y caídas relativas hasta 200 $ y de 200 a 400 $."""
    v0, a, b = theta
    v200 = v0 * (1 - a)
    v400 = v200 * (1 - b)
    return (0, 800, 200, [v0, v200, v400, v400 * SFSN_TAIL[0], v400 * SFSN_TAIL[1]])


def _tfr_model(theta, t1, inputs):
    from world4.core.l0v import model
    tables = None if theta is None else {"sfsn": sfsn_table(theta)}
    r, _ = model.run(tables=tables, t1=float(t1), inputs=inputs)
    t = np.asarray(r.time)
    ok = np.isclose(t, np.round(t)) & (t >= inputs.obs_years[0] + 1)
    return np.round(t[ok]).astype(int), r["tf"][ok, 0]


def errors(theta, cutoff: int, inputs=None, last: int | None = None) -> dict:
    """Error medio de la fecundidad total (%) hasta ``cutoff`` y después, con la tabla ``theta`` (None: World3)."""
    from world4.core.l0v import model
    inputs = inputs or model.world_inputs()
    last = last or int(inputs.obs_years[-1])
    years, tf = _tfr_model(theta, last, inputs)
    obs = np.interp(years, inputs.obs_years, inputs.tfr)
    e = np.abs(tf / obs - 1)
    return {"in_sample_pct": float(100 * e[years <= cutoff].mean()),
            "out_of_sample_pct": float(100 * e[years > cutoff].mean())}


def fit_fertility(cutoff: int = 2000, inputs=None) -> dict:
    """Ajusta ``theta`` con la fecundidad observada hasta ``cutoff`` y devuelve el ajuste y sus dos errores."""
    from scipy.optimize import minimize
    from world4.core.l0v import model
    inputs = inputs or model.world_inputs()

    def loss(theta):
        if not (0.5 < theta[0] < 2.5 and 0 < theta[1] < 0.9 and 0 < theta[2] < 0.9):
            return 1e3
        years, tf = _tfr_model(theta, cutoff, inputs)
        obs = np.interp(years, inputs.obs_years, inputs.tfr)
        return float(np.mean((np.log(tf) - np.log(obs)) ** 2))

    res = minimize(loss, X0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 400})
    theta = tuple(float(v) for v in res.x)
    return {"cutoff": cutoff, "theta": theta, "sfsn": sfsn_table(theta),
            "fitted": errors(theta, cutoff, inputs), "world3": errors(None, cutoff, inputs)}


if __name__ == "__main__":
    import json
    print(json.dumps(fit_fertility(), indent=1))
