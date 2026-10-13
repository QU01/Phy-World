"""Dominancia de lazos sobre el L0: puntuación de lazos de *Loops that Matter* (Schoenberg, Davidsen y
Eberlein 2020, System Dynamics Review 36(2):158-190; arXiv 1908.11434). Implementación propia: el método
sólo existe dentro de STELLA, que es comercial. La fórmula se escribió desde la descripción del artículo
[S: no se contrastó con su código] y se valida con la logística, donde el cambio de dominancia del lazo
de nacimientos (+) al de muertes (-) tiene que caer en P = K/2 (prueba).

Para cada intervalo (t-1, t):

* enlace stock -> flujo:  LS(s→f) = |Δ_s f / Δf| · signo(Δ_s f / Δs), con Δ_s f el cambio que habría tenido
  f si sólo hubiera cambiado s (el resto en t-1) y Δf el cambio total (0 si Δf = 0);
* enlace flujo -> stock:  LS(f→s) = ±Δf / Δ(flujo neto de s)  (+ entrada, - salida);
* lazo: el producto de sus enlaces; puntuación relativa = lazo / Σ |lazos|.

Vista MUNDIAL del L0 regional: cada nodo es un TIPO de stock (los de las 10 regiones se perturban a la
vez) y cada flujo es la suma de sus regiones; así hay 13 nodos y los lazos se pueden enumerar (todos los
ciclos elementales de hasta ``MAX_LEN`` stocks). Lo que no es lazo de stocks (los flujos en diferencias
que dependen del año anterior: Δly, ΔlnP, ΔT) queda como cambio no atribuido y se informa.
"""
from __future__ import annotations

import numpy as np

MAX_LEN = 4


def link_scores(stocks, flows_map, eval_flows, S_prev, S_now, prev_prev, prev_now):
    """Puntuaciones de enlace del intervalo. ``eval_flows(states, prevs)`` evalúa los flujos de una lista
    de estados y devuelve {flujo: array (B,)} (ya agregados). Devuelve (LS_sf, LS_fs, dS) con
    LS_sf[(s, f)] y LS_fs[f]."""
    n = len(stocks)
    states = [S_prev] + [{**S_prev, s: S_now[s]} for s in stocks] + [S_now]
    prevs = [prev_prev] * (n + 1) + [prev_now]
    F = eval_flows(states, prevs)
    dF = {f: F[f][-1] - F[f][0] for f in flows_map}
    dnet = {s: 0.0 for s in stocks}
    for f, (tgt, sign) in flows_map.items():
        dnet[tgt] = dnet[tgt] + sign * dF[f]
    LS_sf, LS_fs = {}, {}
    for f, (tgt, sign) in flows_map.items():
        LS_fs[f] = sign * dF[f] / dnet[tgt] if dnet[tgt] != 0 else 0.0
        for k, s in enumerate(stocks):
            ds_f = F[f][k + 1] - F[f][0]
            dS = _delta(S_now[s], S_prev[s])
            if dF[f] == 0 or ds_f == 0 or dS == 0:
                continue
            LS_sf[(s, f)] = abs(ds_f / dF[f]) * np.sign(ds_f / dS)
    return LS_sf, LS_fs


def _delta(a, b):
    return float(np.sum(np.asarray(a) - np.asarray(b)))


def cycles(stocks, flows_map, edges, max_len=MAX_LEN):
    """Ciclos elementales (s1 -f1-> s2 -f2-> ... -> s1) del grafo de stocks, de hasta ``max_len`` stocks.
    ``edges``: conjunto de (s, f) con enlace s→f."""
    out_edges = {s: [(f, flows_map[f][0]) for (ss, f) in edges if ss == s] for s in stocks}
    found = []
    order = {s: i for i, s in enumerate(stocks)}

    def dfs(start, node, path):
        for f, nxt in out_edges[node]:
            if nxt == start:
                found.append(tuple(path + [(node, f)]))
            elif order[nxt] > order[start] and nxt not in [p[0] for p in path] + [node] and len(path) + 1 < max_len:
                dfs(start, nxt, path + [(node, f)])

    for s in stocks:
        dfs(s, s, [])
    return found


def loop_name(loop) -> str:
    return " -> ".join(f"{s} -[{f}]" for s, f in loop) + f" -> {loop[0][0]}"


def loop_polarity(loop, LS_sf, LS_fs):
    v = 1.0
    for s, f in loop:
        v *= LS_sf.get((s, f), 0.0) * LS_fs.get(f, 0.0)
    return v


def analyse(stocks, flows_map, eval_flows, states, prevs, years, max_len=MAX_LEN) -> dict:
    """Puntuación relativa de cada lazo en cada intervalo. ``states[t]`` y ``prevs[t]`` son el estado y la
    memoria del año anterior en el año t."""
    LS = []
    edges = set()
    for t in range(1, len(states)):
        a, b = link_scores(stocks, flows_map, eval_flows, states[t - 1], states[t], prevs[t - 1], prevs[t])
        LS.append((a, b))
        edges |= {k for k, v in a.items() if v != 0}
    loops = cycles(stocks, flows_map, edges, max_len)
    scores = np.zeros((len(LS), len(loops)))
    for t, (a, b) in enumerate(LS):
        for j, lp in enumerate(loops):
            scores[t, j] = loop_polarity(lp, a, b)
    tot = np.abs(scores).sum(1, keepdims=True)
    rel = np.where(tot > 0, scores / np.where(tot > 0, tot, 1.0), 0.0)
    names = [loop_name(lp) for lp in loops]
    dom = [names[int(np.argmax(np.abs(rel[t])))] if tot[t, 0] > 0 else None for t in range(len(LS))]
    return {"years": list(years[1:]), "loops": names, "relative": rel, "dominant": dom, "raw": scores}


# ---------------------------------------------------------------------- el L0 regional, vista mundial
def l0_world_view(model_obj, run_out):
    """Prepara ``analyse`` para el L0 regional (``model.L0R`` corrido con ``keep_states=True``)."""
    from world4.core.l0r import model as m
    p, ini = model_obj.p, model_obj.ini
    stocks = list(m.REG_STOCKS) + list(m.WORLD_STOCKS)
    flows_map = dict(m.FLOWS)
    states = run_out["states"]
    T = len(states)
    probe = m.L0R(p, ini, M=len(stocks) + 2)
    probe._consts()
    base_net0 = model_obj._net0
    # memoria del año anterior en cada año (la misma que usa ``run``)
    auxes = []
    for st in states:
        model_obj._net0 = base_net0
        auxes.append(model_obj.aux(st))
    prevs = [{"ly": auxes[0]["ly"], "lnP": auxes[0]["lP"], "Treg": auxes[0]["Treg"]}] + \
            [{"ly": a["ly"], "lnP": a["lP"], "Treg": a["Treg"]} for a in auxes[:-1]]

    def eval_flows(sts, pvs):
        B = len(sts)
        batch = {k: np.concatenate([s[k] for s in sts], 0) for k in sts[0]}
        prev = {k: np.concatenate([pv[k] for pv in pvs], 0) for k in pvs[0]}
        probe._net0 = np.broadcast_to(base_net0, (B,) + base_net0.shape[1:])
        if probe.M != B:
            raise ValueError("probe size")
        fl = probe.flows(batch, None, prev)
        out = {}
        for f, v in fl.items():
            v = np.asarray(v, float)
            v = np.broadcast_to(v, (B,) + v.shape[1:]) if v.ndim else np.full((B, 1), float(v))
            out[f] = v.reshape(B, -1).sum(1)
        return out

    return stocks, flows_map, eval_flows, states, prevs, run_out["years"][:T]


def dominance_table(res, top: int = 3, step: int = 10) -> list:
    """Los ``top`` lazos de mayor |puntuación relativa| cada ``step`` años."""
    rows = []
    for t in range(0, len(res["years"]), step):
        r = res["relative"][t]
        idx = np.argsort(-np.abs(r))[:top]
        rows.append({"year": int(res["years"][t]),
                     "loops": [(res["loops"][i], round(float(r[i]), 3)) for i in idx if r[i] != 0]})
    return rows

