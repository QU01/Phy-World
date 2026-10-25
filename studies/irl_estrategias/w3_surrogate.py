"""Emulador de World3 para el aprendizaje por refuerzo: las 6 señales del acoplamiento -> 4 resultados del L0
(P(colapso <= 2070), P(colapso severo), pico de contaminación, HWI 2100) sobre los 400 estados del mundo.

World3 sobre 400 mundos cuesta ~1,3 s por miembro; un agente de RL necesita la recompensa de miles de
episodios. Se entrena un emulador (red neuronal pequeña, entrada estandarizada) con un diseño de hipercubo
latino y se valida en datos apartados. La evaluación FINAL de cada política usa World3 de verdad.
"""
import time

import numpy as np

import common as c

BOX = {"fioac2_factor": (0.8, 1.25), "resource_tech_rate": (-0.01, 0.04), "pollution_tech_rate": (-0.01, 0.05),
       "services_factor": (0.7, 1.5), "icor_factor": (0.85, 1.15), "family_factor": (0.85, 1.15)}
OUTS = ("p_collapse_2070", "p_severe", "ppolx_peak", "hwi_2100")


def design(n, seed=0):
    rng = np.random.default_rng(seed)
    u = (np.stack([rng.permutation(n) for _ in BOX], 1) + rng.uniform(size=(n, len(BOX)))) / n
    lo, hi = np.array([v[0] for v in BOX.values()]), np.array([v[1] for v in BOX.values()])
    return lo + u * (hi - lo)


def build(n=600, seed=0):
    ens = c.world3_ensemble(400)
    X = design(n, seed)
    Y = []
    t = time.time()
    for i, x in enumerate(X):
        r = c.world3_metrics(dict(zip(BOX, x.tolist())), ens)
        Y.append([r[k] for k in OUTS])
        if i % 100 == 0:
            print(f"  World3 {i}/{n} ({time.time() - t:.0f}s)", flush=True)
    c.dump("w3_design.json", {"X": X, "Y": np.array(Y), "keys": list(BOX), "outs": OUTS})


class Surrogate:
    def __init__(self, hidden=64, seed=0):
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        d = c.load("w3_design.json")
        X, Y = np.array(d["X"]), np.array(d["Y"])
        self.mx, self.sx = X.mean(0), X.std(0)
        self.my, self.sy = Y.mean(0), Y.std(0)
        n = len(X)
        idx = np.random.default_rng(seed).permutation(n)
        te, tr = idx[: n // 5], idx[n // 5:]
        net = torch.nn.Sequential(torch.nn.Linear(6, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                  torch.nn.SiLU(), torch.nn.Linear(hidden, 4))
        opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-5)
        xt = torch.tensor((X - self.mx) / self.sx, dtype=torch.float32)
        yt = torch.tensor((Y - self.my) / self.sy, dtype=torch.float32)
        for ep in range(3000):
            opt.zero_grad()
            loss = ((net(xt[tr]) - yt[tr]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = net(xt[te]).numpy()
        self.r2 = dict(zip(OUTS, (1 - ((p - yt[te].numpy()) ** 2).mean(0) / yt[te].numpy().var(0)).round(3).tolist()))
        self.net = net

    def __call__(self, M):
        """M: (B, 6) en el orden de BOX (se recorta a la caja) -> (B, 4)."""
        lo, hi = np.array([v[0] for v in BOX.values()]), np.array([v[1] for v in BOX.values()])
        M = np.clip(M, lo, hi)
        with self.torch.no_grad():
            p = self.net(self.torch.tensor((M - self.mx) / self.sx, dtype=self.torch.float32)).numpy()
        return p * self.sy + self.my


if __name__ == "__main__":
    build()
    s = Surrogate()
    print("R2 fuera de muestra", s.r2)
