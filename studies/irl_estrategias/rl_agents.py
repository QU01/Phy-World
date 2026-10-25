"""Agentes de aprendizaje por refuerzo para ``rl_env.Env`` (PyTorch, CPU).

* **On-policy:** PPO (Schulman et al. 2017) con GAE; política gaussiana, valor aprendido.
* **Off-policy:** SAC (Haarnoja et al. 2018, temperatura automática) y TD3 (Fujimoto et al. 2018), con
  memoria de repetición.
* **Basados en modelo:** un ensemble probabilístico de 5 redes aprende la dinámica (Δestado) y la recompensa;
  - PETS-MPC (Chua et al. 2018): en cada decisión, CEM sobre secuencias de acciones en el modelo
    (muestreo de trayectorias TS-1) y control de horizonte móvil;
  - MBPO / Dyna (Janner et al. 2019): SAC entrenado sobre ramas cortas generadas por el modelo desde estados
    reales, más una fracción de datos reales.

Todas las políticas producen acciones en [-0,5, 0,5]^18. Episodio: 5 decisiones, γ = 1 (horizonte finito,
con el índice de paso en el estado).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
A_MAX = 0.5


def mlp(i, o, h=128, n=2, act=nn.SiLU):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Normalizer:
    def __init__(self, X):
        self.m = X.mean(0)
        self.s = X.std(0) + 1e-6

    def __call__(self, x):
        return (x - self.m) / self.s

    def t(self, x):
        return torch.as_tensor(self(x), dtype=torch.float32)


def flatten(trs):
    """Lista de rollouts -> transiciones (s, a, r, s2, done)."""
    S, A, Rw, S2, D = [], [], [], [], []
    for tr in trs:
        K = tr["obs"].shape[0]
        for k in range(K):
            S.append(tr["obs"][k])
            A.append(tr["act"][k])
            Rw.append(tr["rew"][k])
            S2.append(tr["obs"][k + 1] if k + 1 < K else tr["obs_T"])
            D.append(np.full(tr["rew"].shape[1], float(k == K - 1)))
    return tuple(np.concatenate(x) for x in (S, A, Rw, S2, D))


# ================================================================== PPO (on-policy)
class PPO:
    name = "PPO"

    def __init__(self, norm, obs_dim, act_dim, lr=3e-4, clip=0.2, epochs=10, lam=0.95, seed=0):
        torch.manual_seed(seed)
        self.norm = norm
        self.pi = mlp(obs_dim, act_dim, 64, 2, nn.Tanh)
        self.logstd = nn.Parameter(torch.full((act_dim,), np.log(0.18)))
        self.v = mlp(obs_dim, 1, 64, 2, nn.Tanh)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + [self.logstd] + list(self.v.parameters()), lr)
        self.clip, self.epochs, self.lam = clip, epochs, lam
        self.rng = np.random.default_rng(seed)

    def act(self, obs, step, deterministic=False):
        with torch.no_grad():
            mu = A_MAX * torch.tanh(self.pi(self.norm.t(obs)))
            if deterministic:
                return mu.numpy()
            return (mu + torch.exp(self.logstd) * torch.randn_like(mu)).numpy()

    def update(self, trs):
        obs = np.concatenate([tr["obs"] for tr in trs], 1)          # (K, B, d)
        act = np.concatenate([tr["act_raw"] for tr in trs], 1)
        rew = np.concatenate([tr["rew"] for tr in trs], 1)
        K, B, _ = obs.shape
        o = self.norm.t(obs.reshape(K * B, -1))
        a = torch.as_tensor(act.reshape(K * B, -1), dtype=torch.float32)
        with torch.no_grad():
            v = self.v(o).reshape(K, B).numpy()
            mu = A_MAX * torch.tanh(self.pi(o))
            lp_old = torch.distributions.Normal(mu, torch.exp(self.logstd)).log_prob(a).sum(-1)
        adv = np.zeros((K, B))
        last = np.zeros(B)
        for k in reversed(range(K)):
            nv = v[k + 1] if k + 1 < K else 0.0
            delta = rew[k] + nv - v[k]
            last = delta + self.lam * last
            adv[k] = last
        ret = torch.as_tensor((adv + v).reshape(-1), dtype=torch.float32)
        advt = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        advt = (advt - advt.mean()) / (advt.std() + 1e-8)
        n = K * B
        for _ in range(self.epochs):
            for idx in np.array_split(self.rng.permutation(n), max(1, n // 128)):
                mu = A_MAX * torch.tanh(self.pi(o[idx]))
                lp = torch.distributions.Normal(mu, torch.exp(self.logstd)).log_prob(a[idx]).sum(-1)
                ratio = torch.exp(lp - lp_old[idx])
                l1 = -torch.min(ratio * advt[idx], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advt[idx]).mean()
                l2 = ((self.v(o[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                self.opt.zero_grad()
                (l1 + 0.5 * l2).backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), 0.5)
                self.opt.step()
        with torch.no_grad():
            self.logstd.clamp_(np.log(0.02), np.log(0.3))


# ================================================================== SAC y TD3 (off-policy)
class Replay:
    def __init__(self):
        self.data = None

    def add(self, trans):
        self.data = trans if self.data is None else tuple(np.concatenate([a, b]) for a, b in zip(self.data, trans))

    def __len__(self):
        return 0 if self.data is None else len(self.data[0])

    def sample(self, n, rng):
        i = rng.integers(0, len(self), n)
        return tuple(x[i] for x in self.data)


class SAC:
    name = "SAC"

    def __init__(self, norm, obs_dim, act_dim, lr=3e-4, tau=0.01, seed=0):
        torch.manual_seed(seed)
        self.norm, self.tau = norm, tau
        self.actor = mlp(obs_dim, 2 * act_dim)
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
        self.q1t, self.q2t = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.05), requires_grad=True)
        self.target_ent = -float(act_dim)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr)
        self.oal = torch.optim.Adam([self.log_alpha], lr)
        self.rng = np.random.default_rng(seed)
        self.buf = Replay()
        self.act_dim = act_dim

    def _pi(self, o):
        out = self.actor(o)
        mu, ls = out[:, :self.act_dim], out[:, self.act_dim:].clamp(-5, 1)
        std = torch.exp(ls)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        lp = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
        return A_MAX * a, lp, A_MAX * torch.tanh(mu)

    def act(self, obs, step, deterministic=False):
        with torch.no_grad():
            a, _, m = self._pi(self.norm.t(obs))
            return (m if deterministic else a).numpy()

    def train(self, steps, batch=256, buf=None):
        buf = buf or self.buf
        for _ in range(steps):
            s, a, r, s2, d = buf.sample(batch, self.rng)
            s, s2 = self.norm.t(s), self.norm.t(s2)
            a, r, d = (torch.as_tensor(x, dtype=torch.float32) for x in (a, r, d))
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                a2, lp2, _ = self._pi(s2)
                q = torch.min(self.q1t(torch.cat([s2, a2], 1)), self.q2t(torch.cat([s2, a2], 1))).squeeze(-1)
                y = r + (1 - d) * (q - alpha * lp2)
            sa = torch.cat([s, a], 1)
            lq = ((self.q1(sa).squeeze(-1) - y) ** 2).mean() + ((self.q2(sa).squeeze(-1) - y) ** 2).mean()
            self.oq.zero_grad()
            lq.backward()
            self.oq.step()
            an, lp, _ = self._pi(s)
            qn = torch.min(self.q1(torch.cat([s, an], 1)), self.q2(torch.cat([s, an], 1))).squeeze(-1)
            la = (alpha * lp - qn).mean()
            self.oa.zero_grad()
            la.backward()
            self.oa.step()
            lal = -(self.log_alpha * (lp.detach() + self.target_ent)).mean()
            self.oal.zero_grad()
            lal.backward()
            self.oal.step()
            with torch.no_grad():
                for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p)


class TD3:
    name = "TD3"

    def __init__(self, norm, obs_dim, act_dim, lr=3e-4, tau=0.01, noise=0.08, seed=0):
        torch.manual_seed(seed)
        self.norm, self.tau, self.noise = norm, tau, noise
        self.actor, self.actor_t = mlp(obs_dim, act_dim), mlp(obs_dim, act_dim)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
        self.q1t, self.q2t = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.oa = torch.optim.Adam(self.actor.parameters(), lr)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr)
        self.rng = np.random.default_rng(seed)
        self.buf = Replay()
        self.it = 0

    def _mu(self, o, net=None):
        return A_MAX * torch.tanh((net or self.actor)(o))

    def act(self, obs, step, deterministic=False):
        with torch.no_grad():
            m = self._mu(self.norm.t(obs)).numpy()
        return m if deterministic else m + self.noise * self.rng.standard_normal(m.shape)

    def train(self, steps, batch=256):
        for _ in range(steps):
            self.it += 1
            s, a, r, s2, d = self.buf.sample(batch, self.rng)
            s, s2 = self.norm.t(s), self.norm.t(s2)
            a, r, d = (torch.as_tensor(x, dtype=torch.float32) for x in (a, r, d))
            with torch.no_grad():
                a2 = (self._mu(s2, self.actor_t) + (0.05 * torch.randn_like(a)).clamp(-0.1, 0.1)).clamp(-A_MAX, A_MAX)
                q = torch.min(self.q1t(torch.cat([s2, a2], 1)), self.q2t(torch.cat([s2, a2], 1))).squeeze(-1)
                y = r + (1 - d) * q
            sa = torch.cat([s, a], 1)
            lq = ((self.q1(sa).squeeze(-1) - y) ** 2).mean() + ((self.q2(sa).squeeze(-1) - y) ** 2).mean()
            self.oq.zero_grad()
            lq.backward()
            self.oq.step()
            if self.it % 2 == 0:
                la = -self.q1(torch.cat([s, self._mu(s)], 1)).mean()
                self.oa.zero_grad()
                la.backward()
                self.oa.step()
                with torch.no_grad():
                    for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t), (self.actor, self.actor_t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - self.tau).add_(self.tau * p)


# ================================================================== modelo de dinámica (model-based)
class EnsembleModel:
    """Ensemble de 5: dinámica gaussiana (s, a) -> Δs (verosimilitud) y, aparte, recompensa (s, a) -> r con
    redes pequeñas, MSE, regularización y parada temprana (la recompensa terminal es ruidosa y escasa).
    La incertidumbre epistémica es la discrepancia entre miembros (cada uno con su bootstrap)."""

    def __init__(self, norm, obs_dim, act_dim, E=5, h=200, seed=0):
        torch.manual_seed(seed)
        self.norm, self.E, self.od = norm, E, obs_dim
        self.nets = [mlp(obs_dim + act_dim + 1, 2 * obs_dim, h, 3) for _ in range(E)]
        self.rnets = [mlp(obs_dim + act_dim + 1, 1, 64, 2) for _ in range(E)]
        self.rng = np.random.default_rng(seed)
        self.dm = self.ds = None

    def _in(self, s, a, d):
        return torch.cat([s, a / A_MAX, d[:, None]], 1)

    def _train(self, net, X, Y, tr, va, nll, epochs, wd, batch=256, patience=8):
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
        best, state, bad = np.inf, None, 0
        for _ in range(epochs):
            for idx in np.array_split(self.rng.permutation(tr), max(1, len(tr) // batch)):
                out = net(X[idx])
                if nll:
                    mu, lv = out[:, :self.od], out[:, self.od:].clamp(-8, 4)
                    loss = (((mu - Y[idx]) ** 2) * torch.exp(-lv) + lv).mean()
                else:
                    loss = ((out - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = ((net(X[va])[:, :Y.shape[1]] - Y[va]) ** 2).mean().item()
            if v < best - 1e-5:
                best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(state)
        return best

    def fit(self, data, epochs=80):
        S, A, Rw, S2, D = data
        s, s2 = self.norm(S), self.norm(S2)
        delta = s2 - s
        self.dm, self.ds = delta.mean(0), delta.std(0) + 1e-6
        self.rm, self.rs = Rw.mean(), Rw.std() + 1e-6
        X = torch.as_tensor(np.concatenate([s, A / A_MAX, D[:, None]], 1), dtype=torch.float32)
        Y = torch.as_tensor((delta - self.dm) / self.ds, dtype=torch.float32)
        Yr = torch.as_tensor(((Rw - self.rm) / self.rs)[:, None], dtype=torch.float32)
        n = len(X)
        perm = self.rng.permutation(n)
        te, rest = perm[: n // 10], perm[n // 10:]
        for net, rnet in zip(self.nets, self.rnets):
            boot = self.rng.choice(rest, len(rest))
            va = boot[: len(boot) // 8]
            tr = boot[len(boot) // 8:]
            self._train(net, X, Y, tr, va, True, epochs, 1e-5)
            self._train(rnet, X, Yr, tr, va, False, epochs, 1e-3)
        with torch.no_grad():
            mu = torch.stack([nn_(X[te])[:, :self.od] for nn_ in self.nets]).mean(0)
            rp = torch.stack([r(X[te]) for r in self.rnets]).mean(0)
            es = ((mu - Y[te]) ** 2).mean().item()
            dl = D[te] == 1
            rr = ((rp - Yr[te]) ** 2).squeeze(-1).numpy()
            yr = Yr[te].squeeze(-1).numpy()
        return {"r2_state": float(1 - es), "r2_reward": float(1 - rr.mean() / yr.var()),
                "r2_reward_terminal": float(1 - rr[dl].mean() / yr[dl].var()) if dl.any() else None}

    def step(self, s, a, d, member=None, sample=True):
        """s normalizado (B, d) tensor -> (s2 normalizado, r). member (B,) índice del ensemble (TS-1)."""
        x = self._in(s, a, d)
        outs = torch.stack([n(x) for n in self.nets])                 # (E, B, 2d)
        rs = torch.stack([n(x) for n in self.rnets])[..., 0]          # (E, B)
        idx = member if member is not None else torch.as_tensor(self.rng.integers(0, self.E, len(s)))
        ar = torch.arange(len(s))
        out = outs[idx, ar]
        mu, lv = out[:, :self.od], out[:, self.od:].clamp(-8, 4)
        y = mu + (torch.exp(0.5 * lv) * torch.randn_like(mu) if sample else 0)
        y = y * torch.as_tensor(self.ds, dtype=torch.float32) + torch.as_tensor(self.dm, dtype=torch.float32)
        r = rs[idx, ar] * self.rs + self.rm
        return s + y, r


class PETS:
    """MPC con CEM sobre el ensemble (trayectorias TS-1)."""
    name = "PETS-MPC"

    def __init__(self, norm, obs_dim, act_dim, K=5, pop=256, elite=26, iters=5, seed=0):
        self.norm, self.ad, self.K = norm, act_dim, K
        self.model = EnsembleModel(norm, obs_dim, act_dim, seed=seed)
        self.pop, self.elite, self.iters = pop, elite, iters
        self.rng = np.random.default_rng(seed)
        self.data = None
        self.noise = 0.0

    def add(self, trans):
        self.data = trans if self.data is None else tuple(np.concatenate([a, b]) for a, b in zip(self.data, trans))

    def fit(self):
        return self.model.fit(self.data)

    @torch.no_grad()
    def act(self, obs, step, deterministic=False):
        B, H, P = len(obs), self.K - step, self.pop
        mu = torch.zeros(B, H, self.ad)
        sd = torch.full((B, H, self.ad), 0.25)
        s0 = self.norm.t(obs)
        for _ in range(self.iters):
            a = (mu[:, None] + sd[:, None] * torch.randn(B, P, H, self.ad)).clamp(-A_MAX, A_MAX)
            s = s0[:, None].expand(B, P, -1).reshape(B * P, -1)
            mem = torch.as_tensor(self.rng.integers(0, self.model.E, B * P))
            ret = torch.zeros(B * P)
            for h in range(H):
                d = torch.full((B * P,), float(step + h == self.K - 1))
                s, r = self.model.step(s, a[:, :, h].reshape(B * P, -1), d, member=mem)
                ret += r
            ret = ret.reshape(B, P)
            top = ret.topk(self.elite, 1).indices                      # (B, elite)
            el = torch.gather(a, 1, top[:, :, None, None].expand(-1, -1, H, self.ad))
            mu, sd = 0.8 * el.mean(1) + 0.2 * mu, (0.8 * el.std(1) + 0.2 * sd).clamp_min(0.02)
        out = mu[:, 0].numpy()
        if not deterministic and self.noise:
            out = out + self.noise * self.rng.standard_normal(out.shape)
        return out


class MBPO:
    """SAC entrenado sobre ramas del modelo (Dyna) desde estados reales."""
    name = "MBPO-Dyna"

    def __init__(self, norm, obs_dim, act_dim, K=5, seed=0, real_frac=0.1):
        self.model = EnsembleModel(norm, obs_dim, act_dim, seed=seed)
        self.sac = SAC(norm, obs_dim, act_dim, seed=seed)
        self.norm, self.K, self.real_frac = norm, K, real_frac
        self.rng = np.random.default_rng(seed)
        self.data = None

    def add(self, trans):
        self.data = trans if self.data is None else tuple(np.concatenate([a, b]) for a, b in zip(self.data, trans))
        self.sac.buf.add(trans)

    def act(self, obs, step, deterministic=False):
        return self.sac.act(obs, step, deterministic)

    @torch.no_grad()
    def synth(self, n=4000, horizon=2):
        """Ramas de ``horizon`` pasos desde estados reales; el paso k se lee del rasgo de tiempo del estado."""
        S = self.data[0]
        i = self.rng.integers(0, len(S), n)
        s_raw = S[i]
        k = np.clip(np.rint(s_raw[:, 15] * 5).astype(int), 0, self.K - 1)   # rasgo de tiempo: 2020 -> 0, 2022 -> 0,2
        out = []
        s = self.norm.t(s_raw)
        alive = np.ones(n, bool)
        for _ in range(horizon):
            a, _, _ = self.sac._pi(s)
            d = torch.as_tensor((k == self.K - 1).astype(float), dtype=torch.float32)
            s2, r = self.model.step(s, a, d)
            raw_s = s.numpy() * self.norm.s + self.norm.m
            raw_s2 = s2.numpy() * self.norm.s + self.norm.m
            out.append((raw_s[alive], a.numpy()[alive], r.numpy()[alive], raw_s2[alive], d.numpy()[alive]))
            alive &= (k < self.K - 1)
            k = np.minimum(k + 1, self.K - 1)
            s = s2
        return tuple(np.concatenate([o[j] for o in out]) for j in range(5))

    def train(self, steps, batch=256):
        syn = Replay()
        syn.add(self.synth())
        rng = self.rng
        for _ in range(steps):
            nr = int(batch * self.real_frac)
            br = self.sac.buf.sample(nr, rng)
            bs = syn.sample(batch - nr, rng)
            mix = Replay()
            mix.data = tuple(np.concatenate([x, y]) for x, y in zip(br, bs))
            self.sac.train(1, batch, buf=_Fixed(mix.data))


class _Fixed:
    def __init__(self, data):
        self.data = data

    def sample(self, n, rng):
        return self.data
