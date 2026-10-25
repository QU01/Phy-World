"""Entrena un agente de RL sobre ``rl_env.Env`` y lo evalúa fuera de muestra.

    python rl_train.py init            # datos aleatorios compartidos (calentamiento off-policy y modelos)
    python rl_train.py PPO|SAC|TD3|PETS|MBPO [presupuesto de corridas reales]

Curva de aprendizaje: tras cada corrida real (48 episodios) se registra la puntuación media de entrenamiento;
cada 10, la de la política determinista en una semilla de control. Evaluación final idéntica a la del CEM
(``s3_learning``): semillas 101 y 202, R = 32, World3 de verdad sobre 400 mundos.
"""
import sys
import time

import numpy as np
import torch

import common as c
import rl_agents as ag
import rl_env

R_TRAIN = 48
TRAIN_SEEDS = tuple(range(11, 19))
CONTROL_SEED = 7
BUDGET = {"PPO": 180, "SAC": 120, "TD3": 120, "PETS": 50, "MBPO": 50}


def random_policy(rng):
    def pol(o, k):
        u = rng.uniform(0, 1, (len(o), 1))
        return rng.uniform(-0.5, 0.5, (len(o), rl_env.ACT_DIM)) * u
    return pol


def init(n=20):
    env = rl_env.Env(R_TRAIN)
    rng = np.random.default_rng(0)
    trs = []
    t = time.time()
    for i in range(n):
        tr = env.rollout(random_policy(rng), TRAIN_SEEDS[i % len(TRAIN_SEEDS)])
        trs.append({k: tr[k] for k in ("obs", "act", "obs_T", "rew", "score")})
        print(f"  aleatoria {i + 1}/{n}: puntuación media {tr['score'].mean():+.3f} ({time.time() - t:.0f}s)", flush=True)
    np.savez(c.OUT / "rl_random.npz", **{f"{k}_{i}": tr[k] for i, tr in enumerate(trs) for k in tr})


def load_random():
    z = np.load(c.OUT / "rl_random.npz")
    n = len({k.split("_")[-1] for k in z.files})
    return [{k: z[f"{k}_{i}"] for k in ("obs", "act", "obs_T", "rew", "score")} for i in range(n)]


def make(algo, norm):
    od, ad = rl_env.OBS_DIM, rl_env.ACT_DIM
    return {"PPO": ag.PPO, "SAC": ag.SAC, "TD3": ag.TD3, "PETS": ag.PETS, "MBPO": ag.MBPO}[algo](norm, od, ad)


def evaluate_true(agent, seeds=(101, 202), R=32):
    import s2_strategies as s2
    import s3_learning as s3
    env = rl_env.Env(R)
    ens = c.world3_ensemble(400)
    th = s2.theta_matrix()
    sd = s3.scales()
    reps = []
    for seed in seeds:
        tr = env.rollout(lambda o, k: agent.act(o, k, deterministic=True), seed, keep_f=True)
        f = tr["f"]
        base = s3.make_base(R, seed, ens, th)
        m, sig = c.country_metrics(f), c.world_signals(f)
        mp = c.world3_constants(sig, base["sig"])
        w3 = c.world3_metrics(mp, ens)
        w3.pop("_collapse")
        glob = s2.per_country(m, base["m"], th)
        world = s2.world_summary(f, m, sig)
        mv = s3.metric_vector(world, w3, glob, base["entry"])
        score, pil = s3.score(mv, sd)
        acts = tr["act"].mean(1)                                   # (5, 18): acción media por decisión
        reps.append({"seed": seed, "score": score, "pilares": pil, "metrics": dict(zip(s3.METRICS, mv.tolist())),
                     "world": world, "w3": w3, "w3_mapping": mp, "base_world": base["entry"]["world"],
                     "base_w3": base["entry"]["w3"], "surrogate_score_members": float(tr["score"].mean()),
                     "actions_mean": acts.tolist(), "actions_sd": tr["act"].std(1).tolist()})
        print(f"  evaluación semilla {seed}: puntuación {score:+.3f} (emulador por miembro {tr['score'].mean():+.3f})", flush=True)
    return reps


def main(algo, budget):
    t0 = time.time()
    rnd = load_random()
    obs_all = np.concatenate([np.concatenate([tr["obs"].reshape(-1, rl_env.OBS_DIM), tr["obs_T"]]) for tr in rnd])
    norm = ag.Normalizer(obs_all)
    agent = make(algo, norm)
    env = rl_env.Env(R_TRAIN)
    curve, n_eps = [], 0
    if algo != "PPO":                         # off-policy y modelos: arrancan con los datos aleatorios
        trans = ag.flatten(rnd)
        agent.add(trans) if hasattr(agent, "add") else agent.buf.add(trans)
        n_eps = len(rnd) * R_TRAIN
        if algo in ("PETS", "MBPO"):
            print("  modelo:", agent.fit() if algo == "PETS" else agent.model.fit(agent.data), flush=True)
        if algo in ("SAC", "TD3"):
            agent.train(2000)
        if algo == "MBPO":
            agent.train(2000)
    rng = np.random.default_rng(1)
    batch = []
    for it in range(budget):
        raw = [None] * len(rl_env.DECISIONS)

        def pol(o, k):
            a = agent.act(o, k)
            raw[k] = a
            return a
        tr = env.rollout(pol, TRAIN_SEEDS[it % len(TRAIN_SEEDS)])
        tr["act_raw"] = np.stack(raw)
        n_eps += R_TRAIN
        info = {"iter": it, "episodes": n_eps, "train_score": float(tr["score"].mean()), "t": time.time() - t0}
        if algo == "PPO":
            batch.append(tr)
            if len(batch) == 2:
                agent.update(batch)
                batch = []
        else:
            trans = ag.flatten([tr])
            agent.add(trans) if hasattr(agent, "add") else agent.buf.add(trans)
            if algo in ("SAC", "TD3"):
                agent.train(150)
            elif algo == "PETS":
                if it % 2 == 1:
                    info["model"] = agent.fit()
            elif algo == "MBPO":
                info["model"] = agent.model.fit(agent.data, epochs=40)
                agent.train(400)
        if it % 10 == 9 or it == budget - 1:
            g = env.rollout(lambda o, k: agent.act(o, k, deterministic=True), CONTROL_SEED)
            info["greedy_score"] = float(g["score"].mean())
        curve.append(info)
        print(f"  {algo} {it + 1}/{budget}: episodios {n_eps}, entrenamiento {info['train_score']:+.3f}"
              + (f", determinista {info['greedy_score']:+.3f}" if "greedy_score" in info else "")
              + f" ({info['t'] / 60:.1f} min)", flush=True)
    reps = evaluate_true(agent)
    c.dump(f"rl_{algo}.json", {"algo": algo, "family": {"PPO": "on-policy", "SAC": "off-policy", "TD3": "off-policy",
                                                         "PETS": "basado en modelo (MPC)", "MBPO": "basado en modelo (Dyna)"}[algo],
                                "curve": curve, "eval": reps, "budget": budget, "minutes": (time.time() - t0) / 60})
    if hasattr(agent, "pi"):
        torch.save({"pi": agent.pi.state_dict(), "logstd": agent.logstd}, c.OUT / f"rl_{algo}.pt")
    elif hasattr(agent, "actor"):
        torch.save(agent.actor.state_dict(), c.OUT / f"rl_{algo}.pt")


if __name__ == "__main__":
    if sys.argv[1] == "init":
        init()
    else:
        a = sys.argv[1]
        main(a, int(sys.argv[2]) if len(sys.argv) > 2 else BUDGET[a])
