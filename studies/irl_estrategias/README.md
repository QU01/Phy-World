# Política revelada (IRL) y estrategias: Atlas Mundo (L1) × World3 (L0)

Informe técnico: [`docs/Informe_IRL_estrategias_AtlasMundo_World3.pdf`](../../docs/Informe_IRL_estrategias_AtlasMundo_World3.pdf).
Informe divulgativo (`s5_divulgativo.py`): [`docs/Informe_divulgativo_estrategias_mundo.pdf`](../../docs/Informe_divulgativo_estrategias_mundo.pdf).

| Paso | Script | Salida |
|---|---|---|
| 1. IRL (Bajari-Benkard-Levin) con 6 orígenes, θ global, por grupo y por 20 países | `s1_irl.py` | `out/irl.json` |
| 2. 12 estrategias en el L1 2019-2030 (global y cuasi-unilateral) y World3-03 2020-2100 | `s2_strategies.py` | `out/strategies.json` |
| 2b. Atribución y sensibilidad del acoplamiento L1 → L0 | `s2b_sensitivity.py` | `out/sensitivity.json` |
| 3. Aprendizaje CEM: planificador por zonas y mejor respuesta de 20 países | `s3_learning.py` | `out/learning.json` |
| 5a. Emulador de World3 (600 corridas × 400 mundos; R² 0,99-1,00) | `w3_surrogate.py` | `out/w3_design.json` |
| 5b. Aprendizaje por refuerzo en lazo cerrado: PPO, SAC, TD3, PETS-MPC, MBPO-Dyna | `rl_env.py`, `rl_agents.py`, `rl_train.py` | `out/rl_*.json`, `out/rl_*.pt` |
| 4. Figuras e informe PDF | `s4_figures.py`, `s4_report.py` | `out/fig/`, el PDF |

```bash
python world4_cli.py --install-atlas --install-wpp --install-energy
cd studies/irl_estrategias
python s1_irl.py && python s2_strategies.py && python s2b_sensitivity.py && python s3_learning.py
python w3_surrogate.py && python rl_train.py init
for a in PPO SAC TD3 PETS MBPO; do python rl_train.py $a; done   # ~3,5 GB de RAM cada uno
python s4_report.py
```

Necesita además `matplotlib`, `reportlab` y `torch`. Clima: TCRE del Atlas (F4/F5 apagados; sus datos no se pudieron descargar
en el entorno donde se corrió). El acoplamiento L1 → L0 es un supuesto declarado (`common.world3_constants`).
