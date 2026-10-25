# Política revelada (IRL) y estrategias: Atlas Mundo (L1) × World3 (L0)

Informe: [`docs/Informe_IRL_estrategias_AtlasMundo_World3.pdf`](../../docs/Informe_IRL_estrategias_AtlasMundo_World3.pdf).

| Paso | Script | Salida |
|---|---|---|
| 1. IRL (Bajari-Benkard-Levin) con 6 orígenes, θ global, por grupo y por 20 países | `s1_irl.py` | `out/irl.json` |
| 2. 12 estrategias en el L1 2019-2030 (global y cuasi-unilateral) y World3-03 2020-2100 | `s2_strategies.py` | `out/strategies.json` |
| 2b. Atribución y sensibilidad del acoplamiento L1 → L0 | `s2b_sensitivity.py` | `out/sensitivity.json` |
| 3. Aprendizaje CEM: planificador por zonas y mejor respuesta de 20 países | `s3_learning.py` | `out/learning.json` |
| 4. Figuras e informe PDF | `s4_figures.py`, `s4_report.py` | `out/fig/`, el PDF |

```bash
python world4_cli.py --install-atlas --install-wpp --install-energy
cd studies/irl_estrategias
python s1_irl.py && python s2_strategies.py && python s2b_sensitivity.py && python s3_learning.py && python s4_report.py
```

Necesita además `matplotlib` y `reportlab`. Clima: TCRE del Atlas (F4/F5 apagados; sus datos no se pudieron descargar
en el entorno donde se corrió). El acoplamiento L1 → L0 es un supuesto declarado (`common.world3_constants`).
