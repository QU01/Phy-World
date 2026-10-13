# F8 de Phy-World (World4): escenarios y políticas robustas

- **Fecha:** 2026-10-13
- **Estado:** construida y medida.
  - **Palancas θ:** 7, con caja, unidades y signo declarado [M], sobre el L0 regional.
  - **Narrativas:** cinco tipo SSP, como conjuntos de parámetros.
  - **Búsqueda robusta:** `phycore.Designer` (NSGA-II a fidelidad 0) sobre un ensemble de estados del mundo, con PRIM y arrepentimiento.
  - **Conducta alternativa en el L1:** «lo que harían» frente a «lo que les convendría», desde el problema inverso del Atlas.
  - **CLI:** `--theta`, `--levers`, `--design` y `--narrative`.
- **Campaña:** W6 sigue VERDE con las palancas de F8 (7/7 monótonas y el mutante detectado). El resto no cambia.
- **Pruebas:** 133/133; 9 son nuevas.
- **Reproducir:**
  - `python world4_cli.py --design` (unos 10 min);
  - `python world4_cli.py --theta "carbon_price=150,efficiency=0.01"` (unos segundos).

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| Palancas | `scenarios/levers.py`, `core/l0r/model.py` (`POLICY`, `EPS_*`) | Con θ = 0 el L0 es idéntico bit a bit y sigue en 9,6 ms (W10) |
| Narrativas | `scenarios/narratives.py` | SSP1-5 de O'Neill et al. (2017) como desplazamientos de las perillas del posterior de F7 [M] |
| Ensemble | `narratives.sow` | Estados del mundo: posterior de F7 + no identificadas en su caja (incertidumbre profunda) + eficacia de cada grupo de palancas U(0,5; 1,5) |
| Búsqueda | `scenarios/robust.py` | `World4Plugin` cumple `phycore.plugin.validate_plugin`; frente VERIFICADO; 3 representantes |
| PRIM | `robust.prim` | Propio (Friedman y Fisher 1999); prueba con una caja plantada |
| Arrepentimiento | `robust.regret` | Savage, por estado del mundo, sobre representantes y no hacer nada |
| Conducta | `scenarios/conduct.py`, `coupling.forecast(offsets=, raw=)` | Con `offsets` vacío el L1 es idéntico (prueba) |
| Validación | `validate/f8.py`, W6 | Monotonía de cada palanca en el mundo mediano; mutante: el efecto de las renovables invertido |
| API | `api.evaluate_policy`, `api.design` | Exploratorio en todo resultado |

## 2. Palancas (todas se pagan) y sus efectos [M]

| Palanca | Caja | Efecto (por año, pleno tras 10 años de rampa) | Coste |
|---|---|---|---|
| Precio del carbono | 0-200 USD/tCO₂ | Cuota no fósil +0,05·log(1+τ/100) en logit; intensidad −0,01·log(1+τ/100) sobre la parte fósil | PIB −0,3 × recaudación |
| Renovables | 0-2 % del PIB | Cuota no fósil +0,04 en logit por cada 1 % | Sale de la inversión |
| Eficiencia | 0-2 % del PIB | Intensidad −0,5 % por cada 1 % | Inversión |
| I+D agrícola | 0-0,5 % del PIB | Rendimiento +0,3 % por cada 0,1 % | Inversión |
| Ayuda | 0-1 % del PIB donante | NAM, EUR y OPA → SSA y SAS como capital; llega la mitad | Inversión de los donantes |
| Fecundidad | 0-1 % del PIB en SSA, SAS y MEA | Natalidad bruta −1 % por cada 1 % | Inversión |
| Consolidación | 0-1 del déficit | Recorta el déficit | Inversión: 0,5 por unidad recortada |

- **Los tamaños de efecto no están estimados:** la historia 1992-2019 no tiene un precio mundial del carbono ni programas de esta escala. Su incertidumbre entra en el ensemble como factor de eficacia, y por eso PRIM puede decir «falla si la palanca rinde poco».
- **El consumo es (1 − s)·Y:** las palancas se pagan con inversión, así que su coste aparece como menos capital después.

**Validación en el mundo mediano** (valores 0 / ½ / máximo):
- cuota no fósil de 2050 0,154 → 0,222 → 0,271 (carbono) y → 0,234 → 0,337 (renovables);
- energía de 2050 −12 % → −23 %;
- población de SSA en 2100 de 4 313 a 2 727 millones;
- deuda mundial de 2100 de 1,14 a 0,17.
- Las 7 son monótonas y el mutante se detecta.

## 3. Búsqueda robusta

- **Objetivos,** todos robustos por percentil sobre el ensemble:
  - bienestar p10: media descontada al 2 % del log del consumo per cápita, ponderada por población;
  - GMST máxima 2020-2100, p90;
  - consumo de la región más pobre en 2100, p10.
- **Límites** (MORDM) [M], declarados antes de buscar:
  - alimento: ninguna región baja del 90 % de su cereal per cápita de 2019 (p10);
  - tierra: las tierras de cultivo de 2100 no pasan de 1,10 veces las de 2019 (p90);
  - deuda: la deuda/PIB mundial de 2100 no pasa de 1,5 (p90).
- **Ensemble:** 100 estados del mundo por evaluación (26 ms); unas 2 500 evaluaciones por búsqueda.

**Búsqueda declarada (SSP2): el frente degenera en un punto.**
- Ninguna política cumple el límite de tierra: el mejor p90 alcanzable es 1,146, frente al 1,10 declarado.
- En el L0 las tierras de cultivo solo responden a la población, y la única palanca que la mueve es la fecundidad. No hay ahorro de tierra por rendimiento, porque eso no se identificó en la historia (F5/F6).
- La búsqueda devuelve la política de mínima violación (carbono 65 USD/t, fecundidad 1 %, I+D agrícola 0,27 %).

**Variante declarada:** la tierra se informa en vez de restringir. Frentes de 46 a 75 políticas, todos factibles en alimento y deuda:

| Narrativa | Sin política: bienestar p10 / GMST máx. p90 | Frente: GMST máx. p90 | Arrepentimiento p90 de no hacer nada (GMST) | Mejor tierra alcanzable (g) |
|---|---|---|---|---|
| SSP1 | 9,627 / 5,60 °C | 3,15-3,88 | 2,86 °C | +0,002 |
| SSP2 | 9,457 / 6,05 °C | 3,40-4,23 | 3,02 °C | +0,046 |
| SSP3 | 9,089 / 5,40 °C | 3,17-3,83 | 2,54 °C | +0,102 |
| SSP4 | 9,166 / 5,20 °C | 3,03-3,81 | 2,51 °C | +0,069 |
| SSP5 | 9,700 / 7,69 °C | 3,98-4,74 | 4,36 °C | +0,011 |

**Lectura:**
- **Las palancas acaban en el borde de su caja:** carbono en 200 USD/t, fecundidad en 1 % y ayuda cerca de 1 % en todo el frente. La caja es el límite político declarado y es lo que manda: con más margen, la búsqueda pediría más.
- **El compromiso real** está entre renovables y eficiencia (entre 0,1 y 2 % cada una) y en la consolidación, que sube con la deuda de la narrativa: 0,44 en SSP3 frente a 0,01 en SSP1.
- **El bienestar p10 mejora** frente a no hacer nada en TODO el frente: en SSP2, de 9,457 a 9,47-9,49 en log, un 1,5-3,5 % de consumo equivalente.
  - Por mundos no es gratis: la política más climática es peor que no hacer nada en el 39-60 % de los mundos.
  - Arrepentimiento p90 en bienestar: 0,035 para la más climática, frente a 0,05 de no hacer nada.
- **GMST de 2100:** apenas depende del carbono en la línea base, porque el L0 agota el fósil antes de 2100 (F6). Las políticas bajan el MÁXIMO cuando mueven la difusión limpia, la eficiencia o la población; el precio solo, no.

**Vulnerabilidades (PRIM, 1 000 mundos, SSP2):**

| Política | Pasa de 3 °C | Dónde falla (caja PRIM) | Peor bienestar que no hacer nada | Dónde |
|---|---|---|---|---|
| Mejor bienestar | 93 % | TCRE alta, difusión limpia baja | 7 % | Daño bajo, eficacia social baja |
| Mejor clima | 32 % | Eficacia energética < 1, eficacia del carbono baja, mortalidad alta | 50 % | Daño bajo, convergencia alta |
| Compromiso | 48 % | Eficacia energética baja, natalidad alta | 33 % | Daño bajo, PTF baja |

**Lo que dicen las vulnerabilidades:**
- El riesgo de la política climática es casi siempre el mismo: **un daño climático bajo hace que no compense.**
- La política del bienestar falla en clima cuando la **TCRE** es alta.
- Nada de esto es identificable con la historia 1992-2019; por eso estaba en el ensemble como incertidumbre profunda.

## 4. Conducta: lo que harían frente a lo que les convendría (L1, 2019→2030)

**El método.** El problema inverso del Atlas (BBL, `irl.json`) midió qué desvíos de ±0,5 en el logit de la regla de cada canal habrían subido el bienestar de un país, con los demás siguiendo la regla:
- rentables en más de la mitad de los casos: más tecnología propia (r, 92 %), más extracción propia (x, 87 %) y más redistribución (w, 61 %);
- también rentable: menos importación del centro (m, 68 %) y menos inversión en recursos ajenos (f, 56 %).

«Les convendría» = la regla con esos cinco desvíos, todos los países a la vez, con números aleatorios comunes y el L1 F1-F5, 32 miembros:

| Advised − would, 2030 | |
|---|---|
| Consumo per cápita | −0,2 % |
| PIB per cápita | −0,1 % |
| Riesgo de conflicto | +0,0000 |
| Valor en bienestar (recompensa de bienestar del IRL) | +0,0017; mejor en el 66 % de los miembros |
| Valor con la recompensa REVELADA | +0,045; mejor en el 100 % de los miembros |

**Hallazgo: los desvíos «rentables» del Atlas son de equilibrio parcial.**
- Cuando todos los países se desvían a la vez, el bienestar apenas sube y el consumo baja.
- Lo que sube con claridad es la recompensa revelada. Esa recompensa da peso al poder relativo y a la inercia, y en ella el bienestar pesa en negativo.
- En este modelo, «lo que les convendría» a cada país por separado no es lo que le convendría al conjunto.
- El PPO anclado del Atlas (`recommend.py`) necesita `autograd` y horas de cómputo: queda como extra opcional `world4[rl]` y no se entrenó.

## 5. Desviaciones y lo que no está

- **EMA Workbench:** licencia BSD-3, VERIFICADA en su LICENSE el 2026-10-13. Va como extra `world4[dmdu]` (`ema_workbench==2.5.3`) y no se instaló: podría mover las versiones fijadas de las que depende W2. PRIM es propio y no se ha comparado todavía con el de EMA.
- **Búsqueda en el L0, no en el L1:** el L1 cuesta segundos por miembro. La búsqueda sobre el L1 (fidelidad 1, con el sustituto de residuo de `phycore`) sigue pendiente; la conducta sí usa el L1.
- **Restricciones relativas, no las de Richardson et al.:** son no comerciales (F0) y el L0 no tiene bosque ni agua como stock.
- **La tierra queda fuera del alcance de las palancas.** Haría falta una palanca de ahorro de tierra o un acoplamiento rendimiento → tierras que la historia no identificó.
- **Sin palancas de desigualdad interna ni de legitimidad:** el L0 no tiene esos stocks (F6).
- **El efecto de las palancas es [M]:** su incertidumbre está en el ensemble, pero no su forma.

## 6. Qué sigue

1. **F9:** asistente de la CLI, WALLS/TIGHT/LAST BLOCKER completos (ya hay un `LAST BLOCKER` en `--theta`), informe HTML con el frente y las cajas de PRIM, y `models/` dentro del paquete.
2. **La búsqueda en el L1** con el sustituto de residuo (fidelidad 1), y la comparación de PRIM con EMA Workbench.
3. **Una palanca de ahorro de tierra** y estructura por cohortes en el L0 (F7).
