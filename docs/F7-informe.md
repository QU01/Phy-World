# F7 de Phy-World (World4): calibración bayesiana e incertidumbre

- **Fecha:** 2026-10-13
- **Estado:** construida y medida.
  - **Métodos** (`world4/calib/`): emulador GP propio en NumPy, *history matching* por oleadas, ABC-SMC con el error del modelo como núcleo final, discrepancia AR1 con `phycore.uq.DiscrepancyGP`, `models/posterior.npz`.
  - **Calibrados:** el L0 regional (datos 1992-2019) y el L1 (pseudo-hindcasts antes del origen; orígenes 1990 y 2019).
  - **Campaña:**
    - **W4: ROJO.** La prueba limpia (L1 calibrado solo con datos ≤ 1990) deja dentro de banda PIB, cereales y GMST, pero no población, energía, CO₂ ni tierras.
    - **W8: ROJO.** La cobertura entra en 78-82 % en las 10 celdas y la distancia KS baja de 0,06-0,20 a 0,03-0,11, pero solo 4 de 10 celdas llegan a D ≤ 0,05.
- **Pruebas:** 124/124; 10 son nuevas.
- **Reproducir:**
  - `python world4_cli.py --calibrate` (unos 40 min con 8 procesos; escribe `models/posterior.npz`, `runs/f7_*.json` y `runs/w8.json`);
  - `--campaign`, para puntuarlos.

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| Emulador GP | `calib/gp.py` | Matérn 5/2 con ARD, media lineal integrada (krigeado universal), REML con gradiente analítico, LOO cerrado (Dubrule 1983) |
| *History matching* | `calib/history_matching.py` | Implausibilidad con la 2.ª mayor > 3 (Vernon et al. 2010); siembra por LHS y perturbaciones en la NROY; regla de parada; regla de la caja |
| ABC-SMC | `calib/abc.py` | Núcleo gaussiano con ε adaptativo hasta ε = 1: el núcleo final ES el modelo de error declarado, así que el resultado es el posterior exacto bajo ese error (Wilkinson 2013) |
| Discrepancia | `calib/discrepancy.py` | `realidad = ρ·modelo + δ(h)`; la previa de ρ sale de la marca ([V] 0,05, [S] 0,15, [M] 0,30); con [?] se niega a ajustar |
| L0 | `calib/l0r.py` | 11 perillas identificables y 6 que no lo son (viajan con su previa) |
| L1 | `calib/l1.py`, `core/coupling.py` | 6 perillas: desplazamiento por miembro del intercepto de las ecuaciones sectoriales. Con todas a 0 el resultado es idéntico bit a bit (prueba) |
| W8 | `structural/recal.py`, `questions.py` | Recalibrador de sesgo reciente (`ZCalibrator`) y selección anidada entre cuatro candidatos |
| Producto | `calib/posterior.py`, `api.run_l0r(posterior=True)`, `--calibrate` | El posterior viaja en el repo (99 KB) y no lleva datos de terceros |

**Comprobación de los métodos** con un problema de juguete de solución conocida:
- el ABC-SMC reproduce el posterior exacto de una rejilla: media a menos de 0,3 sd y sd a menos del 20 % (en la práctica, 0,346/−0,435 frente a 0,349/−0,431);
- el *history matching* conserva el valor verdadero.

**Velocidad:** el GP con gradiente analítico ajusta 150 corridas en 11 dimensiones en 0,17 s; la primera versión, con gradiente numérico, no terminaba en 120 s.

## 2. Perillas, previas y varianzas (declaradas antes de medir)

- **Perillas del L0** (sobre las ecuaciones de MCO de F6):
  - multiplicadores: ahorro, natalidad, mortalidad, convergencia de la PTF y TCRE;
  - desplazamientos por año: PTF, natalidad, intensidad energética, cuota no fósil, tierras y rendimiento.
- **No identificables con 1992-2019** (quedan en su valor de F6, o se sacan de su previa con `explore=True`): recurso último, daño, EROI, sustitución, déficit y sensibilidad del rendimiento al calor.
- **Perillas del L1:** fecundidad, esperanza de vida, intensidad energética, cuotas no fósiles, rendimiento y tierras de cultivo.
- **Error del dato** [S]: población 1 %, PIB 3 %, energía 3 %, CO₂ 5 %, cereales 4 %, tierras 2 %, GMST 0,10 °C. En cereales y GMST incluye la variabilidad interanual, que el modelo no tiene.
- **Discrepancia** [S]: la mitad de las bandas de P2, que se declararon en F6 antes de medir P2, a 27 años, creciendo con la raíz del horizonte. Así la banda de P2 equivale a unas 2σ del error estructural.
- **Regla de la caja,** declarada antes de ver ningún *hindcast*: si el p10 o el p90 de una perilla quedan a menos del 5 % del ancho de un borde, la caja se dobla y se recalibra, hasta 2 veces. Un borde inferior en 0 es físico y no cuenta.

## 3. El L0 calibrado

### Desarrollo: datos ≤ 2005, banda 2006-2019

| | Dentro de p10-p90 (posterior + discrepancia + dato) | Solo paramétrica | Error de la mediana en 2019 | MCO de F6 |
|---|---|---|---|---|
| Población | 100 % | 29 % | −1,6 % | −0,7 % |
| PIB | 100 % | 71 % | +1,7 % | −4,7 % |
| Energía primaria | 100 % | 14 % | +8,7 % | +10,0 % |
| Cereales | 100 % | 29 % | −7,3 % | −6,4 % |
| Tierras de cultivo | 100 % | 21 % | −4,0 % | −3,1 % |
| GMST | 64 % | 21 % | +0,10 °C | +0,07 °C |

**Esta comprobación se usó para fijar el método, así que no es una prueba.**

**Cambio de método tras medir (declarado).** Al principio la banda sumaba la discrepancia AR1 ajustada con los residuos de la ventana, y salió del 0-43 %:
- tras calibrar, esos residuos ya están absorbidos y la discrepancia sale casi nula (Brynjarsdóttir y O'Hagan 2014);
- 1992 es un año frío por el Pinatubo, y eso inflaba el ρ de la GMST;
- la banda no incluía el error del dato.

**Método congelado desde aquí:** la predicción usa la MISMA discrepancia declarada que la verosimilitud del *history matching* y del ABC, más el error del dato. La AR1 queda como diagnóstico del sesgo de escala (ρ).

**Lectura:**
- Un 100 % sobre una banda del 80 % dice que la discrepancia declarada es **conservadora**.
- La mediana calibrada no mejora a la de MCO: **el aporte es la banda, no el centro.**

### Producción: datos 1992-2019 (`models/posterior.npz`)

- 150 corridas por oleada, 54 resúmenes. La NROY se queda en el 0,3 % de la caja; el ABC necesitó 5 336 simulaciones; el 98 % de las partículas son no implausibles con el simulador (comprobación del emulador).
- **Identificados** (anchura p10-p90 frente a la previa):

| Perilla | Mediana [p10, p90] | Anchura |
|---|---|---|
| PTF, desplazamiento por año | −0,0090 [−0,0106, −0,0073] | 10 % |
| Intensidad energética, por año | −0,0010 [−0,0022, +0,0006] | 18 % |
| Ahorro × | 1,32 [1,21, 1,38] | 26 % |
| Mortalidad × | 1,20 [1,12, 1,25] | 28 % |

- **Sin identificar:** la cuota no fósil (68 %) y la TCRE (66 %).
- **No son parámetros físicos creíbles.** Ahorro ×1,32, mortalidad ×1,20 y −0,9 %/año de PTF son valores **efectivos**: compensan estructura que le falta al L0. Sin cohortes, la mortalidad bruta tiene que subir para que la población regional cuadre; es el mismo fallo de P2.
- **Ahorro, convergencia y PTF forman una cresta** (se compensan entre sí). Tras las dos ampliaciones, `s` y `lam_A` siguen en el borde: **para esas dos perillas la caja todavía decide.**

## 4. El L1 calibrado y W4

Para el origen O: pseudo-hindcasts desde O-15, O-10 y O-5 hasta O, cada uno con TODO estimado con datos ≤ su propio origen. Cada corrida de diseño tiene 3 ventanas × 8 miembros. Una oleada son 48 corridas, unos 2 min en 8 procesos; el ABC corre sobre el emulador.

**Origen 1990, la prueba limpia,** con el método ya congelado.
- La regla de la caja dobló intensidad y rendimiento una vez.
- **Posterior:**
  - fecundidad +0,010 [+0,004, +0,016] por año (la fecundidad bajó MENOS en 1975-1990 de lo que decían sus ecuaciones);
  - intensidad −0,021 [−0,026, −0,016] por año.

Hindcast 1990→2023, años dentro de p10-p90:

| | Sin calibrar | Posterior | **Posterior + discrepancia + dato** | Con la AR1 (informativo) | Error de la mediana al final |
|---|---|---|---|---|---|
| Población | 24 % | 0 % | **27 %** | 0 % | +1,6 % → +15,9 % |
| PIB (hasta 2019) | 90 % | 97 % | **100 %** | 93 % | −8,2 % → −3,5 % |
| Energía primaria | 100 % | 24 % | **30 %** | 42 % | +27 % → −68 % (log) |
| CO₂ fósil | 91 % | 18 % | **30 %** | 39 % | +38 % → −66 % (log) |
| Cereales | 58 % | 97 % | **100 %** | 100 % | +41 % → +28 % (log) |
| Tierras de cultivo | 3 % | 3 % | **30 %** | 42 % | +14 % → +20 % (log) |
| GMST | 82 % | 67 % | **91 %** | 73 % | −0,43 → −0,09 °C |

(Las cuatro primeras columnas usan el mismo L1 F1-F5 para todas las variables. Por eso «sin calibrar» difiere algo de los hindcasts de F1-F5, que usaban la configuración mínima de cada sector.)

**Hallazgo: el sesgo no es estacionario.**
- Las ventanas 1975-1990 aprenden la fuerte caída de la intensidad energética de después de los choques del petróleo y una fecundidad que bajaba despacio. Ninguna de las dos cosas siguió así después de 1990.
- Calibrar con ellas **empeora** energía, CO₂ y población, y mejora cereales, PIB y GMST.
- La discrepancia AR1 lo ve: ρ de PIB 0,30, energía 0,56 y CO₂ 0,41 en 1975-1990, «CALIBRAR: ρ + discrepancia». Pero corregir con ese ρ tampoco basta (39-42 %).
- **Lo que esto dice:** con 15 años de pseudo-historia no se puede separar un sesgo persistente de uno de época. Es el riesgo N = 1 de la hoja de ruta, medido.

**W4: ROJO** en población, energía, CO₂ y tierras.

**Origen 2019, el posterior vivo,** en `models/posterior.npz`: la historia 2004-2019 apenas restringe (NROY 90 % de la caja). Solo la intensidad se mueve: −0,005 [−0,010, −0,0005] por año.

## 5. W8: recalibración por selección anidada

**Candidatos** (los dos remedios que dejó F0.5):
- todos los PIT (el recalibrador de F0.5);
- ventana de 5 años;
- ventana de 5 años con todos los horizontes juntos;
- sesgo reciente: el residuo estandarizado se corrige con la mediana de los últimos 3 años.

En cada fecha se ELIGE el candidato con menor distancia KS sobre los desenlaces de los 8 años ya publicados, transformado cada uno con el recalibrador que existía en SU fecha.

| | Cobertura 80 % | KS D (F0.5 → F7) | D de referencia: calibrado perfecto con correlación en el año (p95) |
|---|---|---|---|
| Crecimiento PIB pc, k2 = 4 / 5 / 6 / 7 / 8 | 0,82 / 0,80 / 0,80 / 0,80 / 0,78 | 0,127 → **0,033** / 0,152 → **0,041** / 0,152 → 0,065 / 0,177 → 0,089 / 0,197 → 0,108 | 0,065 / 0,081 / 0,096 / 0,101 / 0,095 |
| Deuda / PIB, k2 = 4 / 5 / 6 / 7 / 8 | 0,81 / 0,80 / 0,79 / 0,79 / 0,79 | 0,064 → 0,093 / 0,083 → 0,087 / 0,107 → 0,058 / 0,132 → **0,043** / 0,164 → **0,045** | 0,059 / 0,064 / 0,078 / 0,083 / 0,096 |

- **La cobertura cumple en las 10 celdas. La forma, en 4.** En la deuda a 4-5 años F7 empeora a F0.5: la selección cambia de candidato y paga ese cambio.
- **Exploración, declarada:** los candidatos se compararon primero mirando el periodo de prueba. Esa tabla está en `runs/w8.json` (`candidates_on_test_period_ks_d`) y no decide nada. El veredicto usa solo la selección anidada.
- **Hallazgo de método:** los PIT de un mismo año están correlacionados entre países (choques comunes). Un pronosticador perfecto con esa correlación daría D de 0,06-0,10 (p95), así que el listón D ≤ 0,05 es más estricto de lo que parece con n ≈ 5 000.
  - Se informa, pero **el listón no se cambia.**
  - Incluso contra esa referencia, 3 celdas quedan fuera: crecimiento a k2 = 8 y deuda a k2 = 4-5.
- **W8: ROJO.**
- **El estimador estructural ya usa el recalibrador F7** (`describe()` dice qué candidato eligió). MODEL_REVISION pasa a 7.

## 6. Desviaciones y lo que no está

- **sbi no se usó:** necesita torch, que el producto no lleva. ABC-SMC en NumPy, como permitía la hoja de ruta.
- **JAX (`world4[grad]`) no se añadió:** no hace falta. El L0 cuesta 0,09 ms por miembro en lote, el ABC necesitó unas 5 000 simulaciones y Studio prohíbe jax.
- **Previa de Nebel et al. (2023):** no aplica al L0 regional, que no tiene ningún parámetro de World3. Calibrar el World3-03 de F0 con esa previa queda pendiente, y la tabla del artículo sigue sin abrirse.
- **El posterior del L1 no entra todavía en `api.query`:** cambiaría W3, W5 y W9a. Entra en F11 con el estimador completo. El de origen 1990 empeora 4 de 7 agregados, así que no debe entrar a ciegas.
- **`models/posterior.npz` está en la raíz del repo,** como pide la hoja de ruta, y no dentro del paquete: una instalación no editable no lo lleva. Es para F9.
- **Una sola oleada en cada calibración:** la varianza del emulador ya era menos del 3 % del resto (regla de parada). Con 11 perillas y discrepancias anchas, el emulador no es lo que limita.

## 7. Qué sigue

1. **Cohortes en el L0.** Explican a la vez el fallo de P2 y la mortalidad ×1,2 del posterior.
2. **Sesgo no estacionario en el L1.** Más ventanas y más largas (desde 1965 donde haya dato), o una discrepancia con deriva por época en lugar de un desplazamiento fijo.
3. **W8:** candidatos con menos rotación (selección con histéresis) y criterio de forma consciente de la correlación entre países, si el usuario decide cambiar el listón.
4. **Arrastrados:**
   - la fecundidad de F1;
   - la coherencia F2/F3;
   - el presupuesto de carbono de F4;
   - los frenos del sistema alimentario de F5;
   - LTM frente a PPM/LEEA (F6).
