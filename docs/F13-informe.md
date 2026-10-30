# F13 de Phy-World (World4): qué predice los eventos de violencia y un sistema de regresiones de densidad con el Atlas

- **Fecha:** 2026-10-30
- **Pregunta 1:** ¿qué factores del Atlas y de World3 predicen mejor el número de eventos de violencia organizada de un país?
- **Pregunta 2:** ¿se puede predecir, desde cualquier año, la densidad de eventos (eventos por 10 000 km²) de cada país usando el Atlas?
- **Respuesta corta:**
  - La **historia del conflicto** es la base.
  - Sobre ella, el Atlas mejora la predicción un **19-26 % a 1-3 años**, también en 2020-2025, que nunca se usó para elegir nada.
  - Lo que más aporta es la **capacidad del Estado y la exclusión política** (para el orden de magnitud) y la **escala del país**, es decir, superficie y población (para el volumen).
  - **World3** ayuda sólo a través de sus análogos por país (ingreso, rendimientos, rentas de recursos, déficit ecológico), y nada que no den ya las instituciones y la geografía. Su trayectoria mundial **empeora** siempre la predicción.
- **Código:**
  - `world4/conflict/country_panel.py`: el panel país-año.
  - `factors.py`: el estudio.
  - `density.py`: el sistema.
  - `reference/priogrid_isonum.npz` y `scripts/make_cell_iso.mjs`: el país de cada celda.
- **Resultados:**
  - `reports/factores_eventos/resultados.json`: lente de volumen, PPML.
  - `resultados_nb.json`: lente de orden de magnitud, binomial negativa.
  - `backtest.json`: validación del sistema.
- **Uso:**
  - `api.event_density(origin, horizon, cells=True)` y `api.event_density_scenario(origin)`;
  - `world4_cli.py --event-density [JSON] --origin 2025 --horizon 1` y `--study-factors`;
  - la vista "Densidad de eventos" de la interfaz (`--serve`).
- **Pruebas:** `tests/test_f13_density.py` (4).

## 1. Datos

**Resultado.** Eventos de UCDP GED 26.1 (todos los tipos) por país y año, de 1989 a 2025. El país de un evento es el del lugar donde ocurrió: el de su celda PRIO-GRID según la geometría de Natural Earth 1:50m. No es el código Gleditsch-Ward de UCDP, que no coincide con los ISO3 del Atlas. Un evento sin celda fiable usa el país dominante de su código. Los totales por año coinciden exactamente con GED (2 659 en 1989, 25 962 en 2014, 25 770 en 2025).

**Exposición.** Superficie terrestre del país (suma de las celdas). La densidad son eventos por 10 000 km².

**Países.**
- 167 del Atlas con celdas.
- 10 que el Atlas no tiene pero tienen eventos: Afganistán (41 490 eventos), Somalia, Sudán del Sur, Libia, Eritrea, Papúa Nueva Guinea, Timor Oriental, Guyana, Islas Salomón y Sáhara Occidental. Entran en el sistema con su historia y sin variables del Atlas; el estudio usa sólo los del Atlas.

**Predictores (91, en 10 grupos, información del año t).**

| Grupo | Ejemplos |
|---|---|
| Historia (12, la línea base) | eventos en t, t-1 y media de 5 años; densidad; muertes; años desde el último; parte de celdas con eventos; por tipo; vecinos (contigüidad de celdas); conflicto civil UCDP/PRIO |
| Economía (9) | ln PIB per cápita, crecimiento, inflación, deuda, crisis, rentas de recursos y de petróleo, comercio |
| Demografía (10) | población, crecimiento, jóvenes, urbanización, fecundidad, esperanza de vida y su cambio, migración |
| Instituciones (13) | democracia (V-Dem) y régimen intermedio, corrupción, estado de derecho, capacidad fiscal y estatal, exclusión (EPR), golpes, cambio de régimen, edad del Estado |
| Desigualdad (2) | Gini, 10 % más rico |
| Sociedad y geografía (9) | superficie, rugosidad, montaña, fraccionalización, cohesión, excolonia, vecinos |
| Geopolítica (10) | disputas militarizadas, rivalidades, gran potencia, ayuda, sanciones, gasto militar, intervención extranjera, voto en la ONU, poder material |
| Clima y ambiente (10) | anomalía de temperatura, desastres, estrés hídrico, rendimientos y su choque, PM2,5, déficit ecológico, vulnerabilidad, CO₂, bosque |
| Análogos de World3 por país (9) | los de `reports/acople/investigacion.py`: ln PIB pc (iopc), choque de rendimiento (fpc), rentas (nrfr), CO₂ y PM2,5 (ppolx), cambio de esperanza de vida (le), déficit ecológico (hef), población |
| World3 mundial (15) | la corrida estándar de World3 (iopc, fpc, nrfr, ppolx, le, hwi y sus crecimientos), interacciones con la exposición del país y precios reales mundiales observados de petróleo y cereales |

## 2. Método del estudio

- **Fuera de muestra:** tres bloques de años objetivo (2005-09, 2010-14, 2015-19). Cada uno se predice con un modelo entrenado sólo con objetivos anteriores. Horizontes de 1, 3 y 5 años. Datos faltantes: último valor del país, luego la mediana del entrenamiento, con indicador de faltante.
- **Métrica principal:** el RPS sobre órdenes de magnitud del número de eventos (0, 1-9, 10-99, 100-999, ≥ 1000). Es una regla propia y acotada por fila. La log verosimilitud media se descartó para ordenar factores porque la deciden unas pocas sorpresas. Ucrania 2014, con 0,8 eventos esperados y 1 201 observados, pesa más que cientos de país-años. Se reporta junto con la mediana de su cambio por fila, la parte de los país-años que mejoran, la AUPR de "al menos un evento" y el error de la densidad.
- **Comparaciones:**
  - la historia sola (línea base);
  - la historia más cada factor;
  - la historia más cada grupo;
  - el aporte único de cada grupo (quitarlo del modelo completo);
  - todo el Atlas, y todo el Atlas más World3 mundial;
  - selección hacia delante.
- **Dos lentes (dos estimadores):**
  - **Orden de magnitud:** binomial negativa por máxima verosimilitud. Ajusta bien a la mayoría de los país-años, que tienen pocos eventos. Responde dónde habrá violencia y de qué escala.
  - **Volumen:** media por pseudo-máxima verosimilitud de Poisson (PPML, Santos Silva y Tenreyro 2006), que cuadra el total de eventos, con una binomial negativa cuya dispersión depende de la media. Responde cuántos eventos habrá en total, y la deciden los países en guerra.

## 3. Resultados: qué predice mejor

### 3.1 Lente del orden de magnitud (binomial negativa)

| | 1 año | 3 años | 5 años |
|---|---|---|---|
| Sin historia (sólo la densidad media) frente a la historia | −55 % | −18 % | −20 % |
| **Todo el Atlas** sobre la historia | **+36 %** | **+39 %** | **+31 %** |
| AUPR "algún evento": historia → todo el Atlas | 0,83 → 0,92 | 0,72 → 0,89 | 0,77 → 0,87 |
| Selección hacia delante (13-14 factores) | +37 % | +37 % | +31 % |
| Todo el Atlas **+ World3 mundial** | +2 % | +15 % | **−85 %** |

Mejores factores por separado (reducción del RPS sobre la historia):

| | 1 año | 3 años | 5 años |
|---|---|---|---|
| 1 | superficie +18 % | **capacidad estatal +22 %** | **capacidad estatal +16 %** |
| 2 | capacidad fiscal +15 % | capacidad fiscal +21 % | capacidad fiscal +15 % |
| 3 | capacidad estatal +13 % | estado de derecho +18 % | deuda +12 % |
| 4 | edad del Estado +13 % | corrupción +18 % | corrupción +11 % |
| 5 | bosque +12 % | exclusión social (V-Dem) +17 % | estado de derecho +11 % |
| 6 | voto en la ONU (distancia a EE. UU.) +12 % | población excluida (EPR) +17 % | vulnerabilidad climática +11 % |
| 7 | estado de derecho +12 % | población discriminada +16 % | exclusión social +10 % |
| 8 | corrupción +12 % | edad del Estado +16 % | democracias vecinas +10 % |

Por grupo, a 3 años: instituciones **+27 %**, sociedad y geografía +25 %, clima y ambiente +23 %, análogos de World3 +22 %, economía +21 %, geopolítica +20 %, demografía +16 %, desigualdad +13 %, World3 mundial +3 %.

**Aporte único** (cuánto se pierde al quitar el grupo del modelo con todo el Atlas): sociedad y geografía 5-9 %, instituciones 2-3 %, geopolítica 2 %. La economía, la demografía y la desigualdad aportan ≈ 0 una vez que están los demás grupos.

Coeficientes del modelo seleccionado a 3 años, como razón de tasas por desviación estándar con intervalo de 90 % por bootstrap de países:
- capacidad estatal 0,39 (0,28-0,57): más capacidad, menos eventos;
- bosque 0,75 (0,62-0,95);
- edad del Estado 0,58 (0,47-0,93);
- rentas de recursos 0,72 (0,54-0,99);
- poder material 10 (4-15): países grandes y poblados;
- precio mundial de cereales 1,49 (1,07-1,77).

### 3.2 Lente del volumen (PPML)

Cuando se pondera por volumen, la escala manda. La superficie sola mejora el RPS un 25 % a 1 año, un 19 % a 3 y un 14 % a 5. Los eventos crecen mucho menos que proporcionalmente con el área: en el modelo seleccionado de 1 año la razón de tasas por desviación estándar de ln(superficie) es 0,13. Lo que importa es la población.

Todo el Atlas aporta +19 %, +8 % y +11 %. La selección se queda con 4 a 8 factores:
- **1 año:** superficie, voto en la ONU y deuda;
- **3 años:** superficie, régimen intermedio, excolonia e inflación;
- **5 años:** superficie, 10 % más rico, población, corrupción, anomalía de temperatura, disputas militarizadas, gran potencia y Gini.

### 3.3 World3

| | Lente orden de magnitud (1 / 3 / 5 años) | Lente volumen |
|---|---|---|
| Análogos por país (grupo) | **+17 % / +22 % / +14 %** | +6 % / +4 % / +3 % |
| Su aporte único (grupo economía, sobre el resto del Atlas) | −1 % / 0 % / +1 % | −5 % / −3 % / 0 % |
| Trayectoria mundial de World3 y precios mundiales (grupo) | **−23 % / +3 % / −49 %** | **−61 % / −165 % / −290 %** |
| Añadida al modelo con todo el Atlas | 36 % → 2 %; 39 % → 15 %; 31 % → −85 % | 19 % → −38 %; 8 % → −180 %; 11 % → −279 % |

- **Los análogos por país sí predicen.** El déficit ecológico (análogo de la huella), el choque de rendimientos de cereales (análogo del alimento per cápita), las rentas de recursos (los recursos no renovables) y el ingreso mejoran el RPS entre un 5 % y un 14 % cada uno. Pero lo que aportan ya lo dan las instituciones y la geografía.
- **La trayectoria mundial de World3 no sirve para predecir eventos por país.** Sólo cambia con el año, así que funciona como una tendencia que se extrapola. El caso extremo es el crecimiento industrial mundial: −18 % a 1 año y −98 % a 5 en la lente de volumen.
- **Una excepción débil:** el precio real mundial de los cereales (observado, no simulado) entra en la selección a 3 y 5 años con signo positivo. Encaja con la literatura sobre precios de alimentos y conflicto. Con sólo tres bloques de validación es una señal débil.
- **Conclusión para el acoplamiento:** World3 entra mejor por los análogos que el Atlas calcula por país que por sus agregados mundiales. Coincide con lo que ya mostraba `reports/acople/investigacion.py` para el inicio de conflicto civil.

## 4. El sistema de regresiones (`density.py`)

- **Modelos:** uno por horizonte (1 a 5 años), directo, con la media por PPML y la binomial negativa como distribución.
- **Factores:** la historia más los de la selección PPML (§3.2), sin las variables que sólo cambian con el año. Sin ellas, el RPS de 2020-2025 a 1 año queda igual (0,1777 frente a 0,1780) y la calibración mejora.
- **Datos faltantes:** se imputan con la mediana, sin indicadores. Esos indicadores aprendían países concretos: Palestina no tiene régimen en V-Dem y tiene miles de eventos, y eso contagiaba a Brunéi o Belice. A cambio entra siempre la variable "no está en el Atlas".
- **Salida, desde cualquier año de 1990 a 2025 y para cada país:** eventos esperados, densidad, P(al menos un evento), p10/p50/p90 y los tres factores que más mueven su pronóstico.
- **Entrenamiento:** sólo con objetivos ≤ el año de origen, así que cada predicción usa lo que se sabía entonces.
- **Más allá de 2019:** el Atlas queda congelado en su último dato y la historia de UCDP GED sigue al día.

### 4.1 Validación con origen rodante (orígenes 2005-2024)

| Horizonte | RPS sistema / historia, objetivos 2006-2019 | Mejora | RPS sistema / historia, objetivos 2020-2025 (Atlas congelado) | Mejora | AUPR 2020-25 (sistema / historia) | Esperado / observado 2020-25 |
|---|---|---|---|---|---|---|
| 1 año | 0,160 / 0,211 | **+24 %** | 0,178 / 0,239 | **+26 %** | 0,94 / 0,91 | 0,89 |
| 2 años | 0,211 / 0,266 | +21 % | 0,240 / 0,310 | +23 % | 0,92 / 0,88 | 0,74 |
| 3 años | 0,226 / 0,278 | +19 % | 0,275 / 0,340 | +19 % | 0,91 / 0,87 | 0,67 |
| 4 años | 0,249 / 0,294 | +15 % | 0,326 / 0,368 | +12 % | 0,88 / 0,85 | 0,66 |
| 5 años | 0,277 / 0,314 | +12 % | 0,384 / 0,389 | +1 % | 0,84 / 0,83 | 1,16 |

- **Validez de cada período:** los factores se eligieron con objetivos ≤ 2019, así que la columna 2006-2019 es optimista. La de 2020-2025 es la prueba honesta, y sostiene la mejora a 1-4 años.
- **Calibración:** a 3-4 años el sistema se queda corto, con 0,66-0,67 de lo observado, porque 2022-2024 trajo guerras nuevas (Ucrania, Sudán, Myanmar, el Sahel).
- **El estimador elegido:** la binomial negativa por máxima verosimilitud discrimina parecido, pero sobreestimaba el total de eventos hasta 10 veces con el Atlas observado y entre 1,4 y 2,3 veces en 2020-2025.

**Pronóstico desde 2025.**
- **2026:** unos 24 400 eventos en el mundo (en 2025 hubo 25 770). Por eventos esperados: Ucrania 5 559 (97 por 10 000 km²), Palestina 2 959 (la mayor densidad, 4 113 por 10 000 km²), México 2 148, Etiopía 1 217, RD del Congo 1 072, Pakistán 1 031, Nigeria 758, Rusia 718, Myanmar 711 y Burkina Faso 622.
- **2030:** a 5 años, unos 41 100. Es el horizonte menos fiable.

### 4.2 Densidad por celda

`cell_density` reparte los eventos esperados de cada país entre sus celdas. El peso de cada celda es su historia con vida media de 2 años, más un fondo del 5 % de la media del país proporcional al área. Así, una celda sin historia no queda en cero. El total del país se conserva (prueba `test_density_system_predicts_saves_and_downscales`).

### 4.3 Escenario sobre las trayectorias del L1 (exploratorio)

- **Mecánica:** el modelo de 1 año se aplica año a año a lo que simula el Atlas para cada miembro del ensemble: ingreso, crecimiento, deuda, democracia, golpes, crisis y conflicto civil. La historia de eventos se simula con la propia binomial negativa.
- **Horizonte:** desde 2019 usa el modo vivo del L1 hasta 2030; desde 1990-2018, los años con datos.
- **Resultado desde 2019:** 13 000-14 000 eventos al año en el mundo (p10-p90 de 11 000 a 17 000), frente a 13 484 → 30 066 → 25 770 observados en 2020-2025. **No anticipa las guerras nuevas** de 2022-2024: el L1 no las simula y la historia no las contiene.
- **Dos errores que hubo que corregir en el camino:**
  - Rellenar con 0 un dato faltante cambiaba su patrón de faltantes (Afganistán ×3,6).
  - Realimentar con la media fraccionaria hacía que los países en paz contaran como "con eventos" y la trayectoria explotaba.

## 5. Defectos declarados

- **Correlación predictiva, no efectos causales.** La capacidad estatal predice bien, pero no se identificó su efecto.
- **Selección y sobreajuste.** La selección hacia delante usa los mismos bloques que mide (2005-2019), así que su cifra es optimista. Por eso la prueba que cuenta es 2020-2025.
- **Atlas congelado.** Más allá de 2019 sus variables quedan en su último valor. Los países que el Atlas no tiene se apoyan en su historia.
- **Tres bloques de validación.** Las variables mundiales sólo cambian con el año y casi se confunden con ellos: tres bloques son pocos para juzgarlas. Por eso no entran en el sistema.
- **Países pacíficos.** Reciben entre 0,5 y 1 evento esperado al año (P de algún evento ≈ 0,1), algo por encima de lo observado. Un modelo de dos partes (umbral + tamaño) lo mejoraría.
- **Ubicación de los eventos.** Se asignan por la geometría de Natural Earth 1:110m-1:50m. Los eventos de las celdas de frontera pueden caer en el país vecino.
