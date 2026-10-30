# F11 de Phy-World (World4): predicción de conflicto y eventos en celdas del mapa

- **Fecha:** 2026-10-29
- **Estado:** construida y medida fuera de muestra. Cada celda terrestre de PRIO-GRID (0,5° ≈ 55 × 55 km; 64 818 celdas) recibe, para cada uno de los 12 meses siguientes al último dato:
  - la probabilidad de al menos un evento de violencia organizada de cada tipo de UCDP (estatal `sb`, no estatal `ns`, unilateral contra civiles `os`) y de cualquiera (`any`);
  - los eventos y las muertes estatales esperados (con p50 y p90 mensuales);
  - la parte de su riesgo que es fondo estructural o contagio (de su propio pasado, de sus vecinas, de su país).
- **Datos:** UCDP GED 26.1 (1989-2025) + UCDP Candidate 26.01.26.09 (enero-septiembre de 2026) + PRIO-GRID 2.0. `--install-conflict-grid`, SHA-256 en `world4/data/conflict_grid_manifest.json`. No necesita el Atlas.
- **Validación:** `world4_cli.py --validate-conflict-grid` (7 orígenes, 2019-12 … 2025-12; ~1 h). **En la media de los 7 orígenes, la combinación bate a todos los benchmarks del reto ViEWS 2023/24 en cada tipo y métrica** (origen por origen hay 4 excepciones, §4): AUPR 0,466 frente a 0,418 (cell12) y 0,358 (neighbors12); CRPS de muertes estatales en África y Oriente Medio 0,237 frente a 0,250 (neighbors12) y 0,261 (cero). En 2026, contra eventos candidatos que no existían al descargar los datos de entrenamiento: AUPR 0,480 frente a 0,407. Resultados en `reports/conflicto_celdas/validacion.json`; tablas con `reports/conflicto_celdas/tablas.py`.
- **Pruebas:** `tests/test_f11_conflict_grid.py`, 8 nuevas (mundo sintético + datos instalados).
- **Interfaz:** `api.conflict_grid_forecast()` y `world4_cli.py --conflict-grid` devuelven la carga útil `world4.conflict_grid/1` (JSON por columnas), pensada para la interfaz en React que viene después.

## 1. Lo que dice la literatura y lo que se tomó de ella

| Referencia | Qué aporta | Qué se tomó |
|---|---|---|
| Hegre et al. (2019) *ViEWS: a political violence early-warning system*, JPR 56(2); Hegre et al. (2021) *ViEWS2020*, JPR 58(3); Apéndice A de ViEWS2020 | El estándar de predicción subnacional: celda PRIO-GRID × mes, eventos de UCDP GED, umbral de 1 muerte en `pgm`; ~0,06 % de celda-mes con violencia estatal en África: clases muy desbalanceadas; submuestreo de ceros con recalibración; historia del conflicto con funciones de decaimiento por vida media (meses desde el último evento) | La unidad, el umbral, los predictores de historia con decaimiento y el submuestreo de ceros (aquí con pesos 1/probabilidad, que no requiere recalibrar) |
| Hegre, Nordenving y Dale (2024) *Models in VIEWS, Fatalities002* (documentación) | En `pgm` el ensemble promedia sin pesos modelos LightGBM y **hurdle** (clasificador de cero/no cero × regresor del log de muertes) sobre conjuntos de predictores: historia del conflicto, espacio-tiempo, sequía, geografía natural y social de PRIO-GRID | El modelo hurdle directo por horizonte y la idea de promediar modelos distintos |
| Hegre et al. (2025) *The 2023/24 VIEWS Prediction Challenge*, JPR 62(6) (arXiv:2407.11045 y apéndice) | En `pgm` el cero inflado es extremo (99,5 % de ceros): el modelo "exactly zero" es difícil de batir en CRPS (0,45 en 2018-2023, África y Oriente Medio); el mejor benchmark es **conflictology neighbors12** (los 12 últimos meses de la celda y sus 8 vecinas como distribución). Métricas: CRPS, log-score, MIS | Los benchmarks (`benchmarks.py`) y el CRPS sobre muestras como métrica de muertes |
| Browning, Patten, Rousseau y Mengersen, *Bayesian spatiotemporal modelling of political violence … discrete-time Hawkes processes*, JRSS-A, doi 10.1093/jrsssa/qnag039 (arXiv:2408.14940) | Proceso de Hawkes en tiempo discreto para conteos de ACLED: λ = exp(μ₀ + μₓ) + α Σ y g(t−s) h(distancia); núcleo temporal geométrico truncado; la autoexcitación mejora claramente al modelo sólo de fondo; núcleos de ley de potencias no ganan; memorias largas (52 semanas) ajustan mejor | El modelo de Hawkes: fondo espacial (aquí con covariables de PRIO-GRID en vez de un parámetro por región), núcleos geométricos (tres vidas medias: 1, 4 y 24 meses), núcleo espacial discreto por anillos y excitación cruzada entre tipos |
| Held, Höhle y Hofmann (2005), modelo endémico-epidémico, Statistical Modelling; Mohler et al. (2011) *Self-exciting point process modeling of crime*, JASA | La forma "fondo + epidemia" para conteos y su estimación por EM sobre la estructura de ramificación | La estimación por EM (`models.Hawkes.fit`) |
| Schutte y Weidmann (2011) *Diffusion patterns of violence in civil wars*, Political Geography (doi 10.1016/j.polgeo.2011.03.005) | La violencia en guerras civiles se difunde sobre todo por **escalada** (el área se ensancha hacia las vecinas), no por relocalización | Los rezagos espaciales en los anillos 1 y 2 y la excitación de vecinas en el Hawkes |
| Mueller y Rauh (2022) *The hard problem of prediction for conflict prevention*, JEEA (CEPR DP13748) | Los modelos con historia aciertan donde ya hay conflicto; los estallidos en lugares en paz son el problema difícil (allí ayudan las noticias) | El subconjunto `onset` de la validación (celdas sin eventos en ellas ni en sus vecinas en 24 meses), reportado aparte |
| Fritz et al. (2024), *PLY_cal_double_hurdle* en el reto 2023/24 | Un hurdle de GLM, ligero e interpretable, compite en `pgm` | Que el umbral sea una logística penalizada (interpretable: aportes por predictor) y no un árbol |
| Tollefsen, Strand y Buhaug (2012) *PRIO-GRID*, JPR 49(2); Sundberg y Melander (2013) *UCDP GED*, JPR 50(4); Hegre et al. (2020) *UCDP Candidate* | Los datos | Los datos |

## 2. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Datos | `world4/data/conflict_grid.py`, `conflict_grid_manifest.json` | Descarga GED 26.1, los candidatos de 2026 y 15 variables de PRIO-GRID por su API; comprueba SHA-256 (para PRIO-GRID, el de la tabla (gid, valor) ordenada, porque la API no garantiza el orden); escribe `conflict_grid.npz` (eventos con mes, celda, tipo, muertes, país, precisión; celdas con covariables) |
| Malla | `world4/conflict/grid.py` | Vecindades de Chebyshev 1 y 2 como matrices dispersas (con el antimeridiano), país de cada celda, distancia a la celda activa más cercana, eventos como CSR mes × celda |
| Predictores | `world4/conflict/features.py` | Barrido mes a mes que en el mes t sólo ve eventos ≤ t: 32 predictores del umbral y 30 estados de excitación del Hawkes; muestreo de filas estratificado SÓLO por la historia |
| Modelos | `world4/conflict/models.py` | `Logistic` (Newton amortiguado, L2, pesos), `Size` (log1p de muertes, MCO ponderados), `Hawkes` (EM + θ de la binomial negativa) |
| Pronóstico | `world4/conflict/forecast.py` | Ajuste por origen, simulación del Hawkes, combinación, muestras de muertes, descomposición |
| Referencias y métricas | `benchmarks.py`, `metrics.py` | Benchmarks del reto ViEWS; Brier, log-loss, AUROC, AUPR, CRPS, calibración |
| Validación | `validate.py`, `reports/conflicto_celdas/` | Origen rodante, subconjuntos `all`, `ame`, `onset` |
| Servicio, API, CLI | `service.py`, `api.conflict_grid_forecast`, `--install-conflict-grid`, `--conflict-grid`, `--validate-conflict-grid` | El modelo de cada origen se guarda en `data_cache/conflict_grid/models/` y se reutiliza |

### Reglas de los datos (declaradas en `conflict_grid.RULES`)

- Mes del evento = mes de `date_start` (sólo el 1,75 % de los eventos cruza de mes).
- Un evento cuenta en su celda si `where_prec ≤ 4`: punto exacto, radio de 25 km, o centroide de división administrativa de 1.º o 2.º orden. Así quedan **409 082 de 434 601** eventos. Con `where_prec` 5-7 (rasgo lineal, sólo país, aguas internacionales) cuenta sólo para su país. ViEWS imputa esas ubicaciones (Croicu y Hegre 2018); aquí no.
- La celda es la `priogrid_gid` de UCDP, aunque PRIO-GRID asigne la celda a otro país (lo exige el codebook de GED).
- Después de diciembre de 2025, los eventos son los **candidatos**: preliminares (≈ 35 % con `code_status` "Check …"), publicados cada mes con a lo sumo un mes de retraso.

## 3. El modelo

**Predictores del umbral** (en t, para t + h): para cada tipo, las sumas de eventos con vida media de 1, 4 y 24 meses (log1p) y `0,5^(meses desde el último evento / 12)`; muertes estatales con vida media de 3 y 24 meses; los mismos eventos (todos los tipos) sumados en los anillos 1 y 2; el decaimiento del último evento en la celda o sus vecinas; la distancia (en celdas) a la celda con eventos en los últimos 12 meses más cercana; del país, sus eventos (todos, también los que no tienen celda), sus muertes y la parte de sus celdas activas; y la geografía de PRIO-GRID: población (GPW 2010), distancia a la capital y a la frontera, tiempo de viaje a una ciudad, montaña, cobertura urbana, mortalidad infantil, grupos excluidos (EPR) y recursos (petróleo, diamantes, oro, gemas).

**Umbral logístico** (`Logistic`): uno por horizonte h = 1…12 y por resultado, más uno para "al menos un evento en los 12 meses". Penalización L2 = 1 sobre predictores estandarizados.

**Hawkes** (`Hawkes`), uno por tipo de destino v:

λᵥ(i, t+1) = exp(γᵥ · geografíaᵢ) + Σ_u Σ_k [ aᵥᵤₖ^celda · Sᵤₖ(i) + aᵥᵤₖ^anillo1 · Σ_{j∈A₁(i)} Sᵤₖ(j) + aᵥᵤₖ^anillo2 · Σ_{j∈A₂(i)} Sᵤₖ(j) ] + Σ_u aᵥᵤ^país · Sᵤ,₄(país(i)) / celdas(país)

donde Sᵤₖ = Σ_{s≤t} bₖ (1 − bₖ)^{t−s} yᵤ(s) es el estado de un núcleo geométrico normalizado con vida media k ∈ {1, 4, 24} meses, y Y ~ BinNeg(λ, θ). Todas las a ≥ 0. Cada a es el número esperado de "hijos" de tipo v por evento de tipo u; su suma ponderada por el número de celdas de cada anillo es la **razón de ramificación**. Estimación: EM (cuasi-verosimilitud de Poisson para la media), luego θ por máxima verosimilitud. Pronóstico: S trayectorias simuladas mes a mes. La marginal mensual se calcula por Rao-Blackwell, como la media de P(Y ≥ 1 | λ simulada); la probabilidad de la ventana se toma de las trayectorias cuando hay ≥ 20 aciertos y, si no, del producto de las marginales.

**Combinación:** logit(p) = 0,5 · logit(p_logística) + 0,5 · logit(p_Hawkes). El peso **no** se ajustó con la validación.

**Muertes estatales:** Bernoulli(p combinada de sb) × redondeo de expm1(N(μ(x), σ)), con μ y σ de `Size` por horizonte.

**Sin fugas:** el barrido usa en el mes t sólo eventos ≤ t (`test_features_never_look_ahead`). Las filas del horizonte h son meses m ≤ T − h, y su probabilidad de inclusión depende sólo de la historia: 0,25 si hay masa decaída en la celda o sus dos anillos; 0,004 si no. Las filas no se eligen mirando el resultado.

## 4. Validación fuera de muestra

Origen T = diciembre de 2019 … 2025; se entrena con datos ≤ T y se pronostican los 12 meses siguientes (9 para T = 2025-12: la verdad de 2026 son los eventos candidatos hasta septiembre). Las métricas son medias sobre los meses de origen (7 × 12 − 3 = 81), con 64 818 celdas por mes.

**En una línea:** en la media de los orígenes, la combinación supera a todos los modelos de referencia del reto ViEWS en los tres tipos de violencia y en todas las métricas: Brier, log-loss, AUROC y AUPR para la probabilidad de evento; CRPS para las muertes. También en 2026, contra eventos que no existían cuando se descargaron los datos de entrenamiento. Origen por origen (mundo) pierde en 4 comparaciones, siempre por poco:

- el Brier frente a *cell12* en 2019-12 (`ns` y `any`) y en 2020-12 (`ns`);
- el CRPS de muertes frente al cero en 2021-12 (0,1113 frente a 0,1102).

La AUPR gana en todos los orígenes. Donde menos aporta es en los estallidos en celdas sin historia.

### 4.1 Probabilidad de al menos un evento (cualquier tipo), mundo, h = 1…12

| Modelo | Brier | Log-loss | AUROC | AUPR | Eventos esperados / observados |
|---|---|---|---|---|---|
| **Combinación** | **0,00548** | **0,02145** | **0,979** | **0,466** | 38 491 / 41 396 |
| Logística | 0,00549 | 0,02157 | 0,979 | 0,466 | 38 265 / 41 396 |
| Hawkes | 0,00560 | 0,02205 | 0,977 | 0,446 | 41 475 / 41 396 |
| ViEWS neighbors12 | 0,00604 | 0,02763 | 0,947 | 0,358 | 38 003 / 41 396 |
| cell12 | 0,00563 | 0,03830 | 0,881 | 0,418 | 38 758 / 41 396 |
| Frecuencia histórica | 0,00688 | 0,03225 | 0,942 | 0,230 | 20 655 / 41 396 |
| Persistencia | 0,00896 | 0,12382 | 0,685 | 0,165 | 36 405 / 41 396 |

Con la tasa base de ~0,6 % de celda-mes con eventos, el AUROC sale alto para casi todos. La AUPR es la que separa: **0,466 frente a 0,418** del mejor benchmark (+11 %) y **0,358** de *neighbors12* (+30 %).

### 4.2 Por tipo de violencia (AUPR, mundo, h = 1…12)

| Tipo | Combinación | Logística | Hawkes | ViEWS neighbors12 | cell12 |
|---|---|---|---|---|---|
| Estatal (`sb`) | **0,451** | 0,448 | 0,433 | 0,361 | 0,401 |
| No estatal (`ns`) | **0,410** | 0,402 | 0,388 | 0,239 | 0,345 |
| Unilateral (`os`) | **0,259** | 0,257 | 0,235 | 0,185 | 0,201 |

La violencia contra civiles es la más difícil: es más esporádica y la autoexcitación es menor (0,9 frente a 0,97-1,0).

### 4.3 Por origen (AUPR, cualquier evento, mundo)

| Origen | Combinación | ViEWS neighbors12 | cell12 |
|---|---|---|---|
| 2019-12 | **0,525** | 0,424 | 0,504 |
| 2020-12 | **0,477** | 0,358 | 0,433 |
| 2021-12 | **0,360** | 0,240 | 0,315 |
| 2022-12 | **0,458** | 0,368 | 0,411 |
| 2023-12 | **0,487** | 0,359 | 0,420 |
| 2024-12 | **0,481** | 0,386 | 0,432 |
| 2025-12 (verdad: candidatos 2026) | **0,480** | 0,377 | 0,407 |

2022 es el peor año para todos: la invasión rusa de Ucrania abrió un frente donde no había historia reciente.

### 4.4 Al menos un evento en los próximos 12 meses (lo que pinta el mapa por defecto)

| Modelo | Brier | AUROC | AUPR |
|---|---|---|---|
| **Combinación** | 0,01514 | **0,977** | 0,684 |
| Logística | **0,01501** | 0,977 | **0,689** |
| Hawkes | 0,01614 | 0,972 | 0,649 |
| ViEWS neighbors12 (1 − (1 − p)¹²) | 0,02015 | 0,922 | 0,572 |
| cell12 | 0,01829 | 0,797 | 0,507 |
| Persistencia | 0,02484 | 0,597 | 0,187 |

### 4.5 Muertes estatales (CRPS; menor es mejor)

| Modelo | África y Oriente Medio, h = 1…12 | Ídem, 2026 (candidatos) | Mundo, h = 1…12 |
|---|---|---|---|
| **Combinación** | **0,2365** | **0,1810** | **0,0897** |
| ViEWS neighbors12 | 0,2498 | 0,1970 | 0,0954 |
| cell12 | 0,2575 | 0,2497 | 0,0994 |
| ViEWS exactly zero | 0,2606 | 0,2000 | 0,1032 |
| ViEWS last historical | 0,3828 | 0,2796 | 0,1358 |

El orden coincide con el que el reto de ViEWS encontró en `pgm`: *neighbors12* bate al cero y *last historical* es el peor. La combinación bate a los cuatro, aunque por poco (−5 % frente a *neighbors12*, −9 % frente al cero). El cero inflado deja poco margen a cualquier modelo de muertes por celda. Los niveles no son comparables con las tablas de ViEWS (0,45 para el cero en 2018-2023): aquí se cuentan los eventos con `where_prec ≤ 4` sin imputar, los orígenes son diciembre y no octubre, y los años son otros.

### 4.6 Celdas sin historia (el "problema difícil"), cualquier evento

| Modelo | AUROC | AUPR | Eventos esperados / observados |
|---|---|---|---|
| **Combinación** | 0,921 | 0,010 | 1 427 / 2 148 |
| Logística | **0,928** | **0,012** | 1 231 / 2 148 |
| Hawkes | 0,893 | 0,008 | 2 492 / 2 148 |
| Frecuencia histórica | 0,708 | 0,006 | 3 429 / 2 148 |
| ViEWS neighbors12, cell12, persistencia | 0,500 | 0,000 | 0 / 2 148 |

Los benchmarks de historia dicen 0 por construcción. El modelo **ordena** bien las celdas tranquilas (AUROC 0,92): la distancia al conflicto, el país y la geografía informan. Pero la AUPR de 0,01 dice que acertar dónde estallará es casi imposible con estos datos (la tasa base es 0,05 %). Es lo que dicen Mueller y Rauh, y es el lugar donde más aportarían predictores nuevos.

### 4.7 Calibración (combinación, cualquier evento, h = 1…3, todos los orígenes)

| p pronosticada | p media | Frecuencia observada | Celda-mes |
|---|---|---|---|
| [0, 0,001) | 0,0001 | 0,0003 | 1 149 280 |
| [0,001, 0,003) | 0,0017 | 0,0031 | 78 677 |
| [0,003, 0,01) | 0,0058 | 0,0093 | 51 704 |
| [0,01, 0,03) | 0,0173 | 0,0237 | 35 286 |
| [0,03, 0,1) | 0,0559 | 0,0636 | 22 075 |
| [0,1, 0,2) | 0,1430 | 0,1514 | 10 297 |
| [0,2, 0,4) | 0,2832 | 0,2886 | 7 626 |
| [0,4, 0,6) | 0,4878 | 0,4723 | 3 191 |
| [0,6, 0,8) | 0,6941 | 0,6819 | 1 811 |
| [0,8, 1,0) | 0,8942 | 0,8871 | 1 231 |

Por encima de 0,1 está bien calibrada. Por debajo de 0,03 subestima entre 1,4 y 3 veces en términos relativos; en total pronostica un 7 % menos eventos que los observados. Es el mismo defecto que en las celdas sin historia. No se recalibró, para no ajustar nada a la validación: queda como defecto declarado y como candidato a una recalibración Platt con orígenes anteriores a 2019.

### 4.8 El proceso de Hawkes: cuánto contagia un evento

| Origen | sb → sb | ns → ns | os → os | os → sb | sb → os |
|---|---|---|---|---|---|
| 2019-12 | 0,977 | 0,981 | 0,891 | 0,142 | 0,017 |
| 2022-12 | 0,972 | 0,996 | 0,899 | 0,140 | 0,016 |
| 2025-12 | 0,975 | 0,999 | 0,912 | 0,144 | 0,011 |

(Hijos esperados de cada tipo por evento, sumando la celda, los 24 vecinos de los anillos 1-2 y el país.) Las estimaciones son estables entre orígenes. La violencia organizada es casi crítica: cada evento estatal engendra 0,97 eventos estatales más. La excitación cruzada es asimétrica: la violencia contra civiles precede a combates (0,14 eventos estatales por evento unilateral), y no al revés (0,01-0,02).

### 4.9 Tiempos (4 CPU)

Barrido de predictores 1989-2026 en 45 s. Ajuste de un origen en ~5 min: 52 logísticas con ~1,2 millones de filas y 3 Hawkes por EM. Simulación de 200 trayectorias × 12 meses × 64 818 celdas y carga útil en ~1-2 min.

### 4.10 Pronóstico vivo (origen 2026-09, para 2026-10 … 2027-09)

Se esperan 13 128 eventos estatales (en los 12 meses observados hasta 2026-09 hubo 12 492), 5 856 no estatales (5 235) y 2 736 unilaterales (2 710), y unas 53 000 muertes estatales (47 950). Por eventos esperados, los países de mayor riesgo (Gleditsch-Ward) son Ucrania (369), México (70), Colombia (100), Nigeria (475), Pakistán (770), Israel/Palestina (666), Yemen (678), RD del Congo (490), Etiopía (530) y Malí (432). En las celdas de mayor riesgo, el 95-100 % de la intensidad del mes siguiente es autoexcitación de la propia celda: el conflicto en curso.

## 5. Lo que no hace (defectos declarados)

- **Sólo historia del conflicto y geografía.** No usa noticias, política, economía ni clima. Por eso los estallidos en lugares en paz siguen siendo difíciles, como dicen Mueller y Rauh. ViEWS usa V-Dem, el WDI, la sequía y las noticias. El L1 de World4 podría aportar riesgos por país (`conflict_hazard`, golpes, crisis) como predictores del umbral: es el acoplamiento natural siguiente.
- **Fuga de revisión.** Los orígenes anteriores usan como predictores el GED final (revisado), no lo que existía en T (los candidatos).
- **Covariables congeladas.** Las de PRIO-GRID están en su último año publicado (2010-2014) y las celdas conservan el país de 2014.
- **Ubicaciones imprecisas.** `where_prec` 5-7 no se imputan a celdas (≈ 6 % de los eventos).
- **Hawkes casi crítico.** La razón de ramificación sale ≈ 1 (conflicto muy persistente; en `ns` llega a 1,008 en 2021). A 12 meses la simulación es estable, pero el proceso no sirve para horizontes largos sin un tope.
- **Subestima los riesgos bajos** (§4.7) y los estallidos (§4.6), y el total en un 7 %; no se recalibró para no ajustar nada a la validación.
- **Distancia aproximada.** Es euclídea en la malla: no corrige por la latitud.
- **Tamaño de las muertes.** La parte de tamaño es la más débil (σ ≈ 1,3 en log1p) y no tiene colas de ley de potencias.
- **Interacciones.** Un solo modelo logístico lineal en sus predictores transformados; ViEWS usa árboles. Una prueba con árboles (dependencia nueva) queda para después de ver la interfaz.

## 6. La carga útil para la interfaz (`world4.conflict_grid/1`)

`api.conflict_grid_forecast(origin=None, min_p=0.0, monthly=True, bbox=None, countries=None)` devuelve:

- `origin`, `months` (12 etiquetas `AAAA-MM`), `data` (versiones y reglas), `grid`, `outcomes`, `columns` y `units` (qué es cada columna y en qué unidad), `caveats`, `sources`, `exploratory: false`;
- `cells`: columnas paralelas, una entrada por celda devuelta, ordenadas por riesgo:
  - `gid`, `lon`, `lat`, `country_gw`;
  - `p_window_{sb,ns,os,any}`, `p_month_{…}` (listas de 12);
  - `events_expected_{sb,ns,os}`, `deaths_sb_expected`, `deaths_sb_p50`, `deaths_sb_p90`;
  - `driver_{background,own,neighbours,country}` (partes de la intensidad del mes siguiente, que suman 1);
  - `events_last12`, `deaths_last12`.
- `countries`: eventos y muertes esperados por país (Gleditsch-Ward), celdas con P > 0,5 y el máximo por celda;
- `summary`: totales mundiales y la razón de ramificación; `validation`: lo esencial de la última validación.

El polígono de cada celda es `[lon − 0,25, lat − 0,25, lon + 0,25, lat + 0,25]`. El modelo de cada origen se guarda la primera vez (unos 6 minutos); después cada llamada tarda alrededor de 1 minuto (simulación de 200 trayectorias).
