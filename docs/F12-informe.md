# F12 de Phy-World (World4): la capa de eventos (L2), de Oriente Medio al mundo

- **Fecha:** 2026-11-16 (Oriente Medio) y 2026-11-17 (ampliación mundial, §12).
- **Estado:** construida y medida en las cinco regiones de UCDP.
  - **W13: ROJO, también a escala mundial.** El simulador de agentes no supera a sus referencias; causa nombrada y medida (§4 y §12).
  - El puente L2 → L1 sí mejora el evento anual de conflicto (§6; medido sólo en Oriente Medio).
  - El desplazamiento es exploratorio y pierde contra la persistencia (§7 y §12).
- **Pruebas:** 36 nuevas (`tests/test_f12_l2.py`), sobre un mundo sintético salvo tres que usan los paneles reales si están instalados.
- **Orden de lectura:** §1-§11 son la primera región tal como se midió; §12 es la ampliación y manda donde difieran.
- **Máquina:** contenedor Linux de 4 núcleos y 16 GB; Python 3.13. Es la primera vez que World4 corre fuera de Windows (§8).

El nombre «L2» lo dejó libre la hoja de ruta: FaIR y los cultivos, que iban a ser un L2 aparte, viven en el L1 desde F4/F5. F11 sigue reservada al estimador estructural sobre el L1 completo.

## 1. Qué se construyó

| Pieza | Qué hace |
|---|---|
| `data/conflict.py`, `l2_manifest.json` | Descarga y comprueba (SHA-256) UCDP GED 26.1, UCDP Candidate, PRIO-GRID 2.0, ACNUR y WPP; escribe `data_cache/l2/l2_mideast.npz`. |
| `core/l2/regions.py` | Qué países forman cada región. `mideast` son los 15 Estados de la región «Middle East» de UCDP; desde §12, las cinco regiones de UCDP y `world`. |
| `core/l2/panel.py` | El panel por **sitio** (celda, país), celda y país; rasgos con retardo; `truncated(t0)`, el corte «a fecha de». |
| `core/l2/baseline.py` | La vara: `zero`, `last12` (los 12 meses anteriores de la unidad) y `hurdle` (logit de ocurrencia + regresión del log de las muertes, directo por horizonte). |
| `core/l2/agents.py` | El simulador: actores, díadas, huella espacial, nacimiento y muerte de díadas. |
| `core/l2/displacement.py` | Bloques de población: huida, destino (interno o frontera) y retorno. |
| `core/l2/forecast.py` | Ensembles desde un origen; covariable del L1 (L1 → L2) y evento anual de conflicto (L2 → L1). |
| `core/l2/synthetic.py` | Un mundo sintético con las mismas claves que el panel real, para las pruebas. |
| `validate/l2.py` | W13: origen móvil, puntuación contra las referencias, IC por países, y la medición del desplazamiento. |
| API y CLI | `api.l2_forecast()`; `--install-l2`, `--l2-forecast`, `--l2-validate`; criterio W13 en `--campaign`. |

**No se hizo:** herramienta MCP `w4_l2_forecast` (la auditoría W11 llama a cada herramienta sin argumentos y exigiría los datos del L2 en toda instalación), capa del L2 en el globo de `--dashboard`, y ViEWS como referencia (§9).

## 2. Datos

| Fuente | Qué aporta | Licencia |
|---|---|---|
| UCDP GED 26.1 | 128 033 eventos de Oriente Medio, 1989-2025: fecha, lugar, bandos, muertes (`best`, `low`, `high`) | CC BY 4.0 [V] |
| UCDP Candidate 26.0.X | enero-septiembre de 2026; sólo los 2 124 eventos de la región con `code_status == "Clear"` | CC BY 4.0 [V] |
| PRIO-GRID 2.0 | malla de 0,5°, país de la celda, población GPW v3, distancias, montaña, luces, grupos excluidos | sin licencia explícita [V]: no redistribuible |
| ACNUR (`refugees` 2025.12.3) | refugiados y solicitantes por origen y asilo, desplazados internos, tabla de IDMC, 1951-2025 | CC BY 4.0 [V] |
| UN WPP 2024 | población por país, estimada hasta 2023 y proyectada | CC BY 3.0 IGO [S] |

Las licencias marcadas [V] se leyeron en esta sesión (`sources.json` dice dónde). Nada viaja en el repo.

**El panel:** 15 países, 2 782 celdas, 2 866 sitios, 453 meses (1989-01 a 2026-09), 265 díadas y 239 actores (19 gobiernos, 219 grupos no estatales y «civiles»).

**Convenciones que cambian números:**
- **ISR es Israel más los territorios palestinos.** UCDP codifica Gaza y Cisjordania en «Israel» y PRIO-GRID asigna esas celdas al mismo código.
- **Sitios.** El 5,9 % de los eventos localizados ocurre en un país que no es el de su celda (a 0,5° una celda de frontera es de dos países). El sitio (celda, país) conserva el país que dice UCDP.
- **Eventos sin localizar.** El 1,2 % de los eventos (16 % de las muertes, casi todo resúmenes de guerras) sólo tiene país: cuentan en el país y en ninguna celda.
- **Eventos que cruzan de mes** (0,6 %): reparten sus muertes a partes iguales.
- **El candidato de 2026 nunca es verdad de prueba** y subcuenta: sólo entra lo ya revisado.

**Comprobación contra una cifra conocida:** Siria 2014 suma 78 709 muertes, la del GED 26.1 (prueba).

## 3. El simulador

Cada mes, en cada miembro del ensemble:

1. **Combatir o no.** Cada díada viva se activa según un logit agrupado. Rasgos: sus eventos del último mes, de los últimos 3 y 12; meses que lleva callada; edad; actividad de cada uno de sus dos actores **en sus otros frentes**; violencia del resto del país; tipo de violencia; si es entre dos gobiernos.
2. **Cuánto.** Si se activa, el log del número de eventos sigue una regresión con los mismos rasgos y residuos remuestreados.
3. **Con qué letalidad.** Cada evento toma sus muertes de los eventos pasados de su díada, mezclados con los de su tipo si la díada tiene poca historia (encogimiento m / (m + 10)).
4. **Dónde.** Cada díada tiene una huella sobre los sitios: memoria con decaimiento de dónde ha combatido, difundida a las celdas vecinas **del mismo país**. Un suelo pequeño cae fuera de la huella: donde ha habido violencia de cualquier díada del país, mezclado con donde vive la gente. Los eventos simulados actualizan la huella.
5. **Díadas nuevas.** Nacen por país con una tasa de Poisson que crece con la violencia existente y copian tipo y papeles de una díada nacida antes en ese país. Una díada callada 120 meses muere.

**Estimado con eventos anteriores al origen:** los dos modelos de las díadas, los depósitos de letalidad, la tasa de nacimientos y los cuatro parámetros de la huella (decaimiento, difusión, suelo y mezcla, por verosimilitud a un paso de los lugares de los eventos).

**Coeficientes en el origen 2020-01** (para ver qué pesa): callada −1,15 por log-mes; otros frentes del actor A +0,10 y del B +0,07 (un actor ocupado en otro frente combate **más** en éste, no menos); resto del país −0,08; no estatal −0,56; entre gobiernos −0,50.

**Supuesto, no estimado [M]:**
- la semilla espacial de una díada nueva;
- las muertes se reparten entre sitios en proporción a los eventos;
- techo de eventos por díada y mes: 2 veces el máximo histórico de la región;
- memoria de la huella del país: 0,97 por mes.

**Lo que no hay:**
- **Ningún stock de fuerza.** UCDP no observa efectivos ni recursos. El estado de un actor es su actividad en todos sus frentes.
- **Ninguna maquinaria aparte para guerras entre Estados.** En UCDP son díadas de dos gobiernos y aquí también. En la región hay siete en 37 años: esa parte es exploratoria por falta de casos, no por decisión.
- **Protesta, golpes y represión no letal.** UCDP no los registra.

## 4. W13: el simulador contra su vara

**Protocolo, fijado antes de correr.** Orígenes de enero y julio. Todo se estima sobre el panel cortado en el origen y se puntúa contra el GED final. La métrica que decide es el CRPS de ln(1 + muertes mensuales). La diferencia con cada referencia lleva un IC95 por remuestreo de países enteros.

- **Desarrollo** (20 orígenes, 2005-2014): sirvió para elegir la forma del simulador (§5).
- **Prueba** (20 orígenes, 2015-2024): se corrió **una vez** con la forma fijada.

### Prueba, 2015-2024 (CRPS de ln(1 + muertes); menor es mejor)

| Nivel | Meses | N | Agentes | `hurdle` | `last12` | `zero` | Agentes − mejor referencia |
|---|---|---|---|---|---|---|---|
| País | 1 | 300 | 0,322 | **0,305** | 0,335 | 1,854 | +0,017 [−0,010, +0,046] |
| País | 3 | 300 | 0,369 | **0,335** | 0,372 | 1,799 | +0,034 [+0,005, +0,071] |
| País | 6 | 300 | 0,426 | **0,414** | 0,418 | 1,834 | +0,013 [−0,026, +0,051] |
| País | 12 | 300 | 0,513 | **0,457** | 0,489 | 1,798 | +0,056 [−0,002, +0,112] |
| País | 24 | 285 | 0,660 | **0,566** | 0,616 | 1,728 | +0,094 [+0,013, +0,191] |
| País | 36 | 255 | 0,815 | **0,634** | 0,732 | 1,669 | +0,182 [+0,042, +0,342] |
| Celda | 1 | 55 640 | 0,0382 | 0,0356 | **0,0356** | 0,0608 | +0,0026 [+0,0004, +0,0096] |
| Celda | 3 | 55 640 | 0,0423 | 0,0390 | **0,0388** | 0,0612 | +0,0035 [+0,0008, +0,0123] |
| Celda | 6 | 55 640 | 0,0439 | 0,0401 | **0,0395** | 0,0595 | +0,0044 [+0,0011, +0,0146] |
| Celda | 12 | 55 640 | 0,0481 | **0,0411** | 0,0417 | 0,0551 | +0,0070 [+0,0021, +0,0216] |

**Lectura:**

- **El simulador no supera a la mejor referencia en ningún nivel ni horizonte.** Por país queda entre un 3 % y un 29 % peor que `hurdle`; la diferencia es significativa a 3, 24 y 36 meses. Por celda pierde entre un 7 % y un 17 %, con IC que excluye el cero en los cuatro pasos.
- **Contra la persistencia (`last12`), por país empata:** el IC incluye el cero en los seis horizontes; a 1 y 3 meses el simulador sale algo mejor (0,322 frente a 0,335).
- **Acierta mejor si habrá violencia que cuánta.** El Brier de «al menos una muerte» por país es el mejor de los tres a 1, 6, 24 y 36 meses (a un mes: 0,059 frente a 0,066 de `hurdle` y 0,064 de `last12`). Lo que pierde es la magnitud.
- **El error crece con el horizonte más deprisa que el de `hurdle`.** El simulador itera mes a mes y acumula el ruido de sus propios pasos; `hurdle` se ajusta directamente a cada horizonte.
- **Veredicto: ROJO.** Nada del L2 es de grado de pronóstico. Todo lleva `exploratory`, y `api.l2_forecast` entrega la referencia `hurdle` junto al simulador para que quien lea use la que mide mejor.

### Desarrollo, 2005-2014 (misma métrica)

| Nivel | Meses | Agentes, primera versión | Agentes, versión final | `hurdle` | `last12` |
|---|---|---|---|---|---|
| País | 1 | 0,461 | 0,340 | 0,341 | 0,355 |
| País | 3 | 0,448 | 0,326 | 0,338 | **0,319** |
| País | 6 | 0,499 | **0,355** | 0,389 | 0,364 |
| País | 12 | 0,618 | **0,447** | 0,453 | 0,450 |
| País | 24 | 0,776 | **0,606** | 0,642 | 0,619 |
| País | 36 | 0,974 | 0,769 | 0,815 | **0,765** |
| Celda | 1 | 0,0216 | 0,0205 | **0,0198** | 0,0210 |
| Celda | 12 | 0,0344 | 0,0308 | **0,0286** | 0,0294 |

**En desarrollo el simulador empataba por país con la mejor referencia; en prueba pierde.** Parte de esa distancia es la esperada al elegir la forma con esos orígenes. La otra parte es que `hurdle` mejora mucho en 2015-2024 (0,305 frente a 0,341 a un mes): con más historia de guerra en el panel, el modelo estadístico aprende más que el simulador.

## 5. Lo que se probó en desarrollo

| Cambio | Efecto en desarrollo (país, 1 mes) | Decisión |
|---|---|---|
| **La huella no se difunde a través de una frontera** | 0,461 → 0,342 | Se queda. En la primera versión la violencia de Gaza y Cisjordania se derramaba a Jordania: P(alguna muerte) simulada 0,37 frente a 0,01 observada. |
| Suelo de la huella mezclado con la violencia de todo el país | 0,342 → 0,340; celda a 12 meses 0,0315 → 0,0308 | Se queda. La mezcla la elige la verosimilitud, no la puntuación. |
| Letalidad reciente con más peso (vida media de 12 o 36 meses) | sin cambio medible en 10 orígenes | Descartado. |
| **Covariable del L1** (probabilidad anual de conflicto, origen tres años antes) | 0,340 → 0,336 | Apagada por defecto. En prueba: 0,322 → 0,320 a un mes y **peor** a 3, 24 y 36 meses (0,815 → 0,835). No aporta. |

## 6. El puente L2 → L1

`forecast.conflict_members` da, miembro a miembro, el evento `conflict` del estimador estructural: 25 o más muertes de violencia estatal en el año. Medido en los 12 países de la región que están en el L1, años 2015-2024 (120 preguntas; frecuencia 0,49), con el L2 pronosticando desde enero de cada año:

| Estimador | Brier |
|---|---|
| **L2 desde enero** | **0,084** |
| Igual que el año anterior | 0,108 |
| L1, origen tres años antes | 0,256 |

- **El L2 mejora a la persistencia en el evento anual.** No hay IC: son 120 preguntas de 12 países y no se remuestreó.
- **La comparación con el L1 no es justa y se declara:** el L2 ve datos hasta diciembre del año anterior y el L1 hasta tres años antes (el retraso de la PWT). Mide lo que cuesta ese retraso, no la calidad de cada modelo.
- **No está cableado en `structural/estimator.py`:** cambiaría W9a. Es trabajo de F11.

## 7. Desplazamiento

Ocho parámetros comunes y un multiplicador por país, ajustados a los stocks anuales por país de origen. El error se mide en asinh(personas / 10 000), que para stocks grandes es un error logarítmico: 0,69 es un factor 2.

**Dentro de la muestra** (ajuste con datos hasta 2024): error cuadrático medio 1,00, es decir, un factor típico de 2,7. Siria: 2,1 M de refugiados simulados frente a 6,1 M observados en 2024; desplazados internos 5,1 M frente a 7,4 M.

**Fuera de muestra** (orígenes de enero de 2012, 2015, 2018, 2021 y 2023; stocks de fin de año; error absoluto medio, menor es mejor):

| Stock | Años | N | Persistencia | Modelo con las muertes observadas | Pronóstico completo (p50) | Cobertura p10-p90 |
|---|---|---|---|---|---|---|
| Refugiados y solicitantes | 1 | 75 | **0,135** | 0,230 | 0,213 | 0,31 |
| Refugiados y solicitantes | 2 | 75 | **0,228** | 0,361 | 0,368 | 0,33 |
| Refugiados y solicitantes | 3 | 75 | **0,272** | 0,441 | 0,459 | 0,43 |
| Desplazados internos | 1 | 31 | **0,612** | 0,722 | 0,873 | 0,16 |
| Desplazados internos | 2 | 31 | **0,949** | 1,090 | 1,413 | 0,16 |
| Desplazados internos | 3 | 32 | **1,000** | 1,111 | 1,480 | 0,19 |

- **El módulo pierde contra «el stock no cambia» en todos los horizontes**, incluso cuando se le dan las muertes que de verdad ocurrieron. El fallo está en la respuesta (cuánta gente huye por muerte), no en el pronóstico de la violencia.
- **Las bandas son demasiado estrechas:** cubren entre el 16 % y el 43 % de los casos y deberían cubrir el 80 %. Sólo llevan la incertidumbre de la violencia simulada, no la de los parámetros.
- **Dónde sí da señal:** en los arranques. Siria desde enero de 2012: persistencia 0,6 M de desplazados internos, modelo con muertes observadas 1,5 M, observado 3,0 M. Yemen desde 2015: persistencia 0,33 M, modelo 0,53 M, observado 2,53 M. Se mueve en la dirección correcta y se queda muy corto.
- **Uso que se le puede dar hoy:** ninguno como pronóstico. Sirve como contabilidad coherente de escenarios («si la violencia fuera ésta, el orden de magnitud sería éste»).

**Límites declarados:**
- sólo desplazamiento por violencia organizada: ni económico ni climático;
- UCDP no cuenta muertes indirectas (hambre, enfermedad), que en Yemen son la mayoría;
- los desplazados internos de ACNUR empiezan cuando ACNUR empieza a operar en el país;
- los refugiados palestinos bajo mandato de UNRWA no están;
- el reparto por celda no tiene dato contra el que medirse [M].

## 8. Lo que apareció fuera del L2

- **El Atlas no se podía instalar fuera de Windows.** Su manifiesto guarda el SHA-256 de un checkout con `core.autocrlf`: en Linux los mismos 33 ficheros de texto tienen LF y otro hash, y `--install-atlas` daba MISMATCH. `atlas.matches` acepta ahora el fichero byte a byte o salvo los finales de línea (sólo `.csv` y `.json`); un valor cambiado sigue sin pasar (prueba con mutante).
- **W9a y W8 se reprodujeron en Linux** con `--build-store`: Brier de conflicto 0,0851 (World4), 0,1012 (tasa base) y 0,0762 (base condicionada), idénticos a los de `F05-informe.md`.
- **Siete pruebas fallan, en vez de saltarse, cuando faltan los datos de F3-F5** (`test_api_cli_honesty.py`, cinco; `test_f6_l0r.py`, dos). Desde esta red no se pueden instalar: Zenodo, FAOSTAT y el Met Office no están permitidos. No se tocaron.
- **Los procesos del `Pool` consumían los cuatro núcleos cada uno** por los hilos de BLAS: las corridas de W13 van con `OMP_NUM_THREADS=1`.

## 9. ViEWS

W13 pide informar de ViEWS y no se pudo: sus pronósticos se sirven desde `api.viewsforecasting.org` y `data.viewsforecasting.org`, que la red de este entorno no permite (sí permite `viewsforecasting.org`, que sólo es la web). W13 no puede pasar de ámbar sin esa comparación.

## 10. Cómo reproducir

```bash
pip install -e .[test,l2]
python world4_cli.py --install-l2                          # ~110 MB, SHA-256 comprobado
python -m pytest tests/test_f12_l2.py -q                   # 27 pruebas
OMP_NUM_THREADS=1 python world4_cli.py --l2-validate --l2-period dev    # ~80 s
OMP_NUM_THREADS=1 python world4_cli.py --l2-validate                    # prueba + desplazamiento
OMP_NUM_THREADS=1 python world4_cli.py --l2-validate --l1               # ablación (necesita --build-store)
python world4_cli.py --l2-forecast --horizon 12            # desde el último mes con datos
python world4_cli.py --campaign                            # W13 con el resto
```

## 11. Qué sigue

En orden de lo que ataca las causas medidas (§12 refuerza el primer punto: a escala mundial es, con diferencia, el mayor):

1. **Que el simulador no itere a ciegas.** Su error crece con el horizonte más que el de `hurdle`. Dos remedios: calibrar activación e intensidad contra el error a varios pasos (ABC-SMC de F7 sobre estadísticos resumen), o condicionar el total mensual por país a `hurdle` y dejar al simulador el reparto entre díadas, celdas y desplazamiento.
2. **ViEWS como referencia** (§9): hacen falta dos hosts.
3. **Cablear el puente L2 → L1** en el estimador estructural para los países de la región, con su propia medición en W9a.
4. **Desplazamiento:** una respuesta por tipo de violencia y datos mensuales de ACNUR; hoy sólo hay stocks anuales.
5. **Acoplar las regiones** (hecha la ampliación de §12): hoy un actor no ve sus frentes en otra región.
6. **Eventos que UCDP no registra** (protesta, represión): hace falta otra fuente, y ACLED tiene restricciones de uso que hay que abrir antes.

## 12. Ampliación mundial (2026-11-17)

### Qué cambió

- **Regiones.** `africa`, `americas`, `asia` (con Oceanía), `europe` (con Rusia y el Cáucaso) y `mideast`: las cinco de UCDP. `world` no es una región, es la lista de las cinco. Un país con eventos va donde lo pone UCDP; uno sin eventos, por su continente; `mideast` sigue siendo una lista fija.
- **Los mismos ficheros.** El GED, PRIO-GRID, ACNUR y WPP ya eran mundiales: no hay descargas nuevas ni cambia el manifiesto. Los códigos de país salen de `country_converter` al convertir, con cuatro excepciones declaradas (Serbia 345, Yemen 678, Vietnam 816, Chequia 316).
- **Unidades que suman territorios** (así los codifican UCDP y PRIO-GRID): ISR con Palestina, MAR con el Sáhara Occidental, CHN con Hong Kong y Macao.
- **El simulador, por lotes.** Los miembros se simulan de 8 en 8, cada lote con su generador: el resultado no depende de la memoria de la máquina, y los 8 primeros miembros de una corrida de 64 son la corrida de 8 (prueba).
- **Muestreo de sitios por bloques.** Antes una multinomial densa por díada; ahora un índice por evento. La distribución es la misma (prueba contra las frecuencias esperadas); lo que cambia son los sorteos.
- **Memoria.** Rasgos sólo en las columnas que se usan, series en `float32`, exposición del desplazamiento precalculada y sus parámetros guardados en disco.
- **El modelo no se tocó.** La forma se eligió con Oriente Medio 2005-2014. Para las otras cuatro regiones no hubo periodo de desarrollo: todo lo que sigue es prueba.

### El panel

| Región | Países | Celdas | Eventos | Muertes | Díadas | Actores |
|---|---|---|---|---|---|---|
| África | 53 | 10 286 | 77 627 | 2 171 234 | 1 092 | 1 012 |
| Américas | 35 | 20 683 | 58 048 | 252 803 | 262 | 252 |
| Asia y Oceanía | 44 | 14 704 | 102 315 | 605 255 | 354 | 332 |
| Europa | 44 | 16 404 | 60 745 | 562 660 | 86 | 91 |
| Oriente Medio | 15 | 2 782 | 130 157 | 755 320 | 265 | 239 |
| **Mundo** | **191** | **64 859** | **428 892** | **4 347 272** | **2 059** | **1 926** |

El panel de Oriente Medio es idéntico, array por array, al de la primera medición. Las celdas suman 41 más que las 64 818 de PRIO-GRID: las de frontera con eventos de dos regiones están en ambas.

### W13 mundial, prueba 2015-2024 (CRPS de ln(1 + muertes); 20 orígenes por región)

| Nivel | Meses | N | Agentes | `hurdle` | `last12` | `zero` | Agentes − `last12` | Agentes − `hurdle` |
|---|---|---|---|---|---|---|---|---|
| País | 1 | 3 820 | 0,1550 | 0,1579 | **0,1541** | 0,759 | +0,001 [−0,007, +0,009] | −0,003 [−0,009, +0,003] |
| País | 3 | 3 820 | 0,1814 | 0,1838 | **0,1726** | 0,761 | +0,009 [+0,002, +0,016] | −0,002 [−0,009, +0,005] |
| País | 6 | 3 820 | 0,1933 | 0,1923 | **0,1826** | 0,748 | +0,011 [+0,002, +0,019] | +0,001 [−0,007, +0,010] |
| País | 12 | 3 820 | 0,2226 | 0,2124 | **0,2041** | 0,755 | +0,019 [+0,006, +0,033] | +0,010 [−0,002, +0,026] |
| País | 24 | 3 629 | 0,3113 | 0,2525 | **0,2384** | 0,763 | +0,073 [+0,038, +0,115] | +0,059 [+0,029, +0,098] |
| País | 36 | 3 247 | 0,4452 | 0,2863 | **0,2800** | 0,770 | +0,165 [+0,089, +0,253] | +0,159 [+0,087, +0,244] |
| Celda | 1 | 1 297 180 | 0,0094 | **0,0088** | 0,0089 | 0,0134 | +0,0005 [+0,0001, +0,0012] | +0,0006 [+0,0002, +0,0014] |
| Celda | 3 | 1 297 180 | 0,0103 | 0,0095 | **0,0094** | 0,0133 | +0,0009 [+0,0003, +0,0021] | +0,0009 [+0,0003, +0,0020] |
| Celda | 6 | 1 297 180 | 0,0111 | 0,0095 | 0,0095 | 0,0129 | +0,0016 [+0,0006, +0,0033] | +0,0015 [+0,0006, +0,0033] |
| Celda | 12 | 1 297 180 | 0,0132 | **0,0102** | 0,0103 | 0,0130 | +0,0029 [+0,0012, +0,0061] | +0,0030 [+0,0013, +0,0063] |

### Por región (agentes / mejor referencia; entre paréntesis, cuál)

| Región | País, 1 mes | País, 12 meses | País, 36 meses | Celda, 1 mes | Celda, 12 meses |
|---|---|---|---|---|---|
| África | **0,249** / 0,252 (`last12`) | 0,373 / 0,334 (`last12`) | 0,931 / 0,414 (`last12`) | 0,0219 / 0,0211 (`hurdle`) | 0,0321 / 0,0239 (`hurdle`) |
| Américas | 0,138 / 0,134 (`last12`) | 0,177 / 0,156 (`last12`) | 0,255 / 0,217 (`last12`) | 0,0051 / 0,0045 (`last12`) | 0,0057 / 0,0049 (`last12`) |
| Asia y Oceanía | 0,103 / 0,099 (`last12`) | 0,134 / 0,115 (`last12`) | 0,215 / 0,174 (`last12`) | 0,0099 / 0,0096 (`hurdle`) | 0,0138 / 0,0107 (`hurdle`) |
| Europa | 0,053 / 0,046 (`last12`) | **0,071** / 0,077 (`last12`) | **0,115** / 0,121 (`last12`) | 0,0017 / 0,0014 (`hurdle`) | 0,0046 / 0,0026 (`last12`) |
| Oriente Medio | 0,318 / 0,305 (`hurdle`) | 0,500 / 0,457 (`hurdle`) | 0,816 / 0,634 (`hurdle`) | 0,0382 / 0,0356 (`last12`) | 0,0476 / 0,0411 (`hurdle`) |

**Lectura:**

- **A un mes y por país el simulador empata con las dos referencias** (0,155 frente a 0,154 y 0,158; los dos IC incluyen el cero). Es lo único que no pierde.
- **De 3 a 12 meses pierde contra la persistencia** con IC que excluye el cero, y empata con `hurdle`.
- **A 24 y 36 meses pierde contra todo, y mucho.** A 36 meses su error es un 59 % mayor que el de la persistencia. En África es más del doble (0,931 frente a 0,414): iterar tres años con casi 500 díadas vivas hace que la violencia simulada se dispare.
- **Por celda pierde en los cuatro horizontes,** entre un 7 % y un 30 %. A 12 meses es peor que pronosticar cero en África (0,0321 frente a 0,0272) y en Europa (0,0046 frente a 0,0036), y en el agregado mundial queda igual que cero (0,0132 frente a 0,0130).
- **Fuera de Oriente Medio la mejor referencia por país es la persistencia, no el modelo estadístico.** `hurdle` sólo gana a `last12` en Oriente Medio. Lo que en §4 parecía una vara general era una propiedad de esa región.
- **Europa es la excepción a favor:** a 12 y 36 meses el simulador sale mejor que las dos referencias (0,071 frente a 0,077 y 0,087), sin IC que lo separe. Es la región con menos díadas (20 recientes) y una guerra que domina todo.
- **Sigue acertando mejor si habrá violencia que cuánta:** el Brier de «al menos una muerte» a un mes es el mejor o empata en África (0,0552 frente a 0,0556 de `last12`), Américas (0,0430 frente a 0,0437) y Oriente Medio (0,0606 frente a 0,0642); en Asia y Europa gana la persistencia.
- **Veredicto: ROJO.** `--campaign` lee esta corrida. Nada del L2 es de grado de pronóstico en ninguna región.

### Oriente Medio, dos corridas del mismo modelo

Los números de §4 son de la primera corrida. Con el muestreo nuevo (misma distribución, otros sorteos) la corrida mundial repite la región:

| Nivel | Meses | Agentes, primera | Agentes, segunda | `hurdle` | `last12` |
|---|---|---|---|---|---|
| País | 1 | 0,3216 | 0,3181 | 0,3049 | 0,3345 |
| País | 3 | 0,3689 | 0,3692 | 0,3351 | 0,3721 |
| País | 6 | 0,4261 | 0,4296 | 0,4135 | 0,4177 |
| País | 12 | 0,5125 | 0,4999 | 0,4566 | 0,4886 |
| País | 24 | 0,6602 | 0,6527 | 0,5659 | 0,6161 |
| País | 36 | 0,8151 | 0,8155 | 0,6336 | 0,7323 |
| Celda | 1 | 0,0382 | 0,0382 | 0,0356 | 0,0356 |
| Celda | 12 | 0,0481 | 0,0476 | 0,0411 | 0,0417 |

Las referencias no se mueven en el cuarto decimal y el simulador se mueve como máximo un 2,5 % (a 12 meses): ruido de 64 miembros. El veredicto es el mismo.

### Desplazamiento mundial

Orígenes de enero de 2012, 2015, 2018, 2021 y 2023; error absoluto medio de asinh(personas / 10 000):

| Stock | Años | N | Persistencia | Modelo con las muertes observadas | Pronóstico completo (p50) | Cobertura p10-p90 |
|---|---|---|---|---|---|---|
| Refugiados y solicitantes | 1 | 955 | **0,084** | 0,138 | 0,142 | 0,25 |
| Refugiados y solicitantes | 2 | 955 | **0,159** | 0,232 | 0,251 | 0,33 |
| Refugiados y solicitantes | 3 | 955 | **0,210** | 0,290 | 0,327 | 0,38 |
| Desplazados internos | 1 | 282 | **0,319** | 0,467 | 0,481 | 0,22 |
| Desplazados internos | 2 | 296 | **0,546** | 0,671 | 0,708 | 0,28 |
| Desplazados internos | 3 | 298 | **0,723** | 0,816 | 0,948 | 0,27 |

- **Pierde contra la persistencia en el mundo como perdía en Oriente Medio,** y las bandas siguen cubriendo un tercio de lo que deberían.
- **Por región, a un año:** empata en Américas (refugiados 0,102 frente a 0,103) y en los desplazados internos de Asia (0,288 frente a 0,285); pierde en el resto, y donde más en África (refugiados 0,238 frente a 0,124).

### Límites que añade la escala mundial

- **Las regiones no se ven entre sí.** Un actor con frentes en dos regiones (el Estado Islámico, los gobiernos que intervienen fuera) no acopla unos con otros.
- **Nada está correlacionado entre regiones.** La banda de un total mundial sale demasiado estrecha: no se debe sumar.
- **El país de una celda es el de 2014** (PRIO-GRID) y su población, el reparto de 2010 escalado al total de la ONU.
- **La forma del modelo se eligió con una región.** En las otras cuatro puede haber formas mejores que no se buscaron.

### Coste (4 núcleos, `OMP_NUM_THREADS=1`)

- Conversión de las cinco regiones: 23 s.
- W13 mundial (100 corridas de origen): 26 min. Desplazamiento (25 ajustes y pronósticos): 16 min.
- Una corrida de origen de África (64 miembros, 36 meses, con las referencias): 70 s y 1,2 GB.

### Cómo reproducir

```bash
python world4_cli.py --install-l2 --l2-region world
OMP_NUM_THREADS=1 python world4_cli.py --l2-validate --l2-region world     # runs/l2_w13_world.json
python world4_cli.py --l2-forecast --l2-region world --horizon 12
```

