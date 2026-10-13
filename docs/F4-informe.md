# F4 de Phy-World (World4): clima y carbono

- **Fecha:** 2026-10-12
- **Estado:** construida y medida. D5 cerrado: la temperatura ya no sale de la TCRE instantánea del Atlas sobre su extracción, sino de **FaIR v2.2.4** con el conjunto calibrado y restringido 1.6.0, una configuración por miembro. FaIR corre año a año con las emisiones que genera el propio modelo (F1-F3). Los daños del Atlas (BHM local y desastres) ven ya la temperatura de FaIR. `api.query` usa F1-F4 (`MODEL_REVISION` 5).
  - **W3: ÁMBAR** (la salvedad de F1). La GMST mejora un 38 / 46 / 17 % frente a la del Atlas a 1 / 5 / 10 años.
  - **W4: ROJO** por la población de F1 y porque la GMST desde 1990 queda dentro de banda el 79 % de los años, rozando el listón del 80 %. Desde 2005 la GMST queda dentro el 83 % de los años y el CO₂ el 100 %.
  - **W5: VERDE** (60/60 celdas, con la capa F4).
  - **W6: ÁMBAR.** Se mide la relación de F4.
  - **W7: ÁMBAR.** El carbono cierra a 2e-16; falta el nitrógeno.
  - **W12: ÁMBAR.** La licencia de los datos de calibración de FaIR queda verificada (CC BY 4.0), lo que cierra H17.
  - W1 y W2 siguen en verde (W2 bit a bit).
- **Pruebas:** 99/99; 11 son nuevas.
- **Validación:** `world4_cli.py --validate` (49 orígenes × 8 modos más 9 *hindcasts* y el contraste con AR6; unos 35 min).

## 1. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Datos | `world4/data/climate.py`, `climate_manifest.json` | Calibración de FaIR 1.6.0 (Zenodo 18828694, CC BY 4.0): 841 configuraciones posteriores, propiedades de 62 especies, emisiones históricas CMIP7 1750-2023 y forzamientos solar, volcánico, de uso del suelo y de riego. HadCRUT5 5.1.0.0 (OGL v3) como verdad de la GMST; se fija el SHA-256 de las filas 1850-2023 porque el fichero se actualiza cada mes |
| Paso a paso | `world4/sectors/_fair_step.py` | El bucle de `FAIR.run` partido en `prepare()` + `step(i)` (derivado de FaIR, Apache-2.0, ver NOTICE). Con las mismas emisiones da **exactamente** lo mismo que `FAIR.run` (diferencia 0,0) |
| Sector | `world4/sectors/climate.py` | Montaje de la calibración, reglas de emisión, anclaje a la observación, inflexiones (apagadas), TCRE y presupuesto restante |
| Acoplamiento | `coupling.py` (`F4Config`) | Tras cada año del L1: emisiones → FaIR → GMST, que el L1 usa el año siguiente para la temperatura local (BHM), los desastres y el término climático del conflicto |
| Validación | `rolling.py` (modo `f4`, métricas `gmst` y `co2`), `hindcast.run_climate`, `run_climate_observed`, `run_carbon_budget` | W3, W5, W4, contraste con AR6 |
| CLI / API | `--install-climate`; `--query gmst`, `co2_emissions_world`, `co2_concentration` | Extra `pip install -e .[climate]` (`fair==2.2.4`) |

**El montaje es el de la calibración.** Con las emisiones históricas, el calentamiento de 2004-2023 de cada configuración coincide con el que publica la calibración (`warming_baselines.csv`) a 1e-5 °C (prueba).

**Versión de la calibración.** Se eligió la 1.6.0 («fastmip v1», marzo de 2026) porque corresponde al paquete instalado y trae todos los ficheros de una corrida histórica. La revisada por pares es la 1.4.1 (Smith et al. 2024, GMD 17, 8569), para FaIR 2.1.3; el método es el mismo. No se usó la 1.6.1 (7 de octubre de 2026): tiene cinco días y el autor avisa de que la más nueva no siempre es la adecuada.

## 2. Emisiones: de dónde salen después del origen

Todo lo que se ajusta usa datos ≤ origen (prueba de fuga). Hasta el origen, FaIR corre con las emisiones observadas; después, cada miembro tiene las suyas:

| Especie | Regla | Marca |
|---|---|---|
| CO₂ fósil e industrial | Valor del origen × F(t)/F(t0) × [(1 − s_o)·I(t) + s_o]. **F** es el consumo fósil de F3 (la cantidad). **I** es la intensidad de carbono de la mezcla de combustibles: el reparto entre petróleo, gas y carbón sale de las producciones de F2, ponderadas por su peso en el CO₂ y en la energía del origen. **s_o** (cemento, antorchas) sigue a F | [S] en s_o |
| CO₂ de uso del suelo | Media de 10 años + AR(1) con la persistencia y la escala de los 30 años previos | [S], hasta F5 |
| SO₂, BC, OC, NOx, CO, COV | Parte antropogénica (sobre las emisiones naturales de 1750) × CO₂ fósil relativo × exp(g·h); g es la tendencia de la intensidad en 20 años, acotada a [−6 %, +2 %] anual | [S] |
| N₂O, NH₃ | Ídem, frente a la población mundial (F1) | [S], hasta F5 |
| CH₄ | Ídem, frente a una mezcla de 35 % fósil y 65 % población | [S] |
| Halogenados y gases fluorados | Parte antropogénica × exp(g·h), con la tendencia de 10 años acotada a [−15 %, +5 %] | [S] |
| Solar / volcánico / uso del suelo y riego | Media del último ciclo / 0 (no se conocen erupciones futuras) / valor del origen | [S] |

**Dos correcciones que obligaron los datos:**

1. **Emisiones naturales.** La primera versión escalaba la emisión **total**. Con la caída a cero del experimento del presupuesto, las especies quedaban por debajo de su nivel natural de 1750 y la interacción aerosol-nube daba +0,47 W/m². Ahora las reglas solo mueven la parte antropogénica.
2. **F2 no reparte el fósil de F3.** Desde 1990, el fósil de F3 se multiplica por 2,07 hasta 2023 (observado ×1,69). En F2, en cambio, el petróleo cae a ×0,84, el gas a ×0,97 y el carbón sube a ×1,82, porque cada clase sigue su tendencia propia. La primera versión sacaba el CO₂ de las cantidades de F2, y acertaba en 2023 (+1,8 %) solo porque se compensaban dos errores. Ahora la cantidad es la de F3 y F2 solo da el reparto. Es coherente con la física, pero hereda el exceso de energía de F3 desde 1990 (§4). La incoherencia F2/F3 queda anotada en el informe de F3.

**Temperatura del L1.** La GMST es la respuesta forzada del miembro más la diferencia con la observación en el origen, que tiene dos partes:

- una **lenta**, la media de los 10 años previos, que se mantiene;
- una **rápida**, la del año del origen, que decae con la persistencia anual estimada (ρ ≈ 0,09).

La historia es determinista; la variabilidad interna (la estocástica calibrada de FaIR) solo entra después del origen.

## 3. W3 y W5: origen móvil (49 orígenes, R = 32)

| | Atlas (TCRE) | F1+F2+F3 | **F1-F4** | Mejor referencia |
|---|---|---|---|---|
| CRPS GMST, 1/5/10 años (°C) | 0,105 / 0,129 / 0,104 | 0,105 / 0,129 / 0,103 | **0,065 / 0,070 / 0,087** | persistencia 0,109 / 0,149; tendencia de 10 años 0,177 |
| F4/Atlas, IC95 | | | 0,62 [0,49, 0,79] / 0,54 [0,43, 0,66] / 0,83 [0,62, 1,12] | |
| Cobertura p10-p90 de la GMST | ~0 (casi sin dispersión) | ~0 | 0,41 / 0,69 / 0,62 | nominal 0,80 |
| CRPS log CO₂ fósil mundial | — | — | 0,055 / 0,097 / 0,167 | tendencia de 5 años **0,016 / 0,071 / 0,144** |
| CRPS PIB pc / Atlas | | 0,892 / 0,950 / 0,966 | 0,892 / 0,952 / 0,970 | |
| RMSE rentas / Atlas | | 0,80 / 0,91 / 0,97 | 0,80 / 0,91 / 0,97 | |

- **La GMST mejora claramente** y bate a sus referencias en los tres horizontes. A 10 años el IC incluye el 1.
- **El CO₂ mundial NO bate a la tendencia de 5 años** en ningún horizonte. El error a 1 año (~5 %) es el ruido de un año del fósil de F2/F3; el modelo no gana a extrapolar. Es la misma debilidad de F3 a corto plazo.
- **F4 casi no mueve la economía:** el PIB pc pasa de 0,966 a 0,970 a 10 años. Los daños BHM cambian poco en 10 años.
- **W5:** con la capa F4, la deuda, la democracia, la población, la energía y la `psi` no se mueven fuera de su banda (60/60 celdas en las cuatro capas).

## 4. W4: *hindcast* libre (hasta 2023)

| Origen | GMST dentro de p10-p90 | MAE de la mediana (°C) | Atlas TCRE | Tendencia de 10 años | CO₂ fósil dentro | CO₂ en 2023 (mediana) | ppm dentro (hasta 2019) |
|---|---|---|---|---|---|---|---|
| **1990** | **79 %** (listón 80 %) | 0,118 | 0,094 | 0,110 | 85 % | **+60 %** | 69 % |
| 2005 | 83 % | 0,090 | 0,095 | 0,119 | 100 % | +7 % | 100 % |
| FaIR con emisiones **observadas**, 1950-2023 (en muestra) | 86 % | 0,086 | | | | | |

- **Desde 2005 se cumple** el listón en GMST y en CO₂.
- **Desde 1990 no:** el L1 genera demasiada energía fósil (F3: +40 % de energía primaria en 2023), el CO₂ acaba en +60 % y la concentración se pasa. La GMST, sin embargo, sale **baja**, con un error de −0,52 °C en 2023: más fósil trae más SO₂ y los aerosoles enfrían a corto plazo. La causa es la de F3, no FaIR.
- **2023 se escapa siempre** (−0,3 °C desde 2005): fue un año excepcionalmente cálido y la variabilidad interna de FaIR no llega a tanto.
- **Fuga declarada:** el posterior de FaIR se restringió con la GMST observada hasta 2023, así que los parámetros del clima están en muestra. Lo que estos *hindcasts* miden fuera de muestra es la cadena economía → energía → emisiones → clima.

## 5. Coherencia con AR6

| | World4 (841 configuraciones) | AR6 |
|---|---|---|
| TCRE (1pctCO₂, °C por 1000 GtCO₂), p17 / p50 / p83 | 0,32 / **0,39** / 0,47 | probable 0,27-0,63; mejor estimación 0,45 |
| Presupuesto para 1,5 °C desde 2020, 50 % | **agotado**: el 83 % de las configuraciones pasa de 1,5 °C incluso con la caída más rápida, 10 años y unos 210 GtCO₂ | 500 GtCO₂ |
| Presupuesto para 2 °C desde 2020, 50 % | **465 GtCO₂** (17 %: 1800) | 1350 GtCO₂ |
| Ídem, con el no-CO₂ «como AR6» (CH₄ −50 %, N₂O y NH₃ −20 % al llegar a cero) [S] | 1,5 °C: agotado (72 %); 2 °C: 855 | |

- **La TCRE es coherente con AR6:** la mediana cae dentro del rango probable, algo por debajo de la mejor estimación.
- **El presupuesto restante NO es coherente: sale mucho menor.** La causa medida son los forzamientos distintos del CO₂ con las reglas de F4. Al bajar el CO₂ fósil caen también el SO₂ y el resto de aerosoles, que hoy enfrían en torno a −1 W/m² en este conjunto, y su desaparición calienta de golpe (+0,3-0,4 °C en una década). Además, el N₂O y el 65 % del CH₄ siguen a la población. Con recortes de no-CO₂ como los de AR6 el presupuesto para 2 °C sube de 465 a 855 GtCO₂, pero sigue lejos de 1350.
- Las actualizaciones de los presupuestos posteriores a AR6 (Indicators of Global Climate Change, del mismo equipo que la calibración) también los rebajaron [S, sin verificar las cifras], pero no tanto. **Criterio de salida de F4 «presupuesto coherente con AR6 dentro de banda»: NO se cumple.** Queda como rojo con causa nombrada dentro de F4; no hay un W que lo puntúe.

## 6. W6 y W7

- **W6 (relación de F4).** Con el fósil a ×0,5 / ×1 / ×1,5 desde 2019, la GMST de 2079 sale 2,52 / 2,91 / 3,28 °C (mediana), monótona en todos los miembros. **A 10 años la relación se invierte:** más fósil da una mediana **más fría** (enmascaramiento por aerosoles); a 30 años ya es más cálida en la mediana. Las pruebas lo fijan así: es física, no un fallo.
- **W7 (carbono).** Lo emitido es igual a lo que queda en la atmósfera más lo absorbido; la carga atmosférica es la suma de las cuatro cajas, y el exceso de ppm es la carga por ppm/Gt. Cierra con un error máximo de 2,3e-16.

## 7. Corrida exploratoria 2019 → 2100 (no es un pronóstico)

F1-F4, R = 8:

| Año | GMST p10 / p50 / p90 (°C) | CO₂ fósil (Gt/año, mediana) | ppm | Fósil en la energía | Población (miles de millones) |
|---|---|---|---|---|---|
| 2030 | 1,41 / 1,48 / 1,69 | 42 | 443 | 81 % | 8,3 |
| 2050 | 1,90 / 2,20 / 2,44 | 47 | 500 | 71 % | 9,1 |
| 2100 | 3,42 / **4,37** / 5,92 | 67 | 696 | 52 % | 7,6 |

- **Sin política climática** (las palancas llegan en F8), con la transición lenta de F3 y el PIB pc mundial creciendo de 16.500 a 40.000 USD, el modelo da +4,4 °C en 2100.
- **Los daños BHM del Atlas apenas frenan el PIB.** Ese es el supuesto de la función de daño de BHM, con su debate fuera de muestra (Barker 2024).
- **Con los elementos de inflexión encendidos:** +0,04 °C en 2100. Sus impactos son ilustrativos [M] y no deben leerse.

## 8. Desviaciones y lo que no está

- **«GMST 1950-2023 en *hindcast* libre»:** el L1 no puede arrancar en 1950 (no hay datos para estimar antes). Se mide FaIR con las emisiones observadas 1950-2023 (en muestra) y el modelo completo desde 1990 y 2005.
- **Gases distintos del CO₂ «por sector»:** no hay sectores; se usan reglas por especie ligadas a la energía, a la población o a la tendencia [S]. Agricultura y uso del suelo esperan a F5.
- **Elementos de inflexión:**
  - umbrales de Armstrong McKay et al. (2022) como triangulares [S]: deshielo abrupto del permafrost 1,0 / 1,5 / 2,3 °C y muerte regresiva de la Amazonía 2,0 / 3,5 / 6,0 °C;
  - impactos ilustrativos (1,0 y 0,9 GtCO₂/año; 10 MtCH₄/año) [M, sin verificar ni elicitar];
  - apagados por defecto, exploratorios y separables por ablación;
  - no hay probabilidades elicitadas.
- **Kotz et al. (2024)** no se usa (retractado). Los daños son los del Atlas (BHM y desastres), sin cambios.
- **Mortalidad por desastres o calor extremo en F1:** el calor ya entraba en F1; los desastres siguen fuera.
- **Variabilidad interna:** la cobertura de la GMST es del 41 % a 1 año. El ensemble no captura toda la variabilidad interanual (El Niño), lo que apunta a la recalibración (D).

## 9. Qué sigue

1. **F5 (alimentos, suelo, agua, N):** CO₂ de uso del suelo, CH₄ y N₂O agrícolas endógenos, y el balance de N de W7.
2. **Coherencia F2/F3:** que los mercados de F2 repartan el fósil de F3. Arregla el CO₂ desde 1990 y quizá la GMST de W4; obliga a revalidar F2 y F3.
3. **Presupuesto de carbono:** revisar la regla del SO₂ (que hoy cae al mismo ritmo que el CO₂ fósil) frente a trayectorias publicadas, antes de dar presupuestos.
4. **La fecundidad de F1** sigue siendo el rojo de W4.
