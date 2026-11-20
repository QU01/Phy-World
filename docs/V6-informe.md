# V6 de Phy-World (World4): World3 vigente — los recursos no renovables acotados con datos

- **Fecha:** 2026-11-17
- **Estado:** hecho el primer punto de «Qué sigue» de V5: el número que decidía el máximo industrial, los recursos no renovables iniciales, ya no es un rango citado de memoria. Se acota con dos fuentes descargables y pasa de ser un depósito que se vacía a ser una curva de coste de extracción.
  - **Lo que dicen los datos:** las reservas probadas de combustibles fósiles más lo ya quemado suman 4.310 GtCO₂; el stock de World3, en la misma moneda, son 4.296. El stock de World3 es lo que ya está probado: un suelo, no un valor central. Todo lo que hay bajo tierra, según las curvas de oferta de GCAM, son 47.588 GtCO₂, once veces más, a coste creciente.
  - **Resultado principal:** con la curva, el 90 % de los modelos que la historia no descarta no tiene máximo industrial antes de 2100 (en V5 la mediana era 2071). El agotamiento de recursos deja de ser lo que frena al modelo en este siglo.
  - **El límite cambia de sitio:** pasa a la contaminación persistente y la comida. La esperanza de vida cae antes de 2100 en el 66 % de los miembros. Ese límite descansa en tablas de World3 que ningún dato fija (sección 5).
  - **Lo que no se mueve:** el máximo de población, entre 2055 y 2065, con 9.200 a 10.000 millones.
  - **La historia no distingue entre stock y curva:** el ajuste de 1971-2023 y las pruebas fuera de muestra salen iguales o algo mejor con la curva.
- **El escenario 1 no cambia.** W1 sigue verde. Sin `resources`, todo lo de V1-V4 da lo mismo que antes (prueba).
- **Pruebas:** 147 pasan (11 nuevas, todas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:**
  - `python world4_cli.py --install-world` (descarga además las curvas de GCAM y las reservas de BP);
  - `python world4_cli.py --world3-score` (el marcador con la columna nueva y las pruebas fuera de muestra con la curva);
  - `python world4_cli.py --world3-bands --members 100` (los números de las secciones 4 y 5);
  - `python -m world4.core.l0v.technology` y `python -m world4.core.l0v.ensemble` (recalculan lo guardado).

## 1. Los datos

| Fuente | Qué da | Marca |
|---|---|---|
| Curvas de oferta de recursos fósiles de GCAM (`A11.fos_curves.csv`, JGCRI/gcam-core, derivadas de Rogner 1997) | Por combustible y grado: energía disponible (EJ) y coste de extracción (dólares de 1975 por GJ) | [V], ECL-2.0 |
| Contenido de carbono por combustible de GCAM (`A_PrimaryFuelCCoef.csv`) | 27,3 / 19,6 / 14,2 kg de carbono por GJ de carbón, petróleo y gas | [V] |
| Relación reservas/producción de BP (2016) y producción fósil (BP y Shift), vía OWID | Reservas probadas de 2015 = años de reservas × producción | [S] |
| Emisiones acumuladas por combustible (Global Carbon Budget, vía OWID; ya instalado) | Cuánto se ha extraído de cada combustible, en CO₂ | [V] |

No se versiona ningún dato de terceros: se instalan en `data_cache/world/raw` con huella SHA-256, como el resto.

**Reservas probadas frente al stock de World3.** World3 no tiene unidades de recurso; se le ponen con la misma equivalencia que las emisiones de V4 (las emisiones reales de 1970 por el uso de recursos del modelo en 1970).

| GtCO₂ | Carbón | Petróleo | Gas | Total |
|---|---|---|---|---|
| Emitido hasta 2015 | 715 | 529 | 208 | 1.452 |
| Reservas probadas en 2015 | 1.846 | 664 | 347 | 2.857 |
| Emitido + reservas | 2.561 | 1.193 | 555 | **4.310** |
| Stock inicial de World3 (1900) | | | | **4.296** |
| Curva de GCAM, todos los grados | 26.742 | 8.066 | 12.780 | **47.588** |

- **La coincidencia entre 4.310 y 4.296 es del 0,3 %.** No es un ajuste: el stock de World3 es de 1972 y la equivalencia en carbono viene de V4.
- **Lo que significa:** el escenario 1 de World3 supone que no se encontrará ni se hará extraíble nada más de lo que hoy está probado. Las reservas probadas se reponen con recursos que antes no lo eran (por eso los años de reservas no bajan; citado de memoria, sin serie descargada).
- **Las reservas caben en la curva:** lo probado de cada combustible está en el grado de coste actual de la curva o en el siguiente (prueba).

**La curva de coste** (GtCO₂ acumuladas al final de cada grado; coste en dólares de 1975 por GJ):

| Combustible | Grados |
|---|---|
| Carbón | 115 a 0,34 · 3.184 a 0,37 · 6.914 a 1,2 · 12.215 a 1,7 · 19.478 a 2,0 · 26.742 a 2,3 |
| Petróleo | 140 a 0,50 · 289 a 0,55 · 531 a 0,92 · 777 a 1,3 · 2.676 a 1,8 · 2.929 a 2,1 · 3.344 a 2,7 · 7.056 a 3,3 · 8.066 a 10,8 |
| Gas | 44 a 0,62 · 325 a 0,68 · 569 a 1,1 · 1.024 a 1,7 · 1.315 a 3,1 · 1.859 a 3,7 · 12.780 a 6,5 |

Dónde está la historia sobre la curva:

| Año | Emitido (GtCO₂) | Carbón | Petróleo | Gas | Coste medio | Capital para obtener recursos |
|---|---|---|---|---|---|---|
| 1970 | 418 | 0,37 | 0,50 | 0,62 | 0,50 | 5,0 % |
| 1990 | 782 | 0,37 | 0,55 | 0,68 | 0,53 | 5,4 % |
| 2000 | 1.008 | 0,37 | 0,92 | 0,68 | 0,67 | 6,7 % |
| 2023 | 1.726 | 0,37 | 1,30 | 0,68 | 0,81 | 8,1 % |

- **El carbón sigue en su grado barato** (833 de 3.184 GtCO₂). El petróleo ha subido tres grados desde 1970; el gas, uno.
- El coste medio pesa cada combustible por su parte de la energía de los últimos diez años (carbón 32 %, petróleo 36 %, gas 31 %).

## 2. Cómo entra en el modelo

World3 tiene un stock (`nri`) y una tabla (`fcaor`) que dice qué parte del capital industrial hace falta para obtener recursos según la fracción que queda: 5 % mientras sobra, 50 % cuando queda el 30 %, 90 % cuando queda el 10 %. V6 cambia las dos cosas (`world4/core/l0v/resources.py`, `model.run(resources="estimated")`):

- **El stock** es el total de la curva, 47.588 GtCO₂, colocado de modo que en 1970 al modelo le quede lo mismo que a la curva. El acumulado del modelo sigue al real: 1.742 GtCO₂ en 2023 frente a 1.726 observadas.
- **La tabla** es `5 % × coste medio de extracción / coste medio de 1970`, con tope en 95 %. El 5 % es el valor de World3 con recursos abundantes, que es lo que había en 1970.
- **Después del último dato** el coste depende de dos cosas que la historia no puede fijar, y son los dos ejes nuevos del conjunto:
  - `substitution`, de 0 a 1. Con 0 cada combustible mantiene su cuota de los últimos diez años; con 1 se extrae siempre el grado más barato que quede de cualquiera (hoy, carbón).
  - `resource_scale`, de 0,5 a 2 [S]: lo que queda por extraer de cada grado, de la mitad al doble de lo que dice la curva, que es una evaluación de 1997 y no una medida.
- **Sin fuga de información:** una corrida con corte en un año solo ve la mezcla de combustibles observada hasta ese año (`Resources.cutoff`).

Capital para obtener recursos según lo emitido (sin sustitución / central / con sustitución):

| Emitido (GtCO₂) | 2.500 | 4.000 | 6.000 | 10.000 |
|---|---|---|---|---|
| V6 | 11 % / 10 % / 8 % | 13 % / 11 % / 8 % | 18 % / 15 % / 12 % | 37 % / 27 % / 17 % |
| World3 (stock de 4.296, su tabla) | 18 % | 93 % | agotado | agotado |

## 3. La historia no distingue entre stock y curva

La tecnología se volvió a estimar con la curva (`l0v_technology.json`, bloque `by_cutoff_curve`; el bloque de V3 se conserva).

| Corte | Pérdida con el stock | Pérdida con la curva | `kappa` con el stock | `kappa` con la curva |
|---|---|---|---|---|
| 1990 | 0,00690 | 0,00672 | 0,02 | −0,02 |
| 2000 | 0,00731 | 0,00719 | −0,12 | −0,23 |
| 2010 | 0,00897 | 0,00848 | −0,30 | −0,39 |
| 2023 | 0,01192 | 0,01130 | −0,32 | −0,40 |

- **El ajuste es algo mejor con la curva en los cuatro cortes,** pero la diferencia es pequeña.
- **Lo absorbe una sola perilla:** el coste de capital (`kappa`) baja para compensar el capital que ahora se va en recursos. Las tasas de rendimiento, de uso de recursos y de contaminación no se mueven (menos del 5 %).
- **Consecuencia:** la historia no puede decir cuál de las dos descripciones es la buena. Lo que decide entre ellas son los datos de la sección 1, no el ajuste.

El marcador (error medio de 2000-2023, datos hasta 2000) y las pruebas fuera de muestra:

| Indicador | World3-03 | V5 (stock) | V6 (curva) |
|---|---|---|---|
| Población | 4,7 % | 1,0 % | 1,0 % |
| Esperanza de vida | 8,4 % | 0,7 % | 0,8 % |
| Producción | 26,7 % | 6,9 % | 5,6 % |
| Alimento por persona | 14,1 % | 2,6 % | 2,3 % |
| Uso de recursos | 106,4 % | 13,7 % | 12,9 % |
| CO₂ atmosférico sobre 280 ppm | 32,7 % | 2,6 % | 2,5 % |

| Origen (datos hasta ese año, error hasta 2023) | Producción | Alimento | Uso de recursos | Población |
|---|---|---|---|---|
| 1990, stock / curva | 2,4 / 2,4 | 14,7 / 14,6 | 11,5 / 11,6 | 2,2 / 2,2 |
| 2000, stock / curva | 7,6 / 6,2 | 2,8 / 2,5 | 14,2 / 13,4 | 1,0 / 1,0 |
| 2010, stock / curva | 2,4 / 2,4 | 9,3 / 9,3 | 13,4 / 12,6 | 0,3 / 0,3 |

## 4. Lo que cambia hasta 2100

**La corrida central** desde 2023:

| | Stock de World3 | Curva de coste |
|---|---|---|
| Máximo de producción industrial por persona | 2047 | no llega antes de 2100 |
| Producción industrial por persona en 2100, sobre la de 2023 | 0,24 | 9,2 |
| Capital para obtener recursos en 2100 | 80 % | 13 % |
| Máximo de población | 2060, 9.573 millones | 2060, 9.558 millones |
| Población en 2100 (millones) | 7.889 | 7.579 |
| Contaminación persistente en 2100 (1970 = 1) | 2,5 | 39,7 |
| Alimento por persona en 2100 (kg; subsistencia: 230) | 484 | 264 |
| Esperanza de vida: máximo y valor en 2100 | 84,3 y 84,1 | 84,2 (en 2080) y 74,8 |
| Temperatura en 2100 sobre 1951-1980 | 2,2 °C | 2,8 °C |

- **Es el paso del escenario 1 al escenario 2 del libro:** con más recursos, la industria no se frena y la crisis llega por la contaminación, que destruye la fertilidad de la tierra. Aquí ocurre unas décadas más tarde que en el libro porque la tecnología estimada es mejor que la de World3.

**El conjunto** (174 modelos de 48.000 pasan el descarte con datos hasta 2023; 100 miembros; p10 / p50 / p90):

| | p10 | p50 | p90 |
|---|---|---|---|
| Año del máximo de producción industrial por persona | 2100 o después | 2100 o después | 2100 o después |
| Año del máximo de población | 2055 | 2059 | 2065 |
| Población máxima (millones) | 9.175 | 9.491 | 9.957 |
| Año del máximo de alimento por persona | 2047 | 2072 | 2096 |
| Año del máximo de esperanza de vida | 2070 | 2080 | 2100 |

| En 2050 | p10 | p50 | p90 |
|---|---|---|---|
| Población (millones) | 9.143 | 9.399 | 9.700 |
| Esperanza de vida (años) | 78,1 | 80,2 | 82,6 |
| Producción industrial por persona, sobre la de 2023 | 1,74 | 2,00 | 2,28 |
| Capital para obtener recursos | 8 % | 10 % | 13 % |
| Contaminación persistente (1970 = 1) | 5,3 | 6,2 | 7,4 |
| Temperatura sobre 1951-1980 (°C) | 1,23 | 1,58 | 1,94 |

| En 2100 | p10 | p50 | p90 |
|---|---|---|---|
| Población (millones) | 5.951 | 7.491 | 8.571 |
| Esperanza de vida (años) | 37,5 | 76,3 | 88,9 |
| Hijos por mujer | 1,00 | 1,27 | 2,21 |
| Alimento por persona (kg; subsistencia: 230) | 185 | 398 | 750 |
| Producción industrial por persona, sobre la de 2023 | 4,9 | 7,7 | 11,5 |
| Capital para obtener recursos | 10 % | 13 % | 21 % |
| Contaminación persistente (1970 = 1) | 9,8 | 34,3 | 77,8 |
| CO₂ atmosférico (ppm) | 542 | 617 | 735 |
| Temperatura sobre 1951-1980 (°C) | 1,73 | 2,96 | 4,09 |

- **El máximo industrial sale del siglo.** En 90 de 100 miembros no llega antes de 2100; en los otros 10 cae entre 2091 y 2099. En V5 el p10 era 2037.
- **Los dos ejes de la curva no lo cambian:** ni sin sustitución ni con la mitad de lo evaluado el capital para recursos pasa del 21 % (p90) en 2100.
- **Hasta 2050 las bandas son estrechas** y casi iguales a las de V5, salvo la producción industrial, que ya no tiene cola baja (p10 de 1,74 frente a 0,81).
- **La población es lo más firme, otra vez:** mismo máximo y misma ventana que en V5.
- **La segunda mitad del siglo es peor que en V5:** la esperanza de vida de 2100 baja de 65 años en el 33 % de los miembros (14 % en V5) y el alimento cae por debajo de la subsistencia en el 21 % (5 % en V5).

## 5. El límite que queda y lo poco que se sabe de él

- **La contaminación persistente pasa de 20 veces la de 1970 en el 71 % de los miembros** en 2100. En World3 eso reduce la fertilidad de la tierra y la esperanza de vida.
- **Ningún eje lo decide solo.** Correlación de rangos con la esperanza de vida de 2100: freno a la expansión de tierra −0,50, ritmo de mejora del rendimiento +0,45, suelo de contaminación −0,30, techo del rendimiento −0,29. En V5 un solo eje tenía 0,92.
- **Más recursos baratos es peor, no mejor:** con sustitución alta (más de 0,67) la esperanza de vida mediana de 2100 es 74,9 años; con baja (menos de 0,33), 81,1. El coste de extracción era el freno que quedaba a la industria.
- **Este límite no está estimado.** La «contaminación persistente» de World3 es un índice abstracto. V3 calibró cuánta se genera (contra las emisiones fósiles), pero el daño que hace a la tierra y a la salud son tablas de 1972 sin dato detrás. Las bandas de la segunda mitad del siglo dependen de ellas.
- **El clima, que sí está estimado, no es lo que hunde la comida:** el daño declarado es del 5 % del rendimiento por grado, y el calentamiento desde 2023 es de 1,8 °C en la corrida central.

## 6. La prueba de las bandas

Con datos hasta 2000, fracción de 2001-2023 dentro de p10-p90 (100 miembros; con el error del modelo / solo parámetros):

| Indicador | V5 | V6 |
|---|---|---|
| Población | 1,00 / 0,04 | 1,00 / 0,09 |
| Esperanza de vida | 1,00 / 0,83 | 1,00 / 0,83 |
| Mortalidad | 0,91 / 0,70 | 0,91 / 0,70 |
| Producción | 1,00 / 0,33 | 1,00 / 1,00 |
| Tierra cultivable | 1,00 / 0,38 | 1,00 / 1,00 |
| Uso de recursos | 0,91 / 0,65 | 1,00 / 0,70 |
| CO₂ atmosférico | 0,96 / 0,61 | 0,96 / 0,91 |
| Temperatura | 0,74 / 0,30 | 0,74 / 0,30 |

- **Siguen anchas de más** (entre el 91 % y el 100 % en 11 indicadores) y la de temperatura sigue corta (74 %). V6 no arregla eso.
- **Solo con los parámetros, cubren mejor que en V5** la producción, la tierra y el CO₂.

## 7. Desviaciones y lo que no está

- **Solo combustibles fósiles.** El stock de World3 incluye metales y minerales; aquí no hay dato de ellos. Si alguno escasea antes, este modelo no lo ve.
- **La curva es una evaluación de 1997** (Rogner, vía GCAM 3.0) en dólares de 1975. Los grados no convencionales son especulativos: el último del gas, a 6,5 dólares por GJ, son 10.900 GtCO₂, el 85 % de todo el gas. Por eso el eje `resource_scale`, cuyo rango (de la mitad al doble) es declarado [S].
- **El 5 % de 1970 es el valor base de World3,** no un dato. La proporcionalidad entre coste de extracción y capital necesario es un supuesto.
- **No se descuenta la menor intensidad:** la tecnología estimada hace que cada unidad de producción use menos recursos, y la tabla de World3 no lo tiene en cuenta. Esto exagera el capital para recursos; corregirlo alejaría aún más el límite.
- **Pesos fijos de combustible en el coste medio** (los de los últimos diez años antes del corte).
- **Renovables y nuclear no están.** No hay una energía que sustituya a los fósiles; `substitution` solo mueve la mezcla entre carbón, petróleo y gas.
- **Las reservas de BP son de 2015** (la serie de OWID descargable no llega más lejos). Son una comprobación, no una entrada del modelo.
- **El conjunto guardado de V5 se sustituye por el de V6.** Los números de `docs/V5-informe.md` son los del commit de V5; con `resources=None` y el bloque `by_cutoff` se reproduce una corrida suelta de V5, no su conjunto.
- **No hay criterio nuevo en la campaña.**

## 8. Qué sigue

1. **Estimar la contaminación persistente,** que es ahora lo que decide la segunda mitad del siglo. Hace falta decidir qué es (candidatos con dato: nitrógeno reactivo, plásticos, partículas finas y su mortalidad) y sustituir las tablas de daño de World3 por relaciones estimadas.
2. **Una energía no fósil,** con su coste y su ritmo de despliegue observados: sin ella el modelo solo puede crecer quemando más.
3. **Fase 6: la prueba fuera de muestra como criterio de la campaña.**
