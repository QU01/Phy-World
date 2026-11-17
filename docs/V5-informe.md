# V5 de Phy-World (World4): World3 vigente — bandas de incertidumbre

- **Fecha:** 2026-11-17
- **Estado:** construida y medida la fase 5 del plan: en lugar de una trayectoria, el conjunto de modelos que la historia no descarta, con el error del modelo sumado y una prueba de cobertura.
  - **Resultado principal:** el máximo de la producción industrial por persona cae en 2037 / 2071 / después de 2100 (p10 / p50 / p90). Lo decide un solo número, los recursos no renovables iniciales, que la historia casi no fija.
  - **Lo que sí es firme:** la población toca techo entre 2055 y 2066, con 9.200 a 10.000 millones, en todos los miembros.
  - **La prueba de las bandas:** con datos hasta 2000, las bandas cubren lo ocurrido en 2001-2023 en 11 de 12 indicadores, pero son anchas de más (cubren entre el 91 % y el 100 %, cuando deberían rondar el 80 %). La de temperatura se queda corta (74 %).
  - **Nota posterior (V6):** el eje de los recursos iniciales se sustituyó por una curva de coste acotada con datos, y el conjunto guardado es ahora el de V6; los números de este informe son los del commit de V5 y el resultado principal cambia (ver `docs/V6-informe.md`).
- **El escenario 1 no cambia.** W1 sigue verde.
- **Pruebas:** 136 pasan (12 nuevas, 6 de ellas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:**
  - `python world4_cli.py --world3-bands --members 100` (los números de este informe, unos 70 s; por defecto 32 miembros, 22 s);
  - `python -m world4.core.l0v.ensemble` (reconstruye el conjunto guardado, unos 3 min).

## 1. Cómo se construyen las bandas

Cada miembro del conjunto es una versión distinta del modelo. Difieren en tres cosas:

| Fuente de incertidumbre | Cómo entra | Fichero |
|---|---|---|
| Las cinco perillas de tecnología (la historia las fija a medias) y cinco ejes que no fija | Muestreo de rangos declarados y descarte por la historia | `core/l0v/ensemble.py` |
| Los coeficientes de fecundidad y esperanza de vida | Cada miembro lleva una estimación con los países remuestreados | `equations.bootstrap` |
| Las rectas de CO₂ y temperatura | Cada miembro lleva una estimación con los años remuestreados en bloques de 10 | `carbon.bootstrap` |
| El error del modelo | Deriva por año de horizonte, medida en la partición anterior, sumada a cada miembro | `ensemble.discrepancy` |

Las reglas de V4 no se muestrean: son decisiones.

**El descarte.** Un miembro se queda si reproduce la historia de 1971 al corte sin ser peor que el doble del mejor ajuste en ninguno de los cinco indicadores (rendimiento, uso de recursos, generación de contaminación, producción y tierra). Es descarte, no ponderación.

- De 48.000 modelos muestreados quedan 445 con datos hasta 2000 y 116 con datos hasta 2023.
- El muestreo va por cuatro oleadas, cada una uniforme en la caja que envuelve a los que pasaron la anterior con un listón más laxo.
- Las corridas de la historia van vectorizadas, cientos de miembros a la vez, y dan lo mismo que uno a uno (prueba).

**Los cinco ejes que la historia no fija y sus rangos** [S]:

| Eje | Rango | Lo que queda tras el descarte (p5 / p50 / p95) |
|---|---|---|
| Recursos no renovables iniciales, en veces los de World3 | 0,7 – 3,0 | 0,82 / 1,46 / 2,50 |
| Suelo del uso de recursos por unidad de producción | 0,05 – 0,35 | 0,10 / 0,26 / 0,34 |
| Suelo de la contaminación por unidad de actividad | 0,05 – 0,35 | 0,09 / 0,26 / 0,34 |
| Techo de la tecnología de rendimiento | 1,6 – 3,0 | 1,69 / 2,07 / 2,79 |
| Rendimiento perdido por grado de calentamiento | 2 % – 8 % | 2,4 % / 5,2 % / 7,6 % |

- **El descarte apenas los estrecha:** es la confirmación medida de que la historia no los fija.
- **Los recursos van de 3.000 a 12.900 GtCO₂ en carbono:** de algo menos que las reservas probadas de combustibles fósiles a todos los recursos (órdenes de magnitud citados de memoria, sin fuente descargada).
- **Lo que la historia sí descarta:** con 0,4 veces los recursos de World3 el modelo no reproduce la historia (prueba), y el recargo de capital de la tecnología de World3 tampoco pasa (la perilla queda entre −0,62 y −0,13).

**El error del modelo.** Aunque todos los parámetros fueran ciertos, el modelo se equivoca. Se mide con el modelo estimado hasta 2000 frente a lo observado en 2001-2023, como una deriva proporcional al horizonte:

| Indicador | Deriva por año | En 2100 (77 años) |
|---|---|---|
| Población | 0,10 % | ±8 % |
| Esperanza de vida | 0,09 % | ±7 % |
| Fecundidad | 0,64 % | ±49 % |
| Producción | 0,46 % | ±35 % |
| Uso de recursos | 1,22 % | ±94 % |
| CO₂ atmosférico sobre 280 ppm | 0,24 % | ±19 % |
| Temperatura | 0,012 °C | ±0,9 °C |

Es una desviación típica por miembro; no se retroalimenta al modelo. Las series sin dato (producción industrial por persona, recursos que quedan) no llevan este error.

## 2. La prueba de las bandas

El conjunto no descartado con datos hasta 2000, con su error del modelo medido en la partición 1990-2000, proyectado hasta 2023. Fracción de los años de 2001-2023 con dato que caen dentro de p10-p90 (100 miembros):

| Indicador | Con el error del modelo | Solo parámetros |
|---|---|---|
| Población | 1,00 | 0,04 |
| Esperanza de vida | 1,00 | 0,83 |
| Fecundidad | 1,00 | 0,13 |
| Natalidad | 1,00 | 0,13 |
| Mortalidad | 0,91 | 0,70 |
| Producción | 1,00 | 0,33 |
| Alimento por persona | 1,00 | 1,00 |
| Rendimiento de la tierra | 1,00 | 1,00 |
| Tierra cultivable | 1,00 | 0,38 |
| Uso de recursos | 0,91 | 0,65 |
| CO₂ atmosférico | 0,96 | 0,61 |
| Temperatura | 0,74 | 0,30 |

- **Sin el error del modelo las bandas son inútiles en demografía:** cubren el 4 % de la población. La incertidumbre de los parámetros no es la que importa a 20 años.
- **Con él son anchas de más.** El criterio de la fase era cubrir cerca del 80 %; cubren entre el 91 % y el 100 % en 11 indicadores. La deriva se midió con el modelo estimado hasta 1990, que es peor que el de 2000.
- **La de temperatura se queda corta (74 %)** y es la más inestable: entre 0,57 y 0,83 según el tamaño del conjunto y la semilla. El modelo da menos calentamiento del observado desde 2000.
- **No se ajustó nada para acercar la cobertura al 80 %.** Hacerlo con el periodo de prueba habría sido mirar la respuesta.
- **Las bandas de 2023 son más estrechas que las probadas:** usan la deriva de la partición 2000-2023, que es menor. Su cobertura no se puede medir todavía.

## 3. Lo que dice el conjunto

Arrancando en 2023, 100 miembros (p10 / p50 / p90; modos de comportamiento, no pronóstico):

| | p10 | p50 | p90 |
|---|---|---|---|
| Año del máximo de producción industrial por persona | 2037 | 2071 | 2100 o después |
| Año del máximo de población | 2055 | 2059 | 2066 |
| Población máxima (millones) | 9.178 | 9.478 | 9.952 |

| En 2050 | p10 | p50 | p90 |
|---|---|---|---|
| Población (millones) | 9.144 | 9.387 | 9.701 |
| Esperanza de vida (años) | 77,9 | 80,1 | 82,4 |
| Hijos por mujer | 1,36 | 1,73 | 2,09 |
| Producción industrial por persona, sobre la de 2023 | 0,81 | 1,86 | 2,22 |
| Recursos no renovables que quedan | 29 % | 56 % | 74 % |
| Temperatura sobre 1951-1980 (°C) | 1,17 | 1,52 | 1,95 |

| En 2100 | p10 | p50 | p90 |
|---|---|---|---|
| Población (millones) | 6.731 | 7.671 | 8.475 |
| Esperanza de vida (años) | 58,1 | 80,6 | 89,1 |
| Hijos por mujer | 1,00 | 1,27 | 2,17 |
| Alimento por persona (kg; subsistencia: 230) | 273 | 480 | 978 |
| Producción industrial por persona, sobre la de 2023 | 0,14 | 0,90 | 9,26 |
| Recursos no renovables que quedan | 13 % | 20 % | 52 % |
| CO₂ atmosférico (ppm) | 500 | 574 | 689 |
| Temperatura sobre 1951-1980 (°C) | 1,18 | 2,57 | 3,71 |

- **El máximo industrial no está fijado.** En el 26 % de los miembros no llega antes de 2100; en el 13 % llega antes de 2040.
- **Lo mueven los recursos iniciales:** correlación de rangos de 0,92 con el año del máximo. El siguiente eje, el suelo de eficiencia, se queda en −0,27.

| Recursos iniciales, en veces los de World3 | Miembros | Máximo industrial (mediana) | Temperatura en 2100 (mediana) |
|---|---|---|---|
| 0,7 – 1,0 | 21 | 2037 | 1,88 °C |
| 1,0 – 1,5 | 32 | 2054 | 2,39 °C |
| 1,5 – 2,2 | 29 | 2089 | 3,04 °C |
| 2,2 – 3,0 | 18 | 2100 o después | 2,98 °C |

- **Hay un intercambio:** con pocos recursos la industria toca techo pronto y el clima se calienta menos; con muchos, la industria sigue y la temperatura pasa de 3 °C.
- **La población es lo más firme del modelo:** el máximo cae en una ventana de 11 años y la de 2100 queda entre 6.700 y 8.500 millones con cualquier cantidad de recursos.
- **La esperanza de vida de 2100 tiene una cola baja:** en el 14 % de los miembros baja de 65 años, y en el 5 % el alimento cae por debajo de la subsistencia.

## 4. Desviaciones y lo que no está

- **Descarte en lugar de un posterior bayesiano.** El repo tiene ABC-SMC y emulador (`world4/calib`); aquí no hacen falta, porque 6.000 corridas de la historia tardan 10 s. La contrapartida: todos los miembros que pasan pesan lo mismo.
- **El listón del descarte (el doble del mejor ajuste) es una decisión** [S]. Con 1,5 veces pasaban 6 de 6.000 en la primera caja.
- **Los rangos de los ejes son declarados, no estimados.** El resultado depende de ellos, sobre todo del de los recursos.
- **El error del modelo crece en línea recta con el horizonte** [S]. Más allá de los 23 años medidos es una extrapolación.
- **La cobertura no es del 80 %** (sección 2).
- **Bandas de 116 miembros como máximo** con datos hasta 2023; el informe usa 100.
- **`kappa` se muestrea en 15 niveles,** porque es una tabla de World3 y la corrida vectorizada la comparte.
- **No hay criterio nuevo en la campaña.**

## 5. Qué sigue

1. **Acotar los recursos iniciales con datos.** Es el eje que decide el resultado y el único cuyo rango viene de memoria. Hacen falta reservas y recursos de combustibles fósiles de una fuente descargable.
2. **Fase 6: la prueba fuera de muestra como criterio de la campaña,** con las tablas de origen móvil y esta prueba de cobertura.
3. **Temperatura:** es la banda que falla. Una TCRE efectiva que cambia con el tiempo, o el clima de F4 (FaIR) si sus datos son alcanzables.
