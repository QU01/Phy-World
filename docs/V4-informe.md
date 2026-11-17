# V4 de Phy-World (World4): World3 vigente — carbono, clima y reglas para la escasez

- **Fecha:** 2026-11-17
- **Estado:** construidas y medidas dos cosas: la parte de la fase 4 que los datos alcanzables permiten (carbono y clima) y las reglas explícitas que cierran la decisión abierta de V2 (cómo responde la demografía a la escasez).
  - **Carbono y clima** (`world4/core/l0v/carbon.py`): las emisiones salen del uso de recursos del modelo; dos rectas estimadas con Mauna Loa y GISTEMP dan el CO₂ atmosférico y la temperatura. El calentamiento desde el origen resta rendimiento agrícola.
  - **Reglas** (`equations.Rules`): seis reglas declaradas para lo que los datos no cubren.
  - **Resultado en la historia:** con datos hasta 2000, el CO₂ atmosférico de 2001-2023 sale con 3,4 ppm de error y la temperatura con 0,13 °C. Las reglas no cambian ninguna prueba con datos.
  - **Resultado en la proyección:** las reglas deciden el escenario con escasez (sin tecnología, la población de 2100 pasa de 7.763 a 4.128 millones). Con la tecnología estimada apenas actúan antes de 2100.
  - **Lo que no se hizo de la fase 4:** separar energía de materiales y estimar el coste de extracción. No hay datos alcanzables para ninguna de las dos.
- **El escenario 1 no cambia.** W1 sigue verde.
- **Pruebas:** 124 pasan (12 nuevas, 5 de ellas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:** `python world4_cli.py --install-world` (añade GISTEMP) y `python world4_cli.py --world3-score`.

## 1. Carbono y clima

**Por qué.** La contaminación persistente de World3 dura unos dos años en el ambiente. Compararla con el CO₂, que dura siglos, daba un 110 % de error que no medía nada. El marcador compara ahora el CO₂ que implican las emisiones del modelo con el de Mauna Loa, y la contaminación de World3 sigue dentro del modelo con sus efectos pero sin comparación.

**Las relaciones.**

    emisiones fósiles e industria = las reales de 1970 · uso de recursos(t) / uso de recursos(1970)
    emisiones por uso del suelo   = las observadas; después, la media de los diez años anteriores [S]
    CO₂ atmosférico (ppm)         = c0 + airborne · emisiones acumuladas / 7,82
    temperatura global (°C)       = t0 + tcre · emisiones acumuladas / 1000

| Datos hasta | `airborne` | `tcre` (°C por 1000 GtCO₂) | Error después del corte, con las emisiones reales |
|---|---|---|---|
| 1990 | 0,435 | 0,465 | 0,6 ppm · 0,10 °C |
| 2000 | 0,428 | 0,468 | 0,8 ppm · 0,11 °C |
| 2010 | 0,431 | 0,496 | 0,8 ppm · 0,10 °C |
| 2024 | 0,434 | 0,524 | — |

- **La fracción que se queda en el aire es estable** (0,43) y predice el CO₂ con menos de 1 ppm de error.
- **La TCRE es efectiva:** carga al CO₂ todo el calentamiento observado, también el del metano y el enfriamiento de los aerosoles. Sube con los datos recientes (de 0,47 a 0,52), así que estimada hasta 2000 se queda corta.
- **Son rectas en las emisiones acumuladas.** Valen mientras se siga emitiendo; cuando las emisiones caen, el CO₂ real baja despacio y la recta lo deja fijo.
- **`c0` libre:** con la recta forzada a 280 ppm el CO₂ de 2023 salía 11 ppm bajo.

**El daño es una regla, no una estimación:** el rendimiento agrícola pierde un 5 % por cada grado de calentamiento global desde el origen. Sale del rango que F5 midió con FAOSTAT (de −2,7 % por grado local en un país templado a −5 % en uno cálido) y de que la tierra se calienta más que la media del planeta [S].

**Los recursos de World3 en la moneda del carbono.** Con la misma equivalencia que las emisiones, el stock inicial de recursos no renovables de World3 son 4.293 GtCO₂. El modelo ha emitido 1.726 hasta 2023 (reales, fósil e industria: 1.809) y le quedan 2.567. Es del orden de las reservas probadas de combustibles fósiles, no de los recursos totales, que son varias veces mayores (órdenes de magnitud citados de memoria, sin fuente descargada).

## 2. Las reglas

Las ecuaciones de V2 se estimaron con una muestra en la que la esperanza de vida casi siempre sube y los servicios casi nunca caen. No dicen qué pasa en una escasez. Las reglas son decisiones, cada una con su interruptor en `equations.Rules`:

| # | Regla | De dónde sale |
|---|---|---|
| 1 | Por debajo de la subsistencia, la esperanza de vida sigue la tabla de alimento de World3 | Ya estaba en V2 |
| 2 | La contaminación multiplica con la tabla de World3 | Ya estaba en V2 |
| 3 | Si los servicios por persona de una región caen por debajo de los del origen, su esperanza de vida se multiplica por la tabla de salud de World3 relativa al origen, con su retardo de 20 años | Nueva |
| 4 | El término del tiempo (progreso sanitario y difusión de la baja fecundidad) avanza a un ritmo proporcional a los servicios sobre los del origen, con máximo 1 | Nueva |
| 5 | En la fecundidad entra la esperanza de vida máxima alcanzada, no la del momento: una crisis no devuelve la fecundidad arriba | Nueva |
| 6 | Suelo de la fecundidad en 1,0 hijos por mujer, el valor regional más bajo observado (1,02 en Asia oriental en 2023) | Antes 0,8 |

- **Principio común:** donde hay datos mandan los datos; donde no, manda World3 o una regla que se pueda leer y apagar.
- **Las reglas 3 a 5 no cambian nada mientras los servicios y la esperanza de vida suban.** En las pruebas con datos el error de población, esperanza de vida y natalidad es el mismo con y sin ellas (prueba).
- **El suelo en 1,0 sí cambia las estimaciones,** pero poco: la prueba de origen móvil de V2 se mueve como mucho 0,2 puntos.

## 3. Lo medido en la historia

**El marcador** (`--world3-score`, error medio 2000-2023 en %; la última columna con historia observada hasta 2000 y datos ≤ 2000):

| Indicador | World3-03 | Regiones (V2) | Modelo completo |
|---|---|---|---|
| CO₂ atmosférico sobre 280 ppm | 32,7 | 14,7 | **2,6** |
| Producción | 26,7 | 27,5 | 6,9 |
| Alimento por persona | 14,1 | 17,9 | 2,6 |
| Uso de recursos | 106,4 | 94,6 | 13,7 |
| Población | 4,7 | 1,2 | 1,0 |
| Esperanza de vida | 8,4 | 2,1 | 0,7 |
| Mortalidad | 6,9 | 8,0 | 2,6 |
| Natalidad | 11,6 | 3,2 | 6,0 |

**Carbono y clima fuera de muestra** (`technology_hindcast`: todo estimado con datos ≤ origen, emisiones del modelo, error medio de origen + 1 a 2023; entre paréntesis, con la tecnología de World3):

| Origen | CO₂ (ppm) | Temperatura (°C) |
|---|---|---|
| 1990 | 3,0 (21,6) | 0,13 (0,12) |
| 2000 | 3,4 (18,8) | 0,13 (0,10) |
| 2010 | 2,7 (11,7) | 0,11 (0,08) |

- **El CO₂ mejora mucho** porque el uso de recursos ya sigue a las emisiones reales.
- **La temperatura no mejora:** queda en 0,11-0,13 °C con y sin tecnología. Desde 2000 el modelo da 0,81 °C en 2023 y lo observado es 1,17. Dos causas que se suman: la TCRE estimada hasta 2000 es baja y el modelo emite algo menos que el mundo.
- **Con la tecnología de World3 la temperatura sale algo mejor** porque sus emisiones, demasiado altas, compensan esa TCRE baja.

## 4. La proyección

Arrancando en 2023 con todo estimado hasta 2023 (modo de comportamiento, no pronóstico):

| | Sin tecnología, sin reglas | Sin tecnología, con reglas | Modelo completo | Completo, tecnología congelada en 2023 | Completo, doble de recursos |
|---|---|---|---|---|---|
| Población máxima | 9.161 M en 2047 | 8.840 M en 2039 | 9.573 M en 2060 | 9.538 M en 2059 | 9.558 M en 2059 |
| Población en 2100 | 7.763 M | 4.128 M | 7.889 M | 7.589 M | 7.585 M |
| Esperanza de vida en 2050 / 2100 | 45,1 / 28,3 | 45,3 / 38,8 | 80,8 / 84,1 | 79,6 / 74,5 | 80,5 / 75,2 |
| Hijos por mujer en 2100 | 3,99 | 2,30 | 1,28 | 1,42 | 1,26 |
| Alimento por persona en 2100 (subsistencia: 230 kg) | — | — | 484 | 337 | 262 |
| Máximo de producción industrial por persona | 2012 | 2012 | 2047 | 2044 | después de 2100 |
| CO₂ en 2100 (ppm) | — | 491 | 549 | 548 | 613 |
| Temperatura en 2100 sobre 1951-1980 (°C) | — | 1,65 | 2,19 | 2,19 | 2,79 |

- **Las reglas deciden el escenario con escasez.** Sin tecnología el alimento baja de la subsistencia en 2037; sin reglas la fecundidad volvía a 4 hijos y la población se quedaba en 7.763 millones con 28 años de esperanza de vida. Con reglas la población baja a 4.128 millones.
- **Con la tecnología estimada las reglas casi no actúan antes de 2100.** Los servicios por persona suben hasta 2052 y no bajan del nivel de 2023 hasta 2077; en 2100 son el 45 % de los de 2023. El efecto llegaría después de 2100.
- **Por regiones en 2100:** la esperanza de vida va de 78,8 años en África subsahariana (la única donde la regla 3 pesa) a 89,6 en la OCDE del Pacífico; la fecundidad, de 1,00 a 1,45.
- **El clima resta poco al alimento:** 1,23 °C más que en 2023 quitan un 6 % de rendimiento en 2100.
- **Temperatura en 2100:** 2,19 °C sobre 1951-1980, unos 2,5 °C sobre 1900-1920.
- **Con el doble de recursos el límite vuelve a ser el alimento:** la industria no toca techo, pero la contaminación de World3 degrada el suelo y el clima resta un 9 %; el alimento cae a 262 kg y la esperanza de vida a 75 años.

**Lo que sigue sin fijar.** El máximo industrial de 2047 depende de los recursos iniciales (4.293 GtCO₂ en carbono) y del suelo de eficiencia, como en V3. Las reglas no lo mueven.

## 5. Desviaciones y lo que no está

- **Energía y materiales siguen en un solo stock.** No hay series alcanzables de extracción de materiales.
- **El coste de extracción es el de World3** (tabla `fcaor`). No hay datos alcanzables de inversión en extracción.
- **Los recursos iniciales siguen siendo los de World3.** Ahora se sabe qué significan en carbono, pero no se han estimado.
- **GISTEMP en lugar de HadCRUT5,** que es la referencia de F4 y no era alcanzable. Anomalías sobre 1951-1980.
- **El daño climático solo toca el rendimiento agrícola.** No hay daño a la producción ni a la salud.
- **Antes del origen no hay daño aparte:** la curva de rendimiento se estimó con los rendimientos reales, que ya lo llevan.
- **El clima necesita la tecnología estimada:** el daño actúa sobre su curva de rendimiento.
- **La regla 3 usa la tabla de salud de World3, que se satura:** una región rica puede perder la mitad de sus servicios sin que su esperanza de vida baje.
- **No hay criterio nuevo en la campaña.**

## 6. Qué sigue

1. **Fase 5: bandas.** Los ejes que la historia no fija ya están identificados: recursos iniciales, suelo de eficiencia, techo del rendimiento, daño climático y las reglas de esta fase, que se pueden apagar una a una.
2. **Fase 6: prueba fuera de muestra como criterio de la campaña.** Las tablas de origen móvil ya existen; falta fijar los listones.
3. **Datos pendientes:** FAOSTAT posterior a 2013, PIB anual, materiales, reservas e inversión en extracción, HadCRUT5.
