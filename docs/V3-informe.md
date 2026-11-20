# V3 de Phy-World (World4): World3 vigente — tecnología estimada con datos

- **Fecha:** 2026-11-17
- **Estado:** construida y medida la fase 2 del plan: rendimiento agrícola, uso de recursos, contaminación generada, coste de capital de la tecnología y tierra cultivable, estimados con las series mundiales.
  - **Resultado:** con datos solo hasta 2000, el error de 2000-2023 baja de 26,7 % a 6,6 % en producción, de 14,1 % a 2,2 % en alimento por persona y de 106 % a 13,6 % en uso de recursos. Se cumplen las tres metas de la fase (10 %, 3 % y 20 %).
  - **Dos supuestos de World3 que los datos contradicen:** la tecnología no encarece el capital, y la erosión ligada al rendimiento habría hecho desaparecer una tierra de cultivo que sigue ahí.
  - **El límite cambia de sitio.** Con la tecnología estimada ya no hay hambruna; lo que frena al modelo son los recursos no renovables, y cuándo lo hacen depende de dos supuestos que la historia no fija (sección 5).
  - **Lo que empeora:** la natalidad desde 2000 (de 3,3 % a 6,8 % de error) y el rendimiento agrícola pronosticado desde 1990.
- **El escenario 1 no cambia.** W1 sigue verde.
- **Pruebas:** 112 pasan (12 nuevas, 7 de ellas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:**
  - `python world4_cli.py --install-world` (añade las tierras de cultivo de HYDE);
  - `python world4_cli.py --world3-score` (marcador y las dos pruebas fuera de muestra, unos 6 s);
  - `python -m world4.core.l0v.technology` (reestima y reescribe `world4/data/l0v_technology.json`, unos 6 min).

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| Curvas de tecnología | `core/l0v/technology.py` (`Tech`, `TechHook`) | Los stocks `lyt`, `rct` y `ppt` de World3 se sustituyen por curvas en el tiempo desde 1975 |
| Estimación | `technology.calibrate`, `data/l0v_technology.json` | Cinco perillas, mínimos cuadrados en logaritmos, dos arranques del optimizador; guardadas para los cortes 1990, 2000, 2010 y 2023 |
| Ganchos compuestos | `technology.compose` | La tecnología y la demografía por regiones actúan a la vez sobre `l0.run` |
| Series nuevas | `data/world.py` | Tierras de cultivo (HYDE) y alimento en equivalente vegetal |
| Prueba | `validate/world3.py` (`technology_hindcast`) | El modelo completo con demografía y tecnología estimadas con datos ≤ origen |

**Las curvas.** Desde 1975, el año de políticas de World3:

    rendimiento   lyt(t) = 2 − (2 − 1) · exp(−r_y · (t − 1975))            sube hacia un techo de 2
    recursos      rct(t) = 0,2 + 0,8 · exp(−r_r · (t − 1975))              baja hacia un suelo de 0,2
    contaminación ppt(t) = 0,2 + 0,8 · exp(−r_p · (t − 1975))

**Las cinco perillas y con qué dato se estima cada una:**

| Perilla | Qué es | Dato (índice 1970 = 1) |
|---|---|---|
| `r_yield` | Ritmo de la tecnología de rendimiento | Rendimiento de cereales (Banco Mundial) |
| `r_resource` | Ritmo de mejora del uso de recursos por unidad de producción | CO₂ de combustibles fósiles |
| `r_pollution` | Ritmo de bajada de la contaminación por unidad de actividad | CO₂ de combustibles fósiles |
| `kappa` | Coste de capital de la tecnología: 1 es World3, 0 es gratis | PIB (Maddison) |
| `land_restraint` | Fracción del cambio neto de tierra de World3 que ocurre | Tierras de cultivo (HYDE), en hectáreas |

En las corridas de estimación la población es la observada (cohortes con fecundidad y esperanza de vida reales).

## 2. Lo que obligó a corregir el dato

| Síntoma | Causa | Corrección |
|---|---|---|
| La curva de recursos caía a su suelo en diez años y el suelo, que es un supuesto, decidía el ajuste | El retardo de 20 años de World3 entre desarrollar una tecnología y usarla | El retardo baja a 1 año: la curva se estima con lo que el mundo hizo, así que ya es tecnología en uso |
| Tierra cultivable en 1.162 millones de hectáreas en 2016 (HYDE: 1.593), 654 en 2023 y 88 en 2050 | La tabla de World3 acorta la vida de la tierra cuando sube el rendimiento; con los rendimientos reales la erosiona en una generación | Rige desde 1975 el control de la erosión del escenario 5 del propio libro [S] |
| Sin esa erosión, 1.995 millones de hectáreas en 2016 | World3 rotura tierra mientras el alimento deseado crece con el ingreso | Perilla `land_restraint` |
| Cambiar la tierra potencial total dejaba 1970 un 15 % bajo | Esa constante mueve toda la historia desde 1900 | Se descartó como perilla |
| El alimento por persona parecía un 21 % alto | El dato eran kilocalorías, y World3 cuenta en equivalente vegetal | Serie nueva: las kilocalorías animales cuentan por 7, como en World3 |

- **La erosión es una elección, no una estimación.** El ajuste a la tierra no distingue más erosión de menos roturación: con la erosión de World3 y roturación libre el modelo sigue a HYDE hasta 2000. Lo que lo descarta es que en 2016 queda un 27 % por debajo y que pierde casi la mitad de la tierra en los siete años siguientes.
- **El equivalente vegetal cambia el marcador de las fases anteriores:** el alimento por persona de World3-03 pasa de 7,7 % a 14,1 % de error. Las kilocalorías animales se calculan con los kilos de carne, leche, huevos y pescado y un contenido energético medio de cada uno [S].

## 3. Lo estimado

| Datos hasta | `r_yield` | `r_resource` | `r_pollution` | `kappa` | `land_restraint` |
|---|---|---|---|---|---|
| 1990 | 0,029 | 0,054 | 0,033 | 0,02 | 0,40 |
| 2000 | 0,018 | 0,049 | 0,030 | −0,12 | 0,11 |
| 2010 | 0,016 | 0,049 | 0,029 | −0,30 | 0,07 |
| 2023 | 0,017 | 0,045 | 0,026 | −0,32 | 0,07 |

Error cuadrático medio del ajuste (datos ≤ 2023): rendimiento 4,9 %, uso de recursos 7,2 %, generación de contaminación 5,5 %, producción 2,7 %, tierra 2,2 %.

- **La tecnología no encarece el capital.** `kappa` sale entre 0,02 y −0,32, nunca cerca de 1. Con datos hasta 2023 es negativa: la tecnología abarata el capital, que es lo más parecido a productividad que tiene World3.
- **El uso de recursos por unidad de producción baja cerca de un 5 % al año** hacia su suelo, y la contaminación un 3 %.
- **La tierra cultivable creció entre un 7 % y un 11 %** de lo que World3 roturaría (40 % con datos hasta 1990).
- **Con 15 años de datos (corte 1990) el rendimiento y la tierra no quedan bien fijados;** desde 2000 las cinco perillas son estables.

## 4. Lo medido

**El marcador** (`--world3-score`, error medio 2000-2023 en %; las dos últimas columnas llevan historia observada hasta 2000 y datos ≤ 2000):

| Indicador | World3-03 | Regiones (V2) | Regiones + tecnología |
|---|---|---|---|
| Producción | 26,7 | 27,5 | **6,6** |
| Alimento por persona | 14,1 | 17,9 | **2,2** |
| Rendimiento de la tierra | 23,8 | 24,0 | **4,0** |
| Tierra cultivable | 17,5 | 18,0 | **3,9** |
| Uso de recursos | 106,4 | 94,6 | **13,6** |
| Contaminación | 109,5 | 112,9 | 32,6 |
| Población | 4,7 | 1,2 | 1,1 |
| Esperanza de vida | 8,4 | 2,1 | 0,8 |
| Mortalidad | 6,9 | 8,0 | 2,8 |
| Natalidad | 11,6 | 3,2 | 6,6 |

**El modelo completo fuera de muestra** (`technology_hindcast`: demografía y tecnología con datos ≤ origen, error medio de origen + 1 a 2023; entre paréntesis, con la tecnología de World3):

| Origen | Producción | Alimento | Rendimiento | Tierra | Recursos | Población |
|---|---|---|---|---|---|---|
| 1990 | 2,3 (27,7) | 15,4 (15,4) | 17,2 (20,7) | 11,4 (18,3) | 11,1 (87,9) | 2,1 (1,8) |
| 2000 | 7,2 (30,0) | 2,4 (18,4) | 3,9 (24,5) | 4,1 (18,2) | 13,8 (96,2) | 1,1 (1,2) |
| 2010 | 2,5 (31,4) | 9,3 (24,6) | 2,8 (30,3) | 3,1 (18,8) | 13,2 (92,4) | 0,3 (0,3) |

- **Producción y recursos mejoran mucho en los tres orígenes.**
- **Desde 1990 el alimento no mejora y el rendimiento apenas:** la curva estimada con 15 años de datos sube demasiado deprisa.
- **La contaminación sigue en torno al 33 %.** El contaminante de World3 dura unos 2 años en el ambiente y el CO₂ siglos: no son la misma cosa. Es la fase 4.

**La demografía dentro del modelo completo** (regiones, error medio de origen + 1 a 2023, sin y con tecnología):

| Origen | Esperanza de vida | Mortalidad | Natalidad |
|---|---|---|---|
| 1990 | 1,6 → 3,8 | 4,8 → 11,2 | 5,3 → 3,9 |
| 2000 | 2,2 → 0,8 | 8,4 → 2,9 | 3,3 → 6,8 |
| 2010 | 1,7 → 0,8 | 7,5 → 3,7 | 2,4 → 2,0 |

- **Desde 2000 y 2010 la mortalidad mejora:** el alimento ya no se hunde.
- **Desde 2000 la natalidad empeora.** La esperanza de vida del modelo llega a 74,9 años en 2023 (real 73,2) y la ecuación de fecundidad responde con 2,00 hijos por mujer (real 2,25).
- **Desde 1990 la esperanza de vida se pasa** (77,6 años en 2023): es el rendimiento sobrestimado de ese corte.

## 5. El comportamiento a 2100: el límite son los recursos, y no está fijado

Arrancando en 2023 con todo estimado hasta 2023 (modo de comportamiento, no pronóstico):

| | Sin tecnología (V2) | Tecnología que sigue | Tecnología congelada en 2023 |
|---|---|---|---|
| Alimento por persona en 2050 (subsistencia: 230 kg) | 158 | 718 | 630 |
| Esperanza de vida en 2050 / 2100 | 44,5 / 24,6 | 81,0 / 84,9 | 79,9 / 82,6 |
| Población máxima | 9.268 M en 2050 | 9.537 M en 2059 | 9.517 M en 2059 |
| Población en 2100 | 8.823 M | 7.745 M | 7.768 M |
| Máximo de producción industrial por persona | 2012 | 2047 | 2044 |
| Producción industrial por persona en 2100, sobre su máximo | — | 16 % | 10 % |
| Recursos no renovables que quedan en 2100 | 17 % | 14 % | 15 % |

- **La hambruna de V2 desaparece.** Venía de la agricultura sin tecnología del escenario 1.
- **El freno pasa a ser el agotamiento de los recursos:** hacia 2047 el capital dedicado a extraerlos empieza a comerse la inversión y en 2100 es el 81 %.
- **La población baja por la fecundidad, no por la mortalidad.** La esperanza de vida sigue subiendo aunque la industria caiga un 84 %, porque en las ecuaciones de V2 responde poco al ingreso y su término del tiempo no se detiene. Es la decisión abierta de V2.

**Cuándo llega ese máximo depende de dos números que la historia no fija:**

| Supuesto | Máximo de producción industrial por persona | En 2100, sobre el máximo |
|---|---|---|
| Como arriba (recursos iniciales de World3, suelo 0,2) | 2047 | 16 % |
| El doble de recursos iniciales | después de 2100 | — |
| Suelo del uso de recursos en 0,1 | 2069 | 72 % |
| La mitad de recursos iniciales | 2010 | 6 % |

- **La historia descarta que haya muchos menos recursos** (el máximo ya habría pasado en 2010), pero no que haya más.
- **Los recursos iniciales (`nri`) no se han estimado:** son los de World3. Junto con el suelo, el techo del rendimiento y el suelo de la fecundidad, son los ejes de incertidumbre para la fase 5.

## 6. Desviaciones y lo que no está

- **Productividad:** no hay un término propio. La recoge `kappa` negativa, que solo actúa mientras avanzan las tres tecnologías.
- **El mismo dato para dos perillas:** el CO₂ fósil mide a la vez el uso de recursos y la generación de contaminación. No incluye metales ni minerales.
- **`land_restraint` escala el cambio neto,** así que también reduce la erosión y la urbanización, no solo la roturación.
- **La tierra sigue creciendo en la proyección** (1.731 millones de hectáreas en 2023, 2.132 en 2050), y en 2016 ya va un 3 % por encima de HYDE.
- **Las curvas siguen después del último dato** al mismo ritmo [S]. `Tech(freeze_at=2023)` las detiene.
- **La estimación usa la población observada,** no la del modelo.
- **No hay criterio nuevo en la campaña.**

## 7. Qué sigue

1. **Fase 4 antes que la 3: recursos y contaminación reales.** Es lo que decide el siglo en el modelo y lo peor medido. Separar energía fósil de materiales, estimar el coste de extracción y cambiar el contaminante genérico por CO₂ y clima.
2. **Decisión abierta de V2, ahora más visible:** que la esperanza de vida no baje con la industria un 84 % menor no es creíble. Hay que decidir cómo responde la demografía a la escasez.
3. **Fase 5:** bandas con los ejes no identificados (recursos iniciales, suelos y techos).
