# V2 de Phy-World (World4): World3 vigente — demografía por grupos de países

- **Fecha:** 2026-11-17
- **Estado:** construida y medida. Es la decisión abierta de `V1-informe.md`: tratar la fecundidad y la esperanza de vida por grupos de países.
  - **Grupos** (`world4/core/l0v/groups.py`): el mundo partido en grupos fijos de países, cada uno con sus cohortes, su ingreso y su alimento relativos. Dos particiones: 10 regiones (por defecto) y 4 grupos de ingreso.
  - **Ecuaciones** (`equations.py`): la esperanza de vida y la fecundidad de cada grupo salen de dos ecuaciones estimadas con el panel de países, no de las tablas de World3.
  - **Origen** (`model.py`): la historia hasta un año origen es dato; desde ahí corre el modelo. Es como se prueba fuera de muestra.
  - **Resultado:** con datos solo hasta el origen, el error de población hasta 2023 es de 2,4 % (origen 1980), 1,8 % (1990), 1,2 % (2000) y 0,3 % (2010), con la economía de World3.
  - **Lo que no está resuelto:** el comportamiento a 2100 depende de extrapolaciones fuera de los datos (sección 5).
- **El escenario 1 no cambia.** W1 sigue verde.
- **Pruebas:** 100 pasan (19 nuevas, 11 de ellas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:**
  - `pip install -e .[world]` y `python world4_cli.py --install-world` (el panel de países necesita `pycountry`);
  - `python world4_cli.py --world3-score` (marcador y prueba con origen móvil, unos segundos);
  - `python -c "from world4.core.l0v import equations as e; print(e.selection())"` (la tabla con la que se eligió la forma).

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| Panel de países | `data/world.py` (`world_countries.npz`) | 236 países de WPP, 1950-2023: fecundidad, esperanza de vida, nacimientos, muertes, PIB (Maddison, 164 países, 95-99 % de la población) y kilocalorías (FAO, 1961-2013) |
| Grupos | `core/l0v/groups.py` | Por grupo: población por edad y sexo, fecundidad y su patrón por edad, tablas de vida, migración neta (residuo contable), ingreso y alimento relativos al mundo |
| Tablas de World3 por grupo | `core/l0v/popsector.py` | Las ecuaciones de población de World3 fuera de su bucle; con un grupo dan exactamente la `le` y la `tf` de `l0.run` |
| Ecuaciones estimadas | `core/l0v/equations.py` | Dos logísticas acotadas, ancladas en lo observado en el origen |
| Acople | `core/l0v/model.py` (`GroupCoupler`) | World3 da la economía del mundo; cada grupo recibe su parte y devuelve nacimientos y muertes |
| Prueba | `validate/world3.py` (`hindcast`, `group_check`) | Origen móvil frente a congelar las tasas |

**Particiones.**
- `regions`: las 10 regiones del L0 regional (`core/l0r/regions.py`).
- `income`: cuatro grupos por PIB por persona en 1970, con cortes en 1.500, 4.000 y 10.000 dólares de 2011. El grupo bajo junta a China, India y parte de África: 3.937 millones en 2023, casi la mitad del mundo.
- `world`: un solo grupo, que con las tablas de World3 reproduce el modelo de V1 (prueba).

**Reparto del ingreso.** El ingreso por persona de un grupo es el del mundo (de World3) por su ingreso relativo observado, renormalizado para que la suma sea el total. Después de 2022 el reparto se queda fijo [S].

## 2. Tres cosas que no funcionaron, medidas

| Intento | Resultado | Por qué |
|---|---|---|
| Las tablas de World3 evaluadas en cada región | Esperanza de vida 17,7 % de error; fecundidad 44 % | Con un quinto del ingreso mundial dan 30 años de vida en África subsahariana (real: 62). La relación real entre ingreso y esperanza de vida es mucho más plana |
| Las tablas de World3 con la economía real (un solo mundo) | Esperanza de vida 86,8 años en 2023 (real 73,2); población 10.284 millones | Con el crecimiento real los servicios de salud saturan la tabla. Las tablas solo cuadran con la economía de World3 |
| Ecuaciones estimadas corriendo libres desde 1950 | 5.167 millones en 2023 (real 8.092) | No representan el arranque de la transición demográfica: hacen bajar la fecundidad desde 1950, cuando hasta 1970 estuvo estable |

El tercero fija el uso del modelo: **arranca en un origen con datos** y las ecuaciones solo dan el cambio desde ahí.

## 3. Las ecuaciones

    logit[(e0 − 20) / (92 − 20)]      = u + b_y · ln(ingreso por persona) + b_k · ln(alimento por persona) + b_t · t
    logit[(TFR − 0,8) / (8,5 − TFR)]  = v + c_e · e0 + c_t · t

Estimadas por mínimos cuadrados con efectos fijos de país, ponderadas por población.

| Datos hasta | b_y (ingreso) | b_k (alimento) | b_t (por año) | c_e (por año de e0) | c_t (por año) |
|---|---|---|---|---|---|
| 1980 | −0,007 | 1,221 | 0,021 | −0,007 | −0,029 |
| 1990 | 0,092 | 0,819 | 0,019 | −0,033 | −0,023 |
| 2000 | 0,100 | 0,722 | 0,018 | −0,045 | −0,024 |
| 2010 | 0,088 | 0,644 | 0,018 | −0,052 | −0,020 |
| 2023 | 0,075 | 0,599 | 0,018 | −0,059 | −0,017 |

- **Estables desde 1990.** Con datos hasta 1980 no: las kilocalorías empiezan en 1961 y hay 20 años.
- **La esperanza de vida responde poco al ingreso y bastante al alimento y al tiempo.** A 73 años, el término del tiempo da 0,25 años por año.
- **Respuesta al alimento, un tercio de la de World3.** Reducir el alimento a la mitad baja la esperanza de vida unos 6 años a 73; la tabla de World3 la baja un 30 %. El 98 % de los datos está entre 1.550 y 3.600 kcal por persona y día: no hay ninguna hambruna mundial en la muestra.
- **El ingreso no entra en la fecundidad.** Las formas con ingreso predicen de más la caída posterior a 2000, cuando el ingreso de Asia creció mucho y la fecundidad se frenó.
- **Lo que los datos no cubren queda como en World3:** por debajo de la subsistencia manda su tabla de alimento, y la contaminación multiplica con su tabla.

**Cómo se eligió la forma** (`equations.selection`, regiones, regresores observados, anclado en el origen; error fuera de muestra hasta 2023):

| | Congelar | Tendencia de 10 años | Forma elegida | Formas descartadas |
|---|---|---|---|---|
| Esperanza de vida mundial, datos ≤ 1990 | 7,6 % | 1,2 % | 1,5 % | sin alimento: 2,2-2,9 % |
| Esperanza de vida mundial, datos ≤ 2000 | 5,4 % | 0,5 % | 0,6 % | sin alimento: 1,9-2,5 % |
| Nacimientos mundiales, datos ≤ 1990 | 36,2 % | 4,3 % | 4,8 % | con ingreso: 6,2-8,0 %; solo e0: 13,0 % |
| Nacimientos mundiales, datos ≤ 2000 | 16,0 % | 9,0 % | 7,3 % | con ingreso: 11,2-14,8 %; solo e0: 4,0 % |

- **La forma elegida iguala a la tendencia, no la supera con claridad.** La diferencia es que depende del estado del modelo (alimento, ingreso, esperanza de vida), y una tendencia no.
- **La fecundidad solo con la esperanza de vida** es la mejor desde 2000 y la peor desde 1990. Se eligió la que va bien en los dos orígenes.
- **Exploración declarada:** se compararon 3 formas de esperanza de vida y 5 de fecundidad con 4 suelos mirando el periodo de prueba. El error de la forma elegida es por eso optimista.
- **El suelo de la fecundidad (0,8) no lo identifica el ajuste:** los cuatro suelos probados dan errores parecidos.

## 4. Lo medido

**El motor por grupos solo** (historia observada, 1950-2023): la población mundial con 0,9 % de error medio y 1,5 % máximo; cada región dentro de ±4,1 % en 2023 (África subsahariana −4,1 %, Asia meridional −2,9 %).

**Prueba con origen móvil** (`hindcast`: regiones, ecuaciones estimadas con datos ≤ origen, error medio de origen + 1 a 2023; entre paréntesis, congelando las tasas de cada región en el origen):

| Origen | Población | Nacimientos | Muertes | Esperanza de vida |
|---|---|---|---|---|
| **Con la economía de World3** | | | | |
| 1980 | 2,4 (7,4) | 7,8 (41,6) | 4,8 (42,9) | 1,8 (12,1) |
| 1990 | 1,8 (5,3) | 5,3 (31,7) | 4,8 (30,8) | 1,6 (8,7) |
| 2000 | 1,2 (0,6) | 3,3 (15,5) | 8,4 (24,8) | 2,2 (6,8) |
| 2010 | 0,3 (0,2) | 2,4 (10,4) | 7,5 (12,9) | 1,7 (3,2) |
| **Con la economía real** | | | | |
| 1980 | 3,5 | 6,6 | 9,4 | 3,3 |
| 1990 | 2,0 | 3,9 | 5,8 | 1,9 |
| 2000 | 1,1 | 6,2 | 1,7 | 0,3 |
| 2010 | 0,3 | 1,9 | 2,8 | 0,5 |

- **Nacimientos, muertes y esperanza de vida mejoran a congelar en los cuatro orígenes.**
- **En población, congelar gana en 2000 y 2010** (0,6 y 0,2 %): sus errores de nacimientos y muertes, del 10 al 25 %, se compensan. Es el mismo tipo de acierto que el de las cuatro cajas.
- **Desde 2000, la población de 2023 sale en 7.873 millones** (real 8.092, −2,7 %): faltan nacimientos.
- **Con la economía de World3 las muertes fallan más desde 2000** (8,4 % frente a 1,7 % con la real): el alimento de World3 cae después de 2000 y arrastra la esperanza de vida a 69,5 años en 2023. Es el error de la economía (fase 2), ya aislado.

**Regiones frente a grupos de ingreso** (economía de World3, orígenes 1980 / 1990 / 2000 / 2010):

| | Población | Nacimientos |
|---|---|---|
| 10 regiones | 2,4 / 1,8 / 1,2 / 0,3 | 7,8 / 5,3 / 3,3 / 2,4 |
| 4 grupos de ingreso | 0,6 / 0,8 / 2,0 / 0,6 | 3,4 / 5,3 / 9,8 / 2,6 |

- **No hay un ganador claro.** Los grupos de ingreso van mejor desde 1980 y peor desde 2000.
- **Por defecto quedan las regiones:** separan África subsahariana, donde estarán la mayoría de los nacimientos futuros, y son las que el repo ya usa. `groups="income"` las cambia.

**El marcador** (`--world3-score`, 2000-2023): la columna nueva, con historia observada hasta 2000 y el modelo desde ahí, da 1,2 % en población (World3-03: 4,7 %; cohortes de V1: 6,7 %), 3,2 % en natalidad y 2,1 % en esperanza de vida.

## 5. Lo que no está resuelto: el comportamiento a 2100

Arrancando en 2023 con el escenario 1 de World3 (modo de comportamiento, no pronóstico):

- la población sube hasta 9.268 millones en 2050 y sigue en 8.823 millones en 2100;
- la esperanza de vida cae a 44,5 años en 2050 y a 24,6 en 2100;
- la fecundidad sube a 3,9 hijos por mujer en 2050 y a 5,0 en 2100.

**No es un resultado utilizable.** Sale de tres extrapolaciones fuera de los datos:

1. **La agricultura del escenario 1 se hunde:** el alimento por persona de World3 pasa de 376 a 158 kg en 2050, por debajo de la subsistencia (230). Es el fallo de rendimientos de la fase 2, agravado porque ahora hay más gente.
2. **La fecundidad responde a la esperanza de vida en los dos sentidos.** El coeficiente se estimó con esperanzas de vida que suben; aplicado a una caída de 48 años devuelve la fecundidad a 5 hijos. Nada en los datos dice que eso ocurra.
3. **El término del tiempo no se detiene** aunque la economía se hunda.

Sin colapso, el problema es el contrario: el suelo de 0,8 y el término del tiempo llevan la fecundidad mundial hacia 1 hijo por mujer en 2100. La ONU proyecta alrededor de 1,8 (cifra citada de memoria, no comprobada en esta sesión).

## 6. Desviaciones y lo que no está

- **Las regiones no son los «3 o 4 grupos de ingreso» que se plantearon.** Los grupos de ingreso están construidos y medidos; el empate se muestra arriba.
- **La migración entre grupos después del origen** es la media de los diez años anteriores [S].
- **El reparto de ingreso y alimento se congela** después del último dato (2022 y 2013) [S]. La convergencia de China o India no continúa.
- **Kilocalorías frente al alimento de World3:** World3 mide en equivalente vegetal, que pesa más la carne. Solo entran cocientes, pero no son la misma magnitud.
- **La forma de las tablas de vida de cada grupo** fuera de su rango observado es la de la familia de países (V1).
- **`world4.core.l0v.fit` (V1)** sigue como experimento; no se usa.
- **No hay criterio nuevo en la campaña.**

## 7. Qué sigue

1. **Fase 2: tecnología estimada** (rendimientos, eficiencia, productividad). Es ahora lo que más limita: la economía de World3 explica el error de muertes desde 2000 y el hundimiento del alimento después de 2023.
2. **Decisión abierta: cómo responde la demografía a la escasez.** Tres piezas sin datos que las fijen: la fecundidad cuando la esperanza de vida baja, el término del tiempo en un colapso y el suelo de la fecundidad. Opciones: dejarlas como ejes de escenario con bandas (fase 5) o fijar reglas explícitas.
3. **Datos pendientes** de la fase 0 (FAOSTAT posterior a 2013, PIB anual, materiales).
