# V1 de Phy-World (World4): World3 vigente — datos, marcador y cohortes

- **Fecha:** 2026-11-17
- **Estado:** construidas y medidas las fases 0 y 1 del plan del «World3 vigente» (un World3 que reproduzca 1970-2023, acierte fuera de muestra y proyecte a 2100 con bandas).
  - **Fase 0:** series mundiales observadas con manifiesto y SHA-256 (`world4/data/world.py`) y marcador del L0 frente a ellas (`world4/validate/world3.py`).
  - **Fase 1:** población por cohortes de edad y sexo acoplada a World3 (`world4/core/l0v`).
  - **Resultado:** el motor de cohortes cumple (población mundial 1950-2023 con 0,7 % de error medio cuando recibe la fecundidad y la esperanza de vida reales). El modelo acoplado **no mejora todavía**: con cohortes el error de población sube de 4,7 % a 6,7 %, porque las cuatro cajas compensaban una ecuación de fecundidad que da demasiados hijos entre 1975 y 2005.
  - **Intento que no pasó:** reestimar esa ecuación con datos hasta 2000 empeora la fecundidad de 2001-2023 (16,3 % frente a 5,1 %). No se adopta.
- **El escenario 1 no cambia:** `l0.run()` sigue idéntico y W1 sigue verde. El gancho de `l0.run` gana una clave, `"set"`.
- **Pruebas:** 81 pasan (16 nuevas, 8 de ellas necesitan los datos); 93 se saltan por datos del Atlas sin instalar, como antes.
- **Reproducir:**
  - `python world4_cli.py --install-world` (unos 35 MB; instala también WPP por edad);
  - `python world4_cli.py --world3-score` (unos segundos);
  - `python -m world4.core.l0v.fit` (el intento de recalibrar la fecundidad, 15 s).

## 1. Fase 0: los datos y el marcador

**Series mundiales** (`data_cache/world/world_obs.npz`, nada de terceros en el repo):

| Serie | Años | Fuente |
|---|---|---|
| Población | 1900-2025 | OWID co2-data (HYDE, Gapminder, ONU) |
| Esperanza de vida, fecundidad total, natalidad, mortalidad, nacimientos, muertes | 1950-2023 | ONU WPP 2024, agregado mundial |
| PIB | 1950, 1960, ..., 2010 y 2015-2022 | Maddison Project vía OWID |
| CO₂ de combustibles fósiles, energía primaria | 1900-2025, 1965-2024 | Global Carbon Budget y Energy Institute vía OWID |
| CO₂ atmosférico | 1959-2025 | NOAA Mauna Loa |
| Kilocalorías por persona | 1961-2013 | FAO vía owid-datasets |
| Rendimiento de cereales (índice) | 1961-2014 | Banco Mundial vía owid-datasets |

- Cada fichero está fijado a un commit y comprobado por SHA-256 (`world_manifest.json`); las cinco fuentes están en `sources.json`, todas con marca [S].
- **Lo que falta:** FAOSTAT posterior a 2013, PIB anual del Banco Mundial, extracción de materiales (UNEP), IDH y huella ecológica. Sus servidores no eran alcanzables desde el entorno de esta sesión.

**Marcador.** Error de un indicador = media de |modelo / real − 1| en 2000-2023, en los años con dato. Población, esperanza de vida, fecundidad y tasas brutas se comparan en sus unidades. Producción, alimento, rendimiento, uso de recursos y contaminación se comparan como índices con 1970 = 1 contra un apoderado declarado (PIB, kilocalorías, rendimiento de cereales, CO₂ fósil, CO₂ atmosférico sobre 280 ppm). Es ajuste al pasado, no una prueba fuera de muestra.

## 2. Fase 1: cohortes

- **Motor** (`cohorts.py`): cada sexo en casillas de edad del ancho del paso (200 casillas hasta los 100 años con `dt = 0,5`, más una abierta). En cada paso toda la población avanza una casilla: una generación numerosa avanza como una ola, sin difuminarse.
- **Mortalidad por edad** (`lifetable.py`): World3 da una esperanza de vida; la familia de tablas de vida la convierte en tasas por edad y sexo.
  - Familia de países: regresión local sobre 3.304 pares país-quinquenio de WPP (e0 de 15 a 86 años).
  - **El mundo no es un país medio.** A igual e0, el agregado mundial tiene más mortalidad infantil y menos de mayores: en 2016-2020, 2,8 veces en 1-4 años y un 30 % menos en 75+. La familia del mundo usa las tablas del propio agregado mundial dentro del rango observado (e0 de 49 a 72) y la forma de la de países fuera [S].
- **Fecundidad por edad:** el patrón por edad del agregado mundial de WPP, por quinquenio; la fecundidad total es la de World3.
- **Acople** (`model.py`): por el gancho de `l0.run`. World3 sigue calculando `le` y `tf`; las cohortes los convierten en muertes y nacimientos y sustituyen las cuatro cajas `p1`..`p4` en cada paso. Arranca al final de 1950 con la población observada por edad (WPP da la población a 31 de diciembre).

**Dos errores que obligó a corregir el dato:**

| Síntoma | Causa | Corrección |
|---|---|---|
| Muertes mundiales de 1951 un 6 % altas | Con `dt = 0,5` la edad 0 se exponía 1,25 años a la mortalidad infantil | Cada casilla lleva la tasa de las edades que recorre en el paso (prueba con mutante) |
| Mayores de 65 un 16 % por debajo en 2023 | La familia de países da al mundo demasiada mortalidad de mayores | Familia del agregado mundial |

## 3. Lo medido

**El motor solo** (cohortes con la fecundidad total y la esperanza de vida reales, comparado con WPP a 31 de diciembre, 1950-2023):

| | Error |
|---|---|
| Población total | 0,70 % medio, 1,03 % máximo |
| Por grupo de edad, máximo | 0-14: 1,0 % · 15-44: 1,1 % · 45-64: 1,5 % · 65+: 2,3 % |
| Nacimientos acumulados por tercios del periodo | −0,2 % · −0,9 % · −0,8 % |
| Muertes acumuladas por tercios del periodo | +1,4 % · +0,3 % · −0,5 % |

**El marcador** (`--world3-score`, escenario 1, error medio 2000-2023 en %):

| Indicador | World3-03 | Cohortes | + fecundidad real | + e0 real | + ambas |
|---|---|---|---|---|---|
| Población | 4,7 | **6,7** | 1,4 | 8,4 | 1,0 |
| Natalidad | 11,6 | 4,4 | 0,9 | 5,3 | 0,2 |
| Mortalidad | 6,9 | 8,8 | 18,1 | 4,0 | 1,1 |
| Esperanza de vida (ecuación de World3) | 8,4 | 2,8 | 5,4 | 3,5 | 6,5 |
| Fecundidad total (ecuación de World3) | 8,3 | 5,4 | 7,3 | 5,6 | 8,1 |
| Alimento por persona | 7,7 | 17,9 | 13,4 | 18,2 | 11,2 |
| Producción | 26,7 | 27,9 | 27,6 | 28,0 | 27,5 |
| Uso de recursos | 106,4 | 86,7 | 90,9 | 87,1 | 94,6 |
| Contaminación | 109,5 | 109,0 | 113,2 | 107,1 | 112,8 |

- **Con cohortes la población pasa de quedarse corta (7.426 millones en 2023) a pasarse (8.613; real 8.092).** La fecundidad de World3 es demasiado alta entre 1975 y 2005 (3,60 frente a 3,31 en 1990) y las cuatro cajas lo tapaban con muy pocas personas en edad fértil.
- **Con la fecundidad real, el error de población baja a 1,4 %.** El error que queda en la población es de la ecuación de fecundidad, no de la estructura.
- **Reparto por edades a finales de 2023** (0-14 / 15-44 / 45-64 / 65+, en %): real 24,9 / 44,1 / 20,9 / 10,1; cohortes 24,8 / 46,6 / 19,5 / 9,1; World3-03 22,7 / 40,3 / 20,3 / 16,6.
- **El alimento por persona empeora** (17,9 %): hay más gente y la agricultura del escenario 1 no produce más. Es el fallo de la fase 2 (rendimientos), ahora sin compensar.
- **La esperanza de vida de World3 acierta más con cohortes** (2,8 %), en parte por una razón equivocada: baja porque falta alimento.
- **Hacia 2100** (modo de comportamiento, no pronóstico): con cohortes el máximo es de 8.782 millones en 2030 (7.467 en 2027 en World3-03) y quedan 3.634 millones en 2100 (3.463).

## 4. El intento de recalibrar la fecundidad

`fit.py` reestima los tres primeros puntos de la tabla `sfsn` (tamaño de familia deseado frente al producto industrial por persona) con la fecundidad observada hasta 2000.

| | Error 1951-2000 | Error 2001-2023 (no visto) |
|---|---|---|
| Tabla de World3-03 | 8,4 % | 5,1 % |
| Tabla reestimada | 4,1 % | **16,3 %** |

- El ajuste prolonga la caída rápida de 1970-2000 y da 1,60 hijos por mujer en 2023 (real 2,25): no ve que el descenso se frena.
- Con esa tabla el error de población del marcador sería 0,9 %, pero con errores del 15 % en la natalidad y del 19 % en la mortalidad: un acierto por compensación, como el de las cuatro cajas.
- **No se adopta.** Una sola curva mundial de fecundidad frente al ingreso medio no reproduce las dos fases.

## 5. Desviaciones y lo que no está

- **El criterio de la fase 1 (población bajo 1 %) lo cumple el motor, no el modelo acoplado.** El acoplado necesita una fecundidad y una esperanza de vida que funcionen fuera de muestra.
- **Tablas de vida por debajo de e0 = 30:** los datos de esa zona son guerras y genocidios, con mucha mortalidad de adultos jóvenes. Un colapso de World3 por hambre tendría otro perfil [S].
- **El patrón de fecundidad por edad se congela en el de 2016-2020** después de 2023.
- **Fotos por edad:** `CohortRun` guarda la población por edad simple y sexo a 31 de diciembre de cada año.
- **No hay criterio nuevo en la campaña.** El marcador es ajuste al pasado; el criterio vendrá con la prueba fuera de muestra (fase 6).
- **El L0 regional (`l0r`) sigue con tasas brutas.** El motor de cohortes sirve para él, pero no se ha conectado.

## 6. Qué sigue

1. **Decisión abierta: fecundidad y esperanza de vida por grupos de países.** Lo medido aquí dice que el promedio mundial no basta. La opción es 3 o 4 grupos de ingreso con el panel regional que el repo ya tiene.
2. **Fase 2:** tecnología estimada (rendimientos, eficiencia, productividad). Con cohortes el alimento por persona es ahora el indicador que más empeora.
3. **Datos pendientes** de la fase 0, cuando sus servidores sean alcanzables.
