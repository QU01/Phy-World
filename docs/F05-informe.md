# F0.5 de Phy-World (World4): puente con Prime Radiant

- **Fecha:** 2026-10-11
- **Estado:** construida y medida.
  - Integración en Prime Radiant: verde (167/167 pruebas, 19 nuevas, incluida una de integración contra el World4 real).
  - **W9a: ROJO.**
  - **W8: ROJO.**
  - Ambas con causa nombrada y medida.
- **Pruebas:** World4 60/60; Prime Radiant 167/167 (antes 148/148).

## 1. Qué se construyó

### En World4 (`world4/structural/`)

| Pieza | Qué hace |
|---|---|
| `engine.py` | Ensembles del L1 (32 miembros) con **origen honesto** y **población honesta**. Una pregunta hecha a mediados del año Y se pronostica desde el origen Y-3, porque el reanclaje usa el PIB de la PWT, que se publica con 3 años de retraso. Después del origen, la población y el capital humano se extrapolan con la tendencia de 5 años; **esto cierra D1 en el estimador**, mientras el port del Atlas sigue intacto (W2). El modo vivo parte de 2019 con WPP 2024. |
| `observed.py` | Desenlaces observados y **tasa base con datos (A)**: frecuencia del país en los últimos 20 años publicados, encogida hacia su zona de renta. También una segunda referencia condicionada al último estado conocido, solo para informar. |
| `store.py` | Un ensemble por origen (1970-2018) más el origen vivo (2019→2030): 50 ensembles en 61 s. Vive en `data_cache/`. |
| `recal.py` | **Recalibración (D).** Eventos: isotónica (≥ 200 pares), Platt (30-199) o identidad, la misma escalera que Prime Radiant. Numéricas: recalibración por cuantiles con núcleo t de Student. |
| `questions.py` | 138 735 preguntas históricas de eventos y 71 329 numéricas, todas respondidas con lo que se sabía en su fecha. Evaluación W8 y W9a. |
| `estimator.py` | Petición JSON → bloque. Diagnósticos con nombre: `unbound`, `out_of_domain`, `asof_violation`, `unavailable`. |
| CLI | `--build-store` (ensembles, preguntas, W8 y W9a) y `--structural REQUEST.json` (JSON de entrada y de salida; es lo que llama Studio). |

### En Quasar Studio (Prime Radiant)

Los cambios están en `docs/F05-qstudio.patch`, que contiene **solo** los míos: el repo de Studio tenía 124 cambios tuyos sin commit.

- **Nuevo `prime_radiant/structural/`:**
  - `runner`: llama a World4 por subproceso, con timeout, y no lanza nunca;
  - `validation`: la compuerta W9b, que **en F0.5 siempre devuelve falso**;
  - `__init__`: el diseño (borrador de ADR 0016).
- **`schemas.py`:** `StructuralBinding` y `Question.structural`. El enlace lo declara quien crea la pregunta; ningún LLM lo propone en la v1.
- **`pipeline.py`:**
  - etapa `structural` después de la evidencia;
  - d1 contrafactual con el bloque → `p_final_withstructural`;
  - campos `p_structural`, `base_rate_data`, `structural_diagnostics`;
  - el pronóstico publicado no cambia.
- **`evaluation.py`, `choice.py`:** los tres estimadores entran en las ablaciones.
- **Router:** `POST /questions` acepta `structural`.
- **Configuración por entorno:**
  - `QUASAR_PR_STRUCTURAL_ENABLED` (apagada por defecto);
  - `QUASAR_PR_WORLD4_PYTHON`, `QUASAR_PR_WORLD4_DIR`, `QUASAR_PR_WORLD4_TIMEOUT_S`.

**Desviación del plan: la configuración no entra en `SwarmConfig`.** Su huella identifica al calibrador y `_baseline_row` la recalcula para normalizar filas antiguas del ledger; añadir un campo cambiaría todas las huellas existentes. Mientras World4 no alimente a d1, la procedencia del estimador publicado no cambia. Meterlo en la huella es trabajo de F11.

**No se hizo (F11):** equipo rojo con el bloque, panel en el frontend, herramientas QUO `w4_*`, ramas de situaciones.

## 2. W9a: el estimador contra la tasa base

Preguntas hechas el 1 de julio de 1990-2018 sobre los años Y+1..Y+h (h = 1..5). Brier (menor es mejor):

| Evento | N | Frecuencia | World4 | Atlas crudo | Tasa base (A) | Base condicionada | World4 − tasa base, IC95 por países |
|---|---|---|---|---|---|---|---|
| Conflicto (UCDP ≥ 25 muertes) | 24 300 | 15 % | **0,085** | 0,085 | 0,101 | 0,076 | −0,016 [−0,026, −0,006] |
| Golpe exitoso | 23 250 | 2 % | 0,025 | 0,050 | 0,026 | 0,024 | −0,001 [−0,003, +0,001] |
| Crisis financiera | 19 665 | 18 % | **0,151** | 0,145 | 0,158 | 0,156 | −0,007 [−0,012, −0,003] |
| Democracia (RoW ≥ 2) al final de Y+h | 21 810 | 52 % | 0,108 | 0,111 | **0,075** | 0,074 | **+0,034 [+0,027, +0,041]** |
| **Conjunto** | 89 025 | | | | | | +0,002 [−0,002, +0,006] |

**Lectura:**

- **Conflicto y crisis:** World4 bate a la tasa base con IC que excluye el 0. En conflicto pierde contra la base condicionada (la persistencia del estado es casi todo); en crisis bate a las dos.
- **Golpe:** empate. La recalibración arregla un Atlas crudo muy mal calibrado (0,050 → 0,025).
- **Democracia:** pierde con claridad, y la causa es estructural. El origen honesto (Y-3) va **tres años por detrás** del último régimen publicado (Y-1), que la tasa base sí ve. El remedio es asimilar lo publicado entre el origen y `as_of` (filtro de partículas sobre los miembros), que es F11; aquí no se coló por la puerta de atrás.
- **ViEWS y PITF no se informan:** no hubo acceso en F0.5. W9a no puede ser verde sin ellos.
- **Veredicto: ROJO** (IC conjunto incluye 0 y democracia es peor).

## 3. W8: las bandas recalibradas

| Variable | Cobertura p10-p90 cruda → recalibrada | Distancia KS cruda → recalibrada |
|---|---|---|
| Crecimiento del PIB per cápita | 0,93-0,95 → **0,84-0,85** | 0,165-0,171 → 0,127-0,197 |
| Deuda / PIB | 0,51-0,55 → **0,79-0,83** | 0,147-0,159 → 0,064-0,164 |

- **La cobertura entra en la banda 75-85 % en las 10 celdas.** Las bandas crudas del Atlas eran anchas para el crecimiento y estrechas para la deuda (D7).
- **La forma no queda uniforme fuera de muestra.** Dentro de la muestra la recalibración deja la distancia en 0,002; fuera de ella sube a 0,06-0,20, porque **el sesgo del modelo cambia de década**: un recalibrador entrenado con 1970-1989 desplaza los PIT de 1990-2019. Por la misma causa la recalibración empeora la crisis en W9a (0,151 frente a 0,145 crudo).
- **Hallazgo de método:** con núcleo gaussiano, una observación a varias desviaciones del ensemble daba PIT = 0 o 1 exactos (el 25 % de los PIT de deuda caían en los deciles extremos), y ninguna recalibración puede volver a repartir esa masa. El núcleo t (3 g. l.) no satura y bajó la distancia de la deuda de 0,159 a 0,064 a 4 años.
- **Cambio de criterio, declarado:** W8 pedía "KS p > 0,05" y ahora pide **distancia D ≤ 0,05**. Con el recalibrador estimado en muestra finita, el p-valor de una sola muestra no es válido a n ≈ 5 000; un control con PIT uniformes puros pasa con p 0,18 y la prueba sintética fallaba con p 0,003 sin defecto alguno. El cambio se hizo **después** de ver los datos, pero **no cambia el veredicto**: los datos reales dan D = 0,06-0,20.
- **Veredicto: ROJO.**

## 4. Lo que la integración real destapó

- **La prueba de integración contra el World4 real** cazó un `KeyError` en el estimador: el número de miembros se leía del sitio equivocado. Las 18 pruebas con un World4 falso no podían verlo.
- **La prueba de honestidad** cazó un mensaje que decía que el L1 está "validated" 1-10 años.
- **Ejemplo vivo** (Colombia, conflicto en 2027, preguntado el 2026-10-11): P cruda 0,66 → recalibrada 0,75; tasa base con datos 0,69; origen 2019, siete años sin asimilar datos, aviso incluido.

## 5. Cómo activarlo en Studio

```powershell
$env:QUASAR_PR_STRUCTURAL_ENABLED = "1"
$env:QUASAR_PR_WORLD4_PYTHON = "C:\Users\nickv\projects\Proyectos\World4\Quasar-World4\.venv\Scripts\python.exe"
$env:QUASAR_PR_WORLD4_DIR = "C:\Users\nickv\projects\Proyectos\World4\Quasar-World4"
```

Después, crear la pregunta con `"structural": {"variable": "conflict", "region": "COL"}`. Antes, una sola vez: `world4_cli.py --install-atlas` y `world4_cli.py --build-store`.

## 6. Qué sigue

- **W9b:** el reloj empieza ahora. Hacen falta 30 preguntas reales resueltas con d1 con y sin el bloque.
- **Mejoras que atacan las causas medidas, en orden:**
  1. asimilar lo publicado entre el origen y `as_of` (democracia, y en general toda pregunta viva desde 2019);
  2. recalibradores con ventana móvil o ponderación temporal (la deriva por décadas de W8 y de crisis);
  3. ViEWS/PITF como referencias en W9a.
