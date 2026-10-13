# F0 de Phy-World (World4): informe

- **Fecha:** 2026-10-11
- **Estado:** F0 cerrada. W1 y W2 en verde, W12 en ámbar con causa, el resto BLOQUEADO con la fase que lo desbloquea.
- **Pruebas:** 43 en verde (`pytest -q`, ~17 s), más la validación completa del L1 (~8 min, `scripts/run_atlas_validation.py`).

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| L0: World3-03 en NumPy | `world4/core/l0.py` | Vectorizado sobre un eje de regiones (`n`); escenarios 1-5 del libro de 2004 |
| L1: Atlas portado | `world4/core/l1_atlas/` | 24 ficheros; solo cambian imports y la constante `ROOT` (`scripts/port_atlas.py` imprime el diff) |
| Datos del Atlas | `world4/data/atlas.py`, `atlas_manifest.json` | Se instalan desde el repo del Atlas en `9e96f6b`; 34 ficheros con SHA-256 |
| Registro de fuentes | `world4/data/sources.json`, `vintages.py` | 28 fuentes con licencia, marca, redistribución y retraso de publicación; `known_time` |
| Paridad W2 | `world4/validate/atlas_parity.py`, `tests/golden/` | Sonda de 81 huellas y validación completa del original en esta plataforma |
| API | `world4/api.py` | `catalog`, `query` (L1, 1-10 años, 1970-2018) y `run_l0` |
| Campaña | `world4/campaign.py` | W1-W12 sobre `phycore.vvuq.Campaign` |
| CLI | `world4_cli.py`, `cli_style.py` | Patrón de la familia; `--info`, `--campaign`, `--install-atlas`, `--l0`, `--query` |
| Licencia | `LICENSE`, `NOTICE` | Apache-2.0, la propuesta de la hoja de ruta, que el usuario no ha confirmado todavía |

## 2. W1: el L0 contra World3-03

**Referencia.** La salida del propio modelo Vensim World3-03 (escenario 1) para alimentos, producción industrial, recursos no renovables, índice de contaminación persistente y población, cada 0,5 años de 1900 a 2100. La publica WorldDynamics.jl en `test/world3_03.jl` (MIT) y está copiada en `tests/reference/world3_03_vensim.csv` con su SHA-256.

**Ecuaciones.** Se transcribieron de pydynamo (dominio público) pero con la semántica de Vensim, no la de DYNAMO:

- en cada instante, todos los auxiliares se calculan con los stocks de ese instante;
- se integra por Euler con dt = 0,5;
- SMOOTH, SMOOTH3 y DELAY3 son stocks explícitos.

**Resultado:** error relativo máximo **6,9e-6** en las cinco variables (listón: 1 %). Vensim imprime 6 cifras, así que es el error de redondeo de la referencia.

**Hallazgos:**

1. **La única comparación estricta del modelo.** Vensim conmuta la tabla `lmhs` con `Time > 1940`, no `>=`. Con `>=` el error sube a 0,4 %. Queda en el código y como mutante en las pruebas.
2. **El listón del 1 % es demasiado laxo para cazar ese error.** Por eso hay además un candado de regresión a 1e-4, que sí lo detecta. Los mutantes de tabla (la de 1972 en lugar de la de 2003) y de constante (`nri` +10 %) sí rompen el 1 %.
3. **pydynamo no es referencia numérica.** Su simulación se aparta de Vensim hasta un **12 %** (`io` en 2100, `ppolx` en 2095), por la temporización `.j` de DYNAMO y la conmutación `>=` de 1940. Sus ecuaciones transcritas sí son correctas.
4. **El "coincide con al menos otra" de P1 se cumple así:** el L0 es una implementación independiente de Vensim y coincide con él a 7e-6.
5. **Rendimiento:** 225 ms por corrida con n = 1, y 1,6 ms por miembro con n = 1000. W10 (< 10 ms) es de F6.

## 3. W2: el L1 contra el Atlas

1. **Sonda bit a bit.** Corrida histórica, pronósticos desde 1990 (regla de conducta, política congelada y ablación sin clima), referencias de comparación y pérdida de calibración en el corte de 1990: **81 de 81 huellas SHA-256 idénticas** entre el Atlas original y el port. Un cambio de 1e-9 en `ALPHA` rompe la paridad (mutante en las pruebas).
2. **Validación completa** (49 orígenes, 8 variantes más la política congelada, R = 32): **780 de 780 valores idénticos** entre el original y el port en esta máquina.
3. **Contra el `validation.json` guardado en el repo del Atlas,** generado en otra máquina: todo coincide a **4,7e-15** salvo los AUC.

**Defecto del Atlas: los AUC con empates no son reproducibles.** El `auc()` del Atlas ordena con `argsort`, y cuando las puntuaciones empatan el resultado depende de cómo desempate la ordenación de cada CPU. Con los mismos datos:

| AUC | Fichero guardado | Esta máquina |
|---|---|---|
| Persistencia, conflicto a 10 años | 0,84 | 0,56 |
| Democracia, ablación "sin instituciones" | — | se mueve 0,07 |
| Resto de variantes de democracia | — | se mueven ~1e-3 |

Afecta a las referencias de puntuación casi binaria y a las probabilidades de democracia (promedio de 32 regímenes binarios). Los AUC de conflicto del modelo y del logit no se mueven. Por tanto, **los AUC de persistencia del README del Atlas no son fiables.**

`world4/validate/metrics.auc` usa rangos promedio y no depende del orden (hay una prueba que lo demuestra contra el del Atlas). El port conserva el defecto a propósito, porque W2 exige no cambiar comportamiento; la corrección entra en F0.5, donde el AUC empieza a contar.

**Otra trampa del Atlas:** `model.build_world` lee `panel_full.csv`, que no está en el repo, y **si falta cae en silencio a `panel.csv`**, con otro resultado. `atlas.install` lo genera con el `merge_sources` portado (sale idéntico byte a byte) y `atlas.require` exige su SHA-256 antes de cualquier corrida del L1.

## 4. W12: datos

- **28 fuentes registradas:**
  - 12 con licencia supuesta [S];
  - 16 con licencia desconocida [?];
  - ninguna [V], porque en esta sesión no se abrió ninguna licencia.
- **Ni un dato de terceros dentro del paquete.** Los ficheros del Atlas se instalan desde su repo y se verifican por SHA-256.
- **EM-DAT no es redistribuible** según sus términos. **Aviso:** el repo público del Atlas (`clima.csv`) lo incluye. Es una decisión del usuario sobre ese repo, no de World4.
- **`known_time`:** `vintages.leakage_report()` mide cuánto se adelanta el Atlas a la publicación real de cada fuente en su validación de origen móvil: hasta 3 años con PWT, COW, GFN y WVS, y 5 con Barro-Lee. Se mide en F0 y se aplica en F0.5 y F1.

## 5. Desviaciones del plan

- **La referencia de P1 no es pyworld3,** que es la versión de 1974 y tiene licencia CeCILL. Es la salida de Vensim; ya estaba anotado en la hoja de ruta.
- **Los parámetros recalibrados de Nebel et al. (2023) como escenario del L0** no se añadieron. Hace falta leer su tabla de parámetros en el artículo y no se abrió en esta sesión. Queda pendiente con causa.
- **`api.query` solo cubre 1970-2018 + 1-10 años.** El L1 portado tiene mundo hasta 2019, y proyectar más allá requiere extender las series exógenas: es F0.5.
- **La CLI no tiene asistente;** sin argumentos imprime `--info` y lo dice. Llega en F9.
- **El paquete se llama `world4` y la distribución `quasar-phy-world`.** El nombre visible es Phy-World. Renombrar el paquete y el repo es una decisión pendiente.
