# F9 de Phy-World (World4): producto independiente

- **Fecha:** 2026-10-13
- **Estado:** construida y medida.
  - **Paquete instalable** con la orden `world4`; un `pip install` limpio en un venv nuevo funciona.
  - **CLI con el diseño de la familia:** asistente, `--help-numbers`, `--fidelity`, WALLS / TIGHT / THE LAST BLOCKER, `--report`, `--dashboard` y `--mcp`.
  - **Informe HTML autocontenido**, **dashboard** con el globo del Atlas y **servidor MCP** propio.
  - **W11 medido** y documento científico en inglés.
- **Campaña:**
  - **W11: VERDE.** 12 caminos públicos sin una cifra sin unidad o banda, `exploratory` correcto en todos, ninguna palabra prohibida y 5/5 mutantes detectados.
  - **W10: VERDE.** El L0 corre en 9,3 ms; el L1 con 64 miembros, en 36 s con 8 procesos.
  - El resto no cambia: W4, W8 y W9a siguen en rojo, con su causa.
- **Pruebas:** 144/144; 11 son nuevas.

## 1. Qué se construyó

| Pieza | Fichero | Nota |
|---|---|---|
| Hogar | `world4/home.py` | Checkout: la raíz del repo, como en F0-F8. Instalado: `W4_HOME`, o `~/.world4` |
| Paquete | `pyproject.toml` | Orden `world4 = world4.cli:main`. Viajan el posterior de F7 (`world4/models/`) y las huellas de W2 (`world4/data/reference/`). La CLI pasa a `world4/cli.py`; `world4_cli.py` y `cli_style.py` quedan como capas finas |
| CLI | `world4/cli.py` | Ver §2 |
| Informe | `world4/report.py` | Ver §3 |
| Dashboard | `world4/dashboard.py`, `data/geo_manifest.json` | Ver §4 |
| MCP | `world4/mcp/tools.py`, `server.py` | Ver §5 |
| W11 | `world4/honesty.py`, campaña | Ver §6 |
| Ciencia | `docs/Quasar_World4_Science.md` | Ver §7 |

## 2. La CLI

- **Asistente.** Sin modo y con terminal, pregunta qué calcular y después solo los números de ese modo. Por una tubería da `--info` y dice por qué.
- **`--help-numbers`:** los números de cada modo y las palancas, con su caja.
- **`--fidelity`:**
  - L1 para `--query`; L0 para `--theta` y `--report`;
  - L2 se degrada con aviso: los emuladores de L2 (FaIR, cultivos) ya están acoplados en el L1 desde F4/F5;
  - una fidelidad que no existe para el modo es un error de especificación (salida 2).
- **WALLS / TIGHT / THE LAST BLOCKER** sobre TODA la población, con la lógica de Phy-Prop:
  - con `--theta`, si la política viola un límite: un hipercubo de 64 políticas;
  - con `--design`: la población de la búsqueda.

  El caso real:
  - el límite de tierra es un **WALL** (ninguna política lo cumple; mejor g = +0,009) y además THE LAST BLOCKER de 29 políticas;
  - el de alimento es **TIGHT** (lo incumple el 53 %);
  - el consejo dice que solo la fecundidad mueve la tierra en el L0.
- **Salidas:** 0 bien, 2 especificación (también si faltan datos: el mensaje dice qué instalar), 3 infactible, 4 `--strict` con degradaciones.
- **Corrección de paso:** sin datos, `--theta` acababa en un *traceback* de pandas. Ahora el L0 exige los datos del Atlas antes de leerlos y la CLI convierte cualquier error de datos en su remedio.

## 3. El informe HTML

Un solo fichero de unos 65 KB, sin CDN, sin JavaScript y sin librería de gráficos: SVG propio, como en Phy-HX y Phy-Prop. Lleva:
- el semáforo W1-W12 con lo medido y la causa;
- 6 trayectorias con banda p10-p90 a 2100, sin política y con política: población, PIB per cápita, energía, GMST (con las líneas de 1,5 / 2 / 3 °C), cereal per cápita y tierras;
- los lazos dominantes por década;
- los objetivos y las restricciones, el primer año en que el p90 de la GMST cruza cada umbral y WALLS/TIGHT;
- las vulnerabilidades de PRIM de la política;
- la procedencia: 33 fuentes con licencia y marca, los manifiestos con sus SHA-256 y el posterior con su corte.

Los datos del informe van en un JSON al lado. Se revisó en el navegador.

**Una lectura que el informe deja a la vista.** Con I+D agrícola al 0,3 % del PIB, el cereal per cápita de 2100 sale ×2,4. Ese efecto es un supuesto [M] fuerte (F8: +0,3 % de rendimiento por año por cada 0,1 % del PIB).

## 4. El dashboard con el globo del Atlas

**Lo que se reutiliza del Atlas:**
- su librería de globo, `globe.gl` 2.46.2, fijada como allí;
- su geometría: Natural Earth 1:110m, `data/raw/countries-110m.json` del repo del Atlas en el commit `9e96f6b`, instalada desde ese commit y verificada por SHA-256 (`2516c915…`).

**Lo que no se reutiliza:** su plantilla, atada a su esquema de datos y a sus capas OSINT.

**Capas de World4:**
- **L1 F1-F5 desde 2019 a 10 años,** por país: crecimiento del PIB per cápita, cambio de la población, riesgo de conflicto y deuda.
  - Son el p10/p50/p90 del ensemble CRUDO, y la página lo dice: las bandas recalibradas de W8 solo existen en `api.query`.
  - Una sola corrida del L1 da los 180 países (13 s).
- **L0 regional a 2100,** exploratorio.

Datos y geometría van dentro del HTML (216 KB), así que se abre sin servidor. Sin red, la página muestra la tabla en lugar del globo. Se revisó en el navegador.

## 5. El servidor MCP

- **Herramientas:** funciones puras en `tools.py` (catálogo, consulta del L1, palancas, evaluar una política, bandas del L0, lazos dominantes). El servidor y cualquier capa HTTP llaman a las mismas funciones, así que no pueden divergir.
- **Servidor:** JSON-RPC 2.0 por stdio, sin dependencias (el paquete `mcp` no está):
  - métodos `initialize`, `ping`, `tools/list` y `tools/call`;
  - stdout es solo del protocolo; lo que imprima el modelo va a stderr;
  - un error de dominio vuelve como resultado con `isError`; un método desconocido, como error JSON-RPC.

## 6. W11: honestidad medida

`honesty.py` aplica tres reglas a la API pública, a todas las herramientas MCP y al informe:

1. **Ninguna cifra sin unidad ni banda.**
   - Una cadena `unit` cubre todo lo que tiene debajo; un dict `units` solo las claves que nombra.
   - Todo `p50` lleva `p10` y `p90`.
   - Una serie sin banda dice por qué no la tiene.
2. **«validated» prohibida** fuera del nombre del horizonte medido. Se excluye `honesty.py`, que la contiene porque es la regla.
3. **`exploratory` correcto en cada camino.**

**Lo que encontró.** La primera versión dejaba pasar todo porque contaba un `units` del nivel superior como unidad de todo. Al endurecerla aparecieron dos huecos reales, corregidos en origen:
- los objetivos y las restricciones de `evaluate_policy` no tenían unidad;
- los años de cruce de umbrales del informe no decían qué eran.

También cazó un texto mío en la campaña.

**Mutantes, todos detectados:** quitar la etiqueta, una unidad o una banda, una serie sin nota y la palabra plantada.

## 7. El documento científico

`docs/Quasar_World4_Science.md`, en inglés y con la regla de admisión de la familia: solo números medidos aquí o de fuentes abiertas. Recorre:
- las capas y sus ecuaciones;
- la estimación y la calibración;
- la recalibración de W8;
- las políticas, la conducta y la honestidad;
- la campaña;
- lo que deliberadamente no contiene: los informes de investigación sin verificar, la tabla de Nebel, Kotz 2024 retractado y Richardson no comercial.

## 8. `pip install` limpio en un venv nuevo

Con Python 3.13.7, un venv vacío y `pip install <checkout>`:
- se instalan numpy, scipy, pandas, statsmodels y phycore (desde git, en su SHA);
- `world4 --info` y `world4 --help-numbers` funcionan;
- el posterior se carga desde `site-packages/world4/models/` (corte 2019, revisión 7);
- el servidor MCP responde.
- `world4 --campaign` deja todo BLOQUEADO salvo W12 (ámbar). **Es el comportamiento correcto:** sin datos instalados nada puede ponerse en verde.

**W12 casi se rompe por un error mío en esta fase.** Al llevar las referencias al paquete copié también la salida de Vensim (un dato de terceros) y W12 pasó a rojo. Se sacó: vive en `tests/` del repo, y una instalación sin ella deja W1 BLOQUEADO diciendo por qué.

## 9. Desviaciones y lo que no está

- **El repo no está publicado,** así que la prueba de instalación se hizo desde el checkout. La instalación además necesita git y red para phycore.
- **El dashboard depende de un CDN** para la librería del globo (como el del Atlas); el informe no. Las texturas de la Tierra del Atlas no se usan: el globo es de un solo color con los países.
- **`--scenario` sigue siendo el escenario de World3-03.** Las narrativas de F8 van con `--narrative`.
- **Sin servidor HTTP propio:** la hoja de ruta lo menciona junto al MCP. Las funciones de `tools.py` están listas para él; llega con el addon de Studio (F10).

## 10. Qué sigue

1. **F10:** el addon de Quasar Studio. El manifiesto, el adaptador, `requirements-world4.txt` y el globo ya existen; también `studio_compat`.
2. **Publicar el repo** (decisión del usuario) y repetir la instalación desde GitHub.
3. **Los rojos con causa:** W4 (sesgo no estacionario), W8 (forma del PIT) y W9a (asimilación entre el origen y `as_of`).
